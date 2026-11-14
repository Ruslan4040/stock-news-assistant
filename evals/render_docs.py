"""Fill document templates with numbers from evals/results/*.json.

Run: python -m evals.render_docs                 render docs/templates/*.md into docs/
     python -m evals.render_docs --print FILE    render one template to the terminal
No number in the rendered documents is typed by hand (AGENTS.md §7).

Placeholders:
  {{advisor:C.hit_rate}}                   the metric value
  {{advisor:C.hit_rate:ci}}                its 95% interval
  {{advisor:C.hit_rate:n}}                 its n
  {{advisor:C.hit_rate:base:always_buy}}   one of its baselines
  {{ui:checks_passed_rate:field:passed}}   another field of the metric row
  {{relevance:@confusion.tp}}              a value from `details`
  {{advisor:#run_date}}                    a top-level field of the results file
Add |pct for percent, |int for a whole number, |raw for the value as stored.
"""

import argparse
import re
from pathlib import Path

from evals import common
from src.config import ROOT

PLACEHOLDER = re.compile(r"\{\{\s*([a-z_]+):([^}|]+?)\s*(?:\|\s*(\w+))?\s*\}\}")
TEMPLATES = ROOT / "docs" / "templates"
MISSING = "n/a"


def fmt(value: object, style: str | None) -> str:
    """Format one value: pct, int, raw, or three decimals by default."""
    if value is None:
        return MISSING
    if style == "raw" or isinstance(value, str | bool | dict | list):
        return str(value)
    if style == "pct":
        return f"{value * 100:.1f}%"
    if style == "int" or isinstance(value, int):
        return f"{round(value):,}"
    return f"{value:.3f}"


def dig(data: object, dotted: str) -> object:
    """Follow a dotted path through nested dicts; None when a key is missing."""
    for key in dotted.split("."):
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def metric_value(row: dict, parts: list[str], style: str | None) -> str:
    """Resolve the part after the metric name: value, ci, n, base:<name> or field:<name>."""
    what = parts[0] if parts else "value"
    if what == "ci":
        return f"[{fmt(row['ci_low'], style)}, {fmt(row['ci_high'], style)}]"
    if what == "base":
        return fmt(row["baselines"].get(parts[1]), style)
    if what == "field":
        return fmt(row.get(parts[1]), style)
    return fmt(row.get({"value": "value", "n": "n", "low": "ci_low", "high": "ci_high"}[what]),
               style)


def resolve(component: str, path: str, style: str | None) -> str:
    """Return the text for one placeholder."""
    record = common.load_result(component)
    if record is None:
        return MISSING
    if path.startswith("@"):
        return fmt(dig(record["details"], path[1:]), style)
    if path.startswith("#"):
        return fmt(record.get(path[1:]), style or "raw")
    name, *parts = path.split(":")
    rows = [row for row in record["metrics"] if row["name"] == name]
    return metric_value(rows[0], parts, style) if rows else MISSING


def render(text: str) -> str:
    """Replace every placeholder in a template."""
    return PLACEHOLDER.sub(lambda m: resolve(m.group(1), m.group(2).strip(), m.group(3)), text)


def render_file(template: Path) -> str:
    """Render one template file and report unresolved placeholders."""
    rendered = render(template.read_text(encoding="utf-8"))
    missing = rendered.count(MISSING)
    if missing:
        print(f"  note: {template.name} has {missing} values shown as '{MISSING}'")
    return rendered


def main() -> None:
    """Render all templates into docs/, or print one template."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--print", dest="single", help="render this template to the terminal")
    args = parser.parse_args()
    if args.single:
        print(render_file(Path(args.single)))
        return
    for template in sorted(TEMPLATES.glob("*.md")):
        target = ROOT / "docs" / template.name
        target.write_text(render_file(template), encoding="utf-8")
        print(f"rendered docs/{template.name}")


if __name__ == "__main__":
    main()
