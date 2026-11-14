"""Run every evaluation and refresh the results table in README.md.

Run: python -m evals.run_all            every eval in full, then the README table
     python -m evals.run_all --smoke    every eval with --limit N (results in results/smoke/)
     python -m evals.run_all --table    only rebuild the README table and docs from results
LLM answers come from the disk cache, so a repeated run costs no API calls.
"""

import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv

from evals import report
from src.config import ROOT, load_config
from src.data import sets

STEPS = ["src.pipeline.batch", "evals.eval_data", "evals.eval_features", "evals.eval_relevance",
         "evals.eval_sentiment", "evals.eval_advisor", "evals.eval_simulator", "evals.eval_judge",
         "evals.eval_lookahead", "evals.eval_ui", "evals.eval_llm_client", "evals.walkthrough",
         "evals.eval_targets"]
NO_LIMIT = {"evals.eval_data", "evals.eval_features", "evals.eval_ui", "evals.eval_llm_client",
            "evals.walkthrough", "evals.eval_targets"}
FULL_ONLY = {"evals.eval_ui", "evals.walkthrough", "evals.eval_targets"}  # no reduced form
START, END = "<!-- EVALS:START -->", "<!-- EVALS:END -->"
KEYS = ["OPENROUTER_API_KEY", "FINNHUB_API_KEY"]
SAMPLES = ["set_a.jsonl", "set_b.jsonl", "set_c.jsonl", "outcomes.csv", "phrasebank_sample.csv"]


def check_ready() -> None:
    """Stop with a clear message when keys or eval samples are missing."""
    load_dotenv(ROOT / ".env")
    missing = [key for key in KEYS if not os.environ.get(key)]
    missing += [name for name in SAMPLES if not sets.processed(name).exists()]
    if missing:
        sys.exit(f"Cannot run the evals, missing: {', '.join(missing)}")


def run_steps(limit: int | None) -> None:
    """Run each eval module as its own process, in dependency order."""
    for module in STEPS:
        command = [sys.executable, "-m", module]
        if limit and module not in NO_LIMIT:
            command += ["--limit", str(limit)]
        if limit and module in FULL_ONLY:
            continue
        print(f"\n=== {' '.join(command[1:])}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def refresh_readme() -> None:
    """Replace the table between the markers in README.md."""
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    before, rest = text.split(START, 1)
    after = rest.split(END, 1)[1]
    table = report.markdown_table(report.table_rows())
    path.write_text(f"{before}{START}\n{table}\n{END}{after}", encoding="utf-8")


def changes(before: list[dict], after: list[dict]) -> list[str]:
    """List the metrics whose value changed since the previous run."""
    old = {(row["Component"], row["Metric"]): row["Value"] for row in before}
    lines = []
    for row in after:
        key = (row["Component"], row["Metric"])
        if key in old and old[key] != row["Value"]:
            lines.append(f"{key[0]} / {key[1]}: {old[key]} -> {row['Value']}")
    return lines


def main() -> None:
    """Parse the options, run the steps and refresh the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="quick run with --limit")
    parser.add_argument("--table", action="store_true", help="only rebuild the README table")
    args = parser.parse_args()
    before = report.table_rows()
    if not args.table:
        check_ready()
        run_steps(load_config()["evals"]["smoke_limit"] if args.smoke else None)
    if not args.smoke:
        refresh_readme()
        subprocess.run([sys.executable, "-m", "evals.render_docs"], cwd=ROOT, check=True)
        changed = changes(before, report.table_rows())
        print("\nChanged metrics (before -> after):" if changed else "\nNo metric changed.")
        print("\n".join(changed))


if __name__ == "__main__":
    main()
