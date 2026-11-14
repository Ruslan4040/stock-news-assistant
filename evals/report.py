"""Collect evals/results/*.json into one table (README, report and the app's Evals tab)."""

from evals import common

ORDER = ["data", "features", "llm_client", "relevance", "sentiment", "advisor", "judge",
         "simulator", "lookahead", "ui"]
HEADER = ["Component", "Metric", "Value", "95% CI", "Baseline", "n", "Prompt", "Model"]


def number(value: object) -> str:
    """Format a metric value for the table."""
    if value is None:
        return "n/a"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    if isinstance(value, int) or float(value).is_integer() and abs(value) >= 2:
        return f"{int(value)}"
    return f"{value:.3f}"


def interval(row: dict) -> str:
    """Format the bootstrap confidence interval of a metric row."""
    if row.get("ci_low") is None:
        return ""
    return f"[{number(row['ci_low'])}, {number(row['ci_high'])}]"


def baselines(row: dict) -> str:
    """Format the baseline values of a metric row."""
    return "; ".join(f"{name} {number(value)}" for name, value in row["baselines"].items())


def table_rows() -> list[dict]:
    """Return one table row per metric of every results file that exists."""
    rows = []
    for component in ORDER:
        record = common.load_result(component)
        if record is None:
            continue
        for metric in record["metrics"]:
            rows.append({
                "Component": component, "Metric": metric["name"], "Value": number(metric["value"]),
                "95% CI": interval(metric), "Baseline": baselines(metric),
                "n": metric["n"], "Prompt": record["prompt_version"] or "",
                "Model": record["model"] or "",
            })
    return rows


def markdown_table(rows: list[dict]) -> str:
    """Render the rows as a Markdown table."""
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    lines += ["| " + " | ".join(str(row[column]) for column in HEADER) + " |" for row in rows]
    return "\n".join(lines)
