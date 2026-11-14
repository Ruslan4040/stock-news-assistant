"""Compare two advisor prompt versions on the same decisions (the improvement loop).

Run: python -m evals.compare_advisor_prompts advisor_v2 [--limit 60]
Uses the first decisions of sets A and C in the fixed run order, with the news labels already
stored for them, so only the advisor call differs. Writes
evals/results/advisor_prompt_comparison.json.
"""

import argparse

import pandas as pd

from evals import common
from evals.eval_advisor import direction_rows, directional, horizon_rows
from src.config import load_config
from src.data import sets
from src.pipeline import advisor, batch
from src.pipeline.run import PipelineResult

COMPONENT = "advisor_prompt_comparison"
SOURCE_SETS = "AC"
DEFAULT_LIMIT = 60


def dev_results(letter: str, limit: int) -> list[PipelineResult]:
    """Return the first valid stored results of a set in the fixed run order."""
    stored = {r.decision.id: r for r in batch.load_results(letter)}
    ordered = [stored[i] for i in batch.run_order(letter) if i in stored]
    valid = [r for r in ordered if not batch.incomplete(r) and not r.advice.error]
    return valid[:limit]


def rows_for(results: list[PipelineResult], version: str) -> pd.DataFrame:
    """Ask the advisor with one prompt version and join the true forward returns."""
    outcomes = sets.load_outcomes()
    advice = common.parallel_map(
        lambda r: advisor.run(r.decision, r.labeled, version=version), results)
    rows = []
    for result, item in zip(results, advice, strict=True):
        truth = outcomes.loc[result.decision.id]
        rows.append({"id": result.decision.id, "action": item.action, "horizon": item.horizon,
                     "confidence": item.confidence, "error": item.error or "", "incomplete": False,
                     "last_move": result.decision.features.ret_1d,
                     "fwd_1d": truth["fwd_ret_1d"], "fwd_1w": truth["fwd_ret_1w"]})
    frame = pd.DataFrame(rows)
    frame["fwd"] = frame["fwd_1d"].where(frame["horizon"] == "1d", frame["fwd_1w"])
    frame["side"] = frame["action"].map({"buy": 1, "sell": -1, "hold": 0})
    return frame


def summarize(name: str, frame: pd.DataFrame) -> tuple[list[dict], dict]:
    """Return the metric rows and the action / horizon mix for one prompt version."""
    calls = directional(frame)
    valid = frame[frame["error"] == ""]
    share_1d = {"value": float((valid["horizon"] == "1d").mean()), "ci_low": None,
                "ci_high": None, "n": len(valid)}
    rows = direction_rows(name, calls, len(valid)) + horizon_rows(name, calls)
    rows.append(common.metric_row(f"{name}.share_horizon_1d", share_1d))
    mix = {"actions": valid["action"].value_counts().to_dict(),
           "horizons": valid["horizon"].value_counts().to_dict(),
           "invalid_output": int((frame["error"] != "").sum()),
           "mean_confidence": round(float(valid["confidence"].mean()), 2)}
    return rows, mix


def main() -> None:
    """Run both prompt versions on the same decisions and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("candidate", help="prompt version to test, e.g. advisor_v2")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="decisions per set")
    args = parser.parse_args()
    current = load_config()["prompts"]["advisor"]
    metrics, details = [], {"decisions_per_set": args.limit, "sets": SOURCE_SETS}
    for letter in SOURCE_SETS:
        results = dev_results(letter, args.limit)
        for version in (current, args.candidate):
            rows, mix = summarize(f"{letter}.{version}", rows_for(results, version))
            metrics += rows
            details[f"{letter}.{version}"] = mix
    data = f"first {args.limit} decisions of sets A and C in the run order"
    path = common.write_result(COMPONENT, metrics, data, f"{current} vs {args.candidate}",
                               details=details)
    common.show(path)


if __name__ == "__main__":
    main()
