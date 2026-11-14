"""Eval of the LLM client: valid JSON rate, errors after retry, latency, cache hit rate.

Run: python -m evals.eval_llm_client
Reads logs/llm_calls.jsonl, which holds every call made by the pipeline and the evals.
"""

import numpy as np
import pandas as pd

from evals import common
from src.llm import trace

COMPONENT = "llm_client"


def load_calls() -> pd.DataFrame:
    """Load the trace log into a frame with one row per logical call."""
    frame = pd.DataFrame(trace.read_calls())
    frame["error"] = frame["error"].fillna("")
    frame["transport_error"] = frame["error"].str.startswith("transport")
    frame["invalid_error"] = frame["error"].str.startswith("invalid_output")
    return frame


def per_component(fresh: pd.DataFrame) -> list[dict]:
    """Summarize the fresh (not cached) calls of each component."""
    rows = []
    for component, part in fresh.groupby("component"):
        answered = part[~part["transport_error"]]
        rows.append({
            "component": component,
            "calls": len(part),
            "valid_first_rate": round(float(answered["valid_first"].mean()), 4),
            "invalid_after_retry": int(part["invalid_error"].sum()),
            "transport_errors": int(part["transport_error"].sum()),
            "mean_latency_s": round(float(part["latency_s"].mean()), 1),
            "mean_attempts": round(float(part["attempts"].mean()), 2),
        })
    return rows


def worst_calls(fresh: pd.DataFrame) -> list[dict]:
    """Return failed calls first, then the slowest ones."""
    ordered = fresh.sort_values(["invalid_error", "transport_error", "latency_s"], ascending=False)
    return [
        {"ts": row["ts"], "component": row["component"], "error": row["error"][:200],
         "first_error": row.get("first_error"), "attempts": row["attempts"],
         "latency_s": row["latency_s"], "output": str(row["output"])[:300]}
        for _, row in ordered.iterrows()
    ]


def main() -> None:
    """Compute the client metrics from the trace log and write the results file."""
    common.limit_argument(__doc__)
    calls = load_calls()
    fresh = calls[~calls["cached"]]
    answered = fresh[~fresh["transport_error"]]
    boot, mean = common.bootstrap, common.mean
    metrics = [
        common.metric_row("valid_json_first_answer_rate",
                          boot(mean, answered["valid_first"].to_numpy(dtype=float))),
        common.metric_row("invalid_after_retry_rate",
                          boot(mean, fresh["invalid_error"].to_numpy(dtype=float)),
                          count=int(fresh["invalid_error"].sum())),
        common.metric_row("transport_error_rate",
                          boot(mean, fresh["transport_error"].to_numpy(dtype=float)),
                          count=int(fresh["transport_error"].sum())),
        common.metric_row("mean_latency_s", boot(mean, fresh["latency_s"].to_numpy(dtype=float))),
        common.metric_row("mean_http_attempts_per_call",
                          boot(mean, fresh["attempts"].to_numpy(dtype=float))),
        common.metric_row("cache_hit_rate", boot(mean, calls["cached"].to_numpy(dtype=float))),
    ]
    details = {"calls_total": len(calls), "calls_fresh": len(fresh),
               "per_component": per_component(fresh),
               "note": "transport errors are API failures (5xx, cut answers); "
                       "invalid_after_retry counts answers that broke the schema twice"}
    path = common.write_result(COMPONENT, metrics, "logs/llm_calls.jsonl (all runs)",
                               details=details)
    common.write_errors(COMPONENT, worst_calls(fresh))
    common.show(path)
    assert np.isfinite(metrics[0]["value"])


if __name__ == "__main__":
    main()
