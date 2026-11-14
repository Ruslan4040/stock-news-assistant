"""Eval of the advisor (component 5) on sets A, B, C: direction, horizon choice, confidence.

Run: python -m evals.eval_advisor [--limit N]
Reads the stored pipeline runs (python -m src.pipeline.batch) and the outcomes file.
Baselines on the same rows: 'always buy' and 'same as last move'; fixed 1d and fixed 1w.
"""

import numpy as np
import pandas as pd

from evals import common
from src.config import load_config
from src.data import sets
from src.pipeline import batch

COMPONENT = "advisor"
SETS = "ABC"


def load_rows(letter: str, limit: int | None = None) -> pd.DataFrame:
    """Return one row per stored decision of a set: the advice and the true forward returns."""
    outcomes = sets.load_outcomes()
    rows = []
    for result in batch.load_results(letter)[:limit]:
        truth = outcomes.loc[sets.real_id(result.decision.id)]
        advice = result.advice
        rows.append({
            "set": letter, "id": result.decision.id, "ticker": truth["ticker"],
            "date": truth["date"], "action": advice.action, "horizon": advice.horizon,
            "confidence": advice.confidence, "news_used": advice.news_used,
            "incomplete": batch.incomplete(result), "error": advice.error or "",
            "last_move": result.decision.features.ret_1d,
            "fwd_1d": truth["fwd_ret_1d"], "fwd_1w": truth["fwd_ret_1w"],
            "explanation": advice.explanation,
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["fwd"] = np.where(frame["horizon"] == "1d", frame["fwd_1d"], frame["fwd_1w"])
        frame["side"] = frame["action"].map({"buy": 1, "sell": -1, "hold": 0})
    return frame


def directional(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the valid buy and sell advice: the rows a direction metric can score."""
    valid = frame[(~frame["incomplete"]) & (frame["error"] == "")]
    return valid[valid["side"] != 0]


def direction_rows(letter: str, calls: pd.DataFrame, valid_n: int) -> list[dict]:
    """Hit rate of the advised direction, with both baselines on the same rows."""
    side, fwd = calls["side"].to_numpy(), calls["fwd"].to_numpy()
    hit = (side * fwd > 0).astype(float)
    baselines = {
        "always_buy": round(float((fwd > 0).mean()), 4) if len(fwd) else None,
        "same_as_last_move": round(float((calls["last_move"].to_numpy() * fwd > 0).mean()), 4)
        if len(fwd) else None,
    }
    coverage = {"value": len(calls) / valid_n if valid_n else None, "ci_low": None,
                "ci_high": None, "n": valid_n}
    return [
        common.metric_row(f"{letter}.hit_rate", common.bootstrap(common.mean, hit), baselines),
        common.metric_row(f"{letter}.directional_share", coverage),
    ]


def horizon_rows(letter: str, calls: pd.DataFrame) -> list[dict]:
    """Mean return in the advised direction: chosen horizon vs fixed 1d and fixed 1w."""
    side = calls["side"].to_numpy()
    chosen = side * calls["fwd"].to_numpy() * 100
    baselines = {
        "fixed_1d": round(float((side * calls["fwd_1d"].to_numpy() * 100).mean()), 4)
        if len(calls) else None,
        "fixed_1w": round(float((side * calls["fwd_1w"].to_numpy() * 100).mean()), 4)
        if len(calls) else None,
    }
    stat = common.bootstrap(common.mean, chosen)
    return [common.metric_row(f"{letter}.mean_return_chosen_horizon_pct", stat, baselines)]


def confidence_rows(letter: str, calls: pd.DataFrame) -> list[dict]:
    """Hit rate per confidence bucket: it should rise with confidence."""
    buckets = load_config()["evals"]["confidence_buckets"]
    hit = (calls["side"] * calls["fwd"] > 0).astype(float)
    rows = []
    for low, high in buckets:
        top = calls["confidence"] <= high if high == buckets[-1][1] else calls["confidence"] < high
        inside = hit[(calls["confidence"] >= low) & top].to_numpy()
        name = f"{letter}.hit_rate_confidence_{low}_{high}"
        rows.append(common.metric_row(name, common.bootstrap(common.mean, inside)))
    return rows


def describe(frame: pd.DataFrame, calls: pd.DataFrame) -> dict:
    """Counts that explain the metrics: actions, horizons, errors, news use."""
    valid = frame[(~frame["incomplete"]) & (frame["error"] == "")]
    hit = calls["side"] * calls["fwd"] > 0
    return {
        "decisions": len(frame),
        "valid": len(valid),
        "incomplete_transport": int(frame["incomplete"].sum()),
        "invalid_output": int(((~frame["incomplete"]) & (frame["error"] != "")).sum()),
        "actions": valid["action"].value_counts().to_dict(),
        "horizons": valid["horizon"].value_counts().to_dict(),
        "news_used_share": round(float(valid["news_used"].mean()), 4) if len(valid) else None,
        "mean_confidence": round(float(valid["confidence"].mean()), 2) if len(valid) else None,
        "hit_rate_by_action": hit.groupby(calls["action"]).mean().round(4).to_dict(),
        "hit_rate_by_horizon": hit.groupby(calls["horizon"]).mean().round(4).to_dict(),
        "hit_rate_by_news_used": hit.groupby(calls["news_used"]).mean().round(4).to_dict(),
    }


def worst_errors(frames: list[pd.DataFrame]) -> list[dict]:
    """Return the most confident wrong calls, largest adverse move first within a confidence."""
    calls = pd.concat([directional(f) for f in frames if len(f)])
    wrong = calls[calls["side"] * calls["fwd"] <= 0].copy()
    wrong["return_against_pct"] = (-wrong["side"] * wrong["fwd"] * 100).round(2)
    wrong = wrong.sort_values(["confidence", "return_against_pct"], ascending=False)
    wrong["explanation"] = wrong["explanation"].str[:300]
    columns = ["set", "id", "action", "horizon", "confidence", "return_against_pct", "explanation"]
    return wrong[columns].to_dict("records")


def main() -> None:
    """Score every set that has stored runs and write the results file."""
    limit = common.limit_argument(__doc__)
    metrics, details, frames = [], {}, []
    for letter in SETS:
        frame = load_rows(letter, limit)
        if not len(frame):
            continue
        calls = directional(frame)
        valid_n = int(((~frame["incomplete"]) & (frame["error"] == "")).sum())
        metrics += direction_rows(letter, calls, valid_n)
        metrics += horizon_rows(letter, calls) + confidence_rows(letter, calls)
        details[letter] = describe(frame, calls)
        frames.append(frame)
    version = load_config()["prompts"][COMPONENT]
    data = "data/processed/runs/advice_{a,b,c}.jsonl + data/processed/outcomes.csv"
    path = common.write_result(COMPONENT, metrics, data, version, limit, details)
    common.write_errors(COMPONENT, worst_errors(frames), limit)
    common.show(path)


if __name__ == "__main__":
    main()
