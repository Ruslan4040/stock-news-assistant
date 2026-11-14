"""Eval of the relevance filter (component 3): precision, recall, F1 on labeled fresh headlines.

Run: python -m evals.eval_relevance [--limit N]
Baselines: keep all headlines; keyword match (the headline names the company).
"""

import numpy as np
import pandas as pd

from evals import common
from src.config import ROOT, load_config
from src.data.terms import mentions_company
from src.pipeline import relevance
from src.schemas import Headline

COMPONENT = "relevance"


def load_labels(limit: int | None) -> pd.DataFrame:
    """Load the labeled fresh headlines (only rows that carry a label)."""
    path = ROOT / load_config()["paths"]["labels_dir"] / "fresh_headlines.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame[frame["relevant"].isin(["y", "n"])].head(limit).reset_index(drop=True)
    frame["truth"] = frame["relevant"] == "y"
    return frame


def predict(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the filter once per ticker; rows of a failed call get no prediction (error)."""
    frame = frame.assign(predicted=False, reason="", error="")
    for ticker, part in frame.groupby("ticker", sort=False):
        headlines = [Headline(id=i + 1, text=text) for i, text in enumerate(part["headline"])]
        result = relevance.run(part["company"].iloc[0], ticker, headlines)
        if result.error:
            frame.loc[part.index, "error"] = result.error
            continue
        frame.loc[part.index, "predicted"] = [item.relevant for item in result.items]
        frame.loc[part.index, "reason"] = [item.reason for item in result.items]
    return frame


def precision(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Return the share of kept headlines that are truly relevant."""
    kept = predicted.sum()
    return float((truth & predicted).sum() / kept) if kept else float("nan")


def recall(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Return the share of relevant headlines that were kept."""
    relevant = truth.sum()
    return float((truth & predicted).sum() / relevant) if relevant else float("nan")


def f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Return the harmonic mean of precision and recall."""
    p, r = precision(truth, predicted), recall(truth, predicted)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def confusion(truth: np.ndarray, predicted: np.ndarray) -> dict:
    """Return the four cells of the confusion matrix."""
    return {
        "tp": int((truth & predicted).sum()),
        "fp": int((~truth & predicted).sum()),
        "fn": int((truth & ~predicted).sum()),
        "tn": int((~truth & ~predicted).sum()),
    }


def errors_table(frame: pd.DataFrame) -> list[dict]:
    """Return the misclassified headlines: lost relevant news first, then noise let through."""
    wrong = frame[frame["truth"] != frame["predicted"]].copy()
    wrong["type"] = np.where(wrong["truth"], "false_negative", "false_positive")
    wrong = wrong.sort_values("type")
    columns = ["id", "ticker", "type", "headline", "reason"]
    return wrong[columns].to_dict("records")


def main() -> None:
    """Run the filter and the baselines on the labeled headlines and write the results."""
    limit = common.limit_argument(__doc__)
    frame = predict(load_labels(limit))
    truth = frame["truth"].to_numpy()
    systems = {
        "llm": frame["predicted"].to_numpy(),
        "keep_all": np.ones(len(frame), dtype=bool),
        "keyword": np.array([mentions_company(t, h) for t, h in
                             zip(frame["ticker"], frame["headline"], strict=True)]),
    }
    metrics = []
    for name, function in (("precision", precision), ("recall", recall), ("f1", f1)):
        baselines = {b: round(function(truth, systems[b]), 4) for b in ("keep_all", "keyword")}
        metrics.append(common.metric_row(name, common.bootstrap(function, truth, systems["llm"]),
                                         baselines))
    details = {
        "confusion": confusion(truth, systems["llm"]),
        "llm_errors": int((frame["error"] != "").sum()),
        "labeled_by": frame["labeled_by"].value_counts().to_dict(),
        "relevant_share": round(float(truth.mean()), 4),
    }
    version = load_config()["prompts"][COMPONENT]
    path = common.write_result(COMPONENT, metrics, "evals/labels/fresh_headlines.csv", version,
                               limit, details)
    common.write_errors(COMPONENT, errors_table(frame), limit)
    common.show(path)


if __name__ == "__main__":
    main()
