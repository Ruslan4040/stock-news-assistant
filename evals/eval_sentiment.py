"""Eval of the sentiment labeler (component 4): accuracy, macro-F1, confusion matrix.

Run: python -m evals.eval_sentiment [--limit N]
Data: the PhraseBank sample and the labeled fresh headlines.
Baselines: FinBERT (local model) and 'always neutral'.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from evals import common
from src.config import ROOT, load_config
from src.data import sets
from src.pipeline import sentiment
from src.pipeline.sentiment_baselines import always_neutral, finbert_labels
from src.schemas import Headline

COMPONENT = "sentiment"
LABELS = ["negative", "neutral", "positive"]
BASELINES = ("finbert", "always_neutral")
OPPOSITE = {("positive", "negative"), ("negative", "positive")}


def label_groups(frame: pd.DataFrame, company_column: str | None) -> pd.DataFrame:
    """Run the labeler group by group (in parallel); rows of a failed call are marked 'error'."""
    frame = frame.assign(predicted="error", reason="")
    parts = [part for _key, part in frame.groupby("group", sort=False)]

    def label(part: pd.DataFrame) -> sentiment.SentimentResult:
        """Label the texts of one group with one LLM call."""
        company = part[company_column].iloc[0] if company_column else None
        headlines = [Headline(id=i + 1, text=text) for i, text in enumerate(part["text"])]
        return sentiment.run(company, headlines)

    for part, result in zip(parts, common.parallel_map(label, parts), strict=True):
        if not result.error:
            frame.loc[part.index, "predicted"] = [item.label for item in result.items]
            frame.loc[part.index, "reason"] = [item.reason for item in result.items]
    return frame


def phrasebank_frame(limit: int | None) -> pd.DataFrame:
    """Load the PhraseBank sample and split it into fixed batches."""
    frame = pd.read_csv(sets.processed("phrasebank_sample.csv")).head(limit)
    frame = frame.rename(columns={"sentence": "text", "label": "truth"})
    frame["group"] = np.arange(len(frame)) // load_config()["evals"]["sentiment_batch"]
    return frame


def fresh_frame(limit: int | None) -> pd.DataFrame:
    """Load the labeled fresh headlines, grouped by ticker."""
    path = ROOT / load_config()["paths"]["labels_dir"] / "fresh_headlines.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame[frame["sentiment"].isin(LABELS)].head(limit).reset_index(drop=True)
    return frame.rename(columns={"headline": "text", "sentiment": "truth", "ticker": "group"})


def accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Return the share of exact label matches."""
    return float((truth == predicted).mean())


def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Return the unweighted mean F1 over the three classes."""
    return float(f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0))


def score(name: str, frame: pd.DataFrame) -> tuple[list[dict], dict]:
    """Score the LLM and both baselines on one dataset; return metric rows and details."""
    truth = frame["truth"].to_numpy()
    systems = {
        "llm": frame["predicted"].to_numpy(),
        "finbert": np.array(finbert_labels(frame["text"].tolist())),
        "always_neutral": np.array(always_neutral(frame["text"].tolist())),
    }
    rows = []
    for metric_name, function in (("accuracy", accuracy), ("macro_f1", macro_f1)):
        baselines = {b: round(function(truth, systems[b]), 4) for b in BASELINES}
        stat = common.bootstrap(function, truth, systems["llm"])
        rows.append(common.metric_row(f"{name}.{metric_name}", stat, baselines))
    matrix = confusion_matrix(truth, systems["llm"], labels=[*LABELS, "error"])[:3]
    details = {
        "confusion_rows_truth_cols_predicted": {"labels": [*LABELS, "error"],
                                                "matrix": matrix.tolist()},
        "class_balance": pd.Series(truth).value_counts().to_dict(),
        "llm_errors": int((systems["llm"] == "error").sum()),
    }
    return rows, details


def errors_table(name: str, frame: pd.DataFrame) -> list[dict]:
    """Return wrong labels, opposite-polarity mistakes first."""
    wrong = frame[frame["truth"] != frame["predicted"]].copy()
    pairs = zip(wrong["truth"], wrong["predicted"], strict=True)
    wrong["severity"] = [2 if pair in OPPOSITE else 1 for pair in pairs]
    wrong = wrong.sort_values("severity", ascending=False).assign(dataset=name)
    return wrong[["dataset", "truth", "predicted", "text", "reason"]].to_dict("records")


def main() -> None:
    """Evaluate on PhraseBank and on the fresh headlines; write results and the gap."""
    limit = common.limit_argument(__doc__)
    phrase = label_groups(phrasebank_frame(limit), None)
    fresh = label_groups(fresh_frame(limit), "company")
    relevant = fresh[fresh["relevant"] == "y"]
    heldout = phrase[phrase["id"] > load_config()["evals"]["sentiment_dev_n"]]
    metrics, details, errors = [], {}, []
    datasets = (("phrasebank", phrase), ("phrasebank_heldout", heldout), ("fresh", fresh),
                ("fresh_relevant", relevant))
    for name, frame in datasets:
        if frame.empty:
            continue
        rows, info = score(name, frame)
        metrics += rows
        details[name] = info
        if name in ("phrasebank", "fresh"):
            errors += errors_table(name, frame)[: common.WORST_ERRORS // 2]
    by_name = {row["name"]: row for row in metrics}
    details["sentiment_gap_accuracy"] = {
        "llm": by_name["phrasebank.accuracy"]["value"] - by_name["fresh.accuracy"]["value"],
        "finbert": by_name["phrasebank.accuracy"]["baselines"]["finbert"]
        - by_name["fresh.accuracy"]["baselines"]["finbert"],
    }
    details["labeled_by"] = fresh["labeled_by"].value_counts().to_dict()
    version = load_config()["prompts"][COMPONENT]
    data = "data/processed/phrasebank_sample.csv + evals/labels/fresh_headlines.csv"
    path = common.write_result(COMPONENT, metrics, data, version, limit, details)
    common.write_errors(COMPONENT, errors, limit)
    common.show(path)


if __name__ == "__main__":
    main()
