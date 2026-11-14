"""Sentiment baselines (no LLM): FinBERT as a local classifier, and 'always neutral'."""

from functools import lru_cache

from src.config import load_config

FINBERT_BATCH = 16  # sentences per forward pass; only affects speed


@lru_cache(maxsize=1)
def _finbert() -> object:
    """Load the FinBERT text classifier once (downloads the model on first use)."""
    from transformers import pipeline

    return pipeline("text-classification", model=load_config()["evals"]["finbert_model"])


def finbert_labels(texts: list[str]) -> list[str]:
    """Label each text with FinBERT: positive, negative or neutral."""
    if not texts:
        return []
    outputs = _finbert()(texts, batch_size=FINBERT_BATCH, truncation=True)
    return [output["label"].lower() for output in outputs]


def always_neutral(texts: list[str]) -> list[str]:
    """Label every text as neutral (the majority-class baseline)."""
    return ["neutral"] * len(texts)
