"""Live mode: one advice per ticker per day, plus a new one when a new relevant headline appears.

Every live advice is appended to data/processed/live/advice.jsonl. These records are set D
of the look-ahead experiment.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from src.config import ROOT, load_config
from src.data import news
from src.data.collector import collect_live
from src.llm.trace import new_trace_id
from src.pipeline.run import (
    PipelineResult,
    advise,
    filter_news,
    label_sentiment,
    relevant_headlines,
)


class LiveAdvice(BaseModel):
    """One stored live advice and why it was produced."""

    ticker: str
    day: str  # UTC date of the request
    as_of: str
    reason: str  # first_of_day / new_relevant_headline
    relevant_keys: list[str]
    result: PipelineResult


def store_path() -> Path:
    """Return the file that holds every live advice."""
    return ROOT / load_config()["paths"]["live_dir"] / "advice.jsonl"


def load_store() -> list[LiveAdvice]:
    """Read every stored live advice."""
    path = store_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [LiveAdvice.model_validate_json(line) for line in lines if line.strip()]


def _append(record: LiveAdvice) -> None:
    """Append one advice to the store."""
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def relevant_keys(partial: PipelineResult) -> list[str]:
    """Return the normalized texts of the headlines that passed the relevance filter."""
    return sorted(news.normalize(h.text) for h in relevant_headlines(partial))


def live_advice(ticker: str, now: datetime | None = None) -> tuple[LiveAdvice, bool]:
    """Return today's advice for a ticker and whether it was produced by this call."""
    now = now or datetime.now(UTC)
    day = now.date().isoformat()
    today = [a for a in load_store() if a.ticker == ticker and a.day == day]
    partial = filter_news(collect_live(ticker, now), new_trace_id())
    keys = relevant_keys(partial)
    known = {key for advice in today for key in advice.relevant_keys}
    if today and (partial.errors or set(keys) <= known):
        return today[-1], False  # nothing new (or the filter failed): keep today's advice
    reason = "new_relevant_headline" if today else "first_of_day"
    result = advise(label_sentiment(partial))
    record = LiveAdvice(ticker=ticker, day=day, as_of=partial.decision.as_of, reason=reason,
                        relevant_keys=keys, result=result)
    _append(record)
    return record, True
