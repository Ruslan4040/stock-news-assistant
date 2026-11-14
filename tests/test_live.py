"""Live rule: one advice per ticker per day, a new one only for a new relevant headline."""

from datetime import UTC, datetime

import pytest

from src.pipeline import live
from src.schemas import Decision, Features, Headline

FEATURES = Features(
    ret_1d=0.01, ret_5d=0.02, ret_20d=0.03, volatility_20d=0.015, volatility_level="medium"
)
NOW = datetime(2026, 11, 14, 15, 0, tzinfo=UTC)
ADVICE = {"action": "hold", "horizon": "1d", "confidence": 55, "explanation": "Mixed signals.",
          "used_headlines": [1]}


def decision(texts: list[str]) -> Decision:
    """Build a live decision with the given headline texts."""
    headlines = [Headline(id=i + 1, text=text) for i, text in enumerate(texts)]
    return Decision(id="live-HD-x", set="live", ticker="HD", company="The Home Depot, Inc.",
                    date="2026-11-14", as_of="2026-11-14T15:00:00Z", features=FEATURES,
                    headlines=headlines)


def relevance(flags: list[bool]) -> dict:
    """Script a relevance answer."""
    return {"items": [{"id": i + 1, "relevant": flag, "reason": "r"}
                      for i, flag in enumerate(flags)]}


def sentiment(ids: list[int]) -> dict:
    """Script a sentiment answer."""
    return {"items": [{"id": i, "label": "neutral", "reason": "r"} for i in ids]}


def test_live_rule(llm, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """First call advises; the same news returns the stored advice; new relevant news re-advises."""
    monkeypatch.setattr(live, "store_path", lambda: tmp_path / "advice.jsonl")
    texts = ["Home Depot Raises Guidance", "Market Wrap"]
    monkeypatch.setattr(live, "collect_live", lambda ticker, now: decision(texts))
    llm.queue(relevance([True, False]), sentiment([1]), ADVICE)
    first, created = live.live_advice("HD", NOW)
    assert created and first.reason == "first_of_day" and len(live.load_store()) == 1

    again, created = live.live_advice("HD", NOW)  # same headlines: no LLM call is scripted
    assert not created and again.as_of == first.as_of and len(live.load_store()) == 1

    texts.insert(0, "Fed Holds Rates")  # a new but irrelevant headline
    llm.queue(relevance([False, True, False]))
    _, created = live.live_advice("HD", NOW)
    assert not created and len(live.load_store()) == 1

    texts.insert(0, "Home Depot To Buy Supplier")  # a new relevant headline
    llm.queue(relevance([True, False, True, False]), sentiment([1, 3]), ADVICE)
    extra, created = live.live_advice("HD", NOW)
    assert created and extra.reason == "new_relevant_headline" and len(live.load_store()) == 2

    other_day = datetime(2026, 11, 15, 15, 0, tzinfo=UTC)
    advice = {**ADVICE, "confidence": 60, "used_headlines": [1, 3]}
    llm.queue(advice)  # relevance and sentiment come from the cache; only the advisor is new
    monkeypatch.setattr(live, "collect_live",
                        lambda ticker, now: decision(texts).model_copy(update={"date": "x"}))
    _, created = live.live_advice("HD", other_day)
    assert created and len(live.load_store()) == 3
