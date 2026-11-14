"""Pipeline components with the LLM mocked: normal, empty and invalid-output cases."""

import pytest

from src.pipeline import advisor, judge, relevance, sentiment
from src.pipeline.run import run_anonymized, run_pipeline
from src.schemas import Decision, Features, Headline, LabeledHeadline

FEATURES = Features(
    ret_1d=-0.004, ret_5d=-0.0312, ret_20d=0.052, volatility_20d=0.0161, volatility_level="medium"
)
HEADLINES = [
    Headline(id=1, text="NVIDIA Beats Q1 Estimates", available_date="2019-05-15"),
    Headline(id=2, text="Stocks That Hit 52-Week Highs", available_date="2019-05-15"),
    Headline(id=3, text="Analyst Cuts NVIDIA Price Target", available_date="2019-05-14"),
]
DECISION = Decision(id="A-NVDA-2019-05-15", set="A", ticker="NVDA", company="NVIDIA Corporation",
                    date="2019-05-15", features=FEATURES, headlines=HEADLINES)
RELEVANCE = {"items": [{"id": 1, "relevant": True, "reason": "results"},
                       {"id": 2, "relevant": False, "reason": "list"},
                       {"id": 3, "relevant": True, "reason": "analyst"}]}
SENTIMENT = {"items": [{"id": 1, "label": "positive", "reason": "beat"},
                       {"id": 3, "label": "negative", "reason": "cut"}]}
ADVICE = {"action": "buy", "horizon": "1w", "confidence": 65,
          "explanation": "Results beat estimates.", "used_headlines": [1, 3]}


def test_relevance_normal_empty_invalid(llm) -> None:
    """Relevance: one verdict per headline; no call when empty; error after bad output."""
    llm.queue(RELEVANCE)
    result = relevance.run("NVIDIA Corporation", "NVDA", HEADLINES)
    assert [item.relevant for item in result.items] == [True, False, True]
    assert llm.last_input()["aliases"] == ["NVIDIA", "Nvidia", "NVDA"]
    assert relevance.run("NVIDIA Corporation", "NVDA", []).items == []
    missing = {"items": RELEVANCE["items"][:2]}
    llm.queue(missing, missing)
    failed = relevance.run("NVIDIA Corporation", "NVDA", HEADLINES[::-1])
    assert failed.items == [] and "expected ids [1, 2, 3]" in failed.error


def test_sentiment_normal_empty_invalid(llm) -> None:
    """Sentiment: labels from a fixed set; no call when empty; error after bad output."""
    llm.queue(SENTIMENT)
    result = sentiment.run("NVIDIA Corporation", [HEADLINES[0], HEADLINES[2]])
    assert [item.label for item in result.items] == ["positive", "negative"]
    assert sentiment.run(None, []).items == []
    bad = {"items": [{"id": 1, "label": "bullish", "reason": "x"}]}
    llm.queue(bad, bad)
    failed = sentiment.run(None, [HEADLINES[0]])
    assert failed.items == [] and failed.error.startswith("invalid_output")


def test_advisor_payload_has_percent_features_and_no_ids_beyond_input(llm) -> None:
    """Advisor: Python turns fractions into percent; cited ids must exist in the input."""
    labeled = [LabeledHeadline(id=1, text="NVIDIA Beats Q1 Estimates", sentiment="positive")]
    llm.queue({**ADVICE, "used_headlines": [1]})
    advice = advisor.run(DECISION, labeled)
    assert (advice.action, advice.horizon, advice.confidence) == ("buy", "1w", 65)
    assert advice.news_used and advice.error is None
    assert llm.last_input()["price_features"] == {
        "ret_1d_pct": -0.4, "ret_5d_pct": -3.12, "ret_20d_pct": 5.2,
        "volatility_20d_pct": 1.61, "volatility_level": "medium"}
    assert advisor.run(DECISION, labeled) == advice  # same input: served from the cache
    other = [LabeledHeadline(id=2, text="Another headline", sentiment="neutral")]
    llm.queue(ADVICE, ADVICE)  # cites headlines 1 and 3, which were not given
    failed = advisor.run(DECISION, other)
    assert "unknown ids [1, 3]" in failed.error
    assert (failed.action, failed.confidence, failed.news_used) == ("hold", 0, False)


def test_advisor_without_news(llm) -> None:
    """No relevant news: price-only advice, flagged news_used = false."""
    llm.queue({**ADVICE, "action": "hold", "confidence": 55, "used_headlines": []})
    advice = advisor.run(DECISION, [])
    assert advice.news_used is False and advice.used_headlines == []
    assert llm.last_input()["headlines"] == []


@pytest.mark.parametrize("confidence", [-1, 101, "high"])
def test_advisor_rejects_bad_confidence(llm, confidence: object) -> None:
    """Confidence must be an integer from 0 to 100."""
    bad = {**ADVICE, "confidence": confidence, "used_headlines": []}
    llm.queue(bad, bad)
    assert advisor.run(DECISION, []).error.startswith("invalid_output")


def test_judge_normal_empty_invalid(llm) -> None:
    """Judge: claims with verdicts; no call for an empty explanation; error after bad output."""
    claims = {"claims": [{"text": "Results beat", "supported": True, "reason": "headline 1"},
                         {"text": "Buyback announced", "supported": False, "reason": "absent"}]}
    llm.queue(claims)
    result = judge.run({"company": "NVIDIA"}, "Results beat. Buyback announced.")
    assert [c.supported for c in result.claims] == [True, False]
    assert judge.run({"company": "NVIDIA"}, "  ").claims == []
    llm.queue("{}", "{}")
    assert judge.run({"company": "NVIDIA"}, "Other text.").error.startswith("invalid_output")


def test_full_pipeline_and_trace(llm) -> None:
    """The three calls run in order; only relevant headlines reach sentiment and the advisor."""
    llm.queue(RELEVANCE, SENTIMENT, ADVICE)
    result = run_pipeline(DECISION, trace_id="t-1")
    assert [h.id for h in result.labeled] == [1, 3] and result.errors == []
    assert result.advice.action == "buy" and result.advice.news_used
    sentiment_input, advisor_input = (call[1]["content"] for call in llm.calls[1:])
    assert "52-Week" not in sentiment_input and "52-Week" not in advisor_input
    assert '"sentiment": "negative"' in advisor_input and '"date": "2019-05-15"' in advisor_input


def test_failed_language_step_gives_price_only_advice(llm) -> None:
    """If relevance stays invalid, the advisor still answers from price features only."""
    llm.queue("oops", "oops", {**ADVICE, "used_headlines": []})
    result = run_pipeline(DECISION)
    assert result.errors and result.errors[0].startswith("relevance:")
    assert result.labeled == [] and result.advice.news_used is False


def test_anonymized_run_reuses_labels_and_hides_identity(llm) -> None:
    """Set B: same labels, anonymized texts, no date and no name in the advisor input."""
    llm.queue(RELEVANCE, SENTIMENT, ADVICE)
    source = run_pipeline(DECISION)
    twin = DECISION.model_copy(update={
        "id": "B-NVDA-2019-05-15", "set": "B", "ticker": "X", "company": "Company X",
        "date": None,
        "headlines": [Headline(id=h.id, text=h.text.replace("NVIDIA", "Company X"))
                      for h in HEADLINES]})
    llm.queue(ADVICE)
    result = run_anonymized(twin, source)
    sent = llm.calls[-1][1]["content"]
    assert "NVIDIA" not in sent and "NVDA" not in sent and "2019" not in sent
    assert "Company X Beats Q1 Estimates" in sent and '"date"' not in sent
    assert [h.sentiment for h in result.labeled] == ["positive", "negative"]
