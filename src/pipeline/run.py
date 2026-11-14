"""Pipeline orchestrator: collector output -> relevance -> sentiment -> advisor."""

from pydantic import BaseModel

from src.llm.trace import new_trace_id
from src.pipeline import advisor, relevance, sentiment
from src.pipeline.advisor import Advice
from src.pipeline.relevance import RelevanceItem
from src.pipeline.sentiment import SentimentItem
from src.schemas import Decision, Headline, LabeledHeadline


class PipelineResult(BaseModel):
    """Everything one request produced, step by step."""

    decision: Decision
    relevance: list[RelevanceItem] = []
    sentiment: list[SentimentItem] = []
    labeled: list[LabeledHeadline] = []
    advice: Advice
    errors: list[str] = []
    trace_id: str


def relevant_headlines(result: PipelineResult) -> list[Headline]:
    """Return the headlines that the relevance filter kept."""
    keep = {item.id for item in result.relevance if item.relevant}
    return [h for h in result.decision.headlines if h.id in keep]


def filter_news(decision: Decision, trace_id: str) -> PipelineResult:
    """Language step 1: the relevance filter. A failure leaves the advisor with no news."""
    partial = PipelineResult(decision=decision, advice=Advice(), trace_id=trace_id)
    filtered = relevance.run(decision.company, decision.ticker, decision.headlines, trace_id)
    if filtered.error:
        partial.errors.append(f"relevance: {filtered.error}")
    partial.relevance = filtered.items
    return partial


def label_sentiment(partial: PipelineResult) -> PipelineResult:
    """Language step 2: sentiment of the relevant headlines."""
    relevant = relevant_headlines(partial)
    labels = sentiment.run(partial.decision.company, relevant, partial.trace_id)
    if labels.error:
        partial.errors.append(f"sentiment: {labels.error}")
        return partial
    partial.sentiment = labels.items
    by_id = {item.id: item.label for item in labels.items}
    partial.labeled = [
        LabeledHeadline(id=h.id, text=h.text, sentiment=by_id[h.id]) for h in relevant
    ]
    return partial


def label_news(decision: Decision, trace_id: str) -> PipelineResult:
    """Run the two language steps; a failed step leaves the advisor with no news."""
    return label_sentiment(filter_news(decision, trace_id))


def advise(result: PipelineResult) -> PipelineResult:
    """Run the advisor on the labeled news of a partial result."""
    result.advice = advisor.run(result.decision, result.labeled, result.trace_id)
    if result.advice.error:
        result.errors.append(f"advisor: {result.advice.error}")
    return result


def run_pipeline(decision: Decision, trace_id: str | None = None) -> PipelineResult:
    """Run the full pipeline for one decision."""
    return advise(label_news(decision, trace_id or new_trace_id()))


def run_anonymized(twin: Decision, source: PipelineResult) -> PipelineResult:
    """Set B: reuse the labels of the set-A twin and give the advisor the anonymized texts."""
    text = {h.id: h.text for h in twin.headlines}
    labeled = [
        LabeledHeadline(id=h.id, text=text[h.id], sentiment=h.sentiment) for h in source.labeled
    ]
    inherited = [error for error in source.errors if not error.startswith("advisor")]
    result = PipelineResult(
        decision=twin, relevance=source.relevance, sentiment=source.sentiment, labeled=labeled,
        advice=Advice(), errors=inherited, trace_id=new_trace_id(),
    )
    return advise(result)
