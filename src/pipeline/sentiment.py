"""Sentiment labeler (component 4, LLM call #2): is each headline good or bad for the stock."""

from pydantic import BaseModel

from src.config import load_config
from src.llm.client import call_llm
from src.pipeline.relevance import same_ids
from src.schemas import Headline, Sentiment

COMPONENT = "sentiment"


class SentimentItem(BaseModel):
    """The label for one headline."""

    id: int
    label: Sentiment
    reason: str


class SentimentOutput(BaseModel):
    """What the LLM must return."""

    items: list[SentimentItem]


class SentimentResult(BaseModel):
    """The component's result; `error` is set when the LLM output stayed invalid."""

    items: list[SentimentItem] = []
    error: str | None = None


def build_payload(company: str | None, headlines: list[Headline]) -> dict:
    """Build the LLM input: the target company (or null) and the numbered texts."""
    return {"company": company, "headlines": [{"id": h.id, "text": h.text} for h in headlines]}


def run(
    company: str | None, headlines: list[Headline], trace_id: str | None = None
) -> SentimentResult:
    """Label each headline as positive, negative or neutral for the company's stock."""
    if not headlines:
        return SentimentResult()
    version = load_config()["prompts"][COMPONENT]
    payload = build_payload(company, headlines)
    rule = same_ids([h.id for h in headlines])
    result = call_llm(COMPONENT, version, payload, SentimentOutput, rule, trace_id)
    if result.data is None:
        return SentimentResult(error=result.error)
    return SentimentResult(items=sorted(result.data.items, key=lambda item: item.id))
