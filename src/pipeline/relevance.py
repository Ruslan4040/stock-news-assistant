"""Relevance filter (component 3, LLM call #1): keep only headlines about the company."""

from pydantic import BaseModel

from src.config import load_config
from src.llm.client import Validator, call_llm
from src.schemas import Headline

COMPONENT = "relevance"


class RelevanceItem(BaseModel):
    """The verdict for one headline."""

    id: int
    relevant: bool
    reason: str


class RelevanceOutput(BaseModel):
    """What the LLM must return."""

    items: list[RelevanceItem]


class RelevanceResult(BaseModel):
    """The component's result; `error` is set when the LLM output stayed invalid."""

    items: list[RelevanceItem] = []
    error: str | None = None


def same_ids(expected: list[int]) -> Validator:
    """Build the rule 'exactly one item per input id'."""

    def check(output: BaseModel) -> str | None:
        """Return a message when the ids differ from the input ids."""
        got = sorted(item.id for item in output.items)
        return None if got == sorted(expected) else f"expected ids {sorted(expected)}, got {got}"

    return check


def build_payload(company: str, ticker: str, headlines: list[Headline]) -> dict:
    """Build the LLM input: the company, its aliases and the numbered headlines."""
    aliases = load_config()["tickers"].get(ticker, {}).get("aliases", [])
    texts = [{"id": h.id, "text": h.text} for h in headlines]
    return {"company": company, "ticker": ticker, "aliases": aliases, "headlines": texts}


def run(
    company: str, ticker: str, headlines: list[Headline], trace_id: str | None = None
) -> RelevanceResult:
    """Say for each headline whether it is relevant for the company."""
    if not headlines:
        return RelevanceResult()
    version = load_config()["prompts"][COMPONENT]
    payload = build_payload(company, ticker, headlines)
    rule = same_ids([h.id for h in headlines])
    result = call_llm(COMPONENT, version, payload, RelevanceOutput, rule, trace_id)
    if result.data is None:
        return RelevanceResult(error=result.error)
    return RelevanceResult(items=sorted(result.data.items, key=lambda item: item.id))
