"""Advisor (component 5, LLM call #3): one explained, calibrated suggestion."""

from pydantic import BaseModel, Field

from src.config import load_config
from src.llm.client import Validator, call_llm
from src.schemas import Action, Decision, Features, Horizon, LabeledHeadline

COMPONENT = "advisor"
PERCENT_DIGITS = 2  # decimals shown to the LLM


class AdviceOutput(BaseModel):
    """What the LLM must return."""

    action: Action
    horizon: Horizon
    confidence: int = Field(ge=0, le=100)
    explanation: str = Field(min_length=1)
    used_headlines: list[int]


class Advice(BaseModel):
    """The component's result. On error it is a zero-confidence hold that never trades."""

    action: Action = "hold"
    horizon: Horizon = "1d"
    confidence: int = 0
    explanation: str = ""
    used_headlines: list[int] = []
    news_used: bool = False
    error: str | None = None


def features_in_percent(features: Features) -> dict:
    """Convert fractions to percent in Python, so the LLM never does arithmetic."""
    values = features.model_dump()
    level = values.pop("volatility_level")
    shown = {f"{name}_pct": round(value * 100, PERCENT_DIGITS) for name, value in values.items()}
    return {**shown, "volatility_level": level}


def build_payload(decision: Decision, labeled: list[LabeledHeadline]) -> dict:
    """Build the LLM input; the date is left out when the decision has none (set B)."""
    payload = {"company": decision.company, "ticker": decision.ticker}
    if decision.date:
        payload["date"] = decision.date
    payload["price_features"] = features_in_percent(decision.features)
    payload["headlines"] = [h.model_dump() for h in labeled]
    return payload


def known_ids(allowed: list[int]) -> Validator:
    """Build the rule 'used_headlines contains only ids from the input'."""

    def check(output: BaseModel) -> str | None:
        """Return a message when an unknown headline id is cited."""
        unknown = sorted(set(output.used_headlines) - set(allowed))
        return f"used_headlines has unknown ids {unknown}" if unknown else None

    return check


def run(
    decision: Decision,
    labeled: list[LabeledHeadline],
    trace_id: str | None = None,
    version: str | None = None,
) -> Advice:
    """Combine price features and labeled relevant headlines into one suggestion.

    `version` overrides the prompt version from config (used to compare prompt versions).
    """
    version = version or load_config()["prompts"][COMPONENT]
    payload = build_payload(decision, labeled)
    rule = known_ids([h.id for h in labeled])
    result = call_llm(COMPONENT, version, payload, AdviceOutput, rule, trace_id)
    if result.data is None:
        return Advice(error=result.error)
    return Advice(**result.data.model_dump(), news_used=bool(labeled))
