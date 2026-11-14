"""Shared pydantic models passed between components."""

from typing import Literal

from pydantic import BaseModel

Sentiment = Literal["positive", "negative", "neutral"]
Action = Literal["buy", "hold", "sell"]
Horizon = Literal["1d", "1w"]
VolLevel = Literal["low", "medium", "high"]


class Headline(BaseModel):
    """One news headline as the pipeline sees it."""

    id: int
    text: str
    published: str | None = None  # ISO date (undated source) or ISO UTC datetime; None in set B
    available_date: str | None = None  # first trading day on which it may be used
    source: str | None = None


class Features(BaseModel):
    """Price trend numbers computed in Python (fractions: 0.012 means 1.2%)."""

    ret_1d: float
    ret_5d: float
    ret_20d: float
    volatility_20d: float
    volatility_level: VolLevel


class Decision(BaseModel):
    """Everything known at decision time for one ticker (the collector's output)."""

    id: str
    set: str
    ticker: str
    company: str
    date: str | None  # decision day (close of that day); None in set B
    as_of: str | None = None  # UTC timestamp of a live request
    features: Features
    headlines: list[Headline]


class LabeledHeadline(BaseModel):
    """A relevant headline with its sentiment label (the advisor's news input)."""

    id: int
    text: str
    sentiment: Sentiment
