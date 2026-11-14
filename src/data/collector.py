"""Data collector (component 1): only the prices and headlines available at decision time."""

from datetime import UTC, date, datetime
from typing import Literal

import pandas as pd

from src.config import load_config
from src.data import align, finnhub, news, prices
from src.data import calendar as cal
from src.data.features import compute_features
from src.schemas import Decision

Mode = Literal["historical", "live"]


def company_name(ticker: str) -> str:
    """Return the company name of a config ticker."""
    return load_config()["tickers"][ticker]["name"]


def decision_day(requested: date) -> date:
    """Snap a requested date to the last trading day at or before it, inside the price data."""
    last = pd.Timestamp(load_config()["dates"]["prices_end"]).date()
    return cal.on_or_before(min(requested, last))


def check_no_lookahead(decision: Decision) -> None:
    """Raise if any headline of a historical decision became available after its day."""
    late = [h.id for h in decision.headlines if not align.is_usable(h.available_date, decision.date)]
    if late:
        raise ValueError(f"Look-ahead in {decision.id}: headlines {late} are after {decision.date}")


def collect_historical(
    ticker: str, requested: date, set_name: str = "hist", fetch_to: str | None = None
) -> Decision:
    """Collect prices up to the close of the decision day and the headlines usable then."""
    day = decision_day(requested)
    closes = prices.load_prices(ticker)["adj_close"]
    decision = Decision(
        id=f"{set_name}-{ticker}-{day.isoformat()}",
        set=set_name,
        ticker=ticker,
        company=company_name(ticker),
        date=day.isoformat(),
        features=compute_features(closes, day),
        headlines=news.headlines_for(ticker, day, fetch_to),
    )
    check_no_lookahead(decision)
    return decision


def collect_live(ticker: str, now: datetime | None = None) -> Decision:
    """Collect the latest close and the headlines of the last days up to now."""
    now = now or datetime.now(UTC)
    frame = prices.live_prices(ticker)
    last_close = frame.index[-1].date()
    headlines, _source = news.live_headlines(ticker, now)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return Decision(
        id=f"live-{ticker}-{stamp}",
        set="live",
        ticker=ticker,
        company=company_name(ticker),
        date=now.date().isoformat(),
        as_of=stamp,
        features=compute_features(frame["adj_close"], last_close),
        headlines=headlines,
    )


def collect(ticker: str, requested: date | None = None, mode: Mode = "historical") -> Decision:
    """Collect the inputs of one request; the app may fetch missing news days on demand."""
    if mode == "live":
        return collect_live(ticker)
    if requested is None:
        raise ValueError("Historical mode needs a date")
    return collect_historical(ticker, requested, fetch_to=finnhub.cache_dir())
