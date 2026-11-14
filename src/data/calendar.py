"""NYSE trading calendar: trading days and real closing times (pure helpers)."""

from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.config import load_config


@lru_cache(maxsize=1)
def schedule() -> pd.DataFrame:
    """Return the exchange schedule: index = trading day, market_close in UTC."""
    cfg = load_config()["market"]
    cal = mcal.get_calendar(cfg["calendar"])
    return cal.schedule(start_date=cfg["calendar_start"], end_date=cfg["calendar_end"])


@lru_cache(maxsize=1)
def _days() -> np.ndarray:
    """Return all trading days as a sorted array of datetime64[D]."""
    return schedule().index.values.astype("datetime64[D]")


@lru_cache(maxsize=1)
def _closes() -> np.ndarray:
    """Return all market-close instants (UTC, naive) as a sorted array."""
    closes = schedule()["market_close"].dt.tz_convert("UTC").dt.tz_localize(None)
    return closes.values.astype("datetime64[s]")


def _key(day: date | str) -> np.datetime64:
    """Convert a date to the array key type."""
    return np.datetime64(pd.Timestamp(day).date(), "D")


def _at(index: int) -> date:
    """Return the trading day stored at an array position."""
    return _days()[index].astype(date)


def is_trading_day(day: date | str) -> bool:
    """Say whether the exchange is open on that day."""
    i = int(np.searchsorted(_days(), _key(day)))
    return i < len(_days()) and _days()[i] == _key(day)


def on_or_after(day: date | str) -> date:
    """Return the first trading day at or after the given day."""
    return _at(int(np.searchsorted(_days(), _key(day), side="left")))


def on_or_before(day: date | str) -> date:
    """Return the last trading day at or before the given day."""
    return _at(int(np.searchsorted(_days(), _key(day), side="right")) - 1)


def next_trading_day(day: date | str) -> date:
    """Return the first trading day strictly after the given day."""
    return _at(int(np.searchsorted(_days(), _key(day), side="right")))


def prev_trading_day(day: date | str) -> date:
    """Return the last trading day strictly before the given day."""
    return _at(int(np.searchsorted(_days(), _key(day), side="left")) - 1)


def shift(day: date | str, sessions: int) -> date:
    """Return the trading day `sessions` sessions after a trading day (negative = before)."""
    if not is_trading_day(day):
        raise ValueError(f"{day} is not a trading day")
    return _at(int(np.searchsorted(_days(), _key(day))) + sessions)


def trading_days(start: date | str, end: date | str) -> list[date]:
    """Return every trading day in [start, end]."""
    lo = int(np.searchsorted(_days(), _key(start), side="left"))
    hi = int(np.searchsorted(_days(), _key(end), side="right"))
    return [d.astype(date) for d in _days()[lo:hi]]


def close_utc(day: date | str) -> pd.Timestamp:
    """Return the real market close of a trading day as a UTC timestamp."""
    if not is_trading_day(day):
        raise ValueError(f"{day} is not a trading day")
    i = int(np.searchsorted(_days(), _key(day)))
    return pd.Timestamp(_closes()[i], tz="UTC")


def first_close_at_or_after(ts_utc: pd.Timestamp) -> date:
    """Return the first trading day whose close is at or after a UTC instant."""
    naive = np.datetime64(ts_utc.tz_convert("UTC").tz_localize(None).floor("s"), "s")
    return _at(int(np.searchsorted(_closes(), naive, side="left")))
