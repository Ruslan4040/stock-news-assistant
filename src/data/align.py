"""Time alignment: on which trading day does a headline become usable (no look-ahead)."""

import re
from datetime import date

import pandas as pd

from src.data import calendar as cal

_OFFSET = re.compile(r"[+-]\d{2}:?\d{2}$|Z$")


def has_time_of_day(raw: str) -> bool:
    """Say whether a raw date string carries a real time (a UTC offset is present)."""
    return bool(_OFFSET.search(raw.strip()))


def available_for_timestamp(ts_utc: pd.Timestamp) -> date:
    """Return the trading day a timed headline belongs to (after the close -> next day)."""
    return cal.first_close_at_or_after(ts_utc)


def available_for_day(day: date | str) -> date:
    """Return the trading day an undated headline belongs to (always the next one)."""
    return cal.next_trading_day(day)


def available_date(raw: str) -> tuple[str, date]:
    """Parse a raw date string; return (normalized publication string, available trading day)."""
    raw = raw.strip()
    if has_time_of_day(raw):
        ts = pd.Timestamp(raw).tz_convert("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ"), available_for_timestamp(ts)
    day = pd.Timestamp(raw).date()
    return day.isoformat(), available_for_day(day)


def is_usable(available: date | str, decision_day: date | str) -> bool:
    """Say whether a headline may be used for a decision at the close of decision_day."""
    return pd.Timestamp(available).date() <= pd.Timestamp(decision_day).date()
