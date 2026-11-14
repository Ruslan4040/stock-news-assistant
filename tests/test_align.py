"""Time alignment: after the close or undated -> next trading day."""

from datetime import date

import pandas as pd
import pytest

from src.data import align
from src.data import calendar as cal


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2019-05-15 00:00:00", date(2019, 5, 16)),  # undated, mid-week
        ("2019-05-17 00:00:00", date(2019, 5, 20)),  # undated Friday -> Monday
        ("2019-05-18 00:00:00", date(2019, 5, 20)),  # undated Saturday -> Monday
        ("2019-12-25 00:00:00", date(2019, 12, 26)),  # undated holiday
        ("2020-06-05 10:30:54-04:00", date(2020, 6, 5)),  # during the session
        ("2020-06-05 16:00:00-04:00", date(2020, 6, 5)),  # exactly at the close
        ("2020-06-05 16:00:01-04:00", date(2020, 6, 8)),  # one second after the close
        ("2020-06-06 09:00:00-04:00", date(2020, 6, 8)),  # weekend
        ("2019-07-03 14:00:00-04:00", date(2019, 7, 5)),  # early close at 13:00, then a holiday
    ],
)
def test_available_date(raw: str, expected: date) -> None:
    """Each raw date string lands on the right trading day."""
    assert align.available_date(raw)[1] == expected


def test_published_string_is_normalized() -> None:
    """Undated rows keep a date; timed rows become UTC timestamps."""
    assert align.available_date("2019-05-15 00:00:00")[0] == "2019-05-15"
    assert align.available_date("2020-06-05 10:30:54-04:00")[0] == "2020-06-05T14:30:54Z"


def test_is_usable() -> None:
    """A headline is usable only on or after its available day."""
    assert align.is_usable("2019-05-15", "2019-05-15")
    assert not align.is_usable("2019-05-16", "2019-05-15")


def test_calendar_helpers() -> None:
    """Trading-day arithmetic skips weekends and holidays."""
    assert not cal.is_trading_day("2019-07-04")
    assert cal.on_or_before("2019-05-19") == date(2019, 5, 17)
    assert cal.on_or_after("2019-05-18") == date(2019, 5, 20)
    assert cal.prev_trading_day("2019-05-20") == date(2019, 5, 17)
    assert cal.shift("2019-05-15", 5) == date(2019, 5, 22)
    assert len(cal.trading_days("2019-01-01", "2019-12-31")) == 252
    assert cal.close_utc("2019-07-03") == pd.Timestamp("2019-07-03 17:00", tz="UTC")
