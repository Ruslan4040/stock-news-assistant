"""Sampling is reproducible; split factors and forward returns are exact."""

from datetime import date

import pandas as pd
import pytest

from src.data import prices, sampling


def test_split_evenly() -> None:
    """300 decisions over 8 tickers: four get 38, four get 37."""
    quota = sampling.split_evenly(300, list("ABCDEFGH"))
    assert sum(quota.values()) == 300
    assert sorted(quota.values()) == [37] * 4 + [38] * 4


def test_sample_points_is_deterministic() -> None:
    """The same seed gives the same points, without repeats."""
    days = [date(2019, 1, d) for d in range(1, 29)]
    candidates = {"AAA": days, "BBB": days[:5]}
    first = sampling.sample_points(candidates, 20, seed=42)
    assert first == sampling.sample_points(candidates, 20, seed=42)
    assert len(first) == len(set(first)) == 15  # 10 for AAA, only 5 exist for BBB
    assert first != sampling.sample_points(candidates, 20, seed=7)


def test_future_split_factor() -> None:
    """A 4-for-1 split on day 3 scales the raw close of the days before it."""
    splits = pd.Series([0.0, 0.0, 4.0, 0.0, 2.0, 0.0])
    assert prices.future_split_factor(splits).tolist() == [8.0, 8.0, 2.0, 2.0, 1.0, 1.0]


def test_forward_return() -> None:
    """Forward returns use adjusted closes and stop at the end of the data."""
    index = pd.bdate_range("2019-05-13", periods=4)
    frame = pd.DataFrame({"adj_close": [100.0, 110.0, 99.0, 121.0]}, index=index)
    assert prices.forward_return(frame, date(2019, 5, 13), 1) == pytest.approx(0.10)
    assert prices.forward_return(frame, date(2019, 5, 14), 2) == pytest.approx(0.10)
    assert prices.forward_return(frame, date(2019, 5, 16), 1) is None
    assert prices.forward_return(frame, date(2019, 5, 18), 1) is None
