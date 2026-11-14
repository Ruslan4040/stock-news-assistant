"""Features: exact values on a hand-made price series."""

import statistics
from datetime import date

import pandas as pd
import pytest

from src.data.features import compute_features, volatility_level


def series(closes: list[float]) -> pd.Series:
    """Build a daily close series that ends on 2019-05-15."""
    index = pd.bdate_range(end="2019-05-15", periods=len(closes))
    return pd.Series(closes, index=index)


def test_constant_growth() -> None:
    """1% a day gives exact returns and zero volatility."""
    closes = [100 * 1.01**i for i in range(25)]
    f = compute_features(series(closes), date(2019, 5, 15))
    assert f.ret_1d == pytest.approx(0.01, abs=1e-12)
    assert f.ret_5d == pytest.approx(1.01**5 - 1, abs=1e-12)
    assert f.ret_20d == pytest.approx(1.01**20 - 1, abs=1e-12)
    assert f.volatility_20d == pytest.approx(0.0, abs=1e-12)
    assert f.volatility_level == "low"


def test_alternating_returns() -> None:
    """Volatility equals the sample standard deviation of the last 20 daily returns."""
    closes = [100.0]
    for i in range(24):
        closes.append(closes[-1] * (1.03 if i % 2 == 0 else 0.98))
    daily = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - 20, len(closes))]
    f = compute_features(series(closes), date(2019, 5, 15))
    assert f.volatility_20d == pytest.approx(statistics.stdev(daily), abs=1e-12)
    assert f.ret_1d == pytest.approx(closes[-1] / closes[-2] - 1, abs=1e-12)
    assert f.volatility_level == "high"


def test_future_prices_do_not_change_features() -> None:
    """No look-ahead: prices after the decision day are ignored."""
    closes = [100 + i * 0.7 + (i % 3) for i in range(40)]
    full = series(closes)
    day = full.index[29].date()
    assert compute_features(full, day) == compute_features(full.iloc[:30], day)


def test_errors() -> None:
    """Too little history or a non-trading day is rejected."""
    with pytest.raises(ValueError):
        compute_features(series([100.0] * 10), date(2019, 5, 15))
    with pytest.raises(ValueError):
        compute_features(series([100.0] * 30), date(2019, 5, 18))


def test_volatility_levels() -> None:
    """Thresholds come from config (1% and 2%)."""
    assert [volatility_level(v) for v in (0.005, 0.015, 0.03)] == ["low", "medium", "high"]
