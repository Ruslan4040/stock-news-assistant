"""Simulator rules on a tiny hand-made price series (expected values computed by hand)."""

from datetime import date

import pytest

from src.sim.simulator import Signal, buy_and_hold, max_drawdown, sharpe, simulate

CFG = {"start_cash": 10000, "confidence_threshold": 70, "position_size": 0.2, "fee": 0.001,
       "min_position_cash": 100, "horizon_days": {"1d": 1, "1w": 5}, "trading_days_per_year": 252}
DAYS = [date(2019, 5, d) for d in (13, 14, 15, 16, 17, 20, 21)]
PRICES = {
    "AAA": dict(zip(DAYS, [100.0, 110.0, 121.0, 121.0, 110.0, 100.0, 100.0], strict=True)),
    "BBB": dict(zip(DAYS, [50.0, 50.0, 55.0, 55.0, 60.0, 60.0, 66.0], strict=True)),
}
FEE = 0.001


def test_buy_hold_for_horizon_and_fees() -> None:
    """A confident 1d buy uses 20% of equity, exits the next day and pays two fees."""
    result = simulate([Signal(DAYS[0], "AAA", "buy", "1d", 80)], PRICES, DAYS, CFG)
    shares = 2000 / (100 * (1 + FEE))
    proceeds = shares * 110 * (1 - FEE)
    trade = result.trades[0]
    assert (trade.entry_day, trade.exit_day, trade.reason) == (DAYS[0], DAYS[1], "horizon")
    assert trade.shares == pytest.approx(shares) and trade.proceeds == pytest.approx(proceeds)
    assert result.equity[DAYS[0]] == pytest.approx(8000 + shares * 100)
    assert result.equity[DAYS[-1]] == pytest.approx(8000 + proceeds)
    assert result.metrics["total_return"] == pytest.approx((8000 + proceeds) / 10000 - 1)
    assert result.metrics["n_trades"] == 1 and result.metrics["win_rate"] == 1.0


def test_threshold_long_only_and_one_position_per_ticker() -> None:
    """Low confidence, hold, sell without a position and a repeated buy do nothing."""
    signals = [
        Signal(DAYS[0], "BBB", "buy", "1w", 69),  # below the threshold
        Signal(DAYS[0], "AAA", "hold", "1d", 95),  # not a buy
        Signal(DAYS[0], "AAA", "sell", "1d", 95),  # nothing to sell: long only
        Signal(DAYS[1], "BBB", "buy", "1w", 70),  # exactly at the threshold: trade
        Signal(DAYS[2], "BBB", "buy", "1w", 99),  # already holding: ignored
    ]
    result = simulate(signals, PRICES, DAYS, CFG)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert (trade.ticker, trade.entry_day, trade.exit_day) == ("BBB", DAYS[1], DAYS[6])
    assert trade.reason == "horizon" and trade.cost == pytest.approx(2000)


def test_sell_advice_closes_early_and_equity_compounds() -> None:
    """A later sell advice closes the position before the horizon; sizes use current equity."""
    signals = [
        Signal(DAYS[0], "AAA", "buy", "1d", 80),
        Signal(DAYS[1], "BBB", "buy", "1w", 75),
        Signal(DAYS[3], "BBB", "sell", "1d", 40),
    ]
    result = simulate(signals, PRICES, DAYS, CFG)
    first_proceeds = 2000 / (100 * (1 + FEE)) * 110 * (1 - FEE)
    equity_day1 = 8000 + first_proceeds  # AAA is sold before BBB is sized
    budget = 0.2 * equity_day1
    second_proceeds = budget / (50 * (1 + FEE)) * 55 * (1 - FEE)
    second = result.trades[1]
    assert (second.exit_day, second.reason) == (DAYS[3], "sell_advice")
    assert second.cost == pytest.approx(budget)
    assert second.proceeds == pytest.approx(second_proceeds)
    assert result.equity[DAYS[-1]] == pytest.approx(equity_day1 - budget + second_proceeds)
    assert result.metrics["win_rate"] == 1.0 and result.metrics["n_trades"] == 2


def test_position_past_the_data_is_closed_at_the_end() -> None:
    """A 1w trade that cannot finish is closed on the last day and can lose money."""
    result = simulate([Signal(DAYS[2], "AAA", "buy", "1w", 90)], PRICES, DAYS, CFG)
    trade = result.trades[0]
    assert (trade.exit_day, trade.reason) == (DAYS[6], "end_of_data")
    assert trade.pnl == pytest.approx(2000 / (121 * (1 + FEE)) * 100 * (1 - FEE) - 2000)
    assert result.metrics["win_rate"] == 0.0 and result.metrics["total_return"] < 0
    assert result.exposure[DAYS[1]] == 0.0 and 0.19 < result.exposure[DAYS[2]] < 0.21


def test_no_trades() -> None:
    """Without trades the equity stays flat and the win rate is undefined."""
    result = simulate([], PRICES, DAYS, CFG)
    assert set(result.equity.values()) == {10000.0}
    assert result.metrics["total_return"] == 0.0 and result.metrics["win_rate"] is None
    assert result.metrics["sharpe"] == 0.0 and result.metrics["max_drawdown"] == 0.0


def test_buy_and_hold_baseline() -> None:
    """The baseline splits the cash equally, holds, and pays the fee on entry and exit."""
    result = buy_and_hold(PRICES, ["AAA", "BBB"], DAYS, CFG)
    a, b = 5000 / (100 * (1 + FEE)), 5000 / (50 * (1 + FEE))
    assert result.equity[DAYS[1]] == pytest.approx(a * 110 + b * 50)
    assert result.equity[DAYS[-1]] == pytest.approx((a * 100 + b * 66) * (1 - FEE))
    assert result.metrics["n_trades"] == 2 and result.metrics["mean_exposure"] == 1.0
    assert result.metrics["win_rate"] == 0.5  # BBB wins, AAA only loses the fees


def test_risk_metrics() -> None:
    """Drawdown and Sharpe match hand calculations."""
    assert max_drawdown([100, 120, 90, 130]) == pytest.approx(0.25)
    assert sharpe([100, 100, 100], 252) == 0.0  # no variance: defined as 0
    returns = [0.10, -0.05]
    mean = sum(returns) / 2
    std = (sum((r - mean) ** 2 for r in returns) / 1) ** 0.5
    assert sharpe([100, 110, 104.5], 252) == pytest.approx(mean / std * 252**0.5)
