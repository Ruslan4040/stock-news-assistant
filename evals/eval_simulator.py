"""Eval of the trader (component 6): return, Sharpe, drawdown, trades, win rate per set.

Run: python -m evals.eval_simulator [--limit N]
Trades the stored advice of sets A, B and C. Baseline: buy & hold of the same tickers
over the same days. Intervals: bootstrap over daily returns (trades for the win rate).
"""

from datetime import date

import numpy as np
import pandas as pd

from evals import common
from evals.eval_advisor import SETS, load_rows
from src.config import load_config
from src.data import calendar as cal
from src.data import prices
from src.sim.simulator import (
    Prices,
    Signal,
    SimResult,
    buy_and_hold,
    max_drawdown,
    sharpe,
    simulate,
)

COMPONENT = "simulator"


def signals_of(frame: pd.DataFrame) -> list[Signal]:
    """Turn the valid advice rows of a set into simulator signals."""
    valid = frame[(~frame["incomplete"]) & (frame["error"] == "")]
    return [
        Signal(date.fromisoformat(row.date), row.ticker, row.action, row.horizon, row.confidence)
        for row in valid.itertuples()
    ]


def trading_period(signals: list[Signal], cfg: dict) -> list[date]:
    """Return the trading days from the first advice to the last possible exit."""
    longest = max(cfg["simulator"]["horizon_days"].values())
    last_day = pd.Timestamp(cfg["dates"]["prices_end"]).date()
    end = min(cal.shift(max(s.day for s in signals), longest), last_day)
    return cal.trading_days(min(s.day for s in signals), end)


def price_table(tickers: list[str], days: list[date]) -> Prices:
    """Return adjusted closes per ticker and day."""
    table = {}
    for ticker in tickers:
        closes = prices.load_prices(ticker)["adj_close"]
        table[ticker] = {day: float(closes.loc[pd.Timestamp(day)]) for day in days}
    return table


def curve_metric(function: str, cfg: dict) -> callable:
    """Build a metric over an array of daily returns (for the bootstrap)."""
    def metric(returns: np.ndarray) -> float:
        """Rebuild an equity curve from daily returns and score it."""
        curve = [1.0, *np.cumprod(1 + returns)]
        if function == "total_return":
            return curve[-1] - 1
        if function == "sharpe":
            return sharpe(curve, cfg["simulator"]["trading_days_per_year"])
        return max_drawdown(curve)

    return metric


def metric_rows(letter: str, sim: SimResult, hold: SimResult, cfg: dict) -> list[dict]:
    """Report each simulator metric with a bootstrap interval and the buy & hold value."""
    values = np.array(list(sim.equity.values()))
    returns = values[1:] / values[:-1] - 1
    rows = []
    for name in ("total_return", "sharpe", "max_drawdown"):
        stat = common.bootstrap(curve_metric(name, cfg), returns)
        stat["value"] = sim.metrics[name]
        rows.append(common.metric_row(f"{letter}.{name}", stat,
                                      {"buy_and_hold": round(hold.metrics[name], 4)}))
    wins = np.array([1.0 if trade.pnl > 0 else 0.0 for trade in sim.trades])
    rows.append(common.metric_row(f"{letter}.win_rate", common.bootstrap(common.mean, wins)))
    count = {"value": len(sim.trades), "ci_low": None, "ci_high": None, "n": len(sim.trades)}
    rows.append(common.metric_row(f"{letter}.n_trades", count))
    return rows


def set_details(sim: SimResult, hold: SimResult, signals: list[Signal], cfg: dict) -> dict:
    """Keep what the UI and the report need: curves, trades and counts."""
    threshold = cfg["simulator"]["confidence_threshold"]
    trades = [
        {"ticker": t.ticker, "entry": t.entry_day, "exit": t.exit_day, "reason": t.reason,
         "return_pct": round(t.return_pct * 100, 3)}
        for t in sim.trades
    ]
    return {
        "signals": len(signals),
        "buy_signals_at_threshold": sum(1 for s in signals
                                        if s.action == "buy" and s.confidence >= threshold),
        "mean_exposure": round(sim.metrics["mean_exposure"], 4),
        "days": [day.isoformat() for day in sim.equity],
        "equity": [round(v, 2) for v in sim.equity.values()],
        "buy_and_hold_equity": [round(v, 2) for v in hold.equity.values()],
        "buy_and_hold": {k: round(v, 4) for k, v in hold.metrics.items() if v is not None},
        "trades": trades,
    }


def main() -> None:
    """Simulate every set that has stored runs and write the results file."""
    limit = common.limit_argument(__doc__)
    cfg = load_config()
    tickers = list(cfg["tickers"])
    metrics, details, worst = [], {}, []
    for letter in SETS:
        frame = load_rows(letter, limit)
        if not len(frame):
            continue
        signals = signals_of(frame)
        days = trading_period(signals, cfg)
        table = price_table(tickers, days)
        sim = simulate(signals, table, days, cfg["simulator"])
        hold = buy_and_hold(table, tickers, days, cfg["simulator"])
        metrics += metric_rows(letter, sim, hold, cfg)
        details[letter] = set_details(sim, hold, signals, cfg)
        worst += [{"set": letter, **trade} for trade in details[letter]["trades"]]
    worst.sort(key=lambda trade: trade["return_pct"])
    data = "stored advice of sets A, B, C + data/raw/prices"
    path = common.write_result(COMPONENT, metrics, data, limit=limit, details=details)
    common.write_errors(COMPONENT, worst, limit)
    common.show(path)


if __name__ == "__main__":
    main()
