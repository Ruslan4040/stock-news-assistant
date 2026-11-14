"""Trading simulator (component 6): long-only trades on confident buy advice. Pure Python.

Rules (docs/PROJECT_SPEC.md §6): start with the configured cash; buy only when action = buy
and confidence >= threshold, with a fixed share of equity; hold for the advised horizon or
sell earlier when a later advice for that ticker says sell; pay a fee on every buy and sell.
Trades execute at the close of the decision day. One open position per ticker.
"""

import math
from dataclasses import dataclass, field
from datetime import date

Prices = dict[str, dict[date, float]]


@dataclass(frozen=True)
class Signal:
    """One advice as the simulator sees it."""

    day: date
    ticker: str
    action: str
    horizon: str
    confidence: int


@dataclass
class Trade:
    """One position from entry to exit."""

    ticker: str
    entry_day: date
    planned_exit: date
    entry_price: float
    shares: float
    cost: float  # cash paid, fee included
    exit_day: date | None = None
    exit_price: float | None = None
    proceeds: float | None = None  # cash received, fee deducted
    reason: str | None = None  # horizon / sell_advice / end_of_data

    @property
    def pnl(self) -> float:
        """Return the profit in dollars after both fees."""
        return (self.proceeds or 0.0) - self.cost

    @property
    def return_pct(self) -> float:
        """Return the profit as a fraction of the cash paid."""
        return self.pnl / self.cost


@dataclass
class SimResult:
    """Equity curve, closed trades and summary metrics of one simulation."""

    equity: dict[date, float] = field(default_factory=dict)
    exposure: dict[date, float] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class _Book:
    """Cash and open positions during a simulation."""

    def __init__(self, cfg: dict, prices: Prices, days: list[date]) -> None:
        """Start with cash only."""
        self.cfg, self.prices, self.days = cfg, prices, days
        self.cash = float(cfg["start_cash"])
        self.open: dict[str, Trade] = {}
        self.closed: list[Trade] = []

    def invested(self, day: date) -> float:
        """Return the market value of all open positions."""
        return sum(t.shares * self.prices[t.ticker][day] for t in self.open.values())

    def close(self, ticker: str, day: date, reason: str) -> None:
        """Sell a position at the close of the day and pay the fee."""
        trade = self.open.pop(ticker)
        trade.exit_day, trade.exit_price, trade.reason = day, self.prices[ticker][day], reason
        trade.proceeds = trade.shares * trade.exit_price * (1 - self.cfg["fee"])
        self.cash += trade.proceeds
        self.closed.append(trade)

    def buy(self, signal: Signal, equity: float) -> None:
        """Open a position worth a fixed share of equity (fee included in the budget)."""
        budget = min(self.cfg["position_size"] * equity, self.cash)
        if budget < self.cfg["min_position_cash"]:
            return
        price = self.prices[signal.ticker][signal.day]
        position = self.days.index(signal.day) + self.cfg["horizon_days"][signal.horizon]
        planned = self.days[position] if position < len(self.days) else date.max
        shares = budget / (price * (1 + self.cfg["fee"]))
        self.cash -= budget
        self.open[signal.ticker] = Trade(signal.ticker, signal.day, planned, price, shares, budget)


def _step(book: _Book, day: date, signals: list[Signal]) -> None:
    """Process one day: due exits, sell advice, then new buys."""
    for ticker in [t for t, trade in book.open.items() if trade.planned_exit <= day]:
        book.close(ticker, day, "horizon")
    for signal in signals:
        if signal.action == "sell" and signal.ticker in book.open:
            book.close(signal.ticker, day, "sell_advice")
    equity = book.cash + book.invested(day)
    for signal in signals:
        confident = signal.confidence >= book.cfg["confidence_threshold"]
        if signal.action == "buy" and confident and signal.ticker not in book.open:
            book.buy(signal, equity)


def max_drawdown(values: list[float]) -> float:
    """Return the largest peak-to-trough fall as a positive fraction."""
    peak, worst = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak)
    return worst


def sharpe(values: list[float], periods_per_year: int) -> float:
    """Return the annualized Sharpe ratio of an equity curve (risk-free rate 0)."""
    returns = [b / a - 1 for a, b in zip(values, values[1:], strict=False)]
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return 0.0 if variance == 0 else mean / math.sqrt(variance) * math.sqrt(periods_per_year)


def summarize(result: SimResult, cfg: dict) -> dict:
    """Compute total return, Sharpe, max drawdown, trade count, win rate and exposure."""
    values = list(result.equity.values())
    wins = sum(1 for trade in result.trades if trade.pnl > 0)
    trades = len(result.trades)
    return {
        "total_return": values[-1] / cfg["start_cash"] - 1,
        "sharpe": sharpe(values, cfg["trading_days_per_year"]),
        "max_drawdown": max_drawdown(values),
        "n_trades": trades,
        "win_rate": wins / trades if trades else None,
        "mean_exposure": sum(result.exposure.values()) / len(result.exposure),
        "days": len(values),
    }


def simulate(signals: list[Signal], prices: Prices, days: list[date], cfg: dict) -> SimResult:
    """Trade the advice stream over the given trading days and return the result."""
    book = _Book(cfg, prices, days)
    by_day: dict[date, list[Signal]] = {}
    for signal in signals:
        by_day.setdefault(signal.day, []).append(signal)
    result = SimResult()
    for day in days:
        _step(book, day, by_day.get(day, []))
        if day == days[-1]:
            for ticker in list(book.open):
                book.close(ticker, day, "end_of_data")
        invested = book.invested(day)
        result.equity[day] = book.cash + invested
        result.exposure[day] = invested / result.equity[day]
    result.trades = book.closed
    result.metrics = summarize(result, cfg)
    return result


def buy_and_hold(prices: Prices, tickers: list[str], days: list[date], cfg: dict) -> SimResult:
    """Baseline: split the cash equally over the tickers on day one and hold to the end."""
    budget = cfg["start_cash"] / len(tickers)
    first, last = days[0], days[-1]
    shares = {t: budget / (prices[t][first] * (1 + cfg["fee"])) for t in tickers}
    result = SimResult()
    for day in days:
        value = sum(shares[t] * prices[t][day] for t in tickers)
        result.equity[day] = value * (1 - cfg["fee"]) if day == last else value
        result.exposure[day] = 1.0
    for ticker in tickers:
        proceeds = shares[ticker] * prices[ticker][last] * (1 - cfg["fee"])
        result.trades.append(Trade(ticker, first, last, prices[ticker][first], shares[ticker],
                                   budget, last, prices[ticker][last], proceeds, "end_of_data"))
    result.metrics = summarize(result, cfg)
    return result
