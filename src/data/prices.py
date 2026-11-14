"""Price loading: adjusted closes for returns, raw closes for the memory probe."""

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from src.config import ROOT, load_config

COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]


def future_split_factor(splits: pd.Series) -> pd.Series:
    """Return, for each day, the product of all split ratios that happen after it."""
    ratios = splits.where(splits > 0, 1.0)
    at_or_after = ratios[::-1].cumprod()[::-1]
    return at_or_after.shift(-1).fillna(1.0)


def tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a yfinance frame: flat snake_case columns, date index, raw_close added."""
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.droplevel(1, axis=1)
    frame = frame.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"
    frame = frame[COLUMNS].dropna(subset=["close"]).sort_index()
    frame["raw_close"] = frame["close"] * future_split_factor(frame["stock_splits"])
    return frame


def fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download daily prices from Yahoo Finance for [start, end]."""
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=True,
        progress=False,
    )
    return tidy(raw)


@lru_cache(maxsize=32)
def load_prices(ticker: str) -> pd.DataFrame:
    """Load the downloaded daily prices of one ticker (index = trading day)."""
    path = ROOT / load_config()["paths"]["prices_dir"] / f"{ticker}.csv"
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def live_prices(ticker: str) -> pd.DataFrame:
    """Fetch recent prices for a live request; fall back to the stored snapshot on failure."""
    days = load_config()["news"]["live_price_days"]
    today = date.today()
    try:
        frame = fetch(ticker, today - timedelta(days=days), today)
        if len(frame) > 0:
            return frame
    except Exception:  # noqa: BLE001 - any network or parsing problem falls back to the snapshot
        pass
    return load_prices(ticker)


def forward_return(prices: pd.DataFrame, day: date, sessions: int) -> float | None:
    """Return the adjusted close-to-close return from `day` to `sessions` sessions later."""
    closes = prices["adj_close"]
    stamp = pd.Timestamp(day)
    if stamp not in closes.index:
        return None
    pos = closes.index.get_loc(stamp) + sessions
    if pos >= len(closes):
        return None
    return float(closes.iloc[pos] / closes.loc[stamp] - 1.0)
