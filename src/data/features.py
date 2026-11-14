"""Features (component 2): price trend numbers computed in plain Python, never by the LLM."""

from datetime import date

import pandas as pd

from src.config import load_config
from src.schemas import Features

LOOKBACKS = {"ret_1d": 1, "ret_5d": 5, "ret_20d": 20}  # sessions, fixed by the field names


def volatility_level(volatility: float) -> str:
    """Bucket a daily volatility into low / medium / high with the config thresholds."""
    cfg = load_config()["features"]
    if volatility < cfg["vol_low"]:
        return "low"
    return "high" if volatility > cfg["vol_high"] else "medium"


def compute_features(adj_close: pd.Series, day: date) -> Features:
    """Compute returns and volatility from adjusted closes up to and including `day`."""
    window = load_config()["features"]["vol_window"]
    history = adj_close.loc[: pd.Timestamp(day)]
    needed = max(window, max(LOOKBACKS.values())) + 1
    if len(history) < needed or history.index[-1] != pd.Timestamp(day):
        raise ValueError(f"Not enough price history for {day} (need {needed} closes ending there)")
    last = float(history.iloc[-1])
    returns = {name: last / float(history.iloc[-1 - n]) - 1.0 for name, n in LOOKBACKS.items()}
    volatility = float(history.pct_change().iloc[-window:].std(ddof=1))
    return Features(
        **returns, volatility_20d=volatility, volatility_level=volatility_level(volatility)
    )
