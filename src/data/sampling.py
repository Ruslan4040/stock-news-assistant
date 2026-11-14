"""Deterministic sampling of decision points (ticker, trading day)."""

import random
from datetime import date

from src.data import calendar as cal


def split_evenly(total: int, tickers: list[str]) -> dict[str, int]:
    """Split `total` decisions across tickers as evenly as possible (first ones get the rest)."""
    base, extra = divmod(total, len(tickers))
    return {ticker: base + (1 if i < extra else 0) for i, ticker in enumerate(tickers)}


def sample_points(
    candidates: dict[str, list[date]], total: int, seed: int
) -> list[tuple[str, date]]:
    """Sample decision days per ticker without replacement, reproducibly."""
    rng = random.Random(seed)
    quota = split_evenly(total, list(candidates))
    points: list[tuple[str, date]] = []
    for ticker, days in candidates.items():
        chosen = rng.sample(sorted(days), min(quota[ticker], len(days)))
        points += [(ticker, day) for day in sorted(chosen)]
    return points


def fresh_points(cfg: dict) -> list[tuple[str, date]]:
    """Return the sampled set-C decision points (period after the model's knowledge cutoff)."""
    days = cal.trading_days(cfg["dates"]["fresh_start"], cfg["dates"]["fresh_end"])
    candidates = {ticker: days for ticker in cfg["tickers"]}
    return sample_points(candidates, cfg["evals"]["sample_n"], cfg["seed"])
