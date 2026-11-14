"""Eval of the data collector (component 1): missing days, duplicates, look-ahead, leaks.

Run: python -m evals.eval_data
"""

from datetime import date

import numpy as np
import pandas as pd

from evals import common
from src.config import load_config
from src.data import align, anonymize, kaggle, news, prices, sets
from src.data import calendar as cal

COMPONENT = "data"


def missing_day_flags(cfg: dict) -> np.ndarray:
    """Return one flag per expected (ticker, trading day): 1 when the price row is missing."""
    flags = []
    for ticker in cfg["tickers"]:
        have = set(prices.load_prices(ticker).index.date)
        expected = cal.trading_days(cfg["dates"]["price_start"], cfg["dates"]["prices_end"])
        flags += [0 if day in have else 1 for day in expected]
    return np.array(flags)


def raw_duplicates(cfg: dict) -> tuple[np.ndarray, list[dict]]:
    """Return duplicate flags over the raw Kaggle rows and the most repeated headlines."""
    rows = kaggle.read_rows(cfg).dropna(subset=["headline", "date"])
    rows = rows.assign(key=rows["headline"].map(news.normalize), day=rows["date"].str[:10])
    flags = rows.duplicated(["ticker", "day", "key"]).to_numpy().astype(int)
    counts = rows.groupby(["ticker", "day", "key"]).size().sort_values(ascending=False)
    worst = [
        {"ticker": t, "day": d, "copies": int(n), "headline": k}
        for (t, d, k), n in counts[counts > 1].head(common.WORST_ERRORS).items()
    ]
    return flags, worst


def set_duplicate_flags(letters: str) -> np.ndarray:
    """Return one flag per headline in the prepared sets: 1 when repeated inside a decision."""
    flags = []
    for letter in letters:
        for decision in sets.load_set(letter):
            keys = [news.normalize(h.text) for h in decision.headlines]
            flags += [1 if keys.index(key) < i else 0 for i, key in enumerate(keys)]
    return np.array(flags)


def lookahead_flags(letters: str) -> np.ndarray:
    """Return one flag per headline: 1 when it was not yet available at the decision close."""
    flags = []
    for letter in letters:
        for decision in sets.load_set(letter):
            day = date.fromisoformat(decision.date)
            late = [not align.is_usable(h.available_date, day) for h in decision.headlines]
            flags += [int(flag) for flag in late]
    return np.array(flags)


def leak_flags() -> np.ndarray:
    """Return one flag per set-B headline: 1 when a name, year or dollar amount survived."""
    flags = []
    for real, hidden in zip(sets.load_set("A"), sets.load_set("B"), strict=True):
        flags += [1 if anonymize.find_leaks(h.text, real.ticker) else 0 for h in hidden.headlines]
    return np.array(flags)


def main() -> None:
    """Compute the data-quality metrics and write the results file."""
    common.limit_argument(__doc__)
    cfg = load_config()
    duplicate_flags, worst = raw_duplicates(cfg)
    outcomes = pd.read_csv(sets.processed("outcomes.csv"))
    missing_outcomes = outcomes[["fwd_ret_1d", "fwd_ret_1w"]].isna().any(axis=1).to_numpy()
    rows = {
        "missing_trading_days_rate": missing_day_flags(cfg),
        "duplicate_headlines_rate_raw": duplicate_flags,
        "duplicate_headlines_rate_in_sets": set_duplicate_flags("AC"),
        "lookahead_violation_rate": lookahead_flags("AC"),
        "set_b_leak_rate": leak_flags(),
        "missing_outcome_rate": missing_outcomes.astype(int),
    }
    metrics = [
        common.metric_row(name, common.bootstrap(common.mean, flags), count=int(flags.sum()))
        for name, flags in rows.items()
    ]
    path = common.write_result(
        COMPONENT, metrics, "all downloaded prices, raw Kaggle rows of the 8 tickers, sets A/B/C"
    )
    common.write_errors(COMPONENT, worst)
    common.show(path)


if __name__ == "__main__":
    main()
