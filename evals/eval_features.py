"""Eval of the features (component 2): exact match with independently computed values.

Run: python -m evals.eval_features
The reference values in evals/labels/features_reference.csv come from a second, independent
implementation (csv + statistics only, no pandas), written once and then kept fixed.
"""

import csv
import random
import statistics
from datetime import date

import numpy as np

from evals import common
from src.config import ROOT, load_config
from src.data import calendar as cal
from src.data import prices
from src.data.features import LOOKBACKS, compute_features

COMPONENT = "features"
TOLERANCE = 1e-6  # docs/PROJECT_SPEC.md §7
FIELDS = [*LOOKBACKS, "volatility_20d"]


def reference_path() -> str:
    """Return the path of the fixed reference file."""
    return str(ROOT / load_config()["paths"]["labels_dir"] / "features_reference.csv")


def independent_features(ticker: str, day: date) -> dict:
    """Recompute the features straight from the price CSV, without the project code."""
    cfg = load_config()
    path = ROOT / cfg["paths"]["prices_dir"] / f"{ticker}.csv"
    with open(path, encoding="utf-8", newline="") as handle:
        rows = [(row["date"][:10], float(row["adj_close"])) for row in csv.DictReader(handle)]
    at = [d for d, _ in rows].index(day.isoformat())
    closes = [close for _, close in rows]
    window = cfg["features"]["vol_window"]
    daily = [closes[k] / closes[k - 1] - 1 for k in range(at - window + 1, at + 1)]
    values = {name: closes[at] / closes[at - n] - 1 for name, n in LOOKBACKS.items()}
    return {**values, "volatility_20d": statistics.stdev(daily)}


def make_reference() -> None:
    """Write the reference file once: 3 tickers x 5 seeded dates (never overwritten)."""
    cfg = load_config()
    spec = cfg["evals"]["features_reference"]
    rng = random.Random(cfg["seed"])
    days = cal.trading_days(f"{cfg['dates']['set_a_year']}-01-01", cfg["dates"]["prices_end"])
    with open(reference_path(), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "date", *FIELDS])
        writer.writeheader()
        for ticker in spec["tickers"]:
            for day in sorted(rng.sample(days, spec["n_dates"])):
                row = independent_features(ticker, day)
                writer.writerow({"ticker": ticker, "date": day.isoformat(),
                                 **{k: repr(v) for k, v in row.items()}})


def compare() -> tuple[np.ndarray, list[dict]]:
    """Return per-case match flags and every case with its largest absolute error."""
    flags, cases = [], []
    with open(reference_path(), encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["date"])
            got = compute_features(prices.load_prices(row["ticker"])["adj_close"], day)
            errors = {f: abs(getattr(got, f) - float(row[f])) for f in FIELDS}
            worst = max(errors, key=errors.get)
            flags.append(1 if errors[worst] <= TOLERANCE else 0)
            cases.append({"ticker": row["ticker"], "date": row["date"], "worst_field": worst,
                          "max_abs_error": errors[worst]})
    return np.array(flags), sorted(cases, key=lambda c: -c["max_abs_error"])


def main() -> None:
    """Create the reference if needed, compare, and write the results file."""
    common.limit_argument(__doc__)
    if not (ROOT / reference_path()).exists():
        make_reference()
    flags, cases = compare()
    largest = max(case["max_abs_error"] for case in cases)
    metrics = [
        common.metric_row("exact_match_rate", common.bootstrap(common.mean, flags),
                          tolerance=TOLERANCE, matches=int(flags.sum())),
        common.metric_row("max_abs_error", {"value": largest, "ci_low": None, "ci_high": None,
                                            "n": len(cases)}),
    ]
    path = common.write_result(
        COMPONENT, metrics, "evals/labels/features_reference.csv (3 tickers x 5 dates)"
    )
    common.write_errors(COMPONENT, [case for case in cases if case["max_abs_error"] > TOLERANCE])
    common.show(path)


if __name__ == "__main__":
    main()
