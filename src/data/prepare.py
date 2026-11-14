"""Build the evaluation samples (run after the download).

Run: python -m src.data.prepare
Writes to data/processed/: kaggle_headlines.csv, set_a.jsonl, set_b.jsonl, set_c.jsonl,
outcomes.csv, phrasebank_sample.csv, prepare_report.json, and evals/labels/fresh_headlines.csv.
"""

import json
import random
from datetime import date

import pandas as pd

from src.config import ROOT, load_config
from src.data import anonymize, kaggle, news, phrasebank, prices, sampling, sets
from src.data import calendar as cal
from src.data.collector import collect_historical
from src.data.terms import mentions_company
from src.schemas import Decision

LABEL_COLUMNS = ["relevant", "sentiment", "labeled_by"]


def set_a_candidates(cfg: dict) -> dict[str, list[date]]:
    """Return, per ticker, the set-A trading days that have at least one usable headline."""
    year = cfg["dates"]["set_a_year"]
    days = cal.trading_days(f"{year}-01-01", f"{year}-12-31")
    return {t: [d for d in days if news.kaggle_rows(t, d, 1)] for t in cfg["tickers"]}


def build_set(points: list[tuple[str, date]], name: str) -> list[Decision]:
    """Collect one decision per sampled point, reading only data that is already on disk."""
    return [collect_historical(ticker, day, set_name=name, fetch_to=None) for ticker, day in points]


def outcomes(decisions: list[Decision], cfg: dict) -> pd.DataFrame:
    """Compute the forward returns that decide whether an advice was a hit."""
    sessions = cfg["simulator"]["horizon_days"]
    rows = []
    for d in decisions:
        frame = prices.load_prices(d.ticker)
        day = date.fromisoformat(d.date)
        row = {"id": d.id, "set": d.set, "ticker": d.ticker, "date": d.date}
        for horizon, n in sessions.items():
            row[f"fwd_ret_{horizon}"] = prices.forward_return(frame, day, n)
        rows.append(row)
    return pd.DataFrame(rows)


def fresh_label_sheet(set_c: list[Decision], cfg: dict) -> pd.DataFrame:
    """Sample fresh headlines, evenly per ticker, for labeling (label columns stay empty)."""
    rng = random.Random(cfg["seed"])
    quota = sampling.split_evenly(cfg["evals"]["fresh_labels_n"], list(cfg["tickers"]))
    rows = []
    for ticker in cfg["tickers"]:
        pool = {}
        for d in (d for d in set_c if d.ticker == ticker):
            for h in d.headlines:
                pool.setdefault(news.normalize(h.text), (d.company, h))
        for key in rng.sample(sorted(pool), min(quota[ticker], len(pool))):
            company, h = pool[key]
            rows.append({"ticker": ticker, "company": company, "published": h.published,
                         "source": h.source, "headline": h.text})
    sheet = pd.DataFrame(rows)
    sheet.insert(0, "id", [f"F{i:03d}" for i in range(1, len(sheet) + 1)])
    return sheet.assign(**{column: "" for column in LABEL_COLUMNS})


def naming_stats(decisions: list[Decision]) -> dict:
    """Count, per decision, the headlines that name the company (real tickers only)."""
    real = [d for d in decisions if d.ticker in load_config()["tickers"]]
    named = [sum(mentions_company(d.ticker, h.text) for h in d.headlines) for d in real]
    if not named:
        return {}
    return {
        "headlines_naming_company_mean": round(sum(named) / len(named), 2),
        "decisions_without_named_headline": sum(1 for n in named if n == 0),
    }


def window_hours(decisions: list[Decision]) -> dict:
    """Return the median hours between the newest and oldest headline, per ticker (timed news)."""
    spans: dict[str, list[float]] = {}
    for d in decisions:
        stamps = [pd.Timestamp(h.published) for h in d.headlines if h.published and "T" in h.published]
        if len(stamps) > 1:
            hours = (max(stamps) - min(stamps)).total_seconds() / 3600
            spans.setdefault(d.ticker, []).append(hours)
    return {t: round(float(pd.Series(v).median()), 1) for t, v in spans.items()}


def describe(decisions: list[Decision]) -> dict:
    """Summarize one set: size, headlines per decision, per-ticker counts."""
    counts = [len(d.headlines) for d in decisions]
    per_ticker = pd.Series([d.ticker for d in decisions]).value_counts().to_dict()
    return {
        **naming_stats(decisions),
        "window_hours_median_by_ticker": window_hours(decisions),
        "decisions": len(decisions),
        "headlines_total": sum(counts),
        "headlines_per_decision_mean": round(sum(counts) / max(len(counts), 1), 2),
        "headlines_per_decision_min": min(counts, default=0),
        "headlines_per_decision_max": max(counts, default=0),
        "decisions_without_headlines": sum(1 for c in counts if c == 0),
        "per_ticker": per_ticker,
    }


def anonymization_leaks(set_a: list[Decision], set_b: list[Decision]) -> int:
    """Count set-B headlines that still show a name, a year or a dollar amount."""
    return sum(
        1 for a, b in zip(set_a, set_b, strict=True) for h in b.headlines
        if anonymize.find_leaks(h.text, a.ticker)
    )


def write_label_sheet(sheet: pd.DataFrame, cfg: dict) -> str:
    """Write the labeling sheet unless one exists already (labels are never overwritten)."""
    path = ROOT / cfg["paths"]["labels_dir"] / "fresh_headlines.csv"
    if path.exists():
        return "kept existing file"
    sheet.to_csv(path, index=False)
    return f"wrote {len(sheet)} rows"


def main() -> None:
    """Build every prepared file and a report with the quality counts."""
    cfg = load_config()
    report = {"kaggle": kaggle.save_table(cfg)}
    points_a = sampling.sample_points(set_a_candidates(cfg), cfg["evals"]["sample_n"], cfg["seed"])
    set_a = build_set(points_a, "A")
    set_b = [anonymize.anonymize_decision(d) for d in set_a]
    set_c = build_set(sampling.fresh_points(cfg), "C")
    for letter, decisions in (("a", set_a), ("b", set_b), ("c", set_c)):
        sets.write_decisions(f"set_{letter}.jsonl", decisions)
        report[f"set_{letter}"] = describe(decisions)
    truth = outcomes(set_a + set_c, cfg)
    truth.to_csv(sets.processed("outcomes.csv"), index=False)
    report["outcomes_missing"] = int(truth.isna().sum().sum())
    report["set_b_leaks"] = anonymization_leaks(set_a, set_b)
    sample = phrasebank.sample_phrasebank()
    sample.to_csv(sets.processed("phrasebank_sample.csv"), index=False)
    report["phrasebank_sample"] = sample["label"].value_counts().to_dict()
    report["fresh_label_sheet"] = write_label_sheet(fresh_label_sheet(set_c, cfg), cfg)
    sets.processed("prepare_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
