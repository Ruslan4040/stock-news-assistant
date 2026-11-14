"""News windows: the headlines available for a decision, from Kaggle, Finnhub or yfinance."""

import re
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache

import pandas as pd

from src.config import ROOT, load_config
from src.data import calendar as cal
from src.data import finnhub
from src.schemas import Headline

KAGGLE_TABLE = "kaggle_headlines.csv"


def normalize(text: str) -> str:
    """Return a de-duplication key: lower case, single spaces, no edge punctuation."""
    return re.sub(r"\s+", " ", text).strip().strip(".!?'\"").lower()


def clean_text(text: str) -> str:
    """Collapse whitespace and cut very long headlines."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[: load_config()["news"]["max_headline_chars"]]


@lru_cache(maxsize=1)
def _kaggle_by_ticker() -> dict[str, pd.DataFrame]:
    """Load the cleaned Kaggle table once and split it by ticker."""
    path = ROOT / load_config()["paths"]["processed_dir"] / KAGGLE_TABLE
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table["order"] = table["order"].astype(int)
    return {ticker: part for ticker, part in table.groupby("ticker")}


def kaggle_rows(ticker: str, day: date, limit: int) -> list[dict]:
    """Return the most recent Kaggle rows whose available day lies in [day - window, day]."""
    part = _kaggle_by_ticker().get(ticker)
    if part is None:
        return []
    low = (day - timedelta(days=load_config()["news"]["window_days"])).isoformat()
    inside = part[(part["available_date"] >= low) & (part["available_date"] <= day.isoformat())]
    ordered = inside.sort_values(
        ["available_date", "published", "order"], ascending=[False, False, True]
    )
    return ordered.drop_duplicates("key").head(limit).to_dict("records")


def window_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (lower, upper] UTC instants of the news window for a decision at the close of day."""
    first = cal.on_or_after(day - timedelta(days=load_config()["news"]["window_days"]))
    return cal.close_utc(cal.prev_trading_day(first)), cal.close_utc(day)


def finnhub_rows(
    ticker: str, lower: pd.Timestamp, upper: pd.Timestamp, limit: int, fetch_to: str | None
) -> list[dict]:
    """Return up to `limit` most recent de-duplicated Finnhub rows published in (lower, upper]."""
    start = pd.Timestamp(load_config()["news"]["finnhub_history_start"]).date()
    rows: list[dict] = []
    seen: set[str] = set()
    day = upper.date()
    while day >= max(lower.date(), start) and len(rows) < limit:
        for item in finnhub.get_day(ticker, day, fetch_to):
            stamp = pd.Timestamp(item["datetime"], unit="s", tz="UTC")
            key = normalize(item["headline"])
            if lower < stamp <= upper and key and key not in seen:
                seen.add(key)
                rows.append({**item, "stamp": stamp})
        day -= timedelta(days=1)
    rows.sort(key=lambda row: row["stamp"], reverse=True)
    return rows[:limit]


def _from_kaggle(rows: list[dict]) -> list[Headline]:
    """Turn Kaggle rows into numbered headlines (1 = most recent)."""
    return [
        Headline(
            id=i + 1,
            text=clean_text(row["headline"]),
            published=row["published"],
            available_date=row["available_date"],
            source=row["publisher"] or None,
        )
        for i, row in enumerate(rows)
    ]


def _from_finnhub(rows: list[dict]) -> list[Headline]:
    """Turn Finnhub rows into numbered headlines (1 = most recent)."""
    return [
        Headline(
            id=i + 1,
            text=clean_text(row["headline"]),
            published=row["stamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            available_date=cal.first_close_at_or_after(row["stamp"]).isoformat(),
            source=row.get("source") or None,
        )
        for i, row in enumerate(rows)
    ]


def headlines_for(ticker: str, day: date, fetch_to: str | None) -> list[Headline]:
    """Return the headlines usable for a decision at the close of `day` (historical mode)."""
    cfg = load_config()
    limit = cfg["news"]["max_headlines"]
    if day <= pd.Timestamp(cfg["dates"]["kaggle_end"]).date():
        return _from_kaggle(kaggle_rows(ticker, day, limit))
    if day >= pd.Timestamp(cfg["news"]["finnhub_history_start"]).date():
        lower, upper = window_bounds(day)
        return _from_finnhub(finnhub_rows(ticker, lower, upper, limit, fetch_to))
    return []


def _yf_rows(ticker: str, lower: pd.Timestamp, upper: pd.Timestamp) -> list[dict]:
    """Return yfinance news items in (lower, upper] shaped like Finnhub rows (live backup)."""
    import yfinance as yf

    rows = []
    for item in yf.Ticker(ticker).news or []:
        content = item.get("content", item)
        stamp = pd.Timestamp(content.get("pubDate")) if content.get("pubDate") else None
        if stamp is not None and lower < stamp <= upper and content.get("title"):
            source = (content.get("provider") or {}).get("displayName")
            rows.append({"headline": content["title"], "stamp": stamp, "source": source})
    return sorted(rows, key=lambda row: row["stamp"], reverse=True)


def live_headlines(ticker: str, now: datetime | None = None) -> tuple[list[Headline], str]:
    """Return the headlines of the last window_days days up to now, and the source used."""
    cfg = load_config()["news"]
    upper = pd.Timestamp(now or datetime.now(UTC))
    lower = upper - pd.Timedelta(days=cfg["window_days"])
    try:
        rows = finnhub_rows(ticker, lower, upper, cfg["max_headlines"], finnhub.cache_dir())
        source = "finnhub"
    except Exception:  # any API problem: use the backup source
        rows = []
        source = "finnhub_failed"
    if not rows:
        rows = _yf_rows(ticker, lower, upper)[: cfg["max_headlines"]]
        source = "yfinance" if rows else source
    return _from_finnhub(rows), source
