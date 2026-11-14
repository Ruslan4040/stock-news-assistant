"""Kaggle headlines: filter the big CSV files to the config tickers, clean and align them."""

import pandas as pd

from src.config import ROOT, load_config
from src.data import align
from src.data.news import KAGGLE_TABLE, clean_text, normalize

RAW_COLUMNS = ["headline", "publisher", "date", "stock"]
CHUNK_ROWS = 200_000  # rows per read; only bounds memory use


def code_map(cfg: dict) -> dict[str, str]:
    """Map every Kaggle stock code (ticker or alias code) to its config ticker."""
    codes = {ticker: ticker for ticker in cfg["tickers"]}
    for ticker, extra in cfg["kaggle_ticker_aliases"].items():
        codes.update({code: ticker for code in extra})
    return codes


def read_rows(cfg: dict) -> pd.DataFrame:
    """Read the raw CSV files and keep only the rows of the config tickers."""
    codes = code_map(cfg)
    parts = []
    for name in cfg["paths"]["kaggle_files"]:
        path = ROOT / cfg["paths"]["kaggle_news_dir"] / name
        reader = pd.read_csv(path, usecols=RAW_COLUMNS, dtype=str, chunksize=CHUNK_ROWS)
        for chunk in reader:
            part = chunk[chunk["stock"].isin(codes)].copy()
            part["source_file"] = name
            parts.append(part)
    rows = pd.concat(parts, ignore_index=True)
    rows["ticker"] = rows["stock"].map(codes)
    return rows


def build_table(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Clean, align and de-duplicate the Kaggle rows; return the table and quality counts."""
    rows = read_rows(cfg)
    read = len(rows)
    rows = rows.dropna(subset=["headline", "date"]).copy()
    rows["headline"] = rows["headline"].map(clean_text)
    rows["key"] = rows["headline"].map(normalize)
    rows = rows[rows["key"] != ""].copy()
    usable = len(rows)
    aligned = rows["date"].map(align.available_date)
    rows["published"] = aligned.map(lambda pair: pair[0])
    rows["available_date"] = aligned.map(lambda pair: pair[1].isoformat())
    rows["timed"] = rows["date"].map(align.has_time_of_day)
    rows["order"] = range(len(rows))  # raw file order: more recent rows come first
    rows["day"] = rows["published"].str[:10]
    table = rows.drop_duplicates(["ticker", "day", "key"]).copy()
    stats = {
        "rows_for_tickers": read,
        "empty_removed": read - usable,
        "duplicates_removed": usable - len(table),
        "duplicate_pct": round(100 * (usable - len(table)) / usable, 2),
        "rows_kept": len(table),
        "timed_rows": int(table["timed"].sum()),
        "date_only_rows": int((~table["timed"]).sum()),
    }
    columns = ["ticker", "published", "available_date", "headline", "publisher", "source_file"]
    return table[columns + ["timed", "key", "order"]].fillna(""), stats


def save_table(cfg: dict) -> dict:
    """Build the table, write it to data/processed and return the quality counts."""
    table, stats = build_table(cfg)
    path = ROOT / cfg["paths"]["processed_dir"] / KAGGLE_TABLE
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return stats
