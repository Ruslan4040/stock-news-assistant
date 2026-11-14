"""Finnhub company-news access: one UTC day per request, stored on disk (no LLM calls)."""

import json
import os
import time
from datetime import UTC, date, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.config import ROOT, load_config

_last_call = [0.0]


def _throttle(rpm: int) -> None:
    """Sleep so that calls stay under the per-minute limit."""
    wait = 60.0 / rpm - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def fetch_day(ticker: str, day: date) -> list[dict]:
    """Download the company news of one UTC day (most recent first, capped by the API)."""
    cfg = load_config()["news"]
    load_dotenv(ROOT / ".env")
    headers = {"X-Finnhub-Token": os.environ["FINNHUB_API_KEY"]}
    params = {"symbol": ticker, "from": day.isoformat(), "to": day.isoformat()}
    status = 0
    for attempt in range(cfg["finnhub_retries"]):
        _throttle(cfg["finnhub_rpm"])
        resp = requests.get(cfg["finnhub_url"], params=params, headers=headers, timeout=30)
        status = resp.status_code
        if status == 200:
            return resp.json()
        time.sleep(cfg["finnhub_backoff"] * (attempt + 1))
    raise RuntimeError(f"Finnhub failed for {ticker} {day}: HTTP {status}")


def _path(base: str, ticker: str, day: date) -> Path:
    """Return the file that stores one ticker-day."""
    return ROOT / base / ticker / f"{day.isoformat()}.json"


def cache_dir() -> str:
    """Return the on-demand cache folder used by the app (never data/raw)."""
    return load_config()["paths"]["news_cache_dir"] + "/finnhub"


def read_day(ticker: str, day: date) -> list[dict] | None:
    """Return a stored day from the raw download or the app cache, or None when absent."""
    for base in (load_config()["paths"]["finnhub_dir"], cache_dir()):
        path = _path(base, ticker, day)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def get_day(ticker: str, day: date, fetch_to: str | None) -> list[dict]:
    """Return one UTC day of news; when missing, fetch it and store it under `fetch_to`."""
    stored = read_day(ticker, day)
    if stored is not None:
        return stored
    if fetch_to is None:
        raise FileNotFoundError(f"No stored Finnhub news for {ticker} {day}. Run the download.")
    items = fetch_day(ticker, day)
    if day < datetime.now(UTC).date():  # a day that is still running is never stored
        path = _path(fetch_to, ticker, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items), encoding="utf-8")
    return items
