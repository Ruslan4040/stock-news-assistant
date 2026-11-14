"""Download raw data once: prices, PhraseBank, Finnhub news windows, yfinance news snapshot.

Run: python -m src.data.download [--only prices phrasebank finnhub yf_news kaggle]
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config
from src.data import news, prices, sampling

STEPS = ["kaggle", "prices", "phrasebank", "yf_news", "finnhub"]


def check_kaggle(cfg: dict) -> dict:
    """Check that the human placed the Kaggle CSV files; return their sizes in bytes."""
    folder = ROOT / cfg["paths"]["kaggle_news_dir"]
    sizes = {}
    for name in cfg["paths"]["kaggle_files"]:
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Download the Kaggle dataset first (README).")
        sizes[name] = path.stat().st_size
    return sizes


def download_prices(cfg: dict) -> dict:
    """Download daily prices per ticker (skips tickers that are already on disk)."""
    folder = ROOT / cfg["paths"]["prices_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(cfg["dates"]["price_start"]).date()
    end = pd.Timestamp(cfg["dates"]["prices_end"]).date()
    rows = {}
    for ticker in cfg["tickers"]:
        path = folder / f"{ticker}.csv"
        if not path.exists():
            prices.fetch(ticker, start, end).to_csv(path)
        rows[ticker] = len(pd.read_csv(path))
    return rows


def download_phrasebank(cfg: dict) -> dict:
    """Download the Financial PhraseBank archive from Hugging Face."""
    from huggingface_hub import hf_hub_download

    spec = cfg["evals"]["phrasebank"]
    folder = ROOT / cfg["paths"]["phrasebank_dir"]
    path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", local_dir=folder)
    return {"file": str(Path(path).relative_to(ROOT)), "bytes": Path(path).stat().st_size}


def download_yf_news(cfg: dict) -> dict:
    """Save one snapshot of yfinance news per ticker (the live-mode backup source)."""
    import yfinance as yf

    folder = ROOT / cfg["paths"]["yf_news_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    counts = {}
    for ticker in cfg["tickers"]:
        path = folder / f"{ticker}.json"
        if not path.exists():
            path.write_text(json.dumps(yf.Ticker(ticker).news or []), encoding="utf-8")
        counts[ticker] = len(json.loads(path.read_text(encoding="utf-8")))
    return counts


def download_finnhub(cfg: dict) -> dict:
    """Download the Finnhub news days needed by the sampled set-C decisions."""
    raw_dir = cfg["paths"]["finnhub_dir"]
    limit = cfg["news"]["max_headlines"]
    points = sampling.fresh_points(cfg)
    for i, (ticker, day) in enumerate(points, start=1):
        lower, upper = news.window_bounds(day)
        news.finnhub_rows(ticker, lower, upper, limit, fetch_to=raw_dir)
        if i % 50 == 0:
            print(f"  finnhub: {i}/{len(points)} decision windows", flush=True)
    files = list((ROOT / raw_dir).glob("*/*.json"))
    return {"decision_windows": len(points), "day_files": len(files)}


def main() -> None:
    """Run the selected download steps and store a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=STEPS, default=STEPS)
    args = parser.parse_args()
    cfg = load_config()
    actions = {
        "kaggle": check_kaggle,
        "prices": download_prices,
        "phrasebank": download_phrasebank,
        "yf_news": download_yf_news,
        "finnhub": download_finnhub,
    }
    path = ROOT / cfg["paths"]["raw_dir"] / "download_summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for step in args.only:
        print(f"[download] {step} ...", flush=True)
        summary[step] = {"at": datetime.now(UTC).isoformat(timespec="seconds"), **actions[step](cfg)}
        print(f"[download] {step}: {summary[step]}", flush=True)
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
