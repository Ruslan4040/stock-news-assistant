"""Generate the data cards in docs/crisp_dm/02_data.md from the files on disk.

Run: python -m src.data.cards
Every number in the cards is computed here; nothing is typed by hand.
"""

import json
from datetime import UTC, datetime

import pandas as pd

from src.config import ROOT, load_config
from src.data import calendar as cal
from src.data import kaggle, phrasebank, prices
from src.data.terms import mentions_company

DOC = "docs/crisp_dm/02_data.md"
START, END = "<!-- CARDS:START -->", "<!-- CARDS:END -->"
EXAMPLES = 5
MAX_CELL = 400  # characters; longer cells are cut in the cards


def _cell(value: object) -> str:
    """Make a value safe and short enough for one table cell."""
    text = str(value).replace("|", "/").replace("\n", " ")
    return text if len(text) <= MAX_CELL else text[: MAX_CELL - 1] + "…"


def md_table(rows: list[dict]) -> str:
    """Render a list of dicts as a Markdown table."""
    if not rows:
        return "_no rows_"
    head = list(rows[0])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        cells = [_cell(row[column]) for column in head]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def card(title: str, facts: dict, examples: list[dict], notes: list[str]) -> str:
    """Assemble one data card: facts table, examples and notes."""
    facts_md = md_table([{"Field": key, "Value": value} for key, value in facts.items()])
    notes_md = "\n".join(f"- {note}" for note in notes)
    return f"### {title}\n{facts_md}\n\n**Examples**\n\n{md_table(examples)}\n\n**Notes**\n{notes_md}\n"


def prices_card(cfg: dict) -> str:
    """Describe the daily price files."""
    frames = {t: prices.load_prices(t) for t in cfg["tickers"]}
    first = min(f.index[0] for f in frames.values()).date()
    last = max(f.index[-1] for f in frames.values()).date()
    expected = len(cal.trading_days(first, last))
    missing_days = {t: expected - len(f) for t, f in frames.items()}
    nans = sum(int(f.isna().sum().sum()) for f in frames.values())
    sample = frames["NVDA"].tail(EXAMPLES).round(4).reset_index()
    sample["date"] = sample["date"].dt.date
    facts = {
        "Source": "Yahoo Finance via `yfinance` (`auto_adjust=False`, `actions=True`)",
        "Files": f"`{cfg['paths']['prices_dir']}/<TICKER>.csv`, {len(frames)} tickers",
        "Size": f"{sum(len(f) for f in frames.values())} rows ({expected} per ticker)",
        "Columns": ", ".join(["date", *frames["NVDA"].columns]),
        "Date range": f"{first} → {last}",
        "Missing values": f"{nans} NaN cells",
        "Missing trading days (vs NYSE calendar)": sum(missing_days.values()),
    }
    notes = [
        "`adj_close` (splits and dividends) is used for returns, features and the simulator.",
        "`raw_close` = `close` × later split ratios: the price people saw that day. "
        "It is the truth for the memory probe.",
    ]
    return card("1. Daily prices", facts, sample.to_dict("records"), notes)


def _naming_share(frame: pd.DataFrame) -> dict:
    """Return, per ticker, the percentage of headlines that name the company."""
    named = [mentions_company(t, h) for t, h in zip(frame["ticker"], frame["headline"], strict=True)]
    share = pd.Series(named, index=frame["ticker"].values).groupby(level=0).mean() * 100
    return share.round(0).astype(int).to_dict()


def kaggle_card(cfg: dict) -> str:
    """Describe the Kaggle headlines for the config tickers."""
    rows = kaggle.read_rows(cfg)
    summary = json.loads((ROOT / cfg["paths"]["raw_dir"] / "download_summary.json").read_text())
    megabytes = {k: round(v / 1e6) for k, v in summary["kaggle"].items() if k.endswith(".csv")}
    year = str(cfg["dates"]["set_a_year"])
    in_year = rows[rows["date"].str.startswith(year)]
    timed = in_year["date"].str.contains(r"[+-]\d\d:\d\d$").sum()
    stats = kaggle.build_table(cfg)[1]
    duplicates = f"{stats['duplicates_removed']} ({stats['duplicate_pct']}%)"
    facts = {
        "Source": "Kaggle `miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests`",
        "Files (MB)": ", ".join(f"`{name}` {mb}" for name, mb in megabytes.items()),
        "Rows for our tickers (all years)": len(rows),
        "Columns used": "headline, publisher, date, stock (plus url, unused)",
        "Date range (our tickers)": f"{rows['date'].min()[:10]} → {rows['date'].max()[:10]}",
        f"Rows in {year}": len(in_year),
        f"Rows per ticker in {year}": in_year["ticker"].value_counts().to_dict(),
        f"Rows in {year} with a time of day": int(timed),
        "Missing values": rows[["headline", "publisher", "date"]].isna().sum().to_dict(),
        "Duplicates (same ticker, day, headline)": duplicates,
        f"Headlines in {year} that name the company (%)": _naming_share(in_year),
        "Top publishers": in_year["publisher"].value_counts().head(5).to_dict(),
    }
    sample = in_year.sample(EXAMPLES, random_state=cfg["seed"])
    examples = sample[["ticker", "date", "publisher", "headline"]].to_dict("records")
    notes = [
        f"No {year} row has a time of day, so every headline goes to the next trading day.",
        "GOOG rows are merged into GOOGL; the same headline often appears under both codes.",
        "Most headlines do not even name the company (see the percentages): the source tags a "
        "ticker on market wrap-ups, lists and fund notes. This is the noise the relevance filter "
        "must remove.",
    ]
    return card("2. Historical headlines (Kaggle)", facts, examples, notes)


def _finnhub_items(cfg: dict) -> pd.DataFrame:
    """Load every downloaded Finnhub item into one frame."""
    rows = []
    for path in sorted((ROOT / cfg["paths"]["finnhub_dir"]).glob("*/*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            rows.append({**item, "ticker": path.parent.name, "day": path.stem})
    return pd.DataFrame(rows)


def finnhub_card(cfg: dict) -> str:
    """Describe the downloaded Finnhub news days."""
    items = _finnhub_items(cfg)
    per_day = items.groupby(["ticker", "day"]).size()
    capped = int((per_day >= cfg["news"]["finnhub_page_cap"]).sum())
    stamps = pd.to_datetime(items["datetime"], unit="s", utc=True)
    facts = {
        "Source": "Finnhub `company-news` API (free key)",
        "Files": f"`{cfg['paths']['finnhub_dir']}/<TICKER>/<UTC day>.json`, "
                 f"{len(per_day)} ticker-days with news",
        "Size": f"{len(items)} news items",
        "Columns": ", ".join(c for c in items.columns if c not in ("ticker", "day")),
        "Date range (UTC)": f"{stamps.min():%Y-%m-%d} → {stamps.max():%Y-%m-%d}",
        "Items per ticker": items["ticker"].value_counts().to_dict(),
        "Median items per ticker-day": per_day.groupby("ticker").median().astype(int).to_dict(),
        "Ticker-days at the API cap": capped,
        "Headlines that name the company (%)": _naming_share(items),
        "Missing values": {"empty headline": int((items["headline"].str.strip() == "").sum()),
                           "empty summary": int((items["summary"].str.strip() == "").sum())},
        "Top sources": items["source"].value_counts().head(5).to_dict(),
    }
    sample = items.sample(EXAMPLES, random_state=cfg["seed"]).assign(
        time_utc=lambda f: pd.to_datetime(f["datetime"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
    )
    examples = sample[["ticker", "time_utc", "source", "headline"]].to_dict("records")
    notes = [
        "Only the days needed by the sampled set-C decisions were downloaded, not the whole year.",
        "One request returns at most about 250 items, most recent first. On capped days the "
        "earliest items are cut; the collector needs only the most recent ones before the close.",
        f"History starts at {cfg['news']['finnhub_history_start']}, after the model's knowledge "
        f"cutoff ({cfg['llm']['knowledge_cutoff']}), so every item is unseen by the model.",
        "Busy tickers get about 200 items a day, so the 15 most recent headlines cover only "
        "the last hours before the close (numbers in the phase 3 log).",
    ]
    return card("3. Fresh headlines (Finnhub)", facts, examples, notes)


def yf_news_card(cfg: dict) -> str:
    """Describe the yfinance news snapshot (live backup source)."""
    rows = []
    for ticker in cfg["tickers"]:
        path = ROOT / cfg["paths"]["yf_news_dir"] / f"{ticker}.json"
        for item in json.loads(path.read_text(encoding="utf-8")):
            content = item.get("content", item)
            provider = (content.get("provider") or {}).get("displayName")
            rows.append({"ticker": ticker, "pubDate": content.get("pubDate"),
                         "provider": provider, "title": content.get("title")})
    frame = pd.DataFrame(rows)
    facts = {
        "Source": "`yfinance` `Ticker.news` (snapshot taken at download time)",
        "Files": f"`{cfg['paths']['yf_news_dir']}/<TICKER>.json`",
        "Size": f"{len(frame)} items ({frame.groupby('ticker').size().min()} per ticker)",
        "Fields used": "content.title, content.pubDate, content.provider.displayName",
        "Date range": f"{frame['pubDate'].min()[:10]} → {frame['pubDate'].max()[:10]}",
        "Missing values": frame.isna().sum().to_dict(),
    }
    examples = frame.sample(EXAMPLES, random_state=cfg["seed"]).to_dict("records")
    notes = ["Used only in live mode when Finnhub returns nothing. Not used in any eval set."]
    return card("4. Live headlines backup (yfinance)", facts, examples, notes)


def phrasebank_card(cfg: dict) -> str:
    """Describe the Financial PhraseBank file used as the sentiment exam."""
    frame = phrasebank.load_phrasebank()
    spec = cfg["evals"]["phrasebank"]
    words = frame["sentence"].str.split().str.len()
    balance = (frame["label"].value_counts(normalize=True) * 100).round(1).to_dict()
    facts = {
        "Source": f"Hugging Face `{spec['repo']}`, file `{spec['member']}` (≥ 75% annotators agree)",
        "License": "CC BY-NC-SA 3.0 (non-commercial; the file is not committed)",
        "Size": f"{len(frame)} sentences",
        "Columns": "sentence, label",
        "Date range": "no dates in the file (the dataset was published in 2014)",
        "Missing values": int(frame.isna().sum().sum()),
        "Class balance (%)": balance,
        "Words per sentence": f"median {int(words.median())}, max {int(words.max())}",
        "Eval sample": f"n = {spec['n']}, seed {cfg['seed']}",
    }
    examples = frame.sample(EXAMPLES, random_state=cfg["seed"]).to_dict("records")
    notes = [
        "Classes are imbalanced (mostly neutral), so macro-F1 is reported next to accuracy.",
        "FinBERT (`ProsusAI/finbert`) was fine-tuned on this dataset, so its score here is "
        "optimistic. The fresh labeled headlines give the fair comparison.",
    ]
    return card("5. Sentiment exam (Financial PhraseBank)", facts, examples, notes)


def labels_card(cfg: dict) -> str:
    """Describe the labeled fresh headlines, once the sheet exists."""
    path = ROOT / cfg["paths"]["labels_dir"] / "fresh_headlines.csv"
    title = "6. Labeled fresh headlines"
    if not path.exists():
        return f"### {title}\n_Not created yet: the sheet is sampled in phase 3._\n"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    labeled = frame[frame["relevant"] != ""]
    facts = {
        "Source": "sampled from the set-C headlines (Finnhub), evenly per ticker",
        "File": f"`{cfg['paths']['labels_dir']}/fresh_headlines.csv`",
        "Size": f"{len(frame)} headlines, {len(labeled)} labeled",
        "Columns": ", ".join(frame.columns),
        "Date range": f"{frame['published'].min()[:10]} → {frame['published'].max()[:10]}",
        "Per ticker": frame["ticker"].value_counts().to_dict(),
        "Labeled by": labeled["labeled_by"].value_counts().to_dict(),
        "Class balance: relevant": labeled["relevant"].value_counts().to_dict(),
        "Class balance: sentiment": labeled["sentiment"].value_counts().to_dict(),
    }
    examples = frame.head(EXAMPLES)[["id", "ticker", "headline", "relevant", "sentiment"]]
    notes = [
        "`labeled_by` says who produced the labels: `assistant_draft` (drafted by the coding "
        "assistant, not yet checked) or `human_reviewed` (checked and approved by the human).",
        "Labeling rules are in `evals/labels/README.md`.",
    ]
    return card(title, facts, examples.to_dict("records"), notes)


def main() -> None:
    """Rebuild all cards and place them between the markers of the phase log."""
    cfg = load_config()
    builders = [prices_card, kaggle_card, finnhub_card, yf_news_card, phrasebank_card, labels_card]
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    body = f"_Generated by `python -m src.data.cards` on {stamp}._\n\n"
    body += "\n".join(build(cfg) for build in builders)
    path = ROOT / DOC
    text = path.read_text(encoding="utf-8")
    before, rest = text.split(START, 1)
    after = rest.split(END, 1)[1]
    path.write_text(f"{before}{START}\n{body}\n{END}{after}", encoding="utf-8")
    print(f"Data cards written to {DOC}")


if __name__ == "__main__":
    main()
