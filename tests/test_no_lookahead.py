"""NO LOOK-AHEAD: nothing published after the decision close may reach a decision."""

import json
from datetime import date

import pandas as pd
import pytest

from src.data import align, news, sets
from src.data import calendar as cal
from src.data.collector import check_no_lookahead, collect_historical, decision_day
from src.schemas import Decision, Features, Headline

FEATURES = Features(
    ret_1d=0.0, ret_5d=0.0, ret_20d=0.0, volatility_20d=0.01, volatility_level="medium"
)


def kaggle_row(published: str, available: str, text: str, order: int) -> dict:
    """Build one row of the cleaned Kaggle table."""
    return {"ticker": "NVDA", "published": published, "available_date": available,
            "headline": text, "publisher": "Benzinga", "source_file": "x.csv",
            "timed": False, "key": news.normalize(text), "order": order}


def test_kaggle_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only rows available in [t - 3 days, t] are returned, most recent first, no duplicates."""
    table = pd.DataFrame([
        kaggle_row("2019-05-15", "2019-05-16", "Published on the decision day, undated", 0),
        kaggle_row("2019-05-14", "2019-05-15", "Usable: available on the decision day", 1),
        kaggle_row("2019-05-14", "2019-05-15", "usable: available on the decision day.", 2),
        kaggle_row("2019-05-10", "2019-05-13", "Usable: Friday news, available Monday", 3),
        kaggle_row("2019-05-09", "2019-05-10", "Too old for the window", 4),
    ])
    monkeypatch.setattr(news, "_kaggle_by_ticker", lambda: {"NVDA": table})
    rows = news.kaggle_rows("NVDA", date(2019, 5, 15), limit=15)
    assert [r["order"] for r in rows] == [1, 3]
    assert news.kaggle_rows("NVDA", date(2019, 5, 15), limit=1)[0]["order"] == 1
    assert news.kaggle_rows("TSLA", date(2019, 5, 15), limit=15) == []


def finnhub_item(stamp: str, text: str) -> dict:
    """Build one Finnhub item from a UTC time string."""
    return {"datetime": int(pd.Timestamp(stamp, tz="UTC").timestamp()), "headline": text,
            "source": "Yahoo"}


def test_finnhub_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Items after the decision close or before the window start are dropped."""
    days = {
        date(2026, 3, 11): [finnhub_item("2026-03-11 20:00:01", "One second after the close"),
                            finnhub_item("2026-03-11 19:59:00", "Just before the close")],
        date(2026, 3, 6): [finnhub_item("2026-03-06 21:00:01", "After Friday's close"),
                           finnhub_item("2026-03-06 20:59:00", "Before Friday's close")],
    }
    monkeypatch.setattr(news.finnhub, "get_day", lambda ticker, day, fetch_to: days.get(day, []))
    lower, upper = news.window_bounds(date(2026, 3, 11))
    assert upper == cal.close_utc("2026-03-11") == pd.Timestamp("2026-03-11 20:00", tz="UTC")
    assert lower == pd.Timestamp("2026-03-06 21:00", tz="UTC")
    rows = news.finnhub_rows("NVDA", lower, upper, limit=15, fetch_to=None)
    assert [r["headline"] for r in rows] == ["Just before the close", "After Friday's close"]


def test_collector_rejects_late_headline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collector itself refuses a decision that contains a leaked headline."""
    index = pd.bdate_range(end="2019-05-15", periods=30)
    frame = pd.DataFrame({"adj_close": [100.0 + i for i in range(30)]}, index=index)
    late = Headline(id=1, text="Tomorrow's news", published="2019-05-15",
                    available_date="2019-05-16")
    monkeypatch.setattr("src.data.collector.prices.load_prices", lambda ticker: frame)
    monkeypatch.setattr("src.data.collector.news.headlines_for", lambda *args: [late])
    with pytest.raises(ValueError, match="Look-ahead"):
        collect_historical("NVDA", date(2019, 5, 15))
    on_time = late.model_copy(update={"available_date": "2019-05-15"})
    monkeypatch.setattr("src.data.collector.news.headlines_for", lambda *args: [on_time])
    decision = collect_historical("NVDA", date(2019, 5, 15), set_name="A")
    assert decision.id == "A-NVDA-2019-05-15"
    assert decision.features.ret_1d == pytest.approx(129 / 128 - 1)


def test_check_no_lookahead_and_decision_day() -> None:
    """The guard raises on a late headline; requested dates snap back to a trading day."""
    late = Headline(id=7, text="x", available_date="2019-05-16")
    decision = Decision(id="A-NVDA-2019-05-15", set="A", ticker="NVDA", company="NVIDIA",
                        date="2019-05-15", features=FEATURES, headlines=[late])
    with pytest.raises(ValueError, match=r"\[7\]"):
        check_no_lookahead(decision)
    assert decision_day(date(2019, 5, 18)) == date(2019, 5, 17)
    assert decision_day(date(2030, 1, 1)) == date(2026, 11, 13)


@pytest.mark.parametrize("letter", ["A", "C"])
def test_prepared_sets_have_no_lookahead(letter: str) -> None:
    """Real data: every headline of every decision was available at the decision close."""
    if not sets.processed(f"set_{letter.lower()}.jsonl").exists():
        pytest.skip("prepared sets are not on disk (run src.data.prepare)")
    for decision in sets.load_set(letter):
        day = date.fromisoformat(decision.date)
        assert cal.is_trading_day(day)
        for h in decision.headlines:
            assert align.is_usable(h.available_date, day), (decision.id, h.id)
            if "T" in h.published:
                assert pd.Timestamp(h.published) <= cal.close_utc(day), (decision.id, h.id)
            else:
                assert date.fromisoformat(h.published) < day, (decision.id, h.id)


def test_prepared_inputs_carry_no_outcomes_or_identity() -> None:
    """Real data: inputs hold no future returns; set B holds no dates."""
    if not sets.processed("set_b.jsonl").exists():
        pytest.skip("prepared sets are not on disk (run src.data.prepare)")
    for letter in "abc":
        for line in sets.processed(f"set_{letter}.jsonl").read_text().splitlines():
            assert "fwd_ret" not in line
            assert set(json.loads(line)) == set(Decision.model_fields)
    for decision in sets.load_set("B"):
        assert decision.date is None and decision.ticker == "X"
        assert all(h.published is None and h.available_date is None for h in decision.headlines)
