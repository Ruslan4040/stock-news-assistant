"""Anonymization for set B hides names, dates and dollar amounts."""

import pytest

from src.data.anonymize import anonymize_decision, anonymize_text, find_leaks
from src.schemas import Decision, Features, Headline

CASES = [
    ("NVDA", "Benchmark Maintains Buy on NVIDIA, Raises Price Target to $275"),
    ("TSLA", "Top 5 Stocks To Watch For Tues., Dec. 24, 2019: TSLA, WCC, ACST"),
    ("TSLA", "Lawyer Says Elon Musk Should Pay At Least $190M In Damages Over Model 3"),
    ("GOOGL", "Google Fined $170M By FTC Over Child Privacy Violations By YouTube"),
    ("AVGO", "Broadcom Inc (AVGO) CEO Hock E Tan Sold $5.4 million of Shares in FY20"),
    ("JPM", "JPMorgan Chase & Co. (NYSE: JPM) Q3 2019 Earnings Beat; Dimon Upbeat by March 2020"),
    ("HD", "The Home Depot's 2019 guide on 11/19: comps up 3.2%"),
]


@pytest.mark.parametrize(("ticker", "text"), CASES)
def test_no_leaks(ticker: str, text: str) -> None:
    """No company name, person, product, year or dollar amount survives."""
    hidden = anonymize_text(text, ticker)
    assert find_leaks(hidden, ticker) == []
    assert hidden != text


def test_meaning_is_kept() -> None:
    """Direction words and percentages stay; only identity is removed."""
    hidden = anonymize_text("Nvidia Shares Fall 5% After Q2 2019 Sales Miss", "NVDA")
    assert hidden == "Company X Shares Fall 5% After Q2 Sales Miss"


def test_anonymize_decision() -> None:
    """The set-B twin has no ticker, no date and no publication dates."""
    features = Features(
        ret_1d=0.01, ret_5d=0.02, ret_20d=0.03, volatility_20d=0.015, volatility_level="medium"
    )
    headline = Headline(
        id=1, text="Tesla Beats Q3 2019 Estimates", published="2019-10-23",
        available_date="2019-10-24", source="Benzinga",
    )
    a = Decision(id="A-TSLA-2019-10-24", set="A", ticker="TSLA", company="Tesla, Inc.",
                 date="2019-10-24", features=features, headlines=[headline])
    b = anonymize_decision(a)
    assert (b.id, b.set, b.ticker, b.company, b.date) == (
        "B-TSLA-2019-10-24", "B", "X", "Company X", None)
    assert b.features == a.features
    assert b.headlines[0].model_dump() == {
        "id": 1, "text": "Company X Beats Q3 Estimates", "published": None,
        "available_date": None, "source": None,
    }
