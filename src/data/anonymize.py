"""Anonymization for set B: hide the company, people, products, dates and dollar amounts."""

import re

from src.config import load_config
from src.data.terms import company_terms, term_pattern
from src.schemas import Decision, Headline

COMPANY = "Company X"
TICKER = "X"
PERSON = "an executive"
ENTITY = "a related business"

_MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_DAY = r"\.?\s+\d{1,2}(?!\d)(?:st|nd|rd|th)?(?:,?\s*\d{4})?"
_DATE_RULES = [
    (re.compile(rf"\b(?:{_MONTHS}|May)\.?\s+(?:19|20)\d{{2}}\b", re.I), "[date]"),
    (re.compile(rf"\b(?:{_MONTHS}|May){_DAY}", re.I), "[date]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"), "[date]"),
    (re.compile(r"\bFY\s?'?\d{2,4}\b", re.I), "FY"),
    (re.compile(r"\b(Q[1-4])\s?'?\d{2,4}\b", re.I), r"\1"),
    (re.compile(r"\b(?:19|20)\d{2}\b"), "[year]"),
    (re.compile(rf"\b(?:{_MONTHS})\b\.?"), "[month]"),
]
_UNITS = r"[KMBT]\b|bn\b|million\b|billion\b|trillion\b"
_MONEY = re.compile(rf"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:{_UNITS}))?", re.I)
_EXCHANGE = re.compile(rf"\(\s*(?:(?:NASDAQ|NYSE)\s*:\s*)?{COMPANY}\s*\)")


def _replace_terms(text: str, terms: list[str], replacement: str) -> str:
    """Replace every listed name with a neutral word."""
    pattern = term_pattern(terms)
    return pattern.sub(replacement, text) if pattern else text


def anonymize_text(text: str, ticker: str) -> str:
    """Hide who the headline is about and when it was written."""
    info = load_config()["tickers"][ticker]
    text = _replace_terms(text, [info["name"], *info["aliases"]], COMPANY)
    text = _replace_terms(text, info.get("people", []), PERSON)
    text = _replace_terms(text, info.get("identity", []), ENTITY)
    text = _EXCHANGE.sub("", text)
    text = _MONEY.sub("$N", text)
    for pattern, replacement in _DATE_RULES:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def find_leaks(text: str, ticker: str) -> list[str]:
    """Return identity, year or dollar-amount fragments that survived anonymization."""
    leaks = term_pattern(company_terms(ticker)).findall(text)
    leaks += re.findall(r"\b(?:19|20)\d{2}\b", text)
    leaks += re.findall(r"\$\s?\d", text)
    return leaks


def anonymize_decision(decision: Decision) -> Decision:
    """Return the set-B twin of a set-A decision: same rows, no name, no dates."""
    headlines = [
        Headline(id=h.id, text=anonymize_text(h.text, decision.ticker)) for h in decision.headlines
    ]
    return Decision(
        id="B" + decision.id[1:],
        set="B",
        ticker=TICKER,
        company=COMPANY,
        date=None,
        features=decision.features,
        headlines=headlines,
    )
