"""Company name terms from config: used by anonymization and by the keyword baseline."""

import re
from functools import lru_cache

from src.config import load_config


def term_pattern(terms: list[str]) -> re.Pattern | None:
    """Build one regex for a list of names, longest first, on word boundaries."""
    if not terms:
        return None
    ordered = "|".join(map(re.escape, sorted(set(terms), key=len, reverse=True)))
    return re.compile(rf"(?<![A-Za-z0-9])(?:{ordered})(?![A-Za-z0-9])", re.I)


def company_terms(ticker: str) -> list[str]:
    """Return every configured name that points to the company: names, people, products."""
    info = load_config()["tickers"][ticker]
    return [info["name"], *info["aliases"], *info.get("people", []), *info.get("identity", [])]


@lru_cache(maxsize=64)
def _company_pattern(ticker: str) -> re.Pattern:
    """Return the cached regex that finds any name of the company."""
    return term_pattern(company_terms(ticker))


def mentions_company(ticker: str, text: str) -> bool:
    """Say whether a text names the company, one of its people or one of its products."""
    return bool(_company_pattern(ticker).search(text))
