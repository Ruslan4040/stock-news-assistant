"""Smoke tests for config.yaml and repo hygiene."""

from src.config import ROOT, load_config

REQUIRED_SECTIONS = {
    "seed", "tickers", "dates", "market", "news", "llm", "simulator", "evals", "paths",
}


def test_config_has_required_sections() -> None:
    """All top-level sections exist."""
    assert REQUIRED_SECTIONS <= load_config().keys()


def test_seed_and_llm_settings() -> None:
    """Seed is fixed and the LLM is deterministic and documented."""
    cfg = load_config()
    assert cfg["seed"] == 42
    llm = cfg["llm"]
    assert llm["model_id"]
    assert llm["temperature"] == 0
    assert llm["model_release_date"]
    assert "knowledge_cutoff" in llm


def test_every_ticker_has_a_name() -> None:
    """The relevance filter and anonymization need a company name per ticker."""
    tickers = load_config()["tickers"]
    assert 5 <= len(tickers) <= 10
    for ticker, info in tickers.items():
        assert info["name"], ticker


def test_secrets_and_data_are_gitignored() -> None:
    """Keys and raw data must never be committed."""
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines
    assert "data/raw/" in lines
