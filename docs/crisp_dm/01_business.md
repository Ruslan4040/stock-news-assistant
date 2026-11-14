# CRISP-DM 1. Business Understanding
Purpose: goals, users, scope in/out, success criteria with numeric targets, risks.
Gate: the human approves the doc.

## 2026-11-14 — Project setup & model choice
- What was done: built the repo skeleton (AGENTS.md §3), `config.yaml`, pinned `requirements.txt` + `.venv`, `git init`. Checked the API keys and the Kaggle CSVs, and compared the free OpenRouter models.
- Decisions & why:
  - **Model:** `google/gemma-4-31b-it`. Its pre-training cutoff is published as Jan 2025 (HF model card), so all Finnhub news (~Nov 2025 → now) is unseen. Rejected: Nemotron 3.5 Lightning (post-training data up to May 2026, so a shorter set C; 3B active parameters). Kept as the fallback.
  - **Tickers:** AAPL/MSFT/AMZN/META → INTC/NFLX/AVGO/HD, because the first four have 0 headlines for 2019 in Kaggle. Rejected: keeping all 8 from the spec, which would put different tickers in A and C.
  - **Rate limits:** BYOK with a free Google AI Studio key in OpenRouter. Rejected: several accounts (OpenRouter ToS §7 forbids this, and it does not fix upstream 429s); $10 credits.
  - **Undated 2019 news:** assigned to the next trading day (`news.date_only_policy`), the safe side for look-ahead.
- Artifacts: `config.yaml`, `src/config.py`, `tests/test_config.py`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `docs/PROJECT_SPEC.md` §3 (ticker list).
- Evidence:
  - Kaggle rows: 1,407,328 (analyst) and 1,845,559 (partner). 100% of 2019 rows have no time of day.
  - 2019 rows per ticker: JPM 1724, INTC 1702, HD 1167, AVGO 1087, NFLX 801, GOOGL 767 (+GOOG 713), TSLA 733, NVDA 508.
  - Finnhub: AAPL, 1 week → 126 items.
  - Gemma 4 31B `:free`: 0/10 successful calls (upstream 429). Lightning: OK in 0.8 s.
  - `ruff` clean; `pytest`: 4 passed.
- Gate status: no. The business doc (goals, users, success criteria, risks) is the next task.
- Update (same day): the human added the Google AI Studio key (BYOK). `google/gemma-4-31b-it:free` + `provider.only: [google-ai-studio]` → 3 of 5 calls OK with `is_byok: true`, 2 × 503 "high demand"; the free-model counter stayed at 0/50 (BYOK calls do not use the 50/day quota), so the LLM client needs retries with backoff. The paid slug is not served by Google AI Studio (404), so `config.yaml` keeps the `:free` slug. Spec §2 and §5b example switched from AAPL to NVDA.
- Open questions for the human: none.
