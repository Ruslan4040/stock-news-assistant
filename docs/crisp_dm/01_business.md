# CRISP-DM 1. Business Understanding
Purpose: goals, users, scope in/out, success criteria with numeric targets, risks.
Gate: the human approves the doc.

## Phase document

### Goal
Help a retail investor follow one company at a time: show the price trend, the news that really concerns the company with a sentiment label, and one explained suggestion (buy / hold / sell, 1 day or 1 week, confidence 0–100). Then measure honestly whether the suggestions are any good, and whether the LLM only looks good on old data because it memorized the future.

The second half matters as much as the first. A stock assistant that cannot show how often it is right is not useful, and an LLM evaluated on 2019 news may simply remember what happened next.

### Users
| User | What they need | Where they get it |
|---|---|---|
| Retail investor (primary) | a quick, explained view of one stock for today or a past date | Advice tab: chart, labeled headlines, advice card |
| Course evaluator / developer | proof that every step works and how well | Evals tab, trace panel, `evals/results/`, `docs/REPORT.md` |

### Scope
**In:** 8 US large caps (GOOGL, NVDA, TSLA, JPM, INTC, NFLX, AVGO, HD); headlines only; daily prices; horizons 1 day and 1 week; historical mode and live mode; a long-only trading simulator; evaluation of every component; the look-ahead experiment (sets A, B, C).

**Out:** real trading, intraday data, short selling, portfolio optimization, full article text, fine-tuning, any claim of financial advice.

### Success criteria
Targets are set before any evaluation is run. Every metric is reported with n, a 95% bootstrap CI and a baseline.

| Component | Metric | Target |
|---|---|---|
| Data collector | missing trading days; duplicate headlines after cleaning; look-ahead violations | 0%; 0%; 0 |
| Features | exact match with the reference values (tolerance 1e-6), 15 cases | 15 / 15 |
| LLM client | valid JSON on the first answer; errors left after the retry | ≥ 95%; ≤ 2% |
| Relevance filter | F1 and precision on 100 labeled fresh headlines | F1 ≥ 0.80, precision ≥ 0.80, F1 above "keep all" |
| Sentiment labeler | macro-F1 on PhraseBank (n = 400) | ≥ 0.80, above "always neutral", within 0.05 of FinBERT |
| Sentiment labeler | macro-F1 on 100 labeled fresh headlines | ≥ 0.70 |
| Advisor: direction | hit rate on set C (unseen period) | above "always buy" and "same as last move" on the same rows |
| Advisor: horizon | mean return with the chosen horizon | ≥ the better of fixed 1d and fixed 1w |
| Advisor: confidence | hit rate per confidence bucket | bucket 80–100 at least 5 points above bucket 50–60 |
| Explanation | claims supported by the inputs; judge vs human agreement | ≥ 90%; ≥ 80% |
| Simulator | total return after fees on set C; max drawdown | return > 0; drawdown smaller than buy & hold |
| Look-ahead experiment | decision by the rule in spec §8 | a clear yes / no with CIs |
| UI | checklist on 3 demo cases | 3 / 3 |

Predicting the direction of a large-cap stock from headlines is close to a coin flip, so the advisor targets are ambitious on purpose. Missing them is an acceptable result if the report explains why.

### Risks
| Risk | Why it matters | Mitigation |
|---|---|---|
| Look-ahead bias in the data | one leaked headline or price makes every later metric too good | news after the close goes to the next trading day; undated news goes to the next trading day; outcomes are stored in a separate file; a pytest fails on any violation |
| LLM memorization | Gemma was trained on text up to Jan 2025 and may "know" 2019 outcomes | the A / B / C / D experiment, the direct memory probe and the sentiment gap check |
| Free-tier limits | Google allows 30 requests per minute and about 40% of calls fail with 500 / 503 | disk cache, retries with backoff, a rate limiter, resumable evals, `--limit 20` smoke runs |
| Data coverage | Kaggle has no 2019 news for AAPL, MSFT, AMZN, META; Finnhub gives only one year | tickers replaced by ones with coverage; set C uses the Finnhub year |
| Label quality | the 100 fresh labels and the 20 judge checks are drafted by the coding assistant | each label file carries `labeled_by`; results say whether the human has reviewed them |
| Small samples | n ≈ 300 gives a CI of about ± 6 points on a hit rate | CIs are always shown; conclusions follow the CI rule, not point estimates |
| Different market regimes | 2019 and 2025–26 differ, so A vs C is not a clean comparison | A vs B uses the same rows; baselines are computed on the same rows as the advisor |
| FinBERT baseline leak | FinBERT was fine-tuned on PhraseBank | compare on the fresh headlines too and say so in the analysis |
| Misuse as financial advice | users may act on a suggestion | the UI and README state that this is a course prototype, not advice |

## Log

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

## 2026-11-14 — Business Understanding document
- What was done: wrote the phase document above (goal, users, scope, success criteria with numeric targets, risks) from `docs/PROJECT_SPEC.md` and the facts found during setup.
- Decisions & why:
  - Targets are fixed before any eval runs, so results cannot shape them. Alternative rejected: setting targets after the first eval.
  - Advisor targets are relative to baselines on the same rows, not absolute numbers, because a hit rate depends on the market period. Alternative rejected: a fixed "≥ 55%".
  - Label provenance is recorded per file (`labeled_by`). The human asked the assistant to draft the 100 fresh labels and the 20 judge checks and to hand them over for review. Alternative rejected: reporting them as labeled by the human from scratch.
- Artifacts: `docs/crisp_dm/01_business.md` (section "Phase document").
- Evidence: 13 success criteria, each with a metric and a numeric target; 9 risks, each with a mitigation. Measured limits used in the risks: 30 requests/min (Google free tier quota message), 11 of 16 calls OK in an 8-worker burst (4 × 500, 1 × 503), Finnhub history starts 2025-11-19.
- Gate status: yes. On 2026-11-14 the human asked for all remaining phases in one run and approved the overall plan, which includes this document. The human can still change targets; any change will be logged here.
- Open questions for the human: none.
