# CRISP-DM 3. Data Preparation
Purpose: cleaning, ticker filtering, time alignment, anonymized set B, eval samples with a fixed seed.
Gate: `pytest` passes, including the no-look-ahead test.

## Prepared files
| File | Content | Built by |
|---|---|---|
| `data/processed/kaggle_headlines.csv` | Kaggle headlines of the 8 tickers, cleaned, de-duplicated, with `available_date` | `src/data/kaggle.py` |
| `data/processed/set_a.jsonl` | 300 decisions from 2019, real names and dates | `src/data/prepare.py` |
| `data/processed/set_b.jsonl` | the same 300 decisions, anonymized | `src/data/anonymize.py` |
| `data/processed/set_c.jsonl` | 300 decisions from the fresh period (after the model's knowledge cutoff) | `src/data/prepare.py` |
| `data/processed/outcomes.csv` | forward 1-day and 1-week returns per decision (ground truth, kept apart from inputs) | `src/data/prepare.py` |
| `data/processed/phrasebank_sample.csv` | 400 PhraseBank sentences, seed 42 | `src/data/phrasebank.py` |
| `evals/labels/fresh_headlines.csv` | 100 fresh headlines to label (`relevant`, `sentiment`) | `src/data/prepare.py` |
| `data/processed/prepare_report.json` | quality counts of all of the above | `src/data/prepare.py` |

## The no-look-ahead rule in code
1. A headline with a time is compared with the real NYSE close of that day (`src/data/calendar.py`, including 13:00 early closes). After the close it belongs to the next trading day.
2. A headline without a time (all Kaggle rows of 2019) always belongs to the next trading day.
3. A decision at the close of day t uses only headlines whose `available_date` is in [t − 3 days, t], and prices up to t.
4. `collector.check_no_lookahead` raises if a decision breaks rule 3. `tests/test_no_lookahead.py` checks the rule on hand-made cases and on every decision of the real sets A and C.
5. Forward returns live only in `outcomes.csv`. The decision schema has no field for them.

## Log

## 2026-11-14 — Cleaning, time alignment and eval sets
- What was done:
  - `src/data/kaggle.py` filters the Kaggle CSVs to the 8 tickers, merges GOOG into GOOGL, removes duplicates and gives each headline its first usable trading day.
  - `src/data/collector.py` (component 1) builds a decision for a ticker and a day: features up to the close plus the 15 most recent usable headlines. The same function feeds the eval sets and the app.
  - `src/data/prepare.py` builds sets A, B, C, the outcomes file, the PhraseBank sample and the labeling sheet.
- Decisions & why:
  - Timed headlines use the real close of the day. Rejected: a fixed 16:00, which would leak news on half-days.
  - Decision points are sampled only from days with at least one usable headline, with an equal share per ticker. Rejected: all ticker-days, where many decisions would test only the price-only path.
  - Outcomes are stored apart from the decisions. Rejected: one file with inputs and outcomes.
  - Set B hides company names, people, products, dates and dollar amounts with rules built from `config.yaml`. It is best effort: other companies named in a headline stay.
  - `tests/test_conventions.py` enforces the AGENTS.md §4 rules (≤ 40 lines, docstring, return type) on every function.
- Artifacts: `src/schemas.py`, `src/data/{align,features,anonymize,collector,prepare,sets}.py`, `tests/test_{align,features,anonymize,sampling_prices,no_lookahead,conventions}.py`, `evals/labels/fresh_headlines.csv`.
- Evidence (from `data/processed/prepare_report.json` and pytest):
  - Set A: 300 decisions, 3,013 headlines (10.04 per decision); 4.12 per decision name the company; 29 decisions have none that do.
  - Set C: 300 decisions, 4,465 headlines (14.88 per decision); 5.85 name the company. The 15 headlines span a median of 1.4 hours for NVDA and 51.5 hours for HD.
  - Set B: 0 leaks of names, years or dollar amounts in 3,013 headlines. Missing outcomes: 0.
  - PhraseBank sample: 244 neutral, 115 positive, 41 negative.
  - `pytest`: 64 passed, including no-look-ahead on all 600 real decisions of sets A and C.
- Gate status: yes.
- Open questions for the human: none.
