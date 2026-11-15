# CRISP-DM 5. Evaluation
Purpose: an eval script per component, results JSON, error analysis, the A/B/C/D experiment.
Gate: every component has a metric and a baseline.

## How to read the results
- One script per component: `evals/eval_<component>.py` → `evals/results/<component>.json` and `<component>_errors.csv` (the ten worst errors).
- Every metric has n and a 95% percentile bootstrap interval (1,000 resamples, seed 42).
- The full table is generated into `README.md` by `python -m evals.run_all`.
- The model (`llm.model_id` in `config.yaml`) is frozen from this phase on.

| Component | Eval data | Metrics | Baselines |
|---|---|---|---|
| Data collector | all prices, raw Kaggle rows, sets A/B/C | missing trading days, duplicates, look-ahead violations, anonymization leaks | target value 0 |
| Features | 15 reference cases from an independent implementation | exact match within 1e-6 | target 15 / 15 |
| LLM client | every call in `logs/llm_calls.jsonl` | valid JSON on the first answer, errors after retry, latency, cache hits | targets from phase 1 |
| Relevance filter | 100 labeled fresh headlines | precision, recall, F1 | keep all; keyword match |
| Sentiment labeler | PhraseBank sample (400) + 100 labeled fresh headlines | accuracy, macro-F1, confusion matrix | FinBERT; always neutral |
| Advisor | sets A, B, C | hit rate, return by horizon, hit rate per confidence bucket | always buy; same as last move; fixed 1d / 1w |
| Explanation judge | 100 explanations; claims of 20 labeled | claims supported; judge vs label agreement | – |
| Trader | advice of sets A, B, C | total return, Sharpe, max drawdown, trades, win rate | buy & hold |
| Look-ahead | sets A–D, memory probe, sentiment gap | decision by the rule of spec §8 | post-cutoff control for the probe |
| UI | 3 demo cases | checklist pass rate | – |

## Log

## 2026-11-15 — All component evals, look-ahead experiment, one improvement loop
- What was done:
  - Pipeline run on 200 decisions per set (A, B, C) and live advice for set D; every eval script run on the results; `evals/run_all.py --table` refreshed `README.md`.
  - Labels drafted by the assistant for the 100 fresh headlines and the 109 claims of 20 explanations (`labeled_by = assistant_draft`).
  - Judge checked with planted false claims (wrong 20-day return or an invented buyback) in addition to label agreement, because almost all real claims are true.
  - Loop 4: `advisor_v2` (horizon rule) compared with v1 on 50 decisions per set and rejected (`iterations.md`).
- Decisions & why:
  - 200 instead of 300 decisions per set: the free quota gives about 8 answers a minute. Rejected: a smaller sample for every set, or paying for an API.
  - Planted-claim test added to the judge eval. Rejected: reporting only label agreement, which equals the "always supported" baseline (0.991).
  - v2 of the advisor not adopted: horizon barely changed, extra directional calls were less accurate. Rejected: running v2 on all 400 decisions without evidence of a gain.
- Artifacts: `evals/results/*.json`, `*_errors.csv`, `evals/labels/judge_claims.csv`, `evals/eval_judge.py` (planted claims, always-supported baseline), `evals/eval_lookahead.py` (A/B same-action share), `evals/eval_llm_client.py` (counts), `docs/templates/REPORT.md`.
- Evidence: `evals/results/targets.json` — 13 of 13 phase-1 criteria met on the point estimate. Advisor on set C 0.531 vs always buy 0.454 (baseline inside the CI); memory probe 2019 0.36 vs 0 after the cutoff; A and B give the same action in 0.93 of decisions → "the LLM peeks: no". Judge caught 20 of 20 planted claims. `pytest`: 138 passed; `ruff`: clean.
- Gate status: yes — every component has a metric, a baseline (or a fixed target for deterministic code), n and a CI.
- Open questions for the human: review `evals/labels/fresh_headlines.csv` and `judge_claims.csv`, then `python -m evals.confirm_labels` and `python -m evals.run_all`. Set D outcomes need a re-run of `eval_lookahead` after 2026-11-17.

## 2026-11-15 — Human review of the labels
- What was done: the human reviewed `evals/labels/fresh_headlines.csv` (100) and `judge_claims.csv` (109); `python -m evals.confirm_labels` set `labeled_by = human_reviewed`; `python -m evals.run_all` re-ran every eval.
- Decisions & why: one change, F003 "Is a 100-Year Bond Worth Buying?" (GOOGL) relevant n → y, because the bond in the news is Alphabet's. Judge claims unchanged. Report wording now says "drafted by the assistant, reviewed by the human", not "labeled by the human".
- Artifacts: both label files, `evals/results/*.json`, `relevance_errors.csv`, `docs/REPORT.md`, `docs/REPORT.pdf`, `docs/demo_script.md`, README table.
- Evidence: values from `evals/results/relevance.json` and `targets.json` after the run (relevance confusion tp 39, fp 2, fn 1); targets met 13 of 13.
- Gate status: yes.
- Open questions for the human: score set D after the close of 2026-11-17.
