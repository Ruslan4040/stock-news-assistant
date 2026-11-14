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
