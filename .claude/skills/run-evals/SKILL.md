---
name: run-evals
description: Run all component evaluations and the look-ahead (memorization) experiment, then refresh the results table in README.md and the evaluation log. Use when the user asks to run evals, update metrics, or compare before/after a change.
---

# Run all evals

1. Check that `.env` has the API keys and that the eval samples exist in `data/processed/`. If something is missing, stop and say what.
2. Run every `evals/eval_*.py`: first a smoke run with `--limit 20`, then the full run. Use cached LLM responses; never clear the cache unless asked.
3. Run the look-ahead experiment from docs/PROJECT_SPEC.md §8:
   - advisor + simulator on sets A, B, C, D;
   - the direct memory probe;
   - the sentiment gap check (PhraseBank vs fresh human labels).
4. Collect all `evals/results/*.json` into one table: component | metric | value | 95% CI | baseline | n | prompt version | model.
5. Replace the table between `<!-- EVALS:START -->` and `<!-- EVALS:END -->` in `README.md`.
6. Append to `docs/crisp_dm/05_evaluation.md`:
   - the table;
   - what is good and what is bad;
   - for every metric below its baseline or with a wide CI: a likely reason (look at `*_errors.csv`) and one improvement idea.
7. If results changed since the previous run, list before → after for every changed metric.
8. Never type a number by hand. Every number must come from a results file.
