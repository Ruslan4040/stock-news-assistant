---
name: add-component
description: Add a new pipeline component (relevance filter, sentiment labeler, advisor, explanation judge, LLM client) the standard way - schema, prompt, code, unit test, eval script, docs. Use whenever the user asks to build or add a part of the system.
---

# Add a pipeline component

Follow the steps in order. After step 1, stop and wait for the human's "ok".

1. **Plan (post it, then wait).** Component name and purpose; input and output (pydantic schemas); LLM call or pure code; prompt file name; test cases; eval metric + baseline (from docs/PROJECT_SPEC.md §7); list of files to create.
2. **Schema.** Define the input and output pydantic models in `src/pipeline/<name>.py`.
3. **Prompt (LLM components only).** Create `prompts/<name>_v1.md` with: role, task, rules (use only the given data, no arithmetic, answer in JSON), the output JSON schema, and 2–3 short examples. Load it through `src/llm/client.py`. Never inline it.
4. **Implementation.** One public function `run(input) -> output`. Handle invalid LLM output (retry once → `error` result). No business logic in the UI.
5. **Unit test.** `tests/test_<name>.py` with the LLM mocked: a normal case, an empty-input case, an invalid-LLM-output case.
6. **Eval script.** `evals/eval_<name>.py`:
   - loads the eval sample from `data/processed/` and runs the component;
   - computes the metric, the baseline and a 95% bootstrap CI;
   - writes `evals/results/<name>.json` and the 10 worst errors to `evals/results/<name>_errors.csv`;
   - supports `--limit N`. Do a smoke run with `--limit 20` first.
7. **Docs.** Add the component to the components table and the Mermaid diagram in `README.md`. Log the work with the `crisp-dm-phase` skill (phase 4).
8. **Check.** Run `ruff check .`, `pytest -q`, then the smoke eval. Report changed files and results.
