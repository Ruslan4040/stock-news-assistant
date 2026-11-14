# AGENTS.md — Stock News Awareness Assistant

Rules for any AI coding agent working in this repo (Claude Code loads this file through `CLAUDE.md`).
`docs/PROJECT_SPEC.md` says WHAT we build. This file says HOW we build it. Read both before every new task.

## 1. Project in short
A Streamlit app. For a chosen ticker and decision date it collects prices and news headlines, filters the relevant headlines with an LLM, labels each one (positive / negative / neutral), and returns advice: action (buy / hold / sell), horizon (1 day / 1 week), confidence (0–100) and a short explanation. A simulator trades on high-confidence advice. Every component is evaluated, and a look-ahead experiment checks whether the LLM "remembers the future" from its training data.

## 2. CRISP-DM is mandatory
We work in CRISP-DM phases.
- Start every task with one line: `CRISP-DM phase: <number and name>`.
- Do not start a phase until the gate of the previous phase is met.
- When you finish work in a phase, update its log with the `crisp-dm-phase` skill.

| # | Phase | What the agent produces | Gate before the next phase | Log file |
|---|---|---|---|---|
| 1 | Business Understanding | goals, users, scope in/out, success criteria with numeric targets, risks | the human approves the doc | `docs/crisp_dm/01_business.md` |
| 2 | Data Understanding | download scripts; a data card per dataset: source, size, columns, date range, missing values, class balance, 5 examples | a data card exists for every dataset in the spec | `docs/crisp_dm/02_data.md` |
| 3 | Data Preparation | cleaning, ticker filtering, time alignment, anonymized set B, eval samples with a fixed seed | `pytest` passes, including the no-look-ahead test | `docs/crisp_dm/03_preparation.md` |
| 4 | Modeling | pipeline components (one per task), prompts, simulator | every component has a passing unit test | `docs/crisp_dm/04_modeling.md` |
| 5 | Evaluation | an eval script per component, results JSON, error analysis, the A/B/C/D experiment | every component has a metric and a baseline | `docs/crisp_dm/05_evaluation.md` |
| 6 | Deployment | Streamlit app with a trace panel, README run instructions, `docs/REPORT.md`, demo script | the app runs from a clean clone by following the README; the UI checklist passes | `docs/crisp_dm/06_deployment.md` |

CRISP-DM is a cycle. If Evaluation shows a problem, go back to the right phase, fix it, re-run the evals, and record the loop (trigger metric, change, before → after) in `docs/crisp_dm/iterations.md`.

## 3. Repository layout
```
app/streamlit_app.py      # UI only, no business logic
src/data/                 # download, load, clean, align (no LLM calls here)
src/llm/client.py         # the ONLY place that calls an LLM
src/pipeline/             # one module per component: relevance.py, sentiment.py, advisor.py, judge.py
src/sim/simulator.py      # trading simulator (pure Python)
prompts/                  # prompt templates with versions, e.g. sentiment_v1.md
evals/                    # eval_<component>.py, labels/, results/
tests/                    # pytest, LLM always mocked
data/raw/  data/processed/  logs/   # gitignored
docs/                     # spec, CRISP-DM logs, diagrams, demo script
config.yaml               # tickers, dates, model, thresholds, seed
```

## 4. Coding conventions
- Python 3.11. Type hints everywhere. Functions ≤ 40 lines. One-line docstrings.
- All parameters (tickers, dates, model name, thresholds, `seed: 42`) live in `config.yaml`. No magic numbers in code.
- Secrets live only in `.env` (`OPENROUTER_API_KEY`, `FINNHUB_API_KEY`). Keep `.env.example` up to date. Never print or commit keys.
- Pin dependencies in `requirements.txt`. Ask before adding a new dependency.
- `ruff check .` and `pytest -q` must pass before you call a task done.
- Prefer plain Python over frameworks. No LangChain or agent frameworks unless the human asks.

## 5. LLM rules
- Every LLM call goes through `src/llm/client.py`: OpenRouter, model taken from config, temperature 0, retry with backoff, and a disk cache keyed by (model, prompt version, input). The cache makes evals reproducible and saves the free-tier quota.
- Log every call to `logs/llm_calls.jsonl` (timestamp, component, prompt version, model, input, output, latency). This is our tracing for the demo.
- Prompts live in `prompts/*.md` with a version suffix. Never inline long prompts in code. To change a prompt, create a new version file and keep the old one.
- LLM output must be JSON validated with pydantic. On invalid output, retry once, then return an `error` result and count it in the evals.
- The LLM never does arithmetic. Returns, % changes and metrics are computed in Python and passed into the prompt.
- `model_id` in `config.yaml` is frozen once phase 5 starts. Never switch models silently: a model change invalidates every eval result.

## 6. Data rules
- Use the real datasets listed in the spec. Do not generate synthetic data unless the human approves; if approved, label it `synthetic` everywhere.
- `data/raw/` is read-only after download. Write outputs to `data/processed/`.
- Download once and cache to disk (Yahoo Finance and Finnhub rate-limit).
- NO LOOK-AHEAD: everything used for a decision at time t must have a timestamp ≤ t. News published after market close (16:00 US/Eastern) belongs to the next trading day. Keep a pytest that fails if this rule is broken.
- Set B (anonymized): replace the company name and ticker with "Company X", remove dates, give prices only as % changes.

## 7. Evaluation rules
- Every component has `evals/eval_<component>.py`. It writes `evals/results/<component>.json` with: metric values, n, 95% bootstrap confidence interval, baseline values, model, prompt version, run date.
- Always report a baseline next to every metric (baselines are listed in the spec).
- An LLM-as-judge must be checked against ≥ 20 human-labeled items. Report the agreement %.
- Save the 10 worst errors per component to `evals/results/<component>_errors.csv` for error analysis.
- Use the `run-evals` skill to run everything and refresh the results table in `README.md`.
- Never type a metric value by hand. Every number in docs comes from a results file.

## 8. How to work with the human
- Before coding a new component, post a short plan (files, functions, test, eval) and wait for "ok".
- One component per task. Do not touch files outside the task scope. Never delete data, cache or results.
- If something in the spec is unclear or contradictory, ask. Do not guess.
- When a task is done, report: changed files, test result, which CRISP-DM log you updated, open issues.
- `docs/agent_log.md` belongs to the human. Do not edit it.

## 9. Definition of done (every task)
Code + unit test + eval script (for components) + CRISP-DM log updated + README updated if run instructions changed.

## 10. Skills in this repo (`.claude/skills/`)
- `crisp-dm-phase` — update the CRISP-DM log for the current phase.
- `add-component` — add a pipeline component the standard way (schema, prompt, code, test, eval, docs).
- `run-evals` — run all evals and the look-ahead experiment, refresh the results table.
