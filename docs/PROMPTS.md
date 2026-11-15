# Requests to the coding agent (in order)

How to use: one request per step. Start each new phase in a fresh session (`/clear`). `CLAUDE.md` reloads the rules and the spec automatically.
After each step, check the result. If the agent did something wrong, fix the rule in `AGENTS.md` or in a skill (not only in the chat), and note it in `docs/agent_log.md`.

---

## Before step 0 (you, ~30 min)
- Install Python 3.11+ and Claude Code. Create a GitHub repo and copy these files into it.
- Get free API keys: OpenRouter and Finnhub.
- Download the Kaggle dataset "Daily Financial News for 6000+ Stocks" (needs a Kaggle account) and put the CSV files into `data/raw/kaggle_news/`.
- Choose one free model on https://openrouter.ai/collections/free-models. Note its id and release date (and knowledge cutoff, if the model page shows it). Give these to the agent in step 2.

## 0. Kick-off (no code)
> Read AGENTS.md and docs/PROJECT_SPEC.md. Do not write code. Reply with: (1) a 5-bullet summary of the system, (2) your task plan per CRISP-DM phase, (3) questions about anything unclear or contradictory.

## 1. Business Understanding
> CRISP-DM phase 1. Create docs/crisp_dm/01_business.md from the spec: goal, users, scope in/out, success criteria with numeric targets, risks (look-ahead bias, LLM memorization, free-tier limits, rate limits). Stop and wait for my review.

## 2. Data Understanding
> CRISP-DM phase 2. Create the project skeleton from AGENTS.md §3 (folders, config.yaml, requirements.txt, .env.example, .gitignore, empty README with an `<!-- EVALS:START --><!-- EVALS:END -->` block). In config.yaml set model_id = <MODEL_ID>, model_release_date = <DATE>, knowledge_cutoff = <DATE or null>. Then write src/data/download.py for: yfinance prices for the config tickers, the Kaggle CSV I placed in data/raw/, PhraseBank from Hugging Face, Finnhub company news for the last 12 months. Run it. Write a data card per dataset in docs/crisp_dm/02_data.md. No modeling yet.

## 3. Data Preparation
> CRISP-DM phase 3. Implement src/data/prepare.py: filter headlines to the config tickers, deduplicate, align headlines to trading days (after 16:00 US/Eastern → next trading day), compute features, and build the eval samples: set A (2019), set B (A anonymized), set C (after the training cutoff in config), PhraseBank sample n=400 seed 42, and a CSV of 100 fresh headlines for me to label (evals/labels/fresh_headlines.csv, empty label columns). Add a pytest that fails on look-ahead. Update the CRISP-DM log.

(Now label the 100 headlines yourself, ~1 hour.)

## 4. Modeling (one request per component)
> Use the add-component skill to build src/llm/client.py (OpenRouter wrapper with cache, retries, JSON validation and logging to logs/llm_calls.jsonl). Plan first.

> Use the add-component skill to build the relevance filter. Plan first.

> Use the add-component skill to build the sentiment labeler, plus the FinBERT and "always neutral" baselines. Plan first.

> Use the add-component skill to build the advisor (action, horizon 1d/1w, confidence, explanation, used headline ids). Plan first.

> CRISP-DM phase 4. Build src/sim/simulator.py exactly by the simulator rules in spec §6 (pure Python, no LLM), with unit tests on a tiny hand-made price series.

> Use the add-component skill to build the explanation judge (eval only). Plan first.

## 5. Evaluation
> Use the run-evals skill: run all component evals, the A/B/C experiment, the memory probe and the sentiment gap check. Then explain in docs/crisp_dm/05_evaluation.md which metrics are weak and why (use the *_errors.csv files).

> CRISP-DM loop: pick the weakest component, make one improvement (e.g. prompt v2 with few-shot examples), re-run its eval, and log before → after in docs/crisp_dm/iterations.md.

## 6. Deployment
> CRISP-DM phase 6. Build app/streamlit_app.py: ticker picker; mode (historical date / live now); price chart; headline table with colored sentiment; advice card; "refresh now" button; trace panel showing the LLM calls of the current request (from logs/llm_calls.jsonl); simulator tab (equity curve vs buy & hold); evals tab (results table). Run the UI checklist from spec §7 on 3 demo cases. Update README: setup, run commands, Mermaid diagram, business rules, results.

> Write docs/REPORT.md following spec §10 (summarize from docs/crisp_dm/ and evals/results/, never invent numbers). Then write docs/demo_script.md for a ≤10-minute video in the order given in spec §10. I will write the agentic-setup reflection part myself from docs/agent_log.md.

> Final check: clone the repo into a temp folder, follow the README from scratch, and fix anything that breaks.
