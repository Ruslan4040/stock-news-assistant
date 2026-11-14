# CRISP-DM 4. Modeling
Purpose: pipeline components (one per task), prompts, simulator.
Gate: every component has a passing unit test.

## Components
| # | Component | Code | Prompt | Unit test |
|---|---|---|---|---|
| 1 | Data collector | `src/data/collector.py` | none (code) | `tests/test_no_lookahead.py` |
| 2 | Features | `src/data/features.py` | none (code) | `tests/test_features.py` |
| – | LLM client | `src/llm/client.py`, `src/llm/trace.py` | loads `prompts/*.md` | `tests/test_client.py` |
| 3 | Relevance filter | `src/pipeline/relevance.py` | `prompts/relevance_v1.md` | `tests/test_pipeline.py` |
| 4 | Sentiment labeler | `src/pipeline/sentiment.py`, baselines in `sentiment_baselines.py` | `prompts/sentiment_v1.md` | `tests/test_pipeline.py` |
| 5 | Advisor | `src/pipeline/advisor.py` | `prompts/advisor_v1.md` | `tests/test_pipeline.py` |
| 6 | Trader (simulator) | `src/sim/simulator.py` | none (code) | `tests/test_simulator.py` |
| 7 | Explanation judge | `src/pipeline/judge.py` | `prompts/judge_v1.md` | `tests/test_pipeline.py` |
| – | Orchestrator, live rule, batch runs | `src/pipeline/{run,live,batch}.py` | none | `tests/test_pipeline.py`, `tests/test_live.py` |

## How one LLM call works
1. The component builds a JSON input and names a prompt version from `config.yaml` → `prompts`.
2. `client.call_llm` looks in the disk cache (key: model + prompt version + prompt text + input).
3. On a miss it calls OpenRouter (temperature 0, one pinned provider, JSON mode) under a rate limiter.
4. API failures (5xx, timeouts, answers cut short) are retried with backoff. A quota error pauses all threads.
5. The answer is validated with pydantic plus one rule per component (for example: one item per input id).
6. An invalid answer gets one repair try that shows the model its error. If it is still invalid, the component returns an `error` result.
7. Every call is appended to `logs/llm_calls.jsonl` with a `trace_id` that groups the calls of one request.

## Log

## 2026-11-14 — Client, four LLM components, simulator
- What was done:
  - `src/llm/client.py` with cache, retries, rate limiting, JSON validation, one repair retry and the trace log.
  - Relevance, sentiment, advisor and judge: pydantic schemas, `prompts/*_v1.md`, a `run()` function each, tests with the LLM mocked (normal, empty input, invalid output).
  - `src/sim/simulator.py` (pure Python) with the rules of spec §6 and a buy & hold baseline; tests on a hand-made 7-day price series with expected values worked out by hand.
  - `src/pipeline/batch.py` runs sets A, B and C in parallel; it is resumable through the cache.
- Decisions & why:
  - The user message of every call is plain JSON; the prompt file is the system message. Rejected: text templates with placeholders, which need formatting code per component.
  - An answer cut short by the provider (`finish_reason` ≠ stop) is a transport error and is retried. Rejected: counting it as invalid JSON, which would blame the model for an API failure.
  - A failed language step leads to price-only advice with `news_used = false`. In batch runs the advisor is skipped instead, so a later run can repeat the missing call. Rejected: keeping all headlines when the filter fails.
  - Set B reuses the relevance and sentiment labels of set A; only the advisor input is anonymized. Rejected: re-running the two language steps on anonymized text (600 more calls, and A and B would then differ in more than one thing).
  - Live rule: a refresh costs one relevance call; sentiment and advisor run only when a new relevant headline appeared.
- Artifacts: the files in the table above, `prompts/{relevance,sentiment,advisor,judge,memory_probe}_v1.md`, `tests/conftest.py` (LLM mock used by every test).
- Evidence:
  - `pytest`: 123 passed; `ruff check .`: clean.
  - Real smoke run on 12 decisions (4 per set): 0 incomplete, all advice valid.
  - Measured on the free API: 30 requests per minute (quota message), about 3 HTTP attempts per successful call, about 30 s per successful answer; answers are identical across repeats at temperature 0.
- Gate status: yes.
- Open questions for the human: none.
