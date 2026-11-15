# Agent log (human-owned — the agent must not edit this file)

How I steered the coding agent. This is the source for the "use of the AI coding assistant" part of the report and video.

## Setup
- Agent: Claude Code (reads `CLAUDE.md`, which imports `AGENTS.md` + `docs/PROJECT_SPEC.md`).
- Context control: rules in `AGENTS.md`, scope in the spec, one CRISP-DM phase per session (`/clear` between phases), skills in `.claude/skills/` for repeated workflows.
- Quality control: plan-before-code rule, a unit test + an eval per component, ruff + pytest gate, a no-look-ahead test.

## Log
| Date | Phase / task | What the agent did wrong or unexpectedly | What I changed (AGENTS.md / skill / prompt) | Result |
|---|---|---|---|---|
| 2026-11-14 | 2 Data | The data check showed 0 headlines for 2019 for AAPL, MSFT, AMZN and META in the Kaggle file. Without this check set A would be half empty. | I replaced them with INTC, NFLX, AVGO, HD in the spec and `config.yaml`. | 8 of 8 tickers have 2019 news. |
| 2026-11-14 | 4 LLM setup | The free OpenRouter limit is 50 requests a day, the run needs about 2,400. My idea with several accounts is against the OpenRouter terms. | I connected my own Google AI Studio key to OpenRouter (BYOK). | No daily limit, 30 requests a minute. |
| 2026-11-14 | 4 LLM client | The client counted answers as "invalid JSON" from the model. The real cause: the provider cut the answers short. | I asked the agent to show the raw answers. Cut answers are now API errors and are retried. A test covers it. | Invalid first answers: 3 of 8 → 0 of 371. |
| 2026-11-14 | 4 LLM client | Quota errors (HTTP 429) used up all retries. The client shortened the error text and lost the "retry in N s" part. | Keep the full error text, pause all threads on a quota error, send at most 30 requests a minute. | Unfinished runs: 6 of 40 → 0 of 80. |
| 2026-11-14 | 4 Pipeline run | The free API gave about 8 answers a minute. 300 decisions per set would not finish before the deadline. | 200 decisions per set: the first 200 of a fixed random order. | The run finished; it can be extended later without changing results. |
| 2026-11-14 | 3 Data prep | Building the news table took 170 s: the config file was read again for every row. | Read the config once (cache). | 2.6 s. |
| 2026-11-14 | 3–4 Tests | Twice the agent wrote a test with a wrong hand-computed expected value. The code was right. | I made it recompute the expected values instead of changing the code to pass the test. | The arithmetic is written in the test comments. |
| 2026-11-15 | 5 Labels | I had no time to label 100 headlines and 109 judge claims myself. | The agent drafted the labels by written rules (`labeled_by = assistant_draft`). I checked both files and fixed 1 label: F003 "Is a 100-Year Bond Worth Buying?" is about Alphabet's bond, so relevant. | Files marked `human_reviewed`, evals re-run. |
| 2026-11-15 | 5 Judge | The judge agreed with the labels in 99% of claims, but a judge that always says "supported" gets the same number. So it proved nothing. | Added a test with planted false claims (wrong 20-day return, invented buyback). | Caught 20 of 20. |
| 2026-11-15 | 5 Advisor | `advisor_v1` chose the 1-week horizon in 198 of 200 decisions, so the horizon metric measured almost nothing. | Tried `advisor_v2` with a horizon rule and two examples. | Hit rate on set C went down (0.538 → 0.500 on 50 decisions). I rejected v2 and kept v1. |

## Reflection (fill in at the end)
- What worked well:
  - Rules in files, not in chat. `AGENTS.md` and the spec reload every session, so the agent kept the CRISP-DM gates and the no-look-ahead rule all the time.
  - Rules checked by tests: function length, docstrings, no look-ahead. The agent cannot break them silently.
  - All numbers in README and the report come from results files. Nobody types a metric by hand.
  - The disk cache for LLM calls: re-running evals costs no quota.
  - Asking the agent to check data and API limits before planning. The two biggest changes (tickers, API key) came from that.
- What did not work:
  - The free API is slow and unstable: about 4 HTTP attempts per answer.
  - Because of the deadline I ran all phases in one long session, not one session per phase. The context had to be summarized, and I had to re-check what was done.
  - The agent first blamed the model for API problems. It found the real cause only after I asked for the raw answers.
  - A metric can look great and mean nothing (judge agreement) until you put a trivial baseline next to it.
- What I would put into AGENTS.md from day one next time:
  - Check data coverage and API limits before writing the spec.
  - Read the raw API answer before calling an output invalid.
  - Every agreement or accuracy metric needs a trivial baseline.
  - Size the eval sets by the API quota and the deadline.
