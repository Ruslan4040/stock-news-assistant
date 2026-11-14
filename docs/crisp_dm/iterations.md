# CRISP-DM iterations (loops back to an earlier phase)

Each entry: the trigger (a metric and its value), the phase revisited, the change, and the metric before → after. Numbers come from `logs/llm_calls.jsonl`, `data/processed/prepare_report.json` or `evals/results/`.

## Loop 1 — 2026-11-14 — Data Understanding → Business Understanding
- Trigger: 4 of the 8 tickers in the spec (AAPL, MSFT, AMZN, META) have 0 headlines for 2019 in the Kaggle files.
- Phase revisited: 1 (scope).
- Change: the four tickers were replaced by INTC, NFLX, AVGO and HD; `docs/PROJECT_SPEC.md` §3 and `config.yaml` were updated (decision by the human).
- Before → after: tickers with 2019 headlines 4 of 8 → 8 of 8; set A can now use the same tickers as set C.

## Loop 2 — 2026-11-14 — Evaluation of the LLM client → Modeling
- Trigger: in the first real smoke run, 3 of 8 answers were not valid JSON on the first try and 1 call still failed after the repair retry.
- Finding: the raw text was `{"items": [ {` and the API reported `finish_reason = error` (`UNAVAILABLE`). The provider had cut the answer short; the model had not made a formatting mistake.
- Phase revisited: 4 (LLM client).
- Change: an answer whose `finish_reason` is not `stop` is now a transport error and is retried with backoff (`src/llm/client.py`, test `test_truncated_answer_is_a_transport_error`).
- Before → after: answers invalid on the first try 3 of 8 → 0 of 371; calls failing after the repair retry 1 of 8 → 0 of 374.

## Loop 3 — 2026-11-14 — Evaluation of the LLM client → Modeling
- Trigger: the first full run left 6 of 40 pipeline tasks incomplete with `transport: HTTP 429`.
- Finding: the client cut error messages to 300 characters, which removed the "retry in N s" hint at the end of Google's quota message, so quota errors were treated as ordinary failures and used up the retries.
- Phase revisited: 4 (LLM client).
- Change: the full message is kept for parsing; a quota error pauses all threads and does not count as a retry; requests are paced at the quota rate of 30 per minute.
- Before → after: incomplete tasks 6 of 40 → 0 of 80. Throughput did not improve (9.1 → 7.3 successful calls per minute while other eval jobs shared the quota): a probe of 60 extra requests in one minute got 50 quota errors, so the limit is the free quota itself, with about 3 HTTP attempts per successful answer.
- Consequence for scope: the run uses 200 decisions per set instead of 300 (`evals.batch_limit`). The first 200 of a fixed random order are used, so the run can be extended later without changing earlier results.

## Loop 4 — 2026-11-15 — Evaluation of the advisor → Modeling (negative result, change rejected)
- Trigger: `advisor_v1` chose the 1-week horizon in 198 of 200 decisions of set A and 198 of 199 of set C (`evals/results/advisor.json`, details → horizons), so the horizon-choice metric measured almost nothing.
- Phase revisited: 4 (advisor prompt).
- Change tried: `prompts/advisor_v2.md` adds a rule for choosing the horizon (1d for a fresh event already moving the price, 1w for a slower story) and two short examples. Compared with `python -m evals.compare_advisor_prompts` on the first 50 decisions of sets A and C in the run order (`evals/results/advisor_prompt_comparison.json`).
- Before → after (v1 → v2):
  - share of 1d horizons: A 0.00 → 0.06, C 0.00 → 0.04;
  - share of buy/sell answers: A 0.62 → 0.78, C 0.52 → 0.84;
  - hit rate: A 0.613 (n 31) → 0.590 (n 39); C 0.538 (n 26) → 0.500 (n 42);
  - mean return with the chosen horizon, set C: +0.95% → −0.38%.
- Decision: v2 is rejected and `advisor_v1` stays in `config.yaml`. The horizon barely changed, and the extra buy/sell answers were less accurate. All CIs overlap, so v2 is not proven worse either, but it gives no evidence of an improvement worth an extra 400 calls of the free quota for the full run. Both prompt files are kept.
