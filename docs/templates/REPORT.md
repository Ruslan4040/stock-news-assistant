# Stock News Awareness Assistant — Report

Assignment 1, "Build an NLP System". Repository: https://github.com/Ruslan4040/stock-news-assistant

Every number in this report is filled in from a file in `evals/results/` by `python -m evals.render_docs`. Results date: {{advisor:#run_date}}. Model: `{{advisor:#model}}`.

## 1. Task and scope

**Task.** Help a retail investor follow one company: show the price trend, the news that really concerns the company with a sentiment label, and one explained suggestion (buy / hold / sell, for 1 day or 1 week, with a confidence from 0 to 100). Then measure honestly whether the suggestions are any good, and whether the LLM only looks good on old data because it memorized what happened next.

**In scope.** {{data:@tickers}} US large caps (GOOGL, NVDA, TSLA, JPM, INTC, NFLX, AVGO, HD); headlines only; daily prices; a historical mode and a live mode; a long-only trading simulator; an evaluation of every component; a look-ahead (memorization) experiment.

**Out of scope.** Real trading, intraday data, short selling, portfolio optimization, full article text, fine-tuning.

**Business rules kept in scope.**
- A decision at the close of day t uses prices up to t and headlines of the last 3 days. News published after the close, or without a time, belongs to the next trading day.
- The advisor may use only what is in its prompt and must list the headlines it relied on. With no relevant news it advises from price features only and says so (`news_used = false`).
- The LLM never calculates. Returns and volatility are computed in Python.
- Live mode: one advice per ticker per day, plus a new one when a new relevant headline appears.
- Simulator: $10,000, long only, buys when the advice is `buy` with confidence ≥ 70, 20% of equity per trade, holds for the advised horizon or until a later `sell`, fee 0.1% per trade.

## 2. Data

| Dataset | Source | Size used | Used for |
|---|---|---|---|
| Daily prices | Yahoo Finance (`yfinance`), 2018-01-02 → 2026-11-13 | {{data:missing_trading_days_rate:n|int}} rows, {{data:@tickers}} tickers | features, ground truth, simulator |
| Historical headlines | Kaggle "Daily Financial News for 6000+ Stocks" | {{data:@kaggle.rows_for_tickers|int}} rows for our tickers, {{data:@kaggle.rows_in_set_a_year|int}} in 2019 | sets A and B |
| Fresh headlines | Finnhub `company-news` (free key) | {{data:@finnhub.items|int}} items in {{data:@finnhub.day_files}} ticker-days, Nov 2025 → Nov 2026 | set C, live mode |
| Live headlines backup | `yfinance` `Ticker.news` | 10 items per ticker | live mode only |
| Sentiment exam | Financial PhraseBank, ≥ 75% annotator agreement | {{data:@phrasebank_sentences|int}} sentences, sample of {{sentiment:phrasebank.accuracy:n}} | sentiment eval |
| Labeled fresh headlines | sampled from the Finnhub headlines | {{relevance:f1:n}} headlines: relevant (y/n) + sentiment | relevance eval, fresh sentiment eval |

No synthetic data is used. Existing datasets were checked first and four were rejected: FNSPID (about 30 GB, too heavy), NIFTY (market level, not per company), EDT (event detection, a different task) and DLT-Sentiment-News (crypto only).

**What Data Understanding changed in the project.**
- AAPL, MSFT, AMZN and META have no 2019 headlines in the Kaggle files. They were replaced by INTC, NFLX, AVGO and HD so that sets A and C use the same tickers.
- No 2019 headline has a time of day, so every 2019 headline is assigned to the next trading day. This is the safe side for look-ahead.
- Most headlines tagged with a ticker are not about the company. On average only {{data:@prepare_report.set_a.headlines_naming_company_mean}} of {{data:@prepare_report.set_a.headlines_per_decision_mean}} headlines per 2019 decision even name it. This is why the relevance filter exists.
- {{data:duplicate_headlines_rate_raw|pct}} of the raw rows are duplicates (the same headline under GOOG and GOOGL, for example). They are removed.

**Evaluation sets.** A: {{data:@prepare_report.set_a.decisions}} decisions (ticker, day) sampled from 2019 with seed 42. B: the same decisions with company names, people, products, dates and dollar amounts removed. C: {{data:@prepare_report.set_c.decisions}} decisions from Nov 2025 – Nov 2026, after the model's knowledge cutoff (January 2025). The free API allowed {{advisor:@A.decisions}} decisions per set to go through the LLM; they are the first ones of a fixed random order.

## 3. Components

```mermaid
flowchart LR
    U[User: ticker + date / now] --> C[1. Data collector]
    P[(Prices: yfinance)] --> C
    N[(News: Kaggle / Finnhub / yfinance)] --> C
    C --> F[2. Features - code]
    C --> R[3. Relevance filter - LLM]
    R --> S[4. Sentiment - LLM]
    F --> A[5. Advisor - LLM]
    S --> A
    A --> UI[8. Streamlit UI]
    A --> SIM[6. Trader / simulator - code]
    SIM --> UI
    A -.eval only.-> J[7. Explanation judge - LLM]
```

| # | Component | Purpose | Calls | Input | Output |
|---|---|---|---|---|---|
| 1 | Data collector | give the pipeline only what was known at decision time | external APIs: yfinance, Finnhub; local files | ticker, date | prices up to t, up to 15 most recent headlines of [t − 3 days, t] |
| 2 | Features | turn prices into trend numbers, because LLMs are bad at arithmetic | code | prices | 1-, 5-, 20-day returns, 20-day volatility and its level |
| 3 | Relevance filter | drop headlines that are not about the company | LLM call 1 | company, aliases, headlines | per headline: relevant yes / no + reason |
| 4 | Sentiment labeler | say whether each relevant headline is good or bad for the stock | LLM call 2 | company, relevant headlines | per headline: positive / negative / neutral + reason |
| 5 | Advisor | combine trend and news into one explained, calibrated suggestion | LLM call 3 | features in percent, labeled headlines | action, horizon, confidence, explanation, ids of headlines used |
| 6 | Trader | measure whether confident advice makes money | code | advice stream, prices | trades, equity curve, metrics |
| 7 | Explanation judge | check that explanations do not invent facts | LLM call 4, evaluation only | explanation + the advisor's inputs | per claim: supported yes / no |
| 8 | UI | show everything and the trace of one request | Streamlit | user input | chart, headline table, advice card, trace panel, simulator tab, evals tab |

All four LLM calls go through one client (`src/llm/client.py`): OpenRouter, temperature 0, a disk cache keyed by model, prompt version and input, retries with backoff, a rate limiter, JSON validation with pydantic, one repair retry on invalid output, and a trace log. Prompts are versioned files in `prompts/`.

### Walkthrough of one input (real values)
Input: `{{walkthrough:@ticker}}`, `{{walkthrough:@date}}`, historical mode.

1. **Collector.** Loads prices up to {{walkthrough:@date}} and {{walkthrough:@headlines}} headlines. The 2019 headlines have no time, so each one counts from the trading day after its date.
2. **Features (code).** 1-day return {{walkthrough:@features.ret_1d_pct}}%, 5-day {{walkthrough:@features.ret_5d_pct}}%, 20-day {{walkthrough:@features.ret_20d_pct}}%, volatility {{walkthrough:@features.volatility_20d_pct}}% ({{walkthrough:@features.volatility_level}}).
3. **Relevance (LLM call 1).** Keeps {{walkthrough:@relevant}} of {{walkthrough:@headlines}} headlines.
4. **Sentiment (LLM call 2).** {{walkthrough:@sentiment_counts.positive}} positive, {{walkthrough:@sentiment_counts.negative}} negative, {{walkthrough:@sentiment_counts.neutral}} neutral.

{{walkthrough:@headline_table|raw}}

5. **Advisor (LLM call 3).** `{{walkthrough:@advice.action}}`, horizon `{{walkthrough:@advice.horizon}}`, confidence {{walkthrough:@advice.confidence}}, headlines used: {{walkthrough:@advice.used_headlines|raw}}.
   > {{walkthrough:@advice.explanation|raw}}
6. **Trader.** {{walkthrough:@trader|raw}}.
7. **Judge (evaluation only).** {{walkthrough:@judge_supported}} of {{walkthrough:@judge_claims}} claims in the explanation are supported by the inputs.
8. **UI.** Shows the chart, the table above with tinted sentiment cells, the advice card and the trace of the LLM calls of this request.
9. **Ground truth.** The price moved {{walkthrough:@outcome.fwd_1d|pct}} over the next day and {{walkthrough:@outcome.fwd_1w|pct}} over the next week, so this advice counts as: **{{walkthrough:@outcome.verdict|raw}}**.

## 4. How CRISP-DM was applied

The phases and their gates are written into `AGENTS.md`, so the coding agent had to name the phase at the start of every task and could not start a phase before the previous gate was met. Each phase has a log in `docs/crisp_dm/`.

| Phase | What was produced | Gate | Evidence |
|---|---|---|---|
| 1. Business Understanding | goal, users, scope, 13 success criteria with numeric targets, 9 risks | the human approves the document | `01_business.md` |
| 2. Data Understanding | download scripts, six data cards generated by code | a card for every dataset | `02_data.md` |
| 3. Data Preparation | cleaning, time alignment, sets A / B / C, outcomes kept apart from inputs | `pytest` passes, including no-look-ahead | `03_preparation.md` |
| 4. Modeling | LLM client, four LLM components, simulator, prompts v1 | a passing unit test per component | `04_modeling.md` |
| 5. Evaluation | one eval script per component, baselines, intervals, error files, the look-ahead experiment | a metric and a baseline per component | `05_evaluation.md` |
| 6. Deployment | Streamlit app with a trace panel, README, this report, demo script | the app runs from a clean clone; the UI checklist passes | `06_deployment.md` |

CRISP-DM is a cycle, and the project went back more than once. The loops are recorded in `docs/crisp_dm/iterations.md`.

| Loop | Trigger | Phase revisited | Change | Before → after |
|---|---|---|---|---|
| 1 | 4 of the 8 tickers had no 2019 headlines | 1. scope | the human replaced them with INTC, NFLX, AVGO, HD | tickers with 2019 news: 4 → 8 of 8 |
| 2 | "invalid JSON" in the first real run | 4. LLM client | an answer cut short by the provider counts as an API failure and is retried | invalid first answers: 3 of 8 → 0 of 371 |
| 3 | 6 of 40 pipeline runs incomplete after quota errors | 4. LLM client | the retry hint is kept; quota errors pause all threads and do not use up retries | incomplete: 6 of 40 → 0 of 80 |
| 4 | the advisor chose one week almost always | 4. advisor prompt | `advisor_v2` with a horizon rule | no gain; v2 rejected (§5.9) |

## 5. Evaluation

Every component has its own script `evals/eval_<component>.py`, a metric, a baseline, n and a 95% bootstrap confidence interval. The ten worst errors of each component are saved in `evals/results/<component>_errors.csv`. The full table is in `README.md`.

### 5.1 Targets from phase 1
The targets were written down before any evaluation was run. "Not measured" means that the bucket or file needed for the check has no data.

{{targets:@table|raw}}

**How to read "yes".** The targets were written as comparisons of point estimates, and every one of them is met on the point estimate. For the three advisor rows that is weaker than it looks:
- **Direction.** The baselines lie inside the advisor's 95% interval (§5.5).
- **Horizon.** The chosen horizon is one week in almost every case, so it nearly equals the fixed one-week baseline.
- **Confidence.** The outer confidence buckets hold only {{advisor:C.hit_rate_confidence_50_60:n}} and {{advisor:C.hit_rate_confidence_80_100:n}} decisions.

These rows are "met, but not shown to be better than chance". The data, feature, client, relevance, sentiment and judge targets are met with margin.

### 5.2 Deterministic parts: data, features, client, UI
| Component | Metric | Result | n |
|---|---|---|---|
| Data collector | missing trading days | {{data:missing_trading_days_rate|pct}} | {{data:missing_trading_days_rate:n|int}} ticker-days |
| Data collector | duplicate headlines in the raw rows, removed in cleaning | {{data:duplicate_headlines_rate_raw|pct}} | {{data:duplicate_headlines_rate_raw:n|int}} rows |
| Data collector | look-ahead violations in sets A and C | {{data:lookahead_violation_rate:field:count}} | {{data:lookahead_violation_rate:n|int}} headlines |
| Data collector | set-B headlines that still show a name, a year or a dollar amount | {{data:set_b_leak_rate:field:count}} | {{data:set_b_leak_rate:n|int}} headlines |
| Features | exact match with the independent reference (tolerance 1e-6) | {{features:exact_match_rate:field:matches}} of {{features:exact_match_rate:n}} | 3 tickers × 5 dates |
| LLM client | valid JSON on the first answer | {{llm_client:valid_json_first_answer_rate|pct}} {{llm_client:valid_json_first_answer_rate:ci|pct}} | {{llm_client:valid_json_first_answer_rate:n|int}} answered calls |
| LLM client | still invalid after the repair retry | {{llm_client:invalid_after_retry_rate|pct}} | {{llm_client:invalid_after_retry_rate:n|int}} calls |
| LLM client | calls lost to API failures after all retries | {{llm_client:transport_error_rate|pct}} | {{llm_client:transport_error_rate:n|int}} calls |
| LLM client | HTTP attempts per answered call; seconds per call | {{llm_client:mean_http_attempts_per_call}}; {{llm_client:mean_latency_s}} | {{llm_client:mean_latency_s:n|int}} calls |
| LLM client | cache hit rate over all logged calls | {{llm_client:cache_hit_rate|pct}} | {{llm_client:cache_hit_rate:n|int}} calls |
| UI | checklist items passed on the demo cases | {{ui:checks_passed_rate:field:passed}} of {{ui:checks_passed_rate:n}} | {{ui:demo_cases_passed_rate:n}} cases |

**Analysis.**
- **Results are not distorted by the client.** Almost every answer is valid JSON on the first try. Calls still invalid after the repair retry: {{llm_client:invalid_after_retry_rate:field:count|raw}} of {{llm_client:invalid_after_retry_rate:n|int}}. Calls lost to the API are counted separately ({{llm_client:transport_error_rate|pct}}), so they never look like model mistakes: an advisor run with a lost language step is marked incomplete, not wrong.
- **The free API is slow, not the model.** One answered call needs {{llm_client:mean_http_attempts_per_call}} HTTP attempts on average (500/503 errors and quota refusals). The mean time per call ({{llm_client:mean_latency_s}} s) is measured from the start of the call to a valid answer: it includes retries and the time the call waited behind 16 parallel threads for the 30-requests-a-minute quota.
- **The cache.** It makes every eval re-run on the same machine free and identical. It is local (`.cache/llm`, about 7 MB, not in the repository), so a fresh clone calls the API again; at temperature 0 the answers were identical across repeats in the phase-4 smoke runs.

### 5.3 Relevance filter
{{relevance:f1:n}} labeled fresh headlines, of which {{relevance:@relevant_share|pct}} are relevant.

| System | Precision | Recall | F1 |
|---|---|---|---|
| LLM filter (`{{relevance:#prompt_version}}`) | {{relevance:precision}} {{relevance:precision:ci}} | {{relevance:recall}} {{relevance:recall:ci}} | {{relevance:f1}} {{relevance:f1:ci}} |
| Baseline: keep all | {{relevance:precision:base:keep_all}} | {{relevance:recall:base:keep_all}} | {{relevance:f1:base:keep_all}} |
| Baseline: keyword (headline names the company) | {{relevance:precision:base:keyword}} | {{relevance:recall:base:keyword}} | {{relevance:f1:base:keyword}} |

**Analysis.** The keep-all baseline shows why the filter exists: only {{relevance:@relevant_share|pct}} of the headlines that Finnhub tags with a ticker are about that company. The LLM filter lost {{relevance:@confusion.fn}} relevant headline: "Is a 100-Year Bond Worth Buying?" is tagged GOOGL, and the human reviewer marked it relevant because the 100-year bond in the news was issued by Alphabet. The headline itself does not name the company, so the model called it a "general finance topic"; the keyword baseline misses it too. Both false positives ({{relevance:@confusion.fp}}) are JPMorgan headlines in which the bank is the commentator, not the subject: "JPMorgan Flips Bullish on Bitcoin Over the Next Ten Years" and "JPMorgan drops blunt take on software stocks AI threat" (`evals/results/relevance_errors.csv`). They are borderline cases, and the keyword baseline makes the same mistake, because the name is in the headline. The keyword baseline is strong on fresh headlines because Finnhub headlines usually name the company they are about. The LLM is better on headlines that name the company only in passing: lists of stocks ("Zacks Investment Ideas feature highlights: JP Morgan, Home Depot, Lennar, …") and the CEO's personal remarks ("Elon Musk Fires Back At AOC …"). Keyword matching keeps them; the LLM drops them. Before the human review this headline was labeled not relevant, so the filter's single miss comes from that one changed label.

### 5.4 Sentiment labeler
| Data | Metric | LLM (`{{sentiment:#prompt_version}}`) | FinBERT | Always neutral |
|---|---|---|---|---|
| PhraseBank sample (n = {{sentiment:phrasebank.accuracy:n}}) | accuracy | {{sentiment:phrasebank.accuracy}} {{sentiment:phrasebank.accuracy:ci}} | {{sentiment:phrasebank.accuracy:base:finbert}} | {{sentiment:phrasebank.accuracy:base:always_neutral}} |
| PhraseBank sample | macro-F1 | {{sentiment:phrasebank.macro_f1}} {{sentiment:phrasebank.macro_f1:ci}} | {{sentiment:phrasebank.macro_f1:base:finbert}} | {{sentiment:phrasebank.macro_f1:base:always_neutral}} |
| Fresh headlines (n = {{sentiment:fresh.accuracy:n}}) | accuracy | {{sentiment:fresh.accuracy}} {{sentiment:fresh.accuracy:ci}} | {{sentiment:fresh.accuracy:base:finbert}} | {{sentiment:fresh.accuracy:base:always_neutral}} |
| Fresh headlines | macro-F1 | {{sentiment:fresh.macro_f1}} {{sentiment:fresh.macro_f1:ci}} | {{sentiment:fresh.macro_f1:base:finbert}} | {{sentiment:fresh.macro_f1:base:always_neutral}} |
| Fresh, relevant only (n = {{sentiment:fresh_relevant.accuracy:n}}) | macro-F1 | {{sentiment:fresh_relevant.macro_f1}} {{sentiment:fresh_relevant.macro_f1:ci}} | {{sentiment:fresh_relevant.macro_f1:base:finbert}} | {{sentiment:fresh_relevant.macro_f1:base:always_neutral}} |

**Analysis.**
- **PhraseBank.** The LLM did not miss a single negative sentence. Almost all of its errors are neutral sentences that it called positive (confusion matrix in `evals/results/sentiment.json`), for example "The combined value of the orders is EUR 45mn." The model reads plain business facts (orders, launches, plans) as good news, while the PhraseBank annotators called them neutral.
- **FinBERT on PhraseBank.** FinBERT scores higher here, but `ProsusAI/finbert` was fine-tuned on Financial PhraseBank, so this test favours it.
- **Fresh headlines.** On real headlines from the fresh period FinBERT falls to a macro-F1 of {{sentiment:fresh.macro_f1:base:finbert}}, while the LLM keeps {{sentiment:fresh.macro_f1}}. This is the data the system actually sees.
- **Errors on fresh headlines.** They are again neutral headlines labeled positive, mostly lists and market wrap-ups ("5 Stocks to Buy if AI Investing Comes Back Into Style"). The relevance filter removes most of these before sentiment is used, which is why the score on relevant headlines alone is higher ({{sentiment:fresh_relevant.macro_f1}}).

### 5.5 Advisor
Hit rate = the share of buy and sell advice where the price moved in the advised direction over the chosen horizon. Hold advice has no direction; its share is shown. Baselines are computed on the same rows.

| | Set A (2019) | Set B (2019, anonymized) | Set C (after cutoff) |
|---|---|---|---|
| Decisions with valid advice | {{advisor:@A.valid}} | {{advisor:@B.valid}} | {{advisor:@C.valid}} |
| Share with a direction (buy or sell) | {{advisor:A.directional_share|pct}} | {{advisor:B.directional_share|pct}} | {{advisor:C.directional_share|pct}} |
| **Hit rate** | {{advisor:A.hit_rate|pct}} {{advisor:A.hit_rate:ci|pct}} | {{advisor:B.hit_rate|pct}} {{advisor:B.hit_rate:ci|pct}} | {{advisor:C.hit_rate|pct}} {{advisor:C.hit_rate:ci|pct}} |
| Baseline: always buy | {{advisor:A.hit_rate:base:always_buy|pct}} | {{advisor:B.hit_rate:base:always_buy|pct}} | {{advisor:C.hit_rate:base:always_buy|pct}} |
| Baseline: same as last move | {{advisor:A.hit_rate:base:same_as_last_move|pct}} | {{advisor:B.hit_rate:base:same_as_last_move|pct}} | {{advisor:C.hit_rate:base:same_as_last_move|pct}} |
| Mean return in the advised direction, chosen horizon | {{advisor:A.mean_return_chosen_horizon_pct}}% | {{advisor:B.mean_return_chosen_horizon_pct}}% | {{advisor:C.mean_return_chosen_horizon_pct}}% |
| Same advice, fixed 1 day / fixed 1 week | {{advisor:A.mean_return_chosen_horizon_pct:base:fixed_1d}}% / {{advisor:A.mean_return_chosen_horizon_pct:base:fixed_1w}}% | {{advisor:B.mean_return_chosen_horizon_pct:base:fixed_1d}}% / {{advisor:B.mean_return_chosen_horizon_pct:base:fixed_1w}}% | {{advisor:C.mean_return_chosen_horizon_pct:base:fixed_1d}}% / {{advisor:C.mean_return_chosen_horizon_pct:base:fixed_1w}}% |
| Hit rate, confidence 50–60 (n) | {{advisor:A.hit_rate_confidence_50_60|pct}} ({{advisor:A.hit_rate_confidence_50_60:n}}) | {{advisor:B.hit_rate_confidence_50_60|pct}} ({{advisor:B.hit_rate_confidence_50_60:n}}) | {{advisor:C.hit_rate_confidence_50_60|pct}} ({{advisor:C.hit_rate_confidence_50_60:n}}) |
| Hit rate, confidence 60–80 (n) | {{advisor:A.hit_rate_confidence_60_80|pct}} ({{advisor:A.hit_rate_confidence_60_80:n}}) | {{advisor:B.hit_rate_confidence_60_80|pct}} ({{advisor:B.hit_rate_confidence_60_80:n}}) | {{advisor:C.hit_rate_confidence_60_80|pct}} ({{advisor:C.hit_rate_confidence_60_80:n}}) |
| Hit rate, confidence 80–100 (n) | {{advisor:A.hit_rate_confidence_80_100|pct}} ({{advisor:A.hit_rate_confidence_80_100:n}}) | {{advisor:B.hit_rate_confidence_80_100|pct}} ({{advisor:B.hit_rate_confidence_80_100:n}}) | {{advisor:C.hit_rate_confidence_80_100|pct}} ({{advisor:C.hit_rate_confidence_80_100:n}}) |

**Analysis.**
- **Direction, set C.** On the unseen period the advisor's hit rate ({{advisor:C.hit_rate|pct}}) is above "always buy" ({{advisor:C.hit_rate:base:always_buy|pct}}) and "same as last move" ({{advisor:C.hit_rate:base:same_as_last_move|pct}}). Both baselines lie inside its interval, so with n = {{advisor:C.hit_rate:n}} this is not proof that the advice beats them.
- **Direction, set A.** In 2019 the advisor is clearly worse than always buy. 2019 was a strongly rising year, so every sell call had to be right against the trend. Only {{advisor:@A.hit_rate_by_action.sell|pct}} of the sell calls in set A were right, against {{advisor:@A.hit_rate_by_action.buy|pct}} of the buy calls. In set C the sell calls were right {{advisor:@C.hit_rate_by_action.sell|pct}} of the time. The advisor reacts to negative news in the same way in both periods; the market did not.
- **Horizon.** The advisor chose one week in {{advisor:@A.horizons.1w}} of {{advisor:@A.decisions}} decisions in set A and {{advisor:@C.horizons.1w}} of {{advisor:@C.valid}} in set C. The chosen horizon is therefore almost the same as "fixed 1 week", and the horizon criterion says little. A prompt with an explicit horizon rule was tried and rejected (§5.9).
- **Confidence.** The hit rate rises from the lowest to the highest confidence bucket in all three sets, which is the expected direction. But the model gives a confidence between 60 and 80 to almost every answer (mean {{advisor:@C.mean_confidence|raw}} on set C), so the outer buckets hold only a few decisions each, and their intervals are wide.
- **A versus B.** Real names and dates (A) and the anonymized copy (B) give almost the same actions and hit rates; see §5.8.

### 5.6 Explanations and the judge
The judge split {{judge:claims_supported_rate:field:explanations}} explanations into {{judge:claims_supported_rate:field:claims}} factual claims. **{{judge:claims_supported_rate|pct}}** {{judge:claims_supported_rate:ci|pct}} of the claims are supported by the advisor's inputs. On the {{judge:judge_label_agreement:n}} claims that also have a label, the judge agrees with the label in **{{judge:judge_label_agreement|pct}}** {{judge:judge_label_agreement:ci|pct}} of cases (labels: {{judge:@labeled_by|raw}}).

**Analysis.**
- **The explanations stay close to their inputs.** They mostly repeat the price features and headlines they were given. The only claim the judge rejected is reasoning, not an invented fact: "Medium volatility suggests a one-week horizon for negative momentum" (`evals/results/judge_errors.csv`).
- **The agreement number proves less than it seems.** The labels mark almost every claim as supported, so a judge that always answers "supported" reaches the same agreement ({{judge:judge_label_agreement:base:always_supported|pct}}). The one disagreement shows that the judge is lenient with paraphrase: it accepted "regulatory scrutiny" for a headline about a senator's scrutiny.
- **A test with known false claims.** To check whether the judge catches hallucinations at all, one false sentence was added to each of the {{judge:planted_false_claims_caught_rate:n}} labeled explanations. Half of them got a wrong 20-day return (off by {{judge:@planted_shift_pct|raw}} points), the other half an invented "$10 billion share buyback". The judge rejected **{{judge:planted_false_claims_caught_rate|pct}}** of the planted claims; a judge that accepts everything catches none of them. It catches plain invented facts and wrong numbers, but it is lenient with loose wording of true facts.

### 5.7 Trader (simulator)
| | Set A | Set B | Set C |
|---|---|---|---|
| Trades | {{simulator:A.n_trades|int}} | {{simulator:B.n_trades|int}} | {{simulator:C.n_trades|int}} |
| Total return | {{simulator:A.total_return|pct}} {{simulator:A.total_return:ci|pct}} | {{simulator:B.total_return|pct}} {{simulator:B.total_return:ci|pct}} | {{simulator:C.total_return|pct}} {{simulator:C.total_return:ci|pct}} |
| Buy & hold, same days | {{simulator:A.total_return:base:buy_and_hold|pct}} | {{simulator:B.total_return:base:buy_and_hold|pct}} | {{simulator:C.total_return:base:buy_and_hold|pct}} |
| Sharpe (buy & hold) | {{simulator:A.sharpe}} ({{simulator:A.sharpe:base:buy_and_hold}}) | {{simulator:B.sharpe}} ({{simulator:B.sharpe:base:buy_and_hold}}) | {{simulator:C.sharpe}} ({{simulator:C.sharpe:base:buy_and_hold}}) |
| Max drawdown (buy & hold) | {{simulator:A.max_drawdown|pct}} ({{simulator:A.max_drawdown:base:buy_and_hold|pct}}) | {{simulator:B.max_drawdown|pct}} ({{simulator:B.max_drawdown:base:buy_and_hold|pct}}) | {{simulator:C.max_drawdown|pct}} ({{simulator:C.max_drawdown:base:buy_and_hold|pct}}) |
| Winning trades | {{simulator:A.win_rate|pct}} | {{simulator:B.win_rate|pct}} | {{simulator:C.win_rate|pct}} |
| Mean share of equity invested | {{simulator:@A.mean_exposure|pct}} | {{simulator:@B.mean_exposure|pct}} | {{simulator:@C.mean_exposure|pct}} |

**Analysis.** The trader buys only on buy advice with confidence of at least 70, so it trades rarely and keeps most of its money in cash: on average it had only {{simulator:@C.mean_exposure|pct}} of its equity invested in set C. Buy & hold is always fully invested, so the two are not like for like.
- **Set A.** The gap to buy & hold in 2019 ({{simulator:A.total_return|pct}} against {{simulator:A.total_return:base:buy_and_hold|pct}}) comes mostly from sitting in cash during a rising year.
- **Set C.** In the flat fresh period the trader made {{simulator:C.total_return|pct}} with a much smaller drawdown than buy & hold. This meets the phase-1 target, but the interval of the return includes zero, and the small drawdown is mostly a result of the small exposure.
- **What to change.** A fairer baseline would be buy & hold scaled to the same exposure. It is listed in §7 as further work.

### 5.8 Look-ahead experiment: does the LLM remember the future?
| Check | Result |
|---|---|
| Hit rate A (2019, real names) | {{lookahead:A.hit_rate|pct}} {{lookahead:A.hit_rate:ci|pct}}, n = {{lookahead:A.hit_rate:n}} |
| Hit rate B (same rows, anonymized) | {{lookahead:B.hit_rate|pct}} {{lookahead:B.hit_rate:ci|pct}}, n = {{lookahead:B.hit_rate:n}} |
| Hit rate C (after the knowledge cutoff) | {{lookahead:C.hit_rate|pct}} {{lookahead:C.hit_rate:ci|pct}}, n = {{lookahead:C.hit_rate:n}} |
| A minus B on the same decisions (paired) | {{lookahead:A_minus_B.hit_rate_paired|pct}} {{lookahead:A_minus_B.hit_rate_paired:ci|pct}}, n = {{lookahead:A_minus_B.hit_rate_paired:n}} |
| Memory probe: 2019 closing prices within ± 5% | {{lookahead:memory_probe.2019.within_5pct|pct}} {{lookahead:memory_probe.2019.within_5pct:ci|pct}}, n = {{lookahead:memory_probe.2019.within_5pct:n}} |
| Memory probe: control dates after the cutoff | {{lookahead:memory_probe.post_cutoff.within_5pct|pct}} {{lookahead:memory_probe.post_cutoff.within_5pct:ci|pct}}, n = {{lookahead:memory_probe.post_cutoff.within_5pct:n}} |
| Sentiment gap: PhraseBank accuracy minus fresh accuracy | {{lookahead:sentiment_gap.accuracy_phrasebank_minus_fresh}} {{lookahead:sentiment_gap.accuracy_phrasebank_minus_fresh:ci}} |
| **Decision by the rule** ({{lookahead:@rule|raw}}) | **the LLM peeks: {{lookahead:@llm_peeks|raw}}** |

**Analysis.**
- **The model does remember 2019 prices, roughly.** Asked for a 2019 close from memory, it is within ± 5% in {{lookahead:memory_probe.2019.within_5pct|pct}} of the questions (a hit counts against the price of that day either before or after later stock splits). For dates after its cutoff it declined to answer all {{lookahead:memory_probe.post_cutoff.within_5pct:field:no_answer|int}} questions instead of guessing. So it knows the 2019 price level of these companies, and it knows what it does not know.
- **The advisor does not use that memory.**
  - Set A is not better than C: its hit rate is lower, and the intervals overlap.
  - On the same decisions, real names and dates (A) and the anonymized copy (B) gave the same action in {{lookahead:@a_b_same_action_share|pct}} of the cases.
  - Wherever both gave a direction, it was the same direction, so the paired difference in hit rate is exactly {{lookahead:A_minus_B.hit_rate_paired|pct}}.
  - The advice is driven by the features and headlines in the prompt, not by the company's name or the date.
- **Sentiment gap.** PhraseBank accuracy is not higher than accuracy on the fresh headlines (difference {{lookahead:sentiment_gap.accuracy_phrasebank_minus_fresh}}, interval includes 0). This gives no sign that the model memorized the PhraseBank labels.
- **Decision.** By the rule fixed in the spec, **the LLM does not peek** in the advice task. The memory probe shows that a weaker design could leak: a prompt that asks "what happened next?" might get remembered answers. Our advisor gets only features and headlines, and it does not.

### 5.9 Improvement loop
The weakest point the evaluation found in the advisor was the horizon: `advisor_v1` almost never chose one day. `prompts/advisor_v2.md` adds an explicit rule for choosing the horizon and two short examples. It was compared with v1 on the first 50 decisions of sets A and C (`python -m evals.compare_advisor_prompts`).

| Set C, first 50 decisions | `advisor_v1` | `advisor_v2` |
|---|---|---|
| Share of 1-day horizons | {{advisor_prompt_comparison:C.advisor_v1.share_horizon_1d|pct}} | {{advisor_prompt_comparison:C.advisor_v2.share_horizon_1d|pct}} |
| Share with a direction | {{advisor_prompt_comparison:C.advisor_v1.directional_share|pct}} | {{advisor_prompt_comparison:C.advisor_v2.directional_share|pct}} |
| Hit rate (n) | {{advisor_prompt_comparison:C.advisor_v1.hit_rate|pct}} ({{advisor_prompt_comparison:C.advisor_v1.hit_rate:n}}) | {{advisor_prompt_comparison:C.advisor_v2.hit_rate|pct}} ({{advisor_prompt_comparison:C.advisor_v2.hit_rate:n}}) |
| Mean return, chosen horizon | {{advisor_prompt_comparison:C.advisor_v1.mean_return_chosen_horizon_pct}}% | {{advisor_prompt_comparison:C.advisor_v2.mean_return_chosen_horizon_pct}}% |
| Set A hit rate (n) | {{advisor_prompt_comparison:A.advisor_v1.hit_rate|pct}} ({{advisor_prompt_comparison:A.advisor_v1.hit_rate:n}}) | {{advisor_prompt_comparison:A.advisor_v2.hit_rate|pct}} ({{advisor_prompt_comparison:A.advisor_v2.hit_rate:n}}) |

v2 barely changed the horizon. It gave more buy and sell answers, and those extra answers were less accurate. All intervals overlap, so v2 is not proven worse either, but there was no evidence of a gain worth a full new run on the free quota. **v2 was rejected and v1 stays.** This loop is recorded as a negative result in `docs/crisp_dm/iterations.md` (loop 4). The other three loops changed the scope and the LLM client; they are summarized in §4.

## 6. Use of the AI coding assistant

The system was built with Claude Code. The agent was steered through files in the repository, not only through chat prompts.

| File | What it controls |
|---|---|
| `docs/PROJECT_SPEC.md` | WHAT to build: scope, data, components, business rules, metrics, the look-ahead experiment |
| `AGENTS.md` | HOW to build: CRISP-DM phases and gates, repository layout, coding conventions, LLM rules, data rules (no look-ahead), evaluation rules, how to work with the human, the definition of done |
| `CLAUDE.md` | loads both files into every session; requires plan mode for multi-file changes; one fresh session per phase |
| `.claude/skills/crisp-dm-phase` | how to write a phase log entry: what was done, decisions with the rejected alternative, evidence, gate status |
| `.claude/skills/add-component` | the fixed recipe for a component: plan, schema, prompt file, code, mocked test, eval script, docs |
| `.claude/skills/run-evals` | how to run all evals and refresh the results table |

**Context control.** The rules live in files that reload in every session, so nothing depends on chat history. The spec says what, `AGENTS.md` says how. The CRISP-DM logs carry the state from one phase to the next. Before any multi-file change the agent writes a plan and waits for approval; decisions that belong to the human are asked as explicit choices with a recommendation.

**Quality control that does not rely on the agent's good behavior.**
- `pytest` with the LLM always mocked ({{ui:#run_date}}: all tests pass), and `ruff`.
- `tests/test_conventions.py` fails when a function is longer than 40 lines or has no docstring or return type.
- `tests/test_no_lookahead.py` fails when any decision of the real sets contains a headline from after its close.
- Numbers in the README and in this report are rendered from results files, so the agent cannot type a metric by hand.

**What went wrong or was unexpected, and what changed.**

| What happened | What changed | Result |
|---|---|---|
| The spec listed AAPL, MSFT, AMZN and META. The agent checked the data before coding and found no 2019 headlines for them. | The human chose replacement tickers; the spec and `config.yaml` were updated. | Sets A and C use the same 8 tickers. |
| The human proposed several OpenRouter accounts to get around the limit of 50 free requests a day. The agent read the terms and the limits page first: several accounts are forbidden and do not raise the limit. | A free Google AI Studio key was connected to OpenRouter instead. | The daily limit no longer applies; the limit became 30 requests a minute. |
| The first real run reported "invalid JSON" from the model. The raw answers showed that the provider had cut them short. | The client treats a cut answer as an API failure and retries it; a test covers it. | Invalid first answers: 3 of 8 before; afterwards {{llm_client:valid_json_first_answer_rate|pct}} of {{llm_client:valid_json_first_answer_rate:n|int}} answers were valid on the first try (§5.2). |
| Quota errors used up the retries, because the client shortened error messages and lost the "retry in N seconds" hint. | The hint is kept; a quota error pauses all threads and is not counted as a retry. | Incomplete pipeline runs: 6 of 40 before, 0 after. |
| Building the news table took 170 seconds because the config file was parsed again for every row. | The config is cached. | 2.6 seconds. |
| Twice the agent wrote a test whose hand-computed expected value was wrong, while the code was right. | The agent re-derived the expected values by hand instead of changing the code to fit. | Tests now document the arithmetic in comments. |
| The free API gave about 8 answers a minute, with about 3 HTTP attempts per answer. Running 300 decisions per set was not possible in the time available. | `evals.batch_limit` = 200, taken as the first 200 of a fixed random order. | The run can be extended later without changing earlier results. |
| The human gave the labeling of the 100 fresh headlines and the judge claims to the agent because of the deadline. | The agent drafted the labels following written rules (`evals/labels/README.md`) and marked them `labeled_by = assistant_draft`. The human then reviewed both files, changed one relevance label, and `python -m evals.confirm_labels` switched them to `human_reviewed`. | Results files and this report say who labeled the data. |

**Lessons.**
- Rules in files work better than rules in chat. The gates, the "no look-ahead" rule and the "one place for LLM calls" rule were followed in every phase because they were reloaded each time.
- A rule the agent can break silently should become a test. Function length, docstrings and look-ahead are checked by `pytest`, and documents take their numbers from results files.
- Ask the agent to check the data and the limits before it plans. Two of the largest changes (the tickers and the API route) came from facts found before coding; the third (the sample size) came from measuring the API in an early smoke run.
- The human keeps the decisions: the model, the tickers and the API route were chosen by the human from options that the agent prepared with evidence. Where work was delegated (the labels), the files record it.

## 7. Limitations

- **Small samples.** {{advisor:@C.decisions}} decisions per set give {{advisor:C.hit_rate:n}} directional calls on set C, so the interval of a hit rate is about ± 9 points. Differences smaller than that cannot be shown.
- **Label provenance.** The 100 fresh labels and the judge claims were drafted by the coding assistant and then reviewed by the human ({{relevance:@labeled_by|raw}}); the review changed one label. The same assistant wrote the prompts, and a reviewer who starts from a draft tends to accept it, so agreement with the model may still be optimistic.
- **Different market periods.** 2019 was a rising market and the fresh period was not, so A and C differ in more than what the model can remember. The A versus B comparison (same rows) is the cleaner test.
- **News window.** Only the 15 most recent headlines are used. For a busy ticker such as NVDA they cover the last 1.4 hours before the close, not 3 days. A keyword pre-filter would widen the window.
- **Daily prices.** Trades execute at the close of the decision day, with no slippage. Undated 2019 news is pushed to the next day, which is safe but throws away same-day information.
- **Anonymization is best effort.** Set B hides names, people, products, dates and dollar amounts, but other companies in a headline remain and can hint at the sector.
- **A free model on a free API.** An answered call needs {{llm_client:mean_http_attempts_per_call}} HTTP attempts on average; the rest fail and are retried. The system works, but a live request can take a minute or more.
- **The judge test is narrow.** Almost all real claims are true, so agreement with the labels equals that of a judge that accepts everything. The planted claims show that it catches obvious inventions, but not whether it catches subtle ones, such as a wrong sentiment attributed to a true headline.
- **Trader baseline.** Buy & hold is fully invested while the trader is mostly in cash, so their returns are not like for like. A buy & hold scaled to the trader's exposure would be a fairer baseline; it was not built.
