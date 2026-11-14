# Label files

| File | Rows | Labels | Used by |
|---|---|---|---|
| `fresh_headlines.csv` | 100 fresh headlines (Finnhub, after the model's knowledge cutoff), evenly per ticker | `relevant` (y / n), `sentiment` (positive / negative / neutral) | `evals/eval_relevance.py`, `evals/eval_sentiment.py` |
| `judge_claims.csv` | the claims of 20 advisor explanations | `human_supported` (y / n) | `evals/eval_judge.py` |
| `features_reference.csv` | 15 feature vectors (3 tickers × 5 dates) from an independent calculation | reference values | `evals/eval_features.py` |

## Who labeled
The column `labeled_by` records it for every row:
- `assistant_draft`: drafted by the coding assistant (Claude), **not yet checked by the human**;
- `human_reviewed`: the human read the row, corrected it if needed and approved it.

Every results file that uses these labels reports the counts of both values. The evaluated model is Gemma, a different model from the one that drafted the labels.

To review: open the CSV, read each row, fix the label where you disagree, then run
`python -m evals.confirm_labels` to mark all rows as `human_reviewed`, and re-run
`python -m evals.run_all` (cached LLM answers are reused, so this takes seconds).

## Rules for `relevant`
The target is the company in the `ticker` / `company` columns.

**y** when the headline states something specific about the target company:
- its results, guidance, products, deals, costs, staff, legal or regulatory issues;
- an analyst rating or price target on it, or a named investor buying or selling it;
- a move of its own share price;
- an opinion or analysis piece whose subject is the target's stock ("Is X a buy?");
- a competitor's move, when the headline itself says what it means for the target;
- the target together with one or two other named companies, when a fact about the target is stated.

**n** otherwise:
- the company is not named and not clearly implied;
- market or sector wrap-ups and lists of four or more stocks with no specific fact about the target;
- news about another company where the target is only a comparison or background;
- the target acting as an analyst or commentator on another asset ("JPMorgan turns bullish on Bitcoin");
- fund or portfolio notes, politicians' trades, general economy or politics;
- an executive's personal or political remarks that are not about the business.

## Rules for `sentiment`
The likely effect on the **target company's** stock, as an investor would read the headline alone.
- **positive**: better results, growth, a new product or deal, an upgrade or raised target, a bullish call or a named buyer, the price rising.
- **negative**: worse results, a downgrade or cut target, a bearish call, layoffs, a lawsuit or fine, a competitive threat, the price falling.
- **neutral**: plain facts, mixed news, questions and outlook pieces without a direction.
- If the headline says nothing about the target company, the label is **neutral**.
- A question that also states a move takes the direction of the move ("X declines 8%: buy, sell or hold?" is negative).

## Rules for `human_supported` (judge_claims.csv)
Each row is one factual claim that the judge extracted from an advisor explanation, next to the inputs the advisor saw.
- **y**: the inputs state the claim or it follows directly from them (a rounded number that matches counts).
- **n**: the inputs do not contain it, contradict it, or it needs outside knowledge.
