"""Look-ahead (memorization) experiment: does the LLM 'remember the future'?

Run: python -m evals.eval_lookahead [--limit N]
1. Advisor hit rate on sets A (2019), B (A anonymized), C (after the cutoff), D (live).
2. Direct memory probe: closing prices of 2019 vs a control period after the cutoff.
3. Sentiment gap: accuracy on PhraseBank vs on the fresh labeled headlines.
Decision rule (spec §8): 'the LLM peeks' only if A beats C and B beyond their intervals.
"""

import random

import numpy as np
import pandas as pd
from pydantic import BaseModel

from evals import common, eval_sentiment
from evals.eval_advisor import directional, load_rows
from src.config import load_config
from src.data import calendar as cal
from src.data import prices
from src.llm.client import Validator, call_llm
from src.pipeline import live

COMPONENT = "lookahead"
PERIODS = {"2019": ("2019-01-01", "2019-12-31"), "post_cutoff": None}  # None = fresh period


class ProbeAnswer(BaseModel):
    """The model's remembered closing price for one question."""

    id: int
    close: float | None


class ProbeOutput(BaseModel):
    """What the LLM must return for a batch of questions."""

    answers: list[ProbeAnswer]


def all_answered(expected: list[int]) -> Validator:
    """Build the rule 'exactly one answer per question id'."""

    def check(output: BaseModel) -> str | None:
        """Return a message when the answer ids differ from the question ids."""
        got = sorted(answer.id for answer in output.answers)
        return None if got == sorted(expected) else f"expected ids {sorted(expected)}, got {got}"

    return check


def probe_questions(period: str, count: int) -> list[dict]:
    """Sample (ticker, day) questions for a period, with the true closes kept aside."""
    cfg = load_config()
    start, end = PERIODS[period] or (cfg["dates"]["fresh_start"], cfg["dates"]["fresh_end"])
    days = cal.trading_days(start, end)
    rng = random.Random(f"{cfg['seed']}-{period}")
    tickers = list(cfg["tickers"])
    questions = []
    for i in range(count):
        ticker, day = tickers[i % len(tickers)], rng.choice(days)
        row = prices.load_prices(ticker).loc[pd.Timestamp(day)]
        questions.append({"id": i + 1, "ticker": ticker, "company": cfg["tickers"][ticker]["name"],
                          "date": day.isoformat(), "raw_close": float(row["raw_close"]),
                          "split_adjusted_close": float(row["close"])})
    return questions


def ask_batch(batch: list[dict]) -> dict[int, float | None]:
    """Ask one batch of price questions (the true prices are not sent)."""
    version = load_config()["prompts"]["memory_probe"]
    asked = [{k: q[k] for k in ("id", "ticker", "company", "date")} for q in batch]
    rule = all_answered([q["id"] for q in batch])
    result = call_llm("memory_probe", version, {"questions": asked}, ProbeOutput, rule)
    if result.data is None:
        return {q["id"]: None for q in batch}
    return {answer.id: answer.close for answer in result.data.answers}


def probe(period: str, count: int) -> tuple[dict, list[dict]]:
    """Run the probe for one period; return the metric row and the scored questions."""
    cfg = load_config()["evals"]
    questions = probe_questions(period, count)
    size = cfg["memory_probe_batch"]
    batches = [questions[i:i + size] for i in range(0, len(questions), size)]
    answers = {k: v for part in common.parallel_map(ask_batch, batches) for k, v in part.items()}
    for q in questions:
        q["answer"] = answers[q["id"]]
        errors = [abs(q["answer"] / q[k] - 1) if q["answer"] else np.inf
                  for k in ("raw_close", "split_adjusted_close")]
        q["pct_error_raw"], q["pct_error_adjusted"] = errors
        q["hit"] = min(errors) <= cfg["memory_probe_tolerance"]
    hits = np.array([float(q["hit"]) for q in questions])
    extra = {
        "hits_vs_raw_close": sum(q["pct_error_raw"] <= cfg["memory_probe_tolerance"]
                                 for q in questions),
        "no_answer": sum(q["answer"] is None for q in questions),
        "median_pct_error_raw": float(np.median([q["pct_error_raw"] for q in questions])),
    }
    row = common.metric_row(f"memory_probe.{period}.within_5pct",
                            common.bootstrap(common.mean, hits), **extra)
    return row, questions


def hits_of(calls: pd.DataFrame) -> pd.Series:
    """Return 1.0 where the advised direction matched the forward return, else 0.0."""
    return (calls["side"] * calls["fwd"] > 0).astype(float)


def paired_difference(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Hit(A) - hit(B) on the decisions where both gave a direction (same rows, paired)."""
    a = a.assign(key=a["id"].str[1:]).set_index("key")
    b = b.assign(key=b["id"].str[1:]).set_index("key")
    both = a.index.intersection(b.index)
    diff = (hits_of(a.loc[both]) - hits_of(b.loc[both])).to_numpy()
    return common.metric_row("A_minus_B.hit_rate_paired", common.bootstrap(common.mean, diff))


def same_action_share(limit: int | None) -> float | None:
    """Share of set-A decisions whose anonymized set-B twin got the same action."""
    a, b = (load_rows(letter, limit) for letter in "AB")
    if not len(a) or not len(b):
        return None
    actions = b.assign(key=b["id"].str[1:]).set_index("key")["action"]
    matched = a.assign(key=a["id"].str[1:]).set_index("key")["action"].align(actions, "inner")
    return round(float((matched[0] == matched[1]).mean()), 4)


def set_rows(limit: int | None) -> tuple[list[dict], dict]:
    """Hit rates of A, B, C, the paired A-B difference, and the decision by the spec rule."""
    frames = {letter: directional(load_rows(letter, limit)) for letter in "ABC"}
    rows, stats = [], {}
    for letter, calls in frames.items():
        always_buy = round(float((calls["fwd"] > 0).mean()), 4) if len(calls) else None
        stats[letter] = common.bootstrap(common.mean, hits_of(calls).to_numpy())
        rows.append(common.metric_row(f"{letter}.hit_rate", stats[letter],
                                      {"always_buy": always_buy}))
    rows.append(paired_difference(frames["A"], frames["B"]))
    low = stats["A"]["ci_low"]
    scored = all(stats[letter]["n"] for letter in "ABC")
    peeks = scored and low > stats["C"]["ci_high"] and low > stats["B"]["ci_high"]
    rule = "A.ci_low > C.ci_high and A.ci_low > B.ci_high (hit rate)"
    return rows, {"llm_peeks": bool(peeks), "rule": rule,
                  "a_b_same_action_share": same_action_share(limit)}


def set_d_row() -> tuple[dict, dict]:
    """Score the live advice whose next-day outcome is already known (1-day horizon)."""
    hits, pending = [], 0
    for advice in live.load_store():
        closes = prices.live_prices(advice.ticker)["adj_close"]
        entry = cal.first_close_at_or_after(pd.Timestamp(advice.as_of))
        days = [pd.Timestamp(entry), pd.Timestamp(cal.next_trading_day(entry))]
        side = {"buy": 1, "sell": -1, "hold": 0}[advice.result.advice.action]
        if not all(day in closes.index for day in days):
            pending += 1
        elif side:
            hits.append(float(side * (closes.loc[days[1]] / closes.loc[days[0]] - 1) > 0))
    row = common.metric_row("D.hit_rate", common.bootstrap(common.mean, np.array(hits)))
    store = live.load_store()
    listed = ", ".join(f"{a.ticker} {a.result.advice.action} ({a.result.advice.confidence})"
                       for a in store)
    recorded = sorted({a.as_of[:10] for a in store})
    return row, {"live_advice_recorded": len(store), "outcome_pending": pending,
                 "live_advice": listed, "live_recorded_on": ", ".join(recorded)}


def sentiment_gap_row(limit: int | None) -> dict:
    """Accuracy on PhraseBank minus accuracy on the fresh headlines (cached LLM answers)."""
    phrase = eval_sentiment.label_groups(eval_sentiment.phrasebank_frame(limit), None)
    fresh = eval_sentiment.label_groups(eval_sentiment.fresh_frame(limit), "company")
    stat = common.bootstrap_difference(phrase["truth"] == phrase["predicted"],
                                       fresh["truth"] == fresh["predicted"])
    return common.metric_row("sentiment_gap.accuracy_phrasebank_minus_fresh", stat)


def main() -> None:
    """Run the three checks and write one results file with the decision."""
    limit = common.limit_argument(__doc__)
    count = limit or load_config()["evals"]["memory_probe_n"]
    metrics, details = set_rows(limit)
    d_row, d_info = set_d_row()
    metrics.append(d_row)
    questions = []
    for period in PERIODS:
        row, scored = probe(period, count)
        metrics.append(row)
        questions += [{"period": period, **q} for q in scored]
    metrics.append(sentiment_gap_row(limit))
    details.update(d_info)
    version = load_config()["prompts"]["memory_probe"]
    data = "advisor runs on sets A/B/C, live advice (D), price files, sentiment eval data"
    path = common.write_result(COMPONENT, metrics, data, version, limit, details)
    questions.sort(key=lambda q: -min(q["pct_error_raw"], q["pct_error_adjusted"]))
    common.write_errors(COMPONENT, questions, limit)
    common.show(path)


if __name__ == "__main__":
    main()
