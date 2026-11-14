"""Trace one input through the whole system and store it for the report (spec §5b).

Run: python -m evals.walkthrough
Writes evals/results/walkthrough.json: every step of the first demo case with real values.
"""

from datetime import date

from evals import common
from src.config import load_config
from src.data import prices
from src.data.collector import collect
from src.llm import trace
from src.pipeline import advisor, judge
from src.pipeline.run import PipelineResult, run_pipeline

COMPONENT = "walkthrough"
MARK = {True: "yes", False: "no"}


def headline_table(result: PipelineResult) -> str:
    """Render the collected headlines with their relevance and sentiment as Markdown."""
    verdicts = {item.id: item for item in result.relevance}
    labels = {item.id: item.label for item in result.sentiment}
    lines = ["| # | Published | Headline | Relevant | Sentiment |", "|---|---|---|---|---|"]
    for h in result.decision.headlines:
        relevant = MARK[verdicts[h.id].relevant] if h.id in verdicts else ""
        text = h.text.replace("|", "/")
        lines.append(f"| {h.id} | {h.published} | {text} | {relevant} | {labels.get(h.id, '')} |")
    return "\n".join(lines)


def trader_step(result: PipelineResult) -> str:
    """Say what the simulator rule does with this advice."""
    threshold = load_config()["simulator"]["confidence_threshold"]
    advice = result.advice
    if advice.action != "buy":
        return f"action is {advice.action}, not buy → no trade"
    sign = "≥" if advice.confidence >= threshold else "<"
    verdict = "buy with 20% of equity" if advice.confidence >= threshold else "no trade"
    return f"buy with confidence {advice.confidence} {sign} {threshold} → {verdict}"


def outcome(result: PipelineResult) -> dict:
    """Return the forward returns after the decision (ground truth, hidden from the pipeline)."""
    frame = prices.load_prices(result.decision.ticker)
    day = date.fromisoformat(result.decision.date)
    sessions = load_config()["simulator"]["horizon_days"]
    values = {h: prices.forward_return(frame, day, n) for h, n in sessions.items()}
    chosen = values[result.advice.horizon]
    side = {"buy": 1, "sell": -1, "hold": 0}[result.advice.action]
    verdict = "no direction advised" if side == 0 else ("hit" if side * chosen > 0 else "miss")
    return {**{f"fwd_{h}": v for h, v in values.items()}, "chosen": chosen, "verdict": verdict}


def main() -> None:
    """Run the first demo case end to end and store every step."""
    common.limit_argument(__doc__)
    case = load_config()["evals"]["demo_cases"][0]
    result = run_pipeline(collect(case["ticker"], date.fromisoformat(case["date"])))
    inputs = advisor.build_payload(result.decision, result.labeled)
    verdict = judge.run(inputs, result.advice.explanation, result.trace_id)
    calls = trace.read_calls(result.trace_id)
    details = {
        "ticker": case["ticker"], "date": result.decision.date,
        "company": result.decision.company,
        "features": advisor.features_in_percent(result.decision.features),
        "headlines": len(result.decision.headlines),
        "relevant": len(result.labeled),
        "sentiment_counts": {label: sum(h.sentiment == label for h in result.labeled)
                             for label in ("positive", "negative", "neutral")},
        "headline_table": headline_table(result),
        "advice": result.advice.model_dump(),
        "trader": trader_step(result),
        "judge_claims": len(verdict.claims),
        "judge_supported": sum(claim.supported for claim in verdict.claims),
        "outcome": outcome(result),
        "llm_calls": [call["component"] for call in calls],
        "errors": result.errors,
    }
    path = common.write_result(COMPONENT, [], f"demo case {case['ticker']} {case['date']}",
                               details=details)
    print(f"walkthrough stored in {path.name}: {details['trader']}; {details['outcome']}")


if __name__ == "__main__":
    main()
