"""Eval of the UI (component 8): a checklist on the demo cases, run headlessly.

Run: python -m evals.eval_ui
The app is driven with streamlit.testing; what it shows is compared with the pipeline output.
"""

from datetime import date

import numpy as np
from streamlit.testing.v1 import AppTest

from evals import common
from src.config import ROOT, load_config
from src.data.collector import collect
from src.pipeline.advisor import features_in_percent
from src.pipeline.run import PipelineResult, run_pipeline

COMPONENT = "ui"
APP = "app/streamlit_app.py"
TIMEOUT_S = 900  # a first, uncached request waits for the free API
HORIZON_TEXT = {"1d": "1 day", "1w": "1 week"}
FEATURE_LABELS = {"ret_1d_pct": "1-day return", "ret_5d_pct": "5-day return",
                  "ret_20d_pct": "20-day return", "volatility_20d_pct": "20-day volatility"}


def open_case(ticker: str, day: date) -> AppTest:
    """Start the app, fill the sidebar and press the button."""
    app = AppTest.from_file(str(ROOT / APP), default_timeout=TIMEOUT_S)
    app.run()
    app.sidebar.selectbox[0].select(ticker)
    app.sidebar.date_input[0].set_value(day)
    app.sidebar.button[0].click()
    return app.run()


def advice_matches(metrics: dict, expected: PipelineResult) -> bool:
    """Compare the advice card with the pipeline output."""
    advice = expected.advice
    shown = (metrics.get("Action"), metrics.get("Horizon"), metrics.get("Confidence"),
             metrics.get("News used"))
    return shown == (advice.action.upper(), HORIZON_TEXT[advice.horizon], str(advice.confidence),
                     "yes" if advice.news_used else "no")


def features_match(metrics: dict, expected: PipelineResult) -> bool:
    """Compare the feature tiles with the numbers the advisor received."""
    numbers = features_in_percent(expected.decision.features)
    return all(metrics.get(label) == f"{numbers[key]:.2f}%"
               for key, label in FEATURE_LABELS.items())


def headlines_match(app: AppTest, expected: PipelineResult) -> bool:
    """Compare the headline table with the collected headlines and their labels."""
    tables = [d.value for d in app.dataframe if "Headline" in d.value.columns]
    if not tables:
        return not expected.decision.headlines
    table = tables[0]
    relevant = sum(item.relevant for item in expected.relevance)
    labels = sorted(item.label for item in expected.sentiment)
    shown = sorted(text.split(" ", 1)[1] for text in table["Sentiment"] if text)
    return (table["Headline"].tolist() == [h.text for h in expected.decision.headlines]
            and int((table["Relevant"] == "yes").sum()) == relevant and shown == labels)


def trace_matches(app: AppTest, expected: PipelineResult) -> bool:
    """Check that the trace panel lists exactly the LLM calls of this request."""
    calls = 1 + bool(expected.decision.headlines) + bool(expected.labeled)
    return any(e.label == f"Trace: {calls} LLM calls of this request" for e in app.expander)


def check_case(ticker: str, day: date) -> list[dict]:
    """Run one demo case and return the checklist items with pass / fail."""
    expected = run_pipeline(collect(ticker, day))
    app = open_case(ticker, day)
    metrics = {m.label: m.value for m in app.metric}
    checks = {
        "app runs without an exception": not app.exception,
        "three tabs render": [t.label for t in app.tabs] == ["Advice", "Simulator", "Evals"],
        "advice card equals the pipeline output": advice_matches(metrics, expected),
        "price features equal the pipeline output": features_match(metrics, expected),
        "headline table equals the pipeline output": headlines_match(app, expected),
        "trace panel shows the calls of this request": trace_matches(app, expected),
        "simulator and evals tabs show stored results":
            sum("Metric" in d.value.columns for d in app.dataframe) >= 2,
    }
    case = f"{ticker} {day.isoformat()}"
    return [{"case": case, "check": name, "passed": bool(ok)} for name, ok in checks.items()]


def main() -> None:
    """Run the checklist on every demo case and write the results file."""
    common.limit_argument(__doc__)
    cases = load_config()["evals"]["demo_cases"]
    rows = [row for case in cases
            for row in check_case(case["ticker"], date.fromisoformat(case["date"]))]
    passed = np.array([float(row["passed"]) for row in rows])
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row["passed"])
    cases_ok = np.array([float(all(flags)) for flags in by_case.values()])
    metrics = [
        common.metric_row("checks_passed_rate", common.bootstrap(common.mean, passed),
                          passed=int(passed.sum())),
        common.metric_row("demo_cases_passed_rate", common.bootstrap(common.mean, cases_ok),
                          passed=int(cases_ok.sum())),
    ]
    path = common.write_result(COMPONENT, metrics, f"{len(cases)} demo cases from config.yaml",
                               details={"checklist": rows})
    common.write_errors(COMPONENT, [row for row in rows if not row["passed"]])
    common.show(path)


if __name__ == "__main__":
    main()
