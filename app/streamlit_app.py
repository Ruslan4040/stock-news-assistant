"""Streamlit UI (component 8): shows what the pipeline produced. No business logic here.

Run: streamlit run app/streamlit_app.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import charts  # noqa: E402
from evals import common, report  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import prices  # noqa: E402
from src.data.collector import collect, decision_day  # noqa: E402
from src.llm import trace  # noqa: E402
from src.pipeline.advisor import features_in_percent  # noqa: E402
from src.pipeline.live import live_advice  # noqa: E402
from src.pipeline.run import PipelineResult, run_pipeline  # noqa: E402

HISTORICAL, LIVE = "Historical date", "Live (now)"
HORIZON_TEXT = {"1d": "1 day", "1w": "1 week"}
SENTIMENT_MARK = {"positive": "▲ positive", "negative": "▼ negative", "neutral": "● neutral"}
SENTIMENT_TINT = {"▲ positive": "rgba(12,163,12,0.18)", "▼ negative": "rgba(208,59,59,0.18)",
                  "● neutral": "rgba(137,135,129,0.18)"}
CHART_DAYS = 120  # trading days shown in the price chart
SET_NAMES = {"A": "A: 2019, real names and dates", "B": "B: 2019, anonymized",
             "C": "C: fresh period (after the model's knowledge cutoff)"}


def is_dark() -> bool:
    """Say whether the viewer uses the dark Streamlit theme."""
    try:
        return st.context.theme.type == "dark"
    except Exception:  # theme info is missing in tests and in old Streamlit versions
        return False


def sidebar() -> tuple[str, str, date, bool]:
    """Draw the request form; return ticker, mode, date and whether the button was pressed."""
    cfg = load_config()
    names = {ticker: info["name"] for ticker, info in cfg["tickers"].items()}
    st.sidebar.header("Request")
    ticker = st.sidebar.selectbox("Ticker", list(names), format_func=lambda t: f"{t}: {names[t]}")
    mode = st.sidebar.radio("Mode", [HISTORICAL, LIVE])
    first = cfg["evals"]["demo_cases"][0]["date"]
    day = st.sidebar.date_input(
        "Decision date", value=date.fromisoformat(first), disabled=mode == LIVE,
        min_value=date(cfg["dates"]["set_a_year"] - 1, 3, 1),
        max_value=date.fromisoformat(cfg["dates"]["prices_end"]),
    )
    label = "Refresh now" if mode == LIVE else "Get advice"
    pressed = st.sidebar.button(label, type="primary")
    st.sidebar.caption("News: Kaggle headlines up to mid-2020, Finnhub from Nov 2025. "
                       "Dates in between get advice from price features only.")
    return ticker, mode, day, pressed


def run_request(ticker: str, mode: str, day: date) -> None:
    """Call the pipeline for the request and keep the result in the session."""
    with st.spinner("Collecting data and asking the model (the free API can take a minute)..."):
        if mode == LIVE:
            record, created = live_advice(ticker)
            note = ("New advice" if created else "Today's stored advice (no new relevant "
                    "headline)") + f", reason: {record.reason}"
            st.session_state["view"] = {"result": record.result, "live": True, "note": note}
        else:
            result = run_pipeline(collect(ticker, day))
            snapped = decision_day(day)
            note = "" if snapped == day else f"{day} is not a trading day: using {snapped}."
            st.session_state["view"] = {"result": result, "live": False, "note": note}


def advice_card(result: PipelineResult) -> None:
    """Show the suggestion: action, horizon, confidence and the explanation."""
    advice = result.advice
    columns = st.columns(4)
    columns[0].metric("Action", advice.action.upper())
    columns[1].metric("Horizon", HORIZON_TEXT[advice.horizon])
    columns[2].metric("Confidence", str(advice.confidence))
    columns[3].metric("News used", "yes" if advice.news_used else "no")
    if advice.error:
        st.error(f"The advisor returned no valid answer: {advice.error}")
        return
    st.info(advice.explanation)
    if advice.used_headlines:
        used = ", ".join(f"#{i}" for i in advice.used_headlines)
        st.caption(f"Headlines the advisor relied on: {used}")
    if not advice.news_used:
        st.caption("No relevant news: this advice comes from price features only.")


def feature_row(result: PipelineResult) -> None:
    """Show the price features exactly as the advisor received them."""
    shown = features_in_percent(result.decision.features)
    labels = {"ret_1d_pct": "1-day return", "ret_5d_pct": "5-day return",
              "ret_20d_pct": "20-day return", "volatility_20d_pct": "20-day volatility"}
    columns = st.columns(len(labels))
    for column, (key, label) in zip(columns, labels.items(), strict=True):
        column.metric(label, f"{shown[key]:.2f}%")
    level = shown["volatility_level"]
    st.caption(f"Volatility level: {level}. Computed in Python, not by the LLM.")


def price_figure(result: PipelineResult, live: bool) -> None:
    """Draw the price chart up to the decision close, with a table view."""
    ticker = result.decision.ticker
    frame = prices.live_prices(ticker) if live else prices.load_prices(ticker)
    end = frame.index[-1] if live else pd.Timestamp(result.decision.date)
    closes = frame.loc[:end, "adj_close"].tail(CHART_DAYS)
    title = f"{ticker}: adjusted close, last {len(closes)} trading days"
    figure = charts.price_chart(list(closes.index), closes.tolist(), title, is_dark())
    st.plotly_chart(figure, theme=None, width="stretch")
    with st.expander("Price table"):
        st.dataframe(closes.round(2).rename("adjusted close").to_frame(), width="stretch")


def headline_table(result: PipelineResult) -> pd.DataFrame:
    """Build the headline table: relevance verdict and sentiment per headline."""
    verdicts = {item.id: item for item in result.relevance}
    labels = {item.id: item for item in result.sentiment}
    rows = []
    for h in result.decision.headlines:
        verdict, label = verdicts.get(h.id), labels.get(h.id)
        rows.append({
            "#": h.id, "Published": h.published or "", "Source": h.source or "",
            "Headline": h.text,
            "Relevant": "" if verdict is None else ("yes" if verdict.relevant else "no"),
            "Sentiment": SENTIMENT_MARK[label.label] if label else "",
            "Why": (label.reason if label else verdict.reason if verdict else ""),
        })
    return pd.DataFrame(rows)


def tint(value: str) -> str:
    """Return the cell style of a sentiment label (the text itself carries the meaning)."""
    color = SENTIMENT_TINT.get(value)
    return f"background-color: {color}" if color else ""


def headlines_section(result: PipelineResult) -> None:
    """Show the headlines with a tinted sentiment cell (the label carries the meaning)."""
    table = headline_table(result)
    if table.empty:
        st.caption("No headlines were available in the 3 days before this decision.")
        return
    relevant = int((table["Relevant"] == "yes").sum())
    st.caption(f"{len(table)} headlines collected, {relevant} relevant.")
    styled = table.style.map(tint, subset=["Sentiment"])
    st.dataframe(styled, hide_index=True, width="stretch")
    for error in result.errors:
        st.warning(f"A pipeline step failed: {error}")


def outcome_section(result: PipelineResult) -> None:
    """Historical mode only: reveal what the price did after the decision."""
    with st.expander("What happened next (never shown to the pipeline)"):
        frame = prices.load_prices(result.decision.ticker)
        day = date.fromisoformat(result.decision.date)
        sessions = load_config()["simulator"]["horizon_days"]
        columns = st.columns(len(sessions))
        for column, (horizon, n) in zip(columns, sessions.items(), strict=True):
            value = prices.forward_return(frame, day, n)
            text = "not known yet" if value is None else f"{value * 100:+.2f}%"
            column.metric(f"Return over the next {HORIZON_TEXT[horizon]}", text)


def trace_panel(result: PipelineResult) -> None:
    """Show the LLM calls of this request from logs/llm_calls.jsonl."""
    calls = trace.read_calls(result.trace_id)
    with st.expander(f"Trace: {len(calls)} LLM calls of this request", expanded=False):
        if not calls:
            st.caption("No calls logged for this request.")
            return
        columns = ["ts", "component", "prompt_version", "model", "cached", "latency_s",
                   "attempts", "valid_first", "error"]
        st.dataframe(pd.DataFrame(calls)[columns], hide_index=True, width="stretch")
        for call in calls:
            st.markdown(f"**{call['component']}** · `{call['prompt_version']}`")
            left, right = st.columns(2)
            left.caption("Input")
            left.json(call["input"], expanded=False)
            right.caption("Output")
            right.code(call["output"] or call["error"] or "", language="json")


def advice_tab() -> None:
    """Render the main tab for the stored request."""
    view = st.session_state.get("view")
    if view is None:
        st.write("Choose a ticker and a date in the sidebar, then press the button.")
        return
    result: PipelineResult = view["result"]
    decision = result.decision
    when = f"live, as of {decision.as_of}" if view["live"] else f"close of {decision.date}"
    st.subheader(f"{decision.company} ({decision.ticker}), {when}")
    if view["note"]:
        st.caption(view["note"])
    advice_card(result)
    feature_row(result)
    price_figure(result, view["live"])
    st.markdown("#### Headlines")
    headlines_section(result)
    if not view["live"]:
        outcome_section(result)
    trace_panel(result)


def simulator_tab() -> None:
    """Show the stored simulator results: equity curve, metrics and trades per set."""
    record = common.load_result("simulator")
    if record is None:
        st.write("No simulator results yet. Run `python -m evals.eval_simulator`.")
        return
    letter = st.radio("Advice stream", list(record["details"]), horizontal=True,
                      format_func=lambda s: SET_NAMES.get(s, s))
    info = record["details"][letter]
    figure = charts.equity_chart(info["days"], info["equity"], info["buy_and_hold_equity"],
                                 is_dark())
    st.plotly_chart(figure, theme=None, width="stretch")
    rows = [m for m in report.table_rows() if m["Component"] == "simulator"
            and m["Metric"].startswith(f"{letter}.")]
    st.dataframe(pd.DataFrame(rows)[["Metric", "Value", "95% CI", "Baseline", "n"]],
                 hide_index=True, width="stretch")
    st.caption(f"{info['signals']} advice items, {info['buy_signals_at_threshold']} confident "
               f"buys, mean exposure {info['mean_exposure']:.1%}. Rules: spec §6.")
    with st.expander(f"Trades ({len(info['trades'])})"):
        st.dataframe(pd.DataFrame(info["trades"]), hide_index=True, width="stretch")
    with st.expander("Equity table"):
        table = pd.DataFrame({"day": info["days"], "simulator": info["equity"],
                              "buy & hold": info["buy_and_hold_equity"]})
        st.dataframe(table, hide_index=True, width="stretch")


def evals_tab() -> None:
    """Show the evaluation table built from evals/results/*.json."""
    rows = report.table_rows()
    if not rows:
        st.write("No results yet. Run `python -m evals.run_all`.")
        return
    st.caption("Every number comes from a file in evals/results/. "
               "Intervals are 95% bootstrap confidence intervals.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    decision = (common.load_result("lookahead") or {}).get("details", {})
    if decision:
        st.markdown(f"**Look-ahead decision:** the LLM peeks = `{decision.get('llm_peeks')}` "
                    f"(rule: {decision.get('rule')}).")
    with st.expander("Raw results files"):
        for component in report.ORDER:
            record = common.load_result(component)
            if record:
                st.markdown(f"`evals/results/{component}.json`")
                st.code(json.dumps(record["metrics"], indent=1)[:4000], language="json")


def main() -> None:
    """Lay out the page: sidebar request, then the three tabs."""
    st.set_page_config(page_title="Stock News Awareness Assistant", layout="wide")
    st.title("Stock News Awareness Assistant")
    st.caption("A course prototype. It is not financial advice.")
    ticker, mode, day, pressed = sidebar()
    if pressed:
        try:
            run_request(ticker, mode, day)
        except Exception as error:  # show the problem instead of a stack trace
            st.error(f"The request failed: {error}")
    advice, simulator, evals = st.tabs(["Advice", "Simulator", "Evals"])
    with advice:
        advice_tab()
    with simulator:
        simulator_tab()
    with evals:
        evals_tab()


main()
