"""Eval helpers: the metric code is tested like any other code."""

import numpy as np
import pandas as pd
import pytest

from evals import common, eval_advisor, eval_judge, eval_relevance, eval_sentiment, report
from evals.eval_simulator import curve_metric


def test_bootstrap_is_deterministic_and_brackets_the_value() -> None:
    """Same seed, same interval; the interval contains the point estimate."""
    flags = np.array([1.0] * 60 + [0.0] * 40)
    first, second = common.bootstrap(common.mean, flags), common.bootstrap(common.mean, flags)
    assert first == second
    assert first["value"] == pytest.approx(0.6) and first["n"] == 100
    assert 0.45 < first["ci_low"] < 0.6 < first["ci_high"] < 0.75
    assert common.bootstrap(common.mean, np.array([]))["value"] is None


def test_bootstrap_difference() -> None:
    """Two independent samples: the difference of means with an interval around it."""
    stat = common.bootstrap_difference([1.0] * 90 + [0.0] * 10, [1.0] * 70 + [0.0] * 30)
    assert stat["value"] == pytest.approx(0.2)
    assert stat["ci_low"] < 0.2 < stat["ci_high"] and stat["ci_low"] > 0


def test_relevance_metrics() -> None:
    """Precision, recall and F1 on a tiny hand-checked case."""
    truth = np.array([True, True, True, False, False])
    predicted = np.array([True, True, False, True, False])
    assert eval_relevance.precision(truth, predicted) == pytest.approx(2 / 3)
    assert eval_relevance.recall(truth, predicted) == pytest.approx(2 / 3)
    assert eval_relevance.f1(truth, predicted) == pytest.approx(2 / 3)
    assert eval_relevance.confusion(truth, predicted) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    keep_all = np.ones(5, dtype=bool)
    assert eval_relevance.recall(truth, keep_all) == 1.0
    assert eval_relevance.precision(truth, keep_all) == pytest.approx(0.6)


def test_sentiment_metrics() -> None:
    """Macro-F1 punishes a model that only predicts the majority class."""
    truth = np.array(["neutral"] * 8 + ["positive", "negative"])
    always_neutral = np.array(["neutral"] * 10)
    assert eval_sentiment.accuracy(truth, always_neutral) == pytest.approx(0.8)
    assert eval_sentiment.macro_f1(truth, always_neutral) == pytest.approx((16 / 18) / 3)
    assert eval_sentiment.macro_f1(truth, truth) == 1.0


def advice_frame() -> pd.DataFrame:
    """Six decisions: wins and losses for buy and sell, one hold, one invalid."""
    rows = [
        ("buy", "1d", 55, 0.02, -0.01, 0.01),   # hit over 1d
        ("buy", "1w", 65, 0.01, -0.03, -0.01),  # miss over 1w
        ("sell", "1d", 85, -0.03, 0.02, -0.02),  # hit over 1d
        ("sell", "1w", 100, -0.01, 0.01, 0.04),  # miss over 1w
        ("hold", "1d", 60, 0.05, 0.05, 0.01),   # not directional
        ("buy", "1d", 90, 0.05, 0.05, 0.01),   # invalid output: excluded
    ]
    frame = pd.DataFrame(rows, columns=["action", "horizon", "confidence", "fwd_1d", "fwd_1w",
                                        "last_move"])
    frame["incomplete"] = False
    frame["error"] = ["", "", "", "", "", "invalid_output: schema"]
    frame["fwd"] = np.where(frame["horizon"] == "1d", frame["fwd_1d"], frame["fwd_1w"])
    frame["side"] = frame["action"].map({"buy": 1, "sell": -1, "hold": 0})
    return frame


def test_advisor_direction_and_baselines() -> None:
    """Hit rate, both baselines and the directional share on a hand-checked frame."""
    calls = eval_advisor.directional(advice_frame())
    assert len(calls) == 4
    hit, share = eval_advisor.direction_rows("A", calls, valid_n=5)
    assert hit["value"] == pytest.approx(0.5) and hit["n"] == 4
    assert hit["baselines"] == {"always_buy": 0.5, "same_as_last_move": 1.0}
    assert share["value"] == pytest.approx(0.8)


def test_advisor_horizon_and_confidence_buckets() -> None:
    """Returns follow the advised side; bucket edges are [50,60), [60,80), [80,100]."""
    calls = eval_advisor.directional(advice_frame())
    row = eval_advisor.horizon_rows("A", calls)[0]
    assert row["value"] == pytest.approx((2 - 3 + 3 - 1) / 4)  # percent, side applied
    assert row["baselines"] == {"fixed_1d": pytest.approx(1.75), "fixed_1w": pytest.approx(-1.75)}
    buckets = {r["name"]: (r["value"], r["n"]) for r in eval_advisor.confidence_rows("A", calls)}
    assert buckets == {"A.hit_rate_confidence_50_60": (1.0, 1),
                       "A.hit_rate_confidence_60_80": (0.0, 1),
                       "A.hit_rate_confidence_80_100": (0.5, 2)}


def test_curve_metrics_from_daily_returns() -> None:
    """The bootstrap metrics rebuild an equity curve from daily returns."""
    cfg = {"simulator": {"trading_days_per_year": 252}}
    returns = np.array([0.10, -0.50, 0.20])
    assert curve_metric("total_return", cfg)(returns) == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert curve_metric("max_drawdown", cfg)(returns) == pytest.approx(0.5)


def test_report_formatting() -> None:
    """Numbers, intervals and baselines are printed in one consistent way."""
    assert [report.number(v) for v in (None, 0.5, 300, 4.0, 1.0, True)] == [
        "n/a", "0.500", "300", "4", "1.000", "True"]
    row = {"ci_low": 0.1, "ci_high": 0.2, "baselines": {"keep_all": 0.561, "x": None}}
    assert report.interval(row) == "[0.100, 0.200]"
    assert report.baselines(row) == "keep_all 0.561; x n/a"
    table = report.markdown_table([dict.fromkeys(report.HEADER, "v")])
    assert table.splitlines()[0].startswith("| Component | Metric")


def test_render_docs_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document templates take every number from a results record."""
    from evals import render_docs

    record = {
        "run_date": "2026-11-14", "details": {"confusion": {"tp": 37}},
        "metrics": [{"name": "C.hit_rate", "value": 0.5312, "ci_low": 0.47, "ci_high": 0.59,
                     "n": 173, "baselines": {"always_buy": 0.48}, "passed": 18}],
    }
    monkeypatch.setattr(render_docs.common, "load_result",
                        lambda component: record if component == "advisor" else None)
    text = ("{{advisor:C.hit_rate}} {{advisor:C.hit_rate|pct}} {{advisor:C.hit_rate:ci|pct}} "
            "{{advisor:C.hit_rate:n}} {{advisor:C.hit_rate:base:always_buy|pct}} "
            "{{advisor:C.hit_rate:field:passed}} {{advisor:@confusion.tp}} {{advisor:#run_date}} "
            "{{advisor:missing_metric}} {{judge:anything}}")
    assert render_docs.render(text) == (
        "0.531 53.1% [47.0%, 59.0%] 173 48.0% 18 37 2026-11-14 n/a n/a")


def test_target_checks_handle_missing_results() -> None:
    """A criterion is met, not met, or not measured when a number is missing."""
    from evals.eval_targets import all_met, at_least

    assert at_least(0.81, 0.80) is True and at_least(0.79, 0.80) is False
    assert at_least(None, 0.80) is None and at_least(0.5, None) is None
    assert all_met(True, True) is True
    assert all_met(True, None) is None
    assert all_met(True, None, False) is False


def test_planted_false_claims() -> None:
    """Even items get the invented buyback, odd items a 20-day return off by the config shift."""
    item = {"explanation": "Up 2%.", "inputs": {"price_features": {"ret_20d_pct": 3.0},
                                                 "headlines": []}}
    text, key = eval_judge.planted(item, 0)
    assert text == f"Up 2%. {eval_judge.PLANTED_FACT}" and key == "buyback"
    text, key = eval_judge.planted(item, 1)
    assert key == "9.50" and text == "Up 2%. The 20-day return is 9.50%."  # 3.0 + 6.5
    with_buyback = {**item, "inputs": {**item["inputs"], "headlines": [{"text": "Buyback news"}]}}
    assert eval_judge.planted(with_buyback, 0)[1] == "9.50"
