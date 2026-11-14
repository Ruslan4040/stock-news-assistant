"""Check the success criteria of phase 1 against the results files.

Run: python -m evals.eval_targets
Writes evals/results/targets.json with one row per criterion: target, result, met.
The targets were fixed in docs/crisp_dm/01_business.md before any eval was run.
"""

from collections.abc import Callable

import numpy as np

from evals import common
from evals.report import number

COMPONENT = "targets"
MET = {True: "yes", False: "no", None: "not measured"}


class Results:
    """Reads metric values from the results files."""

    def __init__(self) -> None:
        """Load every results file once."""
        self.records: dict[str, dict | None] = {}

    def get(self, component: str, name: str, field: str = "value") -> float | None:
        """Return a field of a metric row, a baseline (field 'base:<name>') or None."""
        if component not in self.records:
            self.records[component] = common.load_result(component)
        record = self.records[component] or {"metrics": []}
        row = next((r for r in record["metrics"] if r["name"] == name), None)
        if row is None:
            return None
        if field.startswith("base:"):
            return row["baselines"].get(field[5:])
        return row.get(field)


def at_least(value: float | None, bound: float | None) -> bool | None:
    """Return value >= bound, or None when either number is missing."""
    return None if value is None or bound is None else bool(value >= bound)


def all_met(*checks: bool | None) -> bool | None:
    """Combine checks: False if any fails, None if any is missing, else True."""
    if any(check is False for check in checks):
        return False
    return None if any(check is None for check in checks) else True


def data_rows(r: Results) -> list[tuple]:
    """Criteria for the deterministic components: data, features, client, UI."""
    g = r.get
    zero = [g("data", "missing_trading_days_rate"), g("data", "duplicate_headlines_rate_in_sets"),
            g("data", "lookahead_violation_rate")]
    return [
        ("Data collector", "missing days, duplicates in sets, look-ahead violations: all 0",
         " / ".join(number(v) for v in zero),
         all_met(*[None if v is None else v == 0 for v in zero])),
        ("Features", "exact match on 15 of 15 cases",
         f"{number(g('features', 'exact_match_rate', 'matches'))} of "
         f"{number(g('features', 'exact_match_rate', 'n'))}",
         at_least(g("features", "exact_match_rate"), 1.0)),
        ("LLM client", "valid JSON on the first answer ≥ 0.95; invalid after retry ≤ 0.02",
         f"{number(g('llm_client', 'valid_json_first_answer_rate'))}; "
         f"{number(g('llm_client', 'invalid_after_retry_rate'))}",
         all_met(at_least(g("llm_client", "valid_json_first_answer_rate"), 0.95),
                 at_least(0.02, g("llm_client", "invalid_after_retry_rate")))),
        ("UI", "all demo cases pass the checklist",
         f"{number(g('ui', 'demo_cases_passed_rate', 'passed'))} of "
         f"{number(g('ui', 'demo_cases_passed_rate', 'n'))}",
         at_least(g("ui", "demo_cases_passed_rate"), 1.0)),
    ]


def language_rows(r: Results) -> list[tuple]:
    """Criteria for relevance, sentiment and the explanations."""
    g = r.get
    f1, macro = g("relevance", "f1"), g("sentiment", "phrasebank.macro_f1")
    finbert = g("sentiment", "phrasebank.macro_f1", "base:finbert")
    neutral = g("sentiment", "phrasebank.macro_f1", "base:always_neutral")
    return [
        ("Relevance filter", "F1 ≥ 0.80, precision ≥ 0.80, F1 above keep-all",
         f"F1 {number(f1)}, precision {number(g('relevance', 'precision'))}, "
         f"keep-all F1 {number(g('relevance', 'f1', 'base:keep_all'))}",
         all_met(at_least(f1, 0.80), at_least(g("relevance", "precision"), 0.80),
                 at_least(f1, g("relevance", "f1", "base:keep_all")))),
        ("Sentiment (PhraseBank)", "macro-F1 ≥ 0.80, above always-neutral, within 0.05 of FinBERT",
         f"{number(macro)} (FinBERT {number(finbert)}, always-neutral {number(neutral)})",
         all_met(at_least(macro, 0.80), at_least(macro, neutral),
                 None if finbert is None else at_least(macro, finbert - 0.05))),
        ("Sentiment (fresh headlines)", "macro-F1 ≥ 0.70",
         number(g("sentiment", "fresh.macro_f1")),
         at_least(g("sentiment", "fresh.macro_f1"), 0.70)),
        ("Explanations", "claims supported ≥ 0.90; judge vs label agreement ≥ 0.80",
         f"{number(g('judge', 'claims_supported_rate'))}; "
         f"{number(g('judge', 'judge_label_agreement'))}",
         all_met(at_least(g("judge", "claims_supported_rate"), 0.90),
                 at_least(g("judge", "judge_label_agreement"), 0.80))),
    ]


def advisor_rows(r: Results) -> list[tuple]:
    """Criteria for the advisor and the trader, on set C (the unseen period)."""
    g = r.get
    hit = g("advisor", "C.hit_rate")
    buy, last = g("advisor", "C.hit_rate", "base:always_buy"), g(
        "advisor", "C.hit_rate", "base:same_as_last_move")
    chosen = g("advisor", "C.mean_return_chosen_horizon_pct")
    fixed = [g("advisor", "C.mean_return_chosen_horizon_pct", f"base:fixed_{h}")
             for h in ("1d", "1w")]
    low, high = g("advisor", "C.hit_rate_confidence_50_60"), g(
        "advisor", "C.hit_rate_confidence_80_100")
    ret, draw = g("simulator", "C.total_return"), g("simulator", "C.max_drawdown")
    hold_draw = g("simulator", "C.max_drawdown", "base:buy_and_hold")
    best_fixed = None if None in fixed else max(fixed)
    return [
        ("Advisor: direction (set C)", "hit rate above always-buy and same-as-last-move",
         f"{number(hit)} vs {number(buy)} and {number(last)}",
         all_met(at_least(hit, buy), at_least(hit, last))),
        ("Advisor: horizon (set C)", "mean return with the chosen horizon ≥ the better fixed one",
         f"{number(chosen)}% vs {number(best_fixed)}%", at_least(chosen, best_fixed)),
        ("Advisor: confidence (set C)", "bucket 80–100 at least 5 points above bucket 50–60",
         f"{number(high)} vs {number(low)}",
         None if high is None or low is None else at_least(high, low + 0.05)),
        ("Trader (set C)", "total return > 0 and max drawdown below buy & hold",
         f"return {number(ret)}, drawdown {number(draw)} vs {number(hold_draw)}",
         all_met(None if ret is None else ret > 0, at_least(hold_draw, draw))),
    ]


def lookahead_row() -> tuple:
    """The look-ahead experiment must end in a clear decision."""
    record = common.load_result("lookahead") or {"details": {}}
    decision = record["details"].get("llm_peeks")
    text = "no result" if decision is None else f"the LLM peeks: {decision}"
    return ("Look-ahead experiment", "a clear yes / no by the rule of spec §8", text,
            None if decision is None else True)


def table(rows: list[tuple]) -> str:
    """Render the criteria as a Markdown table."""
    lines = ["| Component | Target (set in phase 1) | Result | Met |", "|---|---|---|---|"]
    lines += [f"| {c} | {target} | {result} | {MET[met]} |" for c, target, result, met in rows]
    return "\n".join(lines)


def main() -> None:
    """Evaluate every criterion and write the results file."""
    common.limit_argument(__doc__)
    r = Results()
    builders: list[Callable[[Results], list[tuple]]] = [data_rows, language_rows, advisor_rows]
    rows = [row for build in builders for row in build(r)] + [lookahead_row()]
    measured = np.array([float(met) for *_rest, met in rows if met is not None])
    metrics = [common.metric_row("criteria_met_rate", common.bootstrap(common.mean, measured),
                                 met=int(measured.sum()), not_measured=len(rows) - len(measured))]
    details = {"table": table(rows),
               "rows": [dict(zip(("component", "target", "result", "met"), row, strict=True))
                        for row in rows]}
    path = common.write_result(COMPONENT, metrics, "all results files", details=details)
    print(details["table"])
    common.show(path)


if __name__ == "__main__":
    main()
