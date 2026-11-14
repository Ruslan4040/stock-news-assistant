"""Eval of the explanations with the judge (component 7) and of the judge itself.

Run: python -m evals.eval_judge [--limit N]
1. The judge splits each sampled explanation into claims and checks them against the inputs.
   Metric: share of claims supported.
2. The claims of the first explanations go to evals/labels/judge_claims.csv for labeling.
   Metric: agreement between the judge and those labels.
"""

import json

import numpy as np
import pandas as pd

from evals import common
from src.config import ROOT, load_config
from src.pipeline import advisor, batch, judge
from src.pipeline.run import PipelineResult

COMPONENT = "judge"
SOURCE_SETS = "AC"
SHEET_COLUMNS = ["explanation_id", "claim_no", "claim", "inputs", "explanation",
                 "human_supported", "labeled_by"]


def sample_results(limit: int | None) -> list[PipelineResult]:
    """Pick the first valid explanations of each source set in the fixed run order.

    The run order is random but fixed, so the sample stays the same when more decisions
    are run later.
    """
    per_set = (limit or load_config()["evals"]["judge_n"]) // len(SOURCE_SETS)
    picked = []
    for letter in SOURCE_SETS:
        stored = {r.decision.id: r for r in batch.load_results(letter)}
        ordered = [stored[i] for i in batch.run_order(letter) if i in stored]
        valid = [r for r in ordered if not batch.incomplete(r) and not r.advice.error]
        picked.append(valid[:per_set])
    return [result for pair in zip(*picked, strict=False) for result in pair]


def judge_one(result: PipelineResult) -> dict:
    """Judge one explanation; keep the inputs so that a human can check the same claims."""
    inputs = advisor.build_payload(result.decision, result.labeled)
    verdict = judge.run(inputs, result.advice.explanation)
    return {"id": result.decision.id, "inputs": inputs, "explanation": result.advice.explanation,
            "claims": [claim.model_dump() for claim in verdict.claims], "error": verdict.error}


def supported_share(supported: np.ndarray, total: np.ndarray) -> float:
    """Return supported claims / all claims over a group of explanations."""
    return float(supported.sum() / total.sum()) if total.sum() else float("nan")


def sheet_path() -> str:
    """Return the path of the claim-labeling sheet."""
    return str(ROOT / load_config()["paths"]["labels_dir"] / "judge_claims.csv")


def write_sheet(judged: list[dict]) -> None:
    """Create the labeling sheet for the first explanations (never overwrites labels)."""
    if (ROOT / sheet_path()).exists():
        return
    rows = []
    for item in judged[: load_config()["evals"]["judge_human_n"]]:
        for number, claim in enumerate(item["claims"], start=1):
            rows.append({"explanation_id": item["id"], "claim_no": number, "claim": claim["text"],
                         "inputs": json.dumps(item["inputs"], ensure_ascii=False),
                         "explanation": item["explanation"], "human_supported": "",
                         "labeled_by": ""})
    pd.DataFrame(rows, columns=SHEET_COLUMNS).to_csv(sheet_path(), index=False)


def agreement(judged: list[dict]) -> tuple[dict, list[dict], dict]:
    """Compare the judge with the labeled claims; return the metric, disagreements, provenance."""
    sheet = pd.read_csv(sheet_path(), dtype=str, keep_default_na=False)
    labeled = sheet[sheet["human_supported"].isin(["y", "n"])]
    verdicts = {(item["id"], n): claim
                for item in judged for n, claim in enumerate(item["claims"], start=1)}
    same, disagreements = [], []
    for row in labeled.itertuples():
        claim = verdicts.get((row.explanation_id, int(row.claim_no)))
        if claim is None:
            continue
        human = row.human_supported == "y"
        same.append(1.0 if human == claim["supported"] else 0.0)
        if human != claim["supported"]:
            disagreements.append(error_row("judge_vs_label", row.explanation_id, claim,
                                           "y" if human else "n"))
    provenance = labeled["labeled_by"].value_counts().to_dict()
    return common.bootstrap(common.mean, np.array(same)), disagreements, provenance


def error_row(kind: str, explanation_id: str, claim: dict, label: str = "") -> dict:
    """Build one row of the errors file."""
    return {"type": kind, "id": explanation_id, "claim": claim["text"],
            "judge_supported": claim["supported"], "label_supported": label,
            "judge_reason": claim["reason"]}


def main() -> None:
    """Judge the sampled explanations, build the sheet, score agreement, write the results."""
    limit = common.limit_argument(__doc__)
    judged = common.parallel_map(judge_one, sample_results(limit))
    done = [item for item in judged if not item["error"]]
    supported = np.array([sum(c["supported"] for c in item["claims"]) for item in done])
    total = np.array([len(item["claims"]) for item in done])
    if not limit:
        write_sheet(done)
    stat, disagreements, provenance = agreement(done) if not limit else (
        {"value": None, "ci_low": None, "ci_high": None, "n": 0}, [], {})
    metrics = [
        common.metric_row("claims_supported_rate",
                          common.bootstrap(supported_share, supported, total),
                          claims=int(total.sum()), explanations=len(done)),
        common.metric_row("judge_label_agreement", stat),
    ]
    unsupported = [error_row("unsupported_claim", item["id"], claim)
                   for item in done for claim in item["claims"] if not claim["supported"]]
    details = {"judge_errors": len(judged) - len(done), "labeled_by": provenance,
               "claims_per_explanation": round(float(total.mean()), 2) if len(total) else None}
    version = load_config()["prompts"][COMPONENT]
    data = "100 advisor explanations (sets A and C) + evals/labels/judge_claims.csv"
    path = common.write_result(COMPONENT, metrics, data, version, limit, details)
    common.write_errors(COMPONENT, disagreements + unsupported, limit)
    common.show(path)


if __name__ == "__main__":
    main()
