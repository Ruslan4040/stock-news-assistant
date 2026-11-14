"""Run the pipeline over the prepared sets in parallel and store the results.

Run: python -m src.pipeline.batch [--limit N]
Without --limit it runs `evals.batch_limit` decisions per set (config.yaml), always the first
ones of a fixed random order, so a larger limit later only adds decisions.
Resumable: every successful LLM answer is in the disk cache, so a second run only repeats
the calls that failed. Results go to data/processed/runs/advice_<set>.jsonl.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import ROOT, load_config
from src.data import sets
from src.llm.trace import new_trace_id
from src.pipeline.advisor import Advice
from src.pipeline.run import PipelineResult, advise, label_news, run_anonymized
from src.schemas import Decision

SKIPPED = "skipped: upstream transport error"
PROGRESS_EVERY = 20


def runs_path(letter: str) -> Path:
    """Return the results file of one set."""
    return ROOT / load_config()["paths"]["runs_dir"] / f"advice_{letter.lower()}.jsonl"


def incomplete(result: PipelineResult) -> bool:
    """Say whether a result is missing because the API failed (not because of the model)."""
    return any("transport" in error or error.endswith(SKIPPED) for error in result.errors)


def run_one(decision: Decision) -> PipelineResult:
    """Run one decision; do not ask the advisor when a language step failed in transport."""
    partial = label_news(decision, new_trace_id())
    if incomplete(partial):
        partial.advice = Advice(error=SKIPPED)
        partial.errors.append(f"advisor: {SKIPPED}")
        return partial
    return advise(partial)


def run_pair(decision: Decision, twin: Decision | None) -> list[PipelineResult]:
    """Run a decision and, for set A, its anonymized set-B twin right after it."""
    result = run_one(decision)
    if twin is None:
        return [result]
    if incomplete(result):
        skipped = PipelineResult(decision=twin, advice=Advice(error=SKIPPED),
                                 errors=[f"advisor: {SKIPPED}"], trace_id=new_trace_id())
        return [result, skipped]
    return [result, run_anonymized(twin, result)]


def run_order(letter: str) -> list[str]:
    """Return the decision ids of set A or C in the fixed random order used for runs."""
    rng = random.Random(load_config()["seed"])
    orders = {}
    for name in "AC":  # one generator, always drawn in the same sequence
        ids = [decision.id for decision in sets.load_set(name)]
        rng.shuffle(ids)
        orders[name] = ids
    return orders[letter]


def shuffled_tasks(limit: int | None) -> list[tuple[Decision, Decision | None]]:
    """Return (decision, twin) tasks, A and C interleaved, in the fixed random order."""
    by_id = {d.id: d for letter in "ABC" for d in sets.load_set(letter)}
    pairs_a = [(by_id[i], by_id["B" + i[1:]]) for i in run_order("A")[:limit]]
    singles_c = [(by_id[i], None) for i in run_order("C")[:limit]]
    return [task for pair in zip(pairs_a, singles_c, strict=True) for task in pair]


def save(results: list[PipelineResult]) -> dict[str, int]:
    """Write the results per set (sorted by id) and return how many each file holds."""
    counts = {}
    for letter in "ABC":
        rows = sorted((r for r in results if r.decision.set == letter), key=lambda r: r.decision.id)
        path = runs_path(letter)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(r.model_dump(), ensure_ascii=False) for r in rows]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[letter] = len(rows)
    return counts


def load_results(letter: str) -> list[PipelineResult]:
    """Load the stored results of one set."""
    path = runs_path(letter)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PipelineResult.model_validate_json(line) for line in lines if line.strip()]


def run_all(limit: int | None = None) -> list[PipelineResult]:
    """Run every task in a thread pool, saving a checkpoint as results arrive."""
    tasks = shuffled_tasks(limit)
    results: list[PipelineResult] = []
    with ThreadPoolExecutor(load_config()["llm"]["max_workers"]) as pool:
        futures = [pool.submit(run_pair, decision, twin) for decision, twin in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            results.extend(future.result())
            if done % PROGRESS_EVERY == 0 or done == len(tasks):
                missing = sum(1 for r in results if incomplete(r))
                print(f"[batch] {done}/{len(tasks)} tasks, saved {save(results)}, "
                      f"incomplete {missing}", flush=True)
    return results


def main() -> None:
    """Parse the options and run the sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    default = load_config()["evals"]["batch_limit"]
    parser.add_argument("--limit", type=int, default=default, help="decisions per set")
    args = parser.parse_args()
    results = run_all(args.limit)
    failed = [r.decision.id for r in results if incomplete(r)]
    print(f"[batch] finished: {len(results)} results, {len(failed)} incomplete (re-run to retry)")


if __name__ == "__main__":
    main()
