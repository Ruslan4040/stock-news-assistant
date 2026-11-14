"""Shared eval helpers: bootstrap confidence intervals, result files, parallel map."""

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.config import ROOT, load_config

WORST_ERRORS = 10  # rows kept in <component>_errors.csv (AGENTS.md §7)


def bootstrap(metric: Callable[..., float], *columns: Sequence) -> dict:
    """Return a metric over aligned columns with its n and a percentile bootstrap CI."""
    cfg = load_config()
    arrays = [np.asarray(column) for column in columns]
    n = len(arrays[0])
    if n == 0:
        return {"value": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(cfg["seed"])
    draws = []
    for _ in range(cfg["evals"]["bootstrap_n"]):
        rows = rng.integers(0, n, n)
        draws.append(metric(*(array[rows] for array in arrays)))
    tail = (1 - cfg["evals"]["ci_level"]) / 2 * 100
    low, high = np.nanpercentile(np.asarray(draws, dtype=float), [tail, 100 - tail])
    return {"value": float(metric(*arrays)), "ci_low": float(low), "ci_high": float(high), "n": n}


def bootstrap_difference(first: Sequence, second: Sequence) -> dict:
    """Return mean(first) - mean(second) for two independent samples, with a bootstrap CI."""
    cfg = load_config()
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if not len(a) or not len(b):
        return {"value": None, "ci_low": None, "ci_high": None, "n": min(len(a), len(b))}
    rng = np.random.default_rng(cfg["seed"])
    draws = [
        a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
        for _ in range(cfg["evals"]["bootstrap_n"])
    ]
    tail = (1 - cfg["evals"]["ci_level"]) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return {"value": float(a.mean() - b.mean()), "ci_low": float(low), "ci_high": float(high),
            "n": min(len(a), len(b))}


def mean(values: np.ndarray) -> float:
    """Return the mean of a numeric or boolean array (NaN when empty)."""
    return float(np.mean(values)) if len(values) else float("nan")


def metric_row(name: str, stat: dict, baselines: dict | None = None, **extra: object) -> dict:
    """Build one row of the `metrics` list of a results file."""
    return {"name": name, **stat, "baselines": baselines or {}, **extra}


def results_dir(limit: int | None) -> Path:
    """Return evals/results, or its smoke subfolder for --limit runs."""
    base = ROOT / load_config()["paths"]["results_dir"]
    folder = base / "smoke" if limit else base
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_result(
    component: str,
    metrics: list[dict],
    eval_data: str,
    prompt_version: str | None = None,
    limit: int | None = None,
    details: dict | None = None,
) -> Path:
    """Write evals/results/<component>.json with the fields required by AGENTS.md §7."""
    record = {
        "component": component,
        "run_date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "model": load_config()["llm"]["model_id"] if prompt_version else None,
        "prompt_version": prompt_version,
        "eval_data": eval_data,
        "limit": limit,
        "metrics": metrics,
        "details": details or {},
    }
    path = results_dir(limit) / f"{component}.json"
    text = json.dumps(record, indent=1, ensure_ascii=False, default=str)
    path.write_text(text, encoding="utf-8")
    if prompt_version and not limit:
        archive = results_dir(None) / "archive"
        archive.mkdir(exist_ok=True)
        (archive / f"{component}__{prompt_version}.json").write_text(text, encoding="utf-8")
    return path


def write_errors(component: str, rows: list[dict], limit: int | None = None) -> Path:
    """Write the worst errors (already sorted, worst first) to <component>_errors.csv."""
    path = results_dir(limit) / f"{component}_errors.csv"
    rows = rows[:WORST_ERRORS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path


def load_result(component: str) -> dict | None:
    """Read a full (not smoke) results file, or None when it does not exist."""
    path = results_dir(None) / f"{component}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def parallel_map(function: Callable, items: Sequence) -> list:
    """Apply a function to items in a thread pool, keeping the input order."""
    with ThreadPoolExecutor(load_config()["llm"]["max_workers"]) as pool:
        return list(pool.map(function, items))


def limit_argument(description: str) -> int | None:
    """Parse the standard --limit option of an eval script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N items")
    return parser.parse_args().limit


def show(path: Path) -> None:
    """Print a short summary of a results file."""
    record = json.loads(path.read_text(encoding="utf-8"))
    print(f"[{record['component']}] -> {path.relative_to(ROOT)}")
    for row in record["metrics"]:
        value = "n/a" if row["value"] is None else f"{row['value']:.4f}"
        interval = "" if row["ci_low"] is None else f" [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        print(f"  {row['name']}: {value}{interval} n={row['n']} baselines={row['baselines']}")
