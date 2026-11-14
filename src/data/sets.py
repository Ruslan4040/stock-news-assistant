"""Readers and writers for the prepared evaluation sets."""

import json
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config
from src.schemas import Decision


def processed(name: str) -> Path:
    """Return a path inside data/processed."""
    return ROOT / load_config()["paths"]["processed_dir"] / name


def write_decisions(name: str, decisions: list[Decision]) -> None:
    """Write decisions as JSON lines."""
    path = processed(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(d.model_dump(), ensure_ascii=False) for d in decisions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_set(letter: str) -> list[Decision]:
    """Load one prepared set (A, B or C)."""
    path = processed(f"set_{letter.lower()}.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Decision.model_validate_json(line) for line in lines if line.strip()]


def load_outcomes() -> pd.DataFrame:
    """Load the ground-truth forward returns (kept apart from the model inputs)."""
    return pd.read_csv(processed("outcomes.csv")).set_index("id")


def real_id(decision_id: str) -> str:
    """Map a set-B id to the id of its set-A twin (other ids are returned unchanged)."""
    return "A" + decision_id[1:] if decision_id.startswith("B-") else decision_id
