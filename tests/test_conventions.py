"""Coding conventions from AGENTS.md §4, checked automatically."""

import ast
from pathlib import Path

import pytest

from src.config import ROOT

MAX_LINES = 40
FOLDERS = ["src", "evals", "app"]
FILES = sorted(p for folder in FOLDERS for p in (ROOT / folder).rglob("*.py"))


def functions(path: Path) -> list[ast.FunctionDef]:
    """Return every function and method defined in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_functions_are_short_documented_and_typed(path: Path) -> None:
    """Every function has at most 40 lines, a docstring and a return annotation."""
    for node in functions(path):
        length = node.end_lineno - node.lineno + 1
        assert length <= MAX_LINES, f"{node.name}: {length} lines"
        assert ast.get_docstring(node), f"{node.name}: missing docstring"
        assert node.returns is not None, f"{node.name}: missing return type"
