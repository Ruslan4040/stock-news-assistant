"""Read the LLM trace log (used by the UI trace panel and the client eval)."""

import json
import uuid

from src.llm import client


def new_trace_id() -> str:
    """Return a fresh id that groups the LLM calls of one request."""
    return uuid.uuid4().hex


def read_calls(trace_id: str | None = None) -> list[dict]:
    """Return logged calls, optionally only those of one request."""
    path = client._log_path()
    if not path.exists():
        return []
    calls = []
    with path.open(encoding="utf-8") as log:
        for line in log:
            if line.strip() and (trace_id is None or f'"{trace_id}"' in line):
                call = json.loads(line)
                if trace_id is None or call.get("trace_id") == trace_id:
                    calls.append(call)
    return calls
