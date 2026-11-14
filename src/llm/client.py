"""LLM client: the ONLY place that calls an LLM.

OpenRouter, model from config, temperature 0, disk cache keyed by (model, prompt version,
input), retries with backoff, a rate limiter, JSON validation with pydantic, one repair
retry on invalid output, and a trace log in logs/llm_calls.jsonl.
"""

import hashlib
import json
import os
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from src.config import ROOT, load_config

Validator = Callable[[BaseModel], str | None]
FATAL_STATUS = {400, 401, 402, 403, 404}  # retrying cannot fix these
MAX_ERROR_CHARS = 2000  # long enough to keep the "retry in Ns" hint of a quota error
LOGGED_ERROR_CHARS = 300
REPAIR = (
    "Your previous answer was not valid ({error}). "
    "Answer again with the corrected JSON only, exactly in the required format."
)
_LOG_LOCK = threading.Lock()


@dataclass
class LLMResult:
    """Outcome of one logical LLM call."""

    data: BaseModel | None
    error: str | None = None
    raw: str | None = None
    cached: bool = False
    latency_s: float = 0.0
    attempts: int = 0
    valid_first: bool = True
    first_error: str | None = None
    usage: dict = field(default_factory=dict)


class RateLimiter:
    """Thread-safe spacing of request starts, with a shared pause after a quota error."""

    def __init__(self) -> None:
        """Start with no waiting time."""
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        """Block until this thread may start a request."""
        interval = 60.0 / load_config()["llm"]["max_rpm"]
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + interval
        if start > now:
            time.sleep(start - now)

    def pause(self, seconds: float) -> None:
        """Push every future request start back (the quota is exhausted)."""
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


_LIMITER = RateLimiter()


def _cache_dir() -> Path:
    """Return the folder of the disk cache."""
    return ROOT / load_config()["paths"]["llm_cache_dir"]


def _log_path() -> Path:
    """Return the trace log file."""
    return ROOT / load_config()["paths"]["llm_log"]


def load_prompt(version: str) -> str:
    """Read a prompt template, e.g. 'sentiment_v1' -> prompts/sentiment_v1.md."""
    path = ROOT / load_config()["paths"]["prompts_dir"] / f"{version}.md"
    return path.read_text(encoding="utf-8")


def cache_key(model: str, version: str, prompt: str, payload: dict) -> str:
    """Hash (model, prompt version, prompt text, input) into a cache key."""
    blob = json.dumps([model, version, prompt, payload], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_cache(key: str) -> str | None:
    """Return the cached raw answer, or None."""
    path = _cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["raw"]


def _write_cache(key: str, version: str, raw: str) -> None:
    """Store a validated raw answer atomically."""
    folder = _cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    record = {"model": load_config()["llm"]["model_id"], "prompt_version": version, "raw": raw}
    temp = folder / f"{key}.{threading.get_ident()}.tmp"
    temp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    temp.replace(folder / f"{key}.json")


def _post(messages: list[dict]) -> tuple[int, dict]:
    """Send one chat request; return (HTTP status, parsed body). Status 0 = network error."""
    cfg = load_config()["llm"]
    load_dotenv(ROOT / ".env")
    body = {
        "model": cfg["model_id"],
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
        "provider": cfg["provider"],
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    try:
        reply = requests.post(
            cfg["api_url"], json=body, headers=headers, timeout=cfg["timeout_seconds"]
        )
        return reply.status_code, reply.json()
    except (requests.RequestException, ValueError) as exc:
        return 0, {"error": {"code": 0, "message": type(exc).__name__}}


def _extract(status: int, body: dict) -> tuple[str | None, int, str | None]:
    """Return (content, error code, error text) from an API reply."""
    problem = body.get("error")
    if status != 200 or problem:
        problem = problem or {}
        code = int(problem.get("code") or status)
        detail = str(problem.get("metadata", {}).get("raw") or problem.get("message", ""))
        return None, code, f"HTTP {code}: {' '.join(detail.split())[:MAX_ERROR_CHARS]}"
    choice = (body.get("choices") or [{}])[0]
    content = choice.get("message", {}).get("content")
    finish = choice.get("finish_reason", "stop")
    if not content or finish != "stop":  # the provider sometimes cuts an answer short
        return None, 0, f"incomplete response (finish_reason={finish})"
    return content, 200, None


def _quota_pause(error: str) -> bool:
    """Pause every thread when the API says the quota is used up; return True if it did."""
    hint = re.search(r"retry in ([\d.]+)s", error)
    if hint:
        _LIMITER.pause(min(float(hint.group(1)), load_config()["llm"]["quota_pause_seconds"]))
    return bool(hint)


def _backoff(attempt: int) -> None:
    """Sleep before the next try, longer after each failure."""
    cfg = load_config()["llm"]
    delay = min(cfg["backoff_seconds"] * 2**attempt, cfg["max_backoff_seconds"])
    time.sleep(delay * (0.5 + random.random() / 2))


def _request(messages: list[dict]) -> tuple[str | None, str | None, int, dict]:
    """Call the API with retries; return (content, error, attempts, token usage)."""
    cfg = load_config()["llm"]
    error, failures, waits, attempts = "no attempt", 0, 0, 0
    while failures <= cfg["max_retries"] and waits <= cfg["max_quota_waits"]:
        _LIMITER.wait()
        attempts += 1
        status, body = _post(messages)
        content, code, error = _extract(status, body)
        if content is not None:
            return content, None, attempts, body.get("usage") or {}
        if code in FATAL_STATUS:
            break
        if _quota_pause(error):
            waits += 1
            continue
        failures += 1
        if failures <= cfg["max_retries"]:
            _backoff(failures - 1)
    return None, f"transport: {error[:LOGGED_ERROR_CHARS]}", attempts, {}


def parse_output(
    raw: str, schema: type[BaseModel], validator: Validator | None = None
) -> tuple[BaseModel | None, str | None]:
    """Validate raw model text against the schema and the extra rule; return (data, error)."""
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        data = schema.model_validate_json(text)
    except ValidationError as exc:
        first = exc.errors()[0]
        return None, f"schema: {first['msg']} at {list(first['loc'])}"
    problem = validator(data) if validator else None
    return (None, f"rule: {problem}") if problem else (data, None)


def _fresh_call(
    system: str, payload: dict, schema: type[BaseModel], validator: Validator | None
) -> LLMResult:
    """Ask the model; on invalid output ask once more with the error, then give up."""
    user = json.dumps(payload, ensure_ascii=False, indent=1)
    base = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    messages, result = base, LLMResult(None, "no attempt", valid_first=False)
    for round_no in range(load_config()["llm"]["json_retries"] + 1):
        raw, error, tries, usage = _request(messages)
        result.attempts += tries
        if raw is None:
            result.error = error
            return result
        result.raw, result.usage = raw, usage
        result.data, error = parse_output(raw, schema, validator)
        if result.data is not None:
            result.error, result.valid_first = None, round_no == 0
            return result
        result.first_error = result.first_error or error
        result.error = f"invalid_output: {error}"
        repair = {"role": "user", "content": REPAIR.format(error=error)}
        messages = [*base, {"role": "assistant", "content": raw}, repair]
    return result


def _log(
    component: str, version: str, payload: dict, result: LLMResult, trace_id: str | None
) -> None:
    """Append one call to the trace log (never contains the API key)."""
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "trace_id": trace_id,
        "component": component,
        "prompt_version": version,
        "model": load_config()["llm"]["model_id"],
        "input": payload,
        "output": result.raw,
        "error": result.error,
        "latency_s": result.latency_s,
        "cached": result.cached,
        "attempts": result.attempts,
        "valid_first": result.valid_first,
        "first_error": result.first_error,
        "usage": {k: result.usage.get(k) for k in ("prompt_tokens", "completion_tokens")},
    }
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK, path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_llm(
    component: str,
    prompt_version: str,
    payload: dict,
    schema: type[BaseModel],
    validator: Validator | None = None,
    trace_id: str | None = None,
) -> LLMResult:
    """Run one logical LLM call: cache, request, validate, one repair retry, log."""
    system = load_prompt(prompt_version)
    key = cache_key(load_config()["llm"]["model_id"], prompt_version, system, payload)
    started = time.monotonic()
    result = None
    raw = _read_cache(key)
    if raw is not None:
        data, _error = parse_output(raw, schema, validator)
        if data is not None:
            result = LLMResult(data, None, raw, cached=True)
    if result is None:
        result = _fresh_call(system, payload, schema, validator)
        if result.data is not None:
            _write_cache(key, prompt_version, result.raw)
    result.latency_s = round(time.monotonic() - started, 3)
    _log(component, prompt_version, payload, result, trace_id)
    return result
