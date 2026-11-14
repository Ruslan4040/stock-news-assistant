"""LLM client: cache, retries, JSON validation, repair retry, trace log (network mocked)."""

import json

import pytest
from pydantic import BaseModel

from src.llm import client, trace


class Answer(BaseModel):
    """A tiny schema for the tests."""

    label: str
    score: int


PAYLOAD = {"text": "Company X beats estimates"}
GOOD = {"label": "positive", "score": 1}


def call(payload: dict = PAYLOAD, validator: client.Validator | None = None) -> client.LLMResult:
    """Run one call with a real prompt file."""
    return client.call_llm("sentiment", "sentiment_v1", payload, Answer, validator, "trace-1")


def test_valid_answer_is_parsed_logged_and_cached(llm) -> None:
    """A valid answer is returned, logged once and served from the cache next time."""
    llm.queue(GOOD)
    first = call()
    assert first.data == Answer(**GOOD) and not first.cached and first.attempts == 1
    assert llm.calls[0][0]["role"] == "system" and llm.last_input() == PAYLOAD
    second = call()  # no scripted reply left: a network call would fail the test
    assert second.cached and second.data == first.data and len(llm.calls) == 1
    log = trace.read_calls("trace-1")
    assert [c["cached"] for c in log] == [False, True]
    assert log[0]["component"] == "sentiment" and log[0]["prompt_version"] == "sentiment_v1"
    assert log[0]["input"] == PAYLOAD and json.loads(log[0]["output"]) == GOOD


def test_cache_key_depends_on_model_version_and_input() -> None:
    """Any change of model, prompt version, prompt text or input gives a new key."""
    base = client.cache_key("m", "v1", "prompt", {"a": 1})
    others = [("m2", "v1", "prompt", {"a": 1}), ("m", "v2", "prompt", {"a": 1}),
              ("m", "v1", "prompt!", {"a": 1}), ("m", "v1", "prompt", {"a": 2})]
    assert all(client.cache_key(*other) != base for other in others)
    assert client.cache_key("m", "v1", "prompt", {"a": 1}) == base


def test_invalid_json_is_repaired_once(llm) -> None:
    """Invalid output triggers one retry that carries the error back to the model."""
    llm.queue("not json at all", GOOD)
    result = call()
    assert result.data == Answer(**GOOD) and not result.valid_first and result.attempts == 2
    assert result.first_error.startswith("schema: Invalid JSON")
    assert "not valid" in llm.calls[1][-1]["content"]
    assert llm.calls[1][-2] == {"role": "assistant", "content": "not json at all"}


def test_invalid_twice_returns_error_and_is_not_cached(llm) -> None:
    """After the single retry the client returns an error result; nothing is cached."""
    llm.queue('{"label": "positive"}', '{"label": 5}')
    result = call()
    assert result.data is None and result.error.startswith("invalid_output: schema")
    llm.queue(GOOD)
    assert call().data == Answer(**GOOD)  # the failed call was not cached
    assert [c["error"] is None for c in trace.read_calls()] == [False, True]


def test_validator_rule_and_fenced_json(llm) -> None:
    """An extra rule can reject a schema-valid answer; code fences are tolerated."""
    def positive_only(answer: Answer) -> str | None:
        """Reject scores below 1."""
        return None if answer.score > 0 else "score must be positive"

    llm.queue({"label": "x", "score": 0}, "```json\n" + json.dumps(GOOD) + "\n```")
    result = call(validator=positive_only)
    assert result.data == Answer(**GOOD) and not result.valid_first
    assert "score must be positive" in llm.calls[1][-1]["content"]


def test_transport_errors_are_retried_with_backoff(llm, monkeypatch: pytest.MonkeyPatch) -> None:
    """503 and 500 are retried; the sleeps grow."""
    sleeps: list[float] = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)
    busy = (503, {"error": {"code": 503, "message": "high demand"}})
    broken = (502, {"error": {"code": 502, "metadata": {"raw": "Internal error"}}})
    llm.queue(busy, broken, (0, {"error": {"code": 0, "message": "Timeout"}}), GOOD)
    result = call()
    assert result.data == Answer(**GOOD) and result.attempts == 4 and result.valid_first
    assert len(sleeps) == 3 and sleeps[2] > sleeps[0]


def test_truncated_answer_is_a_transport_error(llm, monkeypatch: pytest.MonkeyPatch) -> None:
    """An answer cut short by the provider is retried, not counted as invalid JSON."""
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    cut = {"choices": [{"message": {"content": '{"label": "pos'}, "finish_reason": "length"}]}
    llm.queue((200, cut), GOOD)
    result = call()
    assert result.data == Answer(**GOOD) and result.valid_first and result.attempts == 2


def test_gives_up_after_max_retries(llm, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead API yields a transport error result instead of an exception."""
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    retries = client.load_config()["llm"]["max_retries"]
    llm.queue(*[(503, {"error": {"code": 503, "message": "high demand"}})] * (retries + 1))
    result = call()
    assert result.data is None and result.error.startswith("transport: HTTP 503")
    assert result.attempts == retries + 1


def test_fatal_status_is_not_retried(llm) -> None:
    """A wrong key or a bad request fails at once."""
    llm.queue((401, {"error": {"code": 401, "message": "No auth credentials found"}}))
    result = call()
    assert result.attempts == 1 and "HTTP 401" in result.error


def test_quota_error_pauses_all_threads(llm, monkeypatch: pytest.MonkeyPatch) -> None:
    """A quota message with a retry hint pauses the shared limiter."""
    pauses: list[float] = []
    monkeypatch.setattr(client._LIMITER, "pause", pauses.append)
    raw = "You exceeded your current quota. " + "x" * 600 + " Please retry in 12.5s."
    quota = (429, {"error": {"code": 429, "metadata": {"raw": raw}}})
    llm.queue(quota, GOOD)
    assert call().data == Answer(**GOOD)
    assert pauses == [12.5]


def test_log_never_contains_the_key(llm, monkeypatch: pytest.MonkeyPatch) -> None:
    """The API key is not written to the trace log."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-secret-123")
    llm.queue(GOOD)
    call()
    assert "sk-test-secret-123" not in client._log_path().read_text()
