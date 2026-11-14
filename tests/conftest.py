"""Shared test setup: the LLM is always mocked and never touches the real cache or log."""

import json

import pytest

from src.llm import client


class FakeLLM:
    """A scripted stand-in for the OpenRouter API."""

    def __init__(self) -> None:
        """Start with no scripted replies."""
        self.replies: list = []
        self.calls: list[list[dict]] = []

    def queue(self, *replies: object) -> None:
        """Script replies: a str or dict is an answer; a (status, body) tuple is an error."""
        self.replies.extend(replies)

    def post(self, messages: list[dict]) -> tuple[int, dict]:
        """Play the next scripted reply; fail the test on an unexpected call."""
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("Unexpected LLM call: no scripted reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, tuple):
            return reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        choice = {"message": {"content": content}, "finish_reason": "stop"}
        return 200, {"choices": [choice], "usage": usage}

    def last_input(self) -> dict:
        """Return the JSON payload of the most recent request."""
        return json.loads(self.calls[-1][1]["content"])


@pytest.fixture(autouse=True)
def llm(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Redirect cache and log to a temp folder and replace the network call."""
    fake = FakeLLM()
    monkeypatch.setattr(client, "_cache_dir", lambda: tmp_path / "llm_cache")
    monkeypatch.setattr(client, "_log_path", lambda: tmp_path / "llm_calls.jsonl")
    monkeypatch.setattr(client._LIMITER, "wait", lambda: None)
    monkeypatch.setattr(client, "_post", fake.post)
    return fake
