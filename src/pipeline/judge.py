"""Explanation judge (component 7, LLM call #4, eval only): does the explanation invent facts."""

from pydantic import BaseModel

from src.config import load_config
from src.llm.client import call_llm

COMPONENT = "judge"


class Claim(BaseModel):
    """One factual claim of the explanation and the verdict on it."""

    text: str
    supported: bool
    reason: str


class JudgeOutput(BaseModel):
    """What the LLM must return."""

    claims: list[Claim]


class JudgeResult(BaseModel):
    """The component's result; `error` is set when the LLM output stayed invalid."""

    claims: list[Claim] = []
    error: str | None = None


def run(advisor_inputs: dict, explanation: str, trace_id: str | None = None) -> JudgeResult:
    """Split an explanation into claims and check each one against the advisor's inputs."""
    if not explanation.strip():
        return JudgeResult()
    version = load_config()["prompts"][COMPONENT]
    payload = {"inputs": advisor_inputs, "explanation": explanation}
    result = call_llm(COMPONENT, version, payload, JudgeOutput, None, trace_id)
    if result.data is None:
        return JudgeResult(error=result.error)
    return JudgeResult(claims=result.data.claims)
