# Agent log (human-owned — the agent must not edit this file)

How I steered the coding agent. This is the source for the "use of the AI coding assistant" part of the report and video.

## Setup
- Agent: Claude Code (reads `CLAUDE.md`, which imports `AGENTS.md` + `docs/PROJECT_SPEC.md`).
- Context control: rules in `AGENTS.md`, scope in the spec, one CRISP-DM phase per session (`/clear` between phases), skills in `.claude/skills/` for repeated workflows.
- Quality control: plan-before-code rule, a unit test + an eval per component, ruff + pytest gate, a no-look-ahead test.

## Log
| Date | Phase / task | What the agent did wrong or unexpectedly | What I changed (AGENTS.md / skill / prompt) | Result |
|---|---|---|---|---|
| | | | | |

## Reflection (fill in at the end)
- What worked well:
- What did not work:
- What I would put into AGENTS.md from day one next time:
