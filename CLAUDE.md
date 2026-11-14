@AGENTS.md
@docs/PROJECT_SPEC.md

## Claude Code specifics
- Use plan mode for any task that creates a new component or changes more than 3 files.
- Project skills live in `.claude/skills/`. Use them instead of improvising the same steps.
- Start each CRISP-DM phase in a fresh session (`/clear`) to keep the context small. These two files reload automatically.
