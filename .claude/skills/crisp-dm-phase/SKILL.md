---
name: crisp-dm-phase
description: Update the CRISP-DM log for the current phase. Use at the end of every task, when moving to the next phase, or when the user says "log the phase" or "update CRISP-DM".
---

# Update the CRISP-DM log

1. Identify the current phase (1–6, see AGENTS.md §2). If unclear, ask.
2. Open `docs/crisp_dm/0<N>_<phase>.md`. If it does not exist, create it from the template below.
3. Append a dated entry. Never rewrite earlier entries.

```
## <YYYY-MM-DD> — <task title>
- What was done: 2–4 bullets.
- Decisions & why: each decision with the alternative that was considered.
- Artifacts: files created or changed (paths).
- Evidence: numbers (row counts, test results, metric values). No claim without a number or a file.
- Gate status: is the gate from AGENTS.md §2 met? yes / no + what is missing.
- Open questions for the human.
```

4. If this task went back to an earlier phase because of an evaluation result, also append to `docs/crisp_dm/iterations.md`: the trigger (metric + value), the phase revisited, the change made, and the metric before → after.
5. Keep each entry under ~15 lines.

## Template for a new phase file
```
# CRISP-DM <N>. <Phase name>
Purpose: <one sentence from AGENTS.md §2>
Gate: <gate from AGENTS.md §2>
```
