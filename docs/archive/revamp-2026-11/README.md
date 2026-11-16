# UI revamp — November 2026 (archived)

Record of the unattended, human-gated revamp of the Glow Streamlit UI, run on branch `revamp`
from 2026-11-13 to 2026-11-16. All 22 items (A1–G3) were completed; commits are prefixed
`revamp: <id> <title>` and listed per item in `PROGRESS.md`.

| File | What it is |
|---|---|
| `UX_REVIEW.md` | The spec: usability review of the pre-revamp app, proposals, and an implementation-status table with commit hashes (§0). |
| `PROGRESS.md` | The ordered checklist that drove the work: status, commit hash and "partial" notes per item, reviewer notes, run log. |
| `REVIEW.md` | Final state of the human/agent handshake file (idle). The per-item design notes and review requests are in the git history of this file (`git log -p --follow -- docs/archive/revamp-2026-11/REVIEW.md`). |
| `revamp.log` | Full runner log: every headless pass, its end-of-run summary, and each review decision. |
| `revamp_prompt.md` | The 8-step protocol each headless `claude -p` pass followed. |
| `run_revamp.sh` | The runner loop (pass → `WAITING_FOR_REVIEW` / `CONTINUE` / `DONE`; hourly poll of `REVIEW.md`). |
| `claude-settings.local.json.example` | Permission allow/deny list used by the headless passes. |

`tools/apptest_walk.py`, the end-to-end AppTest harness written for the revamp, stays in `tools/`
because it is the project's general smoke test.

## Reusing the runner for a future round

The scripts assume the repo-root layout they were written for. To run another round:

1. Copy `revamp_prompt.md`, `run_revamp.sh`, `PROGRESS.md` and `REVIEW.md` back to where the
   prompt expects them (`tools/` for the first two, repo root for the other two), write a new
   checklist in `PROGRESS.md`, reset `REVIEW.md` to `Decision: idle`.
2. `cp claude-settings.local.json.example .claude/settings.local.json` (gitignored).
3. Create/checkout a branch named `revamp` (the runner refuses to run on any other branch).
4. `tmux new -s glow`, then `MODEL=claude-opus-5-5 ./tools/run_revamp.sh`.

Open follow-ups left by this round are listed in the "Implementation status" table of
`UX_REVIEW.md` and the `partial:` notes in `PROGRESS.md` (C2–C7, E1, F2, G2).
