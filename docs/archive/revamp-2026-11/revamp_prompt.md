You are continuing the Glow UI revamp as an unattended run. Work only inside this repository.

## Files that define the work
- `UX_REVIEW.md` — the spec (findings + proposals). Read the sections relevant to the item you work on.
- `PROGRESS.md` — the ordered checklist with status and gate tags. It is the single source of truth.
- `REVIEW.md` — the handshake file with the human reviewer (Xiaowei). See its header for the protocol.
- `tools/apptest_walk.py` — end-to-end check (must pass before every commit).

## Protocol for this run (follow in order)

1. **Branch and tree.** Run `git status -sb` and `git branch --show-current`. You must be on `revamp`.
   If the tree is dirty, those are leftovers from an interrupted run: run the checks; if they pass,
   commit them for the `[~]` item; otherwise `git stash` them, note it in the Run log, and continue.

2. **Pending review?** Read `REVIEW.md`.
   - `Decision: pending` → print exactly `WAITING_FOR_REVIEW` on its own line and stop.
   - `Decision: approved` → if the item was `gate: before`, it is now cleared to implement (continue to
     step 4 with it); if `gate: after`, mark it `[x] (<hash>)`. Reset REVIEW.md to `Decision: idle` with
     empty sections.
   - `Decision: changes: …` → apply the requested changes to that item, run the checks, commit
     `revamp: <id> revise — <short>`, write a new summary in REVIEW.md with `Decision: pending`, print
     `WAITING_FOR_REVIEW`, stop.
   - `Decision: skip` → mark the item `[x] (skipped)`, reset REVIEW.md to idle, continue.

3. **Reviewer notes.** Read "Notes for next run" in `PROGRESS.md`. Act on each note (it may change
   priorities or add items), then move it to "Notes applied" with a one-line description of what you did.

4. **Next item.** Take the first item that is `[~]`, else the first `[ ]`, in file order.
   - If `gate: before` and it has not been approved in step 2: write a design note in `REVIEW.md`
     (≤ 1 page: what will change, which files/functions, how it will be verified, risks), set the item
     to `[?]`, set `Decision: pending`, print `WAITING_FOR_REVIEW`, stop. Do not edit code for it.
   - Otherwise set it `[~]`, implement it per `UX_REVIEW.md`, keeping the change scoped to the item.

5. **Verify.** Run `/home/shared/envs/glow_envs/bin/python -I tools/apptest_walk.py` and
   `/home/shared/envs/glow_envs/bin/python -m pytest tests -q`. Both must pass. If you cannot make
   them pass with reasonable effort, `git restore .` (and `git clean` is NOT allowed — leave new files
   and list them), write the failure in REVIEW.md under "Blocked tools / problems" with
   `Decision: pending`, print `WAITING_FOR_REVIEW`, stop.

6. **Commit.** `git add -A` (never add files under `data/`), then commit with message
   `revamp: <id> <title>` and body = 2–4 lines on what changed, ending with the attribution trailer
   `Co-Authored-By: Claude <noreply@anthropic.com>`. Mark the item `[x] (<short hash>)`.

7. **Gate after?** If the item is `gate: after`: write in REVIEW.md the commit hash, a summary of the
   change, the test output tail, and anything you were unsure about; set `[?]`, `Decision: pending`,
   print `WAITING_FOR_REVIEW`, stop.

8. **Continue.** Append one line to the Run log in `PROGRESS.md` (date, item, result). If you have
   completed fewer than 3 items in this run and items remain, go back to step 4. Otherwise print
   `CONTINUE` if items remain, or `DONE` if every item is `[x]`.

## Rules
- Never push. Never `git reset --hard`, rebase, or rewrite history. Never touch files under `data/`
  except reading the test FCS files. Never modify `.claude/`.
- If a command is denied by permissions, do not work around it: record it under "Blocked tools /
  problems" in REVIEW.md, set `Decision: pending`, print `WAITING_FOR_REVIEW`, stop.
- Keep the app runnable at every commit. Prefer small, readable functions; match existing code style.
- When UX_REVIEW.md and PROGRESS.md disagree, PROGRESS.md (and the reviewer's notes) win.
- Do not use screenshots or browsers; `tools/apptest_walk.py` is the validation harness. Extend it
  when an item adds a user-visible step worth checking.
- End every run with exactly one of these as the final line: `WAITING_FOR_REVIEW`, `CONTINUE`, `DONE`.
