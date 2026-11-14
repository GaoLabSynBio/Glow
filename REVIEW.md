# Review handshake

<!-- The automated runner writes a request below and stops. You answer by editing the
     Decision line, then the runner picks it up (it polls this file every 60 min while waiting).

     Decision values:
       pending                      - waiting for you (runner wrote this)
       approved                     - go ahead (for gate:before) / accept the commit (for gate:after)
       changes: <what to change>    - runner revises, commits, and asks again
       skip                         - skip this item, mark it [x] (skipped), continue
       idle                         - nothing pending (runner resets to this after reading your decision)
-->

Decision: pending

## Item

A5 — Replace write-once `base_experiment` with a file-list hash (gate: after) — commit `3be5b46`

## Request

- `ui/gating/ui_gating.py`: new `_file_list_key(loaded_files)` (sha1 of sorted file paths). The gating tab
  compares it with `st.session_state['_base_files_key']`; `base_experiment` is dropped and rebuilt only when
  the file list differs. Metadata/label edits don't change the key, so they never rebuild.
- `ui/gating/ui_file_loader.py`: `clear_experiment_state` also clears `_base_files_key`.
- Removed the stale "Configuring Gating..." line (this was also C1's first bullet).

**Tests**: `apptest_walk.py` → ALL CHECKS PASSED; `pytest tests -q` → 26 passed (tail: `26 passed in 1.43s`).

**Unsure**: the hash uses paths only, so re-uploading a different file to the same path would not rebuild
(upload paths are processed copies, so this seems unlikely). I added no dedicated test for the key;
the walk only covers the unchanged-list path.

## Blocked tools / problems

None. (A `python3` heredoc needed approval and was not run; I used the Edit tool instead.)
