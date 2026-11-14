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

A4 — Shared metadata editor (gate: after) — commit `e88cde8`

## Request

**Summary**
- New `ui/common/metadata_editor.py`: `render_metadata_editor(context)` = collapsed "Edit metadata"
  expander (form + `data_editor`: Experiment Name, Replicate, Cell Line, extra columns; optional "Add column").
  Rendered on Gating, Analysis and Plot tabs.
- `st.session_state.sample_metadata` (indexed by Sample) is built from the loaded experiment's events
  (`metadata_from_events`) and rebuilt if the sample set changes; cleared with `clear_experiment_state`.
- Save → `apply_sample_metadata` on base/gated/analysis experiments (no re-gate), `reinfer_replicates`
  when names change (hand-edited replicates kept), blank text → "NA", `loaded_files` entries updated so a
  later rebuild keeps edits.
- Cell_Line: `cell_line_changes` separates pure renames (gates in `cell_line_gates` / `cell_line_singlet_gates`
  re-keyed) from reassignments (affected old+new lines stored in `lines_need_regate`, shown as a warning in the expander).
- New pure helpers + tests (`tests/test_sample_metadata.py`, `tests/test_metadata_editor.py`); `apptest_walk.py`
  checks the expander and `sample_metadata` on the Plot tab. `apply_sample_metadata` also updates `MockExperiment.conditions`.
- The old "edit metadata in File Loading above" caption in the Gating "Not locked" expander was replaced.

**Tests**: `apptest_walk.py` → ALL CHECKS PASSED; `pytest tests -q` → 26 passed.

**Unsure / not done**
- Step 1 (pre-load) editors still use their own two `data_editor`s from A2; they are not yet the shared
  component (they work on file paths before any experiment exists). Say if you want them unified here or in C4.
- `lines_need_regate` is only surfaced in the editor expander; the gating step chip is C4/C2 work. Gates of flagged
  lines are not deleted.
- AppTest cannot drive `data_editor` cells, so the save path is covered by unit tests with a fake experiment, not end-to-end.
- `sample_metadata` is a DataFrame, so it is skipped by session save/restore (the loader file list carries the edits).
- Cache clearing keyed on metadata hash is not implemented (no such caches found yet).

## Blocked tools / problems

None (heredoc-with-python commands were denied by permissions, so I used the Edit/Write tools instead).
