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

B2 — Remove per-tab scaling branches (gate: after). Commit 5626ee8. Builds on B1 (14e3034).

## Request

**Summary**
- `ui/plot/view.py`: removed the high-res block (warning, "Show Raw Values" toggle, per-tab ÷factor copy).
- `ui/plot/gating_component.py`: `apply_gates_to_dataframe(df, gates)` no longer takes metadata/show_raw and applies no unit correction; the 4 plot UIs updated.
- `ui/analysis/view.py`: removed toggle, `apply_scaling`/`scaling_factor` and all multiply/divide branches; gates are stored and drawn in displayed units.
- `glow/analysis/positive.py`: `get_channel_data_for_samples` lost `apply_scaling`; returns data as loaded.
- New `ui/common/units.py`: one caption "Values shown ÷10,000; exports are raw." on Plot and Analysis (falls back to base_experiment metadata if the derived experiment dropped it).
- `ui/compensation/view.py` still calls `detect_and_scale_high_res`; it is a no-op now.
- Note: after B1 alone, Plot would have double-scaled (it keyed on `is_high_res`), so B2 was done in the same run.

**Verification**: apptest_walk passes (new checks: caption on Plot/Analysis, no raw toggle, loaded values < 1e6); pytest 30 passed.

**Unsure / please check**
- Analysis gates were already stored in scaled units, but `gating_component` treated them as raw and divided again. Now consistent, but I only verified via the smoke walk, not with a saved gate on high-res data.
- Gating-tab density plots use `log10(x+1)` for display; with scaled values the +1 offset is relatively larger (cosmetic). Model inputs use raw units via `raw_unit_factor`.

## Blocked tools / problems

None.
