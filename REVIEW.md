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
D1 revise — analysis gate-selection lag (commit 123378c)

## Request
Fixed the ~2s lag in Analysis → 1D Histogram when clicking/dragging to set a threshold (`ui/analysis/view.py`):
- Removed the 5 `st.rerun()` calls after applying a selection (state is set before the inputs/figure render, so the extra full rerun only doubled the delay).
- Cached the parent-gate data fetch (`get_channel_data_for_samples`) across reruns, keyed on experiment/channel/samples/parent gate/gates.
- Cached KDE curves per (data signature, sample, log, axis range); KDE data capped at 20k points (fixed seed) for speed.
- Replaced the per-element Python NaN filter with vectorised numpy.
- Tests: apptest ALL CHECKS PASSED; pytest 30 passed.
- Unsure: not timed in a real browser (no harness for it). Other analysis plots (quadrant/scatter paths around lines 990, 1326, 1417, 1615) still use `st.rerun()` after selection and were not touched; say if you see lag there too. KDE subsampling only changes curves for samples with >20k events.

## Blocked tools / problems
Inline `python3 - <<EOF` edits are denied by permissions; used the Edit tool.
