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
D1 revise #2 — unreliable gate click + residual lag (commit ab6a024)

## Request
Your feedback: still some lag; clicks to set range fail ~half the time.
- Click failure (root cause found): Plotly only reports a click when it lands on a trace point. Range mode had no click catcher at all (only curve markers), and binary mode's catcher only covered the y=0 baseline. Now one invisible marker grid (200 x-points × 12 y-levels, spanning the full y extent) covers the whole plot in both modes, so a click anywhere registers. Log mode uses the log-spaced x_domain.
- Lag: no additional change this round beyond the previous one; the remaining cost is the full-script Streamlit rerun that any selection triggers. Not timed in a browser. If still laggy, next step is wrapping the Analysis plot area in `st.fragment` so only it reruns — larger change, say if you want it.
- Tests: apptest ALL CHECKS PASSED; pytest 30 passed.
- Unsure: the click fix can't be exercised by apptest (no browser); please try it. Clicking an already-selected point toggles it off in Plotly (harmless, value stays).

## Previous round (commit 123378c)
Fixed the ~2s lag in Analysis → 1D Histogram when clicking/dragging to set a threshold (`ui/analysis/view.py`):
- Removed the 5 `st.rerun()` calls after applying a selection (state is set before the inputs/figure render, so the extra full rerun only doubled the delay).
- Cached the parent-gate data fetch (`get_channel_data_for_samples`) across reruns, keyed on experiment/channel/samples/parent gate/gates.
- Cached KDE curves per (data signature, sample, log, axis range); KDE data capped at 20k points (fixed seed) for speed.
- Replaced the per-element Python NaN filter with vectorised numpy.
- Tests: apptest ALL CHECKS PASSED; pytest 30 passed.
- Unsure: not timed in a real browser (no harness for it). Other analysis plots (quadrant/scatter paths around lines 990, 1326, 1417, 1615) still use `st.rerun()` after selection and were not touched; say if you see lag there too. KDE subsampling only changes curves for samples with >20k events.

## Blocked tools / problems
Inline `python3 - <<EOF` edits are denied by permissions; used the Edit tool.
