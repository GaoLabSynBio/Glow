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
D1 revise #3 — first click sometimes ignored (commit 837ebb6)

## Request
Your feedback: a click on the histogram sometimes does nothing; clicking the same spot again works.
- Likely cause (not browser-verified): after a click, Plotly keeps that point "selected" in the chart's widget state. The next click on that same point, or on a neighbouring invisible catcher marker that Plotly resolves to the same point, toggles the selection off. That sends an empty selection, so nothing changes. The second click re-selects it, which matches what you saw.
- Fix (`ui/analysis/view.py`): the chart's widget key now includes a nonce. The nonce is bumped each time a point click is consumed, so the chart is rebuilt with no lingering selection and every click is a fresh select. Box (drag) selection is unchanged, since it already works.
- Tests: apptest ALL CHECKS PASSED; pytest 30 passed.
- Unsure: apptest has no browser, so I couldn't reproduce or confirm the fix. Please try rapid clicks at the same spot and at nearby spots. Side effect: the chart remounts after each click, so there may be a brief flicker. If clicks still drop, the next step is `st.fragment` or a custom click component.

## Blocked tools / problems
Inline `python3 - <<EOF` edits, `sed -i` and shell redirects are denied by permissions; used the Edit and Write tools.
