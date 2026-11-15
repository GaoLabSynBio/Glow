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

Decision: changes: here's a problem i observed at analysis phase. now the gate selection has a big (~2s) lag between I click or drag on the region to gate and the gate to be actually shown. Can you minimize the time lag?

## Item
D1 Sidebar stepper (commit 807260f)

## Request
- Sidebar radio labels: "Gating ✓" / "Analysis ✓" when complete; "(unavailable)" + reason caption (load files first / finish gating first). Navigation is NOT blocked (st.radio can't disable options, and the harness visits Plot before finalize).
- 175 px sidebar CSS and "Navigation" title removed.
- Full banner only on landing page; one-line "Glow · Flow Analysis Pipeline" once base_experiment exists.
- Developer Mode → sidebar "Advanced" expander. Start New Experiment → sidebar popover with confirm button.
- Tests: apptest ALL CHECKS PASSED; pytest 30 passed.
- Unsure: Plot/Canvas get no ✓ (no completion signal exists); popover/expander not exercised by apptest; Analysis is labelled "unavailable" after load until gating_complete, though it can partly work on a gated experiment.

## Blocked tools / problems
An inline `python3 - <<EOF` edit script was denied by permissions; I did the edits with the Edit tool instead.
