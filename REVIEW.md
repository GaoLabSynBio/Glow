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

C2 — Live/Dead step card (gate: after). Commit 567f222. (C1 done earlier this run: 58a9884.)

## Request

**Summary (PARTIAL implementation of C2)**
- `ui/gating/ui_gating.py`: Scale / Interaction / Colormap moved out of the left column into a 3-column toolbar at the top of the plot column (same widget keys).
- Model-context selector: now a caption "Model context: X (from Cell Line metadata)" when the cell line maps to a known model; the selector appears only when the line is not recognised.

**Not done (please say if you want these, or whether to defer)**
- Accept / "Adjust gate manually" / Add-to-canvas row already sits under the plot in `col_plots`, but I did not reorder it or add the "Adjust gate manually" button.
- Single result line under the plot: the two green banners ("Gate Applied", "Final Live Cells") and "Gated Lines" are still produced by the existing flow (~lines 1160-1330) and I did not merge them. That block is deeply nested with st.rerun/return paths; refactoring it blind, with only the smoke walk as validation, seemed risky.
- Representative-file selector is still where it was, not in the toolbar.

**Verification**: apptest_walk ALL CHECKS PASSED; pytest 30 passed. The walk does not exercise the MLP branch with a named cell line or the toolbar interaction, so the layout is untested visually.

## Blocked tools / problems

`sed -i` and compound shell commands were prompted for approval in this session, so I made file edits with the Edit tool instead. Nothing else blocked.
