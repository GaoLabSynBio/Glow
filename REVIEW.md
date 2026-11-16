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
E2 Live preview (gate: after) — commit 40b87a7

## Request
Summary: histogram, bar and scatter plots now render immediately and re-render on every settings change; the "Generate Plot" buttons are gone. Preview uses at most 50k events (fixed seed) with a caption showing "Preview: 50,000 of N events". A "Render full data" toggle renders everything. `ui/plot/base.py` has the helpers `render_full_toggle`, `preview_subsample` and a guarded `add_plot_to_canvas`, which refuses to add a subsampled preview to the canvas. Pairwise plot unchanged (it already samples and keeps its button).

Tests: `apptest_walk.py` ALL CHECKS PASSED; pytest 30 passed.

Unsure about:
- The walk script doesn't exercise plot rendering, so the live render path is untested by automation.
- Bar plot means/error bars come from the subsample when >50k events; the caption says so, but you may prefer bar plots to always use full data.
- Download SVG buttons export the previewed figure (not blocked, unlike canvas).
- The Plot tab can now run heavy renders on every widget change for datasets ≤50k; if that feels slow, caching is the next step.
- Not done: the "≤50k → full on Finalize" is manual via the toggle.

## Blocked tools / problems
