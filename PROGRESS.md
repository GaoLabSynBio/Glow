# Glow UI revamp — progress checklist

Spec: `UX_REVIEW.md`. Branch: `revamp` (off `update`). One commit per item: `revamp: <id> <title>`.
Protocol for the automated runner: `tools/revamp_prompt.md`. Review handshake: `REVIEW.md`.

Status: `[ ]` todo · `[~]` in progress · `[x] (<hash>)` done · `[?]` waiting for review.
Gate: `before` = design note must be approved before any edit · `after` = commit, then wait
for approval before the next item · `none` = implement, test, commit, continue.

Verification for every item: `tools/apptest_walk.py` passes and `pytest tests -q` passes
before the commit. A failing run must not be committed.

---

## Phase A — Metadata foundations (UX_REVIEW §2, §2b)

- [x] (427f508) A1  `glow/data_loading/sample_metadata.py`: `build_sample_metadata(preview_meta)`, `default_experiment_names()`, `infer_replicates()`, `sample_label_map()`, `apply_sample_metadata(experiment, df)` + unit tests (`tests/test_sample_metadata.py`)   gate: before
- [x] (0cd4692) A2  Metadata editor defaults: Experiment Name ← file stem, Replicate ← inferred integer (re-inferred on name edit), Cell Line ← path guess or blank (never `HEK`), auto-filled values visibly marked, "fill down" helper   gate: none
- [x] (11b06f6) A3  Stable `Sample` = file stem (folder-prefix only on collision); display label = Experiment Name (+ `rep N`) in gating file selectors, singlet selector, Analysis stats table, overlay multiselect, legends; remove `_`-join in `glow/analysis/stats.py:68`   gate: none
- [x] (94d0dc4) A4  Shared `render_metadata_editor(context)`; used in Step 1 and as a collapsed "Edit metadata" expander on Gating / Analysis / Plot tabs; save → `apply_sample_metadata` on base/gated/analysis experiments, re-infer replicates, no re-gate; Cell_Line rename re-keys `cell_line_gates`/`cell_line_singlet_gates`; genuine reassignment flags only affected lines   gate: after
- [x] (3be5b46) A5  Replace write-once `base_experiment` (`ui_gating.py:397`) with a file-list hash: rebuild only when the file list changes; label edits never rebuild   gate: after

## Phase B — One unit convention (UX_REVIEW §5.2, B2)

- [?] B1  Scale high-res data once at load in `DataLoader` (divide channel columns by `scaling_factor`, set `is_scaled=True`); export writes raw (multiply back) with the `GlowMetadata` header; CSV import respects `is_scaled`; MLP inputs unaffected (models centre/standardise log inputs)   gate: before
- [ ] B2  Remove per-tab scaling branches: `ui/plot/view.py:30-70`, `ui/plot/gating_component.py:199-243`, `ui/analysis/view.py` raw-toggle paths, `glow/analysis/positive.py:384-396`; replace the three yellow warnings with one caption "Values shown ÷10,000; exports are raw"; gates stored in displayed units everywhere   gate: after

## Phase C — Gating tab layout (UX_REVIEW §3, §4)

- [ ] C1  Remove stale/duplicate text: "Configuring Gating..." (`ui_gating.py:394`), duplicated headers before the plot, "Applying N filters..." leftovers in all four plot UIs, pandas `observed=False` warning in `bar_ui.py:305`   gate: none
- [ ] C2  Live/Dead step card: compact toolbar (scale · colormap · interaction · representative file) directly above the plot; Accept / "Adjust gate manually" / Add-to-canvas row directly below; single result line under the plot; model-context selector read-only (derived from metadata Cell Line) with an override link only when unknown   gate: after
- [ ] C3  Singlet step: identical card layout to C2 (same column widths, toolbar, button row, result line)   gate: none
- [ ] C4  Collapse completed steps into summary chips with an Edit link ("Live/Dead ✓ MLP · 87.6 % · Edit"); collapse Step 1 after load into "12 files · N experiments · Edit metadata"; remove the duplicate "Loaded Metadata Example" table; bulk-metadata upload expander collapsed by default and placed below the editor   gate: none
- [ ] C5  Consolidate resets: one "Redo" per step chip; remove the extra global reset buttons; keep per-cell-line reset inside the step   gate: none
- [ ] C6  Rename "Final Data Settings" → "Channels to analyze" as a compact multiselect with structural channels as a caption; move Export and "Run compensation" onto a completion card ("Gating complete — Next: Analysis ▶ · Export CSV · Compensation"); sticky "Next ▶" in the stepper   gate: none
- [ ] C7  Cache `auto_gate_live_cells` and singlet prediction by parameter hash (`st.cache_data`) so slider changes don't re-gate from scratch   gate: none

## Phase D — Navigation (UX_REVIEW §1)

- [ ] D1  Sidebar stepper with completion state (✓ marks, unavailable steps greyed with a reason); drop the fixed 175 px sidebar CSS; compact one-line header after the landing page; Developer Mode moved into an "Advanced" expander   gate: after
- [ ] D2  "Start New Experiment" moved to the bottom of the sidebar behind a confirmation popover   gate: none

## Phase E — Plot tab (UX_REVIEW §6)

- [ ] E1  Plot type selector moves from the sidebar to the top of the settings panel; "Gating (Optional)" → collapsed "Filter events" expander with an active-filter chip; remove the redundant "Grouping by" info box; scatter default Y ≠ X; actionable notice when the only group is `NA` (link to metadata editor); standardise scale options and download buttons across plot types   gate: none
- [ ] E2  Live preview: histogram/bar/scatter re-render on settings change using a ≤50k-event subsample, with "Render full" for canvas/export   gate: after

## Phase F — Analysis tab (UX_REVIEW §5)

- [ ] F1  Summary table: Experiment Name first, Replicate, other condition columns, then metrics; `Sample` hidden by default; CSV download; "group by experiment" toggle (mean ± SD across replicates)   gate: none
- [ ] F2  Gate creator as one card: channel + parent on one row, draggable threshold line (Plotly editable shape), percentile quick-picks from a chosen control sample, live % positive per sample before saving   gate: after

## Phase G — Canvas & polish (UX_REVIEW §7, §8)

- [ ] G1  Canvas: text ↑/↓ buttons, delete confirmation, two-pane layout (thumbnail list + selected plot with its controls), "Export all" ZIP   gate: none
- [ ] G2  Messaging pass: one button system (primary = next step, secondary = other, destructive = confirm), sticky infos → captions, `st.progress` during file loading, consistent toast/success usage   gate: none
- [ ] G3  Wrap-up: README workflow section updated, UX_REVIEW.md items annotated with commit hashes, full `tools/apptest_walk.py` + pytest run, final summary in REVIEW.md   gate: after

---

## Notes for next run
<!-- Xiaowei: write anything here (corrections, priorities, "skip E2", "also do X"). The runner
     reads this section first, acts on it, and moves each note to "Notes applied" below. -->

## Notes applied
<!-- runner appends: date · note · what was done -->

## Run log
<!-- runner appends one line per run: date · items touched · result -->
2026-11-13 · A1 · done (427f508), pytest 20 passed, apptest passed
2026-11-13 · A2 · done (0cd4692); blank Cell_Line single-population gating path hangs apptest (>300s) when banner filter added, deferred to C2 — gating currently shows 'NA' as a cell line
2026-11-13 · A3 · done (11b06f6); 3 items this run (A1-A3)
2026-11-13 · A4 · committed (e88cde8), pytest 26 passed, apptest passed; awaiting review (gate: after)
2026-11-14 · A4 revise · hid condition/folder column in metadata editors; pytest 26 passed, apptest passed; awaiting review
2026-11-14 · A4 · approved, marked done (94d0dc4)
2026-11-14 · A5 · committed (3be5b46), pytest 26 passed, apptest passed; awaiting review (gate: after)
2026-11-14 · A5 · approved, marked done (3be5b46)
2026-11-14 · B1 · design note written; awaiting review (gate: before)
