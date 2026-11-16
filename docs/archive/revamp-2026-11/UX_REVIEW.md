# Glow — UI & Workflow Usability Review

_Review date: 2026-11-13 · Scope: the Streamlit UI (`ui/`) as a user experiences it, end to end._

**How this review was done.** I read every UI module, then ran the app from the shared
conda env (`/home/shared/envs/glow_envs`, `streamlit run ui/app.py`) and walked the full
workflow twice on the 12 FCS files in `data/uploaded_raw/My_Experiment_1`: once with
Streamlit's `AppTest` harness (records every widget, message and session key at each step)
and once in a real Chromium browser (Playwright) taking full-page screenshots and timing every
click. Both runs completed without exceptions: load → MLP live gate → MLP singlets → finalize →
analysis gate → histogram / scatter / bar / pairwise plots → canvas. Every click responded in
1.8–4.9 s. **No code was modified.** This document complements `IMPROVEMENTS.md` (architecture,
robustness); it focuses on what the user sees and does.

Severity: 🔴 blocks or misleads users · 🟡 friction / confusion · 🟢 polish.

---

## 0. Executive summary

The app works, but it reads like a long, linear log of every step rather than a tool that
guides you. The six issues that matter most:

1. 🔴 **Everything is one scrolling page.** After loading, the user has to scroll past the
   full file-loading form, the metadata editor *and* the "Loaded Metadata Example" table
   (~2,000 px) to find the gating plot, and the plot stays below the fold on every rerun.
2. 🔴 **Metadata has no defaults and can't be fixed later.** The editor shows blank
   `Experiment Name` / `Replicate` columns; Confirm with blanks fills `NA`, so every group is
   called `NA`, statistics are disabled, and sample names become
   `HEK_Xiaowei-20260808 B2M stain DKO control cell lines-20260808-1719_NA_NA_Xiaowei-…_D1.fcs`.
   Fixing it means going back to Step 1 and reloading, which discards the gates.
3. 🔴 **Units are inconsistent across tabs.** Analysis says "scaled by 1/10000" but shows raw
   values (threshold default 55,053,998; axis 100k–100M); Plot actually scales (axis 10⁰–10⁵);
   Canvas shows raw. The same channel has three different axis ranges in three tabs.
4. 🟡 **Gating controls are split and disordered.** Plot-affecting controls (Scale, Interaction,
   Colormap) sit *below* the plot in the left column; the "Visualization" header, reset buttons,
   and success banners pile up underneath; step numbering (2 → 3 → 4 → 5 → 6) jumps between
   columns.
5. 🟡 **Plot setup is heavy before you see anything.** "Gating (Optional)" with an expanded
   "No local gates defined" box occupies the top of every plot tab; settings are split between
   the sidebar (truncated to "1D Hi…" by the 175 px sidebar), a left column, and a right
   "Configuration" column; there is no live preview — you must press Generate each time.
6. 🟡 **No concept of "where am I / what's next".** The sidebar lists four tabs that are all
   always clickable, with no completion state, and the bottom "Finalize" / "Finish" buttons are
   the only way to progress.

The recommended shape of the fix is a **stepper-based Gating tab** (Load → Metadata → Live/Dead →
Singlets → Export, one step visible at a time, with a summary strip of completed steps), a
**single plot-settings panel with live preview**, **metadata with sensible defaults that stays
editable on every tab without re-gating** (§2b), and **one unit convention** chosen at load
time. Details and a phased plan follow.

---

## Implementation status (branch `revamp`, Nov 2026)

Item ids and full notes are in `PROGRESS.md`. "Partial" means the listed part is done and
the rest is still open.

| § / finding | What was done | Commit(s) | Status |
|---|---|---|---|
| §1 · 1.1, 1.2, 1.3, 1.5 | Sidebar stepper with ✓ / greyed steps, default sidebar width, one-line header after landing, Developer Mode moved to "Advanced" (D1) | 807260f, 123378c, ab6a024 | Done |
| §1 · 1.4 | "Start new experiment" at the bottom of the sidebar behind a confirmation popover (D2) | 807260f | Done |
| §2 · 2.2, 2.3, B4 | Metadata defaults: Experiment Name ← file stem, Replicate inferred, Cell Line guessed or blank (never `HEK`), auto values marked, fill down (A1, A2) | 427f508, 0cd4692 | Done |
| §2 · 2.4, 5.1 | Stable `Sample` = file stem; Experiment Name (+ rep N) shown in selectors, stats, legends; no `_`-joined names (A3) | 11b06f6 | Done |
| §2 · 2.5, 2.7 | Duplicate "Loaded Metadata Example" removed; bulk upload collapsed (C4) | 21e5d6e | Partial: no step-1 chip after load; upload not moved below the editor |
| §2 · 2.1, 2.6, 2.8, 2.9 | — | — | Open |
| §2b | `sample_metadata` as the source of truth, shared "Edit metadata" expander on every tab, relabel without re-gating, cell-line rename re-keys gates (A4); `base_experiment` rebuilt only when the file list changes (A5) | 94d0dc4, 3be5b46 | Done |
| §3 · 3.3, 3.4, B5, B6, B7 | Stale "Configuring…" / "Applying N filters…" texts, duplicate headers and the pandas `observed` warning removed (C1) | 58a9884 | Done |
| §3 · 3.1, 3.6 | Compact toolbar above the live/dead plot; model context read-only, derived from Cell Line (C2) | 567f222 | Partial: button row / result line not moved |
| §3 · 3.9 | Live/dead auto-gate cached by data hash + parameters (C7) | 67de006 | Partial: singlet prediction not cached |
| §3 · 3.10 | Extra "Reset ALL" buttons removed; per-step buttons are "Redo" (C5) | b135dc8 | Partial: no chip-based Redo |
| §3 · 3.2, 3.5, 3.7, 3.8 | — | — | Open (3.7 partly via C2) |
| §4 · 4.1 | Singlet step uses the same toolbar layout as live/dead (C3) | 8938d93 | Partial |
| §4 · 4.3, 4.4 | "Channels to analyze" multiselect, structural channels as a caption, "Gating complete" header (C6) | 635d5d9 | Partial: export/compensation not merged into one completion card |
| §4 · 4.5, 4.6 | — | — | Open (sticky Next not done) |
| §5 · 5.2, B2 | One unit convention: high-res data scaled once at load, exports raw (B1); per-tab scaling branches removed, one caption "Values shown ÷10,000; exports are raw" (B2) | 14e3034, 5626ee8 | Done |
| §5 · 5.1 | Summary table: Experiment Name, Replicate, conditions, then metrics; Sample hidden; CSV download; group-by-experiment mean ± SD (F1) | 02336e5 | Done |
| §5 · 5.3, 5.4 | Gate creator as one card; percentile quick-picks from a control sample; live % in gate per overlaid sample (F2) | b5cb584 | Partial: no draggable threshold line (Streamlit returns no relayout events); click-to-set kept |
| §5 · 5.5, 5.6 | — | — | Open |
| §6 · 6.1, 6.2, 6.4, 6.5, B3 | Plot type as a radio at the top; "Filter events" expander with active chip; "Grouping by" box removed; scatter Y ≠ X; actionable `NA` notice (E1) | 1f8a6b5 | Partial: scale options / download sets not standardised (6.7, 6.9) |
| §6 · 6.3 | Live preview for histogram / bar / scatter on a ≤50k-event subsample, "Render full" for canvas/export (E2) | 40b87a7 | Done |
| §7 · 7.1, 7.4 | Two-pane canvas (thumbnail list + selected plot with controls beside it), ↑/↓ text buttons, confirmed delete, "Export all (ZIP)" with PNG/SVG/PDF per plot + one multi-page PDF (G1) | bb3963e | Done |
| §7 · 7.2 | Customize controls now sit beside the plot, always visible (G1) | bb3963e | Partial: font list unchanged |
| §7 · 7.3 | Covered by the single unit convention (B1/B2) | 14e3034, 5626ee8 | Done |
| §8 | Emoji prefixes dropped from buttons; Accept/Apply primary; "Adjust gate manually"; toast for action confirmations, success only for lasting results; always-on infos turned into captions; `st.progress` while preparing/loading files (G2) | d66bd3c | Partial: during the single bulk FCS read, progress shows only start and end |

---

## 1. Navigation & global layout

| # | Finding | Where | Sev |
|---|---------|-------|-----|
| 1.1 | Sidebar is hard-fixed to 175 px; the "Select Plot Type" dropdown truncates to `1D Hi…` / `Bar P…`. Navigation lives in the sidebar but the *workflow* progress does not. | `ui/app.py:27-37`, screenshot of Plot tab | 🟡 |
| 1.2 | Large logo + "Flow Analysis Pipeline" banner (≈240 px) repeats on every tab and pushes content down. | `ui/app.py:56-60` | 🟢 |
| 1.3 | Tabs have no state indicator: Analysis/Plot are always listed even before any data is loaded, and show "No experiment loaded" warnings when clicked. Nothing says "Gating ✓ · Analysis · Plot". | `ui/app.py:82-97` | 🟡 |
| 1.4 | "🔄 Start New Experiment (Reset everything)" is the **first** button on the page, above the title, with no confirmation. One mis-click wipes the whole session. | `ui/gating/view.py:18-20` | 🔴 |
| 1.5 | "Developer Mode" toggle is in the user-facing header. It only reveals validation plots (`ui_gating.py:978,1045,1068,2211`). | `ui/gating/view.py:27` | 🟢 |
| 1.6 | Session save/restore is hidden in a collapsed sidebar expander; nothing on the page says sessions exist. | `ui/app.py:111` | 🟢 |

**Proposal**
- Replace the radio with a **progress stepper** in the sidebar (or a top `st.segmented_control`)
  showing `Load ✓ → Gate ✓ → Analyze → Plot → Canvas`, with the current step highlighted and
  unavailable steps greyed with a tooltip ("Finish gating first").
- Move "Start New Experiment" to the bottom of the sidebar behind a confirmation
  (`st.popover` / two-click confirm) and let the sidebar width be the Streamlit default.
- Collapse the banner to a one-line header (small icon + name) after the landing page.
- Gate Developer Mode behind a query param or settings expander.

---

## 2. Step 1 — File loading & metadata

### Observed flow (directory path)
`Enter path → Scan Directory → Generate Preview from Directory → (edit table) → Confirm and Load`.
Four clicks and two buttons that only exist to reveal the next button.

| # | Finding | Where | Sev |
|---|---------|-------|-----|
| 2.1 | **Three separate buttons for one action.** "Scan Directory" and "Generate Preview" could be a single step that runs on path entry; the user sees "Found 12 files" and then has to click again to see them. | `ui_file_loader.py:288-308` | 🟡 |
| 2.2 | **Required metadata is not enforced or even hinted.** `Experiment Name` and `Replicate` are blank, the README says they are "important", but the UI never says so. Confirming with blanks silently fills `NA` (`loader.py:341`), the Replicate auto-formatter skips rows (`ui_file_loader.py:29`), and later the bar plot says "No Replicate column… significance can't be computed". | `ui_file_loader.py:400-414`, `loader.py:319-341` | 🔴 |
| 2.3 | **Cell line is guessed as `HEK` for everything** unless a known name is in the path (`detect_cell_line_from_path` returns `"HEK"` as the fallback). The test data were DKO control lines, not HEK, yet "Detected cell line in 'Cell_Line': HEK" is shown as fact and feeds the MLP model context. | `ui_file_loader.py:48-61` | 🔴 |
| 2.4 | **Sample names are unreadable.** Directory mode prefixes the folder name onto every file (`folder_name` strategy) and the summary-stats table then joins *every* condition column with `_` (`stats.py:68`), producing 120-character names that are truncated everywhere (stats table, legend, overlay multiselect). | `loader.py:206-208`, `stats.py:68` | 🔴 |
| 2.5 | "Bulk Metadata Upload" expander is `expanded=True` and sits *above* the editor, so the most common path (manual edit) is pushed down by a feature most users won't use. | `ui_metadata.py:15` | 🟡 |
| 2.6 | The editor is inside a form with three submit buttons ("Add New Column", "Update View / Save Progress", "Confirm and Load") and a note "Click 'Update View' or 'Load Files' to apply changes". Users don't know when edits are saved. The column multiselect only takes effect on submit. | `ui_file_loader.py:381-414` | 🟡 |
| 2.7 | After loading, the full editor **stays open** and a second, read-only "Loaded Metadata Example" table is appended below it — the two tables look identical and double the scroll distance. | `ui_file_loader.py:450-452` | 🟡 |
| 2.8 | Drag-and-drop upload: the batch name field defaults to `My_Experiment_1` and is reused on every upload, so repeat uploads collide in the same folder. Subfolder structure is now preserved (fixed today) but the user is not told where files went. | `ui_file_loader.py:473` | 🟢 |
| 2.9 | "Enter Directory Path" is the default input method, but desktop (packaged) users will mostly upload; the path field is an empty box with no example and no file picker. | `ui_file_loader.py:211,286` | 🟢 |

**Proposal**
- Merge scan + preview into one action ("Load folder") that immediately shows the editor.
- Make the editor the hero of the step: show it full-width, with **sensible defaults instead
  of required fields**, so Confirm always works and the user only edits what they want:
  - `Experiment Name` ← the FCS **file stem** (`D1`, `E2`, …) when blank.
  - `Replicate` ← **inferred from Experiment Name**: rows that share an Experiment Name are
    numbered `1, 2, 3…` in file order; a unique name gets `1`. Re-inferred whenever the user
    edits Experiment Name (so typing `DKO` on three rows turns them into DKO rep 1/2/3). Keep a
    plain integer, not `DKO_1`, so it's usable as a numeric replicate column in stats. The
    current `auto_format_replicates` (`ui_file_loader.py:8-45`) already loops this way; it only
    needs the default-name step and integer output.
  - `Cell Line` ← best guess from the path when a known name is present, otherwise **blank**
    (never a hard-coded `HEK`). Blank cell line means "one population" and the MLP uses its
    default context; the "Detected cell line" banner only appears for genuine matches.
  - Defaults are visibly marked (e.g. lighter text or an "auto" caption) so users know which
    values they haven't touched.
- Keep "fill down / apply to all" helpers (most experiments have 1–3 distinct values across
  12–96 wells) and a `SelectboxColumn` for Cell Line with the model's known types + free text.
- **Separate the stable sample ID from the display label.** Keep `Sample` = file stem as the
  internal key (used by gates, overlays, stats joins), and show **Experiment Name** (plus
  `rep N` when a name has >1 replicate) everywhere the user sees a sample: gating
  "Representative File" selector, singlet selector, Analysis stats table, overlay multiselect,
  plot legends. Stop joining all conditions into the sample name in `stats.py:68`; show the
  condition columns as their own table columns.
- Move bulk metadata upload into a collapsed "Import metadata from spreadsheet" expander below
  the editor, and the "Loaded Metadata Example" into a collapsed "Preview" after load; once
  loaded, collapse the whole loading step into a one-line summary chip ("12 files · HEK · 3
  conditions · [Edit]").
- Keep edits live (no form) with a single "Confirm and Load" button; Streamlit's `data_editor`
  already persists edits in `session_state`.

### 2b. Metadata editable at any stage (without re-gating)

**Goal.** A user gates with the default names, then at the Analysis or Plot step renames
experiments, fixes replicates, or adds a `Treatment` column — and the gates, statistics and
plots simply relabel, with no re-run of Live/Dead or Singlet detection.

**Why it's feasible.** Gates never depend on the label columns. Live/Dead and singlet gates
are stored as polygons / model parameters keyed by *cell line* (`cell_line_gates`,
`cell_line_singlet_gates`, `live_gate_params`), and the gated/analysis experiments carry the
decisions as boolean columns (`cells`, `singlets`). Positive gates in Analysis reference
channels and thresholds, plus a `samples` list for display (`ui/analysis/view.py:975`). So
everything except `Cell_Line` is a **row-relabel**, not a recompute.

**Backend design**
1. **Single source of truth:** `st.session_state.sample_metadata` — a DataFrame with one row
   per file, indexed by the stable `Sample` id (file stem), columns = `Experiment_Name`,
   `Replicate`, `Cell_Line`, `Condition` (folder) and any user columns. The loader builds it
   at load time; it is JSON-serialisable so the existing session save/restore
   (`ui/common/session.py`) covers it for free.
2. **`apply_sample_metadata(experiment, meta_df)`** in `glow/data_loading/` (pure, no
   Streamlit): for every experiment object in session (`base_experiment`,
   `gated_experiment`, `analysis_experiment`), `merge`/`map` the metadata columns onto the
   event table by `Sample`, add any new columns to `experiment.metadata` as
   `{'type': 'condition', 'class': 'category'}`, and drop removed ones. This is a vectorised
   `Series.map` per column over ~600k rows — well under a second — versus 3–5 s for a
   re-gate. Cytoflow `Experiment` objects accept new condition columns this way (the
   per-cell-line subset code at `ui_gating.py:502-513` already rebuilds metadata like this).
3. **One editor component** `render_metadata_editor(context)` used in three places:
   - Step 1 (full editor, before load);
   - a collapsed **"Edit metadata"** expander on the Gating, Analysis and Plot tabs showing the
     same `data_editor`. On save it updates `sample_metadata`, calls
     `apply_sample_metadata` on every loaded experiment, re-infers Replicate when Experiment
     Name changed, clears cached per-group computations (`st.cache_data` entries keyed on
     metadata hash), and `st.rerun()`s — no gating state is touched.
4. **Cell_Line is the one special column.** Gates are keyed by cell-line value, so:
   - *Rename* a cell line (A→B on all its rows): re-key `cell_line_gates[A]` → `[B]`
     (and singlet gates); no recompute.
   - *Reassign* files between lines or introduce a new line: the gate for the affected line
     no longer covers those files. Mark only those lines as "needs re-gating" and show it in
     the gating step summary chip ("Live/Dead ✓ HEK · ⚠ DKO needs gating"); unaffected lines
     keep their gates. Finalize re-applies stored polygons to the new row subsets, which is
     what it already does (`ui_gating.py:3060-3120`).
5. **Replace `base_experiment`-is-write-once** (`ui_gating.py:397`) with a metadata-hash
   check: the experiment is rebuilt only when the *file list* changes; label changes go
   through `apply_sample_metadata`.
6. Positive gates keep working because they reference `Sample` ids; only their display labels
   come from `sample_metadata` at render time.

**UI consequence.** The "Edit metadata" expander appears on every tab after load; Step 1's
big editor collapses to a chip after load ("12 files · 4 experiments · 3 replicates each ·
[Edit metadata]").

---

## 3. Step 2 — Live/Dead gating

### Observed layout (full-page screenshot)
Left column: "Live/Dead Gating Parameters" expander (channels, cell-line info, method, model
context, boundary slider, outlier checkbox) → *then* "Visualization" header with Scale /
Interaction / Colormap → "Gated Lines" + two reset buttons → two green banners.
Right column: "3. Live Gating Verification" header → info box → representative file selector →
"Gating Classification" header → "Gating Result" plot → Accept / Reject / Add-to-Canvas row.

| # | Finding | Where | Sev |
|---|---------|-------|-----|
| 3.1 | **Display controls (Scale, Interaction, Colormap) are rendered outside the parameters expander and end up ~500 px below the plot they control**, under a separate "Visualization" subheader. | `ui_gating.py:570-576` | 🟡 |
| 3.2 | **Step numbering is confusing**: "2. Live Gating Setup" (left/top), "3. Live Gating Verification" (right column), then "4. Singlet Detection", "5. Final Data Settings", "6. Export Data (Optional)" (an expander). Setup and verification are not separate steps; they are the same screen. | `ui_gating.py:200,643,1625,2614,2703` | 🟡 |
| 3.3 | Stale text "Configuring Gating..." stays on screen permanently under the step header. | `ui_gating.py:394` | 🟢 |
| 3.4 | Three stacked headers before the plot ("3. Live Gating Verification", "Gating Classification", plot title "Gating Result") say the same thing. | `ui_gating.py:643-700` | 🟢 |
| 3.5 | The plot shows only **5,000 of 50,000 events of one representative file** and the gate is drawn from that; the user cannot see how the gate fits the other 11 files without switching the selector and re-running. There is no small-multiple / overlay check before accepting. | `ui_gating.py:658-659` | 🟡 |
| 3.6 | "Cell Line (Model Context)" is a second cell-line selector right under "Select Cell Line to Gate", with the model's internal vocabulary (`colo/hek/hug1n/snu/b16/cho`). Users don't know which one matters. | `ui_gating.py:493,547` | 🟡 |
| 3.7 | Accept/Reject buttons sit below the plot; after accepting, the button label becomes "Gate Applied" (disabled) and *two* green banners ("Gate Applied." and "Final Live Cells: …") appear in the **left** column, far from the button that caused them, plus "Gated Lines: HEK" and two reset buttons. The result number (87.6 %) should be next to the plot. | `ui_gating.py:1147-1302,1432-1436` | 🟡 |
| 3.8 | **Manual gating is click-5-points + a raw text area of vertices.** Pan/Box-Select is a radio in the left column; the hint says "Hold Shift to Multi-Select". There is no lasso/polygon draw tool, no visual feedback of the polygon until ≥3 points, and the "Apply" button is above the text area while "Undo"/"Exit" are below. | `ui_gating.py:1440-1470` | 🟡 |
| 3.9 | Changing *any* parameter triggers a full re-gate (`should_run_gating`) with a spinner, including purely cosmetic ones like colormap — because `cmap`/`scale` are not in `current_gating_params` they don't re-gate, but channel/method/slider changes re-gate the full 600k rows before the plot updates. No debounce; sliders feel laggy. | `ui_gating.py:578-603` | 🟡 |
| 3.10 | The "🔓 Not locked — redo gating or edit metadata" expander is a good idea but is phrased as a negative ("Not locked") and hides the only "Redo" button; meanwhile *four* other reset buttons exist further down (Reset Gate for HEK, Reset ALL Cell Line Gates, Reset Singlet Gate for HEK, Reset ALL Singlet Gates). | `ui_gating.py:425-439,1276-1302,2280-2290` | 🟡 |
| 3.11 | Live gating plot is log-scaled on raw high-res values (axis 1M–1B) while the singlet plot right below is linear (15M–35M) and the Plot tab is scaled (10⁰–10⁵). | `ui_gating.py:572,1755` | 🟡 (see §6) |

**Proposal**
- Reorganise the gating step as **one card per step with a fixed 1 : 3 layout**:
  - Left: *Parameters* (channels, method, method-specific params) — nothing else.
  - Right: plot with a **compact toolbar directly above it** (scale · colormap · representative
    file · "show all files" toggle) and **Accept / Adjust / Manual** buttons directly below it,
    followed by a single result line ("Live: 525,799 / 600,000 (87.6 %)").
  - Reset/redo lives in one place: an "Edit" link on the completed-step summary chip.
- Collapse a step after acceptance into a summary row ("Live/Dead ✓ MLP · 87.6 % · [Edit]") and
  auto-scroll/expand the next step. This alone removes most of the scrolling.
- Replace numbered subheaders with the stepper; drop the duplicated headers and the
  "Configuring Gating..." placeholder.
- Add a **"Check all files" mini-grid** (12 small density plots with the proposed gate
  overlaid, e.g. 4×3, subsampled) before Accept, so the user can see outliers without
  switching the selector.
- Merge the two cell-line selectors: derive the model context from the metadata Cell Line
  (already done at `ui_gating.py:538-546`) and show it as read-only text with an "override"
  link only when the metadata value isn't in the model vocabulary.
- Manual gate: use Plotly's built-in `drawclosedpath` / lasso (`config={'modeBarButtonsToAdd':
  ['drawclosedpath']}` with `on_select`) so the user draws the polygon in one gesture; keep the
  vertex text area under an "Advanced" expander.
- Debounce parameter changes: compute on an explicit "Update gate" button *or* only when the
  slider is released (`st.slider` already fires on release — the cost is the 600k-row re-gate; cache
  `auto_gate_live_cells` per parameter hash with `st.cache_data`).

---

## 4. Step 3 — Singlet gating, final settings, export

| # | Finding | Where | Sev |
|---|---------|-------|-----|
| 4.1 | Singlet step repeats the same pattern as live gating but with a *different* layout: its own "Settings" header, cell-line selector rendered full-width above the columns, "Singlet Settings" subheader placed after the representative-file selector. Inconsistent with step 2. | `ui_gating.py:1648,1749-1764,2089` | 🟡 |
| 4.2 | After accepting singlets the plot recolors to density and the legend changes from Singlet/Doublet to "Singlet Gate", so the user loses the classification view they just approved. | `ui_gating.py:2117-2127` | 🟢 |
| 4.3 | "5. Final Data Settings" is a channel checkbox grid (16 checkboxes in 3 columns) plus Select All / Clear All and a "Structural channels included automatically" banner. It's unclear *why* the user is choosing channels here (it filters Plot/Analysis channel lists) — the label says "Unselected channels will be hidden" but hidden where? | `ui_gating.py:2614-2700` | 🟡 |
| 4.4 | The default selection drops `-H`/`-W` fluorescence channels and all of `Event Info`, `TMSW`, `TLSW`, `TIME` — sensible, but the message "8 channels selected. 8 channels dropped." sounds alarming. | `ui_gating.py:2697` | 🟢 |
| 4.5 | Export ("6. Export Data (Optional)") is an expander *between* channel selection and the Finalize button; the "Run Compensation before Analysis" checkbox is a lone checkbox directly above the primary button. Compensation feels like an afterthought. | `ui_gating.py:2703,2996` | 🟡 |
| 4.6 | "✅ Finalize Gating & Proceed to Analysis" is the last widget on a ~6,000 px page. Nothing at the top says this button exists. | `ui_gating.py:2998` | 🟡 |

**Proposal**
- Use the exact same step card for Singlets as for Live/Dead (same column widths, same
  toolbar, same Accept/Adjust/Manual row, same summary chip).
- Rename "Final Data Settings" → **"Channels to analyze"**, show it as a compact multiselect
  of fluorescence channels (structural ones listed as a caption), and explain in one line that it
  controls which channels appear in Analysis and Plot.
- Make Export and Compensation **optional actions on the completion card** ("Gating complete —
  Next: Analysis ▶ · Export CSV · Run compensation first"), not inline sections.
- Add a sticky "Next ▶" affordance in the stepper so the user never has to find the bottom of
  the page.

---

## 5. Analysis tab

| # | Finding | Where | Sev |
|---|---------|-------|-----|
| 5.1 | Summary-statistics table: `Sample` column is a 120-char concatenation (see 2.4); the useful columns (Live %, Singlets %) are visible, but conditions aren't separate columns so the table can't be sorted by condition. | `stats.py:40-70` | 🔴 |
| 5.2 | **Scaling claim is wrong.** The tab says "High-resolution data detected. Data logic is scaled by default (1/10000)" with a "Show Raw Values" toggle off, yet the density plot axis reads 100k–100M and the default threshold is `55053998.08`. `get_channel_data_for_samples` assumes `is_high_res ⇒ already scaled` (`positive.py:391-392`), but the loader only *tags* high-res; nothing in the gating path divides the data (only `ui/compensation/view.py:128` calls `detect_and_scale_high_res`). So Analysis shows raw values while *calling* them scaled, and gates are stored in raw units. The Plot tab then divides by 10,000 (`ui/plot/view.py:30-70`) and re-converts imported gates (`gating_component.py:199-243`). It works by coincidence of two compensating assumptions and is impossible for a user to reason about. | `positive.py:384-396`, `ui/analysis/view.py:40-45` | 🔴 |
| 5.3 | Gate creator: left column stacks five headers ("New Gate Definition", "Analysis Settings", "Hist Parameters", "Axis & Overlay", "Threshold Selection") for ~8 widgets. "Select Samples to Overlay" defaults to one sample. | `ui/analysis/view.py:325-620` | 🟡 |
| 5.4 | Threshold is set by "Select a point on the plot" (a Plotly click) *or* a number input showing 8 significant digits. No draggable line, no "set at X % of negative control" helper, no percentile quick-picks. | `ui/analysis/view.py:532-617` | 🟡 |
| 5.5 | Defined gates list shows "Visualization not available for 2D gates in this summary view." | `ui/analysis/view.py:165,171` | 🟢 |
| 5.6 | "Finish Analysis & Proceed to Plotting" is required to reach plots? No — Plot tab works without it (falls back to `base_experiment`), but the button suggests it's mandatory. | `ui/analysis/view.py:1650` | 🟢 |

**Proposal**
- Pick **one unit convention at load time**: either scale the data once in the loader (set
  `is_scaled=True`, keep `scaling_factor` in metadata for export/raw toggles) or never scale and
  only change axis *formatting*. Then delete the per-tab scaling branches. Show the convention
  once ("Values shown ÷10,000; raw in exports") rather than a yellow warning on three tabs.
- Statistics table: first column = **Experiment Name** (with `Replicate` as the next column
  when >1 per name), then the other condition columns, then metrics; `Sample` (file stem)
  available but hidden by default. Same labels in the "Select Samples to Overlay" multiselect
  and plot legends. Add a download button and a "group by experiment" toggle (mean ± SD per
  Experiment Name across replicates).
- Gate creator: single "Gate" card — channel + parent on one row, plot with a **draggable
  threshold line** (Plotly `shapes` with `editable=True` → `on_select`/relayout) and quick-picks
  ("99th percentile of <control sample>"), then Save. Show the resulting % positive per sample
  live next to the plot before saving.

---

## 6. Plot tab

### Observed layout (histogram)
Full-width "1. Data Selection" (filter category + multiselect) → yellow scaling warning + raw toggle
→ "Gating (Optional)" with three expanders, the middle one *expanded* just to say "No local gates
defined for this plot." → two-column settings ("Plot Settings" left, "Configuration" right, plot
type selector in the sidebar) → red "Generate Plot" → "Applying 2 filters..." → plot → stats table
→ Download SVG / Add to Canvas.

| # | Finding | Where | Sev |
|---|---------|-------|-----|
| 6.1 | **Gating block dominates the top of every plot type**, expanded by default, usually empty. The bar plot shows it even though it's conceptually a filter. | `gating_component.py:9-104` | 🟡 |
| 6.2 | Plot settings are split three ways: plot *type* in the sidebar (truncated), channel/scale/style in a left column, grouping/legend in a right column titled "Configuration" (histogram) or "Grouping Configuration" (bar). The same concept ("group by") appears as an info box "Grouping by: Experiment_Name" *and* a selectbox "Color/Group By". | `histogram_ui.py:29-120`, `bar_ui.py:36-50` | 🟡 |
| 6.3 | **No live preview.** Every change requires "Generate Plot"; the previous plot stays visible meanwhile, so it's easy to look at a stale plot. The transient "Applying 2 filters..." text remains after render. | `histogram_ui.py:183-212`, `bar_ui.py:209-236` | 🟡 |
| 6.4 | With unfilled metadata the only group is `NA`, so the bar plot is a single violin labelled "NA", comparisons can't be added, and "Significance Correction" shows a disabled select with "Correction disabled (Define ≥2 comparisons)". The user has no idea the fix is back in Step 1. | screenshot; `bar_ui.py:84-120` | 🔴 (root cause 2.2) |
| 6.5 | 2D scatter defaults X and Y to the **same channel** (`EGFP-A` vs `EGFP-A`) when no mCherry channel exists. | `scatter_ui.py:30-42` | 🟡 |
| 6.6 | Channel list is filtered by name patterns (`fsc, fc, ssc, time, replicate, experiment, sample, well, plate`) — the `fc` pattern would hide any channel containing "fc" (e.g. a hypothetical `FcR-A`). | `ui/plot/view.py:119-120` | 🟢 |
| 6.7 | Scale selector offers `log / linear / logicle` here, `Linear / Log` in gating, auto-detected in Analysis. | various | 🟢 |
| 6.8 | Pairwise plot samples 2,000 events by default with no indication on the plot. | `pairwise_ui.py:49` | 🟢 |
| 6.9 | Download buttons appear in inconsistent sets (SVG only here; SVG/PNG/PDF on Canvas). | `histogram_ui.py:427-462`, `canvas/view.py:834-852` | 🟢 |

**Proposal**
- **One settings panel, one plot.** Put plot type at the top of the left panel (not the sidebar),
  then *Data* (channel(s), scale), *Grouping* (group by, facet by, order), *Style*, each as a
  small expander. Right side: the plot, rendered **live** (histogram/bar on ≤600k events already
  render in <3 s; subsample for preview and render full on "Finalize/Add to canvas").
- Move the gating block into a collapsed **"Filter events"** expander inside the left panel,
  with a chip summary when active ("Live ✓ Singlets ✓ EGFP+").
- When the only group is `NA`, show an actionable notice: "No Experiment Name was entered in
  Step 1 — [Go to metadata]".
- Default scatter Y to the first channel ≠ X.
- Standardise scale options and download sets across all plot types.

---

## 7. Canvas tab

| # | Finding | Where | Sev |
|---|---------|-------|-----|
| 7.1 | Move-up / move-down buttons render as tiny glyph boxes (the `⬆`/`⬇` emoji don't render in the headless font; on user machines they may). Delete is a trash icon with no confirmation. | `canvas/view.py:526-534,734` | 🟢 |
| 7.2 | "Customize" is per plot and collapsed; width/height/font/dot scale are four sliders without a preview of the effect until rerun. Font family list is `default/sans-serif/serif/monospace` only. | `canvas/view.py:534-577` | 🟡 |
| 7.3 | Canvas plots keep raw units (axis 10⁶–10⁹) even though the Plot tab is scaled. | screenshot | 🟡 (see 5.2) |
| 7.4 | No "export all" / multi-panel figure assembly; each plot has its own three download buttons. | `canvas/view.py:834-852` | 🟡 |

**Proposal**
- Two-pane canvas: thumbnail list (reorder by drag or ↑/↓ text buttons) on the left, selected
  plot large on the right with its customize controls beside it, and a single "Export all
  (ZIP / multi-page PDF)".

---

## 8. Wording, messaging & feedback

- Emoji-prefixed buttons (✅ ❌ 🔄 🗑️ ↩️ 💾) are used inconsistently; some primary actions are
  plain, some are red `type="primary"`. Pick one system (primary = next step, secondary =
  everything else, destructive = confirm).
- Several info boxes repeat the obvious ("Processing singlets for detected cell line.",
  "Multiple files for HEK. Select representative:") and stay forever. Prefer captions.
- The "Reject / Manual Gate" wording implies rejecting *the method*; "Adjust gate" is closer to
  what users do.
- Loading has no progress for 12 files (spinner only); for 96-well plates the user will stare at
  "Loading Experiment Data..." for a minute. Use `st.progress` per file.
- Success messages go to `st.success` (sticky) in some places and `st.toast` in others.

---

## 9. Bugs found while testing (worth fixing regardless of the redesign)

| # | Bug | Where |
|---|-----|-------|
| B1 | Folder drag-and-drop upload failed with `FileNotFoundError` because sub-folder paths in `uploaded_file.name` were not created — **fixed today** in `loader.py:73-85` (with a path-traversal guard). | `glow/data_loading/loader.py` |
| B2 | Analysis tab reports values as scaled while they are raw (see 5.2). Gates created there are stored in raw units; the Plot tab compensates. Any new consumer of `positive_gates` will get the units wrong. | `glow/analysis/positive.py:389-396` |
| B3 | 2D scatter defaults X == Y. | `ui/plot/scatter_ui.py:30-42` |
| B4 | `detect_cell_line_from_path` returns `"HEK"` for any unrecognised path and that value is shown as "Detected". | `ui/gating/ui_file_loader.py:48-61` |
| B5 | "Configuring Gating..." placeholder text is never cleared. | `ui/gating/ui_gating.py:394` |
| B6 | "Applying N filters..." status text persists after the plot renders. | `histogram_ui.py:212`, `scatter_ui.py:191`, `bar_ui.py:236`, `pairwise_ui.py:99` |
| B7 | pandas `FutureWarning` (`observed=False`) on every bar plot groupby. | `ui/plot/bar_ui.py:305` |

---

## 10. Proposed phased plan

**Phase 1 — quick wins (days, no structural change)**
1. Metadata defaults: Experiment Name ← file stem, Replicate ← inferred integer from
   Experiment Name, Cell Line ← guess-or-blank (never `HEK`, B4); mark auto-filled values;
   add "fill down".
2. Sample naming: stable `Sample` = file stem; show **Experiment Name (+ rep N)** in the gating
   file selectors, Analysis stats table, overlay selectors and legends; conditions as separate
   columns in stats (2.4, 5.1).
3. Collapse the loading step and the metadata example after load; collapse "Bulk Metadata
   Upload" by default (2.5, 2.7).
4. Move gating display controls above the plot; put result line + Accept/Adjust directly
   under the plot; remove duplicate headers and stale texts (3.1, 3.3, 3.4, B5, B6).
5. Plot tab: collapse "Gating (Optional)"; remove the redundant "Grouping by" info; fix
   scatter default Y (6.1, 6.2, B3); move plot type out of the sidebar.
6. Move "Start New Experiment" behind a confirmation (1.4).

**Phase 2 — units & state (1–2 weeks)**
7. Decide and implement one scaling convention at load; remove per-tab scaling code; fix B2.
8. Introduce an explicit workflow state (`current_step`, per-step `accepted` record) that the
   sidebar stepper reads; grey out unavailable tabs.
8b. **Metadata editable anywhere** (§2b): `sample_metadata` as the single source of truth,
   `apply_sample_metadata()` relabel function, shared editor expander on Gating/Analysis/Plot,
   cell-line rename re-keys gates; only genuinely re-assigned cell lines are flagged for
   re-gating.
9. Cache `auto_gate_live_cells` / singlet prediction by parameter hash so slider changes are
   cheap.

**Phase 3 — stepper redesign (2–4 weeks)**
10. Split `render_gating_section` (3,229 lines) into `step_load.py`, `step_metadata.py`,
    `step_live.py`, `step_singlets.py`, `step_finish.py`, each rendering one card with the
    shared layout; completed steps collapse to summary chips with Edit.
11. "Check all files" mini-grid before Accept; lasso/polygon draw for manual gates.
12. Live-preview plot panel with a single settings column; draggable threshold in Analysis.
13. Canvas two-pane layout with export-all.

**Design principles to apply throughout**
- One screen = one decision. Show only the current step's controls; summarise the rest.
- Controls next to what they control; results next to the action that produced them.
- Every metadata field has a usable default, and every field stays editable at every step;
  nothing is discovered as `NA` three tabs later.
- Same layout for things that are the same (Live/Dead vs Singlets; the four plot types).
- One unit convention, stated once.

---

### Appendix — test run evidence

- Harness: `AppTest.from_file("ui/app.py")`, Streamlit 1.52.2, Python 3.12 (shared env).
- Browser: Chromium via Playwright; viewport 1500 × 3200; full-page screenshots per stage.
- Click latencies (browser): Scan 1.8 s · Preview 1.8 s · Confirm & Load 3.0 s · Reject/Manual
  2.9 s · Accept live 4.1 s · Accept singlets 4.9 s · Finalize 4.4 s · Add gate 2.3 s ·
  Generate histogram 1.8 s · Generate bar 2.3 s.
- Results on the 12-file dataset: Live 525,799 / 600,000 (87.6 %), Singlets 517,888 / 600,000
  (86.3 %), analysis population 495,460 events. No exceptions in the server log.
- Screenshots are in the session scratchpad
  (`/tmp/claude-1000/-home-xwz-Projects-test-Glow/4dc9fc2f-82c3-4e16-8535-3efe9935b0df/scratchpad/t_*.png`).
