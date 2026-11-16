# Glow — Codebase Review & Improvement Plan

_Review date: 2026-07-10 · Scope: full repository (`glow/` library + `ui/` Streamlit app, ~16k LOC Python)_

Glow is a well-featured flow-cytometry pipeline with a genuinely rich Streamlit UI (loading, metadata merging, ML-assisted gating, active learning, plotting, canvas). The functionality is impressive. The main risks are **maintainability** (a few very large files/functions), **robustness/error reporting**, **state durability**, and **cross-platform packaging**. This document lists concrete, prioritized improvements, starting with the five known issues you raised and then broadening to the rest of the codebase.

Severity legend: 🔴 high · 🟡 medium · 🟢 low / polish.

---

## Part 1 — The five known issues

### 1. 🔴 Box-plot grouping is hard-locked to Experiment Name

**Where:** `ui/plot/bar_ui.py:51-64`.

The bar/box plot UI *forces* the grouping variable to the first match among
`["Experiment_Name", "Experiment Name", "Sample", "sample"]` and only falls back to a
user-selectable dropdown when none of those columns exist:

```python
for t in target_group_cols:
    if t in conditions:
        found_group_col = t
        break
if found_group_col:
    hue_var = found_group_col
    st.info(f"Grouping by **{hue_var}**")   # user cannot change this
else:
    hue_var = st.selectbox("Group By (X-Axis)", conditions)
```

Because these columns almost always exist, the user can never group by Cell Line, Condition,
Treatment, Dose, etc. In addition, the backend `plot_means_with_significance()` in
`glow/plotting/categorical.py:7` accepts only a single `group_by` string — there is no
secondary grouping / sub-grouping (hue) or faceting.

**Recommendation**
- Always render a `selectbox` (or `multiselect`) over `conditions`; use the detected
  experiment/sample column only as the *default index*, not a lock.
- Add an optional **secondary grouping ("split by")** argument threaded into
  `plot_means_with_significance()` so users can, e.g., group bars by Condition and split by Cell Line.
- Consider a "facet by" option to produce small-multiples across a third variable.
- Apply the same flexibility to the CSV-export grouping logic (`bar_ui.py:299-321`), which
  currently hard-codes `['Sample', 'Experiment_Name', 'Experiment Name', replicate_col]`.

### 2. 🔴 No way to save / restore where the user left off

**Where:** state lives entirely in Streamlit `st.session_state` (in-memory). There is **no
serialization of session state anywhere** in the repo, and two code paths actively *delete*
working data:
- `ui/app.py:63-68` calls `DataLoader.clear_upload_cache()` on every fresh start.
- `ui/gating/ui_gating.py:3166` calls `clear_upload_cache()` again after gating completes.
- Tab position (`st.session_state.current_tab`) is not reflected in the URL, so a browser
  refresh drops the user back to the Gating tab and loses all progress.

The practical effect: a refresh, a crash, or coming back tomorrow means re-doing the whole
analysis.

**Recommendation**
- Introduce a **project / session concept**: serialize the meaningful parts of session state
  (loaded file list, metadata, accepted gates as vertices/parameters, channel selections, tab)
  to a `.glow` file (JSON preferred over pickle for safety — see §Security). Offer
  **"Save Session"** (download) and **"Load Session"** (upload) buttons, plus optional
  autosave to a `sessions/` directory keyed by an experiment name.
- Persist lightweight navigation state in URL query params (`st.query_params`) so refresh
  restores the current tab.
- Save *gate definitions* (polygon vertices, thresholds, method + params), not gated
  DataFrames — they are small, portable, and re-applyable to the raw data.
- Stop unconditionally clearing the upload cache on startup; scope cache cleanup to an
  explicit "Start New Experiment" action (which already exists at `ui/gating/view.py:18`).

### 3. 🔴 Gating steps are locked; can't go back and edit metadata

**Where:** `ui/gating/ui_gating.py`.

- The base experiment is built **once** and cached forever:
  `if 'base_experiment' not in st.session_state:` (`ui_gating.py:395`). If the user edits
  metadata afterwards, nothing rebuilds the experiment, so edits don't propagate downstream.
- Accept buttons get disabled once a gate is accepted:
  `disabled=is_accepted` at `ui_gating.py:1124` and `1376`, gated on
  `st.session_state.gate_accepted`. There's no first-class "unlock / redo this step" path.
- The workflow is effectively one long linear function (`render_gating_section`, ~3000 lines)
  with implicit stage flags (`gate_accepted`, `gating_complete`), so there is no clean way to
  jump back to an earlier stage.

**Recommendation**
- Add an explicit **"Edit metadata & re-apply"** action that invalidates
  `base_experiment` (and dependent gated state) and rebuilds from the current
  `preview_meta`, rather than treating `base_experiment` as write-once.
- Represent the workflow as **navigable steps** (e.g., a stepper: Metadata → Live/Dead →
  Singlets → Export) where each step can be revisited. Re-entering a step should re-open its
  controls and re-run downstream steps that depend on it (invalidation, not deletion).
- Replace scattered boolean flags with a small explicit **state machine** (current step +
  per-step "accepted" record) so "go back" is a single, testable transition.
- Because gates are stored as vertices/parameters, re-applying them to an updated experiment
  is cheap — lean on that.

### 4. 🔴 Error messages are generic and unactionable

**Where:** the codebase has **118 broad `except Exception`** handlers and **4 bare
`except:`**. Many simply echo the raw exception, and some leak debug output:
- `st.error(f"Error: {e}")` in `ui/plot/bar_ui.py:336`, `ui/plot/scatter_ui.py:341`,
  `ui/plot/histogram_ui.py:553`, and others.
- `st.write(e)  # Debug` at `ui/gating/ui_gating.py:3174` — a raw exception shown as debug text.
- `traceback.print_exc()` to the terminal at `ui/gating/ui_metadata.py:249` — invisible in a
  packaged desktop app where there is no console.
- Generic `st.error(f"Error reading/merging file: {e}")` (`ui_metadata.py:246`) with no
  guidance on the common causes (wrong header row, mismatched keys, encoding).

**Recommendation**
- Create a small **error-reporting helper** (e.g., `ui/common/errors.py`) that renders:
  (1) a plain-language summary, (2) the likely cause, (3) a concrete fix, and (4) an
  expandable "technical details" section with the traceback for bug reports.
- Replace the broadest `except Exception` blocks with **specific exceptions** where the
  failure mode is known (missing channel → `KeyError`, bad file → `FileNotFoundError`,
  empty selection, non-positive values for log scale, etc.) and map each to targeted guidance.
  Several of these are already partially handled well (e.g., the log-scale guard at
  `bar_ui.py:238-241`) — extend that pattern.
- Add **file logging** (Python `logging` to a rotating file in the user's app-data dir) so
  packaged Mac/Windows users can attach a log to issue reports; replace `print("DEBUG: …")`
  calls in `glow/data_loading/loader.py` and `glow/plotting/categorical.py:26` with `logging`.
- Validate inputs *before* the risky operation and fail with a friendly message (e.g., "Live/Dead
  gating needs FSC/SSC-like channels; none detected — pick channels manually").

### 5. 🔴 Environment is not portable; packaging for older macOS / Windows is undocumented

**Where:** `environment.yml` and repo packaging.

- `environment.yml` is a **full 408-package conda export with Linux-only build strings**
  (e.g., `alsa-lib=1.2.15.1=hb03c661_0`, `argon2-cffi-bindings=…=py312h4c3975b_2`,
  `python=3.12.12=hd63d673_1_cpython`). These pinned build hashes and Linux-only packages
  (`alsa-lib`, `_libgcc_mutex`, `_openmp_mutex`) will **fail to solve on macOS/Windows**.
- There is **no cross-platform dependency spec** (no `requirements.txt`, no `pyproject.toml` —
  in fact `.gitignore` explicitly excludes `pyproject.toml` and `pytest.ini`), and **no
  packaging config** in the repo for the promised `.exe` / Electron `.app` (no PyInstaller
  spec, no `package.json`, no build scripts).
- The app relies on **CWD-relative paths** that only work when launched from the repo root:
  `st.set_page_config(page_icon="assets/glow_icon.png")` and `st.image("assets/…")`
  (`ui/app.py:6, 48-52`), `os.makedirs("data/uploaded_raw")` (`ui/app.py:67`), default
  `target_dir="data/uploaded_raw"` in `loader.py`. In a bundled app the CWD is unpredictable,
  so these break.

**Recommendation**
- Ship a **hand-authored, cross-platform environment**: a minimal `environment.yml` with only
  direct deps and *no* build strings (let conda solve per-platform), plus a
  `requirements.txt`/`pyproject.toml` for pip users. Keep the current full export as
  `environment.lock.linux.yml` for reproducible Linux dev only.
- Make all resource paths **relative to the module** (`pathlib.Path(__file__).parent`) or use
  `importlib.resources`; write user data to a per-user app-data directory
  (`platformdirs.user_data_dir`) instead of `./data`.
- Add **packaging configuration to the repo** (PyInstaller `.spec` or the Electron build
  files) and a `BUILD.md` documenting how the Mac/Windows bundles are produced.
- Define and document **minimum OS targets** (e.g., macOS 11+, Windows 10+). For older macOS,
  build on the oldest supported SDK / set `MACOSX_DEPLOYMENT_TARGET`; test the produced binary
  on a clean VM. Note that cytoflow + native BLAS/Qt stacks are the usual portability
  pain points — verify these specifically.
- Add a smoke-test CI matrix (Linux/macOS/Windows) that at minimum imports the package and
  launches Streamlit headless.

---

## Part 2 — Architecture & maintainability

- 🔴 **God functions / files.** `ui/gating/ui_gating.py` is **3188 lines**, and
  `render_gating_section` is a single ~3000-line function; `ui/analysis/view.py` is 1653 lines
  and `ui/canvas/view.py` 937 lines. These are very hard to read, test, or modify safely.
  Break the gating flow into per-step render functions (metadata, live/dead, singlet, export),
  each with its own small state, and split analysis/canvas similarly.
- 🟡 **`importlib.reload()` left in production UI code.** Present in
  `ui/plot/bar_ui.py:218-219`, `ui/plot/histogram_ui.py:203-204`, `ui/plot/scatter_ui.py:170-171`,
  `ui/plot/gating_component.py:106`, `ui/gating/ui_active_learning.py:27`. This is a dev
  hot-reload hack: it re-imports modules on every plot generation (wasteful) and can cause
  subtle bugs where reloaded classes fail `isinstance` checks against already-imported ones.
  Remove them.
- 🟡 **Duplicated helpers.** `download_btn()` is copy-pasted in three files
  (`bar_ui.py:12`, `histogram_ui.py:15`, `scatter_ui.py:11`); `get_default_channel` is defined
  twice inside `ui_gating.py` (`:224`, `:421`); `categorical.py:137-145` contains a literally
  duplicated comment/assignment block. Extract shared plotting/UI utilities into one module.
- 🟡 **Fragile import wiring.** `ui/app.py` does `sys.path.append(... '..')` and then imports
  `from gating.view import …` relying on Streamlit injecting the script directory onto
  `sys.path`. `ui/` is not consistently treated as a package. Make `ui` a proper package with
  absolute imports (`from ui.gating.view import …`) and a single, documented launch entry point.
- 🟢 **Dead / stub code.** `GatingManager.filter_by_condition` is a `pass` stub
  (`glow/gating/manager.py:206-210`); `ui_gating.py:3184` is an unreachable second
  `elif not CYTOFLOW_AVAILABLE:` after the same condition at `:3179`; several commented-out
  code blocks remain (e.g., `glow/pipeline/runner.py:57-60`). Remove or implement.
- 🟢 **Naming schema inconsistency.** The code juggles `"Experiment Name"` vs
  `"Experiment_Name"`, `"Cell Line"` vs `"Cell_Line"` throughout (see the unify logic at
  `ui_metadata.py:197-211` and the target lists in `bar_ui.py`). Define one **canonical
  metadata schema** (constants module) and normalize at load time so downstream code matches a
  single name.

---

## Part 3 — Correctness & data integrity

- 🔴 **Missing channels are silently filled with 0.0.** In the channel-union fallback,
  `glow/data_loading/loader.py:366-367` does
  `combined_df[numeric_cols].fillna(0.0)`. Fabricating zeros for channels a tube never measured
  will corrupt gating and statistics for mixed-panel datasets, with no warning to the user.
  Prefer keeping them `NaN`, excluding those tubes from analyses that need the channel, and
  surfacing a clear notice of which files lack which channels.
- 🟡 **Statistics can be silently meaningless.** In `plot_means_with_significance`
  (`glow/plotting/categorical.py`), when there is no replicate column each group collapses to a
  single pseudo-replicate (`linear_reps = [grand_mean]`, `categorical.py:177-180`), so the
  t-test at `:347-352` is skipped and p is forced to `1.0` — the plot shows brackets/fold-change
  with no valid test and no explanation. Also `fc = m2/m1 if m1 != 0 else 0` (`:342`) reports a
  fold change of 0 on a zero denominator. Recommend: require/validate a replicate column for
  significance testing, display the actual **n** per group, and offer non-parametric
  (Mann–Whitney) and ANOVA/Kruskal options rather than t-test only.
- 🟡 **Mutable default arguments.** `plot_means_with_significance(..., comparisons=[],
  annotation_cols=[], ...)` (`categorical.py:7-12`) uses shared mutable defaults — a classic
  latent bug. Use `None` and initialize inside.
- 🟡 **High-resolution heuristic is fragile.** `_detect_and_tag_high_res` flags any dataset
  with a numeric value `> 1e6` and hard-codes a `scaling_factor = 10000.0`
  (`loader.py:224-250`). A single outlier event, a Time channel in ms, or an index-like column
  can trip this and silently rescale all channels by 10⁴. Make detection channel-aware, make the
  factor configurable, and show what was rescaled.
- 🟢 **Legacy state aliasing.** e.g., `bar_ui.py:32-38` syncs `bar_gates` ↔
  `bar_plot_local_gates` in both directions to preserve backward compatibility. This kind of
  dual-key bookkeeping is error-prone; pick one key and migrate.

---

## Part 4 — Security & robustness

- 🔴 **Zip Slip (path traversal) on folder upload.** `save_uploaded_folder_zip` calls
  `zip_ref.extractall(target_dir)` (`glow/data_loading/loader.py:82-83`) without validating
  member paths. A malicious/crafted zip with `../` entries can write outside `target_dir`.
  Validate each member resolves inside the target directory before extracting.
- 🟡 **Untrusted pickle.** The app loads pickled models from `data/*.pkl` and uses
  `pickle.loads(pickle.dumps(fig))` in `glow/canvas/functions.py:24`. Pickle is fine for
  self-produced model files, but **do not** build the future "save session" feature on pickle —
  loading a session file is loading untrusted input. Use JSON for session state, and document
  that model `.pkl` files must come from a trusted source.
- 🟡 **No upload limits.** Uploaded files and extracted zips are written to disk with no size or
  count cap (`save_uploaded_files`, `save_uploaded_folder_zip`), and the full experiment lives in
  memory. Large uploads can exhaust disk/RAM. Add configurable limits and clear messaging.
- 🟢 **Arbitrary directory scan.** "Enter Directory Path" walks any path the process can read
  (`loader.scan_directory`). Acceptable for a local desktop tool, but worth a note/guard.

---

## Part 5 — Performance

- 🟡 **No caching of expensive steps.** Experiment loading, KDE/violin computation, and gating
  run on every Streamlit rerun. Wrap pure, expensive functions in `@st.cache_data` /
  `@st.cache_resource` (keyed on inputs) to avoid recomputation on every widget interaction.
- 🟡 **`importlib.reload` on the hot path** (see Part 2) re-imports plotting modules every time a
  plot is generated.
- 🟢 **Liberal deep copies.** Frequent `copy.copy(experiment)` / `.clone()` / `df.copy()` (e.g.,
  `ui/plot/view.py:38-43`, `bar_ui.py:247`) duplicate potentially large DataFrames. Copy only
  what's mutated, or operate on views/index masks where possible.

---

## Part 6 — Testing, tooling & project hygiene

- 🔴 **Effectively no automated tests.** The only `test_*` file is
  `glow/active_learning/test_pretrained.py`, which is a manual script, and `.gitignore` excludes
  `/tests/`, `tests/`, and `pytest.ini` — so tests are deliberately kept out of the repo. Add a
  real `tests/` suite (pytest): metadata parsing, key-merge logic, channel resolution, the
  channel-union loader, gating strategies on synthetic data, and the stats/fold-change math.
- 🟡 **No CI / linting / formatting config.** Add GitHub Actions (lint + tests on a
  Linux/macOS/Windows matrix), and adopt `ruff`/`black` + a pre-commit hook. This would catch the
  duplicate branches, bare `except:`, and unused code automatically.
- 🟢 **Large binaries committed to git.** `data/gating_model_full.pkl`,
  `data/singlet_model_full.pkl`, and `assets/glow_icon_breath.gif` (~1.0 MB),
  `assets/glow_icon_breath.svg` (~0.9 MB), `assets/glow_icon.png` (~0.55 MB) live in the repo.
  Consider Git LFS or attaching models to releases to keep clone size down.
- 🟢 **Debug prints in library code.** `print("DEBUG: …")` in `loader.py` and `categorical.py:26`
  should become `logging.debug(...)`.
- 🟢 **README gaps.** Windows/Mac sections say installers "will be provided" but give no build
  steps; the conda env name in the README (`glow_env`) should be kept in sync with the shared
  env at `/home/shared/envs/glow_envs`. Document the actual launch command's CWD requirement
  (must run from repo root) until the path-handling fix lands.

---

## Suggested priority order

1. **Robustness & data trust (fastest wins):** friendlier errors + logging (§1.4), fix Zip Slip
   (§4), stop fabricating 0.0 for missing channels or at least warn (§3), remove
   `importlib.reload` and debug prints (§2, §5).
2. **The UX blockers you called out:** flexible box-plot grouping (§1.1), don't lock gating /
   allow metadata edits (§1.3).
3. **Durability:** save/restore session + URL tab state, and stop auto-clearing the cache (§1.2).
4. **Portability & packaging:** cross-platform env, module-relative paths, packaging config +
   `BUILD.md`, OS targets (§1.5).
5. **Structural health (ongoing):** split the gating/analysis/canvas monoliths, add tests + CI,
   centralize the metadata schema (§2, §6).

---

_This review is intentionally specific with file:line references so each item can be picked up
independently. None of the changes above require a rewrite — they are incremental and can be
sequenced by the priority list._
