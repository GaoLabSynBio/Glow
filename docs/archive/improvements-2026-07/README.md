# Codebase review & improvement plan — July 2026 (archived)

`IMPROVEMENTS.md` is the full-repository review from 2026-07-10. Its five headline issues and
most of the hardening items were implemented in commit `4e26fc6` ("Implement reviewed
improvements (issues #1–#5 + hardening)"); issue #3 (locked gating / metadata edits) was
completed properly by the November 2026 UI revamp (`../revamp-2026-11/`, items A4, A5, D1).
`BUILD.md` (packaging/install guide) was a deliverable of that commit and stays at the repo
root as live documentation.

## Status as of 2026-11-16 (verified against the code)

| Item | Status | Where |
|---|---|---|
| 1.1 Box-plot grouping locked to Experiment Name | ✅ done — selectbox over all conditions, experiment column as default | `ui/plot/bar_ui.py:36-48`; `comparisons=None` etc. in `categorical.py:10` |
| 1.2 Save / restore session | ✅ done — JSON session (`ui/common/session.py`), tab in URL (`st.query_params`), no cache clearing on start | `ui/app.py:50,101` |
| 1.3 Gating locked, metadata not editable | ✅ done (revamp) — `base_experiment` keyed by file-list hash, metadata editor on every tab, stepper with Redo | revamp A4 `94d0dc4`, A5 `3be5b46`, D1 `807260f` |
| 1.4 Generic error messages | 🟡 partial — `ui/common/errors.py` helper + rotating file log (`glow/logging_config.py`), worst offenders replaced; **113 `except Exception` and 2 bare `except:` remain** | — |
| 1.5 Portability | 🟡 partial — `glow/paths.py` (module-relative assets, `$GLOW_DATA_DIR`), cross-platform `environment.yml` + Linux lockfile, `requirements.txt`, `BUILD.md`; **no PyInstaller/Electron config in repo, no Windows CI runner** | — |
| 2 God functions (`ui_gating.py`) | ❌ open — `ui_gating.py` is now 3 227 lines, `analysis/view.py` 1 709 | — |
| 2 `importlib.reload` hacks | ✅ removed | — |
| 2 Duplicated `download_btn` | ✅ consolidated (`ui/common/widgets.py`) | — |
| 2 Duplicated `get_default_channel` | ❌ open — still defined twice | `ui_gating.py:245,473` |
| 2 Dead code (`filter_by_condition` stub, unreachable `elif`) | ✅ removed | — |
| 2 Canonical metadata schema (`Experiment Name` vs `Experiment_Name`) | 🟡 partial — `glow/data_loading/sample_metadata.py` centralises the editor; both spellings still used in ~15 files | — |
| 3 Missing channels filled with 0.0 | ✅ done — kept `NaN`, `partial_channels` recorded and shown | `loader.py:421-433` |
| 3 Meaningless stats without replicates | ❌ open — single pseudo-replicate path still present | `categorical.py:183,187` |
| 3 Mutable default args | ✅ done | — |
| 3 High-res heuristic | 🟡 partial — scaling now happens once at load (revamp B1) and is shown to the user; detection threshold/factor still fixed | `loader.py:_detect_and_tag_high_res` |
| 3 Legacy state aliasing (`bar_gates` ↔ `bar_plot_local_gates`) | ❌ open | `bar_ui.py:17` |
| 4 Zip Slip | ✅ done — `_safe_extract` | `loader.py:93` |
| 4 Untrusted pickle for sessions | ✅ done — sessions are JSON | — |
| 4 Upload size/count limits | ❌ open | — |
| 5 Caching of expensive steps | 🟡 partial — live/dead gating cached (revamp C7); loading/KDE/singlets not | — |
| 5 Liberal deep copies | ❌ open | — |
| 6 Tests | ✅ done — 14 pytest modules in `tests/` + `tools/apptest_walk.py` | — |
| 6 CI | 🟡 partial — `.github/workflows/ci.yml` (byte-compile + tests on ubuntu/macos); no lint/format config, no Windows | — |
| 6 Large binaries in git | ❌ open — 2.6 MB tracked under `assets/` + `data/*.pkl` | — |
| 6 Debug prints | ✅ done — none left | — |
| 6 README gaps | 🟡 partial — env name consistent; Windows/Mac section still says installers "will be provided" | `README.md:25` |

## Open follow-ups worth a future round

1. Split `ui/gating/ui_gating.py` and `ui/analysis/view.py` into per-step modules.
2. Replace broad `except Exception` handlers with the `ui/common/errors.py` pattern.
3. Replicate-aware statistics (require replicate column, show *n*, non-parametric option).
4. Upload size/count limits; Git LFS for binaries; ruff + pre-commit; Windows CI runner.
