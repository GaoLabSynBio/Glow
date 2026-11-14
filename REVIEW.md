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

B1 — Scale high-res data once at load in `DataLoader` (gate: before). **Design note; no code edited yet.**

## Request

**What changes**
- `glow/data_loading/loader.py` (high-res tagging, ~l.270-290): after detecting max > 1e6, divide all numeric
  channel columns by `scaling_factor` (10,000) and set `is_scaled=True`, `is_high_res=True`,
  `scaling_factor=10000`. Skip if metadata already says `is_scaled`.
  `glow/utils.py::detect_and_scale_high_res` stays as the idempotent fallback (it returns early when
  `is_scaled`), so the compensation view (`ui/compensation/view.py:128`) becomes a no-op.
- `ui/gating/ui_gating.py` export (~l.2966-2999): multiply channel columns back by `scaling_factor` before
  writing; keep the `# GlowMetadata:` header so exports are raw.
- `ui/gating/ui_file_loader.py` CSV import (~l.190): CSV with header `is_scaled`/`scaling_factor` is treated as
  raw-with-factor and scaled on import via the same helper; CSVs without it that exceed 1e6 are scaled like FCS.
- `positive.py`, Plot and Analysis are NOT touched (B2 removes their branches). They already read `is_scaled`,
  so they keep working in the interim.

**Risks**
- Live/singlet gating currently runs on raw values: hard-coded axis ranges, default thresholds, `auto_gate_live_cells`,
  MLP inputs. The spec says the MLP centres/standardises log inputs, so it should be unaffected, but log10 shifts by
  a constant 4. I will grep `ui_gating.py` and `glow/gating/*` for absolute limits, clamps and `log10(x+1)` offsets
  and fix them in this item.
- Gates stored in raw units live only in session state, so no migration is needed.

**Verification**
- New unit tests: loader scales high-res data once (idempotent on a second call); export round-trips to raw;
  CSV import with and without `is_scaled`.
- `tools/apptest_walk.py` extended to assert loaded `experiment.data` max < 1e6 for the high-res test FCS and that the
  exported CSV max equals the raw max. Plus pytest.

**Question**: if I find hard-coded raw-unit constants (axis ranges, default thresholds), may I derive them from data
quantiles, or should I just rescale them by 1/10,000? Default if you don't say: rescale by 1/10,000.

## Blocked tools / problems

None. (A `python3` heredoc needed approval and was not run; I used the Write tool instead.)
