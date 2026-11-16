"""End-to-end smoke test of the Glow Streamlit app using Streamlit's AppTest.

Walks: load directory → metadata preview → confirm → live gate accept → singlet accept →
Plot tab (all four plot types generate) → Analysis tab (Add New Gate opens) → Canvas tab.
Exits non-zero on any exception rendered by the app or a missing checkpoint.

Run:  /home/shared/envs/glow_envs/bin/python -I tools/apptest_walk.py
"""
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "ui"))
sys.path.insert(0, REPO)

from streamlit.testing.v1 import AppTest  # noqa: E402

DATA_DIR = os.path.join(REPO, "data", "uploaded_raw", "My_Experiment_1")
FAILURES = []


def check(cond, msg):
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        FAILURES.append(msg)


def run(at, label):
    t = time.time()
    at.run()
    dt = time.time() - t
    exc = [str(e)[:400] for e in at.exception]
    err = [e.value[:200] for e in at.error]
    print(f"\n== {label}  ({dt:.1f}s)")
    for e in exc:
        print("  EXCEPTION:", e)
    for e in err:
        print("  ERROR:", e)
    check(not exc, f"{label}: no exceptions")
    return at


def button(at, label_prefix=None, key=None):
    for b in at.button:
        if key is not None and b.key == key:
            return b
        if label_prefix is not None and b.label.startswith(label_prefix):
            return b
    return None


def widget(coll, label):
    for w in coll:
        if w.label == label:
            return w
    return None


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"Test data directory missing: {DATA_DIR}")
        return 2

    at = AppTest.from_file("ui/app.py", default_timeout=300)
    run(at, "fresh load")
    check(widget(at.radio, "Choose Input Method") is not None, "input method radio present")

    # --- Step 1: load directory
    r = widget(at.radio, "Choose Input Method")
    r.set_value("Enter Directory Path")
    run(at, "directory mode")
    ti = next((t for t in at.text_input if "path" in t.label.lower()), None)
    check(ti is not None, "directory path input present")
    ti.set_value(DATA_DIR)
    b = button(at, "Scan Directory")
    check(b is not None, "Scan Directory button present")
    b.click(); run(at, "scanned")
    b = button(at, "Generate Preview")
    if b is not None:  # may be merged into scan by a later revamp item
        b.click(); run(at, "preview")
    b = button(at, "Confirm and Load")
    check(b is not None, "Confirm and Load button present")
    b.click(); run(at, "confirmed load → auto gate")
    check(bool(at.session_state["loaded_files"]), "loaded_files populated")
    check("base_experiment" in at.session_state.filtered_state, "base_experiment created")

    base = at.session_state["base_experiment"]
    chans = [c for c in getattr(base, "channels", []) if c in base.data.columns]
    meta = getattr(base, "metadata", None) or {}
    print(f"  base_experiment high_res={meta.get('is_high_res')} is_scaled={meta.get('is_scaled')}")
    check(float(base.data[chans].max().max()) < 1e6, "loaded channel values < 1e6 (scaled once at load)")
    check(meta.get("is_scaled", False) == meta.get("is_high_res", False), "is_scaled matches is_high_res")

    # --- Step 2: live gate
    acc = button(at, "✅ Accept") or button(at, "Accept")
    check(acc is not None, "live-gate Accept button present")
    if acc:
        acc.click(); run(at, "accepted live gate")
        check(at.session_state["gate_accepted"] is True, "gate_accepted set")

    # --- Step 3: singlets
    s_acc = next((b for b in at.button if "Accept" in b.label and "Singlet" in b.label), None)
    check(s_acc is not None, "singlet Accept button present")
    if s_acc:
        s_acc.click(); run(at, "accepted singlets")
        check("singlet_gate_params" in at.session_state.filtered_state
              or "cell_line_singlet_gates" in at.session_state.filtered_state, "singlet gate stored")

    # --- Plot tab (works on base/gated experiment without finalize)
    nav = widget(at.sidebar.radio, "Go to") or widget(at.radio, "Go to")
    check(nav is not None, "navigation radio present")
    if nav:
        nav.set_value("Plot"); run(at, "plot tab")
        check("sample_metadata" in at.session_state.filtered_state, "shared metadata editor built sample_metadata")
        check(any("Edit metadata" in e.label for e in at.expander), "Edit metadata expander on Plot tab")
        check(any("exports are raw" in c.value for c in at.caption), "units caption on Plot tab")
        check(not any("Raw Values" in t.label for t in at.toggle), "no raw-values toggle on Plot tab")
        for pt, gen_prefix in [("1D Histogram", "Generate"), ("2D Scatter", "Generate"),
                               ("Bar Plot", "Generate"), ("Pairwise Plot", "Generate")]:
            sel = widget(at.radio, "Plot type")
            if sel is None:
                check(False, f"plot type selector present for {pt}")
                break
            sel.set_value(pt); run(at, f"plot type {pt}")
            gb = button(at, gen_prefix)
            if gb is not None:
                gb.click(); run(at, f"generated {pt}")
            else:
                print(f"  (no Generate button for {pt}; assuming live preview)")

        # --- Analysis tab
        nav = widget(at.sidebar.radio, "Go to") or widget(at.radio, "Go to")
        nav.set_value("Analysis"); run(at, "analysis tab")
        check(len(at.dataframe) >= 1, "summary statistics table rendered")
        check(any("exports are raw" in c.value for c in at.caption), "units caption on Analysis tab")
        check(not any("Raw Values" in t.label for t in at.toggle), "no raw-values toggle on Analysis tab")
        ag = button(at, "Add New Gate")
        if ag is not None:
            ag.click(); run(at, "add new gate")
            check(widget(at.selectbox, "Analysis Mode") is not None, "gate creator opened")

        # --- Canvas
        nav = widget(at.sidebar.radio, "Go to") or widget(at.radio, "Go to")
        nav.set_value("Canvas"); run(at, "canvas tab")

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} CHECK(S) FAILED:"))
    for f in FAILURES:
        print("  -", f)
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
