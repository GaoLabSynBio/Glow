"""Shared "Edit metadata" editor for the Gating, Analysis and Plot tabs.

``st.session_state.sample_metadata`` (one row per Sample id) is the source of truth for the
label columns. Saving relabels every loaded experiment in place (`apply_sample_metadata`);
nothing is re-gated. Only a *Cell_Line* change touches gating state: a pure rename re-keys
the stored gates, any other reassignment flags the affected lines in
``st.session_state.lines_need_regate``.
"""
import pandas as pd
import streamlit as st

from glow.data_loading import sample_metadata as sm

EXPERIMENT_KEYS = ("base_experiment", "gated_experiment", "analysis_experiment")
GATE_KEYS = ("cell_line_gates", "cell_line_singlet_gates")
_EDITABLE_EXTRA_SKIP = {"Experiment_Name", "Replicate", "Cell_Line", "Sample", "Condition"}


def _loaded_experiments():
    return [(k, st.session_state[k]) for k in EXPERIMENT_KEYS
            if st.session_state.get(k) is not None and hasattr(st.session_state[k], "data")]


def _extra_columns(experiment):
    """User-defined condition columns (category/str) other than the standard ones."""
    skip = _EDITABLE_EXTRA_SKIP | sm._NON_METADATA
    out = []
    for col, info in (getattr(experiment, "metadata", None) or {}).items():
        if col in skip or col not in experiment.data.columns:
            continue
        if isinstance(info, dict) and info.get("type") == "condition":
            out.append(col)
    return out


def ensure_sample_metadata():
    """Return the sample-metadata table, (re)building it when the loaded samples changed."""
    exps = _loaded_experiments()
    if not exps:
        return None
    exp = exps[0][1]
    data = exp.data
    if sm.SAMPLE_COL not in data.columns:
        return None
    samples = set(data[sm.SAMPLE_COL].astype(str).unique())
    meta = st.session_state.get("sample_metadata")
    if meta is None or set(meta.index) != samples:
        # Use the base experiment for completeness (gated ones only hold surviving events).
        base = st.session_state.get("base_experiment")
        base = exp if base is None else base
        meta = sm.metadata_from_events(base.data, extra=_extra_columns(base))
        st.session_state.sample_metadata = meta
    return meta


def _clean_blank(meta):
    """Blank text -> 'NA' for categorical columns (what the loader writes for unknowns)."""
    meta = meta.copy()
    for col in meta.columns:
        if col == sm.REPLICATE_COL:
            continue
        blank = "HEK" if col == sm.CELL_LINE_COL else "NA"
        meta[col] = meta[col].map(lambda v, b=blank: b if sm._is_blank(v) else str(v).strip())
    return meta


def save_sample_metadata(old, new):
    """Apply ``new`` to every loaded experiment and gate dict. Returns (renames, affected)."""
    new = new.copy()
    rep = pd.to_numeric(new[sm.REPLICATE_COL], errors="coerce")
    new[sm.REPLICATE_COL] = rep.fillna(old[sm.REPLICATE_COL]).astype(int)
    new = sm.default_experiment_names(new)
    new = sm.reinfer_replicates(old, new)
    new = _clean_blank(new)
    new[sm.REPLICATE_COL] = new[sm.REPLICATE_COL].astype(int)

    renames, affected = sm.cell_line_changes(old, new)
    for _, exp in _loaded_experiments():
        sm.apply_sample_metadata(exp, new)
    for key in GATE_KEYS:
        if renames and key in st.session_state:
            st.session_state[key] = sm.rekey_gates(st.session_state[key], renames)
    for old_line, new_line in renames.items():  # a rename carries any pending flag with it
        pending = set(st.session_state.get("lines_need_regate", ()))
        if old_line in pending:
            st.session_state.lines_need_regate = (pending - {old_line}) | {new_line}
    if affected:
        st.session_state.lines_need_regate = set(st.session_state.get("lines_need_regate", ())) | affected

    # Keep the loader's file list in step so a rebuild of the experiment keeps the edits.
    for entry in st.session_state.get("loaded_files") or []:
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("display_name", "")).rsplit(".", 1)[0]
        if sid in new.index:
            for col in new.columns:
                if col != sm.CONDITION_COL:
                    entry[col] = new.at[sid, col]
    st.session_state.sample_metadata = new
    return renames, affected


def render_metadata_editor(context):
    """Collapsed "Edit metadata" expander; edits relabel without re-gating."""
    meta = ensure_sample_metadata()
    if meta is None:
        return
    with st.expander("✏️ Edit metadata (names, replicates, cell lines)", expanded=False):
        st.caption("Changes relabel the data everywhere — gates are kept and nothing is re-run. "
                   "Renaming a cell line keeps its gates; moving files between cell lines "
                   "marks those lines as needing re-gating.")
        ver = st.session_state.get("_meta_editor_ver", 0)
        view = meta.reset_index()
        view[sm.SAMPLE_COL] = view[sm.SAMPLE_COL].astype(str)
        view[sm.REPLICATE_COL] = pd.to_numeric(view[sm.REPLICATE_COL], errors="coerce").fillna(1).astype(int)
        with st.form(f"meta_form_{context}"):
            edited = st.data_editor(
                view,
                column_config={
                    sm.SAMPLE_COL: st.column_config.TextColumn("Sample", disabled=True),
                    sm.NAME_COL: st.column_config.TextColumn("Experiment Name"),
                    sm.REPLICATE_COL: st.column_config.NumberColumn(
                        "Replicate", min_value=1, step=1,
                        help="Renumbered per name when you rename; edit a value to override."),
                    sm.CELL_LINE_COL: st.column_config.TextColumn(
                        "Cell Line", help="Rename = gates follow. Reassign = affected lines need re-gating."),
                    sm.CONDITION_COL: None,  # folder name is internal; hidden to keep the editor compact
                },
                disabled=[sm.SAMPLE_COL, sm.CONDITION_COL],
                hide_index=True, num_rows="fixed", width="stretch",
                key=f"meta_editor_{context}_{ver}",
            )
            c1, c2 = st.columns([0.7, 0.3])
            new_col = c1.text_input("Add column (optional)", key=f"meta_newcol_{context}_{ver}",
                                    placeholder="e.g. Treatment")
            saved = st.form_submit_button("Save metadata", type="primary")
        if saved:
            new = edited.set_index(sm.SAMPLE_COL)
            new = new.reindex(meta.index)
            if new_col and new_col.strip():
                name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in new_col.strip())
                if name and name not in new.columns:
                    new[name] = "NA"
            save_sample_metadata(meta, new)
            st.session_state._meta_editor_ver = ver + 1
            st.toast("Metadata updated — no re-gating needed.", icon="✏️")
            st.rerun()
        stale = sorted(st.session_state.get("lines_need_regate", ()))
        if stale:
            st.warning("Cell lines needing re-gating after reassignment: " + ", ".join(map(str, stale)))
