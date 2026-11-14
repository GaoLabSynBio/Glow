"""Sample metadata: stable sample ids, default names, replicates and relabelling.

Pure pandas, no Streamlit. ``Sample`` (the file stem) is the stable key used by gates,
overlays and stats joins; every other column here is a *label* that can change at any time
without re-gating.
"""
import os
from typing import Dict, Iterable, List, Optional

import pandas as pd

SAMPLE_COL = "Sample"
NAME_COL = "Experiment_Name"
REPLICATE_COL = "Replicate"
CELL_LINE_COL = "Cell_Line"
CONDITION_COL = "Condition"

# Keys of a preview_meta entry that are file bookkeeping, not user metadata.
_FILE_KEYS = {"original_path", "display_name", "condition", "Sample", "Condition"}


def _is_blank(value) -> bool:
    if value is None:
        return True
    if not isinstance(value, str) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def _sample_ids(paths: List[str]) -> List[str]:
    """File stems; the parent folder is prefixed only for stems that collide."""
    stems = [_stem(p) for p in paths]
    counts = pd.Series(stems).value_counts()
    ids = []
    for path, stem in zip(paths, stems):
        if counts[stem] > 1:
            folder = os.path.basename(os.path.dirname(str(path))).replace(" ", "_")
            stem = f"{folder}_{stem}" if folder else stem
        ids.append(stem)
    seen: Dict[str, int] = {}
    for i, sid in enumerate(ids):  # still colliding (same folder name): number them
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] > 1:
            ids[i] = f"{sid}_{seen[sid]}"
    return ids


def infer_replicates(names: Iterable) -> pd.Series:
    """Integer replicate numbers: rows sharing a name are numbered 1, 2, 3 in order."""
    s = pd.Series(list(names), dtype="object").astype(str)
    return (s.groupby(s).cumcount() + 1).astype(int)


def default_experiment_names(meta: pd.DataFrame) -> pd.DataFrame:
    """Fill blank Experiment_Name with the Sample id (file stem). Returns a copy."""
    out = meta.copy()
    if NAME_COL not in out.columns:
        out[NAME_COL] = ""
    names = out[NAME_COL].astype(object)
    blank = names.map(_is_blank).to_numpy()
    names[blank] = out.index[blank]
    out[NAME_COL] = names.astype(str)
    return out


def build_sample_metadata(preview_meta: List[dict]) -> pd.DataFrame:
    """One row per file, indexed by Sample id.

    Columns: Experiment_Name (default: file stem), Replicate (inferred int), Cell_Line
    (as given, blank if unknown), Condition (folder) and any extra user columns.
    """
    rows = [dict(m) for m in preview_meta]
    paths = [str(m.get("original_path") or m.get("display_name") or "") for m in rows]
    ids = _sample_ids(paths)

    records = []
    for m, path in zip(rows, paths):
        rec = {
            NAME_COL: m.get(NAME_COL, ""),
            REPLICATE_COL: m.get(REPLICATE_COL),
            CELL_LINE_COL: "" if _is_blank(m.get(CELL_LINE_COL)) else str(m[CELL_LINE_COL]),
            CONDITION_COL: m.get("condition", m.get(CONDITION_COL)) or os.path.basename(os.path.dirname(path)),
        }
        for k, v in m.items():
            if k not in _FILE_KEYS and k not in rec:
                rec[k] = v
        records.append(rec)

    meta = pd.DataFrame(records, index=pd.Index(ids, name=SAMPLE_COL))
    meta = default_experiment_names(meta)
    given = pd.to_numeric(meta[REPLICATE_COL], errors="coerce")
    meta[REPLICATE_COL] = given.where(given.notna(), infer_replicates(meta[NAME_COL]).to_numpy()).astype(int)
    return meta


def sample_label_map(meta: pd.DataFrame) -> Dict[str, str]:
    """Sample id -> display label: Experiment Name, plus ' rep N' when the name repeats."""
    names = meta[NAME_COL].astype(str)
    multi = names.map(names.value_counts()) > 1
    if REPLICATE_COL in meta.columns:
        reps = meta[REPLICATE_COL]
    else:
        reps = pd.Series(infer_replicates(names).to_numpy(), index=meta.index)
    return {
        sid: (f"{n} rep {r}" if m else n)
        for sid, n, r, m in zip(meta.index, names, reps, multi)
    }


def label_map_from_data(df: pd.DataFrame) -> Dict[str, str]:
    """Sample id -> display label derived from an event table (identity if no names)."""
    if SAMPLE_COL not in df.columns:
        return {}
    cols = [c for c in (SAMPLE_COL, NAME_COL, REPLICATE_COL) if c in df.columns]
    per_sample = df[cols].drop_duplicates(SAMPLE_COL).astype({SAMPLE_COL: str}).set_index(SAMPLE_COL)
    if NAME_COL not in per_sample.columns:
        return {s: s for s in per_sample.index}
    return sample_label_map(per_sample)


def apply_sample_metadata(experiment, meta: pd.DataFrame, drop: Optional[Iterable[str]] = None):
    """Relabel an experiment in place from ``meta`` (indexed by Sample); no re-gating.

    Each metadata column is mapped onto the event table by ``Sample`` and registered as a
    condition. Columns listed in ``drop`` are removed. Returns the experiment.
    """
    data = experiment.data
    if SAMPLE_COL not in data.columns:
        raise KeyError(f"Experiment has no '{SAMPLE_COL}' column")
    for col in meta.columns:
        if col == SAMPLE_COL:
            continue
        mapped = data[SAMPLE_COL].astype(str).map(meta[col])
        if col == REPLICATE_COL:
            data[col] = mapped
            experiment.metadata[col] = {"type": "condition", "class": "int"}
        else:
            data[col] = mapped.astype("object").fillna("NA").astype("category")
            experiment.metadata[col] = {"type": "condition", "class": "category"}
        conds = getattr(experiment, "conditions", None)
        if isinstance(conds, dict):  # MockExperiment keeps a plain dict (cytoflow derives it)
            conds[col] = "int" if col == REPLICATE_COL else "category"
    for col in drop or ():
        if col in data.columns and col not in meta.columns and col != SAMPLE_COL:
            data.drop(columns=col, inplace=True)
            experiment.metadata.pop(col, None)
    return experiment


# Columns of the event table that are gating/bookkeeping state, never user metadata.
_NON_METADATA = {SAMPLE_COL, CONDITION_COL, "cells", "singlets", "Is_Live", "Is_Singlet",
                 "LiveGate", "SingletGate"}


def metadata_from_events(df: pd.DataFrame, extra: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Rebuild the per-sample metadata table (indexed by Sample) from an event table."""
    cols = [c for c in (NAME_COL, REPLICATE_COL, CELL_LINE_COL) if c in df.columns]
    cols += [c for c in (extra or ()) if c in df.columns and c not in cols and c not in _NON_METADATA]
    keys = df[SAMPLE_COL].astype(str).to_numpy()
    meta = df[cols].astype(object).groupby(keys, sort=False).first()
    meta.index.name = SAMPLE_COL
    meta = default_experiment_names(meta)
    if REPLICATE_COL in meta.columns:
        rep = pd.to_numeric(meta[REPLICATE_COL], errors="coerce")
        meta[REPLICATE_COL] = rep.where(rep.notna(), infer_replicates(meta[NAME_COL]).to_numpy()).astype(int)
    else:
        meta[REPLICATE_COL] = infer_replicates(meta[NAME_COL]).to_numpy()
    return meta


def reinfer_replicates(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """After an edit: if names changed, renumber replicates, keeping any the user edited by hand."""
    new = new.copy()
    if NAME_COL not in new.columns or not (old[NAME_COL].astype(str) != new[NAME_COL].astype(str)).any():
        return new
    inferred = pd.Series(infer_replicates(new[NAME_COL]).to_numpy(), index=new.index)
    if REPLICATE_COL in old.columns and REPLICATE_COL in new.columns:
        by_hand = old[REPLICATE_COL].astype(str) != new[REPLICATE_COL].astype(str)
        inferred = inferred.where(~by_hand, pd.to_numeric(new[REPLICATE_COL], errors="coerce"))
    new[REPLICATE_COL] = inferred.astype(int)
    return new


def cell_line_changes(old: pd.DataFrame, new: pd.DataFrame):
    """Compare Cell_Line before/after an edit.

    Returns ``(renames, affected)``. ``renames`` maps old->new for lines whose every file
    moved together to a brand-new name (gates can simply be re-keyed). ``affected`` holds the
    lines (old and new) involved in any other change; their gates no longer cover the right files.
    """
    if CELL_LINE_COL not in old.columns or CELL_LINE_COL not in new.columns:
        return {}, set()
    before = old[CELL_LINE_COL].astype(str)
    after = new[CELL_LINE_COL].astype(str).reindex(before.index)
    changed = before != after
    renames: Dict[str, str] = {}
    affected = set()
    for line in before[changed].unique():
        rows = before == line
        targets = set(after[rows])
        target = next(iter(targets)) if len(targets) == 1 else None
        if target is not None and bool(changed[rows].all()) and target not in set(before):
            renames[line] = target
        else:
            affected.add(line)
            affected.update(targets)
    return renames, affected


def rekey_gates(gates: Optional[dict], renames: Dict[str, str]) -> dict:
    """Return ``gates`` with keys renamed per ``renames`` (no recompute)."""
    return {renames.get(k, k): v for k, v in (gates or {}).items()}
