"""Shared sample grouping for analysis tables.

Events are grouped by the stable ``Sample`` id (file stem) so sample names stay short and
metadata labels (Experiment Name, Replicate, ...) can change without altering the keys.
"""
from typing import List

# Internal gating columns that must never split a sample.
EXCLUDE_CONDITIONS = [
    'cells', 'singlets', 'IsSinglet', 'Is_Singlet',
    'Population', 'AutoLiveGate', 'LiveGate', 'SingletGate',
    'Is_Live', 'Live', 'Density'
]

_FALLBACK_COLS = ['Sample', 'sample', 'Name', 'name', 'folder_name']


def resolve_group_cols(experiment, df) -> List[str]:
    """Columns that define one sample: ``Sample`` when present, else the legacy conditions."""
    if 'Sample' in df.columns:
        return ['Sample']
    group_cols = [c for c in _condition_names(experiment)
                  if c not in EXCLUDE_CONDITIONS and not c.startswith('__')]
    if not group_cols:
        for c in _FALLBACK_COLS:
            if c in df.columns:
                return [c]
    return [c for c in group_cols if c in df.columns]


def _condition_names(experiment) -> List[str]:
    """Condition names, tolerating data columns that lack a metadata entry.

    cytoflow's ``Experiment.conditions`` raises KeyError when a column (e.g. the gating
    helper ``_temp_live_labels``) was added to ``data`` without metadata.
    """
    meta = getattr(experiment, 'metadata', None)
    if isinstance(meta, dict):
        return [k for k, v in meta.items() if isinstance(v, dict) and v.get('type') == 'condition']
    try:
        return list(experiment.conditions.keys())
    except Exception:
        return []


def condition_columns(experiment, df, group_cols) -> List[str]:
    """Condition columns (other than the group key) to show as their own table columns."""
    conds = _condition_names(experiment)
    return [c for c in conds
            if c in df.columns and c not in group_cols
            and c not in EXCLUDE_CONDITIONS and not c.startswith('__')]
