import pandas as pd
import numpy as np

def get_summary_statistics(experiment):
    """
    Returns a DataFrame with summary statistics for the experiment.
    Include: Sample Name, Total Events, Live %, Singlets %.

    Args:
        experiment: Cytoflow Experiment object.

    Returns:
        pd.DataFrame
    """
    if experiment is None:
        return pd.DataFrame()

    results = []
    data = experiment.data

    # Identify sample grouping column
    group_cols = []

    # Define internal gating columns to exclude from grouping
    # These are added as 'conditions' for metadata validity but should not split the sample summary
    EXCLUDE_CONDITIONS = [
        'cells', 'singlets', 'IsSinglet', 'Is_Singlet',
        'Population', 'AutoLiveGate', 'LiveGate', 'SingletGate',
        'Is_Live', 'Live', 'Density'
    ]

    try:
        # Cytoflow property access can trigger lazy evaluation looking up cached keys in metadata
        # If metadata[x] missing for a key it thinks is a condition, it crashes.
        if hasattr(experiment, 'conditions'):
            all_conditions = list(experiment.conditions.keys())
            # Filter out excluded conditions
            group_cols = [
                c for c in all_conditions
                if c not in EXCLUDE_CONDITIONS
                and not c.startswith('__') # exclude internal python attrs if any
            ]
    except Exception:
        # Fallback if Cytoflow metadata is corrupted or mismatching data columns
        pass

    # If no conditions found in experiment metadata, check columns for typical names
    if not group_cols:
        possible_cols = ['Sample', 'sample', 'Name', 'name', 'folder_name']
        for c in possible_cols:
            if c in data.columns:
                group_cols = [c]
                break

    if group_cols:
        # Check if columns actually exist in data to avoid KeyError
        valid_group_cols = [c for c in group_cols if c in data.columns]
        if valid_group_cols:
            grouped = data.groupby(valid_group_cols, observed=True)
        else:
             grouped = [('All', data)]
    else:
        # specific handling if no grouping (single file?)
        grouped = [('All', data)]

    for name, group in grouped:
        # Format sample name
        if isinstance(name, tuple):
            sample_name = "_".join(map(str, name))
        else:
            sample_name = str(name)

        total_events = len(group)

        # --- Resolve LIVE Column ---
        # Candidates for Live/Dead flag
        # Order: 'cells' (CSV standard), 'Is_Live', 'AutoLiveGate', 'Live'
        live_col_name = None
        for cand in ['cells', 'Cells', 'Is_Live', 'AutoLiveGate', 'Live', 'LiveGate']:
            if cand in group.columns:
                live_col_name = cand
                break

        # Calculate Live Mask
        if live_col_name:
            col = group[live_col_name]
            if pd.api.types.is_bool_dtype(col):
                live_mask = col
            else:
                # Numeric (GMM/DBSCAN): Assume >= 0 is "Kept"
                vals = pd.to_numeric(col, errors='coerce')
                live_mask = (vals >= 0)
        else:
            # No gate found -> Assume all live? Or 0?
            # Typically 0 implies missing data or failed gating.
            live_mask = pd.Series([False] * total_events, index=group.index)

        live_count = live_mask.sum()

        # --- Resolve SINGLETS Column ---
        # Candidates for Singlet flag
        # Order: 'singlets', 'Singlets', 'Is_Singlet', 'Singlet', 'LearnedSinglet'
        singlet_col_name = None
        for cand in ['singlets', 'Singlets', 'Is_Singlet', 'IsSinglet', 'Singlet', 'LearnedSinglet']:
            if cand in group.columns:
                singlet_col_name = cand
                break

        # Calculate Singlet Mask (Subset of Live)
        # Requirement: "singlets percentage should be further determined among cells=TRUE"
        if singlet_col_name:
            col = group[singlet_col_name]
            if pd.api.types.is_bool_dtype(col):
                raw_singlet_mask = col
            else:
                vals = pd.to_numeric(col, errors='coerce')
                raw_singlet_mask = (vals >= 0)

            # Intersection: Must be Live AND Singlet
            # This ensures we count the subset.
            final_singlet_mask = live_mask & raw_singlet_mask
            singlet_count = final_singlet_mask.sum()
        else:
            singlet_count = 0

        # Calculate percentages
        # Live %: Fraction of Total
        live_pct = (live_count / total_events * 100) if total_events > 0 else 0

        # Singlets %: Fraction of Live Population (per user instruction)
        singlet_pct = (singlet_count / live_count * 100) if live_count > 0 else 0

        row = {
            'Sample': sample_name,
            'Total Events': total_events,
            'Live Count': live_count,
            'Live %': round(live_pct, 2),
            'Singlets Count': singlet_count,
            'Singlets %': round(singlet_pct, 2)
        }
        results.append(row)

    return pd.DataFrame(results)
