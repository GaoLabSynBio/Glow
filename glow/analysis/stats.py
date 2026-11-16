import pandas as pd

from glow.analysis.grouping import condition_columns, resolve_group_cols

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

    # Group by the stable Sample id; other conditions become their own columns.
    valid_group_cols = resolve_group_cols(experiment, data)
    extra_cols = condition_columns(experiment, data, valid_group_cols)
    if valid_group_cols:
        grouped = data.groupby(valid_group_cols, observed=True)
    else:
        grouped = [('All', data)]

    for name, group in grouped:
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
            **{c: group[c].iloc[0] for c in extra_cols if len(group)},
            'Total Events': total_events,
            'Live Count': live_count,
            'Live %': round(live_pct, 2),
            'Singlets Count': singlet_count,
            'Singlets %': round(singlet_pct, 2)
        }
        results.append(row)

    return pd.DataFrame(results)


_LEADING_CONDITIONS = ['Experiment_Name', 'Replicate']


def summary_condition_columns(experiment):
    """Condition columns (excluding the ``Sample`` key) that label rows of the summary table."""
    if experiment is None:
        return []
    data = experiment.data
    return condition_columns(experiment, data, resolve_group_cols(experiment, data))


def order_summary_columns(df, condition_cols):
    """Experiment Name, Replicate, other conditions, ``Sample``, then metric columns."""
    present = [c for c in condition_cols if c in df.columns]
    leading = [c for c in _LEADING_CONDITIONS if c in present]
    leading += [c for c in present if c not in leading]
    if 'Sample' in df.columns:
        leading.append('Sample')
    metrics = [c for c in df.columns if c not in leading]
    return df[leading + metrics]


def group_by_experiment(df, condition_cols):
    """Mean and SD of every numeric metric per Experiment Name across replicates.

    Other conditions are kept when constant within an experiment. Returns ``df`` unchanged
    when there is no ``Experiment_Name`` column.
    """
    if df.empty or 'Experiment_Name' not in df.columns:
        return df
    key = 'Experiment_Name'
    skip = set(condition_cols) | {'Sample', 'Replicate', key}
    metrics = [c for c in df.columns
               if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(key, sort=False, observed=True, dropna=False)

    sizes = grouped.size()
    out = pd.DataFrame(index=sizes.index)
    for c in condition_cols:
        if c in df.columns and c not in (key, 'Replicate', 'Sample'):
            if (grouped[c].nunique(dropna=False) <= 1).all():
                out[c] = grouped[c].first()
    out['Replicates'] = sizes
    for m in metrics:
        out[f'{m} mean'] = grouped[m].mean().round(2)
        out[f'{m} SD'] = grouped[m].std(ddof=1).round(2)
    return out.reset_index()
