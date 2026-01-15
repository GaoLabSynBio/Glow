import pandas as pd
import numpy as np

def calculate_positive_stats(experiment, channel, threshold, suffix=""):
    """
    Returns positive % per sample for a given channel and threshold.

    Args:
        experiment: Cytoflow Experiment
        channel: Channel name (str)
        threshold: Cutoff value (float)
        suffix: Optional suffix for the result column (default: "Positive %")

    Returns:
        pd.DataFrame: ['Sample', '{Channel} Positive %']
    """
    df = experiment.data
    results = []

    # Identify sample grouping (shared logic)
    group_cols = []

    # Robust Check: attributes might raise KeyError if metadata is partial
    try:
        # We manually access the list to see if it works
        # hasattr doesn't catch KeyError
        if hasattr(experiment, 'conditions'):
             group_cols = list(experiment.conditions.keys())
    except Exception:
        group_cols = []


    if not group_cols:
        possible_cols = ['Sample', 'sample', 'Name', 'name', 'folder_name']
        for c in possible_cols:
            if c in df.columns:
                group_cols = [c]
                break

    if group_cols:
        grouped = df.groupby(group_cols)
    else:
        grouped = [('All', df)]

    col_name = f"{channel} Positive %" if not suffix else suffix
    count_col_name = f"{channel} Positive Count"

    for name, group in grouped:
        if isinstance(name, tuple):
            sample_name = "_".join(map(str, name))
        else:
            sample_name = str(name)

        # -----------------------------------------------------------
        # Filter for Singlet Population (Live & Singlet) if possible
        # -----------------------------------------------------------

        # 1. Identify Live Mask
        live_mask = pd.Series([True] * len(group), index=group.index)
        for cand in ['cells', 'Cells', 'Is_Live', 'AutoLiveGate', 'Live', 'LiveGate']:
            if cand in group.columns:
                col = group[cand]
                if pd.api.types.is_bool_dtype(col):
                    live_mask = col
                else:
                    live_mask = (pd.to_numeric(col, errors='coerce') >= 0)
                break

        # 2. Identify Singlet Mask
        singlet_mask = pd.Series([True] * len(group), index=group.index)
        for cand in ['singlets', 'Singlets', 'Is_Singlet', 'Singlet', 'LearnedSinglet']:
            if cand in group.columns:
                col = group[cand]
                if pd.api.types.is_bool_dtype(col):
                    singlet_mask = col
                else:
                    singlet_mask = (pd.to_numeric(col, errors='coerce') >= 0)
                break

        # Combined Filter
        valid_population_mask = live_mask & singlet_mask

        # Filter Data
        filtered_group = group[valid_population_mask]

        pos_count = 0
        pct = 0.0

        if len(filtered_group) > 0:
            vals = filtered_group[channel]
            count = len(vals)
            # Assuming Positive is > threshold
            # Robust check for Nans? Arrays should be numeric.
            pos_count = (vals > threshold).sum()
            pct = (pos_count / count * 100)

        results.append({
            'Sample': sample_name,
            count_col_name: int(pos_count),
            col_name: round(pct, 2)
        })

    return pd.DataFrame(results)

def get_recursive_mask(df, parent_name, defined_gates):
    """
    Recursively computes the boolean mask for a given parent population.

    Args:
        df: The DataFrame containing the data.
        parent_name: The name of the parent population (e.g., "Singlets", "GFP+").
        defined_gates: List of dicts describing existing gates.
                       [{'label': 'GFP+', 'channel': 'GFP-A', 'threshold': 100, 'parent': 'Singlets'}, ...]

    Returns:
        pd.Series (bool): The mask for the parent population.
    """
    # 0. Base Case: Singlets / Cells (Root)
    # We treat 'Singlets' or 'root' as the base filtered population (Live + Singlet)
    if parent_name in ['Singlets', 'root', 'None', None]:
        # Copied logic for base mask
        live_mask = pd.Series([True] * len(df), index=df.index)
        for cand in ['cells', 'Cells', 'Is_Live', 'AutoLiveGate']:
             if cand in df.columns:
                col = df[cand]
                if pd.api.types.is_bool_dtype(col):
                    live_mask = col
                else:
                    live_mask = (pd.to_numeric(col, errors='coerce') >= 0)
                break

        singlet_mask = pd.Series([True] * len(df), index=df.index)
        for cand in ['singlets', 'Singlets', 'Is_Singlet']:
            if cand in df.columns:
                col = df[cand]
                if pd.api.types.is_bool_dtype(col):
                    singlet_mask = col
                else:
                    singlet_mask = (pd.to_numeric(col, errors='coerce') >= 0)
                break
        return live_mask & singlet_mask

    # 1. Recursive Case: Find the gate definition
    target_gate = next((g for g in defined_gates if g.get('label') == parent_name), None)

    if not target_gate:
        # If unknown parent, fallback to base
        return get_recursive_mask(df, 'Singlets', [])

    # 2. Get the mask of the grandparent
    grandparent_mask = get_recursive_mask(df, target_gate.get('parent', 'Singlets'), defined_gates)

    # 3. Apply the current gate's threshold
    channel = target_gate['channel']

    # Handle Range vs Binary
    if target_gate.get('type') == 'range':
        min_val = target_gate.get('min', -np.inf)
        max_val = target_gate.get('max', np.inf)
        current_gate_mask = (df[channel] >= min_val) & (df[channel] <= max_val)
    else:
        # Backward compatibility or standard binary
        threshold = target_gate['threshold']
        if target_gate.get('direction') == 'negative':
            current_gate_mask = (df[channel] <= threshold)
        else:
            current_gate_mask = (df[channel] > threshold)

    return grandparent_mask & current_gate_mask

def calculate_positive_stats_recursive(experiment, gate, defined_gates):
    """
    Calculates stats for a specific gate, respecting its hierarchy.

    Args:
        experiment: Cytoflow Experiment
        gate: The gate dict to calculate stats for (must include 'label', 'channel', 'threshold', 'parent')
        defined_gates: List of all defined gates (context for recursion)
    """
    df = experiment.data
    results = []

    channel = gate['channel']
    # threshold = gate['threshold'] # access locally inside loop or use getter
    parent_name = gate.get('parent', 'Singlets')
    label = gate.get('label', f"{channel}+")

    # Grouping Logic
    group_cols = []
    try:
        if hasattr(experiment, 'conditions'): group_cols = list(experiment.conditions.keys())
    except: group_cols = []

    if not group_cols:
        possible_cols = ['Sample', 'sample', 'Name', 'name']
        for c in possible_cols:
            if c in df.columns: group_cols = [c]; break

    if group_cols: grouped = df.groupby(group_cols)
    else: grouped = [('All', df)]

    col_name = f"{label} %"
    count_col_name = f"{label} Count"

    for name, group in grouped:
        if isinstance(name, tuple): sample_name = "_".join(map(str, name))
        else: sample_name = str(name)

        # 1. Get the Parent Population Mask for this specific group chunk
        # Note: recursion usually runs on full DF, here we run on group.
        # It's fine as long as logic is row-independent.
        parent_mask = get_recursive_mask(group, parent_name, defined_gates)

        # 2. Filter
        parent_population = group[parent_mask]
        parent_count = len(parent_population)

        pos_count = 0
        pct = 0.0

        if parent_count > 0:
            # 3. Apply This Gate
            vals = parent_population[channel]

            if gate.get('type') == 'range':
                 min_val = gate.get('min', -np.inf)
                 max_val = gate.get('max', np.inf)
                 pos_count = ((vals >= min_val) & (vals <= max_val)).sum()
            else:
                 threshold = gate['threshold']
                 if gate.get('direction') == 'negative':
                     pos_count = (vals <= threshold).sum()
                 else:
                     pos_count = (vals > threshold).sum()

            pct = (pos_count / parent_count * 100) # Percentage of PARENT? Or Total?
            # Standard flow convention: usually % of Parent.
            # Let's stick to % of Parent as it's "Sub-gating".
            # If user wants % of Total, that's different. usually "% Parent" is default.

        results.append({
            'Sample': sample_name,
            count_col_name: int(pos_count),
            col_name: round(pct, 2)
        })

    return pd.DataFrame(results)

def get_channel_data_for_samples(experiment, channel, sample_names, parent_gate_name=None, defined_gates=None):
    """
    Retrieve data vectors for the specified channel, filtered by sample names AND parent gating.
    """
    df = experiment.data
    data_map = {}

    # ----------------------------------------------------
    # Apply Parent Gate Filtering Globally (Optimization?)
    # ----------------------------------------------------
    # It might be faster to compute mask once for full DF, then group.

    full_mask = pd.Series([True] * len(df), index=df.index)
    if parent_gate_name and parent_gate_name != 'Singlets':
        if defined_gates is None: defined_gates = []
        full_mask = get_recursive_mask(df, parent_gate_name, defined_gates)
    else:
        # Default match logic for Singlets
        full_mask = get_recursive_mask(df, 'Singlets', [])

    filtered_df = df[full_mask]

    # Grouping Logic reuse
    group_cols = []
    try:
        if hasattr(experiment, 'conditions'): group_cols = list(experiment.conditions.keys())
    except: pass
    if not group_cols:
        for c in ['Sample', 'sample', 'Name', 'name']:
            if c in df.columns: group_cols = [c]; break

    if group_cols: grouped = filtered_df.groupby(group_cols)
    else: grouped = [('All', filtered_df)]

    for name, group in grouped:
        if isinstance(name, tuple): s_name = "_".join(map(str, name))
        else: s_name = str(name)

        if s_name in sample_names:
            data_map[s_name] = group[channel].values

    return data_map
