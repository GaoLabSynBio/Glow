import pandas as pd
import numpy as np
from scipy import stats
from typing import Any, cast
import importlib.util
from glow.utils import safe_log10
try:
    import statsmodels.stats.multitest as mt
except ImportError:
    mt = None

# Try to import cytoflow, but don't fail if not present
if importlib.util.find_spec("cytoflow") is not None:
    CYTOFLOW_AVAILABLE = True
else:
    CYTOFLOW_AVAILABLE = False

def get_channels(experiment):
    """Returns a list of available channels in the experiment."""
    # Priority: explicit property -> data columns -> empty
    try:
        if hasattr(experiment, 'channels'):
             return sorted(experiment.channels)
    except Exception:
        pass # Fallback if cytoflow property crashes

    # Fallback if channels attribute is missing but data exists
    if hasattr(experiment, 'data'):
        # Filter for numeric columns
        return [c for c in experiment.data.columns if pd.api.types.is_numeric_dtype(experiment.data[c])]

    return []

def get_conditions(experiment):
    """Returns a list of available conditions (metadata columns) in the experiment."""
    try:
        if hasattr(experiment, 'conditions'):
            if isinstance(experiment.conditions, dict):
                return sorted(experiment.conditions.keys())
            return sorted(experiment.conditions) # if list
    except Exception:
        pass # Fallback if cytoflow property crashes

    # Fallback
    if hasattr(experiment, 'data'):
         return sorted([c for c in experiment.data.columns if not pd.api.types.is_numeric_dtype(experiment.data[c])])

    return []

def filter_experiment(experiment, condition, values):
    """
    Filters experiment to keep only events where 'condition' is in 'values'.
    Returns a new experiment-like object (or filters the dataframe).
    """
    data = experiment.data
    if condition not in data.columns:
        return experiment

    subset_df = data[data[condition].isin(values)].copy()

    # Return a new object of the same type if possible, or a basic wrapper
    # Using the same class if possible
    cls = type(experiment)
    try:
        # Check if constructor accepts just data, or try to mimic
        # Pass metadata if available to preserve channel/condition definitions
        if hasattr(experiment, 'metadata'):
            return cls(subset_df, metadata=experiment.metadata)
        return cls(subset_df)
    except Exception:
        # Fallback to a simple container
        class SimpleExp:
            def __init__(self, d, orig=None):
                self.data = d
                self._orig = orig

            @property
            def channels(self):
                # Try to inherit from original experiment to ensure consistency
                if self._orig and hasattr(self._orig, 'channels'):
                    return [c for c in self._orig.channels if c in self.data.columns]

                # Robust fallback: Numeric but NOT Boolean (excludes flags like 'cells')
                # Also exclude common metadata names explicitly in case they are numeric (e.g. Replicates)
                excludes = ['cells', 'singlets', 'Is_Live', 'Is_Singlet', 'LiveGate', 'SingletGate',
                            'Replicate', 'replicate', 'Experiment_Name', 'Experiment Name', 'Time', 'time',
                            'Concentration', 'Timepoint', 'Group', 'group']

                return [c for c in self.data.columns if pd.api.types.is_numeric_dtype(self.data[c])
                        and not pd.api.types.is_bool_dtype(self.data[c])
                        and c not in excludes]

            @property
            def conditions(self):
                # Try to inherit
                if self._orig and hasattr(self._orig, 'conditions'):
                    return self._orig.conditions

                return {c: "category" for c in self.data.columns if c not in self.channels}

            def clone(self):
                return SimpleExp(self.data.copy(), orig=self._orig)

        return SimpleExp(subset_df, orig=experiment)

def get_point_density(x, y, bins=100, log_x=False, log_y=False):
    """
    Calculates density for scatter plots using 2D histogram lookup.
    """
    # Filter NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    # We use safe checking via safe_log10 implicitly if we wanted, but to keep logic similar:
    # If we use safe_log10, it returns NaNs for <= 0.
    # But we want to filter them out for histogram2d.

    # Original logic explicitly removed delicate values.
    # Let's keep the explicit filter as it's efficient for histogram2d which needs clean data.
    if log_x:
        mask &= (x > 0)
    if log_y:
        mask &= (y > 0)

    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.array([]), mask

    # Transform to log space for density calculation if needed
    # (Matches visual density on log-scaled plots)
    # Using safe_log10 adds an extra layer of safety even if filtered
    tx = safe_log10(x) if log_x else x
    ty = safe_log10(y) if log_y else y

    # Histogram
    hh, locx, locy = np.histogram2d(tx, ty, bins=bins)

    # Compute density for each point
    # Digitizing coordinates
    # np.searchsorted gives index to insert.
    # Bin i is [edges[i], edges[i+1]).
    # searchsorted returns i if x < edges[i].
    # We want bin index.

    ix = np.searchsorted(locx, tx) - 1
    iy = np.searchsorted(locy, ty) - 1

    # Clip indices
    ix = np.clip(ix, 0, bins-1)
    iy = np.clip(iy, 0, bins-1)

    # Lookup density
    z = hh[ix, iy]

    # Log normalize
    z = np.log1p(z)

    # Normalize 0..1
    if z.max() > z.min():
        z = (z - z.min()) / (z.max() - z.min())
    else:
        z = np.zeros_like(z)

    return z, mask

def calculate_gated_mean(df, gate_channel, value_channel, threshold, statistic='mean'):
    """
    Calculates the mean of value_channel for events where gate_channel > threshold.
    If threshold is a tuple/list of length 2, it is treated as (min, max).
    If threshold is single value, it is treated as lower bound (val > threshold).
    """
    if isinstance(threshold, (tuple, list)) and len(threshold) == 2:
        low, high = threshold
        gated_df = df[(df[gate_channel] >= low) & (df[gate_channel] <= high)]
    else:
        gated_df = df[df[gate_channel] > threshold]

    if len(gated_df) == 0:
        return np.nan

    vals = gated_df[value_channel]

    if statistic == 'mean':
        return vals.mean()
    elif statistic == 'median':
        return vals.median()
    elif statistic == 'geom_mean':
        # Geometric mean (ignore <= 0)
        pos_vals = vals[vals > 0]
        if len(pos_vals) == 0:
            return np.nan
        return np.exp(np.log(pos_vals).mean())

    return np.nan

def get_percentile_threshold(df, channel, percentile=99.9):
    """
    Returns the value at the given percentile for the channel.
    percentile should be 0-100.
    """
    if channel not in df.columns:
        return 0
    return np.percentile(df[channel].dropna(), percentile)

def get_percentile_of_value(df, channel, value):
    """
    Returns the percentile (0-100) of a given raw value in the distribution of the channel.
    """
    from scipy import stats
    if channel not in df.columns:
        return 0
    vals = df[channel].dropna()
    if len(vals) == 0:
        return 0
    # percentileofscore returns 0-100
    try:
        return stats.percentileofscore(vals, value)
    except Exception:
        return 0

def calculate_significance(data1, data2):
    """
    Calculates independent t-test between two arrays.
    Returns (p_value, star_label, fold_change).
    Fold change is mean(data2) / mean(data1).
    """
    d1 = np.array(data1)
    d2 = np.array(data2)
    # Remove NaNs
    d1 = d1[np.isfinite(d1)]
    d2 = d2[np.isfinite(d2)]

    if len(d1) < 2 or len(d2) < 2:
        return 1.0, "ns", 1.0

    # Calculate Fold Change (Mean based)
    m1 = np.mean(d1)
    m2 = np.mean(d2)
    fc = m2 / m1 if m1 != 0 else 0

    try:
        res = stats.ttest_ind(d1, d2, equal_var=False)
        p = float(cast(Any, res)[1])
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"
        return p, sig, fc
    except Exception:
        return 1.0, "ns", fc

def get_significance_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def adjust_pvalues(p_values, method='t-test'):
    """
    Adjusts a list of p-values for multiple testing.
    Methods: 't-test' (none), 'bonferroni', 'fdr'
    Returns adjusted p-values.
    """
    if not p_values:
        return []

    if method in ['t-test', 'none'] or mt is None:
        return p_values

    method_map = {
        'bonferroni': 'bonferroni',
        'fdr': 'fdr_bh', # False Discovery Rate (Benjamini-Hochberg)
        'fdr_bh': 'fdr_bh'
    }

    m = method_map.get(method.lower(), 'fdr_bh')

    try:
        _, pvals_corrected, _, _ = mt.multipletests(p_values, method=m)
        return pvals_corrected
    except Exception as e:
        print(f"Error in p-value adjustment: {e}")
        return p_values
