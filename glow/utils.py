import numpy as np
import pandas as pd
import warnings

HIGH_RES_THRESHOLD = 1_000_000
HIGH_RES_FACTOR = 10000.0

# Columns that are metadata even when numeric; never rescaled.
_NON_CHANNEL_COLS = {
    'cells', 'singlets', 'sample', 'experiment_name', 'experiment name', 'replicate',
    'cell_line', 'condition', 'timepoint', 'concentration', 'dose', 'group',
}


def _scalable_columns(df, channels=None):
    """Numeric (non-bool, non-metadata) columns eligible for high-res scaling."""
    cols = [c for c in (channels if channels else df.columns) if c in df.columns]
    return [c for c in cols
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            and str(c).lower() not in _NON_CHANNEL_COLS and not str(c).startswith('_temp_')]


def is_high_res_data(df, channels=None, threshold=HIGH_RES_THRESHOLD):
    cols = _scalable_columns(df, channels)
    return bool(cols) and bool((df[cols].max() > threshold).any())


def scale_high_res_df(df, channels=None, factor=HIGH_RES_FACTOR, inverse=False):
    """Divide (or multiply when inverse=True) channel columns by `factor`, in place."""
    cols = _scalable_columns(df, channels)
    if cols:
        vals = df[cols].astype(float)
        df[cols] = vals * factor if inverse else vals / factor
    return df


def scale_experiment_once(experiment):
    """Scale a high-res experiment's channel columns at load time. Idempotent.

    Sets metadata is_high_res / is_scaled / scaling_factor. Returns True if scaling was applied.
    """
    if experiment is None or getattr(experiment, 'data', None) is None:
        return False
    meta = dict(getattr(experiment, 'metadata', None) or {})
    if meta.get('is_scaled', False):
        return False
    channels = list(getattr(experiment, 'channels', None) or [])
    if not (meta.get('is_high_res', False) or is_high_res_data(experiment.data, channels)):
        return False
    factor = float(meta.get('scaling_factor', HIGH_RES_FACTOR))
    scale_high_res_df(experiment.data, channels, factor)
    meta.update(is_high_res=True, is_scaled=True, scaling_factor=factor)
    experiment.metadata = meta
    return True


def raw_unit_factor(experiment):
    """Factor that converts this experiment's loaded values back to raw instrument units.

    1.0 unless the loader scaled the data. Model inputs (log10(x+1) features) must be computed
    in raw units to match training.
    """
    meta = getattr(experiment, 'metadata', None) or {}
    return float(meta.get('scaling_factor', HIGH_RES_FACTOR)) if meta.get('is_scaled') else 1.0


def detect_and_scale_high_res(experiment):
    """
    Detects if the experiment data appears to be from a high-resolution cytometer
    (e.g., BioRad ZE5 with 32-bit integers) and scales it down if necessary.

    This modifies the experiment in-place.

    Args:
        experiment: The experiment object containing .data (DataFrame) and .metadata (dict).

    Returns:
        bool: True if scaling was applied, False otherwise.
    """
    if experiment is None or experiment.data is None:
        return False

    # Idempotent fallback: the loader normally scales at load time (is_scaled=True).
    if experiment is None or experiment.data is None:
        return False
    return scale_experiment_once(experiment)


def logicle_transform(x, cofactor=150):
    """
    Logicle-like transform using hyperbolic arcsine (asinh).

    This is a common approximation for the logicle/biexponential transform used in
    flow cytometry. It handles negative values gracefully (important for compensated data)
    and transitions smoothly from linear near zero to log-like at larger values.

    The transform is: asinh(x / cofactor) / ln(10)

    This makes the scale comparable to log10 for large positive values while
    remaining well-defined for negative values.

    Args:
        x (array-like): Input data (can include negative values).
        cofactor (float): Controls the transition point from linear to log-like behavior.
                         Typical values: 150-500 for fluorescence, 5 for scatter.
                         Smaller cofactor = earlier transition to log-like.

    Returns:
        np.ndarray: Transformed data.
    """
    x = np.asarray(x, dtype=float)
    # asinh(x/c) / ln(10) makes scale comparable to log10 for large values
    return np.arcsinh(x / cofactor) / np.log(10)


def safe_log10(x, min_val=None, replace_with_nan=True):
    """
    Computes log10 safely by handling non-positive values.

    Args:
        x (array-like): Input data.
        min_val (float, optional): If provided, values <= 0 are clipped to this value before taking log.
                                   If None and replace_with_nan is True, values <= 0 become NaN.
        replace_with_nan (bool): If True and min_val is None, values <= 0 are replaced with NaN.

    Returns:
        np.ndarray: Log10 transformed data.
    """
    x = np.array(x, dtype=float)

    if min_val is not None:
        # Clip values to min_val
        x[x <= 0] = min_val
        return np.log10(x)

    if replace_with_nan:
        # Replace non-positive with NaN
        # Use where to avoid RuntimeWarning for invalid value in log10
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # We can't easily avoid the warning without pre-filtering,
            # but masking is cleaner.
            # Actually, `np.log10` emits warnings on NaN/inf generation usually.
            mask = x <= 0
            if np.any(mask):
                x_safe = x.copy()
                x_safe[mask] = np.nan
                return np.log10(x_safe) # log10(nan) is nan, no warning?
            else:
                return np.log10(x)

    # If neither, just do it and let it warn/return -inf/nan as numpy does
    return np.log10(x)

def safe_log(x, min_val=None, replace_with_nan=True):
    """
    Computes natural log safely by handling non-positive values.

    Args:
        x (array-like): Input data.
        min_val (float, optional): If provided, values <= 0 are clipped to this value before taking log.
        replace_with_nan (bool): If True and min_val is None, values <= 0 are replaced with NaN.

    Returns:
        np.ndarray: Natural log transformed data.
    """
    x = np.array(x, dtype=float)

    if min_val is not None:
        x[x <= 0] = min_val
        return np.log(x)

    if replace_with_nan:
        mask = x <= 0
        if np.any(mask):
            x_safe = x.copy()
            x_safe[mask] = np.nan
            return np.log(x_safe)

    return np.log(x)
