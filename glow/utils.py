import numpy as np
import warnings


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

    # Check if already scaled
    if hasattr(experiment, 'metadata') and experiment.metadata.get('is_scaled', False):
        return False

    # Heuristic: Check numeric columns for very large values
    # Typical 18-bit flow data maxes around 262,144.
    # 32-bit signed int max is ~2.14e9.
    # If max > 1,000,000 (10^6), flag it.
    threshold = 1000000

    numeric_cols = experiment.data.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        return False

    max_vals = experiment.data[numeric_cols].max()
    is_high_res = (max_vals > threshold).any()

    # Check metadata tag if loader already flagged it but didn't scale
    if hasattr(experiment, 'metadata') and experiment.metadata.get('is_high_res', False):
        is_high_res = True

    if is_high_res:
        scaling_factor = 10000.0
        print(f"Detected high-resolution data. Applying in-place scaling (factor={scaling_factor}).")

        # Apply scaling to numeric columns (filtering out internal boolean/string cols)
        # Avoid scaling internal masks (usually boolean) or integer clusters if they are small
        # But here checking max > threshold protects against scaling small IDs.

        # Be careful not to scale channel columns that don't need it (e.g. Time?)
        # Usually Time is monotone increasing, might be large. Scaling time? Usually OK.

        # Also include other flow channels that might be low-signal but are part of the set (e.g. fluorophores).
        # It's better to scale ALL flow channels or just the ones triggered?
        # Usually, high-res applies to all detectors.
        # But we don't want to scale 'Time' if it's just seconds.
        # If Time is in ticks (large), scaling is fine.

        # Safer strategy: Scale ALL numeric columns if we trigger the mode,
        # except maybe known categorical ones (though numeric categorical is rare in raw flow).

        experiment.data[numeric_cols] = experiment.data[numeric_cols] / scaling_factor

        if experiment.metadata is None:
            experiment.metadata = {}

        experiment.metadata['is_high_res'] = True
        experiment.metadata['is_scaled'] = True
        experiment.metadata['scaling_factor'] = scaling_factor

        return True

    return False


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
