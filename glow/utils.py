import numpy as np
import warnings


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
