"""Threshold helpers for the 1D gate creator (quick-picks and live % in gate)."""
import numpy as np


def _finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def percentile_threshold(values, pct):
    """``pct``-th percentile of the finite values, or None when there are none."""
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, pct))


def percent_in_gate(values, threshold, direction="positive"):
    """Percentage of finite values inside a binary or range gate.

    ``threshold`` is a number (binary gate; ``direction`` 'positive' keeps values above it,
    'negative' keeps values at or below it) or a ``(min, max)`` pair (range gate, inclusive).
    Returns None when there are no finite values.
    """
    arr = _finite(values)
    if arr.size == 0:
        return None
    if isinstance(threshold, (tuple, list, np.ndarray)):
        lo, hi = sorted(float(t) for t in threshold[:2])
        mask = (arr >= lo) & (arr <= hi)
    elif direction == "negative":
        mask = arr <= float(threshold)
    else:
        mask = arr > float(threshold)
    return float(mask.mean() * 100.0)
