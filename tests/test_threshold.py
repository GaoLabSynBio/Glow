import numpy as np

from glow.analysis.threshold import percent_in_gate, percentile_threshold


def test_percentile_threshold_ignores_nan():
    assert percentile_threshold([1, 2, 3, np.nan], 50) == 2.0
    assert percentile_threshold([np.nan], 99) is None


def test_percent_in_gate_binary_matches_positive_py_semantics():
    vals = [1, 2, 3, 4]
    assert percent_in_gate(vals, 2) == 50.0                      # > 2
    assert percent_in_gate(vals, 2, direction="negative") == 50.0  # <= 2


def test_percent_in_gate_range_inclusive_and_unordered():
    assert percent_in_gate([1, 2, 3, 4], (3, 2)) == 50.0


def test_percent_in_gate_empty():
    assert percent_in_gate([], 1) is None
