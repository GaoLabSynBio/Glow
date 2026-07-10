from glow.plotting.utils import adjust_pvalues, get_significance_label


def test_significance_labels():
    assert get_significance_label(0.0001) == "***"
    assert get_significance_label(0.005) == "**"
    assert get_significance_label(0.03) == "*"
    assert get_significance_label(0.5) == "ns"


def test_adjust_pvalues_none_is_identity():
    raw = [0.01, 0.2, 0.5]
    assert adjust_pvalues(raw, method="t-test") == raw


def test_adjust_pvalues_bonferroni_increases_pvalues():
    raw = [0.01, 0.04]
    adjusted = list(adjust_pvalues(raw, method="bonferroni"))
    assert all(a >= r for a, r in zip(adjusted, raw))


def test_adjust_pvalues_empty():
    assert adjust_pvalues([], method="bonferroni") == []
