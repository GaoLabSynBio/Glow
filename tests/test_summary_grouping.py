import pandas as pd

from glow.analysis.stats import get_summary_statistics
from glow.data_loading.sample_metadata import label_map_from_data


class FakeExp:
    def __init__(self, data):
        self.data = data
        self.conditions = {c: "category" for c in ("Sample", "Experiment_Name", "Replicate", "cells")}


def _exp():
    df = pd.DataFrame({
        "Sample": ["D1", "D1", "D2", "D2"],
        "Experiment_Name": ["DKO", "DKO", "DKO", "DKO"],
        "Replicate": [1, 1, 2, 2],
        "cells": [True, False, True, True],
        "singlets": [True, True, True, False],
    })
    return FakeExp(df)


def test_summary_groups_by_sample_with_condition_columns():
    out = get_summary_statistics(_exp())
    assert out["Sample"].tolist() == ["D1", "D2"]  # no "_"-joined names
    assert out["Experiment_Name"].tolist() == ["DKO", "DKO"]
    assert out["Replicate"].tolist() == [1, 2]
    assert out["Live %"].tolist() == [50.0, 100.0]


def test_label_map_from_data():
    assert label_map_from_data(_exp().data) == {"D1": "DKO rep 1", "D2": "DKO rep 2"}
