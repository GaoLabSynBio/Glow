import pandas as pd

from glow.analysis.stats import get_summary_statistics, group_by_experiment, order_summary_columns
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


def test_order_summary_columns_puts_experiment_first_and_sample_after_conditions():
    df = pd.DataFrame({"Sample": ["a"], "Cell_Line": ["X"], "Replicate": [1],
                       "Experiment_Name": ["E"], "Live %": [1.0], "CD4+ %": [2.0]})
    out = order_summary_columns(df, ["Cell_Line", "Replicate", "Experiment_Name"])
    assert list(out.columns) == ["Experiment_Name", "Replicate", "Cell_Line", "Sample",
                                 "Live %", "CD4+ %"]


def test_group_by_experiment_mean_sd_across_replicates():
    df = pd.DataFrame({
        "Experiment_Name": ["WT", "WT", "KO"],
        "Replicate": [1, 2, 1],
        "Cell_Line": ["X", "X", "X"],
        "Sample": ["w1", "w2", "k1"],
        "Live %": [80.0, 90.0, 50.0],
    })
    out = group_by_experiment(df, ["Experiment_Name", "Replicate", "Cell_Line"])
    assert out["Experiment_Name"].tolist() == ["WT", "KO"]
    assert out["Replicates"].tolist() == [2, 1]
    assert out["Cell_Line"].tolist() == ["X", "X"]
    assert out["Live % mean"].tolist() == [85.0, 50.0]
    assert out.loc[0, "Live % SD"] == round(50 ** 0.5, 2)
    assert pd.isna(out.loc[1, "Live % SD"])
    assert "Sample" not in out.columns


def test_condition_columns_tolerate_data_column_without_metadata():
    from glow.analysis.grouping import condition_columns

    class MetaExp:
        metadata = {"Sample": {"type": "condition"}, "Experiment_Name": {"type": "condition"},
                    "EGFP-A": {"type": "channel"}}

        @property
        def conditions(self):
            raise KeyError("_temp_live_labels")

    df = pd.DataFrame({"Sample": ["a"], "Experiment_Name": ["E"], "_temp_live_labels": [0]})
    assert condition_columns(MetaExp(), df, ["Sample"]) == ["Experiment_Name"]


def test_group_by_experiment_without_names_is_identity():
    df = pd.DataFrame({"Sample": ["a"], "Live %": [1.0]})
    assert group_by_experiment(df, []) is df
