import pandas as pd

from glow.data_loading.sample_metadata import (
    apply_sample_metadata, build_sample_metadata, default_experiment_names,
    infer_replicates, sample_label_map)


class FakeExp:
    def __init__(self, data):
        self.data = data
        self.metadata = {"Sample": {"type": "condition"}}


def _preview():
    return [
        {"original_path": "/x/a/D1.fcs", "display_name": "a_D1.fcs", "condition": "a", "Cell_Line": ""},
        {"original_path": "/x/a/D2.fcs", "display_name": "a_D2.fcs", "condition": "a", "Cell_Line": "DKO"},
        {"original_path": "/x/a/E1.fcs", "display_name": "a_E1.fcs", "condition": "a"},
    ]


def test_infer_replicates_integers():
    assert infer_replicates(["A", "B", "A", "A"]).tolist() == [1, 1, 2, 3]


def test_build_defaults():
    meta = build_sample_metadata(_preview())
    assert meta.index.tolist() == ["D1", "D2", "E1"]
    assert meta["Experiment_Name"].tolist() == ["D1", "D2", "E1"]
    assert meta["Replicate"].tolist() == [1, 1, 1]
    assert meta["Cell_Line"].tolist() == ["", "DKO", ""]  # never HEK
    assert "original_path" not in meta.columns


def test_stem_collision_prefixes_folder():
    pm = [{"original_path": "/x/a/D1.fcs"}, {"original_path": "/x/b/D1.fcs"}, {"original_path": "/x/b/E1.fcs"}]
    assert build_sample_metadata(pm).index.tolist() == ["a_D1", "b_D1", "E1"]


def test_default_names_and_labels():
    meta = build_sample_metadata(_preview())
    meta["Experiment_Name"] = ["DKO", "DKO", ""]
    meta = default_experiment_names(meta)
    assert meta["Experiment_Name"].tolist() == ["DKO", "DKO", "E1"]
    meta["Replicate"] = infer_replicates(meta["Experiment_Name"]).to_numpy()
    assert sample_label_map(meta) == {"D1": "DKO rep 1", "D2": "DKO rep 2", "E1": "E1"}


def test_apply_sample_metadata_relabels_only():
    meta = build_sample_metadata(_preview())
    exp = FakeExp(pd.DataFrame({"Sample": ["D1", "D1", "D2", "E1"], "FSC-A": [1.0, 2.0, 3.0, 4.0],
                                "cells": [True, False, True, True], "Old": ["x"] * 4}))
    apply_sample_metadata(exp, meta, drop=["Old"])
    assert exp.data["Experiment_Name"].astype(str).tolist() == ["D1", "D1", "D2", "E1"]
    assert exp.data["Replicate"].tolist() == [1, 1, 1, 1]
    assert exp.data["cells"].tolist() == [True, False, True, True]
    assert "Old" not in exp.data.columns and "Old" not in exp.metadata
    assert exp.metadata["Cell_Line"]["type"] == "condition"
