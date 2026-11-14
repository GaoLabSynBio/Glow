"""B1: high-res data is scaled once at load; exports/models see raw units."""
import numpy as np
import pandas as pd

from glow.utils import (detect_and_scale_high_res, raw_unit_factor, scale_experiment_once,
                        scale_high_res_df)


class Exp:
    def __init__(self, df, channels, metadata=None):
        self.data = df
        self.channels = channels
        self.metadata = metadata or {}


def _high_res():
    df = pd.DataFrame({
        "FSC-A": [1e6 + 5, 2e7, 5e3],
        "FL1-A": [10.0, 2e6, 3e4],
        "Replicate": [1, 2, 3],
        "cells": [True, False, True],
    })
    return Exp(df, ["FSC-A", "FL1-A"])


def test_scales_channels_once_and_is_idempotent():
    exp = _high_res()
    raw = exp.data.copy()
    assert scale_experiment_once(exp) is True
    assert exp.metadata["is_scaled"] and exp.metadata["is_high_res"]
    assert exp.metadata["scaling_factor"] == 10000.0
    assert exp.data["FSC-A"].max() < 1e6
    np.testing.assert_allclose(exp.data["FSC-A"], raw["FSC-A"] / 10000.0)
    # metadata-ish columns untouched
    assert exp.data["Replicate"].tolist() == [1, 2, 3]
    assert exp.data["cells"].tolist() == [True, False, True]
    once = exp.data.copy()
    assert scale_experiment_once(exp) is False
    assert detect_and_scale_high_res(exp) is False  # compensation fallback is a no-op
    pd.testing.assert_frame_equal(exp.data, once)


def test_low_res_data_untouched():
    exp = Exp(pd.DataFrame({"FSC-A": [1.0, 2e5]}), ["FSC-A"])
    assert scale_experiment_once(exp) is False
    assert "is_scaled" not in exp.metadata
    assert raw_unit_factor(exp) == 1.0


def test_export_round_trip_is_raw():
    exp = _high_res()
    raw = exp.data.copy()
    scale_experiment_once(exp)
    out = scale_high_res_df(exp.data.copy(), exp.channels,
                            exp.metadata["scaling_factor"], inverse=True)
    np.testing.assert_allclose(out[["FSC-A", "FL1-A"]], raw[["FSC-A", "FL1-A"]])
    assert raw_unit_factor(exp) == 10000.0


def test_csv_import_scaled_with_and_without_header():
    # Header says high-res (exported file): scaled on import even if values are small.
    exp = Exp(pd.DataFrame({"FSC-A": [5e3, 2e5]}), ["FSC-A"], {"is_high_res": True, "scaling_factor": 10000.0})
    assert scale_experiment_once(exp)
    assert exp.data["FSC-A"].tolist() == [0.5, 20.0]
    # No header, values > 1e6: treated like an FCS load.
    exp = Exp(pd.DataFrame({"FSC-A": [5e3, 2e7]}), ["FSC-A"])
    assert scale_experiment_once(exp)
    assert exp.data["FSC-A"].max() == 2000.0
    # Already scaled metadata: skipped.
    exp = Exp(pd.DataFrame({"FSC-A": [0.5, 20.0]}), ["FSC-A"], {"is_high_res": True, "is_scaled": True})
    assert not scale_experiment_once(exp)
    assert exp.data["FSC-A"].tolist() == [0.5, 20.0]
