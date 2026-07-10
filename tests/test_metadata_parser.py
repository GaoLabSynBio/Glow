import pandas as pd
import pytest

from glow.data_loading.metadata_parser import parse_metadata_file


def test_header_and_data_slice():
    df_raw = pd.DataFrame([
        ["Sample", "Cell Line", ""],
        ["a.fcs", "HeLa", "x"],
        ["b.fcs", "K562", "y"],
    ])
    out = parse_metadata_file(None, header_row_idx=0, data_start_idx=1, df_raw=df_raw)
    assert list(out.columns)[:2] == ["Sample", "Cell Line"]
    assert len(out) == 2
    # Blank header cell becomes an "Untitled_*" column.
    assert any("Untitled" in c for c in out.columns)


def test_duplicate_headers_disambiguated():
    df_raw = pd.DataFrame([
        ["Dose", "Dose"],
        [1, 2],
    ])
    out = parse_metadata_file(None, header_row_idx=0, data_start_idx=1, df_raw=df_raw)
    assert "Dose" in out.columns and "Dose_1" in out.columns


def test_header_out_of_bounds_raises():
    df_raw = pd.DataFrame([["a"]])
    with pytest.raises(ValueError):
        parse_metadata_file(None, header_row_idx=5, data_start_idx=6, df_raw=df_raw)
