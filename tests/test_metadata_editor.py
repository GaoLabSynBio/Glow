import pandas as pd
import streamlit as st

from ui.common import metadata_editor as me


class FakeExp:
    def __init__(self, data):
        self.data = data
        self.metadata = {}


def _setup(monkeypatch):
    state = {}
    monkeypatch.setattr(st, "session_state", type("S", (dict,), {
        "__getattr__": lambda s, k: s[k], "__setattr__": lambda s, k, v: s.__setitem__(k, v)})(state))
    df = pd.DataFrame({"Sample": ["a", "b", "c", "d"], "Experiment_Name": ["a", "b", "c", "d"],
                       "Cell_Line": ["HEK", "HEK", "DKO", "DKO"], "Replicate": [1, 1, 1, 1]})
    st.session_state["base_experiment"] = FakeExp(df.copy())
    st.session_state["gated_experiment"] = FakeExp(df.iloc[:3].copy())
    st.session_state["cell_line_gates"] = {"HEK": {"g": 1}, "DKO": {"g": 2}}
    return df


def test_rename_and_relabel_keeps_gates(monkeypatch):
    _setup(monkeypatch)
    old = me.ensure_sample_metadata()
    new = old.copy()
    new["Experiment_Name"] = ["x", "x", "c", "d"]
    new["Cell_Line"] = ["WT", "WT", "DKO", "DKO"]
    renames, affected = me.save_sample_metadata(old, new)
    assert renames == {"HEK": "WT"} and not affected
    assert set(st.session_state["cell_line_gates"]) == {"WT", "DKO"}
    for key in ("base_experiment", "gated_experiment"):
        d = st.session_state[key].data
        assert d.loc[d.Sample == "b", "Replicate"].iloc[0] == 2
        assert d.loc[d.Sample == "a", "Cell_Line"].astype(str).iloc[0] == "WT"
    assert "lines_need_regate" not in st.session_state


def test_reassign_flags_only_affected_lines(monkeypatch):
    _setup(monkeypatch)
    old = me.ensure_sample_metadata()
    new = old.copy()
    new.loc["b", "Cell_Line"] = "DKO"
    renames, affected = me.save_sample_metadata(old, new)
    assert not renames and affected == {"HEK", "DKO"}
    assert st.session_state["lines_need_regate"] == {"HEK", "DKO"}
