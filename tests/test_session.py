import json
import types

import pytest

from common import session as S


def _fake_state(state):
    """Replace the module's streamlit reference with a fake holding a plain dict."""
    S.st = types.SimpleNamespace(session_state=state)


def test_roundtrip_skips_opaque_objects():
    _fake_state({
        "current_tab": "Plot",
        "loaded_files": [{"display_name": "a.fcs"}],
        "cell_line_gates": {"HeLa": [[1.0, 2.0], [3.0, 4.0]]},
        "base_experiment": object(),          # opaque -> skipped
        "active_learner_gating": object(),    # denylisted prefix -> skipped
    })
    payload = json.loads(S.save_session_bytes())
    assert payload["__glow_session__"] == S.SESSION_VERSION
    assert payload["state"]["current_tab"] == "Plot"
    assert "base_experiment" not in payload["state"]
    assert "active_learner_gating" not in payload["state"]

    fresh = {}
    _fake_state(fresh)
    n = S.restore_session(json.dumps(payload).encode("utf-8"))
    assert fresh["current_tab"] == "Plot"
    assert fresh["cell_line_gates"]["HeLa"] == [[1.0, 2.0], [3.0, 4.0]]
    assert n == 3


def test_rejects_non_session_file():
    _fake_state({})
    with pytest.raises(ValueError):
        S.restore_session(b'{"foo": 1}')
