"""Save and restore a Glow analysis session.

Streamlit keeps everything in memory, so a browser refresh or an app restart
loses your place. This persists the *lightweight, JSON-serialisable* parts of the
session -- loaded file list, metadata, gate definitions, channel selections, plot
settings, current tab -- to a file the user can download and re-upload later.

Heavy or opaque objects (the in-memory cytoflow experiment, matplotlib figures,
cached models) are deliberately skipped: they are rebuilt from the restored file
list and metadata when the Gating tab re-runs. Using JSON (not pickle) also means
loading a session file can never execute arbitrary code.
"""
from __future__ import annotations

import json

import streamlit as st

from glow.logging_config import get_logger

log = get_logger("ui.session")

SESSION_VERSION = 1

# Keys that must never be serialised (large, opaque, or purely transient).
_DENYLIST_KEYS = {
    "base_experiment", "analysis_experiment", "gated_experiment",
    "cleanup_done", "startup_done",
}
_DENYLIST_PREFIXES = ("active_learner_",)


def _json_safe(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def serializable_state() -> dict:
    """Return the subset of session_state that can be safely written to JSON."""
    out = {}
    for key, value in st.session_state.items():
        if key in _DENYLIST_KEYS or any(key.startswith(p) for p in _DENYLIST_PREFIXES):
            continue
        if _json_safe(value):
            out[key] = value
    return out


def save_session_bytes() -> bytes:
    """Serialise the current session to JSON bytes for download."""
    payload = {"__glow_session__": SESSION_VERSION, "state": serializable_state()}
    return json.dumps(payload, indent=2).encode("utf-8")


def restore_session(raw_bytes) -> int:
    """Restore session keys from a previously saved session file.

    Returns the number of keys restored. Raises ValueError if the file is not a
    recognised Glow session file.
    """
    text = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else raw_bytes
    payload = json.loads(text)
    if not isinstance(payload, dict) or "__glow_session__" not in payload:
        raise ValueError("This file is not a Glow session file.")

    state = payload.get("state", {})
    for key, value in state.items():
        st.session_state[key] = value
    log.info("Restored %d session keys", len(state))
    return len(state)
