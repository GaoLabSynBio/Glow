"""User-facing error reporting for the Streamlit UI.

Instead of dumping a raw exception with ``st.error(f"Error: {e}")``, render a
plain-language summary, the likely cause, a concrete fix, and an expandable
"technical details" section with the full traceback for bug reports. Everything
is also written to the log file.
"""
from __future__ import annotations

import traceback
from contextlib import contextmanager

import streamlit as st

from glow.logging_config import get_logger

log = get_logger("ui")


def show_error(summary, cause=None, fix=None, exc=None):
    """Render a structured error message and log it."""
    st.error(f"**{summary}**")
    if cause:
        st.caption(f"Likely cause: {cause}")
    if fix:
        st.info(f"How to fix: {fix}")
    if exc is not None:
        log.exception(summary)
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        with st.expander("Technical details (for bug reports)"):
            st.code(details)
    else:
        log.error(summary)


@contextmanager
def friendly_error(summary, cause=None, fix=None):
    """Context manager that turns any exception into a structured error message.

    Usage::

        with friendly_error("Could not load the metadata file",
                            cause="the header row may be wrong",
                            fix="check the Header Row Index setting"):
            df = pd.read_csv(...)
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 -- deliberately broad at the UI boundary
        show_error(summary, cause=cause, fix=fix, exc=exc)
