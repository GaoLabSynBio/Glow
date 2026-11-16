"""Progress bar for long file operations, driven by the loader's progress callback."""
from contextlib import contextmanager

import streamlit as st


@contextmanager
def file_progress(text):
    """Show an ``st.progress`` bar; yields a ``callback(done, total, label)`` for the loader.

    The bar is removed when the block exits (also on errors).
    """
    bar = st.progress(0.0, text=text)

    def update(done, total, label=""):
        fraction = min(done / total, 1.0) if total else 0.0
        bar.progress(fraction, text=f"{label} ({done}/{total})" if total else (label or text))

    try:
        yield update
    finally:
        bar.empty()
