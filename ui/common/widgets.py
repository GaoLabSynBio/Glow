"""Small shared Streamlit widgets used across the plotting tabs."""
from __future__ import annotations

import io
import warnings

import streamlit as st


def download_svg_button(fig, key, label="Download SVG"):
    """Render a button that downloads a matplotlib figure as an SVG."""
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="This figure includes Axes that are not compatible with tight_layout",
        )
        fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=f"{key}.svg",
        mime="image/svg+xml",
        key=f"dl_{key}",
    )
