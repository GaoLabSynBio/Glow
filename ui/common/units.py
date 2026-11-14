"""One shared note about the unit convention for high-resolution data."""
import streamlit as st


def render_units_caption(experiment):
    """Show a single caption when the loader scaled the data (B1/B2)."""
    # Derived (gated/analysis) experiments may drop custom metadata; the base experiment is authoritative.
    meta = getattr(experiment, 'metadata', None) or {}
    if not meta.get('is_scaled'):
        meta = getattr(st.session_state.get('base_experiment'), 'metadata', None) or {}
    if meta.get('is_scaled'):
        factor = float(meta.get('scaling_factor', 10000.0))
        st.caption(f"Values shown ÷{factor:,.0f}; exports are raw.")
