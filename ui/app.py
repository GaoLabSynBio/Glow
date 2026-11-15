import os
import sys

# Make the repo root importable so ``import glow`` works regardless of the
# current working directory (e.g. when launched from a packaged app bundle).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from glow import paths as glow_paths

# Set page config must be the first streamlit command
st.set_page_config(
    page_title="Glow Flow Analysis",
    page_icon=str(glow_paths.asset_path("glow_icon.png")),
    layout="wide",
)

# Import tab views
from gating.view import render_gating_tab, clear_all_cache
from analysis.view import render_analysis_tab
from plot.view import render_plot_tab
from canvas.view import render_canvas_tab
from compensation.view import render_compensation_tab
from common import session as glow_session

try:
    from glow.data_loading.loader import DataLoader
    from glow.gating.manager import GatingManager
    import importlib.util
    CYTOFLOW_AVAILABLE = importlib.util.find_spec("cytoflow") is not None
except ImportError as e:
    st.error(f"Cytoflow or pipeline modules not found: {e}. Running in Mock Mode.")
    CYTOFLOW_AVAILABLE = False
    DataLoader = None
    GatingManager = None

# Try to import ActiveLearningLogger, define safely if fails
try:
    from glow.active_learning.storage import ActiveLearningLogger
except ImportError:
    ActiveLearningLogger = None

# Initialize Session State for Workflow
if 'gating_complete' not in st.session_state:
    st.session_state.gating_complete = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_tab' not in st.session_state:
    # Restore the tab from the URL so a browser refresh keeps your place.
    st.session_state.current_tab = st.query_params.get("tab", "Gating")

# Ensure writable data directories exist. We intentionally do NOT clear the upload
# cache here: keeping the files lets the user refresh or come back later and pick
# up where they left off. Use "Start new experiment" (sidebar) to clear.
if 'startup_done' not in st.session_state:
    glow_paths.uploads_dir()
    st.session_state.startup_done = True

# Sidebar Navigation
st.sidebar.image(str(glow_paths.asset_path("glow_icon.png")), width=64)

# Determine available tabs based on state
# User request: Always show Analysis, Plot, and Canvas
tabs = ["Gating", "Analysis", "Plot", "Canvas"]

# Conditional Compensation Tab
if st.session_state.get('run_compensation', False) or st.session_state.current_tab == "Compensation":
    if "Compensation" not in tabs:
        tabs.insert(1, "Compensation")

# Ensure current tab is valid
if st.session_state.current_tab not in tabs:
    st.session_state.current_tab = "Gating" # Default fallback

# Use a key to sync state, but handle manual changes via callback if needed.
# Simple approach: use the radio value.
_has_data = "base_experiment" in st.session_state
_gated = bool(st.session_state.get("gating_complete"))
_done = {"Gating": _gated, "Analysis": bool(st.session_state.get("analysis_complete"))}
_reason = {}
if not _has_data:
    _reason["Analysis"] = _reason["Plot"] = _reason["Canvas"] = "load files first"
elif not _gated:
    _reason["Analysis"] = "finish gating first"


def _step_label(tab):
    if _done.get(tab):
        return f"{tab} ✓"
    return f"{tab} (unavailable)" if tab in _reason else tab


selected_tab = st.sidebar.radio(
    "Go to", tabs, index=tabs.index(st.session_state.current_tab),
    format_func=_step_label,
)
if selected_tab in _reason:
    st.sidebar.caption(f"{selected_tab}: {_reason[selected_tab]}.")

# Keep the URL in sync so a refresh restores the current tab.
if st.query_params.get("tab") != selected_tab:
    st.query_params["tab"] = selected_tab

# Handle tab switch
if selected_tab != st.session_state.current_tab:
    st.session_state.current_tab = selected_tab
    st.rerun()

st.sidebar.divider()

# --- Session Save / Restore ------------------------------------------------
with st.sidebar.expander("💾 Session", expanded=False):
    st.caption("Save your progress to a file and restore it later.")
    st.download_button(
        "Save session",
        data=glow_session.save_session_bytes(),
        file_name="glow_session.json",
        mime="application/json",
        help="Downloads your metadata, gate settings and selections (not the raw data).",
        width="stretch",
    )
    restore_file = st.file_uploader("Restore session", type=["json"], key="restore_session_file")
    if restore_file is not None and st.button("Load this session", width="stretch"):
        try:
            n = glow_session.restore_session(restore_file.getvalue())
            st.success(f"Restored {n} settings. Reloading…")
            st.rerun()
        except Exception as e:
            st.error(f"Could not restore session: {e}")

st.sidebar.divider()

# --- Advanced + new experiment (bottom of sidebar) -------------------------
with st.sidebar.expander("Advanced", expanded=False):
    st.toggle("Developer Mode", key="developer_mode", value=False,
              help="Show advanced tools and validation plots.")

with st.sidebar.popover("🔄 Start new experiment", width="stretch"):
    st.warning("This clears all loaded data, gates and settings.")
    if st.button("Yes, clear everything", key="refresh_experiment", type="primary"):
        clear_all_cache()
        st.rerun()

# Compact one-line header once data is loaded; full banner on the landing page.
if _has_data:
    st.markdown("##### Glow · Flow Analysis Pipeline")
else:
    _c1, _c2 = st.columns([1, 2.5], vertical_alignment="center")
    with _c1:
        st.image(str(glow_paths.asset_path("glow_icon_breath.svg")), width=500)
    with _c2:
        st.markdown('<h1 style="padding-top: 0px; position: relative; top: 1px;">   Flow Analysis Pipeline</h1>', unsafe_allow_html=True)

# Render Selected Tab
if selected_tab == "Gating":
    # Render the gating tab content
    render_gating_tab(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE)

elif selected_tab == "Compensation":
    render_compensation_tab()

elif selected_tab == "Analysis":
    render_analysis_tab()

elif selected_tab == "Plot":
    render_plot_tab()

elif selected_tab == "Canvas":
    render_canvas_tab()
