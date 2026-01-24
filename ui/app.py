import streamlit as st
import os
import sys

# Set page config must be the first streamlit command
st.set_page_config(page_title="Glow Flow Analysis", page_icon="assets/glow_icon.png", layout="wide")

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from glow.data_loading.loader import DataLoader
    from glow.gating.manager import GatingManager
    import cytoflow
    CYTOFLOW_AVAILABLE = True
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

# Import tab views
from gating.view import render_gating_tab
from analysis.view import render_analysis_tab
from plot.view import render_plot_tab
from canvas.view import render_canvas_tab
from compensation.view import render_compensation_tab
from gating.ui_file_loader import clear_experiment_state

col1, col2 = st.columns([1, 2.5], vertical_alignment="center")
with col1:
    st.image("assets/glow_icon_breath.gif")
with col2:
    st.markdown('<h1 style="padding-top: 0px; position: relative; top: 1px;">   Flow Analysis Pipeline</h1>', unsafe_allow_html=True)

# Initialize Session State for Workflow
if 'gating_complete' not in st.session_state:
    st.session_state.gating_complete = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Gating"
    
# Initial Cache Cleanup (Optimized Data Management)
if 'cleanup_done' not in st.session_state and DataLoader:
    # Clear active learning/temp data on fresh start
    DataLoader.clear_upload_cache()
    # Create necessary dirs if missing
    os.makedirs("data/uploaded_raw", exist_ok=True)
    st.session_state.cleanup_done = True


# Sidebar Navigation
st.sidebar.image("assets/glow_icon.png", width=80)
st.sidebar.title("Navigation")

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
selected_tab = st.sidebar.radio("Go to", tabs, index=tabs.index(st.session_state.current_tab))

# Handle tab switch
if selected_tab != st.session_state.current_tab:
    st.session_state.current_tab = selected_tab
    st.rerun()

st.sidebar.divider()

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
