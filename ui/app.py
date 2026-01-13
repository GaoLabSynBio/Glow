import streamlit as st
import os
import sys

# Set page config must be the first streamlit command
st.set_page_config(page_title="Glow Flow Analysis", layout="wide")

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
from gating.ui_file_loader import clear_experiment_state

st.title("Glow Flow Analysis Pipeline")

# Initialize Session State for Workflow
if 'gating_complete' not in st.session_state:
    st.session_state.gating_complete = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Gating"

# Sidebar Navigation
st.sidebar.title("Navigation")

# Determine available tabs based on state
tabs = ["Gating"]
if st.session_state.gating_complete:
    tabs.append("Analysis")
# if st.session_state.analysis_complete: # Bypassed for debugging
tabs.append("Plot")
tabs.append("Canvas")

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
    # Logic for read-only view if gating is complete
    if st.session_state.gating_complete:
        st.warning("Gating is marked as complete. To modify gating, please unlock it below. Unlocking will reset Analysis progress.")
        if st.button("Unlock Gating (Reset Analysis)"):
            st.session_state.gating_complete = False
            st.session_state.analysis_complete = False
            clear_experiment_state() # Ensure fresh state when unlocking
            st.rerun()
            
    # Render the gating tab content
    render_gating_tab(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE)

elif selected_tab == "Analysis":
    render_analysis_tab()

elif selected_tab == "Plot":
    render_plot_tab()

elif selected_tab == "Canvas":
    render_canvas_tab()
