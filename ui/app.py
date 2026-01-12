import streamlit as st
import os
import sys

# Set page config must be the first streamlit command
st.set_page_config(page_title="Glow Flow Analysis", layout="wide", page_icon="assets/xjgao.jpg")

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

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    img_b64 = get_base64_image("assets/xjgao.jpg")
except Exception:
    img_b64 = ""

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@800&display=swap');

    .glow-header {{
        display: flex;
        align-items: center;
        gap: 20px;
        padding-bottom: 20px;
        border-bottom: 1px solid #333;
        margin-bottom: 30px;
    }}
    
    @keyframes move-gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    .shiny-text {{
        font-family: 'Montserrat', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(
            to right,
            #BF953F,
            #FCF6BA,
            #B38728,
            #FBF5B7,
            #AA771C
        );
        background-size: 200% auto;
        animation: move-gradient 4s linear infinite;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-shadow: 0px 0px 10px rgba(191, 149, 63, 0.4);
        letter-spacing: -2px;
    }}
    
    .logo-container {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        overflow: hidden;
        border: 4px solid transparent;
        background: linear-gradient(
            to right,
            #BF953F,
            #FCF6BA,
            #B38728,
            #FBF5B7,
            #AA771C
        ) border-box;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
    }}
    
    .logo-img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
</style>

<div class="glow-header">
    <div class="logo-container">
        <img src="data:image/jpeg;base64,{img_b64}" class="logo-img">
    </div>
    <div class="shiny-text">GLOW</div>
</div>
""", unsafe_allow_html=True)

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
