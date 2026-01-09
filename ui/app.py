import streamlit as st
import os
import sys

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from glow.base_pipeline.data_loader import DataLoader
    from glow.base_pipeline.gating import GatingManager
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

# Import local modules
# We need to make sure python can find the modules directory.
# Since app.py is in ui/, and modules is in ui/modules, 
# and streamlit runs relative to the script usually.
from modules.ui_file_loader import render_file_loader
from modules.ui_gating import render_gating_section

st.title("Glow Flow Analysis Pipeline")

st.sidebar.header("Configuration")

# 1. File Loading Module
render_file_loader(DataLoader, CYTOFLOW_AVAILABLE)

# 2. Gating Module
render_gating_section(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE)

st.info("Active Learning Module to be integrated here.")
