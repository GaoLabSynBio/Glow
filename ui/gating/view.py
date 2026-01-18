import streamlit as st
import numpy as np
import pandas as pd
from .ui_file_loader import render_file_loader
from .ui_active_learning import get_active_learner
from .ui_gating import render_gating_section

def render_gating_tab(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE):
    # -------------------------------------------------------------------------
    # 0. Header & Developer Mode
    # -------------------------------------------------------------------------
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.header("Gating Workflow")
    with col_h2:
        # Developer mode toggle at the top right
        st.toggle("Developer Mode", key="developer_mode", value=False, help="Show advanced tools and validation plots.")

    # 1. File Loading Module
    render_file_loader(DataLoader, CYTOFLOW_AVAILABLE)

    # 2. Gating Module
    render_gating_section(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE)

