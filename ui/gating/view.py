import streamlit as st
import numpy as np
import pandas as pd
from .ui_file_loader import render_file_loader
from .ui_active_learning import get_active_learner
from .ui_gating import render_gating_section

def render_gating_tab(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE):
    st.header("Gating Workflow")

    # 1. File Loading Module
    render_file_loader(DataLoader, CYTOFLOW_AVAILABLE)

    # 2. Gating Module
    render_gating_section(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE)

