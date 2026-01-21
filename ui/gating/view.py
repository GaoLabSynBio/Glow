import streamlit as st
from .ui_file_loader import render_file_loader
from .ui_gating import render_gating_section

def clear_all_cache():
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Clear Streamlit caches
    st.cache_data.clear()
    st.cache_resource.clear()
    st.toast("Experiment State and Cache Cleared.", icon="🔄")

def render_gating_tab(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE):
    # -------------------------------------------------------------------------
    # 0. Header & Developer Mode
    # -------------------------------------------------------------------------
    if st.button('🔄 Start New Experiment (Reset everything)', key='refresh_experiment', help='Clear all cache and start a new experiment.'):
        clear_all_cache()
        st.rerun()

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

