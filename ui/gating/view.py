import streamlit as st
from .ui_file_loader import render_file_loader
from .ui_gating import render_gating_section

def render_gating_tab(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE):
    st.header("Gating Workflow")
    
    # 1. File Loading Module
    render_file_loader(DataLoader, CYTOFLOW_AVAILABLE)

    # 2. Gating Module
    render_gating_section(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE)
    
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Finish Gating & Proceed to Analysis", type="primary"):
            # Check if critical data exists
            if 'base_experiment' in st.session_state and st.session_state.base_experiment is not None:
                # You might want to add more specific checks here (e.g. singlets gated)
                st.session_state.gating_complete = True
                st.session_state.current_tab = "Analysis"
                st.rerun()
            else:
                st.error("Please load data and perform gating before proceeding.")
