import streamlit as st
from glow.plotting import plots
from .base import render_data_selection
from .histogram_ui import render_histogram_ui
from .scatter_ui import render_scatter_ui
from .bar_ui import render_bar_ui

def render_plot_tab():
    st.header("Plotting")
    
    if 'base_experiment' not in st.session_state or st.session_state.base_experiment is None:
        st.warning("No experiment loaded. Please load data in the Gating tab first.")
        return

    experiment = st.session_state.base_experiment
    if experiment is None:
        st.error("Experiment is None.")
        return

    # 1. Common Data Selection
    filtered_experiment, conditions = render_data_selection(experiment)
    if filtered_experiment is None: return # Selection failed

    # 2. Main Plot Type Selection
    channels = plots.get_channels(filtered_experiment)
    
    with st.sidebar:
        st.subheader("Plot Settings")
        plot_type = st.selectbox(
            "Select Plot Type",
            ["1D Histogram", "2D Scatter", "Bar Plot"]
        )
        st.divider()

    # hue_var and facet_var will be handled inside the specific UI modules now
    hue_var = None
    facet_var = None

    # 3. Global Scale Selector (Except for Bar Plot usually, but let's keep it common or pass it down)
    scale = 'log'
    if plot_type in ["1D Histogram", "2D Scatter"]:
        st.subheader("2. Plot Configuration")
        scale_opts = ["log", "linear", "logicle"]
        col_s, _ = st.columns([1,2])
        with col_s:
            scale = st.selectbox("Scale", scale_opts, index=0)
    
    # 4. Delegate to specific sub-modules
    if plot_type == "1D Histogram":
        render_histogram_ui(filtered_experiment, channels, hue_var, conditions, scale)
    elif plot_type == "2D Scatter":
        render_scatter_ui(filtered_experiment, channels, hue_var, facet_var, conditions, scale)
    elif plot_type == "Bar Plot":
        render_bar_ui(filtered_experiment, channels, hue_var, conditions)

