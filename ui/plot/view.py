import streamlit as st
import pandas as pd
import numpy as np
import copy
from glow.plotting import plots
from .base import render_data_selection
from .histogram_ui import render_histogram_ui
from .scatter_ui import render_scatter_ui
from .bar_ui import render_bar_ui
from .pairwise_ui import render_pairwise_ui

def render_plot_tab():
    st.header("Plotting")

    if 'analysis_experiment' in st.session_state and st.session_state.analysis_experiment is not None:
        experiment = st.session_state.analysis_experiment
    elif 'base_experiment' in st.session_state and st.session_state.base_experiment is not None:
        experiment = st.session_state.base_experiment
    else:
        st.warning("No experiment loaded. Please load data in the Gating tab first.")
        return
    if experiment is None:
        st.error("Experiment is None.")
        return

    # 1. Common Data Selection
    filtered_experiment, conditions = render_data_selection(experiment)
    if filtered_experiment is None:
        return # Selection failed

    # --- High Resolution Scaling Logic ---
    if hasattr(experiment, 'metadata') and experiment.metadata.get('is_high_res', False):
         st.warning("High-resolution data detected. Data logic is scaled by default (1/10000).")
         use_raw = st.toggle("Show Raw Values (Unscaled)", value=False, key="plot_show_raw_vals")
         
         if not use_raw:
             scaling_factor = experiment.metadata.get('scaling_factor', 10000.0)
             # Create a safe copy for modification
             if hasattr(filtered_experiment, 'clone'):
                 filtered_experiment = filtered_experiment.clone()
             else:
                 filtered_experiment = copy.copy(filtered_experiment)
                 # Ensure data is also a copy to avoid reference modification
                 filtered_experiment.data = filtered_experiment.data.copy()

             # Perform Scaling on Numeric Channels
             # EXCLUDE Flags and Time columns to prevent breaking gates (boolean logic) or time-series
             EXCLUDE_FROM_SCALING = [
                'cells', 'singlets', 'IsSinglet', 'Is_Singlet',
                'Population', 'AutoLiveGate', 'LiveGate', 'SingletGate',
                'Is_Live', 'Live', 'Density', 'Time', 'time'
             ]

             numeric_cols = []
             for c in filtered_experiment.data.columns:
                 # Check dtype
                 if pd.api.types.is_numeric_dtype(filtered_experiment.data[c]):
                     # Check exclusion list
                     if c in EXCLUDE_FROM_SCALING:
                         continue
                     # Check if it's strictly boolean (sometimes stored as bool dtype)
                     if pd.api.types.is_bool_dtype(filtered_experiment.data[c]):
                         continue
                     numeric_cols.append(c)

             if numeric_cols:
                 filtered_experiment.data[numeric_cols] = filtered_experiment.data[numeric_cols].astype(float) / scaling_factor

    # 2. Main Plot Type Selection
    all_available_channels = plots.get_channels(filtered_experiment)

    # Filter channels based on selection in Gating tab if available
    # Use persistent state to survive tab switch
    user_selected = st.session_state.get('persistent_selected_channels')

    # --- Compensation Logic (Plot Tab) ---
    all_cols = filtered_experiment.data.columns
    comp_cols = [c for c in all_cols if c.startswith("comp_")]
    has_comp = len(comp_cols) > 0
    use_comp = False

    if has_comp:
        use_comp = st.checkbox("Use Compensated Data", value=True, key="use_comp_plot")

    if user_selected is not None:
        raw_channels = [c for c in all_available_channels if c in user_selected]
    else:
        raw_channels = all_available_channels

    # Apply Substitution
    channels = []
    if use_comp:
        for c in raw_channels:
            comp_name = f"comp_{c}"
            if comp_name in all_cols:
                channels.append(comp_name)
            else:
                # Keep original if comp version doesn't exist (e.g. FSC/SSC)
                channels.append(c)
    else:
        # If not using comp, ensure we don't show comp_cols (if they leaked into all_available)
        channels = [c for c in raw_channels if not c.startswith("comp_")]

    with st.sidebar:
        st.subheader("Plot Settings")
        plot_type = st.selectbox(
            "Select Plot Type",
            ["1D Histogram", "2D Scatter", "Bar Plot", "Pairwise Plot"]
        )
        st.divider()

    # Exclude FSC/FC, SSC, and Metadata-like channels for Plotting
    if plot_type in ["1D Histogram", "2D Scatter", "Bar Plot", "Pairwise Plot"]:
        # Extended exclusion list based on user request (fluorophores only)
        # Excludes Scatter (FSC/SSC), Time, and metadata-like numeric columns (Replicate, etc.)
        exclude_patterns = ['fsc', 'fc', 'ssc', 'time', 'replicate', 'experiment', 'sample', 'well', 'plate']
        channels = [c for c in channels if not any(x in c.lower() for x in exclude_patterns)]

    # hue_var and facet_var will be handled inside the specific UI modules now
    hue_var = None
    facet_var = None

    # Scale is now handled inside individual UIs, default placeholder here
    scale = 'log' 

    # 4. Delegate to specific sub-modules
    if plot_type == "1D Histogram":
        render_histogram_ui(filtered_experiment, channels, hue_var, conditions, scale)
    elif plot_type == "2D Scatter":
        render_scatter_ui(filtered_experiment, channels, hue_var, facet_var, conditions, scale)
    elif plot_type == "Bar Plot":
        render_bar_ui(filtered_experiment, channels, hue_var, conditions)
    elif plot_type == "Pairwise Plot":
        render_pairwise_ui(filtered_experiment, channels, conditions, scale)

