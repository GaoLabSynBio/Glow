import streamlit as st
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

    # Exclude FSC/FC and SSC channels for 1D, 2D, and Bar Plot
    if plot_type in ["1D Histogram", "2D Scatter", "Bar Plot", "Pairwise Plot"]:
        # User requested excluding fc and ssc channels. 'fc' likely implies FSC.
        channels = [c for c in channels if not any(x in c.lower() for x in ['fsc', 'fc', 'ssc'])]

    # hue_var and facet_var will be handled inside the specific UI modules now
    hue_var = None
    facet_var = None

    # 3. Global Scale Selector (Except for Bar Plot usually, but let's keep it common or pass it down)
    scale = 'log'
    if plot_type in ["1D Histogram", "2D Scatter", "Pairwise Plot"]:
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
    elif plot_type == "Pairwise Plot":
        render_pairwise_ui(filtered_experiment, channels, conditions, scale)

