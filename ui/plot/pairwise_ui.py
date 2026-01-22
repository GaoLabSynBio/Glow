import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal, cast
from glow.canvas.functions import add_plot_to_canvas

def render_pairwise_ui(experiment, channels, conditions, scale='log'):
    """
    Renders the UI for Pairwise Plots (Seaborn pairplot).
    """
    st.markdown("### Pairwise Plot Configuration")

    # 1. Settings
    col1, col2 = st.columns(2)

    with col1:
        # Channel Selection
        # Default to first few channels if available
        default_channels = channels[:4] if len(channels) >= 4 else channels
        selected_channels = st.multiselect(
            "Select Channels to Compare",
            channels,
            default=default_channels,
            help="Select at least 2 channels."
        )

    with col2:
        # Hue Selection
        hue_options = ["None"] + conditions
        # Add Sample if not present
        if "Sample" in experiment.data.columns and "Sample" not in hue_options:
            hue_options.append("Sample")

        selected_hue = st.selectbox("Select Hue (Group/Color)", hue_options, index=0)

    # 2. Advanced / Performance
    with st.expander("Advanced Settings & Performance", expanded=True):
        max_events = len(experiment.data)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            # Downsampling
            sample_size = st.number_input(
                "Sample Size (Events)",
                min_value=50,
                max_value=max_events,
                value=min(2000, max_events),
                step=100,
                help="Smaller sample size is faster. < 2000 recommended for KDE pairplots."
            )

        with s_col2:
            kind = st.selectbox("Plot Kind", ["kde", "scatter", "hist", "reg"], index=0, help="Type of plot for the off-diagonal subplots.")
            corner = st.checkbox("Corner Plot", value=False, help="Hide the upper triangle of the plot to save space.")

    # 3. Generate Button
    # We use a session state key to store the resulting figure so it persists
    # across interactions (like 'Add to Canvas')

    plot_key = "current_pairwise_fig"

    if st.button("Generate Pairwise Plot", type="primary"):
        if len(selected_channels) < 2:
            st.error("Please select at least 2 channels.")
            # Clear previous if any
            if plot_key in st.session_state:
                del st.session_state[plot_key]
        else:
            with st.spinner("Generating plot... This may take a moment."):
                try:
                    # Prepare Data
                    df = experiment.data

                    # Sampling
                    if sample_size < len(df):
                        df = df.sample(n=int(sample_size), random_state=42)

                    # Columns to include
                    cols = list(selected_channels)
                    hue = None
                    if selected_hue != "None":
                        hue = selected_hue
                        if hue not in cols:
                            cols.append(hue)

                    plot_df = df[cols].copy()

                    # Apply Log Scale if requested
                    if scale == 'log':
                        # Filter positive values for selected channels
                        for col in selected_channels:
                             plot_df = plot_df[plot_df[col] > 0]

                        # Apply Log10
                        for col in selected_channels:
                             plot_df[col] = np.log10(plot_df[col])

                    # Generate Seaborn PairGrid/Plot
                    # Note: Using plt.figure() context to ensure we don't interfere with other plots?
                    # Seaborn creates its own figure usually.

                    plt.close('all') # Close open figures to free memory/context

                    # Define plot keywords to reduce dot size
                    plot_kws = {}
                    if kind == "scatter":
                        plot_kws = {'s': 10} # Reduced by half (assuming default ~20)
                    elif kind == "reg":
                        plot_kws = {'scatter_kws': {'s': 6}} # Reduced to a third (assuming default ~20)

                    grid = sns.pairplot(
                        plot_df,
                        hue=hue,
                        kind=cast(Literal["kde", "scatter", "hist", "reg"], kind),
                        corner=corner,
                        diag_kind="kde" if kind=="kde" else "auto",
                        plot_kws=plot_kws
                    )

                    # Store figure
                    st.session_state[plot_key] = grid.figure
                    st.session_state['pairwise_title'] = f"Pairwise ({kind}): {', '.join(selected_channels)}"

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    if plot_key in st.session_state:
                        del st.session_state[plot_key]

    # 4. Display Result
    if plot_key in st.session_state and st.session_state[plot_key]:
        st.divider()
        fig = st.session_state[plot_key]
        st.pyplot(fig)

        if st.button("Add to Canvas", key="btn_add_pairwise_canvas"):
            title = st.session_state.get('pairwise_title', "Pairwise Plot")
            add_plot_to_canvas(fig, title)
            st.success("Plot added to canvas!")
