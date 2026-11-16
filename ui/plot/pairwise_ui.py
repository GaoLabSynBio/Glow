import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal, cast
from glow.canvas.functions import add_plot_to_canvas
from .gating_component import render_gating_component, apply_gates_to_dataframe

def render_pairwise_ui(experiment, channels, conditions, scale='log'):
    """
    Renders the UI for Pairwise Plots (Seaborn pairplot).
    """
    # --- Gating (Unified Component) ---
    local_gates, analysis_gates_to_apply = render_gating_component(experiment, channels, key_prefix="pairwise_plot")

    col_opts, col_plot = st.columns([1, 3])

    with col_opts:
        st.markdown("### Configuration")

        # 1. Settings
        # Channel Selection
        # Default to first few channels if available
        default_channels = channels[:4] if len(channels) >= 4 else channels
        selected_channels = st.multiselect(
            "Channels",
            channels,
            default=default_channels,
            help="Select at least 2 channels."
        )

        # Hue Selection
        hue_options = ["None"] + conditions
        # Add Sample if not present
        if "Sample" in experiment.data.columns and "Sample" not in hue_options:
            hue_options.append("Sample")

        selected_hue = st.selectbox("Group/Color", hue_options, index=0)

        # Scale Selector
        scale = st.selectbox("Scale", ["log", "linear"], index=0, help="Axis scale transformation.")

        # 2. Advanced / Performance
        st.markdown("---")
        st.markdown("**Advanced**")
        max_events = len(experiment.data)

        # Downsampling
        sample_size = st.number_input(
            "Sample Size",
            min_value=50,
            max_value=max_events,
            value=min(2000, max_events),
            step=100,
            help="Smaller sample size is faster. < 2000 recommended for KDE pairplots."
        )

        kind = st.selectbox("Plot Kind", ["scatter", "kde", "hist", "reg"], index=0, help="Type of plot for the off-diagonal subplots.")
        corner = st.checkbox("Corner Plot", value=False, help="Hide the upper triangle of the plot to save space.")

        # 3. Generate Button
        plot_key = "current_pairwise_fig"
        st.markdown("---")

    # 4. Display Result
    with col_plot:
        if st.button("Generate Pairwise Plot", type="primary"):
            st.session_state.run_pairwise_gen = True

        # Logic Process (Between columns or inside one? Better inside plot column or conditional)
        if st.session_state.get('run_pairwise_gen'):
            st.session_state.run_pairwise_gen = False

            if len(selected_channels) < 2:
                st.error("Please select at least 2 channels.")
                if plot_key in st.session_state:
                    del st.session_state[plot_key]
            else:
                with st.spinner("Generating plot... This may take a moment."):
                    try:
                        # Apply Gates (Filtering + Scaling)
                        all_applied_gates = local_gates + analysis_gates_to_apply

                        filtered_df, processed_gates = apply_gates_to_dataframe(
                            experiment.data,
                            all_applied_gates
                        )

                        if filtered_df.empty:
                            st.warning("No events remain after applying gates. Please adjust your gates.")
                            # Stop execution of the plot generation block
                            # Since we are inside a loop flag, just returning or breaking out is tricky if not in a function top level.
                            # render_pairwise_ui IS a function.
                            return

                        # Prepare Data
                        df = filtered_df

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

                        plt.close('all') # Close open figures

                        # Define plot keywords
                        plot_kws = {}
                        if kind == "scatter":
                            plot_kws = {'s': 6, 'alpha': 0.6}
                        elif kind == "reg":
                            plot_kws = {'scatter_kws': {'s': 3, 'alpha': 0.4}}

                        grid = sns.pairplot(
                            plot_df,
                            hue=hue,
                            kind=cast(Literal["scatter", "kde", "hist", "reg"], kind),
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

        if plot_key in st.session_state and st.session_state[plot_key]:
            fig = st.session_state[plot_key]
            st.pyplot(fig)

            if st.button("Add to Canvas", key="btn_add_pairwise_canvas"):
                title = st.session_state.get('pairwise_title', "Pairwise Plot")
                add_plot_to_canvas(fig, title)
                st.toast("Plot added to canvas.")
