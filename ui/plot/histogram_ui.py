import streamlit as st
import matplotlib.pyplot as plt
import importlib
import warnings
from glow.plotting import plots
from glow.canvas.functions import add_plot_to_canvas
import io

def download_btn(fig, key):
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
        fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
         label="Download SVG",
         data=buf.getvalue(),
         file_name=f"{key}.svg",
         mime="image/svg+xml",
         key=f"dl_{key}"
    )

def render_histogram_ui(filtered_experiment, channels, hue_var, conditions, scale):
    """
    Renders UI and plot logic for 1D Histogram.
    """
    col1, col2 = st.columns(2)
    with col1:
        selected_channel = st.selectbox("Select Channel", channels)

    with col2:
        # User requested: 1D plot groupby should be fixed to Experiment Name if available
        target_cols = ["Experiment Name", "Experiment_Name", "Sample", "sample"]
        found_col = None
        for t in target_cols:
            if t in conditions:
                found_col = t
                break

        if found_col:
            hue_var = found_col
        else:
            compare_by = st.selectbox(
                "Color/Group By",
                ["None"] + conditions,
                index=0 if not conditions else 1,
                help="Select attribute for coloring/grouping histograms"
            )
            hue_var = None if compare_by == "None" else compare_by

    # Histogram specific settings
    show_merged = True
    show_individual = False
    legend_columns = None
    legend_font_size = 10

    st.markdown("#### Histogram Rendering")
    smooth_sigma = st.slider(
        "Smoothing (0 = none)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.2,
        help="Controls smoothing strength of the histogram curve. Higher values = smoother curve."
    )

    normalization_mode = "total"
    if hue_var:
        normalization_mode = st.selectbox(
            "Merged Plot Normalization",
            options=[
                ("Original (raw counts)", "original"),
                ("Modal normalization (FlowJo-style)", "modal"),
                ("Total count normalization", "total"),
            ],
            index=2,
            format_func=lambda x: x[0],
            help="Original: preserve event count differences. Modal: scale each group so peaks match. Total: scale each group so total area sums to 1."
        )[1]

    c1, c2 = st.columns(2)
    with c1: show_merged = st.checkbox("Show Merged/Main Plot", value=True)
    with c2: show_individual = st.checkbox("Show Individual Groups (below)", value=False)

    if hue_var:
         # Legend Column Customization
         st.markdown("##### Legend Configuration")
         extra_meta = [c for c in conditions if c != hue_var]
         avail_cols = [hue_var] + extra_meta + ["Mean", "Median", "Count"]

         # Default: hue_var + Mean
         default_cols = [hue_var, "Mean"]
         legend_columns = st.multiselect("Legend Table Columns", avail_cols, default=default_cols)
         legend_font_size = st.number_input("Legend Font Size", min_value=5, max_value=20, value=10, step=1)

    if st.button("Generate Plot", type="primary", key="btn_hist"):
        st.session_state.hist_generated = True

    if st.session_state.get('hist_generated'):
        # Reload plots module
        from glow.plotting import histogram, plots as main_plots
        importlib.reload(histogram)
        importlib.reload(main_plots)

        with st.spinner("Generating plots..."):
            try:
                # Merged Plot
                if show_merged:
                    st.subheader("Merged Plot")
                    fig = main_plots.plot_1d_histogram(
                        filtered_experiment,
                        channel=selected_channel,
                        hue=hue_var,
                        scale=scale,
                        legend_cols=legend_columns,
                        legend_font_size=legend_font_size,
                        smooth_sigma=smooth_sigma,
                        normalization=normalization_mode
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
                        st.pyplot(fig)

                    download_btn(fig, "hist_merged")

                    canvas_title = f"Histogram: {selected_channel} by {hue_var}" if hue_var else f"Histogram: {selected_channel}"
                    if st.button("Add Merged Plot to Canvas", key="add_merged_hist_to_canvas"):
                        add_plot_to_canvas(fig, canvas_title)
                        st.success("Plot added to canvas!")

                    plt.close(fig)

                # Individual Plots
                if show_individual and hue_var:
                    st.subheader("Individual Plots")
                    groups = sorted(filtered_experiment.data[hue_var].unique())

                    import seaborn as sns
                    palette = sns.color_palette("bright", len(groups))

                    cols = st.columns(2)
                    for i, group in enumerate(groups):
                        sub_exp = plots.filter_experiment(filtered_experiment, hue_var, [group])
                        with cols[i % 2]:
                            fig = main_plots.plot_1d_histogram(
                                sub_exp,
                                channel=selected_channel,
                                hue=None,
                                scale=scale,
                                color=palette[i],
                                title=f"Experiment: {group}",
                                smooth_sigma=smooth_sigma,
                                normalization="total"
                            )
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
                                st.pyplot(fig)

                            download_btn(fig, f"hist_ind_{i}")
                            if st.button(f"Add \"{group}\" Histogram to Canvas", key=f"add_hist_ind_{i}_to_canvas"):
                                add_plot_to_canvas(fig, f"Histogram: {selected_channel} ({group})")
                                st.success("Plot added to canvas!")
                            plt.close(fig)
            except Exception as e:
                st.error(f"Error: {e}")
