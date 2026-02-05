import streamlit as st
import matplotlib.pyplot as plt
import importlib
import copy
import io
import warnings
from glow.plotting import plots
from glow.canvas.functions import add_plot_to_canvas
from .gating_component import render_gating_component, apply_gates_to_dataframe

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

def render_scatter_ui(filtered_experiment, channels, hue_var, facet_var, conditions, scale):
    """
    Renders UI and plot logic for 2D Scatter.
    """
    # --- Gating (Unified Component) ---
    local_gates, analysis_gates_to_apply = render_gating_component(filtered_experiment, channels, key_prefix="scatter_plot")

    # Create split view: Options (Left), Plot (Right)
    col_opts, col_plot = st.columns([1, 3])

    with col_opts:
        st.markdown("#### Plot Settings")
        
        x_channel = st.selectbox("X Axis Channel", channels, index=0)
        y_channel = st.selectbox("Y Axis Channel", channels, index=1 if len(channels) > 1 else 0)
        scale = st.selectbox("Scale", ["log", "linear", "logicle"], index=0, help="Axis scale transformation.")

        compare_by = st.selectbox(
            "Color/Group By",
            ["None"] + conditions,
            index=0 if not conditions else 1
        )
        hue_var = None if compare_by == "None" else compare_by

        # Default to Experiment Name if available
        default_idx = 0
        if "Experiment Name" in conditions:
            try:
                default_idx = (["None"] + conditions).index("Experiment Name")
            except ValueError:
                default_idx = 0
        facet_by = st.selectbox("Facet By (Subplots)", ["None"] + conditions, index=default_idx)
        facet_var = None if facet_by == "None" else facet_by

        # Scatter specific settings
        st.markdown("---")
        st.markdown("**Rendering Options**")
        
        show_trend = st.checkbox("Show Trend Line (Moving Mean)", value=True)
        show_dots = st.checkbox("Show Dots (Density Color)", value=True)
        show_grid = st.checkbox("Show Grid", value=False)

        # Backend Selection
        backend = st.radio("Plot Backend", ["Matplotlib", "Plotly"], index=0, horizontal=True)

        show_merged = True
        show_individual = False

        # Logic for merging/individual view selection based on trend mode
        show_merged = st.checkbox("Show Merged/Main Plot", value=True)
        if not show_trend:
             # Plotly handles facets easily, but splitting by group manually is distinct.
             # Disable individual splits for Plotly for now to keep it simple, or allow it?
             # Let's allow it for consistency.
             show_individual = st.checkbox("Show Individual Groups", value=False)
        else:
             st.caption("Individual splitting handled by Facet selection")

        st.markdown("---")

    with col_plot:
        if st.button("Generate Plot", type="primary", key="btn_scatter"):
            st.session_state.scatter_generated = True

        if st.session_state.get('scatter_generated'):
            from glow.plotting import scatter, plots as main_plots
            importlib.reload(scatter)
            importlib.reload(main_plots)

            base_title = f"Scatter: {x_channel} vs {y_channel}"
            if hue_var:
                base_title += f" by {hue_var}"
            if facet_var:
                base_title += f" (facet: {facet_var})"
            if show_trend:
                base_title += " + trend"
            if show_dots:
                base_title += " + dots"

            with st.spinner("Generating plots..."):
                try:
                    # Apply Gates (Filtering + Scaling)
                    all_applied_gates = local_gates + analysis_gates_to_apply
                    
                    filtered_df, processed_gates = apply_gates_to_dataframe(
                        filtered_experiment.data, 
                        all_applied_gates, 
                        experiment_metadata=getattr(filtered_experiment, 'metadata', None), 
                        show_raw=st.session_state.get('plot_show_raw_vals', False)
                    )
                    
                    if filtered_df.empty:
                        st.warning("No events remain after applying gates. Please adjust your gates.")
                        return

                    if scale == 'log':
                        # Check for positive values to avoid matplotlib crash
                        x_ok = (filtered_df[x_channel] > 0).any()
                        y_ok = (filtered_df[y_channel] > 0).any()
                        if not x_ok or not y_ok:
                            st.warning(f"Data has no positive values in {x_channel} or {y_channel}, and therefore cannot be log-scaled. Consider switching to Linear or Logicle scale, or adjusting gates.")
                            return

                    if processed_gates:
                        st.write(f"Applying {len(processed_gates)} filters...")

                    # Create temp experiment for backend
                    plot_experiment = copy.copy(filtered_experiment)
                    plot_experiment.data = filtered_df

                    if show_merged:
                        if not show_trend:
                            st.subheader("Merged Plot")

                        if backend == "Matplotlib":
                            res = main_plots.plot_2d_scatter(
                                plot_experiment,
                                x_channel=x_channel,
                                y_channel=y_channel,
                                hue=hue_var,
                                facet=facet_var,
                                xscale=scale,
                                yscale=scale,
                                show_trend=show_trend,
                                show_dots=show_dots,
                                show_grid=show_grid
                            )

                            if isinstance(res, tuple):
                                fig_facets, fig_summary = res
                                st.subheader("Merged Comparison")
                                st.pyplot(fig_summary)
                                download_btn(fig_summary, "scatter_summary")
                                if st.button("Add Summary Plot to Canvas", key="add_scatter_summary_to_canvas"):
                                    if add_plot_to_canvas(fig_summary, base_title + " (summary)"):
                                        st.success("Plot added to canvas!")

                                if not getattr(fig_summary, '_glow_is_direct_ref', False):
                                    plt.close(fig_summary)

                                st.subheader("Separate Panels (Detailed)")
                                st.pyplot(fig_facets)
                                download_btn(fig_facets, "scatter_facets")
                                if st.button("Add Facet Plot to Canvas", key="add_scatter_facets_to_canvas"):
                                    if add_plot_to_canvas(fig_facets, base_title + " (facets)"):
                                        st.success("Plot added to canvas!")

                                if not getattr(fig_facets, '_glow_is_direct_ref', False):
                                    plt.close(fig_facets)
                            else:
                                st.pyplot(res)
                                download_btn(res, "scatter_plot")
                                if st.button("Add Plot to Canvas", key="add_scatter_to_canvas"):
                                    if add_plot_to_canvas(res, base_title):
                                        st.success("Plot added to canvas!")

                                if not getattr(res, '_glow_is_direct_ref', False):
                                    plt.close(res)
                        else:
                            # Plotly Backend
                            if show_trend:
                                st.warning("Trend lines not yet supported in Plotly mode. Using standard scatter.")

                            fig = scatter.plot_2d_scatter_plotly(
                                plot_experiment,
                                x_channel=x_channel,
                                y_channel=y_channel,
                                hue=hue_var,
                                facet=facet_var,
                                scale=scale,
                                show_grid=show_grid,
                                title=base_title
                            )
                            st.plotly_chart(fig, width="stretch")
                            if st.button("Add Plot to Canvas", key="add_plotly_scatter"):
                                 add_plot_to_canvas(fig, base_title)
                                 st.success("Added to Canvas!")


                    if show_individual and hue_var and not show_trend:
                        st.subheader("Individual Plots")
                        groups = sorted(filtered_experiment.data[hue_var].unique())
                        cols = st.columns(2)
                        for i, group in enumerate(groups):
                            sub_exp = plots.filter_experiment(plot_experiment, hue_var, [group])
                            with cols[i % 2]:
                                fig = main_plots.plot_2d_scatter(
                                    sub_exp,
                                    x_channel=x_channel,
                                    y_channel=y_channel,
                                    hue=None,
                                    xscale=scale,
                                    yscale=scale,
                                    show_trend=False,
                                    title=f"Experiment: {group}"
                                )
                                st.pyplot(fig)
                                download_btn(fig, f"scatter_ind_{i}")
                                if st.button(f"Add \"{group}\" to Canvas", key=f"add_scatter_ind_{i}_to_canvas"):
                                    if add_plot_to_canvas(fig, base_title + f" ({group})"):
                                        st.success("Plot added to canvas!")

                                if not getattr(fig, '_glow_is_direct_ref', False):
                                    plt.close(fig)

                except Exception as e:
                    st.error(f"Error: {e}")
