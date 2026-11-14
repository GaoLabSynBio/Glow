import streamlit as st
import matplotlib.pyplot as plt
import copy
from glow.plotting import plots
from glow.canvas.functions import add_plot_to_canvas
from common.widgets import download_svg_button as download_btn
from common.errors import show_error
from .gating_component import render_gating_component, apply_gates_to_dataframe

def render_scatter_ui(filtered_experiment, channels, hue_var, facet_var, conditions, scale):
    """
    Renders UI and plot logic for 2D Scatter.
    """
    # --- Gating (Unified Component) ---
    local_gates, analysis_gates_to_apply = render_gating_component(filtered_experiment, channels, key_prefix="scatter_plot")

    # Create split view: Options (Left), Plot (Right)
    col_opts, col_plot = st.columns([1, 3])

    # Variables to track group selection
    selected_groups = None
    final_order = None

    with col_opts:
        st.markdown("#### Plot Settings")

        # Determine default indices for mCherry (X) and GFP (Y)
        default_x_index = 0
        default_y_index = 1 if len(channels) > 1 else 0

        for i, c in enumerate(channels):
            if "mcherry" in c.lower():
                default_x_index = i
                break

        for i, c in enumerate(channels):
            if "gfp" in c.lower():
                default_y_index = i
                break

        x_channel = st.selectbox("X Axis Channel", channels, index=default_x_index)
        y_channel = st.selectbox("Y Axis Channel", channels, index=default_y_index)
        scale = st.selectbox("Scale", ["log", "linear", "logicle"], index=0, help="Axis scale transformation.")

        # Scatter specific settings
        st.markdown("---")
        st.markdown("**Rendering Options**")

        show_trend = st.checkbox("Show Trend Line (Moving Mean)", value=True)

        c1, c2 = st.columns(2)
        with c1:
             show_dots = st.checkbox("Show Dots (Individual)", value=True)
        with c2:
             show_dots_merged = st.checkbox("Show Dots (Merged)", value=False)

        merged_size = st.slider("Merged Plot Size", min_value=2.0, max_value=8.0, value=3.5, step=0.5, help="Size of the merged comparison plot (width & height).")

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
        st.markdown("#### Grouping Configuration")

        # Filter conditions to remove 'cells' and 'singlets'
        filtered_conditions = [c for c in conditions if c not in ["cells", "singlets"]]

        # Determine defaults
        default_group_idx = 0
        default_facet_idx = 0

        # Candidates for defaults order: Experiment Name > Sample
        candidates = ["Experiment Name", "Experiment_Name", "Sample", "sample"]

        # Find best candidate present in filtered conditions
        best_candidate = None
        for c in candidates:
             if c in filtered_conditions:
                 best_candidate = c
                 break

        # Group Options
        group_options = ["None"] + filtered_conditions

        # Set defaults if candidate found
        if best_candidate:
            if best_candidate in group_options:
                 default_group_idx = group_options.index(best_candidate)
            if best_candidate in group_options:
                 default_facet_idx = group_options.index(best_candidate)

        gc1, gc2 = st.columns(2)
        with gc1:
            compare_by = st.selectbox(
                "Color/Group By",
                group_options,
                index=default_group_idx
            )
            hue_var = None if compare_by == "None" else compare_by

        with gc2:
            facet_by = st.selectbox(
                "Facet By (Subplots)",
                group_options,
                index=default_facet_idx
            )
            facet_var = None if facet_by == "None" else facet_by

        if hue_var:
            all_groups = sorted(filtered_experiment.data[hue_var].unique())

            # Use columns for layout
            sg1, sg2 = st.columns([3, 1])
            with sg1:
                selected_groups = st.multiselect("Select Groups to Plot", all_groups, default=all_groups)
            with sg2:
                st.write("") # Spacer
                st.write("")
                manual_order = st.checkbox("Manual Group Order", value=False)

            final_order = selected_groups

            if manual_order and selected_groups:
                order_text = st.text_area("Specify Order (comma separated)", value=", ".join(selected_groups), height=68)
                if order_text:
                    custom_order = [x.strip() for x in order_text.split(",") if x.strip()]
                    final_order = [x for x in custom_order if x in selected_groups]
                    remaining = [x for x in selected_groups if x not in final_order]
                    final_order.extend(remaining)

        st.write("") # Spacer

        if st.button("Generate Plot", type="primary", key="btn_scatter"):
            st.session_state.scatter_generated = True

        if st.session_state.get('scatter_generated'):
            from glow.plotting import scatter, plots as main_plots

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
                        all_applied_gates
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

                    # Apply Group Filter (if Groups Selected)
                    if hue_var and final_order is not None:
                        # Filter dataframe to only keep rows where hue_var is in selected groups
                        filtered_df = filtered_df[filtered_df[hue_var].isin(final_order)]
                        if filtered_df.empty:
                            st.warning("No events match the selected groups.")
                            return

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
                                show_dots_summary=show_dots_merged,
                                show_grid=show_grid,
                                title=base_title,
                                merged_figsize=(merged_size, merged_size)
                            )

                            if isinstance(res, tuple):
                                # Correct unpacking for 3 items (facets, summary, legend) or 2 items
                                fig_legend = None
                                if len(res) == 3:
                                     fig_facets, fig_summary, fig_legend = res
                                else:
                                     fig_facets, fig_summary = res

                                st.subheader("Merged Comparison")

                                if fig_legend:
                                    # Legend on top, scaled proportionally (using a column to restrict width)
                                    # Use a ratio that creates a reasonable size (e.g., 20-25% of width)
                                    # This ensures it scales with browser but doesn't fill screen width (which would distort vertical legend)
                                    lc1, lc2 = st.columns([1.5, 4])
                                    with lc1:
                                        st.pyplot(fig_legend, width="content")

                                st.pyplot(fig_summary, width="content")

                                download_btn(fig_summary, "scatter_summary")
                                if st.button("Add Summary Plot to Canvas", key="add_scatter_summary_to_canvas"):
                                    if add_plot_to_canvas(fig_summary, base_title + " (summary)"):
                                        st.success("Plot added to canvas!")

                                # Close figures to free memory
                                if not getattr(fig_summary, '_glow_is_direct_ref', False):
                                    plt.close(fig_summary)
                                if fig_legend and not getattr(fig_legend, '_glow_is_direct_ref', False):
                                    plt.close(fig_legend)

                                st.subheader("Separate (Detailed) Panels")
                                st.pyplot(fig_facets, width="content")
                                download_btn(fig_facets, "scatter_facets")
                                if st.button("Add Facet Plot to Canvas", key="add_scatter_facets_to_canvas"):
                                    if add_plot_to_canvas(fig_facets, base_title + " (facets)"):
                                        st.success("Plot added to canvas!")

                                if not getattr(fig_facets, '_glow_is_direct_ref', False):
                                    plt.close(fig_facets)
                            else:
                                st.pyplot(res, width="content")
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
                                    title=f"Experiment: {group}",
                                    figsize=(3, 3)
                                )
                                st.pyplot(fig, width="content")
                                download_btn(fig, f"scatter_ind_{i}")
                                if st.button(f"Add \"{group}\" to Canvas", key=f"add_scatter_ind_{i}_to_canvas"):
                                    if add_plot_to_canvas(fig, base_title + f" ({group})"):
                                        st.success("Plot added to canvas!")

                                if not getattr(fig, '_glow_is_direct_ref', False):
                                    plt.close(fig)

                except Exception as e:
                    show_error(
                        "Could not generate the scatter plot.",
                        cause="the selected X/Y channels or gates may leave no plottable data",
                        fix="pick different channels, widen your gates, or adjust the scale settings",
                        exc=e,
                    )
