import streamlit as st
import matplotlib.pyplot as plt
import importlib
from glow.plotting import plots

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
                        legend_font_size=legend_font_size
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    from glow.canvas.functions import add_plot_to_canvas
                    if st.button("Add Merged Plot to Canvas", key="add_merged_hist_to_canvas"):
                        add_plot_to_canvas(fig, f"Histogram: {selected_channel} by {hue_var}" if hue_var else f"Histogram: {selected_channel}")
                        st.success("Plot added to canvas!")
                
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
                            # st.markdown(f"**{group}**")
                            fig = main_plots.plot_1d_histogram(
                                sub_exp, 
                                channel=selected_channel, 
                                hue=None, 
                                scale=scale,
                                color=palette[i],
                                title=f"Experiment: {group}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
