import streamlit as st
import matplotlib.pyplot as plt
import importlib
from glow.plotting import plots

def render_scatter_ui(filtered_experiment, channels, hue_var, facet_var, conditions, scale):
    """
    Renders UI and plot logic for 2D Scatter.
    """
    col1, col2 = st.columns(2)
    with col1: x_channel = st.selectbox("X Axis Channel", channels, index=0)
    with col2: y_channel = st.selectbox("Y Axis Channel", channels, index=1 if len(channels) > 1 else 0)
    
    # Move selection here
    c_grp1, c_grp2 = st.columns(2)
    with c_grp1:
        compare_by = st.selectbox(
            "Color/Group By",
            ["None"] + conditions,
            index=0 if not conditions else 1
        )
        hue_var = None if compare_by == "None" else compare_by
        
    with c_grp2:
        # Default to Experiment Name if available
        default_idx = 0
        if "Experiment Name" in conditions: 
            try:
                default_idx = (["None"] + conditions).index("Experiment Name")
            except:
                default_idx = 0
        facet_by = st.selectbox("Facet By (Subplots)", ["None"] + conditions, index=default_idx)
        facet_var = None if facet_by == "None" else facet_by
    
    # Scatter specific settings
    c_opt1, c_opt2, c_opt3 = st.columns(3)
    with c_opt1: show_trend = st.checkbox("Show Trend Line (Moving Mean)", value=True)
    with c_opt2: show_dots = st.checkbox("Show Dots (Density Color)", value=True)
    with c_opt3: show_grid = st.checkbox("Show Grid", value=False)
    
    show_merged = True
    show_individual = False
    
    # Logic for merging/individual view selection based on trend mode
    c1, c2 = st.columns(2)
    with c1: show_merged = st.checkbox("Show Merged/Main Plot", value=True)
    if not show_trend:
         with c2: show_individual = st.checkbox("Show Individual Groups (below)", value=False)
    else:
         with c2: st.caption("Individual splitting handled by Facet selection")

    if st.button("Generate Plot", type="primary", key="btn_scatter"):
        from glow.plotting import scatter, plots as main_plots
        importlib.reload(scatter)
        importlib.reload(main_plots)
        
        with st.spinner("Generating plots..."):
            try:
                if show_merged:
                    if not show_trend:
                        st.subheader("Merged Plot")
                        
                    res = main_plots.plot_2d_scatter(
                        filtered_experiment,
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
                        plt.close(fig_summary)
                        
                        st.subheader("Separate Panels (Detailed)")
                        st.pyplot(fig_facets)
                        plt.close(fig_facets)
                    else:
                        st.pyplot(res)
                        plt.close(res)
                
                if show_individual and hue_var and not show_trend:
                    st.subheader("Individual Plots")
                    groups = sorted(filtered_experiment.data[hue_var].unique())
                    cols = st.columns(2)
                    for i, group in enumerate(groups):
                        sub_exp = plots.filter_experiment(filtered_experiment, hue_var, [group])
                        with cols[i % 2]:
                            # st.markdown(f"**{group}**")
                            fig = main_plots.plot_2d_scatter(
                                sub_exp, x_channel=x_channel, y_channel=y_channel, 
                                hue=None, xscale=scale, yscale=scale, show_trend=False,
                                title=f"Experiment: {group}"
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                            
            except Exception as e:
                st.error(f"Error: {e}")
