import streamlit as st
import plotly.graph_objects as go
import plotly.io

from glow.canvas.functions import get_canvas_plots, remove_plot_from_canvas


def render_canvas_tab():
    st.header("Plot Canvas")

    plots = get_canvas_plots()
    if not plots:
        st.info("No plots added to canvas yet. Use 'Add to Canvas' buttons from other tabs.")
        return

    for title, plot_json in list(plots.items()):
        st.subheader(title)
        fig = go.Figure(plotly.io.from_json(plot_json))
        st.plotly_chart(fig, use_container_width=True)

        if st.button(f"Remove: {title}", key=f"remove_canvas_{title}"):
            remove_plot_from_canvas(title)
            st.rerun()

        st.divider()
