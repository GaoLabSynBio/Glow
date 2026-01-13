import base64
import io

import plotly.graph_objects as go
import plotly.io
import streamlit as st


def _matplotlib_to_plotly_image(fig, title: str) -> go.Figure:
    import matplotlib.pyplot as plt

    if not isinstance(fig, plt.Figure):
        raise TypeError("Expected a matplotlib.figure.Figure")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")

    img_fig = go.Figure()
    img_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="markers", marker=dict(opacity=0)))
    img_fig.update_xaxes(visible=False, range=[0, 1])
    img_fig.update_yaxes(visible=False, range=[0, 1])

    img_fig.add_layout_image(
        dict(
            source="data:image/png;base64," + img_str,
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            layer="below",
        )
    )
    img_fig.update_layout(title=title, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white")
    return img_fig


def add_plot_to_canvas(fig, title: str):
    """Store a plot in Streamlit session state for display in the Canvas tab.

    Supports Plotly figures directly; Matplotlib figures are converted to a Plotly
    image container so they can be edited/viewed consistently.
    """

    if "canvas_plots" not in st.session_state:
        st.session_state.canvas_plots = {}

    # Matplotlib Figure -> convert
    try:
        import matplotlib.pyplot as plt

        if isinstance(fig, plt.Figure):
            fig = _matplotlib_to_plotly_image(fig, title)
    except Exception:
        # If matplotlib isn't available (or any other conversion issue), fall back.
        pass

    # Plotly Figure -> JSON storage
    st.session_state.canvas_plots[title] = fig.to_json()


def get_canvas_plots():
    return st.session_state.get("canvas_plots", {})


def remove_plot_from_canvas(title: str):
    if "canvas_plots" in st.session_state and title in st.session_state.canvas_plots:
        del st.session_state.canvas_plots[title]


def update_canvas_plot(title: str, fig):
    if "canvas_plots" in st.session_state:
        st.session_state.canvas_plots[title] = fig.to_json()
