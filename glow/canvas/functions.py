import io
import streamlit as st
import warnings

def add_plot_to_canvas(fig, title: str):
    """Store a plot in Streamlit session state for display in the Canvas tab.

    Stores Matplotlib figures as PNG bytes (for display) and SVG bytes (for export).
    """

    if "canvas_plots" not in st.session_state:
        st.session_state.canvas_plots = {}

    try:
        import matplotlib.pyplot as plt
        if isinstance(fig, plt.Figure):
            # 1. Save PNG for display
            buf_png = io.BytesIO()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
                fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=300)
            buf_png.seek(0)

            # 2. Save SVG for export
            buf_svg = io.BytesIO()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
                fig.savefig(buf_svg, format="svg", bbox_inches="tight")
            buf_svg.seek(0)

            # Store
            st.session_state.canvas_plots[title] = {
                "type": "png",
                "data": buf_png.getvalue(),
                "svg_data": buf_svg.getvalue()
            }
            return

        import plotly.graph_objects as go
        if isinstance(fig, go.Figure):
            # Store Plotly figure directly to avoid kaleido/chrome dependency
            st.session_state.canvas_plots[title] = {
                "type": "plotly",
                "data": fig
            }
            return
    except ImportError:
        pass
    except Exception as e:
        st.error(f"Failed to process plot: {e}")
        return

    st.warning(f"Plot type {type(fig)} not supported for Canvas storage.")


def get_canvas_plots():
    return st.session_state.get("canvas_plots", {})


def remove_plot_from_canvas(title: str):
    if "canvas_plots" in st.session_state and title in st.session_state.canvas_plots:
        del st.session_state.canvas_plots[title]
