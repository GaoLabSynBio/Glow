import io
import streamlit as st
import warnings
import pickle
import copy

def add_plot_to_canvas(fig, title: str):
    """Store a plot in Streamlit session state for display in the Canvas tab.

    Stores the figure object directly to allow for dynamic resizing and updates.
    Returns True if added successfully.
    """

    if "canvas_plots" not in st.session_state:
        st.session_state.canvas_plots = {}

    try:
        import matplotlib.pyplot as plt

        # Check for Matplotlib Figure
        if isinstance(fig, plt.Figure):
            # Create a deep copy to ensure the figure persists even if the original is closed
            try:
                # Force draw to ensure artists are initialized
                fig.canvas.draw()
                fig_copy = pickle.loads(pickle.dumps(fig))
            except Exception as e:
                # Fallback if pickling fails (e.g. complex backends), though less safe vs plt.close()
                fig_copy = fig
                # Tag as direct reference to prevent closing
                fig_copy._glow_is_direct_ref = True

            # Store Matplotlib Figure
            st.session_state.canvas_plots[title] = {
                "type": "matplotlib",
                "fig": fig_copy, 
                "settings": {"width_cols": 1, "font_scale": 1.0, "dot_scale": 1.0}
            }
            return True

        import plotly.graph_objects as go
        # Check for Plotly Figure
        if isinstance(fig, go.Figure):
            # Store Plotly figure
            st.session_state.canvas_plots[title] = {
                "type": "plotly",
                "data": fig,
                "settings": {"width_cols": 1, "font_scale": 1.0, "dot_scale": 1.0}
            }
            return True
    except ImportError:
        pass
    except Exception as e:
        st.error(f"Failed to process plot: {e}")
        return False

    st.warning(f"Plot type {type(fig)} not supported for Canvas storage.")
    return False


def get_canvas_plots():
    return st.session_state.get("canvas_plots", {})


def add_asperatio_to_canvas(data, title: str):
    """Store asperatio data in Streamlit session state for display in the Canvas tab."""
    if "canvas_plots" not in st.session_state:
        st.session_state.canvas_plots = {}
    st.session_state.canvas_plots[title] = {
        "type": "asperatio",
        "data": data,
        "settings": {"width_cols": 1, "font_scale": 1.0, "dot_scale": 1.0}
    }


def remove_plot_from_canvas(title: str):
    if "canvas_plots" in st.session_state and title in st.session_state.canvas_plots:
        del st.session_state.canvas_plots[title]


def move_plot_in_canvas(title: str, direction: str):
    """Move a plot up or down in the canvas order."""
    if "canvas_plots" not in st.session_state:
        return

    plots = st.session_state.canvas_plots
    keys = list(plots.keys())

    if title not in keys:
        return

    idx = keys.index(title)
    new_idx = idx - 1 if direction == "up" else idx + 1

    if 0 <= new_idx < len(keys):
        # Create a new dictionary with swapped retrieval order
        keys[idx], keys[new_idx] = keys[new_idx], keys[idx]
        st.session_state.canvas_plots = {k: plots[k] for k in keys}
