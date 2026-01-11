import streamlit as st
import plotly.graph_objects as go
import plotly.io

def add_plot_to_canvas(fig, title):
    """Adds a Plotly figure to the canvas storage in session state."""
    if 'canvas_plots' not in st.session_state:
        st.session_state.canvas_plots = {}
    
    # Store as JSON to ensure it's serializable and persistent across reruns
    st.session_state.canvas_plots[title] = fig.to_json()

def get_canvas_plots():
    """Retrieves all plots from the canvas storage."""
    return st.session_state.get('canvas_plots', {})

def remove_plot_from_canvas(title):
    """Removes a plot from the canvas."""
    if 'canvas_plots' in st.session_state and title in st.session_state.canvas_plots:
        del st.session_state.canvas_plots[title]

def update_canvas_plot(title, fig):
    """Updates an existing plot on the canvas."""
    if 'canvas_plots' in st.session_state:
        st.session_state.canvas_plots[title] = fig.to_json()
