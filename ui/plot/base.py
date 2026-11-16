import streamlit as st
from glow.plotting import plots

def render_data_selection(experiment):
    """
    Renders common data selection and filtering UI.
    Returns: (filtered_experiment, conditions)
    """
    st.subheader("1. Data Selection")
    conditions = plots.get_conditions(experiment)

    # 1. Choose categorization/filtering column
    filter_col = st.selectbox("Select Category to Filter By", ["None"] + conditions, index=1 if conditions else 0)

    filtered_experiment = experiment
    if filter_col != "None":
        unique_vals = sorted(experiment.data[filter_col].unique())
        selected_vals = st.multiselect(f"Select {filter_col} to Include", unique_vals, default=unique_vals)

        if not selected_vals:
            st.warning("No data selected.")
            return None, conditions

        filtered_experiment = plots.filter_experiment(experiment, filter_col, selected_vals)
        st.caption(f"Filtered data contains {len(filtered_experiment.data)} events.")
    else:
        st.caption(f"Using full dataset ({len(experiment.data)} events).")

    return filtered_experiment, conditions


PREVIEW_MAX_EVENTS = 50_000


def render_full_toggle(key):
    """Preview/full switch shown above a live plot. Returns True when the full data should be used."""
    full = st.toggle("Render full data", value=False, key=f"render_full_{key}",
                     help=f"Live preview uses at most {PREVIEW_MAX_EVENTS:,} events. "
                          "Turn on for the final plot, canvas and export.")
    st.session_state["plot_preview_active"] = False
    return full


def preview_subsample(df, full):
    """Subsample df to PREVIEW_MAX_EVENTS (fixed seed) unless full rendering was requested."""
    if full or len(df) <= PREVIEW_MAX_EVENTS:
        return df
    st.session_state["plot_preview_active"] = True
    st.caption(f"Preview: {PREVIEW_MAX_EVENTS:,} of {len(df):,} events. "
               "Switch on 'Render full data' before adding to canvas.")
    return df.sample(PREVIEW_MAX_EVENTS, random_state=0)


def add_plot_to_canvas(fig, title):
    """Canvas add that refuses subsampled previews."""
    from glow.canvas.functions import add_plot_to_canvas as _add
    if st.session_state.get("plot_preview_active"):
        st.warning("This is a subsampled preview. Turn on 'Render full data' to add it to the canvas.")
        return False
    return _add(fig, title)
