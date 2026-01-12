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
