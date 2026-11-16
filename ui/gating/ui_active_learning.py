import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from glow.active_learning.learner import ActiveLearner
from glow.logging_config import get_logger
from glow import paths
from common.errors import show_error

log = get_logger("ui.active_learning")

def get_active_learner(model_type='gating'):
    key = f"active_learner_{model_type}"

    if key not in st.session_state:
        log.info("Initializing ActiveLearner for %s", model_type)
        model_file = "gating_model_full.pkl" if model_type == 'gating' else "singlet_model_full.pkl"
        model_path = str(paths.model_path(model_file))
        try:
             st.session_state[key] = ActiveLearner(model_path=model_path, model_type=model_type)
        except Exception as e:
             show_error(
                 f"Could not load the {model_type} model for active learning.",
                 cause="the bundled model file may be missing or incompatible",
                 fix="reinstall/redownload the app so the model files under data/ are present",
                 exc=e,
             )
             return None

    # Auto-heal: if a stale instance is cached, rebuild from the current class.
    learner = st.session_state[key]
    if not hasattr(learner, 'reset_model'):
        st.session_state[key] = ActiveLearner(model_path=learner.model_path, model_type=learner.model_type)

    return st.session_state[key]

def render_reset_controls(model_type='gating'):
    """
    Renders a button to reset the model to its initial state.
    """
    st.divider()
    label = "Start Over (Reset Model)"
    if model_type == 'singlet':
        label = "Start Over (Reset Singlet Model)"

    if st.button(label, help="Reverts the model to the initial pre-trained state. All interactive learning will be lost.", key=f"reset_btn_{model_type}"):
        learner = get_active_learner(model_type)
        if learner:
            with st.spinner("Resetting model..."):
                success = learner.reset_model()

            if success:
                st.toast("Model reset.")
                # Clear UI state - slightly drastic but safe
                # We should only clear relevant state
                if model_type == 'gating':
                    if 'uncertain_indices' in st.session_state:
                        del st.session_state.uncertain_indices
                    if 'gated_experiment' in st.session_state:
                        del st.session_state.gated_experiment
                elif model_type == 'singlet':
                    if 'doublets_removed' in st.session_state:
                        del st.session_state.doublets_removed

                # Force refresh
                st.session_state.plot_id = st.session_state.get('plot_id', 0) + 1
                st.rerun()
            else:
                st.error("Could not reset model. Base model not found.")

def render_uncertainty_controls(experiment, x_col, y_col, model_type='gating'):
    """
    Renders the button to Release the Hounds (find uncertainties).
    """
    # if model_type != 'gating': return # Not implementing singlet uncertainty viz yet

    st.markdown("### 🧠 Active Learning")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Find Confusion", key=f"find_conf_{model_type}"):
            with st.spinner("Querying model..."):
                learner = get_active_learner(model_type)
                if not learner:
                    return

                df = experiment.data

                # Data Prep (Assume linear UI -> log Model)
                is_x_log = (df[x_col].min() < 0) or (df[x_col].max() < 10)
                vals_x = df[x_col].values
                vals_y = df[y_col].values

                if not is_x_log:
                    vals_x = np.log1p(vals_x)
                    vals_y = np.log1p(vals_y)

                if model_type == 'singlet':
                     query_df = pd.DataFrame({
                        "Log_FSC_A": vals_x,
                        "Log_FSC_H": vals_y
                    }, index=df.index)
                else:
                    query_df = pd.DataFrame({
                        "Log_FSC": vals_x,
                        "Log_SSC": vals_y
                    }, index=df.index)

                _, indices = learner.query_uncertainty(query_df, n_instances=5)
                st.session_state.uncertain_indices = indices
                st.rerun()

    with col2:
        if st.button("Clear Highlights", key=f"clear_highlights_{model_type}"):
            st.session_state.uncertain_indices = []
            st.rerun()

def get_uncertainty_trace(experiment, x_col, y_col):
    """
    Returns a Plotly trace for the uncertain points.
    """
    if 'uncertain_indices' not in st.session_state:
        return None

    indices = st.session_state.uncertain_indices
    # Robust check for empty list or empty numpy array
    if indices is None or len(indices) == 0:
        return None

    df = experiment.data

    try:
        # Filter indices that might be out of bounds if data changed
        valid_indices = [i for i in indices if i in df.index]

        # NOTE: indices returned by the learner are dataframe index labels (not positional)
        subset = df.loc[valid_indices]

        trace = go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode='markers',
            marker=dict(
                size=16,
                color='rgba(255, 0, 0, 0.08)',
                symbol='circle',
                line=dict(width=6, color='red')
            ),
            # Store original dataframe index so clicks on this trace can be mapped
            customdata=subset.index.astype(str),
            name="Uncertainty",
            hovertemplate="Uncertain point<br>Click to label<extra></extra>",
            hoverlabel=dict(bgcolor='red', font_size=14)
        )
        return trace
    except Exception:
        return None

def render_labeling_controls(experiment, x_col, y_col, selection, plot_df=None, model_type='gating'):
    """
    Renders buttons to label the selected points.
    """
    if not selection or not selection.get('selection') or not selection['selection'].get('points'):
        return

    pts = selection['selection']['points']
    count = len(pts)
    if count == 0:
        return

    st.divider()
    st.markdown(f"**📝 Label {count} Selected Points**")
    st.caption("Tip: you can select any cells (not just red-circled) using box select or Shift+Click, then label them.")
    strength = st.slider(
        "Learning strength",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
        help="Higher values up-weight your labels so the boundary moves more visibly.",
        key=f"strength_{model_type}"
    )

    col1, col2 = st.columns(2)

    learner = get_active_learner(model_type)
    if not learner:
        st.error("Active Learning Model not initialized.")
        return

    # Labels depend on context
    lbl_pos = "✅ In Gate"
    lbl_neg = "🚫 Not In Gate"

    if model_type == 'singlet':
        lbl_pos = "✅ Singlet"
        lbl_neg = "🚫 Doublet"

    if col1.button(lbl_pos, key=f"lbl_pos_{model_type}"):
        with st.spinner("Updating model..."):
            _teach_batch(learner, experiment, x_col, y_col, pts, label=1, plot_df=plot_df, strength=strength, model_type=model_type)
    if col2.button(lbl_neg, key=f"lbl_neg_{model_type}"):
        with st.spinner("Updating model..."):
            _teach_batch(learner, experiment, x_col, y_col, pts, label=0, plot_df=plot_df, strength=strength, model_type=model_type)

def _teach_batch(learner, experiment, x_col, y_col, points, label, plot_df=None, strength=10, model_type='gating'):
    def _coerce_index_value(value):
        if value is None:
            return None
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, str):
            s = value.strip()
            if s.isdigit():
                return int(s)
            return s
        return value

    def _extract_custom_index(point):
        for key in ('customdata', 'customData', 'custom_data'):
            if key not in point:
                continue
            raw = point.get(key)
            if raw is None:
                continue
            # px.scatter(custom_data=[...]) yields an array-like per point
            if isinstance(raw, (list, tuple, np.ndarray)):
                if len(raw) == 0:
                    continue
                return _coerce_index_value(raw[0])
            return _coerce_index_value(raw)
        return None

    # Prefer customdata (stable across traces), fallback to pointIndex/pointNumber.
    real_indices = []
    plot_indices = []

    for p in points:
        rid = _extract_custom_index(p)
        if rid is not None:
            real_indices.append(rid)
            continue

        idx = p.get('pointIndex')
        if idx is None:
            idx = p.get('point_index')
        if idx is None:
            idx = p.get('pointNumber')
        if idx is None:
            idx = p.get('point_number')

        if idx is not None:
            plot_indices.append(int(idx))

    # De-duplicate while preserving order (box select + trace overlays can double-report)
    if real_indices:
        real_indices = list(dict.fromkeys(real_indices))
    if plot_indices:
        plot_indices = list(dict.fromkeys(plot_indices))

    if not real_indices and not plot_indices:
        debug_keys = list(points[0].keys()) if points else "Empty"
        st.warning(
            f"No valid points selected. Debug info: Found {len(points)} raw points. Keys in first point: {debug_keys}"
        )
        return

    # Map selection -> dataframe rows
    try:
        if real_indices:
            subset = experiment.data.loc[real_indices]
        elif plot_df is not None:
            # Map positional plot indices to dataframe index
            try:
                # Handle downsampled plot_df
                real_indices = plot_df.iloc[plot_indices].index
                subset = experiment.data.loc[real_indices]
            except Exception:
                # Fallback: Maybe plot_df IS the subset
                subset = plot_df.iloc[plot_indices]
        else:
            subset = experiment.data.iloc[plot_indices]
    except Exception as e:
        st.error(
            f"Error mapping selection to data: {e}. Plot DF Size: {len(plot_df) if plot_df is not None else 'None'}"
        )
        return

    # Feature Prep
    is_x_log = (subset[x_col].min() < 0) or (subset[x_col].max() < 10)
    vals_x = subset[x_col].values
    vals_y = subset[y_col].values

    if not is_x_log:
        vals_x = np.log1p(vals_x + 1e-1) # Added epsilon safety
        vals_y = np.log1p(vals_y + 1e-1)

    X_new = np.column_stack([vals_x, vals_y])
    y_new = np.full(len(X_new), label)

    # Dynamic weighting based on selection size and user strength
    repetition_factor = int(strength) if strength is not None else 5
    repetition_factor = max(1, repetition_factor)

    effective_X = []
    effective_y = []

    for i in range(len(X_new)):
        X_repeated = np.tile(X_new[i], (repetition_factor, 1))
        y_repeated = np.full(repetition_factor, y_new[i])
        effective_X.append(X_repeated)
        effective_y.append(y_repeated)

    X_train = np.vstack(effective_X)
    y_train = np.concatenate(effective_y)

    print(f"[ActiveLearning:{model_type}] Updating with {len(X_new)} samples (Expanded to {len(X_train)} training points via strength {repetition_factor}).")

    # Single update step
    learner.teach(X_train, y_train)

    # Clear generic state that might impact viz
    if 'uncertain_indices' in st.session_state:
        st.session_state.uncertain_indices = []

    # Reset specific state to force re-render with new model
    if model_type == 'gating':
        if 'gated_experiment' in st.session_state:
            del st.session_state.gated_experiment
        if 'current_auto_boundary' in st.session_state:
            del st.session_state.current_auto_boundary
    elif model_type == 'singlet':
        # Don't delete 'doublets_removed' or we lose the page state completely?
        # Maybe we want to just invalidate the viz cache.
        pass

    # Increment plot ID to force clear selection state
    st.session_state.plot_id = st.session_state.get('plot_id', 0) + 1

    st.toast(f"Updated model with {len(X_new)} points.")
    st.rerun()
