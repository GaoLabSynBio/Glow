import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from glow.active_learning.learner import ActiveLearner

def get_active_learner():
    if 'active_learner' not in st.session_state:
        st.session_state.active_learner = ActiveLearner()
    return st.session_state.active_learner

def render_uncertainty_controls(experiment, x_col, y_col):
    """
    Renders the button to Release the Hounds (find uncertainties).
    """
    st.markdown("### 🧠 Active Learning")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Find Confusion"):
            with st.spinner("Querying model..."):
                learner = get_active_learner()
                df = experiment.data
                
                # Data Prep (Assume linear UI -> log Model)
                is_x_log = (df[x_col].min() < 0) or (df[x_col].max() < 10)
                vals_x = df[x_col].values
                vals_y = df[y_col].values
                
                if not is_x_log:
                    vals_x = np.log10(vals_x + 1)
                    vals_y = np.log10(vals_y + 1)
                
                query_df = pd.DataFrame({
                    "Log_FSC": vals_x,
                    "Log_SSC": vals_y
                }, index=df.index)
                
                _, indices = learner.query_uncertainty(query_df, n_instances=5)
                st.session_state.uncertain_indices = indices
                st.rerun()
    
    with col2:
        if st.button("Clear Highlights"):
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
    except Exception as e:
        return None

def render_labeling_controls(experiment, x_col, y_col, selection, plot_df=None):
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
    )
    
    col1, col2 = st.columns(2)
    
    learner = get_active_learner()
    
    if col1.button("✅ In Gate"):
        with st.spinner("Updating model..."):
            _teach_batch(learner, experiment, x_col, y_col, pts, label=1, plot_df=plot_df, strength=strength)
        
    if col2.button("🚫 Not In Gate"):
        with st.spinner("Updating model..."):
            _teach_batch(learner, experiment, x_col, y_col, pts, label=0, plot_df=plot_df, strength=strength)

def _teach_batch(learner, experiment, x_col, y_col, points, label, plot_df=None, strength=10):
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
            real_indices = plot_df.iloc[plot_indices].index
            subset = experiment.data.loc[real_indices]
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
        vals_x = np.log10(vals_x + 1)
        vals_y = np.log10(vals_y + 1)

    X_new = np.column_stack([vals_x, vals_y])
    y_new = np.full(len(X_new), label)
    # Dynamic aggressive up-weighting with Iterative Correction
    # Goal: Force the model to immediately respect these manual corrections.
    # Strategy: Flatten errors with extreme prejudice (repetition + iterative partial_fit).
    
    current_preds = learner.predict(pd.DataFrame({'Log_FSC': vals_x, 'Log_SSC': vals_y}))
    
    # Base strength increased to 200 (was 50)
    # If wrong, multiplier is 10x (was 4x) -> 2000x repetitions
    base_strength = 200 
    
    effective_X = []
    effective_y = []
    
    # Slider aggressiveness: 10 -> 5x, 30 -> 15x
    s_val = int(strength) if strength is not None else 10
    user_mult = max(1, s_val // 2)

    for i in range(len(X_new)):
        is_wrong = (current_preds[i] != y_new[i])
        
        instance_strength = base_strength * 10 if is_wrong else base_strength
        instance_strength *= user_mult
        
        # Cap to reasonable memory limits per point (e.g. 10k)
        instance_strength = min(instance_strength, 10000)

        X_repeated = np.tile(X_new[i], (instance_strength, 1))
        y_repeated = np.full(instance_strength, y_new[i])
        
        effective_X.append(X_repeated)
        effective_y.append(y_repeated)
        
    X_train = np.vstack(effective_X)
    y_train = np.concatenate(effective_y)

    print(f"[ActiveLearning] Iterative Boost: Expanded {len(X_new)} samples to {len(X_train)} training points. Starting force-fit loop...")

    # Force Fit Loop
    # Keep teaching until the model gets it right, or we hit a safety limit.
    max_loops = 10
    accuracy = 0.0
    
    for attempt in range(max_loops):
        learner.teach(X_train, y_train)
        
        # Verification
        check_preds = learner.predict(pd.DataFrame({'Log_FSC': vals_x, 'Log_SSC': vals_y}))
        accuracy = np.mean(check_preds == y_new)
        
        if accuracy >= 0.999: # Allow for tiny float oddities, practically 1.0
            print(f"[ActiveLearning] Convergence reached at iteration {attempt+1}")
            break
        print(f"[ActiveLearning] Iteration {attempt+1}: Accuracy {accuracy:.2f}. Retrying...")
    
    if accuracy < 1.0:
        print(f"[ActiveLearning] Warning: Capped at {max_loops} iterations with accuracy {accuracy:.2f}")

    st.session_state.uncertain_indices = []  # Clear highlights

    # Force re-gating next cycle
    if 'gated_experiment' in st.session_state:
        del st.session_state.gated_experiment

    if 'current_auto_boundary' in st.session_state:
        del st.session_state.current_auto_boundary

    # Increment plot ID to force clear selection state
    st.session_state.plot_id = st.session_state.get('plot_id', 0) + 1

    st.success(f"Updated model. Force-fit applied on {len(X_new)} points ({attempt+1} iterations, expanded to {len(X_train)}). Boundary should move immediately.")
    st.rerun()
