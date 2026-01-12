import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cytoflow
import sys

# Ensure project root is in path so we can import 'glow'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from glow.gating.manager import GatingManager

# Configure Page
st.set_page_config(page_title="Training Data Generator", layout="wide")

SOURCE_DIR = "training/source_fcs"
OUTPUT_DIR = "training/labeled_data"

# Ensure directories exist
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_fcs(filepath):
    """Load FCS file into Cytoflow Experiment"""
    try:
        tube = cytoflow.Tube(file=filepath)
        import_op = cytoflow.ImportOp(tubes=[tube])
        experiment = import_op.apply()
        return experiment
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return None

def manual_gate_ui(experiment, x_channel, y_channel, key_prefix, info_text=""):
    """
    Reusable Manual Gating UI.
    Returns: vertices (list of tuples) or None
    """
    st.markdown(f"### {key_prefix} Gating: {x_channel} vs {y_channel}")
    if info_text:
        st.info(info_text)

    # Log Scale Toggle
    col_log, _ = st.columns([1, 4])
    with col_log:
        is_log = st.checkbox("Log Scale", value=True, key=f"{key_prefix}_log")

    # Prepare Data for Plotting
    df_viz = experiment.data.copy()
    
    # --- Density Calculation (Fast 2D Histogram Method) ---
    color_col = None
    try:
        if is_log:
             # Safe Log10 for density calculation
             x_data = df_viz[x_channel].copy()
             y_data = df_viz[y_channel].copy()
             # Replace 0 or negative with NaN for log
             x_data[x_data <= 0] = np.nan
             y_data[y_data <= 0] = np.nan
             x_data_log = np.log10(x_data)
             y_data_log = np.log10(y_data)
        else:
             x_data_log = df_viz[x_channel]
             y_data_log = df_viz[y_channel]

        # Remove infs/nans for calculation
        mask = np.isfinite(x_data_log) & np.isfinite(y_data_log)
        x_clean, y_clean = x_data_log[mask], y_data_log[mask]
        
        if len(x_clean) > 0:
            # Fast density via histogram
            H, xedges, yedges = np.histogram2d(x_clean, y_clean, bins=100)
            
            # Map density back to points
            x_bins = np.digitize(x_clean, xedges) - 1
            y_bins = np.digitize(y_clean, yedges) - 1
            
            x_bins = np.clip(x_bins, 0, H.shape[0]-1)
            y_bins = np.clip(y_bins, 0, H.shape[1]-1)
            
            # Assign density values
            df_viz['Density'] = 0.0
            df_viz.loc[mask, 'Density'] = H[x_bins, y_bins]
            color_col = 'Density'
    except Exception:
        color_col = None

    # Use a fraction of points for smoother plotting if large
    if len(df_viz) > 5000:
        df_plot = df_viz.sample(n=5000, random_state=42)
        st.caption(f"Plotting 5,000 representative events (out of {len(df_viz)}) for performance.")
    else:
        df_plot = df_viz

    # Setup Plot
    fig = px.scatter(
        df_plot, 
        x=x_channel, 
        y=y_channel, 
        color=color_col,
        color_continuous_scale='Viridis',
        opacity=0.8, 
        render_mode='svg',
        log_x=is_log,
        log_y=is_log,
        title=f"{key_prefix} Setup"
    )

    if color_col:
        fig.update_traces(marker=dict(size=8)) # Increased size for easier clicking
        fig.update_layout(coloraxis_colorbar=dict(title="Density"))

    # State Keys
    pending_key = f"{key_prefix}_pending"
    input_key = f"{key_prefix}_input"
    
    # Initialize pending if not exists
    if pending_key not in st.session_state:
        st.session_state[pending_key] = []

    # --- Pending Gate Visualization (Immediate Feedback) ---
    # Add traces for the polygon being drawn
    if st.session_state[pending_key]:
        p_verts = st.session_state[pending_key]
        # Close loop for visual
        p_path_x = [v[0] for v in p_verts] + [p_verts[0][0]]
        p_path_y = [v[1] for v in p_verts] + [p_verts[0][1]]
        
        fig.add_trace(go.Scatter(x=p_path_x, y=p_path_y, mode="lines+markers", 
                                line=dict(color="cyan", width=4, dash='dash'), 
                                marker=dict(size=8, color="cyan"),
                                name="New Selection (Pending)"))

    # Improved Plot Layout (White Background, Black Axes) matches reference
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black'), fixedrange=True),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black'), fixedrange=True),
        font=dict(color="black"),
        dragmode=False, # Disable dragging to make point click easier
        clickmode='event+select',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # 1. Render Plot
    selection = st.plotly_chart(
        fig, 
        use_container_width=True, 
        on_select="rerun", 
        selection_mode="points", 
        key=f"{key_prefix}_plot"
    )

    # 2. Logic to accumulate points (Exact copy of ui_gating.py logic)
    if selection and selection.get("selection") and selection["selection"].get("points"):
        pts = selection["selection"]["points"]
        count = len(pts)
        
        if count > 0:
            # Extract new points
            new_x = [p["x"] for p in pts]
            new_y = [p["y"] for p in pts]
            new_candidates = list(zip(new_x, new_y))
            
            # Get current pending points
            current_pending = st.session_state[pending_key]
            updated_pending = list(current_pending)
            
            has_changes = False
            for cand in new_candidates:
                # Check if close to existing
                is_close = False
                for ex in current_pending:
                    # Match tolerance from ui_gating.py
                    if np.isclose(cand[0], ex[0], atol=1e-5) and np.isclose(cand[1], ex[1], atol=1e-5):
                        is_close = True
                        break
                if not is_close:
                    updated_pending.append(cand)
                    has_changes = True
            
            if has_changes:
                # Sort if >= 3 to make polygon (Convex/Angular sort)
                final_pending = updated_pending
                if len(updated_pending) >= 3:
                    try:
                        arr = np.array(updated_pending)
                        cx, cy = np.mean(arr[:,0]), np.mean(arr[:,1])
                        angles = np.arctan2(arr[:,1] - cy, arr[:,0] - cx)
                        indices = np.argsort(angles)
                        final_pending = [updated_pending[i] for i in indices]
                    except: pass
                
                st.session_state[pending_key] = final_pending
                st.rerun()

    # 3. Actions / Confirmation
    if st.session_state[pending_key]:
        st.info(f"Selected {len(st.session_state[pending_key])} points (Polygon).")
        col1, col2 = st.columns(2)
        with col1:
             if st.button("Clear Points", key=f"{key_prefix}_clear"):
                st.session_state[pending_key] = []
                st.rerun()
    
    # Return vertices if they exist
    if len(st.session_state[pending_key]) >= 3:
        return st.session_state[pending_key]
    return None

def render_training_gui():
    st.title("Glow Training Data Generator")

    # --- Step 0: Setup / Folder Selection ---
    if "source_dir" not in st.session_state:
        st.session_state.source_dir = "training/source_fcs"
    
    col_src, col_btn = st.columns([4, 1])
    with col_src:
        source_input = st.text_input("Source FCS Folder", value=st.session_state.source_dir)
        st.session_state.source_dir = source_input
    
    # --- Step 1: Scan & Annotate ---
    if "fcs_df" not in st.session_state:
        st.session_state.fcs_df = pd.DataFrame(columns=["Filename", "FullPath", "CellType"])
    
    if st.button("Scan Folder"):
        if os.path.isdir(st.session_state.source_dir):
            files = []
            for root, _, filenames in os.walk(st.session_state.source_dir):
                for f in filenames:
                    if f.lower().endswith(".fcs"):
                        files.append(os.path.join(root, f))
            
            files = sorted(files)
            data = []
            for f in files:
                fname = os.path.basename(f)
                # Try to infer cell type from parent folder 
                parent = os.path.basename(os.path.dirname(f))
                
                # Heuristic: If folder seems to be a known cell line, use it. 
                # Otherwise default to 'hek' as requested, avoiding random folder names as types.
                if parent.lower() in ['snu', 'colo', 'hug1n', 'hug1']:
                    ctype = parent
                else:
                    ctype = "hek"
                
                data.append({"Filename": fname, "FullPath": f, "CellType": ctype})
            
            st.session_state.fcs_df = pd.DataFrame(data)
            st.session_state.step = "annotation"
        else:
            st.error("Invalid Directory")

    if st.session_state.get("step") == "annotation":
        st.subheader("1. Annotate Cell Types")
        st.info("Assign a Cell Type to each file. Files with the same Cell Type will be gated together.")
        
        edited_df = st.data_editor(
            st.session_state.fcs_df[["Filename", "CellType"]],
            disabled=["Filename"],
            num_rows="fixed",
            use_container_width=True,
            height=400
        )
        
        # Update session state with edits
        st.session_state.fcs_df.update(edited_df)
        
        if st.button("Confirm Annotations & Proceed"):
            # Clean up empty or Unknown
            final_df = st.session_state.fcs_df.copy()
            final_df = final_df[final_df["CellType"].str.strip() != ""]
            st.session_state.fcs_df = final_df
            st.session_state.unique_types = sorted(final_df["CellType"].unique().tolist())
            st.session_state.current_type_idx = 0
            st.session_state.step = "gating"
            st.rerun()

    # --- Step 2: Aggregated Gating ---
    if st.session_state.get("step") == "gating":
        unique_types = st.session_state.unique_types
        idx = st.session_state.current_type_idx
        
        if idx >= len(unique_types):
            st.success("All Cell Types Processed! 🎉")
            if st.button("Start Over"):
                st.session_state.clear()
                st.rerun()
            return

        current_type = unique_types[idx]
        
        # Sidebar Progress
        st.sidebar.title("Progress")
        st.sidebar.progress((idx) / len(unique_types))
        st.sidebar.write(f"Processing: **{current_type}**")
        st.sidebar.write(f"Queue: {unique_types[idx:]}")

        st.header(f"Step 2: Gating for '{current_type}'")
        
        # Get files for this type
        type_files = st.session_state.fcs_df[st.session_state.fcs_df["CellType"] == current_type]["FullPath"].tolist()
        st.write(f"Found {len(type_files)} files for {current_type}.")
        
        # Load Aggregate Data (session state cache)
        # We need a key that changes when type changes
        agg_key = f"agg_data_{current_type}"
        
        if agg_key not in st.session_state:
            with st.spinner(f"Loading representative data for {current_type}..."):
                # Load max 2000 events per file to keep it snappy, up to 100k total
                dfs = []
                # Limit input files if too many? No, we need representation.
                # Just sample small amount.
                sample_size = 2000 
                
                for fpath in type_files:
                    try:
                        tube = cytoflow.Tube(file=fpath)
                        # We can't easily partial load with cytoflow without full read first usually
                        # unless we use low level io. 
                        # Optimization: Read full, sample, discard.
                        import_op = cytoflow.ImportOp(tubes=[tube])
                        exp = import_op.apply()
                        
                        d = exp.data
                        if len(d) > sample_size:
                            d = d.sample(n=sample_size, random_state=42)
                        dfs.append(d)
                    except:
                        pass
                
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    # Create a dummy experiment wrapper for our UI functions
                    # We can use the last loaded experiment object structure just for metadata (channels)
                    agg_exp = exp.clone()
                    agg_exp.data = combined_df
                    st.session_state[agg_key] = agg_exp
                else:
                    st.error("Could not load any data.")
                    return

        agg_exp = st.session_state[agg_key]
        
        # --- Gating UI (Same as before but on aggregated data) ---
        
        # Channel Selection (assume consistent across type)
        cols = agg_exp.data.columns.tolist()
        fsc_a = next((c for c in cols if "FSC" in c and "-A" in c), cols[0])
        ssc_a = next((c for c in cols if "SSC" in c and "-A" in c), cols[1] if len(cols)>1 else cols[0])
        
        c1, c2 = st.columns(2)
        lx = c1.selectbox("X Channel (Live)", cols, index=cols.index(fsc_a) if fsc_a in cols else 0, key=f"lx_{current_type}")
        ly = c2.selectbox("Y Channel (Live)", cols, index=cols.index(ssc_a) if ssc_a in cols else 0, key=f"ly_{current_type}")
        
        live_verts = manual_gate_ui(agg_exp, lx, ly, f"Live_{current_type}", f"Define Live Gate for ALL {current_type} data.")
        
        if live_verts and len(live_verts) >= 3:
            st.success("Live Gate Defined.")
            
            # Filter aggregate for singlets
            gater = GatingManager()
            # We assume gater can work on this dummy experiment
            # But GatingManager might rely on standard flow. Let's replicate manual filter.
            # Actually apply_manual_gate creates a copy.
            
            # Apply to AGGREGATE for visualization
            live_agg = gater.apply_manual_gate(agg_exp, "TempLive", lx, ly, live_verts)
            if "TempLive" in live_agg.data.columns:
                 live_filtered_data = live_agg.data[live_agg.data["TempLive"]].copy()
            else:
                 live_filtered_data = live_agg.data
            
            live_filtered_exp = agg_exp.clone()
            live_filtered_exp.data = live_filtered_data

            st.divider()
            st.subheader("Singlet Gating")
            
            fsc_h = next((c for c in cols if "FSC" in c and "-H" in c), cols[0])
            c3, c4 = st.columns(2)
            sx = c3.selectbox("X Channel (Singlet)", cols, index=cols.index(fsc_a) if fsc_a in cols else 0, key=f"sx_{current_type}")
            sy = c4.selectbox("Y Channel (Singlet)", cols, index=cols.index(fsc_h) if fsc_h in cols else 0, key=f"sy_{current_type}")

            singlet_verts = manual_gate_ui(live_filtered_exp, sx, sy, f"Singlets_{current_type}", f"Define Singlet Gate for {current_type}.")

            if singlet_verts and len(singlet_verts) >= 3:
                st.success("Singlet Gate Defined.")
                
                st.divider()
                if st.button(f"🚀 Process & Save All '{current_type}' Files", type="primary"):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    # Create Output Dir
                    type_out_dir = os.path.join(OUTPUT_DIR, current_type)
                    os.makedirs(type_out_dir, exist_ok=True)
                    
                    gater = GatingManager() # Re-init
                    
                    for i, fpath in enumerate(type_files):
                        fname = os.path.basename(fpath)
                        status_text.text(f"Processing {i+1}/{len(type_files)}: {fname}")
                        
                        try:
                            # Load FULL file
                            tube = cytoflow.Tube(file=fpath)
                            import_op = cytoflow.ImportOp(tubes=[tube])
                            full_exp = import_op.apply()
                            
                            df_out = full_exp.data.copy()
                            
                            # Apply Gates
                            exp_L = gater.apply_manual_gate(full_exp, "Live_Flag", lx, ly, live_verts)
                            df_out['cells'] = exp_L.data['Live_Flag'].astype(bool)
                            
                            exp_S = gater.apply_manual_gate(full_exp, "Singlet_Flag", sx, sy, singlet_verts)
                            
                            # Ensure singlets are subset of cells (Requirement: no cells=FALSE and singlets=TRUE)
                            raw_singlets = exp_S.data['Singlet_Flag'].astype(bool)
                            df_out['singlets'] = raw_singlets & df_out['cells']
                            
                            # Save
                            # Format: {folder_name}_{fcs_file_name_no_ext}.csv.gz (spaces to underscores)
                            parent_folder = os.path.basename(os.path.dirname(fpath))
                            # Sanitize naming
                            safe_parent = parent_folder.replace(" ", "_")
                            
                            # Remove extension using splitext
                            fname_no_ext = os.path.splitext(fname)[0]
                            safe_fname = fname_no_ext.replace(" ", "_")
                            
                            out_name = f"{safe_parent}_{safe_fname}.csv.gz"
                            out_path = os.path.join(type_out_dir, out_name)
                            df_out.to_csv(out_path, index=False, compression='gzip')
                            
                        except Exception as e:
                            st.error(f"Failed to process {fname}: {e}")
                        
                        progress_bar.progress((i + 1) / len(type_files))
                    
                    st.success(f"Completed {current_type}!")
                    
                    # Advance State
                    st.session_state.current_type_idx += 1
                    # Clear memory
                    del st.session_state[agg_key] 
                    st.rerun()

    # Clear lingering states from previous runs to avoid confusion if needed



if __name__ == "__main__":
    render_training_gui()
