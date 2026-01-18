import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import io
from glow.compensation.manager import CompensationManager

def get_running_mean(df, x_col, y_col, bins=50):
    try:
        # Create bins
        # Handle log scale visualization often used in flow
        # But for rolling mean calculation, usually done on linear or log-transformed data directly.
        # Assuming data in df is what we are plotting (likely log-transformed for display if that's the mode, 
        # but pure data for compensation is usually linear area).
        # Compensation is linear algebra on LINEAR data. 
        # Visualization usually Logicle/Log. 
        # We will compute running mean on the data provided (which comes from experiment.data, usually linear for 'area' channels?)
        # Wait, usually Cytoflow/Glow keeps data linear until plotted.
        
        # We'll just do simple binning on min-max range
        x_min, x_max = df[x_col].min(), df[x_col].max()
        if x_max == x_min:
            return np.array([x_min]), np.array([df[y_col].mean()])
            
        bins_edges = np.linspace(x_min, x_max, bins)
        # digitize
        bin_idxs = np.digitize(df[x_col], bins_edges)
        
        means_x = []
        means_y = []
        for i in range(1, len(bins_edges)):
            mask = bin_idxs == i
            if np.any(mask):
                means_x.append(df[x_col][mask].mean())
                means_y.append(df[y_col][mask].mean())
                
        return means_x, means_y
    except Exception as e:
        print(f"Running mean failed: {e}")
        return [], []

def render_pairwise_plots(df, channels, suffix="", title_prefix=""):
    """
    Render pairwise plots (triangle)
    """
    import itertools
    pairs = list(itertools.combinations(channels, 2))
    
    if not pairs:
        st.warning("Need at least 2 channels for pairwise plots.")
        return

    # Limit rows to avoid crashing browser
    if len(pairs) > 6:
        st.info(f"Showing first 6 pairs of {len(pairs)}.")
        pairs = pairs[:6]
        
    cols = st.columns(2)
    
    for i, (chX, chY) in enumerate(pairs):
        col = cols[i % 2]
        with col:
            # Downsample for plotting speed
            plot_df = df.sample(min(2000, len(df))) if len(df) > 2000 else df
            
            x_label = f"{chX}{suffix}"
            y_label = f"{chY}{suffix}"
            
            # Simple Scatter
            fig = px.scatter(plot_df, x=chX, y=chY, opacity=0.3, 
                             title=f"{title_prefix} {chX} vs {chY}",
                             render_mode='svg') # svg for speed on small points? or webgl
            
            # Running Mean
            rm_x, rm_y = get_running_mean(plot_df, chX, chY)
            if len(rm_x) > 0:
                fig.add_trace(go.Scatter(x=rm_x, y=rm_y, mode='lines', 
                                         line=dict(color='red', width=2),
                                         name='Running Mean'))
            
            # Layout log if necessary? Usually data is large. 
            # If values > 1000, probably reasonable to log for Viz
            if plot_df[chX].max() > 1000 or plot_df[chY].max() > 1000:
                fig.update_xaxes(type="log")
                fig.update_yaxes(type="log")
                
            st.plotly_chart(fig, use_container_width=True)


def render_compensation_tab():
    st.header("Compensation")

    # Use session state to hold manager
    if 'comp_manager' not in st.session_state:
        st.session_state.comp_manager = CompensationManager()

    # Get Data
    if 'base_experiment' not in st.session_state or st.session_state.base_experiment is None:
        st.error("No data loaded. Go back to Gating tab.")
        return

    exp = st.session_state.base_experiment
    df = exp.data
    
    # 1. Setup Panel
    with st.expander("1. Compensation Setup", expanded=True):
        # Identify Fluorophores
        
        COMMON_FLUORS = [
            "GFP", "EGFP", "mCherry", "Cherry", "APC", "FITC", "PE", "DAPI", "PerCP", 
            "BV", "Brilliant", "Cy", "Cy3", "Cy5", "Cy7", "BFP", "EBFP", "TagBFP", "SB", 
            "RFP", "dsRed", "tdTomato", "YFP", "EYFP", "mNeon", "AmCyan", "Pacific Blue", 
            "Pacific Orange", "Alexa", "eFluor", "Hoechst", "PI", "7-AAD", "Zombie", "Live/Dead", "DraQ"
        ]

        all_channels = list(df.columns)
        
        def is_target_fluorophore(c):
             # 1. Must contain a common fluorophore name
             has_name = any(f.upper() in c.upper() for f in COMMON_FLUORS)
             # 2. Must end in -A (Area) (User Request: "fluorophore-A")
             is_area = c.endswith("-A")
             # 3. Must exclude Scatter/Time just in case
             not_scatter = not any(x in c.upper() for x in ['FSC', 'SSC', 'TIME'])
             
             return has_name and is_area and not_scatter

        fluorophores = [c for c in all_channels if is_target_fluorophore(c)]
        
        selected_channels = st.multiselect("Select Fluorophores to Compensate", 
                                           options=fluorophores,
                                           default=fluorophores[:2] if len(fluorophores) >=2 else fluorophores)
        
        if len(selected_channels) < 2:
            st.warning("Select at least 2 fluorophores.")
            # stop?
        
        st.write("#### Assign Controls")
        
        # Get List of Samples
        if 'Sample' in df.columns:
            samples = sorted(df['Sample'].unique().astype(str))
        else:
            samples = ["All Events"] 
            st.warning("No 'Sample' column found. Cannot distinguish controls.")
        
        # Interactive Layout
        c1, c2 = st.columns(2)
        with c1:
            unstained_sample = st.selectbox("Unstained (Blank) Control", options=samples)
        
        controls_map = {}
        with c2:
            st.write("Single-Color Controls:")
            for ch in selected_channels:
                # Try to auto-guess based on name
                # e.g. if Channel is FITC, look for "FITC" in sample name
                default_idx = 0
                for i, s in enumerate(samples):
                    if ch in s or s in ch: # simple fuzzy
                        default_idx = i
                        break
                        
                sel = st.selectbox(f"Control for {ch}", options=samples, index=default_idx, key=f"sel_comp_{ch}")
                controls_map[ch] = {'Sample': sel}

        if st.button("Calculate Compensation Matrix", type="primary"):
            with st.spinner("Calculating Spillover..."):
                try:
                    unstained_spec = {'Sample': unstained_sample}
                    matrix = st.session_state.comp_manager.compute_spillover_matrix(
                        exp, unstained_spec, controls_map, selected_channels
                    )
                    st.session_state.spillover_matrix = matrix
                    st.session_state.comp_channels = selected_channels
                    st.success("Calculation Complete!")
                except Exception as e:
                    st.error(f"Calculation Failed: {e}")

    # 2. Results & Apply
    if st.session_state.get('spillover_matrix') is not None:
        st.divider()
        st.subheader("2. Review & Apply")
        
        matrix = st.session_state.spillover_matrix
        channels = st.session_state.comp_channels
        
        t1, t2 = st.tabs(["Spillover Matrix", "Pairwise Visualization"])
        
        with t1:
            st.dataframe(matrix.style.background_gradient(cmap='Reds', axis=None).format("{:.2%}"))
            st.caption("Rows: Source (Control) | Columns: Detector (Spillover)")
            
        with t2:
            st.write("#### Before Compensation")
            # We need to construct a df with just the control data? 
            # Or the whole dataset? "before and after compensation"
            # Usually we visualize the CONTROLS to see if they are aligned?
            # Or the user wants to see the whole dataset to see if populations orthogonalize?
            # Prompt: "plot pairwise... before and after ... apply compensation to all data"
            # It implies using "All Data" for visualization? 
            # Or usually we check if the SINGLE STAINS are compensated correctly.
            # Let's show the full dataset (downsampled) so they can see the correction on real cells.
            
            show_sample = st.selectbox("Select Sample to Visualize", options=["All Combined"] + list(df['Sample'].unique().astype(str)))
            
            if show_sample == "All Combined":
                viz_df = df
            else:
                viz_df = df[df['Sample'] == show_sample]
            
            st.write("##### Uncompensated")
            render_pairwise_plots(viz_df, channels, title_prefix="Raw")
            
            st.write("##### Compensated Preview")
            # Apply temporarily for viz
            comp_viz_df = st.session_state.comp_manager.apply_compensation(viz_df, channels, rename=False)
            render_pairwise_plots(comp_viz_df, channels, title_prefix="Compensated")


        st.divider()

        # Add optional export button here
        st.subheader("3. Export Compensated Data")
        if st.checkbox("Enable Export"):
            
            export_scope = st.radio("Export Scope", 
                                    ["Full Data (All Events)", "Gated Population (Live & Singlets Only)"],
                                    index=0)
            
            if st.button("Prepare Download"):
                with st.spinner("Processing & Zipping..."):
                    try:
                        # 1. Start with Analysis Experiment (Has Gating Flags)
                        # We use analysis_experiment because it contains the 'cells' and 'singlets' columns generated during the Proceed step.
                        # Using base_experiment would lose this gating information.
                        if 'analysis_experiment' in st.session_state:
                             temp_exp = st.session_state.analysis_experiment.clone()
                        else:
                             # Fallback (Gating flags will be missing)
                             temp_exp = st.session_state.base_experiment.clone()
                             st.warning("Analysis experiment not found. Gating flags (cells/singlets) may be missing.")
                        
                        # 2. Apply Compensation (Creates new columns)
                        # Use the channels selected in Step 1
                        comp_channels = st.session_state.comp_channels
                        temp_exp.data = st.session_state.comp_manager.apply_compensation(
                            temp_exp.data, comp_channels, rename=True
                        )
                        
                        df_out = temp_exp.data.copy()
                        
                        # 3. Filter Rows (Gating)
                        if "Gated" in export_scope:
                            # Check for flags
                            if 'cells' in df_out.columns and 'singlets' in df_out.columns:
                                df_out = df_out[df_out['cells'] & df_out['singlets']]
                            else:
                                st.warning("Gating flags (cells/singlets) not found. Exporting all events.")
                        
                        # 4. Filter Columns (Selection + Internal Cleanup)
                        # Logic A: Drop internal columns
                        internal_cols = [c for c in df_out.columns if c.startswith('_temp_') or c in ['urows', 'quality']]
                        if internal_cols:
                            df_out.drop(columns=internal_cols, inplace=True)
                            
                        # Logic B: Drop unselected raw channels (matching Gating Panel logic)
                        # But KEEP the compensated versions of selected channels
                        user_selected = st.session_state.get('selected_channels')
                        
                        if user_selected:
                            # Identify what to keep: 
                            # 1. Structural/Metadata (Sample, Condition, Time, FSC, SSC)
                            # 2. User Selected Fluorophores
                            # 3. Compensated versions of User Selected Fluorophores
                            
                            # It's easier to identify what to DROP:
                            # Any channel in 'base.channels' that is NOT in 'user_selected'
                            
                            # Get known acquisition channels from metadata if available, or heuristic
                            all_cols = list(df_out.columns)
                            # We assume columns ending in -A, -H, -W are channels, others metadata
                            # Or rely on 'persistent_selected_channels' logic
                            
                            keep_set = set(user_selected)
                            
                            # Check every column
                            to_drop = []
                            for col in all_cols:
                                # Skip metadata/structural
                                if col in ['Sample', 'Condition', 'Time', 'cells', 'singlets', 'Is_Live', 'Is_Singlet']:
                                    continue
                                if any(x in col for x in ['FSC', 'SSC']):
                                    continue
                                if col.startswith('comp_'):
                                    # Keep if the base channel is selected?
                                    base_name = col.replace('comp_', '')
                                    if base_name not in keep_set:
                                        to_drop.append(col)
                                    continue
                                    
                                # It's a raw channel?
                                # If it looks like a channel (float/int) and NOT in keep_set
                                if col not in keep_set and df_out[col].dtype.kind in 'fi':
                                    to_drop.append(col)
                                    
                            if to_drop:
                                df_out.drop(columns=to_drop, inplace=True)

                        # 5. Create ZIP with Single CSV
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            # Single CSV for all samples
                            csv_data = df_out.to_csv(index=False)
                            zf.writestr("compensated_data.csv", csv_data)
                                
                        st.download_button(
                            label="Download Compensated Data (.zip)",
                            data=zip_buffer.getvalue(),
                            file_name="compensated_data.zip",
                            mime="application/zip"
                        )
                        
                    except Exception as e:
                        st.error(f"Export failed: {e}")

        st.divider()
        if st.button("✅ Accept & Apply to Analysis Dataset", type="primary"):
            with st.spinner("Applying Compensation..."):
                target_exp = st.session_state.get('analysis_experiment')
                if target_exp is None:
                    # Fallback if accessed directly (shouldn't happen in flow)
                    target_exp = st.session_state.base_experiment.clone()
                
                # Apply
                new_df = st.session_state.comp_manager.apply_compensation(target_exp.data, channels, rename=True)
                target_exp.data = new_df
                
                # Update Metadata to reflect new columns?
                for ch in channels:
                    new_col = f"comp_{ch}"
                    # Inherit metadata from original channel
                    if ch in target_exp.metadata:
                        target_exp.metadata[new_col] = target_exp.metadata[ch].copy()
                        
                        
                st.session_state.analysis_experiment = target_exp
                st.session_state.run_compensation = False # Done
                st.session_state.current_tab = "Analysis"
                
                # Let's refresh so they see the new Analysis tab
                st.rerun()


