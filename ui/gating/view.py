import streamlit as st
import numpy as np
import pandas as pd
from .ui_file_loader import render_file_loader
from .ui_gating import render_gating_section

def render_gating_tab(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE):
    st.header("Gating Workflow")
    
    # 1. File Loading Module
    render_file_loader(DataLoader, CYTOFLOW_AVAILABLE)

    # 2. Gating Module
    render_gating_section(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE)
    
    st.divider()
    col1, col2 = st.columns([3, 1])
    
    # Initialize persistent selected channels if not present
    if 'persistent_selected_channels' not in st.session_state and 'base_experiment' in st.session_state and st.session_state.base_experiment:
        all_ch = list(st.session_state.base_experiment.channels)
        # Default to fluorophores (exclude scatter/time)
        defaults = [c for c in all_ch if not any(x in c.upper() for x in ["FSC", "SSC", "TIME"])]
        if not defaults: defaults = all_ch
        st.session_state.persistent_selected_channels = defaults

    with col1:
        if 'base_experiment' in st.session_state and st.session_state.base_experiment is not None:
            all_channels = st.session_state.base_experiment.channels
            st.subheader("Final Data Settings")
            st.info("Select fluorescence channels to include in your analysis. Unselected channels will be hidden.")
            
            # Action buttons for quick management
            b1, b2, _ = st.columns([1, 1, 2])
            if b1.button("Select All", key="gating_sel_all"):
                st.session_state.persistent_selected_channels = list(all_channels)
                st.rerun()
            if b2.button("Clear All", key="gating_clear_all"):
                st.session_state.persistent_selected_channels = []
                st.rerun()

            # Dynamic Checkbox Selection (Click 1 to Include, Click again to Delete/Exclude)
            current_selected = set(st.session_state.get('persistent_selected_channels', []))
            new_selection = []
            
            # Display channels in a structured grid for easier selection
            with st.expander("Available Channels List", expanded=True):
                n_cols = 3
                grid_cols = st.columns(n_cols)
                for i, ch in enumerate(all_channels):
                    with grid_cols[i % n_cols]:
                        # A checkbox is the most literal "Click to Include / Click to Delete" UI
                        if st.checkbox(ch, value=(ch in current_selected), key=f"checkbox_ch_{ch}"):
                            new_selection.append(ch)
            
            # Sync the final selection back to session state
            st.session_state.persistent_selected_channels = new_selection
            st.session_state.selected_channels = new_selection
        else:
            st.info("Load data to enable channel selection.")

    with col2:
        if st.button("Finish Gating & Proceed to Analysis", type="primary"):
            # Check if critical data exists
            if 'base_experiment' in st.session_state and st.session_state.base_experiment is not None:
                
                # --- Reconstruct Fully Annotated Experiment for Analysis ---
                # We need to apply the specific gates to the ORIGINAL dataset to get boolean flags ('cells', 'singlets')
                # instead of filtering rows. This allows correct calculation of Live % and Singlets %.
                
                with st.spinner("Finalizing analysis dataset..."):
                    try:
                        # 1. Start with Clone of Base
                        base = st.session_state.base_experiment
                        full_exp = base.clone() if hasattr(base, 'clone') else base 
                        
                        # Note: We NO LONGER drop channels here to keep the full dataset intact.
                        # The channel selection will be used to filter UI options in Analysis and Plot tabs.

                        # Initialize flags to False if not present (safeguard)
                        # We use 'cells' and 'singlets' as the standard flags
                        # If columns don't exist, we add them as boolean False generally?
                        # Actually, better to compute them fresh.
                        
                        gater = GatingManager()
                        
                        # 2. Re-Apply LIVE Gate
                        l_params = st.session_state.get('live_gate_params')
                        live_mask_col = None
                        
                        if l_params:
                            if l_params['type'] == 'polygon':
                                # Check if manual vertices exist
                                full_exp = gater.apply_manual_gate(full_exp, "LiveGate", l_params['channels'][0], l_params['channels'][1], l_params['vertices'])
                                live_mask_col = "LiveGate"
                                
                            elif l_params['type'] == 'label':
                                # Re-run auto gating to get labels if needed
                                # Note: If AutoLiveGate is already in base, we good.
                                # Check if AutoLiveGate exists in full_exp
                                if 'AutoLiveGate' not in full_exp.data.columns:
                                    gp = st.session_state.get('last_gating_params', {})
                                    if gp:
                                        full_exp = gater.auto_gate_live_cells(
                                            full_exp,
                                            x_channel=gp.get('x'),
                                            y_channel=gp.get('y'),
                                            method=gp.get('method', 'density').lower() if isinstance(gp.get('method'), str) else 'density',
                                            num_components=gp.get('comp', 2),
                                            keep=gp.get('keep', 0.9),
                                            sigma=gp.get('sigma', 2.0),
                                            eps=gp.get('eps', 0.05),
                                            min_samples=gp.get('min_samples', 50),
                                            use_outlier_removal=gp.get('use_outliers', False)
                                        )
                                # Define mask from label
                                lbl = l_params['label']
                                full_exp.data['cells'] = (full_exp.data['AutoLiveGate'] == lbl)
                            
                        # If we have a named gate column, map it to 'cells'
                        if live_mask_col and live_mask_col in full_exp.data.columns:
                             full_exp.data['cells'] = full_exp.data[live_mask_col].astype(bool)
                        elif 'cells' not in full_exp.data.columns:
                             # Fallback: Assume all live if no gate? Or False?
                             # If file loaded via CSV bypass, 'cells' might already be there.
                             if 'cells' not in full_exp.data.columns: # Check again
                                 full_exp.data['cells'] = False
                        
                        # 3. Re-Apply SINGLET Gate
                        s_params = st.session_state.get('singlet_gate_params')
                        
                        if s_params:
                            if s_params['type'] == 'polygon':
                                full_exp = gater.apply_manual_gate(full_exp, "SingletGate", s_params['channels'][0], s_params['channels'][1], s_params['vertices'])
                                if 'SingletGate' in full_exp.data.columns:
                                    full_exp.data['singlets'] = full_exp.data['SingletGate'].astype(bool)
                            
                            elif s_params['type'] == 'gmm_mahalanobis':
                                # Calculate manually on full data
                                X = full_exp.data[s_params['channels']].values
                                d_mean = s_params['mean']
                                inv_cov = np.linalg.inv(s_params['cov'])
                                threshold = s_params.get('threshold', 2.5)
                                
                                diff = X - d_mean
                                # dist = sqrt( (x-mu)T . Sigma^-1 . (x-mu) )
                                # Batched: sum( (diff @ inv_cov) * diff, axis=1 )
                                left = diff @ inv_cov
                                dist_sq = np.sum(left * diff, axis=1)
                                dist = np.sqrt(dist_sq)
                                
                                full_exp.data['singlets'] = (dist <= threshold)
                        
                        # Ensure 'singlets' exists
                        if 'singlets' not in full_exp.data.columns:
                            full_exp.data['singlets'] = False

                        # Save to session
                        st.session_state.analysis_experiment = full_exp
                        st.session_state.gating_complete = True
                        st.session_state.current_tab = "Analysis"
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error preparing analysis data: {e}")
            else:
                st.error("Please load data and perform gating before proceeding.")
