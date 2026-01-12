
import sys

# Read file
lines = []
with open("/home/js/Documents/Glow/ui/gating/ui_gating.py", "r") as f:
    lines = f.readlines()

new_lines = []
step1_active = False

for i, line in enumerate(lines):
    # Detect CSV block end (Line 50 is st.selectbox key="csv_cmap")
    # Line 51 is blank.
    # Line 52 is `else:` or start of main block.
    
    # Logic:
    # 1. Start writing `if st.session_state.gating_step == 1:` around line 53.
    # 2. Indent everything until line 814.
    # 3. Replace lines 814+ with new logic.
    
    # START OF STEP 1
    if "st.write(\"Configuring Gating...\")" in line:
        # This is where standard gating starts.
        new_lines.append("        if st.session_state.gating_step == 1:\n")
        new_lines.append("            " + line) # Indented
        step1_active = True
        continue
            
    # END OF STEP 1 / START OF STEP 2
    if "# --- Step 3: Singlet Detection ---" in line:
        step1_active = False
        new_lines.append(line) 
        continue

    if "if st.session_state.get('gate_accepted'):" in line:
        # REPLACE with Step 2 check
        new_lines.append(line.replace("st.session_state.get('gate_accepted')", "st.session_state.gating_step == 2"))
        
        # ADD BACK BUTTON for Step 2
        # We need to insert it inside the block.
        # But here we just write the header. The indentation of the next lines is already correct for the block.
        # We will inject the button code at the start of the block.
        new_lines.append("                        # Wizard Navigation: Back Button\n")
        new_lines.append("                        st.button('⬅ Back to Live Gating', on_click=lambda: st.session_state.update(gating_step=1))\n")
        continue

    # START OF STEP 3
    if "if st.session_state.get('doublets_removed'):" in line:
        # REPLACE with Step 3 check
        new_lines.append(line.replace("st.session_state.get('doublets_removed')", "st.session_state.gating_step == 3"))
        
        # ADD BACK BUTTON for Step 3
        new_lines.append("                            st.button('⬅ Back to Singlet Detection', on_click=lambda: st.session_state.update(gating_step=2))\n")
        continue

    # INDENTATION FOR STEP 1
    if step1_active:
        # Original indentation is 8 spaces (inside else).
        # We need to add 4 spaces -> 12 spaces.
        if line.strip():
            new_lines.append("    " + line)
        else:
            new_lines.append(line) # Keep blank lines blank
    else:
        new_lines.append(line)

# Write back
with open("/home/js/Documents/Glow/ui/gating/ui_gating.py", "w") as f:
    f.writelines(new_lines)

print("Refactor complete.")
