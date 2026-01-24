
import sys
import os
import traceback
import re
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

def detect_cell_line_from_path(path):
    if not path:
        return "HEK"
    filename = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))
    full_str = f"{dirname}_{filename}"
    match = re.search(r'(?:cell[_\-]?line|cl)[_\-]?([a-zA-Z0-9]+)', full_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    known_lines = ["colo", "hek", "hug1n", "snu", "b16", "cho", "hela", "jurkat"]
    for line in known_lines:
        if line.lower() in full_str.lower():
            return line.upper() # Double upper from original code? "line.upper().upper()"
    return "HEK"

try:
    from glow.data_loading.loader import DataLoader
    
    loader = DataLoader()
    
    # Test 1: Scan Directory
    dir_path = "/Users/acheron/Desktop/Glow/Exp_20260121_1"
    print(f"Scanning {dir_path}...")
    files = loader.scan_directory(dir_path)
    print(f"Found {len(files)} files.")
    for f in files[:3]:
        print(f" - {f}")

    if not files:
        print("Scanned 0 files. Exiting.")
        sys.exit(0)

    # Test 2: Generate Preview
    print("\nGenerating Preview...")
    meta = loader.generate_preview_metadata(files)
    
    # Simulate UI Logic
    for m in meta:
        m['Cell_Line'] = detect_cell_line_from_path(m.get('original_path'))
        
    print(f"Generated {len(meta)} metadata entries.")
    print("Sample Metadata 0:", meta[0])
    
    # Test 3: Load Individually to check columns
    print("\nChecking Column Consistency...")
    from glow.data_loading.loader import DataLoader
    # Create temp loader for individual file inspection
    # We use cytoflow directly or simple read to get columns
    import cytoflow
    
    first_cols = None
    mismatch_found = False
    
    for i, meta_entry in enumerate(meta):
        path = meta_entry.get('current_path', meta_entry.get('original_path')) 

        try:
            # We can use a lightweight read if possible, but load_as_cytoflow logic uses Tube
            # Let's just use the loader's method for a single file subset
            single_meta = [meta_entry]
            # Mock the process_files output structure if needed
            if 'current_path' not in meta_entry:
                 meta_entry['current_path'] = meta_entry['original_path']
            
            # Use Cytoflow directly to peek columns
            tube = cytoflow.Tube(file=path)
            # We need to act like import_op to get the data
            op = cytoflow.ImportOp(tubes=[tube])
            ex = op.apply()
            cols = list(ex.data.columns)
            cols.sort()
            
            if first_cols is None:
                first_cols = cols
                print(f"File 1 Template ({os.path.basename(path)}): {len(cols)} columns")
            else:
                if cols != first_cols:
                    print(f"MISMATCH in {os.path.basename(path)}!")
                    # Find diff
                    s1 = set(first_cols)
                    s2 = set(cols)
                    print(f" - Missing: {s1 - s2}")
                    print(f" - Extra: {s2 - s1}")
                    mismatch_found = True
                    
        except Exception as e:
            print(f"Error reading {os.path.basename(path)}: {e}")

    if not mismatch_found:
        print("All files have identical column sets (sorted).")
    else:
        print("Column mismatches detected.")

    # Test 4: Full Load
    print("\nAttempting Full Load...")


except Exception as e:
    print(f"\nCaught Exception:")
    traceback.print_exc()
