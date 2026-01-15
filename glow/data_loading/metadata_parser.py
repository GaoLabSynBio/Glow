import pandas as pd
import numpy as np

def parse_metadata_file(uploaded_file, header_row_idx, data_start_idx, global_exp_name_coords=None, df_raw=None):
    """
    Parses a CSV or Excel file into a DataFrame.

    Args:
        uploaded_file: The file object (from st.file_uploader or open()). Used if df_raw is None.
        header_row_idx: Index of the row containing headers.
        data_start_idx: Index of the row where data starts.
        global_exp_name_coords: Optional tuple (row, col) to extract experiment name.
        df_raw: Optional pre-loaded DataFrame (raw with no header).

    Returns:
        pd.DataFrame: The parsed metadata.
    """
    # 1. Read Raw
    if df_raw is None:
        if hasattr(uploaded_file, 'name') and uploaded_file.name.lower().endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, header=None)
        else:
            # Assumes Excel if not CSV, or let pandas handle it
            df_raw = pd.read_excel(uploaded_file, header=None)


    # 2. Extract Global Experiment Name if requested
    global_exp_name = None
    if global_exp_name_coords:
        g_row, g_col = global_exp_name_coords
        try:
            val = df_raw.iloc[g_row, g_col]
            if not pd.isna(val) and str(val).strip():
                global_exp_name = str(val)
        except Exception:
            pass

    # 3. Headers
    if header_row_idx >= len(df_raw):
         raise ValueError("Header row index out of bounds")

    header_series = df_raw.iloc[header_row_idx]
    headers = []
    for i, val in enumerate(header_series):
        if pd.isna(val) or str(val).strip() == "":
            headers.append(f"Untitled_{i}")
        else:
            headers.append(str(val))

    # Duplicate headers handling
    seen = {}
    final_headers = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            final_headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            final_headers.append(h)

    # 4. Data Slice
    df_parsed = df_raw.iloc[data_start_idx:].copy()

    # 5. Assign Headers
    if len(df_parsed.columns) == len(final_headers):
        df_parsed.columns = final_headers

    df_parsed.reset_index(drop=True, inplace=True)

    # 6. Apply Global Experiment Name
    if global_exp_name:
        # Determine if we overwrite or fillna.
        # For simplicity, if we extracted it globally, we probably want to apply it globally.
        df_parsed["Experiment_Name"] = global_exp_name

    return df_parsed
