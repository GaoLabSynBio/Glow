import cytoflow
import os
import shutil
import pandas as pd
import re
import zipfile
import pyarrow.parquet as pq
import numpy as np

from ..logging_config import get_logger
from .. import paths as glow_paths

log = get_logger("loader")

class ParquetBatchLoader:
    def __init__(self, file_path, batch_size=100000, columns=None, row_group_indices=None):
        self.file_path = file_path
        self.batch_size = batch_size
        self.columns = columns
        self.parquet_file = pq.ParquetFile(file_path)
        self.num_row_groups = self.parquet_file.num_row_groups
        self.total_rows = self.parquet_file.metadata.num_rows

        if row_group_indices is not None:
            self.row_group_indices = row_group_indices
        else:
            self.row_group_indices = range(self.num_row_groups)

    def __iter__(self):
        # Iterate over row groups.
        # Note: A single row group might be larger or smaller than batch_size.
        # For strict memory control, we could read row groups and yield chunks.
        # But iteratively reading row groups is the standard zero-copy(ish) way.
        for i in self.row_group_indices:
            table = self.parquet_file.read_row_group(i, columns=self.columns)
            yield table

class DataLoader:
    def __init__(self):
        """
        Initialize the DataLoader.
        It can hold references to loaded files.
        """
        self.loaded_files_metadata = []

    @staticmethod
    def sanitize_key(name):
        import re
        import numpy as np
        # Replace invalid chars with underscore
        clean = re.sub(r'[^0-9a-zA-Z_]', '_', str(name))
        # Cannot start with number
        if clean and clean[0].isdigit():
            clean = "_" + clean
        # Ensure not empty
        if not clean:
            clean = "condition_" + str(np.random.randint(1000))
        return clean

    def _natural_sort_key(self, text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(text))]

    def save_uploaded_files(self, uploaded_files, target_dir=None, folder_name=None):
        if target_dir is None:
            target_dir = str(glow_paths.uploads_dir())
        if folder_name:
            folder_name = "".join(x for x in folder_name if (x.isalnum() or x in "._-"))
            target_dir = os.path.join(target_dir, folder_name)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        saved_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(target_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(os.path.abspath(file_path))

        saved_paths.sort(key=self._natural_sort_key)
        return saved_paths

    @staticmethod
    def _safe_extract(zip_ref, target_dir):
        """Extract a zip while rejecting members that would escape target_dir.

        Guards against the "Zip Slip" path-traversal attack, where an archive
        entry named e.g. ``../../evil`` writes outside the intended directory.
        """
        dest_root = os.path.realpath(target_dir)
        for member in zip_ref.namelist():
            member_path = os.path.realpath(os.path.join(target_dir, member))
            if member_path != dest_root and not member_path.startswith(dest_root + os.sep):
                raise ValueError(f"Unsafe path in zip archive (path traversal): {member!r}")
        zip_ref.extractall(target_dir)

    def save_uploaded_folder_zip(self, uploaded_zip, target_dir=None):
        if target_dir is None:
            target_dir = str(glow_paths.user_data_dir() / "uploaded_raw_zip")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir) # Clean up previous uploads
        os.makedirs(target_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                self._safe_extract(zip_ref, target_dir)
        except zipfile.BadZipFile:
            log.error("Invalid zip file uploaded")
            return []
        except ValueError as e:
            log.error("Rejected unsafe zip archive: %s", e)
            return []

        saved_paths = []
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.lower().endswith('.fcs'):
                    saved_paths.append(os.path.abspath(os.path.join(root, file)))

        saved_paths.sort(key=self._natural_sort_key)
        return saved_paths

    def scan_directory(self, directory_path):
        paths = []
        if not os.path.isdir(directory_path):
            log.warning("Directory not found: %s", directory_path)
            return paths

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.fcs'):
                    full_path = os.path.join(root, file)
                    paths.append(full_path)

        paths.sort(key=self._natural_sort_key)
        return paths

    @staticmethod
    def clear_upload_cache(target_dir=None):
        """
        Clears the upload cache directory to free up space.
        """
        if target_dir is None:
            target_dir = str(glow_paths.uploads_dir())
        if os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
                os.makedirs(target_dir, exist_ok=True)
                log.info("Cache cleared: %s", target_dir)
            except Exception as e:
                log.warning("Failed to clear cache %s: %s", target_dir, e)

    def parse_file_paths_from_text(self, file_path_list_file):
        paths = []
        if not os.path.exists(file_path_list_file):
            raise FileNotFoundError(f"File list not found: {file_path_list_file}")

        with open(file_path_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(',')
                    path = parts[0].strip()
                    custom_name = parts[1].strip() if len(parts) > 1 else None

                    if os.path.exists(path):
                        paths.append({'path': path, 'name': custom_name})
                    else:
                        log.warning("File not found at %s", path)

        paths.sort(key=lambda x: self._natural_sort_key(x['path']))

        return paths

    def generate_preview_metadata(self, paths, rename_strategy='folder_name'):
        preview_metadata = []
        for item in paths:
            if isinstance(item, dict):
                path = item['path']
                custom_name = item.get('name')
            else:
                path = str(item)
                custom_name = None

            if not os.path.exists(path):
                continue

            dirname = os.path.basename(os.path.dirname(path))
            filename = os.path.basename(path)

            if not filename.lower().endswith('.fcs'):
                continue

            if custom_name:
                new_name = custom_name
                if not new_name.lower().endswith('.fcs'):
                    new_name += '.fcs'
            elif rename_strategy == 'folder_name':
                name_prefix = dirname.replace(" ", "_")
                new_name = f"{name_prefix}_{filename}"
            else:
                new_name = filename

            file_meta = {
                'original_path': path,
                'display_name': new_name,
                'condition': dirname
            }
            preview_metadata.append(file_meta)
        return preview_metadata

    def process_files(self, metadata_list, output_dir=None):
        self.loaded_files_metadata = []

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        internal_keys = ['original_path', 'current_path', 'path', 'display_name', 'condition']

        for meta in metadata_list:
            original = meta['original_path']
            new_name = meta['display_name']

            final_path = original

            if output_dir:
                dest_path = os.path.join(output_dir, new_name)
                try:
                    shutil.copy2(original, dest_path)
                    final_path = dest_path
                except Exception as e:
                    log.error("Error copying %s: %s", original, e)

            # Sanitize metadata keys immediately upon loading/processing
            sanitized_meta = {}
            for k, v in meta.items():
                if k in internal_keys:
                    sanitized_meta[k] = v
                else:
                    clean_k = self.sanitize_key(k)
                    sanitized_meta[clean_k] = v

            sanitized_meta['current_path'] = final_path
            self.loaded_files_metadata.append(sanitized_meta)

        return self.loaded_files_metadata

    def prepare_fcs_files(self, paths, rename_strategy='folder_name', output_dir=None):
        meta = self.generate_preview_metadata(paths, rename_strategy)
        return self.process_files(meta, output_dir)

    def _detect_and_tag_high_res(self, experiment):
        """
        Detects if the experiment data appears to be from a high-resolution cytometer
        (e.g., BioRad ZE5 with 32-bit integers) and tags metadata accordingly.
        """
        if experiment is None or experiment.data is None:
            return experiment
        
        # Heuristic: Check numeric columns for very large values
        # Typical 18-bit flow data maxes around 262,144.
        # 32-bit signed int max is ~2.14e9.
        # If max > 1,000,000 (10^6), flag it.
        threshold = 1000000 
        
        numeric_cols = experiment.data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            return experiment

        max_vals = experiment.data[numeric_cols].max()
        if (max_vals > threshold).any():
            log.info("Detected high-resolution data. Tagging metadata.")
            new_meta = experiment.metadata.copy() if experiment.metadata else {}
            new_meta['is_high_res'] = True
            new_meta['scaling_factor'] = 10000.0
            experiment.metadata = new_meta
            
        return experiment

    def load_as_cytoflow_experiment(self, metadata_list=None):
        if metadata_list is None:
            metadata_list = self.loaded_files_metadata

        if not metadata_list:
            log.info("No files to load.")
            return None

        internal_keys = ['original_path', 'current_path', 'path', 'display_name', 'condition']

        all_condition_keys = set()
        all_condition_keys.add('Sample')
        all_condition_keys.add('Condition')

        # Collect all unique keys (sanitized)
        for meta in metadata_list:
             for key in meta.keys():
                 if key not in internal_keys:
                     clean_key = self.sanitize_key(key)
                     all_condition_keys.add(clean_key)

        log.debug("All condition keys: %s", all_condition_keys)

        import_conditions_schema = {}
        for key in all_condition_keys:
            import_conditions_schema[key] = "category"

        tubes = []
        for meta in metadata_list:
            tube_conditions = {}
            # Always ensure Sample and Condition are present
            tube_conditions['Sample'] = str(meta.get('display_name', 'Unknown'))

            if 'condition' in meta and meta['condition']:
                tube_conditions['Condition'] = str(meta['condition'])
            else:
                 tube_conditions['Condition'] = "Unspecified"

            # Populate other conditions
            for orig_key, val in meta.items():
                if orig_key in internal_keys:
                    continue

                clean_key = self.sanitize_key(orig_key)
                if clean_key in ['Sample', 'Condition']:
                    continue

                if val is not None and str(val).strip() != "":
                    tube_conditions[clean_key] = str(val)

            # Fill missing keys with NA
            for key in import_conditions_schema.keys():
                if key not in tube_conditions:
                    tube_conditions[key] = "NA"

            try:
                tube = cytoflow.Tube(file=meta['current_path'], conditions=tube_conditions)
                tubes.append(tube)
            except Exception as e:
                log.error("Failed to create Tube for %s: %s", meta.get('current_path'), e)

        # Modified Loading Logic: Handle disjoint channels by unioning per-tube Experiments
        if tubes:
            # Attempt standard import first (Optimized)
            try:
                import_op = cytoflow.ImportOp(tubes=tubes, conditions=import_conditions_schema)
                experiment = import_op.apply()
                return self._detect_and_tag_high_res(experiment)
            except Exception as e:
                # Check for channel mismatch error or generic CytoflowOpError
                msg = str(e)
                if "doesn't have the same channels" in msg or "CytoflowOpError" in str(type(e)):
                    log.info("Standard import failed (%s). Attempting manual channel union...", msg)

                    # Manual Union Strategy
                    loaded_dfs = []
                    combined_metadata = {}

                    for tube in tubes:
                        try:
                            # Import single tube
                            op = cytoflow.ImportOp(tubes=[tube], conditions=import_conditions_schema)
                            ex = op.apply()
                            loaded_dfs.append(ex.data)

                            # Merge Metadata
                            # Cytoflow requires metadata for all columns (channels & conditions)
                            # We accumulate unique metadata entries.
                            for k, v in ex.metadata.items():
                                if k not in combined_metadata:
                                    # Copy to avoid reference issues
                                    combined_metadata[k] = v.copy() if isinstance(v, dict) else v
                                else:
                                    # If it exists, merge for conditions
                                    existing = combined_metadata[k]
                                    if isinstance(existing, dict) and existing.get('type') == 'condition':
                                        current_vals = set(existing.get('values', []))
                                        new_vals = v.get('values', [])
                                        # Update values list with new seen values
                                        updated_vals = list(current_vals.union(new_vals))
                                        updated_vals.sort()
                                        existing['values'] = updated_vals
                                    # For channels, we assume metadata (voltage etc) is acceptable from the first source

                        except Exception as e_inner:
                            log.error("Error loading individual tube %s: %s", tube.file, e_inner)

                    if not loaded_dfs:
                        return None

                    # Concatenate all tubes; disjoint columns become NaN where absent.
                    per_tube_cols = [set(d.columns) for d in loaded_dfs]
                    combined_df = pd.concat(loaded_dfs, axis=0, ignore_index=True, sort=False)

                    # Identify channels missing from at least one tube. We deliberately
                    # keep these as NaN instead of fabricating 0.0: filling with zero
                    # invents a dense artificial population at the origin that corrupts
                    # gating and statistics. NaN is skipped by aggregates and excluded by
                    # the ``value > 0`` filters used in plotting. We record coverage so the
                    # UI can warn the user about mixed panels.
                    partial_channels = sorted(
                        str(c) for c in combined_df.columns
                        if pd.api.types.is_numeric_dtype(combined_df[c])
                        and any(c not in cols for cols in per_tube_cols)
                    )
                    if partial_channels:
                        log.warning(
                            "Mixed panels detected: %d channel(s) are absent from some "
                            "files and will be NaN there (not 0): %s",
                            len(partial_channels), ", ".join(partial_channels),
                        )
                        combined_metadata = dict(combined_metadata)
                        combined_metadata["partial_channels"] = partial_channels

                    # Create unified experiment manually
                    try:
                        unified_experiment = cytoflow.Experiment()
                        unified_experiment.data = combined_df
                        unified_experiment.metadata = combined_metadata
                        return self._detect_and_tag_high_res(unified_experiment)
                    except Exception as e_final:
                         log.error("Failed to create unified experiment: %s", e_final)
                         return None
                else:
                    # Propagate other errors
                    raise e
        else:
            return None
