import cytoflow
import os
import shutil
import pandas as pd
import re
import zipfile
import pyarrow.parquet as pq
import numpy as np

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

    def _natural_sort_key(self, text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(text))]

    def save_uploaded_files(self, uploaded_files, target_dir="data/uploaded_raw", folder_name=None):
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

    def save_uploaded_folder_zip(self, uploaded_zip, target_dir="data/uploaded_raw_zip"):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir) # Clean up previous uploads
        os.makedirs(target_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        except zipfile.BadZipFile:
            print("Error: Invalid zip file")
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
            print(f"Directory not found: {directory_path}")
            return paths

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.fcs'):
                    full_path = os.path.join(root, file)
                    paths.append(full_path)

        paths.sort(key=self._natural_sort_key)
        return paths

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
                        print(f"Warning: File not found at {path}")

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
                    print(f"Error copying {original}: {e}")

            processed_meta = meta.copy()
            processed_meta['current_path'] = final_path
            self.loaded_files_metadata.append(processed_meta)

        return self.loaded_files_metadata

    def prepare_fcs_files(self, paths, rename_strategy='folder_name', output_dir=None):
        meta = self.generate_preview_metadata(paths, rename_strategy)
        return self.process_files(meta, output_dir)

    def load_as_cytoflow_experiment(self, metadata_list=None):
        if metadata_list is None:
            metadata_list = self.loaded_files_metadata

        if not metadata_list:
            print("No files to load.")
            return None

        all_condition_keys = set()
        all_condition_keys.add('Sample')
        all_condition_keys.add('Condition')

        internal_keys = ['original_path', 'current_path', 'path', 'display_name', 'condition']
        for meta in metadata_list:
             for key in meta.keys():
                 if key not in internal_keys:
                     all_condition_keys.add(key)

        import_conditions_schema = {}
        for key in all_condition_keys:
            import_conditions_schema[key] = "category"

        tubes = []
        for meta in metadata_list:
            tube_conditions = {}
            tube_conditions['Sample'] = str(meta.get('display_name', 'Unknown'))

            if 'condition' in meta and meta['condition']:
                tube_conditions['Condition'] = str(meta['condition'])
            else:
                 tube_conditions['Condition'] = "Unspecified"

            for key in all_condition_keys:
                if key in ['Sample', 'Condition']: continue
                if key in meta and meta[key] is not None and str(meta[key]).strip() != "":
                    tube_conditions[key] = str(meta[key])
                else:
                    tube_conditions[key] = "NA"

            try:
                tube = cytoflow.Tube(file=meta['current_path'], conditions=tube_conditions)
                tubes.append(tube)
            except Exception as e:
                print(f"Failed to create Tube for {meta.get('current_path')}: {e}")

        if tubes:
            import_op = cytoflow.ImportOp(tubes=tubes, conditions=import_conditions_schema)
            experiment = import_op.apply()
            return experiment
        else:
            return None
