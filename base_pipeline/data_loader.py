import cytoflow
import os
import shutil
import pandas as pd
import re

class DataLoader:
    def __init__(self):
        """
        Initialize the DataLoader. 
        It can hold references to loaded experiments or file metadata.
        """
        self.loaded_files_metadata = []

    def _natural_sort_key(self, text):
        """
        Helper key function for natural sorting (e.g., A1, A2, A10 instead of A1, A10, A2).
        """
        return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(text))]

    def scan_directory(self, directory_path):
        """
        Recursively finds all .fcs files in a directory.
        Returns them naturally sorted.
        """
        paths = []
        if not os.path.isdir(directory_path):
            print(f"Directory not found: {directory_path}")
            return paths

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.fcs'):
                    full_path = os.path.join(root, file)
                    paths.append(full_path)
        
        # Apply natural sort
        paths.sort(key=self._natural_sort_key)
        return paths

    def parse_file_paths_from_text(self, file_path_list_file):
        """
        Reads a file containing a list of FCS file paths (one per line).
        """
        paths = []
        if not os.path.exists(file_path_list_file):
            raise FileNotFoundError(f"File list not found: {file_path_list_file}")
            
        with open(file_path_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines or comments
                if line and not line.startswith("#"):
                    parts = line.split(',')
                    path = parts[0].strip()
                    custom_name = parts[1].strip() if len(parts) > 1 else None
                    
                    if os.path.exists(path):
                        paths.append({'path': path, 'name': custom_name})
                    else:
                        print(f"Warning: File not found at {path}")
        
        # Apply natural sort to user provided paths as well
        # We sort based on the file path string
        paths.sort(key=lambda x: self._natural_sort_key(x['path']))
        
        return paths

    def generate_preview_metadata(self, paths, rename_strategy='folder_name'):
        """
        Generates metadata and proposed names without moving files.
        """
        preview_metadata = []
        for item in paths:
            # Normalize input
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
            
            # Verify extension
            if not filename.lower().endswith('.fcs'):
                continue

            # Determine new name
            if custom_name:
                new_name = custom_name
                if not new_name.lower().endswith('.fcs'):
                    new_name += '.fcs'
            elif rename_strategy == 'folder_name':
                name_prefix = dirname.replace(" ", "_")
                # Strategy: foldername_originalname
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
        """
        Finalizes the files: copies them to output_dir if specified, and updates 'current_path'.
        
        Args:
            metadata_list (list): List of dicts with 'original_path' and 'display_name'.
            output_dir (str): Directory to save files to.
        """
        self.loaded_files_metadata = []
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for meta in metadata_list:
            original = meta['original_path']
            new_name = meta['display_name']
            
            final_path = original
            
            if output_dir:
                dest_path = os.path.join(output_dir, new_name)
                # Avoid overwriting check could be added here
                try:
                    shutil.copy2(original, dest_path)
                    final_path = dest_path
                except Exception as e:
                    print(f"Error copying {original}: {e}")
            
            # Update metadata with the actual path used for loading
            processed_meta = meta.copy()
            processed_meta['current_path'] = final_path
            self.loaded_files_metadata.append(processed_meta)
            
        return self.loaded_files_metadata

    def prepare_fcs_files(self, paths, rename_strategy='folder_name', output_dir=None):
        """
        Legacy/Wrapper: Prepares FCS files for the pipeline in one shot.
        """
        meta = self.generate_preview_metadata(paths, rename_strategy)
        return self.process_files(meta, output_dir)

    def load_as_cytoflow_experiment(self, metadata_list=None):
        """
        Loads the prepared files into a cytoflow.Experiment object.
        Explicitly declares conditions in ImportOp to avoid type inference errors.
        """
        if metadata_list is None:
            metadata_list = self.loaded_files_metadata
            
        if not metadata_list:
            print("No files to load.")
            return None

        # 1. First pass: Determine all unique condition keys and their types across all files
        # We need this to setup the ImportOp schema correctly
        all_condition_keys = set()
        all_condition_keys.add('Sample')
        all_condition_keys.add('Condition')
        
        internal_keys = ['original_path', 'current_path', 'path', 'display_name', 'condition']
        
        # Determine all possible keys
        for meta in metadata_list:
             for key in meta.keys():
                 if key not in internal_keys:
                     all_condition_keys.add(key)
        
        # 2. Build explicit schema for ImportOp
        # For robustness with dynamic user input, we treat most things as 'category' (string)
        # to avoid the "Metadata dataset X should be []" error which happens when types mismatch
        # or when new unexpected keys appear without being declared.
        import_conditions_schema = {}
        for key in all_condition_keys:
            import_conditions_schema[key] = "category" 

        tubes = []
        for meta in metadata_list:
            # 3. Build per-tube conditions dict matching the schema EXACTLY
            tube_conditions = {}
            
            # Essential: Sample (Unique ID)
            tube_conditions['Sample'] = str(meta.get('display_name', 'Unknown'))
            
            # Folder Condition
            if 'condition' in meta and meta['condition']:
                tube_conditions['Condition'] = str(meta['condition'])
            else:
                 tube_conditions['Condition'] = "Unspecified"

            # Dynamic Metadata
            for key in all_condition_keys:
                if key in ['Sample', 'Condition']: continue
                
                # Check directly in meta
                if key in meta and meta[key] is not None and str(meta[key]).strip() != "":
                    tube_conditions[key] = str(meta[key])
                else:
                    # Fill missing values with "NA" to ensure dense matrix for Cytoflow
                    tube_conditions[key] = "NA"
            
            try:
                tube = cytoflow.Tube(file=meta['current_path'], conditions=tube_conditions)
                tubes.append(tube)
            except Exception as e:
                print(f"Failed to create Tube for {meta.get('current_path')}: {e}")

        if tubes:
            # 4. Create ImportOp with EXPLICIT conditions schema
            # This tells Cytoflow exactly what metadata columns to expect and their types
            import_op = cytoflow.ImportOp(tubes=tubes, conditions=import_conditions_schema)
            experiment = import_op.apply()
            return experiment
        else:
            return None
