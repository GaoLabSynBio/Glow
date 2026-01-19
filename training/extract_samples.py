import os
import random
import shutil
import argparse
import sys

def extract_samples(source_root, dest_root):
    """
    Scans source_root for directories containing .fcs files.
    Picks one random .fcs file > 1MB from each directory.
    Copies it to dest_root with renamed format {folder_name}_{file_name}.fcs.
    Replaces spaces with underscores.
    """
    
    # Ensure destination exists
    if not os.path.exists(dest_root):
        try:
            os.makedirs(dest_root)
            print(f"Created destination directory: {dest_root}")
        except OSError as e:
            print(f"Error creating destination directory: {e}")
            return

    print(f"Scanning source: {source_root}")
    print(f"Destination: {dest_root}")
    print("-" * 50)

    count = 0
    folders_processed = 0

    # Walk the directory tree
    for root, dirs, files in os.walk(source_root):
        
        # Filter for valid FCS files in this current directory
        fcs_candidates = []
        
        for file in files:
            if file.lower().endswith('.fcs'):
                file_path = os.path.join(root, file)
                
                # Check size > 1MB (1 * 1024 * 1024 bytes)
                try:
                    if os.path.getsize(file_path) > 1 * 1024 * 1024:
                        fcs_candidates.append(file)
                except OSError as e:
                    print(f"Could not check size for {file_path}: {e}")

        if fcs_candidates:
            folders_processed += 1
            
            # Randomly pick 1 file
            chosen_file = random.choice(fcs_candidates)
            source_path = os.path.join(root, chosen_file)
            
            # Get folder name logic
            # os.path.basename(root) gives the current folder name
            folder_name = os.path.basename(root)
            
            # If root is the source_root and it has files, folder_name is source_root's name
            # This is generally what is requested.
            
            # Sanitize names (replace spaces with underscores)
            safe_folder_name = folder_name.replace(" ", "_")
            safe_file_name = chosen_file.replace(" ", "_")
            
            # Construct new filename
            # chosen_file ends with .fcs, so we don't need to add it again unless we stripped it
            # Requirement: {folder_name}_{file_name}.fcs
            # If filename is "Sample 1.fcs", safe is "Sample_1.fcs"
            # Result: "Folder_Sample_1.fcs"
            
            new_filename = f"{safe_folder_name}_{safe_file_name}"
            dest_path = os.path.join(dest_root, new_filename)
            
            # Copy
            try:
                # Check if file already exists to avoid overwriting or duplicate work if run multiple times
                if os.path.exists(dest_path):
                    print(f"Skipping (exists): {new_filename}")
                else:
                    shutil.copy2(source_path, dest_path)
                    print(f"Extracted: {new_filename} (from {folder_name})")
                    count += 1
            except Exception as e:
                print(f"Error copying {source_path}: {e}")
                
    print("-" * 50)
    print(f"Completed. Found FCS candidates in {folders_processed} folders.")
    print(f"Successfully copied {count} files to {dest_root}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract random FCS samples > 1MB from subfolders.")
    parser.add_argument("source_dir", help="Path to the root folder containing your FCS file data structure")
    parser.add_argument("--dest_dir", default="training/source_fcs", help="Destination folder (default: training/source_fcs)")
    
    args = parser.parse_args()
    
    # Check source
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' does not exist.")
        sys.exit(1)
        
    extract_samples(os.path.abspath(args.source_dir), os.path.abspath(args.dest_dir))
