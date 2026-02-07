import os

# --- CONFIGURATION ---
# The name of the output file
OUTPUT_FILE = "codebase_dump.txt"

# Folders to completely ignore (deeply nested to save time/space)
IGNORE_DIRS = {
    ".git", "__pycache__", "venv", "env", "node_modules", 
    ".idea", ".vscode", "dist", "build", "coverage"
}

# File extensions to ignore (Jupyter, Weights, Images, Archives)
IGNORE_EXTENSIONS = {
    # Notebooks
    ".ipynb",
    # Model Weights & Serialized Data
    ".pt", ".pth", ".h5", ".onnx", ".pkl", ".bin", ".tflite",
    # Data & Archives
    ".csv", ".json", ".zip", ".tar", ".gz",
    # Images/Media
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4",
    # Compiled/System
    ".pyc", ".exe", ".dll", ".so", ".DS_Store"
}

def is_text_file(filepath):
    """Simple check to see if a file is readable text (avoids binary junk)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            f.read(1024)
        return True
    except (UnicodeDecodeError, PermissionError):
        return False

def bundle_repo():
    root_dir = os.getcwd()
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        outfile.write(f"=== REPOSITORY DUMP ===\n")
        outfile.write(f"Generated from: {root_dir}\n")
        outfile.write(f"=======================\n\n")

        for root, dirs, files in os.walk(root_dir):
            # Modify 'dirs' in-place to prune ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                
                # Skip output file itself and ignored extensions
                if file == OUTPUT_FILE or file == "bundle_repo.py" or ext in IGNORE_EXTENSIONS:
                    continue
                
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, root_dir)

                if is_text_file(filepath):
                    print(f"Adding: {rel_path}")
                    outfile.write(f"\n{'=' * 50}\n")
                    outfile.write(f"FILE: {rel_path}\n")
                    outfile.write(f"{'=' * 50}\n")
                    
                    try:
                        with open(filepath, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"[ERROR READING FILE: {e}]")
                    
                    outfile.write("\n")

    print(f"\nSuccess! All code bundled into: {OUTPUT_FILE}")

if __name__ == "__main__":
    bundle_repo()