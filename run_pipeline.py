import argparse
from glow.base_pipeline.pipeline import run_base_pipeline

def main():
    parser = argparse.ArgumentParser(description="Glow Pipeline Entry Point")
    parser.add_argument("--file_list", help="Path to a text file containing list of FCS file paths")
    parser.add_argument("--input_dir", help="Directory to recursively search for FCS files")
    parser.add_argument("--output_dir", default="data/processed_fcs", help="Directory to save renamed files")
    parser.add_argument("--rename", default="folder_name", choices=["folder_name", "keep"], help="Renaming strategy")
    
    args = parser.parse_args()
    
    if args.file_list or args.input_dir:
        run_base_pipeline(file_list_path=args.file_list, input_dir=args.input_dir, output_dir=args.output_dir, rename_strategy=args.rename)
    else:
        print("Usage: python run_pipeline.py [--file_list <path>] [--input_dir <path>] [options]")

if __name__ == "__main__":
    main()
