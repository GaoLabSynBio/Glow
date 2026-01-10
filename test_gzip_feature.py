import pandas as pd
import io
import os

def test_gzip_export():
    print("Testing DataFrame export to CSV.GZ...")
    
    # Create dummy data
    df = pd.DataFrame({
        'A': range(100),
        'B': range(100, 200)
    })
    
    # 1. Test In-Memory Buffer
    try:
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False, compression='gzip')
        size = buffer.getbuffer().nbytes
        print(f"SUCCESS: Exported {len(df)} rows to memory buffer. Compressed size: {size} bytes.")
    except Exception as e:
        print(f"FAILURE: In-memory export failed: {e}")
        return

    # 2. Test File Write
    filename = "test_export.csv.gz"
    try:
        df.to_csv(filename, index=False, compression='gzip')
        if os.path.exists(filename):
            print(f"SUCCESS: Exported to file '{filename}'.")
            
            # Verify Read
            df_read = pd.read_csv(filename, compression='gzip')
            if len(df_read) == 100:
                print("SUCCESS: Verified file can be read back.")
            else:
                print("FAILURE: Read back row count mismatch.")
        else:
            print("FAILURE: File not created.")
    except Exception as e:
        print(f"FAILURE: File export failed: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_gzip_export()
