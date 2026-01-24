import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from glow.utils import safe_log10, safe_log

def test_safe_log10():
    print("Testing safe_log10...")
    
    # CASE 1: Basic Positive
    arr1 = np.array([1, 10, 100])
    res1 = safe_log10(arr1)
    assert np.allclose(res1, [0, 1, 2]), f"Failed Positive: {res1}"
    print("  Positive: OK")
    
    # CASE 2: Negative with NaN replacement (Default)
    arr2 = np.array([-1, 0, 10])
    res2 = safe_log10(arr2)
    # Expected: [NaN, NaN, 1]
    assert np.isnan(res2[0]), "Failed Nan Replace (Neg)"
    assert np.isnan(res2[1]), "Failed Nan Replace (Zero)"
    assert np.isclose(res2[2], 1.0), "Failed Positive Mixing"
    print("  NaN Replace: OK")
    
    # CASE 3: Clipping
    arr3 = np.array([-100, 0, 100])
    res3 = safe_log10(arr3, min_val=0.1)
    # Expected: log10(0.1), log10(0.1), log10(100) -> -1, -1, 2
    assert np.allclose(res3, [-1, -1, 2]), f"Failed Clipping: {res3}"
    print("  Clipping: OK")
    
    print("safe_log10 PASSED")

def test_safe_log():
    print("\nTesting safe_log (ln)...")
    
    arr = np.array([np.e, 1, 0, -1])
    res = safe_log(arr, replace_with_nan=True)
    
    assert np.isclose(res[0], 1.0)
    assert np.isclose(res[1], 0.0)
    assert np.isnan(res[2])
    assert np.isnan(res[3])
    print("  Basic: OK")
    
    # Clipping
    res_c = safe_log(arr, min_val=1.0)
    # log(e)=1, log(1)=0, log(clip(0->1))=0, log(clip(-1->1))=0
    assert np.allclose(res_c, [1, 0, 0, 0])
    print("  Clipping: OK")
    
    print("safe_log PASSED")

if __name__ == "__main__":
    try:
        test_safe_log10()
        test_safe_log()
        print("\nALL TRIALS SUCCESSFUL")
    except ImportError as e:
        print(f"Import Error (likely missing numpy): {e}")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
