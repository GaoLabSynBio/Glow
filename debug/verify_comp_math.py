import pandas as pd
import numpy as np

def verify_math():
    print("Verifying Compensation Math...")
    
    # 1. Simulate Source Spills
    # FITC spills 20% into PE.
    # PE spills 0% into FITC.
    
    # True Signals
    # Event 1: Pure FITC 1000.
    # Event 2: Pure PE 1000.
    true_data = pd.DataFrame({
        'FITC': [1000., 0.],
        'PE':   [0., 1000.]
    })
    
    # Spillover Matrix S
    # Row=Source, Col=Detector
    # S_fitc_fitc = 1, S_fitc_pe = 0.2
    # S_pe_fitc = 0, S_pe_pe = 1
    
    S = pd.DataFrame([
        [1.0, 0.2], # FITC Source
        [0.0, 1.0]  # PE Source (Note: If PE spilled into FITC, this would be [0.05, 1.0])
    ], index=['FITC', 'PE'], columns=['FITC', 'PE'])
    
    print("Spillover Matrix S:")
    print(S)
    
    # Observe Data = True * S (Pandas: True.dot(S))
    # True (N x F). S (F x D).
    observed_data = true_data.dot(S)
    print("\nObserved Data (with spillover):")
    print(observed_data)
    # Event 1: FITC=1000, PE=200. (Correct 20% spill)
    # Event 2: FITC=0, PE=1000.
    
    # 2. Compensation
    # C = Inv(S)
    C_values = np.linalg.inv(S.values)
    C = pd.DataFrame(C_values, index=['FITC', 'PE'], columns=['FITC', 'PE'])
    
    print("\nCompensation Matrix C = Inv(S):")
    print(C)
    # Inverse of [[1, 0.2], [0, 1]] is [[1, -0.2], [0, 1]]
    
    # 3. Apply Compensation
    # Corrected = Observed * C
    # This seems to be the contentious part.
    # Is it Observed * C? Or Observed * C.T?
    
    # Let's try Observed.dot(C)
    compensated_1 = observed_data.dot(C)
    print("\nAttempt 1: Observed.dot(C)")
    print(compensated_1)
    
    # Check Result
    # Ev 1 (FITC only): Should be [1000, 0].
    # Obs: [1000, 200].
    # C: [[1, -0.2], [0, 1]] (Rows=Source? Cols=Det?)
    # Wait.
    # If C = Inv(S). C has same index/column structure as S.
    # S: Index=Fluor, Col=Det.
    # C: Index=Fluor, Col=Det.
    
    # Dot Product: A.dot(B) -> Sum over A's axis 1 (Cols) and B's axis 0 (Index).
    # A (Observed) Cols = Detectors (FITC, PE).
    # B (C) Index = Fluorophores? (FITC, PE).
    # Since names match, it aligns.
    
    # Calculation for Ev 1 [1000, 200] dot C:
    # Res_FITC (Col 0 of C): 1000*C[0,0] + 200*C[1,0] = 1000*1 + 200*0 = 1000.
    # Res_PE   (Col 1 of C): 1000*C[0,1] + 200*C[1,1] = 1000*(-0.2) + 200*1 = -200 + 200 = 0.
    
    # This works Perfect.
    
    # 4. What if values were swapped?
    # S = [[1, 0], [0.2, 1]] (PE spills to FITC)
    # C = [[1, 0], [-0.2, 1]]
    
    # Observed PE event (0, 1000) -> Spills to FITC 200. Obs: [200, 1000].
    # Comp: [200, 1000] dot [[1, 0], [-0.2, 1]]
    # Res_FITC: 200*1 + 1000*(-0.2) = 200 - 200 = 0.
    # Res_PE:   200*0 + 1000*1 = 1000.
    # Works.
    
    print("Math verification passed.")

if __name__ == "__main__":
    verify_math()
