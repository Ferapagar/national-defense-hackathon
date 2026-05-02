import numpy as np
from scipy.optimize import lsq_linear

def compute_pairwise_dt(history_a: np.ndarray, history_b: np.ndarray) -> float:
    """
    Computes dt_ij between two cameras given their historical frame features.
    Uses cross-correlation to find the time shift that maximizes agreement.
    """
    if len(history_a) == 0 or len(history_b) == 0:
        return 0.0
    
    # Correlate the two signals
    correlation = np.correlate(history_a, history_b, mode='full')
    
    # Find the shift that yields maximum correlation
    shift = np.argmax(correlation) - (len(history_b) - 1)
    return float(shift)

def compute_global_dt(dt_matrix: list, num_cameras: int) -> np.ndarray:
    """
    Given pairwise dt_ij estimates, solves for global dt_i using least squares.
    dt_matrix should be a list of tuples: (i, j, dt_ij)
    """
    if num_cameras < 2 or not dt_matrix:
        return np.zeros(num_cameras)
        
    A = []
    b = []
    
    for (i, j, dt_ij) in dt_matrix:
        row = np.zeros(num_cameras)
        row[i] = 1
        row[j] = -1
        A.append(row)
        b.append(dt_ij)
        
    # Anchor camera 0 (dt_0 = 0) to ensure full rank
    row = np.zeros(num_cameras)
    row[0] = 1
    A.append(row)
    b.append(0.0)
    
    A = np.array(A)
    b = np.array(b)
    
    # Solve argmin ||A * x - b||^2
    res = lsq_linear(A, b)
    return res.x
