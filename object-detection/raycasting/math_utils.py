import numpy as np

def rotation_matrix_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Generates a 3x3 rotation matrix from yaw, pitch, and roll in degrees.
    """
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    
    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [0,          0,         1]
    ])
    Ry = np.array([
        [np.cos(r),  0, np.sin(r)],
        [0,          1,         0],
        [-np.sin(r), 0, np.cos(r)]
    ])
    Rx = np.array([
        [1,         0,          0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p),  np.cos(p)]
    ])
    return Rz @ Ry @ Rx

def ray_aabb_intersection(origins: np.ndarray, directions: np.ndarray, box_min: np.ndarray, box_max: np.ndarray):
    """
    Vectorized Axis-Aligned Bounding Box (AABB) intersection check.
    
    Args:
        origins: (N, 3) numpy array of ray origins
        directions: (N, 3) numpy array of ray directions (normalized)
        box_min: (3,) numpy array
        box_max: (3,) numpy array
        
    Returns:
        valid_mask: (N,) boolean array indicating which rays intersect the box
        t_entry: (N,) float array of entry distances
        t_exit: (N,) float array of exit distances
    """
    # To avoid division by zero warnings
    with np.errstate(divide='ignore'):
        inv_dir = 1.0 / directions
    
    t1 = (box_min - origins) * inv_dir
    t2 = (box_max - origins) * inv_dir
    
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    
    t_entry = np.max(tmin, axis=1)
    t_exit = np.min(tmax, axis=1)
    
    valid_mask = (t_entry <= t_exit) & (t_exit >= 0)
    return valid_mask, t_entry, t_exit
