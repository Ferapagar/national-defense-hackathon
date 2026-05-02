import numpy as np
from .math_utils import ray_aabb_intersection

def cast_rays_into_grid_sampled(
    origins: np.ndarray, 
    directions: np.ndarray, 
    intensities: np.ndarray, 
    grid: np.ndarray, 
    voxel_grid_extent: list, 
    voxel_grid_size: tuple, 
    step_size: float = None
) -> np.ndarray:
    """
    Vectorized step-based ray traversal into a 3D grid.
    
    Args:
        origins: (N, 3) array of ray origins
        directions: (N, 3) array of normalized ray directions
        intensities: (N,) array of pixel/ray intensities
        grid: (Dx, Dy, Dz) 3D numpy array to accumulate intensities
        voxel_grid_extent: List of tuples [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        voxel_grid_size: Tuple (Dx, Dy, Dz)
        step_size: Step size along the ray. If None, computes an optimal default.
        
    Returns:
        grid: The updated 3D numpy array
    """
    if len(origins) == 0:
        return grid
        
    box_min = np.array([voxel_grid_extent[0][0], voxel_grid_extent[1][0], voxel_grid_extent[2][0]])
    box_max = np.array([voxel_grid_extent[0][1], voxel_grid_extent[1][1], voxel_grid_extent[2][1]])
    
    valid, t_entry, t_exit = ray_aabb_intersection(origins, directions, box_min, box_max)
    
    # We only care about the segment from max(0, t_entry) to t_exit
    t_entry = np.maximum(t_entry, 0.0)
    valid &= (t_entry <= t_exit)
    
    if not np.any(valid):
        return grid
        
    # Filter valid rays
    v_orig = origins[valid]
    v_dir = directions[valid]
    v_inten = intensities[valid]
    v_t_entry = t_entry[valid]
    v_t_exit = t_exit[valid]
    
    if step_size is None:
        # Default step size: half of the smallest voxel dimension
        vx = (box_max[0] - box_min[0]) / voxel_grid_size[0]
        vy = (box_max[1] - box_min[1]) / voxel_grid_size[1]
        vz = (box_max[2] - box_min[2]) / voxel_grid_size[2]
        step_size = min(vx, vy, vz) * 0.5
        
    max_dist = np.max(v_t_exit - v_t_entry)
    max_steps = int(max_dist / step_size) + 1
    
    # To prevent extreme memory consumption or infinite loops
    if max_steps > 2000:
        max_steps = 2000 
        
    # Vectorized step loop. NumPy favors few iterations with heavy vectorization.
    s_arr = np.arange(max_steps) * step_size
    grid_size_arr = np.array(voxel_grid_size)
    
    for step in s_arr:
        # Check which rays are still active at this step
        active = (v_t_entry + step) <= v_t_exit
        if not np.any(active):
            break
            
        # Compute 3D points
        # p shape: (N_active, 3)
        p = v_orig[active] + v_dir[active] * (v_t_entry[active] + step)[:, None]
        
        # Map world coordinates to voxel indices
        norm_p = (p - box_min) / (box_max - box_min)
        idx = (norm_p * grid_size_arr).astype(int)
        
        # Filter indices strictly within bounds (should mostly be true due to AABB)
        in_bounds = (idx[:, 0] >= 0) & (idx[:, 0] < grid.shape[0]) & \
                    (idx[:, 1] >= 0) & (idx[:, 1] < grid.shape[1]) & \
                    (idx[:, 2] >= 0) & (idx[:, 2] < grid.shape[2])
                    
        valid_idx = idx[in_bounds]
        valid_inten = v_inten[active][in_bounds]
        
        # Accumulate intensities
        np.add.at(grid, (valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]), valid_inten)
        
    return grid
