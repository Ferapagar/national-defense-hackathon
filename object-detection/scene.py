import numpy as np
from typing import List, Tuple, Optional
from raycasting.raycaster import cast_rays_into_grid_sampled
from raycasting.calibration import compute_pairwise_dt, compute_global_dt

class RayBatch:
    """
    Holds a collection of rays in a vectorized format for performance.
    """
    def __init__(self, camera_id: int, local_timestamp: float, global_timestamp: float, origins: np.ndarray, directions: np.ndarray, intensities: np.ndarray):
        self.camera_id = camera_id
        self.local_timestamp = local_timestamp
        self.global_timestamp = global_timestamp
        self.origins = origins
        self.directions = directions
        self.intensities = intensities

class Camera:
    """
    Holds the intrinsic and extrinsic parameters of each camera, 
    and wraps the ray generation logic.
    """
    def __init__(self, camera_id: int, fov: float, resolution: Tuple[int, int], position: np.ndarray, rotation: np.ndarray):
        self.camera_id = camera_id
        self.fov = fov
        self.resolution = resolution
        self.position = position
        self.rotation = rotation
        self.dt_i = 0.0
        self.frame_history = []  # Used for computing calibration offsets

    def generate_rays(self, image: np.ndarray, local_timestamp: float) -> RayBatch:
        """
        Extracts rays from an image mask/intensity map based on camera extrinsics and intrinsics.
        """
        h, w = self.resolution
        cx, cy = w / 2.0, h / 2.0
        
        # Focal length from fov (in degrees)
        focal_length = cx / np.tan(np.radians(self.fov) / 2.0)
        
        y, x = np.nonzero(image > 0)
        intensities = image[y, x]
        
        if len(x) == 0:
            return RayBatch(self.camera_id, local_timestamp, local_timestamp + self.dt_i, np.empty((0,3)), np.empty((0,3)), np.empty((0,)))
            
        x_cam = x - cx
        y_cam = y - cy
        z_cam = np.full_like(x_cam, focal_length)
        
        dirs_cam = np.stack((x_cam, y_cam, z_cam), axis=-1)
        dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
        
        # Transform into world coordinates
        dirs_world = dirs_cam @ self.rotation.T
        origins_world = np.tile(self.position, (len(x), 1))
        
        return RayBatch(self.camera_id, local_timestamp, local_timestamp + self.dt_i, origins_world, dirs_world, intensities)

    def add_history_frame(self, feature: float):
        """ Adds a historical feature (e.g., total motion intensity) for calibration """
        self.frame_history.append(feature)

class DetectedObject:
    def __init__(self, object_id: int, position: np.ndarray, confidence: float, t_global: Optional[float] = None):
        self.object_id = object_id
        self.position = position
        self.confidence = confidence
        self.t_global = t_global

class FrozenScene:
    def __init__(self, timestamp: float, voxel_grid_snapshot: np.ndarray, detected_objects_snapshot: List[DetectedObject]):
        self.timestamp = timestamp
        self.voxel_grid_snapshot = voxel_grid_snapshot
        self.detected_objects_snapshot = detected_objects_snapshot

class GlobalScene:
    """
    The main continuous structure containing the state of the world, handling the wrapping 
    of calibration, aggregation, and object detection.
    """
    def __init__(self, voxel_grid_extent: List[Tuple[float, float]], voxel_grid_size: Tuple[int, int, int]):
        self.voxel_grid_extent = voxel_grid_extent
        self.voxel_grid_size = voxel_grid_size
        self.voxel_grid = np.zeros(voxel_grid_size, dtype=np.float32)
        self.cameras: List[Camera] = []
        self.rays: List[RayBatch] = []
        self.detected_objects: List[DetectedObject] = []

    def add_camera(self, camera: Camera):
        self.cameras.append(camera)

    def calibrate(self):
        """
        Calibrates the cameras using their historical frames to compute and set dt_i.
        """
        n = len(self.cameras)
        if n < 2:
            return
            
        dt_matrix = []
        for i in range(n):
            for j in range(i+1, n):
                hist_a = np.array(self.cameras[i].frame_history)
                hist_b = np.array(self.cameras[j].frame_history)
                dt_ij = compute_pairwise_dt(hist_a, hist_b)
                dt_matrix.append((i, j, dt_ij))
                
        global_dts = compute_global_dt(dt_matrix, n)
        for i, cam in enumerate(self.cameras):
            cam.dt_i = global_dts[i]
            
    def aggregate_rays(self, ray_batch: RayBatch):
        """
        Casts a batch of rays into the 3D voxel grid.
        """
        self.rays.append(ray_batch)
        self.voxel_grid = cast_rays_into_grid_sampled(
            ray_batch.origins,
            ray_batch.directions,
            ray_batch.intensities,
            self.voxel_grid,
            self.voxel_grid_extent,
            self.voxel_grid_size
        )
        
    def detect_objects(self, threshold: float = 1.0, t_global: Optional[float] = None) -> List[DetectedObject]:
        """
        Thresholds the voxel grid and converts high-intensity voxels into detected objects.
        Optionally tags each object with the global timestamp it was observed at.
        """
        points = np.argwhere(self.voxel_grid > threshold)
        objects = []
        ext = self.voxel_grid_extent
        sz = self.voxel_grid_size

        for i, pt in enumerate(points):
            # Map grid coordinates back to world position (centre of voxel).
            x = ext[0][0] + ((pt[0] + 0.5) / sz[0]) * (ext[0][1] - ext[0][0])
            y = ext[1][0] + ((pt[1] + 0.5) / sz[1]) * (ext[1][1] - ext[1][0])
            z = ext[2][0] + ((pt[2] + 0.5) / sz[2]) * (ext[2][1] - ext[2][0])

            pos = np.array([x, y, z])
            conf = self.voxel_grid[pt[0], pt[1], pt[2]]
            objects.append(DetectedObject(i, pos, conf, t_global=t_global))

        self.detected_objects = objects
        return objects

    def clear_grid(self):
        """Reset voxel grid for the next per-timestamp aggregation window."""
        self.voxel_grid = np.zeros(self.voxel_grid_size, dtype=np.float32)
        self.detected_objects = []

    def freeze(self, timestamp: float) -> FrozenScene:
        """
        Creates an immutable snapshot of the scene for a given timeframe.
        """
        return FrozenScene(
            timestamp=timestamp,
            voxel_grid_snapshot=self.voxel_grid.copy(),
            detected_objects_snapshot=list(self.detected_objects)
        )
