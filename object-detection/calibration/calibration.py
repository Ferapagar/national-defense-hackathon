import numpy as np
import cv2
import os
from scipy.optimize import least_squares
from typing import Tuple, Dict

class Image:
    def __init__(self, data: np.ndarray, title: str):
        self.data = data
        self.title = title

    @staticmethod
    def from_file(file_path: str) -> "Image":
        data = cv2.imread(file_path)
        if data is None:
            raise FileNotFoundError(f"Could not read image from {file_path}")
        title = os.path.splitext(os.path.basename(file_path))[0]
        return Image(data, title)

    @staticmethod
    def from_video(file_path: str) -> "Image":
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video {file_path}")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read frame from {file_path}")
        title = os.path.splitext(os.path.basename(file_path))[0]
        return Image(frame, title)

# Global library cache for transformation matrices
_TRANSFORMATION_LIBRARY = {}

def get_relative_pos(img_i: Image, img_j: Image) -> Tuple[np.ndarray, float, float]:
    """
    Computes a 3D transformation matrix M_{ij} that transforms coordinates 
    between camera i and camera j using SIFT and least squares optimization,
    along with focal lengths f_i and f_j.
    """
    # 1. Library Check
    pair_key = (img_i.title, img_j.title)
    if pair_key in _TRANSFORMATION_LIBRARY:
        return _TRANSFORMATION_LIBRARY[pair_key]
    
    # 2. Feature Matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_i.data, None)
    kp2, des2 = sift.detectAndCompute(img_j.data, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 8:
        raise ValueError("Not enough good matches found between images")
        
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    N = len(pts1)
    
    def euler_to_matrix(yaw, pitch, roll):
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        return R_yaw @ R_pitch @ R_roll
        
    def loss_func(params):
        t_phi, t_theta = params[0], params[1]
        r_yaw, r_pitch, r_roll = params[2], params[3], params[4]
        f_i = params[5]
        f_j = params[6]
        z_prime = params[7:7+N]
        
        # Unit vector t_ij from spherical coordinates
        t_ij = np.array([
            np.sin(t_theta) * np.cos(t_phi),
            np.sin(t_theta) * np.sin(t_phi),
            np.cos(t_theta)
        ])
        
        R_ij = euler_to_matrix(r_yaw, r_pitch, r_roll)
        
        # Vectorized 3D points in camera j
        q_3d_base = np.hstack((pts2, np.full((N, 1), f_j)))
        q_3d = q_3d_base * z_prime[:, None]
        
        # Apply transformation (N, 3) @ (3, 3).T + (3,)
        transformed_q = q_3d @ R_ij.T + t_ij
        
        # Proj_{xy}(x,y,z) = (x/z, y/z)
        proj_q = transformed_q[:, :2] / transformed_q[:, 2:]
        
        # Target projection mapped to f_i
        target_p = proj_q * f_i
        
        # Calculate residuals and flatten to 1D array
        residuals = pts1 - target_p
        return residuals.ravel()

    # Initial guess
    initial_params = np.zeros(7 + N)
    initial_params[1] = np.pi / 2  # t_theta = 90 deg, so cos(theta) = 0
    initial_params[5] = 1.0        # f_i = 1.0
    initial_params[6] = 1.0        # f_j = 1.0
    initial_params[7:] = 100.0     # initial depths
    
    res = least_squares(loss_func, initial_params)
    opt_params = res.x
    
    t_phi, t_theta = opt_params[0], opt_params[1]
    r_yaw, r_pitch, r_roll = opt_params[2], opt_params[3], opt_params[4]
    f_i = opt_params[5]
    f_j = opt_params[6]
    
    t_ij = np.array([
        np.sin(t_theta) * np.cos(t_phi),
        np.sin(t_theta) * np.sin(t_phi),
        np.cos(t_theta)
    ])
    R_ij = euler_to_matrix(r_yaw, r_pitch, r_roll)
    
    M_ij = np.eye(4)
    M_ij[:3, :3] = R_ij
    M_ij[:3, 3] = t_ij
    
    _TRANSFORMATION_LIBRARY[pair_key] = (M_ij, f_i, f_j)
    
    # Store M_ji with normalized translation to maintain consistency
    M_ji = np.eye(4)
    M_ji[:3, :3] = R_ij.T  # No division by s_ij, as proven mathematically
    M_ji[:3, 3] = -R_ij.T @ t_ij
    
    pair_key_ji = (img_j.title, img_i.title)
    _TRANSFORMATION_LIBRARY[pair_key_ji] = (M_ji, f_j, f_i)
    
    return M_ij, f_i, f_j


class ReferenceSystem:
    def __init__(self, img_i: Image, img_j: Image):
        self.ref_img_i = img_i
        self.ref_img_j = img_j
        self.M_ij, f_i, f_j = get_relative_pos(img_i, img_j)
        self.camera_params: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        
        # Add trivial parameters for img_i
        self.camera_params[img_i.title] = (np.zeros(3), np.eye(3), f_i)
        
        # Add parameters for img_j
        t_ij = self.M_ij[:3, 3]
        R_ij = self.M_ij[:3, :3]
        self.camera_params[img_j.title] = (t_ij, R_ij, f_j)

    def get_coords(self, img_k: Image) -> Tuple[np.ndarray, np.ndarray, float]:
        # 1. Cache Check
        if img_k.title in self.camera_params:
            return self.camera_params[img_k.title]
            
        # 2. Compute transformations
        M_ik, _, f_k = get_relative_pos(self.ref_img_i, img_k)
        M_jk, _, _ = get_relative_pos(self.ref_img_j, img_k)
        
        # 3. Extract translation vectors
        t_ij = self.M_ij[:3, 3]
        t_ik = M_ik[:3, 3]
        t_jk = M_jk[:3, 3]
        
        # 4. Calculate rotation matrix
        R_ik = M_ik[:3, :3]
        U_k_ij = R_ik
        
        # Helper to compute angle between two vectors
        def angle(u, v):
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            if norm_u == 0 or norm_v == 0:
                return 0.0
            cos_val = np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0)
            return np.arccos(cos_val)
            
        # 5. Calculate angles using dot product formula
        alpha_i = angle(t_ij, t_ik)
        alpha_j = angle(-t_ij, t_jk)
        alpha_k = angle(t_ik, t_jk)
        
        # 6. Use Law of Sines to find distance
        if np.sin(alpha_k) == 0:
            d_k_ij = 0.0
        else:
            d_k_ij = np.sin(alpha_i) / np.sin(alpha_k)
            
        # 7. Compute the coordinate vector
        v_k_ij = t_ik * d_k_ij
        
        # 8. Store and return
        self.camera_params[img_k.title] = (v_k_ij, U_k_ij, f_k)
        return v_k_ij, U_k_ij, f_k

    def to_rays(self, px_coords: np.ndarray, cam_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the origin and direction of rays given an array of pixel coordinates.
        px_coords: n x 2 array of pixel coordinates.
        cam_id: The ID of the camera.
        """
        # 1. Retrieve camera params
        if cam_id not in self.camera_params:
            raise KeyError(f"Camera ID '{cam_id}' not found in the reference system.")
        v, U, f_k = self.camera_params[cam_id]
        
        n = px_coords.shape[0]
        
        # 2. Directions matrix in local frame
        D_cam = np.hstack((px_coords, np.full((n, 1), f_k)))
        
        # 3. Transform directions to global frame
        D_global = D_cam @ U.T
        
        # 4. Origins matrix
        O_global = np.tile(v, (n, 1))
        
        # 5. Return (origins, directions)
        return O_global, D_global
