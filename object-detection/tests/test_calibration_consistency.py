import os
import sys
import itertools
import numpy as np
from pathlib import Path

# Add object-detection directory to sys.path to resolve imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from calibration.estimate_relative_pose import estimate

def make_M(R, t):
    """ Constructs a 4x4 homogeneous transformation matrix. """
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.array(t)
    return M

def check_consistency(M_AC, M_AB, M_BC):
    """
    Checks the consistency between M_AC and the combination of M_AB and M_BC.
    Returns the rotation error (in degrees), and the absolute/relative Frobenius 
    norm of the 4x4 matrix difference.
    """
    # Test both multiplication orders and pick the best one
    M_pred1 = M_AB @ M_BC
    M_pred2 = M_BC @ M_AB
    
    R_AC = M_AC[:3, :3]
    R_pred1 = M_pred1[:3, :3]
    R_pred2 = M_pred2[:3, :3]
    
    # Calculate angle of residual rotation in degrees
    trace1 = np.clip(np.trace(R_pred1.T @ R_AC), -1.0, 3.0)
    err_rot1 = np.degrees(np.arccos((trace1 - 1.0) / 2.0))
    
    trace2 = np.clip(np.trace(R_pred2.T @ R_AC), -1.0, 3.0)
    err_rot2 = np.degrees(np.arccos((trace2 - 1.0) / 2.0))
    
    # Pick the convention that actually yields the correct consistency
    if err_rot1 < err_rot2:
        M_pred = M_pred1
        err_rot = err_rot1
    else:
        M_pred = M_pred2
        err_rot = err_rot2
        
    abs_err_4x4 = np.linalg.norm(M_AC - M_pred)
    # Avoid division by zero just in case
    norm_M_AC = np.linalg.norm(M_AC)
    rel_err_4x4 = abs_err_4x4 / norm_M_AC if norm_M_AC > 1e-9 else float('inf')
    
    return err_rot, abs_err_4x4, rel_err_4x4

def test_calibration_consistency(image_paths, tolerance_deg=5.0):
    """
    Given a list of image paths, tests every triplet (A, B, C)
    to check if their relative pose matrices are consistent.
    """
    out_dir = ROOT_DIR / "tests" / "temp_out"
    out_dir.mkdir(exist_ok=True)
    
    poses = {}
    print("--- Pre-computing pairwise poses ---")
    
    # Sort to ensure consistent i < j ordering
    image_paths = sorted(image_paths)
    
    # Only calculate i < j to save time
    for (img_a, img_b) in itertools.combinations(image_paths, 2):
        print(f"Estimating pose: {img_a.name} -> {img_b.name}")
        try:
            res = estimate(img_a, img_b, out_dir=out_dir)
            M = make_M(res.R, res.t)
            poses[(img_a, img_b)] = M
            # Compute inverse for j -> i
            poses[(img_b, img_a)] = np.linalg.inv(M)
        except Exception as e:
            print(f"  [!] Skipping pair {img_a.name}->{img_b.name} due to error: {e}")
            poses[(img_a, img_b)] = None
            poses[(img_b, img_a)] = None

    print("\n--- Verifying triplets ---")
    passed = 0
    total = 0
    
    # Check combinations of 3 images
    for (A, B, C) in itertools.combinations(image_paths, 3):
        M_AB = poses.get((A, B))
        M_BC = poses.get((B, C))
        M_AC = poses.get((A, C))
        
        if M_AB is None or M_BC is None or M_AC is None:
            continue
            
        total += 1
        
        err_rot, abs_err, rel_err = check_consistency(M_AC, M_AB, M_BC)
        
        status = "PASS" if err_rot < tolerance_deg else "FAIL"
        if status == "PASS": passed += 1
            
        print(f"Triplet ({A.name}, {B.name}, {C.name}): {status}")
        print(f"  - Rotation error:   {err_rot:.2f}°")
        print(f"  - 4x4 Abs. Diff:    {abs_err:.4f}")
        print(f"  - 4x4 Rel. Diff:    {rel_err:.4f}")
            
    if total == 0:
        print("\nNo valid triplets found to test.")
        return False
        
    print(f"\nConsistency test finished: {passed}/{total} triplets passed (Rotation Tol: <{tolerance_deg}°).")
    return passed == total

def run_tests_on_directory(directory="images/4_images"):
    """
    Finds all images in the target directory and runs the consistency test.
    """
    repo_root = ROOT_DIR.parent
    img_dir = repo_root / directory
    
    if not img_dir.exists() or not img_dir.is_dir():
        print(f"Error: Directory {img_dir} does not exist.")
        return
        
    # Gather images
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(list(img_dir.glob(ext)))
        
    if len(image_paths) < 3:
        print(f"Need at least 3 images to test consistency, found {len(image_paths)} in {img_dir}.")
        return
        
    print(f"Found {len(image_paths)} images. Starting test...")
    test_calibration_consistency(image_paths)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test calibration consistency for triplets of images.")
    parser.add_argument("--dir", type=str, default="images/4_images", 
                        help="Path to directory containing images (relative to repo root)")
    parser.add_argument("--tol", type=float, default=5.0, 
                        help="Tolerance for rotation matrix difference in degrees")
    args = parser.parse_args()
    
    run_tests_on_directory(args.dir)
