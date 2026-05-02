# Calibration of cameras

## Overview
The goal of this component is to calibrate the cameras' position, orientation and focal length into a unified coordinate system. 
For that, we will take a single initial picture from each camera, and use feature detection to triangulate relative positions between cameras. 
Once the relative positions of the cameras are known, we can use two cameras and a single landmark to establish a global coordinate system and calculate the extrinsic parameters of all cameras.
Extrisic parameters of a camera:
- Position: (c_x, c_y, c_z)
- Orientation: (c_theta, c_phi, c_psi)
- Focal length: (f_x, f_y)
These coordinates will be taken with respect to an arbitrary global coordinate system which will be defined around an arbitrary camera:
 - x-axis: the horizontal ray of the optical axis from the first camera
 - y-axis: the direction such that (x, y, z) form a right handed coordinate system
 - z-axis: the vertical ray of the optical axis from the first camera
 - Center of reference: the first camera will be located at (0,0,0) (with no rotation)
 - Distance measurement: every distance will be calculated relative to the distance between two given reference points.

Finally, this component will have a function that takes as input some camera info and the pixel coordinates of detected objects in each camera and outputs the 3D coordinates of the corresponding rays, which will be of the form:
- Origin: (r_x, r_y, r_z)
- Direction: (r_theta, r_phi) in spherical coordinates


## Components
- `main_calibration.py`: Main script to perform the calibration. It will take as input a set of images from different cameras and output the extrinsic parameters of each camera with respect to an arbitrary global coordinate system.
- `estimate_relative_pose.py`: Two-view relative pose estimator. Recovers rotation `R` and unit-norm translation direction `t` between two images of the same static scene using SIFT features, the essential matrix, and OpenCV `recoverPose`.

## Relative pose estimation

> **Looking for a step-by-step walkthrough?** See [USAGE.md](USAGE.md) for a
> guide to taking two photos, running the estimator, and converting the result
> into a real-world distance in metres.

### Inputs
- Reference image: `../../images/ref-0.jpg`
- Test image: `../../images/test-0.jpg`
- Camera intrinsics `K`: resolved in priority order — explicit `--K` flag → EXIF `FocalLengthIn35mmFilm` → Pixel 8 default (`6.90 mm`, `24 mm` 35mm-equivalent). For a 2268×4032 Pixel 8 photo this gives `fx = fy ≈ 2688 px`, `cx = 1134`, `cy = 2016`.

### Dependencies
- `opencv-python`
- `numpy`
- `Pillow` (optional, for EXIF intrinsics fallback)

### Run
From this directory:

```bash
python estimate_relative_pose.py
# alternate pair / custom K:
python estimate_relative_pose.py --ref ../../images/ref-1.jpg --test ../../images/test-1.jpg
python estimate_relative_pose.py --K 2688,2688,1134,2016
```

### Outputs
Written to `out/` by default:
- `pose.json` — `R`, `t`, Euler angles (yaw/pitch/roll, deg), inlier counts, mean reprojection error (px), median triangulated depth, intrinsics matrix, the `K`-resolution path, the selected geometric `model` (`"essential"` or `"homography"`), and `plane_normal` when the homography path is used.
- `matches.png` — all SIFT matches surviving Lowe's ratio test.
- `inliers.png` — only the matches accepted by the winning model.

### Model selection (essential vs. homography)
The script estimates **both** the essential matrix and a homography from the same SIFT correspondences, then picks the appropriate model:

- **Essential matrix** is used for general 3D scenes with real camera translation. Returns a unit-norm `t` (direction only — scale is fundamentally ambiguous).
- **Homography** is used when the scene is near-planar or the motion is pure rotation. The script switches to this path automatically when (a) the homography has substantially more RANSAC inliers, or (b) the essential matrix's cheirality (positive-depth) check collapses — the classic signature of pure rotation. The homography is decomposed via `cv2.decomposeHomographyMat` and filtered by `filterHomographyDecompByVisibleRefpoints`, with the surviving candidate scored by triangulated-point cheirality and reprojection error.

This is the homography fallback motivated by the reference at [`references/homographies/`](../../references/homographies/), adapted to use OpenCV's RANSAC homography solver instead of hand-labelled correspondences.

### Caveats
- **Scale ambiguity:** monocular two-view geometry recovers `t` only up to scale.
- **Pure rotation:** when the homography path is selected with `‖t‖ ≈ 0`, only the rotation is reliable — the translation direction has high uncertainty and the script logs a warning.
- **Intrinsics:** `K` is approximated from the EXIF 35mm-equivalent focal length using horizontal-FOV equivalence. For best accuracy, calibrate the camera (e.g. checkerboard) and pass the result via `--K`.
