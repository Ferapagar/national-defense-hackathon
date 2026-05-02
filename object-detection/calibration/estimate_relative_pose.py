"""Two-view relative pose estimation.

Estimate the relative camera pose (rotation R, translation direction t up to scale)
between two images of the same static scene using SIFT features, the essential
matrix, and OpenCV's recoverPose.

Inputs (default): ../../images/ref-0.jpg, ../../images/test-0.jpg
Outputs (default): ./out/pose.json, ./out/matches.png, ./out/inliers.png

Usage:
    python estimate_relative_pose.py
    python estimate_relative_pose.py --ref ../../images/ref-1.jpg --test ../../images/test-1.jpg
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

DEFAULT_REF = REPO_ROOT / "images" / "ref-0.jpg"
DEFAULT_TEST = REPO_ROOT / "images" / "test-0.jpg"
DEFAULT_OUT = HERE / "out"

# Camera defaults: Google Pixel 8 main camera (user-provided).
PIXEL8_FOCAL_MM = 6.90
PIXEL8_FOCAL_35MM_EQ = 24.0  # from EXIF FocalLengthIn35mmFilm
FULL_FRAME_LONG_SIDE_MM = 36.0

# Feature / matching tuning.
FEATURE_TYPE = "SIFT"
RATIO_TEST = 0.75
MIN_INLIERS = 30
RANSAC_THRESH_PX = 1.0
RANSAC_PROB = 0.999


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    source: str  # how K was resolved: "user" | "exif:..." | "pixel8-default"

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class PoseResult:
    R: list  # 3x3 rotation, world->camera2 in camera1 frame
    t: list  # 3-vector translation (unit norm if model="essential", small if pure-rotation)
    euler_yaw_pitch_roll_deg: list
    inlier_count: int
    total_matches: int
    mean_reproj_error_px: float
    median_depth: float
    K: list
    intrinsic_source: str
    model: str  # "essential" | "homography"
    plane_normal: Optional[list] = None  # only set when model == "homography"


# ---------------------------------------------------------------------------
# Intrinsics resolution
# ---------------------------------------------------------------------------

def _read_exif_focal_35mm(path: Path) -> Optional[float]:
    try:
        from PIL import Image, ExifTags
    except ImportError:
        return None
    try:
        img = Image.open(path)
        exif = img.getexif()
        ifd = exif.get_ifd(0x8769)  # ExifIFD
        for tag_id, value in ifd.items():
            tag = ExifTags.TAGS.get(tag_id)
            if tag == "FocalLengthIn35mmFilm" and value:
                return float(value)
    except Exception:
        return None
    return None


def resolve_intrinsics(
    image_path: Path,
    width: int,
    height: int,
    user_K: Optional[np.ndarray] = None,
) -> Intrinsics:
    """Resolve K in priority order: explicit user K > EXIF > Pixel 8 default.

    Convention: long-side / 36mm focal-length equivalence (horizontal-FOV match).
    """
    if user_K is not None:
        return Intrinsics(
            fx=float(user_K[0, 0]),
            fy=float(user_K[1, 1]),
            cx=float(user_K[0, 2]),
            cy=float(user_K[1, 2]),
            source="user",
        )

    long_side = float(max(width, height))
    cx, cy = width / 2.0, height / 2.0

    f_35 = _read_exif_focal_35mm(image_path)
    if f_35 is not None:
        f_px = (long_side / FULL_FRAME_LONG_SIDE_MM) * f_35
        return Intrinsics(fx=f_px, fy=f_px, cx=cx, cy=cy, source=f"exif:f35={f_35}")

    f_px = (long_side / FULL_FRAME_LONG_SIDE_MM) * PIXEL8_FOCAL_35MM_EQ
    return Intrinsics(fx=f_px, fy=f_px, cx=cx, cy=cy, source="pixel8-default")


# ---------------------------------------------------------------------------
# Feature detection + matching
# ---------------------------------------------------------------------------

def _make_detector():
    if FEATURE_TYPE == "SIFT" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(), "SIFT", cv2.NORM_L2
    if hasattr(cv2, "AKAZE_create"):
        return cv2.AKAZE_create(), "AKAZE", cv2.NORM_HAMMING
    return cv2.ORB_create(nfeatures=8000), "ORB", cv2.NORM_HAMMING


def detect_and_match(img1_gray: np.ndarray, img2_gray: np.ndarray):
    detector, name, norm = _make_detector()
    kp1, des1 = detector.detectAndCompute(img1_gray, None)
    kp2, des2 = detector.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        raise RuntimeError(
            f"Insufficient features: kp1={len(kp1) if kp1 else 0}, "
            f"kp2={len(kp2) if kp2 else 0}"
        )

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

    if len(good) < MIN_INLIERS:
        raise RuntimeError(
            f"Only {len(good)} matches survived ratio test (need >= {MIN_INLIERS}). "
            "Images may not overlap enough or lack texture."
        )

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return kp1, kp2, good, pts1, pts2, name


# ---------------------------------------------------------------------------
# Pose recovery
# ---------------------------------------------------------------------------

# Threshold for preferring the homography model over the essential matrix.
# When the scene is planar or the motion is pure rotation, the essential matrix
# is rank-deficient and recoverPose's cheirality check collapses to ~0 inliers.
# A homography (planar projective transform) is the right model in that case;
# this idea is taken from the reference at references/homographies/.
HOMOGRAPHY_PREFERENCE_RATIO = 1.05


def _recover_pose_essential(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray):
    E, mask_E = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=RANSAC_PROB,
        threshold=RANSAC_THRESH_PX,
    )
    if E is None or mask_E is None:
        return None
    cheir_count, R, t, mask_cheir = cv2.recoverPose(E, pts1, pts2, K, mask=mask_E.copy())
    return {
        "R": R,
        "t": t.reshape(3),
        "ransac_inliers": int(mask_E.sum()),
        "cheirality_inliers": int(cheir_count),
        "mask_inliers": mask_cheir.ravel().astype(bool),
        "plane_normal": None,
    }


def _recover_pose_homography(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray):
    H, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, RANSAC_THRESH_PX * 3.0)
    if H is None or mask_H is None:
        return None
    n_sols, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)

    # Drop physically impossible candidates: points must lie in front of both
    # cameras when reprojected through the recovered (R, t, n).
    inlier_mask = mask_H.ravel().astype(bool)
    inlier_pts1 = pts1[inlier_mask].reshape(-1, 1, 2)
    inlier_pts2 = pts2[inlier_mask].reshape(-1, 1, 2)
    visible_idx = cv2.filterHomographyDecompByVisibleRefpoints(
        Rs, ns, inlier_pts1, inlier_pts2,
    )

    if visible_idx is None or len(visible_idx) == 0:
        candidates = list(range(n_sols))
    else:
        candidates = [int(i) for i in visible_idx.ravel()]

    # Tie-break: pick the candidate with the most positive-depth triangulations
    # in both cameras, then by smallest reprojection error.
    best = None
    for idx in candidates:
        R, t, n = Rs[idx], ts[idx].reshape(3), ns[idx].reshape(3)
        n_pos, mean_err = _score_pose_candidate(pts1, pts2, K, R, t, inlier_mask)
        score = (n_pos, -mean_err)
        if best is None or score > best["score"]:
            best = {"R": R, "t": t, "n": n, "score": score, "mean_err": mean_err}

    if best is None:
        return None

    return {
        "R": best["R"],
        "t": best["t"],
        "ransac_inliers": int(mask_H.sum()),
        "cheirality_inliers": int(mask_H.sum()),  # H decomposition has no separate cheirality stage
        "mask_inliers": inlier_mask,
        "plane_normal": best["n"].tolist(),
    }


def _score_pose_candidate(pts1, pts2, K, R, t, mask):
    if mask.sum() < 4:
        return 0, float("inf")
    p1 = pts1[mask].T.astype(np.float64)
    p2 = pts2[mask].T.astype(np.float64)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X_h = cv2.triangulatePoints(P1, P2, p1, p2)
    w = X_h[3]
    safe = np.abs(w) > 1e-9
    X_h = X_h[:, safe]
    X = (X_h[:3] / X_h[3]).T
    z1 = X[:, 2]
    X2 = (R @ X.T + t.reshape(3, 1)).T
    z2 = X2[:, 2]
    in_front = (z1 > 0) & (z2 > 0)
    n_pos = int(in_front.sum())
    if n_pos == 0:
        return 0, float("inf")
    proj1 = (P1 @ X_h)
    proj1 = (proj1[:2] / proj1[2]).T
    proj2 = (P2 @ X_h)
    proj2 = (proj2[:2] / proj2[2]).T
    err = 0.5 * (np.linalg.norm(proj1 - p1.T, axis=1) + np.linalg.norm(proj2 - p2.T, axis=1))
    return n_pos, float(np.mean(err[in_front]))


def recover_pose(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray):
    """Try both essential matrix and homography models, return the winner.

    Selection rule: prefer homography when it has substantially more RANSAC
    inliers OR when the essential-matrix cheirality check collapses (signature
    of planar scene or pure rotation). The 'model' field in the output records
    which path won so the caller can trust translation magnitude appropriately.
    """
    e = _recover_pose_essential(pts1, pts2, K)
    h = _recover_pose_homography(pts1, pts2, K)

    if e is None and h is None:
        raise RuntimeError("Both essential-matrix and homography estimation failed.")
    if e is None:
        return _format_pose(h, "homography")
    if h is None:
        return _format_pose(e, "essential")

    e_collapsed = e["cheirality_inliers"] < max(MIN_INLIERS, 0.1 * e["ransac_inliers"])
    h_dominates = h["ransac_inliers"] > HOMOGRAPHY_PREFERENCE_RATIO * e["ransac_inliers"]

    if e_collapsed or h_dominates:
        return _format_pose(h, "homography")
    return _format_pose(e, "essential")


def _format_pose(rec: dict, model: str):
    return (
        rec["R"],
        rec["t"],
        rec["mask_inliers"],
        int(rec["cheirality_inliers"]),
        int(rec["ransac_inliers"]),
        rec["plane_normal"],
        model,
    )


def rotation_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Yaw / pitch / roll (Z-Y-X intrinsic), in degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


# ---------------------------------------------------------------------------
# Validation: triangulation, reprojection error, depth
# ---------------------------------------------------------------------------

def triangulate_and_validate(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
):
    if mask.sum() < 8:
        return float("nan"), float("nan"), 0

    p1 = pts1[mask].T  # 2 x N
    p2 = pts2[mask].T
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    X_h = cv2.triangulatePoints(P1, P2, p1, p2)
    X = (X_h[:3] / X_h[3]).T  # N x 3

    proj1 = (P1 @ X_h)
    proj1 = (proj1[:2] / proj1[2]).T
    proj2 = (P2 @ X_h)
    proj2 = (proj2[:2] / proj2[2]).T
    err1 = np.linalg.norm(proj1 - pts1[mask], axis=1)
    err2 = np.linalg.norm(proj2 - pts2[mask], axis=1)
    mean_err = float(np.mean(np.concatenate([err1, err2])))

    depths_cam1 = X[:, 2]
    X_cam2 = (R @ X.T + t.reshape(3, 1)).T
    depths_cam2 = X_cam2[:, 2]
    positive = (depths_cam1 > 0) & (depths_cam2 > 0)
    median_depth = float(np.median(depths_cam1[positive])) if positive.any() else float("nan")
    return mean_err, median_depth, int(positive.sum())


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _scale_keypoints(kps, s: float):
    return [cv2.KeyPoint(k.pt[0] * s, k.pt[1] * s, k.size * s) for k in kps]


def save_match_visualizations(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1, kp2, matches,
    inlier_mask: Optional[np.ndarray],
    out_dir: Path,
    max_side: int = 1600,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    s = min(1.0, max_side / max(h1, w1, h2, w2))
    if s < 1.0:
        small1 = cv2.resize(img1, (int(w1 * s), int(h1 * s)))
        small2 = cv2.resize(img2, (int(w2 * s), int(h2 * s)))
    else:
        small1, small2 = img1, img2
    kp1_s = _scale_keypoints(kp1, s)
    kp2_s = _scale_keypoints(kp2, s)

    all_viz = cv2.drawMatches(
        small1, kp1_s, small2, kp2_s, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out_dir / "matches.png"), all_viz)

    if inlier_mask is not None:
        inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
        in_viz = cv2.drawMatches(
            small1, kp1_s, small2, kp2_s, inlier_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(out_dir / "inliers.png"), in_viz)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def estimate(
    ref_path: Path,
    test_path: Path,
    out_dir: Path,
    user_K: Optional[np.ndarray] = None,
) -> PoseResult:
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test image not found: {test_path}")

    img1 = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(test_path), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise RuntimeError("Failed to load one or both images.")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        print(f"[warn] image sizes differ: ref={w1}x{h1}, test={w2}x{h2}", file=sys.stderr)

    intr = resolve_intrinsics(ref_path, w1, h1, user_K=user_K)
    K = intr.to_matrix()
    print(f"[info] intrinsics source: {intr.source}")
    print(f"[info] K = fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, kp2, matches, pts1, pts2, det_name = detect_and_match(g1, g2)
    print(f"[info] detector={det_name}, kp ref={len(kp1)} test={len(kp2)}, "
          f"good matches={len(matches)}")

    R, t, inlier_mask, n_cheir, n_ransac, plane_n, model = recover_pose(pts1, pts2, K)
    inlier_ratio = n_ransac / max(1, len(matches))
    print(f"[info] model={model}  RANSAC inliers={n_ransac}/{len(matches)} "
          f"({inlier_ratio:.1%})  cheirality_inliers={n_cheir}")

    n_for_threshold = n_ransac if model == "homography" else n_cheir
    if n_for_threshold < MIN_INLIERS:
        raise RuntimeError(
            f"Only {n_for_threshold} pose inliers (need >= {MIN_INLIERS}). "
            "Insufficient overlap or texture between the two views."
        )
    if inlier_ratio < 0.25:
        print("[warn] low inlier ratio — pose may be unreliable.", file=sys.stderr)
    if model == "homography" and np.linalg.norm(t) < 0.05:
        print("[warn] near-zero translation under homography — pure rotation suspected; "
              "rotation is reliable but translation direction has high uncertainty.",
              file=sys.stderr)

    mean_err, median_depth, _ = triangulate_and_validate(pts1, pts2, K, R, t, inlier_mask)
    yaw, pitch, roll = rotation_to_euler_deg(R)
    t_norm = float(np.linalg.norm(t))
    t_unit = (t / t_norm).tolist() if t_norm > 1e-9 else t.tolist()

    print()
    print(f"=== Relative Pose ({model} model — camera2 in camera1 frame) ===")
    print(f"R =\n{np.array2string(R, precision=4, suppress_small=True)}")
    print(f"euler  yaw={yaw:+.2f}deg  pitch={pitch:+.2f}deg  roll={roll:+.2f}deg")
    print(f"t        = [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]  (||t||={t_norm:.4f})")
    print(f"t (unit) = [{t_unit[0]:+.4f}, {t_unit[1]:+.4f}, {t_unit[2]:+.4f}]   (scale ambiguous)")
    if plane_n is not None:
        print(f"plane n  = [{plane_n[0]:+.4f}, {plane_n[1]:+.4f}, {plane_n[2]:+.4f}]")
    print(f"mean reprojection error = {mean_err:.3f} px")
    print(f"median triangulated depth (camera1-relative units) = {median_depth:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_match_visualizations(img1, img2, kp1, kp2, matches, inlier_mask, out_dir)

    result = PoseResult(
        R=R.tolist(),
        t=t.tolist(),
        euler_yaw_pitch_roll_deg=[yaw, pitch, roll],
        inlier_count=n_for_threshold,
        total_matches=len(matches),
        mean_reproj_error_px=mean_err,
        median_depth=median_depth,
        K=K.tolist(),
        intrinsic_source=intr.source,
        model=model,
        plane_normal=plane_n,
    )
    (out_dir / "pose.json").write_text(json.dumps(asdict(result), indent=2))
    print(f"\n[info] artifacts written to {out_dir}/  (pose.json, matches.png, inliers.png)")
    return result


def _parse_K_arg(arg: str) -> np.ndarray:
    vals = [float(x) for x in arg.split(",")]
    if len(vals) == 4:
        fx, fy, cx, cy = vals
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if len(vals) == 9:
        return np.array(vals, dtype=np.float64).reshape(3, 3)
    raise argparse.ArgumentTypeError(
        "--K must be 4 values 'fx,fy,cx,cy' or 9 values for a full 3x3"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--test", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--K",
        type=_parse_K_arg,
        default=None,
        help="Override intrinsics: 'fx,fy,cx,cy' or 9 comma-separated values.",
    )
    args = ap.parse_args()
    estimate(args.ref, args.test, args.out, user_K=args.K)


if __name__ == "__main__":
    main()
