"""Build a single PNG that visually explains what the relative-pose
estimator did for an image pair.

For an essential-matrix result it draws inlier matches plus epipolar
lines on the second image. For a homography result it warps the first
image into the second image's frame and shows the alignment, since that
is the natural geometric explanation for a planar / pure-rotation case.

Usage:
    python make_inspection.py --ref ../../images/ref-0.jpg --test ../../images/test-0.jpg --out out/pair0/inspection.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from estimate_relative_pose import (
    detect_and_match,
    recover_pose,
    resolve_intrinsics,
)


SCALE_LONG = 900  # max long side per panel, keeps output small enough to view


def _resize(img: np.ndarray):
    h, w = img.shape[:2]
    s = SCALE_LONG / max(h, w)
    if s >= 1.0:
        return img.copy(), 1.0
    return cv2.resize(img, (int(w * s), int(h * s))), s


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(out, text, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _draw_essential_panel(img1, img2, pts1_in, pts2_in, K) -> np.ndarray:
    """Side-by-side inlier matches + epipolar lines on img2."""
    s1_img, s = _resize(img1)
    s2_img, _ = _resize(img2)
    pts1 = pts1_in * s
    pts2 = pts2_in * s

    F, _ = cv2.findFundamentalMat(pts1_in, pts2_in, cv2.FM_RANSAC, 1.0, 0.999)
    if F is None:
        F = np.eye(3)

    n = min(40, len(pts1))
    idx = np.linspace(0, len(pts1) - 1, n).astype(int)
    pts1_sub = pts1[idx]
    pts2_sub = pts2[idx]

    h1, w1 = s1_img.shape[:2]
    h2, w2 = s2_img.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = s1_img
    canvas[:h2, w1:w1 + w2] = s2_img

    rng = np.random.default_rng(0)
    for (p1, p2) in zip(pts1_sub, pts2_sub):
        color = tuple(int(c) for c in rng.integers(60, 230, size=3))
        a = (int(round(p1[0])), int(round(p1[1])))
        b = (int(round(p2[0]) + w1), int(round(p2[1])))
        cv2.circle(canvas, a, 4, color, 2)
        cv2.circle(canvas, b, 4, color, 2)
        cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)

    lines = cv2.computeCorrespondEpilines(pts1_in[idx].reshape(-1, 1, 2), 1, F)
    if lines is not None:
        lines = lines.reshape(-1, 3)
        for line in lines:
            a, b, c = line
            a_s, b_s, c_s = a / s, b / s, c / s
            x0, x1 = 0, w2 - 1
            if abs(b_s) < 1e-6:
                continue
            y0 = int(-(a_s * x0 + c_s) / b_s)
            y1 = int(-(a_s * x1 + c_s) / b_s)
            cv2.line(canvas, (x0 + w1, y0), (x1 + w1, y1),
                     (0, 220, 220), 1, cv2.LINE_AA)

    return _label(canvas, "ESSENTIAL: inlier matches + epipolar lines on right image")


def _draw_homography_panel(img1, img2, pts1_in, pts2_in) -> np.ndarray:
    """Warp img1 -> img2 frame via the inlier-fitted homography and overlay."""
    s1_img, s = _resize(img1)
    s2_img, _ = _resize(img2)
    h2, w2 = s2_img.shape[:2]

    H_full, _ = cv2.findHomography(pts1_in, pts2_in, cv2.RANSAC, 3.0)
    if H_full is None:
        H_full = np.eye(3)
    S = np.diag([s, s, 1.0])
    H_disp = S @ H_full @ np.linalg.inv(S)

    warped = cv2.warpPerspective(s1_img, H_disp, (w2, h2))
    overlay = cv2.addWeighted(s2_img, 0.5, warped, 0.5, 0)
    diff = cv2.absdiff(s2_img, warped)
    diff_vis = cv2.applyColorMap(
        np.clip(diff.mean(axis=2) * 2, 0, 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )

    a = _label(s2_img, "TEST image")
    b = _label(warped, "REF warped via H into TEST frame")
    c = _label(overlay, "Overlay 50/50 -- sharp regions = good alignment")
    d = _label(diff_vis, "Per-pixel difference (warped vs test)")
    top = np.hstack([a, b])
    bot = np.hstack([c, d])
    return np.vstack([top, bot])


def build_inspection(ref_path: Path, test_path: Path, out_path: Path):
    img1 = cv2.imread(str(ref_path))
    img2 = cv2.imread(str(test_path))
    if img1 is None or img2 is None:
        raise RuntimeError("Failed to load images.")

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, _, _, pts1, pts2, _ = detect_and_match(g1, g2)

    K = resolve_intrinsics(ref_path, img1.shape[1], img1.shape[0]).to_matrix()
    _R, _t, mask, n_cheir, n_ransac, _n_plane, model = recover_pose(pts1, pts2, K)

    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    if model == "essential":
        panel = _draw_essential_panel(img1, img2, pts1_in, pts2_in, K)
    else:
        panel = _draw_homography_panel(img1, img2, pts1_in, pts2_in)

    header = np.zeros((60, panel.shape[1], 3), dtype=np.uint8)
    info = (f"{ref_path.name} -> {test_path.name}   "
            f"model={model}   ransac_inliers={n_ransac}   cheirality={n_cheir}")
    cv2.putText(header, info, (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    final = np.vstack([header, panel])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"wrote {out_path}  ({final.shape[1]}x{final.shape[0]} px)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build_inspection(args.ref, args.test, args.out)


if __name__ == "__main__":
    main()
