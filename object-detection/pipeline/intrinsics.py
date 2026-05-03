"""Helpers for converting between intrinsic-matrix `K` and the scalar `fov`
used by `scene.Camera.generate_rays`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np


def K_to_fov_deg(K: np.ndarray, image_width_px: int) -> float:
    """Horizontal full-FOV (degrees) from a pinhole intrinsic matrix.

    Camera.generate_rays uses a single isotropic FOV and recomputes the
    focal length as `cx / tan(fov/2)`, so we mirror that convention.
    """
    fx = float(K[0, 0])
    half_w = image_width_px / 2.0
    return float(np.degrees(2.0 * np.arctan(half_w / fx)))


def fov_deg_to_focal_px(fov_deg: float, image_width_px: int) -> float:
    half_w = image_width_px / 2.0
    return float(half_w / np.tan(np.radians(fov_deg) / 2.0))


def load_K_from_pose_json(pose_json_path: str | Path) -> Tuple[np.ndarray, str]:
    """Load (K, intrinsic_source) from a JSON file with `K` and `intrinsic_source` keys.

    Kept as a utility; the current pipeline expects `fov_deg` directly in
    cameras.json rather than a `K` matrix.
    """
    with open(pose_json_path, "r") as f:
        payload = json.load(f)
    K = np.asarray(payload["K"], dtype=float)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 K, got shape {K.shape} in {pose_json_path}")
    return K, str(payload.get("intrinsic_source", "unknown"))


def resolution_from_K_and_principal(K: np.ndarray) -> Tuple[int, int]:
    """Recover (width, height) from the principal point of K, assuming it is image-centred."""
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return int(round(2 * cx)), int(round(2 * cy))
