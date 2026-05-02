"""Frame-difference motion extraction.

Each frame produces a binary intensity mask whose nonzero pixels are the
moving pixels (used by `scene.Camera.generate_rays`), plus a scalar feature
(total motion) used by `raycasting/calibration.py` to estimate per-camera
time offsets `dt_i`.

T0.2 of the pipeline. Pixel-level masks (per user spec, not centroids).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionFrame:
    camera_id: int
    frame_index: int
    t_local: float        # seconds from start of this video
    mask: np.ndarray      # uint8 (H, W); nonzero = motion pixel intensity 0..255
    motion_total: float   # scalar = mask.sum() / 255.0; calibration feature


def _grayscale(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def extract_motion(
    video_path: str | Path,
    camera_id: int,
    threshold: int = 25,
    blur_ksize: int = 5,
    morph_open_ksize: int = 3,
    max_frames: int | None = None,
) -> Iterator[MotionFrame]:
    """Yield one MotionFrame per video frame (skipping the first, which has no
    predecessor to subtract from).

    Parameters
    ----------
    threshold : int
        Per-pixel absolute-difference threshold (0..255). Pixels above this
        survive into the mask.
    blur_ksize : int
        Gaussian blur kernel before differencing; reduces sensor noise. 0 = off.
    morph_open_ksize : int
        Morphological-open kernel size; removes 1-pixel speckle. 0 = off.
    max_frames : int | None
        If set, stop after this many frames (useful for the calibration warm-up).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt_per_frame = 1.0 / fps

    ok, prev_bgr = cap.read()
    if not ok:
        cap.release()
        return
    prev_gray = _grayscale(prev_bgr)
    if blur_ksize > 0:
        prev_gray = cv2.GaussianBlur(prev_gray, (blur_ksize, blur_ksize), 0)

    morph_kernel = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open_ksize, morph_open_ksize))
        if morph_open_ksize > 0 else None
    )

    frame_index = 1
    try:
        while True:
            ok, cur_bgr = cap.read()
            if not ok:
                break
            cur_gray = _grayscale(cur_bgr)
            if blur_ksize > 0:
                cur_gray = cv2.GaussianBlur(cur_gray, (blur_ksize, blur_ksize), 0)

            diff = cv2.absdiff(cur_gray, prev_gray)
            _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            if morph_kernel is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)
            # Keep the *intensity* of the difference where the binary mask passes,
            # so brighter motion contributes more confidence to the voxel grid.
            intensity = np.where(mask > 0, diff, 0).astype(np.uint8)

            yield MotionFrame(
                camera_id=camera_id,
                frame_index=frame_index,
                t_local=frame_index * dt_per_frame,
                mask=intensity,
                motion_total=float(intensity.sum()) / 255.0,
            )

            prev_gray = cur_gray
            frame_index += 1
            if max_frames is not None and frame_index > max_frames:
                break
    finally:
        cap.release()


def collect_calibration_history(
    video_path: str | Path,
    camera_id: int,
    n_frames: int = 30,
    **kwargs,
) -> tuple[list[MotionFrame], np.ndarray]:
    """Run the warm-up window for time-sync calibration.

    Returns
    -------
    frames : list of the first n_frames MotionFrame records (kept for the
        T1 ray-generation step so we don't re-decode the video).
    history : 1-D float array of `motion_total` per frame; this is what
        `raycasting/calibration.compute_pairwise_dt` cross-correlates.
    """
    frames: list[MotionFrame] = []
    for mf in extract_motion(video_path, camera_id, max_frames=n_frames, **kwargs):
        frames.append(mf)
    history = np.asarray([f.motion_total for f in frames], dtype=float)
    return frames, history
