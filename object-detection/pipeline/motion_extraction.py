"""Frame-difference motion extraction.

Each frame produces a binary intensity mask whose nonzero pixels are the
moving pixels (used by `scene.Camera.generate_rays`), plus a scalar feature
(total motion) used by `raycasting/calibration.py` to estimate per-camera
time offsets `dt_i`.

T0.2 of the pipeline. Pixel-level masks (per user spec, not centroids).

Detection combines two complementary channels (both ported from the
reference implementations under raycasting/prev_project/):

1. INTENSITY CHANNEL — `ray_voxel.cpp::detect_motion`
   Per-pixel absolute difference |I_t - I_{t-stride}| > threshold, with ±1
   uniform dither added per frame (`load_image_gray` in ray_voxel.cpp) so
   sub-threshold changes probabilistically surface across frames.

2. EDGE CHANNEL — adapted from `PixelationDecensorer.py::make_edge_ring_template`
   Per-pixel absolute difference of Sobel gradient magnitudes, after each
   frame's magnitude is z-score normalised (mean=0, std=1). Moving edges
   produce strong residuals exactly where raw intensity diffs vanish — this
   is what catches slow / distant / low-contrast motion that the intensity
   channel alone misses.

The two masks are OR'ed; the output intensity per pixel is the max of the
scaled diffs from each channel, so downstream ray-generation still gets a
0..255 intensity field.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np


DetectMode = Literal["intensity", "edge", "combined"]


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


def _dither(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add uniform noise in [-1, +1] and clamp to uint8.

    Ported from ray_voxel.cpp `load_image_gray`. Stochastic dithering makes
    sub-threshold pixel changes probabilistically cross the detection threshold
    over multiple frames, improving sensitivity to slow/distant objects.
    """
    noise = rng.uniform(-1.0, 1.0, gray.shape)
    return np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _edge_magnitude_normalised(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Sobel gradient magnitude, z-score normalised to (mean=0, std=1).

    Adapted from `PixelationDecensorer.py::make_edge_ring_template` and
    `track_window_next`: the same per-frame normalisation lets us compare
    edge maps across frames despite global brightness shifts.

    Returns a float32 array with the same shape as `gray`.
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    mean, std = cv2.meanStdDev(mag)
    s = float(std[0, 0])
    if s < 1e-6:
        return mag - float(mean[0, 0])
    return ((mag - float(mean[0, 0])) / (s + 1e-6)).astype(np.float32)


# Heuristic factor that maps a z-scored edge-diff (typically [0, 5]) into
# the 0..255 byte range so the output intensity stays comparable with the
# raw absdiff channel.
_EDGE_DIFF_TO_BYTE = 64.0


def extract_motion(
    video_path: str | Path,
    camera_id: int,
    threshold: int = 3,
    blur_ksize: int = 0,
    morph_open_ksize: int = 3,
    max_frames: int | None = None,
    dither: bool = True,
    stride: int = 1,
    detect_mode: DetectMode = "combined",
    edge_threshold: float = 0.5,
    edge_ksize: int = 3,
) -> Iterator[MotionFrame]:
    """Yield one MotionFrame per video frame (skipping the first, which has no
    predecessor to subtract from).

    Parameters
    ----------
    threshold : int
        Per-pixel absolute-difference threshold for the INTENSITY channel
        (0..255). Ported from ray_voxel.cpp (motion_threshold=2.0); default 3
        catches sub-pixel motion invisible to the old threshold of 25.
    blur_ksize : int
        Gaussian blur kernel before differencing. Default 0 (off) — blur
        suppresses the small differences we now want to detect.
    morph_open_ksize : int
        Morphological-open kernel size; removes isolated noise pixels. 0 = off.
    max_frames : int | None
        If set, stop after this many frames (useful for the calibration warm-up).
    dither : bool
        Add ±1 uniform noise before differencing (ray_voxel.cpp technique).
        Surfaces sub-threshold changes stochastically. Default True.
    stride : int
        Compare frame i against frame i-stride instead of i-1. Larger values
        accumulate more displacement for slow/distant objects.
    detect_mode : {"intensity", "edge", "combined"}
        Which detection channel(s) to run.
        - "intensity": pure absdiff (ray_voxel.cpp behaviour).
        - "edge": absdiff of z-normalised Sobel magnitudes only.
        - "combined" (default): OR of both — recommended.
    edge_threshold : float
        Threshold on the z-normalised edge-magnitude difference. Z-scored
        magnitudes have unit-ish std, so 0.5 catches motion on edges that
        the intensity channel can't see (slow/distant/low-contrast objects).
    edge_ksize : int
        Sobel kernel size for the edge channel (1, 3, 5, or 7). Default 3.
    """
    if detect_mode not in ("intensity", "edge", "combined"):
        raise ValueError(f"unknown detect_mode={detect_mode!r}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt_per_frame = 1.0 / fps

    rng = np.random.default_rng()

    # Buffer of the last `stride+1` frames to support stride > 1.
    buf: deque[np.ndarray] = deque(maxlen=stride + 1)

    ok, first_bgr = cap.read()
    if not ok:
        cap.release()
        return
    first_gray = _grayscale(first_bgr)
    if blur_ksize > 0:
        first_gray = cv2.GaussianBlur(first_gray, (blur_ksize, blur_ksize), 0)
    buf.append(first_gray)

    morph_kernel = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open_ksize, morph_open_ksize))
        if morph_open_ksize > 0 else None
    )

    use_intensity = detect_mode in ("intensity", "combined")
    use_edge = detect_mode in ("edge", "combined")

    frame_index = 1
    try:
        while True:
            ok, cur_bgr = cap.read()
            if not ok:
                break

            cur_gray = _grayscale(cur_bgr)
            if blur_ksize > 0:
                cur_gray = cv2.GaussianBlur(cur_gray, (blur_ksize, blur_ksize), 0)
            buf.append(cur_gray)

            if len(buf) == stride + 1:
                prev_gray = buf[0]
                a = _dither(cur_gray, rng) if dither else cur_gray
                b = _dither(prev_gray, rng) if dither else prev_gray

                mask_total = np.zeros(cur_gray.shape, dtype=bool)
                intensity = np.zeros(cur_gray.shape, dtype=np.uint8)

                if use_intensity:
                    diff_int = cv2.absdiff(a, b)
                    mask_int = diff_int > threshold
                    mask_total |= mask_int
                    intensity = np.maximum(intensity, np.where(mask_int, diff_int, 0).astype(np.uint8))

                if use_edge:
                    mag_cur = _edge_magnitude_normalised(a, ksize=edge_ksize)
                    mag_prev = _edge_magnitude_normalised(b, ksize=edge_ksize)
                    diff_edge = cv2.absdiff(mag_cur, mag_prev)  # float32
                    mask_edge = diff_edge > edge_threshold
                    mask_total |= mask_edge
                    diff_edge_u8 = np.clip(diff_edge * _EDGE_DIFF_TO_BYTE, 0, 255).astype(np.uint8)
                    intensity = np.maximum(intensity, np.where(mask_edge, diff_edge_u8, 0))

                if morph_kernel is not None:
                    mask_u8 = mask_total.astype(np.uint8) * 255
                    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, morph_kernel)
                    mask_total = mask_u8 > 0

                intensity = np.where(mask_total, intensity, 0).astype(np.uint8)
            else:
                intensity = np.zeros(cur_gray.shape, dtype=np.uint8)

            yield MotionFrame(
                camera_id=camera_id,
                frame_index=frame_index,
                t_local=frame_index * dt_per_frame,
                mask=intensity,
                motion_total=float(intensity.sum()) / 255.0,
            )

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
