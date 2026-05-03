"""Render a 2x2 sync-verification video.

Layout (both rows are time-aligned in global time):

    +----------------------+----------------------+
    |  cam 0 raw           |  cam 1 raw           |
    |  (anchor)            |  (shifted by dt_1)   |
    +----------------------+----------------------+
    |  cam 0 motion mask   |  cam 1 motion mask   |
    +----------------------+----------------------+

Time mapping
------------
At output frame k (global time `t_global = t_start + k / fps_out`):
    cam i shows its frame at local time `t_global - dt_i`.

`dt_i` is the per-camera offset (seconds) — pass either the values from
debug_sync.py or from run_pipeline.py.

Usage
-----
    python pipeline/render_sync_grid.py \
        --videos data/IMG_5436.MOV data/PXL_20260502_222532213.MP4 \
        --dt-seconds 0.0 0.0218 \
        --out data/sync_grid.mp4 --seconds 10
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CamReader:
    cap: cv2.VideoCapture
    fps: float
    w: int
    h: int
    dt: float                # seconds; cam local-time origin in global frame
    label: str
    last_idx: int = -1
    prev_gray: np.ndarray | None = None
    cur_gray: np.ndarray | None = None
    cur_bgr: np.ndarray | None = None


def _open_reader(video_path: Path, dt: float, label: str) -> CamReader:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return CamReader(cap=cap, fps=float(fps), w=w, h=h, dt=dt, label=label)


def _advance_to(reader: CamReader, target_idx: int) -> bool:
    """Advance reader.cap to absolute frame `target_idx`. Maintains prev_gray /
    cur_gray for motion differencing.
    """
    if target_idx < 0:
        return False
    # If we're behind, read forward; never seek backwards (mp4 keyframe seek
    # is unreliable). For the small jumps we do here, sequential reads are
    # fast enough and correct.
    if target_idx <= reader.last_idx:
        # Reset and seek if user asked us to go backwards.
        reader.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_idx))
        reader.last_idx = target_idx - 1
        reader.prev_gray = None
        reader.cur_gray = None

    while reader.last_idx < target_idx:
        ok, frame = reader.cap.read()
        if not ok:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reader.prev_gray = reader.cur_gray
        reader.cur_gray = gray
        reader.cur_bgr = frame
        reader.last_idx += 1
    return True


def _mask(reader: CamReader, threshold: int = 3,
          edge_threshold: float = 0.5,
          rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a uint8 motion-intensity image (same logic as
    motion_extraction.extract_motion in detect_mode='combined').
    """
    cur = reader.cur_gray
    prev = reader.prev_gray
    if cur is None or prev is None:
        return np.zeros((reader.h, reader.w), dtype=np.uint8)

    if rng is not None:
        a = np.clip(cur.astype(np.float32)
                    + rng.uniform(-1.0, 1.0, cur.shape), 0, 255).astype(np.uint8)
        b = np.clip(prev.astype(np.float32)
                    + rng.uniform(-1.0, 1.0, prev.shape), 0, 255).astype(np.uint8)
    else:
        a, b = cur, prev

    # Intensity channel
    diff_int = cv2.absdiff(a, b)
    mask_int = diff_int > threshold
    intensity = np.where(mask_int, diff_int, 0).astype(np.uint8)

    # Edge channel (z-normalised Sobel magnitudes)
    def _edge(gray):
        g = gray.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        m = mag.mean(); s = mag.std()
        if s < 1e-6:
            return mag - m
        return (mag - m) / (s + 1e-6)

    diff_edge = cv2.absdiff(_edge(a), _edge(b))
    mask_edge = diff_edge > edge_threshold
    edge_u8 = np.clip(diff_edge * 64.0, 0, 255).astype(np.uint8)
    intensity = np.maximum(intensity, np.where(mask_edge, edge_u8, 0))

    mask_total = mask_int | mask_edge
    return np.where(mask_total, intensity, 0).astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def render(video_paths: list[Path], dt_seconds: list[float],
           out_path: Path, seconds: float, fps_out: float = 30.0,
           tile_h: int = 540) -> None:
    if len(video_paths) != 2 or len(dt_seconds) != 2:
        raise ValueError("Need exactly 2 videos and 2 dt values.")

    readers = [
        _open_reader(video_paths[0], dt_seconds[0], "cam 0 (anchor)"),
        _open_reader(video_paths[1], dt_seconds[1], f"cam 1  dt={dt_seconds[1]*1000:+.1f} ms"),
    ]

    # Use the earlier camera's clock as t=0 in global time, but skip ahead
    # so that no camera is asked for negative local time.
    t_start = max(0.0, max(dt_seconds))

    aspect = readers[0].w / readers[0].h
    tile_w = int(round(tile_h * aspect))
    out_w = tile_w * 2
    out_h = tile_h * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {out_path}")

    rng = np.random.default_rng(0)
    n_out = int(round(seconds * fps_out))

    written = 0
    try:
        for k in range(n_out):
            t_global = t_start + k / fps_out
            row_top: list[np.ndarray] = []
            row_bot: list[np.ndarray] = []

            for r in readers:
                t_local = t_global - r.dt
                target_idx = int(round(t_local * r.fps))
                if not _advance_to(r, target_idx):
                    raw = np.zeros((r.h, r.w, 3), dtype=np.uint8)
                    mask = np.zeros((r.h, r.w), dtype=np.uint8)
                else:
                    raw = r.cur_bgr if r.cur_bgr is not None else np.zeros((r.h, r.w, 3), dtype=np.uint8)
                    mask = _mask(r, rng=rng)

                mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_INFERNO)

                raw_resized = cv2.resize(raw, (tile_w, tile_h))
                mask_resized = cv2.resize(mask_color, (tile_w, tile_h))

                tag = (
                    f"{r.label}  t_global={t_global:.3f}s  "
                    f"local_idx={target_idx}  motion={int(mask.sum() / 255)}"
                )
                row_top.append(_label(raw_resized, tag))
                row_bot.append(_label(mask_resized, tag))

            top = np.hstack(row_top)
            bot = np.hstack(row_bot)
            grid = np.vstack([top, bot])
            writer.write(grid)
            written += 1
    finally:
        for r in readers:
            r.cap.release()
        writer.release()

    print(f"wrote {written} frames → {out_path} "
          f"({written / fps_out:.2f}s @ {fps_out:.1f} fps, {out_w}x{out_h})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", required=True, nargs=2, type=Path,
                   help="Exactly two videos, in cam-id order (cam 0, cam 1).")
    p.add_argument("--dt-seconds", required=True, nargs=2, type=float,
                   help="Per-camera offsets in seconds; first is the anchor (typically 0.0).")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=30.0,
                   help="Output framerate.")
    p.add_argument("--tile-h", type=int, default=540,
                   help="Per-tile height in px; tile width is computed from cam 0 aspect.")
    args = p.parse_args()

    render(
        video_paths=args.videos,
        dt_seconds=list(args.dt_seconds),
        out_path=args.out,
        seconds=args.seconds,
        fps_out=args.fps,
        tile_h=args.tile_h,
    )


if __name__ == "__main__":
    main()
