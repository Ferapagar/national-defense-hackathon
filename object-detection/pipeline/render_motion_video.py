"""Render a side-by-side preview of motion detection: raw frame | motion mask.

Used for visual inspection of `extract_motion` output (T0.2 of the pipeline).
The mask is the OR of the intensity + edge channels.

Usage:
    python pipeline/render_motion_video.py \
        --video data/IMG_5436.MOV --out data/motion_cam0_10s.mp4 --seconds 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.motion_extraction import extract_motion  # noqa: E402


def render(
    video_path: Path,
    out_path: Path,
    seconds: float = 10.0,
    detect_mode: str = "combined",
    threshold: int = 3,
    edge_threshold: float = 0.5,
    stride: int = 1,
    morph_open_ksize: int = 3,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    n_frames = int(round(seconds * fps))

    out_w = w * 2
    out_h = h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {out_path}")

    cap = cv2.VideoCapture(str(video_path))
    motion_iter = extract_motion(
        video_path,
        camera_id=0,
        threshold=threshold,
        edge_threshold=edge_threshold,
        stride=stride,
        morph_open_ksize=morph_open_ksize,
        detect_mode=detect_mode,
        max_frames=n_frames,
    )

    written = 0
    try:
        for mf in motion_iter:
            ok, frame = cap.read()
            if not ok:
                break

            # Skip frames before the first motion frame (extract_motion drops the
            # very first frame because it has no predecessor). Re-align here.
            while written == 0 and mf.frame_index > 1:
                ok, frame = cap.read()
                if not ok:
                    break
                if mf.frame_index - 1 == 0:
                    break
                break

            mask_u8 = mf.mask
            mask_color = cv2.applyColorMap(
                cv2.normalize(mask_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_INFERNO,
            )

            label = (
                f"f={mf.frame_index}  motion_total={mf.motion_total:.1f}  "
                f"mode={detect_mode}  thr={threshold}  edge_thr={edge_threshold}"
            )
            cv2.putText(frame, "raw", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(mask_color, label, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            side = np.hstack([frame, mask_color])
            writer.write(side)
            written += 1
            if written >= n_frames:
                break
    finally:
        cap.release()
        writer.release()

    print(f"wrote {written} frames → {out_path} ({written / fps:.2f}s @ {fps:.1f} fps)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--mode", choices=["intensity", "edge", "combined"], default="combined")
    p.add_argument("--threshold", type=int, default=3)
    p.add_argument("--edge-threshold", type=float, default=0.5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--morph", type=int, default=3, help="morph-open kernel size; 0 to disable")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(
        video_path=args.video,
        out_path=args.out,
        seconds=args.seconds,
        detect_mode=args.mode,
        threshold=args.threshold,
        edge_threshold=args.edge_threshold,
        stride=args.stride,
        morph_open_ksize=args.morph,
    )


if __name__ == "__main__":
    main()
