"""Headless 3D animation of FrozenScene snapshots → MP4.

Uses matplotlib's 3D scatter (no Open3D / no display required) so the result
can be inspected as a video. Each frame shows:
  - Camera frusta (blue)
  - Voxel detections for that time bin (coloured by confidence)
  - World-axis frame
The view rotates slowly so the 3D structure is readable without interaction.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene import FrozenScene  # noqa: E402


def _frustum_segments(position: np.ndarray, rotation: np.ndarray,
                      fov_deg: float, scale: float = 0.5) -> np.ndarray:
    """Return Nx2x3 line segments for a small camera pyramid in world coords."""
    half = float(np.tan(np.radians(fov_deg) / 2.0) * scale)
    pts_cam = np.array([
        [0.0, 0.0, 0.0],
        [-half, -half, scale],
        [half, -half, scale],
        [half, half, scale],
        [-half, half, scale],
    ])
    pts_world = (rotation @ pts_cam.T).T + position
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return np.array([[pts_world[a], pts_world[b]] for a, b in edges])


def render(
    scenes_pkl: Path,
    cameras_json: Path,
    out_path: Path,
    fps: float = 30.0,
    extent: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    max_points_per_frame: int = 4000,
) -> None:
    with open(scenes_pkl, "rb") as f:
        frozen: list[FrozenScene] = pickle.load(f)
    cameras = json.loads(cameras_json.read_text())["cameras"]

    if extent is None:
        # Auto-fit: use camera positions + a generous box.
        cam_pos = np.array([c["position"] for c in cameras])
        center = cam_pos.mean(axis=0)
        extent = (
            (float(center[0] - 12), float(center[0] + 12)),
            (float(center[1] - 12), float(center[1] + 12)),
            (float(center[2] - 2),  float(center[2] + 30)),
        )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    n_frames = len(frozen)
    print(f"rendering {n_frames} frozen scenes → {out_path}")

    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=4000)
    with writer.saving(fig, str(out_path), dpi=120):
        for i, s in enumerate(frozen):
            ax.clear()

            # Frusta
            for cam in cameras:
                segs = _frustum_segments(
                    np.asarray(cam["position"]),
                    np.asarray(cam["rotation"]),
                    cam["fov_deg"], scale=0.6,
                )
                for a, b in segs:
                    ax.plot(*zip(a, b), color="#3aa0ff", linewidth=1.0, alpha=0.9)
                ax.scatter(*cam["position"], color="#3aa0ff", s=20)

            # Detections
            objs = s.detected_objects_snapshot
            if objs:
                pts = np.array([o.position for o in objs])
                confs = np.array([o.confidence for o in objs])
                if len(pts) > max_points_per_frame:
                    idx = np.argsort(-confs)[:max_points_per_frame]
                    pts = pts[idx]; confs = confs[idx]
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c=confs, cmap="inferno", s=6, alpha=0.85,
                    vmin=float(confs.min()), vmax=float(confs.max()),
                )

            ax.set_xlim(extent[0]); ax.set_ylim(extent[1]); ax.set_zlim(extent[2])
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
            ax.set_title(
                f"t={s.timestamp:.2f}s   detections={len(objs)}   bin {i+1}/{n_frames}"
            )

            # Slow orbit
            azim = -60.0 + (i / max(n_frames - 1, 1)) * 60.0
            ax.view_init(elev=20.0, azim=azim)

            writer.grab_frame()

            if (i + 1) % 30 == 0:
                print(f"  frame {i+1}/{n_frames}  (t={s.timestamp:.2f}s, "
                      f"{len(objs)} dets)")

    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", required=True, type=Path)
    p.add_argument("--cameras", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--extent", type=str, default=None,
                   help="xmin,xmax,ymin,ymax,zmin,zmax (defaults to a box around the cameras).")
    args = p.parse_args()

    extent = None
    if args.extent:
        v = [float(x) for x in args.extent.split(",")]
        if len(v) != 6:
            p.error("--extent must be six floats")
        extent = ((v[0], v[1]), (v[2], v[3]), (v[4], v[5]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(args.scenes, args.cameras, args.out, fps=args.fps, extent=extent)


if __name__ == "__main__":
    main()
