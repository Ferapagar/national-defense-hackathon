"""3D viewer for FrozenScene snapshots produced by run_pipeline.py.

Two modes:
- 'static'   : aggregate all detected objects from all frozen scenes into one
               point cloud, colored by global timestamp. Best for seeing
               the full trajectory at once.
- 'animate'  : step through frozen scenes one at a time. Useful for inspecting
               temporal evolution.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Allow `python pipeline/viewer.py …` from object-detection/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import open3d as o3d  # noqa: E402

from scene import FrozenScene  # noqa: E402


def _camera_frustum_lines(
    position: np.ndarray, rotation: np.ndarray, fov_deg: float, scale: float = 0.4
) -> o3d.geometry.LineSet:
    """Build a small wireframe pyramid showing the camera pose."""
    half = np.tan(np.radians(fov_deg) / 2.0) * scale
    pts_cam = np.array([
        [0, 0, 0],
        [-half, -half, scale],
        [half, -half, scale],
        [half, half, scale],
        [-half, half, scale],
    ])
    pts_world = (rotation @ pts_cam.T).T + position
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.paint_uniform_color([0.2, 0.6, 1.0])
    return ls


def _load_cameras(cameras_json: Path | None) -> list[dict]:
    if cameras_json is None or not cameras_json.exists():
        return []
    return json.loads(cameras_json.read_text())["cameras"]


def _scene_to_pointcloud(scene: FrozenScene, color: list[float] | None = None) -> o3d.geometry.PointCloud:
    pts = np.array([d.position for d in scene.detected_objects_snapshot]) if scene.detected_objects_snapshot else np.empty((0, 3))
    confs = np.array([d.confidence for d in scene.detected_objects_snapshot])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if color is not None and len(pts) > 0:
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
    elif len(pts) > 0:
        # Default: confidence → red intensity.
        norm = (confs - confs.min()) / max(confs.max() - confs.min(), 1e-9)
        rgb = np.stack([norm, np.zeros_like(norm), 1 - norm], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def render_static(frozen: list[FrozenScene], cameras: list[dict]) -> None:
    """All frozen scenes overlaid; color each scene by its timestamp."""
    geoms: list[o3d.geometry.Geometry] = []
    if frozen:
        ts = np.array([s.timestamp for s in frozen])
        t_min, t_max = float(ts.min()), float(ts.max())
        for s in frozen:
            ratio = (s.timestamp - t_min) / max(t_max - t_min, 1e-9)
            color = [ratio, 0.5 * (1 - ratio), 1 - ratio]
            geoms.append(_scene_to_pointcloud(s, color=color))

    for cam in cameras:
        geoms.append(_camera_frustum_lines(
            position=np.asarray(cam["position"]),
            rotation=np.asarray(cam["rotation"]),
            fov_deg=cam["fov_deg"],
        ))

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    o3d.visualization.draw_geometries(geoms, window_name="Detected objects (static)")


def render_animated(frozen: list[FrozenScene], cameras: list[dict], frame_delay_s: float = 0.1) -> None:
    """Step through frozen scenes one at a time."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Detected objects (animated)")

    for cam in cameras:
        vis.add_geometry(_camera_frustum_lines(
            position=np.asarray(cam["position"]),
            rotation=np.asarray(cam["rotation"]),
            fov_deg=cam["fov_deg"],
        ))
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    for s in frozen:
        new_pcd = _scene_to_pointcloud(s)
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(frame_delay_s)

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True, type=Path, help="frozen_scenes.pkl from run_pipeline.py")
    parser.add_argument("--cameras", type=Path, default=None, help="cameras.json (optional, draws frusta)")
    parser.add_argument("--mode", choices=["static", "animate"], default="static")
    parser.add_argument("--delay-ms", type=float, default=100.0, help="Per-frame delay in animate mode.")
    args = parser.parse_args()

    with open(args.scenes, "rb") as f:
        frozen: list[FrozenScene] = pickle.load(f)
    cameras = _load_cameras(args.cameras)

    print(f"Loaded {len(frozen)} frozen scenes, {sum(len(s.detected_objects_snapshot) for s in frozen)} total detections.")

    if args.mode == "static":
        render_static(frozen, cameras)
    else:
        render_animated(frozen, cameras, frame_delay_s=args.delay_ms / 1000.0)


if __name__ == "__main__":
    main()
