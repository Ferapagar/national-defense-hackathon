"""End-to-end driver: cameras.json + N videos → list of FrozenScene snapshots.

Stages map onto the user's T0..T3:
  T0.1 — already done by build_world_frame; produces cameras.json
  T0.2 — motion_extraction.extract_motion: per-frame intensity masks + scalar
  T1   — Camera.generate_rays converts mask → RayBatch with global timestamp
  T2.B — GlobalScene.calibrate solves dt_i from warm-up motion histories
  T2.C — GlobalScene.aggregate_rays casts rays into the voxel grid; clear_grid
         + freeze gives one FrozenScene per global-time bin
  T3   — viewer.py reads the saved scenes and renders them
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Allow running as `python pipeline/run_pipeline.py …` from object-detection/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene import Camera, GlobalScene, FrozenScene  # noqa: E402
from pipeline.motion_extraction import (  # noqa: E402
    MotionFrame,
    collect_calibration_history,
    extract_motion,
)


@dataclass(frozen=True)
class CameraVideo:
    camera: Camera
    video_path: Path
    fps: float


def _read_video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def build_scene_from_cameras_json(
    cameras_json: Path,
    video_paths: list[Path],
    voxel_grid_extent: list[tuple[float, float]],
    voxel_grid_size: tuple[int, int, int],
) -> tuple[GlobalScene, list[CameraVideo]]:
    payload = json.loads(cameras_json.read_text())
    records = payload["cameras"]
    if len(records) != len(video_paths):
        raise ValueError(
            f"cameras.json has {len(records)} cams but {len(video_paths)} videos given."
        )

    scene = GlobalScene(voxel_grid_extent=voxel_grid_extent, voxel_grid_size=voxel_grid_size)
    cam_videos: list[CameraVideo] = []
    for rec, vid in zip(records, video_paths):
        # scene.Camera.resolution is (H, W); cameras.json stores [W, H].
        w, h = rec["resolution_wh"]
        cam = Camera(
            camera_id=rec["camera_id"],
            fov=rec["fov_deg"],
            resolution=(h, w),
            position=np.asarray(rec["position"], dtype=float),
            rotation=np.asarray(rec["rotation"], dtype=float),
        )
        scene.add_camera(cam)
        cam_videos.append(CameraVideo(camera=cam, video_path=Path(vid), fps=_read_video_fps(Path(vid))))

    return scene, cam_videos


def calibrate_time_offsets(
    scene: GlobalScene,
    cam_videos: list[CameraVideo],
    calibration_n_frames: int,
    motion_kwargs: dict,
) -> list[list[MotionFrame]]:
    """Run the warm-up window per camera, push the motion-total history into
    each Camera, then call scene.calibrate(). Convert frame-shift dt_i values
    into seconds using each camera's video fps.

    Returns the warm-up MotionFrame lists so the caller can include them in
    the main aggregation pass without re-decoding.
    """
    warmup_frames_per_cam: list[list[MotionFrame]] = []
    for cv_ in cam_videos:
        frames, history = collect_calibration_history(
            cv_.video_path, cv_.camera.camera_id,
            n_frames=calibration_n_frames, **motion_kwargs,
        )
        for h in history:
            cv_.camera.add_history_frame(float(h))
        warmup_frames_per_cam.append(frames)

    scene.calibrate()
    # compute_pairwise_dt returns integer frame shifts. Convert to seconds.
    for cv_ in cam_videos:
        cv_.camera.dt_i = float(cv_.camera.dt_i) / cv_.fps

    return warmup_frames_per_cam


def _stream_motion(
    cam_videos: list[CameraVideo],
    warmup_frames_per_cam: list[list[MotionFrame]],
    motion_kwargs: dict,
    max_frames_per_camera: int | None,
) -> Iterable[tuple[float, Camera, MotionFrame]]:
    """Yield (t_global, Camera, MotionFrame) sorted by t_global across all cams."""
    merged: list[tuple[float, Camera, MotionFrame]] = []

    for cv_, warmup_frames in zip(cam_videos, warmup_frames_per_cam):
        # Replay warm-up frames first (already decoded), then continue from where we left off.
        for mf in warmup_frames:
            merged.append((mf.t_local + cv_.camera.dt_i, cv_.camera, mf))

        n_warmup = len(warmup_frames)
        if max_frames_per_camera is not None and n_warmup >= max_frames_per_camera:
            continue

        # Re-open video and skip past the warm-up window.
        # extract_motion always yields frame_index starting at 1; we grab everything
        # then drop the first `n_warmup` to avoid double counting.
        remaining_cap = (
            None if max_frames_per_camera is None else max_frames_per_camera
        )
        seen = 0
        for mf in extract_motion(cv_.video_path, cv_.camera.camera_id,
                                  max_frames=remaining_cap, **motion_kwargs):
            seen += 1
            if seen <= n_warmup:
                continue
            merged.append((mf.t_local + cv_.camera.dt_i, cv_.camera, mf))

    merged.sort(key=lambda x: x[0])
    return merged


def run_pipeline(
    cameras_json: Path,
    video_paths: list[Path],
    voxel_grid_extent: list[tuple[float, float]],
    voxel_grid_size: tuple[int, int, int] = (64, 64, 64),
    time_bin_s: float = 1.0 / 30.0,
    detection_threshold: float = 1.0,
    calibration_n_frames: int = 30,
    max_frames_per_camera: int | None = None,
    motion_kwargs: dict | None = None,
) -> list[FrozenScene]:
    motion_kwargs = motion_kwargs or {}
    scene, cam_videos = build_scene_from_cameras_json(
        cameras_json, video_paths, voxel_grid_extent, voxel_grid_size,
    )

    print(f"Calibrating time offsets across {len(cam_videos)} cameras "
          f"(warm-up = {calibration_n_frames} frames each)…")
    warmup = calibrate_time_offsets(scene, cam_videos, calibration_n_frames, motion_kwargs)
    for cv_ in cam_videos:
        print(f"  cam {cv_.camera.camera_id}: dt_i = {cv_.camera.dt_i*1000:+.1f} ms (fps={cv_.fps:.1f})")

    print("Streaming motion frames and aggregating rays into voxel grid…")
    frozen: list[FrozenScene] = []
    bin_start: float | None = None
    n_bins = 0

    for t_global, cam, mf in _stream_motion(cam_videos, warmup, motion_kwargs, max_frames_per_camera):
        if bin_start is None:
            bin_start = t_global
        if (t_global - bin_start) >= time_bin_s:
            scene.detect_objects(threshold=detection_threshold, t_global=bin_start)
            frozen.append(scene.freeze(timestamp=bin_start))
            scene.clear_grid()
            bin_start = t_global
            n_bins += 1

        ray_batch = cam.generate_rays(mf.mask, mf.t_local)
        scene.aggregate_rays(ray_batch)

    # Flush the last bin.
    if bin_start is not None:
        scene.detect_objects(threshold=detection_threshold, t_global=bin_start)
        frozen.append(scene.freeze(timestamp=bin_start))

    print(f"Produced {len(frozen)} frozen scenes across {n_bins+1} time bins "
          f"(bin width = {time_bin_s*1000:.1f} ms).")
    return frozen


def save_frozen_scenes(frozen: list[FrozenScene], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(frozen, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", required=True, type=Path, help="cameras.json from build_world_frame.py")
    parser.add_argument("--videos", required=True, nargs="+", type=Path,
                        help="One video per camera, in cameras.json order.")
    parser.add_argument("--extent", required=True, type=str,
                        help="Voxel-grid extent as 'xmin,xmax,ymin,ymax,zmin,zmax' (metres).")
    parser.add_argument("--grid-size", type=str, default="64,64,64", help="'Dx,Dy,Dz'")
    parser.add_argument("--time-bin-ms", type=float, default=33.3,
                        help="Width of each time bin (ms). One FrozenScene per bin.")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Voxel intensity threshold above which a voxel becomes a DetectedObject.")
    parser.add_argument("--calibration-frames", type=int, default=30,
                        help="Warm-up frames per camera for time-sync.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap on frames decoded per camera (debug).")
    parser.add_argument("--out", type=Path, default=Path("frozen_scenes.pkl"))
    args = parser.parse_args()

    extent_vals = [float(v) for v in args.extent.split(",")]
    if len(extent_vals) != 6:
        parser.error("--extent must be six floats: xmin,xmax,ymin,ymax,zmin,zmax")
    extent = [(extent_vals[0], extent_vals[1]), (extent_vals[2], extent_vals[3]),
              (extent_vals[4], extent_vals[5])]
    grid_size = tuple(int(v) for v in args.grid_size.split(","))
    if len(grid_size) != 3:
        parser.error("--grid-size must be three ints: Dx,Dy,Dz")

    frozen = run_pipeline(
        cameras_json=args.cameras,
        video_paths=args.videos,
        voxel_grid_extent=extent,
        voxel_grid_size=grid_size,
        time_bin_s=args.time_bin_ms / 1000.0,
        detection_threshold=args.threshold,
        calibration_n_frames=args.calibration_frames,
        max_frames_per_camera=args.max_frames,
    )
    save_frozen_scenes(frozen, args.out)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
