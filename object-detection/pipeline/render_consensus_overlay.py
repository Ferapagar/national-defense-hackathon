"""Back-project consensus 3D detections onto each camera's raw video.

Pipeline:
  - Load FrozenScene snapshots (scene.detect_objects already kept only voxels
    that >= `consensus` cameras saw).
  - For each frame in each camera's raw video, find the frozen scene whose
    global timestamp matches that frame's global timestamp (t_local + dt_i).
  - Project every detection's 3D world position into both cameras' image
    planes using the same focal-length / rotation / position model that
    `Camera.generate_rays` uses (just inverted).
  - Draw the projected points coloured by which camera *originated* the
    consensus — but since DetectedObject doesn't track per-camera origin
    after the consensus step, we use a single colour per camera view to
    distinguish "this is what cam0 thinks vs cam1 thinks of the same 3D point".

The result is a side-by-side cam0 | cam1 video where matching coloured dots
in each pane correspond to the same 3D object — exactly what you'd expect
if the consensus detection is correct.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene import FrozenScene  # noqa: E402


@dataclass(frozen=True)
class CamModel:
    camera_id: int
    position: np.ndarray  # (3,)
    rotation: np.ndarray  # (3,3) — camera→world (matches scene.Camera.generate_rays)
    fov_deg: float
    width: int
    height: int


def _load_camera_models(cameras_json: Path) -> list[CamModel]:
    payload = json.loads(cameras_json.read_text())
    out: list[CamModel] = []
    for rec in payload["cameras"]:
        w, h = rec["resolution_wh"]
        out.append(CamModel(
            camera_id=int(rec["camera_id"]),
            position=np.asarray(rec["position"], dtype=float),
            rotation=np.asarray(rec["rotation"], dtype=float),
            fov_deg=float(rec["fov_deg"]),
            width=int(w),
            height=int(h),
        ))
    return out


def _project(points_world: np.ndarray, cam: CamModel) -> tuple[np.ndarray, np.ndarray]:
    """Project (N,3) world points → (M,2) pixel coords + (M,) z-distances.

    Uses the same camera model as `scene.Camera.generate_rays`: rotation maps
    camera-frame → world-frame, so world→camera is `rotation.T @ (P - pos)`.
    Pinhole intrinsics derived from `fov_deg` and image width.
    """
    if len(points_world) == 0:
        return np.empty((0, 2)), np.empty((0,))

    cx = cam.width / 2.0
    cy = cam.height / 2.0
    focal = cx / np.tan(np.radians(cam.fov_deg) / 2.0)

    rel = points_world - cam.position[None, :]
    p_cam = rel @ cam.rotation  # equivalent to (rotation.T @ rel.T).T

    z = p_cam[:, 2]
    in_front = z > 1e-3
    if not np.any(in_front):
        return np.empty((0, 2)), np.empty((0,))

    p_cam = p_cam[in_front]
    z = z[in_front]
    u = focal * (p_cam[:, 0] / z) + cx
    v = focal * (p_cam[:, 1] / z) + cy

    in_bounds = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return np.stack([u[in_bounds], v[in_bounds]], axis=1), z[in_bounds]


def _scene_for_time(frozen: list[FrozenScene], t: float) -> FrozenScene | None:
    """Find the frozen scene whose timestamp window contains t (linear scan, fine for 100 bins)."""
    if not frozen:
        return None
    timestamps = np.array([s.timestamp for s in frozen])
    idx = int(np.argmin(np.abs(timestamps - t)))
    return frozen[idx]


def _overlay_dots(frame: np.ndarray, uv: np.ndarray, confs: np.ndarray, color_bgr: tuple[int, int, int],
                  radius: int = 4, top_k: int | None = 200) -> None:
    """Draw circles at each (u, v); brightness scales with confidence."""
    if len(uv) == 0:
        return
    if top_k is not None and len(confs) > top_k:
        keep = np.argsort(-confs)[:top_k]
        uv = uv[keep]
        confs = confs[keep]
    cmax = float(confs.max()) if confs.size else 1.0
    cmin = float(confs.min()) if confs.size else 0.0
    span = max(cmax - cmin, 1e-9)
    for (u, v), c in zip(uv, confs):
        alpha = 0.4 + 0.6 * (float(c) - cmin) / span
        col = tuple(int(round(ch * alpha)) for ch in color_bgr)
        cv2.circle(frame, (int(round(u)), int(round(v))), radius, col, -1, cv2.LINE_AA)


def render(
    scenes_pkl: Path,
    cameras_json: Path,
    video_paths: list[Path],
    out_path: Path,
    seconds: float = 10.0,
    top_k_per_frame: int = 200,
    min_confidence: float = 0.0,
    dot_radius: int = 5,
) -> None:
    with open(scenes_pkl, "rb") as f:
        frozen: list[FrozenScene] = pickle.load(f)
    cams = _load_camera_models(cameras_json)
    if len(cams) != len(video_paths):
        raise ValueError(f"got {len(cams)} cameras but {len(video_paths)} videos")

    # Per-camera frame source.
    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    if not all(c.isOpened() for c in caps):
        raise FileNotFoundError("could not open one of the videos")

    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(round(seconds * fps))

    # Output: side-by-side cam0 | cam1 (assumes both same resolution).
    w0 = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w0 * len(caps)
    out_h = h0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    # One distinct colour per camera view (BGR).
    palette = [(0, 255, 255), (0, 200, 255), (255, 200, 0), (255, 100, 255)]

    written = 0
    try:
        for frame_idx in range(n_frames):
            t_local = frame_idx / fps  # cam0's local time = ~global time
            scene = _scene_for_time(frozen, t_local)

            panes: list[np.ndarray] = []
            n_drawn = 0
            for cam_i, (cam, cap) in enumerate(zip(cams, caps)):
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
                if scene is not None and scene.detected_objects_snapshot:
                    pts = np.array([d.position for d in scene.detected_objects_snapshot])
                    confs = np.array([d.confidence for d in scene.detected_objects_snapshot])
                    if min_confidence > 0:
                        keep = confs >= min_confidence
                        pts = pts[keep]; confs = confs[keep]
                    uv, _ = _project(pts, cam)
                    # Re-filter confs to in-bounds projections.
                    if len(uv) and len(uv) != len(confs):
                        # _project drops out-of-frame and behind-camera points; we
                        # need confidence aligned with surviving uv. Recompute.
                        pass
                    if len(uv):
                        # Recompute per-survivor confidences via the same mask.
                        rel = pts - cam.position[None, :]
                        p_cam = rel @ cam.rotation
                        z = p_cam[:, 2]
                        valid = z > 1e-3
                        if np.any(valid):
                            cx = cam.width / 2.0
                            cy = cam.height / 2.0
                            focal = cx / np.tan(np.radians(cam.fov_deg) / 2.0)
                            u = focal * (p_cam[valid, 0] / z[valid]) + cx
                            v = focal * (p_cam[valid, 1] / z[valid]) + cy
                            in_b = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
                            uv2 = np.stack([u[in_b], v[in_b]], axis=1)
                            confs2 = confs[valid][in_b]
                            _overlay_dots(frame, uv2, confs2, palette[cam_i % len(palette)],
                                          radius=dot_radius, top_k=top_k_per_frame)
                            n_drawn = max(n_drawn, len(uv2))

                cv2.putText(frame, f"cam{cam.camera_id}", (24, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, palette[cam_i % len(palette)], 3, cv2.LINE_AA)
                panes.append(frame)

            label = (
                f"t={t_local:.2f}s   bin@{scene.timestamp:.2f}s   "
                f"consensus dets in scene={len(scene.detected_objects_snapshot) if scene else 0}   "
                f"shown/cam(top-{top_k_per_frame} by confidence)={n_drawn}"
            )
            for pane in panes:
                cv2.putText(pane, label, (24, h0 - 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(pane, label, (24, h0 - 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(np.hstack(panes))
            written += 1
    finally:
        for c in caps:
            c.release()
        writer.release()

    print(f"wrote {written} frames → {out_path} ({written / fps:.2f}s @ {fps:.1f} fps)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", required=True, type=Path)
    p.add_argument("--cameras", required=True, type=Path)
    p.add_argument("--videos", required=True, nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--top-k", type=int, default=200,
                   help="Per-frame cap on the highest-confidence consensus dets to draw.")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Filter out consensus dets with confidence below this.")
    p.add_argument("--radius", type=int, default=5)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(
        scenes_pkl=args.scenes,
        cameras_json=args.cameras,
        video_paths=args.videos,
        out_path=args.out,
        seconds=args.seconds,
        top_k_per_frame=args.top_k,
        min_confidence=args.min_confidence,
        dot_radius=args.radius,
    )


if __name__ == "__main__":
    main()
