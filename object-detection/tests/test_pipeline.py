"""Smoke tests for the new pipeline/* modules.

These tests fabricate inputs synthetically — they do not require recorded
video data or a calibration run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make `pipeline.*` and `scene` importable when tests are run from anywhere.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.intrinsics import K_to_fov_deg, fov_deg_to_focal_px  # noqa: E402
from pipeline.build_world_frame import build_world_cameras  # noqa: E402
from pipeline.motion_extraction import extract_motion, collect_calibration_history  # noqa: E402
from pipeline.run_pipeline import run_pipeline  # noqa: E402


# ---------------- intrinsics ----------------

def test_K_to_fov_deg_roundtrip():
    width = 1920
    fov_in = 60.0
    fx = fov_deg_to_focal_px(fov_in, width)
    K = np.array([[fx, 0, width / 2.0], [0, fx, 540], [0, 0, 1]])
    assert K_to_fov_deg(K, width) == pytest.approx(fov_in, abs=1e-6)


# ---------------- build_world_frame ----------------

def _synthetic_multi_view_report(tmp: Path) -> Path:
    """Three cameras: cam 0 at origin, cam 1 at +X (baseline=1 unit before scaling),
    cam 2 at +X with twice the displacement (scale_0_2 == 2.0)."""
    K = [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]

    # OpenCV convention: X_j = R @ X_i + t. With cam 1 to the +X of cam 0 and
    # cam 1 looking in the same direction as cam 0, R = I and t points from
    # cam1's origin BACK toward cam0 (i.e. -X in cam1's frame). So t_unit = (-1, 0, 0).
    pair_01 = {"i": 0, "j": 1, "R": np.eye(3).tolist(), "t_unit": [-1.0, 0.0, 0.0],
               "model": "essential", "n_inliers": 200, "n_total_matches": 250,
               "mean_reproj_error_px": 0.3, "median_depth": 10.0, "plane_normal": None}
    pair_02 = {"i": 0, "j": 2, "R": np.eye(3).tolist(), "t_unit": [-1.0, 0.0, 0.0],
               "model": "essential", "n_inliers": 200, "n_total_matches": 250,
               "mean_reproj_error_px": 0.3, "median_depth": 10.0, "plane_normal": None}
    pair_12 = {"i": 1, "j": 2, "R": np.eye(3).tolist(), "t_unit": [-1.0, 0.0, 0.0],
               "model": "essential", "n_inliers": 200, "n_total_matches": 250,
               "mean_reproj_error_px": 0.3, "median_depth": 10.0, "plane_normal": None}
    triplet_012 = {"i": 0, "j": 1, "k": 2,
                   "rotation_residual_deg": 0.0, "n_shared_tracks": 100,
                   "scale_ik": 2.0, "scale_jk": 1.0,
                   "loop_residual_norm": 0.0, "loop_residual_pct": 0.0,
                   "translation_check_skipped": False, "skip_reason": None}

    payload = {
        "images": ["cam0.jpg", "cam1.jpg", "cam2.jpg"],
        "K": K, "intrinsic_source": "synthetic",
        "pairs": [pair_01, pair_02, pair_12],
        "pair_failures": [],
        "triplets": [triplet_012],
    }
    p = tmp / "multi_view_report.json"
    p.write_text(json.dumps(payload))
    return p


def test_build_world_frame_anchors_cam0_at_origin(tmp_path: Path):
    report = _synthetic_multi_view_report(tmp_path)
    cams = build_world_cameras(report, baseline_m=2.0, resolution_wh=(1920, 1080))
    assert len(cams) == 3
    assert np.allclose(cams[0].position, [0, 0, 0])
    assert np.allclose(cams[0].rotation, np.eye(3))


def test_build_world_frame_uses_metric_baseline(tmp_path: Path):
    report = _synthetic_multi_view_report(tmp_path)
    cams = build_world_cameras(report, baseline_m=2.0, resolution_wh=(1920, 1080))
    # cam 1 sits at -R^T @ t * baseline. With R=I and t=(-1,0,0), -R^T @ t = (1,0,0),
    # so position = (baseline, 0, 0) = (2, 0, 0).
    assert np.allclose(cams[1].position, [2.0, 0.0, 0.0])


def test_build_world_frame_chains_through_triplet(tmp_path: Path):
    report = _synthetic_multi_view_report(tmp_path)
    cams = build_world_cameras(report, baseline_m=2.0, resolution_wh=(1920, 1080))
    # cam 2 has scale_ik = 2.0 (||t_02|| = 2 * ||t_01||), so position = (4, 0, 0).
    assert np.allclose(cams[2].position, [4.0, 0.0, 0.0])


def test_build_world_frame_rejects_homography_anchor(tmp_path: Path):
    report = _synthetic_multi_view_report(tmp_path)
    payload = json.loads(report.read_text())
    payload["pairs"][0]["model"] = "homography"
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="homography"):
        build_world_cameras(report, baseline_m=2.0, resolution_wh=(1920, 1080))


# ---------------- motion_extraction ----------------

def _write_synthetic_video(path: Path, n_frames: int, w: int = 64, h: int = 48,
                           fps: float = 30.0, motion_radius: int = 3) -> None:
    """Black background, single white square moving rightward each frame."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cx = (i * 2) % (w - motion_radius * 2 - 1) + motion_radius
        cy = h // 2
        frame[cy - motion_radius:cy + motion_radius, cx - motion_radius:cx + motion_radius] = 255
        writer.write(frame)
    writer.release()


def test_motion_extraction_picks_up_moving_pixels(tmp_path: Path):
    video = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video, n_frames=10)
    frames = list(extract_motion(video, camera_id=0, threshold=20))
    assert len(frames) >= 5
    # Expect every frame to register some motion (the square keeps moving).
    assert all(f.motion_total > 0 for f in frames)


def test_collect_calibration_history_returns_n_frames(tmp_path: Path):
    video = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video, n_frames=15)
    frames, history = collect_calibration_history(video, camera_id=0, n_frames=8)
    assert len(frames) == 8
    assert history.shape == (8,)


# ---------------- end-to-end smoke ----------------

def test_run_pipeline_end_to_end(tmp_path: Path):
    """Build synthetic 2-cam calibration + 2 fake videos; run the pipeline; assert non-empty."""
    report = _synthetic_multi_view_report(tmp_path)
    cams = build_world_cameras(report, baseline_m=1.0, resolution_wh=(64, 48))
    cams_payload = {
        "world_unit": "metres",
        "anchor": {"baseline_m": 1.0, "cam_a": 0, "cam_b": 1},
        "intrinsic_source": "synthetic",
        "cameras": [
            {"camera_id": c.camera_id, "image_path": c.image_path,
             "fov_deg": c.fov_deg, "resolution_wh": list(c.resolution_wh),
             "position": c.position.tolist(), "rotation": c.rotation.tolist()}
            for c in cams[:2]  # 2-cam smoke
        ],
    }
    cameras_json = tmp_path / "cameras.json"
    cameras_json.write_text(json.dumps(cams_payload))

    vid0 = tmp_path / "cam0.mp4"
    vid1 = tmp_path / "cam1.mp4"
    _write_synthetic_video(vid0, n_frames=20)
    _write_synthetic_video(vid1, n_frames=20)

    frozen = run_pipeline(
        cameras_json=cameras_json,
        video_paths=[vid0, vid1],
        voxel_grid_extent=[(-5, 5), (-5, 5), (-5, 15)],
        voxel_grid_size=(16, 16, 16),
        time_bin_s=0.5,
        detection_threshold=0.5,
        calibration_n_frames=5,
        max_frames_per_camera=15,
    )
    assert len(frozen) >= 1
    # At least one bin should have produced detections (given the moving square).
    assert any(len(s.detected_objects_snapshot) > 0 for s in frozen)
