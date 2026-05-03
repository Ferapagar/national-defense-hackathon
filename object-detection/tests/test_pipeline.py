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
from pipeline.build_world_frame import (  # noqa: E402
    WorldCamera,
    world_cameras_from_reference_system,
)
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

class _StubReferenceSystem:
    """Minimal duck-type for `ReferenceSystem`; just exposes `camera_params`.

    Matches the dict layout that `calibration.calibration.ReferenceSystem`
    populates: `{title: (position_native, rotation_cam_to_world)}`.
    """

    def __init__(self, camera_params):
        self.camera_params = camera_params


def _three_cam_stub() -> _StubReferenceSystem:
    """cam 0 at origin, cam 1 one native unit along +X, cam 2 two units along +X."""
    return _StubReferenceSystem({
        "cam0": (np.zeros(3), np.eye(3)),
        "cam1": (np.array([1.0, 0.0, 0.0]), np.eye(3)),
        "cam2": (np.array([2.0, 0.0, 0.0]), np.eye(3)),
    })


def test_world_cameras_anchor_cam0_at_origin():
    rs = _three_cam_stub()
    cams = world_cameras_from_reference_system(
        rs, ["cam0.jpg", "cam1.jpg", "cam2.jpg"],
        baseline_m=2.0, resolution_wh=(1920, 1080), fov_deg=60.0,
    )
    assert len(cams) == 3
    assert np.allclose(cams[0].position, [0, 0, 0])
    assert np.allclose(cams[0].rotation, np.eye(3))


def test_world_cameras_apply_metric_baseline():
    rs = _three_cam_stub()
    cams = world_cameras_from_reference_system(
        rs, ["cam0.jpg", "cam1.jpg", "cam2.jpg"],
        baseline_m=2.0, resolution_wh=(1920, 1080), fov_deg=60.0,
    )
    # ||t_01_native|| = 1, baseline_m = 2.0, so scale = 2.0.
    assert np.allclose(cams[1].position, [2.0, 0.0, 0.0])


def test_world_cameras_scale_chained_cameras():
    rs = _three_cam_stub()
    cams = world_cameras_from_reference_system(
        rs, ["cam0.jpg", "cam1.jpg", "cam2.jpg"],
        baseline_m=2.0, resolution_wh=(1920, 1080), fov_deg=60.0,
    )
    # cam 2 is 2 native units → 2 * 2.0 = 4.0 metres along +X.
    assert np.allclose(cams[2].position, [4.0, 0.0, 0.0])


def test_world_cameras_reject_coincident_anchor_pair():
    rs = _StubReferenceSystem({
        "cam0": (np.zeros(3), np.eye(3)),
        "cam1": (np.zeros(3), np.eye(3)),
    })
    with pytest.raises(ValueError, match="coincident"):
        world_cameras_from_reference_system(
            rs, ["cam0.jpg", "cam1.jpg"],
            baseline_m=2.0, resolution_wh=(1920, 1080), fov_deg=60.0,
        )


def test_world_cameras_reject_unregistered_camera():
    rs = _three_cam_stub()
    with pytest.raises(KeyError, match="not in the reference system"):
        world_cameras_from_reference_system(
            rs, ["cam0.jpg", "cam1.jpg", "cam_missing.jpg"],
            baseline_m=2.0, resolution_wh=(1920, 1080), fov_deg=60.0,
        )


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


def _write_low_contrast_video(path: Path, n_frames: int, w: int = 64, h: int = 48,
                              fps: float = 30.0, motion_radius: int = 4,
                              bg: int = 120, fg: int = 132) -> None:
    """Mid-grey background with a slightly brighter square that moves rightward.

    The bg/fg gap (12 levels) is below the legacy threshold of 25 used by the
    intensity-only detector, so the edge channel is what surfaces the motion.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), bg, dtype=np.uint8)
        cx = (i * 2) % (w - motion_radius * 2 - 1) + motion_radius
        cy = h // 2
        frame[cy - motion_radius:cy + motion_radius,
              cx - motion_radius:cx + motion_radius] = fg
        writer.write(frame)
    writer.release()


def test_edge_channel_detects_low_contrast_motion(tmp_path: Path):
    """The edge channel must surface motion that the intensity threshold misses."""
    video = tmp_path / "low_contrast.mp4"
    _write_low_contrast_video(video, n_frames=10)

    # Intensity-only with the legacy threshold of 25: the 12-level fg/bg gap
    # never crosses the threshold, so no motion is reported.
    intensity_only = list(
        extract_motion(video, camera_id=0, threshold=25, detect_mode="intensity",
                       dither=False, morph_open_ksize=0)
    )
    assert all(f.motion_total == 0 for f in intensity_only), (
        "intensity-only at threshold=25 should miss the low-contrast square"
    )

    # Edge-only: should pick up the moving edges of the square.
    edge_only = list(
        extract_motion(video, camera_id=0, detect_mode="edge", dither=False)
    )
    assert any(f.motion_total > 0 for f in edge_only), (
        "edge channel should detect moving edges even when |ΔI| is small"
    )


def test_combined_mode_at_least_as_sensitive(tmp_path: Path):
    """Combined mode must dominate either single-channel mode in motion_total."""
    video = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video, n_frames=10)

    common = dict(camera_id=0, threshold=20, dither=False, morph_open_ksize=0)
    intensity = list(extract_motion(video, detect_mode="intensity", **common))
    edge = list(extract_motion(video, detect_mode="edge", **common))
    combined = list(extract_motion(video, detect_mode="combined", **common))

    n = min(len(intensity), len(edge), len(combined))
    assert n > 0
    for i, e, c in zip(intensity[:n], edge[:n], combined[:n]):
        # OR mask → combined cannot be below either single channel.
        assert c.motion_total >= i.motion_total - 1e-9
        assert c.motion_total >= e.motion_total - 1e-9


def test_invalid_detect_mode_raises(tmp_path: Path):
    video = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video, n_frames=3)
    with pytest.raises(ValueError, match="detect_mode"):
        list(extract_motion(video, camera_id=0, detect_mode="bogus"))  # type: ignore[arg-type]


def test_collect_calibration_history_returns_n_frames(tmp_path: Path):
    video = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video, n_frames=15)
    frames, history = collect_calibration_history(video, camera_id=0, n_frames=8)
    assert len(frames) == 8
    assert history.shape == (8,)


# ---------------- consensus detection ----------------

def test_consensus_suppresses_single_camera_artifacts():
    """A voxel hit only by cam 0's rays must be rejected when consensus>=2."""
    from scene import Camera, GlobalScene  # noqa: WPS433 (test-local)

    extent = [(-2, 2), (-2, 2), (0, 4)]
    sz = (16, 16, 16)
    scene = GlobalScene(voxel_grid_extent=extent, voxel_grid_size=sz)

    cam0 = Camera(camera_id=0, fov=60.0, resolution=(8, 8),
                   position=np.array([0.0, 0.0, 0.0]),
                   rotation=np.eye(3))
    cam1 = Camera(camera_id=1, fov=60.0, resolution=(8, 8),
                   position=np.array([1.0, 0.0, 0.0]),
                   rotation=np.eye(3))
    scene.add_camera(cam0)
    scene.add_camera(cam1)

    # cam 0 sees a bright pixel at the centre of its image (= ray straight forward)
    img0 = np.zeros((8, 8), dtype=np.uint8)
    img0[4, 4] = 200
    scene.aggregate_rays(cam0.generate_rays(img0, local_timestamp=0.0))

    # cam 1 sees nothing (blank frame).
    img1 = np.zeros((8, 8), dtype=np.uint8)
    scene.aggregate_rays(cam1.generate_rays(img1, local_timestamp=0.0))

    # consensus=2: nothing should pass (cam 1 contributed no rays).
    objs = scene.detect_objects(threshold=1.0, consensus=2)
    assert objs == []
    # consensus=1: cam 0's rays alone should still light up some voxels.
    objs1 = scene.detect_objects(threshold=1.0, consensus=1)
    assert len(objs1) > 0


def test_consensus_finds_intersection_when_both_cameras_agree():
    """Two cameras converging at a target produce a detection at the intersection."""
    from scene import Camera, GlobalScene  # noqa: WPS433

    extent = [(-2, 2), (-2, 2), (0, 4)]
    sz = (32, 32, 32)
    scene = GlobalScene(voxel_grid_extent=extent, voxel_grid_size=sz)

    # Two cameras 2 units apart on X axis, looking +Z.
    cam0 = Camera(0, fov=60.0, resolution=(16, 16),
                   position=np.array([-1.0, 0.0, 0.0]), rotation=np.eye(3))
    cam1 = Camera(1, fov=60.0, resolution=(16, 16),
                   position=np.array([1.0, 0.0, 0.0]), rotation=np.eye(3))
    scene.add_camera(cam0)
    scene.add_camera(cam1)

    # Aim rays at each other so they actually intersect.
    # cam 0 ray tilted toward +X (cam-frame x_cam>0 → world +X since rot=I).
    img0 = np.zeros((16, 16), dtype=np.uint8)
    img0[8, 12] = 200  # column 12, right of centre (cx=8) → +x_cam
    scene.aggregate_rays(cam0.generate_rays(img0, local_timestamp=0.0))

    # cam 1 ray tilted toward -X.
    img1 = np.zeros((16, 16), dtype=np.uint8)
    img1[8, 4] = 200   # column 4, left of centre → -x_cam
    scene.aggregate_rays(cam1.generate_rays(img1, local_timestamp=0.0))

    objs = scene.detect_objects(threshold=1.0, consensus=2)
    assert len(objs) > 0, "expected at least one consensus voxel where rays intersect"
    xs = np.array([o.position[0] for o in objs])
    assert (np.abs(xs) < 0.7).any(), f"no detections near X=0, got X range [{xs.min()}, {xs.max()}]"


# ---------------- end-to-end smoke ----------------

def test_run_pipeline_end_to_end(tmp_path: Path):
    """Hand-built 2-cam cameras.json + 2 fake videos; run the pipeline; assert non-empty."""
    # Synthetic frames don't yield SIFT matches, so skip calibration entirely
    # and synthesise a known camera rig directly.
    cams = [
        WorldCamera(camera_id=0, image_path="cam0.mp4", fov_deg=60.0,
                    resolution_wh=(64, 48), position=np.zeros(3), rotation=np.eye(3)),
        WorldCamera(camera_id=1, image_path="cam1.mp4", fov_deg=60.0,
                    resolution_wh=(64, 48), position=np.array([1.0, 0.0, 0.0]),
                    rotation=np.eye(3)),
    ]
    cams_payload = {
        "world_unit": "metres",
        "anchor": {"baseline_m": 1.0, "cam_a": 0, "cam_b": 1},
        "intrinsic_source": "synthetic",
        "cameras": [
            {"camera_id": c.camera_id, "image_path": c.image_path,
             "fov_deg": c.fov_deg, "resolution_wh": list(c.resolution_wh),
             "position": c.position.tolist(), "rotation": c.rotation.tolist()}
            for c in cams
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
