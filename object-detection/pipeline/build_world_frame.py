"""Convert a `multi_view_report.json` (from calibration/multi_view.py) into a
world-frame `cameras.json` consumable by `scene.Camera`.

Conventions (matching OpenCV / calibration/estimate_relative_pose.py):
- For pair (i, j), `R, t_unit` satisfy `X_j = R @ X_i + t` for a 3-D point X.
- Hence camera-j centre in cam-i frame = `-R^T @ t`.
- The Camera object in scene.py interprets `rotation` as cam-frame → world-frame
  (since `dirs_world = dirs_cam @ rotation.T`).

Anchoring (T0.1 step):
- Cam 0 sits at world origin with identity rotation.
- Pair (0, 1) is the metric anchor: `||t_01|| := baseline_m`.
- For every other camera k, pull `scale_0k` from triplet (0, 1, k); then
  `position_k = baseline_m * scale_0k * (-R_0k.T @ t_unit_0k)`,
  `rotation_k = R_0k.T`.

The script refuses to chain through any pair whose `model == "homography"`,
because the homography fallback yields unreliable translation magnitude.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Allow `python pipeline/build_world_frame.py …` from object-detection/.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.intrinsics import K_to_fov_deg
else:
    from .intrinsics import K_to_fov_deg


@dataclass(frozen=True)
class WorldCamera:
    camera_id: int
    image_path: str
    fov_deg: float
    resolution_wh: tuple[int, int]
    position: np.ndarray  # (3,)
    rotation: np.ndarray  # (3, 3) cam-frame → world-frame


def _index_pairs(report: dict) -> dict[tuple[int, int], dict]:
    return {(p["i"], p["j"]): p for p in report["pairs"]}


def _index_triplets(report: dict) -> dict[tuple[int, int, int], dict]:
    return {(t["i"], t["j"], t["k"]): t for t in report["triplets"]}


def _camera_from_pair_with_zero(
    cam_id: int,
    pair_0k: dict,
    metric_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (position_in_world, rotation_cam_to_world) for camera `cam_id`,
    given the pair record for (0, cam_id) and the metric scale of t_0k."""
    if pair_0k["model"] == "homography":
        raise ValueError(
            f"pair (0, {cam_id}) is homography; translation magnitude unreliable. "
            "Recapture so the essential model can be used, or anchor through a "
            "different camera."
        )
    R = np.asarray(pair_0k["R"], dtype=float)
    t_unit = np.asarray(pair_0k["t_unit"], dtype=float)
    t_unit = t_unit / max(np.linalg.norm(t_unit), 1e-12)
    position = metric_scale * (-R.T @ t_unit)
    rotation_cam_to_world = R.T
    return position, rotation_cam_to_world


def _resolve_metric_scale_for_k(
    cam_id: int,
    triplets: dict[tuple[int, int, int], dict],
    baseline_m: float,
) -> float:
    """Look up scale_0k from triplet (0, 1, k)."""
    if cam_id == 1:
        return baseline_m
    triplet = triplets.get((0, 1, cam_id))
    if triplet is None:
        raise KeyError(
            f"No triplet (0, 1, {cam_id}) in report; cannot resolve scale for cam {cam_id}. "
            "Make sure all cameras share enough features with cams 0 and 1."
        )
    if triplet.get("translation_check_skipped"):
        raise ValueError(
            f"triplet (0, 1, {cam_id}) skipped translation: {triplet.get('skip_reason')}"
        )
    scale_0k = triplet["scale_ik"]  # convention: anchored at ||t_01|| = 1
    if not math.isfinite(scale_0k):
        raise ValueError(
            f"triplet (0, 1, {cam_id}) reports non-finite scale_ik; cannot anchor cam {cam_id}."
        )
    return baseline_m * scale_0k


def build_world_cameras(
    report_path: str | Path,
    baseline_m: float,
    resolution_wh: Optional[tuple[int, int]] = None,
) -> list[WorldCamera]:
    """Top-level entry: load report, return one WorldCamera per image."""
    report = json.loads(Path(report_path).read_text())
    images: list[str] = report["images"]
    if len(images) < 2:
        raise ValueError("Need at least 2 cameras (and 3 to chain beyond pair (0,1)).")

    K = np.asarray(report["K"], dtype=float)
    if resolution_wh is None:
        # Recover from K's principal point (assumes image-centred).
        resolution_wh = (int(round(2 * K[0, 2])), int(round(2 * K[1, 2])))
    fov_deg = K_to_fov_deg(K, image_width_px=resolution_wh[0])

    pairs = _index_pairs(report)
    triplets = _index_triplets(report)

    if (0, 1) not in pairs:
        raise KeyError("Report has no pair (0, 1); needed as the metric anchor.")
    if pairs[(0, 1)]["model"] == "homography":
        raise ValueError(
            "Pair (0, 1) is homography; translation magnitude is unreliable. "
            "Recapture so the essential model is selected, or rerun with --anchor a b "
            "pointing at a non-homography pair (not yet supported)."
        )

    cameras: list[WorldCamera] = [
        WorldCamera(
            camera_id=0,
            image_path=images[0],
            fov_deg=fov_deg,
            resolution_wh=resolution_wh,
            position=np.zeros(3),
            rotation=np.eye(3),
        )
    ]

    for k in range(1, len(images)):
        if (0, k) not in pairs:
            raise KeyError(f"Report has no pair (0, {k}); cam {k} cannot be chained from cam 0.")
        scale = _resolve_metric_scale_for_k(k, triplets, baseline_m)
        pos, R_world = _camera_from_pair_with_zero(k, pairs[(0, k)], scale)
        cameras.append(
            WorldCamera(
                camera_id=k,
                image_path=images[k],
                fov_deg=fov_deg,
                resolution_wh=resolution_wh,
                position=pos,
                rotation=R_world,
            )
        )

    return cameras


def cameras_to_json(cameras: list[WorldCamera], baseline_m: float, intrinsic_source: str) -> dict:
    return {
        "world_unit": "metres",
        "anchor": {"baseline_m": baseline_m, "cam_a": 0, "cam_b": 1},
        "intrinsic_source": intrinsic_source,
        "cameras": [
            {
                "camera_id": c.camera_id,
                "image_path": c.image_path,
                "fov_deg": c.fov_deg,
                "resolution_wh": list(c.resolution_wh),
                "position": c.position.tolist(),
                "rotation": c.rotation.tolist(),
            }
            for c in cameras
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Path to multi_view_report.json")
    parser.add_argument("--baseline-m", type=float, required=True,
                        help="Tape-measured distance (metres) between cam 0 and cam 1.")
    parser.add_argument("--resolution", type=str, default=None,
                        help="Image resolution as 'W,H'. Defaults to inferring from K's principal point.")
    parser.add_argument("--out", default="cameras.json", help="Output JSON path.")
    args = parser.parse_args()

    res = None
    if args.resolution:
        w, h = args.resolution.split(",")
        res = (int(w), int(h))

    cams = build_world_cameras(args.report, args.baseline_m, res)
    report = json.loads(Path(args.report).read_text())
    payload = cameras_to_json(cams, args.baseline_m, report.get("intrinsic_source", "unknown"))
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(cams)} cameras → {args.out}")
    for c in cams:
        print(f"  cam {c.camera_id}: pos={c.position.round(3).tolist()} m, fov={c.fov_deg:.1f}°")


if __name__ == "__main__":
    main()
