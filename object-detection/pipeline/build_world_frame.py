"""Build a world-frame `cameras.json` from N camera images/videos using
`calibration/calibration.py`.

Pipeline:
1. Load each input as a `calibration.Image` (still or first video frame).
2. Build a `ReferenceSystem` anchored on cams 0 and 1.
3. Register every additional camera via `ReferenceSystem.get_coords`
   (law-of-sines triangulation on pairwise translation directions).
4. Apply metric scale: `||position_1|| := baseline_m`, then serialise to
   the same `cameras.json` schema that `pipeline.run_pipeline` consumes.

Conventions match `scene.Camera`:
- `position` is the camera centre in world frame (= cam-0 frame), in metres.
- `rotation` maps camera-frame directions to world-frame directions
  (i.e. `dirs_world = dirs_cam @ rotation.T`).
- `fov_deg` is the horizontal full-FOV in degrees; supplied via the CLI
  because the new calibration does NOT estimate intrinsics.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow `python pipeline/build_world_frame.py …` from object-detection/.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.calibration import Image, ReferenceSystem  # noqa: E402

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class WorldCamera:
    camera_id: int
    image_path: str
    fov_deg: float
    resolution_wh: tuple[int, int]
    position: np.ndarray  # (3,) metres in world frame
    rotation: np.ndarray  # (3, 3) cam-frame → world-frame


def _load_image(path: str | Path) -> Image:
    p = Path(path)
    if p.suffix.lower() in _VIDEO_SUFFIXES:
        return Image.from_video(str(p))
    return Image.from_file(str(p))


def world_cameras_from_reference_system(
    rs: ReferenceSystem,
    image_paths: list[str | Path],
    baseline_m: float,
    resolution_wh: tuple[int, int],
    fov_deg: float,
) -> list[WorldCamera]:
    """Pure-function variant: convert an already-populated `ReferenceSystem`
    into `WorldCamera`s.

    `image_paths` must include the two anchor images (in the order passed to
    `ReferenceSystem.__init__`) followed by every other camera that has
    already been registered via `rs.get_coords`.
    """
    if len(image_paths) < 2:
        raise ValueError("need at least 2 cameras")

    cam1_title = Path(image_paths[1]).stem
    if cam1_title not in rs.camera_params:
        raise KeyError(
            f"cam 1 ({image_paths[1]}) is not in the reference system; "
            "did you build the ReferenceSystem with this image?"
        )
    t_01_native = np.asarray(rs.camera_params[cam1_title][0], dtype=float)
    native_baseline = float(np.linalg.norm(t_01_native))
    if native_baseline < 1e-12:
        raise ValueError(
            f"|t_01| in ReferenceSystem is ~0 ({native_baseline:.3e}); "
            "cam 0 and cam 1 appear coincident — recapture with more separation."
        )
    metric_scale = baseline_m / native_baseline

    cameras: list[WorldCamera] = []
    for cam_id, path in enumerate(image_paths):
        title = Path(path).stem
        if title not in rs.camera_params:
            raise KeyError(
                f"cam {cam_id} ({path}) is not in the reference system; "
                "register it via rs.get_coords(Image.from_file(path)) first."
            )
        position_native, rotation = rs.camera_params[title]
        cameras.append(
            WorldCamera(
                camera_id=cam_id,
                image_path=str(path),
                fov_deg=fov_deg,
                resolution_wh=tuple(resolution_wh),
                position=np.asarray(position_native, dtype=float) * metric_scale,
                rotation=np.asarray(rotation, dtype=float),
            )
        )
    return cameras


def build_world_cameras(
    image_paths: list[str | Path],
    baseline_m: float,
    resolution_wh: tuple[int, int],
    fov_deg: float,
) -> list[WorldCamera]:
    """End-to-end: load images, run calibration, return world cameras."""
    if len(image_paths) < 2:
        raise ValueError("need at least 2 cameras (the first two seed the reference system)")

    images = [_load_image(p) for p in image_paths]
    rs = ReferenceSystem(images[0], images[1])
    for img in images[2:]:
        rs.get_coords(img)

    return world_cameras_from_reference_system(
        rs, image_paths, baseline_m, resolution_wh, fov_deg
    )


def cameras_to_json(
    cameras: list[WorldCamera], baseline_m: float, intrinsic_source: str
) -> dict:
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
    parser.add_argument(
        "--images", nargs="+", required=True,
        help="Two or more image OR video paths. The first two are the metric anchor pair.",
    )
    parser.add_argument(
        "--baseline-m", type=float, required=True,
        help="Tape-measured distance (metres) between cam 0 and cam 1.",
    )
    parser.add_argument(
        "--resolution", type=str, required=True,
        help="Image resolution as 'W,H'.",
    )
    parser.add_argument(
        "--fov-deg", type=float, required=True,
        help="Horizontal full-FOV in degrees (calibration.py does not estimate K).",
    )
    parser.add_argument("--out", default="cameras.json", help="Output JSON path.")
    parser.add_argument(
        "--intrinsic-source", default="cli:--fov-deg",
        help="Free-form note recorded into cameras.json.",
    )
    args = parser.parse_args()

    w, h = (int(x) for x in args.resolution.split(","))
    cams = build_world_cameras(args.images, args.baseline_m, (w, h), args.fov_deg)
    payload = cameras_to_json(cams, args.baseline_m, args.intrinsic_source)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(cams)} cameras → {args.out}")
    for c in cams:
        print(f"  cam {c.camera_id}: pos={c.position.round(3).tolist()} m, fov={c.fov_deg:.1f}°")


if __name__ == "__main__":
    main()
