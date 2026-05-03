# Runbook

Operational procedures for the multi-camera aerial detection pipeline.

## End-to-end deployment (static video)

### Step 1 — Capture calibration images

Take one photo per camera of the same static scene (10–60 % overlap). Cameras must have a real translational baseline (not just pan/tilt).

### Step 2 — Estimate camera poses

```bash
# Two cameras
python object-detection/calibration/estimate_relative_pose.py \
    --ref  images/cam0.jpg \
    --test images/cam1.jpg \
    --out  cal/

# Three or more cameras
python object-detection/calibration/multi_view.py \
    --images images/cam0.jpg images/cam1.jpg images/cam2.jpg \
    --out cal/
```

Outputs: `cal/multi_view_report.json`, `cal/multi_view_report.md`

Check the `.md` report for per-triplet **GREEN / YELLOW / RED** verdicts:

| Verdict | Condition |
|---------|-----------|
| GREEN | rotation cycle < 2°, loop residual < 2.5% |
| YELLOW | loop residual 2.5–5%, or any homography pair |
| RED | rotation cycle > 2°, or loop residual > 5% |

If any triplet is RED: reshoot that pair with a clearer translational baseline and non-planar scene.

### Step 3 — Visual sanity check

```bash
python object-detection/calibration/make_inspection.py \
    --ref  images/cam0.jpg \
    --test images/cam1.jpg \
    --out  cal/inspection.png
```

Epipolar lines must pass through the matched feature on the right image. Systematic misalignment → wrong intrinsics (pass `--K fx,fy,cx,cy`).

### Step 4 — Build metric world frame

Tape-measure the distance between cam 0 and cam 1 in metres.

```bash
python object-detection/pipeline/build_world_frame.py \
    --report    cal/multi_view_report.json \
    --baseline-m 1.5 \
    --out       cameras.json
```

Outputs `cameras.json` with world-frame positions and rotations for every camera.

### Step 5 — Run the detection pipeline

```bash
python object-detection/pipeline/run_pipeline.py \
    --cameras  cameras.json \
    --videos   data/cam0.mp4 data/cam1.mp4 \
    --extent   "-10,10,-10,10,0,30" \
    --grid-size 64,64,64 \
    --time-bin-ms 33.3 \
    --threshold 1.0 \
    --calibration-frames 30 \
    --out      frozen_scenes.pkl
```

The pipeline prints per-camera `dt_i` offsets (ms) and a total frozen-scene count on completion.

### Step 6 — View results

```bash
python object-detection/pipeline/viewer.py frozen_scenes.pkl
```

## Tuning parameters

| Parameter | Effect | When to adjust |
|-----------|--------|----------------|
| `--threshold` | Minimum voxel intensity for a detection | Raise to reduce false positives; lower to catch faint objects |
| `--grid-size` | Spatial resolution of the voxel grid | Increase for finer localisation; increases memory and compute |
| `--extent` | World-space bounds of the voxel grid (metres) | Must cover the airspace of interest |
| `--time-bin-ms` | Width of each time bin | Match to target object speed; narrower = finer temporal resolution |
| `--calibration-frames` | Warm-up frames for time-sync | Increase if cameras have irregular frame timing |
| `motion threshold` (in `motion_extraction.py`) | Per-pixel frame-difference threshold | Raise in windy/noisy environments; lower for slow-moving objects |

## Common issues

### No detections produced

- Confirm video files are in the same order as entries in `cameras.json`.
- Lower `--threshold` (try `0.5`).
- Check that the voxel `--extent` actually covers where the objects are.
- Print `dt_i` values — very large offsets (> 1 s) suggest calibration failed; increase `--calibration-frames`.

### Pipeline errors on `cameras.json`

- `cameras.json has N cams but M videos given` → pass exactly one video per camera entry.
- `pair (0, k) is homography` → reshoot pair (0, k) with clear translational motion; or pass `--resolution W,H` if resolution inference is wrong.

### Poor calibration (YELLOW/RED triplets)

- Reshoot from positions with a real baseline (not pure pan/tilt).
- Ensure 10–60 % scene overlap between every pair.
- Avoid mostly-planar scenes (single flat wall).
- Provide explicit intrinsics via `--K fx,fy,cx,cy` if mixing camera models.

### Build / import errors

```bash
uv sync          # reinstall dependencies
ruff check .     # check for syntax / import errors
pytest object-detection/tests/ -v   # confirm tests still pass
```

## Rollback

The pipeline writes a single output file (`frozen_scenes.pkl` by default). To roll back, re-run the pipeline with the previous `cameras.json` and videos, or simply keep the previous pickle alongside the new one:

```bash
python object-detection/pipeline/run_pipeline.py ... --out frozen_scenes_v2.pkl
```

## Health check

```bash
# Confirm the pipeline end-to-end with synthetic data (no cameras/videos needed):
pytest object-detection/tests/test_pipeline.py::test_run_pipeline_end_to_end -v
```

Expected output: `1 passed`.
