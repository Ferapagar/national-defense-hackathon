# Multi-Camera Aerial Object Detection

Detect aerial objects from multiple static cameras using frame-difference motion extraction, raycasting, and voxel-grid aggregation. Cameras are synchronized automatically; only pixel coordinates of detected objects are transmitted to the central unit, so any camera type (visible, IR, thermal) can be integrated.

## Architecture

The pipeline runs in four stages:

```
T0.1  Camera calibration    → cameras.json
T0.2  Motion extraction     → per-frame intensity masks (pixel-level)
T1    Ray generation        → RayBatch (origin + direction per moving pixel)
T2.B  Time calibration      → per-camera dt_i offset (cross-correlation)
T2.C  Voxel aggregation     → 3-D voxel grid accumulating ray intensities
T3    Scene freeze          → FrozenScene snapshots with DetectedObject list
```

### Key modules

| Path | Purpose |
|------|---------|
| `object-detection/calibration/estimate_relative_pose.py` | Two-view (or N-view) camera pose from SIFT features + essential/homography model |
| `object-detection/calibration/multi_view.py` | N-view consistency check (rotation cycle + translation loop closure) |
| `object-detection/calibration/make_inspection.py` | Visual sanity check: epipolar lines or warp overlay |
| `object-detection/pipeline/build_world_frame.py` | Converts `multi_view_report.json` → metric `cameras.json` |
| `object-detection/pipeline/motion_extraction.py` | Frame-difference motion masks + calibration history |
| `object-detection/pipeline/run_pipeline.py` | End-to-end driver: `cameras.json` + videos → `FrozenScene` list |
| `object-detection/pipeline/viewer.py` | Visualise saved `FrozenScene` pickles |
| `object-detection/scene.py` | Core data model: `Camera`, `RayBatch`, `GlobalScene`, `FrozenScene` |
| `object-detection/raycasting/raycaster.py` | Sampled ray-into-grid casting |
| `object-detection/raycasting/calibration.py` | Pairwise `dt_ij` cross-correlation + global `dt_i` least-squares |

## Quick start

### 1. Install dependencies

```bash
uv sync
```

### 2. Calibrate cameras (one-time)

Take one photo per camera of the same static scene, then:

```bash
# Two cameras — recover relative pose
python object-detection/calibration/estimate_relative_pose.py \
    --ref images/cam0.jpg --test images/cam1.jpg --out cal/

# Three or more cameras — N-view consistency check
python object-detection/calibration/multi_view.py \
    --images images/cam0.jpg images/cam1.jpg images/cam2.jpg --out cal/

# Visual sanity check
python object-detection/calibration/make_inspection.py \
    --ref images/cam0.jpg --test images/cam1.jpg --out cal/inspection.png

# Convert to metric cameras.json (provide tape-measured baseline in metres)
python object-detection/pipeline/build_world_frame.py \
    --report cal/multi_view_report.json \
    --baseline-m 1.5 \
    --out cameras.json
```

### 3. Run the detection pipeline

```bash
python object-detection/pipeline/run_pipeline.py \
    --cameras cameras.json \
    --videos data/cam0.mp4 data/cam1.mp4 \
    --extent "-10,10,-10,10,0,20" \
    --grid-size 64,64,64 \
    --time-bin-ms 33.3 \
    --threshold 1.0 \
    --out frozen_scenes.pkl
```

### 4. View results

```bash
python object-detection/pipeline/viewer.py frozen_scenes.pkl
```

## CLI reference

<!-- AUTO-GENERATED -->
### `run_pipeline.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--cameras` | *(required)* | `cameras.json` produced by `build_world_frame.py` |
| `--videos` | *(required)* | One video file per camera, in `cameras.json` order |
| `--extent` | *(required)* | Voxel-grid bounds: `xmin,xmax,ymin,ymax,zmin,zmax` (metres) |
| `--grid-size` | `64,64,64` | Voxel-grid resolution `Dx,Dy,Dz` |
| `--time-bin-ms` | `33.3` | Width of each time bin in ms; one `FrozenScene` per bin |
| `--threshold` | `1.0` | Voxel intensity threshold for object detection |
| `--calibration-frames` | `30` | Warm-up frames per camera for time-sync |
| `--max-frames` | `None` | Cap on frames decoded per camera (debug) |
| `--out` | `frozen_scenes.pkl` | Output pickle path |

### `build_world_frame.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--report` | *(required)* | `multi_view_report.json` from calibration |
| `--baseline-m` | *(required)* | Tape-measured distance between cam 0 and cam 1 (metres) |
| `--resolution` | inferred from K | Image resolution as `W,H` |
| `--out` | `cameras.json` | Output JSON path |

### `estimate_relative_pose.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--ref` | `../../images/ref-0.jpg` | Reference image |
| `--test` | `../../images/test-0.jpg` | Test image |
| `--images` | — | Three or more images for N-view mode |
| `--K` | EXIF / Pixel 8 default | Intrinsics as `fx,fy,cx,cy` |
| `--out` | `out/` | Output directory |
| `--rotation-tolerance-deg` | `2.0` | N-view rotation cycle tolerance |
| `--cycle-tolerance-pct` | `2.5` | N-view translation loop closure tolerance |
<!-- END AUTO-GENERATED -->

## Data structures

```
RayBatch      — vectorised (origins, directions, intensities) for one motion frame
Camera        — intrinsics + extrinsics + per-frame history for time calibration
GlobalScene   — voxel grid + cameras + aggregation/calibration logic
FrozenScene   — immutable snapshot: timestamp + voxel grid + DetectedObject list
DetectedObject — world-space position (m), confidence, global timestamp
```

## Running tests

```bash
pytest object-detection/tests/
# With coverage:
pytest object-detection/tests/ --cov=object-detection --cov-report=term-missing
```

## Dependencies

<!-- AUTO-GENERATED -->
| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.26 | Array math, ray vectorisation |
| `scipy` | ≥1.11 | Least-squares time calibration |
| `opencv-python` | ≥4.9 | Video decode, SIFT, RANSAC |
| `Pillow` | ≥10.0 | EXIF intrinsics extraction |
| `open3d` | ≥0.18 | 3-D visualisation |
| `pytest` | ≥8.0 | Test framework (dev) |
| `ruff` | ≥0.5 | Linter (dev) |
<!-- END AUTO-GENERATED -->

## Next steps

- [ ] Combine calibration + raycasting into a single end-to-end script with no manual steps
- [ ] Record real video data and test the full pipeline
- [ ] Add real-time streaming mode (static video must remain supported)
