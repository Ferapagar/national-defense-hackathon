# How to use the relative-pose estimator

A practical guide to taking two photos of the same scene and recovering the
relative position and orientation of the second camera with respect to the
first.

> **Read this first.** A single pair of monocular images can only tell you
> the *direction* of camera motion, not metric distance. To get an actual
> distance in metres you need one extra piece of information — a known
> baseline, a known object size, or a calibration target. See section 4.

---

## 1. What you need

- **Two images** of the same static scene, from slightly different camera
  positions (and/or orientations). JPEG or PNG.
- **Python 3.10+** with:
  ```bash
  pip install opencv-python numpy pillow
  ```
- The two scripts in this directory:
  - `estimate_relative_pose.py` — runs the estimation.
  - `make_inspection.py` — renders a visual sanity-check image.

### Capture tips
- Same camera and same focal length / zoom for both shots. Mixed cameras work
  but you must supply intrinsics for *each* via `--K`.
- 10–60 % overlap of scene content. Too little overlap → not enough matches.
- For real distance, the second photo must come from a different *position*,
  not just a different angle. Pure pan/tilt collapses to homography-only
  output, where translation magnitude is meaningless.
- Avoid mostly-planar scenes (one flat wall) for the same reason.

---

## 2. Quick start

From this directory:

```bash
python estimate_relative_pose.py \
    --ref  /path/to/first.jpg \
    --test /path/to/second.jpg
```

Defaults to writing `out/pose.json`, `out/matches.png`, `out/inliers.png`.
Override the output directory with `--out path/to/dir`.

Visual sanity check:

```bash
python make_inspection.py \
    --ref  /path/to/first.jpg \
    --test /path/to/second.jpg \
    --out  out/inspection.png
```

---

## 3. Reading the output

`pose.json` looks like this:

```json
{
  "R": [[...], [...], [...]],
  "t": [tx, ty, tz],
  "euler_yaw_pitch_roll_deg": [yaw, pitch, roll],
  "inlier_count": 247,
  "total_matches": 458,
  "mean_reproj_error_px": 0.32,
  "median_depth": 11.77,
  "K": [[...], [...], [...]],
  "intrinsic_source": "exif:f35=24.0",
  "model": "essential",
  "plane_normal": null
}
```

| Field | Meaning |
|---|---|
| `R` | 3×3 rotation taking points from camera-1's frame into camera-2's frame. |
| `t` | 3-vector translation. **Direction of camera-2's optical centre as seen from camera-1, expressed in camera-1's frame.** |
| `euler_yaw_pitch_roll_deg` | Same `R` as ZYX intrinsic Euler angles, in degrees. |
| `inlier_count` / `total_matches` | RANSAC accepted / Lowe-ratio matches. Higher inlier ratio → more reliable pose. |
| `mean_reproj_error_px` | Average pixel error after triangulating inlier matches and re-projecting. < 1 px is healthy. |
| `median_depth` | Median triangulated point depth in *camera-1 units*. Multiply by your scale factor to get metres. |
| `K` | Camera intrinsic matrix used. |
| `intrinsic_source` | How `K` was obtained — `user`, `exif:f35=...`, or `pixel8-default`. |
| `model` | `"essential"` for general 3D motion, `"homography"` for planar / pure-rotation scenes. |
| `plane_normal` | Dominant plane's normal in camera-1's frame; only set for homography results. |

### Scales of `t` you should expect

| `model` | `‖t‖` | Interpretation |
|---|---|---|
| `essential` | always 1.0 | Direction only. Multiply by your metric scale to get camera-2's position in metres. |
| `homography` | small (e.g. 0.04) | Motion is dominated by rotation. Trust `R`, not the direction of `t`. |

---

## 4. Recovering metric distance

The estimator returns `t` as a *direction*. To turn that into a real-world
distance you need exactly one extra measurement.

### Option A — Tape-measure the baseline once
Place camera 1 and camera 2 by hand, measure the distance between the two
positions, then:

```python
import json, numpy as np

pose = json.load(open("out/pose.json"))
if pose["model"] != "essential":
    raise SystemExit("Pure rotation; no distance to recover.")

t = np.array(pose["t"])
baseline_m = 0.50            # metres, you measured this
camera2_position_m = baseline_m * t / np.linalg.norm(t)
print(camera2_position_m)    # camera 2 in metres, in camera 1's frame
```

For an essential-matrix result `‖t‖ = 1`, so the multiplier equals your
baseline directly. You can also convert any triangulated point's depth into
metres by multiplying by the same factor.

### Option B — Use a known object size
Identify an object visible in both images whose real size you know
(A4 paper = 0.297 m, license plate, your own height). Triangulate two corners
of it from the inlier matches and divide the real length by the triangulated
length:

```python
scale = real_object_length_m / triangulated_object_length
camera2_position_m = scale * t
```

`pose.json["median_depth"]` × `scale` then gives the typical scene depth in
metres.

### Option C — Add a third known-scale view
A third image with a known displacement from one of the others lets you fix
scale by sharing triangulated points. That is small-scale SfM and is out of
scope for this script.

---

## 5. Camera intrinsics

The script needs to know the camera's focal length and principal point.
Sources are tried in priority order:

1. `--K fx,fy,cx,cy` — most accurate. Run a one-time checkerboard calibration
   with `cv2.calibrateCamera` and pass the result.
2. EXIF `FocalLengthIn35mmFilm` from the reference image. Computes
   `fx = fy = (long_side_px / 36 mm) * f_35mm_eq`, principal point at the
   image centre.
3. Hard-coded Pixel 8 default (`6.90 mm` actual, `24 mm` 35mm-equivalent).

Whichever path was used is recorded as `intrinsic_source` in `pose.json`.
If you mix two cameras, EXIF auto-resolution is wrong (it only reads from
the reference image) — pass `--K` explicitly.

```bash
python estimate_relative_pose.py \
    --ref  first.jpg --test second.jpg \
    --K 2688,2688,1134,2016
```

---

## 6. Visual sanity check

Always run `make_inspection.py` before trusting the numbers. The output PNG
depends on which model the estimator selected:

- **Essential model** — left and right images side by side with inlier match
  lines and yellow epipolar lines on the right. The epipolar lines should
  pass *through* the matching feature on the right; if they're systematically
  off, your intrinsics are wrong.
- **Homography model** — a 2×2 grid (test, ref-warped, 50/50 overlay,
  per-pixel difference). Sharp regions in the overlay = good alignment.
  Yellow blobs in the difference image = misaligned regions, usually the
  closest foreground objects under residual parallax.

---

## 7. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `Only N matches survived ratio test` (N small) | Images don't overlap or lack texture | Re-shoot with more shared content; confirm both images are right-side-up. |
| `Only N pose inliers (need >= 30)` | Repeated structure (tiles, foliage) producing wrong matches | Try different scenes, or relax `RATIO_TEST` in the script (raise to 0.8). |
| Inspection shows mismatched lines crossing wildly | Very different viewpoints or one image is rotated 90° | Check EXIF orientation; some apps don't bake the rotation into pixels. |
| `model=homography` and `‖t‖ ≈ 0` | Pure-rotation capture — you panned but didn't move | Re-shoot with a real translation between cameras. |
| Recovered angles plausible but `t` direction wrong | Wrong intrinsics (fallback used for a different camera) | Pass `--K` explicitly. |

---

## 8. End-to-end example: distance between two phone shots

1. Take two photos from positions you can tape-measure, e.g. 0.5 m apart.
2. Save them as `first.jpg` and `second.jpg`.
3. Run:
   ```bash
   python estimate_relative_pose.py --ref first.jpg --test second.jpg
   python make_inspection.py --ref first.jpg --test second.jpg --out out/inspection.png
   ```
4. Open `out/inspection.png` to confirm the geometry looks right.
5. Compute camera-2's position in metres:
   ```python
   import json, numpy as np
   p = json.load(open("out/pose.json"))
   if p["model"] != "essential":
       raise SystemExit("Pure rotation; no distance to recover.")
   t = np.array(p["t"])
   baseline_m = 0.50
   pos = baseline_m * t / np.linalg.norm(t)
   print(f"camera 2 is at {pos} m relative to camera 1")
   ```

That's it.
