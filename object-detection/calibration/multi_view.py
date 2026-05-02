"""N-view consistency calibration.

Run pairwise relative-pose estimation across N >= 2 images. For triplets
(i, j, k) we cross-check the geometry two ways:

1. Rotation cycle:  R_jk · R_ij  should equal  R_ik  (no scale needed).
2. Translation loop closure: after using shared 3-D points triangulated in
   each pair to link the three independently-scaled translations to a single
   reference scale, the chained translation R_jk · t_ij + s_jk · t_jk
   should equal s_ik · t_ik.

Important caveat: monocular two-view geometry recovers translation only up
to scale, so we cannot naively compare the *magnitudes* of t_ij, t_jk and
t_ik — each pair has its own arbitrary scale. We resolve the relative
scales from triangulation before checking the loop residual.

Usage:
    python multi_view.py --images A.jpg B.jpg C.jpg
    python multi_view.py --images A.jpg B.jpg C.jpg D.jpg --out out/cal
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from estimate_relative_pose import (
    PairwisePose,
    _parse_K_arg,
    pairwise_pose,
    resolve_intrinsics,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "out"

ROTATION_TOLERANCE_DEG = 2.0
CYCLE_TOLERANCE_PCT = 5.0
MIN_SHARED_TRACKS_FOR_SCALE = 8


# ---------------------------------------------------------------------------
# Triplet consistency container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TripletConsistency:
    i: int
    j: int
    k: int
    rotation_residual_deg: float
    n_shared_tracks: int
    scale_ik: float           # multiplier on t_ik_unit; NaN if unsolvable
    scale_jk: float           # multiplier on t_jk_unit; NaN if unsolvable
    loop_residual_norm: float  # in (i,j)-pair scale units
    loop_residual_pct: float   # as percent of mean ‖t‖
    translation_check_skipped: bool
    skip_reason: Optional[str]


# ---------------------------------------------------------------------------
# Rotation + scale + loop helpers
# ---------------------------------------------------------------------------

def _angle_between_rotations_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_err = R_a.T @ R_b
    rvec, _ = cv2.Rodrigues(R_err)
    return float(np.degrees(np.linalg.norm(rvec)))


def _triangulate_pair(
    K: np.ndarray, R: np.ndarray, t: np.ndarray,
    pts_a: np.ndarray, pts_b: np.ndarray,
) -> np.ndarray:
    """Triangulate corresponding pixel pairs into camera-a's frame at the
    pair's own scale (i.e. assuming ‖t‖ = pair's unit translation).

    Returns (N, 3) float64 array; rows where triangulation diverges are NaN.
    """
    P_a = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_b = K @ np.hstack([R, t.reshape(3, 1)])
    X_h = cv2.triangulatePoints(
        P_a, P_b,
        pts_a.T.astype(np.float64),
        pts_b.T.astype(np.float64),
    )
    w = X_h[3]
    safe = np.abs(w) > 1e-9
    out = np.full((X_h.shape[1], 3), np.nan, dtype=np.float64)
    if safe.any():
        out[safe] = (X_h[:3, safe] / w[safe]).T
    return out


def _build_track_map(pp: PairwisePose) -> dict[int, int]:
    """For one pair return {kp_idx_in_image_i: kp_idx_in_image_j} for inliers."""
    if pp.inlier_match_pairs.size == 0:
        return {}
    return {int(a): int(b) for a, b in pp.inlier_match_pairs}


def _shared_three_view_tracks(
    pij: PairwisePose, pjk: PairwisePose, pik: PairwisePose,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover (kp_i_idx, kp_j_idx, kp_k_idx) tuples for tracks visible in
    all three images.

    A track exists when (i_idx, j_idx) is an inlier in pair (i,j),
    (j_idx, k_idx) is an inlier in pair (j,k), AND (i_idx, k_idx) is also
    an inlier in pair (i,k) with the same k_idx — both routes must agree.
    """
    ij_map = _build_track_map(pij)
    jk_map = _build_track_map(pjk)
    ik_map = _build_track_map(pik)

    tracks = []
    for i_idx, j_idx in ij_map.items():
        if j_idx not in jk_map or i_idx not in ik_map:
            continue
        if jk_map[j_idx] == ik_map[i_idx]:
            tracks.append((i_idx, j_idx, ik_map[i_idx]))

    if not tracks:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty, empty
    arr = np.array(tracks, dtype=np.int32)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _kp_pts(kp_list, indices: np.ndarray) -> np.ndarray:
    return np.array([kp_list[int(idx)].pt for idx in indices], dtype=np.float32)


def _resolve_pair_scales(
    K: np.ndarray,
    pij: PairwisePose, pjk: PairwisePose, pik: PairwisePose,
    kp_i, kp_j, kp_k,
) -> tuple[float, float, int, Optional[str]]:
    """Find α_ik and α_jk (relative to ‖t_ij‖ := 1) using shared 3-view tracks.

    Triangulating in each pair with its own unit translation gives three
    point clouds that differ from each other by a single global scale per
    pair. The median ratio of point norms locks the scales together.
    """
    track_i, track_j, track_k = _shared_three_view_tracks(pij, pjk, pik)
    n_shared = int(track_i.shape[0])
    if n_shared < MIN_SHARED_TRACKS_FOR_SCALE:
        return float("nan"), float("nan"), n_shared, "insufficient shared 3-view tracks"

    pts_i = _kp_pts(kp_i, track_i)
    pts_j = _kp_pts(kp_j, track_j)
    pts_k = _kp_pts(kp_k, track_k)

    # Cam-i frame, pair-(i,j) scale (||t_ij|| = 1, our anchor).
    X_ij = _triangulate_pair(K, pij.R, pij.t_unit, pts_i, pts_j)
    # Cam-i frame, pair-(i,k) arbitrary scale.
    X_ik = _triangulate_pair(K, pik.R, pik.t_unit, pts_i, pts_k)
    # Cam-j frame, pair-(j,k) arbitrary scale.
    X_jk = _triangulate_pair(K, pjk.R, pjk.t_unit, pts_j, pts_k)

    norms_ij = np.linalg.norm(X_ij, axis=1)
    norms_ik = np.linalg.norm(X_ik, axis=1)
    valid_ik = np.isfinite(norms_ij) & np.isfinite(norms_ik) & (norms_ik > 1e-6)
    alpha_ik = (
        float(np.median(norms_ij[valid_ik] / norms_ik[valid_ik]))
        if valid_ik.sum() >= MIN_SHARED_TRACKS_FOR_SCALE
        else float("nan")
    )

    # Move (i,j)-cloud into cam-j frame to compare with X_jk.
    X_ij_in_j = (pij.R @ X_ij.T + pij.t_unit.reshape(3, 1)).T
    norms_ij_in_j = np.linalg.norm(X_ij_in_j, axis=1)
    norms_jk = np.linalg.norm(X_jk, axis=1)
    valid_jk = np.isfinite(norms_ij_in_j) & np.isfinite(norms_jk) & (norms_jk > 1e-6)
    alpha_jk = (
        float(np.median(norms_ij_in_j[valid_jk] / norms_jk[valid_jk]))
        if valid_jk.sum() >= MIN_SHARED_TRACKS_FOR_SCALE
        else float("nan")
    )

    return alpha_ik, alpha_jk, n_shared, None


def _check_triplet(
    K: np.ndarray,
    pij: PairwisePose, pjk: PairwisePose, pik: PairwisePose,
    kp_i, kp_j, kp_k,
) -> TripletConsistency:
    rot_residual = _angle_between_rotations_deg(pjk.R @ pij.R, pik.R)

    # Homography path gives unreliable translation magnitudes. Rotation
    # check is still meaningful, but skip the loop closure on translation.
    if pij.model != "essential" or pjk.model != "essential" or pik.model != "essential":
        return TripletConsistency(
            i=pij.i, j=pij.j, k=pjk.j,
            rotation_residual_deg=rot_residual,
            n_shared_tracks=0,
            scale_ik=float("nan"),
            scale_jk=float("nan"),
            loop_residual_norm=float("nan"),
            loop_residual_pct=float("nan"),
            translation_check_skipped=True,
            skip_reason="one or more pairs used the homography model",
        )

    alpha_ik, alpha_jk, n_shared, skip_reason = _resolve_pair_scales(
        K, pij, pjk, pik, kp_i, kp_j, kp_k,
    )

    if not (math.isfinite(alpha_ik) and math.isfinite(alpha_jk)):
        return TripletConsistency(
            i=pij.i, j=pij.j, k=pjk.j,
            rotation_residual_deg=rot_residual,
            n_shared_tracks=n_shared,
            scale_ik=alpha_ik,
            scale_jk=alpha_jk,
            loop_residual_norm=float("nan"),
            loop_residual_pct=float("nan"),
            translation_check_skipped=True,
            skip_reason=skip_reason or "scale resolution failed",
        )

    # All translations expressed in pair-(i,j) scale (‖t_ij‖ := 1).
    t_ij = pij.t_unit
    t_jk_scaled = alpha_jk * pjk.t_unit
    t_ik_scaled = alpha_ik * pik.t_unit

    # Composition: x_k = R_jk · (R_ij · x_i + t_ij) + t_jk
    #            = (R_jk · R_ij) · x_i + (R_jk · t_ij + t_jk).
    t_ik_predicted = pjk.R @ t_ij + t_jk_scaled
    residual = t_ik_predicted - t_ik_scaled
    res_norm = float(np.linalg.norm(residual))

    mean_t = (
        float(np.linalg.norm(t_ij))
        + float(np.linalg.norm(t_jk_scaled))
        + float(np.linalg.norm(t_ik_scaled))
    ) / 3.0
    res_pct = float(res_norm / mean_t * 100.0) if mean_t > 1e-9 else float("nan")

    return TripletConsistency(
        i=pij.i, j=pij.j, k=pjk.j,
        rotation_residual_deg=rot_residual,
        n_shared_tracks=n_shared,
        scale_ik=alpha_ik,
        scale_jk=alpha_jk,
        loop_residual_norm=res_norm,
        loop_residual_pct=res_pct,
        translation_check_skipped=False,
        skip_reason=None,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _verdict(tc: TripletConsistency, rot_tol: float, cyc_tol: float) -> str:
    if tc.rotation_residual_deg > rot_tol:
        return "RED — rotation cycle exceeds tolerance"
    if tc.translation_check_skipped:
        return f"YELLOW — translation check skipped ({tc.skip_reason})"
    if tc.loop_residual_pct > cyc_tol:
        return "RED — translation loop closure exceeds tolerance"
    if tc.loop_residual_pct > 0.5 * cyc_tol:
        return "YELLOW — translation loop closure within warning band"
    return "GREEN"


def _write_json_report(
    out_dir: Path,
    image_paths: list[Path],
    K: np.ndarray, intr_source: str,
    pairs: dict, triplets: list, pair_failures: list,
):
    payload = {
        "images": [str(p) for p in image_paths],
        "K": K.tolist(),
        "intrinsic_source": intr_source,
        "pairs": [
            {
                "i": pp.i, "j": pp.j,
                "R": pp.R.tolist(), "t_unit": pp.t_unit.tolist(),
                "model": pp.model,
                "n_inliers": pp.n_inliers,
                "n_total_matches": pp.n_total_matches,
                "mean_reproj_error_px": pp.mean_reproj_error_px,
                "median_depth": pp.median_depth,
                "plane_normal": (
                    pp.plane_normal.tolist() if pp.plane_normal is not None else None
                ),
            }
            for pp in sorted(pairs.values(), key=lambda x: (x.i, x.j))
        ],
        "pair_failures": [
            {"i": i, "j": j, "error": err} for (i, j, err) in pair_failures
        ],
        "triplets": [asdict(t) for t in triplets],
    }
    (out_dir / "multi_view_report.json").write_text(json.dumps(payload, indent=2))


def _write_md_report(
    out_dir: Path,
    image_paths: list[Path],
    pairs: dict, triplets: list, pair_failures: list,
    rot_tol: float, cyc_tol: float,
):
    lines = ["# N-view Calibration Report", ""]
    lines.append(f"Images ({len(image_paths)}):")
    for idx, p in enumerate(image_paths):
        lines.append(f"- index `{idx}`: `{p.name}`")
    lines.append("")

    lines.append("## Pairwise poses")
    lines.append("")
    lines.append("| pair | model | inliers/total | mean reproj (px) | median depth |")
    lines.append("|---|---|---|---|---|")
    for pp in sorted(pairs.values(), key=lambda x: (x.i, x.j)):
        lines.append(
            f"| ({pp.i},{pp.j}) | {pp.model} | "
            f"{pp.n_inliers}/{pp.n_total_matches} | "
            f"{pp.mean_reproj_error_px:.3f} | {pp.median_depth:.3f} |"
        )
    lines.append("")

    if pair_failures:
        lines.append("### Pair failures")
        lines.append("")
        for i, j, err in pair_failures:
            lines.append(f"- ({i},{j}): {err}")
        lines.append("")

    if triplets:
        lines.append("## Triplet consistency")
        lines.append("")
        lines.append(
            f"Tolerances: rotation cycle < **{rot_tol}°**, "
            f"translation loop residual < **{cyc_tol}%**."
        )
        lines.append("")
        lines.append(
            "| triplet | rot Δ° | shared tracks | scale i→k | scale j→k | "
            "loop residual | verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for tc in triplets:
            res_pct = (
                "n/a" if not math.isfinite(tc.loop_residual_pct)
                else f"{tc.loop_residual_pct:.2f}%"
            )
            s_ik = "n/a" if not math.isfinite(tc.scale_ik) else f"{tc.scale_ik:.3f}"
            s_jk = "n/a" if not math.isfinite(tc.scale_jk) else f"{tc.scale_jk:.3f}"
            lines.append(
                f"| ({tc.i},{tc.j},{tc.k}) | {tc.rotation_residual_deg:.3f} | "
                f"{tc.n_shared_tracks} | {s_ik} | {s_jk} | {res_pct} | "
                f"{_verdict(tc, rot_tol, cyc_tol)} |"
            )
        lines.append("")
        lines.append("### How to read it")
        lines.append("")
        lines.append(
            "- **rot Δ°** — angle between the directly-measured rotation `R_ik` "
            "and the chained rotation `R_jk · R_ij`. Should be near zero."
        )
        lines.append(
            "- **scale i→k**, **scale j→k** — multipliers applied to "
            "`t_ik_unit` and `t_jk_unit` so their magnitudes match the "
            "(i,j) pair's unit-translation reference. They depend on scene "
            "geometry, not just camera motion."
        )
        lines.append(
            "- **loop residual** — "
            "‖(R_jk · t_ij + s_jk · t_jk) − s_ik · t_ik‖ as a percentage of "
            "the average ‖t‖. Small values mean the three pairs see a "
            "geometrically consistent triangle of camera positions."
        )
        lines.append("")
        lines.append("### Important caveat on \"distance\" comparison")
        lines.append("")
        lines.append(
            "Two-view monocular geometry recovers translation **only up to "
            "scale**. You cannot directly compare the magnitudes of `t_ij`, "
            "`t_jk`, `t_ik` because each pair has its own arbitrary scale. "
            "The check this report performs is *cycle closure* — after using "
            "shared 3-D points to link the three pairs to a common scale, "
            "the translations must form a closed triangle. That is the "
            "geometric self-consistency check; it is *not* the same as "
            "\"x + y ≈ z in metres\"."
        )
    else:
        lines.append("## Triplet consistency")
        lines.append("")
        lines.append(
            "Need 3+ images and at least one full (i,j,k) triangle of "
            "successful pairs to run the consistency check."
        )

    (out_dir / "multi_view_report.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def estimate_n_view(
    image_paths: list[Path],
    out_dir: Path,
    user_K: Optional[np.ndarray] = None,
    rotation_tolerance_deg: float = ROTATION_TOLERANCE_DEG,
    cycle_tolerance_pct: float = CYCLE_TOLERANCE_PCT,
):
    if len(image_paths) < 2:
        raise ValueError("need at least 2 images")

    images, grays = [], []
    for p in image_paths:
        if not p.exists():
            raise FileNotFoundError(p)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to load {p}")
        images.append(img)
        grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    h0, w0 = images[0].shape[:2]
    intr = resolve_intrinsics(image_paths[0], w0, h0, user_K=user_K)
    K = intr.to_matrix()
    print(f"[info] intrinsics source: {intr.source}")
    print(
        f"[info] K = fx={intr.fx:.1f} fy={intr.fy:.1f} "
        f"cx={intr.cx:.1f} cy={intr.cy:.1f}"
    )

    cached_features: dict = {}
    pairs: dict[tuple[int, int], PairwisePose] = {}
    pair_failures: list[tuple[int, int, str]] = []
    n = len(image_paths)

    for i, j in combinations(range(n), 2):
        try:
            pp = pairwise_pose(
                i, j, grays[i], grays[j], K,
                cached_features=cached_features,
            )
            pairs[(i, j)] = pp
            print(
                f"[info] pair ({i},{j}): model={pp.model}  "
                f"inliers={pp.n_inliers}/{pp.n_total_matches}  "
                f"mean_reproj={pp.mean_reproj_error_px:.3f}px"
            )
        except Exception as e:  # noqa: BLE001 — surface every reason in the report
            pair_failures.append((i, j, str(e)))
            print(f"[warn] pair ({i},{j}) failed: {e}")

    triplets: list[TripletConsistency] = []
    if n >= 3:
        for i, j, k in combinations(range(n), 3):
            if (i, j) not in pairs or (j, k) not in pairs or (i, k) not in pairs:
                continue
            kp_i = cached_features[i][0]
            kp_j = cached_features[j][0]
            kp_k = cached_features[k][0]
            tc = _check_triplet(
                K, pairs[(i, j)], pairs[(j, k)], pairs[(i, k)],
                kp_i, kp_j, kp_k,
            )
            triplets.append(tc)
            verdict = _verdict(tc, rotation_tolerance_deg, cycle_tolerance_pct)
            res_str = (
                "n/a" if not math.isfinite(tc.loop_residual_pct)
                else f"{tc.loop_residual_pct:.2f}%"
            )
            print(
                f"[info] triplet ({i},{j},{k}): "
                f"rot Δ={tc.rotation_residual_deg:.3f}°  loop={res_str}  "
                f"-> {verdict}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json_report(out_dir, image_paths, K, intr.source, pairs, triplets, pair_failures)
    _write_md_report(
        out_dir, image_paths, pairs, triplets, pair_failures,
        rotation_tolerance_deg, cycle_tolerance_pct,
    )
    print(
        f"\n[info] N-view report written to {out_dir}/  "
        f"(multi_view_report.json, multi_view_report.md)"
    )
    return pairs, triplets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--images", type=Path, nargs="+", required=True,
        help="Two or more image paths.",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--K", type=_parse_K_arg, default=None,
        help="Override intrinsics: 'fx,fy,cx,cy' or 9 comma-separated values.",
    )
    ap.add_argument(
        "--rotation-tolerance-deg", type=float, default=ROTATION_TOLERANCE_DEG,
    )
    ap.add_argument(
        "--cycle-tolerance-pct", type=float, default=CYCLE_TOLERANCE_PCT,
    )
    args = ap.parse_args()

    estimate_n_view(
        args.images, args.out,
        user_K=args.K,
        rotation_tolerance_deg=args.rotation_tolerance_deg,
        cycle_tolerance_pct=args.cycle_tolerance_pct,
    )


if __name__ == "__main__":
    main()
