"""Synthetic 3-camera consistency tests for multi_view.py.

We bypass SIFT entirely and exercise the geometric layer directly: build
``PairwisePose`` instances from a known 3-camera ground-truth, then verify
that ``_check_triplet`` recovers zero rotation residual, the correct
relative translation scales, and zero loop-closure residual when fed
consistent input — and flags inconsistency when fed bad input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make the calibration package importable when running tests directly.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from estimate_relative_pose import PairwisePose  # noqa: E402
from multi_view import (  # noqa: E402
    _angle_between_rotations_deg,
    _check_triplet,
    _resolve_pair_scales,
    _shared_three_view_tracks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_K(f: float = 1000.0, cx: float = 500.0, cy: float = 500.0) -> np.ndarray:
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


def _make_kp(pts: np.ndarray) -> list:
    return [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in pts]


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3-D points (N,3 in world frame) through camera (R, t world->cam)."""
    Xc = (R @ X.T + t.reshape(3, 1)).T
    Xh = (K @ Xc.T).T
    return Xh[:, :2] / Xh[:, 2:3]


def _make_pairwise(
    i: int, j: int,
    R: np.ndarray, t: np.ndarray,
    idx_i: np.ndarray, idx_j: np.ndarray,
    model: str = "essential",
) -> PairwisePose:
    pairs = np.column_stack([idx_i, idx_j]).astype(np.int32)
    n = len(idx_i)
    return PairwisePose(
        i=i, j=j,
        R=R.astype(np.float64),
        t_unit=(t / np.linalg.norm(t)).astype(np.float64),
        model=model,
        n_inliers=n,
        n_total_matches=n,
        mean_reproj_error_px=0.0,
        median_depth=10.0,
        inlier_match_pairs=pairs,
        pts_i_inliers=np.zeros((n, 2), dtype=np.float32),
        pts_j_inliers=np.zeros((n, 2), dtype=np.float32),
        plane_normal=None,
    )


# ---------------------------------------------------------------------------
# Fixture: 3 cameras at known positions, all looking +Z, with a 3-D point cloud.
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_triplet():
    """Cameras: A=(0,0,0), B=(1,0,0), C=(0,1,0). All identity rotation.

    OpenCV convention (x_target = R · x_source + t):
        R_AB = I, t_AB = (-1, 0, 0)         baseline AB = 1
        R_AC = I, t_AC = ( 0,-1, 0)         baseline AC = 1
        R_BC = I, t_BC = ( 1,-1, 0)         baseline BC = sqrt(2)

    Anchor scale to ||t_AB|| := 1, so we expect:
        scale_ik (A->C) ≈ 1
        scale_jk (B->C) ≈ sqrt(2)
    """
    K = _make_K()
    rng = np.random.default_rng(0)

    n_pts = 50
    X = np.column_stack([
        rng.uniform(-3.0, 3.0, n_pts),
        rng.uniform(-3.0, 3.0, n_pts),
        rng.uniform(8.0, 12.0, n_pts),
    ])

    R_A, t_A = np.eye(3), np.zeros(3)
    R_B, t_B = np.eye(3), -np.array([1.0, 0.0, 0.0])
    R_C, t_C = np.eye(3), -np.array([0.0, 1.0, 0.0])

    px_A = _project(K, R_A, t_A, X)
    px_B = _project(K, R_B, t_B, X)
    px_C = _project(K, R_C, t_C, X)

    kp_A = _make_kp(px_A)
    kp_B = _make_kp(px_B)
    kp_C = _make_kp(px_C)

    idx = np.arange(n_pts)
    pij = _make_pairwise(0, 1, np.eye(3), np.array([-1.0, 0.0, 0.0]), idx, idx)
    pjk = _make_pairwise(1, 2, np.eye(3), np.array([1.0, -1.0, 0.0]), idx, idx)
    pik = _make_pairwise(0, 2, np.eye(3), np.array([0.0, -1.0, 0.0]), idx, idx)

    return K, pij, pjk, pik, kp_A, kp_B, kp_C, X


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rotation_cycle_zero_for_consistent_triplet(synthetic_triplet):
    _K, pij, pjk, pik, _kpA, _kpB, _kpC, _X = synthetic_triplet
    angle = _angle_between_rotations_deg(pjk.R @ pij.R, pik.R)
    assert angle < 1e-6


def test_shared_tracks_recovered(synthetic_triplet):
    _K, pij, pjk, pik, _kpA, _kpB, _kpC, X = synthetic_triplet
    ti, tj, tk = _shared_three_view_tracks(pij, pjk, pik)
    assert len(ti) == X.shape[0]
    assert (ti == np.arange(X.shape[0])).all()
    assert (tj == np.arange(X.shape[0])).all()
    assert (tk == np.arange(X.shape[0])).all()


def test_scale_resolution_recovers_baseline_ratios(synthetic_triplet):
    K, pij, pjk, pik, kpA, kpB, kpC, _X = synthetic_triplet
    alpha_ik, alpha_jk, n_shared, skip = _resolve_pair_scales(
        K, pij, pjk, pik, kpA, kpB, kpC,
    )
    assert skip is None
    assert n_shared == 50
    assert alpha_ik == pytest.approx(1.0, abs=1e-3)
    assert alpha_jk == pytest.approx(np.sqrt(2.0), abs=1e-3)


def test_loop_closure_zero_for_consistent_triplet(synthetic_triplet):
    K, pij, pjk, pik, kpA, kpB, kpC, _X = synthetic_triplet
    tc = _check_triplet(K, pij, pjk, pik, kpA, kpB, kpC)
    assert not tc.translation_check_skipped
    assert tc.rotation_residual_deg < 1e-6
    assert tc.loop_residual_norm < 1e-3
    assert tc.loop_residual_pct < 0.1
    assert tc.scale_ik == pytest.approx(1.0, abs=1e-3)
    assert tc.scale_jk == pytest.approx(np.sqrt(2.0), abs=1e-3)


def test_rotation_cycle_flags_wrong_rotation(synthetic_triplet):
    """Inject a 10° yaw error into the (j, k) pair; the rotation cycle
    check must fire even though translations and tracks are otherwise fine.
    """
    K, pij, pjk, pik, kpA, kpB, kpC, _X = synthetic_triplet
    bad_R = cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(10.0)]))[0]
    bad_pjk = _make_pairwise(
        pjk.i, pjk.j,
        bad_R, np.array([1.0, -1.0, 0.0]),
        pjk.inlier_match_pairs[:, 0], pjk.inlier_match_pairs[:, 1],
    )
    tc = _check_triplet(K, pij, bad_pjk, pik, kpA, kpB, kpC)
    assert 9.0 < tc.rotation_residual_deg < 11.0


def test_homography_pair_skips_translation_check(synthetic_triplet):
    """Any pair using the homography model invalidates the translation
    loop check (magnitudes are unreliable). Rotation check still runs.
    """
    K, pij, pjk, pik, kpA, kpB, kpC, _X = synthetic_triplet
    pjk_h = PairwisePose(
        i=pjk.i, j=pjk.j, R=pjk.R, t_unit=pjk.t_unit,
        model="homography",
        n_inliers=pjk.n_inliers, n_total_matches=pjk.n_total_matches,
        mean_reproj_error_px=pjk.mean_reproj_error_px,
        median_depth=pjk.median_depth,
        inlier_match_pairs=pjk.inlier_match_pairs,
        pts_i_inliers=pjk.pts_i_inliers, pts_j_inliers=pjk.pts_j_inliers,
        plane_normal=None,
    )
    tc = _check_triplet(K, pij, pjk_h, pik, kpA, kpB, kpC)
    assert tc.translation_check_skipped
    assert tc.skip_reason is not None and "homography" in tc.skip_reason
    assert tc.rotation_residual_deg < 1e-6


def test_loop_closure_flags_inconsistent_translation(synthetic_triplet):
    """If (i,k) translation direction can't be reconciled by any scale with
    the (i,j)->(j,k) chain, the loop residual must exceed the tolerance —
    even though the rotation cycle still passes.
    """
    K, pij, pjk, pik_good, kpA, kpB, kpC, _X = synthetic_triplet
    # Replace pik's direction with something the chain cannot match.
    bad_pik = _make_pairwise(
        pik_good.i, pik_good.j,
        pik_good.R, np.array([1.0, 0.0, 0.0]),
        pik_good.inlier_match_pairs[:, 0], pik_good.inlier_match_pairs[:, 1],
    )
    tc = _check_triplet(K, pij, pjk, bad_pik, kpA, kpB, kpC)
    assert tc.rotation_residual_deg < 1e-6
    assert tc.loop_residual_pct > 5.0
