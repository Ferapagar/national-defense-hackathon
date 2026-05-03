"""Diagnose temporal alignment between camera videos.

Reads cameras.json + N videos, extracts the per-frame motion_total signal
over a long window (default 10 s), and reports:

  * integer-frame offset from cross-correlation (current pipeline behaviour)
  * sub-frame offset from parabolic interpolation around the peak
  * sigma in ms from the peak curvature (uncertainty)
  * prominence ratio (best peak / 2nd-best peak)
  * least-squares residual across all pairs (consistency, only useful for N>=3)

Optionally writes two PNGs:
  - motion histories overlaid before / after applying the inferred dt_i
  - the normalised cross-correlation curve, peak marked

Does NOT modify the pipeline. Pure diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.motion_extraction import extract_motion  # noqa: E402
from raycasting.calibration import compute_global_dt  # noqa: E402


@dataclass(frozen=True)
class CamSignal:
    camera_id: int
    fps: float
    history: np.ndarray  # motion_total per frame
    n_frames: int


@dataclass(frozen=True)
class PairwiseSync:
    i: int
    j: int
    int_shift_frames: int
    sub_shift_frames: float
    sigma_frames: float
    prominence: float
    fps_ref: float  # fps used to convert frames -> seconds for this pair


def _video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def collect_signal(video_path: Path, camera_id: int, n_frames: int,
                   motion_kwargs: dict) -> CamSignal:
    fps = _video_fps(video_path)
    history = []
    for mf in extract_motion(video_path, camera_id,
                             max_frames=n_frames, **motion_kwargs):
        history.append(mf.motion_total)
    return CamSignal(camera_id=camera_id, fps=fps,
                     history=np.asarray(history, dtype=float),
                     n_frames=len(history))


def normalised_xcorr(
    a: np.ndarray,
    b: np.ndarray,
    min_overlap_frac: float = 0.5,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lags, ncc) for valid lags only.

    Lags are in samples; positive lag means b is delayed relative to a.

    Lags whose overlap is less than `min_overlap_frac` of min(len(a), len(b))
    are dropped (they're statistically unreliable — a few samples can correlate
    spuriously). If `max_lag` is given, |lag| is capped to it.

    Signals are zero-mean / unit-std normalised, and divided by the per-lag
    overlap count so each entry is a proper Pearson correlation.
    """
    a = a - a.mean()
    b = b - b.mean()
    sa = a.std()
    sb = b.std()
    if sa < 1e-9 or sb < 1e-9:
        return np.array([0]), np.array([0.0])
    a = a / sa
    b = b / sb

    raw = np.correlate(a, b, mode="full")
    lags_full = np.arange(-(len(b) - 1), len(a))
    overlap = np.minimum.reduce([
        np.full_like(lags_full, len(a)),
        np.full_like(lags_full, len(b)),
        len(a) - np.abs(lags_full),
    ])
    overlap = np.clip(overlap, 1, None)
    ncc_full = raw / overlap

    min_overlap = int(min_overlap_frac * min(len(a), len(b)))
    keep = overlap >= min_overlap
    if max_lag is not None:
        keep &= np.abs(lags_full) <= max_lag

    return lags_full[keep], ncc_full[keep]


def parabolic_peak(corr: np.ndarray, k: int) -> tuple[float, float]:
    """Fit a parabola through corr[k-1], corr[k], corr[k+1].

    Returns (delta, curvature) where delta is the sub-sample offset of the
    peak from index k (in samples), and curvature is -y''(peak). A sharper
    peak -> larger curvature -> smaller sigma.
    """
    if k <= 0 or k >= len(corr) - 1:
        return 0.0, 0.0
    y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return 0.0, 0.0
    delta = 0.5 * (y0 - y2) / denom
    curvature = -denom  # >0 if it's a local max
    return float(delta), float(curvature)


def estimate_sync(a: CamSignal, b: CamSignal,
                  max_shift_s: float | None = None,
                  min_overlap_frac: float = 0.5) -> PairwiseSync:
    """Estimate how much b is shifted relative to a, in frames of the
    common (averaged) sample rate.

    With a positive shift, b's signal is delayed; equivalently, b's
    timestamps need a positive dt added to align with a.
    """
    fps_ref = 0.5 * (a.fps + b.fps)
    max_lag = None if max_shift_s is None else int(max_shift_s * fps_ref)
    lags, ncc = normalised_xcorr(a.history, b.history,
                                 min_overlap_frac=min_overlap_frac,
                                 max_lag=max_lag)
    k = int(np.argmax(ncc))
    int_shift = int(lags[k])

    delta, curvature = parabolic_peak(ncc, k)
    sub_shift = int_shift + delta

    # 1-sigma from peak curvature: assume noise std on ncc ~= 1/sqrt(N_overlap)
    n_overlap = max(1, min(a.n_frames, b.n_frames) - abs(int_shift))
    noise_std = 1.0 / np.sqrt(n_overlap)
    sigma_frames = float(np.sqrt(noise_std / curvature)) if curvature > 0 else float("inf")

    # Prominence: best peak vs 2nd-best in a region away from the main lobe.
    main_lobe_radius = max(3, int(0.05 * len(ncc)))
    masked = ncc.copy()
    lo = max(0, k - main_lobe_radius)
    hi = min(len(ncc), k + main_lobe_radius + 1)
    masked[lo:hi] = -np.inf
    second_best = float(masked.max()) if np.any(np.isfinite(masked)) else 0.0
    prominence = float(ncc[k] / max(second_best, 1e-9))

    return PairwiseSync(
        i=a.camera_id, j=b.camera_id,
        int_shift_frames=int_shift,
        sub_shift_frames=sub_shift,
        sigma_frames=sigma_frames,
        prominence=prominence,
        fps_ref=fps_ref,
    )


def plot_diagnostics(signals: list[CamSignal], pair: PairwiseSync,
                     out_dir: Path,
                     max_shift_s: float | None = None,
                     min_overlap_frac: float = 0.5) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    a = signals[0]
    b = signals[1]
    fps_ref = pair.fps_ref
    dt_s = pair.sub_shift_frames / fps_ref

    # 1. Motion histories before/after alignment.
    t_a = np.arange(a.n_frames) / a.fps
    t_b = np.arange(b.n_frames) / b.fps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(t_a, a.history, label=f"cam {a.camera_id}")
    ax1.plot(t_b, b.history, label=f"cam {b.camera_id}", alpha=0.8)
    ax1.set_title("Raw motion_total per frame (before alignment)")
    ax1.set_ylabel("motion_total")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(t_a, a.history, label=f"cam {a.camera_id}")
    ax2.plot(t_b + dt_s, b.history,
             label=f"cam {b.camera_id} shifted by {dt_s*1000:+.1f} ms",
             alpha=0.8)
    ax2.set_title("After applying inferred dt (overlay)")
    ax2.set_xlabel("seconds")
    ax2.set_ylabel("motion_total")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    p1 = out_dir / "sync_overlay.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # 2. Cross-correlation curve.
    max_lag = None if max_shift_s is None else int(max_shift_s * fps_ref)
    lags, ncc = normalised_xcorr(a.history, b.history,
                                 min_overlap_frac=min_overlap_frac,
                                 max_lag=max_lag)
    lags_ms = lags * 1000.0 / fps_ref

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(lags_ms, ncc, lw=1)
    ax.axvline(pair.int_shift_frames * 1000.0 / fps_ref,
               color="C1", ls="--", label=f"int peak: {pair.int_shift_frames} fr")
    ax.axvline(dt_s * 1000.0,
               color="C3", ls=":", label=f"sub-frame: {dt_s*1000:+.1f} ms")
    ax.set_title(
        f"Normalised cross-correlation (cam {a.camera_id} vs cam {b.camera_id}) — "
        f"σ ≈ {pair.sigma_frames * 1000.0 / fps_ref:.1f} ms, "
        f"prominence = {pair.prominence:.2f}"
    )
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("normalised correlation")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "sync_xcorr.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)

    print(f"  wrote {p1}")
    print(f"  wrote {p2}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cameras", required=True, type=Path)
    p.add_argument("--videos", required=True, nargs="+", type=Path)
    p.add_argument("--seconds", type=float, default=10.0,
                   help="Window length per camera (s) used for cross-correlation.")
    p.add_argument("--out-dir", type=Path, default=Path("data/sync_debug"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--motion-stride", type=int, default=1)
    p.add_argument("--motion-threshold", type=int, default=3)
    p.add_argument("--edge-threshold", type=float, default=0.5)
    p.add_argument("--detect-mode", default="combined")
    p.add_argument("--max-shift-s", type=float, default=5.0,
                   help="Cap |lag| to this many seconds; rejects edge artifacts. "
                        "Set to 0 to disable.")
    p.add_argument("--min-overlap-frac", type=float, default=0.5,
                   help="Reject lags whose overlap is < this fraction of the window.")
    args = p.parse_args()
    max_shift_s = None if args.max_shift_s <= 0 else args.max_shift_s

    motion_kwargs = dict(
        detect_mode=args.detect_mode,
        threshold=args.motion_threshold,
        edge_threshold=args.edge_threshold,
        stride=args.motion_stride,
    )

    payload = json.loads(args.cameras.read_text())
    records = payload["cameras"]
    if len(records) != len(args.videos):
        raise SystemExit(
            f"cameras.json has {len(records)} cams but {len(args.videos)} videos given.")

    signals: list[CamSignal] = []
    for rec, vid in zip(records, args.videos):
        cam_id = rec["camera_id"]
        fps = _video_fps(Path(vid))
        n_frames = int(round(args.seconds * fps))
        print(f"cam {cam_id}: extracting motion over {args.seconds:.1f} s "
              f"({n_frames} frames @ {fps:.4f} fps) from {vid}")
        sig = collect_signal(Path(vid), cam_id, n_frames, motion_kwargs)
        print(f"  got {sig.n_frames} frames; "
              f"motion mean={sig.history.mean():.1f}, std={sig.history.std():.1f}")
        signals.append(sig)

    if len(signals) < 2:
        raise SystemExit("Need at least 2 cameras to estimate sync.")

    print("\nPairwise sync estimates")
    print("-" * 78)
    print(f"{'i':>3} {'j':>3} {'int Δ fr':>10} {'sub Δ fr':>11} "
          f"{'Δ ms':>10} {'σ ms':>9} {'prom':>7}")

    pairs: list[PairwiseSync] = []
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            pair = estimate_sync(signals[i], signals[j],
                                 max_shift_s=max_shift_s,
                                 min_overlap_frac=args.min_overlap_frac)
            pairs.append(pair)
            dt_ms = pair.sub_shift_frames * 1000.0 / pair.fps_ref
            sigma_ms = pair.sigma_frames * 1000.0 / pair.fps_ref
            print(f"{pair.i:>3} {pair.j:>3} {pair.int_shift_frames:>10d} "
                  f"{pair.sub_shift_frames:>11.3f} {dt_ms:>10.2f} "
                  f"{sigma_ms:>9.2f} {pair.prominence:>7.2f}")

    # Global LS solve in seconds (so cross-fps cameras are handled correctly).
    n = len(signals)
    if n >= 2:
        dt_matrix_s = [(p.i, p.j, p.sub_shift_frames / p.fps_ref) for p in pairs]
        # compute_global_dt anchors index 0; pass indices in the same order as
        # signals[].
        idx = {s.camera_id: k for k, s in enumerate(signals)}
        dt_matrix_idx = [(idx[i], idx[j], v) for (i, j, v) in dt_matrix_s]
        dt_per_cam_s = compute_global_dt(dt_matrix_idx, n)

        # LS residual: per-equation deviation in ms. With n=2 this is 0.
        if n >= 3:
            residuals_ms = []
            for (i_idx, j_idx, dt_obs) in dt_matrix_idx:
                pred = dt_per_cam_s[i_idx] - dt_per_cam_s[j_idx]
                residuals_ms.append(abs(pred - dt_obs) * 1000.0)
            res = float(np.sqrt(np.mean(np.square(residuals_ms))))
            print(f"\nGlobal LS RMS residual (consistency): {res:.2f} ms")
        else:
            print("\nGlobal LS residual: not informative for n=2 (exact solve).")

        print("\nPer-camera dt_i (seconds, anchored at cam 0):")
        for k, s in enumerate(signals):
            print(f"  cam {s.camera_id}: dt = {dt_per_cam_s[k]*1000:+.2f} ms")

    if not args.no_plots and len(signals) == 2:
        print("\nWriting diagnostic plots…")
        plot_diagnostics(signals, pairs[0], args.out_dir,
                         max_shift_s=max_shift_s,
                         min_overlap_frac=args.min_overlap_frac)


if __name__ == "__main__":
    main()
