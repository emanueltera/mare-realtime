"""Classical reference estimators on E2 tiles (no learning): masked phase correlation with
Guizar-Sicairos DFT up-sampling for the sub-pixel peak. A pure translation of the content
by (dx, dy) tile pixels between a and b (b(x) = a(x − d)) is returned as d = (dx, dy) in
(col = x, row = y) order, so the camera velocity in tile axes is −d·gsd/dt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage


def _taper(valid: np.ndarray) -> np.ndarray:
    """A soft version of a validity mask (eroded, then blurred), so a mask edge does not
    correlate as a feature of its own."""
    if valid.all():
        return np.ones(valid.shape)
    inner = ndimage.binary_erosion(valid, iterations=4)
    return ndimage.gaussian_filter(inner.astype(np.float64), 2.0) * inner


def _prep(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    a = np.where(w > 0, a, 0.0).astype(np.float64)
    m = (a * w).sum() / max(w.sum(), 1e-12)
    return np.where(w > 0, a - m, 0.0) * w


def _upsampled_dft(data, up, region, offsets):
    """DFT of `data` evaluated on a `region` (rows, cols) grid at 1/up pixel spacing around
    `offsets` (row, col), by two matrix products (Guizar-Sicairos et al. 2008)."""
    nr, nc = data.shape
    rr, rc = region
    kc = np.fft.ifftshift(np.arange(nc)) - nc // 2
    kr = np.fft.ifftshift(np.arange(nr)) - nr // 2
    col_k = np.exp((-2j * np.pi / (nc * up)) * (np.arange(rc) - offsets[1])[:, None] * kc[None, :])
    row_k = np.exp((-2j * np.pi / (nr * up)) * (np.arange(rr) - offsets[0])[:, None] * kr[None, :])
    return row_k @ data @ col_k.T


def _pc(a, b, upsample, joint=False):
    va, vb = np.isfinite(a), np.isfinite(b)
    if joint:  # aligned crops: one mask for both
        va = vb = va & vb
    n0, n1 = a.shape
    win = np.outer(np.hanning(n0), np.hanning(n1))
    A = np.fft.fft2(_prep(a, win * _taper(va)))
    B = np.fft.fft2(_prep(b, win * _taper(vb)))
    R = B * np.conj(A)
    # Whitened, but softly (|R| + λ, λ = the mean |R|) and with a Gaussian low-pass (σ = ¼
    # cycle/px): pure whitening hands the bins where interpolation error and grain outweigh
    # the texture as much weight as the rest, and at a fraction of a pixel that bias shows.
    f0 = np.fft.fftfreq(n0)[:, None]
    f1 = np.fft.fftfreq(n1)[None, :]
    mag = np.abs(R)
    R = R / (mag + mag.mean() + 1e-30) * np.exp(-0.5 * (f0 * f0 + f1 * f1) / 0.25**2)
    cc = np.fft.ifft2(R).real
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    shifts = np.array([p if p <= s // 2 else p - s for p, s in zip(peak, cc.shape)], dtype=np.float64)
    if upsample > 1:
        # refine around the integer peak at 1/upsample px
        span = np.ceil(upsample * 1.5)
        off = np.trunc(span / 2.0) - shifts * upsample
        up = _upsampled_dft(np.conj(R), upsample, (int(span), int(span)), off).conj().real
        mx = np.unravel_index(np.argmax(up), up.shape)
        shifts = shifts + (np.array(mx, dtype=np.float64) - np.trunc(span / 2.0)) / upsample
    return np.array([shifts[1], shifts[0]]), float(cc.max())


def phase_correlation(a: np.ndarray, b: np.ndarray, upsample: int = 50, min_overlap: float = 0.2):
    """Shift d = (dx, dy) in tile pixels with b(x) ≈ a(x − d); NaN in either tile is masked.
    Two stages: an integer estimate on the whole tiles, then the sub-pixel residual between
    crops that follow the content (a fixed window over moving content biases the peak toward
    zero by ~1 px at a 30 px shift). Returns (d, peak), peak being the normalised
    correlation height of the second stage (a quality hint); (NaN, NaN), 0 when the tiles
    share too little valid area."""
    nan = np.array([np.nan, np.nan])
    if min(np.isfinite(a).mean(), np.isfinite(b).mean()) < min_overlap:
        return nan, 0.0
    d0, _ = _pc(a, b, 1)
    dx, dy = int(d0[0]), int(d0[1])
    n0, n1 = a.shape
    m0, m1 = n0 - abs(dy), n1 - abs(dx)
    if min(m0, m1) < 32:
        return nan, 0.0
    ra, ca = max(0, -dy), max(0, -dx)
    ac = a[ra:ra + m0, ca:ca + m1]
    bc = b[ra + dy:ra + dy + m0, ca + dx:ca + dx + m1]
    if min(np.isfinite(ac).mean(), np.isfinite(bc).mean()) < min_overlap:
        return nan, 0.0
    r, peak = _pc(ac, bc, upsample, joint=True)
    return d0 + r, peak


def velocity_pairs(tiles: np.ndarray, gsd: float, dt: float, step: int = 1, upsample: int = 50):
    """Per consecutive pair (i, i + step): camera velocity in tile axes, −d·gsd/(step·dt).
    tiles: [F, n, n] float (NaN masked). Returns [F − step, 2] (NaN where it failed)."""
    out = np.full((tiles.shape[0] - step, 2), np.nan)
    for i in range(tiles.shape[0] - step):
        d, _ = phase_correlation(tiles[i].astype(np.float64), tiles[i + step].astype(np.float64), upsample)
        out[i] = -d * gsd / (step * dt)
    return out


def _phasecorr_run(root: str, run_id: str, step: int):
    from .data import TileSet  # noqa: PLC0415 - worker import

    ts = TileSet(root)
    a = np.asarray(ts.tiles(run_id), dtype=np.float64)
    fps = ts.fps(run_id)
    v = velocity_pairs(a, ts.gsd(run_id), 1.0 / fps, step)
    keys = [(ts.name, run_id, w, 2, step / fps) for w in range(len(v))]
    return keys, v


def phasecorr_predictions(root, out_csv, runs=None, sea_rate=None, step: int = 1, workers: int = 16):
    """Phase correlation between frames w and w + step for every w (K = 2 windows), written
    in the standard prediction format."""
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    from .data import PRED_COLUMNS, TileSet, prediction_frame

    ts = TileSet(root)
    ids = runs or ts.run_ids
    if sea_rate is not None:
        ids = [r for r in ids if float(ts.runs.at[r, "sea_rate"]) == sea_rate]
    keys, est = [], []
    with ProcessPoolExecutor(workers) as pool:
        for k, v in pool.map(_phasecorr_run, [str(root)] * len(ids), ids, [step] * len(ids)):
            keys += k
            est.append(v)
    df = prediction_frame(ts, keys, np.concatenate(est))[PRED_COLUMNS]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    # one BLAS thread per worker process: run as  OMP_NUM_THREADS=1 python -m e2.baselines ...
    import argparse

    ap = argparse.ArgumentParser(prog="python -m e2.baselines")
    ap.add_argument("--set-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sea-rate", type=float, default=None, help="only runs rendered at this render.seaRate")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    phasecorr_predictions(a.set_root, a.out, sea_rate=a.sea_rate, step=a.step, workers=a.workers)
