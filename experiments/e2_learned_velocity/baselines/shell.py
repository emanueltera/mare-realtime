"""B2: E1's dispersion-shell fit on E2 tiles.

The fit itself is E1's, imported unchanged from experiments/e1_wave_bias/shell_fit.py
(prepare → 3-D windowed power spectrum, fit_velocity → grid search of the shell energy
over V, then a fine grid and a quadratic refinement). Only the input changes: instead of
E1's camera-fixed ENU tile (256 cells over 0.6 × the across-track footprint, about 0.7 ×
the height) the fit reads E2's tile (384 cells of the nadir image GSD, 0.62 × the height,
oriented at the run's random θ). Both tiles translate with the camera and index
[row = +y, col = +x], so E1's (E, N) become E2's tile axes (x, y) with no sign change; the
velocity comes out in tile axes.

Two models, as in E1:
- dispersive: ω' = ±√(g|k|) − k·V (the waves' own motion modelled: velocity over the water)
- static:     ω' = −k·V (a frozen pattern: a spectral pattern tracker)

The prior is E1's: the TRUE window velocity plus a random offset, uniform in ±5 m/s per
component (seeded by run, window start and T), searched over ±8 m/s around it. That
uses the label to place the search and is GENEROUS to the baseline: a fit with no prior
would have to resolve the shell's ambiguities alone.

Windows: K = 32 and 64 frames (E1's T = 3.2 s and 6.4 s at 10 fps; the harness's T is the
span, 3.1 s and 6.3 s), starting every 16 frames (E1's 1.6 s step). A run of 81 frames
gives 4 windows at K = 32 and 2 at K = 64; an e1-core run (121 frames) 6 and 4.

    OMP_NUM_THREADS=2 python -m baselines.shell --set-root /workspace/data/e2/sets/test-iid \\
        --out-dir /workspace/data/e2/baselines/shell/test-iid --jobs 8
writes pred-dispersive.csv, pred-static.csv (standard prediction CSVs, K = 32 and 64 rows)
and diag.csv (one row per window and model: prior, J values, at_edge, quad_ok, λ_c).
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "e1_wave_bias"))
from e2.data import PRED_COLUMNS, TileSet, prediction_frame  # noqa: E402

KS = (32, 64)
STEP = 16


def _shell_module(chunk: int):
    """E1's shell_fit, with the GPU objective evaluated `chunk` velocities at a time (E1's 2048
    needs several GB per process; the box's GPU is shared)."""
    import shell_fit as sf  # noqa: PLC0415

    if not getattr(sf, "_e2_patched", False):
        sf._objective_torch = functools.partial(sf._objective_torch, chunk=chunk)
        sf._e2_patched = True
    return sf


def fit_run(set_root: str, run_id: str, Ks=KS, step=STEP, models=("dispersive", "static"),
            prior_half=5.0, box=8.0, seed=1, chunk=512, kmax_frac=0.8):
    """Every window of one run → list of row dicts (key, estimate, diagnostics)."""
    sf = _shell_module(chunk)
    ts = TileSet(set_root)
    a = np.asarray(ts.tiles(run_id), dtype=np.float32)  # NaN = masked; prepare() zero-fills
    fps = ts.fps(run_id)
    t = ts._fr[run_id]["t"].to_numpy()
    dt = float(np.median(np.diff(t)))
    gsd = ts.gsd(run_id)
    lb = ts._lb[run_id]
    pxy = lb[["cam_x", "cam_y"]].to_numpy()  # labels: used for the prior only, as in E1
    rows = []
    for K in Ks:
        for s0 in range(0, len(t) - K + 1, step):
            s1 = s0 + K - 1
            Vt = (pxy[s1] - pxy[s0]) / (t[s1] - t[s0])
            prior = Vt + sf.prior_offset(run_id, float(t[s0]), K / fps, prior_half, seed)
            sp = sf.prepare(a[s0:s0 + K], dt, gsd, kmax_frac=kmax_frac)
            for model in models:
                V, J, info = sf.fit_velocity(sp, prior, box=box, model=model)
                _, _, lam_c = sf.shell_stats(sp, V, 1.0, model)
                rows.append({"set": ts.name, "run_id": run_id, "w_start": s0, "K": K, "T": float(t[s1] - t[s0]),
                             "model": model, "v_est_x": V[0], "v_est_y": V[1], "prior_x": prior[0], "prior_y": prior[1],
                             "J_est": J, "J_truth": float(sf.objective(sp, Vt[None], 1.0, model)[0]),
                             "J_second": info["J_second"], "J_median": info["J_median"], "quad_ok": info["quad_ok"],
                             "at_edge": info["at_edge"], "lambda_c": lam_c, "nk": len(sp.kmag)})
    return rows


def _worker(args):
    root, rid, kw = args
    t0 = time.time()
    try:
        rows = fit_run(root, rid, **kw)
    except Exception as e:  # keep the batch going; the run then shows as missing in the harness notes
        print(f"FAILED {rid}: {e!r}", flush=True)
        return []
    print(f"{rid}: {len(rows)} fits, {time.time() - t0:.1f} s", flush=True)
    return rows


def run_set(set_root, out_dir, jobs=8, limit=None, sea_rate=None, **kw):
    import multiprocessing as mp

    ts = TileSet(set_root)
    ids = ts.run_ids[:limit] if limit else ts.run_ids
    if sea_rate is not None:
        ids = [r for r in ids if float(ts.runs.at[r, "sea_rate"]) == sea_rate]
    os.environ.setdefault("SHELL_FFT_WORKERS", "4")
    rows = []
    with mp.get_context("spawn").Pool(jobs) as pool:
        for r in pool.imap_unordered(_worker, [(str(set_root), rid, kw) for rid in ids]):
            rows.extend(r)
    diag = pd.DataFrame(rows).sort_values(["model", "run_id", "K", "w_start"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    diag.to_csv(out / "diag.csv", index=False)
    for model, g in diag.groupby("model"):
        keys = list(zip(g["set"], g["run_id"], g["w_start"], g["K"], g["T"]))
        prediction_frame(ts, keys, g[["v_est_x", "v_est_y"]].to_numpy())[PRED_COLUMNS].to_csv(
            out / f"pred-{model}.csv", index=False)
    return diag


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m baselines.shell")
    ap.add_argument("--set-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=512, help="velocities per GPU objective batch")
    ap.add_argument("--kmax-frac", type=float, default=0.8,
                    help="shortest wave used, as a fraction of the tile's Nyquist (E1's default 0.8)")
    ap.add_argument("--sea-rate", type=float, default=None, help="only runs rendered at this render.seaRate")
    a = ap.parse_args()
    run_set(a.set_root, a.out_dir, jobs=a.jobs, limit=a.limit, sea_rate=a.sea_rate, chunk=a.chunk, kmax_frac=a.kmax_frac)
