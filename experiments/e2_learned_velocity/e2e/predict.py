"""Predict every window (stride 1, step 1, min_valid 0) of a set with a frozen model-A
checkpoint, in the standard prediction format.

    python -m e2e.predict --ckpt /workspace/data/e2/e2e/main/best.pt --set val \\
        --out /workspace/data/e2/preds/e2e/val.csv

Run by run (E2E.forward_run: each frame is encoded once and every window of the run is
estimated from the shared per-frame spectra; tests/test_e2e.py checks this equals the
window-by-window forward). Besides the CSV, an .npz next to it keeps the phase-correlation
start and every iteration's estimate (m/s, tile axes) for analysis.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from e2.data import PRED_COLUMNS, TileSet, prediction_frame
from e2e.model import E2E

SETS = "/workspace/data/e2/sets"


def load(ckpt: str, device) -> tuple[E2E, dict]:
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = E2E(n_iter=c["args"]["n_iter"], **c["args"]["arch"]).to(device).eval()
    m.load_state_dict(c["model"])
    return m, c


@torch.no_grad()
def predict_set(model: E2E, ts: TileSet, K: int, device, n_iter=None):
    """Every K-frame window (stride 1, step 1) of every run → the window table (ts.windows)
    and v, sigma, v0 [N, 2], vs [N, n_iter, 2] (m/s, tile axes), in the table's order."""
    idx = ts.windows(K, 1, 1, 0.0).reset_index(drop=True)
    out = {"v": [], "sigma": [], "v0": [], "vs": []}
    for rid, g in idx.groupby("run_id", sort=False):
        tiles = torch.from_numpy(np.asarray(ts.tiles(rid), dtype=np.float16)).to(device)
        starts = torch.as_tensor(g["w_start"].to_numpy(), device=device)
        gsd, dt = float(g["gsd"].iloc[0]), float(g["dt"].iloc[0])
        assert np.allclose(g["dt"], dt), rid
        o = model.forward_run(tiles, gsd, dt, K, starts, n_iter)
        scale = gsd / dt
        out["v"].append(o["v"].cpu().numpy())
        out["sigma"].append(o["sigma_v"].cpu().numpy())
        out["v0"].append((-o["u0"] * scale).cpu().numpy())
        out["vs"].append(np.stack([(-u * scale).cpu().numpy() for u in o["us"]], 1))
    # groupby(sort=False) keeps each run's rows together in first-seen order: the table's order
    idx = idx.loc[np.concatenate([g.index.to_numpy() for _, g in idx.groupby("run_id", sort=False)])].reset_index(drop=True)
    return idx, {k: np.concatenate(v) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m e2e.predict")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--sets", default=SETS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--gpu-mem-gb", type=float, default=20.0)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cudnn.benchmark = True
    model, ck = load(a.ckpt, dev)
    ts = TileSet(f"{a.sets}/{a.set}")
    t0 = time.time()
    rows, p = predict_set(model, ts, ck["args"]["K"], dev, a.n_iter)
    keys = list(zip(rows["set"], rows["run_id"], rows["w_start"].astype(int), rows["K"].astype(int), rows["T"]))
    df = prediction_frame(ts, keys, p["v"], p["sigma"])[PRED_COLUMNS]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    np.savez_compressed(out.with_suffix(".npz"), run_id=rows["run_id"].to_numpy().astype(str), w_start=rows["w_start"].to_numpy(),
                        v_true=rows[["v_true_x", "v_true_y"]].to_numpy(), v=p["v"], sigma=p["sigma"], v0=p["v0"], vs=p["vs"])
    print(f"{a.set}: {len(df)} windows, {len(set(rows.run_id))} runs in {time.time() - t0:.0f} s "
          f"(checkpoint step {ck['step']}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB) -> {out}", flush=True)


if __name__ == "__main__":
    main()
