"""Predict every window of a set with a frozen spectral checkpoint (standard prediction CSV).

    cd /workspace/mare-realtime/experiments/e2_learned_velocity
    /workspace/venv/bin/python -m spectral.predict --ckpt /workspace/data/e2/spectral/k32/best.pt \
        --set val --out /workspace/data/e2/preds/spectral/val.csv

Every window (stride 1, step 1, min_valid 0) is predicted, without augmentation unless --tta
(the mean over the 8 dihedral elements, each mapped back to tile axes). Beside the CSV,
<out>.diag.csv keeps per-window diagnostics: the pattern velocity used as the centre, whether
it was missing (phase correlation failed on every pair: centre 0), and whether the truth lay
inside the first stage's grid (diagnostic only; the model never sees the truth).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e2.data import PRED_COLUMNS, TileSet, augment_vector, prediction_frame, unaugment_vector

from .data import SpectralWindows, augment_tiles
from .train import SETS, build, collate, to_dev


@torch.no_grad()
def predict(model, ds, dev, bs=16, workers=8, tta=False):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=collate, pin_memory=True)
    n = len(ds)
    V, S, inside = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros(n, dtype=bool)
    gs = range(8) if tta else [0]
    for b in dl:
        b = to_dev(b, dev)
        i = b["i"].cpu().numpy()
        vs, ss = [], []
        for g in gs:
            gt = torch.full((len(i),), g, device=dev)
            tiles = augment_tiles(b["tiles"], gt) if g else b["tiles"]
            cen = torch.as_tensor(augment_vector(b["centre"].cpu().numpy(), g), device=dev, dtype=torch.float32)
            out = model(tiles, b["gsd"], b["dt"], cen)
            vs.append(unaugment_vector(out["v"].cpu().numpy(), g))
            ss.append(np.abs(unaugment_vector(out["logsig"].exp().cpu().numpy(), g)))
            if g == 0:
                c, o, _ = out["stages"][0]
                inside[i] = ((b["target"] - c).abs() <= o.abs().max()).all(-1).cpu().numpy()
        V[i] = np.mean(vs, 0)
        S[i] = np.sqrt(np.mean(np.square(ss), 0))
    return V, S, inside


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spectral.predict")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args(argv)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    targs = argparse.Namespace(**ck["args"])
    dev = torch.device("cuda")
    model = build(targs, targs.K * targs.pad).to(dev)
    model.load_state_dict(ck["model"])
    ts = TileSet(SETS / a.set)
    ds = SpectralWindows(ts, targs.K, step=1, register=targs.stages != "noreg")
    t0 = time.time()
    V, S, inside = predict(model, ds, dev, a.bs, a.workers, a.tta)
    keys = list(zip(ds.index["set"], ds.index.run_id, ds.index.w_start.astype(int), ds.index.K.astype(int), ds.index["T"]))
    df = prediction_frame(ts, keys, V, S)[PRED_COLUMNS]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    diag = ds.index[["run_id", "w_start", "K", "vpat_x", "vpat_y", "pat_missing"]].copy()
    diag["inside_stage0"] = inside
    diag.to_csv(str(a.out).replace(".csv", ".diag.csv"), index=False)
    print(f"{a.set}: {len(df)} windows in {time.time() - t0:.0f}s (checkpoint epoch {ck['epoch']}, tta {a.tta}); "
          f"pattern missing {diag.pat_missing.mean():.4f}, truth inside stage-0 grid {inside.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
