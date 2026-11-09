"""The same pipeline without learning: E1's whitened dispersion-shell objective (every k
weighted equally after whitening), searched on the spectral model's grids around the same
classical pattern velocity, argmax per stage. A reference that isolates what the learned
weights and the CNN add; it uses no truth (unlike E1's prior, which was the truth + noise).

    python -m spectral.classical --set val --K 32 --out /workspace/data/e2/preds/spectral-classical/val.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from e2.data import PRED_COLUMNS, TileSet, prediction_frame

from . import spectrum as spx
from .data import SpectralWindows
from .model import grid
from .train import SETS, STAGES, collate, to_dev


@torch.no_grad()
def classical(b, stages, prep):
    sp = spx.prepare(b["tiles"], b["gsd"], b["dt"], **prep)
    c = b["centre"].float()
    for half, step, kmax in stages:
        kidx = None if kmax is None else torch.nonzero(sp.f.norm(dim=-1) <= kmax)[:, 0]
        nk = sp.k.shape[1] if kidx is None else len(kidx)
        w = torch.full((c.shape[0], nk), 1.0 / nk, device=c.device)
        o = grid(half, step, c.device)
        C = spx.shell_cost(sp, w, c[:, None, :] + o[None], kidx)
        c = c + o[C.argmax(-1)]
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spectral.classical")
    ap.add_argument("--set", required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--stages", default="reg", choices=list(STAGES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--step", type=int, default=1, help="window start spacing (diagnostics; the protocol is 1)")
    ap.add_argument("--band", type=float, default=1.0)
    ap.add_argument("--kmax-frac", type=float, default=0.8)
    a = ap.parse_args(argv)
    prep = dict(bin=2, pad=max(1, 64 // a.K), band_bins=a.band, kmax_frac=a.kmax_frac)
    ts = TileSet(SETS / a.set)
    ds = SpectralWindows(ts, a.K, step=a.step, register=a.stages != "noreg")
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=a.workers, collate_fn=collate, pin_memory=True)
    V = np.zeros((len(ds), 2))
    t0 = time.time()
    for b in dl:
        b = to_dev(b, "cuda")
        V[b["i"].cpu().numpy()] = classical(b, STAGES[a.stages], prep).cpu().numpy()
    keys = list(zip(ds.index["set"], ds.index.run_id, ds.index.w_start.astype(int), ds.index.K.astype(int), ds.index["T"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prediction_frame(ts, keys, V)[PRED_COLUMNS].to_csv(a.out, index=False)
    print(f"{a.set}: {len(ds)} windows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
