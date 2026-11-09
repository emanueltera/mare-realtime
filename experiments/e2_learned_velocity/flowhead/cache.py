"""Cache the frozen backbone's pair features for a fixed set of windows, so the head can be
trained for many epochs (training through SEA-RAFT on the fly gets ~4–8 windows/s on the
shared GPU: too few for the head to learn anything in the time budget).

    python -m flowhead.cache --set train --runs 1000 --first 0 --out /workspace/data/e2/flowhead/cache/train.0
    python -m flowhead.cache --set train --runs 1000 --first 1000 --out /workspace/data/e2/flowhead/cache/train.1
    python -m flowhead.cache --set val --out /workspace/data/e2/flowhead/cache/val.0

For each run, windows start every `--start-step` frames (8 with K = 8:
back to back) on the tile's centred `--crop` square. With the backbone frozen, a pair's
features do not depend on anything that is trained, so they are computed once
(FlowHeadNet.pair_features, as predict.py does) for the gaps 1, 2, 4, 7 — the gap-1 model's
pairs are the first 7 of them. Written as flat arrays (np.save, memory-mappable):

    pf.npy      float16 [N, P, PAIR_CH, h, w]   pair features (u / U_SCALE etc.)
    u8.npy      float32 [N, P, 2, h, w]         raw velocity per cell (m/s)
    frames.npy  float16 [N, 2, crop, crop]      first and last frame, z-scored (the appearance stem's input)
    valid8.npy  float16 [N, 1, h, w]            fraction of each cell valid in every frame
    windows.csv run_id, w_start, K, T, gsd, dt, v_true_x, v_true_y (labels / bookkeeping)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e2.data import TileSet  # noqa: E402
from flowhead.model import FlowHeadNet, chain_init, phase_shift, zscore_frames  # noqa: E402
from flowhead.train import raise_open_files  # noqa: E402

SETS = Path("/workspace/data/e2/sets")


@torch.no_grad()
def run_features(model, tiles, t, gsd, K, crop, starts, dev, chunk=32):
    """Pair features of the windows starting at `starts` of one run."""
    a = np.asarray(tiles, dtype=np.float32)
    n = a.shape[-1]
    o = (n - crop) // 2
    a = a[:, o:o + crop, o:o + crop]
    Fn = a.shape[0]
    mask = torch.from_numpy(np.isfinite(a)).to(dev)
    x = torch.from_numpy(np.nan_to_num(a, nan=0.0)).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = zscore_frames(x, mask)
        used = sorted({f for w in starts for f in range(w, w + K)})
        fmap = torch.zeros((Fn, 256, crop // 8, crop // 8), device=dev, dtype=torch.bfloat16)
        for s in range(0, len(used), chunk):
            idx = torch.tensor(used[s:s + chunk], device=dev)
            fmap[idx] = model.encode_frames(z[idx]).to(fmap.dtype)
        s1 = torch.cat([phase_shift(z[s:min(s + chunk, Fn - 1)], z[s + 1:min(s + chunk, Fn - 1) + 1])
                        for s in range(0, Fn - 1, chunk)])
        pairs = sorted({(w + i, w + j) for w in starts for i, j in model.pairs})
        pid = {p: k for k, p in enumerate(pairs)}
        ia = torch.tensor([p[0] for p in pairs], device=dev)
        ib = torch.tensor([p[1] for p in pairs], device=dev)
        init = chain_init(s1, ia, ib)
        tt = torch.as_tensor(t, device=dev, dtype=torch.float32)
        dtp = (tt[ib] - tt[ia]) / (ib - ia).float()
        gs = torch.full((len(pairs),), float(gsd), device=dev)
        pf, u8 = [], []
        for s in range(0, len(pairs), chunk):
            sl = slice(s, s + chunk)
            f_, u_ = model.pair_features(z, mask, fmap, ia[sl], ib[sl], init[sl], gs[sl], dtp[sl])
            pf.append(f_.float()); u8.append(u_.float())
    pf, u8 = torch.cat(pf), torch.cat(u8)
    widx = torch.tensor([[pid[(w + i, w + j)] for i, j in model.pairs] for w in starts], device=dev)
    frames = torch.stack([torch.stack([z[w], z[w + K - 1]]) for w in starts]).float()
    valid8 = torch.stack([F.avg_pool2d(mask[w:w + K].all(0, keepdim=True).float()[None], 8)[0] for w in starts])
    return pf[widx], u8[widx], frames, valid8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--sets", default=str(SETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=0, help="n runs from --first (0 = all)")
    ap.add_argument("--first", type=int, default=0, help="first run (to split a set over processes)")
    ap.add_argument("--start-step", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--gaps", default="1,2,4,7")
    ap.add_argument("--crop", type=int, default=256)
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    raise_open_files()
    model = FlowHeadNet(K=args.K, gaps=[int(g) for g in args.gaps.split(",")], backbone="frozen").to(dev).eval()
    ts = TileSet(Path(args.sets) / args.set)
    runs = ts.run_ids[args.first:args.first + args.runs] if args.runs else ts.run_ids[args.first:]
    idx = ts.windows(args.K, 1, args.start_step, 0.0, runs)
    N, P, h = len(idx), model.P, args.crop // 8
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mm = lambda name, dt, shape: np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=dt, shape=shape)  # noqa: E731
    pf_a = mm("pf", np.float16, (N, P, model.PAIR_CH, h, h))
    u8_a = mm("u8", np.float32, (N, P, 2, h, h))
    fr_a = mm("frames", np.float16, (N, 2, args.crop, args.crop))
    v8_a = mm("valid8", np.float16, (N, 1, h, h))
    print(f"{args.set}: {len(runs)} runs, {N} windows, {P} pairs each → {out}", flush=True)
    t0, k = time.time(), 0
    for r, rid in enumerate(runs):
        w = idx[idx.run_id == rid]
        starts = w.w_start.astype(int).tolist()
        fr = ts._fr[rid]
        pf, u8, frames, valid8 = run_features(model, ts.tiles(rid), fr["t"].to_numpy(), float(fr["gsd"].iloc[0]),
                                              args.K, args.crop, starts, dev)
        sl = slice(k, k + len(starts))
        pf_a[sl] = pf.cpu().numpy().astype(np.float16)
        u8_a[sl] = u8.cpu().numpy()
        fr_a[sl] = frames.cpu().numpy().astype(np.float16)
        v8_a[sl] = valid8.cpu().numpy().astype(np.float16)
        k += len(starts)
        if r % 50 == 0:
            print(f"  {r + 1}/{len(runs)} runs, {time.time() - t0:.0f} s", flush=True)
    for a in (pf_a, u8_a, fr_a, v8_a):
        a.flush()
    idx[["run_id", "w_start", "K", "T", "gsd", "dt", "v_true_x", "v_true_y"]].to_csv(out / "windows.csv", index=False)
    print(f"done: {N} windows in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
