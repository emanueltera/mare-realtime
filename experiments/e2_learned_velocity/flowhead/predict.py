"""Predict every window of a set with a frozen flowhead checkpoint → the standard prediction CSV.

    cd /workspace/mare-realtime/experiments/e2_learned_velocity
    python -m flowhead.predict --ckpt /workspace/data/e2/flowhead/main/best.pt --set val \
        --out /workspace/data/e2/preds/flowhead/val.csv

Every window (step 1, min_valid 0) of the set. Per run, the frames are standardised and
encoded once and every pair (i, i + g) that some window uses goes through SEA-RAFT once;
each window then gathers its pairs' features and runs the head. This gives the same numbers
as FlowHeadNet.forward on the window (the pair features depend only on the two frames; each
pair's dt is (t_j − t_i) / (j − i) from frames.csv, equal to the window's dt at a fixed frame
rate) at a third of the cost. The tile is cropped to its centred `--crop` square, as in
validation during training.

--tta g1,g2,… also predicts on those dihedral elements of the tiles and averages the
un-augmented estimates (and σ). A second CSV `<out>.diag.csv` holds the head's
un-corrected flow average v_raw per window (diagnostic only).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e2.data import PRED_COLUMNS, TileSet, augment_array, prediction_frame, unaugment_vector  # noqa: E402
from flowhead.model import FlowHeadNet, chain_init, phase_shift, zscore_frames  # noqa: E402
from flowhead.train import raise_open_files  # noqa: E402

SETS = Path("/workspace/data/e2/sets")


def load_model(ckpt, dev):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = st["args"]
    model = FlowHeadNet(K=a["K"], gaps=[int(g) for g in a["gaps"].split(",")], iters=a["iters"],
                        backbone=a.get("backbone", "frozen"))
    model.load_state_dict(st["model"])
    return model.to(dev).eval(), st


@torch.no_grad()
def predict_run(model, tiles, t, gsd, K, crop, g, dev, chunk, amp):
    """All windows of one run: tiles [F, n, n] float16 with NaN = masked, t [F] (s).
    Returns v [W, 2], sigma [W, 2], v_raw [W, 2] in un-augmented tile axes."""
    a = np.asarray(tiles, dtype=np.float32)
    n = a.shape[-1]
    if crop and crop < n:
        o = (n - crop) // 2
        a = a[:, o:o + crop, o:o + crop]
    if g:
        a = augment_array(a, g)
    Fn = a.shape[0]
    mask = torch.from_numpy(np.isfinite(a)).to(dev)
    x = torch.from_numpy(np.nan_to_num(a, nan=0.0)).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        z = zscore_frames(x, mask)
        fmap = torch.cat([model.encode_frames(z[s:s + chunk]) for s in range(0, Fn, chunk)])
        app = torch.cat([model.frame_features(z[s:s + chunk]) for s in range(0, Fn, chunk)]).float()
        # every pair any window uses
        W = Fn - K + 1
        pairs = sorted({(w + i, w + j) for w in range(W) for i, j in model.pairs})
        pid = {p: k for k, p in enumerate(pairs)}
        ia = torch.tensor([p[0] for p in pairs], device=dev)
        ib = torch.tensor([p[1] for p in pairs], device=dev)
        tt = torch.as_tensor(t, device=dev, dtype=torch.float32)
        dtp = (tt[ib] - tt[ia]) / (ib - ia).float()
        s1 = torch.cat([phase_shift(z[s:min(s + chunk, Fn - 1)], z[s + 1:min(s + chunk, Fn - 1) + 1])
                        for s in range(0, Fn - 1, chunk)])
        init = chain_init(s1, ia, ib)
        gs = torch.full((len(pairs),), float(gsd), device=dev)
        pf, u8 = [], []
        for s in range(0, len(pairs), chunk):
            sl = slice(s, s + chunk)
            f_, u_ = model.pair_features(z, mask, fmap, ia[sl], ib[sl], init[sl], gs[sl], dtp[sl])
            pf.append(f_.float()); u8.append(u_.float())
        pf, u8 = torch.cat(pf), torch.cat(u8)
        # windows
        widx = torch.tensor([[pid[(w + i, w + j)] for i, j in model.pairs] for w in range(W)], device=dev)
        valid_all = mask.float()
        vs, ss, rs = [], [], []
        for s in range(0, W, 64):
            wi = widx[s:s + 64]
            B = wi.shape[0]
            ws = torch.arange(s, s + B, device=dev)
            v8 = torch.stack([F.avg_pool2d(valid_all[w:w + K].min(0, keepdim=True).values[None], 8)[0] for w in ws.tolist()])
            dtw = (tt[ws + K - 1] - tt[ws]) / (K - 1)
            v, sig, d = model.head(pf[wi], u8[wi], app[ws], app[ws + K - 1], v8,
                                   torch.full((B,), float(gsd), device=dev), dtw)
            vs.append(v.float()); ss.append(sig.float()); rs.append(d["v_raw"].float())
    v, sig, raw = (torch.cat(q).cpu().numpy().astype(np.float64) for q in (vs, ss, rs))
    if g:
        v, raw = unaugment_vector(v, g), unaugment_vector(raw, g)
        sig = np.abs(unaugment_vector(sig, g))
    return v, sig, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--sets", default=str(SETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=None, help="default: the training crop")
    ap.add_argument("--tta", default="0", help="dihedral elements to average, e.g. 0 or 0,3,5,6")
    ap.add_argument("--chunk", type=int, default=32, help="frames / pairs per SEA-RAFT call")
    ap.add_argument("--runs", type=int, default=0, help="only the first n runs (quick checks)")
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    raise_open_files()
    model, st = load_model(args.ckpt, dev)
    a = st["args"]
    K = a["K"]
    crop = args.crop or a["crop"]
    amp = bool(a.get("amp", 1))
    tta = [int(g) for g in args.tta.split(",")]
    ts = TileSet(Path(args.sets) / args.set)
    runs = ts.run_ids[:args.runs] if args.runs else ts.run_ids
    idx = ts.windows(K, 1, 1, 0.0, runs)
    print(f"{args.ckpt} (step {st['step']}) on {args.set}: {len(runs)} runs, {len(idx)} windows, crop {crop}, "
          f"tta {tta}", flush=True)
    t0 = time.time()
    keys, est, sig, raw = [], [], [], []
    for k, rid in enumerate(runs):
        fr = ts._fr[rid]
        t = fr["t"].to_numpy()
        gsd = float(fr["gsd"].iloc[0])
        parts = [predict_run(model, ts.tiles(rid), t, gsd, K, crop, g, dev, args.chunk, amp) for g in tta]
        v = np.mean([p[0] for p in parts], 0)
        s = np.mean([p[1] for p in parts], 0)
        r = np.mean([p[2] for p in parts], 0)
        w = idx[idx.run_id == rid]
        assert len(w) == len(v) and (w.w_start.to_numpy() == np.arange(len(v))).all()
        keys += [(ts.name, rid, int(ws), K, float(T)) for ws, T in zip(w.w_start, w["T"])]
        est.append(v); sig.append(s); raw.append(r)
        if k % 25 == 0:
            print(f"  {k + 1}/{len(runs)} runs, {time.time() - t0:.0f} s", flush=True)
    est, sig, raw = np.concatenate(est), np.concatenate(sig), np.concatenate(raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = prediction_frame(ts, keys, est, sig)[PRED_COLUMNS]
    df.to_csv(out, index=False)
    diag = df[["set", "run_id", "w_start", "K", "T", "v_true_x", "v_true_y"]].assign(v_raw_x=raw[:, 0], v_raw_y=raw[:, 1])
    diag.to_csv(str(out).removesuffix(".csv") + ".diag.csv", index=False)
    print(f"wrote {out}: {len(df)} windows in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
