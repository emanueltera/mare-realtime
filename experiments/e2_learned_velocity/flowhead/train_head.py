"""Train the flowhead head on cached frozen-backbone features (flowhead.cache); select on val.

    python -m flowhead.train_head --cache /workspace/data/e2/flowhead/cache --out /workspace/data/e2/flowhead/head-main \
        --gaps 1,2,4,7 --minutes 40

Same head, loss and inputs as flowhead.train with the backbone frozen; only the source of
the pair features differs (computed once on the centred 256 crop for back-to-back windows,
instead of on the fly on random crops). Augmentation: the 8 rotations / flips of the square,
applied to the cached maps — exact for the head's problem (the scalar channels are moved,
the vector channels u and the initial shift are also transformed, like the target) though
SEA-RAFT itself is not exactly equivariant. The checkpoint (best.pt by val median b on the
cached val windows, last.pt) holds the whole FlowHeadNet, so predict.py runs it unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e2.data import DIHEDRAL  # noqa: E402
from flowhead.model import FlowHeadNet, make_pairs  # noqa: E402
from flowhead.train import loss_fn  # noqa: E402

CACHE_GAPS = (1, 2, 4, 7)


class Cache:
    """One cached set (the parts <name>.0, <name>.1, … written by flowhead.cache) in host
    memory (~0.7 MB per window; batches go to the GPU as they are drawn), restricted to the
    model's pairs."""

    def __init__(self, base, name, pairs, dev, limit=0):
        self.dev = dev
        parts = sorted(Path(base).glob(f"{name}.*"), key=lambda p: int(p.suffix[1:]))
        sel = [make_pairs(8, CACHE_GAPS).index(p) for p in pairs]
        self.win = pd.concat([pd.read_csv(p / "windows.csv") for p in parts], ignore_index=True)
        n = len(self.win) if not limit else min(limit, len(self.win))
        self.win = self.win.iloc[:n]

        def load(key, take=None):
            a = np.concatenate([np.load(p / f"{key}.npy", mmap_mode="r")[:, take] if take else
                                np.load(p / f"{key}.npy", mmap_mode="r") for p in parts])[:n]
            return torch.from_numpy(np.ascontiguousarray(a))

        self.pf = load("pf", sel)
        self.u8 = load("u8", sel)
        self.frames = load("frames")
        self.valid8 = load("valid8")
        self.gsd = torch.tensor(self.win.gsd.to_numpy(), dtype=torch.float32, device=dev)
        self.dt = torch.tensor(self.win.dt.to_numpy(), dtype=torch.float32, device=dev)
        self.target = torch.tensor(self.win[["v_true_x", "v_true_y"]].to_numpy(), dtype=torch.float32, device=dev)

    def __len__(self):
        return len(self.win)


def augment(t, g, vec=()):
    """Dihedral element g on maps t [B, C, h, w] (as e2.data.augment_array); `vec` lists the
    channel pairs (x, y) that are vectors in tile axes (as e2.data.augment_vector)."""
    fx, fy, tr = DIHEDRAL[g]
    t = t.clone()
    if fx:
        t = t.flip(-1)
        for cx, _ in vec:
            t[:, cx] = -t[:, cx]
    if fy:
        t = t.flip(-2)
        for _, cy in vec:
            t[:, cy] = -t[:, cy]
    if tr:
        t = t.transpose(-1, -2)
        for cx, cy in vec:
            t[:, [cx, cy]] = t[:, [cy, cx]]
    return t


def batch(model, c: Cache, ix, g):
    """Head inputs for windows ix (a CPU index tensor) of the cache under dihedral element g
    (one per batch), on the GPU."""
    B, P = len(ix), model.P
    get = lambda t: t[ix].to(c.dev, non_blocking=True).float()  # noqa: E731
    pf = get(c.pf).reshape(B * P, -1, *c.pf.shape[-2:])
    pf = augment(pf, g, vec=((0, 1), (2, 3))).reshape(B, P, -1, *pf.shape[-2:])
    u8 = augment(get(c.u8).reshape(B * P, 2, *c.u8.shape[-2:]), g, vec=((0, 1),)).reshape(B, P, 2, *pf.shape[-2:])
    fr = augment(get(c.frames), g)
    v8 = augment(get(c.valid8), g)
    tg = augment(c.target[ix.to(c.dev)][:, :, None, None], g, vec=((0, 1),))[:, :, 0, 0]
    return pf, u8, fr, v8, tg


def run_head(model, pf, u8, fr, v8, gsd, dt):
    B = fr.shape[0]
    app = model.frame_features(fr.reshape(-1, *fr.shape[-2:])).reshape(B, 2, -1, *pf.shape[-2:])
    return model.head(pf, u8, app[:, 0], app[:, 1], v8, gsd, dt)


@torch.no_grad()
def evaluate(model, c: Cache, bs=128):
    model.eval()
    est, raw, sig = [], [], []
    for s in range(0, len(c), bs):
        ix = torch.arange(s, min(s + bs, len(c)))
        pf, u8, fr, v8, _ = batch(model, c, ix, 0)
        ixd = ix.to(c.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v, sg, d = run_head(model, pf, u8, fr, v8, c.gsd[ixd], c.dt[ixd])
        est.append(v.float()); raw.append(d["v_raw"].float()); sig.append(sg.float())
    model.train()
    tru = c.target.cpu().numpy()
    out = {}
    for name, a in (("", torch.cat(est)), ("raw_", torch.cat(raw))):
        e = a.cpu().numpy() - tru
        m = pd.DataFrame({"run_id": c.win.run_id, "ex": e[:, 0], "ey": e[:, 1]}).groupby("run_id").mean()
        bb = np.hypot(m.ex, m.ey).to_numpy()
        out[name + "b_med"] = float(np.median(bb))
        out[name + "b_p90"] = float(np.quantile(bb, 0.9))
        out[name + "b_frac017"] = float((bb <= 0.17).mean())
        out[name + "rms"] = float(np.sqrt((e ** 2).sum(1).mean()))
    e = torch.cat(est).cpu().numpy() - tru
    out["z_rms"] = float(np.sqrt(((e / torch.cat(sig).cpu().numpy()) ** 2).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/data/e2/flowhead/cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gaps", default="1,2,4,7")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--nll-weight", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--train-limit", type=int, default=0, help="first n cached windows only (smoke test)")
    ap.add_argument("--val-on-train", action="store_true", help="smoke test: validate on the training windows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gaps = [int(g) for g in args.gaps.split(",")]
    model = FlowHeadNet(K=8, gaps=gaps, backbone="frozen").to(dev).train()
    tr = Cache(args.cache, "train", model.pairs, dev, args.train_limit)
    va = tr if args.val_on_train else Cache(args.cache, "val", model.pairs, dev)
    print(f"pairs {model.pairs}; train windows {len(tr)}, val windows {len(va)}", flush=True)
    # what predict.py needs to rebuild the model
    meta = {"K": 8, "gaps": args.gaps, "iters": model.iters, "crop": tr.frames.shape[-1], "amp": 1,
            "backbone": "frozen", "trained_on": "cache", **vars(args)}
    (out / "args.json").write_text(json.dumps(meta, indent=1))
    params = [p for n, p in model.named_parameters() if not n.startswith("raft.")]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    lw = csv.writer(open(out / "train_log.csv", "w", newline=""))
    lw.writerow(["step", "time_s", "loss", "huber", "nll", "abs_err", "lr"])
    vrows, best, t0 = [], math.inf, time.time()
    acc = np.zeros(4)
    for step in range(1, args.steps + 1):
        prog = min(1.0, max(step / args.steps, (time.time() - t0) / (args.minutes * 60)))
        for gr in opt.param_groups:
            gr["lr"] = args.lr * min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * prog))
        ix = torch.from_numpy(rng.integers(0, len(tr), args.batch))
        pf, u8, fr, v8, tg = batch(model, tr, ix, int(rng.integers(8)))
        ixd = ix.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v, sg, _ = run_head(model, pf, u8, fr, v8, tr.gsd[ixd], tr.dt[ixd])
        loss, hub, nll, e = loss_fn(v.float(), sg.float(), tg, args.nll_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        acc += [loss.item(), hub.item(), nll.item(), e.norm(dim=1).mean().item()]
        if step % 100 == 0:
            lw.writerow([step, round(time.time() - t0, 1), *(acc / 100), opt.param_groups[0]["lr"]])
            acc[:] = 0
        done = step == args.steps or prog >= 1.0
        if step % args.val_every == 0 or done:
            m = evaluate(model, va)
            m.update(step=step, time_s=round(time.time() - t0, 1))
            vrows.append(m)
            pd.DataFrame(vrows).to_csv(out / "val_log.csv", index=False)
            print("VAL", json.dumps(m), flush=True)
            state = {"model": model.state_dict(), "step": step, "args": meta, "val": m}
            torch.save(state, out / "last.pt")
            if m["b_med"] < best:
                best = m["b_med"]
                torch.save(state, out / "best.pt")
        if done:
            break
    print(f"done: {step} steps in {time.time() - t0:.0f} s; best cached-val median b {best:.4f}; "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
