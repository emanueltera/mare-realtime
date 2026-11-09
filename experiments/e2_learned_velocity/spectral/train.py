"""Train the spectral model (model C) on the E2 train set, with a validation curve.

    cd /workspace/mare-realtime/experiments/e2_learned_velocity
    /workspace/venv/bin/python -m spectral.train --out /workspace/data/e2/spectral/k32 --K 32

Every epoch draws `--samples` random windows (stride 1, any start) from train with a random
dihedral element, and scores a fixed subset of val (every `--val-step`-th window start, no
augmentation): per-run b = |mean(v_est − v_true)| over that subset, its median and p90, the
window error, and the share of windows whose truth the first stage's grid contains. The
checkpoint with the lowest val median b is kept as best.pt (selection on val only).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing

# Batches come back from the loader workers through shared memory files, not descriptors.
torch.multiprocessing.set_sharing_strategy("file_system")

from e2.data import TileSet

from .data import SpectralWindows, augment_tiles, worker_init
from .model import SpectralNet, losses

SETS = Path("/workspace/data/e2/sets")
STAGES = {
    # first stage centred on the classical pattern velocity
    "reg": ((6.0, 0.25, None), (0.8, 0.05, None)),
    # no registration: the first stage searches the whole speed range on the long waves
    "noreg": ((84.0, 1.0, 0.1), (6.0, 0.25, None), (0.8, 0.05, None)),
}


def collate(items):
    return {"tiles": torch.stack([it["tiles"] for it in items]),
            **{k: torch.as_tensor(np.array([it[k] for it in items])) for k in ("gsd", "dt", "centre", "target", "aug", "i")}}


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def build(args, ntp):
    return SpectralNet(ntp=ntp, stages=STAGES[args.stages], hid=args.hid,
                       prep=dict(bin=args.bin, pad=args.pad, band_bins=args.band))


@torch.no_grad()
def run_eval(model, ds, dev, bs=16, workers=6, limit=None):
    """Predict ds (no augmentation). Returns (v_est [n, 2], sigma [n, 2], inside0 [n]) in ds.index order."""
    model.eval()
    n = len(ds) if limit is None else min(limit, len(ds))
    sub = torch.utils.data.Subset(ds, range(n))
    dl = torch.utils.data.DataLoader(sub, batch_size=bs, num_workers=workers, collate_fn=collate, pin_memory=True)
    V = np.zeros((n, 2))
    S = np.zeros((n, 2))
    inside = np.zeros(n, dtype=bool)
    for b in dl:
        b = to_dev(b, dev)
        out = model(b["tiles"], b["gsd"], b["dt"], b["centre"])
        i = b["i"].cpu().numpy()
        V[i] = out["v"].cpu().numpy()
        S[i] = out["logsig"].exp().cpu().numpy()
        c, o, _ = out["stages"][0]
        inside[i] = ((b["target"] - c).abs() <= o.abs().max()).all(-1).cpu().numpy()
    model.train()
    return V, S, inside


def val_metrics(ds, V, S, inside):
    idx = ds.index.iloc[:len(V)]
    err = V - idx[["v_true_x", "v_true_y"]].to_numpy()
    df = pd.DataFrame({"run_id": idx.run_id.to_numpy(), "ex": err[:, 0], "ey": err[:, 1],
                       "speed": np.hypot(idx.v_true_x, idx.v_true_y)})
    g = df.groupby("run_id").agg(ex=("ex", "mean"), ey=("ey", "mean"), speed=("speed", "mean"))
    b = np.hypot(g.ex, g.ey)
    e = np.hypot(err[:, 0], err[:, 1])
    z = np.abs(err) / np.maximum(S, 1e-6)
    return {"val_b_median": float(np.median(b)), "val_b_p90": float(np.quantile(b, 0.9)),
            "val_b_le_017": float((b <= 0.17).mean()), "val_bpct_median": float(np.median(b / g.speed * 100)),
            "val_win_err_median": float(np.median(e)), "val_win_err_p90": float(np.quantile(e, 0.9)),
            "val_inside0": float(inside.mean()), "val_z_within1": float((z <= 1).mean()),
            "val_sigma_median": float(np.median(S))}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spectral.train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--stages", default="reg", choices=list(STAGES))
    ap.add_argument("--bin", type=int, default=2)
    ap.add_argument("--pad", type=int, default=None, help="temporal zero-padding factor (default: 64 bins)")
    ap.add_argument("--band", type=float, default=1.0, help="shell band half-width, unpadded ω' bins")
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--samples", type=int, default=12000, help="train windows per epoch")
    ap.add_argument("--max-minutes", type=float, default=150.0, help="stop after this much wall time")
    ap.add_argument("--teacher-steps", type=int, default=3000)
    ap.add_argument("--val-step", type=int, default=7)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--train-runs", type=int, default=None, help="use only the first N train runs (smoke test)")
    ap.add_argument("--val-on-train", action="store_true", help="score the train subset instead of val (smoke test)")
    ap.add_argument("--bench", type=int, default=0, help="time this many steps and exit")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    pad = a.pad or max(1, 64 // a.K)
    a.pad = pad
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    dev = torch.device("cuda")
    register = a.stages != "noreg"

    tr_ts = TileSet(SETS / "train")
    runs = tr_ts.run_ids[:a.train_runs] if a.train_runs else None
    train = SpectralWindows(tr_ts, a.K, augment=True, register=register, runs=runs)
    if a.val_on_train:
        val = SpectralWindows(tr_ts, a.K, step=a.val_step, register=register, runs=runs)
    else:
        val = SpectralWindows(TileSet(SETS / "val"), a.K, step=a.val_step, register=register)
    print(f"train windows {len(train)} (pattern missing {train.index.pat_missing.mean():.4f}), val {len(val)}", flush=True)

    model = build(a, a.K * pad).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * (a.samples // a.bs)
    t0 = time.time()

    def progress(s):
        # by steps or by wall time, whichever is further along, so the cosine ends with the budget
        return min(1.0, max(s / total, (time.time() - t0) / 60 / a.max_minutes))

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress(s)))))

    curve, step, best = [], 0, float("inf")
    log = open(out / "train_log.csv", "a")
    for ep in range(a.epochs):
        sampler = torch.utils.data.RandomSampler(train, replacement=False, num_samples=min(a.samples, len(train)))
        dl = torch.utils.data.DataLoader(train, batch_size=a.bs, sampler=sampler, num_workers=a.workers,
                                         collate_fn=collate, pin_memory=True, drop_last=True,
                                         worker_init_fn=worker_init, persistent_workers=False, prefetch_factor=4)
        acc, te = {}, time.time()
        for b in dl:
            b = to_dev(b, dev)
            b["tiles"] = augment_tiles(b["tiles"], b["aug"])
            teacher = max(0.0, 0.5 * (1 - step / max(1, a.teacher_steps)))
            o = model(b["tiles"], b["gsd"], b["dt"], b["centre"], target=b["target"], teacher=teacher)
            loss, parts = losses(o, b["target"], model.stages)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            acc["loss"] = acc.get("loss", 0.0) + float(loss)
            acc["n"] = acc.get("n", 0) + 1
            if step % 50 == 0:
                msg = " ".join(f"{k}={v / acc['n']:.4f}" for k, v in acc.items() if k != "n")
                print(f"ep {ep} step {step} {time.time() - te:.0f}s lr {sched.get_last_lr()[0]:.2e} teacher {teacher:.2f} {msg}",
                      flush=True)
                log.write(json.dumps({"ep": ep, "step": step, **{k: v / acc["n"] for k, v in acc.items() if k != "n"}}) + "\n")
                log.flush()
            if a.bench and step >= a.bench:
                dtime = time.time() - te
                print(f"bench: {step} steps in {dtime:.1f}s, {step * a.bs / dtime:.1f} windows/s, "
                      f"max mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
                return
            if (time.time() - t0) / 60 > a.max_minutes:
                break
        V, S, inside = run_eval(model, val, dev)
        m = val_metrics(val, V, S, inside)
        row = {"epoch": ep, "step": step, "minutes": (time.time() - t0) / 60,
               **{f"train_{k}": v / max(acc.get("n", 1), 1) for k, v in acc.items() if k != "n"}, **m,
               "max_mem_gib": torch.cuda.max_memory_allocated() / 2**30}
        curve.append(row)
        pd.DataFrame(curve).to_csv(out / "curve.csv", index=False)
        print("VAL", json.dumps(row), flush=True)
        ck = {"model": model.state_dict(), "args": vars(a), "epoch": ep, "step": step, "val": m}
        torch.save(ck, out / "last.pt")
        if m["val_b_median"] < best:
            best = m["val_b_median"]
            torch.save(ck, out / "best.pt")
        if (time.time() - t0) / 60 > a.max_minutes:
            print("time budget reached", flush=True)
            break
    print(f"done in {(time.time() - t0) / 60:.1f} min, best val median b {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
