"""Train model B ('flowhead') on the E2 train set; select on val only.

    cd /workspace/mare-realtime/experiments/e2_learned_velocity
    python -m flowhead.train --out /workspace/data/e2/flowhead/main --gaps 1,2,4,7 --steps 30000

Loss per window, e = v_est − v_true (m/s, tile axes, augmented with the tiles):
    Huber(e, δ = 0.5 m/s)  +  λ · Gaussian NLL of the *detached* e under σ
so σ learns to describe the error without the NLL down-weighting hard windows in the fit of v.
Every `--val-every` steps the model predicts val windows (every `--val-step`-th start, centred
crop) and logs per-run bias b = |mean_w(v_est − v_true)| (median, p90, fraction ≤ 0.17 m/s), the
same b of the head's un-corrected flow average (v_raw) and the window RMS. The checkpoint with the
lowest val median b is kept as best.pt; last.pt is the end of training. The test sets are never
read here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e2.data import TileSet, WindowDataset  # noqa: E402
from flowhead.model import FlowHeadNet  # noqa: E402

SETS = Path("/workspace/data/e2/sets")


def to_dev(batch, dev):
    return (batch["tiles"].to(dev, non_blocking=True), batch["mask"].to(dev, non_blocking=True),
            batch["gsd"].to(dev).float(), batch["dt"].to(dev).float(), batch["target"].to(dev).float())


def collate(items):
    out = {k: torch.as_tensor(np.stack([it[k] for it in items])) for k in ("tiles", "mask", "gsd", "dt", "target", "aug")}
    out["tiles"] = out["tiles"].half()   # halves the host → device copy; luminance needs no more
    out["key"] = [it["key"] for it in items]
    return out


def raise_open_files():
    # TileSet keeps one memory map (one file descriptor) per run it has read: a worker that
    # samples all 4500 train runs needs more than the default soft limit of 1024
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def worker_init(wid):
    # every worker gets its own augmentation stream (the dataset's rng is copied on fork)
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng([info.seed % 2**32, wid])


def loss_fn(v, sigma, target, nll_weight):
    e = v - target
    hub = F.huber_loss(v, target, delta=0.5)
    nll = (0.5 * (e.detach() / sigma) ** 2 + torch.log(sigma)).mean()
    return hub + nll_weight * nll, hub, nll, e


@torch.no_grad()
def evaluate_windows(model, ds, dev, batch, amp, workers=6):
    """Predict ds (no augmentation) → per-run b of v and of v_raw, window RMS."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate)
    est, raw, sig = [], [], []
    for b in dl:
        tiles, mask, gsd, dt, _ = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            v, s, d = model(tiles.float(), mask, gsd, dt)
        est.append(v.float().cpu().numpy())
        raw.append(d["v_raw"].float().cpu().numpy())
        sig.append(s.float().cpu().numpy())
    model.train()
    idx = ds.index
    tru = idx[["v_true_x", "v_true_y"]].to_numpy()
    out = {}
    for name, a in (("", np.concatenate(est)), ("raw_", np.concatenate(raw))):
        e = a - tru
        df = idx[["run_id"]].assign(ex=e[:, 0], ey=e[:, 1])
        m = df.groupby("run_id")[["ex", "ey"]].mean()
        bb = np.hypot(m.ex, m.ey).to_numpy()
        out[name + "b_med"] = float(np.median(bb))
        out[name + "b_p90"] = float(np.quantile(bb, 0.9))
        out[name + "b_frac017"] = float((bb <= 0.17).mean())
        out[name + "rms"] = float(np.sqrt((e ** 2).sum(1).mean()))
    e = np.concatenate(est) - tru
    s = np.concatenate(sig)
    out["z_rms"] = float(np.sqrt(((e / s) ** 2).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-set", default=str(SETS / "train"))
    ap.add_argument("--val-set", default=str(SETS / "val"))
    ap.add_argument("--gaps", default="1,2,4,7")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--backbone", default="frozen", choices=["frozen", "update", "all"])
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--max-hours", type=float, default=2.5, help="stop (and keep best/last) after this wall time")
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--nll-weight", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=1500)
    ap.add_argument("--val-step", type=int, default=4, help="val windows: every n-th start")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--amp", type=int, default=1, help="bf16 autocast")
    ap.add_argument("--runs", type=int, default=0, help="use only the first n train runs (smoke test)")
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights (e.g. fine-tune a frozen-backbone model)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    raise_open_files()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    dev = torch.device("cuda")

    # tiles are memory-mapped: the page cache holds them once for every process on the box
    ts = TileSet(args.train_set, preload=False)
    runs = ts.run_ids[:args.runs] if args.runs else None
    ds = WindowDataset(ts, K=args.K, augment=True, crop=args.crop, normalize=None, seed=args.seed, runs=runs)
    vs = TileSet(args.val_set, preload=False)
    if args.runs:   # smoke test: "val" is the same train runs, un-augmented (it must overfit)
        vds = WindowDataset(ts, K=args.K, step=args.val_step, crop=args.crop, normalize=None, runs=runs)
    else:
        vds = WindowDataset(vs, K=args.K, step=args.val_step, crop=args.crop, normalize=None)
    print(f"train windows {len(ds)}, val windows {len(vds)}", flush=True)

    gaps = [int(g) for g in args.gaps.split(",")]
    model = FlowHeadNet(K=args.K, gaps=gaps, iters=args.iters, backbone=args.backbone).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
    model.train()
    head_params = [p for n, p in model.named_parameters() if not n.startswith("raft.")]
    groups = [{"params": head_params, "lr": args.lr_head, "base_lr": args.lr_head}]
    bb_params = [p for p in model.raft.parameters() if p.requires_grad]
    if bb_params:
        groups.append({"params": bb_params, "lr": args.lr_backbone, "base_lr": args.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    print(f"pairs {model.pairs}; head params {sum(p.numel() for p in head_params) / 1e6:.2f} M; "
          f"backbone {args.backbone} ({sum(p.numel() for p in bb_params) / 1e6:.2f} M trained)", flush=True)

    sampler = RandomSampler(ds, replacement=True, num_samples=args.steps * args.batch)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, collate_fn=collate,
                    worker_init_fn=worker_init, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    log = open(out / "train_log.csv", "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["step", "time_s", "loss", "huber", "nll", "abs_err", "lr_head", "mem_gb"])
    vlog_path = out / "val_log.csv"
    vrows = []
    best = math.inf
    t0 = time.time()
    acc = {"loss": 0.0, "huber": 0.0, "nll": 0.0, "abs": 0.0, "n": 0}
    step = 0
    for batch in dl:
        step += 1
        # cosine decay over whichever runs out first, the steps or the wall-time budget
        prog = min(1.0, max(step / args.steps, (time.time() - t0) / (args.max_hours * 3600)))
        f = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        tiles, mask, gsd, dt, target = to_dev(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
            v, sigma, _ = model(tiles.float(), mask, gsd, dt)
        loss, hub, nll, e = loss_fn(v.float(), sigma.float(), target, args.nll_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc["loss"] += loss.item(); acc["huber"] += hub.item(); acc["nll"] += nll.item()
        acc["abs"] += e.norm(dim=1).mean().item(); acc["n"] += 1
        if step % 50 == 0:
            n = acc["n"]
            row = [step, round(time.time() - t0, 1), acc["loss"] / n, acc["huber"] / n, acc["nll"] / n, acc["abs"] / n,
                   opt.param_groups[0]["lr"], torch.cuda.max_memory_allocated() / 1e9]
            lw.writerow(row); log.flush()
            print(" ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in row), flush=True)
            acc = {k: 0.0 for k in acc}; acc["n"] = 0
        done = step >= args.steps or (time.time() - t0) > args.max_hours * 3600
        if step % args.val_every == 0 or done:
            m = evaluate_windows(model, vds, dev, 2 * args.batch, bool(args.amp))
            m.update(step=step, time_s=round(time.time() - t0, 1))
            vrows.append(m)
            with open(vlog_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(vrows[0].keys()))
                w.writeheader(); w.writerows(vrows)
            print("VAL", json.dumps(m), flush=True)
            state = {"model": model.state_dict(), "step": step, "args": vars(args), "val": m}
            torch.save(state, out / "last.pt")
            if m["b_med"] < best:
                best = m["b_med"]
                torch.save(state, out / "best.pt")
        if done:
            break
    print(f"done: {step} steps in {time.time() - t0:.0f} s; best val median b {best:.4f}; "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
