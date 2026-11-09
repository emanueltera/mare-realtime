"""Train model A ('e2e').

    cd experiments/e2_learned_velocity
    python -m e2e.train --out /workspace/data/e2/e2e/main --K 8 --steps 30000

Loss, per window, in m/s of camera velocity (tile axes):
- every iteration i of the read-out: smooth-L1(v_i − v_true) (β = 0.05 m/s), weighted
  0.8^(n_iter − 1 − i) (the last counts most; u is detached between iterations, RAFT-style);
- the last iteration's σ: Gaussian NLL of the (detached) error, ½(e/σ)² + log σ per axis,
  weight 0.2, so σ learns the error's scale without pulling on the estimate.

Model selection: the EMA weights (decay 0.999) are scored every `--val-every` steps on val
windows (every `--val-step`-th start of every val run); the checkpoint with the lowest
median per-run bias b (the harness's headline metric) is kept as best.pt. Test sets are
never read here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from e2.data import TileSet
from e2e.loader import ScaleBatches, Windows, to_device
from e2e.model import E2E, n_params

SETS = "/workspace/data/e2/sets"


def per_run_bias(run_id, err) -> pd.DataFrame:
    """b = |run-mean of (v_est − v_true)| per run (the harness's b, on whatever windows given)."""
    d = pd.DataFrame({"run_id": run_id, "ex": err[:, 0], "ey": err[:, 1]})
    g = d.groupby("run_id")[["ex", "ey"]].mean()
    return np.hypot(g.ex, g.ey)


@torch.no_grad()
def evaluate(model, ds: Windows, device, batch=32, workers=6, n_iter=None):
    """Estimates for every window of ds (no augmentation) and the val summary."""
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    vs, ss, u0s, idx = [], [], [], []
    for b in dl:
        b = to_device(b, device)
        o = model(b["tiles"], b["gsd"], b["dt"], n_iter=n_iter)
        vs.append(o["v"].cpu())
        ss.append(o["sigma_v"].cpu())
        u0s.append((-o["u0"] * (b["gsd"] / b["dt"])[:, None]).cpu())
        idx.append(b["i"].cpu())
    v, s, v0, i = (torch.cat(a).numpy() for a in (vs, ss, u0s, idx))
    t = ds.target[i]
    err = v - t
    b = per_run_bias(ds.run_id[i], err)
    b0 = per_run_bias(ds.run_id[i], v0 - t)
    z = np.abs(err) / s
    summ = {"val_b_median": float(b.median()), "val_b_mean": float(b.mean()), "val_b_p90": float(b.quantile(0.9)),
            "val_rms": float(np.sqrt((err ** 2).sum(1).mean())), "val_mae": float(np.abs(err).mean()),
            "val_frac_b_le_017": float((b <= 0.17).mean()), "val_b0_median": float(b0.median()),
            "val_nll": float((0.5 * z ** 2 + np.log(s)).mean()), "val_cov1": float((z <= 1).mean()),
            "val_cov2": float((z <= 2).mean())}
    model.train()
    return summ, {"v": v, "sigma": s, "v0": v0, "i": i}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m e2e.train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default=SETS)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--n-iter", type=int, default=4)
    ap.add_argument("--max-gap", type=int, default=7)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--p-bin", type=float, default=0.25, help="probability of a 2×2-binned (gsd × 2) batch")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=1500)
    ap.add_argument("--val-step", type=int, default=4, help="val windows start every n frames")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--train-runs", type=int, default=None, help="first n train runs only (smoke test)")
    ap.add_argument("--val-on-train", action="store_true", help="score on the training runs (smoke test)")
    ap.add_argument("--max-hours", type=float, default=2.5)
    ap.add_argument("--gpu-mem-gb", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cudnn.benchmark = True

    tr_ts = TileSet(f"{a.sets}/train")                      # memory-mapped: shared page cache
    runs = tr_ts.run_ids[:a.train_runs] if a.train_runs else None
    tr = Windows(tr_ts, a.K, step=1, runs=runs, augment=True, seed=a.seed)
    if a.val_on_train:
        va = Windows(tr_ts, a.K, step=a.val_step, runs=runs)
    else:
        va = Windows(TileSet(f"{a.sets}/val"), a.K, step=a.val_step)
    print(f"train windows {len(tr)}, val windows {len(va)}", flush=True)

    a.arch = {"K": a.K, "max_gap": a.max_gap, "radius": a.radius}
    model = E2E(n_iter=a.n_iter, **a.arch).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"parameters {n_params(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)

    def lr_at(step, hours):
        # linear warm-up, then a cosine over the run's progress: the larger of the step and
        # the wall-time fractions, so the schedule ends on time even if the GPU is busy
        prog = min(1.0, max(step / a.steps, hours / a.max_hours))
        return a.lr * min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * prog))
    dl = torch.utils.data.DataLoader(tr, batch_sampler=ScaleBatches(len(tr), a.batch, a.steps, a.p_bin, a.seed),
                                     num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    log, best, t0 = [], float("inf"), time.time()
    acc = {"loss": 0.0, "nll": 0.0, "n": 0, **{f"l1_it{i}": 0.0 for i in range(a.n_iter)}, "l1_init": 0.0}
    w_it = [0.8 ** (a.n_iter - 1 - i) for i in range(a.n_iter)]
    step = 0
    for batch in dl:
        b = to_device(batch, dev)
        tgt = b["target"].float()
        scale = (b["gsd"] / b["dt"])[:, None]
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, (time.time() - t0) / 3600)
        opt.zero_grad(set_to_none=True)
        o = model(b["tiles"], b["gsd"], b["dt"])
        loss = 0
        for i, u in enumerate(o["us"]):
            v = -u * scale
            loss = loss + w_it[i] * F.smooth_l1_loss(v, tgt, beta=0.05)
            acc[f"l1_it{i}"] += (v.detach() - tgt).abs().mean().item()
        nll = (0.5 * ((o["v"].detach() - tgt) / o["sigma_v"]) ** 2 + torch.log(o["sigma_v"])).mean()
        loss = loss + 0.2 * nll
        loss.backward()
        acc["loss"] += loss.item()
        acc["nll"] += nll.item()
        acc["l1_init"] += (-o["u0"] * scale - tgt).abs().mean().item()
        acc["n"] += 1
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        hours = (time.time() - t0) / 3600
        last = step == a.steps or hours >= a.max_hours
        if step % 100 == 0 or last:
            n = acc["n"]
            row = {"step": step, "hours": hours, "lr": opt.param_groups[0]["lr"], "grad_norm": float(gn),
                   **{k: v / n for k, v in acc.items() if k != "n"},
                   "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            if step % a.val_every == 0 or last:
                s, _ = evaluate(ema, va, dev)
                row.update(s)
                torch.save({"model": ema.state_dict(), "args": vars(a), "step": step}, out / "last.pt")
                if s["val_b_median"] < best:
                    best = s["val_b_median"]
                    torch.save({"model": ema.state_dict(), "args": vars(a), "step": step, "val": s}, out / "best.pt")
            log.append(row)
            pd.DataFrame(log).to_csv(out / "log.csv", index=False)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            acc = {k: 0.0 for k in acc} | {"n": 0}
        if last:
            break
    print(f"done: {step} steps, {(time.time() - t0) / 3600:.2f} h, best val b median {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
