"""Train E3's relative-pose network (or its sensor-only twin with --blind).

    python -m e3.train --train <sets>/train --val <sets>/val --out <dir> [--blind] [--hours 3]

Checkpoints: <out>/last.pt each epoch, <out>/best.pt at the lowest validation translation
error (mean |dp − dp̂| in metres over val's pairs, fixed per-run sensor noise).
<out>/log.csv has one row per epoch.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairDataset
from .model import PoseNet, laplace_nll


def evaluate(model, loader, dev) -> dict:
    model.eval()
    err_m, step_m, rot_net, rot_gyro = [], [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for img, x, y, h, _, _ in loader:
            img, x, y, h = img.to(dev, non_blocking=True), x.to(dev), y.to(dev), h.to(dev)
            mu, _ = model(img, x)
            hm = h[:, None] / 10
            err_m.append(((mu[:, :3] - y[:, :3]) * hm).norm(dim=1).cpu())
            step_m.append((y[:, :3] * hm).norm(dim=1).cpu())
            rot_net.append(((mu[:, 3:] - y[:, 3:]) / 100).norm(dim=1).cpu())
            rot_gyro.append((y[:, 3:] / 100).norm(dim=1).cpu())
    e, s = torch.cat(err_m), torch.cat(step_m)
    model.train()
    return dict(t_err_m=e.mean().item(), t_err_med_pct=(e / s).median().item() * 100,
                rot_net_deg=torch.cat(rot_net).mean().item(), rot_gyro_deg=torch.cat(rot_gyro).mean().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--limit", type=int, default=None, help="use only the first N training runs")
    p.add_argument("--fx", action="store_true", help="train and validate through the camera effects (e3/fx.py)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-airspeed", action="store_true", help="zero the airspeed inputs (a platform without an airspeed sensor)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"
    air = not a.no_airspeed
    tr = PairDataset(a.train, train=True, limit=a.limit, images=not a.blind, fx=a.fx, airspeed=air)
    va = PairDataset(a.val, train=False, images=not a.blind, fx=a.fx, airspeed=air)
    if not va.cam.matches(tr.cam):
        raise SystemExit(f"validation camera {va.cam} is not the training camera {tr.cam}")
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    vl = DataLoader(va, 128, shuffle=False, num_workers=min(16, a.workers), pin_memory=True)
    model = PoseNet(blind=a.blind, cam=tr.cam).to(dev).to(memory_format=torch.channels_last)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, camera {tr.cam}, train pairs {len(tr)}, val pairs {len(va)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.02)
    total = a.epochs * len(tl)
    t0, best, step = time.time(), float("inf"), 0
    # Cosine over whichever runs out first, the epochs or the time cap, so the rate is annealed either way.
    progress = lambda s: min(1.0, max(s / total, (time.time() - t0) / (a.hours * 3600)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 1000) * 0.5 * (1 + math.cos(math.pi * progress(s))))
    log = open(out / "log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "step", "hours", "train_loss", "val_t_err_m", "val_t_err_med_pct", "val_rot_net_deg", "val_rot_gyro_deg", "lr"])
    stop = False
    for ep in range(a.epochs):
        losses = []
        for img, x, y, _, _, _ in tl:
            img = img.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, lb = model(img, x)
            loss = laplace_nll(mu, lb, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            if step % 500 == 0:
                print(f"ep {ep} step {step} loss {np.mean(losses[-500:]):.4f} {(time.time() - t0) / 3600:.2f} h", flush=True)
            if progress(step) >= 1:
                stop = True
                break
        m = evaluate(model, vl, dev)
        hours = (time.time() - t0) / 3600
        w.writerow([ep, step, f"{hours:.3f}", f"{np.mean(losses):.5f}", f"{m['t_err_m']:.5f}", f"{m['t_err_med_pct']:.4f}",
                    f"{m['rot_net_deg']:.6f}", f"{m['rot_gyro_deg']:.6f}", f"{sched.get_last_lr()[0]:.2e}"])
        log.flush()
        print(f"== epoch {ep} val {m} {hours:.2f} h", flush=True)
        ck = dict(model=model.state_dict(), blind=a.blind, epoch=ep, val=m, cam=tr.cam.as_dict(), airspeed=air)
        torch.save(ck, out / "last.pt")
        if m["t_err_m"] < best:
            best = m["t_err_m"]
            torch.save(ck, out / "best.pt")
        if stop:
            break
    print("done", flush=True)


if __name__ == "__main__":
    main()
