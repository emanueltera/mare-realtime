"""Throughput and memory of one training step (forward + backward) on random tiles:
python -m e2e.bench [--batch 16 --n 384 --Ks 2 4 8]."""

from __future__ import annotations

import argparse
import time

import torch

from e2e.model import E2E, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--Ks", type=int, nargs="*", default=[2, 4, 8])
    ap.add_argument("--n", type=int, default=384)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    for K in a.Ks:
        tiles = torch.rand(a.batch, K, a.n, a.n, device=dev).half()
        gsd = torch.full((a.batch,), 0.1, device=dev)
        dt = torch.full((a.batch,), 0.1, device=dev)
        m = E2E(K=K).to(dev)
        opt = torch.optim.AdamW(m.parameters(), 1e-4)
        torch.cuda.reset_peak_memory_stats()
        for k in range(13):
            if k == 3:
                torch.cuda.synchronize()
                t0 = time.time()
            opt.zero_grad()
            o = m(tiles, gsd, dt)
            (sum(u.square().mean() for u in o["us"]) + o["log_sigma"].mean()).backward()
            opt.step()
        torch.cuda.synchronize()
        s = (time.time() - t0) / 10
        print(f"K={K}: {n_params(m)} params, {s * 1000:.0f} ms/step, {a.batch / s:.1f} windows/s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
