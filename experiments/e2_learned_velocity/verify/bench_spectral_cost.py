"""Time the spectral model's fused (torch.compile) shell cost against the plain-torch version
on one val batch of 12 windows at K = 32 (stage-1 grid, 49² candidates). Read-only.

    cd /workspace/mare-realtime/experiments/e2_learned_velocity
    /workspace/venv/bin/python -m verify.bench_spectral_cost
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2.data import TileSet  # noqa: E402
from spectral import spectrum as spx  # noqa: E402
from spectral.data import SpectralWindows  # noqa: E402
from spectral.model import grid  # noqa: E402
from spectral.train import collate, to_dev  # noqa: E402


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def main():
    ds = SpectralWindows(TileSet("/workspace/data/e2/sets/val"), 32, step=7)
    b = to_dev(collate([ds[i] for i in range(0, 12 * 50, 50)]), "cuda")
    sp = spx.prepare(b["tiles"], b["gsd"], b["dt"], bin=2, pad=2, band_bins=1.0)
    w = torch.rand(12, sp.k.shape[1], device="cuda")
    w = w / w.sum(1, keepdim=True)
    V = b["centre"][:, None, :] + grid(6.0, 0.25, "cuda")[None]
    args = (sp.Sband.contiguous(), sp.k.contiguous(), sp.omega.contiguous(), sp.dwp, V.contiguous(), w.contiguous())
    with torch.no_grad():
        ref = spx._cost_fwd(*args)
        fus = spx._cost_fwd_c(*args)
        print("max |fused - plain| / max|plain|:", float((ref - fus).abs().max() / ref.abs().max()))
        tf = timeit(lambda: spx._cost_fwd_c(*args))
        tp = timeit(lambda: spx._cost_fwd(*args), n=5)
    print(f"forward, batch 12, nk {sp.k.shape[1]}, M {V.shape[1]}: fused {tf:.1f} ms, plain {tp:.1f} ms, ratio {tp / tf:.0f}x")


if __name__ == "__main__":
    main()
