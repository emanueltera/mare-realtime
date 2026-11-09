"""B1 pair velocities: pretrained SEA-RAFT-M on consecutive E2 tiles.

For every run and every consecutive pair of tiles (i, i + 1) the flow model gives a flow f
(tile pixels) at each cell of tile i. The tile translates exactly with the camera, so a
static sea would move through it at −v·dt/gsd; the camera velocity a pattern tracker
reports is therefore

    v_pattern = −median(f)·gsd / dt          (component-wise median, tile axes)

over the cells that are valid in tile i, whose match lands inside tile i + 1 (with a
margin) on a valid cell. On a live sea this is biased by the waves' own motion: that bias
is what E1 measured on full frames and what B1 measures on the tiles.

The model sees the tile as an 8-bit grey image: linear luminance re-encoded with the sRGB
transfer curve (the tiles were decoded from sRGB frames, so this approximates the rendered
image's grey level) and copied to three channels; masked cells are 0. SEA-RAFT-M is loaded
and run exactly as in E1 (experiments/e1_wave_bias/flow_bias.py: checkpoint
MemorySlices/Tartan-C-T-TSKH-spring540x960-M, config spring-M, 4 iterations, full
resolution), here on 384 × 384 tiles.

Output: the standard prediction CSV with K = 2 (one row per pair) plus three diagnostic
columns the harness ignores: flow_valid_frac (fraction of cells used), flow_spread_x/y
(1.4826·MAD of the per-cell velocity, m/s).

    python -m baselines.flowpairs --set-root /workspace/data/e2/sets/val --out pairs/searaft/val.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2.data import PRED_COLUMNS, TileSet, prediction_frame  # noqa: E402

SEA_RAFT_DIR = Path("/workspace/models/SEA-RAFT")
SEA_RAFT_CKPT = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
EXTRA_COLUMNS = ["flow_valid_frac", "flow_spread_x", "flow_spread_y"]


def to_image(tiles: np.ndarray) -> np.ndarray:
    """[F, n, n] linear luminance (NaN = masked) → [F, n, n] uint8 sRGB-encoded grey, 0 where masked."""
    y = np.clip(np.nan_to_num(np.asarray(tiles, dtype=np.float32), nan=0.0), 0.0, 1.0)
    s = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return np.rint(255 * s).astype(np.uint8)


class SeaRaft:
    """SEA-RAFT-M as E1 ran it (fp32, full resolution, 4 iterations)."""

    def __init__(self, device, iters: int = 4):
        import torch

        sys.path.insert(0, str(SEA_RAFT_DIR / "core"))
        sys.path.insert(0, str(SEA_RAFT_DIR))
        from raft import RAFT  # noqa: PLC0415

        args = argparse.Namespace(**json.loads((SEA_RAFT_DIR / "config/eval/spring-M.json").read_text()))
        args.iters = iters
        self.args = args
        self.m = RAFT.from_pretrained(SEA_RAFT_CKPT, args=args).to(device).eval()
        self.torch = torch

    def __call__(self, a, b):
        """a, b: (B, 3, H, W) float 0..255 → flow (B, 2, H, W) in pixels, channel 0 = x (col)."""
        with self.torch.no_grad():
            out = self.m(a, b, iters=self.args.iters, test_mode=True)
        return out["flow"][-1].float()


def pair_velocities(model, imgs, valid, gsd: float, dt: np.ndarray, chunk: int = 40, margin: float = 16.0):
    """imgs: uint8 [F, n, n] (torch, on the device), valid: bool [F, n, n]; dt [F − 1] seconds.
    Returns v [F − 1, 2] (tile axes, m/s; NaN when fewer than 1 % of cells are usable),
    valid fraction [F − 1] and robust spread [F − 1, 2]."""
    import torch

    F, n, _ = imgs.shape
    dev = imgs.device
    rows = torch.arange(n, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(rows, rows, indexing="ij")  # cell centres in (row, col) index units
    v_out, frac_out, spread_out = [], [], []
    for s in range(0, F - 1, chunk):
        e = min(F - 1, s + chunk)
        a = imgs[s:e].float()[:, None].expand(-1, 3, -1, -1)
        b = imgs[s + 1:e + 1].float()[:, None].expand(-1, 3, -1, -1)
        f = model(a.contiguous(), b.contiguous())  # (B, 2, n, n)
        x2 = xx + f[:, 0]
        y2 = yy + f[:, 1]
        inside = (x2 >= margin) & (x2 <= n - 1 - margin) & (y2 >= margin) & (y2 <= n - 1 - margin)
        xi = x2.round().clamp(0, n - 1).long()
        yi = y2.round().clamp(0, n - 1).long()
        v2 = valid[s + 1:e + 1]
        land = torch.gather(v2.reshape(e - s, -1), 1, (yi * n + xi).reshape(e - s, -1)).reshape(e - s, n, n)
        ok = valid[s:e] & inside & land & torch.isfinite(f).all(1)
        dtt = torch.as_tensor(dt[s:e], device=dev, dtype=torch.float32).view(-1, 1)
        vel = -f.reshape(e - s, 2, -1) * (gsd / dtt)[:, None, :]  # (B, 2, n²) m/s per cell
        vel = torch.where(ok.reshape(e - s, 1, -1), vel, torch.full_like(vel, float("nan")))
        med = vel.nanmedian(dim=2).values  # (B, 2)
        mad = (vel - med[..., None]).abs().nanmedian(dim=2).values * 1.4826
        frac = ok.reshape(e - s, -1).float().mean(1)
        med[frac < 0.01] = float("nan")
        v_out.append(med.cpu().numpy())
        frac_out.append(frac.cpu().numpy())
        spread_out.append(mad.cpu().numpy())
    return np.concatenate(v_out), np.concatenate(frac_out), np.concatenate(spread_out)


class _Runs:
    """One item per run: (run_id, uint8 images, valid mask, gsd, dt); read in DataLoader workers."""

    def __init__(self, ts: TileSet, ids):
        self.ts, self.ids = ts, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        rid = self.ids[k]
        a = np.asarray(self.ts.tiles(rid))
        t = self.ts._fr[rid]["t"].to_numpy()
        return rid, to_image(a), np.isfinite(a), self.ts.gsd(rid), np.diff(t)


def run_set(set_root, out_csv, runs=None, device="cuda", chunk=40, limit=None, log_every=50, workers=6):
    import torch

    # TF32 matmuls and convolutions (the correlation volume is a matmul): ~3× faster than
    # strict fp32 on this GPU. E1 ran with torch's defaults (TF32 convolutions only).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ts = TileSet(set_root)
    ids = runs or ts.run_ids
    if limit:
        ids = ids[:limit]
    model = SeaRaft(device)
    loader = torch.utils.data.DataLoader(_Runs(ts, ids), batch_size=None, num_workers=workers, prefetch_factor=2)
    keys, est, extra = [], [], []
    t0 = time.time()
    done = 0
    for rid, img, valid, gsd, dt in loader:
        img, valid, gsd, dt = np.asarray(img), np.asarray(valid), float(gsd), np.asarray(dt)
        v, frac, spread = pair_velocities(model, torch.as_tensor(img, device=device),
                                          torch.as_tensor(valid, device=device), gsd, dt, chunk)
        t = ts._fr[rid]["t"].to_numpy()
        keys += [(ts.name, rid, w, 2, float(t[w + 1] - t[w])) for w in range(len(v))]
        est.append(v)
        extra.append(np.column_stack([frac, spread]))
        done += 1
        if done % log_every == 0:
            el = time.time() - t0
            print(f"{ts.name}: {done}/{len(ids)} runs, {el:.0f} s, {done * len(v) / el:.1f} pairs/s, "
                  f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    df = prediction_frame(ts, keys, np.concatenate(est))[PRED_COLUMNS]
    df[EXTRA_COLUMNS] = np.concatenate(extra)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    el = time.time() - t0
    print(f"{ts.name}: wrote {len(df)} pairs to {out_csv} in {el:.0f} s; GPU peak "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m baselines.flowpairs")
    ap.add_argument("--set-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=40, help="pairs per forward pass")
    ap.add_argument("--limit", type=int, default=None, help="first N runs only (B3's train subset, smoke tests)")
    ap.add_argument("--workers", type=int, default=6, help="DataLoader workers reading tiles")
    a = ap.parse_args()
    run_set(a.set_root, a.out, chunk=a.chunk, limit=a.limit, workers=a.workers)
