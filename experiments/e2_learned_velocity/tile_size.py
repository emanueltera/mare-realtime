"""How much of the tile falls outside the image for each tile size N (cell = the nadir image
GSD, so side = N·cam_u/fx), from the runs' frames.csv alone (no pixels needed).
Writes results/tile_size.csv: one row per (set, N).

    /workspace/venv/bin/python tile_size.py --render /workspace/data/e2/render/val/e2-val --set val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from e2 import geom
from e2.rectify import theta_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--sizes", default="256,320,384,512")
    ap.add_argument("--every", type=int, default=2, help="use every k-th frame")
    ap.add_argument("--out", default="results/tile_size.csv")
    a = ap.parse_args()
    rows = []
    dirs = sorted(d for d in Path(a.render).iterdir() if (d / "frames.csv").exists())
    for n in map(int, a.sizes.split(",")):
        per_run, worst = [], []
        for d in dirs:
            fr = geom.read_frames(d)
            spec = geom.load_run(d)
            tile = geom.run_tile(fr, n, geom.theta_for(theta_key(a.set, spec["id"])))
            off = tile.cell_offsets()
            W, H = fr[0]["width"], fr[0]["height"]
            v = []
            for r in fr[:: a.every]:
                u, vv, _ = geom.pixel_map(r, tile, off)
                v.append((np.isfinite(u) & (u >= 0.5) & (vv >= 0.5) & (u <= W - 0.5) & (vv <= H - 0.5)).mean())
            per_run.append(1 - np.mean(v))
            worst.append(1 - np.min(v))
        per_run = np.array(per_run)
        rows.append({"set": a.set, "runs": len(dirs), "n": n, "side_over_height": n / fr[0]["fx"],
                     "masked_mean": per_run.mean(), "masked_run_p90": np.quantile(per_run, 0.9),
                     "masked_run_max": per_run.max(), "masked_frame_max": max(worst)})
        print(rows[-1])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[old.set != a.set], df], ignore_index=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
