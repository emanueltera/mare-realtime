"""Check that predict.py's per-run cached path gives the same estimates as FlowHeadNet.forward
on each window (on the box, any checkpoint):

    python -m flowhead.check_cache --ckpt /workspace/data/e2/flowhead/smoke/last.pt --set val --runs 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from e2.data import TileSet, WindowDataset, unaugment_vector  # noqa: E402
from flowhead.predict import SETS, load_model, predict_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", default="val")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--amp", type=int, default=None, help="override the checkpoint's bf16 setting")
    args = ap.parse_args()
    dev = torch.device("cuda")
    model, st = load_model(args.ckpt, dev)
    a = st["args"]
    if args.amp is not None:
        a["amp"] = args.amp
    ts = TileSet(SETS / args.set)
    worst = 0.0
    for rid in ts.run_ids[:args.runs]:
        fr = ts._fr[rid]
        for g in (0, 5):
            v_c, s_c, _ = predict_run(model, ts.tiles(rid), fr["t"].to_numpy(), float(fr["gsd"].iloc[0]), a["K"],
                                      a["crop"], g, dev, 32, bool(a["amp"]))
            ds = WindowDataset(ts, K=a["K"], crop=a["crop"], normalize=None, runs=[rid])
            for i in (0, 17, len(ds) - 1):
                x = ds.item(i, g)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a["amp"])):
                    v, s, _ = model(torch.from_numpy(x["tiles"])[None].to(dev), torch.from_numpy(x["mask"])[None].to(dev),
                                    torch.tensor([x["gsd"]], device=dev), torch.tensor([x["dt"]], device=dev))
                v = unaugment_vector(v.float().cpu().numpy()[0], g)
                d = float(np.abs(v - v_c[i]).max())
                worst = max(worst, d)
                print(f"{rid[:32]} g={g} window {i}: forward {v.round(4)} cached {v_c[i].round(4)} |diff| {d:.2e} m/s")
    print(f"worst |forward − cached| = {worst:.2e} m/s")


if __name__ == "__main__":
    main()
