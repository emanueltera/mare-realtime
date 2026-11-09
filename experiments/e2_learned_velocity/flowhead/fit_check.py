"""Under- or over-fitting? The cached heads' per-run bias on training windows (the first
`--limit` cached train windows, i.e. the first runs, back-to-back windows) next to the same
numbers on the cached val windows.

    python -m flowhead.fit_check --heads head-g1247-lr5e-4,head-g1-lr5e-4 --out fit_check.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from flowhead.predict import load_model  # noqa: E402
from flowhead.train_head import Cache, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/data/e2/flowhead")
    ap.add_argument("--heads", default="head-g1247-lr5e-4,head-g1-lr5e-4")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    rows = []
    for h in args.heads.split(","):
        m, st = load_model(Path(args.root) / h / "best.pt", dev)
        for name, limit in (("train", args.limit), ("val", 0)):
            c = Cache(Path(args.root) / "cache", name, m.pairs, dev, limit=limit)
            rows.append({"head": h, "step": st["step"], "windows": name, "runs": c.win.run_id.nunique(),
                         "n_windows": len(c), **evaluate(m, c)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
