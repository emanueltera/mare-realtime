"""B3: B1's pattern velocity plus a learned correction that never looks at the wave field.

    v_est = v_pattern + MLP(features(v_pattern, gsd))

v_pattern is B1's K = 8 window estimate (median SEA-RAFT-M flow over the window's 7 pairs,
baselines.windows). The features are only
    log gsd (m/px), |v_pattern| (m/s), and the pattern direction in tile axes (cos φ, sin φ)
— nothing computed from the tiles beyond B1's single velocity. If the learned models beat
B3 clearly, they read the waves rather than subtract an average correction for the
conditions (altitude via gsd, speed via |v_pattern|).

Two parameterisations (`--features`):
- full:   the four features above → a 2-D correction in tile axes. Trained with the 8
          dihedral augmentations applied to v_pattern and the target together (θ is random
          per run, so tile axes carry no world direction; this keeps it so).
- radial: (log gsd, |v_pattern|) → a correction (along, across) v_pattern's direction,
          rotated back to tile axes: equivariant by construction.
A third, `const`, adds the train-mean correction (a fixed vector) and has no parameters.

Data: the windows of the first 1500 train runs (B1 was run on those only; the 4 inputs do
not need more) for training; val for selection (median per-run b over val's K = 8 windows,
then p90); the test sets are only predicted, with the frozen checkpoint.

    python -m baselines.correction smoke   --preds /workspace/data/e2/preds/b1-searaft
    python -m baselines.correction train   --preds /workspace/data/e2/preds/b1-searaft --out /workspace/data/e2/baselines/b3
    python -m baselines.correction predict --preds /workspace/data/e2/preds/b1-searaft --ckpt /workspace/data/e2/baselines/b3/best.pt \\
        --out /workspace/data/e2/preds/b3-mlp --sets val test-iid test-sea test-alt test-rate e1-core
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2.data import DIHEDRAL, PRED_COLUMNS, TileSet, prediction_frame  # noqa: E402

SETS_ROOT = Path("/workspace/data/e2/sets")
K = 8


# ---- data ---------------------------------------------------------------------------------


def load_windows(preds_dir: Path, set_name: str, sets_root: Path = SETS_ROOT) -> pd.DataFrame:
    """B1's K = 8 rows of one set, with the run's tile gsd (from frames.csv, a model input)."""
    df = pd.read_csv(Path(preds_dir) / f"{set_name}.csv")
    df = df[df.K == K].reset_index(drop=True)
    fr = pd.read_csv(Path(sets_root) / set_name / "frames.csv")
    gsd = fr.groupby("run_id")["gsd"].first()
    df["gsd"] = df.run_id.map(gsd).to_numpy()
    return df


def dihedral_torch(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Apply dihedral element g[i] (e2.data.DIHEDRAL) to vector v[i] (…, 2) in tile axes."""
    tab = torch.tensor(DIHEDRAL, dtype=torch.bool)
    fx, fy, tr = tab[g, 0], tab[g, 1], tab[g, 2]
    x = torch.where(fx, -v[:, 0], v[:, 0])
    y = torch.where(fy, -v[:, 1], v[:, 1])
    return torch.stack([torch.where(tr, y, x), torch.where(tr, x, y)], 1)


# ---- model --------------------------------------------------------------------------------


class Correction(nn.Module):
    """MLP correction. forward(v_pattern [B, 2], gsd [B]) → Δv [B, 2] in tile axes."""

    def __init__(self, features: str = "full", hidden: int = 64, depth: int = 2):
        super().__init__()
        self.features = features
        nin = {"full": 4, "radial": 2}[features]
        layers, w = [], nin
        for _ in range(depth):
            layers += [nn.Linear(w, hidden), nn.SiLU()]
            w = hidden
        layers.append(nn.Linear(w, 2))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)  # start at "no correction"
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, vp: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        speed = vp.norm(dim=1).clamp_min(1e-6)
        u = vp / speed[:, None]
        base = [torch.log(gsd / 0.12), speed / 30.0]  # roughly centred: train gsd median 0.125, speeds 3–79
        if self.features == "full":
            return self.net(torch.stack(base + [u[:, 0], u[:, 1]], 1))
        out = self.net(torch.stack(base, 1))  # (along, across) v_pattern
        return out[:, :1] * u + out[:, 1:] * torch.stack([-u[:, 1], u[:, 0]], 1)


def predict_df(model, df: pd.DataFrame, const: np.ndarray | None = None) -> np.ndarray:
    vp = df[["v_est_x", "v_est_y"]].to_numpy()
    if const is not None:
        return vp + const
    ok = np.isfinite(vp).all(1)
    out = np.full_like(vp, np.nan)
    with torch.no_grad():
        d = model(torch.as_tensor(vp[ok], dtype=torch.float32), torch.as_tensor(df.gsd.to_numpy()[ok], dtype=torch.float32))
    out[ok] = vp[ok] + d.numpy()
    return out


def run_bias(df: pd.DataFrame, est: np.ndarray) -> pd.Series:
    """Per-run b = |mean(v_est − v_true)| (the harness's definition; for selection on val)."""
    e = est - df[["v_true_x", "v_true_y"]].to_numpy()
    g = pd.DataFrame(e, columns=["x", "y"]).groupby(df.run_id.to_numpy()).mean()
    return np.hypot(g.x, g.y)


# ---- training -----------------------------------------------------------------------------


def fit(train: pd.DataFrame, val: pd.DataFrame, features="full", hidden=64, depth=2, loss="mse", lr=2e-3,
        epochs=40, batch=2048, seed=0, log=None):
    torch.manual_seed(seed)
    ok = np.isfinite(train[["v_est_x", "v_est_y"]].to_numpy()).all(1)
    vp = torch.as_tensor(train.loc[ok, ["v_est_x", "v_est_y"]].to_numpy().copy(), dtype=torch.float32)
    tgt = torch.as_tensor(train.loc[ok, ["v_true_x", "v_true_y"]].to_numpy(), dtype=torch.float32) - vp
    gsd = torch.as_tensor(train.loc[ok, "gsd"].to_numpy(), dtype=torch.float32)
    model = Correction(features, hidden, depth)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(vp) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    crit = nn.MSELoss() if loss == "mse" else nn.HuberLoss(delta=1.0)
    curve, best = [], (np.inf, np.inf, None, -1)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(vp))
        tot = 0.0
        for i in range(0, len(vp), batch):
            j = perm[i:i + batch]
            g = torch.randint(0, 8, (len(j),))
            v_in, t_in = dihedral_torch(vp[j], g), dihedral_torch(tgt[j], g)
            l = crit(model(v_in, gsd[j]), t_in)
            opt.zero_grad()
            l.backward()
            opt.step()
            sched.step()
            tot += float(l.detach()) * len(j)
        model.eval()
        est = predict_df(model, val)
        b = run_bias(val, est)
        err = est - val[["v_true_x", "v_true_y"]].to_numpy()
        r = {"epoch": ep + 1, "train_loss": tot / len(vp), "val_mse": float(np.nanmean((err ** 2).sum(1))),
             "val_b_median": float(b.median()), "val_b_p90": float(b.quantile(0.9))}
        curve.append(r)
        if log:
            log(r)
        if (r["val_b_median"], r["val_b_p90"]) < best[:2]:
            best = (r["val_b_median"], r["val_b_p90"], {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[2])
    return model, pd.DataFrame(curve), {"val_b_median": best[0], "val_b_p90": best[1], "epoch": best[3]}


GRID = {"features": ["full", "radial"], "hidden": [32, 128], "depth": [2, 3], "loss": ["mse", "huber"]}


def cmd_train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, va = load_windows(a.preds, "train"), load_windows(a.preds, "val")
    print(f"train {len(tr)} windows ({tr.run_id.nunique()} runs), val {len(va)} ({va.run_id.nunique()} runs)", flush=True)
    rows, curves, best, best_by = [], [], None, {}
    # The parameter-free reference: B1 itself, and B1 + the train-mean correction.
    vp_tr = tr[["v_est_x", "v_est_y"]].to_numpy()
    const = np.nanmean(tr[["v_true_x", "v_true_y"]].to_numpy() - vp_tr, axis=0)
    for name, est in [("b1", va[["v_est_x", "v_est_y"]].to_numpy()), ("const", predict_df(None, va, const))]:
        b = run_bias(va, est)
        rows.append({"config": name, "val_b_median": float(b.median()), "val_b_p90": float(b.quantile(0.9))})
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID, vals), epochs=a.epochs)
        name = "-".join(f"{v}" for v in vals)
        t0 = time.time()
        model, curve, sel = fit(tr, va, **cfg)
        curve["config"] = name
        curves.append(curve)
        r = {"config": name, **cfg, **sel, "params": sum(p.numel() for p in model.parameters()),
             "seconds": time.time() - t0}
        rows.append(r)
        print(json.dumps(r), flush=True)
        score = (sel["val_b_median"], sel["val_b_p90"])
        if best is None or score < best[0]:
            best = (score, cfg, model)
        if cfg["features"] not in best_by or score < best_by[cfg["features"]][0]:
            best_by[cfg["features"]] = (score, cfg, model)
    pd.DataFrame(rows).to_csv(out / "selection.csv", index=False)
    pd.concat(curves).to_csv(out / "curves.csv", index=False)
    def save(entry, path):
        _, cfg, model = entry
        torch.save({"state_dict": model.state_dict(), "config": {k: cfg[k] for k in ("features", "hidden", "depth")},
                    "train_config": cfg, "const": const.tolist()}, path)

    save(best, out / "best.pt")  # the selected B3
    for f, entry in best_by.items():  # the best of each parameterisation (ablation)
        save(entry, out / f"best-{f}.pt")
    print(f"selected on val: {best[1]}", flush=True)


def cmd_smoke(a):
    """A small subset: the training loss must fall (64 windows: fit them almost exactly;
    4096 windows: settle at the noise floor the four features leave)."""
    tr = load_windows(a.preds, "train")
    tr = tr[np.isfinite(tr[["v_est_x", "v_est_y"]].to_numpy()).all(1)].sample(4096, random_state=0).reset_index(drop=True)
    for n, ep in [(64, 3000), (4096, 300)]:
        sub = tr.iloc[:n].reset_index(drop=True)
        _, curve, _ = fit(sub, sub, hidden=256, depth=3, epochs=ep, batch=min(n, 512), lr=3e-3)
        c = curve.iloc[[0, len(curve) // 10, len(curve) // 2, -1]]
        print(f"smoke n={n}:\n{c[['epoch', 'train_loss', 'val_mse', 'val_b_median']].to_string(index=False)}", flush=True)
        curve.to_csv(Path(a.out) / f"smoke_n{n}.csv", index=False) if a.out else None


def cmd_predict(a):
    ck = torch.load(a.ckpt, weights_only=False)
    model = Correction(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    const = np.array(ck["const"]) if a.const else None
    for s in a.sets:
        df = load_windows(a.preds, s)
        ts = TileSet(SETS_ROOT / s)
        keys = list(zip(df["set"], df.run_id, df.w_start, df.K, df["T"]))
        prediction_frame(ts, keys, predict_df(model, df, const))[PRED_COLUMNS].to_csv(out / f"{s}.csv", index=False)
        print(f"{s}: {len(df)} windows", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    ap = argparse.ArgumentParser(prog="python -m baselines.correction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--preds", required=True)
    p.add_argument("--out", default=None)
    p = sub.add_parser("train")
    p.add_argument("--preds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p = sub.add_parser("predict")
    p.add_argument("--preds", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--const", action="store_true", help="the parameter-free B1 + train-mean correction instead")
    a = ap.parse_args()
    {"smoke": cmd_smoke, "train": cmd_train, "predict": cmd_predict}[a.cmd](a)
