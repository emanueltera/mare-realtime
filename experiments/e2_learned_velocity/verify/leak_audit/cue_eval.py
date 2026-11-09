"""Score A (e2e K=8) and C (spectral K=32) and B0 pairs on the cue-probe sets; paired against ctrl."""
import os, sys, math, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
A = Path("/workspace/data/e2/audit_leak")
os.environ["SPECTRAL_PATTERN_DIRS"] = str(A / "cue_pairs")
sys.path.insert(0, "/workspace/mare-realtime/experiments/e2_learned_velocity")
from e2.data import TileSet
from e2.baselines import phasecorr_predictions
from e2e.predict import load, predict_set
from spectral.data import SpectralWindows
from spectral.predict import predict as c_predict
from spectral.train import build
torch.cuda.set_per_process_memory_fraction(16 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
dev = torch.device("cuda")
VARS = ["ctrl", "gate0", "gaterev", "foam0", "cloud0"]
mA, ckA = load("/workspace/data/e2/e2e/main/best.pt", dev)
ckC = torch.load("/workspace/data/e2/spectral/k32/best.pt", map_location="cpu", weights_only=False)
targs = argparse.Namespace(**ckC["args"])
mC = build(targs, targs.K * targs.pad).to(dev); mC.load_state_dict(ckC["model"]); mC.eval()
(A / "cue_pairs").mkdir(exist_ok=True)
rows = []
for v in VARS:
    root = A / "cue_sets" / v
    if not (root / "set.json").exists():
        print("missing", v); continue
    ts = TileSet(root)
    if not (A / "cue_pairs" / f"{ts.name}.csv").exists():
        phasecorr_predictions(str(root), str(A / "cue_pairs" / f"{ts.name}.csv"), workers=16)
    # A
    idx, p = predict_set(mA, ts, ckA["args"]["K"], dev)
    e = p["v"] - idx[["v_true_x", "v_true_y"]].to_numpy()
    d = pd.DataFrame({"run_id": idx.run_id, "ex": e[:, 0], "ey": e[:, 1]}); d["method"] = "A"
    # C
    ds = SpectralWindows(ts, targs.K, step=1, register=targs.stages != "noreg")
    V, S, _ = c_predict(mC, ds, dev, 16, 8, False)
    e = V - ds.index[["v_true_x", "v_true_y"]].to_numpy()
    dc = pd.DataFrame({"run_id": ds.index.run_id, "ex": e[:, 0], "ey": e[:, 1]}); dc["method"] = "C"
    # B0 pairs (K = 2)
    pr = pd.read_csv(A / "cue_pairs" / f"{ts.name}.csv")
    db = pd.DataFrame({"run_id": pr.run_id, "ex": pr.v_est_x - pr.v_true_x, "ey": pr.v_est_y - pr.v_true_y}); db["method"] = "B0-K2"
    for dd in (d, dc, db):
        g = dd.groupby(["method", "run_id"])[["ex", "ey"]].mean().reset_index()
        g["variant"] = v
        rows.append(g)
    print(v, "done", flush=True)
R = pd.concat(rows, ignore_index=True)
val = pd.read_csv("/workspace/data/e2/sets/val/runs.csv"); val["slug"] = val.run_id.str.replace(r"^e2-val-\d+-", "", regex=True)
R["slug"] = R.run_id.str.split("-", n=3).str[3]
R = R.merge(val[["slug", "theta_deg", "wind_speed", "wind_direction", "ground_speed_mean", "altitude"]], on="slug", how="left")
th = np.radians(R.theta_deg); wd = np.radians(R.wind_direction)
R["e_e"] = R.ex * np.cos(th) - R.ey * np.sin(th); R["e_n"] = R.ex * np.sin(th) + R.ey * np.cos(th)
R["along"] = R.e_e * np.sin(wd) + R.e_n * np.cos(wd); R["cross"] = R.e_e * np.cos(wd) - R.e_n * np.sin(wd)
R["b"] = np.hypot(R.ex, R.ey)
R.to_csv(A / "cue_per_run.csv", index=False)
c = R[R.variant == "ctrl"].set_index(["method", "slug"])
out = []
for (m, v), g in R.groupby(["method", "variant"]):
    g = g.set_index(["method", "slug"])
    dal = g.along - c.loc[g.index, "along"]; dcr = g.cross - c.loc[g.index, "cross"]
    dv = np.hypot(g.ex - c.loc[g.index, "ex"], g.ey - c.loc[g.index, "ey"])
    out.append({"method": m, "variant": v, "runs": len(g), "b_median": g.b.median(), "frac_b_le_017": (g.b <= 0.17).mean(),
                "along_median": g.along.median(), "d_along_vs_ctrl_median": dal.median(), "d_along_p10": dal.quantile(.1),
                "d_along_p90": dal.quantile(.9), "d_cross_vs_ctrl_median": dcr.median(), "d_vec_vs_ctrl_median": dv.median()})
S = pd.DataFrame(out); S.to_csv(A / "cue_summary.csv", index=False)
print(S.round(3).to_string())
