"""Independent checks of model B ('flowhead'). Run on the box:

    cd /workspace/mare-realtime/experiments/e2_learned_velocity
    /workspace/venv/bin/python -m verify.verify_flowhead --out /workspace/data/e2/verify_flowhead/checks

Writes CSVs (one per check) into --out:
  leak_sets.csv        run-id overlap between every pair of sets; cached windows' runs ⊂ train
  leak_inputs.csv      frames.csv gsd == runs.csv gsd; dt from frames.csv t == window dt; θ spread per set
  rerun_vs_reported.csv my re-predictions (verify_flowhead/preds) against the reported prediction CSVs
  rerun_eval.csv       'all' slice of my harness re-run next to the reported slices.csv
  headline_recomputed.csv per (method, set) per-run b recomputed from the prediction CSVs, against summary_all
  seaRate_recomputed.csv test-rate median b per seaRate recomputed from the predictions
  raw_vs_head_recomputed.csv head vs v_raw from diag CSVs
  e1_compare_recomputed.csv medians from e1_compare.csv
  selection.csv        best cached-val step of every candidate (val_log.csv)
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SETS = Path("/workspace/data/e2/sets")
PREDS = Path("/workspace/data/e2/preds")
EVAL = Path("/workspace/data/e2/eval")
FH = Path("/workspace/data/e2/flowhead")
V = Path("/workspace/data/e2/verify_flowhead")
SET_NAMES = ["train", "val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]


def per_run_b(df, est=("v_est_x", "v_est_y")):
    e = df.assign(ex=df[est[0]] - df.v_true_x, ey=df[est[1]] - df.v_true_y)
    g = e.groupby("run_id")
    m = g[["ex", "ey"]].mean()
    return np.hypot(m.ex, m.ey)


def stats(b, speed=None):
    out = {"runs": len(b), "b_median": float(b.median()), "b_p90": float(b.quantile(0.9)),
           "frac_b_le_017": float((b <= 0.17).mean())}
    return out


def leak_sets(out):
    ids = {s: set(pd.read_csv(SETS / s / "runs.csv").run_id) for s in SET_NAMES}
    seeds = {}
    rows = []
    for s in SET_NAMES:
        r = pd.read_csv(SETS / s / "runs.csv")
        if {"traj_seed", "sea_seed"} <= set(r.columns):
            seeds[s] = set(zip(r.traj_seed, r.sea_seed))
    for a, b in itertools.combinations(SET_NAMES, 2):
        row = {"a": a, "b": b, "run_id_overlap": len(ids[a] & ids[b])}
        if a in seeds and b in seeds:
            row["seed_overlap"] = len(seeds[a] & seeds[b])
        rows.append(row)
    for part in sorted((FH / "cache").glob("*.*")):
        w = pd.read_csv(part / "windows.csv")
        base = part.name.split(".")[0]
        r = set(w.run_id)
        rows.append({"a": f"cache/{part.name}", "b": base, "run_id_overlap": len(r & ids[base]),
                     "n_runs": len(r), "outside_own_set": len(r - ids[base]),
                     "in_any_test": sum(len(r & ids[t]) for t in SET_NAMES if t.startswith(("test", "e1")))})
    pd.DataFrame(rows).to_csv(out / "leak_sets.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def leak_inputs(out):
    from e2.data import TileSet
    rows = []
    for s in SET_NAMES:
        fr = pd.read_csv(SETS / s / "frames.csv")
        runs = pd.read_csv(SETS / s / "runs.csv").set_index("run_id")
        g = fr.groupby("run_id")
        gsd_spread = (g.gsd.max() - g.gsd.min()).max()
        gsd_vs_runs = (g.gsd.first() - runs.gsd.reindex(g.gsd.first().index)).abs().max()
        dtf = g.t.apply(lambda t: np.diff(np.sort(t.to_numpy())))
        dts = np.concatenate(dtf.to_numpy())
        th = runs.theta_deg.to_numpy()
        hist = np.histogram(th % 360, bins=8, range=(0, 360))[0]
        rows.append({"set": s, "runs": len(runs), "gsd_within_run_spread_max": gsd_spread,
                     "gsd_frames_minus_runs_absmax": gsd_vs_runs, "dt_min": dts.min(), "dt_max": dts.max(),
                     "theta_min": th.min(), "theta_max": th.max(), "theta_octant_counts": " ".join(map(str, hist)),
                     "frames_columns": " ".join(fr.columns)})
    pd.DataFrame(rows).to_csv(out / "leak_inputs.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def rerun(out):
    rows, ev = [], []
    for f in sorted((V / "preds").glob("*.csv")):
        if f.name.endswith(".diag.csv"):
            continue
        s = f.stem
        mine = pd.read_csv(f)
        rep = pd.read_csv(PREDS / "flowhead" / f.name)
        j = mine.merge(rep, on=["run_id", "w_start"], suffixes=("_m", "_r"))
        d = np.hypot(j.v_est_x_m - j.v_est_x_r, j.v_est_y_m - j.v_est_y_r)
        bm, br = per_run_b(mine), per_run_b(rep)
        db = (bm - br.reindex(bm.index)).abs()
        rows.append({"set": s, "windows_mine": len(mine), "windows_reported": len(rep), "joined": len(j),
                     "win_diff_median": d.median(), "win_diff_p99": d.quantile(0.99), "win_diff_max": d.max(),
                     "frac_win_diff_gt_1e-2": float((d > 1e-2).mean()),
                     "run_b_absdiff_median": db.median(), "run_b_absdiff_max": db.max(),
                     "b_median_mine": bm.median(), "b_median_reported": br.median(),
                     "frac_b_le_017_mine": float((bm <= 0.17).mean()), "frac_b_le_017_reported": float((br <= 0.17).mean())})
        for who, d_ in (("mine", V / "eval" / s), ("reported", EVAL / "flowhead" / s)):
            p = d_ / "slices.csv"
            if p.exists():
                sl = pd.read_csv(p)
                a = sl[sl.slice == "all"].iloc[0].to_dict()
                ev.append({"set": s, "who": who, **a})
    pd.DataFrame(rows).to_csv(out / "rerun_vs_reported.csv", index=False)
    pd.DataFrame(ev).to_csv(out / "rerun_eval.csv", index=False)
    print(pd.DataFrame(rows).T.to_string())
    print(pd.DataFrame(ev).T.to_string())


def headline(out):
    rows, raw, rate = [], [], []
    for m in ["flowhead", "flowhead-gap1", "flowhead-onthefly", "flowhead-ft"]:
        for f in sorted((PREDS / m).glob("*.csv")):
            if f.name.endswith(".diag.csv"):
                continue
            s = f.stem
            df = pd.read_csv(f)
            b = per_run_b(df)
            runs = pd.read_csv(SETS / s / "runs.csv").set_index("run_id")
            bp = 100 * b / runs.ground_speed_mean.reindex(b.index)
            sl = pd.read_csv(EVAL / m / s / "slices.csv")
            a = sl[sl.slice == "all"].iloc[0]
            rows.append({"method": m, "set": s, **stats(b), "bpct_median": float(bp.median()),
                         "frac_bpct_le_028": float((bp <= 0.28).mean()), "frac_bpct_le_05": float((bp <= 0.5).mean()),
                         "harness_b_median": a.b_median, "harness_b_p90": a.b_p90,
                         "harness_bpct_median": a.b_pct_median, "harness_frac_b_le_017": a.frac_b_le_017,
                         "harness_frac_bpct_le_028": a.frac_bpct_le_028, "harness_frac_bpct_le_05": a.frac_bpct_le_05,
                         "harness_drift_pct_median": a.drift_pct_median})
            dg = Path(str(f).removesuffix(".csv") + ".diag.csv")
            if dg.exists():
                d = pd.read_csv(dg)
                br = per_run_b(d, ("v_raw_x", "v_raw_y"))
                raw.append({"method": m, "set": s, "b_median_head": b.median(), "b_median_raw": br.median()})
            if s == "test-rate":
                rr = runs.sea_rate
                for k, g in b.groupby(rr.reindex(b.index)):
                    rate.append({"method": m, "sea_rate": k, "runs": len(g), "b_median": g.median()})
                for _, x in sl[sl.slice == "sea_rate"].iterrows():
                    rate.append({"method": m, "sea_rate": f"harness {x['bin']}",
                                 "runs": x.runs, "b_median": x.b_median})
    pd.DataFrame(rows).to_csv(out / "headline_recomputed.csv", index=False)
    pd.DataFrame(raw).to_csv(out / "raw_vs_head_recomputed.csv", index=False)
    pd.DataFrame(rate).to_csv(out / "seaRate_recomputed.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(raw).to_string(index=False))
    print(pd.DataFrame(rate).to_string(index=False))


def e1(out):
    rows = []
    for m in ["flowhead", "flowhead-gap1", "flowhead-ft"]:
        p = EVAL / m / "e1-core" / "e1_compare.csv"
        if not p.exists():
            continue
        c = pd.read_csv(p)
        num = c.select_dtypes("number")
        rows.append({"method": m, "runs": len(c), **{f"median_{k}": float(num[k].median()) for k in num.columns}})
    pd.DataFrame(rows).to_csv(out / "e1_compare_recomputed.csv", index=False)
    print(pd.DataFrame(rows).T.to_string())


def selection(out):
    rows = []
    for d in sorted(FH.glob("*/val_log.csv")):
        v = pd.read_csv(d)
        i = v.b_med.idxmin()
        rows.append({"run": d.parent.name, "evals": len(v), "best_step": int(v.step[i]), "best_b_med": v.b_med[i],
                     "last_b_med": v.b_med.iloc[-1], "best_raw_b_med": v.raw_b_med[i]})
    import torch
    for r in rows:
        p = FH / r["run"] / "best.pt"
        if p.exists():
            st = torch.load(p, map_location="cpu", weights_only=False)
            r["ckpt_step"] = st["step"]
            r["ckpt_val_b_med"] = st["val"]["b_med"]
    pd.DataFrame(rows).to_csv(out / "selection.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def calib(out):
    """sigma calibration recomputed with my own code (window z, coverage, Spearman via ranks,
    run-level z with sigma / sqrt(windows / 8)), per set and per seaRate."""
    rows = []
    for f in sorted((PREDS / "flowhead").glob("*.csv")):
        if f.name.endswith(".diag.csv"):
            continue
        df = pd.read_csv(f)
        groups = [(f.stem, df)]
        if f.stem == "test-rate":
            rr = pd.read_csv(SETS / "test-rate" / "runs.csv").set_index("run_id").sea_rate
            groups += [(f"test-rate seaRate {k:g}", g) for k, g in df.groupby(df.run_id.map(rr))]
        for name, g in groups:
            e = np.concatenate([g.v_est_x - g.v_true_x, g.v_est_y - g.v_true_y])
            sg = np.concatenate([g.sigma_x, g.sigma_y])
            z = e / sg
            rs = pd.Series(sg).rank().to_numpy()
            ra = pd.Series(np.abs(e)).rank().to_numpy()
            rho = np.corrcoef(rs, ra)[0, 1]
            gg = g.assign(ex=g.v_est_x - g.v_true_x, ey=g.v_est_y - g.v_true_y).groupby("run_id")
            m = gg[["ex", "ey", "sigma_x", "sigma_y"]].mean()
            nw = gg.size()
            zr = np.concatenate([m.ex / (m.sigma_x / np.sqrt(nw / 8)), m.ey / (m.sigma_y / np.sqrt(nw / 8))])
            rows.append({"set": name, "windows": len(g), "z_rms": np.sqrt(np.mean(z ** 2)),
                         "frac_z_le_1": np.mean(np.abs(z) <= 1), "frac_z_le_2": np.mean(np.abs(z) <= 2),
                         "sigma_median": np.median(sg), "spearman": rho, "run_z_rms": np.sqrt(np.mean(zr ** 2))})
    pd.DataFrame(rows).to_csv(out / "calibration_recomputed.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def slices_check(out):
    """test-iid altitude / wind / sun slices and e1-core comparisons recomputed from per-run b."""
    rows = []
    df = pd.read_csv(PREDS / "flowhead" / "test-iid.csv")
    b = per_run_b(df)
    runs = pd.read_csv(SETS / "test-iid" / "runs.csv").set_index("run_id").reindex(b.index)
    for col, bins in (("altitude", [0, 50, 80, 130, 200]), ("wind_speed", [0, 6, 10, 14, 16]),
                      ("sun_elevation", [-90, 5, 15, 35, 90])):
        for k, g in b.groupby(pd.cut(runs[col], bins), observed=True):
            rows.append({"set": "test-iid", "slice": col, "bin": str(k), "runs": len(g), "b_median": g.median()})
    c = pd.read_csv(EVAL / "flowhead" / "e1-core" / "e1_compare.csv")
    rows.append({"set": "e1-core", "slice": "all", "bin": "frac b < e1 searaft g1", "runs": len(c),
                 "b_median": float((c.b < c.e1_flow_searaft_g1).mean())})
    for a, g in c.groupby("altitude"):
        rows.append({"set": "e1-core", "slice": "altitude", "bin": f"{a} flowhead / e1 searaft medians",
                     "runs": len(g), "b_median": g.b.median(), "e1_searaft_median": g.e1_flow_searaft_g1.median(),
                     "frac_better": float((g.b < g.e1_flow_searaft_g1).mean())})
    pd.DataFrame(rows).to_csv(out / "slices_recomputed.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(V / "checks"))
    ap.add_argument("--only", default="leak_sets,leak_inputs,rerun,headline,e1,selection,calib,slices_check")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only.split(","):
        print(f"==== {name}", flush=True)
        globals()[name](out)


if __name__ == "__main__":
    main()
