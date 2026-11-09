"""Independent verification of model A ('e2e', 'e2e-K2', 'e2e-K4'). Read-only on model A's
outputs; writes only under --out (default /workspace/data/e2/verify_e2e).

    cd /workspace/mare-realtime/experiments/e2_learned_velocity
    OMP_NUM_THREADS=4 /workspace/venv/bin/python -m verify.verify_e2e static
    /workspace/venv/bin/python -m verify.verify_e2e predict      # re-infer from frozen checkpoints (GPU, <= 8 GB)
    /workspace/venv/bin/python -m verify.verify_e2e window       # window-level forward + equivariance probe (GPU)
    /workspace/venv/bin/python -m verify.verify_e2e harness      # e2.evaluate on the re-inferred predictions

`static` recomputes every headline number from the prediction CSVs with code written here
(not e2.evaluate, not e2e.collect / analyse / calib) and compares with model A's tables.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

SETS = Path("/workspace/data/e2/sets")
PREDS = Path("/workspace/data/e2/preds")
EVAL = Path("/workspace/data/e2/eval")
CKPT = Path("/workspace/data/e2/e2e")
RUNS = {"e2e": "main", "e2e-K2": "abl-K2", "e2e-K4": "abl-K4"}
ALL_SETS = ["val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]
PY = "/workspace/venv/bin/python"
# re-inference plan: (method, set)
REDO = [("e2e", "test-iid"), ("e2e", "e1-core"), ("e2e", "test-alt"),
        ("e2e-K2", "test-iid"), ("e2e-K2", "test-alt"), ("e2e-K4", "test-iid"), ("e2e-K4", "test-alt")]


def save(out: Path, name: str, df: pd.DataFrame):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    print(f"== {name}\n{df.to_string(index=False, max_rows=120)}", flush=True)


def rot(th_deg):
    t = np.radians(th_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# ---- independent per-run metrics ----------------------------------------------------------


def my_truth(pred: pd.DataFrame, s: str) -> pd.DataFrame:
    """v_true recomputed here from labels.csv and theta (tile axes from ENU), per row."""
    lb = pd.read_csv(SETS / s / "labels.csv")
    runs = pd.read_csv(SETS / s / "runs.csv").set_index("run_id")
    out = []
    for rid, g in pred.groupby("run_id", sort=False):
        l = lb[lb.run_id == rid].sort_values("frame")
        t, pe = l["t"].to_numpy(), l[["cam_e", "cam_n"]].to_numpy()
        w = g["w_start"].to_numpy(int)
        e = w + np.rint(g["T"].to_numpy() * float(runs.at[rid, "fps"])).astype(int)
        ve = (pe[e] - pe[w]) / (t[e] - t[w])[:, None]
        vx = ve @ rot(float(runs.at[rid, "theta_deg"]))     # ENU -> tile: x = v·x̂, y = v·ŷ
        g = g.copy()
        g["my_vx"], g["my_vy"], g["my_ve"], g["my_vn"] = vx[:, 0], vx[:, 1], ve[:, 0], ve[:, 1]
        out.append(g)
    return pd.concat(out)


def my_per_run(pred: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in pred.groupby("run_id", sort=False):
        m = runs.loc[rid]
        th = float(m["theta_deg"])
        ex = g[["v_est_x", "v_est_y"]].to_numpy() - g[["my_vx", "my_vy"]].to_numpy()
        b = ex.mean(0)
        be = rot(th) @ b
        vt = g[["my_vx", "my_vy"]].to_numpy()
        spd = np.linalg.norm(vt, axis=1).mean()
        wd = np.radians(float(m["wind_direction"]))
        uw = np.array([np.sin(wd), np.cos(wd)])
        vte = g[["my_ve", "my_vn"]].to_numpy().mean(0)
        ut = vte / max(np.linalg.norm(vte), 1e-9)
        # drift over non-overlapping windows: greedy, next window starts at the previous end
        w = g["w_start"].to_numpy(int)
        span = np.rint(g["T"].to_numpy() * float(m["fps"])).astype(int)
        order = np.argsort(w)
        sel, nxt = [], -1
        for j in order:
            if w[j] >= nxt:
                sel.append(j)
                nxt = w[j] + span[j]
        sel = np.array(sel)
        T = g["T"].to_numpy()[sel]
        dist = (np.linalg.norm(vt[sel], axis=1) * T).sum()
        drift = 100 * np.linalg.norm((ex[sel] * T[:, None]).sum(0)) / dist
        rows.append({"run_id": rid, "b": np.hypot(*b), "b_pct": 100 * np.hypot(*b) / spd, "drift_pct": drift,
                     "along_wind": be @ uw, "cross_wind": be @ np.array([uw[1], -uw[0]]),
                     "along_track": be @ ut, "speed": spd, "sea_rate": m.get("sea_rate", np.nan)})
    return pd.DataFrame(rows)


def static(out: Path):
    # 1. checkpoints vs logs
    rows = []
    for meth, d in RUNS.items():
        import torch
        for f in ("best.pt", "last.pt"):
            c = torch.load(CKPT / d / f, map_location="cpu", weights_only=False)
            lg = pd.read_csv(CKPT / d / "log.csv")
            v = lg.dropna(subset=["val_b_median"])
            rows.append({"method": meth, "file": f, "step": c["step"], "K": c["args"]["K"], "sets_arg": c["args"]["sets"],
                         "val_on_train": c["args"]["val_on_train"], "train_runs": c["args"]["train_runs"],
                         "ckpt_val_b_median": c.get("val", {}).get("val_b_median", np.nan),
                         "log_argmin_step": int(v.loc[v.val_b_median.idxmin(), "step"]),
                         "log_min_val_b": float(v.val_b_median.min()), "log_last_step": int(lg.step.iloc[-1]),
                         "log_last_hours": float(lg.hours.iloc[-1]), "log_peak_gpu_gb": float(lg.gpu_mem_gb.max()),
                         "val_curve": " ".join(f"{s}:{b:.3f}" for s, b in zip(v.step, v.val_b_median)),
                         "n_params": sum(int(np.prod(p.shape)) for p in c["model"].values())})
    save(out, "ckpt_vs_log", pd.DataFrame(rows))

    # 2. disjointness of sets and theta uniformity
    from scipy import stats
    allr = {s: pd.read_csv(SETS / s / "runs.csv") for s in ["train"] + ALL_SETS}
    rows = []
    names = list(allr)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = allr[a], allr[b]
            pa = set(zip(ra.traj_seed, ra.sea_seed))
            pb = set(zip(rb.traj_seed, rb.sea_seed))
            rows.append({"a": a, "b": b, "run_id_overlap": len(set(ra.run_id) & set(rb.run_id)),
                         "seed_pair_overlap": len(pa & pb), "sea_seed_overlap": len(set(ra.sea_seed) & set(rb.sea_seed)),
                         "traj_seed_overlap": len(set(ra.traj_seed) & set(rb.traj_seed))})
    save(out, "set_overlap", pd.DataFrame(rows))
    rows = []
    for s, r in allr.items():
        th = r.theta_deg.to_numpy()
        rows.append({"set": s, "runs": len(r), "theta_min": th.min(), "theta_max": th.max(),
                     "ks_p_uniform_0_360": stats.kstest(th / 360, "uniform").pvalue,
                     "corr_theta_vs_wind_dir": np.corrcoef(th, r.wind_direction)[0, 1],
                     "corr_theta_vs_start_heading": np.corrcoef(th, r.start_heading)[0, 1],
                     "c_run_constant_per_run": r.groupby("run_id")[["c_run_e", "c_run_n"]].nunique().max().max() == 1})
    save(out, "theta", pd.DataFrame(rows))

    # 3. independent per-run metrics from every prediction CSV
    summ, sea, prs = [], [], {}
    for meth in RUNS:
        for s in ALL_SETS:
            p = pd.read_csv(PREDS / meth / f"{s}.csv")
            runs = pd.read_csv(SETS / s / "runs.csv").set_index("run_id", drop=False)
            n_expect = len(runs) * (int(runs.n_frames.iloc[0]) - (8 if meth == "e2e" else int(meth[-1])) + 1)
            p = my_truth(p, s)
            dv = np.abs(p[["my_vx", "my_vy"]].to_numpy() - p[["v_true_x", "v_true_y"]].to_numpy()).max()
            # ENU consistency of v_est with theta
            ve = np.stack([rot(runs.at[r, "theta_deg"]) @ v for r, v in zip(p.run_id, p[["v_est_x", "v_est_y"]].to_numpy())])
            de = np.abs(ve - p[["v_est_e", "v_est_n"]].to_numpy()).max()
            pr = my_per_run(p, runs)
            prs[(meth, s)] = pr
            summ.append({"method": meth, "set": s, "rows": len(p), "rows_expected_all_windows": n_expect,
                         "runs": len(pr), "max_dv_true": dv, "max_d_enu_est": de,
                         "b_median": pr.b.median(), "b_p90": pr.b.quantile(0.9), "b_pct_median": pr.b_pct.median(),
                         "drift_pct_median": pr.drift_pct.median(), "frac_b_le_017": (pr.b <= 0.17).mean(),
                         "frac_bpct_le_028": (pr.b_pct <= 0.28).mean(), "frac_bpct_le_05": (pr.b_pct <= 0.5).mean(),
                         "sigma_nan": int(p[["sigma_x", "sigma_y"]].isna().sum().sum())})
            if s == "test-rate":
                for r, g in pr.groupby("sea_rate"):
                    sea.append({"method": meth, "sea_rate": r, "runs": len(g), "b_median": g.b.median(),
                                "frac_b_le_017": (g.b <= 0.17).mean(), "along_wind_median": g.along_wind.median(),
                                "cross_wind_median": g.cross_wind.median(), "along_track_median": g.along_track.median(),
                                "abs_along_wind_median": g.along_wind.abs().median(),
                                "abs_cross_wind_median": g.cross_wind.abs().median()})
    mine = pd.DataFrame(summ)
    rep = pd.read_csv(HERE / "results/e2e/summary_by_set.csv")
    rep = rep[rep.method.isin(RUNS)]
    cmp = mine.merge(rep, on=["method", "set"], suffixes=("", "_reported"))
    for c in ["b_median", "b_p90", "b_pct_median", "drift_pct_median", "frac_b_le_017", "frac_bpct_le_028", "frac_bpct_le_05"]:
        cmp[f"d_{c}"] = cmp[c] - cmp[f"{c}_reported"]
    save(out, "recomputed_by_set", cmp[[c for c in cmp if not c.endswith("_reported") or c.startswith("b_median")] ])

    # phase-correlation reference from the baselines' own CSVs, recomputed
    pc = []
    for s in ALL_SETS:
        f = PREDS / "b0-phasecorr" / f"{s}.csv"
        if f.exists():
            p = pd.read_csv(f)
            p = p[p.K == 2]
            runs = pd.read_csv(SETS / s / "runs.csv").set_index("run_id", drop=False)
            pr = my_per_run(my_truth(p, s), runs)
            prs[("phasecorr", s)] = pr
            pc.append({"set": s, "rows": len(p), "runs": len(pr), "b_median": pr.b.median()})
            if s == "test-rate":
                for r, g in pr.groupby("sea_rate"):
                    sea.append({"method": "phasecorr", "sea_rate": r, "runs": len(g), "b_median": g.b.median(),
                                "frac_b_le_017": (g.b <= 0.17).mean(), "along_wind_median": g.along_wind.median(),
                                "cross_wind_median": g.cross_wind.median(), "along_track_median": g.along_track.median(),
                                "abs_along_wind_median": g.along_wind.abs().median(),
                                "abs_cross_wind_median": g.cross_wind.abs().median()})
    save(out, "phasecorr_recomputed", pd.DataFrame(pc))
    save(out, "sea_rate_recomputed", pd.DataFrame(sea))

    # paired per-condition correction: model along-wind minus phase-correlation along-wind, same run
    if ("phasecorr", "test-rate") in prs:
        rows = []
        for meth in RUNS:
            a = prs[(meth, "test-rate")].merge(prs[("phasecorr", "test-rate")], on="run_id", suffixes=("", "_pc"))
            a["corr_along"] = a.along_wind - a.along_wind_pc
            for r, g in a.groupby("sea_rate"):
                rows.append({"method": meth, "sea_rate": r, "runs": len(g), "correction_along_median": g.corr_along.median(),
                             "correction_along_p10": g.corr_along.quantile(0.1),
                             "correction_along_p90": g.corr_along.quantile(0.9)})
        save(out, "sea_rate_paired_correction", pd.DataFrame(rows))

    # 4. calibration and iterations, recomputed
    cal, its = [], []
    for meth in RUNS:
        for s in ALL_SETS:
            p = pd.read_csv(PREDS / meth / f"{s}.csv")
            e = p[["v_est_x", "v_est_y"]].to_numpy() - p[["v_true_x", "v_true_y"]].to_numpy()
            sg = p[["sigma_x", "sigma_y"]].to_numpy()
            z = np.abs(e) / sg
            cal.append({"method": meth, "set": s, "cov1_axis": (z <= 1).mean(), "cov2_axis": (z <= 2).mean(),
                        "sigma_median_2d": np.median(np.hypot(*sg.T)), "err_median_2d": np.median(np.hypot(*e.T))})
            n = np.load(PREDS / meth / f"{s}.npz", allow_pickle=True)
            assert np.allclose(n["v"], p[["v_est_x", "v_est_y"]].to_numpy(), atol=1e-5)
            row = {"method": meth, "set": s}
            for name, v in [("init", n["v0"])] + [(f"it{i + 1}", n["vs"][:, i]) for i in range(n["vs"].shape[1])]:
                d = pd.DataFrame({"r": n["run_id"], "x": v[:, 0] - n["v_true"][:, 0], "y": v[:, 1] - n["v_true"][:, 1]})
                gm = d.groupby("r")[["x", "y"]].mean()
                row[f"b_median_{name}"] = float(np.hypot(gm.x, gm.y).median())
            its.append(row)
    save(out, "calibration_recomputed", pd.DataFrame(cal))
    save(out, "iterations_recomputed", pd.DataFrame(its))

    # 5. e1 comparison medians
    rows = []
    for meth in RUNS:
        c = pd.read_csv(EVAL / meth / "e1-core" / "e1_compare.csv")
        mine_b = prs[(meth, "e1-core")].set_index("run_id").b
        jj = c.set_index("run_id").join(mine_b.rename("b_mine"))
        rows.append({"method": meth, "runs": len(c), "max_d_b_vs_mine": float((jj.b - jj.b_mine).abs().max()),
                     **{f"median_{x}": c[x].median() for x in c if x == "b" or x.startswith("e1_")},
                     **{f"frac017_{x}": (c[x] <= 0.17).mean() for x in c if x == "b" or x.startswith("e1_")}})
    save(out, "e1_compare_recomputed", pd.DataFrame(rows))

    # 6. slices quoted in NOTES (test-iid), recomputed: ground speed, wind, swell, sun
    p = my_truth(pd.read_csv(PREDS / "e2e" / "test-iid.csv"), "test-iid")
    runs = pd.read_csv(SETS / "test-iid" / "runs.csv").set_index("run_id", drop=False)
    pr = my_per_run(p, runs).set_index("run_id").join(runs[["wind_speed", "swell", "sun_elevation", "altitude"]])
    rows = []
    for name, col, bins in [("ground_speed", "speed", [0, 20, 35, 50, 65, 200]), ("wind", "wind_speed", [0, 6, 10, 14, 16, 30]),
                            ("sun", "sun_elevation", [-90, 5, 15, 35, 90]), ("swell", "swell", [-0.5, 0.5, 1.5])]:
        for k, g in pr.groupby(pd.cut(pr[col], bins), observed=True):
            rows.append({"slice": name, "bin": str(k), "runs": len(g), "b_median": g.b.median(), "b_pct_median": g.b_pct.median()})
    save(out, "test_iid_slices_recomputed", pd.DataFrame(rows))


# ---- re-inference --------------------------------------------------------------------------


def predict(out: Path):
    import torch
    from e2.data import PRED_COLUMNS, TileSet, prediction_frame
    from e2e.predict import load, predict_set
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(8 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    rows = []
    for meth, s in REDO:
        model, ck = load(str(CKPT / RUNS[meth] / "best.pt"), dev)
        ts = TileSet(SETS / s)
        idx, p = predict_set(model, ts, ck["args"]["K"], dev)
        keys = list(zip(idx["set"], idx["run_id"], idx["w_start"].astype(int), idx["K"].astype(int), idx["T"]))
        df = prediction_frame(ts, keys, p["v"], p["sigma"])[PRED_COLUMNS]
        f = out / "preds" / meth / f"{s}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, index=False)
        orig = pd.read_csv(PREDS / meth / f"{s}.csv")
        j = df.merge(orig, on=["run_id", "w_start"], suffixes=("", "_o"))
        dv = np.abs(j[["v_est_x", "v_est_y"]].to_numpy() - j[["v_est_x_o", "v_est_y_o"]].to_numpy())
        ds = np.abs(j[["sigma_x", "sigma_y"]].to_numpy() - j[["sigma_x_o", "sigma_y_o"]].to_numpy())
        rows.append({"method": meth, "set": s, "ckpt_step": ck["step"], "rows_new": len(df), "rows_orig": len(orig),
                     "rows_joined": len(j), "max_abs_dv": dv.max(), "p99_abs_dv": np.quantile(dv, 0.99),
                     "max_abs_dsigma": ds.max()})
        print(rows[-1], flush=True)
        del model
        torch.cuda.empty_cache()
    save(out, "repredict_vs_orig", pd.DataFrame(rows))


def harness(out: Path):
    rows = []
    for meth, s in REDO:
        d = out / "eval" / meth / s
        subprocess.run([PY, "-m", "e2.evaluate", "--pred", str(out / "preds" / meth / f"{s}.csv"), "--sets", str(SETS),
                        "--out", str(d), "--name", meth, "--require-all"], check=True, cwd=HERE,
                       stdout=subprocess.DEVNULL)
        a = pd.read_csv(d / "slices.csv")
        a = a[a.slice == "all"].iloc[0]
        r = pd.read_csv(EVAL / meth / s / "slices.csv")
        r = r[r.slice == "all"].iloc[0]
        rows.append({"method": meth, "set": s, **{f"{c}": a[c] for c in ["runs", "b_median", "b_p90", "frac_b_le_017", "b_pct_median"]},
                     **{f"{c}_reported": r[c] for c in ["b_median", "b_p90", "frac_b_le_017", "b_pct_median"]}})
    save(out, "harness_rerun_vs_reported", pd.DataFrame(rows))


def window(out: Path):
    """Window-by-window forward (not forward_run) through e2e.loader.Windows on a deterministic
    subset, against the stored predictions; and a dihedral-equivariance probe."""
    import torch
    from e2.data import TileSet, augment_array, unaugment_vector
    from e2e.loader import Windows
    from e2e.predict import load
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(8 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    model, ck = load(str(CKPT / "main" / "best.pt"), dev)
    rows = []
    for s in ["test-iid", "e1-core"]:
        ts = TileSet(SETS / s)
        ds = Windows(ts, 8, step=1, augment=False)
        sel = np.where(np.isin(ds.w_start, [0, 37, 73]) & np.isin(ds.run_id, ts.run_ids[::5]))[0]
        orig = pd.read_csv(PREDS / "e2e" / f"{s}.csv").set_index(["run_id", "w_start"])
        for i0 in range(0, len(sel), 16):
            items = [ds[int(i)] for i in sel[i0:i0 + 16]]
            assert set(items[0]) == {"tiles", "gsd", "dt", "target", "aug", "i"}
            x = torch.from_numpy(np.stack([it["tiles"] for it in items])).to(dev)
            gsd = torch.tensor([it["gsd"] for it in items], device=dev)
            dt = torch.tensor([it["dt"] for it in items], device=dev)
            with torch.no_grad():
                v = model(x, gsd, dt)["v"].float().cpu().numpy()
                # equivariance: rotate/flip the stack, un-rotate the answer
                g = 3
                xa = torch.from_numpy(augment_array(np.stack([it["tiles"] for it in items]), g)).to(dev)
                va = unaugment_vector(model(xa, gsd, dt)["v"].float().cpu().numpy(), g)
            for k, it in enumerate(items):
                key = (ds.run_id[it["i"]], int(ds.w_start[it["i"]]))
                o = orig.loc[key]
                rows.append({"set": s, "run_id": key[0], "w_start": key[1], "vx": v[k, 0], "vy": v[k, 1],
                             "vx_orig": o.v_est_x, "vy_orig": o.v_est_y, "vx_aug3": va[k, 0], "vy_aug3": va[k, 1],
                             "tx": it["target"][0], "ty": it["target"][1]})
    df = pd.DataFrame(rows)
    df["d_orig"] = np.hypot(df.vx - df.vx_orig, df.vy - df.vy_orig)
    df["d_aug"] = np.hypot(df.vx - df.vx_aug3, df.vy - df.vy_aug3)
    df["err"] = np.hypot(df.vx - df.tx, df.vy - df.ty)
    df.to_csv(out / "window_forward.csv", index=False)
    s = df.groupby("set").agg(n=("d_orig", "size"), d_orig_max=("d_orig", "max"), d_orig_median=("d_orig", "median"),
                              d_aug_median=("d_aug", "median"), d_aug_p90=("d_aug", lambda x: x.quantile(0.9)),
                              err_median=("err", "median")).reset_index()
    save(out, "window_forward_summary", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["static", "predict", "harness", "window"])
    ap.add_argument("--out", default="/workspace/data/e2/verify_e2e")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    {"static": static, "predict": predict, "harness": harness, "window": window}[a.what](out)


if __name__ == "__main__":
    main()
