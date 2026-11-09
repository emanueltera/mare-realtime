"""Independent verification of model C ('spectral', its K=16 and no-registration ablations).

Read-only on the spectral agent's outputs; writes only under --out
(default /workspace/data/e2/verify/spectral/checks).

    cd /workspace/mare-realtime/experiments/e2_learned_velocity
    OMP_NUM_THREADS=1 /workspace/venv/bin/python -m verify.verify_spectral

Metrics are recomputed here from the prediction CSVs with v_true recomputed from labels.csv
(own code, not e2.evaluate). Re-inferred predictions (made with spectral.predict from the frozen
checkpoint into /workspace/data/e2/verify/spectral/preds) are compared row by row with the
agent's predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

SETS = Path("/workspace/data/e2/sets")
PREDS = Path("/workspace/data/e2/preds")
EVAL = Path("/workspace/data/e2/eval")
SPEC = Path("/workspace/data/e2/spectral")
VER = Path("/workspace/data/e2/verify/spectral")
RES = HERE / "results"
ALLSETS = ["train", "val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]
EVALSETS = ["val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]
METHODS = ["spectral", "spectral-k16", "spectral-noreg"]

OUT: Path


def emit(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(OUT / f"{name}.csv", index=False)
        print(f"== {name}\n{obj.to_string(index=False, max_rows=120)}", flush=True)
    else:
        (OUT / f"{name}.txt").write_text(str(obj))
        print(f"== {name}\n{obj}", flush=True)


_runs, _labels = {}, {}


def runs(s):
    if s not in _runs:
        _runs[s] = pd.read_csv(SETS / s / "runs.csv")
    return _runs[s]


def labels(s):
    if s not in _labels:
        _labels[s] = {rid: g.sort_values("frame") for rid, g in pd.read_csv(SETS / s / "labels.csv").groupby("run_id")}
    return _labels[s]


# ---------------------------------------------------------------- integrity / leak checks
def check_disjoint():
    rows, ids, seeds = [], {}, {}
    for s in ALLSETS:
        r = runs(s)
        ids[s] = set(r.run_id)
        seeds[s] = set(zip(r.traj_seed, r.sea_seed)) if {"traj_seed", "sea_seed"} <= set(r) else set()
    for i, a in enumerate(ALLSETS):
        for b in ALLSETS[i + 1:]:
            rows.append({"a": a, "b": b, "shared_run_ids": len(ids[a] & ids[b]),
                         "shared_traj_sea_seed_pairs": len(seeds[a] & seeds[b]),
                         "shared_sea_seeds": len({x[1] for x in seeds[a]} & {x[1] for x in seeds[b]}),
                         "shared_traj_seeds": len({x[0] for x in seeds[a]} & {x[0] for x in seeds[b]})})
    emit("disjoint_sets", pd.DataFrame(rows))


def check_theta():
    rows = []
    for s in ALLSETS:
        th = runs(s).theta_deg.to_numpy() % 360
        if s == "test-rate":  # paired conditions share θ by design: one per condition
            th = runs(s).drop_duplicates(subset=["traj_seed", "sea_seed"]).theta_deg.to_numpy() % 360
        ks = stats.kstest(th / 360.0, "uniform")
        rows.append({"set": s, "n": len(th), "min": th.min(), "max": th.max(), "ks_p_uniform": ks.pvalue,
                     "c_run_constant_per_run": True})
    emit("theta_uniform", pd.DataFrame(rows))


def pattern_path(s):
    from spectral.pattern import pair_csv
    return pair_csv(s)


def check_pattern_sources(n_pairs=30, seed=0):
    """Every set's pair CSV covers every run and pair; a sample of pairs is recomputed from
    the tiles by e2.baselines.velocity_pairs (so the pattern is tile-derived)."""
    from e2.baselines import phase_correlation
    from e2.data import TileSet

    rng = np.random.default_rng(seed)
    rows, rec = [], []
    for s in ALLSETS:
        p = pattern_path(s)
        df = pd.read_csv(p, usecols=["run_id", "w_start", "K", "T", "v_est_x", "v_est_y"])
        r = runs(s)
        nf = dict(zip(r.run_id, r.n_frames))
        cov = df.groupby("run_id").w_start.agg(["min", "max", "count"])
        ok = all(rid in cov.index and cov.at[rid, "count"] == nf[rid] - 1 and cov.at[rid, "min"] == 0
                 for rid in nf)
        rows.append({"set": s, "file": str(p), "K_values": ",".join(map(str, sorted(df.K.unique()))),
                     "runs_in_set": len(nf), "runs_in_csv": df.run_id.nunique(), "all_pairs_present": ok,
                     "nan_pairs": int(df[["v_est_x", "v_est_y"]].isna().any(axis=1).sum())})
        ts = TileSet(SETS / s)
        sub = df.sample(n=n_pairs, random_state=int(rng.integers(1 << 30)))
        for _, q in sub.iterrows():
            a = np.asarray(ts.tiles(q.run_id)[int(q.w_start)], dtype=np.float64)
            b = np.asarray(ts.tiles(q.run_id)[int(q.w_start) + 1], dtype=np.float64)
            d, _ = phase_correlation(a, b, 50)
            v = -d * ts.gsd(q.run_id) * ts.fps(q.run_id)
            rec.append({"set": s, "run_id": q.run_id, "w_start": int(q.w_start), "csv_x": q.v_est_x, "csv_y": q.v_est_y,
                        "recomputed_x": v[0], "recomputed_y": v[1]})
    emit("pattern_sources", pd.DataFrame(rows))
    rec = pd.DataFrame(rec)
    rec["absdiff"] = np.hypot(rec.csv_x - rec.recomputed_x, rec.csv_y - rec.recomputed_y)
    emit("pattern_recomputed_sample_summary",
         rec.groupby("set").absdiff.agg(["count", "max", "median"]).reset_index())
    rec.to_csv(OUT / "pattern_recomputed_sample.csv", index=False)


def check_diag_centre():
    """The centre used at inference (diag vpat) equals the median of the window's pair velocities."""
    rows = []
    for m in METHODS:
        for s in EVALSETS:
            d = pd.read_csv(PREDS / m / f"{s}.diag.csv")
            if m == "spectral-noreg":
                rows.append({"method": m, "set": s, "n": len(d), "centre_all_zero": bool((d[["vpat_x", "vpat_y"]] == 0).all().all()),
                             "max_absdiff_vs_recomputed": np.nan, "inside_stage0": d.inside_stage0.mean(),
                             "pat_missing": d.pat_missing.mean()})
                continue
            pr = pd.read_csv(pattern_path(s), usecols=["run_id", "w_start", "K", "v_est_x", "v_est_y"])
            pr = pr[pr.K == 2]
            pairs = {rid: g.sort_values("w_start")[["v_est_x", "v_est_y"]].to_numpy() for rid, g in pr.groupby("run_id")}
            vp = np.array([np.nanmedian(pairs[r][w:w + K - 1], axis=0) for r, w, K in zip(d.run_id, d.w_start, d.K)])
            rows.append({"method": m, "set": s, "n": len(d), "centre_all_zero": False,
                         "max_absdiff_vs_recomputed": float(np.abs(vp - d[["vpat_x", "vpat_y"]].to_numpy()).max()),
                         "inside_stage0": d.inside_stage0.mean(), "pat_missing": d.pat_missing.mean()})
    emit("diag_centre", pd.DataFrame(rows))


# ---------------------------------------------------------------- independent metrics
def truth(s, pred):
    """v_true recomputed from labels (window end = w_start + round(T·fps))."""
    lb = labels(s)
    fps = dict(zip(runs(s).run_id, runs(s).fps))
    vt = np.zeros((len(pred), 2))
    T = np.zeros(len(pred))
    for rid, idx in pred.groupby("run_id").indices.items():
        g = lb[rid]
        t = g.t.to_numpy()
        px = g[["cam_x", "cam_y"]].to_numpy()
        w = pred.w_start.to_numpy()[idx].astype(int)
        e = w + np.rint(pred["T"].to_numpy()[idx] * fps[rid]).astype(int)
        T[idx] = t[e] - t[w]
        vt[idx] = (px[e] - px[w]) / T[idx, None]
    return vt, T


def per_run_metrics(s, pred):
    vt, T = truth(s, pred)
    err_truth = np.abs(vt - pred[["v_true_x", "v_true_y"]].to_numpy()).max()
    ex = pred[["v_est_x", "v_est_y"]].to_numpy() - vt
    rows = []
    for rid, idx in pred.groupby("run_id").indices.items():
        e = ex[idx]
        sp = np.linalg.norm(vt[idx], axis=1).mean()
        b = np.linalg.norm(e.mean(0))
        # drift: greedy non-overlapping windows (each starts at or after the previous one's end frame)
        w = pred.w_start.to_numpy()[idx]
        span = np.rint(pred["T"].to_numpy()[idx] * 10).astype(int)
        o = np.argsort(w, kind="stable")
        sel, end = [], -1
        for i in o:
            if w[i] >= end:
                sel.append(i)
                end = w[i] + span[i]
        sel = np.array(sel)
        dist = (np.linalg.norm(vt[idx][sel], axis=1) * T[idx][sel]).sum()
        drift = 100 * np.linalg.norm((e[sel] * T[idx][sel, None]).sum(0)) / dist
        rows.append({"run_id": rid, "b": b, "b_pct": 100 * b / sp, "speed": sp, "rms": np.sqrt((e ** 2).sum(1).mean()),
                     "drift_pct": drift, "n": len(idx)})
    return pd.DataFrame(rows), err_truth


def summarise(pr):
    return {"runs": len(pr), "b_median": pr.b.median(), "b_p90": pr.b.quantile(0.9), "b_pct_median": pr.b_pct.median(),
            "frac_b_le_017": (pr.b <= 0.17).mean(), "frac_bpct_le_028": (pr.b_pct <= 0.28).mean(),
            "frac_bpct_le_05": (pr.b_pct <= 0.5).mean(), "drift_pct_median": pr.drift_pct.median()}


def check_headlines():
    rep = pd.read_csv(SPEC / "report" / "summary_all.csv")
    rows, per = [], {}
    for m in METHODS:
        for s in EVALSETS:
            pred = pd.read_csv(PREDS / m / f"{s}.csv")
            r = runs(s)
            K = int(pred.K.iloc[0])
            n_exp = int(((r.n_frames - K + 1).clip(lower=0)).sum())
            pr, et = per_run_metrics(s, pred)
            per[(m, s)] = pr
            mine = summarise(pr)
            rr = rep[(rep.method == m) & (rep.set == s)]
            row = {"method": m, "set": s, "K": K, "windows": len(pred), "windows_expected_all_starts": n_exp,
                   "dup_windows": int(pred.duplicated(["run_id", "w_start"]).sum()),
                   "max_vtrue_diff_vs_labels": et, **{f"mine_{k}": v for k, v in mine.items()}}
            if len(rr):
                rr = rr.iloc[0]
                for k in ("b_median", "b_p90", "b_pct_median", "frac_b_le_017", "frac_bpct_le_028", "frac_bpct_le_05",
                          "drift_pct_median"):
                    row[f"rep_{k}"] = rr[k]
                row["max_absdiff"] = max(abs(mine[k] - rr[k]) for k in ("b_median", "b_p90", "b_pct_median",
                                                                         "frac_b_le_017", "frac_bpct_le_028",
                                                                         "frac_bpct_le_05", "drift_pct_median"))
            rows.append(row)
    emit("headlines_recomputed", pd.DataFrame(rows))
    return per


def check_references():
    rows = []
    for name, path, s in [("spectral-classical", PREDS / "spectral-classical" / "val.csv", "val"),
                          ("spectral-tta", PREDS / "spectral-tta" / "val.csv", "val"),
                          ("phasecorr", EVAL / "phasecorr" / "val.csv", "val"),
                          ("phasecorr", EVAL / "phasecorr" / "e1-core.csv", "e1-core")]:
        pred = pd.read_csv(path)
        pred = pred[pred.set == s]
        pr, et = per_run_metrics(s, pred)
        rows.append({"method": name, "set": s, "windows": len(pred), "K": int(pred.K.iloc[0]), **summarise(pr)})
    emit("references_recomputed", pd.DataFrame(rows))


def slice_tables(per):
    rows = []
    edges = {"sun_elevation": [-90, 5, 15, 35, 90], "altitude": [0, 50, 80, 130, 200, 260, 330, 400.01]}
    for m in METHODS:
        for s, col in [("test-iid", "sun_elevation"), ("val", "sun_elevation"), ("test-alt", "altitude"),
                       ("test-iid", "altitude"), ("test-rate", "sea_rate")]:
            pr = per[(m, s)].merge(runs(s)[["run_id", col]], on="run_id")
            key = pd.cut(pr[col], edges[col]).astype(str) if col in edges else pr[col]
            for k, g in pr.groupby(key):
                rows.append({"method": m, "set": s, "slice": col, "bin": k, "runs": len(g), "b_median": g.b.median(),
                             "b_p90": g.b.quantile(0.9)})
    # phase correlation by sea rate (its own predictions, K = 2)
    pred = pd.read_csv(EVAL / "phasecorr" / "test-rate.csv")
    pr, _ = per_run_metrics("test-rate", pred)
    pr = pr.merge(runs("test-rate")[["run_id", "sea_rate"]], on="run_id")
    for k, g in pr.groupby("sea_rate"):
        rows.append({"method": "phasecorr", "set": "test-rate", "slice": "sea_rate", "bin": k, "runs": len(g),
                     "b_median": g.b.median(), "b_p90": g.b.quantile(0.9)})
    emit("slices_recomputed", pd.DataFrame(rows))


def e1_compare(per):
    e1 = pd.read_csv(HERE.parent / "e1_wave_bias" / "results" / "per_run.csv")
    e1 = e1[e1.set == "core"]
    r = runs("e1-core")[["run_id", "sun_elevation", "altitude"]]
    rows = []
    for m in METHODS:
        pr = per[(m, "e1-core")].merge(r, on="run_id")
        pr["sun"] = np.where(pr.sun_elevation >= 35, "midday", "low sun")
        for (mm, p) in [("shell:dispersive", "T=3.2"), ("shell:dispersive", "T=6.4"), ("flow:searaft", "g=1")]:
            sub = e1[(e1.method == mm) & (e1.param == p)].set_index("run_id").b_abs
            pr[f"{mm}_{p}"] = pr.run_id.map(sub)
        for sun, g in pr.groupby("sun"):
            rows.append({"method": m, "sun": sun, "runs": len(g), "sun_values": ",".join(map(str, sorted(g.sun_elevation.unique()))),
                         "b_median": g.b.median(), "frac_b_le_017": (g.b <= 0.17).mean(),
                         "e1_shell_T3.2": g["shell:dispersive_T=3.2"].median(),
                         "e1_shell_T6.4": g["shell:dispersive_T=6.4"].median(),
                         "e1_searaft_g1": g["flow:searaft_g=1"].median(),
                         "e1_missing": int(g["shell:dispersive_T=3.2"].isna().sum())})
        for (sun, alt), g in pr.groupby(["sun", "altitude"]):
            rows.append({"method": m, "sun": sun, "altitude": alt, "runs": len(g), "b_median": g.b.median()})
    emit("e1_compare_recomputed", pd.DataFrame(rows))


def calibration():
    rows = []
    for m in METHODS:
        for s in EVALSETS:
            p = pd.read_csv(PREDS / m / f"{s}.csv")
            vt, _ = truth(s, p)
            e = np.abs(p[["v_est_x", "v_est_y"]].to_numpy() - vt)
            sg = p[["sigma_x", "sigma_y"]].to_numpy()
            z = e / sg
            rows.append({"method": m, "set": s, "within1": (z <= 1).mean(), "within2": (z <= 2).mean()})
            if s == "test-rate":
                rate = p.run_id.map(dict(zip(runs(s).run_id, runs(s).sea_rate))).to_numpy()
                for r in sorted(set(rate)):
                    zz = z[rate == r]
                    rows.append({"method": m, "set": f"test-rate sea_rate={r}", "within1": (zz <= 1).mean(),
                                 "within2": (zz <= 2).mean()})
            if s == "val" and m == "spectral":
                sm = sg.ravel()
                em = e.ravel()
                q = pd.qcut(sm, 10, labels=False)
                dec = pd.DataFrame({"d": q, "s": sm, "e": em}).groupby("d").agg(sigma_mean=("s", "mean"),
                                                                                 rms=("e", lambda x: np.sqrt(np.mean(x ** 2))))
                emit("val_sigma_deciles", dec.reset_index())
    emit("calibration_recomputed", pd.DataFrame(rows))


def training_curves():
    import torch

    rows = []
    for run in ["k32", "k16", "k32-noreg", "smoke"]:
        c = pd.read_csv(SPEC / run / "curve.csv")
        ck = torch.load(SPEC / run / "best.pt", map_location="cpu", weights_only=False)
        nparam = sum(v.numel() for v in ck["model"].values())
        col = "val_b_median"
        rows.append({"run": run, "epochs": len(c), "minutes_last": c.minutes.iloc[-1], "steps_last": c.step.iloc[-1],
                     "windows_seen": int(c.step.iloc[-1]) * int(ck["args"]["bs"]),
                     "first_val_b": c[col].iloc[0], "best_val_b": c[col].min(), "argmin_epoch": int(c.epoch[c[col].idxmin()]),
                     "ckpt_epoch": ck["epoch"], "ckpt_val_b": ck["val"]["val_b_median"],
                     "train_loss_first": c.get("train_loss", pd.Series([np.nan])).iloc[0],
                     "train_loss_last": c.get("train_loss", pd.Series([np.nan])).iloc[-1],
                     "val_win_err_first": c.val_win_err_median.iloc[0], "val_win_err_last": c.val_win_err_median.iloc[-1],
                     "max_mem_gib": c.max_mem_gib.max(), "state_dict_params": nparam,
                     "windows_per_s": c.step.iloc[-1] * ck["args"]["bs"] / (c.minutes.iloc[-1] * 60),
                     "K": ck["args"]["K"], "stages": ck["args"]["stages"], "val_on_train": ck["args"].get("val_on_train"),
                     "train_runs": ck["args"].get("train_runs")})
    emit("training_curves", pd.DataFrame(rows))


def rerun_compare():
    rows = []
    for s in ["test-iid", "e1-core", "val"]:
        a = PREDS / "spectral" / f"{s}.csv"
        b = VER / "preds" / f"{s}.csv"
        if not b.exists():
            continue
        A, B = pd.read_csv(a), pd.read_csv(b)
        k = ["run_id", "w_start"]
        J = A.merge(B, on=k, suffixes=("_a", "_b"))
        dv = np.hypot(J.v_est_x_a - J.v_est_x_b, J.v_est_y_a - J.v_est_y_b)
        ds = np.abs(J[["sigma_x_a", "sigma_y_a"]].to_numpy() - J[["sigma_x_b", "sigma_y_b"]].to_numpy())
        sa = pd.read_csv(EVAL / "spectral" / s / "slices.csv")
        sb = pd.read_csv(VER / "eval" / s / "slices.csv")
        ra = sa[sa.slice == "all"].iloc[0] if "slice" in sa else sa.iloc[0]
        rb = sb[sb.slice == "all"].iloc[0] if "slice" in sb else sb.iloc[0]
        rows.append({"set": s, "rows_a": len(A), "rows_b": len(B), "joined": len(J), "v_absdiff_max": dv.max(),
                     "v_absdiff_median": float(np.median(dv)), "frac_v_diff_gt_1mm": float((dv > 1e-3).mean()),
                     "sigma_absdiff_max": ds.max(), "orig_b_median": ra.b_median, "rerun_b_median": rb.b_median,
                     "orig_b_p90": ra.b_p90, "rerun_b_p90": rb.b_p90, "orig_frac_b_le_017": ra.frac_b_le_017,
                     "rerun_frac_b_le_017": rb.frac_b_le_017})
    emit("rerun_compare", pd.DataFrame(rows))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(VER / "checks"))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    OUT = Path(a.out)
    steps = {"disjoint": check_disjoint, "theta": check_theta, "pattern": check_pattern_sources,
             "diag": check_diag_centre, "refs": check_references, "calib": calibration, "train": training_curves,
             "rerun": rerun_compare}
    if a.only:
        for n in a.only.split(","):
            if n == "headlines":
                per = check_headlines()
                slice_tables(per)
                e1_compare(per)
            else:
                steps[n]()
        return
    for n, f in steps.items():
        f()
    per = check_headlines()
    slice_tables(per)
    e1_compare(per)


if __name__ == "__main__":
    main()
