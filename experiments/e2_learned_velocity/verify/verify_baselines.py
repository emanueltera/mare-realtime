"""Independent verification of the E2 baselines (B0-B3). Read-only on the baselines' outputs;
writes only under --out (default /workspace/data/e2/verify_baselines).

    cd /workspace/mare-realtime/experiments/e2_learned_velocity
    OMP_NUM_THREADS=1 /workspace/venv/bin/python -m verify.verify_baselines cpu
    HF_HOME=/workspace/.hf_home /workspace/venv/bin/python -m verify.verify_baselines gpu   # B1 subset (GPU)
    OMP_NUM_THREADS=2 /workspace/venv/bin/python -m verify.verify_baselines shell           # B2 subset
    OMP_NUM_THREADS=1 /workspace/venv/bin/python -m verify.verify_baselines b0              # B0 subset

Every check recomputes from files on disk with code written here (not e2.evaluate), except
`rerun_harness`, which runs e2.evaluate on re-inferred B3 predictions to compare its outputs.
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
BASE = Path("/workspace/data/e2/baselines")
TEST_SETS = ["val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]
PY = "/workspace/venv/bin/python"


def log(out: Path, name: str, obj):
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(out / f"{name}.csv", index=False)
        print(f"== {name}\n{obj.to_string(index=False, max_rows=80)}", flush=True)
    else:
        (out / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
        print(f"== {name}\n{json.dumps(obj, indent=1, default=float)}", flush=True)


# ---- independent per-run b ------------------------------------------------------------------


def labels(set_name):
    lb = pd.read_csv(SETS / set_name / "labels.csv")
    return {rid: g.sort_values("frame") for rid, g in lb.groupby("run_id")}


def my_per_run(pred: pd.DataFrame, lbs: dict) -> pd.DataFrame:
    """b = |mean over finite windows of (v_est - v_true)|, v_true recomputed from labels
    (stride from T: the harness's rule, end = w + round(T*fps) for stride-1 windows)."""
    rows = []
    for (rid, K), g in pred.groupby(["run_id", "K"]):
        lb = lbs[rid]
        t = lb.t.to_numpy()
        xy = lb[["cam_x", "cam_y"]].to_numpy()
        w = g.w_start.to_numpy(int)
        e = w + np.rint(g["T"].to_numpy() * 10).astype(int)
        vt = (xy[e] - xy[w]) / (t[e] - t[w])[:, None]
        truth_err = np.abs(vt - g[["v_true_x", "v_true_y"]].to_numpy()).max()
        ve = g[["v_est_x", "v_est_y"]].to_numpy()
        ok = np.isfinite(ve).all(1)
        ex = ve[ok] - vt[ok]
        b = np.hypot(*ex.mean(0)) if ok.any() else np.nan
        spd = np.linalg.norm(vt, axis=1).mean()
        rows.append({"run_id": rid, "K": int(K), "b": b, "b_pct": 100 * b / spd, "truth_err": truth_err,
                     "n": len(g), "n_ok": int(ok.sum())})
    return pd.DataFrame(rows)


def summ(b: pd.Series) -> dict:
    b = b.dropna()
    return {"runs": len(b), "b_median": b.median(), "b_p90": np.percentile(b, 90), "frac_le_017": (b <= 0.17).mean()}


def cmd_cpu(a):
    out = Path(a.out)
    import torch

    from baselines.correction import Correction, load_windows, predict_df

    # ---- 1. headline table, recomputed from the prediction CSVs ------------------------------
    ov = pd.read_csv(BASE / "tables" / "overview.csv")
    methods = ["b0-phasecorr", "b1-searaft", "b2-shell-dispersive", "b2-shell-static", "b3-mlp", "b3-const",
               "b3-mlp-full", "b3-mlp-radial", "b2-shell-dispersive-kmax0475", "b2-shell-static-kmax0475"]
    rows, perrun = [], {}
    lbcache = {}
    for m in methods:
        for s in TEST_SETS:
            p = PREDS / m / f"{s}.csv"
            if not p.exists():
                continue
            lbcache.setdefault(s, labels(s))
            pred = pd.read_csv(p)
            pr = my_per_run(pred, lbcache[s])
            pr["method"], pr["set"] = m, s
            perrun[(m, s)] = pr
            runs_in_set = len(pd.read_csv(SETS / s / "runs.csv"))
            for K, g in pr.groupby("K"):
                r = {"method": m, "set": s, "K": K, **summ(g.b), "bpct_median": g.b_pct.median(),
                     "all_runs_present": g.run_id.nunique() == runs_in_set, "truth_err_max": g.truth_err.max(),
                     "windows": int(g.n.sum()), "nan_windows": int((g.n - g.n_ok).sum())}
                o = ov[(ov.method == m) & (ov.set == s) & (ov.K == K)]
                if len(o):
                    r["rep_b_median"], r["rep_b_p90"], r["rep_frac"] = o.b_median.iat[0], o.b_p90.iat[0], o.frac_b_le_017.iat[0]
                    r["d_median"] = r["b_median"] - r["rep_b_median"]
                    r["d_p90"] = r["b_p90"] - r["rep_b_p90"]
                rows.append(r)
    tab = pd.DataFrame(rows)
    log(out, "recomputed_overview", tab)
    pd.concat(perrun.values()).to_csv(out / "my_per_run.csv", index=False)

    # sea-rate slice (test-rate) and e1-core sun slice
    runs_tr = pd.read_csv(SETS / "test-rate" / "runs.csv").set_index("run_id")
    runs_e1 = pd.read_csv(SETS / "e1-core" / "runs.csv").set_index("run_id")
    sl = []
    for (m, K) in [("b0-phasecorr", 8), ("b1-searaft", 8), ("b2-shell-dispersive", 32), ("b2-shell-static", 32),
                   ("b3-mlp", 8), ("b3-const", 8)]:
        pr = perrun[(m, "test-rate")]
        pr = pr[pr.K == K].copy()
        pr["sea_rate"] = pr.run_id.map(runs_tr.sea_rate)
        for r, g in pr.groupby("sea_rate"):
            sl.append({"method": m, "K": K, "slice": f"sea_rate={r}", **summ(g.b)})
        pr = perrun[(m, "e1-core")]
        pr = pr[pr.K == K].copy()
        pr["sun"] = pr.run_id.map(runs_e1.sun_elevation)
        for name, msk in [("sun>=35", pr.sun >= 35), ("sun 5-15", (pr.sun >= 5) & (pr.sun < 15))]:
            sl.append({"method": m, "K": K, "slice": name, **summ(pr[msk].b)})
    log(out, "recomputed_slices", pd.DataFrame(sl))

    # ---- 2. B3: checkpoints, re-inference, ablations ------------------------------------------
    cks = {n: torch.load(BASE / "b3" / f"{n}.pt", weights_only=False) for n in ("best", "best-radial", "best-full")}
    same = all(torch.equal(cks["best"]["state_dict"][k], cks["best-radial"]["state_dict"][k]) for k in cks["best"]["state_dict"])
    info = {n: {"config": c["config"], "train_config": c["train_config"], "const": c["const"]} for n, c in cks.items()}
    info["best_equals_best_radial"] = same
    log(out, "b3_checkpoints", info)

    model = Correction(**cks["best"]["config"])
    model.load_state_dict(cks["best"]["state_dict"])
    model.eval()
    re = []
    out_pred = out / "b3_rerun_preds"
    out_pred.mkdir(parents=True, exist_ok=True)
    from e2.data import PRED_COLUMNS, TileSet, prediction_frame

    for s in TEST_SETS:
        df = load_windows(PREDS / "b1-searaft", s)
        est = predict_df(model, df)
        stored = pd.read_csv(PREDS / "b3-mlp" / f"{s}.csv")
        const = pd.read_csv(PREDS / "b3-const" / f"{s}.csv")
        key = ["run_id", "w_start", "K"]
        j = df[key].assign(ex=est[:, 0], ey=est[:, 1], px=df.v_est_x, py=df.v_est_y).merge(
            stored[key + ["v_est_x", "v_est_y"]], on=key, how="outer", indicator=True)
        jc = df[key].assign(px=df.v_est_x, py=df.v_est_y).merge(const[key + ["v_est_x", "v_est_y"]], on=key)
        c = np.array(cks["best"]["const"])
        d = np.abs(j[["ex", "ey"]].to_numpy() - j[["v_est_x", "v_est_y"]].to_numpy())
        dc = np.abs(jc[["px", "py"]].to_numpy() + c - jc[["v_est_x", "v_est_y"]].to_numpy())
        # does B3's input carry anything but B1 and gsd? Inputs: B1 v_est (merge check) and gsd per run
        re.append({"set": s, "windows": len(df), "rows_stored": len(stored), "unmatched": int((j._merge != "both").sum()),
                   "max_abs_diff_b3": float(np.nanmax(d)), "nan_mismatch": int((np.isnan(d).any(1) &
                   ~(np.isnan(j[["ex", "ey"]]).all(1) & np.isnan(j[["v_est_x", "v_est_y"]]).all(1))).sum()),
                   "max_abs_diff_const": float(np.nanmax(dc))})
        if s in ("test-iid", "e1-core"):
            ts = TileSet(SETS / s)
            keys = list(zip(df["set"], df.run_id, df.w_start, df.K, df["T"]))
            prediction_frame(ts, keys, est)[PRED_COLUMNS].to_csv(out_pred / f"{s}.csv", index=False)
    log(out, "b3_reinference", pd.DataFrame(re))

    # B3 retrain of the selected config (determinism of selection) on the same data
    from baselines.correction import fit

    tr, va = load_windows(PREDS / "b1-searaft", "train"), load_windows(PREDS / "b1-searaft", "val")
    tc = cks["best"]["train_config"]
    torch.set_num_threads(8)
    _, curve, sel = fit(tr, va, **tc)
    log(out, "b3_retrain_selected", {"config": tc, "selection": sel, "curve_epochs_1_10": curve.head(10).to_dict("records")})

    # ---- 3. data hygiene: B3 train runs, overlaps -------------------------------------------
    hyg = {}
    tr_runs = pd.read_csv(SETS / "train" / "runs.csv")
    hyg["b3_train_runs"] = int(tr.run_id.nunique())
    hyg["b3_train_runs_are_first_1500"] = set(tr.run_id) == set(tr_runs.run_id.iloc[:1500])
    hyg["b3_train_windows"] = len(tr)
    allruns = {s: pd.read_csv(SETS / s / "runs.csv") for s in ["train"] + TEST_SETS}
    ov_rows = []
    for s in TEST_SETS:
        for col in ("run_id", "traj_seed", "sea_seed"):
            if col in allruns[s]:
                ov_rows.append({"set": s, "col": col, "overlap_with_train": len(set(allruns[s][col]) & set(allruns["train"][col]))})
    hyg["overlap"] = ov_rows
    # theta uniform per set (KS against U[0,360))
    from scipy import stats

    hyg["theta_ks"] = {s: {"n": len(r), "ks_p": float(stats.kstest(r.theta_deg / 360.0, "uniform").pvalue),
                           "min": float(r.theta_deg.min()), "max": float(r.theta_deg.max())} for s, r in allruns.items()}
    log(out, "hygiene", hyg)

    # ---- 4. window medians from pairs (B0, B1), own implementation -----------------------------
    wm = []
    for m, pdir in [("b0-phasecorr", "phasecorr"), ("b1-searaft", "searaft")]:
        for s in ("test-iid", "e1-core", "val"):
            pairs = pd.read_csv(BASE / "pairs" / pdir / f"{s}.csv")
            stored = pd.read_csv(PREDS / m / f"{s}.csv")
            mx = 0.0
            for K in (2, 8, 16):
                st = stored[stored.K == K].set_index(["run_id", "w_start"])
                ests = []
                for rid, g in pairs.groupby("run_id"):
                    v = g.sort_values("w_start")[["v_est_x", "v_est_y"]].to_numpy()
                    for w in range(len(v) - (K - 1) + 1):
                        seg = v[w:w + K - 1]
                        ests.append((rid, w, *[np.nanmedian(seg[:, c]) if np.isfinite(seg[:, c]).any() else np.nan for c in (0, 1)]))
                e = pd.DataFrame(ests, columns=["run_id", "w_start", "x", "y"]).set_index(["run_id", "w_start"])
                jj = e.join(st[["v_est_x", "v_est_y"]], how="outer")
                diff = np.abs(jj[["x", "y"]].to_numpy() - jj[["v_est_x", "v_est_y"]].to_numpy())
                wm.append({"method": m, "set": s, "K": K, "mine": len(e), "stored": len(st),
                           "max_abs_diff": float(np.nanmax(diff)),
                           "nan_disagree": int((np.isnan(jj[["x"]].to_numpy()) != np.isnan(jj[["v_est_x"]].to_numpy())).sum())})
    log(out, "window_medians", pd.DataFrame(wm))

    # ---- 5. E1 cross-check claims --------------------------------------------------------------
    pc = pd.read_csv(BASE / "e1_crosscheck" / "per_run.csv")
    g = pc[pc.pairing == "b1-searaft K=2 vs E1 flow:searaft g=1"].dropna(subset=["b", "b_abs"])
    big = g[g.b > 10]
    small = g[g.b <= 10]
    xc = {"b1k2_runs": len(g), "pearson_all": float(np.corrcoef(g.b, g.b_abs)[0, 1]),
          "spearman_all": float(stats.spearmanr(g.b, g.b_abs).correlation),
          "median_d_b": float((g.b - g.b_abs).median()),
          "n_b_gt_10": len(big), "b_gt_10_range": [float(big.b.min()), float(big.b.max())] if len(big) else None,
          "pearson_without_gt10": float(np.corrcoef(small.b, small.b_abs)[0, 1])}
    log(out, "e1_crosscheck_recomputed", xc)
    # B1 runs with no E1 value, and runs with b > 10 per set
    b1 = []
    for s in TEST_SETS:
        pr = perrun[("b1-searaft", s)]
        pr = pr[pr.K == 8]
        b1.append({"set": s, "runs_b_gt_10": int((pr.b > 10).sum()), "max_b": float(pr.b.max())})
    log(out, "b1_collapses", pd.DataFrame(b1))


def cmd_rerun_harness(a):
    """e2.evaluate on the re-inferred B3 predictions; compare per_run b with the stored ones."""
    out = Path(a.out)
    rows = []
    for s in ("test-iid", "e1-core"):
        od = out / "b3_rerun_eval" / s
        subprocess.run([PY, "-m", "e2.evaluate", "--pred", str(out / "b3_rerun_preds" / f"{s}.csv"), "--sets", str(SETS),
                        "--out", str(od), "--name", "b3-verify", "--require-all"], check=True, cwd=HERE,
                       stdout=subprocess.DEVNULL)
        mine = pd.read_csv(od / "per_run.csv")
        theirs = pd.read_csv(EVAL / "b3-mlp" / s / "per_run.csv")
        j = mine.merge(theirs, on=["run_id", "K"], suffixes=("_v", "_r"))
        sm = pd.read_csv(od / "slices.csv")
        sm = sm[sm.slice == "all"].iloc[0]
        rows.append({"set": s, "runs": len(j), "max_abs_d_b": float((j.b_v - j.b_r).abs().max()),
                     "b_median": sm.b_median, "b_p90": sm.b_p90, "frac_le_017": sm.frac_b_le_017})
    log(out, "b3_harness_rerun", pd.DataFrame(rows))


def subset(s, n):
    """Deterministic subset: every k-th run of the set (spread over the run list)."""
    ids = pd.read_csv(SETS / s / "runs.csv").run_id.tolist()
    return ids[:: max(1, len(ids) // n)][:n]


def compare_pairs(mine: pd.DataFrame, stored_csv: Path, runs) -> dict:
    st = pd.read_csv(stored_csv)
    st = st[st.run_id.isin(runs)]
    j = mine.merge(st, on=["run_id", "w_start"], suffixes=("_v", "_r"))
    d = np.hypot(j.v_est_x_v - j.v_est_x_r, j.v_est_y_v - j.v_est_y_r)
    nan_dis = int((j.v_est_x_v.isna() != j.v_est_x_r.isna()).sum())
    return {"runs": len(runs), "pairs_mine": len(mine), "pairs_matched": len(j), "nan_disagree": nan_dis,
            "diff_median": float(np.nanmedian(d)), "diff_p99": float(np.nanpercentile(d, 99)), "diff_max": float(np.nanmax(d))}


def cmd_gpu(a):
    """B1: SEA-RAFT-M pairs on a deterministic subset of test-iid and e1-core, compared with the stored pairs."""
    from baselines.flowpairs import run_set

    out = Path(a.out)
    res = []
    for s in ("test-iid", "e1-core"):
        runs = subset(s, a.n)
        df = run_set(SETS / s, out / "b1_pairs" / f"{s}.csv", runs=runs, chunk=20, workers=4, log_every=5)
        res.append({"set": s, **compare_pairs(df, BASE / "pairs" / "searaft" / f"{s}.csv", runs)})
    log(out, "b1_rerun_pairs", pd.DataFrame(res))


def cmd_b0(a):
    from e2.baselines import phasecorr_predictions

    out = Path(a.out)
    res = []
    for s in ("test-iid", "e1-core"):
        runs = subset(s, a.n)
        df = phasecorr_predictions(SETS / s, out / "b0_pairs" / f"{s}.csv", runs=runs, workers=16)
        res.append({"set": s, **compare_pairs(df, BASE / "pairs" / "phasecorr" / f"{s}.csv", runs)})
    log(out, "b0_rerun_pairs", pd.DataFrame(res))


def cmd_shell(a):
    import multiprocessing as mp

    from baselines.shell import _worker

    out = Path(a.out)
    res = []
    for s in ("test-iid", "e1-core"):
        runs = subset(s, a.n)
        with mp.get_context("spawn").Pool(6) as pool:
            rows = sum(pool.map(_worker, [(str(SETS / s), r, {"chunk": 256}) for r in runs]), [])
        mine = pd.DataFrame(rows)
        mine.to_csv(out / f"b2_rerun_{s}.csv", index=False)
        for model in ("dispersive", "static"):
            st = pd.read_csv(PREDS / f"b2-shell-{model}" / f"{s}.csv")
            mm = mine[mine.model == model]
            j = mm.merge(st, on=["run_id", "w_start", "K"], suffixes=("_v", "_r"))
            d = np.hypot(j.v_est_x_v - j.v_est_x_r, j.v_est_y_v - j.v_est_y_r)
            res.append({"set": s, "model": model, "windows_mine": len(mm), "matched": len(j),
                        "diff_median": float(d.median()), "diff_max": float(d.max())})
    log(out, "b2_rerun", pd.DataFrame(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cpu", "rerun_harness", "gpu", "b0", "shell"])
    ap.add_argument("--out", default="/workspace/data/e2/verify_baselines")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    {"cpu": cmd_cpu, "rerun_harness": cmd_rerun_harness, "gpu": cmd_gpu, "b0": cmd_b0, "shell": cmd_shell}[a.cmd](a)
