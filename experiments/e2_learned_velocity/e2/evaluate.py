"""E2 evaluation harness: prediction CSV → per-run and per-slice metrics and a markdown summary.

    python -m e2.evaluate --pred preds.csv --sets /workspace/data/e2/sets --out results/<method> [--name <method>]

Prediction CSV, one row per window (e2.data.PRED_COLUMNS):
    set, run_id, w_start, K, T, v_true_x, v_true_y, v_est_x, v_est_y, sigma_x, sigma_y,
    v_true_e, v_true_n, v_est_e, v_est_n
velocities in m/s; x, y in the run's tile axes; e, n in ENU; sigma may be empty (NaN).
v_true is checked against the set's labels (the window's mean camera velocity): a row
that disagrees by more than 1 mm/s stops the evaluation.

Metrics (the same for every method):
- per run: bias b = |mean over the run's windows of (v_est − v_true)| (vector mean, m/s);
  b% = b / the run's mean ground speed (mean |v_true| over its windows) × 100;
  rms = sqrt(mean |v_est − v_true|²) over windows; drift = |Σ (v_est − v_true)·T| / Σ |v_true|·T
  over non-overlapping windows (greedy from the first window: each starts at or after the
  previous one's last frame), × 100 as drift%; and b along / across the wind sea.
- per slice (set, altitude, ground speed, sun elevation, wind speed, swell, gimbal pitch, K,
  seaRate, pattern, masked fraction): runs, median and p90 of b, b%, rms and drift%, and the
  fraction of runs with b ≤ 0.17 m/s, b% ≤ 0.28 and b% ≤ 0.5.
Every number in summary.md is derived from per_run.csv / slices.csv written next to it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

BUDGET_MS = 0.17     # m/s: the water-motion share of the drift budget at 60 m/s (E1)
BUDGET_PCT = 0.28    # % of ground speed
DRIFT_PCT = 0.5      # % of distance: the whole drift budget

REQUIRED = ["set", "run_id", "w_start", "K", "T", "v_true_x", "v_true_y", "v_est_x", "v_est_y"]

SLICES = {
    "altitude": ("altitude", [0, 50, 80, 130, 200, 260, 330, 400.01]),
    "ground_speed": ("ground_speed", [0, 20, 35, 50, 65, 200]),
    "sun_elevation": ("sun_elevation", [-90, 5, 15, 35, 90]),
    "wind_speed": ("wind_speed", [0, 6, 10, 14, 16, 18, 20.01]),
    "gimbal_pitch": ("gimbal_pitch", [-90.01, -80, -70, -59.99]),
    "masked_frac": ("masked_frac_mean", [-0.01, 0.001, 0.01, 0.05, 1.01]),
}
CATEGORICAL = ["set", "swell", "K", "sea_rate", "pattern"]


def nonoverlapping(w_start: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Indices (into the sorted order) of greedily chosen windows that share no frame interval."""
    order = np.argsort(w_start, kind="stable")
    keep, end = [], -1
    for i in order:
        if w_start[i] >= end:
            keep.append(i)
            end = w_start[i] + span[i]
    return np.array(keep, dtype=int)


def load_sets(sets_root: Path, names) -> tuple[pd.DataFrame, dict]:
    runs, labels = [], {}
    for s in names:
        d = sets_root / s
        r = pd.read_csv(d / "runs.csv")
        r["set"] = s
        runs.append(r)
        labels[s] = pd.read_csv(d / "labels.csv")
    return pd.concat(runs, ignore_index=True), labels


def check_truth(pred: pd.DataFrame, runs: pd.DataFrame, labels: dict, tol: float = 1e-3) -> pd.DataFrame:
    """Recompute v_true from the labels for every row; raise if any row disagrees."""
    fps = runs.set_index(["set", "run_id"])["fps"]
    out = []
    for (s, rid), g in pred.groupby(["set", "run_id"], sort=False):
        lb = labels[s]
        lb = lb[lb.run_id == rid].sort_values("frame")
        if lb.empty:
            raise ValueError(f"{s}/{rid}: no such run in the set's labels")
        t = lb["t"].to_numpy()
        pe, px = lb[["cam_e", "cam_n"]].to_numpy(), lb[["cam_x", "cam_y"]].to_numpy()
        f = float(fps.loc[(s, rid)])
        w = g["w_start"].to_numpy(int)
        span = np.rint(g["T"].to_numpy() * f).astype(int)
        e = w + span
        if (e >= len(t)).any() or (w < 0).any():
            raise ValueError(f"{s}/{rid}: window outside the run")
        T = t[e] - t[w]
        vx = (px[e] - px[w]) / T[:, None]
        ve = (pe[e] - pe[w]) / T[:, None]
        err = np.abs(vx - g[["v_true_x", "v_true_y"]].to_numpy()).max()
        if err > tol:
            raise ValueError(f"{s}/{rid}: v_true differs from the labels by {err:.4f} m/s (window/T mismatch?)")
        g = g.copy()
        g["span"] = span
        g["T"] = T
        g["v_true_x"], g["v_true_y"] = vx[:, 0], vx[:, 1]
        g["v_true_e"], g["v_true_n"] = ve[:, 0], ve[:, 1]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def per_run(pred: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    meta = runs.set_index(["set", "run_id"])
    rows = []
    for (s, rid, K), g in pred.groupby(["set", "run_id", "K"], sort=False):
        m = meta.loc[(s, rid)]
        th = math.radians(float(m["theta_deg"]))
        ex = g[["v_est_x", "v_est_y"]].to_numpy() - g[["v_true_x", "v_true_y"]].to_numpy()
        ok = np.isfinite(ex).all(1)
        vt = g[["v_true_x", "v_true_y"]].to_numpy()
        speed = float(np.linalg.norm(vt, axis=1).mean())
        r = {"set": s, "run_id": rid, "K": int(K), "T": float(g["T"].median()), "n_windows": len(g),
             "n_valid": int(ok.sum()), "ground_speed": speed}
        if ok.any():
            b = ex[ok].mean(0)
            r["b_x"], r["b_y"] = b
            be = b[0] * np.array([math.cos(th), math.sin(th)]) + b[1] * np.array([-math.sin(th), math.cos(th)])
            r["b_e"], r["b_n"] = be
            r["b"] = float(np.hypot(*b))
            r["b_pct"] = 100 * r["b"] / speed if speed > 0 else np.nan
            r["rms"] = float(np.sqrt((ex[ok] ** 2).sum(1).mean()))
            wd = math.radians(float(m["wind_direction"]))
            uw = np.array([math.sin(wd), math.cos(wd)])
            r["b_along_wind"] = float(be @ uw)
            r["b_cross_wind"] = float(be @ np.array([uw[1], -uw[0]]))
            sel = nonoverlapping(g["w_start"].to_numpy(), g["span"].to_numpy())
            sel = sel[ok[sel]]
            T = g["T"].to_numpy()[sel]
            dist = float((np.linalg.norm(vt[sel], axis=1) * T).sum())
            r["n_drift_windows"] = len(sel)
            r["drift_distance"] = dist
            r["drift_pct"] = 100 * float(np.linalg.norm((ex[sel] * T[:, None]).sum(0))) / dist if dist > 0 else np.nan
            if {"sigma_x", "sigma_y"} <= set(g.columns):
                r["sigma_mean"] = float(np.nanmean(np.hypot(g["sigma_x"], g["sigma_y"]))) if g[["sigma_x", "sigma_y"]].notna().any().any() else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    keep = ["altitude", "airspeed", "cam_u_mean", "gimbal_pitch", "stabilized", "turbulence", "pattern", "wind_speed",
            "wind_direction", "swell", "swell_height", "swell_period", "sea_hs", "sea_tp", "sun_elevation", "overcast",
            "haze", "sea_rate", "gsd", "theta_deg", "masked_frac_mean", "valid_frac_min", "ground_speed_mean"]
    return out.merge(runs[["set", "run_id"] + [c for c in keep if c in runs]], on=["set", "run_id"], how="left")


def summarise(df: pd.DataFrame) -> dict:
    b, bp, dp, rms = df["b"], df["b_pct"], df["drift_pct"], df["rms"]
    q = lambda s, p: float(np.nanpercentile(s, p)) if s.notna().any() else np.nan  # noqa: E731
    return {
        "runs": len(df), "b_median": q(b, 50), "b_p90": q(b, 90), "b_pct_median": q(bp, 50), "b_pct_p90": q(bp, 90),
        "rms_median": q(rms, 50), "drift_pct_median": q(dp, 50), "drift_pct_p90": q(dp, 90),
        "frac_b_le_017": float((b <= BUDGET_MS).mean()), "frac_bpct_le_028": float((bp <= BUDGET_PCT).mean()),
        "frac_bpct_le_05": float((bp <= DRIFT_PCT).mean()), "frac_drift_le_05": float((dp <= DRIFT_PCT).mean()),
    }


def slices(pr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, gs in pr.groupby("set", sort=True):
        for K, g in gs.groupby("K"):
            rows.append({"set": s, "K": K, "slice": "all", "bin": "all", **summarise(g)})
            for name, (col, edges) in SLICES.items():
                if col not in g:
                    continue
                cut = pd.cut(g[col], edges)
                for interval, gg in g.groupby(cut, observed=True):
                    rows.append({"set": s, "K": K, "slice": name, "bin": str(interval), **summarise(gg)})
            for col in CATEGORICAL:
                if col in ("set", "K") or col not in g:
                    continue
                for val, gg in g.groupby(col):
                    rows.append({"set": s, "K": K, "slice": col, "bin": str(val), **summarise(gg)})
    return pd.DataFrame(rows)


def markdown(name: str, sl: pd.DataFrame, pr: pd.DataFrame, pred_path: str, notes: list[str]) -> str:
    def f(x, nd=3):
        return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"

    lines = [f"# E2 evaluation: {name}", "",
             f"Predictions: `{pred_path}`. Every number below is derived from `per_run.csv` and `slices.csv` "
             f"in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); "
             f"b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ {BUDGET_MS} m/s, "
             f"b% ≤ {BUDGET_PCT}; drift budget {DRIFT_PCT} %.", ""]
    lines += notes
    head = "| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |"
    lines += ["", "## By set", "", head, "|" + "---|" * 13]
    for _, r in sl[sl.slice == "all"].iterrows():
        lines.append(f"| {r.set} | {r.K} | {r.runs} | {f(r.b_median)} | {f(r.b_p90)} | {f(r.b_pct_median)} | {f(r.b_pct_p90)} | "
                     f"{f(r.rms_median)} | {f(r.drift_pct_median)} | {f(r.drift_pct_p90)} | {f(r.frac_b_le_017, 2)} | "
                     f"{f(r.frac_bpct_le_028, 2)} | {f(r.frac_bpct_le_05, 2)} |")
    for slc in [*SLICES, *[c for c in CATEGORICAL if c not in ("set", "K")]]:
        sub = sl[sl.slice == slc]
        if sub.empty:
            continue
        lines += ["", f"## By {slc}", "", "| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |",
                  "|" + "---|" * 10]
        for _, r in sub.iterrows():
            lines.append(f"| {r.set} | {r.K} | {r.bin} | {r.runs} | {f(r.b_median)} | {f(r.b_p90)} | {f(r.b_pct_median)} | "
                         f"{f(r.drift_pct_median)} | {f(r.frac_b_le_017, 2)} | {f(r.frac_bpct_le_028, 2)} |")
    return "\n".join(lines) + "\n"


E1_PER_RUN = Path(__file__).resolve().parents[2] / "e1_wave_bias" / "results" / "per_run.csv"
E1_METHODS = [("flow:searaft", "g=1"), ("flow:ufmrefine980", "g=1"), ("shell:dispersive", "T=3.2"), ("shell:dispersive", "T=6.4")]


def e1_compare(pr: pd.DataFrame, e1_csv: Path) -> pd.DataFrame | None:
    """For set e1-core: this method's per-run b next to E1's on the same runs (b_abs of E1's
    flow and shell estimators, from E1's results/per_run.csv)."""
    mine = pr[pr.set == "e1-core"]
    if mine.empty or not Path(e1_csv).exists():
        return None
    e1 = pd.read_csv(e1_csv)
    e1 = e1[e1.set == "core"]
    out = mine[["run_id", "K", "b", "b_pct", "ground_speed", "altitude", "wind_speed", "sun_elevation"]].copy()
    for m, p in E1_METHODS:
        col = f"e1_{m.replace(':', '_')}_{p.replace('=', '')}"
        sub = e1[(e1.method == m) & (e1.param == p)].set_index("run_id")["b_abs"]
        out[col] = out.run_id.map(sub)
    return out


def evaluate(pred_path, sets_root, out_dir, name=None, sets=None, require_all=False, e1_csv=E1_PER_RUN):
    pred = pd.read_csv(pred_path)
    missing = [c for c in REQUIRED if c not in pred]
    if missing:
        raise ValueError(f"prediction CSV lacks columns {missing}")
    names = sorted(pred["set"].unique()) if sets is None else sets
    pred = pred[pred["set"].isin(names)]
    runs, labels = load_sets(Path(sets_root), names)
    pred = check_truth(pred, runs, labels)
    pr = per_run(pred, runs)
    notes = []
    for s in names:
        have = set(pr.loc[pr.set == s, "run_id"])
        allr = set(runs.loc[runs.set == s, "run_id"])
        miss = sorted(allr - have)
        notes.append(f"- {s}: {len(have)} of {len(allr)} runs predicted" + (f" ({len(miss)} missing)" if miss else "") + ".")
        if require_all and miss:
            raise ValueError(f"{s}: {len(miss)} runs without predictions, e.g. {miss[:3]}")
    nf = int((pr["n_valid"] < pr["n_windows"]).sum())
    if nf:
        notes.append(f"- {nf} runs have windows with a non-finite estimate (left out of their run's metrics).")
    sl = slices(pr)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pr.to_csv(out / "per_run.csv", index=False)
    sl.to_csv(out / "slices.csv", index=False)
    cmp = e1_compare(pr, e1_csv)
    if cmp is not None:
        cmp.to_csv(out / "e1_compare.csv", index=False)
        cols = [c for c in cmp if c.startswith("e1_")]
        notes.append("")
        notes.append("e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method "
                     f"{cmp.b.median():.3f}; " + "; ".join(f"{c[3:]} {cmp[c].median():.3f}" for c in cols) + ".")
    (out / "summary.md").write_text(markdown(name or Path(pred_path).stem, sl, pr, str(pred_path), notes))
    return pr, sl


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m e2.evaluate")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--sets", default="/workspace/data/e2/sets", help="root holding <set>/runs.csv, labels.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="evaluate only these sets")
    ap.add_argument("--require-all", action="store_true", help="fail if a set has runs without predictions")
    a = ap.parse_args(argv)
    pr, sl = evaluate(a.pred, a.sets, a.out, a.name, a.only, a.require_all)
    print(sl[sl.slice == "all"][["set", "K", "runs", "b_median", "b_p90", "b_pct_median", "drift_pct_median",
                                  "frac_b_le_017", "frac_bpct_le_028"]].to_string(index=False))


if __name__ == "__main__":
    main()
