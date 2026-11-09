"""Score E3 predictions: per-step errors, and drift when the steps are chained over a whole run.

    python -m e3.evaluate --set <sets>/<set> --out <dir> name=<preds>.npz [name=<preds>.npz ...]

Every method is also compared with `airspeed`, dead reckoning from the airspeed along the nose
(no wind: what the aircraft knows without a camera), built from the same sensor draw.

Chaining: position p̂_i+1 = p̂_i + R_i · dp̂_i from the true start, with three attitudes R_i:
  true   the true camera attitude (isolates the translation estimate)
  ahrs   the AHRS attitude, with its yaw error (bias σ 2°) — what a real system would use
  chain  the AHRS attitude at frame 0, then the method's own rotations (airspeed: the gyro's)
Drift = horizontal end-point error / horizontal distance flown, in %. Because a loop or
orbit can bring a biased estimate back near the truth, max_drift is also reported: the
largest horizontal error at any frame / the whole distance flown. b = horizontal end-point
error / run duration, m/s (the mean velocity error, as E2's per-run bias). yaw_err is the
heading part of the attitude error at the last step (world-vertical component).

Writes per_run.csv, per_step.csv and summary.md in --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DT, Run, exp_so3, list_runs, log_so3, parse_noise, run_seed, simulate

ATTS = ("true", "ahrs", "chain")


def chain(p0, R_list, dp):
    steps = np.einsum("nij,nj->ni", R_list, dp)
    return np.concatenate([p0[None], p0[None] + np.cumsum(steps, 0)])


def chain_rot(R0, rot):
    Rs = [R0]
    for w in exp_so3(rot):
        Rs.append(Rs[-1] @ w)
    return np.stack(Rs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise", action="append", help="sensor noise override key=value (see data.parse_noise); repeatable. "
                    "Use the same overrides the predictions were made with, except yaw_*, which the network never sees")
    ap.add_argument("preds", nargs="+")
    a = ap.parse_args()
    noise = parse_noise(a.noise)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds = {}
    for spec in a.preds:
        name, path = spec.split("=", 1)
        z = np.load(path)
        preds[name] = z
    runs = [Run.load(d) for d in list_runs(a.set)]
    ids = [r.id for r in runs]
    rows, steps = [], []
    for name, z in preds.items():
        assert list(z["runs"]) == ids, f"{name}: runs differ from the set"
    for ri, run in enumerate(runs):
        s = simulate(run, np.random.default_rng(run_seed(run)), noise)
        dp_true, rot_true = run.labels()
        n = run.n - 1
        methods = {"airspeed": (run.nose[None] * s.airspeed[:-1, None] * DT, s.gyro)}
        for name, z in preds.items():
            m = z["run"] == ri
            order = np.argsort(z["i"][m])
            assert len(order) == n and (z["i"][m][order] == np.arange(n)).all()
            methods[name] = (z["dp_pred"][m][order], z["rot_pred"][m][order])
        hpath = np.linalg.norm(np.diff(run.p[:, :2], axis=0), axis=1).sum()
        for name, (dp, rot) in methods.items():
            e = np.linalg.norm(dp - dp_true, axis=1)
            sl = np.linalg.norm(dp_true, axis=1)
            fwd = dp_true / sl[:, None]
            steps.append(dict(run=run.id, method=name, sea_rate=run.sea_rate, step_err_m=e.mean(), step_err_pct_med=np.median(e / sl) * 100,
                              along_bias_pct=np.mean(((dp - dp_true) * fwd).sum(1)) / sl.mean() * 100,
                              rot_err_deg=np.degrees(np.linalg.norm(rot - rot_true, axis=1)).mean(),
                              gyro_err_deg=np.degrees(np.linalg.norm(s.gyro - rot_true, axis=1)).mean()))
            atts = {"true": run.R[:-1], "ahrs": s.R_ahrs[:-1], "chain": chain_rot(s.R_ahrs[0], rot)[:-1]}
            for att, R_list in atts.items():
                p_hat = chain(run.p[0], R_list, dp)
                errs = np.linalg.norm(p_hat[:, :2] - run.p[:, :2], axis=1)
                err = errs[-1]
                yaw_err = np.degrees(log_so3(R_list[-1] @ run.R[-2].T)[2])
                rows.append(dict(run=run.id, method=name, att=att, sea_rate=run.sea_rate, duration=n * DT, path_m=hpath,
                                 end_err_m=err, drift_pct=err / hpath * 100, max_drift_pct=errs.max() / hpath * 100, b_ms=err / (n * DT),
                                 yaw_err_deg=yaw_err, pattern=run.meta["trajectory"]["pattern"]["type"],
                                 altitude=run.p[0, 2], airspeed=run.airspeed.mean(),
                                 ground_speed=hpath / (n * DT), wind=run.meta["sea"]["windSpeed"],
                                 sun=run.meta["light"]["sunElevation"]))
    pr = pd.DataFrame(rows)
    ps = pd.DataFrame(steps)
    pr.to_csv(out / "per_run.csv", index=False)
    ps.to_csv(out / "per_step.csv", index=False)
    g = pr.groupby(["sea_rate", "method", "att"])
    summ = g.agg(runs=("drift_pct", "size"), drift_med=("drift_pct", "median"), drift_p90=("drift_pct", lambda x: np.percentile(x, 90)),
                 under_0p5=("drift_pct", lambda x: (x <= 0.5).mean() * 100), max_drift_med=("max_drift_pct", "median"),
                 b_med=("b_ms", "median"), yaw_err_abs_med=("yaw_err_deg", lambda x: np.median(np.abs(x)))).reset_index()
    pat = pr[pr.att == "true"].groupby(["sea_rate", "pattern", "method"]).agg(runs=("drift_pct", "size"), drift_med=("drift_pct", "median"),
                                                                           max_drift_med=("max_drift_pct", "median")).reset_index()
    st = ps.groupby(["sea_rate", "method"]).agg(step_err_m=("step_err_m", "mean"), step_err_pct_med=("step_err_pct_med", "median"),
                                                along_bias_pct=("along_bias_pct", "median"), rot_err_deg=("rot_err_deg", "mean"),
                                                gyro_err_deg=("gyro_err_deg", "mean")).reset_index()
    summ.to_csv(out / "summary_runs.csv", index=False)
    pat.to_csv(out / "summary_patterns.csv", index=False)
    st.to_csv(out / "summary_steps.csv", index=False)
    with open(out / "summary.md", "w") as f:
        f.write(f"# {Path(a.set).name}: {len(runs)} runs\n\n## Drift over the run (horizontal end error / distance)\n\n")
        f.write(summ.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## By pattern (true attitude)\n\n")
        f.write(pat.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Per step\n\n")
        f.write(st.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    print(open(out / "summary.md").read())


if __name__ == "__main__":
    main()
