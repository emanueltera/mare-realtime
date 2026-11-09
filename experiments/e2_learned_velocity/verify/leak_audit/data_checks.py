import pandas as pd, numpy as np, json, sys
from pathlib import Path
R = Path("/workspace/data/e2/sets")
sets = ["train","val","test-iid","test-sea","test-alt","test-rate","e1-core","pilot"]
allr = []
for s in sets:
    r = pd.read_csv(R/s/"runs.csv"); r["set"]=s; allr.append(r)
    fr = pd.read_csv(R/s/"frames.csv")
    dts = fr.groupby("run_id")["t"].apply(lambda t: np.diff(np.sort(t.values)))
    d = np.concatenate(dts.values)
    print(s, "runs", len(r), "dt min/max", d.min(), d.max(), "frames/run", fr.groupby("run_id").size().unique()[:5],
          "masked mean", round(r.masked_frac_mean.mean(),4), "frac runs masked>0.01", round((r.masked_frac_mean>0.01).mean(),3),
          "gsd", round(r.gsd.min(),3), round(r.gsd.max(),3))
A = pd.concat(allr, ignore_index=True)
# disjointness
for col in ["run_id","traj_seed","sea_seed"]:
    g = A.groupby(col)["set"].nunique()
    bad = g[g>1]
    print(col, "shared across sets:", len(bad), list(bad.index[:5]))
# within-set dupes besides test-rate
for s in sets:
    a = A[A.set==s]
    print(s, "dup sea_seed", a.sea_seed.duplicated().sum(), "dup traj_seed", a.traj_seed.duplicated().sum())
# correlation of masked fraction with ground speed (train)
t = A[A.set=="train"]
print("train corr(masked, gs)", np.corrcoef(t.masked_frac_mean, t.ground_speed_mean)[0,1],
      "corr(masked, gimbal)", np.corrcoef(t.masked_frac_mean, t.gimbal_pitch)[0,1])
print("train corr(gsd, gs)", np.corrcoef(t.gsd, t.ground_speed_mean)[0,1], "corr(wind, gs)", np.corrcoef(t.wind_speed, t.ground_speed_mean)[0,1])
# theta vs heading
print("theta-heading circ corr train:", np.abs(np.mean(np.exp(1j*np.radians(t.theta_deg - t.start_heading)))))
# held-out regime check on train/val/test-iid
for s in ["train","val","test-iid","test-alt","test-rate"]:
    a = A[A.set==s]
    ho = (a.wind_speed>=16)|((a.swell_height>0)&(a.swell_period>=14))
    print(s, "held-out-regime runs:", int(ho.sum()), "max wind", a.wind_speed.max(), "max swellT(with swell)", a[a.swell_height>0].swell_period.max(), "alt range", a.altitude.min(), a.altitude.max())
a = A[A.set=="test-sea"]; print("test-sea all heldout:", bool(((a.wind_speed>=16)|((a.swell_height>0)&(a.swell_period>=14))).all()))
print(A.groupby("set")[["overcast","haze"]].describe().T.to_string()[:1500])
