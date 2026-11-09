"""Per stretch: crab angle (GNSS track vs image-up, which is the camera's -y axis) against the
signed angle of the predicted step relative to GNSS (positive = prediction to the left), medians.

    python real_crab.py <extract_real.py output dir> <e3.real output dir>
"""
import sys, json, numpy as np
from pathlib import Path
real, ev = Path(sys.argv[1]), Path(sys.argv[2])
z = np.load(ev / "preds.npz")
idx = json.loads((real / "index.json").read_text())["stretches"]
def ang(a, b):  # angle from b to a, positive counterclockwise (left)
    return np.degrees(np.arctan2(b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0], (a * b).sum(1)))
C, P = [], []
print("stretch     crab° (track left of nose)   pred left of track°   pred left of nose°")
for st in idx:
    r = np.load(real / st["name"] / "real.npz"); dp = z[f"{st['name']}/dp"]; R = r["R"][:-1]
    true = np.diff(r["en"], axis=0)
    pred = np.einsum("nij,nj->ni", R, dp)[:, :2]
    nose = -R[:, :2, 1]
    c, p, q = np.median(ang(true, nose)), np.median(ang(pred, true)), np.median(ang(pred, nose))
    C.append(c); P.append(p)
    print(f"{st['name']}   {c:+6.2f}                        {p:+6.2f}                 {q:+6.2f}")
C, P = np.array(C), np.array(P)
k = np.polyfit(C, P, 1)
print(f"pred-vs-track = {k[0]:+.2f} * crab {k[1]:+.2f} deg;  corr {np.corrcoef(C, P)[0,1]:+.2f}")
