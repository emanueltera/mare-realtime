"""Is flight 0087's step-direction bias a water drift (a fixed world vector) or a camera yaw offset
(fixed in the body)? Fits  pred_v - gnss_v = -c + d*J v + (s-1) v  over all steps, with one scale s
per stretch, and prints each stretch's mean velocity error along and left of its track.

    python real_bias.py <extract_real.py output dir> <e3.real output dir>
"""
import sys, json, numpy as np
from pathlib import Path
real, ev = Path(sys.argv[1]), Path(sys.argv[2])
z = np.load(ev / "preds.npz")
idx = json.loads((real / "index.json").read_text())["stretches"]
J = np.array([[0, -1], [1, 0]])
rows, Y, X, per = [], [], [], []
for k, st in enumerate(idx):
    r = np.load(real / st["name"] / "real.npz")
    dp = z[f"{st['name']}/dp"]
    R = r["R"]
    dt = np.diff(r["t"])[:, None]
    pred = np.einsum("nij,nj->ni", R[:-1], dp)[:, :2] / dt
    true = np.diff(r["en"], axis=0) / dt
    d = pred - true
    hdg = np.degrees(np.arctan2(true[:, 0].mean(), true[:, 1].mean())) % 360
    sp = np.linalg.norm(true, axis=1).mean()
    # per stretch: mean difference split into along/cross the mean track
    u = true.mean(0) / np.linalg.norm(true.mean(0)); n = J @ u
    per.append((st["name"], hdg, sp, d.mean(0), d.mean(0) @ u, d.mean(0) @ n))
    for i in range(len(d)):
        v = true[i]
        # unknowns: c_e, c_n, delta, s_k (per stretch)
        a = np.zeros((2, 3 + len(idx))); a[:, 0:2] = -np.eye(2); a[:, 2] = J @ v; a[:, 3 + k] = v
        X.append(a); Y.append(d[i])
X = np.concatenate(X); Y = np.concatenate(Y)
print("stretch     track°  speed m/s   mean(pred−gnss) E,N m/s   along   left")
for name, h, sp, m, al, cr in per:
    print(f"{name}  {h:6.1f}  {sp:6.1f}     {m[0]:+6.2f} {m[1]:+6.2f}            {al:+5.2f}  {cr:+5.2f}")
def fit(cols, label):
    A = X[:, cols]; sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = Y - A @ sol
    print(f"{label:38s} rms residual {np.sqrt((res**2).mean()):.3f} m/s;", end=" ")
    return sol
ns = list(range(3, 3 + len(idx)))
s0 = fit(ns, "scale per stretch only"); print()
s1 = fit([0, 1] + ns, "+ fixed world vector (water drift)"); print(f"c = E {s1[0]:+.2f}, N {s1[1]:+.2f} m/s (|c| {np.hypot(*s1[:2]):.2f})")
s2 = fit([2] + ns, "+ fixed yaw offset (mount/stabiliser)"); print(f"delta = {np.degrees(s2[0]):+.2f} deg")
s3 = fit([0, 1, 2] + ns, "+ both"); print(f"c = E {s3[0]:+.2f}, N {s3[1]:+.2f} m/s, delta = {np.degrees(s3[2]):+.2f} deg")
