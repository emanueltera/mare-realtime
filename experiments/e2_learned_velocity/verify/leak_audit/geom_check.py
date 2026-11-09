"""End-to-end sign/rotation/scale check on real engine renders of a frozen sea (test-rate rate0, e1-frozen):
own numpy phase correlation between consecutive tiles, through the e2e loader (with every dihedral element),
against the loader's label. Content must move at -v*dt/gsd cells (x = col, y = row)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/workspace/mare-realtime/experiments/e2_learned_velocity")
from e2.data import TileSet
from e2e.loader import Windows

def pc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a = np.where(m, a, 0); b = np.where(m, b, 0)
    a = np.log(a + 1e-3) * m; b = np.log(b + 1e-3) * m
    n = a.shape[0]; w = np.hanning(n)[:, None] * np.hanning(n)[None]
    A = np.fft.fft2((a - a[m].mean()) * w * m); B = np.fft.fft2((b - b[m].mean()) * w * m)
    R = B * np.conj(A); R /= np.abs(R) + 1e-9
    c = np.real(np.fft.ifft2(R))
    py, px = np.unravel_index(np.argmax(c), c.shape)
    def sub(cm, c0, cp):
        d = cm - 2 * c0 + cp
        return 0.5 * (cm - cp) / d if abs(d) > 1e-12 else 0.0
    sx = sub(c[py, (px - 1) % n], c[py, px], c[py, (px + 1) % n]); sy = sub(c[(py - 1) % n, px], c[py, px], c[(py + 1) % n, px])
    dx = (px - n if px > n // 2 else px) + sx; dy = (py - n if py > n // 2 else py) + sy
    return np.array([dx, dy])  # b(x) = a(x - d)

rows = []
for SET, sel in [("test-rate", lambda r: r.sea_rate == 0), ("e1-frozen", lambda r: True)]:
    ts = TileSet(f"/workspace/data/e2/sets/{SET}")
    runs = [r for r in ts.run_ids if sel(ts.runs.loc[r])][:40]
    ds = Windows(ts, 2, step=10, runs=runs, augment=True, seed=7)
    rng = np.random.default_rng(0)
    for i in range(len(ds)):
        it = ds[i]
        a = it["tiles"].astype(np.float32)
        d = pc(a[0], a[1])
        v_img = -d * it["gsd"] / it["dt"]          # what the content says the camera did (tile axes, augmented)
        t = it["target"]
        sp = np.linalg.norm(t)
        uhat = t / max(sp, 1e-9); perp = np.array([-uhat[1], uhat[0]])
        e = v_img - t
        rows.append({"set": SET, "run_id": ds.run_id[i], "aug": it["aug"], "speed": sp, "err": np.linalg.norm(e),
                     "err_along": e @ uhat, "err_perp": e @ perp, "shift_px": np.linalg.norm(d)})
D = pd.DataFrame(rows)
D.to_csv("geom_check_windows.csv", index=False)
D["rel_along"] = D.err_along / D.speed; D["rel_perp"] = D.err_perp / D.speed
S = D.groupby(["set", "aug"]).agg(n=("err", "size"), err_median=("err", "median"), along_median=("err_along", "median"),
                                   perp_median=("err_perp", "median"), rel_along_median=("rel_along", "median"),
                                   rel_perp_median=("rel_perp", "median")).reset_index()
S.to_csv("geom_check_summary.csv", index=False)
print(S.round(4).to_string())
print(D.groupby("set").agg(err_median=("err", "median"), err_p90=("err", lambda x: x.quantile(.9)), rel_along=("rel_along", "median"), rel_perp=("rel_perp", "median")).round(4))
