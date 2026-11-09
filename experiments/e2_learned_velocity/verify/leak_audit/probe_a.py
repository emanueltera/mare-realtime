"""Audit probes on model A (e2e main, K=8) on val windows (step 4): perturb the input tiles and
see what the estimate does. Writes probe_a_windows.csv and probe_a_runs.csv."""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "/workspace/mare-realtime/experiments/e2_learned_velocity")
from e2.data import TileSet
from e2e.predict import load
from scipy import ndimage

torch.cuda.set_per_process_memory_fraction(16 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
dev = torch.device("cuda")
model, ck = load("/workspace/data/e2/e2e/main/best.pt", dev)
K = ck["args"]["K"]
SET = sys.argv[1] if len(sys.argv) > 1 else "val"
ts = TileSet(f"/workspace/data/e2/sets/{SET}")
idx = ts.windows(K, 1, 4, 0.0).reset_index(drop=True)
rng = np.random.default_rng(123)
runs = ts.runs

def perturb(a, kind, gsd, r):
    """a [K, n, n] float32 with NaN."""
    K, n, _ = a.shape
    if kind == "base":
        return a
    if kind == "shuffle":
        p = rng.permutation(K)
        while (p == np.arange(K)).all() or (p == np.arange(K)[::-1]).all():
            p = rng.permutation(K)
        return a[p]
    if kind == "reverse":
        return a[::-1].copy()
    if kind == "static0":
        return np.repeat(a[:1], K, 0)
    if kind == "static_mid":
        return np.repeat(a[K // 2:K // 2 + 1], K, 0)
    if kind == "clip99":   # remove the brightest 1% (whitecaps, foam, glint peaks) per window
        q = np.nanpercentile(a, 99)
        return np.minimum(a, q)
    if kind == "clip95":
        q = np.nanpercentile(a, 95)
        return np.minimum(a, q)
    if kind in ("shadow_move", "shadow_static"):
        # a smooth multiplicative brightness pattern (60 m scale, ±30 %), moving 8 m/s along +x tile
        # (like the sim's cloud shadow at U10 = 10) or held fixed to the tile
        yy, xx = np.mgrid[0:n, 0:n] * gsd
        out = np.empty_like(a)
        for k in range(K):
            s = 8.0 * 0.1 * k if kind == "shadow_move" else 0.0
            pat = 1 + 0.3 * np.sin(2 * np.pi * (xx - s) / 60.0 + 0.7) * np.cos(2 * np.pi * yy / 85.0)
            out[k] = a[k] * pat
        return out
    if kind == "highpass":  # remove scales longer than ~30 m
        sig = 10.0 / gsd
        out = np.empty_like(a)
        for k in range(K):
            m = np.isfinite(a[k]); x = np.where(m, a[k], np.nanmean(a[k]))
            fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.fftfreq(n)[None, :]
            lo = np.real(np.fft.ifft2(np.fft.fft2(x) * np.exp(-2 * (np.pi * sig) ** 2 * (fx * fx + fy * fy))))
            out[k] = np.where(m, x - lo + np.nanmean(a[k]), np.nan)
        return np.clip(out, 1e-4, None)
    if kind == "frame_noise_static":   # same frame, independent pixel noise per copy
        b = np.repeat(a[:1], K, 0)
        return b * (1 + 0.02 * rng.standard_normal(b.shape)).astype(np.float32)
    raise ValueError(kind)

KINDS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base", "shuffle", "reverse", "static0", "static_mid", "clip99", "clip95", "shadow_move", "shadow_static", "highpass"]
rows = []
B = 16
for kind in KINDS:
    for s in range(0, len(idx), B):
        g = idx.iloc[s:s + B]
        tl = np.stack([perturb(ts.window_tiles(r.run_id, int(r.w_start), K), kind, float(r.gsd), r) for r in g.itertuples()])
        with torch.no_grad():
            o = model(torch.from_numpy(tl).to(dev), torch.tensor(g.gsd.values, dtype=torch.float32, device=dev),
                      torch.tensor(g.dt.values, dtype=torch.float32, device=dev))
        v = o["v"].cpu().numpy(); sg = o["sigma_v"].cpu().numpy()
        v0 = (-o["u0"] * (torch.tensor(g.gsd.values, device=dev) / torch.tensor(g.dt.values, device=dev))[:, None]).cpu().numpy()
        for j, r in enumerate(g.itertuples()):
            rows.append({"kind": kind, "run_id": r.run_id, "w_start": r.w_start, "vt_x": r.v_true_x, "vt_y": r.v_true_y,
                         "v_x": v[j, 0], "v_y": v[j, 1], "s_x": sg[j, 0], "s_y": sg[j, 1], "v0_x": v0[j, 0], "v0_y": v0[j, 1]})
    print(kind, "done", flush=True)
W = pd.DataFrame(rows)
W.to_csv(f"probe_a_windows_{SET}.csv", index=False)
W["ex"], W["ey"] = W.v_x - W.vt_x, W.v_y - W.vt_y
W["e0x"], W["e0y"] = W.v0_x - W.vt_x, W.v0_y - W.vt_y
W["spd_true"] = np.hypot(W.vt_x, W.vt_y); W["spd_est"] = np.hypot(W.v_x, W.v_y)
W["e_win"] = np.hypot(W.ex, W.ey)
R = W.groupby(["kind", "run_id"]).agg(ex=("ex", "mean"), ey=("ey", "mean"), e0x=("e0x", "mean"), e0y=("e0y", "mean"),
                                       spd_true=("spd_true", "mean"), spd_est=("spd_est", "mean"), sig=("s_x", "mean")).reset_index()
R["b"] = np.hypot(R.ex, R.ey); R["b0"] = np.hypot(R.e0x, R.e0y)
R.to_csv(f"probe_a_runs_{SET}.csv", index=False)
S = R.groupby("kind").agg(b_median=("b", "median"), b_p90=("b", lambda x: x.quantile(0.9)), frac_le_017=("b", lambda x: (x <= 0.17).mean()),
                          b0_median=("b0", "median"), spd_est_median=("spd_est", "median"), spd_true_median=("spd_true", "median"),
                          sigma_median=("sig", "median")).reset_index()
S2 = W.groupby("kind").agg(win_err_median=("e_win", "median")).reset_index()
S = S.merge(S2, on="kind")
S.to_csv(f"probe_a_summary_{SET}.csv", index=False)
print(S.to_string())
