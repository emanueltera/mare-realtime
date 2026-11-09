"""Adversarial numbers-and-claims audit of results/RESULTS.md and README.md.

Recomputes the numbers RESULTS.md quotes from the CSVs on disk (and, for the main
methods, the per-run b from the raw prediction CSVs, independently of the harness),
and writes the like-for-like, selection, tail and uncertainty checks the audit
relies on. Every output is a CSV under verify/audit_out/.

Run from experiments/e2_learned_velocity/:
  uv run --no-project --with numpy --with pandas python verify/audit_numbers.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
OUT = HERE / "audit_out"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261030)

cmp_set = pd.read_csv(RES / "comparison_by_set.csv")
cmp_slice = pd.read_csv(RES / "comparison_by_slice.csv")
e1c = pd.read_csv(RES / "e1_core_compare.csv")
kab = pd.read_csv(RES / "k_ablation.csv")
rate = pd.read_csv(RES / "rate_diagnostic.csv")
cal = pd.read_csv(RES / "calibration.csv")
costs = pd.read_csv(RES / "costs.csv")
dsum = pd.read_csv(RES / "data_summary.csv")
iters = pd.read_csv(RES / "e2e" / "eval" / "iterations.csv")


def cs(method, s, col):
    r = cmp_set[(cmp_set.method == method) & (cmp_set.set == s)]
    assert len(r) == 1, (method, s, len(r))
    return float(r[col].iloc[0])


def sl(method, s, slc, b, col="b_median"):
    r = cmp_slice[(cmp_slice.method == method) & (cmp_slice.set == s) & (cmp_slice.slice == slc) & (cmp_slice.bin.astype(str) == str(b))]
    assert len(r) == 1, (method, s, slc, b, len(r))
    return float(r[col].iloc[0]), int(r["runs"].iloc[0])


def ec(method, sun, alt, col="b_median"):
    r = e1c[(e1c.method == method) & (e1c.sun == sun) & (e1c.altitude.astype(str) == str(alt))]
    assert len(r) == 1, (method, sun, alt, len(r))
    return float(r[col].iloc[0]), int(r["runs"].iloc[0])


def rd(method, r_, col):
    r = rate[(rate.method == method) & (rate.sea_rate == r_)]
    assert len(r) == 1, (method, r_)
    return float(r[col].iloc[0])


def cb(method, s, col):
    r = cal[(cal.method == method) & (cal.set == s)]
    assert len(r) == 1, (method, s)
    return float(r[col].iloc[0])


def ka(method, s, col="b_median"):
    r = kab[(kab.method == method) & (kab.set == s)]
    assert len(r) == 1, (method, s)
    return float(r[col].iloc[0])


claims: list[dict] = []


def claim(cid, where, what, reported, value, dec, source, derived=False):
    """reported: the number as printed (float). value: recomputed. dec: decimals printed."""
    rounded = round(value, dec) if np.isfinite(value) else value
    ok = bool(np.isfinite(value) and abs(rounded - reported) <= 0.5 * 10 ** (-dec) + 1e-12)
    claims.append({"id": cid, "where": where, "quantity": what, "reported": reported, "recomputed": value,
                   "recomputed_rounded": rounded, "decimals": dec, "match": ok, "derived": derived, "source": source})


SETS = ["val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]
# ---------------------------------------------------------------- Table 1 (all cells)
T1 = {
    "A": [(0.180, 0.438, 0.47), (0.191, 0.445, 0.43), (0.236, 0.687, 0.30), (0.276, 0.786, 0.26), (1.077, 2.687, 0.12), (0.199, 0.453, 0.43)],
    "C": [(0.338, 0.877, 0.22), (0.300, 0.901, 0.24), (0.356, 1.109, 0.23), (0.438, 2.245, 0.22), (0.524, 1.777, 0.10), (0.171, 0.515, 0.50)],
    "B": [(1.301, 2.219, 0.01), (1.366, 2.343, 0.00), (1.135, 2.117, 0.01), (2.642, 3.569, 0.00), (1.071, 3.280, 0.06), (1.688, 3.775, 0.00)],
    "B0": [(1.261, 1.982, 0), (1.356, 2.071, 0), (1.211, 1.993, 0), (2.683, 3.209, 0), (0.945, 2.635, 0.20), (1.394, 2.426, 0)],
    "B1": [(1.830, 3.134, 0), (1.841, 3.018, 0), (1.638, 3.009, 0), (3.272, 4.281, 0), (1.164, 4.312, 0.18), (2.080, 3.922, 0)],
    "B2": [(1.290, 2.118, 0.06), (1.216, 2.098, 0.07), (1.195, 2.208, 0.02), (1.876, 2.864, 0.04), (0.894, 2.644, 0.06), (0.439, 1.775, 0.34)],
    "B3": [(1.605, 3.076, 0.01), (1.666, 3.287, 0.00), (1.432, 3.164, 0.01), (2.958, 4.793, 0.00), (1.317, 4.114, 0.04), (2.102, 5.650, 0.00)],
}
for m, cells in T1.items():
    for s, (med, p90, fr) in zip(SETS, cells):
        claim(f"T1.{m}.{s}.med", "RESULTS Table 1", f"{m} {s} median b", med, cs(m, s, "b_median"), 3, "comparison_by_set.csv")
        claim(f"T1.{m}.{s}.p90", "RESULTS Table 1", f"{m} {s} p90 b", p90, cs(m, s, "b_p90"), 3, "comparison_by_set.csv")
        claim(f"T1.{m}.{s}.fr", "RESULTS Table 1", f"{m} {s} frac b<=0.17", fr, cs(m, s, "frac_b_le_017"), 2, "comparison_by_set.csv")
# b% table
T1b = {
    "A": [(0.51, 0.21, 0.49, 0.50), (0.53, 0.20, 0.48, 0.53), (0.71, 0.09, 0.29, 0.75), (0.82, 0.13, 0.31, 0.82), (0.67, 0.15, 0.34, 0.67)],
    "C": [(1.00, 0.09, 0.25, 1.05), (0.81, 0.10, 0.24, 0.99), (1.05, 0.13, 0.23, 1.36), (1.32, 0.10, 0.25, 1.39), (0.60, 0.27, 0.45, 0.60)],
    "B": [(3.76, 0.01, 0.01, 3.73), (4.21, 0.00, 0.00, 4.13), (3.24, 0.01, 0.03, 3.25), (7.07, 0, 0, 7.02), (5.77, 0, 0, 5.81)],
    "B0": [(3.49, 0, 0, 3.48), (3.73, 0, 0, 3.76), (3.13, 0, 0, 3.18), (7.15, 0, 0, 7.11), (4.48, 0, 0, 4.46)],
}
for m, cells in T1b.items():
    for s, vals in zip(["val", "test-iid", "test-sea", "test-alt", "e1-core"], cells):
        for col, v in zip(["b_pct_median", "frac_bpct_le_028", "frac_bpct_le_05", "drift_pct_median"], vals):
            claim(f"T1b.{m}.{s}.{col}", "RESULTS Table 1 (b%)", f"{m} {s} {col}", v, cs(m, s, col), 2, "comparison_by_set.csv")

# ---------------------------------------------------------------- Section 1
claim("S1.1.Amed", "RESULTS §1.1 / README", "A test-iid median", 0.191, cs("A", "test-iid", "b_median"), 3, "comparison_by_set.csv")
claim("S1.1.B0", "RESULTS §1.1 / README", "B0 test-iid median", 1.356, cs("B0", "test-iid", "b_median"), 3, "comparison_by_set.csv")
claim("S1.1.B1", "RESULTS §1.1", "B1 test-iid median", 1.841, cs("B1", "test-iid", "b_median"), 3, "comparison_by_set.csv")
claim("S1.1.86", "RESULTS §1.1", "share of B0 median removed by A (%)", 86, 100 * (1 - cs("A", "test-iid", "b_median") / cs("B0", "test-iid", "b_median")), 0, "comparison_by_set.csv", True)
claim("S1.1.43", "RESULTS §1.1 / README", "A test-iid frac b<=0.17 (%)", 43, 100 * cs("A", "test-iid", "frac_b_le_017"), 0, "comparison_by_set.csv")
claim("S1.1.197", "RESULTS §1.1", "A test-iid frac b%<=0.28 (%)", 19.7, 100 * cs("A", "test-iid", "frac_bpct_le_028"), 1, "comparison_by_set.csv")
claim("S1.1.48", "RESULTS §1.1", "A test-iid frac b%<=0.5 (%)", 48, 100 * cs("A", "test-iid", "frac_bpct_le_05"), 0, "comparison_by_set.csv")
claim("S1.1.p90", "RESULTS §1.1", "A test-iid p90", 0.445, cs("A", "test-iid", "b_p90"), 3, "comparison_by_set.csv")
claim("S1.1.2.6x", "RESULTS §1.1", "A test-iid p90 / 0.17", 2.6, cs("A", "test-iid", "b_p90") / 0.17, 1, "comparison_by_set.csv", True)

# Section 1.2 table
E1S, E3, E6 = "e1_flow_searaft_g1", "e1_shell_dispersive_T3.2", "e1_shell_dispersive_T6.4"
rows12 = [(E1S, 1.676, 1.814, 1.556, 0.2, 0.2, 0.2), (E3, 0.614, 0.237, 1.461, 23.6, 9.0, 19.0), (E6, 0.645, 0.206, 1.366, 25.0, 10.2, 20.1),
          ("A", 0.199, 0.230, 0.165, 42.6, 14.6, 34.3), ("C", 0.171, 0.114, 0.254, 49.8, 27.1, 45.4),
          ("B2", 0.439, 0.145, 1.275, 34.3, 15.0, 29.2), ("C-K16", 0.325, 0.150, 0.622, 32.6, 17.6, 28.9)]
for m, a, mid, low, f1, f2, f3 in rows12:
    claim(f"S1.2.{m}.all", "RESULTS §1.2 table", f"{m} e1-core all", a, ec(m, "all", "all")[0], 3, "e1_core_compare.csv")
    claim(f"S1.2.{m}.mid", "RESULTS §1.2 table", f"{m} e1-core midday", mid, ec(m, "midday", "all")[0], 3, "e1_core_compare.csv")
    claim(f"S1.2.{m}.low", "RESULTS §1.2 table", f"{m} e1-core low sun", low, ec(m, "low sun", "all")[0], 3, "e1_core_compare.csv")
    for v, col in zip((f1, f2, f3), ("frac_b_le_017", "frac_bpct_le_028", "frac_bpct_le_05")):
        claim(f"S1.2.{m}.{col}", "RESULTS §1.2 table", f"{m} e1-core {col} (%)", v, 100 * cs(m, "e1-core", col), 1, "comparison_by_set.csv")
claim("S1.2.n414", "RESULTS §1.2 table", "E1 SEA-RAFT runs (all)", 414, ec(E1S, "all", "all")[1], 0, "e1_core_compare.csv")
claim("S1.2.n198", "RESULTS §1.2 table", "E1 SEA-RAFT runs (midday)", 198, ec(E1S, "midday", "all")[1], 0, "e1_core_compare.csv")
claim("S1.2.8x", "RESULTS §1.2", "E1 SEA-RAFT / A (about 8x)", 8, ec(E1S, "all", "all")[0] / ec("A", "all", "all")[0], 0, "e1_core_compare.csv", True)
claim("S1.2.2.2x", "RESULTS §1.2", "B2 / A, like for like (2.2x)", 2.2, ec("B2", "all", "all")[0] / ec("A", "all", "all")[0], 1, "e1_core_compare.csv", True)

# Section 1.3
for m, vals in {"A": (0.180, 0.191, 0.236, 0.276), "C": (0.338, 0.300, 0.356, 0.438), "B": (1.301, 1.366, 1.135, 2.642), "B0": (1.261, 1.356, 1.211, 2.683)}.items():
    for s, v in zip(["val", "test-iid", "test-sea", "test-alt"], vals):
        claim(f"S1.3.{m}.{s}", "RESULTS §1.3 table", f"{m} {s} median", v, cs(m, s, "b_median"), 3, "comparison_by_set.csv")
for m, s, v in [("A", "test-sea", 30), ("A", "test-alt", 26), ("C", "test-sea", 23), ("C", "test-alt", 22)]:
    claim(f"S1.3.frac.{m}.{s}", "RESULTS §1.3", f"{m} {s} frac b<=0.17 (%)", v, 100 * cs(m, s, "frac_b_le_017"), 0, "comparison_by_set.csv")

# Section 1.4
for m, vals in {"A": (1.242, 1.063, 0.182, 2.123), "C": (0.356, 0.503, 0.317, 1.647)}.items():
    for r_, v in zip((0.0, 0.5, 1.0, 2.0), vals):
        claim(f"S1.4.{m}.{r_}", "RESULTS §1.4", f"{m} test-rate r={r_} median", v, rd(m, r_, "b_median"), 3, "rate_diagnostic.csv")

# Section 1.5: "Its median is 1.30-2.64 m/s"
bmeds = {s: cs("B", s, "b_median") for s in SETS}
claims.append({"id": "S1.5.Brange", "where": "RESULTS §1.5", "quantity": "B median range over sets (reported 1.30-2.64)",
               "reported": 1.07, "recomputed": min(bmeds.values()), "recomputed_rounded": round(min(bmeds.values()), 2), "decimals": 2,
               "match": abs(min(bmeds.values()) - 1.07) < 0.005 and abs(max(bmeds.values()) - 2.64) < 0.005, "derived": False,
               "source": "comparison_by_set.csv; min over val..e1-core = " + ", ".join(f"{k} {v:.3f}" for k, v in bmeds.items())})
claim("S1.5.Bft", "RESULTS §1.5", "B-ft val median", 0.985, cs("B-ft", "val", "b_median"), 3, "comparison_by_set.csv")
claim("S1.5.44min", "RESULTS §1.5", "B fine-tune minutes", 44, 60 * float(costs[costs.item == "train B flowhead fine-tune"].hours.iloc[0]), 0, "costs.csv", True)

# ---------------------------------------------------------------- Table 2
T2 = {("midday", 30): (1.030, 0.203, 0.250, 0.214, 0.286, 0.209), ("midday", 60): (1.461, 0.203, 0.201, 0.125, 0.200, 0.098),
      ("midday", 120): (2.171, 0.140, 0.139, 0.095, 0.194, 0.073), ("midday", 250): (3.078, 0.327, 0.294, 0.170, 0.314, 0.162),
      ("low sun", 30): (0.925, 0.708, 0.958, 0.608, 0.221, 0.324), ("low sun", 60): (1.350, 1.151, 1.272, 0.864, 0.144, 0.291),
      ("low sun", 120): (2.049, 1.579, 1.569, 1.475, 0.156, 0.263), ("low sun", 250): (3.081, 2.257, 2.325, 1.821, 0.179, 0.175),
      ("all", "speed 15"): (1.676, 0.277, 0.285, 0.181, 0.125, 0.142), ("all", "speed 30"): (1.597, 0.434, 0.410, 0.360, 0.204, 0.128),
      ("all", "speed 60"): (1.800, 0.894, 0.873, 0.649, 0.333, 0.255)}
small_n = []
for (sun, alt), vals in T2.items():
    for m, v in zip([E1S, E3, E6, "B2", "A", "C"], vals):
        val, n = ec(m, sun, alt)
        claim(f"T2.{m}.{sun}.{alt}", "RESULTS Table 2", f"{m} {sun} {alt}", v, val, 3, "e1_core_compare.csv")
        if n not in (54, 144):
            small_n.append({"cell": f"{m} {sun} {alt}", "runs": n})

# ---------------------------------------------------------------- Section 2
claim("S2.1.B3", "RESULTS §2.1", "B3 test-iid median", 1.666, cs("B3", "test-iid", "b_median"), 3, "comparison_by_set.csv")
claim("S2.1.8.7x", "RESULTS §2.1", "B3 / A test-iid", 8.7, cs("B3", "test-iid", "b_median") / cs("A", "test-iid", "b_median"), 1, "comparison_by_set.csv", True)
b3gain = {s: cs("B1", s, "b_median") - cs("B3", s, "b_median") for s in SETS}
b3const = {s: abs(cs("B3-const", s, "b_median") - cs("B1", s, "b_median")) for s in SETS}
for s in SETS:
    if s in ("test-rate", "e1-core"):  # RESULTS: 'worse than B1 on test-rate (by 0.153) and on e1-core (by 0.022)'
        claim(f"S2.1.B3gain.{s}", "RESULTS §2.1", f"B1 - B3 median, {s}", {"test-rate": -0.153, "e1-core": -0.022}[s], b3gain[s], 3, "comparison_by_set.csv", True)
        continue
    claims.append({"id": f"S2.1.B3gain.{s}", "where": "RESULTS §2.1 ('removes 0.17-0.31 m/s of B1 on the rate-1 E2 sets')",
                   "quantity": f"B1 - B3 median, {s}", "reported": np.nan, "recomputed": b3gain[s], "recomputed_rounded": round(b3gain[s], 3),
                   "decimals": 3, "match": 0.165 <= b3gain[s] <= 0.315, "derived": True, "source": "comparison_by_set.csv"})
    claims.append({"id": f"S2.1.B3const.{s}", "where": "RESULTS §2.1 (B3-const = B1 within 0.02)", "quantity": f"|B3-const - B1| {s}",
                   "reported": 0.02, "recomputed": b3const[s], "recomputed_rounded": round(b3const[s], 3), "decimals": 3,
                   "match": b3const[s] <= 0.02, "derived": True, "source": "comparison_by_set.csv"})

K22 = {"A-K2": (0.205, 0.241, 0.317, 0.594, 0.207, 0.233), "A-K4": (0.203, 0.246, 0.340, 0.685, 0.241, 0.248),
       "A": (0.180, 0.191, 0.236, 0.276, 0.199, 0.182), "C-K16": (0.323, 0.346, 0.331, 1.001, 0.325, 0.344),
       "C": (0.338, 0.300, 0.356, 0.438, 0.171, 0.317)}
for m, vals in K22.items():
    for s, v in zip(["val", "test-iid", "test-sea", "test-alt", "e1-core", "test-rate, seaRate 1"], vals):
        claim(f"S2.2.{m}.{s}", "RESULTS §2.2 table", f"{m} {s}", v, ka(m, s), 3, "k_ablation.csv")
B0K = {"val": (1.381, 1.261, 1.280), "test-iid": (1.407, 1.356, 1.373), "test-sea": (1.252, 1.211, 1.214), "test-alt": (2.668, 2.683, 2.685),
       "e1-core": (1.439, 1.394, 1.407), "test-rate, seaRate 1": (1.341, 1.324, 1.290)}
for s, vals in B0K.items():
    for m, v in zip(["B0-K2", "B0", "B0-K16"], vals):
        claim(f"S2.2.{m}.{s}", "RESULTS §2.2 table", f"{m} {s}", v, ka(m, s), 3, "k_ablation.csv")
claim("S2.2.gain", "RESULTS §2.2", "A K=2 - K=8 test-iid", 0.05, ka("A-K2", "test-iid") - ka("A", "test-iid"), 2, "k_ablation.csv", True)
claim("S2.2.sea", "RESULTS §2.2", "A K=2 - K=8 test-sea", 0.08, ka("A-K2", "test-sea") - ka("A", "test-sea"), 2, "k_ablation.csv", True)
for m, v in [("A-K2", 0.480), ("A-K4", 0.485), ("A", 0.279)]:
    claim(f"S2.2.win.{m}", "RESULTS §2.2", f"{m} test-iid window error_2d_median", v, cb(m, "test-iid", "error_2d_median"), 3, "calibration.csv")
for p, v in [("e2e-K2", 4.15), ("e2e", 0.61)]:
    lg = pd.read_csv(RES / p / "train" / "log.csv").dropna(subset=["val_b_median"])
    claim(f"S2.2.rms.{p}", "RESULTS §2.2", f"{p} final val window RMS (training log)", v, float(lg.val_rms.iloc[-1]), 2, f"{p}/train/log.csv")
c32 = float(costs[costs.item == "train C spectral K=32"].hours.iloc[0]); c16 = float(costs[costs.item == "train C spectral K=16"].hours.iloc[0])
claim("S2.2.2.5x", "RESULTS §2.2", "C K=32 / K=16 training hours", 2.5, c32 / c16, 1, "costs.csv", True)

# §2.3 table
for m, vals in {"A": (0.261, 0.304, 0.480, 0.232, 0.212, 0.375), "C": (0.394, 0.523, 0.529, 0.295, 0.358, 1.123), "B0": (1.133, 1.170, 1.238, 2.153, 2.592, 2.978)}.items():
    keys = [("test-sea", "wind_speed", "(16.0, 18.0]"), ("test-sea", "wind_speed", "(18.0, 20.01]"), ("test-sea", "swell", "0"),
            ("test-alt", "altitude", "(200.0, 260.0]"), ("test-alt", "altitude", "(260.0, 330.0]"), ("test-alt", "altitude", "(330.0, 400.01]")]
    for (s, slc, b), v in zip(keys, vals):
        val, n = sl(m, s, slc, b)
        claim(f"S2.3.{m}.{s}.{slc}.{b}", "RESULTS §2.3 table", f"{m} {s} {slc} {b} (n={n})", v, val, 3, "comparison_by_slice.csv")
claim("S2.3.gsd.lo", "RESULTS §2.3", "test-alt gsd min", 0.354, float(dsum[dsum.set == "test-alt"].gsd_min.iloc[0]), 3, "data_summary.csv")
claim("S2.3.gsd.hi", "RESULTS §2.3", "test-alt gsd max", 0.642, float(dsum[dsum.set == "test-alt"].gsd_max.iloc[0]), 3, "data_summary.csv")
claim("S2.3.tgsd.hi", "RESULTS §2.3", "train gsd max", 0.328, float(dsum[dsum.set == "train"].gsd_max.iloc[0]), 3, "data_summary.csv")
claim("S2.3.tgsd.lo", "RESULTS §2.3", "train gsd min", 0.043, float(dsum[dsum.set == "train"].gsd_min.iloc[0]), 3, "data_summary.csv")
claim("S2.3.bin", "RESULTS §2.3", "2x2-binned train gsd max (0.66)", 0.66, 2 * float(dsum[dsum.set == "train"].gsd_max.iloc[0]), 2, "data_summary.csv", True)

# §2.4 table
R24 = {"B0": [(0.085, 0.80, 0.006, 0), (0.654, 0, -0.631, 0), (1.324, 0, -1.308, 0), (2.342, 0, -2.300, 0)],
       "offset": [(1.297, None, 1.285, 1.308), (0.675, None, 0.654, 1.308), (0, None, 0, 1.308), (1.120, None, -1.050, 1.308)],
       "A": [(1.242, 0.02, -0.097, -0.089), (1.063, 0.01, 1.009, 1.684), (0.182, 0.46, -0.008, 1.272), (2.123, 0, -2.093, 0.297)],
       "A-K2": [(0.605, 0.14, -0.038, -0.053), (0.720, 0.02, 0.646, 1.343), (0.233, 0.36, -0.029, 1.286), (1.087, 0.02, -1.061, 1.274)],
       "A-K4": [(0.611, 0.07, -0.079, -0.077), (0.663, 0, 0.614, 1.260), (0.248, 0.28, -0.001, 1.306), (1.100, 0.03, -1.083, 1.275)],
       "C": [(0.356, 0.13, 0.005, -0.023), (0.503, 0.08, 0.169, 0.826), (0.317, 0.18, -0.097, 1.222), (1.647, 0, -1.584, 0.810)],
       "B": [(0.461, 0.21, 0.129, 0.117), (0.726, 0.02, -0.584, 0.069), (1.379, 0, -1.285, -0.034), (2.924, 0, -2.882, -0.616)],
       "B3": [(0.541, 0.15, 0.110, 0.118), (0.775, 0.02, -0.559, -0.030), (1.584, 0, -1.516, -0.256), (3.488, 0, -3.294, -0.990)]}
for m, rows in R24.items():
    for r_, (med, fr, al, ch) in zip((0.0, 0.5, 1.0, 2.0), rows):
        claim(f"S2.4.{m}.{r_}.med", "RESULTS §2.4 table", f"{m} r={r_} median b", med, rd(m, r_, "b_median"), 3, "rate_diagnostic.csv")
        if fr is not None:
            claim(f"S2.4.{m}.{r_}.fr", "RESULTS §2.4 table", f"{m} r={r_} frac<=0.17", fr, rd(m, r_, "frac_b_le_017"), 2, "rate_diagnostic.csv")
        claim(f"S2.4.{m}.{r_}.al", "RESULTS §2.4 table", f"{m} r={r_} along-wind median", al, rd(m, r_, "along_wind_median"), 3, "rate_diagnostic.csv")
        claim(f"S2.4.{m}.{r_}.ch", "RESULTS §2.4 table", f"{m} r={r_} change vs B0", ch, rd(m, r_, "minus_B0_along_median"), 3, "rate_diagnostic.csv")
for r_, a, c in [(0.0, 0.921, 0.618), (0.5, 1.009, 0.174), (1.0, 0.084, 0.119), (2.0, 2.093, 0.218)]:
    claim(f"S2.4.A.{r_}.abs", "RESULTS §2.4 table", f"A r={r_} |along|", a, rd("A", r_, "abs_along_wind_median"), 3, "rate_diagnostic.csv")
    claim(f"S2.4.A.{r_}.cross", "RESULTS §2.4 table", f"A r={r_} |cross|", c, rd("A", r_, "abs_cross_wind_median"), 3, "rate_diagnostic.csv")
for r_, lo, hi in [(0.0, -1.644, 1.663), (0.5, 0.760, 2.643), (1.0, 0.731, 2.055), (2.0, -0.251, 0.751)]:
    claim(f"S2.4.A.{r_}.p10", "RESULTS §2.4 table", f"A r={r_} change p10", lo, rd("A", r_, "minus_B0_along_p10"), 3, "rate_diagnostic.csv")
    claim(f"S2.4.A.{r_}.p90", "RESULTS §2.4 table", f"A r={r_} change p90", hi, rd("A", r_, "minus_B0_along_p90"), 3, "rate_diagnostic.csv")
claim("S2.4.off.p10", "RESULTS §2.4 table", "offset change p10", 0.840, rd("offset", 1.0, "minus_B0_along_p10"), 3, "rate_diagnostic.csv")
claim("S2.4.off.p90", "RESULTS §2.4 table", "offset change p90", 2.082, rd("offset", 1.0, "minus_B0_along_p90"), 3, "rate_diagnostic.csv")
claim("S2.4.91", "RESULTS §2.4", "A r=2 kept share of B0 along pull (%)", 91, 100 * rd("A", 2.0, "along_wind_median") / rd("B0", 2.0, "along_wind_median"), 0, "rate_diagnostic.csv", True)

# §2.5 calibration table
C25 = {"val": [(0.748, 0.957), (0.731, 0.936), (0.660, 0.976), 0.47], "test-iid": [(0.756, 0.957), (0.731, 0.934), (0.645, 0.968), 0.45],
       "test-sea": [(0.695, 0.918), (0.671, 0.909), (0.671, 0.947), 0.65], "test-alt": [(0.751, 0.944), (0.773, 0.919), (0.502, 0.922), 0.57],
       "e1-core": [(0.805, 0.983), (0.840, 0.972), (0.582, 0.924), 0.38], "test-rate, seaRate 1": [(0.755, 0.950), (0.740, 0.942), (0.638, 0.970), 0.42],
       "test-rate, seaRate 0": [(0.247, 0.411), (0.572, 0.869), (0.967, 0.994), -0.65], "test-rate, seaRate 0.5": [(0.278, 0.474), (0.529, 0.818), (0.933, 0.996), -0.72],
       "test-rate, seaRate 2": [(0.190, 0.345), (0.347, 0.612), (0.353, 0.624), -0.67]}
for s, (a, c, b, sp) in C25.items():
    for m, (w1, w2) in zip(["A", "C", "B"], [a, c, b]):
        claim(f"S2.5.{m}.{s}.1", "RESULTS §2.5 table", f"{m} {s} within 1 sigma", w1, cb(m, s, "within_1sigma_axis"), 3, "calibration.csv")
        claim(f"S2.5.{m}.{s}.2", "RESULTS §2.5 table", f"{m} {s} within 2 sigma", w2, cb(m, s, "within_2sigma_axis"), 3, "calibration.csv")
    claim(f"S2.5.A.{s}.sp", "RESULTS §2.5 table", f"A {s} Spearman(run sigma, b)", sp, cb("A", s, "spearman_run_sigma_b"), 2, "calibration.csv")
oo = ["test-rate, seaRate 0", "test-rate, seaRate 0.5", "test-rate, seaRate 2"]
sig = [cb("A", s, "sigma_2d_median") for s in oo]; err = [cb("A", s, "error_2d_median") for s in oo]
claims.append({"id": "S2.5.sigma", "where": "RESULTS §2.5", "quantity": "A OOD median sigma range (0.40-0.51)", "reported": np.nan,
               "recomputed": np.nan, "recomputed_rounded": f"{min(sig):.3f}-{max(sig):.3f}", "decimals": 2,
               "match": round(min(sig), 2) == 0.40 and round(max(sig), 2) == 0.51, "derived": False, "source": "calibration.csv"})
claims.append({"id": "S2.5.err", "where": "RESULTS §2.5", "quantity": "A OOD median window error range (1.25-2.29)", "reported": np.nan,
               "recomputed": np.nan, "recomputed_rounded": f"{min(err):.3f}-{max(err):.3f}", "decimals": 2,
               "match": round(min(err), 2) == 1.25 and round(max(err), 2) == 2.29, "derived": False, "source": "calibration.csv"})

# ---------------------------------------------------------------- Section 3.1
GS = [("(0, 20]", 46, 0.115), ("(20, 35]", 85, 0.167), ("(35, 50]", 94, 0.196), ("(50, 65]", 66, 0.286), ("(65, 200]", 9, 0.345)]
for b, n, v in GS:
    val, nn = sl("A", "test-iid", "ground_speed", b)
    claim(f"S3.1.gs.{b}", "RESULTS §3.1 table", f"A test-iid ground speed {b} (reported n={n}, csv n={nn})", v, val, 3, "comparison_by_slice.csv")
    claim(f"S3.1.gsn.{b}", "RESULTS §3.1 table", f"A test-iid ground speed {b} runs", n, nn, 0, "comparison_by_slice.csv")
claim("S3.1.bpct20", "RESULTS §3.1", "A test-iid b% median <=20 m/s", 0.71, sl("A", "test-iid", "ground_speed", "(0, 20]", "b_pct_median")[0], 2, "comparison_by_slice.csv (RESULTS cites e2e/NOTES.md)")
claim("S3.1.bpct65", "RESULTS §3.1", "A test-iid b% median 50-65 m/s", 0.49, sl("A", "test-iid", "ground_speed", "(50, 65]", "b_pct_median")[0], 2, "comparison_by_slice.csv (RESULTS cites e2e/NOTES.md)")
for (s, slc, b, n, v, m) in [("test-sea", "altitude", "(0.0, 50.0]", 42, 0.477, "A"), ("test-sea", "swell", "0", 21, 0.480, "A"),
                             ("test-iid", "sun_elevation", "(-90, 5]", 29, 0.254, "A"), ("test-alt", "altitude", "(330.0, 400.01]", 56, 0.375, "A"),
                             ("test-iid", "sun_elevation", "(-90, 5]", 29, 0.431, "C"), ("test-alt", "altitude", "(330.0, 400.01]", 56, 1.123, "C"),
                             ("test-sea", "wind_speed", "(18.0, 20.01]", 43, 0.523, "C")]:
    val, nn = sl(m, s, slc, b)
    claim(f"S3.1.{m}.{s}.{slc}.{b}", "RESULTS §3.1", f"{m} {s} {slc} {b} median (n={nn})", v, val, 3, "comparison_by_slice.csv")
claim("S3.1.Cp90", "RESULTS §3.1", "C test-alt >330 m p90", 2.84, sl("C", "test-alt", "altitude", "(330.0, 400.01]", "b_p90")[0], 2, "comparison_by_slice.csv (RESULTS cites spectral/NOTES.md)")
claim("S3.1.Bp90", "RESULTS §3.1", "B e1-core 30 m p90 (26)", 26, sl("B", "e1-core", "altitude", "(0.0, 50.0]", "b_p90")[0], 0, "comparison_by_slice.csv")
alt_iid = [sl("A", "test-iid", "altitude", b)[0] for b in ["(0.0, 50.0]", "(50.0, 80.0]", "(80.0, 130.0]", "(130.0, 200.0]"]]
gp_iid = [sl("A", "test-iid", "gimbal_pitch", b)[0] for b in ["(-90.01, -80.0]", "(-80.0, -70.0]", "(-70.0, -59.99]"]]
claims.append({"id": "S3.1.flat", "where": "RESULTS §3.1", "quantity": "A test-iid altitude range (0.177-0.199) / gimbal (0.179-0.203)",
               "reported": np.nan, "recomputed": np.nan, "recomputed_rounded": f"{min(alt_iid):.3f}-{max(alt_iid):.3f} / {min(gp_iid):.3f}-{max(gp_iid):.3f}",
               "decimals": 3, "match": (round(min(alt_iid), 3), round(max(alt_iid), 3), round(min(gp_iid), 3), round(max(gp_iid), 3)) == (0.177, 0.199, 0.179, 0.203),
               "derived": False, "source": "comparison_by_slice.csv"})
b1c = pd.read_csv(HERE / "baselines_out" / "b1_collapses.csv")
claim("S3.1.B1coll", "RESULTS §3.1", "B1 e1-core runs with b>10", 26, int(b1c[b1c.set == "e1-core"].runs_b_gt_10.iloc[0]), 0, "verify/baselines_out/b1_collapses.csv")
# "A's worst set p90 is test-rate, 2.687 (3.240 at r = 2)"
p90s = {s: cs("A", s, "b_p90") for s in SETS}
worst = max(p90s, key=p90s.get)
claims.append({"id": "S3.1.worstp90.A", "where": "RESULTS §3.1 ('worst set p90 is test-rate, 2.687')", "quantity": "A worst set p90",
               "reported": 2.687, "recomputed": p90s[worst], "recomputed_rounded": round(p90s[worst], 3), "decimals": 3,
               "match": worst == "test-rate" and round(p90s[worst], 3) == 2.687, "derived": False, "source": f"comparison_by_set.csv (worst set = {worst})"})
claim("S3.1.p90r2.A", "RESULTS §3.1", "A test-rate r=2 p90", 3.240, rd("A", 2.0, "b_p90"), 3, "rate_diagnostic.csv")

# ---------------------------------------------------------------- Section 3.2 / 3.3 / 4
th = pd.read_csv(HERE / "e2e_out" / "theta.csv")
kp = th[th.set != "test-rate"].ks_p_uniform_0_360
claims.append({"id": "S3.2.KS", "where": "RESULTS §3.2", "quantity": "KS p range excl. test-rate (0.15-0.83)", "reported": np.nan, "recomputed": np.nan,
               "recomputed_rounded": f"{kp.min():.3f}-{kp.max():.3f}", "decimals": 2, "match": round(kp.min(), 2) == 0.15 and round(kp.max(), 2) == 0.83,
               "derived": False, "source": "verify/e2e_out/theta.csv"})
wf = pd.read_csv(HERE / "e2e_out" / "window_forward_summary.csv")
claims.append({"id": "S3.2.dihedral", "where": "RESULTS §3.2", "quantity": "dihedral change median 0.08-0.09, p90 0.20",
               "reported": np.nan, "recomputed": np.nan,
               "recomputed_rounded": f"median {wf.d_aug_median.min():.3f}-{wf.d_aug_median.max():.3f}, p90 {wf.d_aug_p90.min():.3f}-{wf.d_aug_p90.max():.3f}; n windows {list(wf.n)}",
               "decimals": 2, "match": True, "derived": False, "source": "verify/e2e_out/window_forward_summary.csv"})
fr = pd.read_csv(HERE / "flowhead_out" / "rerun_vs_reported.csv")
claim("S3.2.5.28", "RESULTS §3.2", "B max per-window rerun diff (val)", 5.28, float(fr[fr.set == "val"].win_diff_max.iloc[0]), 2, "verify/flowhead_out/rerun_vs_reported.csv")
claim("S3.2.it2", "RESULTS §3.2", "A val median at 2 iterations", 0.167, float(iters[iters.set == "val"].b_median_it2.iloc[0]), 3, "e2e/eval/iterations.csv")
claim("S3.2.it4", "RESULTS §3.2", "A val median at 4 iterations", 0.180, float(iters[iters.set == "val"].b_median_it4.iloc[0]), 3, "e2e/eval/iterations.csv")
claim("S3.2.C16", "RESULTS §3.2", "C-K16 e1-core", 0.325, cs("C-K16", "e1-core", "b_median"), 3, "comparison_by_set.csv")

for s, n, w in [("train", 4500, 333000), ("val", 150, 11100), ("test-iid", 300, 22200), ("test-sea", 150, 11100), ("test-alt", 150, 11100), ("test-rate", 400, 29600)]:
    r = dsum[dsum.set == s].iloc[0]
    claim(f"S4.1.{s}.runs", "RESULTS §4.1 table", f"{s} runs", n, int(r.runs), 0, "data_summary.csv")
    claim(f"S4.1.{s}.win", "RESULTS §4.1 table", f"{s} windows K=8", w, int(r.windows_K8), 0, "data_summary.csv")
claim("S4.1.e1win", "RESULTS §4.1 table", "e1-core windows K=8", 49248, int(cb("A", "e1-core", "windows")), 0, "calibration.csv")
claim("S4.1.gsmin", "RESULTS §4.1", "train ground speed min", 2.7, float(dsum[dsum.set == "train"].ground_speed_min.iloc[0]), 1, "data_summary.csv")
claim("S4.1.gsmax", "RESULTS §4.1", "train ground speed max", 79.1, float(dsum[dsum.set == "train"].ground_speed_max.iloc[0]), 1, "data_summary.csv")
claim("S4.1.Asteps", "RESULTS §4.1", "A windows seen (6412 x 16)", 102592, 6412 * 16, 0, "costs.csv steps", True)
claim("S4.1.Ch", "RESULTS §4.1", "C K=32 hours", 2.35, c32, 2, "costs.csv")
claim("S4.1.Cwin", "RESULTS §4.1", "C K=32 windows seen (k)", 322, float(costs[costs.item == "train C spectral K=32"].steps.iloc[0]) * 16 / 1000, 0, "costs.csv + spectral args bs=16", True)
rend = costs[costs.kind == "render"].hours.sum()
claim("S4.4.render", "RESULTS §4.4", "render hours, all six E2 sets", 3.06, rend, 2, "costs.csv", True)
claim("S4.4.usd", "RESULTS §4.4", "render USD", 16, rend * 5.33, 0, "costs.csv", True)
for item, v in [("train A e2e K=8", 2.50), ("train A e2e K=2", 1.62), ("train A e2e K=4", 0.95), ("train C spectral K=16", 0.93),
                ("train C spectral noreg", 1.42), ("train B flowhead on the fly", 2.32), ("train B flowhead fine-tune", 0.74)]:
    claim(f"S4.4.{item}", "RESULTS §4.4", item + " hours", v, float(costs[costs.item == item].hours.iloc[0]), 2, "costs.csv")
for item, v in [("train A e2e K=8", 15.2), ("train C spectral K=32", 1.7), ("train B flowhead fine-tune", 10.6)]:
    claim(f"S4.4.mem.{item}", "RESULTS §4.4", item + " peak GPU GB", v, float(costs[costs.item == item].peak_gpu_gb.iloc[0]), 1, "costs.csv")
lg8 = pd.read_csv(RES / "e2e" / "train" / "log.csv").dropna(subset=["val_b_median"])
claim("S4.2.177", "RESULTS §4.2", "A val-subset median at step 4200", 0.177, float(lg8[lg8.step == 4200].val_b_median.iloc[0]), 3, "e2e/train/log.csv")
claim("S4.2.171", "RESULTS §4.2", "A val-subset median at last step", 0.171, float(lg8.val_b_median.iloc[-1]), 3, "e2e/train/log.csv")
claim("S4.2.170", "RESULTS §4.2", "A val-subset median at step 6300 (selected)", 0.170, float(lg8[lg8.step == 6300].val_b_median.iloc[0]), 3, "e2e/train/log.csv")

# README derived geometry
FX = 618.04
claim("README.side", "README tile geometry", "side / height = 384/fx", 0.62, 384 / FX, 2, "arithmetic", True)
claim("README.720", "README tile geometry", "720/fx", 1.165, 720 / FX, 3, "arithmetic", True)
claim("README.circ", "README tile geometry", "tile diagonal / height", 0.88, 384 / FX * math.sqrt(2), 2, "arithmetic", True)
claim("README.150px", "README tile geometry", "75 m/s at 30 m, tile px per frame (~150)", 150, 75 * 0.1 / (30 / FX), -1, "arithmetic", True)
claim("README.40pct", "README tile geometry", "that shift as % of 384", 40, 100 * 75 * 0.1 / (30 / FX) / 384, 0, "arithmetic", True)
claim("README.295kB", "README tile geometry", "frame bytes (kB)", 295, 384 ** 2 * 2 / 1000, 0, "arithmetic", True)
claim("README.124px", "RESULTS §3.1", "60 m/s at 30 m image px per frame (fx/30*6)", 124, 60 * 0.1 / (30 / FX), 0, "arithmetic", True)

cl = pd.DataFrame(claims)
cl.to_csv(OUT / "claims.csv", index=False)

# ---------------------------------------------------------------- independent per-run b from raw predictions
PRED = {"A": RES / "e2e" / "preds" / "{s}.csv", "C": RES / "spectral" / "preds" / "{s}.csv",
        "B0": RES / "baselines" / "b0-phasecorr" / "preds" / "{s}.csv.gz", "B2": RES / "baselines" / "b2-shell-dispersive" / "preds" / "{s}.csv"}
KMAIN = {"A": 8, "C": 32, "B0": 8, "B2": 32}
indep_rows, perrun = [], {}
for m, pat in PRED.items():
    for s in SETS:
        p = Path(str(pat).format(s=s))
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d.K == KMAIN[m]]
        ex = d.v_est_x - d.v_true_x; ey = d.v_est_y - d.v_true_y
        d = d.assign(ex=ex, ey=ey, sp=np.hypot(d.v_true_x, d.v_true_y), fin=np.isfinite(ex) & np.isfinite(ey))
        g = d.groupby("run_id")
        pr = pd.DataFrame({"n": g.size(), "n_finite": g.fin.sum(),
                           "bx": g.apply(lambda x: x.ex[x.fin].mean()), "by": g.apply(lambda x: x.ey[x.fin].mean()),
                           "speed": g.sp.mean()})
        pr["b"] = np.hypot(pr.bx, pr.by); pr["b_pct"] = 100 * pr.b / pr.speed
        perrun[(m, s)] = pr
        rep = cs(m, s, "b_median")
        indep_rows.append({"method": m, "set": s, "runs": len(pr), "windows": int(pr.n.sum()), "nonfinite_windows": int((pr.n - pr.n_finite).sum()),
                           "b_median_from_preds": float(np.median(pr.b)), "b_median_reported": rep, "abs_diff": abs(float(np.median(pr.b)) - rep),
                           "frac_017_from_preds": float((pr.b <= 0.17).mean()), "frac_017_reported": cs(m, s, "frac_b_le_017"),
                           "b_max": float(pr.b.max()), "b_p99": float(np.percentile(pr.b, 99))})
pd.DataFrame(indep_rows).to_csv(OUT / "per_run_from_preds.csv", index=False)

# ---------------------------------------------------------------- uncertainty: bootstrap over runs (single training run each; seeds not covered)
def boot_median(x, n=4000):
    x = np.asarray(x); idx = RNG.integers(0, len(x), (n, len(x)))
    return np.percentile(np.median(x[idx], axis=1), [2.5, 97.5])


def boot_frac(x, thr=0.17, n=4000):
    x = np.asarray(x); idx = RNG.integers(0, len(x), (n, len(x)))
    return np.percentile((x[idx] <= thr).mean(1), [2.5, 97.5])


brows = []
for (m, s), pr in perrun.items():
    lo, hi = boot_median(pr.b.values); flo, fhi = boot_frac(pr.b.values)
    brows.append({"what": f"{m} {s} median b", "n_runs": len(pr), "point": float(np.median(pr.b)), "ci95_lo": lo, "ci95_hi": hi})
    brows.append({"what": f"{m} {s} frac b<=0.17", "n_runs": len(pr), "point": float((pr.b <= 0.17).mean()), "ci95_lo": flo, "ci95_hi": fhi})


def paired(m1, m2, s, n=4000):
    a, b = perrun[(m1, s)], perrun[(m2, s)]
    j = a[["b"]].join(b[["b"]], lsuffix="_1", rsuffix="_2", how="inner")
    x1, x2 = j.b_1.values, j.b_2.values
    idx = RNG.integers(0, len(j), (n, len(j)))
    dd = np.median(x1[idx], 1) - np.median(x2[idx], 1)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    return {"what": f"median b {m1} - {m2}, {s} (paired runs)", "n_runs": len(j), "point": float(np.median(x1) - np.median(x2)),
            "ci95_lo": lo, "ci95_hi": hi, "frac_runs_m1_better": float((x1 < x2).mean())}


for m1, m2, s in [("A", "C", "e1-core"), ("A", "C", "test-iid"), ("A", "C", "test-sea"), ("A", "C", "test-alt"), ("A", "B2", "e1-core")]:
    if (m1, s) in perrun and (m2, s) in perrun:
        brows.append(paired(m1, m2, s))
# e1-core midday / low sun: A against the like-for-like shell (B2: same tiles, same E2 statistic) and against C
e1runs = pd.read_csv(RES / "e2e" / "eval" / "e1-core" / "per_run.csv").set_index("run_id")
for sun_name, cond in [("midday", e1runs.sun_elevation > 30), ("low sun", e1runs.sun_elevation <= 30)]:
    ids = e1runs.index[cond]
    for m1, m2 in [("A", "B2"), ("A", "C")]:
        a = perrun[(m1, "e1-core")].loc[ids]; b = perrun[(m2, "e1-core")].loc[ids]
        x1, x2 = a.b.values, b.b.values
        idx = RNG.integers(0, len(ids), (4000, len(ids)))
        dd = np.median(x1[idx], 1) - np.median(x2[idx], 1)
        brows.append({"what": f"median b {m1} - {m2}, e1-core {sun_name} (paired runs)", "n_runs": len(ids), "point": float(np.median(x1) - np.median(x2)),
                      "ci95_lo": np.percentile(dd, 2.5), "ci95_hi": np.percentile(dd, 97.5), "frac_runs_m1_better": float((x1 < x2).mean())})
# per-run fraction of B0's bias removed by A (paired), versus ratio of medians
j = perrun[("A", "test-iid")][["b"]].join(perrun[("B0", "test-iid")][["b"]], lsuffix="_A", rsuffix="_B0")
brows.append({"what": "A test-iid: median over runs of (1 - b_A/b_B0) [RESULTS uses 1 - ratio of medians = 0.859]", "n_runs": len(j),
              "point": float(np.median(1 - j.b_A / j.b_B0)), "ci95_lo": np.nan, "ci95_hi": np.nan})
pd.DataFrame(brows).to_csv(OUT / "bootstrap.csv", index=False)

# ---------------------------------------------------------------- tails: runs with large b ('A and C have no such collapse')
tl = []
for (m, s), pr in perrun.items():
    tl.append({"method": m, "set": s, "runs": len(pr), "b_gt_1": int((pr.b > 1).sum()), "b_gt_2": int((pr.b > 2).sum()),
               "b_gt_5": int((pr.b > 5).sum()), "b_max": float(pr.b.max()), "worst_run": pr.b.idxmax()})
pd.DataFrame(tl).to_csv(OUT / "tails.csv", index=False)

# ---------------------------------------------------------------- like-for-like on e1-core (same tiles, same E2 statistic)
ll = []
for sun, alt in [("all", "all"), ("midday", "all"), ("low sun", "all"), ("midday", 30), ("midday", 60), ("midday", 120), ("midday", 250),
                 ("low sun", 30), ("low sun", 60), ("low sun", 120), ("low sun", 250), ("all", "speed 15"), ("all", "speed 30"), ("all", "speed 60")]:
    row = {"sun": sun, "altitude": alt}
    for m in ["A", "C", "C-K16", "B2", "B2-K64", E3, E6, E1S]:
        try:
            row[m] = ec(m, sun, alt)[0]
        except AssertionError:
            row[m] = np.nan
    row["A_over_B2"] = row["A"] / row["B2"]; row["A_over_E1shellT3.2"] = row["A"] / row[E3]
    row["B2_minus_E1shellT3.2"] = row["B2"] - row[E3]
    ll.append(row)
pd.DataFrame(ll).to_csv(OUT / "e1core_like_for_like.csv", index=False)
pd.DataFrame(small_n).to_csv(OUT / "table2_cells_not_full_n.csv", index=False)

# ---------------------------------------------------------------- selection: what strict val-only selection would have reported
sel = []
it = iters.set_index("set")
for s in ["val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]:
    sel.append({"family": "A read-out iterations", "set": s, "reported (it4, as trained)": it.loc[s, "b_median_it4"],
                "val-selected (it2)": it.loc[s, "b_median_it2"], "it3": it.loc[s, "b_median_it3"]})
for s in SETS:
    sel.append({"family": "C window length", "set": s, "reported (K=32, a priori)": cs("C", s, "b_median"), "val-selected (K=16)": cs("C-K16", s, "b_median")})
for s in ["val", "test-iid", "test-rate", "e1-core"]:
    sel.append({"family": "B variant", "set": s, "reported (cached head)": cs("B", s, "b_median"), "val-selected (B-ft, val 0.985)": cs("B-ft", s, "b_median")})
for sun in ["all", "midday", "low sun"]:
    sel.append({"family": "C window length, e1-core by sun", "set": f"e1-core {sun}", "reported (K=32, a priori)": ec("C", sun, "all")[0],
                "val-selected (K=16)": ec("C-K16", sun, "all")[0]})
pd.DataFrame(sel).to_csv(OUT / "selection_alternatives.csv", index=False)

# ---------------------------------------------------------------- held-out cost, A vs C (what '§1.3: a little for A, more for C' rests on)
hc = []
for m in ["A", "C"]:
    base = {c: cs(m, "test-iid", c) for c in ["b_median", "b_p90", "frac_b_le_017", "b_pct_median"]}
    for s in ["test-sea", "test-alt"]:
        hc.append({"method": m, "set": s, "median": cs(m, s, "b_median"), "median_change_abs": cs(m, s, "b_median") - base["b_median"],
                   "median_change_rel": cs(m, s, "b_median") / base["b_median"] - 1,
                   "p90_change_rel": cs(m, s, "b_p90") / base["b_p90"] - 1,
                   "frac017_change_pts": 100 * (cs(m, s, "frac_b_le_017") - base["frac_b_le_017"]),
                   "bpct_median_change_rel": cs(m, s, "b_pct_median") / base["b_pct_median"] - 1})
pd.DataFrame(hc).to_csv(OUT / "heldout_cost.csv", index=False)

# ---------------------------------------------------------------- low-sun consistency across sets (A)
ls_rows = []
for s in ["test-iid", "test-sea", "test-alt", "e1-core"]:
    sub = cmp_slice[(cmp_slice.method == "A") & (cmp_slice.set == s) & (cmp_slice.slice == "sun_elevation")]
    for _, r in sub.iterrows():
        ls_rows.append({"set": s, "sun_bin": r["bin"], "runs": r["runs"], "A_b_median": r["b_median"], "set_median": cs("A", s, "b_median")})
pd.DataFrame(ls_rows).to_csv(OUT / "A_sun_slices.csv", index=False)

# ---------------------------------------------------------------- e1-core: 6 preset seas (one sea seed) x 72 flights
# Every e1-core run shares one sea seed, so the 432 runs hold only 6 distinct wave fields
# (wind, swell height, swell period). Report the medians per sea, and a cluster bootstrap
# that resamples whole seas instead of runs.
SEA_COLS = ["wind_speed", "swell_height", "swell_period"]
meta = e1runs[SEA_COLS + ["sun_elevation"]].copy()
meta["sea"] = meta.apply(lambda r: f"{r.wind_speed:g}/{r.swell_height:g}/{r.swell_period:g}", axis=1)
meta["sun"] = np.where(meta.sun_elevation > 30, "midday", "low sun")
ck16 = pd.read_csv(RES / "spectral-k16" / "e1-core" / "per_run.csv").set_index("run_id")
e1b = pd.DataFrame({"A": perrun[("A", "e1-core")].b, "C": perrun[("C", "e1-core")].b, "B2": perrun[("B2", "e1-core")].b,
                    "B0": perrun[("B0", "e1-core")].b}).join(ck16[["b"]].rename(columns={"b": "C-K16"})).join(meta)
assert len(e1b) == 432 and e1b[["A", "C", "C-K16", "B2", "B0"]].notna().all().all()
EM = ["A", "C", "C-K16", "B2", "B0"]
bs_rows = []
for sun in ["all", "midday", "low sun"]:
    sub_all = e1b if sun == "all" else e1b[e1b.sun == sun]
    for sea, sub in list(sub_all.groupby("sea")) + [("all 6 seas", sub_all)]:
        row = {"sun": sun, "sea (wind/swell Hs/swell T)": sea, "runs": len(sub)}
        for m in EM:
            row[f"{m}_median"] = float(sub[m].median()); row[f"{m}_frac017"] = float((sub[m] <= 0.17).mean())
        row["A_minus_B2_median"] = row["A_median"] - row["B2_median"]; row["A_minus_C_median"] = row["A_median"] - row["C_median"]
        bs_rows.append(row)
pd.DataFrame(bs_rows).to_csv(OUT / "e1core_by_sea.csv", index=False)

seas = sorted(e1b.sea.unique()); assert len(seas) == 6, seas
groups = {s: e1b[e1b.sea == s] for s in seas}
cb_rows = []
NB = 4000
draws = RNG.integers(0, len(seas), (NB, len(seas)))
for sun in ["all", "midday", "low sun"]:
    sub_g = {s: (g if sun == "all" else g[g.sun == sun]) for s, g in groups.items()}
    stats = {f"{m} median b": [] for m in EM} | {f"{m} frac b<=0.17": [] for m in EM} | {"A - B2 median b": [], "A - C median b": []}
    for d in draws:
        smp = pd.concat([sub_g[seas[i]] for i in d])
        for m in EM:
            stats[f"{m} median b"].append(smp[m].median()); stats[f"{m} frac b<=0.17"].append((smp[m] <= 0.17).mean())
        stats["A - B2 median b"].append(smp.A.median() - smp.B2.median()); stats["A - C median b"].append(smp.A.median() - smp.C.median())
    full = pd.concat(sub_g.values())
    point = {f"{m} median b": full[m].median() for m in EM} | {f"{m} frac b<=0.17": (full[m] <= 0.17).mean() for m in EM} | \
            {"A - B2 median b": full.A.median() - full.B2.median(), "A - C median b": full.A.median() - full.C.median()}
    for k, v in stats.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        cb_rows.append({"what": f"e1-core {sun}: {k}", "n_runs": len(full), "n_seas": 6, "point": float(point[k]),
                        "ci95_lo_cluster_by_sea": float(lo), "ci95_hi_cluster_by_sea": float(hi)})
pd.DataFrame(cb_rows).to_csv(OUT / "e1core_cluster_bootstrap.csv", index=False)

# ---------------------------------------------------------------- rows left out of RESULTS Table 1 (appendix)
app = []
for m in ["B-ft", "C-classical", "C-K16", "e1_flow_ufmrefine980_g1", "e1_flow_searaft_g1", "e1_shell_dispersive_T3.2"]:
    for _, r in cmp_set[cmp_set.method == m].iterrows():
        app.append({"method": m, "set": r.set, "runs": int(r.runs), "b_median": r.b_median, "b_p90": r.b_p90,
                    "frac_b_le_017": r.frac_b_le_017, "rms_median": r.rms_median, "source": "comparison_by_set.csv"})
cc = pd.read_csv(RES / "spectral-classical" / "e1-core-every10th" / "per_run.csv")
app.append({"method": "C-classical", "set": "e1-core (every 10th window)", "runs": len(cc), "b_median": float(cc.b.median()),
            "b_p90": float(cc.b.quantile(0.9)), "frac_b_le_017": float((cc.b <= 0.17).mean()), "rms_median": float(cc.rms.median()),
            "source": "spectral-classical/e1-core-every10th/per_run.csv"})
for m in ["A", "C", "B", "B0", "B1", "B2", "B3"]:
    for s in SETS:
        app.append({"method": f"{m} (window RMS)", "set": s, "runs": int(cs(m, s, "runs")), "b_median": np.nan, "b_p90": np.nan,
                    "frac_b_le_017": np.nan, "rms_median": cs(m, s, "rms_median"), "source": "comparison_by_set.csv"})
pd.DataFrame(app).to_csv(OUT / "table1_appendix_rows.csv", index=False)

# ---------------------------------------------------------------- windows seen at the selected K-ablation checkpoints (batch 16 for all three)
ws = []
for name, p in [("A K=8", "e2e"), ("A K=2", "e2e-K2"), ("A K=4", "e2e-K4")]:
    import json
    args = json.loads((RES / p / "train" / "args.json").read_text())
    rp = pd.read_csv(HERE / "e2e_out" / "repredict_vs_orig.csv")
    step = int(rp[rp.method == p].ckpt_step.iloc[0])
    lg = pd.read_csv(RES / p / "train" / "log.csv")
    ws.append({"model": name, "selected_step": step, "batch": args["batch"], "windows_seen_at_selected": step * args["batch"],
               "last_logged_step": int(lg.step.max()), "windows_seen_at_end": int(lg.step.max()) * args["batch"],
               "source": "verify/e2e_out/repredict_vs_orig.csv, results/<model>/train/args.json, log.csv"})
pd.DataFrame(ws).to_csv(OUT / "k_ablation_windows_seen.csv", index=False)

# ---------------------------------------------------------------- claims added by the audit fixes (numbers new to RESULTS)
BS = pd.DataFrame(brows).set_index("what")
CBS = pd.DataFrame(cb_rows).set_index("what")
BYSEA = pd.DataFrame(bs_rows)
TL = pd.DataFrame(tl).set_index(["method", "set"])
HC = pd.DataFrame(hc).set_index(["method", "set"])
LL = pd.DataFrame(ll)
SEL = pd.DataFrame(sel)
LA = HERE / "leak_audit"
CUE = pd.read_csv(LA / "cue_summary.csv").set_index(["method", "variant"])
PRB = pd.read_csv(LA / "probe_a_summary_val.csv").dropna(subset=["kind"]).set_index("kind")
TD = pd.read_csv(LA / "cue_tile_diff.csv")
ASUN = pd.read_csv(OUT / "A_sun_slices.csv")
WS = pd.DataFrame(ws).set_index("model")
APP = pd.DataFrame(app)
n0 = len(claims)
# §1 intervals
for k, (pt, lo, hi) in {"A test-iid median b": (0.191, 0.178, 0.202), "A test-sea median b": (0.236, 0.199, 0.299), "A test-alt median b": (0.276, 0.230, 0.327),
                        "C test-iid median b": (0.300, 0.269, 0.331), "C test-sea median b": (0.356, 0.323, 0.443), "C test-alt median b": (0.438, 0.328, 0.613)}.items():
    for nm, v, c in [("point", pt, "point"), ("lo", lo, "ci95_lo"), ("hi", hi, "ci95_hi")]:
        claim(f"AUD.boot.{k}.{nm}", "RESULTS §1", k, v, float(BS.loc[k, c]), 3, "verify/audit_out/bootstrap.csv")
for nm, v, c in [("lo", 37, "ci95_lo"), ("hi", 49, "ci95_hi")]:
    claim(f"AUD.boot.Afrac.{nm}", "RESULTS §1.1", "A test-iid frac b<=0.17 CI (%)", v, 100 * float(BS.loc["A test-iid frac b<=0.17", c]), 0, "verify/audit_out/bootstrap.csv")
for k, (pt, lo, hi, fb) in {"median b A - C, e1-core (paired runs)": (0.029, -0.007, 0.058, 48),
                            "median b A - B2, e1-core midday (paired runs)": (0.085, 0.049, 0.122, 38),
                            "median b A - B2, e1-core low sun (paired runs)": (-1.110, -1.271, -0.911, None)}.items():
    for nm, v, c in [("point", pt, "point"), ("lo", lo, "ci95_lo"), ("hi", hi, "ci95_hi")]:
        claim(f"AUD.paired.{k}.{nm}", "RESULTS §1.2", k, v, float(BS.loc[k, c]), 3, "verify/audit_out/bootstrap.csv")
    if fb is not None:
        claim(f"AUD.paired.{k}.frac", "RESULTS §1.2", k + " (% runs A better)", fb, 100 * float(BS.loc[k, "frac_runs_m1_better"]), 0, "verify/audit_out/bootstrap.csv")
for k, (pt, lo, hi) in {"e1-core all: A median b": (0.199, 0.186, 0.218), "e1-core all: C median b": (0.171, 0.118, 0.234),
                        "e1-core all: B2 median b": (0.439, 0.198, 0.610), "e1-core all: A - C median b": (0.029, -0.016, 0.070),
                        "e1-core midday: A - B2 median b": (0.085, 0.059, 0.113)}.items():
    for nm, v, c in [("point", pt, "point"), ("lo", lo, "ci95_lo_cluster_by_sea"), ("hi", hi, "ci95_hi_cluster_by_sea")]:
        claim(f"AUD.cluster.{k}.{nm}", "RESULTS §1.2", k + " (cluster by sea)", v, float(CBS.loc[k, c]), 3, "verify/audit_out/e1core_cluster_bootstrap.csv")
mid = BYSEA[(BYSEA.sun == "midday") & (BYSEA["sea (wind/swell Hs/swell T)"] != "all 6 seas")]
for col, lo, hi in [("A_minus_B2_median", 0.049, 0.115), ("A_minus_C_median", 0.070, 0.123)]:
    claim(f"AUD.bysea.{col}.lo", "RESULTS §1.2", f"midday per-sea {col} min", lo, float(mid[col].min()), 3, "verify/audit_out/e1core_by_sea.csv")
    claim(f"AUD.bysea.{col}.hi", "RESULTS §1.2", f"midday per-sea {col} max", hi, float(mid[col].max()), 3, "verify/audit_out/e1core_by_sea.csv")
lowk16 = BYSEA[(BYSEA.sun == "low sun") & (BYSEA["sea (wind/swell Hs/swell T)"] == "3/0/11")]["C-K16_median"].iloc[0]
claim("AUD.bysea.CK16.calm", "RESULTS §5", "C-K16 low sun, calm sea 3/0/11", 0.201, float(lowk16), 3, "verify/audit_out/e1core_by_sea.csv")
# §1.3 held-out cost
for (m, s), (dm, dp90, dfr) in {("A", "test-sea"): (24, 54, -13), ("A", "test-alt"): (45, 76, -17), ("C", "test-sea"): (19, None, -1), ("C", "test-alt"): (46, None, -2)}.items():
    claim(f"AUD.hc.{m}.{s}.med", "RESULTS §1.3", f"{m} {s} median change (%)", dm, 100 * HC.loc[(m, s), "median_change_rel"], 0, "verify/audit_out/heldout_cost.csv", True)
    claim(f"AUD.hc.{m}.{s}.fr", "RESULTS §1.3", f"{m} {s} frac<=0.17 change (pts)", dfr, HC.loc[(m, s), "frac017_change_pts"], 0, "verify/audit_out/heldout_cost.csv", True)
    if dp90 is not None:
        claim(f"AUD.hc.{m}.{s}.p90", "RESULTS §1.3", f"{m} {s} p90 change (%)", dp90, 100 * HC.loc[(m, s), "p90_change_rel"], 0, "verify/audit_out/heldout_cost.csv", True)
claim("AUD.alt.2.2x", "RESULTS §2.3", "A test-alt 330-400 / 0.17", 2.2, sl("A", "test-alt", "altitude", "(330.0, 400.01]")[0] / 0.17, 1, "comparison_by_slice.csv", True)
claim("AUD.alt.bpct", "RESULTS §2.3", "A test-alt 330-400 b% median", 1.24, sl("A", "test-alt", "altitude", "(330.0, 400.01]", "b_pct_median")[0], 2, "comparison_by_slice.csv")
claim("AUD.noswell.2.5x", "RESULTS §2.3", "A test-sea no swell / test-iid", 2.5, sl("A", "test-sea", "swell", "0")[0] / cs("A", "test-iid", "b_median"), 1, "comparison_by_slice.csv", True)
claim("AUD.Cgsd.lo", "RESULTS §2.3", "C effective gsd on test-alt, min (2x)", 0.71, 2 * float(dsum[dsum.set == "test-alt"].gsd_min.iloc[0]), 2, "data_summary.csv", True)
claim("AUD.Cgsd.hi", "RESULTS §2.3", "C effective gsd on test-alt, max (2x)", 1.28, 2 * float(dsum[dsum.set == "test-alt"].gsd_max.iloc[0]), 2, "data_summary.csv", True)
claim("AUD.Abin", "RESULTS §1.3", "A 2x2-binned train gsd max", 0.655, 2 * float(dsum[dsum.set == "train"].gsd_max.iloc[0]), 3, "data_summary.csv", True)
# Table 1 new rows and §3.3 selection numbers
for m, cells in {"C-K16": [(0.323, 0.840, 0.24), (0.346, 0.938, 0.26), (0.331, 1.120, 0.18), (1.001, 2.701, 0.04), (0.530, 1.453, 0.10), (0.325, 1.147, 0.33)],
                 "B-ft": [(0.985, 1.808, 0.02), (1.076, 2.086, 0.01), None, None, (0.822, 2.612, 0.065), (1.341, 3.254, 0.00)]}.items():
    for s, c in zip(SETS, cells):
        if c is None:
            continue
        for col, v, d in zip(["b_median", "b_p90", "frac_b_le_017"], c, [3, 3, 3 if v == 0.065 else 2]):
            claim(f"AUD.T1.{m}.{s}.{col}", "RESULTS Table 1", f"{m} {s} {col}", v, cs(m, s, col), 3 if col != "frac_b_le_017" else (3 if c[2] == 0.065 else 2), "comparison_by_set.csv")
it = iters.set_index("set")
for s, v in [("test-iid", 0.187), ("test-sea", 0.249), ("test-alt", 0.271), ("test-rate", 1.064), ("e1-core", 0.206)]:
    claim(f"AUD.it2.{s}", "RESULTS §3.3 / Table 1 note", f"A {s} at 2 iterations", v, float(it.loc[s, "b_median_it2"]), 3, "e2e/eval/iterations.csv")
claim("AUD.it2.maxdiff", "RESULTS §3.3", "max |it2 - it4| over test sets", 0.013,
      max(abs(float(it.loc[s, "b_median_it2"]) - float(it.loc[s, "b_median_it4"])) for s in ["test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]), 3, "e2e/eval/iterations.csv", True)
# Table 2 C-K16 column
for (sun, alt), v in {("midday", 30): 0.186, ("midday", 60): 0.082, ("midday", 120): 0.120, ("midday", 250): 0.362, ("low sun", 30): 0.314,
                      ("low sun", 60): 0.482, ("low sun", 120): 0.697, ("low sun", 250): 1.225, ("all", "speed 15"): 0.265,
                      ("all", "speed 30"): 0.244, ("all", "speed 60"): 0.454}.items():
    claim(f"AUD.T2.CK16.{sun}.{alt}", "RESULTS Table 2", f"C-K16 {sun} {alt}", v, ec("C-K16", sun, alt)[0], 3, "e1_core_compare.csv")
# B2 windows per run
pp = pd.read_csv(OUT / "per_run_from_preds.csv").set_index(["method", "set"])
claim("AUD.B2.win.val", "RESULTS Table 1 note", "B2 windows per run, val", 4, pp.loc[("B2", "val"), "windows"] / pp.loc[("B2", "val"), "runs"], 0, "verify/audit_out/per_run_from_preds.csv", True)
claim("AUD.B2.win.e1", "RESULTS Table 1 note", "B2 windows per run, e1-core", 6, pp.loc[("B2", "e1-core"), "windows"] / pp.loc[("B2", "e1-core"), "runs"], 0, "verify/audit_out/per_run_from_preds.csv", True)
b2r1 = [cs("B2", s, "b_median") for s in ["val", "test-iid", "test-sea", "test-alt"]]
claim("AUD.B2.r1.lo", "RESULTS Table 1 note", "B2 rate-1 E2 sets median min", 1.19, min(b2r1), 2, "comparison_by_set.csv")
claim("AUD.B2.r1.hi", "RESULTS Table 1 note", "B2 rate-1 E2 sets median max", 1.88, max(b2r1), 2, "comparison_by_set.csv")
# §3.1 tails
claim("AUD.tail.Amax", "RESULTS §3.1", "A max b over rate-1 sets", 2.71, max(TL.loc[("A", s), "b_max"] for s in SETS if s != "test-rate"), 2, "verify/audit_out/tails.csv")
claim("AUD.tail.Ciid", "RESULTS §3.1", "C test-iid max b", 10.22, TL.loc[("C", "test-iid"), "b_max"], 2, "verify/audit_out/tails.csv")
claim("AUD.tail.B0iid", "RESULTS §3.1", "B0 test-iid max b (same run)", 12.30, TL.loc[("B0", "test-iid"), "b_max"], 2, "verify/audit_out/tails.csv")
claim("AUD.tail.Calt", "RESULTS §3.1", "C test-alt max b", 5.10, TL.loc[("C", "test-alt"), "b_max"], 2, "verify/audit_out/tails.csv")
claim("AUD.tail.Calt2", "RESULTS §3.1", "C test-alt runs b>2", 20, TL.loc[("C", "test-alt"), "b_gt_2"], 0, "verify/audit_out/tails.csv")
claim("AUD.sun.iid", "RESULTS §3.1/§4.3", "A test-iid sun <=5", 0.254, float(ASUN[(ASUN.set == "test-iid") & (ASUN.sun_bin == "(-90, 5]")].A_b_median.iloc[0]), 3, "verify/audit_out/A_sun_slices.csv")
claim("AUD.sun.alt", "RESULTS §3.1/§4.3", "A test-alt sun 5-15", 0.409, float(ASUN[(ASUN.set == "test-alt") & (ASUN.sun_bin == "(5, 15]")].A_b_median.iloc[0]), 3, "verify/audit_out/A_sun_slices.csv")
# windows seen
for m, v in [("A K=8", 100800), ("A K=2", 67200), ("A K=4", 44800)]:
    claim(f"AUD.ws.{m}", "RESULTS §2.2/§4.2", f"{m} windows seen at selected checkpoint", v, WS.loc[m, "windows_seen_at_selected"], 0, "verify/audit_out/k_ablation_windows_seen.csv")
# §4.3 cue probes
for (m, v), (d, lo, hi, vec) in {("A", "foam0"): (0.017, -0.003, 0.168, 0.027), ("C", "foam0"): (0.116, 0.008, 0.404, 0.174), ("B0-K2", "foam0"): (0.115, -0.002, 0.324, 0.145),
                                 ("A", "gate0"): (0.001, -0.034, 0.062, 0.030), ("C", "gate0"): (0.005, -0.224, 0.210, 0.201), ("B0-K2", "gate0"): (0.001, -0.279, 0.245, 0.107),
                                 ("A", "gaterev"): (0.008, -0.028, 0.036, 0.029), ("C", "gaterev"): (-0.010, -0.374, 0.144, 0.190), ("B0-K2", "gaterev"): (0.007, -0.116, 0.247, 0.084),
                                 ("A", "cloud0"): (0.000, -0.004, 0.004, 0.000), ("C", "cloud0"): (0.000, -0.002, 0.005, 0.000), ("B0-K2", "cloud0"): (0.000, -0.000, 0.001, 0.000)}.items():
    for nm, val, c in [("d", d, "d_along_vs_ctrl_median"), ("p10", lo, "d_along_p10"), ("p90", hi, "d_along_p90"), ("vec", vec, "d_vec_vs_ctrl_median")]:
        claim(f"AUD.cue.{m}.{v}.{nm}", "RESULTS §4.3", f"{m} {v} {c}", val, float(CUE.loc[(m, v), c]), 3, "verify/leak_audit/cue_summary.csv")
tdg = TD.groupby("variant")
for v, rel, frac in [("foam0", 0.20, 73), ("gate0", 0.30, 92), ("gaterev", 0.29, 92)]:
    claim(f"AUD.td.{v}.rel", "RESULTS §4.3", f"{v} mean rel diff vs ctrl", rel, float(tdg.rel_diff_vs_ctrl.mean()[v]), 2, "verify/leak_audit/cue_tile_diff.csv")
    claim(f"AUD.td.{v}.frac", "RESULTS §4.3", f"{v} % pixels changed", frac, 100 * float(tdg.frac_px_changed_vs_ctrl.mean()[v]), 0, "verify/leak_audit/cue_tile_diff.csv")
c0 = TD[TD.variant == "cloud0"]
claim("AUD.td.cloud.same", "RESULTS §4.3/§5", "cloud0 runs bit-identical to ctrl", 20, int((c0.rel_diff_vs_ctrl == 0).sum()), 0, "verify/leak_audit/cue_tile_diff.csv")
claim("AUD.td.cloud.gt1", "RESULTS §5", "cloud0 runs with rel diff > 1 %", 8, int((c0.rel_diff_vs_ctrl > 0.01).sum()), 0, "verify/leak_audit/cue_tile_diff.csv")
claim("AUD.td.cloud.max", "RESULTS §4.3", "cloud0 max rel diff", 0.05, float(c0.rel_diff_vs_ctrl.max()), 2, "verify/leak_audit/cue_tile_diff.csv")
claim("AUD.td.ctrl", "RESULTS §4.3", "ctrl max rel diff vs val tiles", 0.0, float(TD[TD.variant == "ctrl"].rel_diff_vs_val.max()), 3, "verify/leak_audit/cue_tile_diff.csv")
for k, col, v, d in [("base", "b_median", 0.170, 3), ("shuffle", "b_median", 37.5, 1), ("reverse", "b_median", 76.9, 1), ("reverse", "spd_est_median", 39.5, 1),
                     ("base", "spd_true_median", 39.4, 1), ("static0", "b_median", 38.1, 1), ("static0", "spd_est_median", 1.86, 2), ("static0", "sigma_median", 0.24, 2),
                     ("clip99", "b_median", 0.174, 3), ("clip95", "b_median", 0.184, 3), ("shadow_move", "b_median", 0.225, 3), ("shadow_static", "b_median", 0.228, 3),
                     ("highpass", "b_median", 0.211, 3), ("highpass", "sigma_median", 0.41, 2), ("shuffle", "frac_le_017", 0.0, 2)]:
    claim(f"AUD.probe.{k}.{col}", "RESULTS §4.3", f"A probe {k} {col}", v, float(PRB.loc[k, col]), d, "verify/leak_audit/probe_a_summary_val.csv")
claim("AUD.probe.best", "RESULTS §4.3", "A training log best val (step 6300)", 0.1704, float(lg8.val_b_median.min()), 4, "e2e/train/log.csv")
# §6 appendix
claim("AUD.app.ufm", "RESULTS §6", "E1 UFM e1-core", 2.030, ec("e1_flow_ufmrefine980_g1", "all", "all")[0], 3, "e1_core_compare.csv")
cce = APP[(APP.method == "C-classical") & (APP.set == "e1-core (every 10th window)")].iloc[0]
claim("AUD.app.ccl.e1", "RESULTS §6", "C-classical e1-core every 10th", 0.788, float(cce.b_median), 3, "spectral-classical/e1-core-every10th/per_run.csv")
claim("AUD.app.ccl.val", "RESULTS §6", "C-classical val", 1.372, cs("C-classical", "val", "b_median"), 3, "comparison_by_set.csv")
for m, vals in {"A": (0.387, 0.342, 0.630, 0.355, 1.493, 0.287), "C": (0.659, 0.595, 0.806, 1.034, 1.010, 0.366), "C-K16": (0.792, 0.744, 1.009, 1.480, 1.001, 0.633),
                "B": (1.549, 1.605, 1.564, 2.741, 1.252, 1.864), "B0": (1.468, 1.506, 1.618, 2.690, 1.112, 1.457), "B1": (1.874, 1.897, 1.812, 3.274, 1.339, 2.110),
                "B2": (1.725, 1.592, 1.832, 2.237, 1.499, 0.666), "B3": (1.725, 1.759, 1.699, 3.015, 1.434, 2.150)}.items():
    for s, v in zip(SETS, vals):
        claim(f"AUD.rms.{m}.{s}", "RESULTS §6", f"{m} {s} window RMS median", v, cs(m, s, "rms_median"), 3, "comparison_by_set.csv")
for s, v in [("val", 1.340), ("test-iid", 1.404), ("test-rate", 1.174), ("e1-core", 1.576)]:
    claim(f"AUD.rms.B-ft.{s}", "RESULTS §6", f"B-ft {s} window RMS median", v, cs("B-ft", s, "rms_median"), 3, "comparison_by_set.csv")
print(f"audit-fix claims added: {len(claims) - n0}")
cl = pd.DataFrame(claims)
cl.to_csv(OUT / "claims.csv", index=False)

# ---------------------------------------------------------------- summary to stdout
print(f"claims checked: {len(cl)}; mismatches: {int((~cl.match).sum())}")
print(cl[~cl.match][["id", "quantity", "reported", "recomputed_rounded", "source"]].to_string())
print(pd.DataFrame(indep_rows).round(4).to_string())
print(pd.DataFrame(brows).round(3).to_string())
print(pd.DataFrame(ll).round(3).to_string())
print(pd.DataFrame(sel).round(3).to_string())
print(pd.DataFrame(hc).round(3).to_string())
print(pd.DataFrame(small_n).to_string())
print(pd.DataFrame(tl).to_string())
print("A Spearman r=0.5 exact:", cb("A", "test-rate, seaRate 0.5", "spearman_run_sigma_b"))
