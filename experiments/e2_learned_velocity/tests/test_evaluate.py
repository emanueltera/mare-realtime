"""The evaluation harness on a synthetic set: an oracle scores zero, a constant offset
scores exactly that offset, and predicting zero scores the whole ground speed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from e2.data import PRED_COLUMNS, TileSet, WindowDataset, prediction_frame
from e2.evaluate import evaluate, nonoverlapping

from fakeset import make_set


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    root = tmp_path_factory.mktemp("sets")
    make_set(root / "fake")
    return root


def predict(sets, fn, K=5):
    ts = TileSet(sets / "fake")
    ds = WindowDataset(ts, K=K)
    keys = list(zip(ds.index.set, ds.index.run_id, ds.index.w_start, ds.index.K, ds.index["T"]))
    est = fn(ds.index[["v_true_x", "v_true_y"]].to_numpy())
    return prediction_frame(ts, keys, est)[PRED_COLUMNS]


def run(sets, tmp_path, fn, K=5):
    p = tmp_path / "pred.csv"
    predict(sets, fn, K).to_csv(p, index=False)
    return evaluate(p, sets, tmp_path / "out", name="t")


def test_oracle_scores_zero(sets, tmp_path):
    pr, sl = run(sets, tmp_path, lambda v: v)
    assert np.allclose(pr[["b", "rms", "drift_pct"]], 0, atol=1e-9)
    assert (sl.frac_b_le_017 == 1).all()
    assert (tmp_path / "out" / "summary.md").exists()


def test_constant_offset_is_the_bias(sets, tmp_path):
    off = np.array([0.06, -0.08])  # |off| = 0.1 m/s
    pr, sl = run(sets, tmp_path, lambda v: v + off)
    assert np.allclose(pr.b, 0.1)
    assert np.allclose(pr.rms, 0.1)
    assert np.allclose(pr.b_pct, 100 * 0.1 / pr.ground_speed)
    # drift = |Σ off·T| / Σ |v|·T = 0.1 / speed for constant velocity
    assert np.allclose(pr.drift_pct, 100 * 0.1 / pr.ground_speed)
    allrow = sl[sl.slice == "all"].iloc[0]
    assert allrow.frac_b_le_017 == 1.0


def test_predicting_zero_scores_the_ground_speed(sets, tmp_path):
    pr, sl = run(sets, tmp_path, lambda v: np.zeros_like(v))
    assert np.allclose(pr.b, pr.ground_speed)
    assert np.allclose(pr.b_pct, 100.0)
    assert np.allclose(pr.drift_pct, 100.0)
    assert (sl.frac_b_le_017 == 0).all() and (sl.frac_bpct_le_05 == 0).all()


def test_wrong_truth_is_caught(sets, tmp_path):
    df = predict(sets, lambda v: v)
    df.loc[0, "v_true_x"] += 0.5
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="differs from the labels"):
        evaluate(p, sets, tmp_path / "o2")


def test_nonoverlapping():
    w = np.array([0, 1, 2, 4, 8, 9, 12])
    span = np.full(7, 4)
    assert list(w[nonoverlapping(w, span)]) == [0, 4, 8, 12]
