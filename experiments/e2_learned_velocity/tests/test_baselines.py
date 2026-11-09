"""The baselines' plumbing: window medians over pair velocities, and B3's dihedral
augmentation matching e2.data's (torch only; skipped without it)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baselines.windows import pair_arrays, window_medians, window_predictions
from e2.data import TileSet, augment_vector

from fakeset import make_set


def test_window_medians_cover_the_window_pairs():
    v = np.stack([np.arange(10.0), -np.arange(10.0)], 1)  # pair i has velocity (i, −i)
    m = window_medians(v, 4)  # windows of 4 frames hold pairs w, w+1, w+2
    assert m.shape == (8, 2)
    assert np.allclose(m[:, 0], np.arange(8) + 1) and np.allclose(m[:, 1], -(np.arange(8) + 1))
    assert np.allclose(window_medians(v, 2), v)  # K = 2 is the pair itself


def test_window_medians_ignore_nan_pairs():
    v = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0], [np.nan, np.nan], [np.nan, np.nan]])
    m = window_medians(v, 3)
    assert np.allclose(m[0], 1.0) and np.allclose(m[1], 3.0) and np.isnan(m[3]).all()


def test_window_predictions_carry_true_labels(tmp_path):
    make_set(tmp_path / "fake")
    ts = TileSet(tmp_path / "fake")
    rows = []
    for rid in ts.run_ids:
        nf = len(ts._fr[rid])
        rows += [{"run_id": rid, "w_start": w, "K": 2, "v_est_x": 1.0, "v_est_y": 2.0} for w in range(nf - 1)]
    df = window_predictions(ts, pd.DataFrame(rows), Ks=(2, 5))
    w = ts.windows(5)
    got = df[df.K == 5].merge(w, on=["run_id", "w_start"], suffixes=("", "_w"))
    assert len(got) == len(w)
    assert np.allclose(got.v_true_x, got.v_true_x_w) and np.allclose(got.v_est_x, 1.0)


def test_dihedral_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    from baselines.correction import dihedral_torch

    v = np.random.default_rng(0).normal(size=(8, 2))
    g = np.arange(8)
    got = dihedral_torch(torch.as_tensor(v), torch.as_tensor(g)).numpy()
    want = np.stack([augment_vector(v[i], i) for i in g])
    assert np.allclose(got, want)
