"""The loader: window labels, allowed inputs only, and exact augmentations (the target
turns with the tiles)."""

from __future__ import annotations

import numpy as np
import pytest

from e2 import geom
from e2.baselines import phase_correlation
from e2.data import (DIHEDRAL, TileSet, WindowDataset, augment_array, augment_vector, collate, prediction_frame,
                     unaugment_vector)

from fakeset import make_set


@pytest.fixture(scope="module")
def ts(tmp_path_factory):
    root = make_set(tmp_path_factory.mktemp("sets") / "fake")
    return TileSet(root)


def test_windows_label_is_mean_velocity(ts):
    w = ts.windows(K=5, stride=2, step=3)
    assert set(w.run_id) == {"r0", "r1", "r2"}
    r1 = w[w.run_id == "r1"].iloc[0]
    assert r1["T"] == pytest.approx(0.8)
    assert r1["dt"] == pytest.approx(0.2)
    assert np.allclose([r1.v_true_e, r1.v_true_n], [-5.0, 20.0])
    assert np.allclose([r1.v_true_x, r1.v_true_y], geom.enu_to_tile([-5.0, 20.0], 33.0))
    # 21 frames, span 8: starts 0, 3, 6, 9, 12
    assert list(w[w.run_id == "r0"].w_start) == [0, 3, 6, 9, 12]


def test_item_has_only_allowed_inputs(ts):
    ds = WindowDataset(ts, K=4)
    x = ds[0]
    assert set(x) == {"tiles", "mask", "gsd", "dt", "target", "aug", "key"}
    assert x["tiles"].shape == (4, 64, 64) and x["tiles"].dtype == np.float32
    assert not x["mask"][:, :2].any() and x["mask"][:, 2:].all()
    assert np.isfinite(x["tiles"]).all()
    b = collate([ds[0], ds[1]])
    assert b["tiles"].shape == (2, 4, 64, 64)


@pytest.mark.parametrize("g", range(8))
def test_augmentation_turns_the_target_with_the_tiles(ts, g):
    ds = WindowDataset(ts, K=2, normalize=None)
    for i in range(len(ds)):
        x = ds.item(i, g)
        a, b = (np.where(x["mask"][k], x["tiles"][k], np.nan) for k in (0, 1))
        d, _ = phase_correlation(a, b)
        expect = -x["target"] * x["dt"] / x["gsd"]
        assert np.abs(d - expect).max() < 0.1, (g, d, expect)
        assert np.allclose(unaugment_vector(x["target"], g), ds.item(i)["target"], atol=1e-5)


def test_dihedral_group_is_complete():
    a = np.arange(9.0).reshape(3, 3)
    seen = {augment_array(a, g).tobytes() for g in range(8)}
    assert len(seen) == 8 == len(DIHEDRAL)
    v = np.array([0.3, -1.7])
    for g in range(8):
        assert np.allclose(unaugment_vector(augment_vector(v, g), g), v)


def test_crop_and_bin(ts):
    ds = WindowDataset(ts, K=2, crop=32, bin=2, augment=True, normalize=None)
    x = ds[0]
    assert x["tiles"].shape == (2, 16, 16)
    assert x["gsd"] == pytest.approx(0.2)


def test_prediction_frame_fills_truth_and_enu(ts):
    ds = WindowDataset(ts, K=3, stride=2)
    keys = [ds[i]["key"] for i in range(len(ds))]
    est = ds.index[["v_true_x", "v_true_y"]].to_numpy() + 0.1
    df = prediction_frame(ts, keys, est)
    assert np.allclose(df[["v_true_x", "v_true_y"]], ds.index[["v_true_x", "v_true_y"]])
    v_en = np.stack([ts.to_enu(r, e) for r, e in zip(df.run_id, est)])
    assert np.allclose(df[["v_est_e", "v_est_n"]], v_en)
