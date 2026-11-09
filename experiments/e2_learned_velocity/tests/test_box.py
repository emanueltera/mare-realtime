"""Checks on the rendered E2 data (GPU box; skipped where /workspace/data/e2 is absent).

On test-rate's frozen-sea runs (render.seaRate 0) the tile content is the same water seen
from a moving camera, so phase correlation between consecutive tiles recovers the camera's
velocity: the per-run bias should sit at the E1 frozen floor (~0.05 m/s median)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SETS = Path(os.environ.get("E2_SETS", "/workspace/data/e2/sets"))
pytestmark = pytest.mark.skipif(not (SETS / "test-rate" / "runs.csv").exists(), reason="no E2 data here")


def test_frozen_runs_phase_correlation_recovers_velocity(tmp_path):
    from e2.baselines import phasecorr_predictions
    from e2.evaluate import evaluate

    pred = tmp_path / "pc.csv"
    phasecorr_predictions(SETS / "test-rate", pred, sea_rate=0.0, workers=16)
    pr, sl = evaluate(pred, SETS, tmp_path / "out", name="phasecorr-frozen")
    pr.to_csv(tmp_path / "per_run.csv", index=False)
    assert len(pr) == 100
    assert float(pr.b.median()) < 0.08, pr.b.describe()


def test_sets_are_complete_and_disjoint():
    seen = {}
    for s in ["train", "val", "test-iid", "test-sea", "test-alt", "test-rate", "e1-core"]:
        r = pd.read_csv(SETS / s / "runs.csv")
        seeds = set(r.sea_seed)
        for other, os_ in seen.items():
            assert not (seeds & os_), (s, other)
        seen[s] = seeds
        t = np.load(SETS / s / "tiles" / f"{r.run_id.iloc[0]}.npy", mmap_mode="r")
        assert t.dtype == np.float16 and t.shape[1:] == (r.n.iloc[0], r.n.iloc[0])
