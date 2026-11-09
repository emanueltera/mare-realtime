"""flowhead: pair layout, the phase-correlation shift's sign and its chaining (CPU, no SEA-RAFT)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flowhead.model import chain_init, make_pairs, phase_shift  # noqa: E402


def test_make_pairs_cover_the_window():
    p = make_pairs(8, (1, 2, 4, 7))
    assert p[:7] == [(i, i + 1) for i in range(7)]
    assert (0, 7) in p and len(p) == 14
    assert make_pairs(8, (1,)) == [(i, i + 1) for i in range(7)]


def test_phase_shift_sign_and_chain():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(512, 512))
    d = np.array([[13, -7], [-20, 5], [31, 2]])          # (dx, dy) per consecutive frame
    pos = np.vstack([[0, 0], d.cumsum(0)])
    # frame k shows the content displaced by pos[k]: z_k(p) = base(p − pos_k)
    frames = np.stack([np.roll(base, (int(y), int(x)), axis=(0, 1))[128:384, 128:384] for x, y in pos])
    z = torch.from_numpy(frames).float()
    s1 = phase_shift(z[:-1], z[1:])
    assert np.array_equal(s1.numpy(), d)
    init = chain_init(s1, torch.tensor([0, 1]), torch.tensor([3, 2]))
    assert np.array_equal(init.numpy(), [pos[3] - pos[0], pos[2] - pos[1]])
