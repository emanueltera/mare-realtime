"""Model A's fixed parts on a frozen texture: the phase-correlation start finds the content
shift with e2.baselines' sign, `align` makes a translating stack stand still, and the
velocity has the label's sign (v = −u · gsd / dt)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.model import E2E, align, cost_volumes, phase_corr_init, prep, sample_windows, spectra, to_velocity  # noqa: E402

from fakeset import fourier_shift  # noqa: E402


def texture(n=128, seed=3):
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    lp = np.exp(-(f[:, None] ** 2 + f[None, :] ** 2) / 0.01)
    return np.exp(0.3 * np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * lp).real / 0.05)


def stack(d, K=6, n=128):
    """K frames of a frozen texture whose content moves by d = (dx, dy) px per frame."""
    a = texture(n)
    return np.stack([fourier_shift(a, k * d[0], k * d[1]) for k in range(K)])


@pytest.mark.parametrize("d", [(3.3, -1.7), (-11.6, 6.2)])
def test_phase_corr_init_finds_the_shift(d):
    t = torch.tensor(stack(d))[None].float()
    x, m = prep(t)
    u = phase_corr_init(x, m)[0].numpy()
    assert np.allclose(u, d, atol=0.35), u


def test_align_freezes_a_translating_stack():
    d = (4.25, -2.5)
    t = torch.tensor(stack(d))[None].float()
    x, m = prep(t)
    xa, ma = align(x, m, torch.tensor([d]))
    both = (ma[0] > 0).all(0)
    both[:8], both[-8:], both[:, :8], both[:, -8:] = False, False, False, False   # periodic texture: skip the seams

    def spread(a):  # mean |frame k − middle frame| over the cells every frame covers
        ref = a[0, a.shape[1] // 2][both]
        return max(float((a[0, k][both] - ref).abs().mean()) for k in range(a.shape[1]))

    right = spread(xa)
    wrong = spread(align(x, m, torch.tensor([[d[0] + 1, d[1]]]))[0])
    # what is left at the right u is bicubic interpolation error only
    assert right < 0.03 and right < 0.1 * wrong, (right, wrong)


def test_velocity_sign_and_untrained_model_returns_its_start():
    u = torch.tensor([[5.0, -2.0]])
    v = to_velocity(u, torch.tensor([0.1]), torch.tensor([0.1]))
    assert torch.allclose(v, torch.tensor([[-5.0, 2.0]]))   # content moves at −v
    d = (2.0, 1.0)
    t = torch.tensor(stack(d, K=4, n=64))[None].float()
    o = E2E(K=4, n_iter=2)(t, torch.tensor([0.1]), torch.tensor([0.1]), amp=False)
    # the head's last layer starts at zero: the estimate is the phase-correlation start
    assert torch.allclose(o["us"][-1], o["u0"]) and o["v"].shape == (1, 2)


def test_cost_volume_peaks_at_the_content_shift():
    """Features = the frames themselves (one channel, unit norm per cell would make them ±1,
    so use the raw normalised values): the gap-g surface peaks at g·d, and the window read
    around g·d is centred on its peak."""
    d = (3.0, -2.0)
    t = torch.tensor(stack(d, K=4, n=64))[None].float()
    x, m = prep(t)
    ff, fm = spectra(x[0][:, None], m[0][:, None])
    corr, ov = cost_volumes(ff[None], fm[None], [1, 2], 64, 64)
    for gi, g in enumerate([1, 2]):
        c = corr[0, gi]
        iy, ix = divmod(int(c.argmax()), c.shape[1])
        assert (ix - 64, iy - 64) == (g * d[0], g * d[1])
    win = sample_windows(torch.stack([corr, ov], 2), torch.tensor([[[3.0, -2.0], [6.0, -4.0]]]), 2)
    assert int(win[0, 0, 0].argmax()) == 12 and int(win[0, 1, 0].argmax()) == 12   # the centre of 5 × 5


def test_run_inference_matches_window_inference():
    """forward_run (features and pair spectra once per run, window sums by cumulative sums)
    gives what forward gives window by window."""
    torch.manual_seed(0)
    K, n, Fr = 4, 64, 9
    model = E2E(K=K, n_iter=2).eval()
    torch.nn.init.normal_(model.readout.out.weight, std=0.05)   # make the read-out move u
    a = texture(n)
    tiles = torch.tensor(np.stack([fourier_shift(a, 2.5 * k, -1.0 * k) for k in range(Fr)])).float()
    tiles[:, :3, :5] = float("nan")                              # a masked corner
    starts = torch.arange(Fr - K + 1)
    with torch.no_grad():
        run = model.forward_run(tiles, 0.12, 0.1, K, starts, amp=False)
        win = model(torch.stack([tiles[s:s + K] for s in starts]), torch.full((len(starts),), 0.12),
                    torch.full((len(starts),), 0.1), amp=False)
    assert torch.allclose(run["u0"], win["u0"], atol=1e-3)
    assert torch.allclose(run["v"], win["v"], atol=1e-3), (run["v"] - win["v"]).abs().max()
    assert torch.allclose(run["sigma_v"], win["sigma_v"], rtol=1e-3)
