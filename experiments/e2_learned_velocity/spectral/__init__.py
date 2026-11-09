"""E2 model C ('spectral'): a learned dispersion-shell fit.

The camera-fixed tile stack's 3-D power spectrum puts a wave of wavevector k at
ω' = √(g|k|) − k·V (wrapped modulo 2π·fps), V the camera's velocity over the water. E1's
classical fit maximises the whitened spectral energy on that shell over V; here a network
chooses how much each k is trusted (and how sharp the peak is) from the spectrum itself, and
the search is centred on a classical, non-learned pattern velocity (phase correlation
between consecutive tiles), so v = v_pattern + δ.

    spectrum.py  tiles → whitened power spectrum S(k, ω'), shell gathers (cost volumes)
    pattern.py   the classical pattern velocity per window (phase correlation, e2.baselines)
    model.py     the network: per-k weights → cost volume → CNN → coarse / fine soft-argmax, σ
    data.py      windows with their pattern velocity, exact dihedral augmentation
    train.py     training with a validation curve; predict.py  prediction CSVs for e2.evaluate
"""
