"""E2 baselines B0–B3: the reference points the learned models are compared with.

B0  masked phase correlation between consecutive tiles (e2.baselines), median over the
    window's pairs: the static-pattern velocity.
B1  pretrained SEA-RAFT-M on consecutive tiles, median flow → velocity: E1's failure mode,
    on E2's tiles.
B2  E1's dispersion-shell fit (experiments/e1_wave_bias/shell_fit.py) on E2 tiles, with
    E1's truth-centred prior (generous to the baseline).
B3  B1's pattern velocity plus a small MLP correction from features that never look at the
    wave field (gsd, |v_pattern|, pattern direction in tile axes).

Every method writes the standard prediction CSV (e2.data.PRED_COLUMNS) and is scored by
e2.evaluate; see results/baselines/NOTES.md.
"""
