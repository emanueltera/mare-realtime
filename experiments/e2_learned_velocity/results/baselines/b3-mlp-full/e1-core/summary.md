# E2 evaluation: b3-mlp-full

Predictions: `/workspace/data/e2/preds/b3-mlp-full/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 2.066; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 432 | 2.066 | 5.543 | 6.584 | 26.207 | 2.094 | 6.646 | 26.506 | 0.00 | 0.00 | 0.00 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 50.0] | 108 | 1.163 | 57.677 | 6.359 | 6.538 | 0.00 | 0.00 |
| e1-core | 8 | (50.0, 80.0] | 108 | 1.504 | 2.278 | 5.012 | 5.013 | 0.00 | 0.00 |
| e1-core | 8 | (80.0, 130.0] | 108 | 2.496 | 3.842 | 7.352 | 7.351 | 0.00 | 0.00 |
| e1-core | 8 | (200.0, 260.0] | 108 | 3.722 | 5.849 | 10.991 | 10.984 | 0.00 | 0.00 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0, 20] | 144 | 1.483 | 4.052 | 9.884 | 9.932 | 0.00 | 0.00 |
| e1-core | 8 | (20, 35] | 144 | 1.643 | 3.926 | 5.477 | 5.467 | 0.00 | 0.00 |
| e1-core | 8 | (50, 65] | 144 | 2.642 | 56.668 | 4.403 | 4.402 | 0.00 | 0.00 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (5, 15] | 216 | 2.088 | 5.183 | 7.042 | 7.014 | 0.00 | 0.00 |
| e1-core | 8 | (35, 90] | 216 | 1.995 | 5.741 | 6.224 | 6.268 | 0.00 | 0.00 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 6.0] | 72 | 2.117 | 5.263 | 7.060 | 7.058 | 0.00 | 0.00 |
| e1-core | 8 | (6.0, 10.0] | 288 | 1.976 | 5.591 | 6.711 | 6.758 | 0.00 | 0.00 |
| e1-core | 8 | (10.0, 14.0] | 72 | 1.908 | 5.604 | 5.432 | 5.521 | 0.00 | 0.00 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-90.01, -80.0] | 432 | 2.066 | 5.543 | 6.584 | 6.646 | 0.00 | 0.00 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-0.01, 0.001] | 432 | 2.066 | 5.543 | 6.584 | 6.646 | 0.00 | 0.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 0 | 288 | 2.088 | 5.139 | 6.513 | 6.555 | 0.00 | 0.00 |
| e1-core | 8 | 1 | 144 | 2.014 | 5.903 | 7.037 | 7.052 | 0.00 | 0.00 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 1 | 432 | 2.066 | 5.543 | 6.584 | 6.646 | 0.00 | 0.00 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | straight | 432 | 2.066 | 5.543 | 6.584 | 6.646 | 0.00 | 0.00 |
