# E2 evaluation: flowhead-gap1

Predictions: `/workspace/data/e2/preds/flowhead-gap1/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 1.802; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 432 | 1.802 | 4.001 | 6.354 | 19.708 | 1.881 | 6.356 | 19.890 | 0.00 | 0.00 | 0.00 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 50.0] | 108 | 1.207 | 22.951 | 6.727 | 6.974 | 0.00 | 0.00 |
| e1-core | 8 | (50.0, 80.0] | 108 | 1.344 | 2.171 | 4.415 | 4.367 | 0.00 | 0.00 |
| e1-core | 8 | (80.0, 130.0] | 108 | 1.965 | 2.772 | 6.143 | 6.118 | 0.00 | 0.00 |
| e1-core | 8 | (200.0, 260.0] | 108 | 2.968 | 3.857 | 9.603 | 9.444 | 0.00 | 0.00 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0, 20] | 144 | 1.324 | 3.110 | 8.825 | 9.285 | 0.00 | 0.00 |
| e1-core | 8 | (20, 35] | 144 | 1.563 | 2.975 | 5.211 | 5.207 | 0.00 | 0.00 |
| e1-core | 8 | (50, 65] | 144 | 2.746 | 19.155 | 4.576 | 4.550 | 0.00 | 0.00 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (5, 15] | 216 | 1.676 | 4.510 | 5.981 | 6.004 | 0.00 | 0.00 |
| e1-core | 8 | (35, 90] | 216 | 1.907 | 3.867 | 6.577 | 6.810 | 0.00 | 0.00 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (0.0, 6.0] | 72 | 2.058 | 3.815 | 7.590 | 7.717 | 0.00 | 0.00 |
| e1-core | 8 | (6.0, 10.0] | 288 | 1.770 | 4.012 | 6.316 | 6.373 | 0.00 | 0.00 |
| e1-core | 8 | (10.0, 14.0] | 72 | 1.472 | 3.780 | 4.990 | 4.895 | 0.00 | 0.00 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-90.01, -80.0] | 432 | 1.802 | 4.001 | 6.354 | 6.356 | 0.00 | 0.00 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | (-0.01, 0.001] | 432 | 1.802 | 4.001 | 6.354 | 6.356 | 0.00 | 0.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 0 | 288 | 1.768 | 3.995 | 6.354 | 6.369 | 0.00 | 0.00 |
| e1-core | 8 | 1 | 144 | 1.818 | 3.992 | 6.339 | 6.333 | 0.00 | 0.00 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | 1 | 432 | 1.802 | 4.001 | 6.354 | 6.356 | 0.00 | 0.00 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 8 | straight | 432 | 1.802 | 4.001 | 6.354 | 6.356 | 0.00 | 0.00 |
