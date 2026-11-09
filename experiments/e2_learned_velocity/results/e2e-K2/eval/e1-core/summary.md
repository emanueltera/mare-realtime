# E2 evaluation: e2e-K2

Predictions: `/workspace/data/e2/preds/e2e-K2/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 0.207; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | 432 | 0.207 | 0.614 | 0.763 | 2.421 | 0.545 | 0.763 | 2.421 | 0.41 | 0.15 | 0.32 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (0.0, 50.0] | 108 | 0.183 | 3.320 | 0.771 | 0.771 | 0.45 | 0.13 |
| e1-core | 2 | (50.0, 80.0] | 108 | 0.155 | 0.310 | 0.501 | 0.501 | 0.62 | 0.20 |
| e1-core | 2 | (80.0, 130.0] | 108 | 0.197 | 0.383 | 0.695 | 0.695 | 0.45 | 0.22 |
| e1-core | 2 | (200.0, 260.0] | 108 | 0.412 | 0.752 | 1.294 | 1.294 | 0.10 | 0.05 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (0, 20] | 144 | 0.176 | 0.461 | 1.175 | 1.175 | 0.49 | 0.01 |
| e1-core | 2 | (20, 35] | 144 | 0.207 | 0.465 | 0.690 | 0.690 | 0.42 | 0.14 |
| e1-core | 2 | (50, 65] | 144 | 0.297 | 1.404 | 0.496 | 0.496 | 0.31 | 0.30 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (5, 15] | 216 | 0.181 | 0.622 | 0.666 | 0.666 | 0.49 | 0.19 |
| e1-core | 2 | (35, 90] | 216 | 0.230 | 0.614 | 0.850 | 0.850 | 0.33 | 0.11 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (0.0, 6.0] | 72 | 0.211 | 0.632 | 0.766 | 0.766 | 0.39 | 0.12 |
| e1-core | 2 | (6.0, 10.0] | 288 | 0.197 | 0.573 | 0.741 | 0.741 | 0.44 | 0.15 |
| e1-core | 2 | (10.0, 14.0] | 72 | 0.278 | 0.662 | 0.856 | 0.856 | 0.31 | 0.17 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (-90.01, -80.0] | 432 | 0.207 | 0.614 | 0.763 | 0.763 | 0.41 | 0.15 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (-0.01, 0.001] | 432 | 0.207 | 0.614 | 0.763 | 0.763 | 0.41 | 0.15 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | 0 | 288 | 0.216 | 0.636 | 0.772 | 0.772 | 0.40 | 0.15 |
| e1-core | 2 | 1 | 144 | 0.200 | 0.579 | 0.755 | 0.755 | 0.43 | 0.15 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | 1 | 432 | 0.207 | 0.614 | 0.763 | 0.763 | 0.41 | 0.15 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | straight | 432 | 0.207 | 0.614 | 0.763 | 0.763 | 0.41 | 0.15 |
