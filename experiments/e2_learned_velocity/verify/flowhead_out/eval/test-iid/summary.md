# E2 evaluation: flowhead-verify

Predictions: `/workspace/data/e2/verify_flowhead/preds/test-iid.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- test-iid: 300 of 300 runs predicted.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test-iid | 8 | 300 | 1.366 | 2.342 | 4.204 | 8.115 | 1.606 | 4.135 | 8.198 | 0.00 | 0.00 | 0.00 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-iid | 8 | (0.0, 50.0] | 72 | 1.116 | 1.845 | 3.359 | 3.414 | 0.00 | 0.00 |
| test-iid | 8 | (50.0, 80.0] | 68 | 1.200 | 1.589 | 3.623 | 3.656 | 0.00 | 0.01 |
| test-iid | 8 | (80.0, 130.0] | 83 | 1.525 | 2.218 | 4.390 | 4.309 | 0.00 | 0.00 |
| test-iid | 8 | (130.0, 200.0] | 77 | 1.902 | 2.737 | 5.015 | 4.980 | 0.01 | 0.00 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-iid | 8 | (0, 20] | 46 | 1.021 | 1.837 | 6.517 | 6.299 | 0.02 | 0.00 |
| test-iid | 8 | (20, 35] | 85 | 1.397 | 2.132 | 4.811 | 4.851 | 0.00 | 0.00 |
| test-iid | 8 | (35, 50] | 94 | 1.635 | 2.347 | 3.955 | 3.906 | 0.00 | 0.00 |
| test-iid | 8 | (50, 65] | 66 | 1.347 | 4.004 | 2.502 | 2.536 | 0.00 | 0.00 |
| test-iid | 8 | (65, 200] | 9 | 0.508 | 1.511 | 0.723 | 0.815 | 0.00 | 0.11 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-iid | 8 | (-90, 5] | 29 | 1.221 | 1.774 | 2.857 | 2.882 | 0.00 | 0.00 |
| test-iid | 8 | (5, 15] | 46 | 1.343 | 2.028 | 4.157 | 4.077 | 0.00 | 0.00 |
| test-iid | 8 | (15, 35] | 81 | 1.507 | 2.363 | 4.632 | 4.788 | 0.00 | 0.01 |
| test-iid | 8 | (35, 90] | 144 | 1.384 | 2.340 | 4.296 | 4.276 | 0.01 | 0.00 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-iid | 8 | (0.0, 6.0] | 89 | 1.734 | 2.714 | 5.099 | 5.126 | 0.00 | 0.00 |
| test-iid | 8 | (6.0, 10.0] | 77 | 1.459 | 2.345 | 4.193 | 4.195 | 0.00 | 0.00 |
| test-iid | 8 | (10.0, 14.0] | 83 | 1.218 | 1.872 | 3.427 | 3.468 | 0.00 | 0.00 |
| test-iid | 8 | (14.0, 16.0] | 51 | 1.100 | 1.635 | 2.722 | 2.909 | 0.02 | 0.02 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-iid | 8 | (-90.01, -80.0] | 95 | 1.464 | 2.320 | 4.290 | 4.195 | 0.00 | 0.00 |
| test-iid | 8 | (-80.0, -70.0] | 113 | 1.309 | 2.327 | 4.095 | 4.017 | 0.01 | 0.00 |
| test-iid | 8 | (-70.0, -59.99] | 92 | 1.383 | 2.350 | 4.302 | 4.248 | 0.00 | 0.01 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-iid | 8 | (-0.01, 0.001] | 284 | 1.351 | 2.327 | 4.204 | 4.135 | 0.00 | 0.00 |
| test-iid | 8 | (0.001, 0.01] | 14 | 1.845 | 2.695 | 4.781 | 4.989 | 0.00 | 0.00 |
| test-iid | 8 | (0.01, 0.05] | 2 | 1.049 | 1.264 | 2.262 | 2.200 | 0.00 | 0.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-iid | 8 | 0 | 90 | 1.465 | 2.488 | 4.472 | 4.327 | 0.00 | 0.00 |
| test-iid | 8 | 1 | 210 | 1.348 | 2.182 | 3.886 | 3.876 | 0.00 | 0.00 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-iid | 8 | 1 | 300 | 1.366 | 2.342 | 4.204 | 4.135 | 0.00 | 0.00 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| test-iid | 8 | figure8 | 40 | 1.417 | 2.329 | 4.441 | 4.258 | 0.00 | 0.00 |
| test-iid | 8 | lawnmower | 56 | 1.253 | 2.256 | 3.534 | 3.409 | 0.00 | 0.02 |
| test-iid | 8 | orbit | 53 | 1.397 | 2.276 | 4.497 | 4.484 | 0.00 | 0.00 |
| test-iid | 8 | racetrack | 54 | 1.348 | 2.203 | 3.808 | 3.830 | 0.00 | 0.00 |
| test-iid | 8 | random | 48 | 1.476 | 2.689 | 4.215 | 3.999 | 0.02 | 0.00 |
| test-iid | 8 | straight | 49 | 1.364 | 2.364 | 4.427 | 4.402 | 0.00 | 0.00 |
