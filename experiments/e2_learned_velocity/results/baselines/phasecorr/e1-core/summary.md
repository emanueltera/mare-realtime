# E2 evaluation: phasecorr-K2

Predictions: `/workspace/data/e2/eval/phasecorr/e1-core.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-core: 432 of 432 runs predicted.

e1-core, the same runs as E1 (`e1_compare.csv`): median per-run b, m/s — this method 1.439; flow_searaft_g1 1.676; flow_ufmrefine980_g1 2.030; shell_dispersive_T3.2 0.614; shell_dispersive_T6.4 0.645.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | 432 | 1.439 | 2.550 | 4.890 | 13.822 | 1.706 | 4.890 | 13.822 | 0.00 | 0.00 | 0.00 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (0.0, 50.0] | 108 | 0.955 | 10.578 | 3.793 | 3.793 | 0.00 | 0.00 |
| e1-core | 2 | (50.0, 80.0] | 108 | 1.102 | 1.590 | 3.894 | 3.894 | 0.00 | 0.00 |
| e1-core | 2 | (80.0, 130.0] | 108 | 1.523 | 1.843 | 5.290 | 5.290 | 0.00 | 0.00 |
| e1-core | 2 | (200.0, 260.0] | 108 | 2.251 | 2.604 | 7.362 | 7.362 | 0.00 | 0.00 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (0, 20] | 144 | 1.412 | 2.395 | 9.412 | 9.412 | 0.00 | 0.00 |
| e1-core | 2 | (20, 35] | 144 | 1.420 | 2.476 | 4.735 | 4.735 | 0.00 | 0.00 |
| e1-core | 2 | (50, 65] | 144 | 1.576 | 7.788 | 2.626 | 2.626 | 0.00 | 0.00 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (5, 15] | 216 | 1.721 | 2.735 | 5.832 | 5.832 | 0.00 | 0.00 |
| e1-core | 2 | (35, 90] | 216 | 1.346 | 2.073 | 3.961 | 3.961 | 0.00 | 0.00 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (0.0, 6.0] | 72 | 1.399 | 2.507 | 4.617 | 4.617 | 0.00 | 0.00 |
| e1-core | 2 | (6.0, 10.0] | 288 | 1.467 | 2.546 | 4.975 | 4.975 | 0.00 | 0.00 |
| e1-core | 2 | (10.0, 14.0] | 72 | 1.428 | 2.707 | 4.941 | 4.941 | 0.00 | 0.00 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (-90.01, -80.0] | 432 | 1.439 | 2.550 | 4.890 | 4.890 | 0.00 | 0.00 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | (-0.01, 0.001] | 432 | 1.439 | 2.550 | 4.890 | 4.890 | 0.00 | 0.00 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | 0 | 288 | 1.418 | 2.541 | 4.780 | 4.780 | 0.00 | 0.00 |
| e1-core | 2 | 1 | 144 | 1.522 | 2.559 | 5.076 | 5.076 | 0.00 | 0.00 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | 1 | 432 | 1.439 | 2.550 | 4.890 | 4.890 | 0.00 | 0.00 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-core | 2 | straight | 432 | 1.439 | 2.550 | 4.890 | 4.890 | 0.00 | 0.00 |
