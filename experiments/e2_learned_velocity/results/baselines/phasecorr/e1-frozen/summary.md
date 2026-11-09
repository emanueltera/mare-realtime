# E2 evaluation: phasecorr-e1-frozen

Predictions: `/workspace/data/e2/eval/phasecorr-e1-frozen.csv`. Every number below is derived from `per_run.csv` and `slices.csv` in this directory (computed by `e2.evaluate`). b = |run-mean (v_est − v_true)| (m/s); b% = b / mean ground speed; drift% over non-overlapping windows. Budget: b ≤ 0.17 m/s, b% ≤ 0.28; drift budget 0.5 %.

- e1-frozen: 24 of 24 runs predicted.

## By set

| set | K | runs | b median | b p90 | b% median | b% p90 | rms median | drift% median | drift% p90 | b≤0.17 | b%≤0.28 | b%≤0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1-frozen | 2 | 24 | 0.030 | 0.184 | 0.087 | 0.386 | 0.174 | 0.087 | 0.386 | 0.88 | 0.79 | 0.92 |

## By altitude

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-frozen | 2 | (50.0, 80.0] | 12 | 0.048 | 0.265 | 0.153 | 0.153 | 0.75 | 0.67 |
| e1-frozen | 2 | (80.0, 130.0] | 12 | 0.027 | 0.117 | 0.081 | 0.081 | 1.00 | 0.92 |

## By ground_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-frozen | 2 | (20, 35] | 12 | 0.026 | 0.120 | 0.087 | 0.087 | 0.92 | 0.75 |
| e1-frozen | 2 | (50, 65] | 12 | 0.044 | 0.193 | 0.074 | 0.074 | 0.83 | 0.83 |

## By sun_elevation

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-frozen | 2 | (35, 90] | 24 | 0.030 | 0.184 | 0.087 | 0.087 | 0.88 | 0.79 |

## By wind_speed

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-frozen | 2 | (6.0, 10.0] | 16 | 0.030 | 0.121 | 0.087 | 0.087 | 0.94 | 0.88 |
| e1-frozen | 2 | (10.0, 14.0] | 8 | 0.072 | 0.310 | 0.162 | 0.162 | 0.75 | 0.62 |

## By gimbal_pitch

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-frozen | 2 | (-90.01, -80.0] | 24 | 0.030 | 0.184 | 0.087 | 0.087 | 0.88 | 0.79 |

## By masked_frac

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-frozen | 2 | (-0.01, 0.001] | 24 | 0.030 | 0.184 | 0.087 | 0.087 | 0.88 | 0.79 |

## By swell

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-frozen | 2 | 0 | 16 | 0.022 | 0.213 | 0.074 | 0.074 | 0.88 | 0.75 |
| e1-frozen | 2 | 1 | 8 | 0.032 | 0.104 | 0.094 | 0.094 | 0.88 | 0.88 |

## By sea_rate

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-frozen | 2 | 0 | 24 | 0.030 | 0.184 | 0.087 | 0.087 | 0.88 | 0.79 |

## By pattern

| set | K | bin | runs | b median | b p90 | b% median | drift% median | b≤0.17 | b%≤0.28 |
|---|---|---|---|---|---|---|---|---|---|
| e1-frozen | 2 | straight | 24 | 0.030 | 0.184 | 0.087 | 0.087 | 0.88 | 0.79 |
