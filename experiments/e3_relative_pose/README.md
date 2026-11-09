# E3 — relative pose between frames, end to end, without GNSS

A simpler question than E2: give a network two consecutive onboard frames and the sensors an
aircraft has without GNSS, and ask for the camera's translation and rotation between them.
Chain the steps over a whole flight and measure the drift. Everything is simulated; nothing
here has seen real water.

**Result** ([results/RESULTS.md](results/RESULTS.md)). On 120 simulated 60 s flights the network
drifts a median 1.18 % of distance flown with the true attitude, 1.76 % with the AHRS attitude
and 2.08 % with its own rotations, against 33 to 36 % for sensor-only dead reckoning that knows no
wind. It takes its scale mostly from the baro: pooled, about 73 % of a run's baro error reaches
its output, a share that rises with altitude to about all of it above 120 m. Its rotation adds
nothing over the gyro, and its drift is 1.6 to 1.8 times higher when the simulated sea is frozen
or runs at double speed, a weaker form of the wave-speed dependence E2 found. These sizes rest on
the assumed sensor noise.

## Setup

- **Flights.** E2's domain-randomised runs (sea, sky, pattern, turbulence and wind aloft as
  `randomRun` draws them; altitude 30–200 m log-uniform; airspeed 15–60 m/s), but the camera
  is **hard-mounted** (`gimbal.stabilized` 0, mount pitch −90…−60°), so it turns with the
  airframe. 720 × 1280 portrait, vfov 92, 10 fps, stored as 288 × 512 luma
  (`e3/shrink.py`).
- **Inputs** (`e3/data.py`). Frames t and t+1, plus a 14-vector:
  - gravity in camera axes at t and t+1, from an AHRS with tilt error (bias σ 0.3°, noise 0.1°);
  - the gyro's rotation over the step (bias σ 0.05°/s per axis, noise 0.003° per step);
  - baro height (bias σ 2 m, noise 0.3 m);
  - airspeed (scale error σ 2 %, noise 0.3 m/s), taken along the nose.

  The network gets no GNSS, wind or heading. Sensor errors are drawn afresh per sample in
  training and from a fixed per-run seed for evaluation.
- **Outputs.**
  - Translation in camera axes of frame t, in tenths of the baro height.
  - A correction to the gyro's rotation.
  - A Laplace scale for each, trained by negative log-likelihood.
- **Network** (`e3/model.py`, 7.8 M params).
  - A shared CNN encoder.
  - Local correlation volumes at 1/8 (±80 px) and 1/16 (±96 px), with ray-direction channels.
  - The sensors broadcast at 1/16 and joined again in an MLP head.
- **Augmentation.** The four exact camera mirrors (labels and sensors transformed with them),
  plus gain, gamma and noise.
- **Baselines.**
  - `airspeed`: dead reckoning from airspeed along the nose with the gyro, without wind
    (the aircraft without a camera).
  - `blind`: the same head without images; it learns only what the sensors imply.

| set | runs | length | use |
|---|---|---|---|
| `train` | 3000 | 12 s | training (360 k pairs) |
| `val` | 100 | 12 s | checkpoint selection |
| `test-long` | 120 | 60 s | drift over a minute |
| `test-rate` | 30 conditions × seaRate 0, 1, 2 | 20 s | does it measure motion, or the sim's waves? (E2's finding) |
| `pilot` | 4 | 12 s | smoke tests |

## Metrics (`e3/evaluate.py`)

Each method's steps are chained from the true start, placed in the world with one of three
attitudes:

- `true`: isolates the translation.
- `ahrs`: the yaw error included; what a real system gets.
- `chain`: the method's own rotations from the AHRS start.

**Drift** is the horizontal end-point error divided by the horizontal distance flown. **b** is
that error divided by the run's duration (m/s, like E2's per-run bias). Per step, the script
reports the mean translation error, the median relative error, the along-track scale bias,
and the rotation error against the gyro alone.

## Running

```bash
node experiments/e3_relative_pose/make_jobs.mjs
bash experiments/e3_relative_pose/render_all.sh <set>
nohup bash experiments/e3_relative_pose/run_e3.sh > /workspace/data/e3/logs/run.log 2>&1 &
```

`render_all.sh` runs once per set (on the box). `run_e3.sh` trains, predicts and scores.
`E3_SET=<sets>/pilot python -m pytest tests` runs the checks.
