# E1 — how much does the sea's own motion bias image-based velocity?

Experiment E1 of [no-gnss-training-plan.md](../../../docs/no-gnss-training-plan.md). All renders are
from mare-realtime on 29 October 2026. Every number below comes from the CSVs in this folder:
`per_run.csv`, `summary_by_slice.csv`, `phase_speed.csv`, `altitude_exponent.csv`,
`decomp_by_layer.csv` and `audit_checks.csv`, which `analyze.py` writes from the per-measurement
files `flow-*.csv` and `shell-*.csv`. Numbers marked *derived* are computed, not measured. An
audit corrected several statements on 29 October 2026; see "Audit" at the end.

## Question

How large is the velocity bias caused by the sea's own motion (wave phase propagation, foam
drift)? It is measured for two estimators:

- **(a)** off-the-shelf dense optical flow, converted to metric velocity with the true attitude
  and height;
- **(b)** a classical dispersion-shell fit (3-D FFT).

Both are swept over sea state, altitude, ground speed, heading relative to the waves, and sun.
A frozen-sea control checks the flow-to-metric geometry and the static spectral pipeline when
the water does not move (midday, 60–120 m, 30–60 m/s only; it cannot test the dispersive fit,
see headline 5). The budget is 0.17 m/s, which is 0.28 % of speed at 60 m/s and is the water-motion share
of an RSS split of the drift budget. The whole drift budget is 0.5 % of distance.

## Headline answers

1. **Pattern-tracking flow is biased by 1–3 m/s, everywhere.** SEA-RAFT-M at full resolution
   (Δt = 0.1 s), median over runs:

   | Altitude | 30 m | 60 m | 120 m | 250 m |
   |---|---|---|---|---|
   | Bias (m/s) | 0.98 | 1.39 | 2.10 | 3.08 |
   | Runs | 90 of 108 ¹ | 108 | 108 | 108 |

   ¹ At 30 m and 60 m/s SEA-RAFT collapses onto the midday glint in all 18 midday runs (see
   "Flagging failures"), so that cell is low-sun only.

   The bias points along the wind sea's travel direction: the apparent pattern velocity has a
   median bearing of 59–61° at every heading, and the wind sea of every preset travels toward 60°.
   For cross-swell, whose swell travels toward 140°, the median bearing is 65° (69 runs). In m/s
   it does not depend on ground speed (1.68, 1.60 and 1.80 m/s at 15, 30 and 60 m/s). As a share
   of speed it is therefore 11.2 %, 5.3 % and 3.0 %.

   Only 1 of the 414 SEA-RAFT runs is under 0.28 % of speed, and that run is suspect (see
   Tables). The median run is 5.8 % of speed, about 10× the 0.17 m/s share (derived). UFM-Refine-980
   is larger still: 1.28–3.48 m/s, median 6.5 % of speed.
2. **Wave phase propagation explains the flow bias. Foam drift does not.** Four results show
   this:
   - The flow tracks waves of one size in its own input pixels. If |a| is read as a deep-water
     phase speed, the matching wavelength is 13–15 ground pixels for SEA-RAFT full at every
     altitude. For SEA-RAFT half it doubles to 27–29 px, because that model sees each pixel at
     half size. (Both derived.)
   - |a| grows as altitude to the power 0.45–0.54. A phase speed at a fixed pixel scale grows
     as the power 0.5.
   - Run on the wave-slope layer alone (the `normals` label mode, with no foam, sun, sky or
     clouds), flow gives the same velocity as on the shaded image.
   - It is about the same on a calm sea as on a strong one: 1.83 against 1.46 m/s.

   Foam drift (0.035·U10 = 0.10–0.42 m/s) is 2–13× too small in the seas that have foam (calm
   has none; derived, per sea × altitude medians), and foam covers 0.3–1.6 % of the pixels.
3. **The dispersion-shell fit removes most of the wave bias, but not reliably.** At midday:
   - Its median per-run bias is 0.24, 0.21 and 0.19 m/s at T = 3.2, 6.4 and 12 s, against 1.81 m/s
     for SEA-RAFT. That is about 8× smaller (derived).
   - The tail is long: the 90th percentile is 0.87–1.05 m/s.
   - 17–19 % of midday runs meet 0.28 % of speed, and 38–46 % meet 0.17 m/s.

   The best cell, selected afterwards from 12 altitude × speed cells and 4 windows, is 120 m at
   60 m/s: median 0.16 m/s (0.27 %) at T = 6.4 and 12 s, with 9 and 10 of its 18 runs ≤ 0.28 %
   (95 % Wilson intervals 29–71 % and 34–75 %). Because it was picked as the best of 48, its share
   is optimistic. T = 1.6 s is too short: the median is 0.52 m/s and reaches 2.5 m/s at 250 m. The fit fails in
   two places:
   - At 30 m and 60 m/s the bias is 1.05–1.21 m/s, because the water crosses the 21 m tile in
     0.35 s.
   - In low sun the median is 1.37–1.64 m/s.
4. **Why the shell fit fails in low sun is open.** Two explanations fit the evidence, and the
   evidence does not tell them apart:
   - *Simulator foam / break gate.* The foam layer moves as a pattern at 1.3–2.4 m/s (static
     shell, T = 3.2, median over seas per altitude), the order of the whitecap break-gate scroll
     (1.98 m/s toward 59°), and a calm sea (no foam) is fine in low sun (0.20 m/s at T = 6.4,
     36 runs).
   - *Low-sun shading itself.* Dark wave faces, shadowing and sky reflection at a 7° sun are real
     physics. Against the foam explanation: the foam layer's pattern speed grows with altitude
     (static shell, T = 3.2, per sea: 1.0–1.5 m/s at 30 m to 2.3–2.5 m/s at 250 m), whereas the gate
     scrolls at a constant 1.98 m/s; the error does not scale with foam cover (moderate and
     cross-swell, foam on 0.26–0.27 % of pixels, fail at 1.34 and 1.63 m/s, about as badly as
     strong at 1.6 % and 1.76 m/s); and calm differs from the other seas in its gentler slopes
     (less shadowing) as well as in having no foam.

   The wave-slope (`normals`) and `foam` label layers do not depend on the sun (their per-pair
   flow is identical in the midday and low-sun copies), so the slope layer, where the dispersive
   fit has a median of 0.06–0.12 m/s at T = 1.6–12 over 16 unique sequences, shows only that shading causes the
   failure, not which part of the shading. The dispersive fit can also lock onto a wrong solution
   with no foam present (worst slope-layer run 0.56 m/s at T ≥ 6.4; the audit's clean synthetic
   seas, see "Audit"). It is not a weighting artefact: with 'energy' weighting the low-sun error
   is larger still (2.1–2.6 m/s). Gate-off and foam-off renders are needed to decide.
5. **The frozen-sea control checks the flow geometry and the static spectral pipeline, not the
   dispersive fit.** On 24 frozen runs (midday, 60–120 m, 30–60 m/s), the per-run |bias| is:

   | Estimator | Median / p90 / max, m/s |
   |---|---|
   | SEA-RAFT full, g = 1 | 0.036 / 0.061 / 0.092 |
   | UFM-Refine-980, g = 1 | 0.040 / 0.084 / 0.102 |
   | Shell static model, T = 1.6–12 | 0.034–0.050 / 0.11–0.17 / 0.14–0.27 |

   The static shell model's p90 (0.169 m/s at T = 3.2) and max (0.27 m/s) reach or exceed the
   0.17 m/s budget. The dispersive fit, which is the E1 estimator, cannot be tested on a frozen
   sea (it gives 1.08–1.31 m/s there, as expected); its measurement floor rests only on the
   synthetic sea and the wave-slope layer. The control does not cover 30 m, 250 m, 15 m/s, low
   sun or the oblique set. **UFM-Base failed it**, at 0.140 (0.260) m/s. The error is the model's
   sub-pixel accuracy, so UFM-Base is reported only for reference, and UFM-Refine-980 replaces it.

**What it means for E1:**

- **Does pattern-tracking flow need wave-phase removal?** Yes, at every altitude, speed, sea
  state, heading and sun tested. No slice is within either budget line.
- **Does the shell fit beat it?** Yes, by about an order of magnitude at midday, with T ≥ 3.2 s.
  But it meets the 0.28 % share only in a minority of runs, mainly at 60–120 m. It is not robust
  to the low-sun imagery, for a reason not yet identified (headline 4).

## Setup (exact)

### Renders

Renders come from mare-realtime `scripts/generate.mjs`. The job files are in `../jobs/`.

**Camera and flight, common to all sets:**
- Portrait 720×1280, vertical fov 92°. K from frames.csv: fx = fy = 618.04 px, cx = 360, cy = 640.
- Gimbal yaw 0, stabilized 1 (exact attitude). Pitch −90 (nadir), or −60 for the oblique set.
- 10 fps, 12 s (121 frames), 8 s warm-up.
- Pattern `transit` (straight), wind `none`, turbulence 0, airspeedVariation 0. Ground speed is
  therefore airspeed, and with no current the water-relative velocity equals the ground velocity.
- The camera height used is `cam_u`, about 0.22 m below the aircraft altitude.

**Render sets** (every set completed with 0 failed runs; frames.csv and PNG counts were checked):

| Set | Runs | Sweep | Notes |
|---|---|---|---|
| core | 432 | sea ∈ {calm, moderate, fresh, strong, moderate-swell, cross-swell} × altitude {30, 60, 120, 250} m × airspeed {15, 30, 60} m/s × heading {60 down-wave, 150 cross, 240 up-wave} × light {midday (sun 60° el., az 180), low-sun (7° el., az 250)} | |
| frozen | 24 | render.seaRate 0 from frame 0 (waves, foam, break gate and clouds stop; the camera flies on); sea {moderate, strong, moderate-swell} × altitude {60, 120} × airspeed {30, 60} × heading {60, 150}; midday | |
| oblique | 36 | pitch −60; sea {moderate, strong, cross-swell} × altitude {60, 120} × airspeed {30, 60} × heading {60, 150, 240}; midday | |
| decomp | 32 | sea {calm, moderate, strong, cross-swell} × altitude {30, 60, 120, 250} × light {midday, low-sun}; airspeed 30, heading 60 | Added; not in the original design. Renders the `normals` and `foam` label modes beside the shaded image, so each image layer can be measured alone. |

The first core render attempt failed every run with "Ocean engine: shader error" while another
job filled the GPU's memory. It rendered with no failures when rerun on a free GPU.

### (a) Flow (`flow_bias.py`)

**Pairs.** Pairs are (i, i+g) with g ∈ {1, 3} (Δt 0.1 s and 0.3 s) and i = 5, 10, …, giving
23 pairs per gap per run.

**From flow to velocity.**
- Every pixel p₁ is back-projected onto the plane up = 0 with the frame-i pose, and p₁ + flow
  with the frame-(i+g) pose.
- The apparent pattern velocity a is the component-wise median of the displacement divided by
  Δt.
- Valid pixels are at least 32 px from the border, with rays within 75° of vertical. UFM pixels
  also need covisibility above 0.5.
- The bias of a static-sea estimator is b = −a.

**Per run.** The per-run value is the median over the pairs that are not flagged.

**Models** (pretrained, no fine-tuning; weights from Hugging Face):

| Name | Checkpoint | Input |
|---|---|---|
| `searaft` | SEA-RAFT-M (`MemorySlices/Tartan-C-T-TSKH-spring540x960-M`, spring-M config, 4 iterations) | the full 720×1280 frame |
| `searaft_half` | the same model | 360×640 |
| `ufmrefine980` | UFM-Refine-980 (`infinity1096/UFM-Refine-980`) | inference resolution 980×644 (W×H), see below |
| `ufm` | UFM-Base (`infinity1096/UFM-Base`) | 560×420 |

Both UFM models see the central 720×960 crop (rows 160–1119), transposed to landscape. UFM
rescales it to its inference resolution itself, which for UFM-Refine-980 is not an isotropic
scale. The UFM wrapper maps the flow back to input pixels, and the synthetic warp test below
covers this.

**Flagging failures.** A pair is flagged when either is true:
- the model's flow has collapsed toward zero: |a − v_cam| < 0.25·ground speed;
- the per-pixel velocities scatter by more than 3 m/s (1.4826·MAD in either component).

The pilot rule was |a| ≥ 0.25·ground speed. It would have flagged real pattern speeds at
15 m/s, so it was replaced before the full runs. On the pilot (828 live and 828 frozen pairs) the
new rule flags 135 live and 56 frozen pairs, the old rule 124 and 49. Every pair the old rule
flags, the new one flags too; the extra 18 are all SEA-RAFT-half, g = 3, at 60 m and 60 m/s,
with |a| of 2.4–14.7 m/s and a scatter above 3 m/s, which the scatter rule catches.

**SEA-RAFT full collapses onto the midday glint, not simply on large flow.** Its failures
depend on the sun (`audit_checks.csv`, shares of flagged pairs, 414 pairs per cell; flow in
pixels is the camera's motion, derived):

| Pairs (flow) | Low sun | Midday |
|---|---|---|
| g = 1, 30 m, 60 m/s (124 px) | 0.02 | 1.00 |
| g = 3, 30 m, 15 m/s (93 px) | 0.31 | 1.00 |
| g = 3, 60 m, 30 m/s (93 px) | 0.04 | 1.00 |
| g = 3, 120 m, 60 m/s (93 px) | 0.06 | 0.96 |
| g = 3, 30 m/30 m/s, 60 m/60 m/s (186 px), 30 m/60 m/s (374 px) | 1.00 | 1.00 |
| all other cells (≤ 62 px) | 0.00 | 0.00 |

- At 30 m and 60 m/s, g = 1, the midday pairs collapse: median measured flow 17 px against the
  camera's 124 px, and |a| ≈ 58 m/s, i.e. the camera's own motion. The model locks onto the glint
  patch, which stays fixed in the image. The 18 low-sun runs of that cell pass at the same 124 px
  (|a| 1.00 m/s). So the SEA-RAFT g = 1 value for that cell, and its 30 m column, come from
  low-sun runs only (18 of 36 runs in the cell).
- In low sun SEA-RAFT full works up to at least 124 px and fails at 186 px. At midday it fails
  from about 90 px.
- At g = 3, 158 SEA-RAFT core runs have all pairs flagged.
- At 250 m the per-pixel scatter of non-collapsed pairs (90th percentile) is 0.46–0.57 m/s for
  SEA-RAFT full and half and UFM-Refine-980, and 0.71–0.99 m/s for UFM-Base, so the scatter rule
  (3 m/s) does not fire on genuine wave motion.

**Headline gap.** The headline uses g = 1 (the camera's real rate). g = 3 gives 2–8 % more |a|
on matched runs.

### (b) Shell fit (`shell_fit.py`)

**Tile.** Each frame is rectified onto a 256² ENU tile fixed to the camera, centred on the
nadir point or, for pitch −60, on the footprint centre. The side is 0.6 × the footprint width
(21 m at 30 m, 42 m at 60 m, 84 m at 120 m, 174 m at 250 m), set from the true attitude and
height only.

**Preparation.**
- Linear luminance.
- Each tile is divided by its mean (the renderer's auto-exposure).
- Each cell's temporal mean is removed (the fixed glint patch).
- A Hann window in x, y and t, and a 3-D FFT padded ×4 in t.

**Fit.**
- Wavelengths from L/3 down to 0.8 × the tile's Nyquist.
- Each k bin is whitened.
- The band is ±1 temporal bin.
- Both branches of ω' = ±√(g|k|) − k·V are used, wrapped modulo 2π·fps (temporal aliasing).
- The prior is the truth plus U(±5) m/s per component. The search covers a ±8 m/s box, with
  grids of 0.2 then 0.02 m/s and a quadratic peak refinement.

**Windows.** T ∈ {1.6, 3.2, 6.4, 12} s, with starts every 1.6 s: 7, 6, 4 and 1 windows per run.
The per-run value is the median over windows.

**Models.** Two models are fitted:
- 'dispersive', the E1 estimator;
- 'static' (ω' = −k·V), a spectral pattern tracker. On a frozen sea it is the correct model.

**Validation.** On a synthetic linear sea in tile space (pilot stage,
`shell-summary-synthetic*.csv`), the median |error| at T = 3.2 is 0.003, 0.008 and 0.026 m/s at
15, 30 and 60 m/s. This test feeds the fit a tile stack directly; it does not cover the image
path (render → rectify → fit). Through the full image path the floor is higher: the audit's clean
synthetic seas gave 0.07–0.28 m/s depending on weight and window, worst at T = 12 (a single
window), and the dispersive fit failed outright in 3 of 6 configurations (see "Audit"). On the
real renders every window with a large error is a maximum of the objective itself
(J_est ≥ J_truth in 100 % of windows with error > 0.5 m/s in core and the slope layer, and
> 0.15 m/s in the frozen set), so these errors come from the objective, not from the search.

### Conventions

- Bias b = v_est − v_true, in m/s.
- Along-track positive means the estimate overstates the speed.
- Cross-track positive is to the right of track.
- Sea directions are compass degrees the waves travel *toward* (src/ocean/scene.js). Every
  preset's wind sea travels toward 60°; moderate-swell's swell toward 75° and cross-swell's
  toward 140° (run.json).
- Along-wave (`b_along_wave`, `a_along_wave`, `err_along_wave`) is the component along the wind
  sea's direction, 60° in every preset. For cross-swell it is therefore along the wind sea, not
  the swell. Along-swell (`b_along_swell`) is the component along the swell's direction, for the
  two swell seas only.
- % of speed = 100·|b| / ground speed.
- "Median over runs" is over the per-run values within a slice. Other variables in the slice
  are pooled.

## Tables

### 1. Core grid (nadir), median |b| by altitude (m/s, then % of speed)

Flow at g = 1; all suns pooled. n = 108 runs per altitude, except SEA-RAFT full at 30 m: 90 runs,
because its 18 midday runs at 60 m/s collapse onto the glint, so its 30 m entry has no midday
runs at 60 m/s.

| Estimator | 30 m | 60 m | 120 m | 250 m |
|---|---|---|---|---|
| SEA-RAFT full | 0.98 (3.6 %), n = 90 | 1.39 (4.6 %) | 2.10 (7.2 %) | 3.08 (10.2 %) |
| UFM-Refine-980 | 1.28 (4.4 %) | 1.73 (5.7 %) | 2.43 (8.1 %) | 3.48 (11.6 %) |
| SEA-RAFT half | 1.35 (4.6 %) | 2.02 (7.2 %) | 2.93 (10.0 %) | 4.01 (13.6 %) |
| UFM-Base (failed control) | 1.61 (5.4 %) | 2.20 (7.4 %) | 2.99 (9.9 %) | 4.13 (13.7 %) |
| Shell static, T = 3.2 | 0.93 (3.1 %) | 1.22 (4.0 %) | 1.71 (5.7 %) | 2.53 (8.5 %) |
| Shell dispersive, T = 3.2 | 0.55 (1.4 %) | 0.64 (1.5 %) | 0.33 (1.0 %) | 0.68 (2.3 %) |
| Shell dispersive, T = 6.4 | 0.63 (1.7 %) | 0.60 (1.3 %) | 0.30 (1.0 %) | 0.77 (2.5 %) |
| Shell dispersive, T = 12 | 0.97 (2.1 %) | 0.75 (1.4 %) | 0.33 (1.0 %) | 0.80 (2.1 %) |

Pooled over the whole core grid:

| Estimator | n runs | Median \|b\| | p90 | Median % of speed | p90 % | Runs ≤ 0.28 % | Runs ≤ 0.5 % |
|---|---|---|---|---|---|---|---|
| SEA-RAFT full | 414 | 1.68 m/s | 3.11 | 5.8 % | 14.9 % | 1 of 414 | 1 of 414 |
| UFM-Refine-980 | 432 | 2.03 m/s | 3.53 | 6.5 % | 16.8 % | 0 | 0 |
| Shell dispersive, T = 3.2 | 432 | 0.61 m/s | 2.09 | 1.66 % | 8.5 % | 9.0 % | 19.0 % |
| Shell dispersive, T = 6.4 | 432 | 0.65 m/s | 2.27 | 1.52 % | 8.7 % | 10.2 % | 20.1 % |
| Shell dispersive, T = 12 | 432 | 0.79 m/s | 2.42 | 1.63 % | 9.5 % | 10.6 % | 19.4 % |

The SEA-RAFT run under 0.28 % (strong, low-sun, 30 m, 60 m/s: 0.14 m/s) sits in the cell where
the model fails, and 6 of its 23 pairs were flagged. Treat it as a partial failure, not as a
good case.

### 2. Flow: ground speed does not change the bias in m/s, only its share

SEA-RAFT full, g = 1. Median |b| in m/s, with % of speed in brackets. n = 36 runs per cell
(both suns), except 30 m at 60 m/s: n = 18, low sun only (the midday runs collapse onto the
glint).

| Altitude \ speed | 15 m/s | 30 m/s | 60 m/s |
|---|---|---|---|
| 30 m | 0.96 (6.4 %) | 1.01 (3.4 %) | 0.90 (1.5 %), low sun only, n = 18 |
| 60 m | 1.42 (9.5 %) | 1.39 (4.6 %) | 1.36 (2.3 %) |
| 120 m | 2.10 (14.0 %) | 2.17 (7.2 %) | 2.07 (3.5 %) |
| 250 m | 3.03 (20.2 %) | 3.07 (10.2 %) | 3.12 (5.2 %) |

The best cell is still 5× the 0.28 % share (derived). For UFM-Refine-980 the cells range from
2.15 % (30 m, 60 m/s) to 23.2 % (250 m, 15 m/s).

### 3. Heading relative to the waves (core; median b in m/s)

n per heading: SEA-RAFT 138 runs (both suns), shell 72 runs (midday).

| Heading | SEA-RAFT along-track | SEA-RAFT cross-track | Shell dispersive T = 6.4, midday, along | Shell dispersive T = 6.4, midday, cross |
|---|---|---|---|---|
| 60 (down-wave) | −1.46 | −0.04 | −0.08 | −0.05 |
| 150 (cross) | 0.05 | +1.95 | 0.00 | +0.11 |
| 240 (up-wave) | +2.08 | 0.04 | +0.23 | +0.06 |

- **SEA-RAFT.** Flying down-wave, flow understates the speed; flying up-wave, it overstates it.
  On cross headings the bias turns into a cross-track error, which is a heading error of about
  atan(|b|/v) (derived: 3.7° at 30 m/s and 1.9 m/s). The median bearing of a is 61.4°, 58.8°
  and 60.9° for headings 60, 150 and 240.
- **Shell dispersive.** At midday its residual follows the same sign pattern, about 9–18× smaller
  (derived from the table: 1.46/0.08, 1.95/0.11, 2.08/0.23).
- Headings are relative to the wind sea (60°). For cross-swell the swell travels toward 140°, so
  "down-wave" and "up-wave" there refer to the wind sea only; that sea's SEA-RAFT bias is 1.62 m/s
  along the wind sea and 0.44 m/s along the swell (median a, 69 runs, `audit_checks.csv`).

### 4. Sea state and sun (core; median |b| in m/s)

n per sea: SEA-RAFT 69 runs (both suns; 3 midday runs at 30 m/60 m/s are excluded per sea),
shell 36 runs per sun.

| Sea | SEA-RAFT g = 1 | Shell dispersive T = 6.4, midday | Shell dispersive T = 6.4, low sun |
|---|---|---|---|
| calm | 1.83 | 0.25 | 0.20 |
| moderate | 1.69 | 0.17 | 1.34 |
| fresh | 1.55 | 0.20 | 1.57 |
| strong | 1.46 | 0.24 | 1.76 |
| moderate-swell | 1.65 | 0.16 | 1.41 |
| cross-swell | 1.63 | 0.19 | 1.63 |

Sun changes the flow bias a little. Over all runs SEA-RAFT gives 1.81 m/s at midday (198 runs)
and 1.56 in low sun (216), but that compares different cells: the 18 extra low-sun runs are the
30 m/60 m/s cell, where midday collapses. On the 198 matched cells (same sea, altitude, speed and
heading) the medians are 1.81 and 1.66 m/s, and the median per-cell difference is −0.10 m/s
(low sun lower). The sun decides whether the shell fit works: at T = 6.4 the median is 0.21 m/s at midday and 1.37 m/s in
low sun.

### 5. Shell dispersive at midday, by altitude × speed (median |b|, m/s, then % of speed)

n = 18 runs per cell (6 seas × 3 headings).

| Altitude | T | 15 m/s | 30 m/s | 60 m/s |
|---|---|---|---|---|
| 30 m | 3.2 | 0.10 (0.63 %) | 0.18 (0.59 %) | 1.11 (1.85 %) |
| 30 m | 6.4 | 0.12 (0.78 %) | 0.18 (0.60 %) | 1.05 (1.75 %) |
| 60 m | 3.2 | 0.12 (0.79 %) | 0.09 (0.28 %) | 0.73 (1.21 %) |
| 60 m | 6.4 | 0.11 (0.76 %) | 0.09 (0.30 %) | 0.52 (0.87 %) |
| 120 m | 3.2 | 0.10 (0.70 %) | 0.13 (0.43 %) | 0.21 (0.35 %) |
| 120 m | 6.4 | 0.10 (0.66 %) | 0.15 (0.50 %) | 0.16 (0.27 %) |
| 120 m | 12 | 0.12 (0.78 %) | 0.16 (0.54 %) | 0.16 (0.27 %) |
| 250 m | 6.4 | 0.19 (1.29 %) | 0.31 (1.05 %) | 0.41 (0.68 %) |

At T = 1.6 the 250 m medians are 2.48–2.64 m/s. The cells where at least half the runs are
≤ 0.28 % are below. They were selected from 12 cells × 4 windows, so their shares are optimistic;
the intervals are 95 % Wilson intervals on n = 18.

| Cell | T | Runs ≤ 0.28 % | 95 % interval |
|---|---|---|---|
| 120 m, 60 m/s | 6.4 | 9 of 18 (50 %) | 29–71 % |
| 120 m, 60 m/s | 12 | 10 of 18 (56 %) | 34–75 % |
| 60 m, 30 m/s | 3.2 | 9 of 18 (50 %) | 29–71 % |

All midday runs, by window length (n = 216):

| T | Median | p90 | Runs ≤ 0.17 m/s | Runs ≤ 0.28 % | Runs ≤ 0.5 % |
|---|---|---|---|---|---|
| 1.6 s | 0.52 m/s | 2.55 | 18 % | 3 % | 12 % |
| 3.2 s | 0.24 m/s | 0.91 | 38 % | 17 % | 31 % |
| 6.4 s | 0.21 m/s | 0.87 | 42 % | 19 % | 33 % |
| 12 s | 0.19 m/s | 1.05 | 46 % | 19 % | 33 % |

### 6. Frozen-sea control (24 runs; per-run |b| in m/s: median / p90 / max)

| Estimator | g = 1 or shorter T | g = 3 or longer T |
|---|---|---|
| SEA-RAFT full | 0.036 / 0.061 / 0.092 | 0.028 / 0.069 / 0.119 (18 runs; 6 runs at 60 m/s and 60 m fully flagged) |
| SEA-RAFT half | 0.053 / 0.148 / 0.174 | 0.076 / 0.279 / 0.330 (40 flagged pairs) |
| UFM-Refine-980 | 0.040 / 0.084 / 0.102 | 0.048 / 0.169 / 0.298 |
| UFM-Base | **0.140 / 0.230 / 0.260** | 0.104 / 0.301 / 0.380 |
| Shell static | T = 1.6: 0.034 / 0.124 / 0.267; T = 3.2: 0.049 / 0.169 / 0.267 | T = 6.4: 0.036 / 0.114 / 0.140; T = 12: 0.050 / 0.127 / 0.139 |

- The control covers midday, 60–120 m and 30–60 m/s only. It says nothing directly about 30 m,
  250 m, 15 m/s, low sun or the oblique set.
- The shell static model's largest residuals are in the strong sea at 60 m, along-track. Its
  per-window RMS reaches 0.72 m/s at T = 1.6. Its p90 (0.17 m/s at T = 3.2) and max (0.27 m/s)
  reach or exceed the 0.17 m/s budget.
- **Why the flow residual is not zero** (from the audit, derived): with exact geometry a frozen
  pair gives a = 0 exactly. The SEA-RAFT g = 1 residual is mostly along-track and negative
  (per-run |a_along| median 0.036 m/s; median a_along −0.015 m/s at 30 m/s and −0.049 at 60 m/s;
  |a_cross| median 0.008) and grows with the flow, so it is mainly the model under-reading large
  displacements. A second, smaller term is parallax: the frozen snapshots' mean
  `seaHeightUnderCamera` is about +0.05 m (moderate, moderate-swell) and +0.17 m (strong), not 0,
  which predicts a ≈ −v·η̄/h, a few hundredths of a m/s with the measured sign; the audit found
  the per-run correlation weak (r = 0.2–0.7, not rechecked). For the live core seas the audit
  derived at most ~0.13 m/s at 30 m and 60 m/s and ≤ 0.02 m/s at ≥ 120 m from the same formula
  (an upper estimate).
- The dispersive model cannot be checked on a frozen sea, since there is no dispersion to fit.
  It gives 1.08–1.31 m/s there, as expected. Its measurement floor rests only on the synthetic
  sea (tile space; and the audit's image-path synthetics, 0.07–0.28 m/s) and on the wave-slope
  layer (Table 8: median 0.06–0.12 m/s, worst run 0.56 m/s).
- UFM-Base did not pass. Its pilot error on exact synthetic warps had the same size
  (`flow-synth-pilot-frozen.csv`), so this is the model's sub-pixel accuracy.
- Of the UFM checkpoints tried on the pilot (`flow-synth-ufm-variants.csv`,
  `flow-pilot-frozen-ufm-variants.csv`), UFM-Refine-980 was best: its synthetic-warp median
  error is 0.049 m/s at g = 1 and 0.026 at g = 3. It replaces UFM-Base as "UFM" in the
  conclusions.

### 7. What the flow tracks: phase speed vs foam drift (core, g = 1)

Median over the seas of each altitude's median |a|. Every column other than |a| is derived.

| Altitude | GSD | SEA-RAFT \|a\| | λ_eq = 2π\|a\|²/g | λ_eq in px | c(λ = 3 px) | c(4 px) | c(16 px) | Foam drift 0.035·U10 |
|---|---|---|---|---|---|---|---|---|
| 30 m | 0.05 m | 1.01 m/s | 0.65 m | 13.5 | 0.48 | 0.55 | 1.10 | 0.10–0.42 |
| 60 m | 0.10 m | 1.42 m/s | 1.28 m | 13.3 | 0.67 | 0.78 | 1.55 | 0.10–0.42 |
| 120 m | 0.19 m | 2.13 m/s | 2.90 m | 14.9 | 0.95 | 1.10 | 2.20 | 0.10–0.42 |
| 250 m | 0.40 m | 3.10 m/s | 6.14 m | 15.2 | 1.38 | 1.59 | 3.18 | 0.10–0.42 |

λ_eq in ground pixels for the other models (30 / 60 / 120 / 250 m):

| Model | λ_eq (px) |
|---|---|
| SEA-RAFT half | 26.8 / 28.6 / 28.7 / 26.8 |
| UFM-Refine-980 | 22.3 / 20.3 / 19.8 / 18.8 |
| UFM-Base | 34.1 / 32.8 / 29.6 / 26.7 |

The slope of log|a| against log h is 0.54 (SEA-RAFT full), 0.49 (half), 0.47 (UFM-Refine-980)
and 0.45 (UFM-Base). Per sea it is 0.46–0.62 for SEA-RAFT full.

**Reading.**
- The flow follows waves of about 13–15 of its input pixels, about four times the shortest
  resolvable wavelength (3–4 px).
- A faster wave shows up with a coarser model input: SEA-RAFT half sees twice the λ_eq in
  ground pixels and 1.4× the speed.
- The phase speed of the shortest resolvable waves, c(3–4 px), is a lower bound. It sits a
  factor of about 2 below the measurement.
- Foam drift is 2–13× too small (seas with foam; derived). The break-gate scroll (1.98 m/s) is
  constant, whereas the measured bias grows with altitude.
- For cross-swell the flow follows the wind sea, not the larger swell (swellHeight 1.8 m; the same
  wind sea alone, the moderate preset, has Hs 1.25 m): median a is 1.62 m/s along
  the wind sea (60°) and 0.44 m/s along the swell (140°), and its bearing is 65° (69 runs). The
  bias follows the short waves at the model's pixel scale, not the dominant waves.

### 8. Decomposition by image layer

The decomp set: 30 m/s, heading 60, median over seas. Each value is a median over 4 seas, one
run each (3 for the foam layer: calm is excluded because it has no foam). Values are median |b|
in m/s at 30 / 60 / 120 / 250 m.

The label layers (`normals`, `foam`) do not depend on the sun: their per-pair flow is identical in
the midday and low-sun copies. So the 32 decomp runs hold 16 unique slope sequences and 16 unique
foam sequences, and the slope and foam columns describe both suns. (The shell values of the two
copies can differ slightly, because the random prior offset is seeded by run id.) On the slope
layer the shell uses the normal's horizontal component toward 60°; for cross-swell, whose swell
travels toward 140°, that weights the swell's slope by cos 80° ≈ 0.17 (derived).

| Estimator | Wave slope only (normals) | Shaded, midday | Shaded, low sun | Foam only |
|---|---|---|---|---|
| SEA-RAFT full, g = 1 | 0.95 / 1.25 / 1.44 / 2.40 | 0.97 / 1.24 / 1.83 / 2.83 | 0.90 / 1.06 / 1.56 / 2.69 | 0.54 / 0.58 / 0.91 / 1.39 |
| UFM-Refine-980, g = 1 | 1.44 / 2.04 / 2.90 / 4.11 | 1.23 / 1.71 / 2.49 / 3.35 | 1.20 / 1.55 / 2.30 / 3.32 | 0.76 / 0.96 / 1.28 / 1.96 |
| Shell static, T = 3.2 | 0.84 / 1.14 / 1.56 / 2.25 | 0.95 / 1.09 / 1.67 / 2.67 | 0.85 / 1.01 / 1.40 / 1.97 | 1.34 / 1.59 / 1.91 / 2.35 |
| Shell dispersive, T = 3.2 | **0.05 / 0.02 / 0.06 / 0.08** | 0.06 / 0.06 / 0.07 / 0.33 | 0.61 / 0.65 / 1.68 / 1.83 | 1.55 / 1.63 / 2.30 / 2.25 |
| Shell dispersive, T = 6.4 | 0.10 / 0.05 / 0.12 / 0.16 | 0.12 / 0.06 / 0.17 / 0.38 | 0.58 / 0.67 / 1.31 / 1.94 | — |

- Flow and the static shell model see about the same pattern velocity on the wave-slope layer as
  on the shaded image. The bias is in the waves themselves.
- On the slope layer the dispersive fit's median over seas is ≤ 0.16 m/s at T = 3.2 and 6.4 at
  every altitude, and over the 16 unique sequences the median is 0.06 (T = 3.2) and 0.12 m/s
  (T = 6.4). Single runs are worse: the worst is 0.24 m/s at T = 3.2, 0.52 at T = 6.4, 0.56 at
  T = 12 and 3.57 at T = 1.6 (all calm, 250 m). Shading at midday adds little, except at 250 m.
  Low-sun shading adds 0.5–1.8 m/s.
- On the foam layer, flow uses 0.26 % (cross-swell), 0.27 % (moderate) and 1.6 % (strong) of the
  pixels. Foam patterns move at 0.5–2.4 m/s (flow 0.54–1.96 m/s, static shell fit 1.34–2.35 m/s,
  medians over seas). Their median bearing (of −b, over the foam runs, per window or gap) is
  59–66° for flow, 61–67° for the static shell fit and 49–62° for the dispersive fit, with a wide
  per-run spread (15–118° for the dispersive fit).

### 9. Oblique (pitch −60, 36 runs, midday)

Median |b|, m/s:

| Estimator | All runs | Matched nadir core runs (same sea, altitude, speed and heading, midday) |
|---|---|---|
| SEA-RAFT full, g = 1 | 1.93 (60 m: 1.65, 120 m: 2.20) | 1.62 |
| UFM-Refine-980, g = 1 | 2.12 | 2.02 |
| Shell static, T = 3.2 | 1.30 | |
| Shell dispersive, T = 3.2 / 6.4 / 12 | 0.35 / 0.27 / 0.35 | 0.23 / 0.22 / 0.17 |

The oblique dispersive fit is good at headings 150 and 240 (0.15–0.18 m/s) and poor at heading
60 (1.23–1.48 m/s). This was not investigated.

SEA-RAFT as % of speed:

| Altitude | 30 m/s | 60 m/s |
|---|---|---|
| 60 m | 5.5 % | 2.8 % |
| 120 m | 7.4 % | 3.6 % |

### 10. Gap and failures

On matched runs, g = 3 gives more |a| than g = 1:

| Model | Matched runs | g = 1 | g = 3 |
|---|---|---|---|
| SEA-RAFT | 274 | 2.19 m/s | 2.30 m/s |
| UFM-Refine-980 | 416 | 2.14 m/s | 2.39 m/s |
| SEA-RAFT half | 363 | 2.74 m/s | 2.84 m/s |
| UFM-Base | 416 | 2.60 m/s | 2.74 m/s |

This fits a longer gap favouring longer, faster waves, but that is an interpretation and was
not tested.

Flagged core pairs at g = 1 / g = 3:

| Model | g = 1 | g = 3 |
|---|---|---|
| SEA-RAFT full | 421 | 3881 |
| SEA-RAFT half | 7 | 1845 |
| UFM-Refine-980 | 4 | 1248 |
| UFM-Base | 3 | 858 |

`summary_by_slice.csv` lists the flagged pairs and the excluded runs for every slice.

## Simulator artefacts that may inflate or deflate the result

| Artefact | Effect on the result | Evidence |
|---|---|---|
| Whitecap break gate scrolls at about 1.98 m/s toward 59°. Fresh whitecaps appear where it allows and ride with the crests. | **May inflate** the shell error in low sun and add a non-dispersive line (one of two open explanations; headline 4). Its effect on flow is small, because foam covers 0.3–1.6 % of the pixels. | The foam layer's static shell speed is 1.3–2.4 m/s toward about 60°, but it grows with altitude while the gate is constant, and the low-sun error does not scale with foam cover. A low-sun calm sea, which has no foam, is fine (0.20 m/s), but calm also has gentler slopes. Not proven: no gate-off or foam-off render was made. |
| Low-sun shading (7° sun): dark wave faces, shadowing and sky reflection | **Inflates** the shell error, from 0.21 to 1.37 m/s at T = 6.4 — either by itself or by leaving foam bright (open). | Flow changes little: on 198 matched cells 1.66 against 1.81 m/s at midday. Whether a real low-sun sea behaves the same is unknown. |
| Every preset's wind sea travels toward 60°; the swells travel toward 75° (moderate-swell) and 140° (cross-swell). Directions are 'toward' (src/ocean/scene.js). | Directions are confounded for the wind sea: wave phase, foam, break gate and cloud drift all point near 60°. | The decomposition by layer separates them, not the direction. Cross-swell's flow bias follows the wind sea (1.62 m/s along 60°, 0.44 along 140°). |
| Cloud shadows and reflections drift at 2 + 0.6·U10 m/s downwind (engine.js). This is 3.8–9.2 m/s, derived. | Could **inflate** flow and shell where clouds are visible. | Overcast is 0.05 at midday and 0.1 in low sun. The normals layer has no clouds and gives the same flow velocity as the shaded image, so clouds are not a large term here. Not measured separately. |
| Wave vectors lie on discrete cascade grids, frequencies are quantised to 2π/1200 s, waves are linear plus choppiness. | May **deflate** the shell error compared with a real nonlinear sea, since the linear shell is exact here. | On the slope layer the dispersive fit's median over seas is ≤ 0.16 m/s at T = 3.2–6.4 (worst run 0.56 m/s). A real sea has bound harmonics and a wind drift layer that are not rendered. |
| No surface current, no wind drift of short waves (the sea has no current) | **Deflates** the bias compared with reality. Real short waves ride a wind-drift layer of about 3 % of U10, which no image method can separate from ego-motion. | That is E4's subject. |
| Midday glint patch fixed in the image | Makes SEA-RAFT full **collapse** onto zero image motion once the camera's flow reaches about 90 px: at midday 100 % of g = 1 pairs at 30 m/60 m/s (124 px) and 96–100 % of g = 3 pairs at 93 px are flagged, against 2 % and 4–31 % in low sun. Below that it pulls flow only slightly. Gives the shell an ω' = 0 line, removed by mean subtraction. | "Flagging failures" table. On the frozen control (≤ 62 px of flow at g = 1) the residuals of SEA-RAFT and UFM-Refine-980 are ≤ 0.10 m/s. |
| Renderer frame-to-frame view-dependent shading (pilot frozen NCC 0.95–0.99) | The floor of the frozen control, a few cm/s. | Table 6. |

## Departures from the design, and why

- **UFM-Base was replaced by UFM-Refine-980.** UFM-Base failed the frozen control: 0.14 m/s
  median and 0.26 max, against a 0.17 m/s budget. UFM-Base still appears in every CSV and plot,
  for reference.
- **SEA-RAFT also runs at half resolution.** This is not a substitute for the full model. It
  shows how the bias depends on scale, and gives a working g = 3 at 30–60 m, where the
  full-resolution model fails.
- **Failure rule.** The pilot rule was replaced by the collapse and scatter rule above, so that
  genuine pattern speeds at 15 m/s are not flagged. On the pilot the new rule flags 135 live and
  56 frozen pairs, the old one 124 and 49; the 18 extra pairs are SEA-RAFT-half g = 3 partial
  failures that only the scatter rule catches.
- **Per-run statistic.** The per-run value is a median over pairs or windows. For the shell,
  `per_run.csv` also carries the window mean (`b_E_mean`, `b_N_mean`) and the RMS.
- **Wave directions.** The along-wave component is along the wind sea (60° in every preset). The
  measurement scripts now read `windDirection` and `swellDirection` from run.json and also write
  an along-swell component; `analyze.py` derives it for the existing CSVs from the preset
  directions (75° moderate-swell, 140° cross-swell, both 'toward').
- **The decomp job is an addition.** It renders 32 runs with the `normals` and `foam` label
  modes. `flow_bias.py` gained `--mode` and `--mask-min-lum` (40/255 for foam), and
  `shell_fit.py` / `rectify.py` gained `--mode`. On normals, the shell uses the normal's
  horizontal component toward 60° as a scalar linear in slope.
- **Shell departures** carried over from the build stage: shell wrapping modulo 2π·fps,
  per-tile mean normalisation, per-cell temporal-mean removal, whitening, and k from L/3 to
  0.8 × Nyquist (see the `shell_fit.py` docstring).

## Next steps

1. Add switches for gate-off and foam-off, then rerun low sun to decide between the two
   explanations in headline 4 (foam / gate against low-sun shading). Then make the gate and whitecaps move with the breaking crests, as on a real
   sea, rather than with an arbitrary scroll.
2. Improve the shell fit:
   - foam/glint masking or robust weighting in k;
   - a band that includes the dwell smear (V·Δk);
   - a tile that moves with the prior at high speed and low altitude. At 30 m and 60 m/s the
     dwell is 0.35 s.

   Then measure again against the 0.28 % line.
3. The flow bias is a phase speed at the model's own pixel scale, so a learned head trained on
   **water-relative labels** must see a stack of frames (dispersion needs time). Test whether a
   multi-frame SEA-RAFT fine-tuned on these renders, with the true velocity as the label, gets
   under 0.17 m/s. The midday shell fit, at 0.19–0.24 m/s median, is the bar.
4. Run real-flight imagery through the same flow-to-metric pipeline. As an order of magnitude,
   expect a SEA-RAFT full bias of order 1–2 m/s at 60–120 m (Table 1 gives 1.4 and 2.1 m/s on
   this simulated sea; the real sea's spectrum and wind drift will change it). Its direction
   should follow the short wind waves at the model's pixel scale, which on a real sea can differ
   from the dominant waves' direction and ride a wind-drift layer. This sim tested that only in
   cross-swell (the bias followed the 60° wind sea, not the 140° swell); every wind sea here
   travels toward 60°, together with its foam, gate and cloud drift.
5. Add sun elevations between 7° and 60°, an overcast sky, and a nonlinear (Stokes) sea, to see
   how the shell fit degrades between the two suns tested.

## Figures

- `fig_bias_vs_altitude.png`: |b| against altitude, one panel per sea state, with the derived
  c(λ = 3 px), c(λ = 16 px) and the foam drift.
- `fig_bias_vs_heading.png`: along- and cross-track bias by heading relative to the waves.
- `fig_bias_vs_speed.png`: |b| in m/s and as % of speed, with the 0.17 m/s, 0.28 % and 0.5 %
  lines.
- `fig_frozen_control.png`: frozen against matched live runs, per estimator.
- `fig_flow_vs_shell.png`: % of speed per estimator and window, by altitude, against the
  budget lines.
- `fig_pct_budget.png`: cumulative distribution of % of speed over all core runs.
- `fig_phase_speed.png`: measured |a| against height with the derived phase speeds, and
  λ_eq in pixels.
- `fig_decomposition.png`: the image-layer decomposition (Table 8).

## Reproducibility and cost

Flow, for all four models on one RTX PRO 6000 Blackwell Max-Q:
- 432 core runs took 5798 s wall, part of it while sharing the GPU with other jobs.
- Model time per pair in the core pass: SEA-RAFT full 64.6 ms, half 12.9 ms, UFM-Base 33.4 ms,
  UFM-Refine-980 147.1 ms.
- On the frozen pass, which had the GPU alone for most of its run, the same times were 49.0, 9.8,
  25.0 and 111.2 ms.

Renders took 90 GB for core, 5.6 GB frozen and 9.4 GB oblique. They stay on the box under
`/workspace/data/e1/`. Commands are in [../README.md](../README.md).

## Audit (29 October 2026)

An audit of geometry and claims raised ten points. Each was checked against the CSVs
(`audit_checks.csv` holds the numbers) before anything was changed.

**Fixed in code**

- **Cross-swell's "along-wave" was along the wind sea, not the swell** (low). Sea directions are
  'toward' (src/ocean/scene.js), and cross-swell's swell travels toward 140°, its wind sea toward
  60°. This settles the earlier open question about the convention. `flow_bias.py` and
  `shell_fit.py` now read `windDirection` and `swellDirection` from run.json and write
  `wind_dir`, `swell_dir` and an along-swell column; `analyze.py` adds `b_along_swell` (from the
  preset directions for the existing CSVs) to `per_run.csv`, `summary_by_slice.csv`,
  `phase_speed.csv` and `decomp_by_layer.csv`. No measurement changes: the components are
  projections of the same a. Measured consequence: cross-swell's SEA-RAFT bias bearing is 65°
  (every other sea 60.2–60.9°), 1.62 m/s along the wind sea and 0.44 m/s along the swell. No
  headline number used along-wave. `pytest tests/` passes 22/22 on the box with the change.

**Corrected in this document**

- **SEA-RAFT failures at 30 m/60 m/s are a midday glint collapse** (high). Confirmed: all 414
  midday g = 1 pairs there are flagged (median flow 17 px instead of 124, |a| ≈ 58 m/s), against
  7 of 414 in low sun; at g = 3 the same split holds at 93 px. Table 1's 30 m entry and Table 2's
  30/60 cell are now marked low-sun only (n = 18 of 36); the sun comparison is given on 198
  matched cells (1.81 vs 1.66 m/s instead of 1.81 vs 1.56); the glint row of the artefact table
  gives the collapse rates.
- **Headline 4 overstated the low-sun attribution** (high). Confirmed: the label layers do not
  depend on the sun (identical per-pair flow), the foam layer's pattern speed grows with altitude,
  and the low-sun error does not scale with foam cover. Headline 4 is now an open hypothesis
  (foam / gate against low-sun shading); "partly confirmed" and "not a limit of the method" are
  gone.
- **The frozen control was claimed to cover every pipeline** (medium). Confirmed: it cannot test
  the dispersive fit and covers midday, 60–120 m and 30–60 m/s only; the static model's p90 and
  max reach 0.17 and 0.27 m/s. The Question, headline 5, Table 6 and the README are narrowed.
- **Pilot failure rules do not flag the same pairs** (medium). Confirmed: new rule 135 / 56,
  old rule 124 / 49; the 18 extra are SEA-RAFT-half g = 3 partial failures. Corrected.
- **Small n reported without n** (medium). Confirmed. Tables 1–5 and 8 now give n; the
  slope-layer result is a median over 4 seas (16 unique sequences) with its worst runs; the
  best cells carry n = 18, Wilson intervals and a note that they were selected.
- **Small numeric errors** (low). Confirmed and corrected: foam drift 2–13× too small (not 3–30×;
  calm has no foam), shell vs SEA-RAFT by heading 9–18× (not 6–10×), UFM-Refine-980 at 250 m
  11.6 % (not 11.7 %), pooled shell T = 12 ≤ 0.28 % 10.6 % (not 10.7 %), 250 m scatter p90 per
  model (UFM-Base reaches 0.99 m/s), foam bearings per estimator.
- **Real-flight direction prediction** (low). Next step 4 now gives an order of magnitude and
  says the direction should follow the short wind waves, which the sim tests only in cross-swell.
- **The shell fit's < 0.05 m/s validation does not cover the image path** (low). The audit's
  image-path synthetic results (0.07–0.28 m/s floor, 3 of 6 dispersive configurations failing)
  could not be rerun here, because its scripts were deleted; they are quoted as the audit's. What
  was checked on the real renders: (1) every large-error window, in core, frozen and the slope
  layer, has J_est ≥ J_truth, so the errors are maxima of the objective, not search failures;
  (2) the dispersive fit errs without foam or sun (slope layer worst run 0.56 m/s at T ≥ 6.4,
  3.57 at T = 1.6), which supports softening headline 4. The audit suggested 'energy' or 'sqrt'
  weighting; both were rerun on real renders (`shell-*-energy.csv`, `shell-frozen-sqrt.csv`,
  methods `shell:<model>/<weight>`). Median per-run |b| in m/s:

  | Set | Model | whiten (default) | energy | sqrt |
  |---|---|---|---|---|
  | frozen (24) | static, T = 1.6 / 3.2 / 6.4 / 12 | 0.034 / 0.049 / 0.036 / 0.050 | 0.048 / 0.072 / 0.078 / 0.154 | 0.031 / 0.053 / 0.046 / 0.056 |
  | slope layer (16 unique) | dispersive, T = 1.6 / 3.2 / 6.4 / 12 | 0.08 / 0.06 / 0.12 / 0.12 | 2.92 / 0.08 / 0.09 / 0.13 | — |
  | core, midday (216) | dispersive, T = 1.6 / 3.2 / 6.4 / 12 | 0.52 / 0.24 / 0.21 / 0.19 | 2.43 / 0.33 / 0.19 / 0.19 | — |
  | core, low sun (216) | dispersive, T = 1.6 / 3.2 / 6.4 / 12 | 1.64 / 1.46 / 1.37 / 1.64 | 2.37 / 2.18 / 2.10 / 2.64 | — |

  On the real renders 'energy' is no better and usually worse than whitening (on the frozen
  control, on the slope layer at T = 1.6, and in low sun), and 'sqrt' is about level with it on
  the frozen control. Whitening stays the default; the low-sun failure is not an artefact of the
  weighting.

**Checked, no change needed**

- Geometry, signs, units and aggregation (the audit's point 2): back-projection, rectification,
  flow-to-velocity, the UFM transpose and every headline number reproduced. Its explanation of
  the frozen flow residual (model under-reading plus η̄ parallax) is added to Table 6, with the
  parts rechecked here (sign, size by speed, mean sea height) marked apart from those that were
  not (the per-run correlation, the live-sea upper estimate).

**Not fixed**

- Gate-off and foam-off renders, which would settle headline 4, need renderer switches that do
  not exist yet (next step 1).
- The oblique heading-60 failure is still not investigated. The audit's oblique synthetic
  (0.12–0.14 m/s at heading 60) suggests rectification is not the cause.
