# Simulation-native ideas

mare-realtime knows things no real flight can: the sea's own clock, every wave's phase and
frequency, where each water parcel started, the exact sun vector and camera pose. It can
also break the physics on purpose (stop the sea, change g, remove one cue) and fly the same
flight over many oceans. The ideas below use those privileges to measure and remove the
*biases* that set drift, not only to make more frames. They serve the generate ⇄ train ⇄
evaluate loop in [no-gnss-training-plan.md](no-gnss-training-plan.md) and its experiments
[E1–E6](no-gnss-training-plan.md#what-the-simulation-experiments-must-answer). Ranking is
value × confidence ÷ effort, with an idea moved up when it unblocks an experiment.

Numbers are tagged as in the plan: **[SIM]** measured in a simulator, **[REAL]** measured on
a recorded flight, **[DERIVED]** arithmetic or general physics done for this note,
**[TARGET]** a goal. Nothing below has been run unless it says so. Pass criteria are
proposals. Paths are relative to the mare-realtime root. "Novelty" is what our searches
found, not a claim that nobody has tried it.

Budget shares used for pass bars (plan §2, root-sum-square split) [DERIVED]: heading bias
≤ 0.17°, scale bias ≤ 0.29 %, water-motion bias ≤ 0.17 m/s at 60 m/s. Each term alone may
take 0.29°, 0.5 % and 0.30 m/s.

## Ranking

| # | Idea | What it exploits in the sim | Budget term / experiment | Novelty | Effort (first test / full) | First test |
|---|---|---|---|---|---|---|
| 1 | [Doppler-shell measurement layer](#1-doppler-shell-measurement-layer) | Exact dispersion ω = √(gk), exact pose and height | water motion, scale / E1, E3, E6 | adjacent prior art (new here): the shell fit is known, but always takes pose from GNSS/INS | S / M | Shell fit on 200 existing batch windows with height perturbed ±5 % |
| 2 | [Counterfactual twins: a per-cause bias ledger](#2-counterfactual-twins-a-per-cause-bias-ledger) | Sea clock separate from flight clock; byte-identical determinism | water motion / E1; artefact gate for all | adjacent prior art (new here) | S / M | 100 frozen-sea triples over swell, no engine change |
| 3 | [Simulator-only cues: find, fix, randomise](#3-simulator-only-cues-find-fix-randomise) | Every fixed constant and drift in the renderer is known | all three terms / gate before training | known technique, new application | S / S–M | Spectrum-line check, gate-scroll and foam-drift diagnostics |
| 4 | [Gravity knob](#4-gravity-knob) | g in the waves' time evolution only | scale / E3 (E1) | adjacent prior art (new here) | S / M | 20 knob clips through a classical shell fit |
| 5 | [Designed-flight bias bench with common random numbers](#5-designed-flight-bias-bench-with-common-random-numbers) | Seeds, designed patterns, exact truth | all terms, separated / every E, and [Scale](#scale) | known technique, new application | S / M–L | Fake estimator with injected σ, δψ, b on 40 designed runs |
| 6 | [Sun-locked compasses: skewed glint first, Brewster patch second](#6-sun-locked-compasses-skewed-glint-first-brewster-patch-second) | Exact sun vector, glitter model, relit twins | heading / E2 | adjacent prior art (new here) | S / M | Gaussian vs Gram–Charlier glitter integral; Brewster check on 0087 |
| 7 | [AHRS error: identify on real logs, calibrate against the swell](#7-ahrs-error-identify-on-real-logs-calibrate-against-the-swell) | Swell direction as a world-fixed reference; sensor pass downstream of render | heading / E2 | known technique, new application | S / M | Cross-validated heading-error fit across 0087's turns |
| 8 | [Same flight, many oceans](#8-same-flight-many-oceans) | `render.seaSeed` independent of the trajectory | bias in any term / E1, E3 | adjacent prior art (new here) | S / M | 16 seeds × 20 conditions, seed-mean error vs standard error |
| 9 | [Name the water: sheared current and a Doppler convention](#9-name-the-water-sheared-current-and-a-doppler-convention) | Current profile and Doppler term under our control | water motion / E1 label, E4 | known technique, new application | S / M | Analytic Ũ(k) table per preset and altitude |
| 10 | [Spectrum-keyed wave-bias correction](#10-spectrum-keyed-wave-bias-correction) | Live-minus-frozen labels per window | water motion / E1 | adjacent prior art (new here) | M / M | 2,000 60-s runs with twins, tree-model ceiling |
| 11 | [Lagrangian labels and a tracer taxonomy](#11-lagrangian-labels-and-a-tracer-taxonomy) | Undisplaced coordinate x0 per fragment; foam, glint, shadow fields | water motion / E1 | adjacent prior art (new here) | S / M | Dense homography-flow ablation, then per-class bias table |
| 12 | [Real-flight twins accepted on navigation bias](#12-real-flight-twins-accepted-on-navigation-bias) | Recorded poses replayed through the renderer | transfer / E6 | adjacent prior art (new here) | M / M–L | Quad-hover twin vs real height trend of apparent velocity |
| 13 | [Tier-0 linear sea inside the training loop](#13-tier-0-linear-sea-inside-the-training-loop) | The cascades' linear physics, ported without the lattice | water motion, scale / E1, E3 pretraining | known technique, new application | M / M–L | Match engine height and slope at the same seed to 1e-5 |
| 14 | [Wave-field map for long dwell](#14-wave-field-map-for-long-dwell) | Phase-resolved field propagated exactly | water motion, scale for quads / E1, E3 | adjacent prior art (new here) | M / L | Oracle propagate-and-register on float slopes |

## The ideas

### 1. Doppler-shell measurement layer

*Merges: Doppler-shell regression layer, amortised spectral phase inversion,
dispersion-focus self-supervision, shell sharpness as a monitor.*

**What.** The plan's shell fit ([§4](no-gnss-training-plan.md#spectral-wave-models-fft-wavelets))
turned around: GNSS-free attitude, a height prior and the IMU/airspeed-predicted motion go
in; residual velocity relative to the water and metric scale come out, with a covariance,
as an EKF measurement. Frames are orthorectified and warped into a water-fixed tile moving
at the predicted velocity, using the oblique band of the frame (view incidence ≈ 25–65°).
Per bin, the complex phase over time is fitted to ±√(g·k_px/s) − k_px·du_px; both branches
jointly, frame interval as a nuisance state. Stages:

1. Classical, no network (the bar for everything else).
2. Learned per-bin weights, or a network that outputs windowed Fourier coefficients of
   height with confidence, feeding a differentiable weighted least-squares layer that
   solves for ΔX and scale ε.
3. On real flights: train the weights without labels by minimising the cross-validated
   shell residual (fit on half the bins, score on the rest).
4. Shell sharpness under the model's own estimates as a GNSS-free transfer monitor.

**Why it might work.** In the pre-warped frame the Doppler term is scale-free in pixels and
the dispersion term carries s, so velocity and height separate. For a planar sea a height
error is a pure scale of the orthorectification. Phase-difference size [DERIVED]: at
λ ≈ 10 m (ω = 2.5 rad/s), Δt = 2 s, ε = 0.5 % shifts each component's phase by 0.0125 rad.
With ~10³ components at 0.1 rad noise, one window gives ≈ 0.1–0.2 % scale precision.
Coherent phase gives far finer velocity than one frequency bin (λ/T, 0.67 m/s for 2 m waves
over 3 s).

**Sim signal.** Exists: exact pose, K and `heightAboveSea` in `frames.csv`
(`src/ocean/engine/truth.js:48`, `scripts/generate.mjs`); ω = √(9.81k) with no current
(`src/ocean/engine/waves.js:245`). The ω quantisation (2π/1200 s, `waves.js:246`) is
invisible over 2–8 s windows. Must add, for oracle checks only: float height and slope on a
ground grid (new modes next to `src/ocean/engine/post.js:19`, float readback in
`src/batch/page.js`, a `.npy` writer in `scripts/generate.mjs`; M), since the 8-bit height
channel clips at ±2 m (`src/ocean/channels.js`).

**First test (S, no simulator change).** A Python orthorectify, FFT and phase-fit script
on 200 existing batch windows: moderate, fresh and swell-only seas, 120 m, 24–32 m/s (the
random draws stop at 32 m/s, `src/conditions/job.js:267`), windows of 2, 3, 5 and 8 s, true
pose with height perturbed −5 … +5 %.
*Pass:* velocity bias < 0.1 m/s along the well-constrained axis in every sea state; height
within 0.3 % of `heightAboveSea`; predicted covariance within ×1.5 of the empirical
spread. Record the minimum window and the along-crest failure on swell-only seas. Repeat on
float height (ΔX error < 1 %, ε within 0.5 % from a ±3 % wrong prior) to separate the fit
from the renderer's modulation transfer. If raw height already fails, drop stage 2.
*Real stage, on 0087:* per 3–5 s window; fit one constant current per 10-min block, GNSS
offline only; residual RMS < 0.3 m/s with no correlation to heading; dispersion height
varying < 0.5 % over 10 min at constant altitude. Shell sharpness should be measurably
better with the true 0.1001 s frame interval than with 0.1 s [REAL interval,
`extract-flight.py:248-249`]; if not, keep the monitor for velocity only.

**Risks.** Only k·du along the wave directions is observed: long-crested swell leaves the
along-crest component weak, and the covariance must say so. Near nadir under a smooth sky
intensity is quadratic in slope (energy moves to 2k), so weight by view geometry. A bin can
hold opposing waves, so the fit needs two branches. Free per-bin weights collapse onto a few
lucky bins: keep an effective-bin-count floor. Real-sea effects the sim lacks: amplitude
dispersion ω ≈ √(gk)(1 + (ka)²/2) moves apparent g/h by ~1 % at ka = 0.1 [DERIVED], bound
harmonics, and wind-drift shear ([idea 9](#9-name-the-water-sheared-current-and-a-doppler-convention)).

**Closest prior art.** Spencer, Shah & Guha, IEEE TIP 2006, scale from water video
(https://stars.library.ucf.edu/facultybib2000/6610). Dugan et al., JGR 2001, airborne
dispersion fit with GPS/INS pose (https://data-ww3.ifremer.fr/BIB/Dugan_etal_JGR2001.pdf).
Streßer et al., IEEE GRSL 2017 (https://data-ww3.ifremer.fr/BIB/Stresser_etal_IEEEGRSL2017.pdf).
Solodoch et al., IEEE TGRS 2025. Serafino et al., normalised scalar product
(https://adgeo.copernicus.org/articles/19/83/2008/adgeo-19-83-2008.pdf). Spectral-leakage
Doppler bias, arXiv 2311.01265. Differentiable phase correlation, DPCN++ arXiv 2206.05707.
Every one takes platform pose as input; no prior art found in our searches estimates the
platform's own velocity and height from the shell, or trains on a self-supervised shell loss.

### 2. Counterfactual twins: a per-cause bias ledger

*Merges: frozen-sea / frozen-camera twins, twin-sea counterfactual renders, per-cause bias
ledger, cue knock-out environments.*

**What.** Render each condition as the factual run plus k twins at the same instants on the
same GPU, each with one cause removed: sea clock frozen (waves, foam, gate and clouds held),
sea reversed, foam drift 0, cloud drift 0, glitter gain 0, exposure replayed from the
factual run, gate scroll off, cascade c removed. For any estimator,
Δv_cause = v(factual) − v(twin), per frame. Add the exact rigid-geometry flow from pose and
float depth, which splits the wave term three ways: view-dependent (frozen-sea fit minus
truth: reflected sky and clouds sit at infinity and do not translate; glint moves),
wave-plus-foam (factual minus frozen), and the non-additive remainder.

Uses, in order: a bias ledger for every checkpoint; hard gates on simulator artefacts
([idea 3](#3-simulator-only-cues-find-fix-randomise)); a reversal test of whether travel
direction is read from dynamics or from wind cues; labels for
[idea 10](#10-spectrum-keyed-wave-bias-correction); nuisance twins (cloud, exposure,
glitter) as invariance pairs. As a training term, penalise the variance across twins of the
windowed *mean signed error*, not of the risk, and leave out twins that remove the target's
own information.

**Why it might work.** The manual engine is byte-identical on the same GPU, driver and
browser (`README.md`, Determinism), so a twin differs from the factual frame only by the
cause, with no seed noise in the difference.

**Sim signal.** Exists: `setSeaTime` (`src/ocean/engine/engine.js:523`), `seek` (`:534`),
`advance(dt, {flight: false})` holds the flight while the sea runs (`:562`); several modes of
one instant without advancing (`src/batch/page.js:233-239`). Must add (S each): a
`render.counterfactual` block in `src/conditions/run.js` (resolveRender); a separate wave
time at `engine.js:1380` (today `waves.update(this.simTime)`); zero foam drift
(`waves.js:587`); a cloud-drift multiplier (`engine.js:1467`); a glitter gain at
`src/ocean/engine/sky.js:328`; exposure replay (`post.js:289`); a gate-scroll switch
(`engine.js:1383`); a sign on `uTime` for reversal (`waves.js:608`); per-cascade knock-out
where the fades are overwritten each draw (`engine.js` draw). Foam is stateful ping-pong
(`foam.js`), so exact branching mid-run needs a foam snapshot (M). Depth is HalfFloat today
(`engine.js:265`), ≈ 0.1 m steps at 150 m [DERIVED].

**First test (S, no engine change).** Below U10 ≈ 3.6 m/s foam is off (`waves.js:577`), so
use U10 ≤ 3.5 m/s plus swell of 0.5, 1 and 2 m, clear sky, 2 sun elevations, 100 triples
at 120 m, built with `setSeaTime` + `seek` in `src/batch/page.js`; SEA-RAFT and a planar fit
with true height.
*Pass:* (i) factual-pair bias ≥ 0.3 m/s and growing with swell; (ii) frozen-sea bias
< 0.05 m/s, or its excess attributed to reflection and glint by masking them;
(iii) factual − frozen within 10 % of the frozen-camera-implied velocity.
*Second test, after the switches:* 40 runs, overcast or sun behind the camera, 30–150 m.
Frozen, glitter-off twin bias < 0.02 m/s (the pipeline adds no bias of its own);
factual − frozen points along the wind sea and grows with the estimator's long-wave weight
as altitude rises. Gates: gate scroll off changes velocity < 0.02 m/s and heading < 0.05°;
cloud drift off changes velocity < 0.02 m/s. Under reversal, a dynamics-based model's
velocity relative to the water moves < 0.05 m/s.
*Training pass, only after the above:* the twin term lowers held-out-seed bias ≥ 2× against
fine-tuning on the same pairs with exact labels alone. Otherwise twins stay a diagnostic.
Do not score sim factual − frozen against the real 0.74–1.63 m/s apparent water velocity
[REAL]: that includes current. Compare how the real value changes with height and look
angle at fixed sea state, which a current does not change.

**Risks.** A frozen sea is off the data manifold: use it as a probe. Cloud shadows drift
with `simTime` (`engine.js:1467`) and must be held, or they leak in as fake sea motion.
Knocked-out seas are inconsistent (a missing cascade changes the Jacobian, so foam, mss and
glint change): fine for an audit, small weight in training. Twins cancel randomness only on
the same GPU, driver and browser build.

**Closest prior art.** Gaidon et al., Virtual KITTI, CVPR 2016 (https://arxiv.org/abs/1605.06457).
Zhou et al., Learning Rigidity in Dynamic Scenes, ECCV 2018 (https://arxiv.org/pdf/1804.04259v1).
DeFlowSLAM (https://astro.paperswithcode.com/paper/deflowslam-self-supervised-scene-motion).
Intervention Design for Effective Sim2Real Transfer, arXiv 2012.02055. Smeltzer & Ellingsen,
arXiv 2311.01265 (simulated, unpaired Doppler bias). No prior art found in our searches that
stops or reverses the clock of a fluid surface to attribute an odometer's bias, and no
synthetic sea flow or VO dataset was found
(`research_notes/Visual front ends beyond UFM/sim_to_real_and_edge.md`).

### 3. Simulator-only cues: find, fix, randomise

*Merges: alternate-physics probes (all but gravity), physics as the only ruler
(equivariant front end), the artefact half of the tracer idea, the artefact arms of the
twin renders.*

**What.** A list of regularities that exist only in the simulator and that a model can use
as a ruler, a compass or a velocity reference, each with a probe and a fix:

| Cue | Where | Why it matters |
|---|---|---|
| Break-gate scroll 1.7·(1, 0.6) m/s ≈ 1.98 m/s on a fixed 59° bearing, whatever the wind | `engine.js:1383-1387` | sim-only velocity and compass cue |
| Every preset sea travels toward 60° | `src/conditions/presets.js:43` | heading shortcut on preset runs |
| Foam drifts 0.035·U10 downwind over waves with no Doppler shift | `waves.js:584-587` | at 10 m/s, a 0.35 m/s foam-vs-wave offset [DERIVED], above the 0.30 m/s fixed-wing water budget; at sea short waves feel the same surface drift, so the relative motion is much smaller |
| Foam lifetime 2.4 s | `waves.js:41` | real residual foam lasts tens of seconds or more |
| Cloud shadows at 2 + 0.6·U10 m/s along the sea's wind | `engine.js:1467` | plausible speed, but tied to the sea's direction |
| Fixed metric periods: patches 1013 m × 6.71⁻ⁿ, gate 768 m, lace 256, clouds 24,576 | `waves.js:31-32`, `foam.js:31`, `surface.js:23`, `sky.js:18` | hidden rulers for scale |
| Repeat every 1013 m (17 s at 60 m/s) and every 1200 s; ω on a 2π/1200 s lattice | `waves.js:31, 36-37, 246` | fake landmarks; a spectral shortcut |

Fixes: a `render.physics` block in `src/conditions/run.js` with gate scroll
{wind-tied, random, off}, foam drift from the declared surface drift
([idea 9](#9-name-the-water-sheared-current-and-a-doppler-convention)) with randomised
coefficient, foam lifetime 2–60 s, cloud drift direction and speed decoupled from the sea,
per-run ±30 % scale on the fixed periods with the spectrum's physical shape fixed, per-run
cascade offsets and rotations, and randomised preset wind directions. Design consequence for
the front end: feed it gravity-aligned, orthorectified, velocity-pre-warped input, have it
output a pixel-unit residual, and let metric scale and heading enter only through the
height state ([ideas 1](#1-doppler-shell-measurement-layer), [4](#4-gravity-knob)) and the
compass heads ([idea 6](#6-sun-locked-compasses-skewed-glint-first-brewster-patch-second)).

**Why it might work.** Each cue is a known constant in the code, so each probe has a known
correct response (zero), and the fix costs little.

**First test (S, no training).** (a) A classical 3-D FFT dispersion fit, and the M2 baseline
when it exists, on 12 runs of 60 s (moderate and moderate-swell, 120 m, 25 m/s) under
{gate scroll on, off} × {patch ×1, ×1.1}. Take the radially averaged power spectrum of
shaded frames and look for lines at k = 2πn/151 m and 2πn/22.5 m (the second and third
patches) and for the 2π/1200 s lattice in the ω histogram. (b) Runs at U10 = 6, 10 and
14 m/s × foam-drift coefficient {0, 0.035} × gate {on, off}, flow on residual foam only.
*Pass:* the dispersion fit's response to patch scale and gate scroll is < 0.5 %; with
patches and offsets randomised, no cascade line rises > 3 dB above the local spectrum; the
foam-only bias shifts by coefficient × U10 ± 20 %; a component along 59°/239° appears with
the gate on and vanishes with it off. Any learned model that responds > 1 % to patch scale,
or > 0.1 m/s or > 0.1° to gate on/off, is using a simulator-only cue. For a trained front
end: ±30 % on every fixed metric constant moves speed and height < 0.1 % and heading
< 0.05°, and a warp offset of ±2 m/s gives an output slope of −1 ± 0.02.

**Risks.** Changing patch size must scale `uKMin/uKMax` and the CPU sampler built at
`engine.js:242`, or `heightAboveSea` breaks. Offset hops mid-run can leave visible seams.
The real camera (fisheye, rolling shutter, stabilisation, mount yaw) breaks the
front end's symmetries unless modelled (plan §5 item 5).

**Closest prior art.** van Dijk & de Croon, How Do Neural Networks See Depth in Single
Images?, ICCV 2019. Geirhos et al., ICLR 2019 (cue conflict). IntPhys (arXiv 1803.07616).
EqNIO (https://arxiv.org/html/2408.06321v3). No prior art found in our searches that audits
a sea renderer's fixed-size textures and drifts as hidden rulers for a navigation model.

### 4. Gravity knob

*Merges: dispersion-only gravity knob, counterfactual gravity (time-only), cue-conflict
gravity test, the g′ arm of the alternate-physics probes.*

**What.** A uniform g_d used only in the waves' time evolution (`waves.js:245`), with the
spectrum (`G` at `waves.js:163`, `density()` at `:194`) left at 9.81, mirrored in the CPU
sampler (`src/ocean/engine/sampler.js:107`). Every single frame is identical across arms;
only how the sea evolves differs. Plus a mode switch {dispersive, non-dispersive
ω = c0|k|, frozen}.

1. *Probe.* For any scale or velocity head, measure d ln Ĥ / d ln g_d. A dispersion user
   gives −1; a single-frame texture ruler gives 0. Run with airspeed, baro and IMU
   translation ablated (pass/fail), and with all inputs (read as the weight on the
   dispersion cue, since airspeed and manoeuvres also carry scale).
2. *Training.* With h and the metric spectrum fixed, ω = √(g_d k) gives exactly the video
   of a geometrically similar sea at h_eff = h·9.81/g_d: lengths ×s with s = 9.81/g_d, times
   and slopes unchanged [DERIVED]. Label h_eff and v_eff = v·s; exclude IMU and pitot (they
   carry the true g). Mixed with normal clips, texture stops predicting height.

**Why it might work.** It is the only intervention that shifts the scale label while
leaving every frame's texture untouched, so only the temporal cue can explain it. The
saturation range of the spectrum is scale-invariant, so the equivalent sea is plausible.

**Sim signal.** Must add (S): the uniform at `waves.js:245`, the sampler mirror, the value
in `meta.json`. Height truth comes from the GPU probe (`probe.js`), so labels follow the
knob. ω quantisation (`waves.js:246`): worst case ±0.0026 rad/s per mode, ≤ 0.12 % in g/h
for the short waves a fixed-wing uses, up to 0.83 % for a 10 s swell fit [DERIVED]; add a
de-quantised batch mode (drop `% LOOP_SECONDS` at `:608`, keep phase precision).

**First test.** (a) Classical, 20 knob clips: a 3-D FFT shell fit returns h_eff within
0.3 % with quantisation off; run it on and off to size the quantisation term; velocity
relative to the water unchanged within 0.05 m/s. (b) Any E3 height head on 30 held-out runs
at s ∈ {0.9, 0.95, 1.0, 1.05, 1.1}, moderate to strong seas, 120 m.
*Pass:* elasticity −1 ± 0.15, linear in log s (R² > 0.9), foam coverage within 5 % across s.
|elasticity| < 0.3 with the other sensors ablated means a shortcut (texture or the JONSWAP
Hs–Tp prior): widen spectral-shape randomisation and retest before using any real flight
for E3. (c) Head A trained on knob clips, head B on normal clips, both tested at g = 9.81
with h varied: A reaches p95 scale error < 0.5 % at Hs ≥ 0.5 m with ≤ 10 s windows.
Non-dispersive control: the expected velocity error is c0·⟨k̂⟩ weighted by the estimator's
response, which equals c0 only for a unidirectional sea; check with narrow swell first.

**Risks.** Real seas have amplitude dispersion (≈ 1 % in apparent g/h at ka = 0.1
[DERIVED]), which a linear sim cannot teach; rerun head A on clips with an ad-hoc (ka)²
frequency correction to size the shift. Foam lifetime and gate scroll stay on the flight
clock, so foam turnover changes relative to the waves at s ≠ 1. Fixed metric periods are
not scaled ([idea 3](#3-simulator-only-cues-find-fix-randomise)).

**Closest prior art.** Spencer & Shah, IEEE TIP 2006. Gravity as a dynamics-randomisation
parameter in Isaac Gym
(https://docs.isaacsim.omniverse.nvidia.com/2023.1.1/isaac_gym_tutorials/tutorial_gym_domain_randomization.html),
for policy robustness. CoPhy, arXiv 1909.12000. Physics-guided wave peak period from video,
arXiv 2606.13302. No prior art found in our searches that changes g only in a renderer's
dispersion to test or train a navigation model's scale.

### 5. Designed-flight bias bench with common random numbers

*Merges: bias-floor scaling laws with CRN and designed trajectories, regret-seeking
condition search.*

**What.** A frozen sim bench of designed flights: reciprocal straight legs at three ground
speeds (e.g. 25, 40 and 60 m/s), orbits and long legs, on fixed sea seeds with tiling
broken. Per run fit

e(t) = σ·d(t) + δψ·J·d(t) + b_E·t + R(ψ_rel)·b_B·t + ε

where σ is scale, δψ heading, b_E an earth-fixed velocity bias (current-like) and b_B a
body- or look-frame bias that rotates with heading relative to the waves and sun. Varying
ground speed separates per-distance terms (σ, δψ) from per-time terms (b); orbits isolate
b_E. Checkpoints are compared under common random numbers (same seeds, conditions and
sensor noise). A search arm runs CMA-ES or Bayesian optimisation over the continuous
condition parameters (categoricals fixed), maximising
regret = |bias(model)| − |bias(attainable oracle)|, where the oracle is the same front end
plus [idea 10](#10-spectrum-keyed-wave-bias-correction)'s regressor fed true sea parameters,
inside a hindcast climatology of (U10, Hs, Tp, swell). Outputs: a failure atlas for the
per-slice report and a 20–30 % replay share in training (plan §7, targeted generation).

**Why it might work.** The aggregate drift hides which budget term is failing; the bench
reports each term against data and model size ([Scale](#scale)). CRN removes seed noise from
comparisons, so small improvements are measurable.

**Sim signal.** Exists: determinism (`README.md`), per-run seeds (`run.js` seaSeedFor),
patterns (`src/trajectory/patterns.js`), `trajectory.csv`. `run.js:419` accepts any positive
airspeed, but the random draws stop at 32 m/s (`job.js:267`) and the flight model above that
is unchecked. Must add: tiling break (`waves.js:657` offsets; M); raw or JPEG frames out of
`src/batch/page.js` (PNG data URLs today, `:233-246`; M); streamed trajectory samples for
long runs (≈ 42 kB per second of flight in memory, `README.md` Limits).

**First test (S, render-free).** A fake estimator equal to truth plus injected σ = 0.3 %,
δψ = 0.2°, b_E = 0.1 m/s and b_B = 0.1 m/s along-track, on 40 designed 10 km runs
(`--no-frames`).
*Pass:* all four recovered within 10 %; the variance of a checkpoint difference under CRN
≥ 5× lower than with independent seeds. Then 3 data sizes × 2 head sizes on a frozen
backbone. Search arm: 300 evaluations over 10 parameters; the top 10 θ keep ≥ 3× the median
regret on 5 fresh seeds; triage every find as a real failure or a sim artefact (tiling,
gate lattice) before training on it.

**Risks.** Without speed variation the regression is under-identified (an along-track bias
looks like scale, a cross-track bias like heading). A regret against the frozen-sea twin
would only climb toward easy, fast, long-wave seas (dropped). Throughput is unverified (see
[Scale](#scale)).

**Closest prior art.** Sartor & Thompson, Neural Scaling Laws in Robotics, arXiv 2405.14005.
Data Scaling Laws for End-to-End Autonomous Driving, arXiv 2504.04338. Bouthillier et al.,
arXiv 2103.03098. Sharma, When Does Pairing Seeds Reduce Variance?, arXiv 2512.24145. PAIRED,
arXiv 2012.02096. AdvSim, CVPR 2021. INS/DVL calibration with designed manoeuvres, Sensors
13:14583 (2013). No prior art found in our searches that splits VO drift by bias term against
data scale.

### 6. Sun-locked compasses: skewed glint first, Brewster patch second

*Merges: relit-twin sun and glint heads, skewed-glitter compass, sun-and-glitter compass
with sun-azimuth pairs, skewness-aware glint compass, the sea as a polariser, sea-mirror
polarisation compass.*

**What.** A head that predicts the sun's azimuth relative to the camera in a
gravity-levelled frame (roll and pitch from the AHRS), with von Mises uncertainty. World
heading = ephemeris azimuth − predicted, fused intermittently into the EKF's heading-bias
state. Trained with sun-azimuth pairs (relit twins: same sea and trajectory, azimuth turned
about the world vertical) and wind direction randomised against the sun.

**The finding that orders the work.** The sim's glitter uses a Gaussian slope pdf
(`sky.js:289`; the FFT slopes are linear, `waves.js:245-262`), while real slopes are skewed
along the wind. With Cox–Munk coefficients at U = 10 m/s, c03 ≈ −0.29 and σ_u ≈ 0.18 move the
slope-pdf mode ≈ 0.026 rad, about 3° in reflected direction downwind [DERIVED]: roughly 10×
the heading budget, and fixed in the earth frame for a flight. The estimator decides how
much of that lands in heading. The mean slope is unchanged, so a linear, unsaturated,
geometry-weighted first moment in slope space is in principle immune; a peak, a threshold or
a saturated-blob centroid (what an 8-bit auto-exposed camera gives) carries the full shift.
Tilt also couples in: an AHRS roll or pitch error δ gives ≈ δ/sin(glint off-nadir angle) of
azimuth error, 0.3–2° for 0.3–1° of tilt [DERIVED], so tilt must be fitted jointly (from
the glitter ellipse).

**Second cue: the Brewster patch.** Near 53° incidence Rp ≈ 0 and Rs ≈ 0.078 [DERIVED],
so reflected skylight is modulated by (1 + P·cos 2χ), with P ≈ 0.5–0.7 at 90° from the sun
in a clear sky. That is a sun-locked darkening an ordinary camera sees, symmetric about the
solar meridian, strongest at low sun (where the glint is weakest). It needs view incidence
of ≈ 45–65° across a useful range of azimuths.

**Sim signal.** Exists: sun azimuth and elevation per run (`engine.js` resolveLight),
`R_wc` per frame (`truth.js:48`), the sky LUT rebuilt each draw, glitter with full slope
covariance and Smith terms (`sky.js:277-295`), per-band and tail mss (`waves.js` bandStats,
tailSlopeVariance). Must add: Gram–Charlier c21, c03, c40, c22, c04 (functions of U10,
rotated to the wind) in the glitter pdf (`sky.js:287-289`; S); a saturating 8-bit camera
response (plan §5 item 5; M); `engine.setLight` so relit twins need not re-enter the scene,
which rebuilds the flight (`engine.js:506-514`; S–M); wind aloft decoupled from the sea's
U10 (`src/trajectory/wind.js:109-120`; S). Only if the real check passes: Rayleigh
polarisation in the sky (`atmosphere.js`, the sky LUT) and Rs/Rp in place of Schlick
(`sky.js:292` and meanFresnel; M).

**First tests (S each, no rendering).** (a) Integrate Gaussian and Gram–Charlier glitter
× Fresnel × shadowing × geometry over the real camera's field of view with saturation, for
U10 3–15 m/s, sun elevation 10–70°, wind-to-sun 0–180°. Report the heading bias of the peak,
the clipped centroid, and a geometry-weighted first-moment fit that solves for tilt.
*Go/no-go:* if the peak or clipped centroid shifts > 0.17° over > 25 % of that space,
skewness and clipping go into the renderer before any E2 training. Also check the rendered
mss against Cox–Munk σ² = 0.003 + 0.00512·U within 10 %. (b) On 0087, GNSS used offline
only for the sun's relative azimuth: mask glint and foam, remove a radial vignetting model,
bin red-channel water brightness at incidence 45–65° by view azimuth relative to the sun.
*Go/no-go:* a two-lobed dark modulation near ±90° from the sun, amplitude ≥ 10 %; if absent,
drop the polarisation cue.
*Training pass, on skewed sim with skewness and peakedness randomised:* |mean bias| < 0.17°
with true attitude and < 0.29° with a 0.3° AHRS tilt error; |slope| of the 100-frame mean
error against wind-to-sun angle < 0.05°/rad. Real: on reciprocal straight legs of 0087, the
mean of (compass heading − GNSS course) < 0.3°, evaluation only.

**Risks.** Skewness varies with slicks (removed) and nonlinearly with wind. 0.3° is ≈ 2 px at
720 px over ~90° [DERIVED], so intrinsics and distortion enter directly. Water-leaving light
is partly polarised (Mobley 2015). Haze and overcast push P toward 0: the variance must
widen, not the bias.

**Closest prior art.** Sun-BCNN, Peretroukhin, Clement & Kelly, ICRA 2017
(https://arxiv.org/pdf/1609.05993), ~12° median sun error. Cox & Munk, JOSA 1954
(https://data-ww3.ifremer.fr/BIB/Cox_Munk_JOSA1954.pdf). Munk, Annu. Rev. Mar. Sci. 2009
(https://data-ww3.ifremer.fr/BIB/Munk_ARMS2009.pdf). Bréon & Henriot, JGR 2006
(https://data-ww3.ifremer.fr/BIB/Breon_Henriot_JGR2006.pdf). Takács et al., Brewster's dark
patch, Applied Optics 2017
(https://arago.elte.hu/sites/default/files/BrewsterDarkPatchOnWater_AO.pdf). Mobley, Applied
Optics 2015. Relighting with light-direction augmentation, arXiv 2012.06444. In-house: 0087's
`solar_heading_correction_rad` column (plan §2). No prior art found in our searches for a
glint-based aircraft heading compass, or for a downward camera using the sea's Brewster
reflection as the analyser.

### 7. AHRS error: identify on real logs, calibrate against the swell

*Merges: sensor system identification from real logs, swell and water-relative-wind compass
swing, the frozen-front-end nuisance head and label-budget curve.*

**What.** Three parts.

1. *Identify.* On 0087, with the GNSS-free camera orientation stream and GNSS ground
   velocity (offline): wind per 3–5 min segment, airspeed per segment with a turn term, and
   δψ = a + b·t + Σ_{n≤2}(c_n sin nψ + d_n cos nψ), terms kept only if cross-validation
   chooses them. IMU residual = real IMU − `src/trajectory/imu.js` on the smoothed post-flight
   trajectory, trusted below ≈ 1 Hz; vibration from straight-leg spectra; an Allan-matched
   residual bank. These set the AHRS and IMU randomisation of plan §5 items 1 and 8.
2. *Calibrate in flight.* Swell direction, and W′ = v_vision − v_air, are world-fixed. Their
   apparent direction, rotated by the AHRS heading, should not change with heading; what does
   change measures the 1ψ and 2ψ harmonics and gyro drift between turns. The constant term is
   left to [idea 6](#6-sun-locked-compasses-skewed-glint-first-brewster-patch-second).
3. *Label budget.* In sim with random mount yaw, focal scale, clock rate, camera-to-IMU
   offset: how many labelled minutes and which manoeuvres pin each term. Mount yaw is shared
   across flights and AHRS bias is per flight, since one flight cannot separate them.

**Why it might work.** The sim can apply any sensor error after the render, so one set of
frames serves many error draws, with truth known. Precision [DERIVED]: N independent spectral
cells give σθ/√N; a 10 km leg with a 300 m swath and λ = 150 m gives N ≈ 130, ≈ 1.3° per leg
at σθ = 15°, and 2 harmonics over 20 legs ≈ 0.4–0.6°. Sub-0.3° is not assured.

**Sim signal.** Exists: `sea.swellDirection` (`waves.js` setSea), true heading and airspeed
in `trajectory.csv`, patterns. `imu.js` is noise-free (`imu.js:3`). Must add: a seeded sensor
pass (`src/trajectory/ahrs.js` or a new `src/sensors/`) with bias, 1ψ/2ψ harmonics of 1–5°,
drift and lag (S–M); a pitot pass (S); mount, intrinsics and timing perturbations after the
render (S–M); position-dependent swell direction (per-segment setSea or a second swell
cascade; M); runs past 1200 s.

**First test (S).** Fit part 1 on half of 0087's turns and predict the other half.
*Pass:* held-out residual RMS < 1° and the chosen model beats a constant offset by ≥ 30 %;
Allan deviation of sim plus bank within 10 % of real for τ = 0.01–100 s. Sim swing:
lawnmower, racetrack and random patterns at 120 m over moderate swell and cross-swell,
normals at 2 fps, AHRS pass applied offline: residual heading-dependent error < 0.3° RMS for
swell/wind-sea energy ≥ 1 and legs ≥ 5 km. Label budget: mount yaw < 0.1° and scale < 0.1 %
from ≤ 10 labelled minutes including ≥ 2 turns.

**Risks.** It is not known whether the camera's orientation stream uses a magnetometer.
0087 has no airspeed sensor, and sideslip in turns (1–3°) is the size of the effect.
Several swell systems, refraction by eddies and a dominant wind sea all degrade the
reference.

**Closest prior art.** Opromolla et al., Sensors 2020
(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7014542/). CMU wind-estimation flight tests
(https://www.ri.cmu.edu/publications/evaluation-different-wind-estimation-methods-flight-tests-fixed-wing-uav).
Online camera-IMU calibration and degenerate motions (https://udel.edu/~pgeneva/downloads/papers/r05.pdf).
Shipborne camera wave direction with IMU correction, MAE 1.14° (Chinese Journal of Ship
Research, doi 10.19693/j.issn.1673-3185.04950). No prior art found in our searches for an
in-flight AHRS harmonic calibration against swell.

### 8. Same flight, many oceans

**What.** Batches of K sea seeds × an identical trajectory, light and sea spec. Loss =
multi-horizon displacement + λ_b·(‖mean_K e_w‖² − tr(Σ̂_w)/K) + NLL under a covariance head
whose target is the ensemble spread + λ_h·HSIC(e_w, nuisance). The nuisance vector holds the
sea spec, sun azimuth relative to course, U10, height, and the phase of the textures that do
not change with the seed (gate, clouds, lace).

**Why it might work.** Per-sample L2 is already bias² + variance in expectation; this
up-weights bias and estimates it inside one batch, where long-window training only sees it
slowly. The unbiased form (minus tr(Σ̂)/K) keeps it from punishing variance.

**Sim signal.** Exists: `render.seaSeed` is its own field (`run.js:462`); only the `seed`
key ties trajectory and sea together (`job.js:171-173`), so sweeping `render.seaSeed` with
`trajectory.seed` fixed gives the same flight over different seas. Gap: random runs are
appended, not crossed with a sweep (`job.js:182-197`): a small job-writer script (S).

**First test.** 16 seeds × 20 conditions × 30 s with the existing generator
(96,000 frames at 10 fps, ≈ 70 min at 23 frames/s [DERIVED]); baseline flow plus planar fit;
per-condition seed-mean error with its standard error.
*Pass (premise):* at ≥ 5 of 20 conditions the seed-mean error exceeds 3 standard errors and
correlates with a condition variable. Then fine-tune with the split loss against plain
multi-horizon L2 on the same frames. *Pass:* held-out conditional bias (new seeds and
conditions) drops ≥ 2× at equal per-frame noise, and filter NEES lies in [0.8, 1.2].

**Risks.** It removes only biases that vary with what the sim varies; sim-to-real gaps are
untouched. Fixed textures show up as "conditional bias", which is useful as an artefact
detector but not physics.

**Closest prior art.** Bahng et al., Learning De-biased Representations with Biased
Representations, ICML 2020 (https://arxiv.org/pdf/1910.02806). HSIC for fair latents, arXiv
2403.02011. arXiv 2512.24145 on paired seeds. No prior art found in our searches on
same-condition environment ensembles in an odometry loss.

### 9. Name the water: sheared current and a Doppler convention

*Merges: depth-sheared current with a velocity reference, k-resolved Doppler head, shear and
wind-driven current, current-front tracking, the tracer-vs-dispersion half of the tracer
idea.*

**What.** A per-run current U(z) = U_deep + U_w·e^{z/δ}, entering the waves as
ω = √(gk) + k·Ũ(k) with the closed form Ũ = U_deep + U_w·2kδ/(2kδ + 1), and moving foam at
the surface drift from the same profile. Whether Stokes drift enters the Doppler shift is
unsettled in the literature (Pizzo et al. 2023 say yes), so the convention is a per-run
switch and is recorded. Navigation output: velocity relative to a *named* reference layer
(the longest well-resolved band, Ũ(k_long)), plus the observable shear
Ũ(k_short) − Ũ(k_long) and the residual-foam-minus-dispersion difference as auxiliary
outputs (the latter feeds the E5 wind state). U_deep stays out of every image target: it is
the E4 current state with a prior. An optional E4 extension tracks only sharp current fronts
(|Δc| ≥ 0.2 m/s within 200 m) by step detection on d(v_vision)/dt − a_IMU.

**Why it might work.** Size [DERIVED]: surface drift 0.03·U10, e-folding 1 m, U10 = 10 m/s
gives U_eff = 0.26 m/s at λ = 2 m and 0.12 m/s at λ = 20 m, so a single-velocity fit is off
by 0.1–0.25 m/s and moves by ≈ 0.14 m/s as altitude shifts the dominant band. That exceeds
the quad budget and is most of the 0.17 m/s fixed-wing share. Real hovers show apparent
water velocity changing with height: East slope −0.0094 m/s per m, 95 % CI
[−0.0132, −0.0062] [REAL] (`dragonsurf/docs/*-07-23-2026-results.md:151-175`).

**Sim signal.** Today there is no current, and foam drifts 0.035·U10 over Doppler-free waves
(`waves.js:584-587`). Must add (M): the Doppler term at `waves.js:245` before quantisation, or
the de-quantised mode of [idea 4](#4-gravity-knob) (a current breaks the 1200 s loop); the
uniform part as advection of the cascade offsets (`waves.js:657`) and of the foam shift; the
CPU sampler (`sampler.js:75, 107`); a current block in `run.js`; labels in `meta.json` and
`frames.csv`. Fronts, render-free: a new `src/ocean/current.js` (S–M).

**First test (S, analytic).** Per sea preset, compute Ũ(k) over the bands resolved at 90, 120
and 150 m (λ ≈ 0.6–50 m, from ground sample distance and an MTF), for δ = 0.3–2 m, surface
drift 1–4 % of U10, Stokes fraction 0–1, with the spectral tail cutoff stated (surface Stokes
drift weights ω³, so the finest cascade and `tailSlopeVariance`, `waves.js:571`, set it).
*Go:* the gap between foam-, short-wave- and long-wave-weighted estimators > 0.1 m/s, or its
variation with altitude > 0.05 m/s. After the sim change: error against the reference layer
has |slope| vs U10 < 0.005 (< 0.05 m/s at 10 m/s); the shear head recovers
Ũ(k_short) − Ũ(k_long) to ±0.05 m/s at 120 m. Fronts: 30-min `--no-frames` flights over
fronts, eddies and slow gradients; a tracked current removes ≥ 50 % of front-step drift,
worsens uniform-field runs ≤ 5 %, false steps < 1 per 10 min wings-level, and the share of
drift from slow (unobservable) gradients is reported.

**Risks.** The convention choice changes what "velocity relative to the water" means; keep it
switchable. AHRS tilt leaks into the front innovation (0.5° ≈ 0.086 m/s² [DERIVED]). Slow
mesoscale and tidal changes stay invisible.

**Closest prior art.** Smeltzer, Æsøy, Ådnøy & Ellingsen, JGR Oceans 2019
(https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2640726). Weichert, Smeltzer & Ellingsen,
arXiv 2408.08197. Pizzo, Lenain et al., JFM 2023 (https://par.nsf.gov/biblio/10430789).
Ardhuin et al., JGR 2021 (https://data-ww3.ifremer.fr/BIB/Ardhuin_etal_JGR2021.pdf).
Chickadel et al., GRSL 2011, IR PIV
(https://depts.washington.edu/uwefm/publications/Chickadel_etal_GRSL2011.pdf). Costanzi et al.,
OCEANS'16, current as a filter state (https://flore.unifi.it/handle/2158/1061129). The
techniques are known; using them to define and report a navigation velocity reference was not
found.

### 10. Spectrum-keyed wave-bias correction

**What.** For a fixed front end F, label its wave bias per window,
b_F = v_F(factual) − v_F(frozen twin) ([idea 2](#2-counterfactual-twins-a-per-cause-bias-ledger)).
Learn b̂_F from a slow descriptor (the ground-projected 2-D image power spectrum plus the
travel-direction sign from the residual temporal frequency, using only components with
c(k) > 2δV, integrated over 30–120 s), height, look and sun angles, ground speed and heading
relative to the waves. The EKF subtracts b̂_F as a slowly varying bias with covariance: the
optical counterpart of the wave-Doppler correction used in SAR current retrieval.

**Why it might work.** A pattern tracker's apparent velocity ≈ v_rel_water + Σ W(k)·c(k)·k̂,
which depends on the directional spectrum and the geometry, both observable (a uniform
current is not). Spectrum and direction are stationary over kilometres and minutes, so they
can be integrated slowly while the flow supplies the fast velocity. W(k) has no analytic
form for a learned front end, which is why paired labels make it practical.

**Sim signal.** Needs idea 2's wave-time freeze. An analytic proxy exists today (run.json sea
parameters through `sampler.js:62-77` or `src/ocean/spectrum.js`), but W(k) must be measured.

**First test (M).** 2,000 random 60-s runs with frozen twins at 60–150 m and 18–60 m/s
(2.4 M frames at 10 fps: ≈ 29 h at 23 frames/s, ≈ 2.7 h render-bound [DERIVED]; lower fps if
needed). (a) Ceiling: gradient-boosted trees from true sea parameters and geometry to b_F;
(b) from image descriptors with true height; (c) the same with ±2 % height error.
*Pass:* held-out-seed residual RMS < 0.1, < 0.2 and < 0.25 m/s for (a), (b), (c), reported
separately for wind sea, swell-dominated and crossing seas, against the uncorrected bias
measured on the same runs.

**Risks.** Metric k ties it to E3 (circular): run with true height first. The sim's slopes
are symmetric, so the single-frame 180° ambiguity must be broken in time. Transfer: the sim
is linear, has no bound harmonics and no short-wave drift, so treat b̂ as a sim prior and
recalibrate on real reciprocal legs.

**Closest prior art.** Mouche et al., IEEE TGRS 2012, the CDOP model
(https://data-ww3.ifremer.fr/BIB/Mouche_etal_TGRS2012.pdf). Collard et al., SEASAR 2008
(https://data-ww3.ifremer.fr/BIB/Collard_etal_SEASAR2008.pdf). SEASTAR wave-Doppler GMFs
(https://seastar-project.readthedocs.io/en/latest/gmfs.html). arXiv 2311.01265. No optical or
odometry version, and none learned from paired sim labels, was found in our searches.

### 11. Lagrangian labels and a tracer taxonomy

*Merges: tracer taxonomy with exact Lagrangian labels, Lagrangian water-parcel flow labels,
material-point correspondence labels, the tracer branch of the foam idea, the off-shell
energy budget.*

**What.** Step 0, no new export: the dense flow of the mean sea plane (a homography from pose
and K) as the per-pixel target of a multi-frame flow head, with an auxiliary target of
teacher flow minus homography, i.e. where and by how much wave motion biases texture flow.
Step 1: export per pixel the undisplaced coordinate x0, float range, the foam material label
(x0 minus the accumulated foam shift), breaking, glint fraction and cloud-shadow factor.
Classes: wave pattern, residual foam, active breaking, glint, cloud-shadow edge. Heads:
segmentation; per-class exact motion; a correspondence loss on shared x0, masked to pixels
where the parcel is actually marked (foam or short-wave energy above a threshold). Ego-velocity
is pooled from material-class pixels with per-class corrections, excluding breaking and
shadow. Material-flow labels are averaged over 1–2 s, since the orbital term is zero-mean.

**Why it might work.** The sim is Lagrangian by construction: each vertex is drawn at
p + D(p, t) and carries `vEN = p` (`src/ocean/engine/surface.js:100, 131`), an exact parcel
label no real dataset has. Image velocimetry fails in wind because waves are tracked as flow
(`research_notes/.../over_water_motion_and_currents.md`); each biased class here has known
physics (shadows at 2 + 0.6·U10 m/s, breaking at crest speed, foam at the foam drift).

**Sim signal.** Internal, not exported: `vEN`, breaking and residual foam (`surface.js:322-323`),
cloud shadow (`sky.js:219`, used at `surface.js:317`), the glitter share (`sky.js:328`). Must
add (M): a fourth float32 render target in `surface.js` (x0 relative to the camera, glint
fraction, class bits); `shadeWater` returning the glitter share; a float truth target (today
HalfFloat, `engine.js:265`, which resolves x0 at hundreds of metres only to 0.06–0.5 m
[DERIVED]); float readback in `src/batch/page.js`; a writer in `scripts/generate.mjs`. Label
runs need foam drift 0, or foam and waves drifted together
([idea 9](#9-name-the-water-sheared-current-and-a-doppler-convention)).

**First test.** Step 0 at a fixed budget of 50 runs: per-frame velocity labels only vs
the same plus dense homography flow. *Pass:* held-out bias drops ≥ 30 %, or matches at half
the data. Step 1: 100 pairs each at U10 = 5, 8, 11 m/s, with and without broken cloud, 120 m;
SEA-RAFT's mean bias per class against material flow. *Pass (sim physics):* shadows ≈ 2 + 0.6·U10,
breaking ≈ crest speed, residual foam ≈ 0.035·U10 relative to waves; the footprint mean of
(material flow − SEA-RAFT flow) equals idea 2's factual − frozen within 0.05 m/s. Then
class-weighted pooling < 0.3 m/s where uniform pooling exceeds 0.7 m/s, and an 8-frame model
< 0.15 m/s and ≥ 2× better than a 2-frame one. Slick and gust-patch layers (each with its own
velocity) are built only if real quad hovers and 0087 windows show off-shell k-ω energy
> 3× the sim's at matched sea state, correlated with tracker bias (r > 0.5).

**Risks.** "Material" is unbiased only in the sim (closed orbits, no Stokes drift). A
correspondence loss where nothing marks the parcel trains on noise. The foam class is
sim-specific until idea 9 lands.

**Closest prior art.** Vrećica, Paletta & Lenain, whitecap segmentation, CCAI@ICML 2021
(https://www.climatechange.ai/papers/icml2021/68). Synthetic river flow videos, WRR 2022
(https://researchportal.ip-paris.fr/en/publications/synthetic-river-flow-videos-for-evaluating-image-based-velocimetr/).
MCFormer, arXiv 2507.04750. Chen et al., Learning by Cheating, CoRL 2019
(https://proceedings.mlr.press/v100/chen20a.html). Senet et al. dispersion filtering
(https://data-ww3.ifremer.fr/BIB/Senet_etal.pdf). No prior art found in our searches that uses
a simulator's per-pixel Lagrangian identity to supervise correspondence on a water surface.

### 12. Real-flight twins accepted on navigation bias

*Merges: pose-replay spectral twin, transfer-gap factor sweep, simulation-based inference,
phase-locked twin, learned camera from HDR radiance, generator tuning against real error.*

**What.** A twin per real flight, quads first. Replay the post-flight poses through the
renderer; fit about 8 sea and light parameters per clip by Bayesian optimisation on the
log-spectral distance of the shell-integrated directional spectrum plus the glint profile
(U10 seeded from Cox–Munk mss); run the same tracker on real and twin frames along the same
poses, and accept the twin on its navigation bias, not its looks. Later stages:

- a 2⁴ factorial over candidate improvements (camera pass, tone mapper, foam threshold,
  sub-resolution roughness) on the *signed* along- and across-track bias difference;
- a neural posterior over sea and light parameters per clip, pooled into an empirical prior
  that replaces the uniform ranges in `job.js` (plus a broad mixture component);
- a learned camera split into a small-receptive-field content operator, checked against sim
  flow truth, and an explicit image-fixed nuisance layer (vignetting, fixed-pattern noise,
  the JPEG grid, the exposure sequence) labelled with zero velocity;
- tuning 3 generator knobs against real GNSS-scored velocity error with piecewise
  current offsets per 5–10 min segment removed, held out by day;
- for quad hovers and λ ≥ 5 m only, a phase-locked twin: invert cascade-0 amplitudes and
  write them into the seeded noise buffer (`waves.js:429-437`).

**Why it might work.** The plan's gap tracking (§7) compares a model with "the sim's closest
match"; a pose-replayed twin makes that match exact in geometry, leaving only the sea, light
and camera to explain the gap.

**Sim signal.** Exists: a trajectory path that places the aircraft on any
`{duration, stateAt}` source (`src/ocean/scene.js:108-150`); exact K and pose (`truth.js:48`);
analytic per-band mss (`waves.js:503-571`). Missing: the batch page builds only generated
trajectories (`src/batch/page.js:130`), so a `recorded` kind reading a pose CSV
(`src/trajectory/index.js`, `io.js:100` parseCsv) admitted by `run.js`; the onboard rig
clamps fov to 12–100° (`src/trajectory/rigs.js:393-395`); a per-run time offset against the
1200 s loop; renderer constants (`uToneMapper`, `JACOBIAN_WEIGHTS`, `BREAK_SIGMAS`, `uDamp`)
exposed through RunSpec. M overall; the camera pass (plan §5 item 5) is shared.

**First test.** The 07-22/07-23 quad hovers at 12.6–110 m. *Pass:* the twin's East height
slope of apparent velocity lies inside the real 95 % CI [−0.0132, −0.0062] m/s per m [REAL],
its North CI includes zero, and its absolute apparent velocity, after a fitted per-site
constant is removed, is within 0.15 m/s per clip. Fixed-wing (0087): wavenumber spectrum and
glint fit only (≈ 4 s dwell gives ≈ 1.6 rad/s frequency resolution [DERIVED]), reporting
|bias_real − bias_twin| per 60 s. Factorial: the factor ranking is stable across 2 seeds and
across frozen vs retrained models, and one factor cuts the bias gap ≥ 40 %, else the gap is
outside the toggles. Phase lock, sim to sim first: height correlation > 0.8 on λ ≥ 5 m and
≥ 6 dB PSNR gain over a random-phase twin.

**Risks.** Removing a per-site constant cancels only a depth-uniform current (shear leaves
≈ 0.05–0.1 m/s [DERIVED]). With 2–3 real training flights the generator tuning overfits
unless regularised toward physical defaults. Pose error limits phase lock to long waves
(0.1° at 120 m ≈ 0.2 m on the sea).

**Closest prior art.** Solodoch et al., IEEE TGRS 2025
(https://cris.huji.ac.il/en/publications/retrieval-of-surface-waves-spectrum-from-uav-nadir-video/).
CopterCurrents (https://ms.hereon.de/coptercurrents/index.php.en). BayesSim, arXiv 1906.01728.
Learning to Simulate, arXiv 1810.02513. AutoFlow, arXiv 2104.14544. Sensor Transfer, arXiv
1809.06256. Enhancing Photorealism Enhancement, arXiv 2105.04619. Desmars et al., OMAE 2018
(https://personal.egr.uri.edu/grilli//Desmars_OMAE2018_78367.pdf). No prior art found in our
searches that fits a real-time ocean renderer per flight and accepts it on a navigation
estimator's bias.

### 13. Tier-0 linear sea inside the training loop

**What.** A PyTorch or JAX generator running inside the data loader. Per sample it draws a
spectrum, swell, wind, trajectory, Ũ(k), sun, camera and optionally g′; synthesises slopes
over the footprint from several FFT patches of incommensurate size and random rotation with
continuous ω (no lattice, no loop); maps them to radiance with a linearised sky and Fresnel
model; and emits labels for velocity relative to the declared water reference, height,
directional spectrum, shear and sun azimuth. Pretrain the dispersion, Doppler and scale heads
here, then fine-tune on mare-realtime renders.

**Why it might work.** It is the same linear physics as the cascades, so it can be checked
against the engine to float precision; it has none of the tiling or frequency lattice, no PNG
I/O, and does not wait on the unverified Linux WebGL path (plan §5).

**Sim signal.** To port: `src/ocean/spectrum.js`, the h0 construction (`waves.js:204-232`,
`util.js` PRNG), `meanFresnel` and `sunGlitter` (`sky.js:264, 277-295`).

**First test.** Gate: tier 0 matches the engine's height and slope at the same seed to
< 1e-5 relative. Then the same small head (velocity relative to the water + height) trained
on tier 0 only vs mare-realtime only, at equal GPU-hours, evaluated on held-out mare-realtime
renders. *Pass:* tier 0's velocity bias within 1.5× of the render-trained head's, or tier 0
plus a 1-hour fine-tune beats it; and tier 0's head shows the −9.1 % height response to
g′ = 1.1 g ([idea 4](#4-gravity-knob)), so it learned dispersion and not a tier-0 artefact.

**Risks.** No occlusion at grazing angles, no nonlinear crests, foam or saturated glint:
restrict it to the physics heads and near-nadir to moderate obliquity.

**Closest prior art.** Rezvov, Krinitskiy et al., Frontiers in Marine Science 2024
(doi 10.3389/fmars.2024.1363135), Fourier-synthesised radar sea images for pretraining.
AutoFlow, arXiv 2104.14544. Mobley 2015 (FFT sea surfaces for polarised ray tracing). Known
technique; no in-loop linear sea with dispersion, shear and current labels for navigation was
found.

### 14. Wave-field map for long dwell

*Merges: the three wave-field SLAM proposals.*

**What.** A water-frame map of complex wave amplitudes on a k grid, propagated exactly by
e^{−iω(k)dt}; each frame's slope field (i·k·h at visible points, with per-pixel uncertainty)
is registered against it by a Gauss–Newton step on velocity relative to the water, height, a
small yaw rate and, jointly, pitch and roll bias ("sea bundle adjustment"). Scoped to quads and
hovering platforms.

**Why it might work.** Coherent phase tracking resolves velocity to ≈ δφ/(k·T): ≈ 0.008 m/s
per mode at k = 3 rad/m, T = 4 s, 0.1 rad phase noise, and frequency precision improves as
T^−1.5 [DERIVED]. A quad at 12 m/s and 120 m keeps a ≈ 200 m footprint in view ≈ 15 s, about
10× better than a 3 s window, and the quad's water budget (0.06–0.12 m/s) needs that. A
fixed-wing at 60 m/s keeps a patch only 3–5 s, so the map adds nothing over
[idea 1](#1-doppler-shell-measurement-layer) there.

**Sim signal.** Exists: cascade displacement and slope textures (`waves.js` evolve), the
fixed-point inversion in `probe.js` (only 4 points, `probe.js:22`), the normals mode. Must
add: a float slope and height export over the footprint (shared with ideas 1 and 11), an N×N
raster probe (S–M), the tiling break, and a per-bin phase random walk with amplitude
dispersion so the sim does not overstate predictability (it is exactly linear and repeats
every 1200 s).

**First test (oracle, M).** Float sim slopes, a 15 s window at 120 m and 8–12 m/s, oblique
camera, wide and narrow directional spread; initialise from the first 5 s, predict 5–15 s under
velocity and height offsets. *Pass:* the minimum lies within 0.05 m/s (well-constrained axis)
and 0.3 % in height; accuracy improves roughly as T^−1.5 against the 3 s windowed fit; and it
beats idea 1 by ≥ 2× in velocity bias, or the idea is dropped. Then add phase random walks with
coherence times of 30–300 s and report the horizon at which registration fails.

**Risks.** Registering a far-range observation with a biased pose amplifies pitch bias (0.1°
moves a patch ≈ 5.8 m at 11° depression from 120 m, a 1 % velocity error over 600 m
[DERIVED]): pitch bias must be a state. Slope from radiance is ill-posed under uniform
overcast and in glint. Real wind sea decorrelates in tens of seconds.

**Closest prior art.** Desmars et al., OMAE 2018 (above). EnKF-HOS phase-resolved
reconstruction (arXiv 2504.13943). Naaijen et al., predictable zones, International
Shipbuilding Progress 2014 (https://content.iospress.com/articles/international-shipbuilding-progress/isp113).
McCann & Bell, IEEE Access 2018 (https://data-ww3.ifremer.fr/BIB/McCann_Bell_IEEEACESS2018.pdf).
No use of a propagated wave field as an odometry map was found in our searches.

## Scale

The loop should measure scaling **per budget term**, not as one drift number: a term whose
floor sits above its share of the budget will not be fixed by more data.

What to measure:

- **Data volume D.** On the [idea 5](#5-designed-flight-bias-bench-with-common-random-numbers)
  bench, fit each term (σ, δψ, b_E, b_B) as A·D^−α + floor, with fresh samples only, at 3 or
  more data sizes. Report the floor and its confidence interval.
- **Diversity vs volume.** At a fixed frame count, vary the number of distinct seeds and
  conditions against frames per condition. [Idea 8](#8-same-flight-many-oceans) gives the
  within-condition spread to compare against.
- **Temporal window T.** Bias against window length (plan E1 asks how much history is
  needed); the shell fit's minimum window ([idea 1](#1-doppler-shell-measurement-layer)) is
  the reference curve.
- **Model size N.** 2 head sizes on a frozen backbone first, then backbone size, at fixed D.
- **Fidelity vs diversity at equal compute.** Camera pass, glitter skewness, current or
  higher resolution against more conditions at lower fidelity, scored on the real bench.
- **Search.** Targeted sampling (regret arm of idea 5) against uniform sampling at equal
  frames; keep the uniform share fixed so other slices do not regress.
- **Transfer.** The same curves on `real-heldout`. A sim curve that keeps falling while the
  real one is flat points at the simulator ([idea 12](#12-real-flight-twins-accepted-on-navigation-bias)).
- **Comparisons under CRN.** Every checkpoint pair scored on identical seeds, conditions and
  sensor draws.

Throughput and cost, from `README.md` (Apple M5, 1280×720, the same pixel count as
720×1280):

| Quantity | Value | Tag |
|---|---|---|
| Render | ≈ 4 ms per frame | [SIM] |
| PNG encode | ≈ 30 ms per frame; ≈ 23 frames/s per page; `--concurrency 2` roughly doubles | [SIM] |
| PNG size | ≈ 2 MB per frame (2.1–2.3 MB measured, moderate sea) | [SIM] |
| One page for a day, PNG | ≈ 2.0 M frames ≈ 55 h of flight at 10 fps ≈ 11,900 km at 60 m/s, ≈ 75× the length of 0087 (158.4 km [REAL]) | [DERIVED] |
| Render-bound ceiling | 250 frames/s ≈ 21.6 M frames/day ≈ 130,000 km/day at 60 m/s | [DERIVED], unverified |
| Raw readback at that rate | 720×1280 RGBA8 = 3.7 MB per frame → ≈ 0.9 GB/s through the browser | [DERIVED]; beyond today's data-URL path, so expect in-browser downscale or JPEG and a lower real rate |
| Storage | 1 M PNG frames ≈ 2 TB | [DERIVED] (plan §5) |
| Twins | k twins multiply render cost by (1 + k), on the same GPU | [DERIVED] |
| Idea 8 first test | 96,000 frames ≈ 70 min at 23 frames/s | [DERIVED] |
| Idea 10 first test | 2.4 M frames ≈ 29 h at 23 frames/s, ≈ 2.7 h render-bound | [DERIVED] |
| Idea 5 search arm | 300 evaluations × 3 seeds × 300 frames = 270,000 frames ≈ 3.3 h at 23 frames/s, ≈ 18 min render-bound | [DERIVED] |

Measure the achieved frames per second of a raw or JPEG stream before planning around the
render-bound figures, and run the Linux GPU smoke test (plan §5) before counting on cloud
capacity.

## Simulator additions the ideas need

A work list for mare-realtime, deduplicated. Ideas in brackets.

1. **Separate sea clock and counterfactual switches** (`render.counterfactual`): freeze or
   reverse wave time (`engine.js:1380`, `waves.js:608`), foam drift (`waves.js:587`), cloud
   drift (`engine.js:1467`), glitter gain (`sky.js:328`), exposure replay (`post.js:289`), gate
   scroll (`engine.js:1383`), per-cascade knock-out (fade uniforms in `engine.js` draw); foam
   snapshot for exact branching (`foam.js`). [2, 3, 10, 11]
2. **Dispersion control**: g_d and a mode {dispersive, linear, frozen} at `waves.js:245` only,
   mirrored at `sampler.js:107`, recorded in `meta.json`. [4, 3, 13]
3. **De-quantised, non-looping sea for batch runs** (`waves.js:246, 608`), with phase
   precision kept over long runs. [4, 9, 5, 14]
4. **Sheared current with a Doppler convention**: Ũ(k) in evolve, uniform part advecting the
   cascade offsets (`waves.js:657`) and foam, sampler mirror, `run.js` block, labels. [9, 11, 12]
5. **Artefact fixes**: gate scroll mode, foam drift from the declared surface drift, foam
   lifetime 2–60 s (`waves.js:41`, `foam.js`), cloud drift decoupled from the sea's wind,
   preset wind directions randomised (`presets.js:43`). [3, 6, 9, 11]
6. **Tiling and hidden rulers**: per-run cascade offsets, rotations and patch scale
   (`waves.js:31-32, 657`, `uKMin/uKMax`, the sampler at `engine.js:242`); per-run scale on
   `LACE_PERIOD`, `GATE_PERIOD`, `CLOUD_PERIOD`. [3, 5, 14]
7. **Float32 truth exports**: x0 relative to the camera, range, glint fraction, cloud shadow,
   breaking and class bits as a fourth target in `surface.js`; a float truth target
   (`engine.js:265`); float height and slope on a ground grid (modes beside `post.js:19`);
   float readback in `src/batch/page.js`; a `.npy` or EXR writer in `scripts/generate.mjs`.
   [1, 2, 11, 12, 14]
8. **Glitter realism**: Gram–Charlier skewness and peakedness in `sunGlitter`
   (`sky.js:287-289`) driven by the wind direction; a saturating camera response. [6, 12]
9. **Relighting without re-entering the scene**: `engine.setLight` (`engine.js:506-514`). [6, 2]
10. **Polarised sky and Rs/Rp Fresnel** (`atmosphere.js`, `sky.js:292`, meanFresnel), only if
    the 0087 Brewster check passes. [6]
11. **Recorded-pose trajectory kind** (`src/trajectory/index.js`, `io.js:100`, `run.js`,
    `src/batch/page.js:130`). [12]
12. **Sensor pass downstream of the render**: AHRS (heading harmonics, drift, tilt), pitot, IMU
    residual bank, frame-interval and offset timing, mount and intrinsics perturbation (plan §5
    items 1, 3, 7, 8). [7, 1, 5]
13. **Throughput and envelope**: raw or JPEG frames from `src/batch/page.js:233-246`; streamed
    long trajectories; random airspeed draws above 32 m/s (`job.js:267`) with the flight model
    checked to ≈ 70 m/s. [5, 10, Scale]
14. **Job writer** crossing K sea seeds with N random conditions (`job.js:182-197`). [8, 5]
15. **N×N raster probe** over the footprint (`probe.js:22`). [14, 1]
16. **Position-dependent swell direction** (per-segment `setSea` or a second swell cascade). [7]
17. **Off-shell modulation layers** (slicks, gust patches) with their own velocities, only if
    the off-shell diagnostic justifies them (`surface.js`, `truth.js`, `post.js:19`). [11]

## Dropped, and why

- **Frozen-camera twin as a flow target.** A non-rigid pattern moving at phase speed has no
  exact correspondence; exact material flow (idea 11) replaces it.
- **Changing g everywhere.** The spectrum's peak and amplitudes depend on g
  (`spectrum.js`, `waves.js` jonswap), so a model reading peak wavelength against U10 also
  shifts and nothing can be attributed. Time-only g (idea 4) instead.
- **Time-only g as plain randomisation against texture rulers.** Texture still predicts
  height, so it does not punish them; idea 4's fixed-height knob clips with h_eff labels, and
  idea 3's metric-constant randomisation, do.
- **Equalising risk across knock-outs (V-REx).** Removing an informative cue raises risk
  because information is gone; idea 2 equalises mean signed error instead.
- **Re-offsetting a cascade mid-run as a tiling probe.** The image jump alone moves any flow
  model; paired runs offset from t = 0 instead.
- **Regret against the frozen-sea twin.** No estimator can attain it, so the search would
  climb toward the easiest seas; idea 5 uses an attainable oracle.
- **Matching sim factual − frozen to the real 0.74–1.63 m/s.** The real figure includes
  current; compare its change with height and look angle instead.
- **Absolute wind-driven current, or wind vs current, from shear.** Only dŨ/dk is observable,
  the signal is ≈ 0.1 m/s at fixed-wing dwell, and the drift ratio and deflection priors are as
  uncertain as the answer; idea 9 keeps only the label definition and the shear output.
- **Far-to-nadir registration with fixed poses.** Grazing geometry turns a 0.1° pitch bias
  into a 1 % velocity error; idea 14 tracks continuously and estimates pitch bias.
- **Wave-field map and phase-locked twin for fixed-wing.** A 3–5 s dwell gives nothing beyond
  the windowed shell fit, and pose error rules out phase lock below λ ≈ 5–10 m.
- **Aircraft shadow as a sun cue.** A few per cent contrast at most, blurred by subsurface
  scattering, and the antisolar point is rarely in a forward-tilted view.
- **Shapley attribution of the transfer gap.** Switching a component off is out of
  distribution and |bias difference| is not additive; idea 12 runs a factorial over candidate
  improvements on the signed difference.
- **A "provably motion-free" learned camera.** Image-fixed patterns, exposure flicker and
  clipping all bias measured flow; idea 12 makes them an explicit zero-velocity layer.
- **One constant current per 45-min flight as the real-data offset.** Currents change over
  158 km; use piecewise offsets per segment with heading diversity.
