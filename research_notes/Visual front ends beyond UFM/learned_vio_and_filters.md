# Learned and hybrid VIO, learned inertial odometry, differentiable filters, and fixed-wing GNSS-denied baselines (for a camera + IMU + air-data aircraft over open water)

Research date: 29 October 2026. Budget-limited pass (about 17 tool calls). Every number below is as reported by the cited source; "reported" means the authors' own claim on their own benchmark. Metric types are stated (ATE = absolute trajectory error after alignment; RPE/t_rel = relative error per frame or segment; % = position error as a share of distance travelled). None of the benchmarks below (EuRoC, TUM-VI, KITTI, TartanAir, Blackbird) contains open water, so none of these numbers transfers directly to the target problem.

## 1. Learned VO / VIO / SLAM front ends: reported accuracy, runtime, and behaviour in low texture

### Takeaway
On standard benchmarks the strongest learned front ends (DROID-SLAM, DPVO/DPV-SLAM, MAC-VO) are reported in ATE metres or per-frame relative error, not % of distance. Monocular versions suffer badly from scale drift outdoors (KITTI DPVO ATE 53.6 m average). None reports a result over open water. The 2025 foundation-model SLAM systems (MASt3R-SLAM, VGGT-SLAM) run at about 10–26 FPS on desktop RTX-4090-class GPUs and depend on static 3D structure and loop closure, which open water does not provide.

### Cited Findings
**DPVO / DPV-SLAM (Princeton; DPVO 2022/NeurIPS 2023, DPV-SLAM ECCV 2024)**
- DPVO on the TartanAir monocular test split (ECCV 2020 SLAM challenge): ATE with scale alignment, median of 5 runs, averages 0.105 (default) and 0.129 (fast config). Reported 4x faster than DROID-SLAM on the TartanAir validation split, AUC 0.80 vs 0.71. — [DPVO paper, arXiv 2208.04726](https://arxiv.org/pdf/2208.04726)
- EuRoC average ATE (m): DROID-SLAM 0.022 (20 FPS, 20 GB GPU), DPVO 0.105, DPV-SLAM 0.024 (50 FPS, 5.0 GB), DPV-SLAM++ 0.023 (50 FPS, 7.0 GB), GO-SLAM 0.024 (6.8 FPS, 14 GB). ORB-SLAM3 fails on V203. — [DPV-SLAM, arXiv 2408.01654](https://arxiv.org/html/2408.01654v1)
- KITTI 00–10 average ATE (m), monocular: DPVO 53.61, DPV-SLAM 53.03, DPV-SLAM++ 25.76. DROID-SLAM fails on 02 and 09. KITTI FPS: DPVO 48, DPV-SLAM 39, DROID-SLAM 17. All timings on an RTX 3090. — [DPV-SLAM, arXiv 2408.01654](https://arxiv.org/html/2408.01654v1)
- The authors say "scale drift is a known challenge in this setting", that proximity loop closure fails under significant scale drift, and that DPV-SLAM "like most monocular systems, sometimes suffers from scale-drift in outdoor environments." — [DPV-SLAM, arXiv 2408.01654](https://arxiv.org/html/2408.01654v1)

**MAC-VO (CMU AirLab; arXiv 2409.09479, v2 March 2025; stereo, learned metrics-aware covariance)**
- Reported as per-frame relative error (t_rel m/frame, r_rel °/frame), not ATE. EuRoC: MAC-VO 0.0024 / 0.0403; DROID-SLAM 0.0024 / 0.0590; DPVO* 0.0027 / 0.0437; ORB-SLAM3 0.0092 / 0.1866; MASt3R-SLAM* 0.0277 / 1.0472; TartanVO-stereo 0.0368 / 0.8346 (* = scale aligned to ground truth). — [MAC-VO, arXiv 2409.09479v2](https://arxiv.org/html/2409.09479v2)
- TartanAir v2 Hard: MAC-VO 0.0141 / 0.1429; DROID-SLAM 0.0370 / 0.0861; DPVO* 0.2406 / 1.246; TartanVO* 0.2316 / 3.020. ORB-SLAM3 and MASt3R-SLAM lost track on every Hard sequence. — [MAC-VO, arXiv 2409.09479v2](https://arxiv.org/html/2409.09479v2)
- KITTI: ORB-SLAM3 0.0258 / 0.0434 beats MAC-VO 0.0420 / 0.0645 on translation. DROID-SLAM 0.0900 / 0.0448; DPVO* 0.1951 / 0.0329. Authors attribute weaker KITTI rotation to having no multi-frame optimisation (two-frame only, no BA, no loop closure). — [MAC-VO, arXiv 2409.09479v2](https://arxiv.org/html/2409.09479v2)
- Runtime at 640×640 on a Ryzen 9 5950X + RTX 3090 Ti: 2.15 FPS raw, 3.25 with TensorRT, 3.96 with multiprocessing + TensorRT, and 10.57 FPS for "MAC-VO Fast" (bf16, lighter model). No embedded/Jetson numbers are reported. — [MAC-VO, arXiv 2409.09479v2](https://arxiv.org/html/2409.09479v2)
- Covariance: a FlowFormer-initialised matcher predicts flow plus 2D uncertainty; a CovGRU refines it in log space; trained with an NLL loss on synthetic TartanAir only and evaluated on real data without fine-tuning. The uncertainty is used both to reject keypoints in "feature-less areas" and to weight pose-graph residuals with full (off-diagonal) 3D covariance. — [MAC-VO, arXiv 2409.09479v2](https://arxiv.org/html/2409.09479v2)
- A CMU MSR thesis (2026) extends this to visual-inertial (MAC-VIO) and reports a 33.4% ATE reduction on EuRoC compared with the visual-only MAC-VO. — [Xiang Fei MSR thesis, CMU, 2026](https://publications.ri.cmu.edu/storage/publications/2026/04/MSR_Thesis_Xiang_Fei_20260501094349.pdf) (seen via search snippet only; tables not read)

**Hybrid learned-dense-BA + IMU factor graph: DBA-Fusion (arXiv 2403.13714, 2024)**
- DBA-Fusion tightly couples DROID-style trainable dense bundle adjustment with IMU preintegration (and optionally GNSS) in a factor graph, giving metric-scale VIO. On TUM-VI magistrale/outdoors it reports the largest gains on outdoor sequences, which the authors attribute to deep optical flow coping with low texture. — [DBA-Fusion, arXiv 2403.13714](https://arxiv.org/pdf/2403.13714)
- An independent paper (VIGS-SLAM, arXiv 2512.02293) reports that DBA-Fusion's vision-only keyframe selection can drift badly on EuRoC when the vehicle starts nearly static, because IMU preintegration intervals grow too long. It forced a keyframe at least every 20 frames to fix this. — [VIGS-SLAM, arXiv 2512.02293](https://arxiv.org/pdf/2512.02293)

**Foundation-model dense SLAM (2025–2026)**
- MASt3R-SLAM (CVPR 2025): the authors report about 15 FPS on an i9-12900K + single RTX 4090, and they subsampled every second frame to simulate real time. It needs no known camera model. Its 7-Scenes ATE average is 0.047 m. — [MASt3R-SLAM, arXiv 2412.12392](https://arxiv.org/pdf/2412.12392); [CVPR 2025 open access](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html)
- VGGT-SLAM (NeurIPS 2025) aligns VGGT feed-forward submaps with 15-DoF homographies optimised on the SL(4) manifold, from uncalibrated monocular RGB. — [VGGT-SLAM, arXiv 2505.12549](https://arxiv.org/pdf/2505.12549)
- A third-party FPS table (FoundationSLAM, arXiv 2512.25008) lists DROID-SLAM 24, MASt3R-SLAM 10, VGGT-SLAM 26, FoundationSLAM 18 FPS. On its own EuRoC run it measures MASt3R-SLAM ATE at 0.041. This setup is not the original authors'. — [FoundationSLAM, arXiv 2512.25008](https://arxiv.org/pdf/2512.25008)
- VGGT-SLAM++ (arXiv 2604.06830, 2026) states that VGGT was trained only on RGB, so on grayscale EuRoC it underperforms classical methods and DROID-SLAM. — [VGGT-SLAM++, arXiv 2604.06830](https://arxiv.org/pdf/2604.06830)
- In MAC-VO's tables, MASt3R-SLAM had 10x worse EuRoC t_rel than DROID-SLAM/MAC-VO and lost track on all TartanAir v2 Hard sequences. — [MAC-VO, arXiv 2409.09479v2](https://arxiv.org/html/2409.09479v2)

**Over water specifically**
- No paper on learned or classical UAV VO over open ocean was found. The closest work (monocular VO on a sea-surface vehicle) says that tracking the water plane is "generally not an option" and tracks shore features beyond the water instead. — [Monocular VO for an unmanned sea-surface vehicle, arXiv 1707.04444](https://arxiv.org/html/1707.04444)
- A general survey of GNSS-denied UAV navigation notes that VO accuracy "can diminish in feature-poor or low-light areas." — [GNSS-denied UAV navigation survey, Satellite Navigation 2025](https://link.springer.com/article/10.1186/s43020-025-00162-z)

### Inferences
- The SLAM half of DROID/DPV-SLAM/MASt3R/VGGT-SLAM (loop closure, global BA, 3D map) adds nothing over open water, where nothing is revisited and the "scene" moves. What remains useful is the dense or patch-flow front end and its learned per-pixel confidence. That is the same role UFM plays in the current design, so the architecture question comes down to which front end gives the best-calibrated flow plus uncertainty on water, not whether to adopt a SLAM system.
- Learned covariance from the front end (MAC-VO's approach) is the most transferable idea. MAC-VO was trained only on synthetic data and transferred to real data without fine-tuning, which is similar to the sim-heavy, real-scarce training regime here.
- Runtime: MAC-VO full runs at about 4 FPS and MAC-VO Fast at about 10.6 FPS on an RTX 3090 Ti at 640×640. Foundation SLAM runs at 10–26 FPS on desktop GPUs. On a Jetson at 10 Hz with 720×1280 input, none of these fits without heavy downscaling, distillation, or TensorRT work. DPVO (48–60 FPS on an RTX 3090) has the most headroom.
- Camera motion over water is measured relative to the moving water surface: waves travel at their phase speed, and the surface drifts with current and wind. A flow-based velocity is therefore biased by the surface's own motion. This bias, unlike noise, is not removed by averaging. (Inference from physics. No paper found that quantifies it for UAV VO.)
- Barometric altitude plus attitude gives metric scale for flow over a near-planar sea, which removes the monocular scale-drift failure seen on KITTI. Wave height relative to altitude limits how good this is. (Inference.)

### Gaps
- No % of distance drift numbers were found for DROID-SLAM, DPVO, MAC-VO or the foundation-model SLAM systems. Their papers report ATE in metres or per-frame error. Converting would need trajectory lengths, which were not extracted.
- Not researched in this pass (budget): TartanVO, DytanVO, iSLAM, DeepVIO, VINet, SelfVIO, transformer VIO, DEIO, and results on UZH-FPV, MidAir or Blackbird for the visual methods. No numbers are given for them here.
- Licences were not verified from LICENSE files. From memory and unverified: DROID-SLAM is BSD-3, DPVO is MIT, and MASt3R/DUSt3R are non-commercial (CC BY-NC-SA). Check each repository before use.
- UFM (the current dense-flow front end) was not benchmarked in any source found here, so there is no head-to-head against DPVO or MAC-VO flow.

## 2. Learned inertial odometry, learned IMU models, and air-data dead reckoning

### Takeaway
Learned inertial odometry for drones has matured (Cioffi et al. 2023, AirIO RA-L 2025, AirIMU), with large relative ATE gains over TLIO on Blackbird. All of it is evaluated on short indoor or racing flights or pedestrian/vehicle data, not on fixed-wing aircraft over tens of km. Classical airspeed-aided fixed-wing estimation without wind estimation gives about 3–5% of distance over short flights. Its weakest state is heading (magnetometer yaw RMSE about 5.5°), roughly 20x larger than the 0.29° budget.

### Cited Findings
- **TLIO** (2020): a network predicts IMU displacement and its covariance, and an EKF fuses that as a measurement. The authors report the learned covariances are statistically consistent, and in ablations learned covariance beats hand-tuned constant covariance on drift and ATE. — [TLIO, arXiv 2007.01867](https://arxiv.org/pdf/2007.01867)
- **Learned inertial odometry for autonomous drone racing** (Cioffi et al., arXiv 2210.15287, RA-L 2023): reported to beat TLIO on every Blackbird sequence, with ATE gains of 12% (sequence "egg") to 80% ("winter"). — [Cioffi et al., arXiv 2210.15287](https://arxiv.org/pdf/2210.15287) (gains seen via search summary; table not read)
- **AirIO** (arXiv 2501.15659, accepted to IEEE RA-L June 2025): for drones, keeping the IMU features in the body frame (rather than rotating them to the world frame, as pedestrian IO does) while retaining gravity improved ATE by 73.9% on Blackbird. Explicitly encoding attitude adds about 23.8% more (secondary summary). Evaluated on Blackbird, EuRoC and a Pegasus-simulator dataset. IMU only, no control inputs or extra sensors. — [AirIO, arXiv 2501.15659](https://arxiv.org/html/2501.15659v1); [summary](https://themoonlight.io/review/airio-learning-inertial-odometry-with-enhanced-imu-feature-observability)
- **AirIMU** (arXiv 2310.04874): learns IMU noise and corrections and propagates covariance through a differentiable on-manifold preintegrator. Reports drift rate reduced by a factor of 2.2–4 on its benchmarks (hand-held, vehicle, helicopter), the largest dataset spanning 262 km. — [AirIMU, arXiv 2310.04874](https://arxiv.org/pdf/2310.04874)
- **Airspeed-aided fixed-wing estimation without GNSS** (Ye, Zeng, Zeng, Zou, Sensors 2022): pitot TAS + IMU + magnetometer + baro on a 6.9 kg fixed-wing, from one flight of about 200 s processed offline against an RTK GNSS/INS reference. Wind and sideslip are ignored (maximum wind about 3 m/s). Horizontal position error under 30 m after 30 s (about 600 m flown, ≈5%) and under 50 m after 90 s (about 1,800 m, ≈2.8%). Velocity within 2 m/s. Yaw MAE/RMSE 4.79°/5.54°, limited by magnetometer disturbance. Authors call the result "still not very precise." — [Ye et al., Sensors 22(9):3156, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9099684/)
- A non-peer-reviewed simulation project on airspeed-aided EKF reports one failure where the EKF "learned a fake wind," causing 2.7 km divergence and a 50° attitude error. A +6% airspeed scale error behaves like unknown wind. — [gnss-denied-nav GitHub project](https://github.com/koksalan/gnss-denied-nav) (hobby project; treat as anecdotal)
- A vendor (UAV Navigation) lists autopilot dead reckoning without visual aid at about 4% of distance travelled. Marketing figure, unverified. — [UAV Navigation, GNSS-denied navigation page](https://www.uavnavigation.com/gnss-denied-navigation)

### Inferences
- At the 0.29° heading budget, the heading error alone gives a cross-track error of sin(0.29°) ≈ 0.51% of distance, which uses the whole 0.5% budget. A magnetometer-based heading at about 5° RMSE (Ye et al.) would contribute about 9%. Heading, not velocity, is likely the dominant term. No learned-IO paper found addresses absolute heading. (Arithmetic inference.)
- Airspeed gives velocity relative to the air mass. Camera flow over water gives velocity relative to the moving surface. Neither is ground velocity. Wind and surface current are both slowly varying biases, and they are only partly separable when the aircraft turns. A filter that estimates wind (and possibly surface drift) as states is needed whatever the front end. (Inference, consistent with the wind-ignoring limitation in Ye et al. and the fake-wind failure in the hobby project.)
- AirIO's main finding (body-frame representation plus explicit attitude input) carries over directly, because attitude is an available input here. A learned IO head conditioned on attitude and airspeed is a plausible fallback when the camera degrades. (Inference.)

### Gaps
- No published learned inertial or air-data odometry result for fixed-wing aircraft over tens of km was found, and no "synthetic air data" or wind-triangle learned estimator paper was retrieved in this pass.
- RoNIN and Tartan IMU (a 2025 IMU foundation model) were not researched here.
- Long-horizon IMU-only drift on Blackbird is short-flight data. Extrapolating to tens of km is unsupported.

## 3. Differentiable / learned filters: does training through the filter reduce bias and drift?

### Takeaway
The best-supported evidence (Kloss, Martius, Bohg 2021) is that end-to-end training through a differentiable filter helps mainly by learning the noise models: heteroscedastic observation noise "significantly" improves results. Fine-tuning the sensor and process models end-to-end can hurt. The recommended recipe is to pretrain components separately, then fine-tune the noise models through a differentiable EKF. KalmanNet-style learned gains help under model mismatch but have shown limits on some odometry tasks. No paper found shows end-to-end filter training removing a persistent bias (as opposed to noise) over long horizons.

### Cited Findings
- Kloss, Martius & Bohg, "How to Train Your Differentiable Filter" (Autonomous Robots 2021): built differentiable EKF/UKF/MC-UKF/PF. Fine-tuning only the noise models improved RMSE and NLL for most filters, while also fine-tuning the sensor and process models sometimes hurt. Best results came from pretraining individually and then fine-tuning end-to-end. Long enough training sequences matter, heteroscedastic observation noise "significantly" improves results, and the differentiable EKF is recommended as the starting point for its simplicity. KITTI-10 was used in the evaluation. — [arXiv 2012.14313](https://arxiv.org/pdf/2012.14313); [MPI page](https://am.is.mpg.de/publications/kloss_auro_2020)
- KalmanNet (Revach et al., arXiv 2107.10043): an RNN learns the Kalman gain, which avoids needing explicit noise statistics. It beats the model-based EKF under model mismatch. On NCLT localisation it reports 22.2 dB MSE vs 25.385 dB for the EKF (vanilla RNN 40.21). — [KalmanNet, arXiv 2107.10043](https://arxiv.org/pdf/2107.10043)
- A 2026 legged-odometry paper found that applying the standard KalmanNet training pipeline "yields limited training and testing performance" on quadruped state estimation, and attributes this to implicit recurrent adaptation. — [IMM proprioceptive odometry, arXiv 2603.29383](https://arxiv.org/pdf/2603.29383)
- TLIO's ablation: learned per-measurement covariance beats constant hand-tuned covariance inside the EKF. — [TLIO, arXiv 2007.01867](https://arxiv.org/pdf/2007.01867)
- AirIMU: a differentiable preintegrator with learned noise and covariance reduces drift rate by 2.2–4x. — [AirIMU, arXiv 2310.04874](https://arxiv.org/pdf/2310.04874)
- MAC-VO: learned metrics-aware covariance (NLL-trained) weights the residuals of the downstream optimiser and is the main source of its gains over DPVO/TartanVO on TartanAir v2 Hard. — [MAC-VO, arXiv 2409.09479v2](https://arxiv.org/html/2409.09479v2)

### Inferences
- The evidence favours a modular design: a learned front end that outputs a measurement (velocity or flow-derived displacement) plus a calibrated heteroscedastic covariance, fused by an EKF/ESKF whose noise models are fine-tuned end-to-end on trajectory loss. A KalmanNet-style black-box gain is less attractive, because it cannot be audited when a sensor drops out.
- Bias suppression comes from making biases states of the filter (wind, surface drift, gyro bias, airspeed scale, flow-scale/altitude bias) with correct observability, not from learning. Learned covariances help with weighting, but none of the cited work shows them removing systematic bias. (Inference.)

### Gaps
- Haarnoja et al. "Backprop KF" (2016), DynaNet and other end-to-end filters were not read in this pass.
- No paper was found that directly measures drift reduction (% of distance) from end-to-end filter training on a long aerial trajectory.

## 4. Classical fixed-wing GNSS-denied results for comparison

### Takeaway
Published fixed-wing camera+IMU or airspeed+IMU dead reckoning is in the 2.5–5% of distance range over a few km. The target of under 0.5% over tens of km is about 5–10x better than any open fixed-wing result found, and none of those flights was over water.

### Cited Findings
- Ellingson, Brink & McLain, "Relative navigation of fixed-wing aircraft in GPS-denied environments," NAVIGATION 67(2), 2020: tightly coupled EKF monocular VIO plus a pose-graph back end on a 1.5 m-span hobby fixed-wing, processed on a Jetson TX2. Flight test of about 6 km over 388 s: 5.3% of distance for the whole flight, and about 2.5% for the 150–350 s segment after convergence. Simulation (60 s, about 620 m): under 1% for the trajectory with the most turns. Straight-and-level flight showed "significant bias errors" because forward velocity is poorly observable with monocular VIO; the flight had a scale bias up to about 1.2x, mostly from initialisation. No airspeed or barometer was used. — [NAVIGATION 67(2):255](https://navi.ion.org/content/67/2/255)
- BYU press coverage of the same line of work describes an 1,800 m flight tracked with under 3% of distance error. — [BYU News](https://news.byu.edu/news/byu-creating-new-breed-drones-can-navigate-without-gps)
- Ye et al. 2022 (airspeed + IMU + magnetometer + baro, no camera): about 2.8–5% of distance over 600–1,800 m (details in section 2). — [Sensors 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9099684/)
- Vendor claim: about 4% of distance for dead reckoning without vision. Unverified marketing. — [UAV Navigation](https://www.uavnavigation.com/gnss-denied-navigation)

### Inferences
- Ellingson et al. found that straight-and-level flight is the worst case for monocular VIO. A long-range transit over water is mostly straight and level, so absolute scale must come from barometric altitude (and airspeed), not from the visual-inertial excitation that VIO relies on. (Inference.)
- The literature figures come from camera+IMU or airspeed+IMU alone. A system that fuses all of camera flow, IMU, attitude, airspeed and baro with wind estimation has no published fixed-wing precedent at the required accuracy, so the 0.5% target should be treated as unproven until measured.

### Gaps
- Hosen/Fossen/Johansen (NTNU) vision-aided fixed-wing observers and other defence-oriented results were not retrieved.
- No public fixed-wing result over open water was found.

## 5. Design implications: sensor dropout, long horizons, bias suppression

### Takeaway
On the evidence found, a dense/patch-flow front end that outputs velocity plus learned heteroscedastic covariance, fused in an error-state EKF that carries wind, sensor-bias and scale states, is the best-supported architecture. Its noise models can be fine-tuned end-to-end, and a learned inertial head (AirIO-style, conditioned on attitude and airspeed) can serve as a dropout fallback. This is broadly the UFM-plus-filter design, upgraded with learned covariance and a differentiable filter. No integrated end-to-end learned VIO found has demonstrated long-horizon, low-drift, open-water performance that would justify replacing it.

### Cited Findings
- Modular filters with learned noise models are the configuration that end-to-end training improves most reliably. — [Kloss et al. 2021](https://arxiv.org/pdf/2012.14313)
- Learned measurement covariance improves drift and ATE over constant covariance in an EKF (TLIO), and in an optimiser (MAC-VO). — [TLIO](https://arxiv.org/pdf/2007.01867); [MAC-VO](https://arxiv.org/html/2409.09479v2)
- Tightly coupled learned dense BA + IMU (DBA-Fusion) can drift badly when its keyframe logic interacts poorly with IMU preintegration, which is a failure mode of tight coupling. — [VIGS-SLAM](https://arxiv.org/pdf/2512.02293)
- Monocular learned VO has outdoor scale drift (DPVO KITTI ATE 53.6 m). — [DPV-SLAM](https://arxiv.org/html/2408.01654v1)
- Airspeed-aided navigation that ignores wind is limited to about 3–5% of distance, and an EKF can lock onto a false wind. — [Ye et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9099684/); [hobby project](https://github.com/koksalan/gnss-denied-nav)

### Inferences
- **Dropout:** a filter with explicit per-sensor measurement models degrades gracefully: missing airspeed or baro means skipping an update and inflating covariance. End-to-end networks (VINet/DeepVIO-type) must be trained with dropout augmentation and give no guarantees. Prefer the filter.
- **Long horizons (tens of km):** with no landmarks, error grows with the integral of velocity bias plus heading error. Heading (0.29°) is the budget-limiting term, and nothing in the learned VO literature found addresses absolute heading over water. External heading references (for example a sun or horizon-referenced heading, or a high-grade gyro) are outside this survey but are likely necessary.
- **Bias vs noise:** learned covariances fix the weighting. They do not fix the mean. The remaining biases (surface drift seen by the camera, wind seen by the pitot, flow scale from altitude error, gyro bias) need to be states. Training on simulated water could also teach the front end a systematic bias that a learned covariance will not reveal. Real-data validation against GNSS is the only check.
- **Candidate upgrades over a plain UFM front end:** (a) add an NLL-trained covariance head (MAC-VO style); (b) consider DPVO-style sparse patches for Jetson runtime headroom; (c) differentiable EKF with noise-model fine-tuning; (d) AirIO/AirIMU-style learned inertial correction as a fallback. All of these are inferences from the cited work, not tested on water.

### Gaps
- No source found compares a dense-flow-plus-filter design against an integrated learned VIO on the same aerial, long-horizon dataset.
- Jetson-class runtime for UFM, MAC-VO or DPVO at 720×1280 was not found.
- Defence-oriented public results on GNSS-denied navigation over water were not found in this pass.
