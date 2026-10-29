# Over-water ego-motion and the motion of the water itself (wave dispersion, surface currents)

Scope: what is published on (a) visual navigation / odometry over water, (b) retrieving surface current and wave motion from optical image sequences, and what that implies for a learned visual front end on a GNSS-free aircraft (10 Hz wide-angle RGB, 30–400 m altitude, 15–70 m/s ground speed; budget ~0.3 m/s unmodelled water motion at 60 m/s; an in-house KLT tracker measured 0.7–1.6 m/s apparent water velocity from hovering quads at 12–110 m). Research cut-off: 29 October 2026. About 18 searches and fetches; several primary PDFs were blocked (HTTP 403), so some accuracy figures come from abstracts or secondary summaries. Those cases are flagged.

## 1. Visual odometry / optical-flow navigation of aircraft over water

### Takeaway
I found no published aerial VO/VIO system with drift figures for open water. The closest work is on surface and underwater vehicles, plus fixed-wing GNSS-denied VIO over land. All of it names the same problems over water: little texture, false features from ripples and reflections, and no static ground plane.

### Cited Findings
- Visual SLAM work on water describes the surface as low-texture, with too few distinct features and with "dynamic invalid features" from ripples and reflections that degrade pose accuracy. The proposed fix is semantic masking of water plus line features. It was evaluated against ORB-SLAM3 on the USVInland dataset, which is a surface vessel, not an aircraft. — [search summary of JMSE/MDPI 2023 paper](https://www2.mdpi.com/2077-1312/11/3/515)
- Monocular VO on an unmanned sea-surface vehicle: there is no ground plane over water to give reliable disparity, so the method leans on IMU orientation. It reports "reliable odometry estimates for trajectories several hundred meters long". Its features are on shore and structures, not on water. — [arXiv 1707.04444 (2017)](https://arxiv.org/pdf/1707.04444)
- Underwater benchmarks: low-texture scenes break ORB-SLAM3 and SVIn2. TURTLMap used generated waves in a tow tank to mimic disturbance (2024). In a DVL-inertial-visual study, ORB-SLAM3 had the lowest coverage because it "consistently failed" under visually hard conditions (2026). — [TURTLMap arXiv 2408.01569](https://arxiv.org/pdf/2408.01569); [DIVO arXiv 2607.04615](https://arxiv.org/pdf/2607.04615)
- Fixed-wing GNSS-denied navigation (Gallo & Barrientos) fuses an INS with semi-direct VO (SVO-type) to bound horizontal drift. It assumes GNSS at takeoff and was tested over land/simulation, not open water. — [Aerospace 10(3):220, 2023](https://doi.org/10.3390/aerospace10030220); [arXiv 2303.03804](https://arxiv.org/pdf/2303.03804)
- Early block-matching optical-flow + IMU + pressure EKF navigation notes that accuracy falls as altitude varies. — [IEEE 6896732](https://ieeexplore.ieee.org/document/6896732/)
- A 2026 maritime UAV–USV system (AUSLUN) uses VIO for a hovering UAV in coastal GNSS-denied settings, with stationary targets. It does not address water drift. — [arXiv 2606.29875](https://arxiv.org/pdf/2606.29875)
- A 2025 survey of GNSS-denied UAV navigation (Springer, Satellite Navigation) covers sensor fusion and localisation methods. I found no open-water drift numbers in the search summary. — [Satellite Navigation 2025](https://link.springer.com/article/10.1186/s43020-025-00162-z)

### Inferences
- Over open water every visual feature sits on a moving surface. A tracker that "succeeds" (KLT locks on) still measures camera motion relative to whatever pattern it locked onto. That pattern can be a wave crest moving at phase speed, a foam or slick patch moving with the current plus Stokes drift, or a glint moving with the sun-surface-camera geometry. This is a different failure mode from the low-texture loss of track the robotics papers describe, and it is silent.
- The in-house figure of 0.7–1.6 m/s fits wave-phase tracking. Deep-water linear phase speed c = sqrt(g/k) = sqrt(gλ/2π) is ≈0.68 m/s at λ = 0.3 m, ≈1.25 m/s at λ = 1 m and ≈1.58 m/s at λ = 1.6 m. These are textbook values I calculated, not taken from a source. Those wavelengths are the ones resolvable from 12–110 m, so the tracker very plausibly follows short-wave crests rather than advected water.

### Gaps
- I found no paper reporting a VO/VIO drift figure (for example, % of distance) for a fixed-wing or multirotor aircraft over open sea with no shoreline or vessel in view. This is a real gap in the open literature, not just a search miss: three differently phrased searches came back empty.
- I found no published comparison of how traditional VO (KLT, ORB, direct) fails on open-sea video against how learned flow (RAFT-type) fails.

## 2. Surface current retrieval from optical image sequences via the dispersion relation

### Takeaway
The established method maps a short video onto a fixed geographic grid, takes a 3-D FFT (x, y, t) and fits the Doppler-shifted dispersion shell ω = sqrt(gk tanh kd) + k·U. Against ADCPs it reaches about 0.06–0.09 m/s RMS from hovering drones, with ~30 s sequences. It needs camera pose in a fixed earth frame, from a hover plus georeferencing or from GPS/INS on aircraft. The fit returns current relative to that frame. With no external reference it returns only current minus camera velocity.

### Cited Findings
- **Streßer, Carrasco & Horstmann (2017), IEEE GRSL 14(11):2027–2031**, "Video-Based Estimation of Surface Currents Using a Low-Cost Quadcopter":
  - Setup: nadir-looking camera on a small quadcopter. The video is geocoded, and wavelength, period and direction are extracted.
  - Method: the linear dispersion relation is fitted with spectral energy maximisation in the wavenumber–frequency domain to find the current-induced frequency shift.
  - Result: RMSE 0.09 m/s with no bias against a boat-mounted ADCP.
  - [IEEE 8049342](https://ieeexplore.ieee.org/document/8049342); [Hereon CopterCurrents page, DOI 10.1109/LGRS.2017.2749120](https://ms.hereon.de/coptercurrents/index.php.en)
  - Accuracy figures come via the search summary; the full PDF returned 403.
- **CopterCurrents** (Hereon, MATLAB, GPLv3) is the open-source implementation. Its example retrieves a current field from a **30 s** video from an off-the-shelf quadcopter. A 2025 talk reported a tidal-estuary validation with r = 0.95, **RMSE 0.06 m/s, bias −0.01 m/s**. — [Hereon CopterCurrents](https://ms.hereon.de/coptercurrents/index.php.en); [IEEE vTools event, 2025](https://events.vtools.ieee.org/m/493102)
- WISE webinar, 22 January 2026:
  - Streßer presented "Wave-Dispersion-Based Retrieval of Surface Currents From Aerial Video": a 3-D FFT finds the current Doppler shift, followed by energy maximisation. He calls small consumer quadcopters effective, non-intrusive platforms.
  - The abstracts give no error metrics and do not say how platform motion is handled.
  - [WISE webinar page](https://wise.discourse.group/t/wise-webinar-on-wave-spectra-and-currents-from-nadir-imagery-on-22-1-2026-at-4-pm-cet/379)
- **Dugan et al. (2001), JGR and JTECH:**
  - Platform: a small manned aircraft with a turret that stares at a fixed geodetic spot. It was designed to fit a UAV later.
  - Processing: camera position and attitude (GPS/INS) map each frame onto a common earth-fixed surface. The space-time cube is then Fourier-transformed, and the dispersion surface of gravity waves was "well resolved". Both surface current and depth were retrieved from it.
  - [Dugan et al. JGR 2001](https://data-ww3.ifremer.fr/BIB/Dugan_etal_JGR2001.pdf); [Dugan et al. JTECH 2001](https://data-ww3.ifremer.fr/BIB/Dugan_etal_JTECH2001.pdf)
  - The method lineage is noted in [EarthRef 44148](https://earthref.org/ERR/44148).
- **ROCIS** (Remote Ocean Current Imaging System; Areté with Fugro):
  - Operation: space-time processing of airborne wave imagery gives real-time current maps every 1 km along track. Post-processing gives 3 km swaths resolved to 100 m.
  - Gulf of Mexico flights in 2012, 2014 and 2015 were compared with in-situ currents, and performance was analysed against altitude, dwell, wind speed and wave height.
  - **The abstract gives no m/s accuracy figures.** — [Anderson, Zuckerman & Stuart, AGU Ocean Sciences 2016](https://studio.m-anage.com/agu/os16/preliminaryview.cgi/Paper91778.html); [US DOE article](https://www.energy.gov/fe/articles/new-airborne-technology-measures-ocean-surface-currents-offshore-energy-production-and)
- **Solodoch, Toledo, Grigorieva & Lehahn (2025), IEEE TGRS 63:4201914**, "Retrieval of Surface Waves Spectrum From UAV Nadir Video":
  - Basis: optical contrast scales with wave steepness.
  - Method: regularised inversion of image spectra gives the 3-D wavenumber–frequency sea-surface-height spectrum.
  - Result: "good agreement" with buoys and bottom-mounted acoustic sensors across several sea trials, depths and sea states. No numbers appear in the abstract.
  - [HUJI CRIS record](https://cris.huji.ac.il/en/publications/retrieval-of-surface-waves-spectrum-from-uav-nadir-video/); [Haifa CRIS record](https://cris.haifa.ac.il/en/publications/retrieval-of-surface-waves-spectrum-from-uav-nadir-video/)
- A Texas A&M group compared drone wave-dispersion current mapping with particle tracking and optical flow. They report that the alternatives were often disrupted by breaking waves, sun glare and the need for tracers. — [Ocean News, "Researchers Trial Aerial Drones to Measure Currents"](https://oceannews.com/news/science-technology/researchers-trial-aerial-drones-to-measure-currents/) (trade press; secondary)

### Inferences
- **Observability.** In a camera frame moving at horizontal velocity V, each wave component appears at ω' = sqrt(gk tanh kd) + k·(U − V). The dispersion fit therefore cleanly removes the wave phase propagation term, which is the 0.7–1.6 m/s contaminant. What it returns is **U − V**: the camera's velocity relative to the surface water, not over ground. U and V cannot be told apart from water-only imagery at any sequence length. Every published retrieval breaks the degeneracy with an external earth-fixed reference: hover plus georeferencing (Streßer; CopterCurrents) or GPS/INS pose (Dugan; ROCIS). This is my derivation from the dispersion relation as stated in these papers. None of them states it as a navigation limit.
- Consequence for a GNSS-free aircraft: a perfect learned front end can at best deliver **water-relative velocity**, much as an airspeed sensor delivers air-relative velocity. The surface current, plus any wind-driven or Stokes drift of the tracked pattern, becomes an unobservable bias on ground track. It stays unobservable unless a fixed feature (coast, moored object, seabed in shallow clear water) or a prior (current model or forecast) is available. Typical coastal and tidal currents of 0.1–1+ m/s are well above the 0.3 m/s budget (general oceanographic knowledge; CopterCurrents' estuary validation implies currents in that range, but I have no single cited distribution).
- What *is* achievable, and worth designing for, is removing the wave-phase term so that the residual error is the current alone, not current plus phase speed. The literature says that needs a coherent space-time window of tens of seconds (30 s in the CopterCurrents example) on an earth- or water-fixed grid. At 10 Hz that is ~300 frames. From a 60 m/s aircraft the ground footprint moves ~1.8 km in that time, so the window must be built by stabilising onto a moving patch, as ROCIS and Dugan do by staring or by along-track processing, not from a fixed image region.
- Per-frame learned flow (two-frame RAFT-type) sees only Δt = 0.1 s. Over that time a λ = 1 m crest moves ~0.125 m, indistinguishable from advection. Separating phase speed from advection needs k-resolved temporal frequency information, which means long temporal windows. Multi-scale input helps, because phase speed depends on wavelength: different scales moving at different speeds are the signature of propagation, while all scales moving together are the signature of advection or ego-motion. That "same velocity at all scales" test is the physical basis IWaVE uses (section 3).

### Gaps
- I found no published retrieval from a fast fixed-wing platform without GNSS/INS pose. ROCIS performance against altitude and dwell is reported only qualitatively in the abstract I could reach.
- I could not get the full Streßer 2017 PDF (403). Flight altitude, exact sequence length, tile size, and how it handled drone drift and yaw in the hover could not be confirmed from source.
- I found no source quantifying the minimum dwell needed for a given current accuracy, for example σ_U in m/s against seconds of video.

## 3. Image velocimetry on water (LSPIV, STIV, deep optical flow) and wave-induced bias

### Takeaway
River and estuary image velocimetry reaches roughly 0.05–0.1 m/s RMSE against ADCP in good conditions. It degrades badly when wind waves are present (RMSE doubling to 0.44 m/s in high wind), because waves are tracked as if they were flow. Methods that model wave dispersion (IWaVE) stay accurate in wind. Deep learning so far mostly cleans inputs rather than solving the physics.

### Cited Findings
- **EGU25-3849 (2025), Dolcetti et al.:**
  - Setup: tidal channel in the Netherlands; LSPIV, STIV and the wave-dispersion method IWaVE were compared with ADCP.
  - Without wind, all three agree with ADCP. With wind, "LSPIV and STIV fail to obtain the correct surface velocity" while IWaVE stays in good agreement.
  - The authors attribute this to IWaVE separating advective motion from wind waves.
  - Conference abstract only. — [EGU25-3849](https://meetingorganizer.copernicus.org/EGU25/EGU25-3849.html); [IRIS print](https://iris.unive.it/retrieve/17329144-60a0-48cf-a5b3-ffb9b82b919e/EGU25-3849-print.pdf)
- River LSPIV studies: surface gravity waves, especially standing waves, are associated with **underestimation** of surface velocity at low seeding density. Gravity-capillary waves travel in different directions at wavelength-dependent celerity and can be confused with advected patterns. — [RiverFlow v3, White Rose eprint](https://eprints.whiterose.ac.uk/165812/1/RiverFlow_v3.pdf)
- Tidal-stream field LSPIV against drifters: **RMSE 0.44 m/s in high wind (31 km/h) vs 0.22 m/s in low wind (10 km/h)**. — via the same search; source listed as [Plymouth PEARL](https://pearl.plymouth.ac.uk/secam-research/518) (attribution from the search summary, not verified in full text)
- Deep-learning-enhanced LSPIV in tidal estuaries (Water 2026):
  - Additions: SegNet water segmentation and YOLOv8 vessel removal.
  - Against ADCP: **RMSE 0.048–0.11 m/s**, NRMSE 3.5–10.3%, R² > 0.895.
  - Segmentation alone cut RMSE by up to 0.046 m/s.
  - [Water 18:468 (2026)](https://mdpi-res.com/d_attachment/water/water-18-00468/article_deploy/water-18-00468.pdf)
- RivVideoFlow (Water 2024): orthorectification from ground control points, then VideoFlow, a multi-frame optical-flow network that uses temporal cues. No accuracy numbers in the snippet. — [Water 16:2784](https://mdpi-res.com/d_attachment/water/water-16-02784/article_deploy/water-16-02784.pdf)
- OF-PhyNet (IEEE JSTARS, January 2026): optical flow with physics-informed constraints for river velocity. No numbers in the snippet. — [DOAJ record](https://doaj.org/article/dc201a484e234c67bbb70ba429aa4ee5)
- IAHR 2023 compared RivQNet (CNN optical flow) with LSPIV and STIV. Results were not in the snippet. — [IAHR library](https://www.iahr.org/library/info?pid=29954)
- Smartphone LSPIV with DeepLabV3+ water masking raised mean surface velocity by 7% by removing non-water pixels. — [Sustainability 17(18):8181 (2025)](https://ideas.repec.org/a/gam/jsusta/v17y2025i18p8181-d1747108.html)
- Dérian & Almar, "Typhoon" (IEEE TGRS 55(10):5790–5797, 2017): wavelet-based optical flow tracks surf-zone foam to give dense 2-D currents from shore and UAV video. Against an ADV it captured wave-to-wave and low-frequency variations despite poor image quality. It depends on foam tracers (surf zone). — [HAL 01573715](https://hal.archives-ouvertes.fr/hal-01573715); [IRD record](https://www.documentation.ird.fr/hor/fdi:010071324)
- A related wave bias in ADCP measurements from quasi-Lagrangian platforms is bounded in order of magnitude by surface Stokes drift. — [arXiv 2405.00763 (2024)](https://arxiv.org/pdf/2405.00763)

### Inferences
- All of these assume a stationary, georeferenced camera (GCPs or a fixed mount). Their 0.05–0.1 m/s accuracies are the best case for measuring water motion with the camera's own motion known. They are not evidence that a moving camera could match it.
- The doubling of LSPIV error in wind is the same mechanism as the in-house KLT 0.7–1.6 m/s: correlation trackers lock onto wave phase. A learned flow model trained only on rigid-scene data will inherit this bias. One trained on water with proper labels (camera-relative-to-water or ground motion, not "pattern motion") may not.
- Tracer-following methods (Typhoon on foam, LSPIV on seeding) measure the water's Lagrangian surface motion. On open sea, foam and slicks are sparse and also drift with wind (windage), so "water-relative" can itself be ambiguous by a few cm/s to tens of cm/s. That figure is not quantified in a source I found.

### Gaps
- I found no full-text accuracy numbers for RivVideoFlow, OF-PhyNet or RivQNet.
- I found no study of LSPIV, STIV or deep flow from a moving (non-hovering) camera.

## 4. Deep learning on ocean wave imagery (spectra, sea state, height; synthetic training)

### Takeaway
Learned wave-parameter estimation is mature for SAR and X-band radar, including synthetic-data pretraining that measurably helps. For optical drone imagery the state of the art is still physics-based spectral inversion (Solodoch 2025). Learned optical work is mostly stereo reconstruction. I found no learned optical model that estimates current from moving-platform video.

### Cited Findings
- Deep learning inversion of the ocean wave spectrum from SAR: trained on collocated real SAR and WAVEWATCH III output, it improved MSE by >65% over the analytical SAR inversion (Tripathi et al., ICASSP 2024). — [Ifremer Archimer 00885/99670](https://archimer.ifremer.fr/doc/00885/99670)
- Synthetic X-band radar sea-clutter images generated from a Pierson–Moskowitz spectrum were used to pretrain a significant-wave-height network, then fine-tuned on real radar. Synthetic pretraining increased Hs accuracy (Rezvov et al., Frontiers in Marine Science 2024). — [Front. Mar. Sci. 10.3389/fmars.2024.1363135](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1363135/pdf)
- Thermal (LWIR) stereo with a stereo network fine-tuned on synthetic water surfaces with ground-truth disparity gave **mean bias < 2.1%** against wave probes in a flume (arXiv, August 2024). — [arXiv 2408.10670](https://www.arxiv.org/abs/2408.10670)
- PSM-Net stereo matching plus SIFT gives wave height, period and direction. The paper notes WASS automates dense point clouds from stereo (Front. Mar. Sci. 2024). — [Front. Mar. Sci. 10.3389/fmars.2024.1508233](https://frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1508233/full)
- Physics-driven CNN for real-time 3-D sea-wave reconstruction (Pistellato et al., 2021): stereo is accurate but expensive and usually limited to short pre-recorded sequences; the CNN makes the final surface step real-time. — [AGRIS record](https://agris.fao.org/search/en/records/65df80914c5aef494fe31313)
- Solodoch et al. (2025) retrieve the full 3-D (k, ω) height spectrum from UAV nadir RGB video by regularised spectral inversion, not deep learning (see section 2). — [HUJI CRIS](https://cris.huji.ac.il/en/publications/retrieval-of-surface-waves-spectrum-from-uav-nadir-video/)

### Inferences
- Precedent from radar and SAR suggests synthetic sea-surface pretraining followed by real fine-tuning is a workable recipe for wave quantities. The domain gap for optical RGB (sky reflection, glint, foam, whitecaps) is larger than for radar clutter, and no optical result I found quantifies it.
- Because the Solodoch and CopterCurrents pipelines produce (k, ω) spectra and U from real drone video, they could auto-label real data with wave spectrum and current. That only holds where a stationary or GNSS-referenced reference flight exists.

### Gaps
- I found no deep-learning paper estimating wave direction, sea state or current from *monocular optical drone or aircraft* video with quantitative results. A targeted search for synthetic-to-real optical ocean training returned nothing relevant.

## 5. Sun glint and texture-poor water

### Takeaway
Glint suppression is studied mainly for imaging targets (ships, benthos), using polarisers at optimal angles, polarisation-difference imaging and inpainting. I found no work that pairs polarisation filtering with velocimetry or odometry over water. Glare and breaking waves are named as reasons optical-flow and particle methods fail, and as a reason to prefer dispersion methods.

### Cited Findings
- UAV-borne polarisation-difference glint suppression on wavy water: +25.3% region contrast and +78.4% SNR against a 90° polarisation image (Optics and Precision Engineering 31(15):2181, 2023). — [researching.cn](https://researching.cn/articles/OJbf8a6a58bf399a02)
- Polariser plus least-squares polynomial fitting of glint regions; line-by-line fitting worked best (Laser & Optoelectronics Progress 58(24), 2021). — [researching.cn](https://m.researching.cn/articles/OJ67171597bb709b5f)
- Red Sea shore field tests: a linear polariser at the optimal angle reduces glint and improves maritime detection imagery. — [BGU CRIS](https://cris.bgu.ac.il/en/publications/improving-object-imaging-with-sea-glinted-background-using-polari/)
- A US patent describes detecting and repairing sun glint in UAV RGB ocean images with an optical-flow-based method (benthic mapping). Method details were not verified. — [USPTO 12277713](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12277713)
- Optical-flow and particle methods are reported as frequently disrupted by sun glare and breaking waves, in contrast to wave-dispersion mapping. — [Ocean News](https://oceannews.com/news/science-technology/researchers-trial-aerial-drones-to-measure-currents/) (secondary)
- The water surface is low-texture, and ripples and reflections create dynamic false features. — [MDPI JMSE 2023](https://www2.mdpi.com/2077-1312/11/3/515)

### Inferences
- For a nadir or near-nadir wide-angle camera, the specular glint patch sits at a fixed position in the image for a given sun geometry and moves with aircraft attitude, not with the water. A learned model could mask it from known sun and attitude, or learn to discount it. A fixed polariser helps only near Brewster-angle viewing and costs light. Both points are physical reasoning, not from a source.
- Low texture is partly a dispersion *benefit*: spectral methods use the whole wave field, not discrete features. Solodoch's finding that contrast tracks wave steepness means calm, glassy water (low steepness) is the regime where both feature tracking and spectral methods lose signal.

### Gaps
- I found no quantitative study of glint's effect on VO or velocimetry accuracy over water, and no polarisation-plus-velocimetry work.

## 6. Implications: what a learned front end would need, and is current observable from a moving platform?

### Takeaway
From water-only imagery the observable is velocity relative to the surface water. Wave phase propagation can be removed physically, given tens of seconds of coherent, stabilised imagery and multi-scale (k-resolved) analysis. The surface current itself cannot be separated from ego-motion without an earth-fixed reference or a prior. The 0.3 m/s budget is therefore met only where the actual current is below about 0.3 m/s, or where it is supplied externally.

### Cited Findings
- All published optical current retrievals rely on earth-fixed camera pose: a hover with georeferencing ([Streßer et al. 2017](https://ieeexplore.ieee.org/document/8049342); [CopterCurrents](https://ms.hereon.de/coptercurrents/index.php.en)), or GPS/INS-mapped frames from a staring aircraft turret ([Dugan et al. 2001](https://data-ww3.ifremer.fr/BIB/Dugan_etal_JGR2001.pdf)), or GNSS-referenced along-track processing ([ROCIS, AGU 2016](https://studio.m-anage.com/agu/os16/preliminaryview.cgi/Paper91778.html)).
- Achieved accuracy with an earth-fixed reference is 0.06–0.09 m/s RMS ([CopterCurrents 2025 talk](https://events.vtools.ieee.org/m/493102); [Streßer 2017](https://ieeexplore.ieee.org/document/8049342)), using about 30 s of video ([CopterCurrents](https://ms.hereon.de/coptercurrents/index.php.en)).
- Methods that model dispersion stay accurate when wind waves are present; pattern trackers do not ([EGU25-3849](https://meetingorganizer.copernicus.org/EGU25/EGU25-3849.html)).
- Multi-frame (temporal) networks are already used for river velocimetry ([RivVideoFlow, Water 2024](https://mdpi-res.com/d_attachment/water/water-16-02784/article_deploy/water-16-02784.pdf)).

### Inferences
These are design implications drawn from the cited findings plus linear wave theory.

1. **Separate the two problems explicitly.**
   - (a) *Wave-phase rejection*: physically solvable, and the source of the 0.7–1.6 m/s KLT error.
   - (b) *Current*: unobservable from water imagery alone, so it should be treated as a slowly varying state estimated from other sources. Options are a current forecast or model; returns to any earth-fixed feature; and comparing water-relative velocity with an inertial/airspeed/wind solution, though that only helps if wind is independently known, which it usually is not without GNSS.
2. **Temporal window.** Two-frame flow at 10 Hz cannot tell phase speed from advection. The literature uses about 30 s (~300 frames) for 0.06–0.09 m/s. A learned model would need a recurrent or long-context temporal component working on a water-stabilised (or attitude- and height-rectified) grid. The trade-off between window length and accuracy is unpublished, and the team would have to measure it.
3. **Multi-scale input.** Advection and ego-motion move all wavelengths at one velocity, whereas waves move at c(k) = sqrt(g/k) (deep water). A model that sees several spatial scales, or a (k, ω) representation, gets the physical cue for free.
   - Resolvable wavelengths depend on ground sample distance. At 400 m altitude with a wide-angle lens, short waves (< 1 m) may be unresolved, leaving longer, faster waves (λ = 10 m → c ≈ 4 m/s). That makes rejection *more* important at altitude.
4. **Known height and intrinsics** are required to turn image-plane frequencies into physical k. Without metric k the dispersion relation cannot be applied, and the scale ambiguity of monocular VO returns. Altitude from a barometer or radar altimeter, plus calibrated intrinsics and attitude, matter as much as the image.
5. **Footprint advection.** At 15–70 m/s the ground footprint (a few hundred m at 400 m altitude with a wide lens) is crossed in a few to tens of seconds. That is short compared with the 30 s windows used from hovering drones, so the achievable dwell per water patch, and with it the phase-rejection accuracy, falls at high speed and low altitude. ROCIS's along-track processing shows it is workable from a manned aircraft, but its accuracy against dwell is not public in what I found.
6. **Training labels.** Labelling learned flow with "pattern motion" will bake in the wave bias. Labels should be camera motion relative to an earth or water frame. Synthetic oceans with known U, spectrum and depth can supply those labels; real data would need GNSS-referenced flights with independent current truth (ADCP or drifters), as in the validation studies.
7. **Residual drifts that remain even with perfect phase rejection:** surface current (unobservable), Stokes drift of tracers (an order-of-magnitude bound, per [arXiv 2405.00763](https://arxiv.org/pdf/2405.00763)) and windage of foam and slicks. These are in addition to the 0.3 m/s budget unless modelled.

### Gaps
- I found no published demonstration of optical current or water-relative-velocity retrieval from a platform without GNSS/INS, and no published error budget for separating ego-motion from current on a moving aircraft. The observability argument above is my derivation from the dispersion relation, not a cited result.
- Typical open-ocean current magnitudes against the 0.3 m/s budget, and Stokes drift magnitudes for given sea states, are not sourced here and would need an oceanography reference.
