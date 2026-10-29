# GNSS-free absolute heading and metric scale (height above water) for an aircraft over open water

Context: the error budget for <0.5% drift of distance flown is about 0.29° total heading bias and about 0.5% height error (about 0.6 m at 120 m). The sensors are a wide-angle RGB camera, an IMU, magnetometer-based attitude, airspeed and a barometer. Research done 29 October 2026. About 17 search/fetch calls; several primary papers were read only through abstracts or search snippets, and that is flagged where it matters. Results are labelled **[FLIGHT]** for flight-demonstrated, **[GROUND/STATIC]** for outdoor non-flight, **[SIM]** for simulation and **[VENDOR]** for manufacturer claims.

## Heading: how accurately can absolute heading be obtained without GNSS (magnetometer AHRS, sun compass, glint, polarization, star trackers, waves)?

### Takeaway
No GNSS-free heading source in the open literature has shown a bias of 0.29° or better in flight on a small aircraft. Magnetometer AHRS reaches about 1° at best after cooperative calibration and is usually off by several degrees. Skylight-polarization compasses are the best camera-class option: about 0.5° RMSE in one drone test and about 2–3° in dynamic flight elsewhere, with sub-0.2° only when static. Sun-from-RGB, sun glint and wave direction have no flight-demonstrated heading accuracy that I could find. Daytime star trackers can meet the budget easily in principle, but published aircraft results report star detection, not heading error.

### Cited Findings
**Magnetometer AHRS on small UAVs**
- **[FLIGHT]** Vitiello et al., *Sensors* (May 2021), estimated onboard magnetic bias using a second "deputy" UAV (CDGNSS + visual). Calibrated heading error stayed below 1° on two custom quadrotors. Uncalibrated magnetic heading, and the commercial autopilot's own filter, were off by "several degrees" in the same tests. This method still uses GNSS on the cooperating platform, so it calibrates but is not GNSS-free in operation — [DOAJ record](https://doaj.org/article/509ef5070dec46eead5334b2a0deeaf8); [PMC full text](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8196794/)
- Opromolla et al. attribute the dominant onboard magnetic bias on small UAVs to fields generated on board by payloads and motors (hard and soft iron that varies with throttle). They state that heading error for low-cost MAV autopilots "can be up to several degrees" — [Magnetometer Calibration for Small UAVs Using Cooperative Flight Data, Sensors 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7014542/); [IRIS UniNa](https://www.iris.unina.it/handle/11588/761188)
- **[SIM]** Wu et al., *Acta Aeronautica et Astronautica Sinica* (2011), report an in-flight rotation-based calibration with magnetic heading error SD <1.2° under strong hard- and soft-iron disturbance, about 8% of the error of max/min calibration — [HKXB abstract](https://hkxb.buaa.edu.cn/EN/abstract/article/1000-6893/12683)
- **[FLIGHT, informal]** A practitioner blog reports that continuous in-flight self-calibration kept the calibrated magnetometer vector "usually within 5 degrees" of the predicted vector. This is not peer-reviewed — [gallinazo.flightgear.org](https://gallinazo.flightgear.org/uas/continuously-self-calibrating-uav-compass/)
- **[VENDOR]** A UAV autopilot compass (UAV Navigation POLAR-300) lists 0.3° heading accuracy, 1σ, "in standard flight with the full sensor suite". The full suite probably includes GNSS aiding, and the conditions are not specified — [UAV Navigation](https://www.uavnavigation.com/products/ahrs-imu/polar-300-autopilot-compass)
- A 2025 arXiv paper (AMO-HEAD) targets MARG-only heading under magnetic disturbance, which shows the problem is still an open research topic — [arXiv 2510.10979](https://arxiv.org/pdf/2510.10979)

**Skylight polarization compass (bio-inspired)**
- **[FLIGHT]** A drone test on dusty days (IEEE, 2024/2025, "Polarized Compass Orientation Framework Based on Real-Time Discrimination of Heading Information in Harsh Weather") reports heading RMSE of about 0.51°. The authors claim this is >81% lower than typical polarized-light methods — [IEEE Xplore 10747186](https://ieeexplore.ieee.org/document/10747186)
- **[FLIGHT]** From a review of passive polarized vision for autonomous vehicles (*Sensors* 24(11):3312, 2024), as summarised in search snippets because the full text returned HTTP 403:
  - A three-unit polarization-sensor array on a 6 kg hexarotor reached <0.2° static but about 1.9–2.9° in dynamic flight.
  - A single quadrotor-mounted unit reached better than 2° outdoors at 10 Hz.

  [MDPI Sensors 24/11/3312](https://www.mdpi.com/1424-8220/24/11/3312)
- **[FLIGHT]** Chinese Journal of Aeronautics (2021): a polarization-camera compass flown on a UAV, with heading precision limited by sensor and output noise. A multiscale-transform denoiser was proposed, using a fiber-optic gyro reference (0.035° heading accuracy). No absolute number was available in the snippet — [ScienceDirect S1000936121001837](https://sciencedirect.com/science/article/pii/S1000936121001837?via=ihub%3D)
- **[FLIGHT]** Polarization sensor/INS tight integration with UAV dynamics (2025/2026): heading RMSE fell 32.5%, 32.3% and 70.2% compared with a loosely coupled method across three multirotor tests. Absolute values were not in the snippet — [ResearchGate 399687096](https://www.researchgate.net/publication/399687096_An_enhanced_polarization_sensorINS_integrated_navigation_method_with_embedded_UAV_dynamics)
- **[GROUND/STATIC]** A ZYNQ real-time polarization heading system (*Sensors* 25(9):2744, 2025) reports outdoor mean heading error of 0.50° and a maximum of 1.43° — [PMC12074092](https://pmc.ncbi.nlm.nih.gov/articles/PMC12074092/)
- **[GROUND, dynamic]** A bio-inspired polarization sensor (*Sensors* 19(16):3448, 2019) showed ±0.5° deviation from an INS in a dynamic outdoor (vehicle) test — [PMC6721297](https://pmc.ncbi.nlm.nih.gov/articles/PMC6721297)
- Sensor tilt couples directly into polarization-derived heading. A dedicated analysis exists — [arXiv 2007.08987, "Influence of sensor tilts on bio-inspired polarized skylight orientation determination"](https://arxiv.org/pdf/2007.08987)

**Sun compass from RGB images**
- **[GROUND, learned]** Ma et al., "Find your Way by Observing the Sun and Other Semantic Cues" (arXiv 1606.07415, 2016), train a CNN (Sun-CNN) to estimate sun direction from a single image and use it as a heading cue for visual odometry. They report state of the art on KITTI-Sun. This is ground-vehicle work, not UAV — [arXiv 1606.07415](https://arxiv.org/pdf/1606.07415)
- **[UAV, non-camera]** "Absolute Orientation for a UAV using Celestial Objects": a hemispherical array of light-intensity sensors locates the sun by least squares and is compared with GPS and magnetic compass. No accuracy was in the snippet — [ResearchGate 268469242](https://www.researchgate.net/publication/268469242_Absolute_Orientation_for_a_UAV_using_Celestial_Objects)
- I found no paper giving an in-flight heading accuracy for a UAV camera sun compass, learned or classical.

**Sun glint on water as a heading cue**
- Glint location and spread are predictable from the law of reflection, sun and sensor geometry, and sea-surface slope statistics (Cox–Munk). Stronger wind spreads the glint farther from the specular centre — [Kay et al., Remote Sensing 1(4):697, 2009](https://www.mdpi.com/2072-4292/1/4/697)
- The Cox–Munk model was the most accurate of the models tested for MODIS glint — [ResearchGate 255788698](https://www.researchgate.net/publication/255788698_Sun_glint_estimation_in_marine_satellite_images_a_comparison_of_results_from_calculation_and_radiative_transfer_modeling)
- Ship-borne polarimetric radiometers track the sun and the vessel's heading to hold a fixed relative sun azimuth and avoid glint. This uses heading-sun geometry, but the other way round — [Characterization of sun and sky glint, SPIE 2015](https://ui.adsabs.harvard.edu/abs/2015SPIE.9613E..0WF/abstract)
- I found no work that uses glint as a heading measurement for an aircraft.

**Star trackers / daytime star trackers**
- **[FLIGHT, detection only]** In a daytime flight test at 16.5 kft, the HRL/Boeing low-SWaP SWIR star tracker detected stars brighter than magnitude 7. Near sea level it detected magnitude <5.5 in cloudless conditions (ground test). It supplied attitude and position references for Boeing's 2024–25 quantum-navigation flight tests. No heading error was published in what I could access — [SPIE 13469, 1346919 (2025)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13469/1346919/Low-cost-low-SWaP-day-night-capable-SWIR-star-tracker/10.1117/12.3057799.short); [Boeing Innovation Quarterly, March 2025](https://www.boeing.com/innovation/innovation-quarterly/2025/03/beyond-gps-quantum-navigation-flight-test)
- **[VENDOR]** Sodern's ASTRADIA daytime star tracker gives day and night attitude when hybridised with an INS. Sodern claims "conclusive in-flight testing" but publishes no accuracy figures — [Sodern news](https://sodern.com/en/news/astradia-a-daytime-star-tracker-diurne-for-a-reliable-and-robust-navigation-system); [Sodern daytime trackers](https://sodern.com/en/viseurs-diurnes/)
- **[FLIGHT, balloon]** DayStar (University of Colorado), a daytime star tracker for high-altitude balloons, reported arcsecond-class attitude accuracy (nighttime pitch and yaw at about the 0.2–0.5 arcsec level as summarised in the snippet). This is a balloon at stratospheric altitude, not an aircraft in the boundary layer — [ResearchGate 261248246](https://www.researchgate.net/publication/261248246_DayStar_Modeling_and_test_results_of_a_balloon-borne_daytime_star_tracker)

**Wave direction**
- No sources found; see Gaps.

### Inferences
- **Size of the budget.** 0.29° is about 5 mrad. Magnetometer AHRS (best flight evidence about 1°, typical several degrees) misses it by 3–10×. That holds even before soft-iron effects from throttle-dependent motor currents, which can change during a flight.
- **Polarization is the strongest camera-class candidate, but no published flight result clearly meets 0.29°.**
  - The best flight number found (about 0.51° RMSE) is still about 1.8× the budget.
  - Moving flight degrades results to about 2–3° unless the INS is tightly coupled.
  - Polarization needs a sky-facing sensor and suffers under overcast skies and at high sun elevation. A downward or forward wide-angle RGB camera cannot measure polarization, so this needs added hardware: a division-of-focal-plane polarization camera pointed up.
- **Sun compass with the existing RGB camera.** Heading error is roughly the sun-azimuth measurement error plus tilt-coupled error. To first order, a tilt (attitude) error δθ projects into azimuth error of about δθ·tan(sun elevation). At 45° sun elevation, a 0.2° roll or pitch error alone therefore uses most of the 0.29° budget. Sun azimuth from the ephemeris needs time and only coarse position: a few km of position error changes sun azimuth by hundredths of a degree. The limiting factor is the camera measurement and the attitude, not the ephemeris. This is my geometry, not a sourced result.
- **Sun glint.** On a nadir or forward camera that sees the glint, the glint centroid lies in the sun's vertical plane, so its image azimuth gives sun azimuth relative to the camera. Cox–Munk spreading means the centroid is a broad, wind-dependent blob, and wave-direction asymmetry can bias it. I expect it to be a useful coarse cue (degree level) rather than sub-0.3° without careful modelling. No published accuracy exists.
- **Wave direction** can at most give relative heading stability over minutes. Dominant wave direction is not tied to north and can be multimodal (swell and wind sea). It is not an absolute reference unless a wave forecast supplies the direction, and forecast direction uncertainty is generally many degrees.
- **Daytime SWIR star trackers** are the only passive option whose underlying accuracy (arcsec to arcmin) is far inside the budget. The evidence at low altitude is weak (detection to magnitude 5.5–7, cloud-limited), and they add a SWIR camera and a stabilised sky view.

### Gaps
- No peer-reviewed flight-test heading error for a daytime star tracker on an aircraft at low altitude. The full HRL/Boeing SPIE paper was not accessible.
- I could not read the MDPI polarization review in full (HTTP 403). Polarization figures from it come from search-snippet summaries, and author names and years for the individual UAV experiments are unverified.
- No sources on wave direction as a heading cue, sun glint as a heading measurement, or learned sun-direction estimation flown on a UAV.
- No polarization-compass flight test over open sea was found. Sea-surface reflections and marine haze could affect downward or low-elevation views, but the zenith sky pattern should be unaffected. This is untested in the sources.

## Gyrocompassing / earth-rate sensing: is it feasible at 0.29°, and at what IMU grade and cost?

### Takeaway
Gyrocompassing reaches 0.29° only with gyro bias of about 0.05°/h at mid-latitudes. That is upper tactical to navigation grade (FOG/RLG/HRG), not MEMS. The first MEMS gyrocompass, announced March 2025, claims only "better than 1° × secant latitude" without GNSS. Gyrocompassing also normally needs stationary or velocity-aided alignment, which is hard in GNSS-free flight.

### Cited Findings
- Earth rate is about 15°/h. Its horizontal (north) component is about 10.6°/h at 45° latitude and about 5.13°/h at 70°, and heading becomes undefined at the pole. To first order, heading error = arctan(bias / north Earth rate). Precision of about 1° needs gyro rate uncertainty better than about 0.2°/h — [US patent 6,477,465 / 6,738,714 text](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6477465)
- A patent states that an IMU with 1°/h gyros is "typically capable of approximately 5 degrees initial heading at mid-latitude" — [same patent family](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6738714)
- **[VENDOR]** SBG Systems (31 March 2025) announced the "world's first MEMS-based north seeking/gyrocompassing IMU": heading better than 1° × secant latitude without GNSS, and initialisation from static or dynamic conditions in about 1 min. Their own page says <0.1° with GNSS fusion, while press coverage repeats <0.01°, so the sources conflict. Size about 52×52×36 mm, <150 g, 2 W. Gyro bias not stated. Products were planned for early 2026 — [SBG Systems news](https://www.sbg-systems.com/news/unveiling-worlds-first-mems-based-gyrocompass/); [Unmanned Systems Technology, April 2025](https://www.unmannedsystemstechnology.com/2025/04/sbg-systems-unveils-first-mems-based-north-seeking-imu/)
- A 2025 review ("Towards learning-based gyrocompassing") says traditional gyrocompassing often fails with low-performance gyros because of limited sensitivity to Earth rate. It proposes learned approaches — [ScienceDirect S0952197625028738](https://www.sciencedirect.com/science/article/pii/S0952197625028738)
- A quaternion-based bias-estimation method matches traditional fine-alignment accuracy in about 20 s instead of 10 min. It still needs a high-grade IMU able to sense Earth rate and works only while stationary — [Scientific Reports 2024](https://www.nature.com/articles/s41598-024-66282-9.pdf)
- **[VENDOR]** Grade reference points:
  - MEMS IMU (Advanced Navigation Orientus): about 3°/h bias instability.
  - FOG (Boreas D90): about 0.001°/h.

  [Advanced Navigation IMU article](https://www.advancednavigation.com/tech-articles/inertial-measurement-unit-imu-an-introduction/)
- All-MEMS tactical IMUs reach about 1°/h gyro bias — [NATO STO EN-SET-116 (2011)](https://www.sto.nato.int/publications/STO%20Educational%20Notes/RTO-EN-SET-116-2011/EN-SET-116(2011)-03.pdf)
- **[VENDOR]** Inertial Labs lists 1°/h tactical MEMS IMUs (IMU-P Tactical, IMU-H100). Prices are on request only — [Inertial Labs IMU-P](https://inertiallabs.com/imup.html); [European Security & Defence, May 2025](https://euro-sd.com/2025/05/major-news/43972/inertial-labs-new-mems-imu)
- Cost reference points, both dated:
  - A survey table puts tactical-grade IMUs at about the US$1,000s level against about $10 for consumer grade — [Satellite Navigation (Springer) 2021, table 7](https://satellite-navigation.springeropen.com/articles/10.1186/s43020-021-00041-3/tables/7)
  - A NASA report says a 0.003°/h HRG IMU costs over $1 million each. A JPL goal was 0.01°/h for under $2,500 in volume; that was a target, not achieved — [NTRS 20010041293](https://ntrs.nasa.gov/citations/20010041293)

### Inferences
- **Arithmetic for the budget** (bias-only, mine): tan(0.29°) ≈ 0.00506.
  - At 45° latitude: allowed bias ≈ 10.6°/h × 0.00506 ≈ 0.054°/h.
  - At 60°: about 0.038°/h.
  - At 30°: about 0.066°/h.

  That needs a navigation-grade or high-end tactical FOG (bias about 0.01–0.05°/h). Such units are typically priced in the tens of thousands of US dollars and up. That is an order-of-magnitude judgement; I found no sourced current price.
- **The SBG MEMS claim (1° × sec φ) is about 3.5× too coarse** at the equator and worse at higher latitude.
- **Gyrocompassing in flight is harder than on a ship or the ground.** It needs the specific force or velocity to be known, so that horizontal acceleration is not mistaken for tilt or Earth rate. Without GNSS, velocity must come from airspeed plus wind or from vision, and errors there couple into heading. A practical design would align on the ground or carrier before launch and then hold heading with the gyro. A 0.05°/h gyro drifts about 0.05° per hour, which is compatible with the budget on flights of one to a few hours.

### Gaps
- No published flight test of in-air gyrocompassing on a small fixed-wing UAV without GNSS.
- No current (2026) list prices for navigation-grade FOG IMUs suitable for a small UAV.
- The SBG unit's bias stability and the exact definition of "secant latitude" accuracy are not published on the page fetched.

## Scale: height above the sea surface without GNSS (barometer, radar/lidar altimeter, wave dispersion, learned monocular depth)

### Takeaway
- **Barometer.** It cannot hold 0.6 m (0.5% at 120 m) on its own over a flight. Weather drifts the pressure reference (1 hPa ≈ 8–9 m), and the temperature or density model scales the reading (about 0.35% per K of column-temperature error).
- **Radar altimeter.** This is the established direct sensor: small units claim 2–6 cm precision within 45–50 m, but longer-range units are coarser (0.37 m to 0.9 m or 1%). The one over-sea flight comparison found shows about 1 m spread at 50 m.
- **Wave dispersion as a scale.** It has a clear precedent (Spencer & Shah, IEEE TIP 2006), but no published accuracy at the 0.5% level and no use on an aircraft for navigation.
- **Learned monocular height over open water.** No published work found.

### Cited Findings
**Barometric**
- Moving weather fronts can change pressure by a few mbar (hPa) per hour, which cannot be told apart from altitude change — [US patent 7,428,466, Self-correcting barometer/altimeter](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7428466)
- **[forum, non-authoritative]** In storms the change can reach about 8 hPa/h, about 40 m of false altitude over 30 min near sea level. Rule of thumb: about 1 hPa per 8–10 m near sea level — [MavicPilots forum](https://mavicpilots.com/threads/can-the-altitude-agl-update-dynamically.51909/post-592669)
- NASA (Orion) lists four primary barometric error sources: sensor, A/D conversion, aerodynamic (static-port/position error) and atmosphere-modelling errors — [NTRS 20110023694](https://ntrs.nasa.gov/api/citations/20110023694/downloads/20110023694.pdf)
- On moving drones a MEMS barometer can sense partly total rather than static pressure. The error depends on airspeed and sensor inertia — [Impact of Static Pressure Measurement Accuracy on Barometric Altimeter Errors, Safety & Defense](https://sd-magazine.eu/index.php/sd/article/view/184); [BazTech 2022 paper](https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-43149a1b-80ce-4c3c-895a-6b7fa40a7d49)
- A first-order temperature correction scales altitude by T_actual/288.15 K — [HMC rocketry assignment](https://pages.hmc.edu/spjut/AdvRoc/Assignments/assign4.md.html)

**Radar / altimeters over water**
- **[FLIGHT, over sea]** Alberi et al., "Accuracy of flight altitude measured with low-cost GNSS, radar and barometer sensors", *Sensors* 17(8):1889 (2017):
  - Setup: 7 altimetric sensors and about 3 h of flight, 35–2194 m.
  - Below 70 m, radar and barometric altimeters performed best, with median standard deviation about 1 m at 50 m.
  - GNSS was used only to calibrate the barometer because sea multipath added large noise.
  - The platform type was not confirmed in the fetched abstract.

  [IRIS UniFe](https://iris.unife.it/handle/11392/2374044)
- **[VENDOR]** Short-range UAV radar altimeters:
  - Roke MRA Type 2: 2 cm precision, 0.2 m minimum, operates over water — [datasheet](https://roke.co.uk/media/2x0dy3qc/ds26_2-0_140125-2025-mra-type-2-datasheet.pdf)
  - Aerotenna µLanding Lite: 5 cm, 0.35–45 m, claims "exceptional performance over water" — [datasheet](https://www.sundance.com/wp-content/uploads/docs/AerotennauLandingLiteDatasheet.pdf)
  - Ainstein US-D1: 4 cm (>1 m), 0.5–50 m, marketed for over water — [datasheet](https://store.sundance.com/docs/Ainstein/US-D1-Data-Sheetv2.pdf)
- **[VENDOR]** Longer-range units:
  - Ainstein LR-D1: altitude precision 0.365 m, range to 500 m, "Over Water Missions" listed — [datasheet](https://www.sundance.com/wp-content/uploads/docs/LR-D1-2024-Data-Sheet-1.pdf)
  - TUALCOM radar altimeter: accuracy 90 cm or 1%, whichever is larger — [datasheet, Feb 2026](https://www.unmannedsystemstechnology.com/wp-content/uploads/2026/02/RADAR_ALTIMETER-Datasheet.pdf)
- DTU's UAWOS page (Geolux radar on UAV over rivers, 30–50 m) reports water-surface elevation profiles with centimetre accuracy. It notes water is an excellent microwave reflector, so water returns dominate — [DTU UAWOS radar altimeter](https://uawos.dtu.dk/payloads/radar-altimeter)

**Wave dispersion as a metric scale**
- Spencer, Shah & Guha, "Determining scale and sea state from water video", *IEEE Trans. Image Processing* 15(6):1525–35 (June 2006), doi 10.1109/tip.2006.871102:
  - An uncalibrated camera's space–time FFT, combined with the water-wave dispersion relation, "recover[s] the real-world scale of an ocean sequence". Sea state (peak wavelength and period, wind speed, wave height) also comes from oceanographic spectra.
  - It was demonstrated on synthetic and real sequences against known geometry. The abstract gives no numeric scale accuracy.

  [PubMed 16764277](https://pubmed.ncbi.nlm.nih.gov/16764277/)
- Uncalibrated monoscopic sea-state video (SN Computer Science, 2021): pixel-intensity spectra were scaled to metres with an empirical ocean model. Significant wave height reached max RMSE 0.37 m and MAPE 16% over Hs 0.5–3.6 m against buoys. That is a wave-height result, not camera-scale accuracy — [Springer s42979-021-00727-0](https://link.springer.com/article/10.1007/s42979-021-00727-0)
- Almar et al., "Sea State from Single Optical Images" (*Remote Sensing* 13(4):679, 2021): the wave-property method was applied to drone and satellite imagery, with drone validation against an ADCP in Chile — [MDPI](https://www.mdpi.com/2072-4292/13/4/679)
- UAV-video nearshore bathymetry uses wave celerity and the finite-depth dispersion relation. Celerities were captured consistently, but Hs was overestimated by 31% — [Springer chapter 978-3-032-15477-4_65](https://link.springer.com/chapter/10.1007/978-3-032-15477-4_65)
- Near-surface currents are routinely retrieved from the measured *departure* from the still-water dispersion relation (Doppler shift k·U). In other words, currents bias any dispersion-based scale — [arXiv 1904.11575](https://arxiv.org/pdf/1904.11575)

**Learned monocular height**
- **[land]** Nadir-image altitude regression with ResNet-50 on more than 300,000 drone images (preprint 2025) — [Preprints 202504.0816](https://www.preprints.org/manuscript/202504.0816)
- **[landing]** VIAE-Net fuses monocular vision and inertial data for fixed-wing landing altitude — [PMC8472930](https://pmc.ncbi.nlm.nih.gov/articles/PMC8472930/)
- **[maritime]** Ship-deck-relative monocular pose with a transformer, including vision-in-the-loop and hardware-in-the-loop flight validation. This is relative to a ship, not height over open water — [arXiv 2502.05409](https://arxiv.org/abs/2502.05409); [arXiv 2606.19176](https://arxiv.org/pdf/2606.19176)

### Inferences
- **Barometer arithmetic** (hypsometric, mine):
  - 0.6 m ≈ 0.07 hPa at sea level. A synoptic tendency of 1 hPa/h therefore breaks the budget in about 4 minutes, unless the reference pressure is refreshed.
  - The scale error from mean column temperature is Δh/h ≈ ΔT/T. 0.5% means knowing the 0–120 m mean virtual temperature to about 1.4 K. A 10 K departure from ISA uncorrected gives about 3.5% (about 4 m at 120 m).
  - Static-port (position) error depends on airspeed and angle of attack and must be calibrated per airframe.
  - Tide and storm surge shift the sea surface relative to a land-referenced QNH by up to metres.
  - Conclusion: the barometer is good for short-term *changes* in height but not for absolute height above water at 0.5%.
- **Radar altimeter.** Over waves the return comes from a footprint-weighted surface, so the reading follows the waves. Averaging over a wave period gives mean sea level relative to the aircraft. For 120 m, a unit with ≤0.1–0.3 m specified accuracy plus wave averaging plausibly meets 0.6 m. The 45–50 m short-range units don't reach 120 m. The coarser long-range specs (0.37 m precision; 0.9 m or 1%) are near or over the budget. Treat all as vendor claims.
- **Wave dispersion for scale.** Deep water gives λ = gT²/2π (ω² = gk). With a camera, the wavenumber is measured in pixels per radian and the frequency in Hz, so ω² = g·k_metric gives metres per pixel and hence height. Error terms (mine):
  - Frequency error doubles: δλ/λ = 2·δT/T.
  - A current or aircraft-motion residual U along the wave shifts the frequency by k·U. Example: for λ = 50 m (c ≈ 8.8 m/s), 0.1 m/s of uncompensated current shifts the apparent frequency by about 1%, and the scale by about 2%.
  - Finite depth is negligible in the open ocean when depth > λ/2.
  - Nonlinear (Stokes) and wind-drift corrections are small but not zero.
  - Because the aircraft moves, the 3D (kx, ky, ω) spectrum is Doppler-shifted by ground velocity. Fitting the dispersion shell can in principle solve jointly for scale and the platform velocity relative to the water, as ship-radar current retrieval does.
  - Absolute current relative to the seabed stays unobservable, so it biases velocity (not height) unless a current model is used. Surface currents in the open ocean are often 0.1–1 m/s.
  - I expect a scale accuracy of a few percent from a single short window, not 0.5%, unless many frequencies are fit jointly and the current is estimated simultaneously. This is unverified, and no paper reports it.
- **Learned monocular depth over featureless water.** This is ill-posed without a physical scale cue. A network would have to learn wave-scale statistics, which is effectively the dispersion or spectral-shape cue above, and that varies with sea state.

### Gaps
- No published accuracy for dispersion-derived metric scale or height from a moving aircraft, and no navigation use found. Spencer & Shah's quantitative accuracy is not in the abstract; the full paper is needed.
- No peer-reviewed over-sea test of small radar altimeters against the 0.5% at 120 m requirement. The Alberi et al. result is the only measured over-sea comparison found.
- No sourced statistic for typical sea-level pressure tendency (hPa/h) in the operating area. The figure used above is from a patent's "a few mbar per hour" and a forum post.
- Lidar altimeters over water were not researched (calls ran out). Specular or dropout behaviour over calm water and at nadir is a known concern but unverified here.
- No learned monocular height-above-open-water model found.

## Learned methods that jointly estimate attitude/heading corrections from imagery for UAVs

### Takeaway
I found little. The closest results are a CNN sun-direction compass for cars (Sun-CNN, 2016), a learned visual compass for high-altitude UAV relative orientation, an omnidirectional-camera attitude/heading estimator validated in flight, and a 2025 learned-gyrocompassing review. None reports absolute heading at sub-degree level in flight over water.

### Cited Findings
- Sun-CNN: a single-image sun-direction CNN used as a heading cue in visual odometry. State of the art on KITTI-Sun; ground vehicle — [arXiv 1606.07415](https://arxiv.org/pdf/1606.07415)
- PLVC: a visual compass from point and line features (masks plus neural networks) for UAV high-altitude *relative* orientation — [ResearchGate 359296393](https://www.researchgate.net/publication/359296393_A_Visual_Compass_Based_on_Point_and_Line_Features_for_UAV_High-Altitude_Orientation_Estimation)
- **[FLIGHT]** Omnidirectional (catadioptric) vision for UAV attitude and heading, validated against onboard inertial data in real flights — [Robotics & Autonomous Systems 2010](https://www.sciencedirect.com/science/article/abs/pii/S0921889010000576)
- Learning-based gyrocompassing (2025 review/paper) aims to extract Earth-rate heading from low-grade gyros — [ScienceDirect S0952197625028738](https://www.sciencedirect.com/science/article/pii/S0952197625028738)

### Inferences
- Over open water, a downward or forward RGB camera's only absolute heading cues are the sun (directly or via glint) and sky polarization (needs a polarization sensor). Learned methods would be fitting those same physical cues. A physically modelled sun or glint estimator with explicit tilt coupling is likely more auditable than an end-to-end network at the 0.29° level.
- The horizon gives roll and pitch, which reduces the tan(elevation) coupling in a sun compass. It does not give heading.

### Gaps
- No search budget was left for a focused review of learned horizon or attitude estimation or sky-image sun localisation on UAVs, such as illumination-estimation networks from computer graphics. Treat this section as incomplete.
