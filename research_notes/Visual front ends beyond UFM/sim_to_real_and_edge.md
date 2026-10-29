# Sim-to-real transfer, self-supervised adaptation and edge deployment for learned optical-flow / VO front ends (GNSS-free aircraft over water)

Research date: 29 October 2026. Setting: front end trained mostly on a real-time rasterised ocean (three.js GPU FFT, JONSWAP spectra) plus scarce real flight video; target 10 Hz on Jetson Orin NX/AGX.

Source-strength legend used below: **[fetched]** = read this session from the primary page/repo; **[repo API]** = licence read from the GitHub licence API / raw LICENSE file this session; **[search summary]** = from a search-result summary, primary not opened; **[not re-verified]** = a well-known paper claim cited to its arXiv page but not re-read this session (the writer should treat as lower confidence or re-check before quoting numbers).

## Synthetic-to-real transfer for flow/VO, and which techniques close the gap

### Takeaway
Every mainstream flow model (RAFT, SEA-RAFT, NeuFlow v2, GMFlow) is pretrained on purely synthetic data (FlyingChairs/FlyingThings or AutoFlow) and still generalises usefully to real imagery (KITTI); the literature's consistent lesson is that variety and the right *training schedule* of synthetic data matter more than photorealism. No public synthetic ocean/water flow or VO dataset was found, and no study of flow/VO trained on rendered sea surfaces transferring to real sea video was found — the team's setting is uncharted.

### Cited Findings
- Mayer, Ilg, Fischer, Hazirbas, Cremers, Dosovitskiy, Brox, "What Makes Good Synthetic Training Data for Learning Disparity and Optical Flow Estimation?" (arXiv Jan 2018, IJCV) argues synthetic data is the practical training source because humans cannot label pixel-accurate flow, studies how dataset properties affect generalisation, and finds benefit in "learning schedules that use different types of data at selected stages" (i.e., curriculum: Chairs → Things). [fetched abstract] — [arXiv 1801.06397](https://arxiv.org/abs/1801.06397)
  - The paper's body-level findings commonly cited (simple, varied data such as Chairs beats more realistic data early; realistic lighting/camera effects help only modestly; mixing in camera artefacts such as blur/noise helps) were **not** re-read this session — see Gaps.
- AutoFlow (Sun et al., Google, submitted 29 Apr 2021, CVPR 2021) renders flow training data with motion, shape and appearance controlled by *learnable hyperparameters* tuned for target-domain performance, and "achieves state-of-the-art accuracy in pre-training both PWC-Net and RAFT" versus hand-designed synthetic sets. [fetched abstract] — [arXiv 2104.14544](https://arxiv.org/abs/2104.14544). Code/data repo is Apache-2.0 [repo API] — [google-research/opticalflow-autoflow](https://github.com/google-research/opticalflow-autoflow)
- NeuFlow v2 (v3 of paper dated 3 Aug 2025) was trained only on FlyingThings for comparability and evaluated on Sintel (synthetic) and KITTI-15 (real); authors claim its refinement module "mitigates generalisation issues" and reports a "huge jump in accuracy on the real-world KITTI dataset" over v1, plus qualitative results on unseen real images. These are author claims. [fetched] — [arXiv 2408.10161v3](https://arxiv.org/html/2408.10161v3)
- KLTNet (Jin, Zou, Yu; arXiv Aug 2026), a learned sparse feature tracker for monocular VIO, is trained only on TartanAir (synthetic) in two stages and evaluated on real datasets (EuRoC, TUM-VI, KITTI-360, UMA-VI, a self-collected low-texture set); the paper does not explicitly analyse the sim-to-real gap. [fetched] — [arXiv 2608.24544](https://arxiv.org/html/2608.24544v1)
- TartanAir/TartanVO: TartanAir is a large photo-realistic synthetic SLAM dataset with heavy diversity (lighting, weather, motion patterns); TartanVO trained only on it generalises to real datasets (KITTI, EuRoC) without fine-tuning, attributed to diversity plus an up-to-scale loss and intrinsics layer. [not re-verified] — [TartanAir arXiv 2003.14338](https://arxiv.org/abs/2003.14338); [TartanVO arXiv 2011.00359](https://arxiv.org/abs/2011.00359). Tooling and TartanVO code are BSD-3-Clause [repo API] — [castacks/TartanVO](https://github.com/castacks/TartanVO)
- Underwater analogue: wflow-TartanVO adds imaging-physics-based weighting (medium transmission map) to the optical flow feeding a pretrained TartanVO and reports considerably reduced ATE on several real underwater datasets *without fine-tuning the VO network*. [search summary] — [arXiv 2407.13159](https://arxiv.org/pdf/2407.13159). This is the closest found precedent of "physics-aware weighting of flow for a water-dominated domain".
- A 2025 GNSS-denied UAV navigation survey notes learning-based methods struggle where features are sparse or uniform, explicitly including open water. [search summary] — [Satellite Navigation / Springer, 2025](https://link.springer.com/article/10.1186/s43020-025-00162-z)
- Existing UAV GNSS-denied localisation datasets (University-1652, SUES-200, SatLoc) are land-dominated; SatLoc combines learned matching, optical flow and altitude at >2 Hz on edge hardware. [search summary] — [SatLoc, Remote Sensing 2025](https://doi.org/10.3390/rs17173048)
- RealFlow (Han et al., ECCV 2022) generates *real-image* training pairs by EM-style iterative rendering from predicted flow + depth; repo licence is a custom/unrecognised licence (GitHub reports NOASSERTION). [repo API for licence; method description not re-verified] — [megvii-research/RealFlow](https://github.com/megvii-research/RealFlow)

### Inferences
- For a rasterised FFT ocean, the transferable lesson from Mayer et al. and AutoFlow is to randomise what the renderer controls cheaply — JONSWAP parameters (wind speed, fetch, peak enhancement), swell direction mix, sun elevation/azimuth, sky radiance, foam/whitecap coverage, haze, altitude, attitude, lens blur/noise/compression/rolling-shutter, exposure — rather than chase photorealism. AutoFlow's idea (tune generator hyperparameters to maximise validation accuracy on a small *real* labelled set) maps directly onto a parametric ocean renderer: the scarce real flights with GNSS can serve as AutoFlow's validation target.
- Water breaks the static-scene assumption that FlyingThings/TartanAir teach implicitly: the surface itself moves (phase speed of waves), so "flow" on water ≠ ego-motion. The renderer gives exact ego-motion and could also give surface-velocity ground truth, letting the model be trained to separate the two — something no public dataset offers.
- A Chairs/Things (or AutoFlow) pretrain followed by ocean-sim fine-tune followed by self-supervised real fine-tune is the standard three-stage schedule consistent with Mayer et al.'s "different data at selected stages".

### Gaps
- No public synthetic ocean/sea-surface dataset for flow or VO learning was found; no paper measuring sim-to-real transfer of flow/VO over open water was found.
- Exact AutoFlow-vs-Things numbers on Sintel/KITTI were not retrieved (abstract only).
- Mayer et al.'s detailed realism-vs-variety findings and any rendering-realism ablations (e.g., Spring, Kubric-based studies) were not re-read this session.
- Domain-randomisation-specific studies for flow (beyond AutoFlow) were not retrieved.

## Self-supervised / unsupervised fine-tuning on unlabeled real video, and behaviour on water

### Takeaway
The unsupervised-flow recipe is mature (census/SSIM photometric loss + occlusion masking + smoothness + self-distillation/augmentation consistency, as in UFlow → ARFlow → SMURF), and its best use is *fine-tuning a synthetically pretrained RAFT-family model on unlabeled target video*. But every one of these losses assumes brightness constancy on a rigid-ish surface; specular glint, sky reflection and wave motion on water violate it directly, and no published work validates these losses on sea surfaces.

### Cited Findings
- "What Matters in Unsupervised Optical Flow" (UFlow, Jonschkowski et al., ECCV 2020) systematically compares photometric losses, occlusion handling and smoothness regularisation to identify the most effective combination. [search summary] — [arXiv 2006.04902](https://arxiv.org/pdf/2006.04902). Code is within google-research (Apache-2.0) [repo API] — [google-research/google-research](https://github.com/google-research/google-research)
- UnFlow replaced raw intensity differences with a bidirectional census loss because brightness constancy is not invariant to the illumination changes common in realistic scenes. [search summary] — [UnFlow, AAAI 2018](https://cdn.aaai.org/ojs/12276/12276-13-15804-1-2-20201228.pdf)
- BrightFlow (WACV 2023) jointly estimates flow and per-pixel brightness change and compensates the photometric loss so that reconstruction errors from shadows or reflections do not corrupt training — the closest published remedy to non-Lambertian effects. [search summary] — [BrightFlow, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Marsal_BrightFlow_Brightness-Change-Aware_Unsupervised_Learning_of_Optical_Flow_WACV_2023_paper.pdf)
- Occluded regions cannot be optimised by photometric loss and are excluded via forward–backward consistency checks (standard in ASFlow and successors). [search summary] — [ASFlow arXiv 2104.03560](https://arxiv.org/pdf/2104.03560)
- ARFlow (Liu et al., CVPR 2020) adds augmentation-as-regularisation (consistency between flow on original and transformed pairs); code MIT. [repo API for licence; method not re-verified] — [lliuz/ARFlow](https://github.com/lliuz/ARFlow); [arXiv 2003.13045](https://arxiv.org/abs/2003.13045)
- SMURF (Stone et al., CVPR 2021) applies a RAFT architecture with full-image warping, multi-frame self-supervision and sequence-loss self-distillation, reporting large error reductions over prior unsupervised methods and approaching supervised PWC-Net/FlowNet2 on KITTI. [not re-verified] — [arXiv 2105.07014](https://arxiv.org/abs/2105.07014)
- Online/test-time adaptation for deep VO: Li et al. (CVPR 2020) use online meta-learning, a ConvLSTM and feature-distribution alignment so a self-supervised VO net keeps adapting without labels, reporting gains in virtual-to-real and outdoor-to-indoor shifts; the CVPR 2021 follow-up refines a single-view depth prior with Bayesian updates and uses an online-learned photometric uncertainty inside a differentiable Gauss–Newton layer (Cityscapes→KITTI, KITTI→TUM). [search summary; numbers not verified] — [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Self-Supervised_Deep_Visual_Odometry_With_Online_Adaptation_CVPR_2020_paper.html); [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalizing_to_the_Open_World_Deep_Visual_Odometry_With_Online_CVPR_2021_paper.pdf)

### Inferences (water-specific failure modes and remedies — reasoned, not published)
- **Specular glint / sun glitter**: view-dependent, moves with surface slope; photometric and census losses will reward matching glints, which track facet orientation, not ground motion. Remedies: mask high-intensity/saturated pixels; BrightFlow-style brightness-change head; learned per-pixel photometric uncertainty (as in Li et al. 2021) so the loss down-weights unreliable pixels.
- **Sky reflection gradients**: low-texture, large-scale; smoothness priors will fill them with plausible but wrong flow — census helps with illumination but not with absence of texture.
- **Moving surface**: wave phase speed means even a perfect flow is not ego-motion. For VO, the loss should be on the *pose/velocity output* rather than on dense photometric agreement alone; or the flow should be decomposed (ego component via a rigid/planar model + residual surface motion). Using a planar-sea homography constraint during self-supervision (ocean surface ≈ plane at known altitude) is a natural regulariser.
- **Exposure/auto-gain in real video**: the census transform or SSIM handles global gain changes better than L1; BrightFlow handles local ones.
- **Practical recipe**: (1) pretrain on Things/AutoFlow; (2) supervised fine-tune on the rendered ocean with full ground truth; (3) self-supervised fine-tune on unlabeled real flight video with census + occlusion mask + glint mask + teacher self-distillation from the stage-2 model (SMURF-style), weighted low to avoid drift; (4) validate only on GNSS-scored real segments.

### Gaps
- No paper found that evaluates UFlow/ARFlow/SMURF/BrightFlow on water or sea-surface video; failure modes above are reasoning, not measurement.
- SemiFlow was not located/verified this session.
- No published quantification of how self-supervised fine-tuning on a target domain trades off against forgetting of synthetic-supervised accuracy for RAFT-family models in a non-driving domain.

## GNSS velocity as weak supervision for VO fine-tuning; test-time adaptation

### Takeaway
No paper was found that uses recorded GNSS velocity as the weak label for fine-tuning a learned flow/VO front end over water. The closest literature is self-supervised online-adaptation VO (Li et al. 2020/2021). Using GNSS velocity (plus IMU attitude and altitude) as a metric supervision on the *output* of the front end is technically straightforward and avoids the photometric-loss problems on water.

### Cited Findings
- Online self-supervised adaptation of deep VO to new environments without labels (meta-learning + ConvLSTM + feature alignment) has been shown in virtual→real settings. [search summary] — [Li et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Self-Supervised_Deep_Visual_Odometry_With_Online_Adaptation_CVPR_2020_paper.html)
- A differentiable Gauss–Newton layer with online-learned photometric uncertainty allows adaptation at test time while keeping geometric structure. [search summary] — [Li et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalizing_to_the_Open_World_Deep_Visual_Odometry_With_Online_CVPR_2021_paper.pdf)
- SC-wLS (Cao et al.) shows self-supervised test-time adaptation of a correspondence-weighting network for visual localisation (7Scenes/Cambridge). [search summary] — [catalyzex author page](https://www.catalyzex.com/author/Yingdian%20Cao)

### Inferences
- With altitude (baro/radar) and attitude (IMU), a nadir/oblique camera over a near-planar sea gives a closed-form mapping from image-plane ego-flow to horizontal velocity. GNSS horizontal velocity therefore supervises the flow → velocity head (and, through it, the flow) without needing per-pixel flow labels; this sidesteps glint/sky photometric failures. Wave-induced surface drift and wind-driven current bias will appear as a systematic residual that should be modelled (or learned) separately.
- Adapter-style fine-tuning (only the final refinement/regression layers or LoRA on the encoder) is the safer choice with scarce real flights, to limit overfitting and preserve synthetic-trained generalisation. This is a common practice inference, not a sourced result for this domain.
- Test-time adaptation in flight (no GNSS) would have to rely on photometric self-supervision, which is precisely what water breaks; offline adaptation on GNSS-labelled flights is lower risk.

### Gaps
- No source found for GNSS-velocity-supervised VO/flow fine-tuning in any domain; no maritime test-time-adaptation result found.

## Jetson Orin performance: published FPS/latency, INT8 accuracy, memory

### Takeaway
The only systematic, same-device flow benchmark found is NeuFlow v2's (author-measured, Jetson **Orin Nano**, FP16, Sintel 1024×436): NeuFlow v2 0.106 s/frame (~9 Hz), RAPIDFlow 0.252 s, GMFlow 0.404 s, SEA-RAFT(M) 0.524 s, SEA-RAFT(L) 0.910 s, RAFT(32 it) 1.060 s. At ~512×384 the authors claim >20 FPS for NeuFlow v2 on Orin Nano. Orin NX/AGX are substantially faster than Nano, so 10 Hz at 320–640 px looks feasible for NeuFlow v2 and plausible for SEA-RAFT(M) with fewer iterations + TensorRT on AGX, but no independent Orin NX/AGX measurements for these flow models were found. INT8 on RAFT-style recurrent flow has been reported to cause major accuracy loss.

### Cited Findings
- NeuFlow v2 Jetson Orin Nano table (half precision; Sintel 1024×436 / KITTI 1242×375, s/frame): SKFlow(32) 4.181/4.478; SEA-RAFT(M)(4) 0.524/0.549; DIP(20) 3.615/3.767; GMFlowNet(32) 2.626/2.761; FlowFormer 11.196/11.172; RPKNet(12) 0.947/0.976; SEA-RAFT(L)(12) 0.910/0.953; GMA(32) 1.245/1.343; RAFT(32) 1.060/1.126; CRAFT crashed the device; RAPIDFlow(12) 0.252/0.254; GMFlow(1) 0.404/0.426; NeuFlow v1 0.064/0.070; NeuFlow v2(9) 0.106/0.114. Author measurements, not independent. [fetched] — [arXiv 2408.10161v3](https://arxiv.org/html/2408.10161v3)
- NeuFlow v2 headline claim: >20 FPS at 512×384 on Jetson Orin Nano, 10×–70× faster than SOTA at similar accuracy; no independent benchmark found. [search summary of paper] — [arXiv 2408.10161](https://arxiv.org/abs/2408.10161)
- RAFT on Jetson **AGX Xavier** (2022, Carvalho, Zaykov, Ukaye): reported PyTorch baseline ~0.95 FPS → ~1.43 FPS with TensorRT runtime, a further ~2.4× with FP16; INT8 quantisation caused major accuracy loss. [abstract fetched confirms RAFT on AGX Xavier with "accuracy preserved" goal; the FPS/INT8 figures are from a search summary, full text not readable] — [arXiv 2208.02808](https://arxiv.org/abs/2208.02808)
- RAFT through TensorRT inside a human-action-recognition pipeline on a Jetson held a steady ~130 ms latency. [search summary] — [arXiv 2409.05662](https://arxiv.org/pdf/2409.05662)
- KLTNet (Aug 2026) runs its CoarseFlowNet (1/4 resolution) and TriPatchRefiner (full resolution) under TensorRT on Jetson AGX Orin and claims real-time tracking with low GPU memory for 25–150 tracks; numbers are only in a figure (Fig. 5), not in text. It compares against RAFT-Sparse and SEA-RAFT-Sparse inside a VINS-Mono front end and lists DPVO/DROID-VO as system-level references. [fetched] — [arXiv 2608.24544](https://arxiv.org/html/2608.24544v1)
- LightGlue on Jetson Orin Nano (MAXN_SUPER, TensorRT 10.3, FP16, one image pair, ALIKED-type features): ~7.9 ms at 512 keypoints, 21.6 ms at 1024, 126.5 ms at 3072 (matcher only). Model-card measurement by the publisher. [search summary] — [kornia/lightglue on Hugging Face](https://huggingface.co/kornia/lightglue)
- Deployment gotcha (2026): ONNX Runtime's TensorRT execution provider stalled at session setup on Orin Nano; compiling an offline device-specific TensorRT FP16 engine avoided it (~2.2 ms/query in that work); engines must be rebuilt per target device. [search summary] — [Jetson-ORB-SLAM3, arXiv 2608.17874](https://arxiv.org/pdf/2608.17874)
- TensorRT is not always faster than PyTorch for transformer-heavy models (Grounding DINO on desktop GPUs), though fastest for CNNs on Jetson in a 2026 industrial benchmark. [search summary] — [arXiv 2607.11356](https://arxiv.org/pdf/2607.11356)
- Other Orin reference point: DEFOM-Stereo variants ran at ~2.2–8.5 FPS TensorRT FP16 on Jetson Orin (Super, 16 GB) — shows that RAFT-style iterative stereo/flow with ViT encoders lands in single-digit FPS on Orin-class hardware. [search summary] — [arXiv 2603.26250](https://arxiv.org/pdf/2603.26250)

### Inferences
- At 320–640 px wide (roughly 0.3–0.6× Sintel pixel count) and on Orin NX/AGX (more GPU than Nano), NeuFlow v2 should clear 10 Hz with margin; SEA-RAFT(M) with 4 iterations is the most likely "accurate RAFT-family" model to reach 10 Hz on AGX Orin. Full RAFT (32 iterations) and GMFlow are unlikely to reach 10 Hz at 640 px on NX without iteration cuts. These are extrapolations from Orin Nano timings, not measurements.
- Prefer FP16; treat INT8 for the recurrent update / correlation-volume parts as high-risk (RAFT INT8 loss reported) — if INT8 is wanted, quantise only the CNN feature/context encoders with QAT and keep the correlation lookup and GRU in FP16.
- Sparse front ends (XFeat + LightGlue at ~512 keypoints, or a KLTNet-style sparse tracker) are an order of magnitude cheaper than dense flow on Orin, but on open water keypoint detectors will fire on transient glints and wave crests — keypoint count and stability over water is the open risk, not latency.

### Gaps
- No independent (non-author) Jetson Orin NX or AGX FPS/latency found for RAFT, SEA-RAFT, NeuFlow v2, GMFlow or DPVO at 320–640 px.
- No published Jetson Orin numbers found for DPVO, DINOv2-S/B backbones, or XFeat.
- No published INT8 accuracy-loss numbers for SEA-RAFT/NeuFlow v2; RAFT INT8 loss reported only qualitatively (via search summary).
- Memory footprints per model on Orin were not found in text form.

## Knowledge distillation of large dense matchers (UFM/RoMa) into small real-time students

### Takeaway
Distillation of dense matchers into edge students is plausible and has precedent (SMURF-style self-distillation in flow; RoMa's authors ship a "Tiny RoMa" on an XFeat backbone), but no published recipe for distilling UFM or RoMa into a Jetson-real-time flow/VO student was found.

### Cited Findings
- RoMa's repo provides Tiny RoMa v1, "built on top of XFeat", scoring MegaDepth-1500 AUC@5/10/20 of 56.4/69.5/79.5 vs XFeat 46.4/58.9/69.2 and XFeat* 51.9/67.2/78.9; the README gives no speed numbers and does not say it was distilled. [fetched] — [Parskatt/RoMa](https://github.com/Parskatt/RoMa)
- SMURF uses sequence-loss self-distillation (teacher predictions as targets on cropped/augmented views) within unsupervised flow. [not re-verified] — [arXiv 2105.07014](https://arxiv.org/abs/2105.07014)

### Inferences
- A workable recipe: run UFM (BSD-3 code) or RoMa (MIT code) offline on the real flight video to produce dense pseudo-labels with confidence; train a NeuFlow v2 / SEA-RAFT(M) student on (rendered ocean GT) ∪ (confidence-filtered teacher pseudo-labels on real video). Teacher errors on glint/sky are inherited, so confidence filtering and the GNSS-velocity consistency check should gate pseudo-labels.
- Licence of the teacher matters even for offline distillation if weights or derivatives are shipped (see licences section): avoid CroCo/DUSt3R/MASt3R-based teachers for a commercial product.

### Gaps
- No paper found distilling UFM or RoMa into a real-time flow student; Tiny RoMa's training method not documented in the README.
- UFM's weight licence (as distinct from code) was not checked.

## Licences of commonly used pretrained code/weights (commercial use)

### Takeaway
Commercially permissive: RAFT, SEA-RAFT, DROID-SLAM, UFM code, TartanVO (BSD-3); DPVO, RoMa, DeDoDe, unimatch/GMFlow-unified, ARFlow (MIT); GMFlow, NeuFlow v1/v2, DINOv2, XFeat, LightGlue code, LoFTR, DISK, AutoFlow, Depth-Anything-V2/3 code (Apache-2.0). Non-commercial: CroCo and MASt3R (CC BY-NC-SA 4.0; DUSt3R/MUSt3R also NOASSERTION, same family), SuperPoint/SuperGlue (Magic Leap academic/non-profit only). DINOv3 is a custom Meta licence that allows commercial use but **prohibits ITAR-regulated and military/warfare end uses**, requires flow-down of the licence, and can be unilaterally amended — material for an aircraft navigation product.

### Cited Findings (all [repo API] or [fetched], checked 29 Oct 2026)
- RAFT — BSD-3-Clause — [princeton-vl/RAFT LICENSE](https://github.com/princeton-vl/RAFT/blob/master/LICENSE)
- SEA-RAFT — BSD-3-Clause — [princeton-vl/SEA-RAFT LICENSE](https://github.com/princeton-vl/SEA-RAFT/blob/main/LICENSE)
- NeuFlow v2 — Apache-2.0 (NeuFlow v1 also Apache-2.0) — [neufieldrobotics/NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2)
- GMFlow — Apache-2.0 — [haofeixu/gmflow LICENSE](https://github.com/haofeixu/gmflow/blob/main/LICENSE); UniMatch (GMFlow/GMStereo unified) — MIT — [autonomousvision/unimatch LICENSE](https://github.com/autonomousvision/unimatch/blob/master/LICENSE)
- DPVO — MIT — [princeton-vl/DPVO LICENSE](https://github.com/princeton-vl/DPVO/blob/main/LICENSE); DROID-SLAM — BSD-3-Clause — [princeton-vl/DROID-SLAM LICENSE](https://github.com/princeton-vl/DROID-SLAM/blob/main/LICENSE)
- TartanVO — BSD-3-Clause — [castacks/TartanVO](https://github.com/castacks/TartanVO); tartanair_tools — BSD-3-Clause — [castacks/tartanair_tools](https://github.com/castacks/tartanair_tools/blob/master/LICENSE)
- DINOv2 — Apache-2.0 — [facebookresearch/dinov2 LICENSE](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)
- DINOv3 — custom "DINOv3 License", last updated 19 Aug 2025: royalty-free, worldwide, non-exclusive, non-transferable licence to use, reproduce, distribute and modify; redistribution of materials *or derivative works* must be under the same agreement with a copy provided; publications must acknowledge use; no reverse engineering; licensee must not use or permit use "for any activities subject to the International Traffic in Arms Regulations (ITAR) or end uses prohibited by Trade Controls, including those related to military or warfare purposes, nuclear industries or applications, espionage…"; patent-litigation termination clause; Meta may modify the agreement with changes effective immediately. Weights are gated (sign-up). [fetched raw LICENSE.md] — [facebookresearch/dinov3 LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md); [Meta DINOv3 License page](https://ai.meta.com/resources/models-and-libraries/dinov3-license); community discussion that the custom licence "requires an ad-hoc review by legal" — [dinov3 issue #31](https://github.com/facebookresearch/dinov3/issues/31)
- XFeat — Apache-2.0 — [verlab/accelerated_features LICENSE](https://github.com/verlab/accelerated_features/blob/main/LICENSE)
- LightGlue code — Apache-2.0 — [cvg/LightGlue LICENSE](https://github.com/cvg/LightGlue/blob/main/LICENSE)
- SuperGlue/SuperPoint (Magic Leap) — "ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY" — [magicleap/SuperGluePretrainedNetwork LICENSE](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE)
- RoMa — MIT ("All our code except DINOv2 is MIT license. DINOv2 has an Apache 2 license") — [Parskatt/RoMa](https://github.com/Parskatt/RoMa); DeDoDe — MIT — [Parskatt/DeDoDe](https://github.com/Parskatt/DeDoDe/blob/main/LICENSE)
- UFM — code BSD-3-Clause — [UniFlowMatch/UFM LICENSE](https://github.com/UniFlowMatch/UFM/blob/main/LICENSE)
- CroCo — CC BY-NC-SA 4.0 — [naver/croco LICENSE](https://github.com/naver/croco/blob/master/LICENSE); MASt3R — CC BY-NC-SA 4.0 — [naver/mast3r LICENSE](https://github.com/naver/mast3r/blob/main/LICENSE); DUSt3R and MUSt3R — GitHub reports NOASSERTION (custom text, same Naver family) — [naver/dust3r LICENSE](https://github.com/naver/dust3r/blob/main/LICENSE)
- ARFlow — MIT — [lliuz/ARFlow](https://github.com/lliuz/ARFlow/blob/master/LICENSE); UFlow/SMURF (google-research monorepo) — Apache-2.0 — [google-research LICENSE](https://github.com/google-research/google-research/blob/master/LICENSE)
- AutoFlow — Apache-2.0 — [google-research/opticalflow-autoflow](https://github.com/google-research/opticalflow-autoflow)
- LoFTR — Apache-2.0 — [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR); EfficientLoFTR — NOASSERTION (custom) — [zju3dv/EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR); DISK — Apache-2.0 — [cvlab-epfl/disk](https://github.com/cvlab-epfl/disk)
- Depth-Anything-V2 and Depth-Anything-3 repos — Apache-2.0 at repo level — [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3); VGGT — NOASSERTION (custom) — [facebookresearch/vggt](https://github.com/facebookresearch/vggt)

### Inferences
- Repo LICENSE files cover code; pretrained weights can carry separate terms (e.g., some Depth-Anything model sizes and VGGT checkpoints are released under different terms than the repo). Every weight file to be shipped needs its own check.
- Weights trained on FlyingThings/Sintel/KITTI inherit dataset terms questions (KITTI is CC BY-NC-SA 3.0; Sintel non-commercial) — retraining on own synthetic ocean + licensed data avoids that ambiguity for the shipped model. (KITTI/Sintel dataset licences were not re-checked this session.)
- DINOv3's ITAR/military clause is the main licence risk if the product could be sold into defence; DINOv2 (Apache-2.0) has no such clause.
- LightGlue's code is Apache-2.0, but pairing it with SuperPoint pulls in Magic Leap's non-commercial terms; pair with XFeat, DISK, ALIKED or DeDoDe instead.

### Gaps
- Not checked: separate weight licences for UFM, SEA-RAFT and NeuFlow v2 checkpoints (assumed to follow repo licences); DUSt3R/MUSt3R/VGGT/EfficientLoFTR/RealFlow exact custom terms; KITTI/Sintel/FlyingThings dataset licences and how they bear on trained weights.
