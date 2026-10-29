# Visual front ends beyond UFM: dense flow, dense matching and correspondence models for GNSS-free over-water navigation

Research date: 29 October 2026. About 17 web searches and fetches. Primary sources are official GitHub READMEs and arXiv abstract pages. Where a figure comes from a search-engine summary of a paper rather than from the paper itself, it is marked "(via search summary)". Licence statements are what the repos say, not legal advice.

## Q1. Why pick UFM? Published results, speed, limitations and exact licence

### Takeaway
UFM is a single transformer that outputs both dense flow and a per-pixel covisibility mask. It covers small-motion optical flow and wide-baseline matching in one model. The authors claim it is 28% more accurate than UniMatch on flow, and that against RoMa it has 62% less error and runs 6.7x faster. The code is BSD-3-Clause. The released checkpoints are **non-commercial (CC BY-NC-SA 4.0)** because they inherit the licences of their training datasets, and an Apache-2.0 release is only "considered". No real-time "UFM-Tiny" had been released as of the repo snapshot.

### Cited Findings
- UFM ("UFM: A Simple Path towards Unified Dense Correspondence with Flow") comes from CMU: Zhang, Keetha, ... Ramanan, Scherer, Wang. v1 was submitted 10 June 2025 and v2 revised 10 February 2026. The repo says it was presented at NeurIPS 2025. — [arXiv 2506.09278](https://arxiv.org/abs/2506.09278); [GitHub UniFlowMatch/UFM](https://github.com/UniFlowMatch/UFM)
- Abstract claims: "28% more accurate than state-of-the-art flow methods (Unimatch)", "62% less error" and "6.7x faster" than dense wide-baseline matchers (RoMa). It is trained on unified data for co-visible pixels. — [arXiv 2506.09278](https://arxiv.org/abs/2506.09278)
- Outputs: the Python API returns `result.flow.flow_output` (the flow field) and `result.covisibility.mask` (a per-pixel covisibility mask). — [GitHub UFM](https://github.com/UniFlowMatch/UFM)
- Runtimes and sizes, all author-reported on an RTX 5090 at default resolution:
  - UFM-Base: 33 ms, 0.4B parameters
  - UFM-Refine: 44 ms, 0.4B
  - UFM-Base-980: 77 ms
  - UFM-Refine-980: 96 ms
  - UFM-Base-DINOv2L-init: 28 ms, 0.3B
  - UFM-Base-DINOv2G-init: 55 ms, 1B

  — [GitHub UFM](https://github.com/UniFlowMatch/UFM)
- Release timeline:
  - 2025-06-10: initial checkpoints
  - 2025-10-08: 980-px models
  - 2025-10-20: benchmark branch
  - 2025-10-21: training scripts
  - 2026-02-20: smaller DINOv2-initialised UFM
  - 2026-04-03: UFM-G (DINOv2-G init)

  "UFM-Tiny" for real-time robotics is listed as upcoming, not released. — [GitHub UFM](https://github.com/UniFlowMatch/UFM)
- Licence wording: the code is under a "fully open-source BSD-3-Clause license". The pre-trained checkpoints "inherit the licenses of the underlying training datasets" and "may not be used for commercial purposes (CC BY-NC-SA 4.0)". The maintainers say they may consider an Apache 2.0 version, which is not a commitment. — [GitHub UFM](https://github.com/UniFlowMatch/UFM)
- The README gives no benchmark tables. The numbers are in the paper and on the benchmark branch. — [GitHub UFM](https://github.com/UniFlowMatch/UFM)

### Inferences
- Why a team would pick UFM:
  - One network covers both regimes Mayfly sees: consecutive frames at 10 Hz (small baseline) and re-acquisition after a gap or a large yaw (wide baseline).
  - The covisibility mask is a ready-made gate for a filter.
  - DINO features are the strongest generic prior available for weakly textured scenes.
- The licence is decisive. The code is clean (BSD-3), but every shipped checkpoint is NC. Commercial use means retraining UFM from the BSD code on data the team can use commercially. Since the weights "inherit" the dataset licences, the training-data choice is the licence problem, not the architecture.
- Speed on the target hardware: 28–44 ms on an RTX 5090 for a 0.3–0.4B ViT-L-class model points to roughly an order of magnitude slower on a Jetson AGX Orin. That puts 10 Hz at 320–640 px doubtful without a smaller encoder (the 2026 "smaller DINOv2" variant) or the unreleased UFM-Tiny. This is an inference from GPU class ratios, not a measurement.
- Choice of encoder: the user's brief says DINOv3-B, but the public checkpoint table names only DINOv2-L and DINOv2-G initialisations. DINOv3's licence carries a military-use exclusion (see Q3), so a DINOv2-initialised build has the cleaner chain.

### Gaps
- I did not open UFM's paper tables. Per-benchmark numbers (Sintel, KITTI, Spring, TA-WB, ETH3D) and the input resolution behind the 5090 timings are not captured here.
- I found no published Jetson or TensorRT measurement of UFM.
- I did not verify that DINOv3-B-initialised UFM weights exist.

## Q2. Optical-flow alternatives: accuracy, runtime (Jetson Orin), licence

### Takeaway
On accuracy, WAFT (Princeton, ICLR 2026) is the current published leader: rank 1 on Spring, Sintel and KITTI with a DAv2 backbone. SEA-RAFT (ECCV 2024, BSD-3) is the accuracy/speed sweet spot, and it also outputs per-pixel uncertainty. NeuFlow v2 (Apache-2.0) is the only candidate with a published Jetson Orin real-time claim: over 20 FPS at 512x384 on an Orin Nano. A 2026 trend is single-pass flow built on frozen foundation-model features (DINOv2 plus a depth model).

### Cited Findings
- **WAFT** (Wang & Deng, Princeton; ICLR 2026)
  - Design: RAFT-like, but the cost volume is replaced by high-resolution feature warping, which gives lower memory use.
  - Accuracy claim: rank 1 on Spring, Sintel and KITTI, plus the best zero-shot KITTI result.
  - With an ImageNet-only Twins backbone it ranks 1st on Spring, 2nd on KITTI and is competitive on Sintel. A depth-pretrained DAv2 backbone is needed to lead everywhere.
  - Speed: 1.3x faster than FlowFormer++ and 4.1x faster than CCMR+.
  - Sintel Final is hurt by one outlier sequence (Ambush 1).
  - Code and weights: github.com/princeton-vl/WAFT.

  — [arXiv 2506.21526](https://arxiv.org/html/2506.21526v3); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10010386)
- **SEA-RAFT** (ECCV 2024)
  - Accuracy: Spring EPE 3.69 and 1px outlier rate 0.36, state of the art at publication.
  - Speed: "at least 2.3x faster than existing methods"; over 20 fps at 1080p on an RTX 3090 (author claim).
  - Cost at 540x960: SEA-RAFT(M) takes 70.96 ms / 486.9 GMACs, against RAFT's 140.7 ms / 938.2 GMACs (via search summary).

  — [arXiv 2405.14793](https://arxiv.org/pdf/2405.14793); [ECCV 2024 page](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1065_ECCV_2024_paper.php)
- SEA-RAFT(L) benchmark scores: Sintel clean 1.31, Sintel final 2.60, KITTI Fl-all 4.30. These come from a secondary summary and need checking against the paper. — [visual-slam-roadmap summary](https://raw.githubusercontent.com/changh95/visual-slam-roadmap/main/level-05-deep-learning/sea-raft.md)
- SEA-RAFT repo:
  - Code licence: BSD-3-Clause. No separate weight licence is stated.
  - Trained with a mixture-of-Laplace loss; `custom.py` visualises "the uncertainty".
  - Checkpoint `Tartan-C-T-TSKH-spring540x960-M` is on Hugging Face (MemorySlices).

  — [GitHub princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
- **NeuFlow v2** (Neufield Robotics, Northeastern)
  - Speed claim: "over 20 FPS at 512x384 resolution on a Jetson Orin Nano", and 10x–70x faster than state-of-the-art methods at comparable accuracy.
  - A German news site quotes about 30 FPS and 10–80x. That contradicts the paper; trust the paper.

  — [arXiv 2408.10161](https://arxiv.org/abs/2408.10161); contradicted by [mind-verse.de](https://mind-verse.de/news/neue-entwicklungen-neuflow-v2-optischer-fluss-edge-geraete)
- NeuFlow v2 comparison table (via search summary of the paper):
  - Orin Nano times: NeuFlow-V2 (9 iterations) 0.106, RAPIDFlow 0.252, SEA-RAFT(M) at 4 iterations 0.524, SEA-RAFT(L) at 12 iterations 0.910. The unit is labelled seconds but is ambiguous.
  - SEA-RAFT(M) at 4 iterations scores Sintel final 4.04 and KITTI 14.2, at about 0.061 s/frame on an RTX 2080.

  — [arXiv 2408.10161 PDF](https://arxiv.org/pdf/2408.10161)
- NeuFlow v2 repo:
  - Code licence: Apache-2.0. A weight licence is not separately stated.
  - Checkpoints: `neuflow_mixed.pth` (default), `neuflow_sintel.pth`, `neuflow_things.pth`.
  - Training data: FlyingChairs, FlyingThings3D, Sintel, KITTI, HD1K.
  - No Jetson or TensorRT instructions in the README.

  — [GitHub NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2)
- **"Rethinking Dense Optical Flow without Test-Time Scaling"** (Chanda & Kumar)
  - Published 8 May 2026 on arXiv, CVPR 2026 ViSCALE workshop.
  - Method: single forward pass, global matching on a frozen DINOv2 plus a monocular-depth foundation model, with no iterative refinement.
  - Result: Sintel Final EPE 2.81 without refinement, claimed better than SEA-RAFT, RAFT, GMFlow and FlowSeek in the same setting.
  - No runtime reported.

  — [arXiv 2605.08000](https://arxiv.org/abs/2605.08000)
- A further September 2026 preprint, "FreeFlow: A Bias-free Hierarchical Transformer for Optical Flow Estimation", surfaced in search. I did not read it. — [arXiv 2609.11486](https://arxiv.org/pdf/2609.11486)
- No SEA-RAFT or RAFT TensorRT benchmark on Jetson Orin turned up in search. The only Orin numbers found were in the NeuFlow v2 paper's table. — [arXiv 2408.10161](https://arxiv.org/pdf/2408.10161)

### Inferences
- For 10 Hz at 320–640 px on an AGX Orin, these models look feasible:
  - NeuFlow v2: it does over 20 FPS on the much smaller Orin Nano at 512x384.
  - SEA-RAFT(S/M) with few iterations under TensorRT FP16: plausible, but unmeasured.
- UFM-class ViT-L models and WAFT with a DAv2 backbone look infeasible at 10 Hz without distillation.
- SEA-RAFT's per-pixel Laplace-mixture uncertainty is the most directly filter-usable output among the fast flow models.
- Licence: SEA-RAFT (BSD-3) and NeuFlow v2 (Apache-2.0) have permissive code. Their weights are trained on FlyingThings, Sintel, KITTI, HD1K, TartanAir and Spring. UFM itself says weights inherit dataset terms, and KITTI is commonly distributed under a non-commercial CC licence. So the commercial status of "permissive" pretrained flow weights is unsettled, and a team should assume it may need to retrain on its own data.

### Gaps
- Not verified this session; these are known models but nothing below was sourced here:
  - Licences, runtimes and benchmark numbers for RAFT (believed BSD-3), FlowFormer/FlowFormer++, GMFlow/UniMatch (believed MIT), VideoFlow, MemFlow, StreamFlow, RAPIDFlow, DPFlow and FlowSeek.
  - WAFT's licence and parameter count.
  - The current top of the Spring, Sintel and KITTI leaderboards as of October 2026.
- No independent (non-author) TensorRT or Jetson AGX Orin measurement was found for any flow model.

## Q3. Dense matchers and 3D foundation models, and backbone licences (DINOv2 vs DINOv3)

### Takeaway
Among dense and semi-dense matchers, RoMa (MIT code, DINOv2 Apache-2.0) and EfficientLoFTR are the strongest, but they target wide-baseline rigid scenes and are slower than flow models. LightGlue with DISK or ALIKED gives a fully permissive sparse pipeline; SuperPoint's weights are restrictively licensed. MASt3R/DUSt3R are CC BY-NC-SA. VGGT now has a commercial checkpoint (VGGT-1B-Commercial) that excludes military use. CoTracker's demo is CC BY-NC. DINOv3 is under a custom Meta licence that permits use but prohibits military and ITAR uses. DINOv2 is Apache-2.0.

### Cited Findings
- **RoMa:** "All our code except DINOv2 is MIT license. DINOv2 has an Apache 2 license." It builds on DKM. No explicit licence for the RoMa weights was found; they are hosted as GitHub release assets. — [GitHub Parskatt/RoMa](https://github.com/Parskatt/roma); [PyPI romatch](https://pypi.org/project/romatch)
- **LightGlue:**
  - Code and weights: Apache-2.0.
  - SuperPoint weights and inference file: "a different, restrictive license".
  - DISK: Apache-2.0. ALIKED: BSD-3-Clause.
  - Speed: about 150 FPS at 1024 keypoints and 50 FPS at 4096 on an RTX 3080 (compiled, adaptive); about 20 FPS at 512 keypoints on an i7 CPU.

  — [GitHub cvg/LightGlue](https://github.com/cvg/LightGlue)
- **EfficientLoFTR** (CVPR 2024): about 2.5x faster than LoFTR, and the authors say it can beat SuperPoint+LightGlue on speed. Licence not found in this session. — [CVF open access](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Efficient_LoFTR_Semi-Dense_Local_Feature_Matching_with_Sparse-Like_Speed_CVPR_2024_paper.html)
- **MASt3R/DUSt3R:** listed under CC BY-NC-SA 4.0, with checkpoints trained on datasets that include a non-commercial one. These sources are secondary (a paper appendix and a model-card mirror). — [arXiv 2506.09883 licence appendix](https://arxiv.org/pdf/2506.09883); [MASt3R checkpoint mirror](https://ai.gitcode.com/hf_mirrors/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric)
- **VGGT:**
  - Since 29 July 2025 the code is under a "commercial-use-friendly license", excluding military applications.
  - Only VGGT-1B-Commercial (gated application) may be used commercially; the original checkpoint stays non-commercial.
  - Co3D AUC@30 is 90.37 for the commercial checkpoint against 89.98 for the original.
  - Reconstructs a scene in under 1 s. The smaller 500M and 200M models are not released.

  — [GitHub facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- **CoTracker:** the Hugging Face demo Space is cc-by-nc-4.0. The repo licence was not confirmed. — [HF cotracker README](https://huggingface.co/spaces/facebook/cotracker/blob/main/README.md); [GitHub co-tracker](https://github.com/facebookresearch/co-tracker)
- **DINOv3 licence:**
  - A custom "DINOv3 License": non-exclusive, worldwide, royalty-free. It does not exclude commercial use.
  - Prohibits ITAR-controlled activities and "military or warfare purposes", plus nuclear, espionage and weapons uses.
  - Requires sanctions and export compliance, carrying the licence onto derivatives, and acknowledgement in publications.
  - Meta may modify the terms.

  — [DINOv3 LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)

### Inferences
- **Frame-to-frame metric motion over a quasi-planar surface:** consecutive frames at 10 Hz are a small-baseline problem, so optical flow (SEA-RAFT, NeuFlow v2, UFM-flow) fits better than wide-baseline matchers. RoMa, MASt3R and VGGT matter mainly for:
  - re-localisation or loop closure across larger gaps, and
  - offline labelling or teacher distillation, when their licences allow.
- **Sparse pipelines:** LightGlue + DISK/ALIKED (all permissive) give a robust homography fit on a near-planar sea, but they depend on keypoints, and keypoint density on open water is low (see Q5).
- **Licence screen for a GNSS-free aircraft navigation product:**
  - DINOv3, VGGT-Commercial and Llama-style Meta licences exclude military use. If the product could be sold into defence, these exclusions matter even though civil commercial use is allowed.
  - DINOv2 (Apache-2.0) is the safer foundation backbone.
- **DINOv2 licence:** I took its Apache-2.0 status from the RoMa README's statement. I did not fetch the DINOv2 LICENSE file directly.

### Gaps
- Not verified this session:
  - XFeat licence (believed Apache-2.0) and speed.
  - DKM and EfficientLoFTR licences.
  - TAPIR/BootsTAP licences.
  - Official MASt3R and DUSt3R LICENSE files.
- MegaDepth and ScanNet pose-AUC numbers for RoMa, EfficientLoFTR and LightGlue were not captured.
- No Jetson numbers were found for any dense matcher.

## Q4. Which models give calibrated per-pixel uncertainty, covisibility or confidence?

### Takeaway
UFM outputs a covisibility mask. SEA-RAFT is trained with a mixture-of-Laplace loss and exposes per-pixel uncertainty. RoMa-family matchers produce match certainty. None of the sources found demonstrate calibration on aerial or water imagery.

### Cited Findings
- UFM returns `result.covisibility.mask` alongside flow. — [GitHub UFM](https://github.com/UniFlowMatch/UFM)
- SEA-RAFT uses a mixture-of-Laplace loss, and its demo visualises "the uncertainty". — [GitHub SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
- In UAV river velocimetry with RAFT, the authors added a variance-based score to filter unreliable flow results. This is a learned flow model with no native uncertainty, made filter-usable after the fact. — [DOAJ, Int. J. Appl. Earth Obs. Geoinf. 2024](https://doaj.org/article/7d5590cb8c66490c97b5181e6d9cf298)

### Inferences
- **What a Kalman or factor-graph filter needs:** a per-pixel or per-region variance, not just a binary mask.
  - SEA-RAFT's Laplace scale is the closest native output.
  - UFM's covisibility answers "is this pixel matchable" rather than "how far wrong".
- **Calibration on water** would need checking against recorded flights in either case, for example via NEES/NIS tests.

### Gaps
- I found no study of uncertainty calibration for any of these models on aerial-over-water footage.
- I did not confirm whether NeuFlow v2 or WAFT output any confidence; the READMEs reviewed mention none.

## Q5. Evidence on water, sea surface or low-texture aerial imagery

### Takeaway
The closest evidence is river surface velocimetry from UAVs, where RAFT (and, in one fixed-camera study, SEA-RAFT) generalises reasonably and can be fine-tuned. I found nothing published on deep flow for aircraft ego-motion over open sea. Classical sea-surface VO notes that water lacks a stable ground plane with enough disparity.

### Cited Findings
- Kriščiūnas et al. (Drones, 2026): a modified multi-frame RAFT for UAV river video, trained on finite-element-flow reference data with a GRU fusion module. Mean angular error is under 15°, and accuracy improves with higher spatial resolution and moderate frame spacing. — [KTU ePubl](https://epubl.ktu.edu/object/elaba:281557088/index.html)
- The same group's 2024 paper applied RAFT to UAV river footage with convolutional preprocessing, and used a variance score to reject unreliable flow. — [DOAJ](https://doaj.org/article/7d5590cb8c66490c97b5181e6d9cf298)
- Tlhomole et al. (J. Hydrology, 2025): RAFT generalised well to lab and field imagery, while a physics-informed approach struggled on drone data. — [Imperial Spiral](https://spiral.imperial.ac.uk/entities/publication/5ce1094f-d59f-4ea3-95b7-12efb7d79091)
- A Kyoto DPRI abstract tests SEA-RAFT against LSPIV and STIV on flood-period rivers, using fixed PTZ cameras rather than drones. — [DPRI C204](https://www.dpri.kyoto-u.ac.jp/hapyo/26/pdf/C204.pdf)
- FIG 2026 abstract: RAFT on surveillance video, with average error under 0.1 m/s for river surface velocity. — [FIG 2026](https://fig.net/resources/proceedings/fig_proceedings/fig2026/papers/ts03c/TS03C_zhang_di_et_al_13815_abs.pdf)
- A micro-UAV landing study found few feature points on water, and stereo breaking down on rotor-induced ripples. Dense Lucas-Kanade flow with a planar-homography error rose sharply near water. — [arXiv 1812.03539](https://arxiv.org/pdf/1812.03539)
- Monocular VO for an unmanned sea-surface vehicle needed IMU orientation, because "there exists no ground plane to provide features with enough disparity". — [arXiv 1707.04444](https://arxiv.org/pdf/1707.04444)
- Underwater work reports that learned flow (RAFT, GMFlow) is more robust to noise and degradation than classical feature or flow methods. — [arXiv 2506.06476](https://arxiv.org/pdf/2506.06476)
- PRGFlow: deep global flow fused with an IMU for aerial-robot 6-DoF estimation, which beats classical matching under noise. Not confirmed to have been tested over water. — [NSF PAR](https://par.nsf.gov/biblio/10371425-prgflow-unified-swapaware-deep-global-optical-flow-aerial-robot-navigation)

### Inferences
- **Flow measures the wrong thing for navigation.** River velocimetry wants the water's own motion; Mayfly wants the camera's motion relative to the Earth. Flow on water measures camera motion plus surface motion (current, wind drift, wave orbital motion and propagation).
  - Any front end must be paired with a way to separate or bound the surface-motion term, for example longer-wavelength texture or wind/sea-state priors.
  - A model's covisibility or uncertainty will not flag that bias, because the surface motion is consistent and well matched. It is a systematic error, not noise.
- **Altitude and texture:** at 30–400 m with a wide-angle lens, ground sample distance is coarse. Capillary waves and glitter vanish and swell or whitecaps dominate. Low-texture failure is likeliest in calm, glassy conditions, which matches the general camera-only limitation.
- **Learned flow on water:** the RAFT results in velocimetry suggest learned flow is usable on water texture. SEA-RAFT or NeuFlow v2 fine-tuned on over-water footage is a reasonable licence-clean path.

### Gaps
- I found no benchmark, dataset or paper on deep flow or matching for aircraft ego-motion over open sea.
- I found no evaluation of UFM, WAFT, RoMa or NeuFlow on water imagery.

## Q6. Licence-clean, commercially usable options, and the cost of training from scratch

### Takeaway
Permissive code: UFM (BSD-3), SEA-RAFT (BSD-3), NeuFlow v2 (Apache-2.0), RoMa (MIT, plus DINOv2 Apache-2.0), LightGlue/DISK/ALIKED (Apache/BSD). Pretrained weights are the problem. UFM says outright that its weights are NC because of their training data, and the same dataset-inheritance logic plausibly applies to other flow models trained on KITTI and similar data. The realistic commercial path is a permissive architecture plus a DINOv2 (Apache) backbone, retrained on the team's own data.

### Cited Findings
- UFM: BSD-3 code; checkpoints CC BY-NC-SA 4.0 because they "inherit the licenses of the underlying training datasets". — [GitHub UFM](https://github.com/UniFlowMatch/UFM)
- SEA-RAFT: BSD-3-Clause code; no weight licence stated. — [GitHub SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
- NeuFlow v2: Apache-2.0 code; weights trained on FlyingChairs, FlyingThings3D, Sintel, KITTI and HD1K. — [GitHub NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2)
- RoMa: MIT code, DINOv2 Apache-2.0. — [GitHub RoMa](https://github.com/Parskatt/roma)
- LightGlue: Apache-2.0 code and weights; SuperPoint restrictive; DISK Apache-2.0; ALIKED BSD-3. — [GitHub LightGlue](https://github.com/cvg/LightGlue)
- VGGT: commercial checkpoint available, with no military use. — [GitHub VGGT](https://github.com/facebookresearch/vggt)
- DINOv3: custom licence, no military or ITAR use. — [DINOv3 LICENSE](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- Training on the team's own physically-modelled reference data has precedent in the water domain: Kriščiūnas et al. trained RAFT on finite-element flow references. — [KTU ePubl](https://epubl.ktu.edu/object/elaba:281557088/index.html)

### Inferences
- **Candidate stacks:**
  - (a) NeuFlow v2 or SEA-RAFT-S/M, retrained on own data. This is real-time-feasible on an Orin. SEA-RAFT adds uncertainty.
  - (b) UFM code with a smaller DINOv2 (Apache) encoder, retrained, then distilled for the Orin.
  - (c) LightGlue + ALIKED/DISK as a permissive sparse fallback or cross-check.
- **Training cost** was not found in sources. RAFT-family training is typically a multi-day multi-GPU job, but treat that as unsourced.

### Gaps
- No source gave GPU-hours for training SEA-RAFT, NeuFlow v2 or UFM from scratch.
- No source gave an authoritative licence for SEA-RAFT or NeuFlow v2 weights, or settled whether dataset terms (KITTI, Sintel, FlyingThings, Spring, TartanAir) carry over to trained weights. That needs legal review.
