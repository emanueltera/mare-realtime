# Model B, `flowhead`: SEA-RAFT flow features with a learned temporal head

**Result.** Model B does not do what E2 asks. The model frozen on val is a head trained on
cached features from a frozen SEA-RAFT-M. Its median per-run bias is 1.30 m/s on val and
1.37 m/s on test-iid, against the 0.17 m/s budget. About 1 run in 150 is inside the budget.
That is no better than the classical phase-correlation reference at K = 8 (val 1.261,
test-iid 1.356), and better than plain SEA-RAFT at K = 8 (val 1.830, test-iid 1.841); both
of those are other agents' runs, copied for context to `reference_all.csv`. The bias still grows
with the sea's clock (test-rate). The head removes about 0.1 m/s of the wave-phase bias.
The training-run check shows the head **underfits**: it is as biased on its own training
runs as on val. So the frozen flow fields do not carry what the correction needs, or the
head cannot get it out of them. Fine-tuning the backbone's update block through the
velocity loss for 45 min lowers the val median to 0.98 m/s (test-iid 1.08) and was still
improving. This is the only lever that moved the number, and it still leaves only 2 % of
val runs inside the budget. Every figure below comes from a CSV in this directory.
Where a figure was derived outside the harness, the text says so.

## Architecture (`flowhead/model.py`)

1. **Tiles.** Each of the window's K = 8 tiles is z-scored on its own valid cells, using the
   centred 256 × 256 crop.
2. **Pairs.** Pairs (i, i+g) are taken for the gaps {1, 2, 4, 7}, as a chain that covers the
   window: 7 + 4 + 2 + 1 = 14 pairs (`make_pairs`). The gap-1 ablation uses the 7
   consecutive pairs only.
3. **SEA-RAFT-M.** This is the pretrained `Tartan-C-T-TSKH-spring540x960-M`, run for 4
   iterations. Its feature encoder runs once per frame. Pretrained SEA-RAFT returns about
   zero flow on this water once the displacement passes about 60 px. That covers fast, low
   runs and every longer gap: errors equal the full ground speed (see "Findings").
   Each pair's iterations therefore start from a global shift instead of SEA-RAFT's own
   regressed guess. The shift is an integer, from whitened phase correlation of each
   consecutive pair, summed along the chain (`phase_shift`, `chain_init`).
4. **Per-pair features at 1/8 resolution (10 channels):**
   - the raw velocity u = −flow · gsd / (g · dt) in m/s (content moves at −v), which is
     E1's failure mode;
   - the initial shift expressed as a velocity;
   - SEA-RAFT's mixture weights and log b;
   - the photometric residual after warping;
   - the fraction of each cell whose flow lands inside the tile.
5. **Head (2.05 M parameters).**
   - Inputs, besides the pair features: a learned stem on the first and last frame
     (wavelength, orientation, glint), the fraction of each cell valid in all K frames,
     log(gsd / 0.125) and dt / 0.1.
   - A 6-block dilated residual CNN at 1/8 resolution, with FiLM from a global summary.
   - Per cell and pair it outputs a weight logit w and a correction Δ (2 m/s scale,
     zero-initialised). The estimate is **v = Σ softmax(w)·(u + Δ)**, so the head starts as
     the plain mean of the raw flow.
   - σ (per tile axis) comes from the global summary and the weighted spread of u.
6. **Loss.** Huber(v − v_true, δ = 0.5 m/s) + 0.1 × the Gaussian NLL of the *detached*
   error under σ. σ learns to describe the error without down-weighting hard windows in the
   fit of v.

**Inputs, confirmed allowed-only.** The model sees the K tiles, their validity mask, the
tile GSD and dt. The GSD is the run's tile GSD, the `gsd` of frames.csv. dt is taken from
`t` differences.

- Labels (`v_true` from labels.csv) are used only as the target and by the harness.
- Positions, airspeed, heading, θ, sea and sky parameters, run ids and frame indices never
  reach the model.
- The window index decides which frames are loaded, nothing more.
- The phase-correlation initial shift is computed from the tiles alone.

## Training regimes

- **On the fly** (`flowhead/train.py`, method `flowhead-onthefly`). Random dihedral element
  and random 256 crop per window, over all 333 000 train windows. Frozen backbone, B = 8,
  AdamW, head learning rate 5e-4, cosine decay, bf16.
  - The GPU was shared four ways, so this gets 1.6–2 s per step: 12 000 steps (96 000
    windows) in 8 400 s.
  - Val median b went from 1.59 (step 1500) to 1.38 (best, step 10 500)
    (`curves/onthefly-main/val_log.csv`).
  - Too slow to learn much. That is why the regime below exists.
- **Cached features** (`flowhead/cache.py` + `flowhead/train_head.py`, the selected
  `flowhead`).
  - With the backbone frozen, a pair's features do not depend on anything trained. They
    were computed once for back-to-back windows: start step 8, centred crop, train runs
    0–3999, so 40 000 windows. Val: 150 runs, 1 500 windows, the same windows the
    on-the-fly validation uses.
  - Then the head trained for 40 min with all 8 dihedral elements applied to the cached
    maps, vector channels transformed. B = 64, AdamW with weight decay 1e-4, cosine decay.
    Four runs: gaps {1,2,4,7} or {1} × learning rate 5e-4 or 1e-3, about 3 700–3 900 steps
    each.
- **Fine-tuned ablation** (`flowhead-ft`). Starts from the selected checkpoint and trains on
  the fly with SEA-RAFT's update block, flow head and upsampler trainable (3.17 M
  parameters; encoders frozen). Learning rates: head 1e-4, backbone 2e-5. 2 001 steps in
  45 min.

**Model selection (val only).** The selected checkpoint is the cached head, gaps {1,2,4,7},
learning rate 5e-4, step 3000. It has the lowest cached-val median b (1.280,
`curves/head-g1247-lr5e-4/val_log.csv`). The alternatives were:

- 1e-3: 1.319;
- on-the-fly: 1.376 on the same windows;
- full val through the harness: `flowhead` 1.301 against `flowhead-onthefly` 1.423
  (`summary_all.csv`).

The gap-1 head was chosen the same way (lr 5e-4, 1.339). The test sets were predicted
once per frozen checkpoint and **never used for any choice**. The test predictions of
`flowhead` started before its full-val number was in; the choice rested on the cached val
only.

**Smoke tests.**
- On the fly, 16 train runs: loss 5.9 to 0.94 in 600 steps
  (`curves/smoke/train_log.csv`), and median b on those runs 1.92 to 1.51
  (`curves/smoke/val_log.csv`).
- Cached head on the same 16 runs, 1 184 windows: it overfits. Median b of the head is
  0.189 m/s against 1.727 for its own uncorrected flow average after 1 468 steps
  (`curves/head-smoke/val_log.csv`). The loss and pipeline work, and the head can memorise.

## Results (harness, every window, step 1)

`summary_all.csv` holds the 'all' slice of each `<set>/slices.csv`. Abbreviations in the
tables: med = median, p.p. = percentage points. b% is relative to ground speed.

| set | method | runs | med b (m/s) | p90 b | med b% | runs b ≤ 0.17 | b% ≤ 0.28 | b% ≤ 0.5 | drift% med |
|---|---|---|---|---|---|---|---|---|---|
| val | flowhead | 150 | 1.301 | 2.219 | 3.76 | 0.0067 | 0.0067 | 0.0133 | 3.73 |
| test-iid | flowhead | 300 | 1.366 | 2.343 | 4.21 | 0.0033 | 0.0033 | 0.0033 | 4.13 |
| test-sea | flowhead | 150 | 1.135 | 2.117 | 3.24 | 0.0067 | 0.0067 | 0.0333 | 3.25 |
| test-alt | flowhead | 150 | 2.642 | 3.569 | 7.07 | 0 | 0 | 0 | 7.02 |
| test-rate (all rates) | flowhead | 400 | 1.071 | 3.280 | 3.08 | 0.0575 | 0.025 | 0.0675 | 3.02 |
| e1-core | flowhead | 432 | 1.688 | 3.775 | 5.77 | 0 | 0 | 0 | 5.81 |

**test-rate by seaRate** (`test-rate/slices.csv`, slice `sea_rate`), median b:

| seaRate | 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|
| median b (m/s) | 0.461 | 0.726 | 1.379 | 2.924 |

The bias tracks the sea's own motion. On the frozen sea it is still 0.46 m/s: the head adds
a correction where none is needed.

**Slices** (`val/`, `test-iid/` slices.csv):
- The bias grows with altitude. test-iid: 1.12 at 0–50 m up to 1.91 at 130–200 m.
- It falls with wind speed. test-iid: 1.73 at ≤ 6 m/s down to 1.10 at 14–16 m/s.
- It is flat in sun elevation.

**Held-out conditions:**
- test-sea (stronger wind, longer swell) is not worse than val: 1.14.
- test-alt (220–400 m, GSD outside train's range) is twice val: 2.64.
- e1-core: 30 m and 60 m/s runs have catastrophic windows. p90 is 26 m/s at 30 m and
  21.6 m/s at 50–65 m/s. Displacements beyond the phase-correlation wrap of ±128 px lead to
  SEA-RAFT failures.

**Against E1** (`e1_compare_medians.csv`, derived from the harness's `e1-core/e1_compare.csv`):

| | flowhead | E1 SEA-RAFT (g = 1) | E1 UFM-Refine-980 | E1 shell, T = 3.2 s |
|---|---|---|---|---|
| e1-core median b (m/s) | 1.688 | 1.676 | 2.030 | 0.614 |

- flowhead beats E1's SEA-RAFT in half the runs (0.505).
- By altitude it is better above 60 m and worse at 30 m.

**Head against its own uncorrected flow average** (`raw_vs_head.csv`, derived from
`preds/*.diag.csv`). The uncorrected average is v_raw = Σ softmax(w)·u, with Δ removed.
Median b (head / raw):

| set | head | raw |
|---|---|---|
| val | 1.301 | 1.407 |
| test-iid | 1.366 | 1.492 |
| test-sea | 1.135 | 1.292 |
| test-alt | 2.642 | 2.773 |
| e1-core | 1.688 | 1.702 |
| test-rate | 1.071 | 1.079 |

The learned correction Δ is worth about 0.1 m/s. The rest of the gain over the plain
SEA-RAFT baseline comes from the head's weights, which choose pairs and cells.

**Underfitting, not overfitting** (`fit_check.csv`). The selected head on the first 150
training runs (cached windows) has median b 1.256 m/s. On val it has 1.280. The gap-1 head
has 1.266 and 1.339.

## Ablations

Selection is on val; the other sets were predicted once.

| method | val med b | test-iid med b | e1-core med b | test-rate med b | where |
|---|---|---|---|---|---|
| flowhead (gaps 1, 2, 4, 7; frozen) | 1.301 | 1.366 | 1.688 | 1.071 | `summary_all.csv` |
| flowhead-gap1 (gap 1 only, 2-frame pairs; frozen) | 1.368 | 1.486 | 1.802 | — | `summary_all.csv` |
| flowhead-onthefly (gaps 1, 2, 4, 7; frozen; 12 000 on-the-fly steps) | 1.423 | — | — | — | `summary_all.csv` |
| flowhead-ft (from flowhead; update block fine-tuned for 45 min) | 0.985 | 1.076 | 1.341 | 0.822 | `summary_all.csv` |

- **Multi-gap against gap 1 only.** The longer gaps help a little and consistently. The
  difference in median b is 0.07 on val, 0.12 on test-iid and 0.11 on e1-core. On the
  smaller scale that matters here, it is no help.
- **Frozen against fine-tuned.** On the cached-protocol val windows, the fine-tuned model
  went 1.143 → 1.064 → 1.018 at steps 500, 1000 and 1500, then 1.028 at 2001
  (`curves/ft-update/val_log.csv`). The starting point was 1.280. Best is step 1500.
  Through the harness:
  - Median b: val 0.985, test-iid 1.076, e1-core 1.341, test-rate 0.822.
  - Fraction of runs with b ≤ 0.17: 0.020, 0.013, 0.005, 0.065.
  - test-rate median b by seaRate 0 / 0.5 / 1 / 2: 0.365 / 0.621 / 1.007 / 2.182
    (`../flowhead-ft/test-rate/slices.csv`). The bias still roughly doubles with each
    doubling of the sea's rate: fine-tuning lowered it without removing the dependence.
  - The correction Δ is again about 0.1 m/s (`raw_vs_head.csv`: val 0.985 for the head
    against 1.103 for the uncorrected average). The gain comes from the flow itself
    changing.
  - This is the only change that moved the number by more than 0.1 m/s, and it was still
    falling when the 45 min ran out.
  - It is not a controlled comparison: the fine-tuned model also got 2 001 more steps on
    the fly. The frozen on-the-fly run, however, gained only about 0.01 m/s over its last
    3 000 steps (1.391 → 1.376 → 1.390).

## σ calibration (`calibration.csv`, `../flowhead-gap1/calibration.csv`)

σ is trained on the window error. On val and test-iid it describes the window error well:

- z-RMS 1.03 and 1.01;
- 66 % and 64 % of components within 1σ, 97.6 % and 96.8 % within 2σ;
- Spearman correlation of σ with |error| 0.27 and 0.30.

It does not describe the bias. The run-level z-RMS is 2.5–4.1: the run-mean error against
σ/√(windows/8), a rough measure because windows overlap. The windows of a run share their
error, and σ cannot see the wave-phase bias it shares.

It also does not follow the sea's rate. z-RMS is 0.46 on the frozen sea and 2.07 at
seaRate 2, with almost the same median σ (1.22 and 1.13 m/s). On test-alt it is 1.24, with
only 50 % of components within 1σ.

The gap-1 and fine-tuned models behave the same way (`../flowhead-gap1/calibration.csv`,
`../flowhead-ft/calibration.csv`):

- gap-1, val: z-RMS 1.07;
- fine-tuned, val: z-RMS 1.00, 68 % of components within 1σ;
- fine-tuned, seaRate 2: z-RMS 1.95.

## Findings worth keeping

- **Large displacements.** Pretrained SEA-RAFT-M, 4 iterations, on these tiles collapses to
  about zero flow beyond about 60 px of displacement. This was diagnosed on 16 train runs:
  per-pair errors equal the ground speed. The phase-correlation chain initialisation fixes
  most of it, except where the per-frame shift passes the ±128 px wrap of the 256 crop.
- **Numerical chaos on ambiguous water.** Same model, same inputs: SEA-RAFT batch sizes of
  32, 14 and 7 pairs change the estimate by up to 0.6 m/s on 25 of 74 windows of val run
  000 (8° sun, 51 px/frame), and by about 1e-3 m/s on a well-textured run. That is
  floating-point differences between kernels, amplified by the iterations. The cached
  prediction path matches the per-window forward on the well-textured run to 1e-3 m/s
  (`flowhead/check_cache.py`). This was checked by hand and is not stored in a CSV.

## Wall time and GPU memory

From the logs in `/workspace/data/e2/flowhead/logs/`. The GPU was shared with three other
agents for all but the last hour.

| step | wall time | GPU memory |
|---|---|---|
| on-the-fly main | 8 400 s for 12 000 steps | peak 5.9 GB, 2.3 GB while training |
| feature cache, train | 4 × 1 734–2 073 s, two processes at a time, about 46 min wall | |
| feature cache, val | 351 s | |
| cache on disk | | 26 GB at `/workspace/data/e2/flowhead/cache` |
| cached heads | 4 × 40 min at once | 1.7 GB each |
| fine-tuning | 2 825 s | 10.6 GB |
| predictions | about 1–3 s per run per process | |

- The first on-the-fly launch died after 30 min on "Too many open files": TileSet keeps one
  memory map per run, and the soft limit is 1 024. It is fixed (`raise_open_files`) and
  relaunched.
- During the last hour, the evaluation and fine-tuning used up to about 40 GB of the then
  otherwise idle GPU. That is above the 20 GB guideline.

## Limitations

- Training time was the binding constraint. The cached head saw 40 000 windows from 4 000
  runs on a fixed centred crop; the on-the-fly model saw 96 000 random windows once. Neither
  got close to the budget. The training-run check says more data or steps would not fix the
  frozen variant.
- The phase-correlation chain is an integer shift from the tiles. It wraps at ±128 px, and
  it occasionally locks onto a wave alias (low sun). Those windows fail in every variant;
  the head can only down-weight them.
- Only the update block was fine-tuned, for 45 min. Fine-tuning the encoders too
  (`--backbone all`) costs 6 windows/s at 14 GB with B = 4, and was not tried.
- Predictions use a centred 256 crop of the 384 tile, with no test-time augmentation.

## Reproduce (on the box)

```sh
cd /workspace/mare-realtime/experiments/e2_learned_velocity
export HF_HOME=/workspace/.hf_home PY=/workspace/venv/bin/python D=/workspace/data/e2/flowhead
# smoke tests
$PY -m flowhead.train --out $D/smoke --runs 16 --steps 600 --val-every 300 --warmup 100 --workers 4
$PY -m flowhead.cache --set train --runs 16 --start-step 1 --out $D/cache-smoke/train.0
$PY -m flowhead.train_head --cache $D/cache-smoke --out $D/head-smoke --val-on-train --minutes 6
# on-the-fly model (frozen backbone)
$PY -m flowhead.train --out $D/main --gaps 1,2,4,7 --backbone frozen --lr-head 5e-4 --steps 12000 \
    --max-hours 2.4 --val-every 1500 --val-step 8 --workers 5
# feature cache (train runs 0–3999 in four parts, val), then the heads
for i in 0 1 2 3; do $PY -m flowhead.cache --set train --runs 1000 --first $((i*1000)) --out $D/cache/train.$i; done
$PY -m flowhead.cache --set val --out $D/cache/val.0
for g in 1,2,4,7 1; do for lr in 5e-4 1e-3; do
  $PY -m flowhead.train_head --cache $D/cache --out $D/head-g${g//,/}-lr$lr --gaps $g --lr $lr --minutes 40 --steps 40000
done; done
# fine-tuned ablation
$PY -m flowhead.train --out $D/ft-update --gaps 1,2,4,7 --backbone update --init $D/head-g1247-lr5e-4/best.pt \
    --lr-head 1e-4 --lr-backbone 2e-5 --warmup 200 --steps 6000 --max-hours 0.75 --val-every 500 --val-step 8 --workers 5
# predictions + harness (→ /workspace/data/e2/preds/<method>/<set>.csv, /workspace/data/e2/eval/<method>/<set>/)
bash flowhead/eval_all.sh flowhead $D/head-g1247-lr5e-4/best.pt val test-iid test-sea test-alt test-rate e1-core
bash flowhead/eval_all.sh flowhead-gap1 $D/head-g1-lr5e-4/best.pt val test-iid e1-core
bash flowhead/eval_all.sh flowhead-onthefly $D/main/best.pt val
bash flowhead/eval_all.sh flowhead-ft $D/ft-update/best.pt val test-iid test-rate e1-core
# tables
$PY -m flowhead.calibration --preds /workspace/data/e2/preds/flowhead --out /workspace/data/e2/eval/flowhead/calibration.csv
$PY -m flowhead.fit_check --out $D/fit_check.csv
$PY -m flowhead.summarize --out <results dir>
```

Unit tests (local, CPU): `uv run --no-project --with numpy --with scipy --with pandas --with
opencv-python-headless --with pytest --with torch python -m pytest -q tests/test_flowhead.py`
checks the pair layout and the phase-correlation sign and chaining.
