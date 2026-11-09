# E1 — wave-motion bias in image-based velocity

How large is the bias in image-based velocity caused by the sea's own motion (wave phase
propagation, foam drift)? Two estimators, scored against the true camera velocity on
rendered nadir/oblique onboard frames:

- **(a) dense optical flow**, pretrained and not fine-tuned (SEA-RAFT-M, UFM), converted to
  metric velocity on the plane up = 0 with the TRUE attitude and height (`flow_bias.py`);
- **(b) a classical dispersion-shell fit**: a camera-fixed rectified tile, 3-D FFT over a window
  of T seconds, energy on ω' = ±√(g|k|) − k·V maximised over V (`shell_fit.py`).

A frozen-sea control (`render.seaRate 0`) checks the flow-to-metric geometry and the static
spectral pipeline when the water does not move (midday, 60–120 m, 30–60 m/s). It cannot test the
dispersive fit, whose floor rests on synthetic seas and the wave-slope label layer. Results and their reading: [results/RESULTS.md](results/RESULTS.md).

## Files

| File | What it does |
|---|---|
| `jobs/core.json`, `jobs/frozen.json`, `jobs/oblique.json` | the render grids (432 / 24 / 36 runs × 121 frames) |
| `jobs/decomp.json` | 32 runs that also write the `normals` and `foam` label modes, for the per-layer decomposition |
| `jobs/pilot.json`, `jobs/pilot-frozen.json` | 6 + 6 pilot runs used to build and check the pipelines |
| `e1geom.py` | frames.csv poses, back-projection onto the sea plane (numpy + torch) |
| `flow_bias.py` | (a): flow models → apparent pattern velocity a per frame pair; `synth` sub-command for the exact-warp sign test |
| `rectify.py`, `shell_fit.py` | (b): camera-fixed ENU tile, 3-D FFT, dispersive and static shell fits |
| `synth.py` | synthetic linear sea for validating the shell fit |
| `analyze.py` | per-run bias, `summary_by_slice.csv`, `phase_speed.csv`, `audit_checks.csv`, plots |
| `flow_summary.py`, `analyze_shell.py`, `diag_shell.py`, `verify_pilot.py` | pilot-stage checks and diagnostics |
| `tests/` | `test_geom.py` (back-projection, exact-flow bound), `test_shell.py` (shell geometry, synthetic recovery) |

## Environment (GPU box)

- Renderer: this repo (`pnpm install`, Playwright Chromium).
- Python: `/workspace/venv` (uv, Python 3.12) with a CUDA 12.8 PyTorch build (the RTX PRO 6000
  Blackwell is sm_120):

```sh
export HF_HOME=/workspace/.hf_home
uv venv /workspace/venv --python 3.12 && . /workspace/venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
mkdir -p /workspace/models && cd /workspace/models
git clone --recursive https://github.com/UniFlowMatch/UFM.git        # used at ff78d6c
git clone https://github.com/princeton-vl/SEA-RAFT.git                # used at 9137517
uv pip install -e UFM/UniCeption -e UFM
uv pip install scipy opencv-python h5py tqdm einops huggingface_hub pandas matplotlib pytest pillow numpy
```

Weights download on first use into `$HF_HOME`: `infinity1096/UFM-Base`,
`infinity1096/UFM-Refine-980`, `MemorySlices/Tartan-C-T-TSKH-spring540x960-M` (SEA-RAFT-M).
UFM's DINOv2 and SEA-RAFT's ResNet34 come through torch hub into `~/.cache/torch`.

## Reproduce

```sh
cd /workspace/mare-realtime

# 1. Render (measured: 1.5 min frozen, 2.5 min oblique, 20 min core at 32 pages)
for s in frozen oblique core decomp; do
  node scripts/generate.mjs experiments/e1_wave_bias/jobs/$s.json \
    --out /workspace/data/e1/$s --port <free port> --concurrency 32
done
# runs land in /workspace/data/e1/<set>/e1-<set>/<run-id>/ ; index.csv there lists status per run

cd experiments/e1_wave_bias
export HF_HOME=/workspace/.hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Flow models (≈ 10 s per run for all four models on one GPU)
for s in frozen oblique core; do
  /workspace/venv/bin/python flow_bias.py run --data /workspace/data/e1/$s/e1-$s \
    --models searaft,searaft_half,ufm,ufmrefine980 --workers 8 --out /workspace/data/e1/flow/full/flow-$s.csv
done

# 2b. Decomposition: flow on each image layer of the decomp set
for m in shaded normals; do
  /workspace/venv/bin/python flow_bias.py run --data /workspace/data/e1/decomp/e1-decomp --mode $m \
    --models searaft,ufmrefine980 --out /workspace/data/e1/flow/full/flow-decomp-$m.csv
done
/workspace/venv/bin/python flow_bias.py run --data /workspace/data/e1/decomp/e1-decomp --mode foam --mask-min-lum 40 \
  --models searaft,ufmrefine980 --out /workspace/data/e1/flow/full/flow-decomp-foam.csv

# 3. Shell fit (12 processes share the GPU)
for s in frozen oblique core; do
  /workspace/venv/bin/python shell_fit.py /workspace/data/e1/$s/e1-$s \
    --out /workspace/data/e1/shell/full/shell-$s.csv --jobs 12
done

for m in shaded normals foam; do
  /workspace/venv/bin/python shell_fit.py /workspace/data/e1/decomp/e1-decomp --mode $m \
    --out /workspace/data/e1/shell/full/shell-decomp-$m.csv --jobs 6
done

# 3b. Audit: the same fit with other k weights (default 'whiten'); script /workspace/data/e1/shell_weight.sh
#     writes shell-{frozen,core,decomp-normals}-energy.csv and shell-frozen-sqrt.csv to /workspace/data/e1/shell/weight/
/workspace/venv/bin/python shell_fit.py /workspace/data/e1/frozen/e1-frozen --weight energy \
  --out /workspace/data/e1/shell/weight/shell-frozen-energy.csv --jobs 16

# 4. Copy flow-*.csv and shell-*.csv into results/, then (anywhere with pandas + matplotlib)
python analyze.py          # writes results/per_run.csv, summary_by_slice.csv, phase_speed.csv,
                           # altitude_exponent.csv, decomp_by_layer.csv, audit_checks.csv and
                           # results/fig_*.png; shell-<set>-<weight>.csv files appear as
                           # methods shell:<model>/<weight>
```

Checks: `pytest tests/` (on the box: geometry and shell tests), and
`python flow_bias.py synth --run /workspace/data/e1/pilot-frozen --models ... --out ...`
(exact-warp sign test: a frozen frame warped by a known camera move plus a known sea motion).

Do not run the flow pass while a render is running if VRAM is short: the core render once
failed every run with "Ocean engine: shader error" while the GPU was full (it rendered
cleanly when rerun on a free GPU).

Directions: sea directions are compass degrees the waves travel *toward* (src/ocean/scene.js).
`flow_bias.py` and `shell_fit.py` read `windDirection` and `swellDirection` from each run.json;
`*_along_wave` is along the wind sea (60° in every E1 preset) and `*_along_swell` along the swell
(75° moderate-swell, 140° cross-swell; NaN without swell). CSVs written before this change lack
the swell columns; `analyze.py` fills them from the preset directions.
