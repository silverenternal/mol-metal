# Official REINVENT learned generation and NLL on AMD (2026-09-13)

## Goal

Locate a real pretrained REINVENT prior, run the official CLI for five molecules, and verify learned model execution independently from the project's existing RDKit proxy bridge. Keep the CUDA installation and core ROCm project environment intact while enabling actual AMD inference.

## Outcome

**PASS for official learned generation and prior likelihood on CPU and AMD GPU.** Both produce five valid molecules and five finite positive sequence negative log likelihoods (NLLs). This is the official ChEMBL 25 pretrained Reinvent RNN: embedding 34×256, three-layer LSTM with 512 hidden units, 34-token output layer, **5,805,602 parameters**. No replacement model, random prior or RDKit scoring proxy was used.

| Path | Runtime | Actual neural execution | Valid sampled molecules |
|---|---|---|---:|
| Original isolated installation | REINVENT 4.8.24; torch 2.12.0+cu126 | CPU: 58 RNN + 58 LSTM sampling calls; 1 each for NLL | 5/5 |
| Separate ROCm project | REINVENT 4.8.24; torch 2.14.0+rocm7.2; torchvision 0.29.0+rocm7.2 | gfx1101: 80 RNN + 80 LSTM sampling calls; 1 each for NLL, parameters and input tensors all `cuda:0` | 5/5 |

The AMD run reports 98 MiB peak reserved GPU memory. All runs use seed 42. CPU and GPU RNG streams generate different molecules; identity across backends is not expected. On the same five fixed SMILES and untouched checkpoint, the largest CPU/GPU NLL difference is **1.0491×10⁻⁵** (see `report.json`). This supports this small inference compatibility check, not exhaustive numerical parity or a performance claim.

## Model provenance

Local asset: `/mnt/storage/models/reinvent4/reinvent_v4.4.22.prior`, **23,226,277 bytes**. The current Zenodo endpoint returned 504/timeouts, so the original official GitHub release asset was used:

- Repository: `https://github.com/MolecularAI/REINVENT4`
- Prior tag: `v4.4.22`; commit `eb61443a1a93a8cbd06510548b25cc77cfd9d3a6`
- Git blob: `0dee328238b3d413b5a34c80e3ddab49e4e0af28` (verified from downloaded bytes)
- SHA256: `b6513ec6dbc54c87ea45cdbf9b4aaefadd7652548b74175366b27f12ec5732fe`
- Runtime source: `ee0d56f4a07472bbb622cd0858184d06f11bff5d`, REINVENT 4.8.24

The legacy prior's internal metadata xxhash check logs an error in the current runtime. The check serializes runtime Python/NumPy objects and is distinct from the downloaded file hash; we have not established the exact reason for the mismatch. **The original file exactly matches the official Git blob and is unchanged.** The warning is retained in both official CLI logs; no metadata is rewritten or validation disabled.

## Environment changes and limits

The new managed uv project lives at `/mnt/storage/env-projects/reinvent4-rocm`, with its own `.venv` and lock. It copies the official source and changes only its dependency index from CUDA 12.6 to ROCm 7.2; all **576 Python source files** match the original. The root environment manifest explicitly overrides upstream's `torch==2.12.0` requirement with `torch==2.14.0` and pins matching torchvision. Pillow, TensorBoard, Polars and Pygments reuse versions from the working CUDA environment. SciPy is added because upstream runtime imports it but omits its dependency declaration. The first missing-SciPy failure is preserved in `rocm_traced/report.json`.

Snapshots of both project manifests, `uv.lock`, the index-only patch and installation logs are under `environment/`. The original `/mnt/storage/envs/reinvent4` installation, core project environment, frozen runtime modules and proxy JSONL bridge are untouched.

**Scoring here means learned SMILES sequence NLL under the prior.** It is not activity, synthesis, docking, ADMET or calibrated task desirability. NLL depends on sequence length and SMILES representation. This experiment does not turn the existing project proxy bridge into a learned multiproperty scoring plugin.

## Reproduce

Existing isolated ROCm environment:

```bash
uv sync --frozen --project /mnt/storage/env-projects/reinvent4-rocm
HIP_VISIBLE_DEVICES=0 uv run --frozen --project /mnt/storage/env-projects/reinvent4-rocm \
  python /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/verify_reinvent_learned_inference.py \
  --device cuda:0 \
  --out-dir /home/hugo/codes/try_triton_on_rocm/molmetal/reports/reinvent4_learned_smoke/rocm_repeat
```

The observer calls the official `reinvent.Reinvent.main_script` with `-d cuda:0 -s 42`, then invokes the official model adapter's `likelihood_smiles` for five fixed molecules. It registers forward observers on the actual RNN and LSTM modules, checks the GPU architecture and prior blob, and refuses CPU fallback. The resulting folder includes its TOML/JSON configuration, generated CSV, exact CLI arguments/log, seed, scores and forward-device evidence.

Direct official CLI (without observers):

```bash
HIP_VISIBLE_DEVICES=0 uv run --frozen --project /mnt/storage/env-projects/reinvent4-rocm \
  reinvent -d cuda:0 -s 42 \
  /home/hugo/codes/try_triton_on_rocm/molmetal/reports/reinvent4_learned_smoke/rocm_scipy_traced/sampling.toml
```

CPU reference:

```bash
uv run --no-project --python /mnt/storage/envs/reinvent4/bin/python \
  python molmetal/scripts/verify_reinvent_learned_inference.py --device cpu \
  --out-dir molmetal/reports/reinvent4_learned_smoke/cpu_repeat
```

For a clean rebuild, copy the runtime source at its recorded commit into the isolated project's `upstream/`, apply `environment/upstream_rocm_index.patch`, restore `environment/pyproject.toml` and `environment/uv.lock`, then run the frozen sync command. The manifest's local editable source path assumes the recorded `/mnt/storage/env-projects/reinvent4-rocm` location. Prior downloads should use the pinned commit URL and verify the recorded Git blob/SHA256 before loading.
