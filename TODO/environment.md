# Environment

## 1. Overview

This project is a uv-managed Python 3.12 stack running ROCm 7.2 + `triton-rocm 3.8.0` against a single AMD RX 7800 XT (gfx1101, RDNA3). The kernel-lib, training driver, and baselines all live in one virtualenv created by `uv sync` from the committed `uv.lock`. There is no `pip install`, no Poetry, no conda — `uv` is the only package manager.

## 2. Environment manager: `uv`

`uv` (https://github.com/astral-sh/uv) replaces `pip`, `poetry`, and `pyenv` in this repo. It is used because:

- **Speed.** A cold `uv sync` against the locked tree is ~10x faster than `pip install` on this lockfile.
- **Lockfile as source of truth.** `uv.lock` pins every wheel's hash + URL; `uv sync --frozen` fails fast if the lock is stale. This is how we guarantee the same `torch-2.14.0+rocm7.2` wheel on every machine.
- **Workspaces.** A future `libs/kernels` + `apps/train` + `apps/eval` split can be expressed in `[tool.uv.workspace]` with one shared lock.
- **Non-PyPI wheels via index sources.** ROCm PyTorch and `triton-rocm` are not on PyPI. We register a custom simple-index in `pyproject.toml` and route the two packages through it (Section 5).

We never run `pip install`. We never edit `[project.dependencies]` by hand — we always go through `uv add`/`uv remove` so the lock stays atomic.

## 3. Python toolchain

- `.python-version` pins the interpreter to `3.12`. `uv` reads it before `pyproject.toml`.
- `pyproject.toml` declares `requires-python = "==3.12.*"` (exact major.minor, patch floats). This pins us to CPython 3.12.x and rejects 3.13 builds (ROCm wheels are cp312 only).
- `[tool.uv] managed = true` tells `uv` to own the virtualenv at `.venv/`.
- Local packages declared in `[tool.setuptools]`: `molmetal`, `molmetal.molmetal_lam`, `molmetal.data`, `molmetal.baselines`, `molmetal_lam`.

```bash
# Confirm pin
cat .python-version           # 3.12
grep requires-python pyproject.toml   # requires-python = "==3.12.*"
```

## 4. ROCm stack

Two ROCm-specific packages are pinned:

- `torch>=2.14,<2.15` — the ROCm 7.2 wheel from PyTorch's index (`torch-2.14.0+rocm7.2`). PyPI's CUDA build is **not** used.
- `triton-rocm==3.8.0` — exact pin. ROCm ships its own `triton` fork under the `triton-rocm` name; importing the PyPI `triton` package alongside it will silently break kernel launches. See Section 9.

GPU target: **AMD Radeon RX 7800 XT, gfx1101 (RDNA3)**. Architecture-specific notes:

- **Wave size 64.** RDNA3 hardware lanes are 32 threads, but Triton-on-ROCm 3.8.0 exposes a wave64 programming model. Use `num_warps ∈ {4, 8, 16}` as the tuning knob (one wave64 = 4 warps).
- **`waves_per_eu` is ignored on gfx1101.** That knob is CDNA-only (MI300/MI355). On RDNA, only `num_warps` + `num_stages` matter.
- **No MFMA at FP16 peak.** RDNA3 lacks the FP16 tensor-core throughput CDNA has, so `torch.compile(mode="max-autotune")` rarely beats hand-written Triton for hot kernels (EGNN edge aggregation, radius-neighbor gather, scatter-add, FP16→FP32 accumulators). Reserve `torch.compile` for the outer step + optimizer.
- **Peak FLOPs** (for reporting `achieved / peak`): ~35 TFLOPS FP32, ~70 TFLOPS FP16.

## 5. Custom index for ROCm wheels

PyTorch's ROCm wheels live at `https://download.pytorch.org/whl/rocm7.2`. We register it as an **explicit** index so it is consulted only for the packages we route to it:

```toml
[[tool.uv.index]]
name = "pytorch-rocm"
url  = "https://download.pytorch.org/whl/rocm7.2"
explicit = true

[tool.uv.sources]
torch       = [{ index = "pytorch-rocm" }]
triton-rocm = [{ index = "pytorch-rocm" }]
```

- `explicit = true` means `uv` does **not** fall back to this index when resolving unmentioned packages. `numpy`, `rdkit`, `scipy`, etc. still come from PyPI.
- Both `torch` and `triton-rocm` resolve from the same simple index — no need for a second `[[tool.uv.index]]` entry.
- Resolving `torch` via this index is what pulls in the `+rocm7.2` build (the wheel filename contains `+rocm7.2`). On PyPI proper the same version string resolves to a CUDA wheel that will not load on this box.

## 6. Setup commands

```bash
# One-time: install uv (Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Per machine: create .venv and install everything in uv.lock
uv sync

# Run any command inside the managed env (no manual activation)
uv run python scripts/_a1_sanity.py
uv run pytest -q
uv run python main.py --config configs/foo.yaml

# Activate the venv if you want a shell (optional)
source .venv/bin/activate
```

`uv run` is the canonical entry point — it boots the interpreter from `.venv` resolved from `.python-version`, so we never need to `source .venv/bin/activate`.

## 7. Adding a dependency

Always go through `uv add`. Do not hand-edit `[project].dependencies` (breaks lock atomicity, will be overwritten).

```bash
# Default: pull from PyPI (numpy, rdkit, …)
uv add scikit-learn

# Bump a version range
uv add "rdkit>=2024.3.2"

# Add a dev/test tool (PEP 735 [dependency-groups], not [project.optional-dependencies])
uv add --dev pytest pytest-cov

# Remove
uv remove pyyaml
```

What happens internally: `uv add foo` resolves `foo` (PyPI by default), edits `pyproject.toml`, regenerates `uv.lock`, and runs `uv sync` to install. Commit both files.

For ROCm-routed packages the command is the same — `uv` already knows `torch` and `triton-rocm` are pinned to the `pytorch-rocm` index from `pyproject.toml`, so you do **not** pass `--index` on the CLI. If you ever need an explicit one-off (e.g. installing into a scratch venv):

```bash
uv pip install --index https://download.pytorch.org/whl/rocm7.2 torch==2.14.0+rocm7.2 triton-rocm==3.8.0
```

## 8. GPU verification

Before attempting dataset downloads, inspect `/mnt/storage/data/molmetal`.
This machine already has `CrossDocked2020_cascadediff.zip`, the extracted
`crossdocked_pocket10.tar.gz`, and `split_by_name.pt`. The maintained loader
in `molmetal/data/crossdocked.py` reads this location by default and reports
100 test pairs plus 100,000 train pairs.

QVina is available as the statically linked vendored binary at
`molmetal/references/SoftMol/gated_mcts/utils/docking/qvina02`; the adapter
discovers it automatically or accepts `MOLMETAL_QVINA_BIN`. REINVENT4 is kept
isolated at `/mnt/storage/envs/reinvent4` because it pins CUDA Torch 2.12 and
cannot share the ROCm environment; invoke its CLI with
`/mnt/storage/envs/reinvent4/bin/reinvent --version`.

This host exposes two AMD devices (RX 7800 XT `gfx1101` and integrated
Radeon 780M `gfx1103`). For reproducible benchmarks, select the discrete
card explicitly with `HIP_VISIBLE_DEVICES=0`; ROCm uses this variable rather
than CUDA's equivalent. `torch.cuda.device_count()` should then report one
device. The current stack reports PyTorch `2.14.0+rocm7.2`, HIP `7.2.53211`,
and `triton` `3.8.0`.

Run after `uv sync` to confirm the ROCm stack is wired correctly:

```bash
# 1. torch can see the GPU
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.hip, torch.cuda.get_device_name(0))"
# Expect: True 7.2.4 ... Radeon RX 7800 XT ...

# 2. triton-rocm is the active Triton (version pinned to 3.8.0)
uv run python -c "import triton; print(triton.__version__)"
# Expect: 3.8.0

# 3. The driver sees the card
rocm-smi
# Expect: one GPU, gfx1101, VBIOS version

# 4. Optional: runtime capabilities
clinfo | grep -E 'Device Name|Version'
```

A quick kernel smoke test (allocates, launches, copies back):

```bash
uv run python -c "import torch, triton, triton.language as tl
@triton.jit
def add1(x): pass
x = torch.zeros(1, device='cuda')
print('alloc ok', x.device)"
```

If `torch.cuda.is_available()` returns `False`, `torch.version.hip` is `None`, or `clinfo` is missing — do not proceed; reinstall ROCm userland first.

## 9. Common issues

- **`triton` vs `triton-rocm` naming conflict.** ROCm ships its fork as `triton-rocm`. Never `uv add triton` — that installs the PyPI package, which collides at import time. The ROCm torch wheel already declares `Requires-Dist: triton-rocm~=3.8.0`, so it is installed transitively. The pin in `pyproject.toml` (`triton-rocm==3.8.0`) makes the version explicit.

- **`ROCTRACER_HOME` / `HSA_*` env vars unset.** Some ROCm tooling expects `ROCTRACER_HOME`, `HSA_RUNTIME_PATH`, or `HIP_PATH`. `uv run` does not source `/etc/profile.d/rocm.sh`. Either `source /etc/profile.d/rocm.sh` before `uv run`, or install ROCm to `/opt/rocm` and symlink. Symptom: `libhsa-runtime64.so` not found.

- **`HIP_VISIBLE_DEVICES` ignored.** On multi-GPU boxes, restrict with `HIP_VISIBLE_DEVICES=0 uv run python …`. ROCm does **not** honor `CUDA_VISIBLE_DEVICES`.

- **MIOpen cache permissions / corruption.** MIOpen writes its compiled-kernel cache under `~/.cache/miopen` (or `$MIOPEN_CACHE_DIR`). If kernels hang or behave nondeterministically after a driver upgrade, nuke the cache: `rm -rf ~/.cache/miopen`. Set `MIOPEN_DISABLE_CACHE=1` to fall back to JIT (slower startup).

- **`triton.autotune` thrashes.** Triton 3.8 caches per-kernel autotune results under `$TRITON_CACHE_DIR` (default `~/.triton/cache`). Set `TRITON_CACHE_DIR=/path/with/space` to a project-local path and commit a `.gitignore` line — the cache is machine-specific.

- **Lock drift.** If a teammate's `uv.lock` does not match yours, run `uv lock --check` in CI to fail fast. Resolve by `uv lock` then commit.

## 10. CI / reproducibility

Optional research data and isolated tools discovered on this workstation:

- CrossDocked2020 is staged under `/mnt/storage/data/molmetal/crossdocked`
  (8.3 GiB, including `split_by_name.pt` and pocket10 archive). Running
  `uv run python molmetal/scripts/stage_crossdocked100.py` generates the
  repository manifest with 100 test pockets.
- REINVENT4 4.8.24 is installed at `/mnt/storage/envs/reinvent4/bin/reinvent`
  in a separate CUDA/PyTorch environment. Its standard CLI is TOML/RL based,
  not the JSON-lines protocol used by the MolMetal adapter. The adapter now
  rejects this mismatch explicitly (`cli_protocol_mismatch`); use the
  isolated `reinvent4_jsonl_worker.py` bridge for protocol smoke tests and
  reserve the standard CLI for a dedicated native REINVENT4 workflow.

Pytest is configured in `pyproject.toml` with `testpaths = ["tests", "molmetal"]`
and excludes `molmetal/references/`. The latter contains vendored projects
with independent pytest plugins and optional dependencies; recursively
collecting it caused collection failures (including duplicate `--device`
options). The supported project-suite command is therefore:

```bash
HIP_VISIBLE_DEVICES=0 uv run pytest -q
```

- **`uv.lock` is committed** at repo root. It pins wheel hashes and URLs for every dep.
- **`.python-version` is committed** at repo root (`3.12`).
- **`.venv/` is gitignored**. `uv sync` re-creates it deterministically from the lock.
- **CI command:** `uv sync --frozen && uv run pytest -q`. `--frozen` refuses to update the lock — if the lock is stale relative to `pyproject.toml`, CI fails with a clear diff instead of silently drifting.
- **Pre-commit hook:** `uv lock --check` on every commit, catches accidental manual edits to `pyproject.toml`.
- **ROCm wheel reproducibility:** the ROCm 7.2 simple index keeps historical wheels available; the version string `torch-2.14.0+rocm7.2` resolves deterministically as long as the index is reachable. If the upstream index is ever retired, mirror it under `vendor/` and point `[[tool.uv.index]].url` at the mirror.


## 11. Genuine AiZynth CPU / ROCm policy (verified 2026-09-13)

AiZynthFinder 4.4.1 runs in the separately locked `environments/aizynth`
Python 3.12 runtime because it needs RDKit 2023.9.6 / NumPy 1.26.4. Use
`bash environments/aizynth/run.sh python ...` for its exact cached environment;
the root ROCm `.venv` is not added to the chemistry process import path.

The working assets are explicitly **official legacy v3 USPTO** weights and
46,695 matching templates, plus the official 17,422,831-entry ZINC stock.
They are staged under `/mnt/storage/data/molmetal/aizynth_public`, with
source snapshots, SHA256, range-download verification and original-model
S3 MD5 evidence. The current upstream Zenodo policy remains unreachable;
do not mix its model or template ordering with the legacy pair.

- `molmetal/configs/aizynth_legacy_v3_cpu.yml` uses the preserved pretrained
  weights converted to ONNX. Independent NumPy evaluation of the original
  Keras inference equations verifies 28 inputs, max probability error 3.28e-6.
- `molmetal/configs/aizynth_legacy_v3_rocm.yml` replaces only the Dense policy
  predictor with a persistent worker in the root ROCm environment. Actual
  parameters and outputs are on RX 7800 XT `gfx1101`, `cuda:0`. Its 28-input
  ONNX comparison has max probability error 2.32e-6 and 100% top-1/10/50
  set agreement. RDKit, templates, stock, MCTS and route analysis remain CPU.
- Genuine CPU and GPU learned searches solve the three declared smoke targets
  with the same final stock routes. Three actual Lambda CuAAC products also
  obtain routes, depths 6/5/5. Small numerical changes can alter intermediate
  MCTS trajectories; this is not a throughput or synthesis-rate benchmark.

Root callers use `build_synthesis_gate("aizynthfinder_isolated", config_path)`
or the Lambda runner's corresponding `synthesis_oracle` setting. Candidate
checks are batched; missing assets and SMARTS fallback never count as learned
success. A total process deadline reclaims launcher and GPU descendants.
Generation denominators include synthesis-rejected products, with retained,
selected and physically evaluated counts reported separately.

```bash
HIP_VISIBLE_DEVICES=0 bash environments/aizynth/run.sh python molmetal/scripts/aizynth_learned_smoke.py \
  --config molmetal/configs/aizynth_legacy_v3_rocm.yml \
  --protocol-label 'official legacy v3; ROCm predictor; CPU chemistry and MCTS' \
  --output /tmp/aizynth_rocm_smoke.json
HIP_VISIBLE_DEVICES=0 uv run --no-sync python molmetal/scripts/verify_aizynth_torch_policy.py
```

See [measured protocols and provenance](../molmetal/reports/aizynth_real_backend_20260913/report.md)
and [isolated runtime instructions](../environments/aizynth/README.md).
