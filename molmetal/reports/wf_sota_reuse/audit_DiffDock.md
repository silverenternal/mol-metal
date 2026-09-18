# Audit: DiffDock (Corso 2023, ICLR; v=L diffdock-L Feb 2024)

Path: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/DiffDock/`
Our existing wrapper: `molmetal/adapters/diffdock_subprocess_adapter.py`
(DiffDockAdapter — currently subprocess-only; GPU-blocked per wf_gpu_auto_recover)

## (1) What we already have hand-rolled that is redundant

- We do NOT re-implement DiffDock — we wrap it. Our wrapper handles
  subprocess invocation, PDB write-out, and pose-id parsing.
- The actual docking diffusion (E(3)-equivariant SO(2) twist sampling,
  score model `models/aa_model.py` + confidence model
  `models/aa_model.py:Confidence`) lives in DiffDock's repo and is ~4k LOC
  of CUDA-targeted PyTorch.

## (2) What is actually different / better in our hand-rolled code

- **Subprocess-only invocation**: we shell out to `inference.py --protein_path
  … --ligand_description …`. This adds ~30 s of process startup per pose
  and is GPU-unfriendly on our RX 7800 XT (the model uses `torch.scatter`
  + custom CUDA kernels in `models/tensor_layers.py`).
- **No Python API wrapping**: DiffDock's `inference.py:main(args)` parses
  YAML + argparse, downloads model weights from Zenodo if missing, sets up
  PyG datasets, runs sampling. Importing `from models.aa_model import
  ScoreModel` requires us to fabricate a fake `InferenceDataset` and
  reproduce the YAML-driven config dict — non-trivial.
- **DiffDock-L confidence model** is mandatory for ranking poses; our
  wrapper has no confidence channel.

## (3) Concrete 3-line patch plan (file + lines + import)

```
# molmetal/adapters/diffdock_subprocess_adapter.py:1-40 (subprocess entry)
# OPTION A: keep subprocess (cheap, GPU-tolerant), add confidence model:
#   just pass --confidence_model_dir /path/to/conf --confidence_ckpt best_model.pt
# OPTION B (preferred for in-process reuse):
import sys; sys.path.insert(0, "/home/hugo/codes/try_triton_on_rocm/molmetal/references/DiffDock")
from utils.sampling import sampling  # 20-step reverse-diffusion
from utils.diffusion_utils import t_to_sigma, get_t_schedule
# then build a thin DiffDockAPIAdapter that calls `sampling(...)` directly,
# bypassing YAML + argparse.
```
Caveat: DiffDock's repo installs `torch_geometric==2.2.0` + `torch_scatter`
+ `torch_cluster`, all CUDA-only builds; ROCm wheel missing. Direct in-process
import would force us to vendor these builds.

## Verdict

**PARTIAL REUSE — keep our subprocess adapter for now.** The honest blocker
is that DiffDock is hard-coded for CUDA 11.6 / PyG 2.2 with PyTorch
`torch.scatter` calls that aren't on ROCm gfx1101. Our subprocess wrapper
already integrates DiffDock-L confidence model via `--confidence_model_dir`;
when GPU recovers AND we have CUDA compatibility, we can write a thin
`diffdock_api_adapter.py` to call `utils.sampling.sampling()` directly
and skip the YAML parsing. **Do not modify** the references/ clone.