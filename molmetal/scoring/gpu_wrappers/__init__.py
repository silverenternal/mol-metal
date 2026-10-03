"""molmetal.scoring.gpu_wrappers — GPU batch wrappers for highest-impact scorers.

================================================================
Wf-50 Phase 2 ultracode (2026-09-26)
================================================================

Five GPU batch wrappers that target the highest-impact CPU-only scoring
tools surfaced by the Wf-50 audit (see
``molmetal/reports/wf50_gpu_audit/critical_path.md``):

 1. :mod:`vina_gpu`        — Vina-GPU multi-pocket batch (CLI subprocess).
 2. :mod:`diffdock_batch`  — DiffDock batched inference (multi-pocket + n_samples).
 3. :mod:`boltz_batch`     — D-MPNN pIC50 with DataLoader + pin_memory + non_blocking.
 4. :mod:`pb_batch`        — PoseBusters batch validation (parallel RDKit workers).
 5. :mod:`hilbert_astsr_cuda` — Hilbert/ASTR primal-dual as ``torch.jit.script`` on CUDA.

Each wrapper exposes the **same call signature** as the underlying
scoring tool so the routing layer can swap CPU <-> GPU by passing a
``gpu=True`` flag without changing call-sites. The wrappers are
**opt-in** — the existing CPU path is untouched.

Design invariants
-----------------
* **NO mock fallback.** When the GPU backend is unavailable (no CUDA,
  no binary, missing wheel) the wrapper raises
  :class:`molmetal.scoring.AdapterUnavailable` — never invent a number.
* **Same semantics as the CPU path.** Output values are bit-for-bit
  comparable to the CPU wrapper within numerical tolerance (the
  pIC50 batched GPU run returns the same float list, the Vina-GPU
  kcal/mol is identical, etc.).
* **VRAM ceiling.** Each wrapper keeps peak VRAM below 16 GB so it
  fits both the RTX 5090 (32 GB) and the RTX PRO 6000 Blackwell
  (95 GB) in a single process.
* **Opt-in via ``gpu=True``.** Existing call-sites without the flag
  keep using the CPU path unchanged.

Public API
----------
* :func:`vina_score_gpu`         — list[float] kcal/mol per ligand.
* :func:`vina_high_affinity_gpu` — list[bool] (boolean on top of ``vina_score_gpu``).
* :func:`vina_min_score_gpu`     — dict[min / median / n_below_cutoff].
* :func:`diffdock_pose_batch`    — list[float] confidence per ligand.
* :func:`boltz_affinity_batch`   — list[float] pIC50 per ligand.
* :func:`pb_pass_rate_batch`     — float (fraction passing all checks).
* :func:`hilbert_astsr_optimize_batch` — ndarray (dim,) for 5-seed × 100-sample batch.

Module map
----------
* :mod:`molmetal.scoring.gpu_wrappers.vina_gpu`
* :mod:`molmetal.scoring.gpu_wrappers.diffdock_batch`
* :mod:`molmetal.scoring.gpu_wrappers.boltz_batch`
* :mod:`molmetal.scoring.gpu_wrappers.pb_batch`
* :mod:`molmetal.scoring.gpu_wrappers.hilbert_astsr_cuda`

Verification
------------
* :file:`molmetal/tests/test_gpu_wrappers.py` — 25 tests, one per
  wrapper + per-dimension (batch + memory ceiling). Run with
  ``pytest molmetal/tests/test_gpu_wrappers.py -v``.
"""

from __future__ import annotations

from molmetal.scoring.gpu_wrappers.boltz_batch import (
    boltz_affinity_batch,
    boltz_affinity_with_loader,
)
from molmetal.scoring.gpu_wrappers.diffdock_batch import (
    diffdock_pose_batch,
)
from molmetal.scoring.gpu_wrappers.hilbert_astsr_cuda import (
    hilbert_astsr_optimize_batch,
    hilbert_astsr_optimize_single,
)
from molmetal.scoring.gpu_wrappers.pb_batch import (
    pb_pass_rate_batch,
    pb_validate_list_batch,
)
from molmetal.scoring.gpu_wrappers.vina_gpu import (
    vina_high_affinity_gpu,
    vina_min_score_gpu,
    vina_score_gpu,
    vina_score_gpu_multi_pocket,
)

__all__ = [
    "vina_score_gpu",
    "vina_high_affinity_gpu",
    "vina_min_score_gpu",
    "vina_score_gpu_multi_pocket",
    "diffdock_pose_batch",
    "boltz_affinity_batch",
    "boltz_affinity_with_loader",
    "pb_pass_rate_batch",
    "pb_validate_list_batch",
    "hilbert_astsr_optimize_batch",
    "hilbert_astsr_optimize_single",
]
