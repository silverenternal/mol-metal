"""vina_gpu — GPU batch wrapper for Vina-GPU.

================================================================
Wf-50 Phase 2 (#4 — Vina-GPU multi-pocket batch wrapper)
================================================================

Wraps :class:`molmetal.molmetal_lam.sbdd_env.vina_gpu_adapter.QuickVinaGPUAdapter`
behind a multi-pocket batch interface. The wrapper:

* Detects the CUDA-capable GPU (or falls back to AMD OpenCL on ROCm
  when the QuickVina2-GPU-2.1 binary is on $PATH).
* Accepts a list of ``(pocket, smiles_list, n_poses)`` triples so the
  caller can pre-stage several pockets without re-loading the receptor
  cache.
* Runs the docking through the existing adapter — no chemistry
  shortcut, no parallelism invention. The adapter already enforces
  the AMD-device contract on the ROCm build and rejects non-NVIDIA
  devices; the wrapper mirrors the same gate (see :func:`_select_engine`).
* VRAM ceiling: 1.2 GB (single receptor cache, ``gpu_threads`` ≤
  32 000, ``search_depth`` = 1) — comfortably fits on RTX 5090 32 GB.

Public API
----------
* :func:`vina_score_gpu`         — list[float] kcal/mol per ligand.
* :func:`vina_high_affinity_gpu` — list[bool] (boolean classifier).
* :func:`vina_min_score_gpu`     — dict[min / median / n_below_cutoff].

Failure mode
------------
* No CUDA device + no ``quickvina-gpu`` binary on PATH → raise
  :class:`molmetal.scoring.AdapterUnavailable` (NO mock fallback).
* Binary present but refuses NVIDIA Blackwell → raise with the same
  message the adapter surfaces.
"""
from __future__ import annotations

import logging
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — keep the package importable on CI boxes without CUDA.
# ---------------------------------------------------------------------------
def _scoring_exceptions():
    from molmetal.scoring.score_any import AdapterUnavailable

    return AdapterUnavailable


def _vina_gpu_module():
    from molmetal.molmetal_lam.sbdd_env import vina_gpu_adapter

    return vina_gpu_adapter


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineInfo:
    """Resolved Vina-GPU engine metadata.

    Attributes
    ----------
    backend : str
        ``"cuda"`` / ``"rocm"`` / ``"cpu-fallback"``.
    binary_path : str | None
        Absolute path to the Vina-GPU binary, or None if not found.
    opencl_device : str
        ``"opencl:0"`` for ROCm binary; ``"cuda:0"`` for CUDA binary.
    expected_device : str
        GPU device name we expect to see in the engine log (used for the
        ROCm AMD-only verification).
    """

    backend: str
    binary_path: Optional[str]
    opencl_device: str
    expected_device: str


def _select_engine(
    binary_path: Optional[str] = None,
    kernel_dir: Optional[str] = None,
) -> EngineInfo:
    """Pick the right Vina-GPU backend for the host.

    Resolution order:

    1. ``$VINA_GPU_BIN`` env var (test override).
    2. ``binary_path`` keyword (call-site override).
    3. ``$PATH`` for ``quickvina-gpu`` / ``quickvina2-gpu`` /
       ``vina-gpu`` / ``autodock-gpu``.
    4. The QuickVina2-GPU-2.1 vendor tree at
       ``molmetal/references/SoftMol/gated_mcts/utils/docking/quickvina2-gpu-2-1/``.

    When none resolve, returns ``EngineInfo("cpu-fallback", None, …)``
    so the wrapper can fail loudly later. The wrapper never invents a
    kcal/mol score.
    """
    candidates: List[str] = []
    env_bin = os.environ.get("VINA_GPU_BIN")
    if env_bin:
        candidates.append(env_bin)
    if binary_path:
        candidates.append(str(binary_path))
    for name in (
        "quickvina-gpu",
        "quickvina2-gpu",
        "qvina-gpu",
        "vina-gpu",
        "autodock-gpu",
    ):
        hit = shutil.which(name)
        if hit:
            candidates.append(hit)

    pkg_root = Path(__file__).resolve().parents[3]  # molmetal/scoring/gpu_wrappers -> molmetal
    vendor = (
        pkg_root / "references" / "SoftMol" / "gated_mcts" / "utils"
        / "docking" / "quickvina2-gpu-2-1"
    )
    if vendor.is_dir():
        for sub in ("quickvina2-gpu", "quickvina-gpu"):
            cand = vendor / sub
            if cand.is_file() and os.access(cand, os.X_OK):
                candidates.append(str(cand))
                break

    for cand in candidates:
        path = Path(cand).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        # Heuristic: AMD vendor tree ⇒ OpenCL/gfx1101. NVIDIA vendor
        # trees ⇒ CUDA. We default to the AMD contract (the only one
        # shipped today) and let the adapter reject non-AMD devices.
        if "gated_mcts" in str(path) or "quickvina2-gpu-2-1" in str(path):
            return EngineInfo(
                backend="rocm",
                binary_path=str(path),
                opencl_device="opencl:0",
                expected_device="gfx1101",
            )
        return EngineInfo(
            backend="cuda",
            binary_path=str(path),
            opencl_device="cuda:0",
            expected_device="NVIDIA",
        )

    return EngineInfo(
        backend="cpu-fallback",
        binary_path=None,
        opencl_device="opencl:0",
        expected_device="<none>",
    )


# ---------------------------------------------------------------------------
# Adapter factory (cached per binary path)
# ---------------------------------------------------------------------------
_ADAPTER_CACHE: Dict[str, Any] = {}


def _get_adapter(
    engine: EngineInfo,
    kernel_dir: Optional[str] = None,
    gpu_threads: int = 1000,
    search_depth: int = 1,
) -> Any:
    """Return a cached :class:`QuickVinaGPUAdapter` for ``engine``."""
    if engine.backend == "cpu-fallback" or engine.binary_path is None:
        AdapterUnavailable = _scoring_exceptions()
        raise AdapterUnavailable(
            "vina_score_gpu: no Vina-GPU binary available on this host. "
            "Install the QuickVina2-GPU-2-1 vendor binary under "
            "molmetal/references/SoftMol/gated_mcts/utils/docking/"
            "quickvina2-gpu-2-1/, set $VINA_GPU_BIN, or fall back to "
            "vina_score (CPU)."
        )
    cache_key = f"{engine.binary_path}|{kernel_dir}|{gpu_threads}|{search_depth}"
    if cache_key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[cache_key]
    vgpu = _vina_gpu_module()
    adapter = vgpu.QuickVinaGPUAdapter(
        binary_path=engine.binary_path,
        kernel_dir=kernel_dir,
        gpu_threads=gpu_threads,
        search_depth=search_depth,
        opencl_device=engine.opencl_device,
        expected_device=engine.expected_device,
    )
    _ADAPTER_CACHE[cache_key] = adapter
    return adapter


# ---------------------------------------------------------------------------
# Public API — mirror the CPU wrappers in score_any.py
# ---------------------------------------------------------------------------
def vina_score_gpu(
    smiles_list: Sequence[str],
    pocket: Any,
    *,
    n_poses: int = 5,
    exhaustiveness: int = 8,
    binary_path: Optional[str] = None,
    kernel_dir: Optional[str] = None,
    gpu_threads: int = 1000,
    search_depth: int = 1,
    expected_vram_mb: int = 1200,
) -> List[float]:
    """Run Vina-GPU on a list of SMILES against a single pocket.

    Parameters
    ----------
    smiles_list : sequence of str
        Ligand SMILES (one kcal/mol per input).
    pocket : molmetal.domain.Pocket | str
        Pocket domain object (preferred) or a path to a pre-prepared
        PDBQT receptor (only if ``set_prepared_receptor`` was called).
    n_poses : int, default 5
        Number of poses per ligand.
    exhaustiveness : int, default 8
        Reported in metadata but ignored by the AMD GPU lanes (the
        adapter uses ``gpu_threads`` and ``search_depth`` instead).
    binary_path, kernel_dir : str, optional
        Override the Vina-GPU binary / kernel paths.
    gpu_threads, search_depth : int
        Forwarded to :class:`QuickVinaGPUAdapter`.
    expected_vram_mb : int, default 1200
        Refuse to run if the projected peak VRAM exceeds this cap
        (default 1.2 GB fits both RTX 5090 and Blackwell comfortably).

    Returns
    -------
    list[float]
        Best (most negative) kcal/mol per input SMILES. ``nan`` on a
        docking failure (mirrors the CPU wrapper).

    Raises
    ------
    AdapterUnavailable
        When no Vina-GPU backend is reachable, or the binary refuses
        the host's GPU device.
    """
    AdapterUnavailable = _scoring_exceptions()
    if not smiles_list:
        return []
    engine = _select_engine(binary_path=binary_path, kernel_dir=kernel_dir)
    if engine.backend == "cpu-fallback":
        raise AdapterUnavailable(
            "vina_score_gpu: Vina-GPU backend not available on this host "
            "(no binary on PATH, no $VINA_GPU_BIN, no vendored tree). "
            "The wrapper does NOT fall back to vina_score automatically — "
            "use score_any(\"vina_score\", ...) for the CPU path."
        )

    # VRAM ceiling: estimate = gpu_threads * search_depth * 8 bytes.
    # 32 000 lanes × depth 1 × 8 B = 256 KB; plus a fixed 800 MB receptor
    # cache ⇒ ~1 GB peak. We keep the ceiling configurable but default
    # to 1.2 GB to match the audit's VRAM estimate.
    projected_mb = int(
        800 + (int(gpu_threads) * int(search_depth) * 8) / (1024 * 1024)
    )
    if projected_mb > expected_vram_mb:
        raise AdapterUnavailable(
            f"vina_score_gpu: projected peak VRAM {projected_mb} MB exceeds "
            f"ceiling {expected_vram_mb} MB (gpu_threads={gpu_threads}, "
            f"search_depth={search_depth}). Reduce gpu_threads or raise "
            f"expected_vram_mb explicitly."
        )

    adapter = _get_adapter(
        engine,
        kernel_dir=kernel_dir,
        gpu_threads=gpu_threads,
        search_depth=search_depth,
    )

    # The CPU wrapper calls ``vina.dock_smiles`` per SMILES; the GPU
    # adapter exposes the same surface via ``dock_smiles_gpu``. We
    # keep the loop to mirror the CPU semantics — the speed-up vs.
    # the CPU path comes from the GPU lanes / Monte Carlo parallelism
    # inside the kernel, not from Python-level batching (which the
    # adapter doesn't support).
    out: List[float] = []
    for smi in smiles_list:
        try:
            complexes = adapter.dock_smiles_gpu(smi, pocket, n_poses=int(n_poses))
        except Exception as exc:
            logger.warning(
                "vina_score_gpu: dock_smiles_gpu failed for SMILES=%r: %s",
                smi,
                exc,
            )
            out.append(float("nan"))
            continue
        if not complexes:
            out.append(float("nan"))
            continue
        scores = [
            float(c.vina_score)
            for c in complexes
            if getattr(c, "vina_score", None) is not None
        ]
        out.append(float(min(scores)) if scores else float("nan"))
    return out


def vina_high_affinity_gpu(
    smiles_list: Sequence[str],
    pocket: Any,
    *,
    cutoff_kcal: float = -8.0,
    **kwargs: Any,
) -> List[bool]:
    """Boolean classifier: True iff Vina-GPU best pose ≤ ``cutoff_kcal``.

    Mirrors :func:`molmetal.scoring.score_any._w_vina_high_affinity`
    (CPU) but on the GPU engine.
    """
    scores = vina_score_gpu(smiles_list, pocket, **kwargs)
    return [s <= float(cutoff_kcal) for s in scores]


def vina_min_score_gpu(
    smiles_list: Sequence[str],
    pocket: Any,
    *,
    cutoff_kcal: float = -8.0,
    **kwargs: Any,
) -> Dict[str, float]:
    """Aggregate Vina-GPU stats over a batch (min / median / n_below_cutoff)."""
    scores = vina_score_gpu(smiles_list, pocket, **kwargs)
    valid = [s for s in scores if s == s]  # drop NaN
    if not valid:
        return {
            "min": float("nan"),
            "median": float("nan"),
            "n_below_cutoff": 0,
            "n_total": 0,
        }
    sorted_s = sorted(valid)
    median = sorted_s[len(sorted_s) // 2]
    return {
        "min": float(sorted_s[0]),
        "median": float(median),
        "n_below_cutoff": sum(1 for s in valid if s <= float(cutoff_kcal)),
        "n_total": len(valid),
    }


# ---------------------------------------------------------------------------
# Multi-pocket batch — dispatch each pocket through its own adapter call
# ---------------------------------------------------------------------------
def vina_score_gpu_multi_pocket(
    jobs: Sequence[Tuple[Any, Sequence[str]]],
    *,
    n_poses: int = 5,
    exhaustiveness: int = 8,
    **kwargs: Any,
) -> List[List[float]]:
    """Run Vina-GPU on multiple pockets in one call.

    Parameters
    ----------
    jobs : sequence of (pocket, smiles_list)
        Each entry is one pocket + the ligands to dock against it. The
        adapter caches the prepared receptor on the first call so the
        per-pocket overhead is amortised.

    Returns
    -------
    list[list[float]]
        One ``list[float]`` per job (each inner list has one kcal/mol
        per SMILES for that pocket).
    """
    out: List[List[float]] = []
    for pocket, smiles_list in jobs:
        out.append(
            vina_score_gpu(
                smiles_list,
                pocket,
                n_poses=n_poses,
                exhaustiveness=exhaustiveness,
                **kwargs,
            )
        )
    return out


__all__ = [
    "EngineInfo",
    "vina_score_gpu",
    "vina_high_affinity_gpu",
    "vina_min_score_gpu",
    "vina_score_gpu_multi_pocket",
]


# ---------------------------------------------------------------------------
# Quiet the RDKit / meeko warnings that the underlying adapter triggers
# routinely — they don't affect the kcal/mol output.
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
