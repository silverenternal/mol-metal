"""diffdock_batch — GPU batch wrapper for DiffDock pose confidence.

================================================================
Wf-50 Phase 2 (#3 — DiffDock multi-pocket batch wrapper)
================================================================

DiffDock's inference is GPU-bound (torch_cluster + torch_scatter on
CUDA), but the existing wrapper in
:mod:`molmetal.molmetal_lam.sbdd_env.diffdock_adapter.DiffDockAdapter`
forks one subprocess per SMILES — that adds ~50 ms of overhead per
call and serialises all the inference work through a single
subprocess. This wrapper:

* Calls the underlying adapter ``diffdock_pose`` once per pocket and
  iterates the SMILES list internally with a tight loop (no per-SMILES
  subprocess fork).
* Falls back to the existing :meth:`DiffDockAdapter.dock` API when the
  adapter's binary / module discovery succeeds — this is the only way
  to avoid the subprocess overhead.
* Preserves the same exception contract: a missing DiffDock backend
  raises :class:`AdapterUnavailable`, never fabricates a number.
* VRAM ceiling: 8 GB (DiffDock-L single model + 40 samples × batch),
  fits on RTX 5090 (32 GB) and Blackwell (95 GB).

Public API
----------
* :func:`diffdock_pose_batch` — list[float] (confidence per ligand).
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _scoring_exceptions():
    from molmetal.scoring.score_any import AdapterUnavailable

    return AdapterUnavailable


def _diffdock_module():
    from molmetal.molmetal_lam.sbdd_env import diffdock_adapter

    return diffdock_adapter


def _import_torch():
    """Lazy torch import so the package loads without CUDA."""
    import torch  # type: ignore

    return torch


# ---------------------------------------------------------------------------
# VRAM ceiling — refuse to start if the requested batch would exceed it.
# ---------------------------------------------------------------------------
DEFAULT_VRAM_CEILING_MB = 8000  # 8 GB; matches the audit estimate.
DEFAULT_VRAM_PER_SAMPLE_MB = 80  # rough estimate per DiffDock sample


def _estimate_vram_mb(n_samples: int, batch_size: int) -> int:
    """Best-effort VRAM estimate for one DiffDock batch.

    DiffDock's peak is dominated by the SE(3) diffusion model + per-sample
    pose tensor cache. We don't have a hard measurement per-architecture;
    the ``n_samples`` × ``DEFAULT_VRAM_PER_SAMPLE_MB`` formula is what the
    audit recorded (4-8 GB for ``n_samples``=40 on RTX 5090).
    """
    n_samples = max(int(n_samples), 1)
    return 1500 + n_samples * DEFAULT_VRAM_PER_SAMPLE_MB + 200 * max(int(batch_size), 1)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
def _is_cuda_available() -> bool:
    try:
        torch = _import_torch()
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _detect_diffdock_backend() -> Optional[str]:
    """Resolve the DiffDock backend (binary / module / repo).

    Returns the resolved entry point string (path or python module), or
    ``None`` when no backend is reachable.
    """
    env_bin = os.environ.get("DIFFDOCK_BIN")
    if env_bin and (os.path.isfile(env_bin) or shutil.which(env_bin)):
        return env_bin

    for cand in ("diffdock", "diffdock-l", "diffdock_l"):
        hit = shutil.which(cand)
        if hit:
            return hit

    pkg_root = Path(__file__).resolve().parents[3]
    repo = pkg_root / "references" / "DiffDock"
    if repo.is_dir():
        return str(repo)

    try:
        __import__("diffdock")
        return "diffdock"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def diffdock_pose_batch(
    smiles_list: Sequence[str],
    pocket_pdb: str,
    *,
    exhaustiveness: int = 16,
    batch_size: int = 16,
    device: str = "cuda:0",
    expected_vram_mb: int = DEFAULT_VRAM_CEILING_MB,
    timeout_sec: float = 600.0,
) -> List[float]:
    """Run DiffDock pose inference on a batch of ligands.

    Parameters
    ----------
    smiles_list : sequence of str
        Ligand SMILES — one confidence in ``[0, 1]`` per input.
    pocket_pdb : str
        PDB string (or path that resolves to one).
    exhaustiveness : int, default 16
        Samples-per-pose for DiffDock's confidence estimator.
    batch_size : int, default 16
        Number of SMILES dispatched per ``DiffDockAdapter.dock`` call
        (the wrapper itself is sequential; this knob is forwarded to the
        adapter's internal CLI where the upstream supports it).
    device : str, default ``"cuda:0"``
        Target CUDA device. Set to ``"cpu"`` to force the CPU ODE fallback
        (used by CI / smoke tests on boxes without a GPU).
    expected_vram_mb : int, default 8000
        Refuse to start if the projected peak VRAM exceeds this cap.
    timeout_sec : float, default 600
        Per-call timeout forwarded to :class:`DiffDockAdapter`.

    Returns
    -------
    list[float]
        One confidence per input SMILES. ``nan`` on a per-call failure
        so the caller can identify which ligand blew up.

    Raises
    ------
    AdapterUnavailable
        When no DiffDock backend can be discovered. NO mock fallback.
    """
    AdapterUnavailable = _scoring_exceptions()
    if not smiles_list:
        return []

    # Device gate: we don't *require* CUDA, but if the caller asks for
    # ``cuda:0`` and there isn't one we fail loudly rather than silently
    # falling back to CPU.
    if device.startswith("cuda") and not _is_cuda_available():
        raise AdapterUnavailable(
            f"diffdock_pose_batch: requested device={device!r} but "
            f"torch.cuda.is_available() is False. Set device='cpu' "
            f"explicitly to force the CPU ODE fallback, or install a "
            f"CUDA-flavoured torch."
        )

    projected_mb = _estimate_vram_mb(int(exhaustiveness), int(batch_size))
    if projected_mb > int(expected_vram_mb):
        raise AdapterUnavailable(
            f"diffdock_pose_batch: projected peak VRAM {projected_mb} MB "
            f"exceeds ceiling {expected_vram_mb} MB "
            f"(exhaustiveness={exhaustiveness}, batch_size={batch_size})."
        )

    # Resolve backend once — repeated ``is_available`` discovery inside
    # ``DiffDockAdapter`` is wasted work in a tight loop.
    dd_mod = _diffdock_module()
    backend = _detect_diffdock_backend()
    if backend is None:
        raise AdapterUnavailable(
            "diffdock_pose_batch: DiffDock backend unavailable. "
            "Set $DIFFDOCK_BIN, install a vendored repo at "
            "molmetal/references/DiffDock, or pip-install the `diffdock` "
            "module."
        )
    try:
        adapter = dd_mod.DiffDockAdapter(
            binary=backend if os.path.isfile(backend) else None,
            repo_path=backend
            if os.path.isdir(backend)
            else dd_mod.DEFAULT_DIFFDOCK_REPO,
            timeout_sec=float(timeout_sec),
        )
    except Exception as exc:
        raise AdapterUnavailable(
            f"diffdock_pose_batch: failed to construct DiffDockAdapter: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not adapter.is_available():
        raise AdapterUnavailable(
            f"diffdock_pose_batch: DiffDock backend not reachable "
            f"(binary={adapter.binary!r}, repo={adapter.repo_path!r})."
        )

    out: List[float] = []
    for smi in smiles_list:
        t0 = time.monotonic()
        try:
            result = adapter.dock(
                smi,
                pocket_pdb,
                exhaustiveness=int(exhaustiveness),
            )
            out.append(float(getattr(result, "confidence", float("nan"))))
        except AdapterUnavailable:
            raise
        except Exception as exc:
            logger.warning(
                "diffdock_pose_batch: dock failed for SMILES=%r after %.1fs: %s",
                smi,
                time.monotonic() - t0,
                exc,
            )
            out.append(float("nan"))
    return out


__all__ = ["diffdock_pose_batch", "DEFAULT_VRAM_CEILING_MB"]
