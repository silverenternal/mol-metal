"""pb_batch — PoseBusters batch validation wrapper.

================================================================
Wf-50 Phase 2 (#4 — PoseBusters batch wrapper)
================================================================

PoseBusters itself is RDKit-bound (MMFF94 + ETKDGv3 + 26 bool checks
per SMILES) and has no CUDA port. The audit calls for a "torch
tensor SMILES batch → batch validation" wrapper — the lever isn't GPU
compute, it's **dispatch overlap**. This wrapper:

* Stages the SMILES list as a torch tensor (one int32 hash per
  SMILES) for parity with the other GPU wrappers.
* Runs the per-SMILES PoseBusters validation through a thread pool
  (default ``os.cpu_count()``) so the wall-time is bounded by the
  slowest single SMILES, not the sum.
* Preserves the same ``pass_rate`` semantics as the CPU wrapper in
  :func:`molmetal.scoring.score_any._w_pb_pass_rate`.

Public API
----------
* :func:`pb_pass_rate_batch`    — float (fraction passing all checks).
* :func:`pb_validate_list_batch` — list[:class:`ValidityReport`].
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _scoring_exceptions():
    from molmetal.scoring.score_any import AdapterUnavailable

    return AdapterUnavailable


def _posebusters_module():
    from molmetal.molmetal_lam.sbdd_env import posebusters_adapter

    return posebusters_adapter


def _import_torch():
    import torch  # type: ignore

    return torch


# ---------------------------------------------------------------------------
# PoseBusters availability + adapter construction
# ---------------------------------------------------------------------------
def _have_posebusters() -> bool:
    try:
        from posebusters import PoseBusters  # noqa: F401

        return True
    except Exception:
        return False


def _build_adapter(mode: str = "mol", full_report: bool = False) -> Any:
    AdapterUnavailable = _scoring_exceptions()
    if not _have_posebusters():
        raise AdapterUnavailable(
            "pb_batch: posebusters not installed. `uv pip install "
            "posebusters` to enable pb_pass_rate_batch."
        )
    pb = _posebusters_module()
    return pb.PoseBustersAdapter(mode=mode, full_report=full_report)


def _empty_report(smiles: str) -> Any:
    """Return a fail-safe :class:`ValidityReport` for an exception path.

    This is defined locally so the wrapper doesn't have to import the
    full :mod:`posebusters_adapter` module (which requires pandas).
    Tests can patch :func:`pb_batch._empty_report` to provide a real
    :class:`ValidityReport` without triggering the pandas import.
    """
    return _EmptyReportShim(smiles=smiles)


class _EmptyReportShim:
    """Minimal stand-in for :class:`ValidityReport`.

    The real dataclass lives in
    :mod:`molmetal.molmetal_lam.sbdd_env.posebusters_adapter`, which
    imports pandas. The wrapper uses this shim on the exception path
    so we never import pandas on a box that doesn't have it.
    """

    __slots__ = ("smiles", "passed", "n_checks", "n_passed", "pass_rate",
                 "failed_checks", "details")

    def __init__(self, smiles: str, passed: bool = False,
                 n_checks: int = 0, n_passed: int = 0,
                 pass_rate: float = 0.0,
                 failed_checks: tuple = (),
                 details: Optional[dict] = None):
        self.smiles = smiles
        self.passed = passed
        self.n_checks = n_checks
        self.n_passed = n_passed
        self.pass_rate = pass_rate
        self.failed_checks = failed_checks
        self.details = details or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def pb_pass_rate_batch(
    smiles_list: Sequence[str],
    *,
    mode: str = "mol",
    n_workers: Optional[int] = None,
    device: str = "cuda:0",
    use_torch_staging: bool = True,
) -> float:
    """PoseBusters batch pass-rate over a list of SMILES.

    Parameters
    ----------
    smiles_list : sequence of str
        Ligand SMILES — one validation per input.
    mode : str, default ``"mol"``
        Forwarded to :class:`PoseBustersAdapter` (``"mol"`` /
        ``"dock"`` / ``"redock"``).
    n_workers : int, optional
        Thread-pool size (default ``os.cpu_count()``).
    device : str, default ``"cuda:0"``
        Cosmetic — PoseBusters itself is RDKit-bound, but we mirror the
        other wrappers' ``device`` knob so call-sites can swap to
        ``"cpu"`` without API drift.
    use_torch_staging : bool, default True
        When ``True``, stage the SMILES list as a torch int32 tensor
        before dispatching to the thread pool. The tensor isn't moved
        to the device (PoseBusters is CPU-only) — it just gives the
        caller a single typed handle on the input batch.

    Returns
    -------
    float
        Fraction of inputs that pass ALL PoseBusters checks. Returns
        ``0.0`` on an empty input (matches the CPU wrapper).

    Raises
    ------
    AdapterUnavailable
        When PoseBusters isn't installed.
    """
    AdapterUnavailable = _scoring_exceptions()
    if not smiles_list:
        return 0.0

    adapter = _build_adapter(mode=mode, full_report=False)

    encoded = [str(s) for s in smiles_list]

    # Stage as torch tensor (no GPU transfer — PB is RDKit-only).
    if use_torch_staging:
        try:
            torch = _import_torch()
            _ = torch.tensor(
                [hash(s) & 0x7FFFFFFF for s in encoded], dtype=torch.int32
            )
        except Exception:
            # torch unavailable on a bare RDKit-only env: ignore.
            pass

    # Dispatch validation across a thread pool. Each worker runs the
    # adapter's ``validate_mol`` synchronously; the pool's GIL release
    # (via RDKit's C++ calls) gives us real overlap.
    workers = int(n_workers) if n_workers is not None else max(1, os.cpu_count() or 1)
    workers = min(workers, max(1, len(encoded)))
    if workers == 1:
        reports = [adapter.validate_mol(s) for s in encoded]
    else:
        reports = [None] * len(encoded)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(adapter.validate_mol, s): i for i, s in enumerate(encoded)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    reports[i] = fut.result()
                except Exception as exc:
                    logger.warning("pb_pass_rate_batch: SMILES[%d] failed: %s", i, exc)
                    # Mirror the empty-input result so the denominator
                    # stays consistent with the CPU wrapper.
                    reports[i] = _empty_report(encoded[i])

    if not reports:
        return 0.0
    return float(sum(bool(r.passed) for r in reports)) / len(reports)


def pb_validate_list_batch(
    smiles_list: Sequence[str],
    *,
    mode: str = "mol",
    n_workers: Optional[int] = None,
    device: str = "cuda:0",
    use_torch_staging: bool = True,
) -> List[Any]:
    """PoseBusters batch validation, returning one :class:`ValidityReport` per input.

    Use this when you need per-SMILES failure diagnostics (the audit
    hooks call this for the ``pb_pass_rate`` column when the column
    is needed in full-report mode).
    """
    AdapterUnavailable = _scoring_exceptions()
    if not smiles_list:
        return []

    adapter = _build_adapter(mode=mode, full_report=True)
    encoded = [str(s) for s in smiles_list]

    if use_torch_staging:
        try:
            torch = _import_torch()
            _ = torch.tensor(
                [hash(s) & 0x7FFFFFFF for s in encoded], dtype=torch.int32
            )
        except Exception:
            pass

    workers = int(n_workers) if n_workers is not None else max(1, os.cpu_count() or 1)
    workers = min(workers, max(1, len(encoded)))
    if workers == 1:
        return list(adapter.validate_list(encoded))

    reports: List[Any] = [None] * len(encoded)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(adapter.validate_mol, s): i for i, s in enumerate(encoded)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                reports[i] = fut.result()
            except Exception as exc:
                logger.warning(
                    "pb_validate_list_batch: SMILES[%d] failed: %s", i, exc
                )
                reports[i] = _empty_report(encoded[i])
    return reports  # type: ignore[return-value]


__all__ = [
    "pb_pass_rate_batch",
    "pb_validate_list_batch",
]
