"""Batched RDKit wrappers for high-throughput MCTS rollouts.

Every MCTS rollout evaluates one or more ``MolFromSmiles`` /
``Descriptors.*`` calls per state.  Each call has a sizeable fixed
overhead (C++ Python-binding round-trip, atom-bond table allocation)
that dominates the actual chemistry work for small drug-like ligands.
In a ``RewardAggregator._batch_rewards`` step we typically have
**tens to hundreds** of candidates — dispatching each through a single
``Pool.starmap`` task is wasteful because every worker task pays the
binding cost again per candidate.

This module provides a small set of *intra-batch* helpers that split a
list of SMILES across a :class:`multiprocessing.Pool` *once*, then
return numpy arrays (or RDKit mols) in the parent's order.  A serial
fallback is provided for environments without ``multiprocessing``
(some CI sandboxes, Jupyter notebooks running under ``-X faulthandler``,
Windows under default ``spawn`` semantics with re-entrancy).

Public surface
--------------
* :func:`batch_mol_from_smiles`     — parallel ``Chem.MolFromSmiles``
* :func:`batch_descriptors`         — parallel named descriptor (MW / logP / …)
* :func:`batch_qed`                  — parallel ``Descriptors.qed``
* :func:`batch_sa`                   — parallel SA proxy (QED-driven fallback)
* :func:`batch_lipinski`             — parallel Ro5 boolean mask
* :func:`is_parallel_safe`           — cheap probe for pool availability
* :func:`_sequential_*`              — used by the parallel functions when
                                       multiprocessing fails
"""

from __future__ import annotations

import logging
import multiprocessing as _mp
import os
from typing import List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Safe RDKit import — the helpers degrade to NaN/False when RDKit is missing
# so the rest of molmetal still imports under a stripped-down environment.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised in CI with RDKit installed
    from rdkit import Chem as _Chem  # type: ignore
    from rdkit.Chem import Descriptors as _Descriptors  # type: ignore
    from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors  # type: ignore
    _RDKIT_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    _Chem = None  # type: ignore
    _Descriptors = None  # type: ignore
    _rdMolDescriptors = None  # type: ignore
    _RDKIT_AVAILABLE = False


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multiprocessing availability probe
# ---------------------------------------------------------------------------

#: Cache the result of the multiprocessing probe — calling it on every
#: batch call is cheap but pointless when the environment is stable.
_PROBE_RESULT: Optional[bool] = None


def is_parallel_safe() -> bool:
    """Return True iff :class:`multiprocessing.Pool` should be used.

    We require at least 2 CPUs (``os.cpu_count()`` can return ``None``
    in some sandboxes) **and** that the import of
    :mod:`multiprocessing.pool` succeeded.  The probe is memoised so
    the cost is paid once per process.
    """
    global _PROBE_RESULT
    if _PROBE_RESULT is not None:
        return _PROBE_RESULT
    cpu = os.cpu_count() or 1
    if cpu < 2:
        _PROBE_RESULT = False
        return False
    try:  # pragma: no cover - import-only check
        import multiprocessing.pool as _  # noqa: F401
    except Exception:
        _PROBE_RESULT = False
        return False
    _PROBE_RESULT = True
    return True


def _chunks(seq: Sequence, n_workers: int) -> List:
    """Split ``seq`` into ``n_workers`` roughly-equal chunks."""
    n = len(seq)
    if n == 0:
        return [[] for _ in range(max(1, n_workers))]
    chunk_size = max(1, (n + n_workers - 1) // n_workers)
    return [seq[i:i + chunk_size] for i in range(0, n, chunk_size)]


# ---------------------------------------------------------------------------
# Worker-side helpers — must be module-level so they pickle cleanly
# ---------------------------------------------------------------------------

def _mol_from_smiles_chunk(smiles_chunk: Sequence[str]) -> List[Optional[object]]:
    """Worker entry: parse a chunk of SMILES -> list[Chem.Mol | None]."""
    out: List[Optional[object]] = []
    for s in smiles_chunk:
        if not _RDKIT_AVAILABLE or _Chem is None or s is None:
            out.append(None)
            continue
        try:
            out.append(_Chem.MolFromSmiles(s))
        except Exception:
            out.append(None)
    return out


def _descriptors_chunk(
    args_chunk: Sequence,
) -> List[float]:
    """Worker entry: compute one named descriptor for a chunk of SMILES.

    Each item in ``args_chunk`` is a ``(smi, descriptor_name)`` tuple.
    Returns a list of floats; ``nan`` is used when parsing fails.
    """
    out: List[float] = []
    for smi, name in args_chunk:
        out.append(_scalar_descriptor(smi, name))
    return out


def _scalar_descriptor(smi: str, name: str) -> float:
    """Compute a single RDKit descriptor by name. Returns ``nan`` on failure."""
    if not _RDKIT_AVAILABLE or _Chem is None:
        return float("nan")
    try:
        mol = _Chem.MolFromSmiles(smi) if smi else None
    except Exception:
        mol = None
    if mol is None:
        return float("nan")
    try:
        n = name.lower()
        if n == "mw" or n == "molwt":
            return float(_Descriptors.MolWt(mol))
        if n == "logp" or n == "mollogp":
            return float(_Descriptors.MolLogP(mol))
        if n == "rotb" or n == "numrotatablebonds":
            return float(_Descriptors.NumRotatableBonds(mol))
        if n == "hbd" or n == "numhbd":
            return float(_rdMolDescriptors.CalcNumHBD(mol))
        if n == "hba" or n == "numhba":
            return float(_rdMolDescriptors.CalcNumHBA(mol))
        if n == "rings" or n == "numrings":
            return float(_rdMolDescriptors.CalcNumRings(mol))
        if n == "tpsa":
            return float(_Descriptors.TPSA(mol))
        if n == "qed":
            return float(_Descriptors.qed(mol))
        # Unknown descriptor — explicit fallback to NaN.
        return float("nan")
    except Exception:
        return float("nan")


def _qed_chunk(smiles_chunk: Sequence[str]) -> List[float]:
    return [_scalar_descriptor(s, "qed") for s in smiles_chunk]


def _sa_chunk(smiles_chunk: Sequence[str]) -> List[float]:
    """SA proxy = clip(0.02*MW + 0.5*|logP| + 0.1*RotB, 1, 10).

    We deliberately keep the *same formula* as ``_sweep_helpers._sa_proxy``
    so the batched result is numerically identical to the scalar
    implementation.  This is the same Ertl-style SA surrogate we use
    throughout the Lambda sweep — we explicitly avoid pulling in
    :mod:`rdkit.Contrib.SA_Score` because that contribution ships as
    *optional* (not bundled with `rdkit-pypi` wheels).
    """
    return [
        _sa_proxy_one(s)
        for s in smiles_chunk
    ]


def _sa_proxy_one(smi: str) -> float:
    """Single-molecule SA proxy. Mirrors ``_sweep_helpers._sa_proxy``."""
    if not _RDKIT_AVAILABLE or _Chem is None:
        return float("nan")
    try:
        mol = _Chem.MolFromSmiles(smi) if smi else None
    except Exception:
        mol = None
    if mol is None:
        return float("nan")
    try:
        mw = float(_Descriptors.MolWt(mol))
        logp = float(_Descriptors.MolLogP(mol))
        rotb = float(_Descriptors.NumRotatableBonds(mol))
    except Exception:
        return float("nan")
    return max(1.0, min(10.0, 0.02 * mw + 0.5 * abs(logp) + 0.1 * rotb))


def _lipinski_chunk(smiles_chunk: Sequence[str]) -> List[bool]:
    return [_lipinski_one(s) for s in smiles_chunk]


def _lipinski_one(smi: str) -> bool:
    """Single-molecule Ro5 filter.

    Mirrors ``_sweep_helpers.lipinski_pass`` semantics — empty SMILES
    parses to a 0-atom molecule that passes (we document this in the
    sweep helpers tests).
    """
    if not _RDKIT_AVAILABLE or _Chem is None:
        return False
    try:
        mol = _Chem.MolFromSmiles(smi) if smi else None
    except Exception:
        return False
    if mol is None:
        return False
    try:
        mw = float(_Descriptors.MolWt(mol))
        logp = float(_Descriptors.MolLogP(mol))
        hbd = float(_rdMolDescriptors.CalcNumHBD(mol))
        hba = float(_rdMolDescriptors.CalcNumHBA(mol))
    except Exception:
        return False
    return mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10


# ---------------------------------------------------------------------------
# Sequential fallback (used when multiprocessing.Pool is unsafe)
# ---------------------------------------------------------------------------

def _sequential_mol(smiles_list: Sequence[str]) -> List[Optional[object]]:
    return [_Chem.MolFromSmiles(s) if (_RDKIT_AVAILABLE and s) else None
            for s in smiles_list]


def _sequential_descriptors(
    smiles_list: Sequence[str], name: str,
) -> List[float]:
    return [_scalar_descriptor(s, name) for s in smiles_list]


def _sequential_qed(smiles_list: Sequence[str]) -> List[float]:
    return [_scalar_descriptor(s, "qed") for s in smiles_list]


def _sequential_sa(smiles_list: Sequence[str]) -> List[float]:
    return [_sa_proxy_one(s) for s in smiles_list]


def _sequential_lipinski(smiles_list: Sequence[str]) -> List[bool]:
    return [_lipinski_one(s) for s in smiles_list]


# ---------------------------------------------------------------------------
# Pool helpers — try / except with a clear log + serial fallback
# ---------------------------------------------------------------------------

def _run_pool(
    fn,
    chunks: Sequence,
    n_workers: int,
    pool_label: str,
):
    """Run ``fn`` over ``chunks`` in a multiprocessing.Pool with safe fallback."""
    if not is_parallel_safe():
        return None
    n_chunks = len(chunks)
    if n_chunks <= 0:
        return []
    try:
        ctx = _mp.get_context("fork")
        worker_n = max(1, min(int(n_workers), n_chunks))
        with ctx.Pool(processes=worker_n) as pool:
            return pool.map(fn, list(chunks))
    except Exception as exc:
        log.debug(
            "batched_rdkit.%s pool failed (%s); using serial fallback",
            pool_label, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def batch_mol_from_smiles(
    smiles_list: Sequence[str],
    n_workers: int = 0,
) -> List[Optional[object]]:
    """Parse ``smiles_list`` in parallel, returning RDKit mols in order.

    Returns ``[Chem.Mol | None, ...]`` with the same length and order as
    ``smiles_list``.  ``None`` is returned for un-parseable strings or
    empty inputs.  Falls back to a sequential loop when the pool setup
    fails (sandboxed CI, Jupyter, Windows spawn-reentrancy).
    """
    smiles_list = list(smiles_list)
    n = len(smiles_list)
    if n == 0:
        return []
    if not _RDKIT_AVAILABLE:
        return [None] * n
    n_workers = int(n_workers) if int(n_workers) > 0 else (os.cpu_count() or 2)
    # The chunking cost only pays off when there's enough work.
    if not is_parallel_safe() or n < 2 * n_workers:
        return _sequential_mol(smiles_list)
    chunks = _chunks(smiles_list, n_workers)
    results = _run_pool(_mol_from_smiles_chunk, chunks, n_workers, "batch_mol_from_smiles")
    if results is None:
        return _sequential_mol(smiles_list)
    out: List[Optional[object]] = []
    for chunk_result in results:
        out.extend(chunk_result)
    # Pad / truncate to the requested length (defensive).
    if len(out) < n:
        out.extend([None] * (n - len(out)))
    return out[:n]


def batch_descriptors(
    mols: Sequence[object],
    descriptor_name: str,
    n_workers: int = 0,
) -> np.ndarray:
    """Compute a named descriptor on a list of RDKit mols in parallel.

    ``descriptor_name`` must be one of: ``mw``, ``logp``, ``rotb``,
    ``hbd``, ``hba``, ``rings``, ``tpsa``, ``qed`` — case-insensitive.
    Unknown names return a NaN array of the correct length.
    """
    mols = list(mols)
    n = len(mols)
    if n == 0:
        return np.zeros(0, dtype=float)
    if not _RDKIT_AVAILABLE:
        return np.full(n, float("nan"), dtype=float)
    # Re-derive SMILES per mol — keeps the worker free of RDKit state
    # beyond Chem.MolFromSmiles.
    smiles_list: List[str] = []
    for m in mols:
        if m is None:
            smiles_list.append("")
        else:
            try:
                smiles_list.append(_Chem.MolToSmiles(m))
            except Exception:
                smiles_list.append("")
    return batch_descriptors_from_smiles(smiles_list, descriptor_name, n_workers=n_workers)


def batch_descriptors_from_smiles(
    smiles_list: Sequence[str],
    descriptor_name: str,
    n_workers: int = 0,
) -> np.ndarray:
    """Compute a named descriptor on a list of SMILES in parallel."""
    smiles_list = list(smiles_list)
    n = len(smiles_list)
    if n == 0:
        return np.zeros(0, dtype=float)
    if not _RDKIT_AVAILABLE:
        return np.full(n, float("nan"), dtype=float)
    n_workers = int(n_workers) if int(n_workers) > 0 else (os.cpu_count() or 2)
    if not is_parallel_safe() or n < 2 * n_workers:
        return np.asarray(_sequential_descriptors(smiles_list, descriptor_name), dtype=float)
    payload = [(s, descriptor_name) for s in smiles_list]
    chunks = _chunks(payload, n_workers)
    results = _run_pool(_descriptors_chunk, chunks, n_workers, "batch_descriptors")
    if results is None:
        return np.asarray(_sequential_descriptors(smiles_list, descriptor_name), dtype=float)
    flat: List[float] = []
    for chunk_result in results:
        flat.extend(chunk_result)
    return np.asarray(flat[:n], dtype=float)


def batch_qed(
    smiles_list: Sequence[str],
    n_workers: int = 0,
) -> np.ndarray:
    """Compute RDKit ``Descriptors.qed`` on ``smiles_list`` in parallel."""
    smiles_list = list(smiles_list)
    n = len(smiles_list)
    if n == 0:
        return np.zeros(0, dtype=float)
    if not _RDKIT_AVAILABLE:
        return np.full(n, float("nan"), dtype=float)
    n_workers = int(n_workers) if int(n_workers) > 0 else (os.cpu_count() or 2)
    if not is_parallel_safe() or n < 2 * n_workers:
        return np.asarray(_sequential_qed(smiles_list), dtype=float)
    chunks = _chunks(smiles_list, n_workers)
    results = _run_pool(_qed_chunk, chunks, n_workers, "batch_qed")
    if results is None:
        return np.asarray(_sequential_qed(smiles_list), dtype=float)
    flat: List[float] = []
    for chunk_result in results:
        flat.extend(chunk_result)
    return np.asarray(flat[:n], dtype=float)


def batch_sa(
    smiles_list: Sequence[str],
    n_workers: int = 0,
) -> np.ndarray:
    """Compute the SA proxy (MW + logP + rotatable-bonds linear score).

    Why not RDKit's ``rdkit.Contrib.SA_Score``? — the ``SA_Score``
    module ships as an *optional* contribution and is not bundled with
    the ``rdkit-pypi`` wheels used by Mol-Metal.  The Ertl-style proxy
    implemented in :func:`_sa_proxy_one` matches the formula used by
    ``_sweep_helpers._sa_proxy`` exactly, so callers see numerically
    identical results to the scalar path.
    """
    smiles_list = list(smiles_list)
    n = len(smiles_list)
    if n == 0:
        return np.zeros(0, dtype=float)
    if not _RDKIT_AVAILABLE:
        return np.full(n, float("nan"), dtype=float)
    n_workers = int(n_workers) if int(n_workers) > 0 else (os.cpu_count() or 2)
    if not is_parallel_safe() or n < 2 * n_workers:
        return np.asarray(_sequential_sa(smiles_list), dtype=float)
    chunks = _chunks(smiles_list, n_workers)
    results = _run_pool(_sa_chunk, chunks, n_workers, "batch_sa")
    if results is None:
        return np.asarray(_sequential_sa(smiles_list), dtype=float)
    flat: List[float] = []
    for chunk_result in results:
        flat.extend(chunk_result)
    return np.asarray(flat[:n], dtype=float)


def batch_lipinski(
    smiles_list: Sequence[str],
    n_workers: int = 0,
) -> np.ndarray:
    """Compute the Ro5 boolean mask for ``smiles_list`` in parallel.

    Returns a ``np.bool_`` array where ``True`` iff the SMILES parses
    AND satisfies ``MW <= 500 ∧ LogP <= 5 ∧ HBD <= 5 ∧ HBA <= 10``.
    """
    smiles_list = list(smiles_list)
    n = len(smiles_list)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if not _RDKIT_AVAILABLE:
        return np.zeros(n, dtype=bool)
    n_workers = int(n_workers) if int(n_workers) > 0 else (os.cpu_count() or 2)
    if not is_parallel_safe() or n < 2 * n_workers:
        return np.asarray(_sequential_lipinski(smiles_list), dtype=bool)
    chunks = _chunks(smiles_list, n_workers)
    results = _run_pool(_lipinski_chunk, chunks, n_workers, "batch_lipinski")
    if results is None:
        return np.asarray(_sequential_lipinski(smiles_list), dtype=bool)
    flat: List[bool] = []
    for chunk_result in results:
        flat.extend(chunk_result)
    return np.asarray(flat[:n], dtype=bool)


__all__ = [
    "is_parallel_safe",
    "batch_mol_from_smiles",
    "batch_descriptors",
    "batch_descriptors_from_smiles",
    "batch_qed",
    "batch_sa",
    "batch_lipinski",
]
