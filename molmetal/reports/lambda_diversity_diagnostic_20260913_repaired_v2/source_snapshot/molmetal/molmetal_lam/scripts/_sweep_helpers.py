"""Shared helpers for the Lambda MCTS sweep scripts.

Consolidates helpers that were previously duplicated (or wired through
the ``importlib.util.spec_from_file_location`` hack in
:mod:`molmetal_lam.scripts.lambda_benchmark`) between

* :mod:`molmetal.scripts.lambda_100pocket_sweep` — full 100-pocket
  CrossDocked sweep, default 204-tile library.
* :mod:`molmetal_lam.scripts.lambda_benchmark` — convergence
  benchmark over (n_simulations, branching) cells.

Public surface
--------------
* :func:`build_seed_smiles` — extract a SMILES from a pocket directory
* :func:`build_seed_term`   — parse the seed into a :class:`MoleculeClosedTerm`
* :func:`build_tile_library_for_branching` — branching → MCTS-ready tile list
* :func:`build_reward`      — SA + QED :class:`RewardAggregator`
* :func:`vina_proxy`        — placeholder docking score (real L-1 oracle not yet wired)
* :func:`lipinski_pass`     — Ro5-style filter
* :func:`smi_of`            — robust SMILES extraction from a closed term

The constants :data:`DEFAULT_CROSSDOCKED_ROOT` and
:data:`BRANCHING_LABELS` live here too so both scripts can refer to a
single source of truth.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

from molmetal_lam.lam_chem.batched_rdkit import (
    batch_lipinski as _batch_lipinski,
    batch_qed as _batch_qed,
    batch_sa as _batch_sa,
    is_parallel_safe as _is_parallel_safe,
)
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.search_alg.proof_search import RewardAggregator
from molmetal_lam.tile_lib.library import (
    FRAGMENT_LIBRARY_200_TILES,
    build_tile_library,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Default CrossDocked2020 root.  Both sweep scripts read this; the
#: benchmark accepts an explicit ``--pockets`` directory which
#: overrides it.
DEFAULT_CROSSDOCKED_ROOT = "/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10"


#: Fallback seed when a pocket directory contains no readable SDF.
DEFAULT_SEED_SMILES = "C1=CCC=C1"


#: Human-readable labels for each supported branching factor.  Mirrors
#: the convention used by the round-0 sweep notes:
#:
#: * 12   — single-rule baseline (``build_tile_library(12)``).
#: * 60   — STANDARD_12 x 5 click rules (round-0 default).
#: * 1020 — 204-tile library x 5 click rules (round-4 L-3).
BRANCHING_LABELS: Dict[int, str] = {
    12:    "minimal 12-tile (single rule)",
    60:    "STANDARD_12 x 5 rules",
    1020:  "extended 204-tile x 5 rules",
}


# ---------------------------------------------------------------------------
# Pocket seed extraction
# ---------------------------------------------------------------------------


def build_seed_smiles(pocket_dir: str) -> Optional[str]:
    """Return the first readable SMILES from ``pocket_dir``'s SDFs.

    Falls back to :data:`DEFAULT_SEED_SMILES` (cyclopentadiene) if no
    SDF is present or none can be parsed.
    """
    if not os.path.isdir(pocket_dir):
        return None
    sdf_files = [f for f in os.listdir(pocket_dir) if f.endswith(".sdf")]
    for fname in sdf_files:
        try:
            suppl = Chem.SDMolSupplier(os.path.join(pocket_dir, fname), removeHs=False)
            for m in suppl:
                if m is not None:
                    return Chem.MolToSmiles(m)
        except Exception:
            continue
    return None


def build_seed_term(pocket_dir: str) -> Optional[MoleculeClosedTerm]:
    """Return a :class:`MoleculeClosedTerm` parsed from the pocket seed.

    Reads the first available SDF in ``pocket_dir`` (or the
    :data:`DEFAULT_SEED_SMILES` fallback) and returns a parsed closed
    term with ``embed_3d=False``.  Returns ``None`` when parsing fails
    so the caller can emit a structured ``seed_parse_fail`` record.
    """
    seed_smiles = build_seed_smiles(pocket_dir) or DEFAULT_SEED_SMILES
    try:
        return MoleculeClosedTerm.from_smiles(seed_smiles, embed_3d=False)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tile library selection
# ---------------------------------------------------------------------------


def build_tile_library_for_branching(branching: int) -> List[MoleculeClosedTerm]:
    """Return a ``MoleculeClosedTerm``-typed tile library of size ``branching``.

    For ``branching in {12, 60}`` we use the canonical Phase-0 library
    (12 tiles).  For ``branching == 1020`` we use the Phase-1 204-tile
    library and concatenate 5 copies (the round-4 ``run_one_pocket``
    convention is ``204 tiles x 5 click rules = 1020`` effective
    branches).  Every :class:`Tile` is cast to a
    :class:`MoleculeClosedTerm` via :meth:`MoleculeClosedTerm.from_smiles`
    so the MCTS code can consume it; tiles that fail to parse are
    silently skipped.
    """
    if branching in (12, 60):
        tiles = build_tile_library(12, include_fragments=False)
    elif branching == 1020:
        tiles = []
        for _ in range(5):
            tiles.extend(FRAGMENT_LIBRARY_200_TILES())
        # Truncate to the documented size in case the pool grew.
        tiles = tiles[:1020]
    else:
        raise ValueError(
            f"unsupported branching factor {branching!r}; "
            f"supported: {sorted(BRANCHING_LABELS)}"
        )
    terms: List[MoleculeClosedTerm] = []
    for t in tiles:
        try:
            terms.append(MoleculeClosedTerm.from_smiles(t.smiles, embed_3d=False))
        except Exception:
            continue
    return terms


# ---------------------------------------------------------------------------
# Reward channels
# ---------------------------------------------------------------------------


def _sa_proxy(smi: str) -> float:
    """1..10 SA proxy via MW + logP + rotatable bonds (Ertl-like).

    Routed through :func:`batch_sa` so the implementation matches the
    batched path exactly (numerical identity guaranteed).
    """
    out = _batch_sa([smi])
    if len(out) == 0:
        return float("nan")
    return float(out[0])


def _qed(smi: str) -> float:
    """RDKit ``Descriptors.qed`` on ``smi`` (NaN on parse failure).

    Routed through :func:`batch_qed` so callers see numerically
    identical values to the batched path.
    """
    out = _batch_qed([smi])
    if len(out) == 0:
        return float("nan")
    return float(out[0])


def batched_lipinski(smiles_list):
    """Public re-export of :func:`batch_lipinski` for sweep callers."""
    return _batch_lipinski(smiles_list)


def batched_qed(smiles_list):
    """Public re-export of :func:`batch_qed` for sweep callers."""
    return _batch_qed(smiles_list)


def batched_sa(smiles_list):
    """Public re-export of :func:`batch_sa` for sweep callers."""
    return _batch_sa(smiles_list)


def build_reward() -> RewardAggregator:
    """Return a :class:`RewardAggregator` with SA + QED channels wired.

    Round-2 verified aggregator; weights match the legacy 100-pocket
    sweep (``w_sa=0.5, w_qed=0.5``).  Callers can add a Vina proxy
    channel via :func:`dataclasses.replace` if desired.
    """
    def r_sa(state) -> float:
        smi = smi_of(state)
        return max(0.0, 1.0 - _sa_proxy(smi) / 10.0)

    def r_qed(state) -> float:
        smi = smi_of(state)
        return _qed(smi=smi)

    return RewardAggregator(r_sa=r_sa, r_qed=r_qed, w_sa=0.5, w_qed=0.5)


def vina_proxy(state) -> float:
    """Placeholder Vina-proxy score (0.0 if parsing fails).

    Coarse proxy: ``-0.04 * MW - 0.1 * logP - 0.5 * ring_n``.  Returns
    0.0 when no real docking oracle is available.  This is **not**
    comparable to SOTA Vina scores; see :mod:`lambda_100pocket_sweep`
    for the cite-only comparison policy.
    """
    try:
        smi = smi_of(state)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        ring_n = rdMolDescriptors.CalcNumRings(mol)
        return -0.04 * mw - 0.1 * logp - 0.5 * ring_n
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Filters + extraction
# ---------------------------------------------------------------------------


def lipinski_pass(smi: str) -> bool:
    """Return ``True`` if ``smi`` satisfies the canonical Ro5 rule.

    ``MW <= 500``, ``LogP <= 5``, ``HBD <= 5``, ``HBA <= 10``.
    Returns ``False`` when SMILES parsing fails.

    Backed by :func:`molmetal_lam.lam_chem.batched_rdkit.batch_lipinski`
    so the parallel path is exercised even on the single-molecule API
    — the batched wrapper falls back to a serial loop when
    ``multiprocessing.Pool`` is unavailable or the input is too small.
    """
    out = _batch_lipinski([smi])
    if len(out) == 0:
        return False
    return bool(out[0])


def smi_of(state) -> str:
    """Robustly extract a SMILES string from a MoleculeClosedTerm-like.

    ``canonical_smiles`` in :mod:`closed_term` is an unbound *method*,
    not a property, so ``state.canonical_smiles`` returns the bound
    method object rather than a string.  We first try the
    ``source_smiles`` attribute (cheap) and only call the method when
    we need the canonical form.
    """
    src = getattr(state, "source_smiles", None)
    if isinstance(src, str):
        return src
    try:
        cs = getattr(state, "canonical_smiles", None)
        if callable(cs):
            return str(cs())
        if isinstance(cs, str):
            return cs
    except Exception:
        pass
    return str(state)


__all__ = [
    "DEFAULT_CROSSDOCKED_ROOT",
    "DEFAULT_SEED_SMILES",
    "BRANCHING_LABELS",
    "build_seed_smiles",
    "build_seed_term",
    "build_tile_library_for_branching",
    "build_reward",
    "vina_proxy",
    "lipinski_pass",
    "smi_of",
]