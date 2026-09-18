"""Evaluation metrics for MolFlow-Triton.

Provides:
- Validity: fraction of generated SMILES that parse as valid molecules.
- Uniqueness: fraction of unique molecules among the valid ones.
- Novelty: fraction of valid molecules not present in the training set.

RDKit is optional; when unavailable all rdkit-backed metrics degrade gracefully.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

try:  # rdkit is optional
    from rdkit import Chem, RDLogger

    _HAS_RDKIT = True
    # Silence RDKit chatter during parsing.
    try:
        RDLogger.DisableLog("rdApp.*")
    except Exception:  # pragma: no cover - defensive
        pass
except Exception:  # pragma: no cover - rdkit missing
    Chem = None  # type: ignore[assignment]
    _HAS_RDKIT = False


def _require_rdkit():
    if not _HAS_RDKIT:
        raise RuntimeError(
            "rdkit is not installed. Install it with `uv pip install rdkit` "
            "(or `pip install rdkit-pypi`) to use this metric."
        )


def _mol_from_smiles(smiles: str):
    """Parse SMILES into a Mol, returning None on failure."""
    if not _HAS_RDKIT:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def validity(smiles_list: Iterable[str]) -> float:
    """Fraction of SMILES that parse as valid molecules.

    Returns 0.0 if rdkit is unavailable. Returns 0.0 for an empty input.
    """
    smiles_list = list(smiles_list)
    if not smiles_list:
        return 0.0
    if not _HAS_RDKIT:
        return 0.0
    n_valid = sum(1 for s in smiles_list if _mol_from_smiles(s) is not None)
    return n_valid / len(smiles_list)


def uniqueness(
    smiles_list: Iterable[str],
    *,
    canonical: bool = True,
) -> float:
    """Fraction of unique molecules among the valid SMILES.

    If `canonical` is True (default) and rdkit is available, molecules are
    compared by canonical SMILES. Otherwise compared by raw string.
    Returns 0.0 if there are no valid molecules.
    """
    if not _HAS_RDKIT or not canonical:
        keys = [s for s in smiles_list if s]
        if not keys:
            return 0.0
        return len(set(keys)) / len(keys)

    canonical_set: List[str] = []
    for s in smiles_list:
        mol = _mol_from_smiles(s)
        if mol is None:
            continue
        try:
            canonical_set.append(Chem.MolToSmiles(mol))
        except Exception:
            continue

    if not canonical_set:
        return 0.0
    return len(set(canonical_set)) / len(canonical_set)


def novelty(
    smiles_list: Iterable[str],
    training_smiles: Sequence[str],
    *,
    canonical: bool = True,
) -> float:
    """Fraction of valid molecules NOT present in the training set.

    Comparison is by canonical SMILES when rdkit is available, otherwise by
    raw string. Returns 0.0 if no valid generated molecules.
    """
    training_set = _canonical_set(training_smiles, canonical=canonical)
    if training_set is None:
        return 0.0

    generated_keys: List[str] = []
    if not _HAS_RDKIT or not canonical:
        generated_keys = [s for s in smiles_list if s]
    else:
        for s in smiles_list:
            mol = _mol_from_smiles(s)
            if mol is None:
                continue
            try:
                generated_keys.append(Chem.MolToSmiles(mol))
            except Exception:
                continue

    if not generated_keys:
        return 0.0
    n_novel = sum(1 for k in generated_keys if k not in training_set)
    return n_novel / len(generated_keys)


def _canonical_set(
    smiles_list: Iterable[str],
    *,
    canonical: bool,
) -> Optional[set]:
    """Return a set of canonical (or raw) SMILES keys for the training set.

    Returns None if rdkit is required but unavailable.
    """
    if not _HAS_RDKIT or not canonical:
        return {s for s in smiles_list if s}
    out: set = set()
    for s in smiles_list:
        mol = _mol_from_smiles(s)
        if mol is None:
            continue
        try:
            out.add(Chem.MolToSmiles(mol))
        except Exception:
            continue
    return out


def evaluate(
    smiles_list: Iterable[str],
    training_smiles: Sequence[str],
) -> dict:
    """Compute validity / uniqueness / novelty in one shot.

    Returns a dict with float values; values are 0.0 when rdkit is missing.
    """
    smiles_list = list(smiles_list)
    return {
        "validity": validity(smiles_list),
        "uniqueness": uniqueness(smiles_list),
        "novelty": novelty(smiles_list, training_smiles),
        "n_samples": len(smiles_list),
    }
