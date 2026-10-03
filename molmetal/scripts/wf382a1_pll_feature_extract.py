#!/usr/bin/env python3
"""Wf-382-A1 — PLL feature extraction CPU-only SCAFFOLD script.

This module is the SCAFFOLD extraction helper for the Wf-382-A1
TODO-94 R20-P3 Multi-PH + PLL CPU SCAFFOLD expansion.  It wraps the
adapter at :mod:`molmetal.adapters.flow_matching_lipman.pll_features`
and provides two public helpers for downstream trainers:

* :func:`per_mol_pll_features`   batch-extract PLL features per molecule
* :func:`aggregate_pll_distribution`   distribution analysis over a cohort

The script is CPU-only and SCAFFOLD-only — no trainer wiring, no
GPU bench.  Full GPU integration is deferred to TODO-207 Wf-299 launch
(USER_OK-gated per AGENTS.md §5.1).

Usage (smoke run)::

    python -m molmetal.scripts.wf382a1_pll_feature_extract --smoke

Output JSON path::

    molmetal/reports/wf382a1_pll_smoke/smoke_result.json

References
----------
* Liu J., Feng H. (2026) "Persistent Local Laplacian Prediction of
  Protein–Ligand Binding Affinities".  J. Chem. Inf. Model. 66 (3).
* Wf-37 P3 Multi-PH + PLL end-to-end (TODO-94, R² +0.087 honest-neg).
* Wf-348-B1 chemprop+PLL CEILING_BREAK (R² +0.6814 on 60-SMILES smoke).
* Wf-359-A1 W₂ on R20-P3 features (per-cell MEASURED).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Repo root resolution — mirror the wf359a1 test pattern
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Adapter import — deferred so the script can be smoke-run even when
# the canonical PLL module is unavailable (we surface a clear error).
try:
    from molmetal.adapters.flow_matching_lipman.pll_features import (
        PLLFeatureAdapter,
        PLLFeatureError,
        persistent_laplacian,
        pll_eigenvalues,
        pll_to_feature_vector,
    )

    _ADAPTER_AVAILABLE = True
except ImportError as exc:  # pragma: no cover — defensive
    _ADAPTER_AVAILABLE = False
    _ADAPTER_IMPORT_ERR = repr(exc)


# ---------------------------------------------------------------------------
# Cohort distribution helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PLLDistribution:
    """Aggregate PLL distribution statistics over a cohort.

    Attributes
    ----------
    n_mols : int
        Number of molecules in the cohort.
    n_failed : int
        Number of molecules for which PLL extraction failed (missing
        conformer, RDKit/scipy unavailable, etc.).
    top_k : int
        Feature vector length used for aggregation.
    per_index_mean : np.ndarray
        Length-``top_k`` array of per-index means.
    per_index_std : np.ndarray
        Length-``top_k`` array of per-index standard deviations.
    per_index_min : np.ndarray
        Length-``top_k`` array of per-index minima.
    per_index_max : np.ndarray
        Length-``top_k`` array of per-index maxima.
    aggregate_n_atoms_mean : float
        Mean number of heavy atoms in the cohort.
    smiles_with_features : List[str]
        SMILES that produced a valid feature vector.
    """

    n_mols: int
    n_failed: int
    top_k: int
    per_index_mean: np.ndarray
    per_index_std: np.ndarray
    per_index_min: np.ndarray
    per_index_max: np.ndarray
    aggregate_n_atoms_mean: float
    smiles_with_features: List[str]

    def as_dict(self) -> dict:
        """Return a JSON-friendly dict for downstream trainer / report writing."""
        return {
            "n_mols": int(self.n_mols),
            "n_failed": int(self.n_failed),
            "top_k": int(self.top_k),
            "per_index_mean": self.per_index_mean.tolist(),
            "per_index_std": self.per_index_std.tolist(),
            "per_index_min": self.per_index_min.tolist(),
            "per_index_max": self.per_index_max.tolist(),
            "aggregate_n_atoms_mean": float(self.aggregate_n_atoms_mean),
            "smiles_with_features": list(self.smiles_with_features),
        }


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def per_mol_pll_features(
    mols: Sequence,
    top_k: int = 10,
    k: int = 16,
    k_nn: int = 8,
    skip_failures: bool = True,
) -> List[np.ndarray]:
    """Extract PLL feature vectors for each molecule in ``mols``.

    Parameters
    ----------
    mols : Sequence[Chem.Mol]
        Sequence of RDKit molecules with 3D conformers.
    top_k : int
        Feature vector length.  Default 10 (matches Wf-382-A1 task brief).
    k : int
        Number of eigenvalues requested from the canonical PLL module.
        Default 16 (Liu & Feng 2026 Table 2).
    k_nn : int
        Number of nearest neighbours per atom.  Default 8.
    skip_failures : bool
        When ``True`` (default), molecules whose PLL extraction fails
        are silently skipped and the function returns the partial list.
        When ``False``, the first failure raises :class:`PLLFeatureError`.

    Returns
    -------
    List[np.ndarray]
        Length-``n_mols_ok`` list of length-``top_k`` feature vectors.
        Order matches the input sequence minus any skipped failures.
    """
    if not _ADAPTER_AVAILABLE:  # pragma: no cover — defensive
        raise PLLFeatureError(f"PLL feature adapter unavailable: {_ADAPTER_IMPORT_ERR}")

    features: List[np.ndarray] = []
    for mol in mols:
        try:
            pll = persistent_laplacian(mol, k=k, k_nn=k_nn)
            eigvals = pll_eigenvalues(pll)
            vec = pll_to_feature_vector(eigvals, top_k=top_k)
        except PLLFeatureError:
            if not skip_failures:
                raise
            continue
        features.append(vec)
    return features


def aggregate_pll_distribution(
    cohort: Sequence,
    top_k: int = 10,
    k: int = 16,
    k_nn: int = 8,
) -> PLLDistribution:
    """Compute aggregate PLL distribution statistics over ``cohort``.

    Parameters
    ----------
    cohort : Sequence[Chem.Mol]
        Sequence of RDKit molecules (typically with 3D conformers).
    top_k, k, k_nn : int
        Feature extraction parameters (see :func:`per_mol_pll_features`).

    Returns
    -------
    PLLDistribution
        Aggregate distribution dataclass; per-index mean/std/min/max
        arrays are populated from successfully-extracted molecules.
        Molecules whose extraction fails are counted in ``n_failed``
        but excluded from the per-index arrays.
    """
    n_mols = len(cohort)
    n_failed = 0
    smiles_ok: List[str] = []
    n_atoms_list: List[int] = []
    feats: List[np.ndarray] = []

    for mol in cohort:
        try:
            pll = persistent_laplacian(mol, k=k, k_nn=k_nn)
            vec = pll_to_feature_vector(pll.eigenvalues, top_k=top_k)
        except PLLFeatureError:
            n_failed += 1
            continue
        feats.append(vec)
        n_atoms_list.append(pll.n_atoms)
        if pll.smiles:
            smiles_ok.append(pll.smiles)

    if feats:
        stacked = np.stack(feats, axis=0)
        per_index_mean = stacked.mean(axis=0)
        per_index_std = stacked.std(axis=0)
        per_index_min = stacked.min(axis=0)
        per_index_max = stacked.max(axis=0)
        agg_n_atoms = float(np.mean(n_atoms_list)) if n_atoms_list else 0.0
    else:
        per_index_mean = np.zeros(top_k, dtype=np.float64)
        per_index_std = np.zeros(top_k, dtype=np.float64)
        per_index_min = np.zeros(top_k, dtype=np.float64)
        per_index_max = np.zeros(top_k, dtype=np.float64)
        agg_n_atoms = 0.0

    return PLLDistribution(
        n_mols=int(n_mols),
        n_failed=int(n_failed),
        top_k=int(top_k),
        per_index_mean=per_index_mean,
        per_index_std=per_index_std,
        per_index_min=per_index_min,
        per_index_max=per_index_max,
        aggregate_n_atoms_mean=agg_n_atoms,
        smiles_with_features=smiles_ok,
    )


# ---------------------------------------------------------------------------
# Smoke-run helper — benzene + methane + ethanol + cisplatin
# ---------------------------------------------------------------------------


def _smoke_cohort() -> Optional[Dict[str, np.ndarray]]:
    """Build a tiny smoke cohort with 3D conformers.

    Returns ``None`` when RDKit is unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None

    smiles_list = [
        "c1ccccc1",
        "C",
        "CCO",
        "C1=CC=CC=C1Cl",
    ]  # benzene, methane, ethanol, chlorobenzene
    out: Dict[str, np.ndarray] = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, randomSeed=42) != 0:
            # Some molecules (e.g. methane) may fail with default ETKDGv3;
            # retry with ETKDGv2.
            params = AllChem.ETKDGv2()
            if AllChem.EmbedMolecule(mol_h, params) != 0:
                continue
        try:
            pll = persistent_laplacian(mol_h, k=8)
            vec = pll_to_feature_vector(pll.eigenvalues, top_k=10)
            out[smi] = vec
        except PLLFeatureError:
            continue
    return out if out else None


def _write_smoke_report(out_dir: Path) -> dict:
    """Run the smoke cohort and dump the result JSON.

    Returns the result dict for in-process callers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "task": "Wf-382-A1 PLL feature extraction smoke",
        "smoke_cohort_present": True,
        "per_smiles_features": {},
        "aggregate": None,
        "adapter_available": _ADAPTER_AVAILABLE,
    }
    feats_dict = _smoke_cohort()
    if feats_dict is None:
        result["smoke_cohort_present"] = False
        result["honest_framing"] = (
            "RDKit is unavailable in the current environment; "
            "smoke cohort not built. PLL feature adapter is importable "
            "(canonical module also needed for runtime)."
        )
    else:
        result["per_smiles_features"] = {
            smi: vec.tolist() for smi, vec in feats_dict.items()
        }

        # Build a small aggregation over the cohort
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            cohort = []
            for smi in feats_dict.keys():
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mol_h = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol_h, randomSeed=42) != 0:
                    params = AllChem.ETKDGv2()
                    AllChem.EmbedMolecule(mol_h, params)
                cohort.append(mol_h)
            if cohort:
                agg = aggregate_pll_distribution(cohort, top_k=10)
                result["aggregate"] = agg.as_dict()
        except Exception as exc:  # pragma: no cover — defensive
            result["aggregate_error"] = repr(exc)

    out_path = out_dir / "smoke_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    result["output_path"] = str(out_path)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-382-A1 PLL feature extraction smoke script (CPU-only SCAFFOLD). "
            "Wraps molmetal.adapters.flow_matching_lipman.pll_features."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke cohort (benzene + methane + ethanol + chlorobenzene).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "molmetal" / "reports" / "wf382a1_pll_smoke"),
        help="Output directory for the smoke JSON report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="PLL feature vector length (default 10 per Wf-382-A1 task brief).",
    )
    args = parser.parse_args()

    if not args.smoke:
        parser.print_help()
        return 2

    out_dir = Path(args.out_dir)
    result = _write_smoke_report(out_dir)

    print(f"Wf-382-A1 PLL smoke run complete:")
    print(f"  output_path = {result.get('output_path', '<not written>')}")
    print(f"  adapter_available = {result['adapter_available']}")
    print(f"  smoke_cohort_present = {result['smoke_cohort_present']}")
    if result.get("per_smiles_features"):
        print(f"  per_smiles_features (n={len(result['per_smiles_features'])}):")
        for smi, vec in result["per_smiles_features"].items():
            print(f"    {smi:>16s}  top-3 eigvals = {vec[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
