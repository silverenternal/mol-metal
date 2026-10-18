# SPDX-License-Identifier: Apache-2.0
"""Ring-SIZE profile — the axis OPT-2 found decisive.

WHY THIS EXISTS
---------------
OPT-2 (metrics/by_metric/cfm_opt2_ring_content.json, MEASURED 2026-10-13)
compared 66 generated molecules against 2,000 real six-metal complexes:

    fraction of rings that are 5- or 6-membered   gen 0.019  real 0.969
    fraction of rings that are 3-membered         gen 0.898  real 0.005
    aromatic rings per molecule                   gen 0.015  real 3.574

That verdict was produced by a ONE-OFF script reading ONE CSV
(molmetal/scripts/wf_cfm_opt2_ring_content_2026_10_13.py), so neither the
trainer's per-epoch report nor the property panel could show the number that
decides the question.  This module is the shared implementation both now use,
so a training run and an inference run cannot disagree about what a ring is.

DEFINITION
----------
SSSR (RDKit's GetRingInfo().AtomRings()) on a SANITISED molecule.  Shares are
over RINGS, not molecules — the same denominator OPT-2 used, so the numbers
are directly comparable to its verdict block.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


def ring_sizes_from_mol(mol) -> List[int]:
    """SSSR ring sizes of a sanitised RDKit Mol, ascending."""
    return sorted(int(len(r)) for r in mol.GetRingInfo().AtomRings())


def summarise_ring_sizes(per_mol_sizes: Iterable[Iterable[int]]) -> Dict[str, Any]:
    """Aggregate per-molecule ring-size lists into the OPT-2-comparable block."""
    hist: Dict[int, int] = {}
    n_mol = 0
    for sizes in per_mol_sizes:
        n_mol += 1
        for k in sizes:
            hist[int(k)] = hist.get(int(k), 0) + 1
    n_rings = int(sum(hist.values()))
    return {
        "ring_size_distribution": {str(k): int(v) for k, v in sorted(hist.items())},
        "rings_per_mol_mean": float(n_rings / n_mol) if n_mol else 0.0,
        "ring_5_6_share": (
            float((hist.get(5, 0) + hist.get(6, 0)) / n_rings) if n_rings else 0.0
        ),
        "ring_3_share": float(hist.get(3, 0) / n_rings) if n_rings else 0.0,
    }


def ring_profile(smiles_list: Iterable[str]) -> Dict[str, Any]:
    """Ring-size profile for a list of SMILES.

    Unparseable strings are SKIPPED and counted in n_unparsed rather than
    silently treated as ring-free molecules — an empty profile means "nothing
    parsed", which is a different statement from "no rings".
    """
    from rdkit import Chem

    sizes: List[List[int]] = []
    n_unparsed = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            n_unparsed += 1
            continue
        sizes.append(ring_sizes_from_mol(mol))
    out = summarise_ring_sizes(sizes)
    out["n_molecules"] = len(sizes)
    out["n_unparsed"] = n_unparsed
    return out
