"""Complex-level ligand featurization (FIX-L1 / DEF-1).

The MSRS cohort stores a metal complex as ``ligand_canonical``: a dot-separated
SMILES of the main ligand **plus** ancillary ligands and counterions, with the
metal itself in a separate column.  ``ligand_main_fragment`` keeps only the
largest organic fragment.

Measured on 2026-10-18 (21,404 exact-potency rows):

* 99.93% of rows have more than one component;
* 5,861 full complexes collapse into 2,846 main fragments;
* ECFP4 on the full complex instead of the main fragment moves R2 from
  0.3366 to 0.4368 under the historical fragment-grouped CV (+0.1002,
  paired p = 2.3e-20), and from 0.3972 to 0.5012 under the stricter
  complex-grouped CV.

So the ancillary ligands carry real signal, not just complex identity.  This
module exposes the blocks separately so an experiment can ask *which* part
matters instead of re-deriving the whole matrix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

#: Components with fewer heavy atoms are counterions / solvent, not ligands.
#: Kept identical to ``msrs_build_endpoints.MIN_LIGAND_HEAVY_ATOMS`` so the
#: "main fragment" here is the same object the rest of the pipeline uses.
MIN_LIGAND_HEAVY_ATOMS = 6


def _rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def component_smiles(smiles: object) -> List[str]:
    """Canonical SMILES of every component, largest (by heavy atoms) first.

    Returns an empty list if the string does not parse.  Sorting by heavy-atom
    count makes ``components[0]`` the main fragment by construction, which is
    exactly how ``ligand_main_fragment`` was defined.
    """
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return []
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return []
    ordered = sorted(frags, key=lambda f: (-f.GetNumHeavyAtoms(), f.GetNumAtoms()))
    return [Chem.MolToSmiles(f) for f in ordered]


def ancillary_smiles(smiles: object) -> Optional[str]:
    """Every component except the largest, joined with dots.

    Returns ``""`` (not ``None``) when the molecule is a single component, so a
    caller can distinguish "no ancillary ligands" from "unparseable".
    Returns ``None`` only when the SMILES does not parse.
    """
    comps = component_smiles(smiles)
    if not comps:
        return None
    return ".".join(comps[1:])


def main_component(smiles: object) -> Optional[str]:
    """Canonical SMILES of the largest component, or ``None``."""
    comps = component_smiles(smiles)
    return comps[0] if comps else None


def _morgan(smiles: Sequence[str], n_bits: int, radius: int) -> sparse.csr_matrix:
    """ECFP-style binary fingerprint; an unparseable SMILES is an all-zero row."""
    from molmetal.scripts.msrs_surface_model import morgan_matrix

    return morgan_matrix(list(smiles), n_bits=n_bits, radius=radius)


def _counterion_onehot(
    counterions: Sequence[object], vocabulary: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """One-hot over a fixed counterion vocabulary, plus an ``other`` column."""
    index = {c: i for i, c in enumerate(vocabulary)}
    other = len(vocabulary)
    out = np.zeros((len(counterions), len(vocabulary) + 1), dtype=np.float32)
    for i, raw in enumerate(counterions):
        key = "" if raw is None else str(raw)
        out[i, index.get(key, other)] = 1.0
    names = [f"counterion_{c}" for c in vocabulary] + ["counterion_other"]
    return out, names


def complex_ligand_blocks(
    frame,
    *,
    n_bits: int = 2048,
    radius: int = 2,
    counterion_vocabulary: Optional[Sequence[str]] = None,
    include: Sequence[str] = ("canonical", "main", "ancillary", "counterion", "n_fragments"),
) -> Tuple[sparse.csr_matrix, List[str], Dict[str, Tuple[int, int]]]:
    """Build the ligand block for a frame with ``ligand_canonical``.

    Parameters
    ----------
    frame
        Must carry ``ligand_canonical``.  ``ligand_main_fragment`` and
        ``counterion`` are optional; the main fragment is recomputed from the
        canonical SMILES so the block cannot drift from the cohort definition.
    include
        Which sub-blocks to emit, in this fixed order.  The default is the
        full complex representation; ablations pass a subset.

    Returns
    -------
    X, names, slices
        ``slices`` maps block name -> ``(start, stop)`` so a caller can point
        at a block without re-deriving column offsets.
    """
    if "ligand_canonical" not in frame.columns:
        raise KeyError("complex_ligand_blocks: frame needs a 'ligand_canonical' column")

    canonical = [("" if s is None else str(s)) for s in frame["ligand_canonical"]]
    main = [main_component(s) or "" for s in canonical]
    ancillary = [ancillary_smiles(s) or "" for s in canonical]

    parts: List[sparse.csr_matrix] = []
    names: List[str] = []
    slices: Dict[str, Tuple[int, int]] = {}

    def add_block(key: str, block: sparse.csr_matrix, block_names: List[str]) -> None:
        start = sum(p.shape[1] for p in parts)
        parts.append(block)
        names.extend(block_names)
        slices[key] = (start, start + block.shape[1])

    if "canonical" in include:
        add_block(
            "canonical",
            _morgan(canonical, n_bits, radius),
            [f"ecfp4_complex_{i}" for i in range(n_bits)],
        )
    if "main" in include:
        add_block(
            "main",
            _morgan(main, n_bits, radius),
            [f"ecfp4_main_{i}" for i in range(n_bits)],
        )
    if "ancillary" in include:
        add_block(
            "ancillary",
            _morgan(ancillary, n_bits, radius),
            [f"ecfp4_ancillary_{i}" for i in range(n_bits)],
        )
    if "counterion" in include:
        vocab = list(counterion_vocabulary or [])
        if "counterion" in frame.columns:
            dense, cnames = _counterion_onehot(frame["counterion"].tolist(), vocab)
            add_block("counterion", sparse.csr_matrix(dense), cnames)
        else:
            dense = np.zeros((len(frame), len(vocab) + 1), dtype=np.float32)
            add_block(
                "counterion",
                sparse.csr_matrix(dense),
                [f"counterion_{c}" for c in vocab] + ["counterion_other"],
            )
    if "n_fragments" in include:
        counts = np.array([len(component_smiles(s)) for s in canonical], dtype=np.float32)
        add_block(
            "n_fragments",
            sparse.csr_matrix(counts.reshape(-1, 1)),
            ["n_components"],
        )

    if not parts:
        raise ValueError("complex_ligand_blocks: 'include' selected no blocks")
    X = sparse.hstack(parts, format="csr")
    assert X.shape[1] == len(names), (X.shape, len(names))
    return X, names, slices
