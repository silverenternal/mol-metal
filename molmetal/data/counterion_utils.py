"""Counter-ion feature extraction + ablation configurations.

Counter-ions (e.g. PF6-, Cl-, OTf-) are *not* part of the covalent ligand
shell but in the Krasnov 2026 MetalCytoToxDB pipeline they appear either
in the ``Counterion`` column or embedded in ``SMILES_Ligands``.  Whether
they should be treated as signal or noise is an open question that
Krasnov did not ablate.  This module provides:

* :func:`extract_counterion_features` — extract a 5-d feature vector from
  a counter-ion SMILES fragment.
* :func:`merge_counterion_features` — add those features to a row dict.
* Three :data:`CONFIGS` describing alternative treatments:

  * ``'no_counterion'``               — drop the field entirely.
  * ``'counterion_features'``         — append the 5 features to Morgan FP.
  * ``'counterion_concat_smiles'``    — concatenate the counter-ion as a
                                         second SMILES component (e.g.
                                         ``complex.cation``).

Counter-ion feature vector (5 dims)
-----------------------------------
============== ==================================================
key             meaning
============== ==================================================
``charge``      formal net charge (sum of formal charges)
``mw``          molecular weight (g/mol)
``num_atoms``   heavy atom count
``is_inorganic``1 if no carbon atoms, else 0
``has_metal``   1 if any atom is a metal (Cu/Fe/Zn/...), else 0
``logp``        Crippen logP proxy (RDKit ``MolLogP``)
============== ==================================================

Defaults for invalid/empty SMILES are zero-filled so downstream code can
treat every row uniformly.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from rdkit import Chem  # used in apply_config for counterion_concat_smiles

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# RDKit periodic-table metal symbol set (subset for fast membership test).
_METAL_SYMBOLS = frozenset(
    {
        "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr",
        "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs",
        "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "Fr", "Ra", "Ac", "Ce", "Pr", "Nd", "Pm",
        "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm",
    }
)

FEATURE_NAMES: List[str] = [
    "charge",
    "mw",
    "num_atoms",
    "is_inorganic",
    "has_metal",
    "logp",
]

# Default vector for invalid / empty counter-ion SMILES (all zeros).
DEFAULT_FEATURES: Dict[str, float] = {k: 0.0 for k in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
#: The three ablation configurations.  Use :func:`apply_config` to dispatch.
CONFIGS = ("no_counterion", "counterion_features", "counterion_concat_smiles")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_counterion_features(counterion_smiles: str) -> Dict[str, float]:
    """Extract a 5-feature dict from a counter-ion SMILES fragment.

    Invalid / empty SMILES → :data:`DEFAULT_FEATURES` (all zeros).  This
    keeps downstream feature matrices rectangular and avoids masking.
    """
    if not counterion_smiles or not isinstance(counterion_smiles, str):
        return dict(DEFAULT_FEATURES)

    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

        RDLogger.DisableLog("rdApp.*")
        # Counter-ions may be multi-component (e.g. "F[B-](F)(F)F.O=S(=O)([O-])C(F)(F)F").
        # Take the FIRST component as the "primary" counter-ion; this matches
        # the convention used by the Krasnov pipeline (largest fragment).
        primary = counterion_smiles.split(".")[0].strip()
        mol = Chem.MolFromSmiles(primary)
        if mol is None:
            return dict(DEFAULT_FEATURES)

        # Net formal charge
        charge = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        # Molecular weight (g/mol)
        mw = float(Descriptors.MolWt(mol))
        # Heavy atom count
        num_atoms = float(mol.GetNumHeavyAtoms())
        # Inorganic flag: 1 iff no carbon
        is_inorganic = float(1.0 if all(a.GetSymbol() != "C" for a in mol.GetAtoms()) else 0.0)
        # Metal flag: 1 iff any heavy atom is a metal
        has_metal = float(
            1.0 if any(a.GetSymbol() in _METAL_SYMBOLS for a in mol.GetAtoms()) else 0.0
        )
        # Crippen logP proxy (clipped to a sane range)
        try:
            logp = float(Descriptors.MolLogP(mol))
        except Exception:
            logp = 0.0
        # Clip extreme values that occasionally come from weird fragments
        logp = float(np.clip(logp, -10.0, 10.0))

        return {
            "charge": charge,
            "mw": mw,
            "num_atoms": num_atoms,
            "is_inorganic": is_inorganic,
            "has_metal": has_metal,
            "logp": logp,
        }
    except Exception:
        return dict(DEFAULT_FEATURES)


def merge_counterion_features(row: Dict[str, object]) -> Dict[str, object]:
    """Return ``row`` augmented with the 5-dim counter-ion feature vector.

    Reads ``row['Counterion']`` (or ``row['counterion']``) — whichever is
    present.  Missing/NaN values yield all-zero features (i.e. the model
    is told "no counter-ion info", not "no counter-ion").
    """
    raw = row.get("Counterion", row.get("counterion", ""))
    smiles = "" if raw is None else str(raw)
    # NaN check (pandas may surface as nan)
    try:
        if isinstance(raw, float) and np.isnan(raw):
            smiles = ""
    except Exception:
        pass
    feats = extract_counterion_features(smiles)
    out: Dict[str, object] = dict(row)
    for k, v in feats.items():
        out[f"counterion_{k}"] = float(v)
    out["counterion_smiles"] = smiles
    return out


# ---------------------------------------------------------------------------
# Config application
# ---------------------------------------------------------------------------
def apply_config(
    smiles_list: Sequence[str],
    counterion_list: Sequence[str],
    config: str,
    morgan_radius: int = 2,
    morgan_nbits: int = 2048,
) -> np.ndarray:
    """Build feature matrix under one of the three ablation configs.

    Parameters
    ----------
    smiles_list : sequence of str
        Ligand SMILES (one per row).
    counterion_list : sequence of str
        Counter-ion SMILES aligned to ``smiles_list``.  May contain
        empty strings / NaN.
    config : {'no_counterion', 'counterion_features', 'counterion_concat_smiles'}

    Returns
    -------
    features : np.ndarray, shape (N, K)
        K = ``morgan_nbits`` for ``no_counterion``;
        K = ``morgan_nbits + len(FEATURE_NAMES)`` for ``counterion_features``;
        K = ``morgan_nbits`` for ``counterion_concat_smiles`` (FP computed
        over the concatenated multi-component SMILES).
    """
    from molmetal.baselines.eval_utils import morgan_features

    if config not in CONFIGS:
        raise ValueError(
            f"Unknown counterion config {config!r}; must be one of {CONFIGS}"
        )

    smiles_list = ["" if s is None else str(s) for s in smiles_list]
    counterion_list = [
        "" if c is None or (isinstance(c, float) and np.isnan(c)) else str(c)
        for c in counterion_list
    ]

    if config == "no_counterion":
        # Counter-ion ignored entirely (Krasnov baseline).
        return morgan_features(smiles_list, morgan_radius, morgan_nbits)

    if config == "counterion_features":
        fp = morgan_features(smiles_list, morgan_radius, morgan_nbits).astype(np.float32)
        n = len(smiles_list)
        extra = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
        for i, c in enumerate(counterion_list):
            feats = extract_counterion_features(c)
            for j, k in enumerate(FEATURE_NAMES):
                extra[i, j] = feats[k]
        return np.concatenate([fp, extra], axis=1)

    # config == "counterion_concat_smiles"
    combined = []
    for s, c in zip(smiles_list, counterion_list):
        c_first = c.split(".")[0].strip() if c else ""
        if c_first:
            combined.append(f"{s}.{c_first}")
        else:
            combined.append(s)
    return morgan_features(combined, morgan_radius, morgan_nbits)


__all__ = [
    "FEATURE_NAMES",
    "DEFAULT_FEATURES",
    "CONFIGS",
    "extract_counterion_features",
    "merge_counterion_features",
    "apply_config",
]
