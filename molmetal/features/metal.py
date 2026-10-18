"""Metal block (FIX-P1 step 2).

Reproduces ``msrs_surface_model.metal_matrix`` bit-for-bit: a one-hot over the
five-cohort metal vocabulary followed by the tabulated descriptors.  A metal
absent from the descriptor CSV gets ``NaN`` rather than a zero — a zero says
"this metal has ionization energy 0", which is a lie a tree model will happily
split on.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Cohort vocabulary, in the order the one-hot columns appear.
METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")


def metal_block(
    metals: Sequence[str],
    descriptors: pd.DataFrame,
    *,
    columns: Sequence[str] = ("ie1_kj", "ie2_kj", "pauling_en"),
    one_hot: bool = True,
    vocabulary: Sequence[str] = METALS,
) -> Tuple[np.ndarray, List[str]]:
    """Dense ``(n, n_metals + n_descriptors)`` float32 block + column names."""
    desc = descriptors.set_index("metal_symbol")
    parts: List[np.ndarray] = []
    names: List[str] = []

    if one_hot:
        onehot = np.zeros((len(metals), len(vocabulary)), dtype=np.float32)
        index = {m: i for i, m in enumerate(vocabulary)}
        for i, m in enumerate(metals):
            j = index.get(m)
            if j is not None:
                onehot[i, j] = 1.0
        parts.append(onehot)
        names.extend(f"metal_onehot_{m}" for m in vocabulary)

    if columns:
        cols = list(columns)
        extra = np.array(
            [
                [desc.at[m, c] if m in desc.index else np.nan for c in cols]
                for m in metals
            ],
            dtype=np.float32,
        )
        parts.append(extra)
        # Historical names, not the CSV spelling: "metal_ie1" / "metal_ie2" /
        # "metal_en" is what every existing artifact's feature list says.
        names.extend(_short_name(c) for c in cols)

    if not parts:
        raise ValueError("metal_block: neither one_hot nor columns selected")
    return np.hstack(parts), names


def _short_name(column: str) -> str:
    aliases = {
        "ie1_kj": "metal_ie1",
        "ie2_kj": "metal_ie2",
        "pauling_en": "metal_en",
    }
    return aliases.get(column, f"metal_{column}")


def state_block(
    frame,
    *,
    metal_col: str = "metal",
    os_col: str = "oxidation_state",
    charge_col: str = "charge_complex",
    valence_col: str = "valence_e",
) -> Tuple[np.ndarray, List[str]]:
    """Oxidation state, complex charge, d-electron count, valence electrons.

    The one variable that changes *within* a metal.  ``metal_block``'s one-hot
    cannot express Ru(II) vs Ru(III), and that difference drives d-electron
    count, LFSE and redox potential — the axis FIX-L2 measured at +0.0444 R2.

    ``d_electrons`` comes from :mod:`molmetal.data.metal_graph` (group number
    minus oxidation state, clipped), so this block does not invent a table.
    """
    from molmetal.data.metal_graph import d_electrons

    os_ = frame[os_col].to_numpy(dtype=np.float32)
    charge = frame[charge_col].to_numpy(dtype=np.float32)
    valence = frame[valence_col].to_numpy(dtype=np.float32)
    metals = frame[metal_col].astype(str).tolist()
    d_e = np.array(
        [d_electrons(m, int(o)) for m, o in zip(metals, os_)], dtype=np.float32
    )
    out = np.column_stack([os_ / 4.0, charge / 4.0, d_e / 10.0, valence / 8.0])
    return out.astype(np.float32), [
        "metal_os",
        "metal_charge",
        "metal_d_electrons",
        "metal_valence_e",
    ]


def load_descriptors(path: Optional[object] = None) -> pd.DataFrame:
    """Read ``metal_chemistry_5.csv``; no silent fallback if it is missing."""
    from pathlib import Path

    target = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "data" / "metal_chemistry_5.csv"
    )
    if not target.is_file():
        raise FileNotFoundError(f"metal descriptor table missing: {target}")
    return pd.read_csv(target)
