"""BLK-PA12 regression: ``ScaffoldSplitter`` must emit real Bemis-Murcko keys.

Before 2026-10-12 ``ScaffoldSplitter._scaffold_for`` called
``MurckoScaffold.GetScaffoldForMol(mol, includeChirality=...)``.  On the
installed rdkit (2026.03.6) that binding takes a *single* argument, so the call
raised ``TypeError`` for EVERY molecule; the surrounding bare ``except`` turned
that into an empty scaffold, and ``__call__`` then keyed every row by its own
``__empty__<smiles>`` — i.e. the "scaffold" split was silently a molecule-level
(random) split, on every cohort (MetalCytoToxDB, nci60, platinai, tmQM A/B).

These tests fail loudly if that degeneracy regresses, or if the desalt step for
multi-fragment metal-organic rows (``ligand.solvent.counterion``) is dropped.
They are CPU-only and need no data files.
"""

from __future__ import annotations

import numpy as np

from molmetal.data.splits import ScaffoldSplitter, _bemis_murcko_key


class _SmilesOnly:
    """Minimal shim exposing the ``.smiles`` / ``__len__`` the splitter reads."""

    def __init__(self, smiles):
        self.smiles = np.asarray(smiles, dtype=object)

    def __len__(self) -> int:
        return len(self.smiles)


def test_single_fragment_organic_scaffold_is_nonempty():
    """The signature bug made this return "" (TypeError swallowed)."""
    assert ScaffoldSplitter()._scaffold_for("Cc1ccccc1") == "c1ccccc1"


def test_multi_fragment_desalts_to_largest_fragment():
    """Whole-molecule Murcko keeps the pyridine ring; desalt must drop it."""
    key = _bemis_murcko_key("c1ccccc1CC.c1ccncc1")
    assert key == "c1ccccc1"
    assert key != "c1ccccc1.c1ccncc1"


def test_splitter_is_not_degenerate_on_multi_fragment_rows():
    smis = [
        "c1ccccc1CC.c1ccncc1",
        "c1ccc2ccccc2c1.CCO",
        "Cc1ccccc1.[Cl-]",
        "c1ccncc1.CCC",
        "O=C(O)c1ccccc1.O",
    ]
    sp = ScaffoldSplitter()
    keys = [sp._scaffold_for(s) for s in smis]
    assert any(keys), "every scaffold key is empty — degeneracy regressed"
    assert len(set(keys)) > 1, (
        "all scaffold keys collapsed to one value — the splitter is "
        "molecule-level, not scaffold-level (BLK-PA12 regressed)"
    )


def test_scaffold_groups_are_disjoint_across_splits():
    """Molecules sharing a scaffold must land in the same split."""
    smis = (
        ["Cc1ccccc1"] * 3
        + ["CCc1ccccc1"] * 3
        + ["c1ccc2ccccc2c1"] * 3
        + ["c1ccncc1"] * 3
        + ["c1ccccc1O"] * 3
    )
    res = ScaffoldSplitter(strategy="largest_first", seed=0)(_SmilesOnly(smis))
    keys = np.array([_bemis_murcko_key(s) for s in smis], dtype=object)
    tr = set(keys[res.train_idx])
    va = set(keys[res.val_idx])
    te = set(keys[res.test_idx])
    assert not (tr & va), f"train/val scaffold overlap: {tr & va}"
    assert not (tr & te), f"train/test scaffold overlap: {tr & te}"
    assert not (va & te), f"val/test scaffold overlap: {va & te}"
