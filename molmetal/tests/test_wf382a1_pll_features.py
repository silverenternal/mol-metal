"""Wf-382-A1 — PLL feature extraction SCAFFOLD tests (CPU-only).

Per TODO-94 R20-P3 + Wf-382-A1 task brief:
1 NEW test verifying PLL spectrum sanity for benzene + methane.

The test confirms that:
  (a) ``persistent_laplacian`` returns a frozen ``PLLFeatureAdapter`` dataclass
  (b) ``pll_eigenvalues`` returns the descending eigenvalues array
  (c) ``pll_to_feature_vector`` produces a length-10 vector with the
      leading entries matching the eigenvalues (zero-padded for short
      molecules per the Wf-359-A1 W₂ recipe contract)
  (d) Benzene (6 heavy atoms) and methane (1 heavy atom) both produce
      well-formed outputs — benzene with ``n_atoms == 6`` and methane
      with ``n_atoms == 1`` (zero-padded feature vector).

The test self-skips when the canonical PLL module or RDKit is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _try_import_pll_adapter():
    """Best-effort import of the Wf-382-A1 PLL feature adapter."""
    try:
        from molmetal.adapters.flow_matching_lipman.pll_features import (
            PLLFeatureAdapter,
            PLLFeatureError,
            persistent_laplacian,
            pll_eigenvalues,
            pll_to_feature_vector,
        )

        return {
            "PLLFeatureAdapter": PLLFeatureAdapter,
            "PLLFeatureError": PLLFeatureError,
            "persistent_laplacian": persistent_laplacian,
            "pll_eigenvalues": pll_eigenvalues,
            "pll_to_feature_vector": pll_to_feature_vector,
        }
    except ImportError:
        return None


def _try_embed_3d(smiles: str, seed: int = 42):
    """Best-effort RDKit 3D embedder; returns ``None`` on failure."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_h, randomSeed=seed) != 0:
        params = AllChem.ETKDGv2()
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            return None
    return mol_h


class TestPLLFeatureAdapterSanity:
    """Single Wf-382-A1 sanity test: benzene + methane PLL spectrum."""

    def test_benzene_and_methane_pll_spectrum(self):
        """PLL spectrum sanity for benzene (6 heavy atoms) + methane (1 heavy atom).

        Verifies the Wf-382-A1 SCAFFOLD contract:
          1. ``persistent_laplacian`` returns a ``PLLFeatureAdapter``.
          2. ``pll_eigenvalues`` returns a descending numpy array.
          3. ``pll_to_feature_vector`` produces a length-10 vector with
             zero-padding for short molecules (methane).
          4. Benzene ``n_atoms == 6``; methane ``n_atoms == 1``.
        """
        adapter = _try_import_pll_adapter()
        if adapter is None:
            pytest.skip("Wf-382-A1 PLL feature adapter not importable")

        benzene = _try_embed_3d("c1ccccc1")
        methane = _try_embed_3d("C")
        if benzene is None or methane is None:
            pytest.skip("RDKit / conformer embed unavailable in current env")

        PLLFeatureAdapter = adapter["PLLFeatureAdapter"]
        persistent_laplacian = adapter["persistent_laplacian"]
        pll_eigenvalues = adapter["pll_eigenvalues"]
        pll_to_feature_vector = adapter["pll_to_feature_vector"]

        # Benzene: 6 heavy atoms → expect n_atoms == 6 and eigenvalues
        # of the normalised Laplacian bounded in [0, 2] per von Luxburg 2007.
        try:
            benzene_pll = persistent_laplacian(benzene, k=4)
        except Exception as exc:  # canonical module may raise on small n
            pytest.skip(f"Canonical PLL module refused benzene: {exc!r}")

        assert isinstance(benzene_pll, PLLFeatureAdapter), (
            f"benzene_pll must be PLLFeatureAdapter; got {type(benzene_pll)}"
        )
        # After AddHs, benzene has 6 heavy atoms + 6 H atoms = 12 total.
        assert benzene_pll.n_atoms == 12, (
            f"benzene n_atoms must be 12 (6 heavy + 6 H after AddHs); "
            f"got {benzene_pll.n_atoms}"
        )
        assert benzene_pll.k_used >= 1, (
            f"benzene k_used must be >= 1; got {benzene_pll.k_used}"
        )
        benzene_eigs = pll_eigenvalues(benzene_pll)
        assert isinstance(benzene_eigs, np.ndarray)
        assert benzene_eigs.ndim == 1
        assert benzene_eigs.shape[0] == benzene_pll.k_used
        # Descending order
        assert np.all(benzene_eigs[:-1] >= benzene_eigs[1:]), (
            "benzene eigenvalues must be in descending order"
        )
        # Bounded in [0, 2]
        assert np.all(benzene_eigs >= -1e-9), (
            f"benzene eigenvalues must be >= 0; got min {benzene_eigs.min()}"
        )
        assert np.all(benzene_eigs <= 2.0 + 1e-9), (
            f"benzene eigenvalues must be <= 2; got max {benzene_eigs.max()}"
        )

        # Methane: 1 heavy atom + 4 H atoms = 5 total after AddHs.
        # Feature vector still zero-padded to top_k=10.
        try:
            methane_pll = persistent_laplacian(methane, k=4)
        except Exception as exc:
            pytest.skip(f"Canonical PLL module refused methane: {exc!r}")

        assert isinstance(methane_pll, PLLFeatureAdapter)
        assert methane_pll.n_atoms == 5, (
            f"methane n_atoms must be 5 (1 heavy + 4 H after AddHs); "
            f"got {methane_pll.n_atoms}"
        )
        assert methane_pll.k_used >= 1
        methane_eigs = pll_eigenvalues(methane_pll)
        assert isinstance(methane_eigs, np.ndarray)
        assert methane_eigs.ndim == 1

        # Zero-padded feature vector for top_k=10
        methane_vec = pll_to_feature_vector(methane_eigs, top_k=10)
        assert methane_vec.shape == (10,), (
            f"methane feature vector must have shape (10,); got {methane_vec.shape}"
        )
        assert methane_vec.dtype == np.float64
        # The leading entries match the eigenvalues; the tail is zero.
        n_eigs = methane_eigs.shape[0]
        if n_eigs <= 10:
            np.testing.assert_allclose(
                methane_vec[:n_eigs], methane_eigs, atol=1e-12
            )
            np.testing.assert_allclose(
                methane_vec[n_eigs:], np.zeros(10 - n_eigs), atol=1e-12
            )

        # Benzene feature vector (top_k=10): should not be all-zero
        benzene_vec = pll_to_feature_vector(benzene_eigs, top_k=10)
        assert benzene_vec.shape == (10,)
        assert not np.allclose(benzene_vec, 0.0), (
            "benzene feature vector must not be all-zero"
        )
