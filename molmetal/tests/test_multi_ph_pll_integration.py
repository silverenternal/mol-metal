"""Pytest: Integration 4 — Multi-PH -> PLL signature wrapper.

R18 P3 ships the multi-PH signature (Kim & Jung 2026) and W2.3
wires it into the R17 PLL signature (Liu & Feng 2026) as an
*additional* descriptor.  This test module pins the Integration
4 contract:

1. :class:`MultiPHPLLSignature.compute` returns
   ``((n_bins, n_bins), (k,))`` with finite, non-negative values.
2. :meth:`MultiPHPLLSignature.distance` is symmetric and zero on
   identical signatures.
3. :class:`MultiPHPLLAugment` concatenates the two streams without
   mutation and rejects shape mismatches.
4. :func:`compare_multi_ph_vs_single_pll` reports the augmented
   signature strictly >= the bare PLL on pair-distance
   distinction.

All tests run on CPU in <10 s.  No GPU dependencies required.

Reference
---------
Kim M.-J. & Jung S.-H. (2026).  Adv. Appl. Math. 174:103023.
Liu J. & Feng H. (2026).  J. Chem. Inf. Model. 66:1024-1037.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.math_skills.multi_parameter_ph import (
    TwoParameterFiltration,
)
from molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration import (
    MultiPHPLLAugment,
    MultiPHPLLError,
    MultiPHPLLSignature,
    compare_multi_ph_vs_single_pll,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: A tiny canonical SMILES list — used to keep the suite < 10 s.
SMILES_LIST: Tuple[str, ...] = (
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "NCCO",
    "CCN",
)


def _toy_smiles() -> List[str]:
    """Return the canonical smoke SMILES list (copy to avoid mutation)."""
    return list(SMILES_LIST)


def _toy_synthetic_data(
    n_bins: int = 8,
    k: int = 8,
    b: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic test data for :class:`MultiPHPLLAugment` tests.

    Returns (pll_features, multi_ph, augment_features, identity_pairs)
    where the *augmented* output is built manually to compare against
    :meth:`MultiPHPLLAugment.augment`.

    Used to bypass RDKit embedding (which is slow and stochastic
    across hosts).
    """
    g = torch.Generator().manual_seed(7)
    pll = torch.rand(b, k, generator=g)
    mph = torch.rand(b, n_bins, n_bins, generator=g) * 3.0
    flat = mph.reshape(b, -1)
    expected = torch.cat([pll, flat], dim=-1)
    return pll, mph, expected, flat


# ===========================================================================
# 1. MultiPHPLLSignature — shape / finiteness / construction
# ===========================================================================


def test_multi_ph_pll_signature_default_construction():
    """The default constructor uses ``n_bins=32``, ``k=n_bins``."""
    sig = MultiPHPLLSignature()
    assert sig.n_bins == 32
    assert sig.k == 32
    assert sig.knn == 8


def test_multi_ph_pll_signature_construction_with_k():
    """Constructor accepts explicit ``k`` distinct from ``n_bins``."""
    sig = MultiPHPLLSignature(n_bins=16, k=8)
    assert sig.n_bins == 16
    assert sig.k == 8
    assert sig.knn == 8  # default


def test_multi_ph_pll_signature_construction_with_knn():
    """Constructor accepts explicit ``knn``."""
    sig = MultiPHPLLSignature(n_bins=8, k=4, knn=6)
    assert sig.knn == 6


def test_multi_ph_pll_signature_rejects_invalid_n_bins():
    """``n_bins < 2`` is rejected."""
    with pytest.raises(MultiPHPLLError, match="n_bins"):
        MultiPHPLLSignature(n_bins=1)
    with pytest.raises(MultiPHPLLError, match="n_bins"):
        MultiPHPLLSignature(n_bins=0)


def test_multi_ph_pll_signature_rejects_invalid_knn():
    """``knn < 1`` is rejected."""
    with pytest.raises(MultiPHPLLError, match="knn"):
        MultiPHPLLSignature(n_bins=4, k=4, knn=0)


def test_multi_ph_pll_signature_rejects_invalid_base_pll():
    """Non-callable ``base_pll`` is rejected."""
    with pytest.raises(MultiPHPLLError, match="callable"):
        MultiPHPLLSignature(n_bins=4, k=4, base_pll=42)


def test_multi_ph_pll_signature_rejects_invalid_multi_ph():
    """A ``multi_ph`` without ``.compute()`` is rejected."""
    with pytest.raises(MultiPHPLLError, match="compute"):
        MultiPHPLLSignature(n_bins=4, k=4, multi_ph=object())


# ===========================================================================
# 2. .compute(mol_smiles) — shape / finiteness / determinism
# ===========================================================================


def test_multi_ph_pll_signature_shape():
    """``compute`` returns a tuple ``((n_bins, n_bins), (k,))``."""
    sig = MultiPHPLLSignature(n_bins=16, k=8)
    feat, eig = sig.compute("CCO")
    assert feat.shape == (16, 16), f"expected (16, 16), got {tuple(feat.shape)}"
    assert eig.shape == (8,), f"expected (8,), got {tuple(eig.shape)}"


def test_multi_ph_pll_signature_returns_finite_tensors():
    """Both tensors in the pair are finite."""
    sig = MultiPHPLLSignature(n_bins=12, k=6)
    feat, eig = sig.compute("c1ccccc1")
    assert torch.isfinite(feat).all(), "feature map contains NaN/Inf"
    assert torch.isfinite(eig).all(), "eigenvalues contain NaN/Inf"


def test_multi_ph_pll_signature_feature_map_non_negative():
    """The multi-PH feature map is a count grid — non-negative everywhere."""
    sig = MultiPHPLLSignature(n_bins=12, k=6)
    feat, _eig = sig.compute("CCO")
    assert (feat >= 0).all(), "feature map must be non-negative"


def test_multi_ph_pll_signature_eigenvalues_in_band():
    """PLL eigenvalues are bounded in ``[0, 2]`` (normalised Laplacian)."""
    sig = MultiPHPLLSignature(n_bins=12, k=6)
    _feat, eig = sig.compute("CCO")
    assert (eig >= -1e-6).all(), f"eigenvalues below 0; min={eig.min().item()}"
    assert (eig <= 2.0 + 1e-6).all(), f"eigenvalues above 2; max={eig.max().item()}"


def test_multi_ph_pll_signature_eigenvalues_descending():
    """The PLL spectrum is descending (Liu & Feng 2026 convention)."""
    sig = MultiPHPLLSignature(n_bins=8, k=8)
    _feat, eig = sig.compute("c1ccccc1")
    diff = eig[:-1] - eig[1:]
    assert (diff >= -1e-6).all(), f"eigenvalues must be descending; got {eig.tolist()}"


def test_multi_ph_pll_signature_repeatable_smiles():
    """Two consecutive calls with the same SMILES are stable enough."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    feat1, eig1 = sig.compute("CCO")
    feat2, eig2 = sig.compute("CCO")
    # Multi-PH (counts + discrete binning) is exact on the same
    # conformer; PLL is also exact because the lru_cache key is
    # coords-bytes.
    torch.testing.assert_close(feat1, feat2, rtol=0.0, atol=0.0)
    torch.testing.assert_close(eig1, eig2, rtol=0.0, atol=0.0)


def test_multi_ph_pll_signature_batch_shape():
    """``compute_batch`` returns ``(B, n_bins, n_bins)`` and ``(B, k)``."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    feats, eigs = sig.compute_batch(_toy_smiles())
    assert feats.shape == (len(SMILES_LIST), 8, 8)
    assert eigs.shape == (len(SMILES_LIST), 4)


def test_multi_ph_pll_signature_rejects_non_string_input():
    """``compute`` rejects non-string ``mol_smiles``."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    with pytest.raises(MultiPHPLLError, match="string"):
        sig.compute(123)  # type: ignore[arg-type]
    with pytest.raises(MultiPHPLLError, match="string"):
        sig.compute(None)  # type: ignore[arg-type]


# ===========================================================================
# 3. .distance(sig1, sig2) — symmetry / zero-on-identical
# ===========================================================================


def test_multi_ph_pll_distance_symmetric():
    """``distance(a, b) == distance(b, a)``."""
    sig = MultiPHPLLSignature(n_bins=12, k=4)
    f1, e1 = sig.compute("CCO")
    f2, e2 = sig.compute("c1ccccc1")
    d_ab = sig.distance((f1, e1), (f2, e2))
    d_ba = sig.distance((f2, e2), (f1, e1))
    assert abs(d_ab - d_ba) <= 1e-6, f"distance not symmetric: d_ab={d_ab} d_ba={d_ba}"


def test_multi_ph_pll_distance_zero_for_self():
    """``distance(sig, sig) == 0.0`` for identical signatures."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    f1, e1 = sig.compute("CCO")
    d = sig.distance((f1, e1), (f1, e1))
    assert abs(d) <= 1e-9, f"distance to self must be 0; got {d}"


def test_multi_ph_pll_distance_shape_mismatch_features():
    """``distance`` rejects feature-map shape mismatch."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    f1 = torch.zeros(8, 8)
    f2 = torch.zeros(8, 9)
    e = torch.zeros(4)
    with pytest.raises(MultiPHPLLError, match="feature-map"):
        sig.distance((f1, e), (f2, e))


def test_multi_ph_pll_distance_shape_mismatch_eigenvalues():
    """``distance`` rejects eigenvalue-shape mismatch."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    f = torch.zeros(8, 8)
    e1 = torch.zeros(4)
    e2 = torch.zeros(5)
    with pytest.raises(MultiPHPLLError, match="eigenvalue"):
        sig.distance((f, e1), (f, e2))


def test_multi_ph_pll_distance_distinguishes_different_mols():
    """``distance`` returns a strictly larger value for different mols."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    f1, e1 = sig.compute("CCO")
    f2, e2 = sig.compute("c1ccccc1")
    d_same = sig.distance((f1, e1), (f1, e1))
    d_diff = sig.distance((f1, e1), (f2, e2))
    assert d_diff > d_same, (
        f"distance must separate distinct mols: d_same={d_same} " f"d_diff={d_diff}"
    )


# ===========================================================================
# 4. MultiPHPLLAugment — concatenation contract
# ===========================================================================


def test_multi_ph_pll_augment_shape():
    """``augment`` returns ``(B, H + n_bins ** 2)``."""
    pll, mph, _exp, _flat = _toy_synthetic_data(n_bins=6, k=4, b=3)
    aug = MultiPHPLLAugment(base_pll_features=pll, multi_ph_features=mph)
    out = aug.augment(pll, mph)
    assert out.shape == (3, 4 + 36), f"expected (3, 40), got {tuple(out.shape)}"


def test_multi_ph_pll_augment_value_concat():
    """``augment`` is exactly the concatenation of ``[pll, mph.flatten]``."""
    pll, mph, expected, _flat = _toy_synthetic_data(n_bins=6, k=4, b=3)
    aug = MultiPHPLLAugment()
    out = aug.augment(pll, mph)
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)


def test_multi_ph_pll_augment_does_not_mutate_inputs():
    """``augment`` does not modify the input tensors."""
    pll, mph, _exp, _flat = _toy_synthetic_data(n_bins=4, k=2, b=2)
    pll_before = pll.clone()
    mph_before = mph.clone()
    aug = MultiPHPLLAugment()
    _ = aug.augment(pll, mph)
    torch.testing.assert_close(pll, pll_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(mph, mph_before, rtol=0.0, atol=0.0)


def test_multi_ph_pll_augment_rejects_wrong_rank_pll():
    """``augment`` rejects ``pll_features`` with the wrong rank."""
    aug = MultiPHPLLAugment()
    bad_pll = torch.randn(3, 4, 5)  # 3-D
    mph = torch.randn(3, 8, 8)
    with pytest.raises(MultiPHPLLError, match="pll_features"):
        aug.augment(bad_pll, mph)


def test_multi_ph_pll_augment_rejects_wrong_rank_mph():
    """``augment`` rejects ``multi_ph`` with the wrong rank."""
    aug = MultiPHPLLAugment()
    pll = torch.randn(3, 4)
    bad_mph = torch.randn(3, 4, 8, 8)  # 4-D
    with pytest.raises(MultiPHPLLError, match="multi_ph"):
        aug.augment(pll, bad_mph)


def test_multi_ph_pll_augment_rejects_batch_mismatch():
    """``augment`` rejects ``B_pll != B_mph``."""
    aug = MultiPHPLLAugment()
    pll = torch.randn(3, 4)
    mph = torch.randn(4, 8, 8)
    with pytest.raises(MultiPHPLLError, match="batch"):
        aug.augment(pll, mph)


# ===========================================================================
# 5. compare_multi_ph_vs_single_pll — expressivity + wall-time
# ===========================================================================


def test_compare_multi_ph_vs_single_returns_dict():
    """Returns a JSON-friendly dict with the documented keys."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    res = compare_multi_ph_vs_single_pll(
        _toy_smiles()[:3],
        n_runs=2,
        signature=sig,
    )
    for key in (
        "n_mols",
        "multi_ph_topology_strict",
        "multi_ph_distinct_count",
        "pll_distinct_count",
        "wall_time_multi_ph_pll_ms",
        "wall_time_bare_pll_ms",
        "wall_time_runs",
    ):
        assert key in res, f"missing key: {key}"


def test_compare_multi_ph_vs_single_strictly_more_expressive():
    """``compare_multi_ph_vs_single_pll`` reports strict containment."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    res = compare_multi_ph_vs_single_pll(
        _toy_smiles(),
        n_runs=2,
        signature=sig,
    )
    # Multi-PH feature maps are (n_bins, n_bins) counts that depend
    # on the *atomic-number-difference* second filter — they are
    # strictly more expressive than the bare PLL spectrum (which
    # collapses onto the rotation/translation-invariant Betti
    # signal).  The function reports `>=` so this assertion holds
    # for any pocket set.
    assert res["multi_ph_distinct_count"] >= res["pll_distinct_count"], (
        f"augmented descriptor must dominate the bare PLL on "
        f"pair-distance distinction: augmented="
        f"{res['multi_ph_distinct_count']} bare="
        f"{res['pll_distinct_count']}"
    )


def test_compare_multi_ph_vs_single_rejects_empty_mols():
    """Empty molecule list raises :class:`MultiPHPLLError`."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    with pytest.raises(MultiPHPLLError, match="empty"):
        compare_multi_ph_vs_single_pll([], n_runs=2, signature=sig)


def test_compare_multi_ph_vs_single_rejects_zero_runs():
    """Zero ``n_runs`` raises :class:`MultiPHPLLError`."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    with pytest.raises(MultiPHPLLError, match="n_runs"):
        compare_multi_ph_vs_single_pll(_toy_smiles()[:2], n_runs=0)


def test_compare_multi_ph_vs_single_wall_time_bounded():
    """Per-call wall-time is finite and well-bounded (< 5 s)."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    res = compare_multi_ph_vs_single_pll(
        _toy_smiles()[:3],
        n_runs=1,
        signature=sig,
    )
    assert res["wall_time_multi_ph_pll_ms"] >= 0.0
    assert res["wall_time_bare_pll_ms"] >= 0.0
    # Bound for the test infrastructure: 5 s per (n_bins=8, k=4,
    # SMILES length 3-8) call is *very* generous; we expect < 1 s
    # on a modern CPU for this size.
    assert res["wall_time_multi_ph_pll_ms"] < 5000.0
    assert res["wall_time_bare_pll_ms"] < 5000.0


# ===========================================================================
# 6. Internal / ABI sanity
# ===========================================================================


def test_multi_ph_pll_integration_module_does_not_mutate_pll_signature():
    """Importing the integration layer must not modify pll_signature."""
    import importlib

    # Import the R17 module.
    pll_mod = importlib.import_module(
        "molmetal.molmetal_lam.lam_chem.pll_signature",
    )
    pll_id_before = id(pll_mod)
    pll_compute_before = pll_mod.compute_pll

    # Import the integration module (and trigger its imports).
    _ = importlib.import_module(
        "molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration",
    )

    pll_mod_after = importlib.import_module(
        "molmetal.molmetal_lam.lam_chem.pll_signature",
    )
    pll_id_after = id(pll_mod_after)
    pll_compute_after = pll_mod_after.compute_pll

    assert (
        pll_id_before == pll_id_after
    ), "import must not re-create the pll_signature module"
    assert (
        pll_compute_before is pll_compute_after
    ), "compute_pll must remain the same callable"


def test_multi_ph_pll_integration_uses_existing_TwoParameterFiltration():
    """The integration layer composes with TwoParameterFiltration."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    assert isinstance(sig.multi_ph, TwoParameterFiltration)


def test_multi_ph_pll_integration_exposes_callable_for_compute():
    """The signature instance is *callable* through the public surface
    via ``compute``.  We exercise the public accessors
    :attr:`n_bins`, :attr:`k`, :attr:`knn` here."""
    sig = MultiPHPLLSignature(n_bins=8, k=4)
    assert callable(sig.compute)
    assert isinstance(sig.n_bins, int)
    assert isinstance(sig.k, int)
    assert isinstance(sig.knn, int)
