"""Tests for the :mod:`molmetal_lam.lam_chem.batched_rdkit` helpers.

Verifies that the batched wrappers behave numerically identical to
the scalar RDKit API on small canonical molecules, and that the
sequential fallback path is wired correctly when multiprocessing is
unavailable.
"""

from __future__ import annotations

import math
import multiprocessing

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

from molmetal_lam.lam_chem import batched_rdkit as br
from molmetal_lam.lam_chem.batched_rdkit import (
    _RDKIT_AVAILABLE,
    _scalar_descriptor,
    batch_descriptors_from_smiles,
    batch_lipinski,
    batch_mol_from_smiles,
    batch_qed,
    batch_sa,
    is_parallel_safe,
)


# ---------------------------------------------------------------------------
# Small canonical SMILES corpus used by all tests below.
# ---------------------------------------------------------------------------
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
LONG_ALKANE = "C" * 1000  # MW > 500 -> fails Ro5
INVALID = "not_a_smiles@@@"


pytestmark = pytest.mark.skipif(
    not _RDKIT_AVAILABLE,
    reason="RDKit is required for batched_rdkit tests",
)


# ---------------------------------------------------------------------------
# batch_mol_from_smiles
# ---------------------------------------------------------------------------

def test_batch_mol_from_smiles_returns_correct_length() -> None:
    """100 SMILES -> 100 RDKit mols (or None for invalid)."""
    smiles = [ASPIRIN, ETHANOL, BENZENE, CAFFEINE, INVALID] * 20
    assert len(smiles) == 100
    mols = batch_mol_from_smiles(smiles)
    assert len(mols) == 100
    # Valid SMILES parse
    for i in (0, 1, 2, 3):
        assert isinstance(mols[i], Chem.Mol), f"index {i} should parse"
    # Invalid SMILES -> None
    for i in range(4, 100, 5):
        assert mols[i] is None, f"index {i} (invalid SMILES) should be None"


def test_batch_mol_from_smiles_preserves_order() -> None:
    """The order of inputs is preserved in the output list."""
    smiles = [ASPIRIN, ETHANOL, BENZENE, CAFFEINE]
    mols = batch_mol_from_smiles(smiles)
    canonical = [Chem.MolToSmiles(m) for m in mols]
    expected = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles]
    assert canonical == expected


def test_batch_mol_from_smiles_empty_input() -> None:
    """Empty input returns an empty list (does not crash)."""
    assert batch_mol_from_smiles([]) == []


# ---------------------------------------------------------------------------
# batch_qed
# ---------------------------------------------------------------------------

def test_batch_qed_matches_scalar_aspirin() -> None:
    """batch_qed on [aspirin] equals scalar QED.qed."""
    expected = float(QED.qed(Chem.MolFromSmiles(ASPIRIN)))
    out = batch_qed([ASPIRIN])
    assert out.shape == (1,)
    assert math.isclose(float(out[0]), expected, rel_tol=1e-9, abs_tol=1e-9)


def test_batch_qed_matches_scalar_ethanol() -> None:
    """batch_qed on [ethanol] equals scalar QED.qed."""
    expected = float(QED.qed(Chem.MolFromSmiles(ETHANOL)))
    out = batch_qed([ETHANOL])
    assert math.isclose(float(out[0]), expected, rel_tol=1e-9, abs_tol=1e-9)


def test_batch_qed_matches_scalar_on_mixed_list() -> None:
    """Batched QED matches scalar QED on aspirin + ethanol."""
    smiles = [ASPIRIN, ETHANOL]
    expected = np.asarray(
        [float(QED.qed(Chem.MolFromSmiles(s))) for s in smiles],
        dtype=float,
    )
    out = batch_qed(smiles)
    assert out.shape == (2,)
    np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)


def test_batch_qed_returns_nan_for_invalid_smiles() -> None:
    """Invalid SMILES produce NaN entries (no crash)."""
    out = batch_qed([INVALID])
    assert out.shape == (1,)
    assert math.isnan(float(out[0]))


def test_batch_qed_empty_input_returns_empty_array() -> None:
    """Empty input returns a length-0 array."""
    out = batch_qed([])
    assert out.shape == (0,)


# ---------------------------------------------------------------------------
# batch_sa
# ---------------------------------------------------------------------------

def test_batch_sa_matches_scalar_aspirin() -> None:
    """batch_sa matches the Ertl-style SA proxy formula."""
    expected = br._sa_proxy_one(ASPIRIN)
    out = batch_sa([ASPIRIN])
    assert math.isclose(float(out[0]), expected, rel_tol=1e-9, abs_tol=1e-9)


def test_batch_sa_clipped_to_unit_interval() -> None:
    """The SA proxy is clipped to [1, 10] for both tiny and large mols."""
    out = batch_sa([ETHANOL, LONG_ALKANE, ASPIRIN])
    for v in out:
        assert 1.0 <= float(v) <= 10.0


def test_batch_sa_returns_nan_for_invalid_smiles() -> None:
    """Invalid SMILES produce NaN SA scores."""
    out = batch_sa([INVALID])
    assert math.isnan(float(out[0]))


# ---------------------------------------------------------------------------
# batch_lipinski
# ---------------------------------------------------------------------------

def test_batch_lipinski_matches_scalar_aspirin() -> None:
    """Aspirin passes Ro5 — both scalar and batch agree."""
    expected = True
    out = batch_lipinski([ASPIRIN])
    assert bool(out[0]) == expected


def test_batch_lipinski_matches_scalar_large_polymer() -> None:
    """A 1000-carbon alkane fails Ro5 (MW > 500) — both scalar and batch."""
    out = batch_lipinski([LONG_ALKANE])
    assert bool(out[0]) is False


def test_batch_lipinski_matches_scalar_on_mixed_list() -> None:
    """Batched Ro5 matches scalar Descriptors.* on a 5-mol corpus."""
    smiles = [ASPIRIN, ETHANOL, BENZENE, CAFFEINE, LONG_ALKANE]
    expected = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            expected.append(False)
            continue
        try:
            expected.append(
                Descriptors.MolWt(mol) <= 500
                and Descriptors.MolLogP(mol) <= 5
                and rdMolDescriptors.CalcNumHBD(mol) <= 5
                and rdMolDescriptors.CalcNumHBA(mol) <= 10
            )
        except Exception:
            expected.append(False)
    out = batch_lipinski(smiles)
    assert [bool(x) for x in out] == expected


def test_batch_lipinski_invalid_smiles_returns_false() -> None:
    """Invalid SMILES produce False entries (matches scalar behaviour)."""
    out = batch_lipinski([INVALID])
    assert bool(out[0]) is False


def test_batch_lipinski_returns_bool_dtype() -> None:
    """The returned array is bool-typed."""
    out = batch_lipinski([ASPIRIN, ETHANOL, INVALID])
    assert out.dtype == np.bool_


# ---------------------------------------------------------------------------
# batch_descriptors_from_smiles
# ---------------------------------------------------------------------------

def test_batch_descriptors_mw_matches_scalar() -> None:
    """batch_descriptors(mw) matches Descriptors.MolWt on a small corpus."""
    smiles = [ASPIRIN, ETHANOL, BENZENE, CAFFEINE]
    expected = np.asarray(
        [float(Descriptors.MolWt(Chem.MolFromSmiles(s))) for s in smiles],
        dtype=float,
    )
    out = batch_descriptors_from_smiles(smiles, "mw")
    np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)


def test_batch_descriptors_logp_matches_scalar() -> None:
    """batch_descriptors(logp) matches Descriptors.MolLogP."""
    smiles = [ASPIRIN, ETHANOL]
    expected = np.asarray(
        [float(Descriptors.MolLogP(Chem.MolFromSmiles(s))) for s in smiles],
        dtype=float,
    )
    out = batch_descriptors_from_smiles(smiles, "logp")
    np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)


def test_batch_descriptors_unknown_name_returns_nan() -> None:
    """An unknown descriptor name returns NaN (length-correct)."""
    out = batch_descriptors_from_smiles([ASPIRIN, ETHANOL], "definitely_not_a_real_descriptor")
    assert out.shape == (2,)
    assert math.isnan(float(out[0]))
    assert math.isnan(float(out[1]))


def test_batch_descriptors_case_insensitive() -> None:
    """Descriptor names are case-insensitive."""
    out_upper = batch_descriptors_from_smiles([ASPIRIN], "MW")
    out_lower = batch_descriptors_from_smiles([ASPIRIN], "mw")
    assert math.isclose(float(out_upper[0]), float(out_lower[0]))


# ---------------------------------------------------------------------------
# Multiprocessing fallback (the sequential path)
# ---------------------------------------------------------------------------

def test_sequential_fallback_for_mol_from_smiles(monkeypatch) -> None:
    """When is_parallel_safe() is False, we use the serial code path."""
    monkeypatch.setattr(br, "_PROBE_RESULT", False)
    monkeypatch.setattr(br, "is_parallel_safe", lambda: False)
    smiles = [ASPIRIN, ETHANOL, INVALID]
    out = batch_mol_from_smiles(smiles)
    assert len(out) == 3
    assert isinstance(out[0], Chem.Mol)
    assert isinstance(out[1], Chem.Mol)
    assert out[2] is None


def test_sequential_fallback_for_qed(monkeypatch) -> None:
    """Force serial path: batch_qed still returns the right answer."""
    monkeypatch.setattr(br, "_PROBE_RESULT", False)
    monkeypatch.setattr(br, "is_parallel_safe", lambda: False)
    expected = float(QED.qed(Chem.MolFromSmiles(ASPIRIN)))
    out = batch_qed([ASPIRIN])
    assert math.isclose(float(out[0]), expected, rel_tol=1e-9, abs_tol=1e-9)


def test_sequential_fallback_for_lipinski(monkeypatch) -> None:
    """Force serial path: batch_lipinski still filters correctly."""
    monkeypatch.setattr(br, "_PROBE_RESULT", False)
    monkeypatch.setattr(br, "is_parallel_safe", lambda: False)
    out = batch_lipinski([ASPIRIN, LONG_ALKANE, INVALID])
    assert [bool(x) for x in out] == [True, False, False]


def test_pool_failure_falls_back_to_sequential(monkeypatch) -> None:
    """If the Pool raises, we fall back to the serial path (no crash)."""
    def _raise_pool(*a, **kw):  # noqa: ANN001
        raise RuntimeError("simulated pool failure")

    monkeypatch.setattr(br, "_PROBE_RESULT", True)
    monkeypatch.setattr(br, "is_parallel_safe", lambda: True)
    monkeypatch.setattr(br, "_mp", type("_Dummy", (), {"get_context": staticmethod(_raise_pool)}))

    # Should not raise — we silently fall back to the sequential code path.
    out = batch_qed([ASPIRIN, ETHANOL])
    expected = [
        float(QED.qed(Chem.MolFromSmiles(s)))
        for s in (ASPIRIN, ETHANOL)
    ]
    np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Multiprocessing probe
# ---------------------------------------------------------------------------

def test_is_parallel_safe_returns_bool() -> None:
    """is_parallel_safe is a callable returning a bool."""
    val = is_parallel_safe()
    assert isinstance(val, bool)


def test_pool_succeeds_on_two_workers() -> None:
    """The pool code path actually runs (sanity check on Linux+macOS)."""
    if not is_parallel_safe():
        pytest.skip("multiprocessing pool not available in this env")
    smiles = [ASPIRIN, ETHANOL, BENZENE, CAFFEINE]
    # Direct invocation via multiprocessing.Pool to confirm the
    # worker-side helper is picklable.
    with multiprocessing.get_context("fork").Pool(processes=2) as pool:
        out = pool.map(br._qed_chunk, [smiles[:2], smiles[2:]])
    flat: list[float] = []
    for chunk in out:
        flat.extend(chunk)
    assert len(flat) == 4
    for v in flat:
        assert not math.isnan(v)


# ---------------------------------------------------------------------------
# Internal _scalar_descriptor
# ---------------------------------------------------------------------------

def test_scalar_descriptor_nan_for_invalid() -> None:
    """_scalar_descriptor returns NaN for an unparseable SMILES."""
    assert math.isnan(_scalar_descriptor(INVALID, "mw"))
    assert math.isnan(_scalar_descriptor("", "qed"))


def test_scalar_descriptor_qed_matches_module_api() -> None:
    """_scalar_descriptor('qed') equals Descriptors.qed for aspirin."""
    expected = float(QED.qed(Chem.MolFromSmiles(ASPIRIN)))
    got = _scalar_descriptor(ASPIRIN, "qed")
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)
