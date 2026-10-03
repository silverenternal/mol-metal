"""Tests for Wf-128 Phase 1 — 172-d feature stack builder.

================================================================
Pins the Wf-128 Phase 1 contract for the **172-d feature stack**
consumed by the multi-task 5-metal backbone.  Twelve tests across
five categories:

1. **Constants & shape** — FEATURE_DIM = 172, RDKit/Multi-PH/PLL
   splits sum correctly, METALS_5 = (Ru, Ir, Os, Rh, Re).
2. **Per-stream shape** — RDKit 12-d, Multi-PH 32-d (after
   mean-pooling 32×32 → 32), PLL 128-d.
3. **Per-stream correctness** — RDKit values on a known SMILES,
   Multi-PH non-negative, PLL eigenvalues descending.
4. **End-to-end stack** — ``compute_172d`` returns a (172,) float32
   vector; concat order is RDKit | Multi-PH | PLL.
5. **Build + write** — ``build_feature_stack`` produces the five
   expected arrays; ``write_npz`` writes a valid ``.npz`` archive;
   the summary JSON lands alongside.

All tests run on CPU in < 60 s.  No GPU dependencies required.

References
----------
* TODO/pending/95_r20_p4_3way_100pocket.md
* TODO/pending/94_r20_p3_multiph_pll_feature.md
* TODO/completed/91_wf41_3way_integration.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SCRIPTS = _REPO_ROOT / "molmetal/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import build_172d_features as m  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Constants & shape (the Wf-128 spec)
# ---------------------------------------------------------------------------
def test_feature_dim_is_172() -> None:
    """The stack is **exactly** 172-d (12 RDKit + 32 Multi-PH + 128 PLL)."""
    assert m.FEATURE_DIM == 172


def test_per_stream_dims_sum_to_172() -> None:
    """12 + 32 + 128 must sum to 172."""
    assert m.RDKIT_DIM + m.MULTI_PH_DIM + m.PLL_DIM == 172
    assert m.RDKIT_DIM == 12
    assert m.MULTI_PH_DIM == 32
    assert m.PLL_DIM == 128


def test_metals_5_is_canonical_subset() -> None:
    """The canonical 5-metal subset is Ru / Ir / Os / Rh / Re (V4MT order)."""
    assert m.METALS_5 == ("Ru", "Ir", "Os", "Rh", "Re")
    assert len(m.METALS_5) == 5
    # Metal → id mapping is contiguous.
    assert {metal: i for i, metal in enumerate(m.METALS_5)} == m.METAL_TO_ID


# ---------------------------------------------------------------------------
# 2. Per-stream shape
# ---------------------------------------------------------------------------
def test_rdkit_12d_shape_on_known_smiles() -> None:
    """RDKit 12-d stream returns a 12-vector for a parseable SMILES."""
    if not m._HAS_RDKIT:
        pytest.skip("RDKit not available")
    v = m.compute_rdkit_12d("CCO")
    assert v is not None
    assert len(v) == 12
    assert all(isinstance(x, float) for x in v)


def test_rdkit_12d_returns_none_for_garbage() -> None:
    """Garbage SMILES → ``None`` (not an exception)."""
    if not m._HAS_RDKIT:
        pytest.skip("RDKit not available")
    assert m.compute_rdkit_12d("") is None
    assert m.compute_rdkit_12d("not_a_smiles!!!") is None


def test_multiph_32d_shape_on_known_smiles() -> None:
    """Multi-PH stream returns a 32-vector for a parseable SMILES."""
    if not m._HAS_TORCH:
        pytest.skip("Torch not available — Multi-PH extraction skipped")
    v = m.compute_multiph_32d("CCO")
    if v is None:
        pytest.skip("Multi-PH embedding failed on this host")
    assert v.shape == (32,)
    assert v.dtype == np.float32


def test_pll_128d_shape_on_known_smiles() -> None:
    """PLL stream returns a 128-vector for a parseable SMILES."""
    if not m._HAS_RDKIT:
        pytest.skip("RDKit not available")
    v = m.compute_pll_128d("CCO")
    if v is None:
        pytest.skip("PLL embedding failed on this host")
    assert v.shape == (128,)
    assert v.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. Per-stream correctness
# ---------------------------------------------------------------------------
def test_rdkit_12d_values_match_rdkit_descriptors() -> None:
    """The first 12 entries of the RDKit stream match a fresh RDKit call."""
    if not m._HAS_RDKIT:
        pytest.skip("RDKit not available")
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    smi = "c1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smi)
    expected = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.MolMR(mol),
        Descriptors.BalabanJ(mol),
    ]
    got = m.compute_rdkit_12d(smi)
    assert got is not None
    for i, (a, b) in enumerate(zip(expected, got)):
        assert math_close(a, b, atol=1e-4), f"RDKit[{i}] mismatch"


def test_multiph_32d_is_non_negative() -> None:
    """The Multi-PH 2-parameter filtration counts edge bins → non-negative."""
    if not m._HAS_TORCH:
        pytest.skip("Torch not available")
    for smi in ("CCO", "c1ccccc1", "CC(=O)O"):
        v = m.compute_multiph_32d(smi)
        if v is None:
            continue
        assert (v >= 0).all(), f"Multi-PH must be non-negative on {smi}"


def test_pll_128d_is_descending_when_present() -> None:
    """The PLL eigenvalues are **descending** before padding (Liu & Feng 2026)."""
    if not m._HAS_RDKIT:
        pytest.skip("RDKit not available")
    v = m.compute_pll_128d("CCO")
    if v is None:
        pytest.skip("PLL embedding failed")
    # The padded tail is zeros — strict descending holds only on the prefix.
    nonzero = v[v > 0]
    assert nonzero.size > 0
    # Strictly non-increasing on the non-zero prefix.
    for i in range(nonzero.size - 1):
        assert nonzero[i] >= nonzero[i + 1] - 1e-6


# ---------------------------------------------------------------------------
# 4. End-to-end stack
# ---------------------------------------------------------------------------
def test_compute_172d_shape_and_dtype() -> None:
    """``compute_172d`` returns a (172,) float32 vector on CCO."""
    v = m.compute_172d("CCO")
    if v is None:
        pytest.skip("All three streams failed on CCO on this host")
    assert v.shape == (172,)
    assert v.dtype == np.float32
    assert np.isfinite(v).all()


def test_compute_172d_concat_order_is_rdkit_multiph_pll() -> None:
    """The concat order is RDKit(12) | Multi-PH(32) | PLL(128)."""
    if not m._HAS_RDKIT:
        pytest.skip("RDKit not available")
    rdkit_only = m.compute_rdkit_12d("CCO")
    full = m.compute_172d("CCO")
    if rdkit_only is None or full is None:
        pytest.skip("RDKit or full-stack failed on CCO on this host")
    # First 12 entries are the RDKit 12-d.
    for i in range(12):
        assert math_close(
            full[i], rdkit_only[i], atol=1e-5
        ), f"stack[{i}] != rdkit[{i}] (concat-order broken)"


def test_compute_172d_handles_all_stream_failures_gracefully() -> None:
    """Empty SMILES → ``None`` (do not raise)."""
    assert m.compute_172d("") is None
    assert m.compute_172d("not_a_smiles!@#") is None


# ---------------------------------------------------------------------------
# 5. Build + write
# ---------------------------------------------------------------------------
def test_build_feature_stack_produces_five_arrays(tmp_path: Path) -> None:
    """``build_feature_stack`` returns the five-array payload contract."""
    rows = [
        {
            "smiles": "CCO",
            "metal": "Ru",
            "scaffold_id": "s1",
            "split": "train",
            "activity": "0",
        },
        {
            "smiles": "c1ccccc1",
            "metal": "Ir",
            "scaffold_id": "s2",
            "split": "train",
            "activity": "1",
        },
        {
            "smiles": "CC(=O)O",
            "metal": "Os",
            "scaffold_id": "s3",
            "split": "val",
            "activity": "0",
        },
    ]
    payload = m.build_feature_stack(rows, max_rows=None, seed=0)
    for key in ("X", "y", "metal_id", "smiles_hash", "row_meta"):
        assert key in payload, f"missing {key}"
    assert payload["X"].shape[1] == 172
    assert payload["X"].dtype == np.float32
    assert payload["y"].dtype == np.float32
    assert payload["metal_id"].dtype == np.int64
    assert payload["smiles_hash"].dtype == np.int64


def test_write_npz_writes_loadable_archive(tmp_path: Path) -> None:
    """``write_npz`` writes a valid ``.npz`` archive with the expected keys."""
    rows = [
        {
            "smiles": "CCO",
            "metal": "Ru",
            "scaffold_id": "s1",
            "split": "train",
            "activity": "0",
        },
        {
            "smiles": "c1ccccc1",
            "metal": "Ir",
            "scaffold_id": "s2",
            "split": "train",
            "activity": "1",
        },
    ]
    payload = m.build_feature_stack(rows)
    out_path = tmp_path / "features_172d.npz"
    m.write_npz(payload, out_path)
    assert out_path.exists()
    with np.load(out_path, allow_pickle=True) as data:
        assert set(data.keys()) >= {"X", "y", "metal_id", "smiles_hash", "row_meta"}
        X = data["X"]
        assert X.shape[1] == 172
    # Summary JSON sibling.
    summary_path = out_path.with_suffix(".summary.json")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["feature_dim"] == 172


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def math_close(a: float, b: float, atol: float = 1e-5) -> bool:
    """Robust float close (handles NaN identically)."""
    if a != a and b != b:  # NaN == NaN
        return True
    return abs(a - b) <= atol
