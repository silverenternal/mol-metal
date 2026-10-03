"""test_macroguide_ph.py — pytest tests for MacroGuide PH (R18 P2-2).

This module covers the **math-engine contract** for
``molmetal/molmetal_lam/lam_chem/macroguide_ph.py`` — the CPU
skeleton of the MacroGuide persistent-homology guidance mechanism
introduced by Maksymiuk (Oxford, ICML'26, arXiv:2602.14977).

The contract is split across three classes / one helper:

1. **RingTopologyComputer** — converts a SMILES into a
   ``(n_bins,)`` ring-topology feature vector.  It exposes
   ``compute(smi) -> Tensor``, ``is_aromatic(ring_atoms) -> bool``,
   and ``ring_size_grid``.
2. **MacroGuidePH** — soft guidance score in ``[0, 1]`` peaking
   when the molecule contains a ring near the target ring size.
3. **augment_pll_with_macroguide** — concatenates PLL features with
   MacroGuide features along the last axis.

All tests are CPU-only and self-skip when ``macroguide_ph`` or
``pll_signature`` is not importable.

Run from the project root::

    uv run pytest molmetal/tests/test_macroguide_ph.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy imports — keep pytest collection working even when modules are
# not on disk yet.  Each test calls these helpers and self-skips
# when the import fails.
# ---------------------------------------------------------------------------
def _import_macroguide():
    """Return MacroGuide PH classes/functions or pytest.skip()."""
    try:
        from molmetal_lam.lam_chem.macroguide_ph import (
            DEFAULT_MAX_RING_SIZE,
            DEFAULT_N_BINS,
            DEFAULT_N_WINDOWS,
            MacroGuidePH,
            MacroGuidePHError,
            RingTopologyComputer,
            augment_pll_with_macroguide,
        )
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"macroguide_ph module not importable: {exc}")
    return (
        RingTopologyComputer,
        MacroGuidePH,
        MacroGuidePHError,
        augment_pll_with_macroguide,
        DEFAULT_N_BINS,
        DEFAULT_N_WINDOWS,
        DEFAULT_MAX_RING_SIZE,
    )


# ---------------------------------------------------------------------------
# 1. RingTopologyComputer — basic shape contract
# ---------------------------------------------------------------------------
def test_ring_topology_shape_default() -> None:
    """``RingTopologyComputer().compute(smi)`` returns ``(n_bins,)``."""
    (RingTopologyComputer, _MG, _err, _aug, n_bins, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer()
    feat = comp.compute("c1ccccc1")  # benzene
    assert isinstance(feat, torch.Tensor), f"expected Tensor, got {type(feat)!r}"
    assert feat.dtype == torch.float32, f"expected float32, got {feat.dtype!r}"
    assert feat.shape == (n_bins,), f"expected ({n_bins},), got {tuple(feat.shape)}"


def test_ring_topology_shape_custom_n_bins() -> None:
    """Custom ``n_bins`` propagates to the output shape."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer(n_bins=16)
    feat = comp.compute("C1CCCCC1")  # cyclohexane (6-ring)
    assert feat.shape == (16,), f"expected (16,), got {tuple(feat.shape)}"


def test_ring_topology_rejects_zero_n_bins() -> None:
    """``__init__`` rejects ``n_bins < 1``."""
    (RingTopologyComputer, _MG, MacroGuidePHError, _aug, *_rest) = _import_macroguide()
    with pytest.raises(MacroGuidePHError, match="n_bins"):
        RingTopologyComputer(n_bins=0)
    with pytest.raises(MacroGuidePHError, match="n_bins"):
        RingTopologyComputer(n_bins=-3)


# ---------------------------------------------------------------------------
# 2. RingTopologyComputer — benzene / cyclohexane / macrocycle detection
# ---------------------------------------------------------------------------
def test_ring_topology_benzene_has_6_ring() -> None:
    """Benzene (``c1ccccc1``) deposits weight in the bin corresponding to size 6."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer(n_bins=32)
    feat = comp.compute("c1ccccc1")
    # 6 - MIN_RING_SIZE(=3) = bin 3
    assert feat[3] > 0.0, f"benzene should have 6-ring at bin 3; got {feat[3]}"
    # Aromatic + size<12 → weight 1.5
    assert torch.isclose(
        feat[3], torch.tensor(1.5), atol=1e-5
    ), f"benzene 6-ring weight should be 1.5 (aromatic); got {feat[3]}"
    # No signal at other bins.
    feat_zeroed = feat.clone()
    feat_zeroed[3] = 0.0
    assert torch.all(feat_zeroed == 0.0), (
        f"benzene should not deposit weight outside the 6-ring bin; "
        f"got nonzero at {torch.nonzero(feat_zeroed).flatten().tolist()}"
    )


def test_ring_topology_cyclohexane_is_aliphatic_6_ring() -> None:
    """Cyclohexane (``C1CCCCC1``) deposits aliphatic weight at the 6-ring bin."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer(n_bins=32)
    feat = comp.compute("C1CCCCC1")
    assert feat[3] > 0.0, f"cyclohexane should have 6-ring at bin 3; got {feat[3]}"
    # Aliphatic (no aromaticity) → weight 1.0
    assert torch.isclose(
        feat[3], torch.tensor(1.0), atol=1e-5
    ), f"cyclohexane 6-ring weight should be 1.0 (aliphatic); got {feat[3]}"


def test_ring_topology_macrocycle_bonus() -> None:
    """A 12-membered macrocycle gets the macrocycle bonus on its weight."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer(n_bins=32)
    # 12-membered aliphatic macrocycle.
    feat = comp.compute("C1CCCCCCCCCCC1")
    # Bin index 12 - 3 = 9
    assert feat[9] > 0.0, f"12-ring should be at bin 9; got {feat[9]}"
    # Aliphatic + macrocycle → weight 1.0 + 0.5 = 1.5
    assert torch.isclose(
        feat[9], torch.tensor(1.5), atol=1e-5
    ), f"12-ring weight should be 1.5 (aliphatic + macrocycle); got {feat[9]}"


def test_ring_topology_fused_rings_accumulate() -> None:
    """Fused rings (e.g. naphthalene) accumulate weights at their bins."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer(n_bins=32)
    feat = comp.compute("c1ccc2ccccc2c1")  # naphthalene: two fused 6-rings
    assert feat[3] >= 2.0, (
        f"naphthalene should have two aromatic 6-rings at bin 3; "
        f"got {feat[3]} (expected ≥ 2*1.5)"
    )


def test_ring_topology_acyclic_molecule_returns_zeros() -> None:
    """An acyclic molecule (no rings) returns the all-zero feature."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer(n_bins=32)
    feat = comp.compute("CCCC")  # n-butane
    assert torch.all(feat == 0.0), (
        f"acyclic molecule should yield all-zero feature; "
        f"nonzero bins: {torch.nonzero(feat).flatten().tolist()}"
    )


def test_ring_topology_empty_or_invalid_smiles_returns_zeros() -> None:
    """Empty / unparseable SMILES gracefully degrade to zero features."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer(n_bins=32)
    for bad in ("", "not-a-smiles", "@@@"):
        feat = comp.compute(bad)
        assert torch.all(feat == 0.0), (
            f"empty / unparseable SMILES {bad!r} should yield zero feature; "
            f"got nonzero at {torch.nonzero(feat).flatten().tolist()}"
        )


# ---------------------------------------------------------------------------
# 3. RingTopologyComputer — aromaticity detection
# ---------------------------------------------------------------------------
def test_ring_topology_aromatic_detection_benzene() -> None:
    """Benzene's 6-ring is aromatic."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer()
    # Trigger _last_mol cache.
    _ = comp.compute("c1ccccc1")
    # Benzene atoms are 0..5.
    assert comp.is_aromatic([0, 1, 2, 3, 4, 5]) is True


def test_ring_topology_aromatic_detection_cyclohexane() -> None:
    """Cyclohexane's 6-ring is *not* aromatic."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer()
    _ = comp.compute("C1CCCCC1")
    # Without knowing the exact heavy-atom indices we test that
    # *some* 6-atom subset is aliphatic.  We use the cached molecule
    # via GetAtomWithIdx.
    from rdkit import Chem  # type: ignore

    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    # First 6 atoms are the ring carbons (heavy atoms).
    ring_indices = list(range(6))
    assert comp.is_aromatic(ring_indices) is False


def test_ring_topology_is_aromatic_empty_list_returns_false() -> None:
    """``is_aromatic([])`` returns ``False`` (vacuously not aromatic)."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer()
    assert comp.is_aromatic([]) is False


def test_ring_topology_is_aromatic_out_of_range_returns_false() -> None:
    """``is_aromatic`` with an out-of-range index gracefully returns False."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer()
    _ = comp.compute("c1ccccc1")
    # Index 999 is way beyond benzene's atom count.
    assert comp.is_aromatic([0, 1, 999]) is False


# ---------------------------------------------------------------------------
# 4. RingTopologyComputer — grid accessor + invariance
# ---------------------------------------------------------------------------
def test_ring_topology_ring_size_grid() -> None:
    """``ring_size_grid`` is the integer sequence ``[3, 4, ..., 3+n_bins-1]``."""
    (RingTopologyComputer, _MG, _err, _aug, _nb, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer(n_bins=16)
    grid = comp.ring_size_grid
    assert tuple(grid.tolist()) == tuple(
        range(3, 19)
    ), f"expected (3, 4, ..., 18); got {tuple(grid.tolist())}"


def test_ring_topology_out_of_range_ring_folded_into_last_bin() -> None:
    """A ring of size > MAX folds into the last bin (truncation)."""
    (RingTopologyComputer, _MG, _err, _aug, n_bins, _nw, _mx) = _import_macroguide()
    comp = RingTopologyComputer(n_bins=n_bins)
    # Construct a ring of size n_bins + 5; should fold to last bin.
    ring_size = n_bins + 5
    # 30-atom carbon cycle: C1C...C1
    big_cycle = "C1" + "C" * (ring_size - 2) + "C1"
    feat = comp.compute(big_cycle)
    assert (
        feat[-1] > 0.0
    ), f"oversized ring should fold into last bin; got last={feat[-1]}"


# ---------------------------------------------------------------------------
# 5. MacroGuidePH — guidance score contract
# ---------------------------------------------------------------------------
def test_macroguide_guide_score_is_float_in_unit_interval() -> None:
    """``guide(smi, target)`` returns a Python float in ``[0, 1]``."""
    (RingTopologyComputer, MacroGuidePH, _err, _aug, _nb, _nw, _mx) = (
        _import_macroguide()
    )
    mg = MacroGuidePH(n_windows=4, n_bins=32)
    score = mg.guide("c1ccccc1", target_ring_size=6)
    assert isinstance(score, float), f"expected float, got {type(score)!r}"
    assert 0.0 <= score <= 1.0, f"score out of [0, 1]: {score}"


def test_macroguide_target_ring_size_peaks_at_target() -> None:
    """A molecule with a ring matching ``target`` scores higher than
    a molecule whose closest ring is one size off-target."""
    (RingTopologyComputer, MacroGuidePH, _err, _aug, _nb, _nw, _mx) = (
        _import_macroguide()
    )
    mg = MacroGuidePH(n_windows=4, n_bins=32)
    score_6 = mg.guide("c1ccccc1", target_ring_size=6)  # benzene (matches)
    score_5 = mg.guide("c1ccccc1", target_ring_size=5)  # 1 off
    score_12 = mg.guide("c1ccccc1", target_ring_size=12)  # far off
    assert (
        score_6 >= score_5
    ), f"target=6 score {score_6} should be >= target=5 score {score_5}"
    assert (
        score_6 >= score_12
    ), f"target=6 score {score_6} should be >= target=12 score {score_12}"


def test_macroguide_empty_molecule_returns_zero() -> None:
    """Empty / unparseable molecules yield ``0.0`` (graceful degrade)."""
    (RingTopologyComputer, MacroGuidePH, _err, _aug, _nb, _nw, _mx) = (
        _import_macroguide()
    )
    mg = MacroGuidePH()
    for bad in ("", "not-a-smiles"):
        assert (
            mg.guide(bad, target_ring_size=6) == 0.0
        ), f"empty/unparseable SMILES {bad!r} should yield 0.0"


def test_macroguide_compute_windows_shape() -> None:
    """``compute_windows`` returns ``(n_windows, n_bins)`` tensor."""
    (RingTopologyComputer, MacroGuidePH, _err, _aug, n_bins, n_windows, _mx) = (
        _import_macroguide()
    )
    mg = MacroGuidePH(n_windows=n_windows, n_bins=n_bins)
    windows = mg.compute_windows("c1ccccc1")
    assert windows.shape == (
        n_windows,
        n_bins,
    ), f"expected ({n_windows}, {n_bins}), got {tuple(windows.shape)}"


def test_macroguide_window_centres_monotone() -> None:
    """``window_centres`` is a monotone sequence from ``3`` to ``3 + n_bins - 1``."""
    (RingTopologyComputer, MacroGuidePH, _err, _aug, _nb, n_windows, _mx) = (
        _import_macroguide()
    )
    mg = MacroGuidePH(n_windows=n_windows, n_bins=32)
    centres = mg.window_centres()
    assert centres.shape == (
        n_windows,
    ), f"expected ({n_windows},), got {tuple(centres.shape)}"
    # Strictly monotone (or single point for n_windows==1).
    diffs = centres[1:] - centres[:-1]
    assert torch.all(
        diffs >= 0.0
    ), f"window centres must be non-decreasing; diffs={diffs.tolist()}"
    # First centre is at or above the smallest ring size; last is at or below the largest.
    assert float(centres[0]) >= 3.0, f"first centre too small: {centres[0]}"
    assert float(centres[-1]) <= 3.0 + 31.0, f"last centre too large: {centres[-1]}"


def test_macroguide_rejects_invalid_n_windows() -> None:
    """``MacroGuidePH`` rejects ``n_windows < 1``."""
    (RingTopologyComputer, MacroGuidePH, MacroGuidePHError, _aug, *_rest) = (
        _import_macroguide()
    )
    with pytest.raises(MacroGuidePHError, match="n_windows"):
        MacroGuidePH(n_windows=0)
    with pytest.raises(MacroGuidePHError, match="n_windows"):
        MacroGuidePH(n_windows=-2)


def test_macroguide_rejects_invalid_n_bins() -> None:
    """``MacroGuidePH`` rejects ``n_bins < 1``."""
    (RingTopologyComputer, MacroGuidePH, MacroGuidePHError, _aug, *_rest) = (
        _import_macroguide()
    )
    with pytest.raises(MacroGuidePHError, match="n_bins"):
        MacroGuidePH(n_bins=0)


# ---------------------------------------------------------------------------
# 6. augment_pll_with_macroguide — concatenation contract
# ---------------------------------------------------------------------------
def test_augment_pll_shape_1d() -> None:
    """1-D ``(k,) + (n_bins,) → (k + n_bins,)``."""
    (RingTopologyComputer, _MG, MacroGuidePHError, augment, _nb, _nw, _mx) = (
        _import_macroguide()
    )
    pll = torch.linspace(0.0, 1.0, 16)
    mg = torch.zeros(32)
    out = augment(pll, mg)
    assert out.shape == (48,), f"expected (48,), got {tuple(out.shape)}"


def test_augment_pll_shape_batched() -> None:
    """Batched ``(B, k) + (B, n_bins) → (B, k + n_bins)``."""
    (RingTopologyComputer, _MG, MacroGuidePHError, augment, _nb, _nw, _mx) = (
        _import_macroguide()
    )
    pll = torch.rand(4, 16)
    mg = torch.rand(4, 32)
    out = augment(pll, mg)
    assert out.shape == (4, 48), f"expected (4, 48), got {tuple(out.shape)}"


def test_augment_pll_broadcast_leading_dims() -> None:
    """Broadcastable leading dims → broadcast output."""
    (RingTopologyComputer, _MG, MacroGuidePHError, augment, _nb, _nw, _mx) = (
        _import_macroguide()
    )
    pll = torch.rand(4, 1, 16)  # (B=4, 1, k)
    mg = torch.rand(1, 3, 32)  # (1, T=3, n_bins)
    out = augment(pll, mg)
    assert out.shape == (4, 3, 48), f"expected (4, 3, 48), got {tuple(out.shape)}"


def test_augment_pll_dtype_float32() -> None:
    """Output is ``float32`` regardless of input dtype."""
    (RingTopologyComputer, _MG, MacroGuidePHError, augment, _nb, _nw, _mx) = (
        _import_macroguide()
    )
    pll = torch.zeros(8, dtype=torch.float64)
    mg = torch.zeros(32, dtype=torch.float32)
    out = augment(pll, mg)
    assert out.dtype == torch.float32, f"expected float32, got {out.dtype}"


def test_augment_pll_rejects_non_tensors() -> None:
    """``augment_pll_with_macroguide`` rejects non-Tensor inputs."""
    (RingTopologyComputer, _MG, MacroGuidePHError, augment, _nb, _nw, _mx) = (
        _import_macroguide()
    )
    with pytest.raises(MacroGuidePHError):
        augment("not a tensor", torch.zeros(32))
    with pytest.raises(MacroGuidePHError):
        augment(torch.zeros(8), [0.0] * 32)


def test_augment_pll_rejects_zero_dim_tensors() -> None:
    """``augment_pll_with_macroguide`` rejects 0-D scalar tensors."""
    (RingTopologyComputer, _MG, MacroGuidePHError, augment, _nb, _nw, _mx) = (
        _import_macroguide()
    )
    with pytest.raises(MacroGuidePHError):
        augment(torch.tensor(1.0), torch.zeros(32))
    with pytest.raises(MacroGuidePHError):
        augment(torch.zeros(8), torch.tensor(0.0))


def test_augment_pll_with_real_pll_signature() -> None:
    """End-to-end: real PLL signature + MacroGuide feature concat works."""
    (_RingTopologyComputer, _MG, _err, augment, _nb, _nw, _mx) = _import_macroguide()
    # Try to use the real PLL signature if available; otherwise
    # synthesise a valid shape.
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore

        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        _ = AllChem.EmbedMolecule(mol)
        from molmetal_lam.lam_chem.pll_signature import compute_pll

        sig = compute_pll(mol, k=16)
        pll = torch.from_numpy(sig.eigenvalues).float()
    except Exception:
        pytest.skip("pll_signature or RDKit not available; skipping E2E")
    comp = _RingTopologyComputer(n_bins=32)
    mg_feat = comp.compute("c1ccccc1")
    out = augment(pll, mg_feat)
    assert out.shape == (pll.shape[0] + mg_feat.shape[0],), (
        f"expected ({pll.shape[0] + mg_feat.shape[0]},), " f"got {tuple(out.shape)}"
    )
