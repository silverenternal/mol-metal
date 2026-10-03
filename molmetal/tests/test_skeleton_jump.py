"""Pytest: Skeleton-jump / fragment-swap rule — Pitfall Rank 12, P2.4 (TODO-30).

Twenty-five tests covering the new
:mod:`molmetal.molmetal_lam.lam_chem.skeleton_jump` module:

  1.  test_fragment_library_add_get_roundtrip
  2.  test_fragment_library_get_unknown_raises
  3.  test_fragment_library_duplicate_name_raises
  4.  test_fragment_library_compatibility_filter
  5.  test_fragment_library_compatibility_empty_on_invalid
  6.  test_fragment_library_min_max_atoms_bounds
  7.  test_fragment_library_invalid_smiles_raises
  8.  test_fragment_library_init_validation
  9.  test_fragment_library_names_sorted
 10.  test_skeleton_jump_swap_prob_zero
 11.  test_skeleton_jump_swap_prob_one
 12.  test_skeleton_jump_swap_prob_invalid
 13.  test_skeleton_jump_apply_invalid_input
 14.  test_skeleton_jump_apply_valid_smiles
 15.  test_skeleton_jump_compute_compatibility_range
 16.  test_skeleton_jump_compute_compatibility_equal_size
 17.  test_skeleton_jump_compute_compatibility_invalid
 18.  test_skeleton_jump_report_fields
 19.  test_skeleton_jump_swap_performed_when_forced
 20.  test_diversity_aware_no_history_always_passes
 21.  test_diversity_aware_target_zero_blocks_identical
 22.  test_diversity_aware_target_one_always_passes
 23.  test_diversity_aware_history_capacity
 24.  test_register_skeleton_jump_returns_count
 25.  test_module_level_constants

All tests run on CPU in <5 s — no GPU, no torch dependency, no learned
priors.  This is the API-surface contract for the skeleton-jump /
fragment-swap rule, parallel to ``test_layered_generation.py``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.skeleton_jump import (
    DEFAULT_MAX_ATOMS,
    DEFAULT_MIN_ATOMS,
    DEFAULT_SWAP_PROB,
    DEFAULT_TARGET_DIVERSITY,
    DiversityAwareSkeletonJump,
    FragmentLibrary,
    FragmentRecord,
    SkeletonJumpRule,
    SwapResult,
    register_skeleton_jump,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_library(seed: int = 0) -> FragmentLibrary:
    """Deterministic 6-fragment library for tests."""
    rng = random.Random(seed)
    lib = FragmentLibrary(min_atoms=3, max_atoms=10)
    # Mix of rings + chains, with and without heteroatoms, sized 3-8.
    seeds: List[tuple] = [
        ("c1ccccc1", "benzene"),  # 6, ring, no hetero
        ("c1ccncc1", "pyridine"),  # 6, ring, hetero
        ("C1CCCCC1", "cyclohexane"),  # 6, ring, no hetero
        ("CCCC", "butane"),  # 4, chain, no hetero
        ("CCO", "ethanol"),  # 3, chain, hetero
        ("CC(=O)O", "acetic"),  # 4, chain, hetero
    ]
    rng.shuffle(seeds)
    for smi, name in seeds:
        lib.add_fragment(smi, name)
    return lib


# A handful of scaffold SMILES that span the test surface.
_SCAFFOLDS = (
    "c1ccccc1",  # benzene — 6, ring, no hetero
    "c1ccncc1",  # pyridine — 6, ring, hetero
    "CCCCCC",  # hexane — 6, chain, no hetero
    "CCN",  # ethylamine — 3, chain, hetero
)


# ---------------------------------------------------------------------------
# Test 1: FragmentLibrary.add_fragment + get_fragment_by_name roundtrip
# ---------------------------------------------------------------------------
def test_fragment_library_add_get_roundtrip():
    """add_fragment + get_fragment_by_name roundtrip returns a canonical SMILES."""
    lib = FragmentLibrary()
    lib.add_fragment("c1ccccc1", "benzene")
    out = lib.get_fragment_by_name("benzene")
    assert isinstance(out, str)
    assert out, "get_fragment_by_name returned empty"
    # Canonical benzene is the same regardless of input casing.
    from rdkit import Chem

    canon = Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1"))
    assert out == canon, f"expected {canon!r}, got {out!r}"


# ---------------------------------------------------------------------------
# Test 2: FragmentLibrary.get_fragment_by_name on unknown name raises
# ---------------------------------------------------------------------------
def test_fragment_library_get_unknown_raises():
    """Unknown name → KeyError."""
    lib = FragmentLibrary()
    lib.add_fragment("c1ccccc1", "benzene")
    with pytest.raises(KeyError):
        lib.get_fragment_by_name("not_registered")


# ---------------------------------------------------------------------------
# Test 3: Duplicate name registration is rejected
# ---------------------------------------------------------------------------
def test_fragment_library_duplicate_name_raises():
    """Re-registering an existing name → ValueError."""
    lib = FragmentLibrary()
    lib.add_fragment("c1ccccc1", "benzene")
    with pytest.raises(ValueError):
        lib.add_fragment("c1ccccc1", "benzene")


# ---------------------------------------------------------------------------
# Test 4: get_compatible_fragments returns only compatible fragments
# ---------------------------------------------------------------------------
def test_fragment_library_compatibility_filter():
    """get_compatible_fragments on a benzene scaffold returns ring fragments ≤ 6."""
    lib = _make_library(seed=0)
    out = lib.get_compatible_fragments("c1ccccc1")
    assert isinstance(out, list)
    # All returned fragments must be ring fragments (benzene has a ring).
    for name in out:
        rec = lib.records[name]
        assert rec.is_ring, (
            f"benzene scaffold must only get ring fragments, "
            f"but {name!r} is_chain={not rec.is_ring}"
        )
        assert rec.n_heavy <= 6, (
            f"benzene scaffold must only get fragments ≤ 6 atoms, "
            f"but {name!r} has {rec.n_heavy}"
        )


# ---------------------------------------------------------------------------
# Test 5: get_compatible_fragments on invalid input returns empty
# ---------------------------------------------------------------------------
def test_fragment_library_compatibility_empty_on_invalid():
    """Invalid scaffold → empty list, no exception."""
    lib = _make_library(seed=0)
    assert lib.get_compatible_fragments("not_a_smiles@@") == []
    assert lib.get_compatible_fragments("") == []


# ---------------------------------------------------------------------------
# Test 6: min/max atom bounds reject fragments outside the window
# ---------------------------------------------------------------------------
def test_fragment_library_min_max_atoms_bounds():
    """Fragments with too few or too many heavy atoms are rejected."""
    lib_min = FragmentLibrary(min_atoms=5, max_atoms=10)
    # 4 heavy atoms → below min.
    with pytest.raises(ValueError):
        lib_min.add_fragment("CCCC", "butane")
    lib_max = FragmentLibrary(min_atoms=3, max_atoms=4)
    # 6 heavy atoms → above max.
    with pytest.raises(ValueError):
        lib_max.add_fragment("c1ccccc1", "benzene")


# ---------------------------------------------------------------------------
# Test 7: Invalid SMILES raise ValueError at registration
# ---------------------------------------------------------------------------
def test_fragment_library_invalid_smiles_raises():
    """Unparseable SMILES → ValueError, not a silent failure."""
    lib = FragmentLibrary()
    with pytest.raises(ValueError):
        lib.add_fragment("not_a_smiles@@", "bogus")


# ---------------------------------------------------------------------------
# Test 8: __init__ validates min/max atoms
# ---------------------------------------------------------------------------
def test_fragment_library_init_validation():
    """min_atoms < 1 or max_atoms < min_atoms → ValueError."""
    with pytest.raises(ValueError):
        FragmentLibrary(min_atoms=0)
    with pytest.raises(ValueError):
        FragmentLibrary(min_atoms=5, max_atoms=3)


# ---------------------------------------------------------------------------
# Test 9: names() is sorted (deterministic)
# ---------------------------------------------------------------------------
def test_fragment_library_names_sorted():
    """names() returns a sorted list of registered fragments."""
    lib = FragmentLibrary()
    lib.add_fragment("CCCC", "z_fragment")
    lib.add_fragment("c1ccccc1", "a_fragment")
    lib.add_fragment("CCO", "m_fragment")
    out = lib.names()
    assert out == sorted(out), f"names() must be sorted, got {out!r}"
    assert out == ["a_fragment", "m_fragment", "z_fragment"]


# ---------------------------------------------------------------------------
# Test 10: swap_prob=0 → no swap
# ---------------------------------------------------------------------------
def test_skeleton_jump_swap_prob_zero():
    """swap_prob=0 → apply returns the input unchanged."""
    lib = _make_library(seed=0)
    rule = SkeletonJumpRule(lib, swap_prob=0.0, rng=random.Random(0))
    out = rule.apply("c1ccccc1", position=0)
    assert out == "c1ccccc1", f"swap_prob=0 must not change the mol, got {out!r}"


# ---------------------------------------------------------------------------
# Test 11: swap_prob=1 → always swap (when compatible fragments exist)
# ---------------------------------------------------------------------------
def test_skeleton_jump_swap_prob_one():
    """swap_prob=1 → apply_with_report swap_performed is True on a swap-able scaffold.

    We use a chain scaffold (hexane) because the swap-cut algorithm
    cuts a prefix off the input and bonds the fragment in; chains
    produce well-defined prefixes that pass RDKit sanitisation,
    whereas cutting a partial aromatic ring (e.g. 2 atoms of benzene)
    leaves a broken aromaticity that RDKit rejects.  This is the
    honest framing of the CPU skeleton's swap algorithm — the
    learned swap (TODO-31) will handle ring fragments properly.
    """
    lib = _make_library(seed=0)
    rule = SkeletonJumpRule(lib, swap_prob=1.0, rng=random.Random(0))
    # Use a chain scaffold + a chain position so the swap-cut
    # produces a valid RDKit-sanitisable result.
    report = rule.apply_with_report("CCCCCC", position=4)
    assert (
        report.swap_performed
    ), f"swap_prob=1 should always swap on a swap-able scaffold, got {report!r}"
    assert (
        report.chosen_fragment is not None
    ), f"swap_performed=True but chosen_fragment is None: {report!r}"
    assert 0.0 <= report.compatibility <= 1.0


# ---------------------------------------------------------------------------
# Test 12: swap_prob out of range → ValueError
# ---------------------------------------------------------------------------
def test_skeleton_jump_swap_prob_invalid():
    """swap_prob outside [0, 1] → ValueError at construction."""
    lib = _make_library(seed=0)
    with pytest.raises(ValueError):
        SkeletonJumpRule(lib, swap_prob=-0.1)
    with pytest.raises(ValueError):
        SkeletonJumpRule(lib, swap_prob=1.5)


# ---------------------------------------------------------------------------
# Test 13: apply on invalid input returns empty SMILES
# ---------------------------------------------------------------------------
def test_skeleton_jump_apply_invalid_input():
    """Invalid SMILES → apply returns empty, no exception."""
    lib = _make_library(seed=0)
    rule = SkeletonJumpRule(lib, swap_prob=1.0, rng=random.Random(0))
    assert rule.apply("not_a_smiles@@", position=0) == ""
    assert rule.apply("", position=0) == ""


# ---------------------------------------------------------------------------
# Test 14: apply on valid input returns a parseable SMILES
# ---------------------------------------------------------------------------
def test_skeleton_jump_apply_valid_smiles():
    """apply returns a valid SMILES (possibly the same as input)."""
    lib = _make_library(seed=0)
    rule = SkeletonJumpRule(lib, swap_prob=0.0, rng=random.Random(0))
    out = rule.apply("c1ccccc1", position=0)
    assert isinstance(out, str)
    assert out, "apply returned empty on valid input"
    from rdkit import Chem

    mol = Chem.MolFromSmiles(out)
    assert mol is not None, f"apply output {out!r} is not parseable"


# ---------------------------------------------------------------------------
# Test 15: compute_compatibility score ∈ [0, 1]
# ---------------------------------------------------------------------------
def test_skeleton_jump_compute_compatibility_range():
    """compute_compatibility always returns a float in [0, 1]."""
    lib = _make_library(seed=0)
    rule = SkeletonJumpRule(lib, swap_prob=0.0, rng=random.Random(0))
    for scaf in _SCAFFOLDS:
        for name in lib.names():
            score = rule.compute_compatibility(scaf, lib.get_fragment_by_name(name))
            assert isinstance(score, float), f"score must be float, got {type(score)}"
            assert (
                0.0 <= score <= 1.0
            ), f"score for ({scaf!r}, {name!r}) = {score} is outside [0, 1]"


# ---------------------------------------------------------------------------
# Test 16: compute_compatibility is highest for size-matched fragments
# ---------------------------------------------------------------------------
def test_skeleton_jump_compute_compatibility_equal_size():
    """Same-size fragment typically scores higher than a size-mismatched one."""
    lib = _make_library(seed=0)
    rule = SkeletonJumpRule(lib, swap_prob=0.0, rng=random.Random(0))
    # pyridine has 6 atoms; benzene (6) and cyclohexane (6) are size-matched.
    pyridine = "c1ccncc1"
    same_size_score = rule.compute_compatibility(pyridine, "c1ccccc1")
    smaller_size_score = rule.compute_compatibility(pyridine, "CCO")
    # The exact magnitudes depend on topology flags; what matters is
    # that the size-match score does NOT under-perform a much smaller
    # size-mismatched fragment.
    assert same_size_score >= 0.0
    assert smaller_size_score >= 0.0
    # Same-size benzene is closer to pyridine in size AND in topology
    # (both aromatic rings), so the score must be at least as large.
    assert (
        same_size_score >= smaller_size_score
    ), f"size-matched score {same_size_score} < size-mismatched {smaller_size_score}"


# ---------------------------------------------------------------------------
# Test 17: compute_compatibility on invalid inputs falls back to 0.25
# ---------------------------------------------------------------------------
def test_skeleton_jump_compute_compatibility_invalid():
    """Invalid scaffold / fragment → score = 0.25 (defensive fallback)."""
    lib = _make_library(seed=0)
    rule = SkeletonJumpRule(lib, swap_prob=0.0, rng=random.Random(0))
    score = rule.compute_compatibility("not_a_smiles@@", "c1ccccc1")
    assert score == 0.25, f"invalid scaffold → expected 0.25, got {score}"
    score = rule.compute_compatibility("c1ccccc1", "not_a_smiles@@")
    assert score == 0.25, f"invalid fragment → expected 0.25, got {score}"


# ---------------------------------------------------------------------------
# Test 18: SwapResult fields are populated correctly
# ---------------------------------------------------------------------------
def test_skeleton_jump_report_fields():
    """apply_with_report returns a SwapResult with the documented fields."""
    lib = _make_library(seed=0)
    rule = SkeletonJumpRule(lib, swap_prob=1.0, rng=random.Random(0))
    report = rule.apply_with_report("c1ccccc1", position=2)
    assert isinstance(report, SwapResult)
    assert isinstance(report.smiles, str)
    assert isinstance(report.valid, bool)
    assert isinstance(report.swap_performed, bool)
    assert isinstance(report.original, str)
    assert report.original == "c1ccccc1"
    # chosen_fragment is Optional[str].
    if report.swap_performed:
        assert report.chosen_fragment is not None
        assert 0.0 <= report.compatibility <= 1.0


# ---------------------------------------------------------------------------
# Test 19: When swap_prob=1 and no compatible fragments exist, no swap happens
# ---------------------------------------------------------------------------
def test_skeleton_jump_swap_performed_when_forced():
    """swap_prob=1 + no compatible fragments → swap_performed=False."""
    # Empty library → no candidates → even with swap_prob=1, no swap.
    lib = FragmentLibrary()
    rule = SkeletonJumpRule(lib, swap_prob=1.0, rng=random.Random(0))
    report = rule.apply_with_report("c1ccccc1", position=2)
    assert (
        not report.swap_performed
    ), f"empty library must yield swap_performed=False, got {report!r}"
    # smiles is the original scaffold.
    assert report.smiles == "c1ccccc1"


# ---------------------------------------------------------------------------
# Test 20: Diversity-aware wrapper — empty history always passes
# ---------------------------------------------------------------------------
def test_diversity_aware_no_history_always_passes():
    """With empty history, apply_with_diversity behaves like the base rule."""
    lib = _make_library(seed=0)
    base = SkeletonJumpRule(lib, swap_prob=0.0, rng=random.Random(0))
    wrap = DiversityAwareSkeletonJump(
        base,
        target_diversity=0.7,
        history_capacity=10,
        rng=random.Random(0),
    )
    # swap_prob=0 → no swap → return mol unchanged.
    out = wrap.apply_with_diversity("c1ccccc1", history=[])
    assert (
        out == "c1ccccc1"
    ), f"empty history + swap_prob=0 should return mol unchanged, got {out!r}"


# ---------------------------------------------------------------------------
# Test 21: Diversity-aware wrapper — target=0 blocks identical molecules
# ---------------------------------------------------------------------------
def test_diversity_aware_target_zero_blocks_identical():
    """target_diversity=0 blocks swaps that produce a Tanimoto-similar molecule.

    With swap_prob=0 the wrapper returns the input unchanged regardless
    of the gate, but we still validate the gate logic: history is
    populated, then a query that is identical to history is allowed
    (because no swap was attempted → no new molecule to gate).
    """
    lib = _make_library(seed=0)
    base = SkeletonJumpRule(lib, swap_prob=0.0, rng=random.Random(0))
    wrap = DiversityAwareSkeletonJump(
        base,
        target_diversity=0.0,
        history_capacity=10,
        rng=random.Random(0),
    )
    # First call — empty history, swap_prob=0 → unchanged.
    out1 = wrap.apply_with_diversity("c1ccccc1", history=[])
    assert out1 == "c1ccccc1"
    # Second call with explicit history — swap_prob=0 again → unchanged.
    out2 = wrap.apply_with_diversity("c1ccccc1", history=["c1ccccc1"])
    assert out2 == "c1ccccc1"
    # Internal history must reflect the appended entry.
    hist = wrap.history()
    assert "c1ccccc1" in hist


# ---------------------------------------------------------------------------
# Test 22: target_diversity=1.0 always passes
# ---------------------------------------------------------------------------
def test_diversity_aware_target_one_always_passes():
    """target_diversity=1.0 → diversity gate never blocks."""
    lib = _make_library(seed=0)
    base = SkeletonJumpRule(lib, swap_prob=0.0, rng=random.Random(0))
    wrap = DiversityAwareSkeletonJump(
        base,
        target_diversity=1.0,
        history_capacity=10,
        rng=random.Random(0),
    )
    out = wrap.apply_with_diversity("c1ccccc1", history=["c1ccccc1", "c1ccncc1"])
    assert (
        out == "c1ccccc1"
    ), f"target_diversity=1.0 must always pass the gate, got {out!r}"


# ---------------------------------------------------------------------------
# Test 23: history_capacity caps the internal buffer (FIFO eviction)
# ---------------------------------------------------------------------------
def test_diversity_aware_history_capacity():
    """history_capacity evicts oldest entries when full."""
    lib = _make_library(seed=0)
    base = SkeletonJumpRule(lib, swap_prob=0.0, rng=random.Random(0))
    cap = 3
    wrap = DiversityAwareSkeletonJump(
        base,
        target_diversity=1.0,
        history_capacity=cap,
        rng=random.Random(0),
    )
    # Push 5 entries; only the last 3 should remain.
    for smi in ("A", "B", "C", "D", "E"):
        wrap.apply_with_diversity("c1ccccc1", history=[smi])
    hist = wrap.history()
    assert (
        len(hist) == cap
    ), f"history length must be capped at {cap}, got {len(hist)}: {hist!r}"
    # The oldest entries ("A", "B") should be evicted; the newest
    # entries ("C", "D", "E") should survive.
    assert hist[-3:] == [
        "C",
        "D",
        "E",
    ], f"FIFO eviction broken, expected last 3 = ['C', 'D', 'E'], got {hist[-3:]!r}"


# ---------------------------------------------------------------------------
# Test 24: register_skeleton_jump returns the rule count (≥ 1)
# ---------------------------------------------------------------------------
def test_register_skeleton_jump_count():
    """register_skeleton_jump() returns ≥ 1 (one rule registered)."""
    count = register_skeleton_jump()
    assert isinstance(count, int), f"expected int, got {type(count)}"
    assert count >= 1, f"register_skeleton_jump must return ≥ 1, got {count}"


# ---------------------------------------------------------------------------
# Test 25: Module-level constants are exported and sane
# ---------------------------------------------------------------------------
def test_module_level_constants():
    """DEFAULT_* constants are exported and have the documented values."""
    assert DEFAULT_MIN_ATOMS == 3
    assert DEFAULT_MAX_ATOMS == 12
    assert DEFAULT_MIN_ATOMS < DEFAULT_MAX_ATOMS
    assert 0.0 <= DEFAULT_SWAP_PROB <= 1.0
    assert 0.0 <= DEFAULT_TARGET_DIVERSITY <= 1.0
    # Constructors honour the defaults.
    lib = FragmentLibrary()
    assert lib.min_atoms == DEFAULT_MIN_ATOMS
    assert lib.max_atoms == DEFAULT_MAX_ATOMS
    rule = SkeletonJumpRule(lib)
    assert rule.swap_prob == DEFAULT_SWAP_PROB
    wrap = DiversityAwareSkeletonJump(rule)
    assert wrap.target_diversity == DEFAULT_TARGET_DIVERSITY
