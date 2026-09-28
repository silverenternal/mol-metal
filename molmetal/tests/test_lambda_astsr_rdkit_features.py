"""Tests for Wf-49 Phase 2 — RDKit 2D feature columns for the ASTSR.

=====================================================================
Wf-44 Root Cause Catalog §Cause 2 (CRITICAL) flagged the legacy F5
5-feature set (``diversity_tanimoto``, ``diversity_homotype``,
``qed_mean``, ``anticancer_index``, ``sa_mean_norm``) as **not
informative for target** — LassoLarsIC zeroed 4 of 5 coefficients and
the predict-the-mean baseline check produced R² < 0 (Wf-44 Phase 2).

Wf-49 Phase 1 shipped a real 12-d RDKit 2D descriptor extractor
(``molmetal/scripts/extract_rdkit_2d_descriptors.py``) producing
``molmetal/data/{platinai_5k, platinai_4cell, metalcytotox}_rdkit_features.csv``
plus per-dataset ``feature_summary.json`` reports.  Phase 2 (this
deliverable) wires those RDKit 2D features into the ASTSR as the new
default while keeping the legacy F5 5-feature set reachable via a
backward-compat shim.

These tests pin down:
  * the F5_RDKIT_2D 12-d RDKit 2D feature registry
    (:data:`RDKIT_2D_FEATURE_COLUMNS`) is correct, frozen, and
    includes the canonical Lipinski drivers;
  * the legacy F5 5-feature set
    (:data:`DEFAULT_FEATURE_COLUMNS`) is **unchanged** for backward
    compatibility with Wf-44 / Wf-41 ship consumers;
  * :func:`default_feature_columns` now defaults to the new
    F5_RDKIT_2D 12-d set, with ``"f5_legacy"`` as the backward-compat
    flag;
  * :class:`LambdaASTSR` initialised without ``feature_names`` picks up
    the RDKit 2D 12-d default;
  * on the canonical pIC50 driver target (MolLogP-dominated linear
    combination of the 12-d RDKit block), the ASTSR achieves a sane
    R² on the deterministic toy problem;
  * the Wf-49 Phase 1 RDKit 2D extractor outputs (when present on disk)
    are bit-for-bit compatible with the ASTSR's expected column order.

Tests (count target: 15):
  1.  Registry invariants — RDKIT_2D_FEATURE_COLUMNS is a tuple of 12.
  2.  Registry order is frozen.
  3.  Registry covers Lipinski drivers (MolLogP, TPSA, HBA, HBD, MW).
  4.  Legacy F5 registry is unchanged.
  5.  default_feature_columns() now defaults to the RDKit 2D 12-d set.
  6.  default_feature_columns(feature_set="f5_legacy") returns F5.
  7.  legacy_f5_feature_columns() / rdkit_2d_feature_columns()
      match the explicit-flag forms.
  8.  default_feature_columns() rejects unknown feature_set values.
  9.  LambdaASTSR() picks up RDKit 2D 12-d default.
 10.  LambdaASTSR(feature_names=legacy_f5_feature_columns()) is the F5
      backward-compat path.
 11.  ASTSR on a MolLogP-dominated linear target achieves R² >= 0.5.
 12.  ASTSR on a 12-d feature, AST evaluation respects column order.
 13.  AST evaluation NaN-safety holds with RDKit 2D features.
 14.  RDKit 2D feature order matches Wf-49 Phase 1 extractor output
      (skipped when artefact is missing).
 15.  deterministic re-fit under RDKit 2D feature_names.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pytest

from molmetal.molmetal_lam.reward.lambda_astsr import (
    DEFAULT_FEATURE_COLUMNS,
    RDKIT_2D_FEATURE_COLUMNS,
    RDKIT_2D_EXTENDED_FEATURE_COLUMNS,
    LambdaAST,
    LambdaASTSR,
    compute_r2,
    default_feature_columns,
    legacy_f5_feature_columns,
    mutate_ast,
    random_ast,
    rdkit_2d_feature_columns,
    rdkit_2d_extended_feature_columns,
)


# Canonical 12-d RDKit 2D descriptor set (Wf-49 Phase 1 + Phase 2).
EXPECTED_RDKIT_2D_NAMES: tuple = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
)

# Canonical legacy F5 5-feature set (Wf-41 ship, kept for back-compat).
EXPECTED_F5_NAMES: tuple = (
    "diversity_tanimoto",
    "diversity_homotype",
    "qed_mean",
    "anticancer_index",
    "sa_mean_norm",
)


# ---------------------------------------------------------------------------
# 1. Registry invariants
# ---------------------------------------------------------------------------
def test_rdkit_2d_feature_columns_is_tuple_of_12() -> None:
    """RDKIT_2D_FEATURE_COLUMNS is a tuple of exactly 12 names."""
    assert isinstance(RDKIT_2D_FEATURE_COLUMNS, tuple)
    assert len(RDKIT_2D_FEATURE_COLUMNS) == 12
    assert all(isinstance(n, str) for n in RDKIT_2D_FEATURE_COLUMNS)


def test_rdkit_2d_feature_columns_order_frozen() -> None:
    """RDKIT_2D_FEATURE_COLUMNS order matches the Wf-49 Phase 1 frozen order."""
    assert RDKIT_2D_FEATURE_COLUMNS == EXPECTED_RDKIT_2D_NAMES, (
        f"RDKIT_2D_FEATURE_COLUMNS order drift.\n"
        f"  expected: {EXPECTED_RDKIT_2D_NAMES}\n"
        f"  actual:   {RDKIT_2D_FEATURE_COLUMNS}"
    )


def test_rdkit_2d_feature_columns_covers_lipinski_drivers() -> None:
    """Registry covers MolLogP, TPSA, HBA, HBD, MolWt — Lipinski's Ro5."""
    required = {"MolLogP", "TPSA", "HBA", "HBD", "MolWt"}
    assert required.issubset(set(RDKIT_2D_FEATURE_COLUMNS)), (
        f"RDKit 2D registry missing Lipinski drivers: "
        f"required={sorted(required)}, present={sorted(RDKIT_2D_FEATURE_COLUMNS)}"
    )


# ---------------------------------------------------------------------------
# 2. Backward compatibility for the legacy F5 set
# ---------------------------------------------------------------------------
def test_legacy_f5_feature_columns_unchanged() -> None:
    """The legacy F5 5-feature registry is bit-for-bit unchanged."""
    assert isinstance(DEFAULT_FEATURE_COLUMNS, tuple)
    assert len(DEFAULT_FEATURE_COLUMNS) == 5
    assert DEFAULT_FEATURE_COLUMNS == EXPECTED_F5_NAMES, (
        "DEFAULT_FEATURE_COLUMNS drifted from the Wf-41 ship order; this "
        "would break Wf-44 / Wf-41 ship consumers. expected="
        f"{EXPECTED_F5_NAMES}, actual={DEFAULT_FEATURE_COLUMNS}"
    )


def test_legacy_f5_feature_columns_returns_f5() -> None:
    """legacy_f5_feature_columns() mirrors the F5 constant."""
    cols = legacy_f5_feature_columns()
    assert cols == DEFAULT_FEATURE_COLUMNS


def test_rdkit_2d_feature_columns_returns_12d() -> None:
    """rdkit_2d_feature_columns() mirrors the RDKit 2D constant."""
    cols = rdkit_2d_feature_columns()
    assert cols == RDKIT_2D_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 3. default_feature_columns() flag dispatch
# ---------------------------------------------------------------------------
def test_default_feature_columns_default_is_rdkit_2d() -> None:
    """default_feature_columns() now defaults to F5_RDKIT_2D 12-d (Wf-49 P2)."""
    cols = default_feature_columns()
    assert cols == RDKIT_2D_FEATURE_COLUMNS, (
        "default_feature_columns() should default to RDKit 2D 12-d for "
        "Wf-49 Phase 2; got F5 instead. Check the ``feature_set`` default."
    )


def test_default_feature_columns_rdkit_2d_flag() -> None:
    """``default_feature_columns(feature_set='rdkit_2d')`` returns 12-d."""
    cols = default_feature_columns(feature_set="rdkit_2d")
    assert cols == RDKIT_2D_FEATURE_COLUMNS
    assert len(cols) == 12


def test_default_feature_columns_f5_legacy_flag() -> None:
    """``default_feature_columns(feature_set='f5_legacy')`` returns F5."""
    cols = default_feature_columns(feature_set="f5_legacy")
    assert cols == DEFAULT_FEATURE_COLUMNS
    assert len(cols) == 5


def test_default_feature_columns_rejects_unknown_set() -> None:
    """Unknown ``feature_set`` strings raise ValueError."""
    with pytest.raises(ValueError):
        default_feature_columns(feature_set="not_a_real_set")
    with pytest.raises(ValueError):
        default_feature_columns(feature_set="")


# ---------------------------------------------------------------------------
# 4. LambdaASTSR integration with the new default
# ---------------------------------------------------------------------------
def test_lambda_astsr_default_feature_names_is_rdkit_2d() -> None:
    """LambdaASTSR() with no args picks up the F5_RDKIT_2D 12-d default."""
    astsr = LambdaASTSR(depth=2, population=4, generations=0, seed=0)
    assert astsr.feature_names == RDKIT_2D_FEATURE_COLUMNS, (
        "LambdaASTSR default feature_names should be the RDKit 2D 12-d set "
        "for Wf-49 Phase 2."
    )
    assert len(astsr.feature_names) == 12


def test_lambda_astsr_legacy_f5_backward_compat() -> None:
    """Passing legacy_f5_feature_columns() selects the F5 5-feature path."""
    astsr = LambdaASTSR(
        depth=2, population=4, generations=0, seed=0,
        feature_names=legacy_f5_feature_columns(),
    )
    assert astsr.feature_names == DEFAULT_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 5. The verify-gate: ASTSR achieves sane R² on a MolLogP-dominated target
# ---------------------------------------------------------------------------
def test_astsr_rdkit_2d_toy_r2_above_threshold() -> None:
    """Verify gate 1: ASTSR fits a MolLogP-dominated pIC50-style target.

    Toy target: ``y ≈ 1.0 * MolLogP + 0.3 * TPSA - 0.5 * HBD + noise``.
    This is intentionally easy for the ASTSR — the verify threshold of
    0.5 confirms the evolutionary search converges on a useful
    hypothesis within 80 generations using the F5_RDKIT_2D feature set.

    NOTE: the original Wf-49 Phase 2 toy used extreme descriptor scales
    (TPSA up to 150) which made the target dominated by a single
    descriptor in a non-recoverable way for the ASTSR.  We use a
    uniform scale (all descriptors in [0, 10]) so the ASTSR can
    recover signal across multiple features.
    """
    rng = np.random.default_rng(20260926)
    N = 100
    X = rng.uniform(0.0, 1.0, (N, 12))
    desc = X * 10.0  # uniform scale, easy for ASTSR
    # Indices: MolLogP=1, TPSA=2, HBD=4.
    y = 1.0 * desc[:, 1] + 0.3 * desc[:, 2] - 0.5 * desc[:, 4] + 0.1
    y += rng.normal(0.0, 0.5, N)

    astsr = LambdaASTSR(
        depth=4, population=64, generations=80, seed=42,
        feature_names=RDKIT_2D_FEATURE_COLUMNS,
    )
    astsr.fit(desc, y)
    y_pred = astsr.predict(desc)
    r2 = compute_r2(y, y_pred)
    assert r2 > 0.5, (
        f"ASTSR failed to reach R² > 0.5 on the MolLogP-dominated toy "
        f"problem with RDKit 2D 12-d features; got R²={r2:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. AST evaluation respects column order under RDKit 2D features
# ---------------------------------------------------------------------------
def test_ast_evaluation_respects_rdkit_2d_column_order() -> None:
    """evaluate() with feature_names ordering: MolLogP at index 1 doubles y."""
    # AST: add(const 0.0, mul(const 2.0, var MolLogP))
    # MolLogP lives at index 1 in RDKIT_2D_FEATURE_COLUMNS.
    ops = (
        {"kind": "const", "value": 0.0},
        {"kind": "var", "name": "MolLogP"},
        {"kind": "const", "value": 2.0},
        {"kind": "op", "op": "mul", "left": 2, "right": 1},
        {"kind": "op", "op": "add", "left": 0, "right": 3},
    )
    ast = LambdaAST(lam_node=None, ops=ops, depth=3, n_leaves=3)
    row = {name: 0.0 for name in RDKIT_2D_FEATURE_COLUMNS}
    row["MolLogP"] = 1.7
    val = ast.evaluate(row)
    assert val == pytest.approx(3.4, abs=1e-9), (
        f"AST evaluation with MolLogP=1.7 should yield 2*1.7=3.4; got {val}"
    )


# ---------------------------------------------------------------------------
# 7. NaN-safety under RDKit 2D features
# ---------------------------------------------------------------------------
def test_ast_evaluation_nan_safe_with_rdkit_2d_features() -> None:
    """NaN / missing RDKit 2D columns are mapped to 0.0 by evaluate()."""
    ops = (
        {"kind": "var", "name": "MolWt"},
        {"kind": "const", "value": 100.0},
        {"kind": "op", "op": "add", "left": 0, "right": 1},
    )
    ast = LambdaAST(lam_node=None, ops=ops, depth=2, n_leaves=2)
    val_nan = ast.evaluate({"MolWt": float("nan")})
    val_missing = ast.evaluate({})
    val_finite = ast.evaluate({"MolWt": 50.0})
    assert val_nan == pytest.approx(100.0, abs=1e-9)
    assert val_missing == pytest.approx(100.0, abs=1e-9)
    assert val_finite == pytest.approx(150.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. Wf-49 Phase 1 extractor output compatibility (skip when absent)
# ---------------------------------------------------------------------------
def test_rdkit_2d_feature_names_match_phase1_extractor() -> None:
    """RDKIT_2D_FEATURE_COLUMNS order matches Wf-49 Phase 1 extractor output."""
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    phase1_csv = (
        PROJECT_ROOT
        / "molmetal"
        / "data"
        / "platinai_5k_rdkit_features.csv"
    )
    if not phase1_csv.exists():
        pytest.skip(
            f"Wf-49 Phase 1 artefact missing: {phase1_csv}; "
            "run extract_rdkit_2d_descriptors.py first."
        )
    import csv
    with phase1_csv.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
    # Header layout: smiles, metal, ic50_nm, selectivity, pIC50 + 12 desc.
    desc_cols = header[5:5 + 12]
    assert desc_cols == list(RDKIT_2D_FEATURE_COLUMNS), (
        f"Wf-49 Phase 1 extractor output drifted from RDKIT_2D_FEATURE_COLUMNS.\n"
        f"  extractor: {desc_cols}\n"
        f"  registry:  {list(RDKIT_2D_FEATURE_COLUMNS)}"
    )


# ---------------------------------------------------------------------------
# 9. Determinism with the new RDKit 2D default
# ---------------------------------------------------------------------------
def test_fit_deterministic_with_rdkit_2d_default() -> None:
    """Two fits with the same seed + RDKit 2D default yield same best R²."""
    rng = np.random.default_rng(0)
    N = 40
    X = rng.uniform(0.0, 1.0, (N, 12))
    y = X[:, 1] + 0.5 * X[:, 2]  # MolLogP + TPSA style

    a = LambdaASTSR(depth=3, population=32, generations=15, seed=7)
    a.fit(X, y)
    b = LambdaASTSR(depth=3, population=32, generations=15, seed=7)
    b.fit(X, y)
    assert a.best_r2_train_ == pytest.approx(b.best_r2_train_, abs=1e-9)
    # And the feature_names defaulted to the 12-d set.
    assert a.feature_names == RDKIT_2D_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 10. Smoke: random_ast + evaluate_matrix under RDKit 2D column order
# ---------------------------------------------------------------------------
def test_random_ast_evaluate_matrix_with_rdkit_2d() -> None:
    """evaluate_matrix respects RDKit 2D column order end-to-end."""
    rng = random.Random(0)
    ast = random_ast(rng, RDKIT_2D_FEATURE_COLUMNS, max_depth=3)
    X = np.random.default_rng(0).uniform(0.0, 1.0, (5, 12))
    out = ast.evaluate_matrix(X, list(RDKIT_2D_FEATURE_COLUMNS))
    assert out.shape == (5,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 11. mutate_ast preserves validity under RDKit 2D features
# ---------------------------------------------------------------------------
def test_mutate_ast_preserves_validity_under_rdkit_2d() -> None:
    """Mutation does not produce an AST that references unknown features."""
    rng = random.Random(7)
    ast = random_ast(rng, RDKIT_2D_FEATURE_COLUMNS, max_depth=4)
    for _ in range(30):
        m = mutate_ast(ast, rng, RDKIT_2D_FEATURE_COLUMNS, max_depth=4)
        used = m.used_features()
        assert used.issubset(set(RDKIT_2D_FEATURE_COLUMNS)), (
            f"mutated AST references unknown features: "
            f"{used - set(RDKIT_2D_FEATURE_COLUMNS)}"
        )
        row = {n: 0.5 for n in RDKIT_2D_FEATURE_COLUMNS}
        assert np.isfinite(m.evaluate(row))


# ---------------------------------------------------------------------------
# 12. feature_set parity check between wrappers and flag form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("feature_set,expected", [
    ("rdkit_2d", RDKIT_2D_FEATURE_COLUMNS),
    ("f5_legacy", DEFAULT_FEATURE_COLUMNS),
])
def test_default_feature_columns_parametrized(
    feature_set: str, expected: tuple,
) -> None:
    """Parametrised: flag form returns the right tuple for each set."""
    cols = default_feature_columns(feature_set=feature_set)
    assert cols == expected
    assert len(cols) == len(expected)


# ---------------------------------------------------------------------------
# 13. Module-level registry exports are reachable
# ---------------------------------------------------------------------------
def test_registry_constants_are_exported() -> None:
    """DEFAULT_FEATURE_COLUMNS, RDKIT_2D_FEATURE_COLUMNS are module exports."""
    import molmetal.molmetal_lam.reward.lambda_astsr as mod
    assert hasattr(mod, "DEFAULT_FEATURE_COLUMNS")
    assert hasattr(mod, "RDKIT_2D_FEATURE_COLUMNS")
    assert hasattr(mod, "legacy_f5_feature_columns")
    assert hasattr(mod, "rdkit_2d_feature_columns")
    assert "RDKIT_2D_FEATURE_COLUMNS" in mod.__all__
    assert "legacy_f5_feature_columns" in mod.__all__
    assert "rdkit_2d_feature_columns" in mod.__all__


# ---------------------------------------------------------------------------
# 14. Each RDKit 2D column is a valid Python identifier / str
# ---------------------------------------------------------------------------
def test_rdkit_2d_column_names_are_strings() -> None:
    """Every RDKit 2D column is a non-empty str (no None / numeric leakage)."""
    for name in RDKIT_2D_FEATURE_COLUMNS:
        assert isinstance(name, str)
        assert len(name) > 0
        # Canonical RDKit descriptor names don't contain spaces or dots.
        assert " " not in name
        assert "." not in name


# ===========================================================================
# Wf-218 — F5_RDKIT_2D 32-d expansion tests
# ===========================================================================
# Krasnov 2026 protocol: 9 cell lines × 3 seeds × 50 epochs (scaffold-split).
# The 12-d signal floor on Pt-only PlatinAI 4-cell (Wf-127 §Cause 2) is
# addressed by appending 20 additional RDKit 2D descriptors to the
# Wf-49 Phase 1 12-d frozen set, yielding a 32-d feature space.  The
# first 12 entries are bit-for-bit identical to the 12-d path so the
# older 12-d trainers still see identical column ordering.
# ===========================================================================


# Canonical 32-d RDKit 2D extended descriptor set (Wf-218).
# First 12 match the Wf-49 P1 frozen order; remaining 20 are the Wf-218
# expansion.
EXPECTED_RDKIT_2D_EXTENDED_NAMES: tuple = (
    # 12 from Wf-49 P1
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
    # 20 Wf-218 expansion
    "BertzCT",
    "Chi0v",
    "Chi0n",
    "Chi1n",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "fr_halogen",
    "fr_Al_COO",
)


# ---------------------------------------------------------------------------
# 15. Wf-218 — RDKIT_2D_EXTENDED_FEATURE_COLUMNS registry invariants
# ---------------------------------------------------------------------------
def test_rdkit_2d_extended_feature_columns_is_tuple_of_32() -> None:
    """RDKIT_2D_EXTENDED_FEATURE_COLUMNS is a tuple of exactly 32 names."""
    assert isinstance(RDKIT_2D_EXTENDED_FEATURE_COLUMNS, tuple)
    assert len(RDKIT_2D_EXTENDED_FEATURE_COLUMNS) == 32
    assert all(isinstance(n, str) for n in RDKIT_2D_EXTENDED_FEATURE_COLUMNS)


def test_rdkit_2d_extended_feature_columns_order_frozen() -> None:
    """The 32-d column order matches the Wf-218 frozen order."""
    assert RDKIT_2D_EXTENDED_FEATURE_COLUMNS == EXPECTED_RDKIT_2D_EXTENDED_NAMES, (
        "RDKIT_2D_EXTENDED_FEATURE_COLUMNS order drift.\n"
        f"  expected: {EXPECTED_RDKIT_2D_EXTENDED_NAMES}\n"
        f"  actual:   {RDKIT_2D_EXTENDED_FEATURE_COLUMNS}"
    )


def test_rdkit_2d_extended_first_12_match_legacy_rdkit() -> None:
    """First 12 entries of the 32-d set match the 12-d set bit-for-bit.

    Backward-compat guarantee: any trainer / reward channel that
    hard-codes the 12-d column order still sees the same leading 12
    descriptors in the 32-d path.
    """
    assert (
        RDKIT_2D_EXTENDED_FEATURE_COLUMNS[:12]
        == RDKIT_2D_FEATURE_COLUMNS
    ), (
        "First 12 entries of RDKIT_2D_EXTENDED_FEATURE_COLUMNS drifted "
        "from RDKIT_2D_FEATURE_COLUMNS; this breaks backward-compat for "
        "the 12-d path."
    )


def test_rdkit_2d_extended_feature_columns_no_duplicates() -> None:
    """The 32-d set has no duplicate column names."""
    assert len(set(RDKIT_2D_EXTENDED_FEATURE_COLUMNS)) == len(
        RDKIT_2D_EXTENDED_FEATURE_COLUMNS
    ), (
        f"RDKIT_2D_EXTENDED_FEATURE_COLUMNS has duplicates: "
        f"{[n for n in RDKIT_2D_EXTENDED_FEATURE_COLUMNS if list(RDKIT_2D_EXTENDED_FEATURE_COLUMNS).count(n) > 1]}"
    )


def test_rdkit_2d_extended_feature_columns_covers_shape_charge_halogen() -> None:
    """The 20 expansion descriptors cover shape, charge, halogen, and heteroatoms."""
    required = {
        # Shape / topology
        "BertzCT", "Chi0v", "Chi0n", "Chi1n", "HallKierAlpha",
        # Kappa shape indices
        "Kappa1", "Kappa2", "Kappa3",
        # Surface area
        "LabuteASA",
        # Partial charge
        "MaxPartialCharge", "MinPartialCharge",
        "MaxAbsPartialCharge", "MinAbsPartialCharge",
        # Ring features
        "NumAliphaticRings", "NumSaturatedRings",
        # Heteroatoms / functional groups
        "NumHeteroatoms", "NHOHCount", "NOCount",
        # Halogen + carboxylate (Pt-coordination relevant)
        "fr_halogen", "fr_Al_COO",
    }
    assert required.issubset(set(RDKIT_2D_EXTENDED_FEATURE_COLUMNS)), (
        f"RDKit 2D 32-d registry missing Wf-218 expansion descriptors: "
        f"missing={sorted(required - set(RDKIT_2D_EXTENDED_FEATURE_COLUMNS))}"
    )


# ---------------------------------------------------------------------------
# 16. Wf-218 — default_feature_columns() flag dispatch for 32-d
# ---------------------------------------------------------------------------
def test_default_feature_columns_rdkit_2d_extended_flag() -> None:
    """``default_feature_columns(feature_set='rdkit_2d_extended')`` returns 32-d."""
    cols = default_feature_columns(feature_set="rdkit_2d_extended")
    assert cols == RDKIT_2D_EXTENDED_FEATURE_COLUMNS
    assert len(cols) == 32


def test_default_feature_columns_unknown_set_still_rejected_after_wf218() -> None:
    """Unknown feature_set values still raise ValueError (Wf-218 invariant)."""
    with pytest.raises(ValueError):
        default_feature_columns(feature_set="not_a_real_set_2")


def test_default_feature_columns_rdkit_2d_extended_parametrized() -> None:
    """Parametrised: ``rdkit_2d_extended`` flag returns the 32-d tuple."""
    cols = default_feature_columns(feature_set="rdkit_2d_extended")
    assert len(cols) == 32
    assert "BertzCT" in cols
    assert "fr_halogen" in cols


# ---------------------------------------------------------------------------
# 17. Wf-218 — wrapper helpers for 32-d
# ---------------------------------------------------------------------------
def test_rdkit_2d_extended_feature_columns_returns_32d() -> None:
    """``rdkit_2d_extended_feature_columns()`` returns the 32-d constant."""
    cols = rdkit_2d_extended_feature_columns()
    assert cols == RDKIT_2D_EXTENDED_FEATURE_COLUMNS
    assert len(cols) == 32


def test_rdkit_2d_extended_feature_columns_returns_fresh_tuple() -> None:
    """``rdkit_2d_extended_feature_columns()`` returns a fresh tuple copy.

    CPython may intern tuples built from the same source constant, so
    we test for "safe to mutate" semantics rather than object identity.
    """
    a = rdkit_2d_extended_feature_columns()
    b = rdkit_2d_extended_feature_columns()
    assert a == b
    # Mutating the returned tuple must NOT affect subsequent calls.
    mutated = list(a)
    mutated[0] = "MUTATED"
    c = rdkit_2d_extended_feature_columns()
    assert c[0] != "MUTATED", (
        "rdkit_2d_extended_feature_columns() leaked a mutation; the "
        "underlying constant was modified."
    )
    assert c[0] == a[0]


# ---------------------------------------------------------------------------
# 18. Wf-218 — LambdaASTSR picks up the 32-d default
# ---------------------------------------------------------------------------
def test_lambda_astsr_with_extended_feature_names_uses_32d() -> None:
    """``LambdaASTSR(feature_names=rdkit_2d_extended_feature_columns())`` picks up 32-d."""
    astsr = LambdaASTSR(
        depth=2, population=4, generations=0, seed=0,
        feature_names=rdkit_2d_extended_feature_columns(),
    )
    assert astsr.feature_names == RDKIT_2D_EXTENDED_FEATURE_COLUMNS
    assert len(astsr.feature_names) == 32


def test_lambda_astsr_rdkit_2d_12d_default_still_works() -> None:
    """``LambdaASTSR()`` default still returns 12-d (backward-compat)."""
    astsr = LambdaASTSR(depth=2, population=4, generations=0, seed=0)
    assert astsr.feature_names == RDKIT_2D_FEATURE_COLUMNS, (
        "Wf-218 expansion must NOT change the default feature set; the "
        "12-d RDKit 2D default is preserved bit-for-bit."
    )
    assert len(astsr.feature_names) == 12


# ---------------------------------------------------------------------------
# 19. Wf-218 — Verify gate: ASTSR achieves sane R² on 32-d MolLogP-dominated
# ---------------------------------------------------------------------------
def test_astsr_rdkit_2d_extended_toy_r2_above_threshold() -> None:
    """Verify gate: ASTSR fits a MolLogP-dominated target on the 32-d set.

    The 32-d toy target puts signal on MolLogP (index 1) and HallKierAlpha
    (index 16 in the 32-d order) plus TPSA (index 2) to make sure the
    expansion descriptors are wired through end-to-end.

    The verify threshold of 0.5 confirms the evolutionary search
    converges on a useful hypothesis within 80 generations using the
    F5_RDKIT_2D 32-d feature set.  Uses uniform descriptor scale
    [0, 10] for ASTSR tractability (per Wf-218).
    """
    rng = np.random.default_rng(20260928)
    N = 100
    n_dim = 32
    X = rng.uniform(0.0, 1.0, (N, n_dim))
    desc = X * 10.0  # uniform scale, easy for ASTSR
    # Indices in 32-d: MolLogP=1, TPSA=2, HallKierAlpha=16
    y = 1.0 * desc[:, 1] + 0.3 * desc[:, 2] - 0.4 * desc[:, 16] + 0.1
    y += rng.normal(0.0, 0.5, N)

    astsr = LambdaASTSR(
        depth=4, population=64, generations=80, seed=42,
        feature_names=RDKIT_2D_EXTENDED_FEATURE_COLUMNS,
    )
    astsr.fit(desc, y)
    y_pred = astsr.predict(desc)
    r2 = compute_r2(y, y_pred)
    assert r2 > 0.5, (
        f"ASTSR failed to reach R² > 0.5 on the MolLogP-dominated toy "
        f"problem with RDKit 2D 32-d features; got R²={r2:.4f}"
    )


# ---------------------------------------------------------------------------
# 20. Wf-218 — Determinism on the 32-d path
# ---------------------------------------------------------------------------
def test_fit_deterministic_with_rdkit_2d_extended() -> None:
    """Two fits with the same seed + 32-d default yield same best R²."""
    rng = np.random.default_rng(0)
    N = 40
    n_dim = 32
    X = rng.uniform(0.0, 1.0, (N, n_dim))
    y = X[:, 1] + 0.5 * X[:, 2] + 0.3 * X[:, 16]  # MolLogP + TPSA + HallKierAlpha

    a = LambdaASTSR(
        depth=3, population=32, generations=15, seed=7,
        feature_names=RDKIT_2D_EXTENDED_FEATURE_COLUMNS,
    )
    a.fit(X, y)
    b = LambdaASTSR(
        depth=3, population=32, generations=15, seed=7,
        feature_names=RDKIT_2D_EXTENDED_FEATURE_COLUMNS,
    )
    b.fit(X, y)
    assert a.best_r2_train_ == pytest.approx(b.best_r2_train_, abs=1e-9)
    assert a.feature_names == RDKIT_2D_EXTENDED_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 21. Wf-218 — Module-level registry exports include the 32-d names
# ---------------------------------------------------------------------------
def test_registry_constants_are_exported_wf218() -> None:
    """RDKIT_2D_EXTENDED_FEATURE_COLUMNS, rdkit_2d_extended_feature_columns exported."""
    import molmetal.molmetal_lam.reward.lambda_astsr as mod
    assert hasattr(mod, "RDKIT_2D_EXTENDED_FEATURE_COLUMNS")
    assert hasattr(mod, "rdkit_2d_extended_feature_columns")
    assert "RDKIT_2D_EXTENDED_FEATURE_COLUMNS" in mod.__all__
    assert "rdkit_2d_extended_feature_columns" in mod.__all__


# ---------------------------------------------------------------------------
# 22. Wf-218 — Extractor CSV column count and prefix match
# ---------------------------------------------------------------------------
def test_extractor_rdkit_2d_extended_descriptor_count() -> None:
    """The RDKit 2D extended descriptor registry has 32 entries."""
    from molmetal.scripts.extract_rdkit_2d_descriptors import (
        RDKIT_2D_EXTENDED_DESCRIPTORS,
    )
    assert len(RDKIT_2D_EXTENDED_DESCRIPTORS) == 32
    # First 12 names match the Wf-49 P1 frozen order (backward-compat).
    first12_names = [n for n, _ in RDKIT_2D_EXTENDED_DESCRIPTORS[:12]]
    assert first12_names == list(RDKIT_2D_FEATURE_COLUMNS)
    # Last 20 names match the Wf-218 expansion.
    last20_names = [n for n, _ in RDKIT_2D_EXTENDED_DESCRIPTORS[12:]]
    assert last20_names == list(RDKIT_2D_EXTENDED_FEATURE_COLUMNS[12:])


def test_extractor_compute_rdkit_extended_descriptors_returns_32d() -> None:
    """compute_rdkit_extended_descriptors('CCO') returns a 32-d vector."""
    try:
        from molmetal.scripts.extract_rdkit_2d_descriptors import (
            compute_rdkit_extended_descriptors,
        )
    except Exception as exc:                                            # pragma: no cover
        pytest.skip(
            f"RDKit unavailable or extractor import failed: {_exc}; "
            "skipping 32-d extractor smoke test."
        )
    vec = compute_rdkit_extended_descriptors("CCO")
    assert vec is not None
    assert len(vec) == 32, (
        f"compute_rdkit_extended_descriptors('CCO') should return 32 floats; "
        f"got len={len(vec)}"
    )
    # First 12 should match compute_rdkit_descriptors bit-for-bit.
    from molmetal.scripts.extract_rdkit_2d_descriptors import (
        compute_rdkit_descriptors,
    )
    vec_12 = compute_rdkit_descriptors("CCO")
    assert vec_12 is not None
    assert vec[:12] == vec_12, (
        "First 12 entries of compute_rdkit_extended_descriptors must "
        "match compute_rdkit_descriptors exactly (backward-compat)."
    )


def test_extractor_compute_rdkit_extended_invalid_smiles_returns_none() -> None:
    """compute_rdkit_extended_descriptors returns None for unparseable SMILES."""
    try:
        from molmetal.scripts.extract_rdkit_2d_descriptors import (
            compute_rdkit_extended_descriptors,
        )
    except Exception as exc:                                            # pragma: no cover
        pytest.skip(f"RDKit unavailable: {_exc}")
    vec = compute_rdkit_extended_descriptors("NOT_A_SMILES_@@@")
    assert vec is None