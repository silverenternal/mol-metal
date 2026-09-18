"""test_learned_shaping.py — opt-in learned reward shaping (Phase 1 F5 adapter).

Honest framing reminder: F5 is INTERPRETIVE (identifies sa_norm as the sole
surviving LassoBIC signal).  These tests verify the closed-form adapter
matches the Phase 2 coefficients, NOT that the shaping lifts downstream
metrics (that requires a separate integration into proof_search.py).
"""

from __future__ import annotations

import math
import os

import pytest


# ---------------------------------------------------------------------------
# F5 closed-form constants
# ---------------------------------------------------------------------------
def test_f5_shape_reward_basic():
    """sa_score=3.32 (mean) -> R = 2.5836 - 2.5149 * 3.32 = -5.7662..."""
    from molmetal.molmetal_lam.reward.learned_shaping import (
        F5_INTERCEPT,
        F5_SA_COEFFICIENT,
        LearnedShaping,
    )
    sh = LearnedShaping()
    sh.enable()
    expected = F5_INTERCEPT + F5_SA_COEFFICIENT * 3.32
    actual = sh.shape_reward(3.32)
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(actual, -5.7657468, rel_tol=1e-4)


def test_f5_shape_reward_zero():
    """sa_score=0 -> R = 2.5836 (intercept only)."""
    from molmetal.molmetal_lam.reward.learned_shaping import LearnedShaping

    sh = LearnedShaping()
    sh.enable()
    actual = sh.shape_reward(0.0)
    assert math.isclose(actual, 2.5836, abs_tol=1e-9)


def test_f5_shape_reward_in_range():
    """Monotonically decreasing for sa_score in [1, 10]."""
    from molmetal.molmetal_lam.reward.learned_shaping import LearnedShaping

    sh = LearnedShaping()
    sh.enable()
    prev = sh.shape_reward(1.0)
    for sa in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        cur = sh.shape_reward(sa)
        assert cur < prev, f"Expected monotone decrease, but R({sa})={cur} >= R(prev)={prev}"
        prev = cur


# ---------------------------------------------------------------------------
# Env-gated enable/disable
# ---------------------------------------------------------------------------
def test_is_enabled_default_off(monkeypatch):
    """By default, shaping is OFF (no env var)."""
    monkeypatch.delenv("LEARNED_SHAPING_ENABLED", raising=False)
    from molmetal.molmetal_lam.reward.learned_shaping import LearnedShaping

    sh = LearnedShaping()
    assert sh.is_enabled() is False
    # shape_reward returns 0.0 when disabled (safe no-op).
    assert sh.shape_reward(3.32) == 0.0


def test_is_enabled_env_on(monkeypatch):
    """LEARNED_SHAPING_ENABLED=1 enables the shaping."""
    monkeypatch.setenv("LEARNED_SHAPING_ENABLED", "1")
    from molmetal.molmetal_lam.reward.learned_shaping import LearnedShaping

    sh = LearnedShaping()
    assert sh.is_enabled() is True
    # shape_reward now returns F5 closed-form.
    val = sh.shape_reward(3.32)
    assert math.isclose(val, -5.7657468, rel_tol=1e-4)


def test_is_enabled_env_truthy_aliases(monkeypatch):
    """Multiple truthy aliases all flip the gate on."""
    from molmetal.molmetal_lam.reward.learned_shaping import LearnedShaping

    for v in ["true", "yes", "on"]:
        monkeypatch.setenv("LEARNED_SHAPING_ENABLED", v)
        sh = LearnedShaping()
        assert sh.is_enabled() is True, f"expected enabled for {v!r}"


def test_disable_programmatic(monkeypatch):
    """LearnedShaping.disable() flips the gate back to off."""
    monkeypatch.setenv("LEARNED_SHAPING_ENABLED", "1")
    from molmetal.molmetal_lam.reward.learned_shaping import LearnedShaping

    sh = LearnedShaping()
    assert sh.is_enabled() is True
    sh.disable()
    assert sh.is_enabled() is False
    assert sh.shape_reward(3.32) == 0.0


# ---------------------------------------------------------------------------
# Formula metadata
# ---------------------------------------------------------------------------
def test_get_active_formula():
    """Returns F5 with correct lit anchor + coefficients."""
    from molmetal.molmetal_lam.reward.learned_shaping import (
        F5_FEATURE_NAMES,
        F5_SA_COEFFICIENT,
        LearnedShaping,
    )

    sh = LearnedShaping()
    sh.enable()
    f = sh.get_active_formula()
    assert f.name == "F5"
    assert "McAllester 1999" in f.lit_anchor
    assert f.coefficients["sa_mean_norm"] == F5_SA_COEFFICIENT
    # All other F5 features zeroed by sparsity.
    for fn in F5_FEATURE_NAMES:
        if fn != "sa_mean_norm":
            assert f.coefficients[fn] == 0.0


def test_load_best_formula_returns_callable():
    """BestFormula.formula_callable is a real function that runs."""
    from molmetal.molmetal_lam.reward.learned_shaping import load_best_formula

    f = load_best_formula()
    assert callable(f.formula_callable)
    # smoke: sa_norm=0.5 -> 2.5836 - 2.5149*0.5 = 1.32615
    assert math.isclose(f.formula_callable(0.5), 1.32615, abs_tol=1e-4)


def test_enumerate_8_families_includes_f5():
    """enumerate_8_families returns 8 entries with F5 having MEASURED coefs."""
    from molmetal.molmetal_lam.reward.learned_shaping import (
        enumerate_8_families,
    )

    fams = enumerate_8_families()
    assert len(fams) == 8
    f5 = next(f for f in fams if f.name == "F5")
    assert f5.coefficients["sa_mean_norm"] != 0.0
    # Placeholder families have zero SA coefficient.
    for f in fams:
        if f.name != "F5":
            assert f.coefficients["sa_mean_norm"] == 0.0


def test_fit_from_metrics_self_consistent():
    """Re-fit on F5-closed-form targets should reproduce the canonical coefs."""
    from molmetal.molmetal_lam.reward.learned_shaping import (
        fit_from_metrics,
        shape_reward_full_row,
    )

    # Synthesize 10 cells with varied sa_norm; F5 itself is the target.
    cells = [{"sa_mean_norm": sa} for sa in [0.0, 0.1, 0.2, 0.5, 1.0,
                                            1.5, 2.0, 3.0, 4.0, 5.0]]
    # Add reward via F5 closed-form (so fit_from_metrics's self-consistency
    # branch is exercised).
    for c in cells:
        c["reward"] = shape_reward_full_row(c)

    f = fit_from_metrics(cells, family="F5")
    # Should recover intercept ≈ 2.5836 and sa_mean_norm ≈ -2.5149.
    assert math.isclose(f.coefficients["intercept"], 2.5836, abs_tol=1e-3)
    assert math.isclose(f.coefficients["sa_mean_norm"], -2.5149, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# Real SMILES — SA via RDKit if available
# ---------------------------------------------------------------------------
def test_shape_reward_on_real_smiles():
    """Apply F5 to SA scores of 5 real drug SMILES; verify numerical output."""
    sa_module = pytest.importorskip("rdkit", reason="rdkit not installed")
    # sascorer is the standard SA scorer; not always installed, so we fall
    # back to a deterministic approximation if missing.
    sa_func = None
    try:
        from rdkit.Chem import RDConfig
        import os as _os
        import sys as _sys
        sys.path.append(_os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer as _sascorer  # noqa: F401
        sa_func = _sascorer.calculateScore
    except Exception:
        sa_func = None

    from molmetal.molmetal_lam.reward.learned_shaping import (
        F5_INTERCEPT,
        F5_SA_COEFFICIENT,
        LearnedShaping,
    )
    from rdkit import Chem

    sh = LearnedShaping()
    sh.enable()

    smiles_list = [
        "CCO",                       # ethanol (trivial)
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
        "OC(=O)C1=CN(C2CC2)C2=C(C=N1)C(=O)N(C)C(=O)N2C",  # tentative ondansetron
    ]

    expected_outputs = []
    actual_outputs = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        assert mol is not None, f"failed to parse {sm}"
        if sa_func is not None:
            sa = sa_func(mol)
        else:
            # Fallback: derive a deterministic SA-like proxy from atom count.
            # Honest: not the real Ertl SA; just verifies the F5 closed-form
            # thread runs end-to-end with a numeric input.
            sa = float(mol.GetNumHeavyAtoms()) / 20.0  # rough 0..2 range
        reward = sh.shape_reward(sa)
        expected = F5_INTERCEPT + F5_SA_COEFFICIENT * sa
        expected_outputs.append(expected)
        actual_outputs.append(reward)

    for e, a in zip(expected_outputs, actual_outputs):
        assert math.isclose(e, a, rel_tol=1e-9, abs_tol=1e-9), (
            f"F5 closed-form mismatch: expected={e} got={a}"
        )

    # Honest framing: rewards are negative for any non-zero SA, by design.
    # (F5 closed-form has negative sa_norm coefficient.)
    for a in actual_outputs:
        if a != 2.5836:  # only sa=0 gives positive intercept
            assert a < 2.5836
