"""Tests for click-chemistry ReactionRule subclasses + rate predictor.

This file covers T8: each ``ReactionRule`` subclass must produce at
least one product on a canonical reactant pair, and the
:class:`HeuristicRegressor`-backed rate predictor must yield
sane [0, 1] predictions on the same pairs.

Run with::

    source .venv/bin/activate && python -m pytest \\
        molmetal/molmetal_lam/tests/test_reaction_operators.py -v
"""

from __future__ import annotations

import pytest

# ------------------------------------------------------------------
# Imports under test
# ------------------------------------------------------------------
from molmetal_lam.reactions.beta_reductions import (
    REACTION_RULES,
    CuAAC,
    SPAAC,
    SPC,
    DielsAlder,
    ThiolEne,
    attach_all_rate_predictors,
)
from molmetal_lam.reactions.rate_predictor import (
    LITERATURE_YIELDS,
    RatePredictor,
    smiles_pair_features,
    train_rate_predictor,
    predict_yield,
)
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _term(smiles: str) -> MoleculeClosedTerm:
    """Parse a SMILES into a MoleculeClosedTerm without 3D embedding."""
    return MoleculeClosedTerm.from_smiles(smiles, embed_3d=False)


# ------------------------------------------------------------------
# 1. Each ReactionRule subclass reduces a canonical pair to >= 1 product
# ------------------------------------------------------------------

@pytest.mark.parametrize("rule_name,smi_a,smi_b,expected_product_atoms", [
    # CuAAC : ethyl azide + propyne -> 8-atom triazole
    ("CuAAC",      "CCN=[N+]=[N-]",  "C#CC",           8),
    # SPAAC : benzyl azide + cyclooctyne -> triazole fused to octane (18 heavy atoms)
    ("SPAAC",      "[N-]=[N+]=NCc1ccccc1", "C1CCCC#CCC1", 18),
    # SPC   : ethyl azide + methylphosphine -> iminophosphorane + N2 (RDKit reports both)
    ("SPC",        "CCN=[N+]=[N-]",  "CP",             7),
    # DA    : butadiene + ethylene -> cyclohexene (6 atoms)
    ("DielsAlder", "C=CC=C",         "C=C",            6),
    # Thiol-ene : methylthiol + propylene -> connected thioether
    ("ThiolEne",   "CS",             "C=CC",           5),
])
def test_rule_reduce_returns_at_least_one_product(
    rule_name, smi_a, smi_b, expected_product_atoms
) -> None:
    """Each click rule must yield >= 1 product on its canonical pair."""
    rule = REACTION_RULES[rule_name]
    a = _term(smi_a)
    b = _term(smi_b)
    products = rule.reduce((a, b))
    assert len(products) >= 1, (
        f"{rule_name}({smi_a!r}, {smi_b!r}) produced no products"
    )
    # The first product must be a closed term.
    p = products[0]
    assert isinstance(p, MoleculeClosedTerm)
    # Heavy-atom conservation (SPC explicitly retains its nitrogen byproduct).
    assert p.n_atoms == expected_product_atoms, (
        f"{rule_name}: expected {expected_product_atoms} atoms, got "
        f"{p.n_atoms} in {p.canonical_smiles()}"
    )


# ------------------------------------------------------------------
# 2. Rate predictor — featurisation shape + sanity bounds
# ------------------------------------------------------------------

def test_features_shape_is_eight() -> None:
    """smiles_pair_features must return an 8-d descriptor."""
    feats = smiles_pair_features("CCN=[N+]=[N-]", "C#CC")
    assert len(feats) == 8
    # First four = a's mw/logP/tpsa/heavy; last four = b's.
    assert feats[0] > 0 and feats[3] > 0
    assert feats[4] > 0 and feats[7] > 0


def test_rate_predictor_returns_yield_in_unit_interval() -> None:
    """predict_yield must return a value in [0, 1] on a held-out pair."""
    fitted = attach_all_rate_predictors()
    rule = REACTION_RULES["CuAAC"]
    y = rule.predict_yield("CCN=[N+]=[N-]", "C#CC")
    assert 0.0 <= y <= 1.0, f"yield {y} out of [0,1]"


# ------------------------------------------------------------------
# 3. Dataset + per-reaction training smoke test
# ------------------------------------------------------------------

@pytest.mark.parametrize("reaction_name", [
    "CuAAC",
    "SPAAC",
    "SPC",
    "DielsAlder",
    "ThiolEne",
    "ClickCuAAC_aryl_variant",
])
def test_dataset_has_ten_rows_per_reaction(reaction_name: str) -> None:
    """The literature dataset must have exactly 10 rows per reaction."""
    rows = LITERATURE_YIELDS[reaction_name]
    assert len(rows) == 10, (
        f"{reaction_name}: expected 10 citations, got {len(rows)}"
    )
    # Every row must carry a DOI pointer.
    for smi_a, smi_b, yld, doi in rows:
        assert smi_a and smi_b
        assert 0.0 <= yld <= 1.0
        assert doi.startswith("10.")


def test_per_reaction_training_succeeds() -> None:
    """Each reaction's HeuristicRegressor should fit on its 10-row dataset."""
    for name in LITERATURE_YIELDS:
        reg = train_rate_predictor(name)
        assert reg is not None
        # Predict a known pair (first row of that reaction's data).
        smi_a, smi_b, yld, _doi = LITERATURE_YIELDS[name][0]
        feats = smiles_pair_features(smi_a, smi_b)
        import numpy as np
        yhat = float(reg.predict(np.asarray(feats).reshape(1, -1))[0])
        # The prediction must lie in [0, 1]; we do not require accuracy
        # at n=10 (the sklearn-linear / sklearn-rf backend can be coarse).
        assert 0.0 <= yhat <= 1.0, (
            f"{name}: prediction {yhat} out of bounds for ({smi_a},{smi_b})"
        )


# ------------------------------------------------------------------
# 4. RatePredictor facade — wired into REACTION_RULES at import
# ------------------------------------------------------------------

def test_rate_predictors_attached_to_rules() -> None:
    """attach_all_rate_predictors must leave every click rule wired."""
    fitted = attach_all_rate_predictors()
    assert "CuAAC" in fitted and "SPAAC" in fitted
    assert "SPC" in fitted and "DielsAlder" in fitted
    assert "ThiolEne" in fitted and "ClickCuAAC_aryl_variant" in fitted
    for rule_name in ("CuAAC", "SPAAC", "SPC", "DielsAlder", "ThiolEne"):
        rule = REACTION_RULES[rule_name]
        assert rule.rate_predictor is not None, (
            f"{rule_name} has no attached rate_predictor"
        )
        assert rule.rate_predictor.reaction_name == rule_name
    # CuAAC also carries an aryl-variant predictor as second model.
    assert hasattr(REACTION_RULES["CuAAC"], "aryl_predictor")
    assert REACTION_RULES["CuAAC"].aryl_predictor.reaction_name == (
        "ClickCuAAC_aryl_variant"
    )


# ------------------------------------------------------------------
# 5. Class identity — each ReactionRule subclass is properly subclassed
# ------------------------------------------------------------------

def test_subclass_identity() -> None:
    """REACTION_RULES instances are the documented ReactionRule subclasses."""
    assert isinstance(REACTION_RULES["CuAAC"], CuAAC)
    assert isinstance(REACTION_RULES["SPAAC"], SPAAC)
    assert isinstance(REACTION_RULES["SPC"], SPC)
    assert isinstance(REACTION_RULES["DielsAlder"], DielsAlder)
    assert isinstance(REACTION_RULES["ThiolEne"], ThiolEne)
    # Every rule carries a ReactionRule name + stoichiometry.
    for name, rule in REACTION_RULES.items():
        assert rule.name == name
        assert rule.stoichiometry == {}
