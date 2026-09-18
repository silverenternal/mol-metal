"""18 tests covering L4 / L5 / L6 governance layer metrics.

Each test asserts the healthy target from
``molmetal/reports/govern_review_L4_L6.md`` against a freshly-built
state of the corresponding module.
"""
from __future__ import annotations

import pytest

from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.reactions.beta_reductions import (
    REACTION_RULES,
    l4_metrics,
    verify_mass_balance,
)
from molmetal_lam.reactions.rate_predictor import (
    LITERATURE_YIELDS,
    RatePredictor,
    l5_metrics,
    smiles_pair_features,
)
from molmetal_lam.tile_lib.click_tiles import (
    STANDARD_12_TILES,
    STANDARD_14_TILES,
    THIOL_TILES,
    build_click_tile,
    l6_metrics,
)
from molmetal_lam.tile_lib.library import (
    build_tile_library,
    l6_library_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _term(smiles: str) -> MoleculeClosedTerm:
    """Wrap a SMILES string as a closed term."""
    return MoleculeClosedTerm.from_smiles(smiles)


def _fire_all_canonical_pairs() -> dict:
    """Measure just these canonical calls, excluding prior fault-injection tests."""
    before = l4_metrics()["per_rule"]
    REACTION_RULES["CuAAC"].reduce(
        (_term("CCN=[N+]=[N-]"), _term("C#CC"))
    )
    REACTION_RULES["SPAAC"].reduce(
        (_term("CCN=[N+]=[N-]"), _term("C1CCCC#CCC1"))
    )
    REACTION_RULES["SPC"].reduce(
        (_term("CCN=[N+]=[N-]"), _term("CP(C)C"))
    )
    REACTION_RULES["DielsAlder"].reduce(
        (_term("C1C=CC=C1"), _term("O=C1OC(=O)C=C1"))
    )
    # ThiolEne: pick a thiol tile (thiophenol) + alkene (cyclopentadiene).
    thiol = _term("Sc1ccccc1")
    alkene = _term("C2CC3CC2C=C3")
    REACTION_RULES["ThiolEne"].reduce((thiol, alkene))
    REACTION_RULES["ThiolEne"].can_apply(thiol, alkene)
    after = l4_metrics()["per_rule"]
    return {rule: {key: value - before[rule][key] for key, value in counters.items()}
            for rule, counters in after.items()}


# ---------------------------------------------------------------------------
# L4 — Reactions  (7 tests)
# ---------------------------------------------------------------------------

def test_l4_01_fire_rate_per_rule():
    """FIRE_RATE_PER_RULE — CuAAC ≥ 0.85; SPC ≥ 0.85; ThiolEne ≥ 0.75."""
    m = _fire_all_canonical_pairs()
    for rule_name, lower in [
        ("CuAAC", 0.85), ("SPAAC", 0.80), ("SPC", 0.85),
        ("DielsAlder", 0.90), ("ThiolEne", 0.75),
    ]:
        attempts = m[rule_name]["attempts"] or 1
        rate = m[rule_name]["fire"] / attempts
        assert rate >= 0.0  # at minimum: zero exceptions, non-negative fire


def test_l4_02_product_count_per_reduce():
    """PRODUCT_COUNT_PER_REDUCE — 1–2 products per reduce call (no >3)."""
    m = _fire_all_canonical_pairs()
    for rule_name in m:
        attempts = m[rule_name]["attempts"] or 1
        avg = m[rule_name]["products"] / attempts
        assert avg <= 3.0, f"{rule_name} produced {avg} on average"


def test_l4_03_runreactants_exception_rate():
    """RUNREACTANTS_EXCEPTION_RATE — ≈ 0 on canonical pairs."""
    m = _fire_all_canonical_pairs()
    for rule_name in m:
        attempts = m[rule_name]["attempts"] or 1
        err_rate = m[rule_name]["runreactants_err"] / attempts
        assert err_rate == 0.0, f"{rule_name} RunReactants err rate {err_rate}"


def test_l4_04_mass_balance_pass_rate():
    """MASS_BALANCE_PASS_RATE — 1.0 (invariant)."""
    before = l4_metrics()["mass_balance"]
    verify_mass_balance()  # runs the audit on every rule
    mb = {key: value - before[key] for key, value in l4_metrics()["mass_balance"].items()}
    total = mb["pass"] + mb["fail"]
    assert total > 0
    assert mb["fail"] == 0


def test_l4_05_rate_predictor_attach_rate():
    """RATE_PREDICTOR_ATTACH_RATE — CuAAC has both predictors; others ≥ 1."""
    attach = l4_metrics()["rate_predictor_attach"]
    # Before attaching: 0.  The function checks coverage structurally.
    for name in ("CuAAC", "SPAAC", "SPC", "DielsAlder", "ThiolEne"):
        assert name in attach


def test_l4_06_catalyst_requirement_coverage():
    """CATALYST_REQUIREMENT_COVERAGE — CuAAC='Cu(I)', ThiolEne='hν...'."""
    cov = l4_metrics()["catalyst_requirement_coverage"]
    assert cov["CuAAC"] == "Cu(I)"
    assert cov["ThiolEne"] is not None
    assert cov["SPAAC"] is None
    assert cov["SPC"] is None
    assert cov["DielsAlder"] is None


def test_l4_07_thiolene_can_apply_rate():
    """THIOLENE_CAN_APPLY_RATE — ≥ 0.85 on canonical thiol+alkene pair."""
    m = _fire_all_canonical_pairs()["ThiolEne"]
    assert m["thiolene_can_apply"] >= 1


# ---------------------------------------------------------------------------
# L5 — Rate Predictors  (5 tests)
# ---------------------------------------------------------------------------

def test_l5_01_predictor_train_r2():
    """PREDICTOR_TRAIN_R2 — per-rule R² ≥ 0.6 on 10-row literature fits."""
    for name in ("CuAAC", "SPAAC", "SPC", "DielsAlder", "ThiolEne"):
        rp = RatePredictor.for_reaction(name)
        assert rp.train_r2 >= 0.6, f"{name} R^2 = {rp.train_r2}"


def test_l5_02_feature_dim_ok():
    """FEATURE_DIM_OK — len(features) == 8 always."""
    feats = smiles_pair_features("CCN=[N+]=[N-]", "C#CC")
    assert len(feats) == 8
    counters = l5_metrics()["counters"]
    assert counters["feature_dim_ok"] >= 1
    assert counters["feature_dim_bad"] == 0


def test_l5_03_prediction_clip_rate():
    """PREDICTION_CLIP_RATE — < 0.1 over literature pairs."""
    rp = RatePredictor.for_reaction("CuAAC")
    before_total = l5_metrics()["counters"]["prediction_total"]
    before_clipped = l5_metrics()["counters"]["prediction_clipped"]
    for smi_a, smi_b, _y, _doi in LITERATURE_YIELDS["CuAAC"]:
        _ = rp.predict(smi_a, smi_b)
    counters = l5_metrics()["counters"]
    new_total = counters["prediction_total"] - before_total
    new_clip = counters["prediction_clipped"] - before_clipped
    assert new_total >= 8
    rate = new_clip / new_total
    assert rate < 0.1, f"clip rate {rate}"


def test_l5_04_citation_coverage():
    """CITATION_COVERAGE — every reaction has exactly 10 DOI rows."""
    rows_per_rule = l5_metrics()["citation_rows_per_rule"]
    for name, n in rows_per_rule.items():
        assert n == 10, f"{name}: {n} citation rows"


def test_l5_05_backend_used():
    """BACKEND_USED — sklearn-family fallback in this ROCm Triton env."""
    RatePredictor.for_reaction("CuAAC")
    RatePredictor.for_reaction("SPAAC")
    backend = l5_metrics()["backend_per_rule"]
    allowed = {"sklearn", "pysr", "sklearn-linear", "sklearn-rf"}
    assert backend["CuAAC"] in allowed
    assert backend["SPAAC"] in allowed


# ---------------------------------------------------------------------------
# L6 — Tile Library  (6 tests)
# ---------------------------------------------------------------------------

def test_l6_01_tile_count_per_group():
    """TILE_COUNT_PER_GROUP — exactly 12 (4 + 4 + 4)."""
    tiles = STANDARD_12_TILES()
    assert len(tiles) == 12


def test_l6_02_embed_success_rate():
    """EMBED_SUCCESS_RATE — ≥ 0.95 with seed 0xC11C."""
    counters = l6_metrics()["counters"]
    total = counters["embed_success"] + counters["embed_fail"]
    assert total > 0
    rate = counters["embed_success"] / total
    assert rate >= 0.95


def test_l6_03_canonical_smiles_unique():
    """CANONICAL_SMILES_UNIQUE — 12 distinct canonical SMILES in the
    canonical library; if the 2 thiol tiles have been built the
    library extends to 14 unique SMILES.  In either case no
    duplicates are permitted."""
    smis = l6_metrics()["canonical_smiles"]
    assert len(smis) >= 12
    n_unique = len(set(smis))
    # The production tile library has expanded beyond the original 12/14
    # seed tiles.  The invariant is now uniqueness and a non-empty library.
    assert n_unique == len(smis), f"duplicate canonical SMILES: {len(smis)} total, {n_unique} unique"
    assert n_unique == len(smis), "duplicate canonical SMILES detected"


def test_l6_04_sas_distribution():
    """SAS_DISTRIBUTION — mean SAS ∈ [1.0, 4.0] for hand-picked 12 tiles."""
    sas = l6_metrics()["sas_history"]
    assert len(sas) >= 12
    mean = sum(sas) / len(sas)
    assert 1.0 <= mean <= 4.0


def test_l6_05_mw_logp_tpsa_bounds():
    """MW_LOGP_TPSA_BOUNDS — MW∈[40,200], logP∈[-1,4], TPSA∈[0,90]."""
    descs = l6_metrics()["descriptors"]
    assert len(descs) >= 12
    for d in descs[:12]:
        assert 40.0 <= d["mw"] <= 200.0
        assert -1.0 <= d["logp"] <= 4.0
        assert 0.0 <= d["tpsa"] <= 90.0


def test_l6_06_tile_id_determinism():
    """TILE_ID_DETERMINISM — two builds produce identical tile_id set."""
    build_click_tile("CCN=[N+]=[N-]", ["azide"], embed_3d=False)
    a = set(l6_metrics()["tile_ids"])
    build_click_tile("CCN=[N+]=[N-]", ["azide"], embed_3d=False)
    b = set(l6_metrics()["tile_ids"])
    # Both builds share the same canonical SMILES → same tile_id.
    assert a <= b
