from molmetal_lam.priors.metal_hydration import MetalHydrationAnalyzer
import inspect


def test_cisplatin_high_aquation():
    m = MetalHydrationAnalyzer()
    assert m.aquation_rate_proxy("N[Pt](N)(Cl)Cl") >= 0.7


def test_carboplatin_lower_than_cisplatin():
    m = MetalHydrationAnalyzer()
    assert m.aquation_rate_proxy("N[Pt]1(OC(=O)CC(=O)O1)N") <= m.aquation_rate_proxy("N[Pt](N)(Cl)Cl")


def test_ruthenium_bpy_inert():
    m = MetalHydrationAnalyzer("Ru", 2)
    s = "n1ccccc1[Ru](n2ccccc2)n3ccccc3"
    assert m.kinetic_inertness_score(s) >= 0.7
    assert m.aquation_rate_proxy(s) <= 0.5


def test_gsh_resistance_bulky():
    m = MetalHydrationAnalyzer()
    assert m.gs_resistance_flag("Cc1cc(C)nc(C)c1[Pt]Cl") is True


def test_reduction_class():
    assert MetalHydrationAnalyzer().reduction_potential_class("Cl[Pt]N") == "moderately_reduced"


def test_bounds_and_invalid_default():
    m = MetalHydrationAnalyzer()
    for s in ("Cl[Pt]Cl", "N[Pt](N)(Cl)Cl", "invalid"):
        assert 0 <= m.aquation_rate_proxy(s) <= 1
        assert 0 <= m.kinetic_inertness_score(s) <= 1
    assert m.aquation_rate_proxy("invalid") == 0.5
    assert m.kinetic_inertness_score("invalid") == 0.5


def test_cisplatin_trans_effect_is_weak():
    result = MetalHydrationAnalyzer().trans_effect_indicator("Cl[Pt](N)(N)Cl")
    assert result["trans_labilization_score"] <= 0.5


def test_phosphine_trans_donor_is_strong():
    result = MetalHydrationAnalyzer().trans_effect_indicator("Cl[Pt](N)(P(C)(C)C)Cl")
    assert result["trans_labilization_score"] >= 0.7


def test_trans_effect_invalid_smiles_fallback():
    result = MetalHydrationAnalyzer().trans_effect_indicator("invalid")
    assert result == {"strongest_trans_donor": None,
                      "trans_labilization_score": 0.5,
                      "labile_position_index": None}


def test_trans_effect_donor_type_and_constructor_signature():
    result = MetalHydrationAnalyzer().trans_effect_indicator("Cl[Pt](N)(N)Cl")
    assert result["strongest_trans_donor"] is None or isinstance(result["strongest_trans_donor"], str)
    signature = inspect.signature(MetalHydrationAnalyzer.__init__)
    assert list(signature.parameters) == ["self", "metal", "oxidation_state", "default_pH"]
