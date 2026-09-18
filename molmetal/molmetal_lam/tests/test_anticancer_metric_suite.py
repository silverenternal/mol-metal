from molmetal_lam.priors.anticancer_metric_suite import AnticancerMetricSuite
from molmetal_lam.search_alg.proof_search import RewardAggregator

def test_invalid_smiles_fallbacks():
    s = AnticancerMetricSuite()
    for fn in (s.logp_anticancer, s.tpsa_iv, s.rotatable_bonds, s.herg_proxy):
        assert fn("not-a-smiles") == 0.5

def test_composite_unit_interval():
    assert 0.0 <= AnticancerMetricSuite().composite_score("Cl[Pt](N)(N)Cl") <= 1.0

def test_l5_snapshot():
    assert len(AnticancerMetricSuite().l5_metrics()) >= 6

def test_register_wires_composite_and_state_smiles():
    suite = AnticancerMetricSuite()
    agg = RewardAggregator()
    agg.register_anticancer_channels(suite)
    class State:
        def canonical_smiles(self):
            return "CCO"
    assert "anticancer_composite" in agg.metrics()
    assert 0.0 <= agg.r_anticancer_composite(State()) <= 1.0

def test_mw_flags_expose_metal_adjusted_protocol():
    suite = AnticancerMetricSuite()
    flags = suite.adjusted_flags("Cl[Pt](N)(N)Cl")
    assert set(flags) == {"mw_lipinski", "mw_metal_adjusted"}
    assert suite.molecular_weight("Cl[Pt](N)(N)Cl") > 290.0
    assert suite.adjusted_flags("not-a-smiles")["mw_lipinski"] is False

def test_descriptor_report_exposes_raw_iv_metrics_and_invalid_flags():
    suite = AnticancerMetricSuite()
    report = suite.descriptor_report("CCO")
    assert set(("logP", "TPSA", "RotB", "MW")).issubset(report)
    assert report["MW"] > 40.0
    assert report["RotB_in_range"] is True
    bad = suite.descriptor_report("not-a-smiles")
    assert bad["MW_lipinski"] is False
    assert bad["logP"] != bad["logP"]  # NaN, explicit invalid marker


import math
import warnings
from types import SimpleNamespace
import pytest
from rdkit import Chem
from molmetal_lam.priors import anticancer_metric_suite as metrics_module


@pytest.mark.parametrize("smiles", ["", " ", "\t\n", "*", "C*", "not-a-smiles", None])
def test_invalid_reports_never_pass_flags(smiles):
    suite = AnticancerMetricSuite()
    row = suite.descriptor_report(smiles)
    assert all(math.isnan(row[k]) for k in ("logP", "TPSA", "RotB", "MW"))
    assert not any(v for v in row.values() if isinstance(v, bool))
    assert not any(suite.adjusted_flags(smiles).values())
    assert math.isnan(suite.molecular_weight(smiles))


def test_empty_rdkit_molecule_is_invalid(monkeypatch):
    suite = AnticancerMetricSuite()
    monkeypatch.setattr(suite, "_mol", lambda s: Chem.MolFromSmiles(""))
    assert not suite.descriptor_report("CCO")["descriptor_valid"]


@pytest.mark.parametrize("mw,expected", [(299.999, False), (300, True), (500, True),
                                        (700, True), (700.001, False), (800, False)])
def test_metal_mw_descriptive_range_boundaries(monkeypatch, mw, expected):
    monkeypatch.setattr(metrics_module.Descriptors, "MolWt", lambda mol: mw)
    suite = AnticancerMetricSuite()
    assert suite.descriptor_report("CCO")["MW_metal_adjusted"] is expected
    assert suite.adjusted_flags("CCO")["mw_metal_adjusted"] is expected
    assert suite.adjusted_flags("CCO")["mw_lipinski"] is (mw <= 500)


@pytest.mark.parametrize("rotb,expected", [(9, True), (10, False), (11, False)])
def test_raw_rotb_is_strict_but_reward_remains_compatible(monkeypatch, rotb, expected):
    monkeypatch.setattr(metrics_module.Descriptors, "NumRotatableBonds", lambda mol: rotb)
    suite = AnticancerMetricSuite()
    assert suite.descriptor_report("CCO")["RotB_in_range"] is expected
    if rotb == 10:
        assert suite.rotatable_bonds("CCO") == 1.0


@pytest.mark.parametrize("descriptor", ["MolWt", "MolLogP", "TPSA", "NumRotatableBonds"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_descriptor_invalidates_entire_report(monkeypatch, descriptor, value):
    monkeypatch.setattr(metrics_module.Descriptors, descriptor, lambda mol: value)
    row = AnticancerMetricSuite().descriptor_report("CCO")
    assert not any(v for v in row.values() if isinstance(v, bool))
    assert all(math.isnan(row[k]) for k in ("logP", "TPSA", "RotB", "MW"))


def test_flags_parse_once(monkeypatch):
    suite = AnticancerMetricSuite()
    original = suite._mol
    calls = []
    def parse(s):
        calls.append(s)
        return original(s)
    monkeypatch.setattr(suite, "_mol", parse)
    suite.adjusted_flags("CCO")
    assert calls == ["CCO"]


def _complex(smiles, vina):
    return SimpleNamespace(molecule=SimpleNamespace(smiles=smiles), vina_score=vina)


def test_pilot_uses_all_input_denominators_and_finite_means():
    from molmetal_lam.scripts.pocket2mol_vs_lambda_1h36 import _aggregate_metrics
    rows = [_complex("CCO", -8), _complex("", None), _complex("*", float("nan")),
            _complex("bad-smiles", float("inf"))]
    metrics = _aggregate_metrics(rows)
    assert metrics["n_ligands"] == 4
    assert metrics["n_descriptor_valid"] == metrics["mw_n_valid"] == 1
    assert metrics["n_descriptor_invalid"] == 3
    assert metrics["mw_lipinski_rate"] == 0.25
    assert metrics["rotb_in_range_rate"] == 0.25
    assert metrics["mw_metal_adjusted_rate"] == 0
    assert metrics["success_rate"] == 0.25
    assert metrics["success_rate_docked"] == 1.0
    assert metrics["mean_vina"] == -8
    assert metrics["mw_mean"] == pytest.approx(AnticancerMetricSuite().molecular_weight("CCO"))


@pytest.mark.parametrize("rows", [[], [_complex("", float("nan")), _complex("*", None)]])
def test_empty_and_all_invalid_aggregation_has_no_nanmean_warning(rows):
    from molmetal_lam.scripts.pocket2mol_vs_lambda_1h36 import _aggregate_metrics
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = _aggregate_metrics(rows)
    assert metrics["mw_n_valid"] == 0
    assert metrics["anticancer_composite_n_valid"] == 0
    assert math.isnan(metrics["mw_mean"])
    assert math.isnan(metrics["anticancer_composite_mean"])
    assert math.isnan(metrics["mean_vina"])
    if rows:
        assert metrics["mw_lipinski_rate"] == 0
    else:
        assert math.isnan(metrics["mw_lipinski_rate"])
