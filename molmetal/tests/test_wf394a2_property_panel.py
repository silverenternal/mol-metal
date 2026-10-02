"""Wf-394-A2 — diversity + SA-aware property panel runner smoke test.

====================================================================
Why this test exists
====================================================================
Wf-394-A2 author :mod:`molmetal.scripts.wf394a2_property_panel_runner`
which composes:

* Wf-264-SA — SA + QED per molecule (Ertl 2009 + Bickerton 2012)
* Wf-380-A6 — SA-aware sampling penalty (TODO-205)
* Wf-380-A3 / Wf-388-A2 — diversity Tanimoto + scaffold pass

This test verifies the runner produces a well-formed panel JSON on a
synthetic SMILES cohort of 4 hand-picked drug-like molecules.

Wf-394-A2 honest framing (party-twenty-spirit §3.5):
* CPU-only smoke test on 4 SMILES.
* PROJECTED wall budget: <1 s for the panel + <1 s for Tanimoto.
* MEASURED wall: smoke-mode runs in well under 1 s on the test host.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# Add molmetal scripts to sys.path so the runner module imports cleanly
# ``__file__`` layout: <repo>/molmetal/tests/test_wf394a2_property_panel.py
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SCRIPT_DIR = REPO_ROOT / "molmetal" / "scripts"
for p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Pre-configure the upstream flow_matching reference shim so the conftest
# preload (which depends on the path) succeeds before our test imports run.
import importlib.util as _iu_loader
_loader_name = "molmetal.adapters.flow_matching_lipman._reference_loader"
_loader_path = (REPO_ROOT / "molmetal" / "adapters" / "flow_matching_lipman"
                / "_reference_loader.py")
if _loader_name not in sys.modules:
    _spec = _iu_loader.spec_from_file_location(_loader_name, str(_loader_path))
    _mod = _iu_loader.module_from_spec(_spec)
    sys.modules[_loader_name] = _mod
    _spec.loader.exec_module(_mod)
    from pathlib import Path as _P
    _mod._reference_root = (
        REPO_ROOT / "molmetal" / "references" / "flow_matching"
        / "flow_matching"
    )


def _import_runner():
    """Import the runner module under test via spec_from_file_location.

    Uses the file path directly so the dataclass-based ``DiversityCfg`` /
    ``SaCfg`` see a properly-registered module name in ``sys.modules``.
    """
    import importlib.util
    runner_path = SCRIPT_DIR / "wf394a2_property_panel_runner.py"
    mod_name = "wf394a2_property_panel_runner"
    spec = importlib.util.spec_from_file_location(mod_name, str(runner_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # CRITICAL for @dataclass
    spec.loader.exec_module(mod)
    return mod


def test_panel_runner_smoke():
    """Smoke test: build a panel on a synthetic 4-SMILES cohort.

    * SA + QED must be computed for all 4 molecules.
    * SA-aware penalty pass must run with --sa-aware flag.
    * Diversity pass must run with --diversity-aware-eval flag and emit
      valid Tanimoto aggregate stats.
    """
    runner = _import_runner()
    assert runner is not None

    # Synthetic 4-molecule cohort — drug-like SMILES (aspirin, caffeine,
    # paracetamol, ibuprofen).
    cohort = [
        "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(=O)NC1=CC=C(O)C=C1",   # paracetamol
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",   # ibuprofen
    ]

    diversity_cfg = runner.DiversityCfg(
        enabled=True,
        sample_size=10,  # tiny — for smoke
        seed = 42,
    )
    sa_cfg = runner.SaCfg(target_sa=2.8, sa_weight=0.3)

    panel = runner.run_property_panel(
        cohort_smiles=cohort,
        diversity_cfg=diversity_cfg,
        sa_cfg=sa_cfg,
    )

    # Top-level shape
    assert panel["panel_version"] == "wf394a2/v1"
    assert panel["n_input"] == 4

    # SA + QED always present
    sa_qed = panel["sa_qed"]
    assert sa_qed["n_parsed"] == 4
    assert sa_qed["n_failed"] == 0
    assert sa_qed["parsed_ratio"] == 1.0
    assert "sa" in sa_qed["aggregate"]
    assert "qed" in sa_qed["aggregate"]
    sa_agg = sa_qed["aggregate"]["sa"]
    assert sa_agg["mean"] is not None
    assert 1.0 <= sa_agg["mean"] <= 10.0
    qed_agg = sa_qed["aggregate"]["qed"]
    assert qed_agg["mean"] is not None
    assert 0.0 <= qed_agg["mean"] <= 1.0

    # SA-aware pass
    sa_aware = panel["sa_aware"]
    assert sa_aware is not None
    assert sa_aware["n_parsed"] == 4
    sa_aw_agg = sa_aware["aggregate"]
    assert "mean_sa" in sa_aw_agg
    assert "fraction_sa_le_2_8" in sa_aw_agg
    assert "mean_penalty" in sa_aw_agg
    assert sa_aw_agg["mean_sa"] is not None

    # Diversity pass
    diversity = panel["diversity"]
    assert diversity is not None
    assert diversity["n_parsed_mol"] == 4
    assert diversity["n_unique_scaffold"] >= 1
    assert diversity["fraction_unique_scaffold"] > 0.0
    assert diversity["tanimoto"]["n_pairs"] >= 1
    tanimoto_mean = diversity["tanimoto"]["mean"]
    assert tanimoto_mean is not None
    assert 0.0 <= tanimoto_mean <= 1.0

    # JSON round-trip — sanity check
    panel_str = json.dumps(panel, default=lambda o: float(o) if isinstance(o, float) else str(o))
    parsed_back = json.loads(panel_str)
    assert parsed_back["n_input"] == 4
    assert parsed_back["panel_version"] == "wf394a2/v1"


def test_panel_runner_no_diversity_no_sa_aware():
    """Smoke test: SA + QED only when --sa-aware and --diversity-aware-eval are off."""
    runner = _import_runner()
    assert runner is not None

    cohort = ["CCO", "CCN", "CCCl"]  # 3 trivial SMILES
    panel = runner.run_property_panel(
        cohort_smiles=cohort,
        diversity_cfg=runner.DiversityCfg(enabled=False),
        sa_cfg=None,  # SA-aware OFF
    )
    assert panel["n_input"] == 3
    assert panel["sa_qed"]["n_parsed"] == 3
    assert panel["sa_aware"] is None
    assert panel["diversity"] is None


def test_aggregate_panel_metrics_basic():
    """aggregate_panel_metrics works on a list of per-mol results."""
    runner = _import_runner()
    assert runner is not None

    per_mol = [
        {"smiles": "CCO",  "sa": 2.5, "qed": 0.6},
        {"smiles": "CCN",  "sa": 2.8, "qed": 0.5},
        {"smiles": "CCCl", "sa": None, "qed": None},
        {"smiles": "CCBr", "sa": 4.0, "qed": 0.4, "penalty": 0.4},
    ]
    agg = runner.aggregate_panel_metrics(per_mol)
    assert agg["n_input"] == 4
    sa = agg["sa"]
    assert sa["n_parsed"] == 3
    assert sa["n_failed"] == 1
    assert sa["mean"] == pytest.approx((2.5 + 2.8 + 4.0) / 3, rel=1e-4)
    qed = agg["qed"]
    assert qed["n_parsed"] == 3
    assert qed["mean"] == pytest.approx((0.6 + 0.5 + 0.4) / 3, rel=1e-4)
    assert "penalty" in agg
    assert agg["penalty"]["mean"] == pytest.approx(0.4, rel=1e-4)


def test_cli_smoke():
    """CLI smoke: --smiles-literal + --sa-aware + --diversity-aware-eval."""
    runner = _import_runner()
    assert runner is not None

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        out_path = fh.name
    try:
        rc = runner.main([
            "--smiles-literal", "CCO,CCN,CCO",
            "--sa-aware",
            "--diversity-aware-eval",
            "--diversity-sample-size", "5",
            "--out-json", out_path,
        ])
        assert rc == 0
        with open(out_path) as fh:
            data = json.load(fh)
        assert data["panel_version"] == "wf394a2/v1"
        assert data["n_input"] == 3
        assert data["sa_qed"]["n_parsed"] == 3
        assert data["sa_aware"] is not None
        assert data["diversity"] is not None
    finally:
        Path(out_path).unlink(missing_ok=True)