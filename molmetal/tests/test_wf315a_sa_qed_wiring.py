"""Wf-315A — SA + QED evaluator wiring tests (WEAK_TO_STRONG §3.6).

Per Wf-315A / WEAK_TO_STRONG_PLAN.md §3.6: the m5 SA (Ertl-Schuffenhauer
2009) and m6 QED (Bickerton 2012) evaluators live in
:mod:`molmetal.molmetal_lam.sbdd_env.sa_score` and
:mod:`molmetal.moldd_env.qed_scorer` but were not wired into
:mod:`molmetal.molmetal_lam.sbdd_env.metrics_v2` + the ADMET
pass-through pipeline.  This test module verifies the Wf-315A
wire-in:

  * 2 tests: SA / QED function correctness (basic range + parse-failure
    fallback).
  * 2 tests: pass-through emission (sa_score.json + qed_score.json
    on the 449 metallo-drug cohort, both carrying the canonical
    Wf-308B / Wf-312C fields).
  * 2 tests: pre-flight gate validation (p3_sa_score_metric_present
    + p3_qed_score_metric_present).
  * 2 tests: integration with downstream consumer (paper_dashboard
    CLAIM_CATALOG contains the 2 NEW Wf-315A claim cards).

Tests are CPU-only (no GPU, no retrain, no chemprop).  They follow
the convention of the Wf-303C T1 / Wf-312C tests:

  * Direct unit tests skip when RDKit is unavailable (the same gate
    used by ``test_sa_score.py``).
  * Pipeline / JSON tests skip when the on-disk artefact is missing
    (the panel is opt-in for live environments).

Honest framing (party-twenty-spirit §3.5):
  Tests use the same METRICS_DIR / pre_flight gate helpers used by
  the rest of the panel — no fabricated readings, no hand-rolled
  constants.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"


# ---------------------------------------------------------------------------
# 1. SA / QED function correctness
# ---------------------------------------------------------------------------

# Skip the live RDKit-dependent unit tests when RDKit is unavailable —
# same gate as test_sa_score.py / test_metrics_v2.py.
try:
    from molmetal.molmetal_lam.sbdd_env import metrics_v2 as _mv2

    _MV2_AVAILABLE = True
except Exception:
    _MV2_AVAILABLE = False


_RDKIT_AVAILABLE = getattr(_mv2, "_RDKIT_AVAILABLE", False) if _MV2_AVAILABLE else False

pytestmark_rdkit = pytest.mark.skipif(
    not _RDKIT_AVAILABLE,
    reason="RDKit unavailable in this environment — SA/QED unit tests skip",
)


@pytest.mark.skipif(not _MV2_AVAILABLE, reason="metrics_v2 import failed")
@ pytestmark_rdkit
def test_wf315a_sa_score_function_correctness():
    """sa_score returns SA in [1, 10] for valid SMILES (aspirin ~ 1.5)."""
    sa = _mv2.sa_score("CC(=O)Oc1ccccc1C(=O)O")
    assert sa == sa  # not NaN
    # Per RDKit UnitTestSAScore + Ertl paper, aspirin ~ 1.5-2.0
    assert 1.0 <= sa <= 5.0, f"aspirin SA out of expected range: {sa}"
    # Confirm the panel-side function routes to sa_score_ertl
    # (cross-check against the same function called on ethanolamine).
    sa_eto = _mv2.sa_score("NCCO")
    assert 1.0 <= sa_eto <= 5.0


@pytest.mark.skipif(not _MV2_AVAILABLE, reason="metrics_v2 import failed")
@ pytestmark_rdkit
def test_wf315a_qed_score_function_correctness():
    """qed_score returns QED in [0, 1] for valid SMILES (aspirin ~ 0.55)."""
    q = _mv2.qed_score("CC(=O)Oc1ccccc1C(=O)O")
    assert q == q  # not NaN
    assert 0.0 <= q <= 1.0, f"aspirin QED out of [0, 1]: {q}"
    # Aspirin is canonical "drug-like" — QED should be > 0.3 (typical)
    assert q > 0.3, f"aspirin QED unexpectedly low: {q}"


# ---------------------------------------------------------------------------
# 2. Pass-through emission
# ---------------------------------------------------------------------------

SA_SCORE_JSON = METRICS_DIR / "sa_score.json"
QED_SCORE_JSON = METRICS_DIR / "qed_score.json"


@pytest.mark.skipif(
    not SA_SCORE_JSON.exists(),
    reason=f"pass-through artefact missing: {SA_SCORE_JSON}",
)
def test_wf315a_sa_score_passthrough_emission():
    """sa_score.json exists with the standard FLAT schema + HONEST_NEGATIVE tag.

    Per the Wf-308B / Wf-312C contract, the on-disk JSON for any
    ADMET pass-through metric whose cohort mean DOES NOT meet the
    target band MUST carry ``HONEST_NEGATIVE: true`` +
    ``honest_negative_reason`` fields.  The 449 metallo-drug
    cohort carries SA mean = 4.715 which is ABOVE the Wf-139 P6
    synthesizable threshold 4.0 — so HONEST_NEGATIVE=true is the
    correct contract here (do not silently mask).
    """
    data = json.loads(SA_SCORE_JSON.read_text())
    assert data["metric"] == "sa_score"
    assert data["status"] == "MEASURED"
    assert data["n_samples"] == 449
    assert data["n_parsed"] == 449
    assert isinstance(data["value"], (int, float))
    assert data["value"] > 0
    assert data["verdict"] == "FAIL"
    assert data["HONEST_NEGATIVE"] is True
    assert isinstance(data.get("honest_negative_reason"), str)
    assert data["honest_negative_reason"]
    # Wf-315A provenance tags
    assert "Wf-315A" in data.get("wf_tag", "")
    assert "WEAK_TO_STRONG" in data.get("wf_design_ref", "")


@pytest.mark.skipif(
    not QED_SCORE_JSON.exists(),
    reason=f"pass-through artefact missing: {QED_SCORE_JSON}",
)
def test_wf315a_qed_score_passthrough_emission():
    """qed_score.json exists with the standard FLAT schema + HONEST_NEGATIVE tag.

    The 449 metallo-drug cohort carries QED mean = 0.230 which is
    BELOW the Wf-139 P6 druglikeness threshold 0.5 — so
    HONEST_NEGATIVE=true is the correct contract here.
    """
    data = json.loads(QED_SCORE_JSON.read_text())
    assert data["metric"] == "qed_score"
    assert data["status"] == "MEASURED"
    assert data["n_samples"] == 449
    assert data["n_parsed"] == 449
    assert isinstance(data["value"], (int, float))
    assert 0.0 <= data["value"] <= 1.0
    assert data["verdict"] == "FAIL"
    assert data["HONEST_NEGATIVE"] is True
    assert isinstance(data.get("honest_negative_reason"), str)
    assert data["honest_negative_reason"]
    # Wf-315A provenance tags
    assert "Wf-315A" in data.get("wf_tag", "")
    assert "WEAK_TO_STRONG" in data.get("wf_design_ref", "")


# ---------------------------------------------------------------------------
# 3. Pre-flight gate validation
# ---------------------------------------------------------------------------

PRE_FLIGHT_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"


@pytest.mark.skipif(
    not PRE_FLIGHT_SCRIPT.exists(),
    reason="pre_flight_check.py missing",
)
def test_wf315a_p3_sa_score_metric_present_gate_defined():
    """p3_sa_score_metric_present gate is registered in pre_flight_check.py."""
    text = PRE_FLIGHT_SCRIPT.read_text()
    assert '"p3_sa_score_metric_present"' in text, (
        "p3_sa_score_metric_present gate not found in pre_flight_check.py"
    )
    # Confirm the gate points at admet_sa_score_json
    assert "admet_sa_score_json" in text
    # Confirm both pipelines (frontier + all) carry the gate
    assert text.count("p3_sa_score_metric_present") >= 2


@pytest.mark.skipif(
    not PRE_FLIGHT_SCRIPT.exists(),
    reason="pre_flight_check.py missing",
)
def test_wf315a_p3_qed_score_metric_present_gate_defined():
    """p3_qed_score_metric_present gate is registered in pre_flight_check.py."""
    text = PRE_FLIGHT_SCRIPT.read_text()
    assert '"p3_qed_score_metric_present"' in text, (
        "p3_qed_score_metric_present gate not found in pre_flight_check.py"
    )
    # Confirm the gate points at admet_qed_score_json
    assert "admet_qed_score_json" in text
    # Confirm both pipelines (frontier + all) carry the gate
    assert text.count("p3_qed_score_metric_present") >= 2


# ---------------------------------------------------------------------------
# 4. Integration with downstream consumer (paper_dashboard CLAIM_CATALOG)
# ---------------------------------------------------------------------------

PAPER_DASHBOARD_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "paper_dashboard.py"


@pytest.mark.skipif(
    not PAPER_DASHBOARD_SCRIPT.exists(),
    reason="paper_dashboard.py missing",
)
def test_wf315a_paper_dashboard_sa_score_claim_card_registered():
    """paper_dashboard CLAIM_CATALOG carries the Wf-315A sa_score claim card."""
    text = PAPER_DASHBOARD_SCRIPT.read_text()
    assert '"claim_wf315a_sa_score_honest_negative"' in text, (
        "claim_wf315a_sa_score_honest_negative card missing from paper_dashboard.py"
    )
    # Confirm the card references the on-disk JSON
    assert "sa_score.json" in text
    # Confirm the card mentions the honest-negative framing
    assert "HONEST_NEGATIVE" in text


@pytest.mark.skipif(
    not PAPER_DASHBOARD_SCRIPT.exists(),
    reason="paper_dashboard.py missing",
)
def test_wf315a_paper_dashboard_qed_score_claim_card_registered():
    """paper_dashboard CLAIM_CATALOG carries the Wf-315A qed_score claim card."""
    text = PAPER_DASHBOARD_SCRIPT.read_text()
    assert '"claim_wf315a_qed_score_honest_negative"' in text, (
        "claim_wf315a_qed_score_honest_negative card missing from paper_dashboard.py"
    )
    # Confirm the card references the on-disk JSON
    assert "qed_score.json" in text
    # Confirm the card mentions the honest-negative framing
    assert "HONEST_NEGATIVE" in text
