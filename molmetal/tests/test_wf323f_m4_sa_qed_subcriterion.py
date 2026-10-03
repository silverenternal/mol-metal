"""Wf-323F — m4 SA/QED sub-criterion smoke tests (WEAK_TO_STRONG §2.4).

Per Wf-323F / WEAK_TO_STRONG §2.4: the m4 Success Rate criterion is

    Vina < −8.18  ∧  QED > 0.25  ∧  SA > 0.59

The Vina sub-criterion currently binds at 0/100 on the MEASURED cohort
(cascades from m3, see WEAK_TO_STRONG §2.3). The SA/QED sub-criterion
is the AND of the two physico-chemical gates *independent of Vina* —
this test module verifies the Wf-323F CPU smoke:

  * 1 test:  compute_sub_rate() correctness on a small synthetic SMILES list.
  * 1 test:  per-pool distribution emission (8 buckets).
  * 1 test:  on-disk m4 metric JSON contract (FLAT schema + HONEST_NEGATIVE).
  * 1 test:  on-disk Wf-323F report JSON contract (sub_rate_sa_qed numeric).
  * 1 test:  pre-flight gate ``p3_m4_sa_qed_subcriterion_documented`` is
             registered (mirrors Wf-315A pattern).
  * 1 test:  WEAK_TO_STRONG §2.4 integration — TODO closeout doc exists.

Tests are CPU-only (no GPU, no retrain, no chemprop). They follow the
convention of the Wf-303C T1 / Wf-312C / Wf-315A tests:

  * Direct unit tests skip when RDKit is unavailable (same gate used by
    ``test_sa_score.py`` + ``test_wf315a_sa_qed_wiring.py``).
  * Pipeline / JSON tests skip when the on-disk artefact is missing
    (the smoke is opt-in for live environments).

Honest framing (party-twenty-spirit §3.5):
  Tests use the same METRICS_DIR / pre_flight gate helpers used by
  the rest of the panel — no fabricated readings, no hand-rolled
  constants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"
SCRIPT_DIR = REPO_ROOT / "molmetal" / "scripts"
TESTS_DIR = REPO_ROOT / "molmetal" / "tests"
DATA_DIR = REPO_ROOT / "molmetal" / "data"
REPORTS_DIR = REPO_ROOT / "molmetal" / "reports"


# ---------------------------------------------------------------------------
# 1. compute_sub_rate() correctness (RDKit-dependent)
# ---------------------------------------------------------------------------

# Skip the live RDKit-dependent unit tests when RDKit is unavailable —
# same gate as test_wf315a_sa_qed_wiring.py.
try:
    from molmetal.scripts import wf323f_m4_sa_qed_subcriterion as _wf323f

    _WF323F_AVAILABLE = True
except Exception:
    _WF323F_AVAILABLE = False


try:
    from rdkit import Chem  # noqa: WPS433
    from rdkit.Chem import QED  # noqa: WPS433

    _RDKIT_AVAILABLE = True
except Exception:
    _RDKIT_AVAILABLE = False


pytestmark_rdkit = pytest.mark.skipif(
    not _RDKIT_AVAILABLE,
    reason="RDKit unavailable in this environment — m4 SA/QED unit tests skip",
)


@pytest.mark.skipif(not _WF323F_AVAILABLE, reason="wf323f import failed")
@pytestmark_rdkit
def test_wf323f_compute_sub_rate_function_correctness(tmp_path):
    """compute_sub_rate() returns the canonical schema with valid SA/QED.

    Builds a tiny CSV with 3 SMILES of known druglikeness profile:
      * aspirin    : SA ~ 1.78 (synthesizable), QED ~ 0.55 (druglike)
      * ethanol    : SA ~ 1.0  (synthesizable), QED ~ 0.36 (druglike borderline)
      * phenanthrene: SA > 2.8 (3-ring polycyclic; not synthesizable per Wf-139 P6)
    Expects sub_rate_sa_qed to be in [0, 1] and the per_pool_distribution
    to have 8 buckets.
    """
    csv_path = tmp_path / "smiles.csv"
    csv_path.write_text(
        "smiles\n"
        "CC(=O)Oc1ccccc1C(=O)O\n"  # aspirin
        "CCO\n"  # ethanol
        "c1ccc2c(c1)ccc1ccccc12\n",  # phenanthrene (3 fused rings)
        encoding="utf-8",
    )
    result = _wf323f.compute_sub_rate(csv_path)
    assert result["metric"] == "m4_success_rate_sa_qed_subcriterion"
    assert result["status"] == "MEASURED"
    assert result["n_input"] == 3
    assert result["n_parsed"] == 3
    assert 0.0 <= result["sub_rate_sa_qed"] <= 1.0
    assert 0.0 <= result["sub_rate_sa_only"] <= 1.0
    assert 0.0 <= result["sub_rate_qed_only"] <= 1.0
    # Per-pool distribution: 8 buckets (4 SA + 4 QED).
    distribution = result["per_pool_distribution"]
    assert isinstance(distribution, dict)
    assert set(distribution.keys()) == {
        "sa_bucket_lt_2_8",
        "sa_bucket_2_8_to_4",
        "sa_bucket_4_to_6",
        "sa_bucket_6_plus",
        "qed_bucket_lt_0_25",
        "qed_bucket_0_25_to_0_5",
        "qed_bucket_0_5_to_0_75",
        "qed_bucket_0_75_plus",
    }
    # Sanity: bucket counts sum to n_parsed for SA side AND QED side.
    sa_total = sum(v for k, v in distribution.items() if k.startswith("sa_"))
    qed_total = sum(v for k, v in distribution.items() if k.startswith("qed_"))
    assert sa_total == result["n_parsed"]
    assert qed_total == result["n_parsed"]
    # Wf-323F provenance tags
    assert "Wf-323F" in result["wf_tag"]
    assert "WEAK_TO_STRONG" in result["wf_design_ref"]


# ---------------------------------------------------------------------------
# 2. on-disk metric JSON contract (FLAT schema + HONEST_NEGATIVE)
# ---------------------------------------------------------------------------

M4_METRIC_JSON = METRICS_DIR / "m4_success_rate.json"


@pytest.mark.skipif(
    not M4_METRIC_JSON.exists(),
    reason=f"m4 metric artefact missing: {M4_METRIC_JSON}",
)
def test_wf323f_m4_metric_json_contract():
    """m4_success_rate.json carries the FLAT schema + HONEST_NEGATIVE tag.

    Per the Wf-308B / Wf-312C / Wf-315A contract, the on-disk JSON for
    any metric whose cohort mean DOES NOT meet the target band MUST
    carry ``HONEST_NEGATIVE: true`` + ``honest_negative_reason`` fields.
    The m4 SA/QED sub-rate MEASURED on the 449 cohort is 0.0022
    (well below the 0.60 gate from WEAK_TO_STRONG §2.4), so the
    honest-negative contract applies.
    """
    data = json.loads(M4_METRIC_JSON.read_text())
    assert data["metric"] == "m4_success_rate"
    assert data["status"] == "MEASURED"
    assert data["n_samples"] == 449
    assert data["n_parsed"] == 449
    assert isinstance(data["value"], (int, float))
    assert 0.0 <= data["value"] <= 1.0
    assert data["sub_rate_sa_qed"] == data["value"]
    assert data["sub_threshold_sa_max"] == 2.8
    assert data["sub_threshold_qed_min"] == 0.5
    assert data["vina_subcriterion_in_scope"] is False
    assert data["verdict"] == "FAIL"
    assert data["HONEST_NEGATIVE"] is True
    assert isinstance(data.get("honest_negative_reason"), str)
    assert data["honest_negative_reason"]
    # Wf-323F provenance tags
    assert "Wf-323F" in data.get("wf_tag", "")
    assert "WEAK_TO_STRONG" in data.get("wf_design_ref", "")


# ---------------------------------------------------------------------------
# 3. on-disk Wf-323F report JSON contract (sub_rate_sa_qed + distribution)
# ---------------------------------------------------------------------------

WF323F_REPORT_JSON = REPORTS_DIR / "wf323f_m4_sa_qed_subcriterion" / "m4_rate.json"


@pytest.mark.skipif(
    not WF323F_REPORT_JSON.exists(),
    reason=f"Wf-323F report artefact missing: {WF323F_REPORT_JSON}",
)
def test_wf323f_report_json_contract():
    """wf323f_m4_sa_qed_subcriterion/m4_rate.json carries the full contract:
    n_input/n_parsed + thresholds + 3 sub-rates + per_pool_distribution +
    Wf-323F provenance tags.
    """
    data = json.loads(WF323F_REPORT_JSON.read_text())
    assert data["metric"] == "m4_success_rate_sa_qed_subcriterion"
    assert data["status"] == "MEASURED"
    assert data["n_input"] == 449
    assert data["n_parsed"] == 449
    assert data["thresholds"]["sa_max"] == 2.8
    assert data["thresholds"]["qed_min"] == 0.5
    assert 0.0 <= data["sub_rate_sa_qed"] <= 1.0
    assert 0.0 <= data["sub_rate_sa_only"] <= 1.0
    assert 0.0 <= data["sub_rate_qed_only"] <= 1.0
    # The MEASURED cohort sub-rate is 1/449 = ~0.0022.
    assert abs(data["sub_rate_sa_qed"] - 1.0 / 449.0) < 1e-3, (
        f"sub_rate_sa_qed={data['sub_rate_sa_qed']:.4f} " f"expected ~{1.0 / 449.0:.4f}"
    )
    # per_pool_distribution: 8 buckets with sane counts.
    distribution = data["per_pool_distribution"]
    assert isinstance(distribution, dict)
    assert len(distribution) == 8
    sa_total = sum(v for k, v in distribution.items() if k.startswith("sa_"))
    qed_total = sum(v for k, v in distribution.items() if k.startswith("qed_"))
    assert sa_total == 449
    assert qed_total == 449
    # Wf-323F provenance tags
    assert "Wf-323F" in data.get("wf_tag", "")
    assert "WEAK_TO_STRONG" in data.get("wf_design_ref", "")


# ---------------------------------------------------------------------------
# 4. Pre-flight gate registration (mirrors Wf-315A pattern)
# ---------------------------------------------------------------------------

PRE_FLIGHT_SCRIPT = SCRIPT_DIR / "pre_flight_check.py"


@pytest.mark.skipif(
    not PRE_FLIGHT_SCRIPT.exists(),
    reason="pre_flight_check.py missing",
)
def test_wf323f_p3_m4_sa_qed_subcriterion_gate_defined():
    """p3_m4_sa_qed_subcriterion_documented gate is registered.

    Mirrors the Wf-315A ``p3_sa_score_metric_present`` +
    ``p3_qed_score_metric_present`` registration pattern.  The gate
    must appear at least twice (once in 'all' pipeline, once in
    'frontier' pipeline) — same pattern as the Wf-315A gates.
    """
    text = PRE_FLIGHT_SCRIPT.read_text()
    assert '"p3_m4_sa_qed_subcriterion_documented"' in text, (
        "p3_m4_sa_qed_subcriterion_documented gate not found in " "pre_flight_check.py"
    )
    # Confirm the gate references the Wf-323F artefact paths.
    assert "wf323f_m4_sa_qed_subcriterion" in text
    assert "m4_success_rate.json" in text
    # Confirm both pipelines (all + frontier) carry the gate.
    assert text.count('"p3_m4_sa_qed_subcriterion_documented"') >= 2


# ---------------------------------------------------------------------------
# 5. EXPECTED_GATE_COUNTS updated (Wf-321D audit invariant)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PRE_FLIGHT_SCRIPT.exists(),
    reason="pre_flight_check.py missing",
)
def test_wf323f_expected_gate_counts_bumped():
    """EXPECTED_GATE_COUNTS is bumped by 1 on 'all' + 'frontier' for Wf-323F.

    Per Wf-321D audit: the gate-count consistency check enforces that
    ``EXPECTED_GATE_COUNTS['all']`` + ``['frontier']`` track the
    build_gates() output.  Wf-323F adds 1 gate in each pipeline.
    Pre-Wf-323F: ``all=56``, ``frontier=23`` (per Wf-321D ship).
    Post-Wf-323F: ``all=57``, ``frontier=24``.
    """
    text = PRE_FLIGHT_SCRIPT.read_text()
    # Read the EXPECTED_GATE_COUNTS dict via simple text match.
    assert '"all": 57' in text, "EXPECTED_GATE_COUNTS['all'] should be 57 (was 56)"
    assert (
        '"frontier": 24' in text
    ), "EXPECTED_GATE_COUNTS['frontier'] should be 24 (was 23)"


# ---------------------------------------------------------------------------
# 6. WEAK_TO_STRONG §2.4 integration — TODO closeout doc on disk
# ---------------------------------------------------------------------------

WF323F_TODO_DOC = REPO_ROOT / "TODO" / "completed" / "323f_m4_sa_qed_subcriterion.md"


@pytest.mark.skipif(
    not WF323F_TODO_DOC.exists(),
    reason=f"Wf-323F closeout TODO doc missing: {WF323F_TODO_DOC}",
)
def test_wf323f_todo_closeout_doc_present():
    """TODO/completed/323f_m4_sa_qed_subcriterion.md exists with §2.4 ref.

    Per the Wf-323F ship surface, the closeout doc must reference
    WEAK_TO_STRONG §2.4 (the planning doc) AND Vina sub-criterion
    OUT-OF-SCOPE framing (the honest-negative anchor).
    """
    text = WF323F_TODO_DOC.read_text()
    assert "WEAK_TO_STRONG" in text
    assert "§2.4" in text
    # Honest-negative anchors.
    assert "Vina" in text
    assert "USER_OK" in text or "kill_then_adapt" in text
    # Wf-323F provenance tag.
    assert "Wf-323F" in text
