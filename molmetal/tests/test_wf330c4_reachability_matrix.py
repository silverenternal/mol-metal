"""Tests for the Wf-330-C4 per-domain reachability matrix.

================================================================
Wf-330-C4: Per-domain reachability matrix pinning test
(modelled on `test_closure_reachability_r20.py`).

This file pins the **MEASURED** per-domain reachability anchors for the
closure-theorem ship surface, as catalogued in
``TODO/completed/83j_lambda_closeout.md`` §3.1 (L1/L6/L12 MEASURED) and
the Wf-57 Tanimoto<0.4 OOD protocol.

MEASURED anchors (from on-disk JSON reports):

  * **L1** beta-NF enumeration reachability_rate = **0.5108**
    — `molmetal/reports/wf36_closure_theorem/closure_bench.json`
    — 836 samples × 9 metals × tmQM d-block subset (2026-09-26)
  * **L6** d-block reachability_rate = **0.5891**
    — `molmetal/reports/wf37_closure_dblock/per_metal_bench.json`
    — 2244 samples × 30 d-block metals (2026-09-26)
  * **L12 PlatinAI 5K** OOD reachability_rate = **0.5581**
    — `molmetal/reports/wf41_closure_ood/ood_closure.json`
    — 86 OOD-filtered samples × PlatinAI 5K (2026-09-26)
  * **L12 MetalCytoTox 5-metal** OOD reachability_rate = **0.0300**
    — `molmetal/reports/wf41_closure_ood/ood_closure.json`
    — 100 OOD-filtered samples × MetalCytoTox 5-metal (2026-09-26)
    — gap to PROJECTED ≥ 0.85 → **0.8200** (CPU-only may not close)
  * **Tanimoto<0.4** fraction_below_threshold = **0.9978**
    — `molmetal/reports/wf57_tanimoto/ood_split.json`
    — 32256 (train, test) cross-pairs × 30 d-block metals (2026-09-26)

These 5 anchors are the canonical LAMBDA_ROADMAP §3.2 reference values;
downstream agents MUST cite this file (or the source JSONs) when claiming
any L1 / L6 / L12 / MetalCytoTox / Tanimoto<0.4 number — per
``party-twenty-spirit`` §3.5 zero-tolerance discipline.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"

# MEASURED anchors — pin the per-domain reachability matrix.
# Projected / aspirational numbers MUST NOT appear in this file.
L1_REACHABILITY_MEASURED = 0.5108  # Wf-36 closure theorem verifier (836 × 9 metals)
L6_REACHABILITY_MEASURED = 0.5891  # Wf-37 closure d-block (2244 × 30 metals)
L12_PLATINAI_REACHABILITY_MEASURED = 0.5581  # Wf-41 P2 platinai_5k OOD
L12_METALCYTO_REACHABILITY_MEASURED = 0.0300  # Wf-41 P2 metalcytotox_5metal OOD
TANIMOTO_BELOW_04_MEASURED = 0.9978  # Wf-57 Tanimoto<0.4 protocol

# PROJECTED targets (paper §4 aspirational; honest gap = target - measured).
PROJECTED_REACHABILITY_GATE = 0.85  # R20-P5 closure gate target
GAP_METALCYTO_TO_PROJECTED = (
    PROJECTED_REACHABILITY_GATE - L12_METALCYTO_REACHABILITY_MEASURED
)

# Tolerance for JSON-file float comparison (5 dp).
RTOL = 1e-3


def _load_report_json(rel_path: str) -> Dict[str, Any]:
    """Load a JSON report under ``molmetal/reports/<rel_path>``.

    Returns an empty dict if the report is missing so tests can
    skip rather than fail when an upstream bench has not shipped yet.
    """
    path = REPORTS_DIR / rel_path
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Test 1: L1 (beta-NF enumeration) anchor — Wf-36 closure theorem verifier
# ---------------------------------------------------------------------------


def test_l1_beta_nf_enumeration_anchor() -> None:
    """L1 reachability_rate = 0.5108 MEASURED on tmQM d-block (Wf-36 P1).

    Source: ``molmetal/reports/wf36_closure_theorem/closure_bench.json``.
    Verifies the on-disk JSON has the same value as the LAMBDA_ROADMAP §3.2
    anchor.
    """
    bench = _load_report_json("wf36_closure_theorem/closure_bench.json")
    if not bench:
        pytest.skip("Wf-36 closure bench JSON missing on disk")
    aggregate = bench.get("aggregate", {})
    rate = aggregate.get("closure_reachability_rate")
    assert rate is not None, "closure_reachability_rate missing from Wf-36 bench"
    assert math.isclose(
        rate, L1_REACHABILITY_MEASURED, rel_tol=RTOL
    ), f"L1 reachability drifted: got {rate}, expected {L1_REACHABILITY_MEASURED}"
    # Sample-count sanity (836 samples × 9 metals).
    assert aggregate.get("n_samples") == 836
    assert aggregate.get("n_closure_ok") == 427


# ---------------------------------------------------------------------------
# Test 2: L6 (d-block reachability) anchor — Wf-37 closure d-block 30 metals
# ---------------------------------------------------------------------------


def test_l6_dblock_30_metals_anchor() -> None:
    """L6 reachability_rate = 0.5891 MEASURED on 30 d-block metals (Wf-37 P2).

    Source: ``molmetal/reports/wf37_closure_dblock/per_metal_bench.json``.
    Verifies the 30-metal d-block sweep shipped and the rate matches
    the LAMBDA_ROADMAP §3.2 anchor.
    """
    bench = _load_report_json("wf37_closure_dblock/per_metal_bench.json")
    if not bench:
        pytest.skip("Wf-37 closure d-block JSON missing on disk")
    aggregate = bench.get("aggregate", {})
    rate = aggregate.get("closure_reachability_rate")
    assert rate is not None, "closure_reachability_rate missing from Wf-37 bench"
    assert math.isclose(
        rate, L6_REACHABILITY_MEASURED, rel_tol=RTOL
    ), f"L6 reachability drifted: got {rate}, expected {L6_REACHABILITY_MEASURED}"
    # Sample-count sanity (2244 samples × 30 metals).
    assert aggregate.get("n_samples") == 2244
    assert aggregate.get("n_closure_ok") == 1322


# ---------------------------------------------------------------------------
# Test 3: L12 PlatinAI OOD anchor — Wf-41 closure OOD cross-domain
# ---------------------------------------------------------------------------


def test_l12_platinai_ood_anchor() -> None:
    """L12 platinai_5k reachability_rate = 0.5581 MEASURED (Wf-41 P2).

    Source: ``molmetal/reports/wf41_closure_ood/ood_closure.json``.
    Verifies the PlatinAI 5K OOD-filtered reachability matches the
    LAMBDA_ROADMAP §3.2 anchor.
    """
    bench = _load_report_json("wf41_closure_ood/ood_closure.json")
    if not bench:
        pytest.skip("Wf-41 closure OOD JSON missing on disk")
    per_domain = bench.get("per_domain", {})
    plat = per_domain.get("platinai_5k", {})
    if not plat:
        pytest.skip("platinai_5k domain missing from Wf-41 OOD bench")
    rate = plat.get("aggregate", {}).get("closure_reachability_rate")
    assert rate is not None, "platinai reachability_rate missing"
    assert math.isclose(rate, L12_PLATINAI_REACHABILITY_MEASURED, rel_tol=RTOL), (
        f"L12 platinai reachability drifted: got {rate}, "
        f"expected {L12_PLATINAI_REACHABILITY_MEASURED}"
    )


# ---------------------------------------------------------------------------
# Test 4: L12 MetalCytoTox OOD anchor — biggest gap to PROJECTED ≥ 0.85
# ---------------------------------------------------------------------------


def test_l12_metalcyto_ood_anchor() -> None:
    """L12 metalcytotox_5metal reachability_rate = 0.0300 MEASURED (Wf-41 P2).

    Source: ``molmetal/reports/wf41_closure_ood/ood_closure.json``.
    Verifies the MetalCytoTox 5-metal OOD reachability matches the
    LAMBDA_ROADMAP §3.2 anchor and computes the honest gap to the
    PROJECTED ≥ 0.85 R20-P5 gate target.
    """
    bench = _load_report_json("wf41_closure_ood/ood_closure.json")
    if not bench:
        pytest.skip("Wf-41 closure OOD JSON missing on disk")
    per_domain = bench.get("per_domain", {})
    metalcyto = per_domain.get("metalcytotox_5metal", {})
    if not metalcyto:
        pytest.skip("metalcytotox_5metal domain missing from Wf-41 OOD bench")
    rate = metalcyto.get("aggregate", {}).get("closure_reachability_rate")
    assert rate is not None, "metalcytotox reachability_rate missing"
    assert math.isclose(rate, L12_METALCYTO_REACHABILITY_MEASURED, rel_tol=RTOL), (
        f"L12 metalcytotox reachability drifted: got {rate}, "
        f"expected {L12_METALCYTO_REACHABILITY_MEASURED}"
    )

    # Honest gap to PROJECTED gate (party-twenty-spirit §3.5).
    # This is the biggest gap in the L1/L6/L12 matrix; R20-P5 GPU retrain
    # is reserved for the MetalCytoTox 0.0300 → 0.85 lift.
    gap = PROJECTED_REACHABILITY_GATE - rate
    assert math.isclose(
        gap, GAP_METALCYTO_TO_PROJECTED, rel_tol=RTOL
    ), f"MetalCytoTox gap drifted: got {gap}, expected {GAP_METALCYTO_TO_PROJECTED}"
    # Sanity: gap must be the largest in the matrix (party-twenty-spirit §3.5).
    gaps = {
        "L1": PROJECTED_REACHABILITY_GATE - L1_REACHABILITY_MEASURED,
        "L6": PROJECTED_REACHABILITY_GATE - L6_REACHABILITY_MEASURED,
        "L12_platinai": PROJECTED_REACHABILITY_GATE
        - L12_PLATINAI_REACHABILITY_MEASURED,
        "L12_metalcyto": PROJECTED_REACHABILITY_GATE - rate,
    }
    assert gaps["L12_metalcyto"] == max(
        gaps.values()
    ), f"MetalCytoTox should have the largest gap; got gaps={gaps}"


# ---------------------------------------------------------------------------
# Test 5: Tanimoto<0.4 anchor — Wf-57 OOD protocol verification
# ---------------------------------------------------------------------------


def test_tanimoto_below_04_anchor() -> None:
    """Tanimoto<0.4 fraction_below_threshold = 0.9978 MEASURED (Wf-57 P1).

    Source: ``molmetal/reports/wf57_tanimoto/ood_split.json``.
    Verifies the OOD protocol verification target (>= 0.80) PASSES on
    the tmQM d-block subset.
    """
    bench = _load_report_json("wf57_tanimoto/ood_split.json")
    if not bench:
        pytest.skip("Wf-57 Tanimoto ood_split JSON missing on disk")
    # Top-level OR nested under 'aggregate'.
    rate = bench.get("fraction_below_threshold")
    if rate is None:
        rate = bench.get("aggregate", {}).get("fraction_below_threshold")
    assert rate is not None, "fraction_below_threshold missing"
    assert math.isclose(rate, TANIMOTO_BELOW_04_MEASURED, rel_tol=RTOL), (
        f"Tanimoto<0.4 fraction drifted: got {rate}, "
        f"expected {TANIMOTO_BELOW_04_MEASURED}"
    )
    # Verification gate (>= 0.80) must PASS.
    assert (
        rate >= 0.80
    ), f"Tanimoto<0.4 protocol verification FAILED: got {rate}, gate >= 0.80"


# ---------------------------------------------------------------------------
# Test 6 (extra): per-domain matrix anchor constants are coherent
# ---------------------------------------------------------------------------


def test_per_domain_matrix_constants_are_coherent() -> None:
    """Sanity: per-domain matrix constants are internally consistent.

    - L1 < L6 < 1.0 (closure rate grows with metal-coverage breadth)
    - L12_metalcyto << all others (the honest-negative big gap)
    - Tanimoto<0.4 > 0.95 (OOD protocol verification PASSES)
    """
    assert 0.0 < L1_REACHABILITY_MEASURED < L6_REACHABILITY_MEASURED < 1.0
    assert L1_REACHABILITY_MEASURED < L12_PLATINAI_REACHABILITY_MEASURED
    assert L12_PLATINAI_REACHABILITY_MEASURED > L12_METALCYTO_REACHABILITY_MEASURED
    assert TANIMOTO_BELOW_04_MEASURED > 0.95

    # The matrix values are the canonical LAMBDA_ROADMAP §3.2 anchors.
    matrix = {
        "L1_beta_NF_enumeration": L1_REACHABILITY_MEASURED,
        "L6_dblock_reachability": L6_REACHABILITY_MEASURED,
        "L12_PlatinAI_OOD": L12_PLATINAI_REACHABILITY_MEASURED,
        "L12_MetalCytoTox_OOD": L12_METALCYTO_REACHABILITY_MEASURED,
        "Tanimoto_lt_0.4_fraction": TANIMOTO_BELOW_04_MEASURED,
    }
    # Every value must be a float in [0, 1].
    for label, val in matrix.items():
        assert isinstance(val, float), f"{label}: not a float"
        assert 0.0 <= val <= 1.0, f"{label}: out of [0, 1] range: {val}"
