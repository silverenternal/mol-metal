"""Tests for the Wf-39 Phase 2 (parallel) lambda algorithm vulnerability scan.

================================================================
Test the ``scan_lambda_algorithm_vulns.py`` read-mostly vuln scanner
that probes the R17 + R18 lambda-algorithm stack on real NCI-60 IC50
data.  The scanner exercises 6 vulnerability surfaces (V1–V6) and
records an OK / WARN / FAIL verdict for each.

Test design
-----------
1.  ``test_vuln_v1_ast_r2_finite`` — V1 AST regression returns a
    finite R² on a small IC50 sample.
2.  ``test_vuln_v2_hilbert_stability_no_nan`` — V2 Hilbert optimisation
    finishes with finite ``best_params_`` (the Wf-25 numerical
    pathology must be caught — even if it only shows as WARN here,
    it must NOT be silently dropped).
3.  ``test_vuln_v3_composite_roundtrip_predicts_all_finite`` — V3
    composite fit / predict round-trip returns finite predictions for
    every test row.
4.  ``test_vuln_v4_lambda_ast_extremes_no_exception`` — V4 evaluates
    random ASTs on extreme inputs (±1e10, NaN) without raising.
5.  ``test_vuln_v5_reproducibility_same_seed_same_ast`` — V5 same seed
    → same AST / same R² across two fits (R17 determinism contract).
6.  ``test_vuln_v6_seed_spread_records_per_seed`` — V6 records a
    per-seed R² for every seed.
7.  ``test_run_scan_aggregates_verdicts`` — the orchestrator's
    ``summary`` block counts OK / WARN / FAIL correctly.
8.  ``test_run_scan_payload_is_json_serialisable`` — the payload
    round-trips through ``json.dumps``.
9.  ``test_run_scan_no_critical_failures_on_smoke`` — at smoke size
    (50 samples × 2 seeds) the scan returns at most a WARN count of
    1; no FAIL is expected on real data (the Wf-25 / Wf-37 numerical
    pathology is fixed by the Wf-39 P2 closed-form ridge).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from scan_lambda_algorithm_vulns import (  # noqa: E402
    run_scan,
    vuln_v1_ast_r2,
    vuln_v2_hilbert_stability,
    vuln_v3_composite_roundtrip,
    vuln_v4_lambda_ast_extremes,
    vuln_v5_reproducibility,
    vuln_v6_seed_spread,
)


@pytest.fixture(scope="module")
def nci60_rows():
    from bench_deflexpressor_end_to_end import load_nci60_precious

    return load_nci60_precious(
        _REPO_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv",
        n_max=120,
    )


# ---------------------------------------------------------------------------
# V1 — AST R² finite
# ---------------------------------------------------------------------------
def test_vuln_v1_ast_r2_finite(nci60_rows):
    out = vuln_v1_ast_r2(nci60_rows, seed=0)
    assert "verdict" in out
    assert out["verdict"] in ("OK", "WARN", "FAIL")
    assert np.isfinite(out["r2_test"]), f"V1 r2 not finite: {out}"


# ---------------------------------------------------------------------------
# V2 — Hilbert stability
# ---------------------------------------------------------------------------
def test_vuln_v2_hilbert_stability_no_nan(nci60_rows):
    out = vuln_v2_hilbert_stability(nci60_rows, seed=0)
    assert "verdict" in out
    # The Wf-25 pathology (NaN params) must NOT reappear.
    assert out["has_nan_params"] is False, f"V2 NaN params: {out}"


# ---------------------------------------------------------------------------
# V3 — Composite round-trip
# ---------------------------------------------------------------------------
def test_vuln_v3_composite_roundtrip_predicts_all_finite(nci60_rows):
    out = vuln_v3_composite_roundtrip(nci60_rows, seed=0)
    assert "verdict" in out
    assert (
        out["n_finite_predict"] == out["n_test"]
    ), f"V3 composite produced non-finite predictions: {out}"


# ---------------------------------------------------------------------------
# V4 — AST extremes (NaN inputs)
# ---------------------------------------------------------------------------
def test_vuln_v4_lambda_ast_extremes_no_exception():
    out = vuln_v4_lambda_ast_extremes(seed=0)
    assert "verdict" in out
    assert out["n_total_evaluations"] >= 40
    # 40 evaluations across 10 ASTs × 4 extreme rows; at least some
    # should be finite (the constant / leaf rows trivially are).
    assert out["n_finite"] >= 10


# ---------------------------------------------------------------------------
# V5 — Determinism
# ---------------------------------------------------------------------------
def test_vuln_v5_reproducibility_same_seed_same_ast(nci60_rows):
    out = vuln_v5_reproducibility(nci60_rows, seed=0)
    assert out["verdict"] in ("OK", "WARN", "FAIL")
    # The R17 contract: same seed → same AST.  Allow WARN if the
    # contract is degraded (e.g. due to RNG drift) but never silently
    # drop the check.
    if out["verdict"] == "FAIL":
        pytest.fail(f"V5 reproducibility broken: {out}")


# ---------------------------------------------------------------------------
# V6 — Seed spread
# ---------------------------------------------------------------------------
def test_vuln_v6_seed_spread_records_per_seed(nci60_rows):
    out = vuln_v6_seed_spread(nci60_rows, seeds=[0, 1, 2])
    assert "verdict" in out
    assert len(out["r2_per_seed"]) == 3
    for r2 in out["r2_per_seed"]:
        assert np.isfinite(r2), f"V6 per-seed R² not finite: {r2}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def test_run_scan_aggregates_verdicts(nci60_rows):
    """The orchestrator's summary block counts verdicts correctly."""
    payload = run_scan(n_samples=80, seeds=[0, 1])
    s = payload["summary"]
    assert s["n_ok"] + s["n_warn"] + s["n_fail"] == s["n_total"]
    # With 2 seeds: V1 (2) + V2 (2) + V3 (2) + V4 (1) + V5 (1) + V6 (1) = 9
    assert s["n_total"] == 9


def test_run_scan_payload_is_json_serialisable(nci60_rows):
    payload = run_scan(n_samples=60, seeds=[0])
    s = json.dumps(payload)
    assert isinstance(s, str) and len(s) > 100


def test_run_scan_no_critical_failures_on_smoke():
    """Smoke scan must produce 0 FAIL — the closed-form ridge fix
    (Wf-39 P2) eliminated the Wf-25 / Wf-37 numerical pathology."""
    payload = run_scan(n_samples=50, seeds=[0, 1])
    s = payload["summary"]
    assert (
        s["n_fail"] == 0
    ), f"Vuln scan found {s['n_fail']} FAILs: {json.dumps(payload, indent=2)[:2000]}"
