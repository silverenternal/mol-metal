"""Pytest: TODO-23 metric credibility audit (read-only).

Verifies the credibility audit script and JSON companion:

1. ``test_audit_loads_all_metrics`` — the audit JSON contains 20+1 metrics.
2. ``test_audit_classifies_strong`` — strong bucket has ≥5 metrics.
3. ``test_audit_classifies_acceptable`` — acceptable bucket has ≥2 metrics.
4. ``test_audit_classifies_weak`` — weak bucket has ≥8 metrics.
5. ``test_audit_classifies_unmeasured`` — unmeasured bucket exists.
6. ``test_audit_gap_calculation`` — gap = measured - sota (sign-flip for m3/m10).
7. ``test_audit_no_overclaim`` — no weak metric is also classified as strong.
8. ``test_audit_ready_for_paper`` — ready_for_paper is True iff ≥20 metrics
   classified, no overclaim, and ≥5 strong.
9. ``test_audit_strong_ids_include_headlines`` — m11, m14, m17 in strong.
10. ``test_audit_weak_ids_include_anchors`` — m1, m3, m7, m19 in weak.
11. ``test_audit_script_runs`` — running the script writes
    credibility_audit.json with the right keys.
12. ``test_audit_script_outputs_n_counts`` — n_strong + n_acceptable +
    n_weak + n_unmeasured + n_design_only ≥ 20.
13. ``test_audit_extras_handled`` — m21 (extra) appears in acceptable_ids.
14. ``test_audit_lower_better_metrics`` — m3, m10 use kcal/mol direction.
15. ``test_audit_no_overclaim_in_buckets`` — no metric appears in two buckets.

All tests run on CPU in <2s — no GPU, no external evaluators, no
modification of paper §4.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.metric_credibility_audit import (  # noqa: E402
    OUT_JSON,
    build_credibility_audit,
    load_audit,
)

LOWER_IS_BETTER = ["m3", "m10"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _audit() -> dict:
    return load_audit()


def _cred() -> dict:
    return build_credibility_audit(_audit())


# ---------------------------------------------------------------------------
# Test 1: audit loads all metrics
# ---------------------------------------------------------------------------
def test_audit_loads_all_metrics():
    audit = _audit()
    metrics = audit.get("metrics", [])
    assert (
        len(metrics) == 20
    ), f"metrics_audit.json should contain 20 metric entries, got {len(metrics)}"
    # m21 is in extras
    assert "metric_21_nci60_precious_9cell_auc" in audit.get("extras", {})


# ---------------------------------------------------------------------------
# Test 2: audit classifies strong
# ---------------------------------------------------------------------------
def test_audit_classifies_strong():
    cred = _cred()
    assert cred["n_strong"] >= 5, f"expected ≥5 strong metrics, got {cred['n_strong']}"
    assert len(cred["strong_ids"]) == cred["n_strong"]


# ---------------------------------------------------------------------------
# Test 3: audit classifies acceptable
# ---------------------------------------------------------------------------
def test_audit_classifies_acceptable():
    cred = _cred()
    assert (
        cred["n_acceptable"] >= 2
    ), f"expected ≥2 acceptable metrics, got {cred['n_acceptable']}"


# ---------------------------------------------------------------------------
# Test 4: audit classifies weak
# ---------------------------------------------------------------------------
def test_audit_classifies_weak():
    cred = _cred()
    assert cred["n_weak"] >= 8, f"expected ≥8 weak metrics, got {cred['n_weak']}"


# ---------------------------------------------------------------------------
# Test 5: audit classifies unmeasured
# ---------------------------------------------------------------------------
def test_audit_classifies_unmeasured():
    cred = _cred()
    assert (
        cred["n_unmeasured"] >= 1
    ), f"expected ≥1 unmeasured metric, got {cred['n_unmeasured']}"


# ---------------------------------------------------------------------------
# Test 6: gap calculation correctness
# ---------------------------------------------------------------------------
def test_audit_gap_calculation():
    audit = _audit()
    metrics = audit.get("metrics", [])
    for m in metrics:
        mid = m["id"]
        measured = m.get("measured")
        tier1 = m.get("tier1_sota")
        gap = m.get("gap")
        if measured is None or tier1 is None or gap is None:
            # unmeasured entries may not have gap as a number
            continue
        if isinstance(measured, dict) or isinstance(tier1, dict):
            # m17 has measured/tier1/gap as per-split dicts
            # verify each key matches
            assert (
                isinstance(measured, dict)
                and isinstance(tier1, dict)
                and isinstance(gap, dict)
            ), f"{mid}: mixed scalar/dict types for measured/tier1/gap"
            for key in gap:
                if measured.get(key) is None or tier1.get(key) is None:
                    continue
                expected = round(measured[key] - tier1[key], 4)
                assert (
                    abs(gap[key] - expected) < 0.01
                ), f"{mid}/{key}: gap={gap[key]}, expected ≈ {expected}"
            continue
        if mid in LOWER_IS_BETTER:
            # gap = measured - tier1 (signed; positive = behind = bad)
            expected = round(measured - tier1, 4)
            assert (
                abs(gap - expected) < 0.01
            ), f"{mid}: gap={gap}, expected ≈ {expected}"
        else:
            # gap = measured - tier1 (signed; negative = behind = bad)
            expected = round(measured - tier1, 4)
            assert (
                abs(gap - expected) < 0.01
            ), f"{mid}: gap={gap}, expected ≈ {expected}"


# ---------------------------------------------------------------------------
# Test 7: no overclaim (no weak metric also classified as strong)
# ---------------------------------------------------------------------------
def test_audit_no_overclaim():
    cred = _cred()
    assert cred["no_overclaim"] is True, cred["overclaim_msg"]
    weak = set(cred["weak_ids"])
    strong = set(cred["strong_ids"])
    assert not (
        weak & strong
    ), f"overclaim detected: {weak & strong} appear in both weak and strong"


# ---------------------------------------------------------------------------
# Test 8: ready_for_paper is True iff ≥20 metrics + no overclaim + ≥5 strong
# ---------------------------------------------------------------------------
def test_audit_ready_for_paper():
    cred = _cred()
    n_total = (
        cred["n_strong"]
        + cred["n_acceptable"]
        + cred["n_weak"]
        + cred["n_unmeasured"]
        + cred["n_design_only"]
    )
    expected_ready = n_total >= 20 and cred["no_overclaim"] and cred["n_strong"] >= 5
    assert cred["ready_for_paper"] == expected_ready


# ---------------------------------------------------------------------------
# Test 9: strong_ids include headline metrics
# ---------------------------------------------------------------------------
def test_audit_strong_ids_include_headlines():
    cred = _cred()
    strong = set(cred["strong_ids"])
    for headline in ("m11", "m14", "m17"):
        assert (
            headline in strong
        ), f"headline metric {headline} should be strong; got {sorted(strong)}"


# ---------------------------------------------------------------------------
# Test 10: weak_ids include known weak anchors
# ---------------------------------------------------------------------------
def test_audit_weak_ids_include_anchors():
    cred = _cred()
    weak = set(cred["weak_ids"])
    for anchor in ("m1", "m3", "m7", "m19"):
        assert (
            anchor in weak
        ), f"anchor metric {anchor} should be weak; got {sorted(weak)}"


# ---------------------------------------------------------------------------
# Test 11: audit script runs and writes credibility_audit.json
# ---------------------------------------------------------------------------
def test_audit_script_runs(tmp_path=None):
    # Use subprocess to run the script as a real CLI invocation.
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "molmetal" / "scripts" / "metric_credibility_audit.py"),
            "--audit",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        result.returncode == 0
    ), f"script failed: stdout={result.stdout} stderr={result.stderr}"
    assert OUT_JSON.exists(), f"{OUT_JSON} not written"
    with OUT_JSON.open() as f:
        payload = json.load(f)
    for key in (
        "audit_date",
        "n_strong",
        "n_acceptable",
        "n_weak",
        "n_unmeasured",
        "strong_ids",
        "acceptable_ids",
        "weak_ids",
        "ready_for_paper",
    ):
        assert key in payload, f"missing key {key} in credibility_audit.json"


# ---------------------------------------------------------------------------
# Test 12: n_counts sum to ≥20
# ---------------------------------------------------------------------------
def test_audit_script_outputs_n_counts():
    payload_path = OUT_JSON
    if not payload_path.exists():
        # Run the script first
        subprocess.run(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "molmetal"
                    / "scripts"
                    / "metric_credibility_audit.py"
                ),
                "--audit",
            ],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            timeout=30,
        )
    with payload_path.open() as f:
        payload = json.load(f)
    total = (
        payload["n_strong"]
        + payload["n_acceptable"]
        + payload["n_weak"]
        + payload["n_unmeasured"]
        + payload["n_design_only"]
    )
    assert total >= 20, f"expected ≥20 metrics total, got {total}"


# ---------------------------------------------------------------------------
# Test 13: extras (m21) handled
# ---------------------------------------------------------------------------
def test_audit_extras_handled():
    cred = _cred()
    assert (
        "m21" in cred["acceptable_ids"]
    ), f"m21 (extra) should be in acceptable_ids; got {cred['acceptable_ids']}"


# ---------------------------------------------------------------------------
# Test 14: lower-is-better metrics (m3, m10) use kcal/mol direction
# ---------------------------------------------------------------------------
def test_audit_lower_better_metrics():
    audit = _audit()
    metrics = {m["id"]: m for m in audit["metrics"]}
    for mid in LOWER_IS_BETTER:
        assert mid in metrics, f"{mid} missing from audit"
        m = metrics[mid]
        assert (
            m["direction"] == "lower"
        ), f"{mid}: direction should be 'lower' (kcal/mol), got {m['direction']}"
        # gap should be positive (we are behind SOTA)
        if m.get("gap") is not None:
            assert (
                m["gap"] > 0
            ), f"{mid}: gap should be >0 (behind SOTA), got {m['gap']}"


# ---------------------------------------------------------------------------
# Test 15: no metric appears in two buckets
# ---------------------------------------------------------------------------
def test_audit_no_overclaim_in_buckets():
    cred = _cred()
    all_ids = (
        cred["strong_ids"]
        + cred["acceptable_ids"]
        + cred["weak_ids"]
        + cred["unmeasured_ids"]
        + cred["design_only_ids"]
    )
    assert len(all_ids) == len(set(all_ids)), (
        f"duplicate metric IDs across buckets: "
        f"{[x for x in all_ids if all_ids.count(x) > 1]}"
    )
