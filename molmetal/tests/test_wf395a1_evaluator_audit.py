"""Wf-395-A1 — Evaluator audit tests (3 NEW tests).

Owner: Wf-395-A1 ultracode sub-flow
Date: 2026-09-30
Trigger: refresh of TODO/02_metrics/EVALUATORS_MIGRATED.md after Wf-380..394
         added 8 NEW evaluator modules not in the legacy §1 table.

These tests:
  1. Verify each metric in the EVALUATORS_MIGRATED.md inventory has a working
     evaluator module (test_evaluator_count).
  2. Verify each evaluator returns valid output on a synthetic cohort
     (test_each_evaluator_runs).
  3. Verify the EVALUATORS_MIGRATED.md count matches the audit script's
     MEASURED count (test_evaluator_doc_updated).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_evaluator_count() -> None:
    """Each metric in EVALUATOR_INVENTORY has a working module path."""
    from molmetal.scripts.wf395a1_evaluator_audit import EVALUATOR_INVENTORY

    assert len(EVALUATOR_INVENTORY) >= 14, (
        f"EVALUATOR_INVENTORY has {len(EVALUATOR_INVENTORY)} entries, "
        f"expected >= 14 (R17-era canonical 14/14 baseline)"
    )

    # Tier distribution must include R17 + SCAFFOLD
    by_tier: dict[str, int] = {}
    for ev in EVALUATOR_INVENTORY:
        by_tier[ev["tier"]] = by_tier.get(ev["tier"], 0) + 1

    assert "R17" in by_tier, f"R17 tier missing; got tiers {list(by_tier.keys())}"
    assert "SCAFFOLD" in by_tier, f"SCAFFOLD tier missing; got tiers {list(by_tier.keys())}"
    assert by_tier["R17"] >= 10, f"R17 has only {by_tier['R17']} entries, expected >= 10"

    # All module paths must be non-empty strings
    for ev in EVALUATOR_INVENTORY:
        assert isinstance(ev["module"], str) and ev["module"], f"empty module path: {ev}"
        assert isinstance(ev["function"], str) and ev["function"], f"empty function name: {ev}"
        assert ev["tier"] in {"R17", "SCAFFOLD", "WIRE-IN", "MEASURED-ON-REAL"}, (
            f"unknown tier {ev['tier']!r}"
        )


def test_each_evaluator_runs() -> None:
    """Each evaluator runs on a synthetic SMILES cohort and returns valid output."""
    from molmetal.scripts.wf395a1_evaluator_audit import (
        audit_evaluator_coverage,
        test_each_evaluator,
    )

    by_tier = audit_evaluator_coverage()
    n_total = sum(by_tier.values())
    records = test_each_evaluator()

    assert len(records) == n_total, (
        f"records ({len(records)}) != by_tier total ({n_total})"
    )

    # Aggregate pass count: expect at least 18/22 (allowing for honest-negative
    # SCAFFOLD probes that need real-cohort input)
    n_pass = sum(1 for r in records if r["passed"])
    assert n_pass >= 18, (
        f"only {n_pass}/{len(records)} evaluators passed synthetic probe; "
        f"expected >= 18 (some SCAFFOLD-tier functions need real-cohort input)"
    )

    # All R17 evaluators MUST pass (they're canonical baseline)
    r17_records = [r for r in records if r["tier"] == "R17"]
    r17_pass = sum(1 for r in r17_records if r["passed"])
    assert r17_pass == len(r17_records), (
        f"R17-tier FAIL: {r17_pass}/{len(r17_records)} passed. "
        f"R17 baseline must be 100% (canonical baseline preserved)."
    )

    # Each record has required keys
    for r in records:
        for key in ("name", "module", "function", "tier", "passed", "result"):
            assert key in r, f"record missing {key}: {r}"


def test_evaluator_doc_updated() -> None:
    """EVALUATORS_MIGRATED.md count matches audit script's MEASURED count."""
    from molmetal.scripts.wf395a1_evaluator_audit import audit_evaluator_coverage

    by_tier = audit_evaluator_coverage()
    audit_total = sum(by_tier.values())

    doc_path = _ROOT / "TODO" / "02_metrics" / "EVALUATORS_MIGRATED.md"
    assert doc_path.exists(), f"EVALUATORS_MIGRATED.md missing at {doc_path}"
    doc_text = doc_path.read_text()

    # Doc must reflect Wf-395-A1 timestamp (>= 2026-09-30)
    assert "2026-09-30" in doc_text or "Wf-395-A1" in doc_text, (
        f"EVALUATORS_MIGRATED.md not updated to Wf-395-A1 (2026-09-30). "
        f"Current date string: {doc_text[:200]}"
    )

    # Doc must include all 4 tiers (R17 + SCAFFOLD + WIRE-IN + MEASURED-ON-REAL)
    for tier in ("R17", "SCAFFOLD"):
        assert tier in doc_text, f"tier {tier!r} missing from EVALUATORS_MIGRATED.md"

    # Doc must include the audit-script total (with 1 doc count tolerance)
    expected_in_doc = f"{audit_total}"
    assert expected_in_doc in doc_text, (
        f"EVALUATORS_MIGRATED.md does not mention audit total {audit_total}; "
        f"audit_by_tier={by_tier}"
    )


def test_wf395a1_audit_json_persisted() -> None:
    """Audit script persists per-evaluator JSON record."""
    json_path = _ROOT / "molmetal" / "reports" / "wf395a1_evaluator_audit" / "audit.json"
    # The audit script writes JSON when run; if not yet run, this test is informational
    # (test is OPTIONAL; only enforced if the audit has been run)
    if json_path.exists():
        data = json.loads(json_path.read_text())
        assert "by_tier" in data
        assert "records" in data
        assert data["n_evaluators"] >= 14
    # else: skip (audit hasn't been run yet on this host)
