"""Regression tests for Wf-427-A4 B5 Wf-125 schema adapter.

Per Wf-425-I5 + Wf-426-A1(e) finding: Wf-125 trainer caller at
molmetal/scripts/train_metal_lomo_wasserstein.py:211 reads
``entry.get("catherine_6of6_pass", False)`` but Wf-387-A2 cohort JSON
per_compound[] has 9 fields (NO catherine_6of6_pass). This caused
RuntimeError because all 500 entries got filtered out.

Wf-427-A4 fix: line 211 now reads
``entry.get("catherine_6of6_pass", False) or entry.get("is_active", False)``.
This accepts both schemas (Option A per Wf-425-I5 finding).

Tests:
  1. Wf-387-A2 schema (is_active field) returns > 0 SMILES (was 0 pre-fix)
  2. Wf-125 P1 legacy schema (catherine_6of6_pass field) still works
  3. Empty fields raise RuntimeError
  4. Mixed schema: catherine_6of6_pass=True OR is_active=1 both pass
  5. Active cohort loads without RuntimeError on real Wf-387-A2 cohort
"""
import ast
import json
import tempfile
import os
import typing
from pathlib import Path


def _load_load_rh_synthetic_smiles():
    """Extract the load_rh_synthetic_smiles function from trainer file."""
    src = open(
        "molmetal/scripts/train_metal_lomo_wasserstein.py"
    ).read()
    tree = ast.parse(src)
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "load_rh_synthetic_smiles"
        ):
            fn_code = ast.get_source_segment(src, node)
            break
    ns = {"Path": Path, "List": typing.List, "json": json}
    exec(fn_code, ns)
    return ns["load_rh_synthetic_smiles"]


def _write_temp_cohort(payload):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(payload, f)
    f.close()
    return f.name


def test_wf387a2_schema_is_active_filter():
    """Wf-387-A2 schema uses is_active field (NO catherine_6of6_pass).
    Pre-fix: 0 SMILES returned. Post-fix: active entries pass."""
    fn = _load_load_rh_synthetic_smiles()
    p = _write_temp_cohort(
        {
            "per_compound": [
                {
                    "compound_id": "a",
                    "smiles": "CCO",
                    "is_active": 1,
                },
                {
                    "compound_id": "b",
                    "smiles": "CCC",
                    "is_active": 0,
                },
                {
                    "compound_id": "c",
                    "smiles": "CCN",
                    "is_active": 1,
                },
            ]
        }
    )
    try:
        smis = fn(Path(p))
        assert len(smis) == 2, f"expected 2 active SMILES, got {len(smis)}"
        assert "CCO" in smis
        assert "CCN" in smis
        assert "CCC" not in smis
    finally:
        os.unlink(p)


def test_wf125_p1_legacy_schema_still_works():
    """Backward-compat: Wf-125 P1 legacy schema with catherine_6of6_pass
    field still filters correctly."""
    fn = _load_load_rh_synthetic_smiles()
    p = _write_temp_cohort(
        {
            "per_compound": [
                {
                    "compound_id": "x",
                    "smiles": "CCO",
                    "catherine_6of6_pass": True,
                },
                {
                    "compound_id": "y",
                    "smiles": "CCC",
                    "catherine_6of6_pass": False,
                },
                {
                    "compound_id": "z",
                    "smiles": "CCN",
                    "catherine_6of6_pass": True,
                },
            ]
        }
    )
    try:
        smis = fn(Path(p))
        assert len(smis) == 2, f"expected 2 PASS SMILES, got {len(smis)}"
        assert "CCO" in smis
        assert "CCN" in smis
        assert "CCC" not in smis
    finally:
        os.unlink(p)


def test_empty_fields_raises_runtime_error():
    """Empty / no field raises RuntimeError per load_rh_synthetic_smiles contract."""
    fn = _load_load_rh_synthetic_smiles()
    p = _write_temp_cohort(
        {"per_compound": [{"compound_id": "q", "smiles": "CCO"}]}
    )
    try:
        try:
            fn(Path(p))
            assert False, "RuntimeError should have been raised"
        except RuntimeError as e:
            assert "No Catherine-6/6 PASS" in str(e)
    finally:
        os.unlink(p)


def test_mixed_schema_or_logic():
    """Mixed schema: catherine_6of6_pass=True OR is_active=1 both pass."""
    fn = _load_load_rh_synthetic_smiles()
    p = _write_temp_cohort(
        {
            "per_compound": [
                {
                    "compound_id": "x1",
                    "smiles": "CCO",
                    "catherine_6of6_pass": True,
                },
                {
                    "compound_id": "x2",
                    "smiles": "CCN",
                    "catherine_6of6_pass": False,
                    "is_active": 1,
                },
                {
                    "compound_id": "x3",
                    "smiles": "CCC",
                    "catherine_6of6_pass": False,
                    "is_active": 0,
                },
                {
                    "compound_id": "x4",
                    "smiles": "CMB",
                    "catherine_6of6_pass": True,
                    "is_active": 0,
                },
            ]
        }
    )
    try:
        smis = fn(Path(p))
        # Expected: CCO (x1: catherine), CCN (x2: is_active), CMB (x4: catherine)
        assert len(smis) == 3, f"expected 3 SMILES, got {len(smis)}: {smis}"
        assert "CCO" in smis
        assert "CCN" in smis
        assert "CMB" in smis
        assert "CCC" not in smis
    finally:
        os.unlink(p)


def test_real_wf387a2_cohort_loads_without_runtime_error():
    """Real Wf-387-A2 balanced 500 cohort: load_rh_synthetic_smiles
    must return 250 active SMILES (no RuntimeError)."""
    cohort = Path(
        "molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json"
    )
    assert cohort.exists(), f"missing cohort {cohort}"
    fn = _load_load_rh_synthetic_smiles()
    smis = fn(cohort)
    assert len(smis) == 250, (
        f"Wf-387-A2 expected 250 active SMILES, got {len(smis)}"
    )
    assert all(isinstance(s, str) and s.strip() for s in smis)


def test_real_cohort_includes_only_active():
    """Verify the 250 returned SMILES correspond to is_active=1 entries."""
    cohort_path = Path(
        "molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json"
    )
    fn = _load_load_rh_synthetic_smiles()
    smis = fn(cohort_path)
    with cohort_path.open() as fh:
        payload = json.load(fh)
    active_smis = {
        str(e.get("smiles", "")).strip()
        for e in payload["per_compound"]
        if e.get("is_active", False)
    }
    assert set(smis) == active_smis, (
        "Returned SMILES do not match is_active=True entries"
    )


if __name__ == "__main__":
    test_wf387a2_schema_is_active_filter()
    print("PASS test_wf387a2_schema_is_active_filter")
    test_wf125_p1_legacy_schema_still_works()
    print("PASS test_wf125_p1_legacy_schema_still_works")
    test_empty_fields_raises_runtime_error()
    print("PASS test_empty_fields_raises_runtime_error")
    test_mixed_schema_or_logic()
    print("PASS test_mixed_schema_or_logic")
    test_real_wf387a2_cohort_loads_without_runtime_error()
    print("PASS test_real_wf387a2_cohort_loads_without_runtime_error")
    test_real_cohort_includes_only_active()
    print("PASS test_real_cohort_includes_only_active")
    print("ALL 6 TESTS PASS")
