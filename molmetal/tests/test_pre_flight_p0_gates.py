"""Tests for Wf-260 pre-flight P0 BLOCKER gates.

Verifies:
1. `gate_yaml_schema` PASSes on the 7 active YAMLs and FAILs on a
   deliberately malformed file.
2. `gate_csv_hash` PASSes on real CSVs and FAILs when a row count
   threshold is breached.
3. `gate_scaffold_zero_leak` (Wf-261-Doc NEW P0 BLOCKER) PASSes on the
   strict scaffold-split v2 CSV and FAILs on the leaky v1 CSV.
4. Both helpers return GateResult objects with the right shape.

Run with: pytest molmetal/tests/test_pre_flight_p0_gates.py -v
"""
from pathlib import Path

import pytest

from molmetal.scripts.pre_flight_check import (
    gate_yaml_schema,
    gate_csv_hash,
    gate_scaffold_zero_leak,
    REPO_ROOT,
)


# ----------------------------------------------------------------------------
# gate_yaml_schema
# ----------------------------------------------------------------------------


def test_yaml_schema_passes_on_active_configs():
    """All 7 active configs/*.yaml must parse via yaml.safe_load."""
    configs_dir = REPO_ROOT / "configs"
    yaml_paths = [
        configs_dir / "cfm_full_retrain.yaml",
        configs_dir / "cfm_per_innovation_ablation_p1.4.yaml",
        configs_dir / "cfm_real_decode_lift_p1.3.yaml",
        configs_dir / "wf124_p3_50epoch_full_sweep_g5.yaml",
        configs_dir / "wf124_p3_per_innovation_ablation_g6.yaml",
        configs_dir / "wf124_p3_remaining_metals_g4.yaml",
        configs_dir / "default.yaml",
    ]
    # Sanity: every file is on disk.
    for p in yaml_paths:
        assert p.is_file(), f"missing config: {p}"

    res = gate_yaml_schema(
        "test_yaml_schema_passes", "P0", "config", yaml_paths
    )
    assert res.status == "PASS"
    assert "7 YAMLs" in res.evidence


def test_yaml_schema_fails_on_missing_file(tmp_path):
    """Missing files surface as FAIL with a list."""
    missing = tmp_path / "does_not_exist.yaml"
    res = gate_yaml_schema(
        "test_yaml_schema_missing", "P0", "config", [missing]
    )
    assert res.status == "FAIL"
    assert "missing" in res.evidence


def test_yaml_schema_fails_on_malformed_yaml(tmp_path):
    """YAML syntax errors surface as FAIL."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: [unclosed bracket\nbar: : :")
    res = gate_yaml_schema(
        "test_yaml_schema_malformed", "P0", "config", [bad]
    )
    assert res.status == "FAIL"
    assert "invalid" in res.evidence


# ----------------------------------------------------------------------------
# gate_csv_hash
# ----------------------------------------------------------------------------


def test_csv_hash_passes_on_real_metalcytotox():
    """The real MetalCytoToxDB CSV has 26802 lines, well over the 26800 floor."""
    cytox = REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
    assert cytox.is_file()
    res = gate_csv_hash(
        "test_csv_hash_cytox", "P0", "data", [(cytox, 26800)]
    )
    assert res.status == "PASS"


def test_csv_hash_passes_on_real_metallo_drugs():
    """The real metallo_drugs_500_train.csv has 450 lines."""
    md = REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
    assert md.is_file()
    res = gate_csv_hash(
        "test_csv_hash_md", "P0", "data", [(md, 449)]
    )
    assert res.status == "PASS"


def test_csv_hash_fails_on_breached_threshold(tmp_path):
    """A 1-row CSV breaches a 100-row threshold → FAIL."""
    tiny = tmp_path / "tiny.csv"
    tiny.write_text("a,b,c\n1,2,3\n")
    res = gate_csv_hash(
        "test_csv_hash_tiny", "P0", "data", [(tiny, 100)]
    )
    assert res.status == "FAIL"
    assert "lines" in res.evidence


def test_csv_hash_fails_on_missing(tmp_path):
    """Missing CSV surfaces as FAIL."""
    res = gate_csv_hash(
        "test_csv_hash_missing", "P0", "data",
        [(tmp_path / "ghost.csv", 1)],
    )
    assert res.status == "FAIL"
    assert "missing" in res.evidence


# ----------------------------------------------------------------------------
# gate_scaffold_zero_leak (Wf-261-Doc NEW P0 BLOCKER)
# ----------------------------------------------------------------------------


def test_scaffold_zero_leak_passes_on_v2():
    """The v2 CSV (post-Wf-261) has 0 scaffold_id overlap across train/val/test.

    Strict Bemis-Murcko scaffold 70/10/20 split (seed=42) per
    `molmetal/reports/wf261_regenerate/regen_report.md` (sha256
    `cc3ea24755f1b14a69c8c3acd827db0bc1dd123301479c9ca490920725adf063`,
    26 632 rows, 3 069 unique scaffolds).
    """
    v2 = REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v2.csv"
    assert v2.is_file()
    # HOLE-5 (Wf-446): gate signature changed to List[Tuple[Path, label]];
    # this call site adapted to the new list form (assertions unchanged).
    res = gate_scaffold_zero_leak(
        "test_scaffold_zero_leak_v2", "P0", "data", [(v2, "metalcytotox_v2")]
    )
    assert res.status == "PASS"
    assert "0 scaffold leak" in res.evidence


def test_scaffold_zero_leak_fails_on_v1():
    """The v1 CSV (pre-Wf-261) has 30/48/32 scaffold_id overlaps.

    v1 was generated by per-metal random 70/10/20 (per
    `molmetal/reports/wf261_audit/split_audit.md` §4); cross-metal scaffold
    spans leak through. This test confirms the gate correctly identifies
    the leak.
    """
    v1 = REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
    assert v1.is_file()
    res = gate_scaffold_zero_leak(
        "test_scaffold_zero_leak_v1", "P0", "data", [(v1, "metalcytotox_v1")]
    )
    assert res.status == "FAIL"
    assert "leaks=" in res.evidence


def test_scaffold_zero_leak_fails_on_missing(tmp_path):
    """A missing CSV surfaces as FAIL."""
    ghost = tmp_path / "ghost_v2.csv"
    res = gate_scaffold_zero_leak(
        "test_scaffold_zero_leak_missing", "P0", "data",
        [(ghost, "metalcytotox_missing")],
    )
    assert res.status == "FAIL"
    assert "missing" in res.evidence.lower()


# ----------------------------------------------------------------------------
# GateResult shape
# ----------------------------------------------------------------------------


def test_gate_result_fields():
    """Every gate result must carry name/priority/pipeline/status/evidence/wall_sec."""
    configs_dir = REPO_ROOT / "configs"
    res = gate_yaml_schema(
        "shape_test", "P0", "config",
        [configs_dir / "cfm_full_retrain.yaml"],
    )
    assert res.name == "shape_test"
    assert res.priority == "P0"
    assert res.pipeline == "config"
    assert res.status in ("PASS", "FAIL", "SKIP")
    assert isinstance(res.evidence, str)
    assert isinstance(res.wall_sec, float)
    assert res.wall_sec >= 0.0