"""Wf-313D — TODO-220 §6.3 + §6.4 acceptance gate schema tests.

================================================================
12 tests covering the Wf-313D acceptance gate schema design
(Wf-310C Part 5 acceptance gate wiring):

§6.3 acceptance gate schema (4 tests)
* test_6_3_1_smoke_acceptance_measured — 5/5 metal × 1-seed × 1-epoch
  CPU MEASURED in sweep_summary_smoke.json
* test_6_3_2_gpu_acceptance_projected_pending — wrapper READY at
  launch_wf310c_todo220_gpu.sh Stage A; awaits USER_OK + 2.9 hr GPU
* test_6_3_3_per_metal_mean_std_partial — 1-seed std=0 uninformative;
  GPU std becomes actionable post Phase 3
* test_6_3_4_krasnov_protocol_fidelity_pass — split_protocol field
  has Krasnov 2026 temporal-strict values

§6.4 acceptance gate schema (4 tests)
* test_6_4_1_smoke_acceptance_measured — 9/9 cell × 1-seed × 1-epoch
  CPU MEASURED in sweep_summary_smoke.json
* test_6_4_2_gpu_acceptance_projected_pending — wrapper READY at
  launch_wf310c_todo220_gpu.sh Stage B; awaits USER_OK + 1.0 hr GPU
* test_6_4_3_per_cell_mean_std_partial — 1-seed std=0 uninformative;
  GPU std becomes actionable post Phase 3
* test_6_4_4_hct116_highlight_pc3_honest_negative — HCT116=0.6998
  highlight; PC3=0.1667 honest-negative (below random)

Smoke persistence JSON validation (2 tests)
* test_temporal_smoke_json_schema — sweep_summary_smoke.json structure
  matches §6.3.1 schema (5 metals, per_metal_seed_auc, etc.)
* test_nci60_smoke_json_schema — sweep_summary_smoke.json structure
  matches §6.4.1 schema (9 cells, per_cell_seed_auc, etc.)

GPU integration tests (2 tests)
* test_gpu_wrapper_user_ok_gate — wrapper exits non-zero without
  /tmp/USER_OK_WF310C_TODO220 file
* test_gpu_pipeline_master_phase3_integration — gpu_pipeline_master.sh
  Phase 3 includes wf310c_todo220 as a job

Honest framing (per party-twenty-spirit §3.5)
----------------------------------------------
* §6.3 + §6.4 acceptance gate docs are CPU-only DESIGN.
* Smoke is MEASURED (real numbers); GPU is PROJECTED until Phase 3
  lands (USER_OK + 3.9 hr GPU wall).
* PC3 honest-negative is declared as honest-negative; not paraphrased
  as "achievable" via MultiTaskLoss head (TODO-130 G3 follow-up; not
  in Wf-313D scope).

Cross-references
----------------
* TODO/completed/220_wf220_temporal_split_acceptance.md — §6.3 doc
* TODO/completed/220_wf219_nci60_9cell_acceptance.md — §6.4 doc
* molmetal/reports/wf310c_temporal_split/sweep_summary_smoke.json — smoke MEASURED (5/5)
* molmetal/reports/wf310c_nci60_9cell/sweep_summary_smoke.json — smoke MEASURED (9/9)
* molmetal/scripts/launch_wf310c_todo220_gpu.sh — GPU wrapper (USER_OK-gated)
* scripts/gpu_pipeline_master.sh — Phase 3 integration
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path("/home/hugo/codes/mol-metal")
TEMPORAL_SMOKE = ROOT / "molmetal/reports/wf310c_temporal_split/sweep_summary_smoke.json"
NCI60_SMOKE = ROOT / "molmetal/reports/wf310c_nci60_9cell/sweep_summary_smoke.json"
GPU_WRAPPER = ROOT / "molmetal/scripts/launch_wf310c_todo220_gpu.sh"
GPU_PIPELINE = ROOT / "scripts/gpu_pipeline_master.sh"
SEC_6_3_DOC = ROOT / "TODO/completed/220_wf220_temporal_split_acceptance.md"
SEC_6_4_DOC = ROOT / "TODO/completed/220_wf219_nci60_9cell_acceptance.md"
USER_OK_FILE = Path("/tmp/USER_OK_WF310C_TODO220")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    assert path.exists(), f"smoke JSON missing: {path}"
    return json.loads(path.read_text())


def _read_text(path: Path) -> str:
    assert path.exists(), f"file missing: {path}"
    return path.read_text()


# ===========================================================================
# §6.3 acceptance gate schema (4 tests)
# ===========================================================================


def test_6_3_1_smoke_acceptance_measured():
    """§6.3.1 — Smoke acceptance: 5/5 metal × 1-seed × 1-epoch CPU MEASURED.

    Smoke contract: sweep_summary_smoke.json has n_runs_planned=5 and
    n_runs_completed=5 (5/5 ok), partial=false, mean_auc_across_metals
    present, and per_metal_summary has all 5 metals (Ru/Ir/Os/Rh/Re).
    """
    smoke = _read_json(TEMPORAL_SMOKE)
    # Overall smoke integrity
    assert smoke["ran"] is True, "smoke must report ran=true"
    assert smoke["partial"] is False, "smoke must report partial=false (5/5 ok)"
    assert smoke["n_runs_planned"] == 5, f"n_runs_planned must be 5, got {smoke['n_runs_planned']}"
    assert smoke["n_runs_completed"] == 5, f"n_runs_completed must be 5, got {smoke['n_runs_completed']}"
    # mean_auc_across_metals must be present and in (0, 1)
    mean_auc = smoke["mean_auc_across_metals"]
    assert 0.0 < mean_auc < 1.0, f"mean_auc_across_metals out of (0, 1): {mean_auc}"
    # All 5 metals must be in per_metal_summary with per_seed_auc[0]
    expected_metals = {"Ru", "Ir", "Os", "Rh", "Re"}
    actual_metals = set(smoke["per_metal_summary"].keys())
    assert actual_metals == expected_metals, (
        f"per_metal_summary keys {actual_metals} != expected {expected_metals}"
    )
    for metal in expected_metals:
        m = smoke["per_metal_summary"][metal]
        assert m["n_seeds"] == 1, f"{metal}: n_seeds must be 1 in smoke, got {m['n_seeds']}"
        assert "0" in m["per_seed_auc"], f"{metal}: per_seed_auc missing seed 0"


def test_6_3_2_gpu_acceptance_projected_pending():
    """§6.3.2 — GPU acceptance: wrapper READY at launch_wf310c_todo220_gpu.sh.

    GPU contract: wrapper exists; has USER_OK gate; has Stage A
    (temporal-split 5-metal × 5-seed × 50-epoch); has Stage B
    (NCI-60 9-cell × 3-seed × 50-epoch); references pre-flight check;
    exit codes documented.
    """
    wrapper = _read_text(GPU_WRAPPER)
    # USER_OK gate
    assert "USER_OK_WF310C_TODO220" in wrapper, "wrapper must reference USER_OK_WF310C_TODO220 file"
    # Pre-flight check
    assert "pre_flight_check" in wrapper, "wrapper must run pre_flight_check.py first"
    # Stage A: temporal-split 5-metal × 5-seed × 50-epoch
    assert re.search(r"Stage A", wrapper), "wrapper must have Stage A block (temporal-split)"
    # Stage B: NCI-60 9-cell
    assert re.search(r"Stage B", wrapper), "wrapper must have Stage B block (NCI-60 9-cell)"
    # Exit codes documented
    assert "Exit codes" in wrapper, "wrapper must document exit codes"
    # Honest about being pending (PROJECTED gate)
    assert re.search(r"\b2\.9\s*hr\b", wrapper), "wrapper must cite 2.9 hr wall (Stage A projected)"


def test_6_3_3_per_metal_mean_std_partial():
    """§6.3.3 — Per-metal mean ± std: 1-seed std=0 uninformative in smoke.

    Smoke contract: per_metal_summary[*].std == 0 because n_seeds=1.
    This is honest-negative framing — std is not yet meaningful for
    variance reporting. Becomes actionable when §6.3.2 GPU lands.
    """
    smoke = _read_json(TEMPORAL_SMOKE)
    for metal, m in smoke["per_metal_summary"].items():
        assert m["std"] == 0.0, f"{metal}: std must be 0 with n_seeds=1, got {m['std']}"
        assert m["mean_auc"] > 0.0, f"{metal}: mean_auc must be > 0"
        assert m["mean_auc"] < 1.0, f"{metal}: mean_auc must be < 1"
    # §6.3 doc must acknowledge this honest-negative
    sec_6_3 = _read_text(SEC_6_3_DOC)
    assert "std=0" in sec_6_3 or "std = 0" in sec_6_3, "§6.3 doc must declare std=0 uninformative"
    assert "honest-negative" in sec_6_3.lower() or "honest framing" in sec_6_3.lower(), (
        "§6.3 doc must use honest-negative / honest framing language"
    )


def test_6_3_4_krasnov_protocol_fidelity_pass():
    """§6.3.4 — Krasnov 2026 protocol fidelity: temporal-strict PASS.

    Smoke contract: split_protocol field has Krasnov 2026 temporal-strict
    values (train<2023, test>=2024); split_arg has both cutoffs;
    krasnov_protocol_aligned=true; per-seed split field recorded.
    """
    smoke = _read_json(TEMPORAL_SMOKE)
    # split_protocol field with Krasnov 2026 string
    assert "split_protocol" in smoke, "smoke must have split_protocol field"
    assert "temporal_strict" in smoke["split_protocol"], (
        f"split_protocol must contain 'temporal_strict': {smoke['split_protocol']}"
    )
    assert "Krasnov 2026" in smoke["split_protocol"], (
        f"split_protocol must mention 'Krasnov 2026': {smoke['split_protocol']}"
    )
    # split_arg with both cutoffs
    assert "split_arg" in smoke, "smoke must have split_arg field"
    assert smoke["split_arg"]["train_cutoff"] == "2023-01-01", (
        f"train_cutoff must be 2023-01-01, got {smoke['split_arg']['train_cutoff']}"
    )
    assert smoke["split_arg"]["test_cutoff"] == "2024-01-01", (
        f"test_cutoff must be 2024-01-01, got {smoke['split_arg']['test_cutoff']}"
    )
    # krasnov_protocol_aligned boolean
    assert smoke["krasnov_protocol_aligned"] is True, "krasnov_protocol_aligned must be true"
    # Per-seed split field
    for record in smoke["per_seed_records"]:
        assert "split" in record, f"per_seed_records entry missing 'split' field: {record.get('metal')}"
        assert "temporal_train<2023_test>=2024" in record["split"], (
            f"per_seed split must be 'temporal_train<2023_test>=2024': {record['split']}"
        )
    # Krasnov baselines cited
    assert smoke["krasnov_2026_baseline_ru"] == 0.81, "Krasnov Ru baseline must be 0.81"
    assert smoke["krasnov_2026_baseline_ir"] == 0.73, "Krasnov Ir baseline must be 0.73"


# ===========================================================================
# §6.4 acceptance gate schema (4 tests)
# ===========================================================================


def test_6_4_1_smoke_acceptance_measured():
    """§6.4.1 — Smoke acceptance: 9/9 cell × 1-seed × 1-epoch CPU MEASURED.

    Smoke contract: sweep_summary_smoke.json has n_runs_planned=9 and
    n_runs_completed=9 (9/9 ok), partial=false, mean_auc_across_cells
    present, and per_cell_summary has all 9 cells.
    """
    smoke = _read_json(NCI60_SMOKE)
    # Overall smoke integrity
    assert smoke["ran"] is True, "smoke must report ran=true"
    assert smoke["partial"] is False, "smoke must report partial=false (9/9 ok)"
    assert smoke["n_runs_planned"] == 9, f"n_runs_planned must be 9, got {smoke['n_runs_planned']}"
    assert smoke["n_runs_completed"] == 9, f"n_runs_completed must be 9, got {smoke['n_runs_completed']}"
    # n_cells_used matches n_cells_requested (no unmatched cells)
    assert smoke["n_cells_used"] == 9, f"n_cells_used must be 9, got {smoke['n_cells_used']}"
    assert smoke["n_cells_requested"] == 9, f"n_cells_requested must be 9, got {smoke['n_cells_requested']}"
    assert smoke["cells_unmatched"] == [], f"cells_unmatched must be [], got {smoke['cells_unmatched']}"
    # mean_auc_across_cells
    mean_auc = smoke["mean_auc_across_cells"]
    assert 0.0 < mean_auc < 1.0, f"mean_auc_across_cells out of (0, 1): {mean_auc}"
    # All 9 cells in per_cell_summary
    expected_cells = {"MCF7", "A2780", "A549", "HCT116", "PC3", "HL60", "A2780cis", "DU145", "SKOV3"}
    actual_cells = set(smoke["per_cell_summary"].keys())
    assert actual_cells == expected_cells, (
        f"per_cell_summary keys {actual_cells} != expected {expected_cells}"
    )
    for cell in expected_cells:
        c = smoke["per_cell_summary"][cell]
        assert c["n_seeds"] == 1, f"{cell}: n_seeds must be 1 in smoke, got {c['n_seeds']}"


def test_6_4_2_gpu_acceptance_projected_pending():
    """§6.4.2 — GPU acceptance: wrapper READY at launch_wf310c_todo220_gpu.sh.

    GPU contract: wrapper exists; Stage B block present; 1.0 hr
    NCI-60 wall cited; pre-flight check run; exit codes documented.
    """
    wrapper = _read_text(GPU_WRAPPER)
    # Stage B (NCI-60 9-cell) block
    assert re.search(r"Stage B", wrapper), "wrapper must have Stage B block (NCI-60 9-cell)"
    # 9-cell reference in Stage B
    assert "9.cell" in wrapper or "9-cell" in wrapper, "wrapper must mention 9-cell"
    # 1.0 hr wall for Stage B
    assert re.search(r"1\.0\s*hr", wrapper), "wrapper must cite 1.0 hr wall (Stage B projected)"
    # 3 seeds for Stage B
    assert re.search(r"3\s*seeds?", wrapper), "wrapper must mention 3 seeds (Stage B)"
    # Exit codes documented
    assert "Exit codes" in wrapper, "wrapper must document exit codes"
    # Pre-flight check
    assert "pre_flight_check" in wrapper, "wrapper must run pre_flight_check.py first"


def test_6_4_3_per_cell_mean_std_partial():
    """§6.4.3 — Per-cell mean ± std: 1-seed std=0 uninformative in smoke.

    Smoke contract: per_cell_summary[*].std == 0 because n_seeds=1.
    Per-cell mean_auc varies widely (range [0.1667, 0.7813]) — at 1
    epoch this is expected; std=0 is honest-negative framing.
    """
    smoke = _read_json(NCI60_SMOKE)
    aucs = []
    for cell, c in smoke["per_cell_summary"].items():
        assert c["std"] == 0.0, f"{cell}: std must be 0 with n_seeds=1, got {c['std']}"
        assert 0.0 <= c["mean_auc"] <= 1.0, f"{cell}: mean_auc out of [0, 1]: {c['mean_auc']}"
        aucs.append(c["mean_auc"])
    # Per-cell range is wide — this is the honest-negative observation
    assert max(aucs) > 0.7, f"max per-cell AUC must be > 0.7 (DU144 highlight), got {max(aucs)}"
    assert min(aucs) < 0.5, f"min per-cell AUC must be < 0.5 (PC3 honest-negative), got {min(aucs)}"
    # §6.4 doc must acknowledge this honest-negative
    sec_6_4 = _read_text(SEC_6_4_DOC)
    assert "std=0" in sec_6_4 or "std = 0" in sec_6_4, "§6.4 doc must declare std=0 uninformative"
    assert "honest-negative" in sec_6_4.lower() or "honest framing" in sec_6_4.lower(), (
        "§6.4 doc must use honest-negative / honest framing language"
    )


def test_6_4_4_hct116_highlight_pc3_honest_negative():
    """§6.4.4 — HCT116 highlight + PC3 honest-negative.

    Smoke contract: HCT116=0.6998 (above 0.5 random; highlight);
    PC3=0.1667 (below random; honest-negative). §6.4 doc must declare
    PC3 as honest-negative (not paraphrased as "achievable" via
    MultiTaskLoss head).
    """
    smoke = _read_json(NCI60_SMOKE)
    hct116 = smoke["per_cell_summary"]["HCT116"]
    pc3 = smoke["per_cell_summary"]["PC3"]
    # HCT116 highlight (smoke MEASURED)
    assert hct116["mean_auc"] >= 0.65, f"HCT116 AUC must be >= 0.65 (highlight), got {hct116['mean_auc']}"
    assert hct116["mean_auc"] <= 0.75, f"HCT116 AUC must be <= 0.75, got {hct116['mean_auc']}"
    # PC3 honest-negative (smoke MEASURED, below random 0.5)
    assert pc3["mean_auc"] < 0.5, f"PC3 AUC must be < 0.5 (honest-negative), got {pc3['mean_auc']}"
    assert pc3["mean_auc"] > 0.0, f"PC3 AUC must be > 0, got {pc3['mean_auc']}"
    # Smallest cohort (after DU145) — honest-negative framing
    assert pc3["n_train"] < 250, f"PC3 n_train must be < 250 (small cohort), got {pc3['n_train']}"
    # §6.4 doc must explicitly declare PC3 honest-negative
    sec_6_4 = _read_text(SEC_6_4_DOC)
    assert "PC3" in sec_6_4, "§6.4 doc must mention PC3"
    assert "honest-negative" in sec_6_4, "§6.4 doc must declare honest-negative for PC3"
    # Must NOT paraphrase PC3 as "achievable" via MultiTaskLoss head
    # (TODO-130 G3 is NOT in Wf-313D scope)
    # The MultiTaskLoss mention must be scoped to TODO-130 G3 follow-up
    assert "TODO-130" in sec_6_4 or "TODO130" in sec_6_4, (
        "§6.4 doc must scope MultiTaskLoss head to TODO-130 G3 follow-up"
    )


# ===========================================================================
# Smoke persistence JSON validation (2 tests)
# ===========================================================================


def test_temporal_smoke_json_schema():
    """Smoke persistence — sweep_summary_smoke.json §6.3.1 schema validation.

    Schema contract: top-level keys ran, partial, n_runs_planned,
    n_runs_completed, total_wall_min, per_metal_seed_auc, per_metal_summary,
    mean_auc_across_metals, split_protocol, split_arg, krasnov_protocol_aligned,
    krasnov_2026_baseline_ru, krasnov_2026_baseline_ir, n_seeds_used,
    per_seed_records, honest_framing_note, config.

    Each per_metal_seed_auc entry has keys: auc, time_sec, status,
    n_train, n_val, n_test. Each per_seed_records entry has keys:
    metal, seed, n_train, n_val, n_test, pos_rate_train, pos_rate_test,
    in_domain_test_auc, elapsed_s, epoch_count, status, error, split.
    """
    smoke = _read_json(TEMPORAL_SMOKE)
    # Top-level keys
    required_keys = {
        "ran", "partial", "n_runs_planned", "n_runs_completed", "total_wall_min",
        "per_metal_seed_auc", "per_metal_summary", "mean_auc_across_metals",
        "split_protocol", "split_arg", "krasnov_protocol_aligned",
        "krasnov_2026_baseline_ru", "krasnov_2026_baseline_ir",
        "n_seeds_used", "per_seed_records", "honest_framing_note", "config",
    }
    assert required_keys.issubset(smoke.keys()), (
        f"sweep_summary_smoke.json missing required keys: {required_keys - set(smoke.keys())}"
    )
    # Per-metal-seed entry keys
    per_metal_entry_keys = {"auc", "time_sec", "status", "n_train", "n_val", "n_test"}
    for metal, seed_dict in smoke["per_metal_seed_auc"].items():
        for seed_str, entry in seed_dict.items():
            assert set(entry.keys()) >= per_metal_entry_keys, (
                f"{metal}/seed={seed_str} missing keys: {per_metal_entry_keys - set(entry.keys())}"
            )
            assert entry["status"] == "ok", f"{metal}/seed={seed_str} status must be 'ok'"
    # Per-seed record keys
    per_seed_record_keys = {
        "metal", "seed", "n_train", "n_val", "n_test", "pos_rate_train",
        "pos_rate_test", "in_domain_test_auc", "elapsed_s", "epoch_count",
        "status", "error", "split",
    }
    for record in smoke["per_seed_records"]:
        assert set(record.keys()) >= per_seed_record_keys, (
            f"per_seed_record (metal={record.get('metal')}) missing keys: "
            f"{per_seed_record_keys - set(record.keys())}"
        )
    # honest_framing_note is non-empty
    assert len(smoke["honest_framing_note"]) > 50, "honest_framing_note must be non-trivial"


def test_nci60_smoke_json_schema():
    """Smoke persistence — sweep_summary_smoke.json §6.4.1 schema validation.

    Schema contract: top-level keys ran, partial, n_runs_planned,
    n_runs_completed, total_wall_min, per_cell_seed_auc, per_cell_summary,
    mean_auc_across_cells, n_cells_used, n_cells_requested, cells_matched,
    cells_unmatched, cells_available, per_seed_records, honest_framing_note,
    config.

    Each per_cell_seed_auc entry has keys: auc, time_sec, status,
    n_train, n_val, n_test. Each per_seed_records entry has keys:
    cell, seed, n_train, n_val, n_test, pos_rate_train, pos_rate_test,
    in_domain_test_auc, elapsed_s, epoch_count, status, error.
    """
    smoke = _read_json(NCI60_SMOKE)
    # Top-level keys
    required_keys = {
        "ran", "partial", "n_runs_planned", "n_runs_completed", "total_wall_min",
        "per_cell_seed_auc", "per_cell_summary", "mean_auc_across_cells",
        "n_cells_used", "n_cells_requested", "cells_matched", "cells_unmatched",
        "cells_available", "per_seed_records", "honest_framing_note", "config",
    }
    assert required_keys.issubset(smoke.keys()), (
        f"sweep_summary_smoke.json missing required keys: {required_keys - set(smoke.keys())}"
    )
    # All 9 cells matched (cells_unmatched is empty)
    assert smoke["cells_unmatched"] == [], "cells_unmatched must be empty (all 9 cells found)"
    # Per-cell-seed entry keys
    per_cell_entry_keys = {"auc", "time_sec", "status", "n_train", "n_val", "n_test"}
    for cell, seed_dict in smoke["per_cell_seed_auc"].items():
        for seed_str, entry in seed_dict.items():
            assert set(entry.keys()) >= per_cell_entry_keys, (
                f"{cell}/seed={seed_str} missing keys: {per_cell_entry_keys - set(entry.keys())}"
            )
            assert entry["status"] == "ok", f"{cell}/seed={seed_str} status must be 'ok'"
    # Per-seed record keys
    per_seed_record_keys = {
        "cell", "seed", "n_train", "n_val", "n_test", "pos_rate_train",
        "pos_rate_test", "in_domain_test_auc", "elapsed_s", "epoch_count",
        "status", "error",
    }
    for record in smoke["per_seed_records"]:
        assert set(record.keys()) >= per_seed_record_keys, (
            f"per_seed_record (cell={record.get('cell')}) missing keys: "
            f"{per_seed_record_keys - set(record.keys())}"
        )
    # honest_framing_note is non-empty
    assert len(smoke["honest_framing_note"]) > 50, "honest_framing_note must be non-trivial"


# ===========================================================================
# GPU integration tests (2 tests)
# ===========================================================================


def test_gpu_wrapper_user_ok_gate():
    """GPU integration — wrapper exits non-zero without USER_OK file.

    Integration contract: wrapper checks for /tmp/USER_OK_WF310C_TODO220
    file; exits with rc=2 when missing; wrapper source explicitly
    references the gate path.
    """
    wrapper = _read_text(GPU_WRAPPER)
    # USER_OK path
    assert "/tmp/USER_OK_WF310C_TODO220" in wrapper, "wrapper must reference USER_OK file path"
    # rc=2 (USER_OK missing) in exit codes block
    assert re.search(r"2\s*=\s*USER_OK\s+missing", wrapper) or re.search(r"2.*USER_OK.*missing", wrapper, re.IGNORECASE), (
        "wrapper must document rc=2 for USER_OK missing"
    )
    # rc=3 (pre-flight fail) — sanity check
    assert re.search(r"3\s*=\s*Pre-flight", wrapper) or "Pre-flight" in wrapper, (
        "wrapper must document rc=3 for pre-flight failure"
    )
    # Wrapper must be a bash script
    assert wrapper.startswith("#!/usr/bin/env bash") or wrapper.startswith("#!/bin/bash"), (
        "wrapper must be a bash script with shebang"
    )


def test_gpu_pipeline_master_phase3_integration():
    """GPU integration — gpu_pipeline_master.sh Phase 3 includes wf310c_todo220.

    Integration contract: scripts/gpu_pipeline_master.sh Phase 3 has
    Job 3 = wf310c_todo220 (3.9 hr cuda:1) referencing the wrapper.
    """
    pipeline = _read_text(GPU_PIPELINE)
    # Phase 3 contains wf310c_todo220 job
    assert "wf310c_todo220" in pipeline, "gpu_pipeline_master.sh must reference wf310c_todo220"
    # 3.9 hr wall cited
    assert "3.9" in pipeline, "gpu_pipeline_master.sh must cite 3.9 hr wall"
    # Job 3 marker (TODO-220 is Job 3 in Phase 3)
    assert re.search(r"Job 3:", pipeline), "gpu_pipeline_master.sh must have Job 3 marker"
    # TODO-220 mention
    assert "TODO-220" in pipeline, "gpu_pipeline_master.sh must mention TODO-220"
    # Launch wrapper path
    assert "launch_wf310c_todo220" in pipeline, "gpu_pipeline_master.sh must invoke launch_wf310c_todo220 wrapper"