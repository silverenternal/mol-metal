#!/usr/bin/env python3
# Copyright (c) 2026 Mol-Metal authors. Licensed under the project's license.
"""
Wf-433 m5 REAL validation tests (CPU-only).

Verifies the canonical per-metric audit script emits ALL_MEASURED for the
5 user-voice MOCK items per AGENTS.md §5.2 honest re-framing. Per
Wf-433 brief, all 5 items are ALREADY MEASURED — this test suite
enforces that contract and validates each metric JSON loads + has a
MEASURED value + provenance chain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make repo root importable when invoked via pytest from project root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.scripts.wf433_m5_real_validation import (  # noqa: E402
    AUDIT_DIR,
    METRIC_FILE_MAP,
    OUT_DIR,
    probe_metric,
)


def test_m3_vina_measured():
    """m3 Vina cisplatin mean kcal/mol is MEASURED per Wf-121 P3."""
    probe = probe_metric("m3_vina", METRIC_FILE_MAP["m3_vina"])
    assert probe.load_ok, f"m3 Vina JSON failed to load: {probe.metric_path}"
    assert probe.provenance_ok, (
        f"m3 Vina probe lacks provenance: top_value={probe.metric_json_top_value}, "
        f"n_measurements={probe.n_measurements}"
    )
    assert probe.metric_json_top_value is not None
    assert probe.metric_json_top_value < 0.0, (
        f"m3 Vina top value should be negative kcal/mol (lower=better binding); "
        f"got {probe.metric_json_top_value}"
    )
    # Wf-121 P3 cisplatin mean: -3.103; Wf-141 v2 ex32: -3.1575;
    # Wf-143 chemotype: -3.1552. Aggregate extractor returns cisplatin_mean_kcal_per_mol.
    assert -3.5 < probe.metric_json_top_value < -2.5, (
        f"m3 Vina cisplatin mean expected in [-3.5, -2.5] kcal/mol band; "
        f"got {probe.metric_json_top_value}"
    )
    assert (
        probe.n_measurements >= 5
    ), f"m3 Vina expected >=5 measurement records; got {probe.n_measurements}"
    assert probe.last_measurement_status.startswith(
        "MEASURED"
    ), f"m3 Vina last status expected MEASURED_REAL_VINA; got {probe.last_measurement_status}"


def test_m15_rmsd_measured():
    """m15 RMSD median across pockets is MEASURED per Wf-246-B redesign."""
    probe = probe_metric("m15_rmsd", METRIC_FILE_MAP["m15_rmsd"])
    assert probe.load_ok
    assert probe.provenance_ok
    assert probe.metric_json_top_value is not None
    # Median RMSD across 5 pockets; SOTA gate is <= 2.0 Å (Trott & Olson 2010).
    assert probe.metric_json_top_value <= 2.0, (
        f"m15 RMSD median should be <= 2.0 Å SOTA gate; "
        f"got {probe.metric_json_top_value}"
    )
    assert (
        probe.metric_json_top_value < 1.0
    ), f"m15 RMSD median expected <1.0 Å (high fidelity); got {probe.metric_json_top_value}"
    # Per Wf-246-B: median 0.6578 Å; tolerance ±0.1.
    assert 0.55 < probe.metric_json_top_value < 0.75, (
        f"m15 RMSD median expected in [0.55, 0.75] Å band; "
        f"got {probe.metric_json_top_value}"
    )
    assert probe.n_measurements == 5, (
        f"m15 RMSD expected exactly 5 measurement records (5 pockets); "
        f"got {probe.n_measurements}"
    )
    assert probe.last_measurement_status == "MEASURED"


def test_m14_cfm_real_measured():
    """m14 CFM decode_ratio on NCI-60 + DrugOOD is MEASURED per Wf-148."""
    probe = probe_metric("m14_cfm_real", METRIC_FILE_MAP["m14_cfm_real"])
    assert probe.load_ok
    assert probe.provenance_ok
    assert probe.metric_json_top_value is not None
    # Real CFM decode_ratio PASSES both internal (>=0.8) + lit TargetDiff (>=0.5).
    assert probe.metric_json_top_value >= 0.5, (
        f"m14 CFM real decode_ratio must be >=0.5 (lit TargetDiff target); "
        f"got {probe.metric_json_top_value}"
    )
    assert probe.metric_json_top_value >= 0.8, (
        f"m14 CFM real decode_ratio must be >=0.8 (internal target); "
        f"got {probe.metric_json_top_value}"
    )
    # Wf-148 MEASURED 0.9980; tolerance ±0.05.
    assert 0.93 < probe.metric_json_top_value <= 1.0, (
        f"m14 CFM real decode_ratio expected in [0.93, 1.0] band; "
        f"got {probe.metric_json_top_value}"
    )
    assert probe.last_measurement_status == "MEASURED_PASS"


def test_m17_drugood_measured():
    """m17 DrugOOD OOD AUC is MEASURED per Wf-310 + Wf-300 + Wf-301."""
    probe = probe_metric("m17_drugood", METRIC_FILE_MAP["m17_drugood"])
    assert probe.load_ok
    assert probe.provenance_ok
    assert probe.metric_json_top_value is not None
    # OOD AUC; iMoLD reference assay 0.7211; honest framing per Wf-433 brief.
    assert probe.metric_json_top_value >= 0.5, (
        f"m17 DrugOOD OOD AUC must be > 0.5 random chance; "
        f"got {probe.metric_json_top_value}"
    )
    # Wf-310 metacytotox_fallback assay 0.891; tolerance ±0.2.
    assert probe.metric_json_top_value > 0.6, (
        f"m17 DrugOOD OOD AUC expected > 0.6 honest partial; "
        f"got {probe.metric_json_top_value}"
    )
    assert probe.n_measurements >= 9, (
        f"m17 DrugOOD expected >=9 measurement records (3 splits x 3 seeds); "
        f"got {probe.n_measurements}"
    )
    assert probe.last_measurement_status == "MEASURED"


def test_m21_nci60_measured():
    """m21 NCI-60 9-cell precious-metal AUC is MEASURED per Wf-310."""
    probe = probe_metric("m21_nci60", METRIC_FILE_MAP["m21_nci60"])
    assert probe.load_ok
    assert probe.provenance_ok
    assert probe.metric_json_top_value is not None
    # Overall mean AUC across 9 cell lines; Krasnov 2026 baseline 0.81.
    assert probe.metric_json_top_value >= 0.5, (
        f"m21 NCI-60 AUC must be > 0.5 random chance; "
        f"got {probe.metric_json_top_value}"
    )
    # Wf-310 MEASURED 0.8007; tolerance ±0.05.
    assert 0.75 < probe.metric_json_top_value < 0.85, (
        f"m21 NCI-60 AUC expected in [0.75, 0.85] band per Wf-310; "
        f"got {probe.metric_json_top_value}"
    )
    assert probe.n_measurements >= 9, (
        f"m21 NCI-60 expected >=9 measurement records (3 seeds x 8 valid cells; "
        f"DU145 null across all 3 seeds); got {probe.n_measurements}"
    )
    assert probe.last_measurement_status == "MEASURED"


def test_audit_json_loads_and_validates():
    """Audit JSON loads + reports ALL_MEASURED + 5 metrics probed."""
    audit_path = AUDIT_DIR / "m5_real_per_metric.json"
    assert audit_path.exists(), f"Audit JSON missing: {audit_path}"
    with open(audit_path, "r") as f:
        payload = json.load(f)
    assert payload["wf"] == "Wf-433"
    assert payload["n_metrics_probed"] == 5
    assert payload["n_metrics_measured"] == 5
    assert payload["n_metrics_load_failed"] == 0
    assert payload["aggregate_status"] == "ALL_MEASURED"
    assert len(payload["per_metric"]) == 5
    # Each metric entry has required provenance fields.
    for entry in payload["per_metric"]:
        for required in (
            "metric_id",
            "metric_path",
            "metric_json_top_value",
            "n_measurements",
            "last_measurement_status",
            "provenance_ok",
            "load_ok",
            "honest_negative_note",
        ):
            assert (
                required in entry
            ), f"Metric entry missing required field {required!r}: {entry['metric_id']}"


def test_wf433_report_result_loads():
    """Wf-433 report result.json loads + matches canonical shape."""
    report_path = OUT_DIR / "result.json"
    assert report_path.exists(), f"Wf-433 report result.json missing: {report_path}"
    with open(report_path, "r") as f:
        payload = json.load(f)
    assert payload["wf"] == "Wf-433"
    assert payload["n_metrics_probed"] == 5
    assert payload["aggregate_status"] == "ALL_MEASURED"
    assert "honest_framing" in payload
    # User voice acknowledgement present.
    assert "user_voice_acknowledgement" in payload["honest_framing"] or (
        "5 MOCK" in str(payload["honest_framing"])
    )
