# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wf-R43 m16 — tests for the D-MPNN pIC50 Pt ckpt-gap guard.

CPU-only; no trainer is invoked and nothing is trained.  8 tests:

  1. the on-disk inventory really is missing Pt (the m16 gap premise)
  2. auditing the real inventory FAILs and names Pt as missing
  3. the FAIL evidence cites the SYNTHETIC-label blocker
  4. a synthetic-framed inventory that covers all 6 metals PASSes
  5. an absent inventory SKIPs rather than raising
  6. malformed JSON FAILs loudly instead of crashing the gate
  7. the flat ``checkpoints``-list JSON shape is also understood
  8. the gate is registered on "all" and SKIPs by default (opt-in)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.wf_r43a1_m16_pt_ckpt_gap import (  # noqa: E402
    EXPECTED_METALS,
    METRIC_KEY,
    OPT_IN_ENV,
    audit_m16,
)

REAL_INVENTORY = REPO_ROOT / "metrics" / "by_metric" / "dmpnn_ckpt_status.json"


def test_m16_gap_premise_pt_absent_from_on_disk_inventory():
    """The m16 item is only real if Pt is genuinely missing today."""
    assert REAL_INVENTORY.is_file(), "ckpt inventory absent — re-verify premise"
    payload = json.loads(REAL_INVENTORY.read_text(encoding="utf-8"))
    inventoried = set(payload.get("per_metal_summary") or {})
    assert "Pt" in EXPECTED_METALS
    assert "Pt" not in inventoried, (
        "Pt now appears in the inventory — m16 may have been closed; "
        "re-run the audit before assuming the gap is still open"
    )


def test_audit_real_inventory_fails_and_names_pt():
    result = audit_m16(REAL_INVENTORY, repo_root=REPO_ROOT)
    assert result["status"] == "FAIL"
    assert "Pt" in result["metals_missing"]
    assert result["n_checkpoints"] == 25
    assert METRIC_KEY in result["evidence"]


def test_fail_evidence_cites_synthetic_label_blocker():
    """The FAIL must say WHY Pt cannot simply be trained."""
    result = audit_m16(REAL_INVENTORY, repo_root=REPO_ROOT)
    ev = result["evidence"]
    assert "SYNTHETIC" in ev
    assert "platinai_4cell.meta.json" in ev
    assert result["real_pt_cohorts"] == [], (
        "a REAL Pt cohort appeared on disk — the m16 blocker has changed "
        "and this guard's evidence needs re-authoring"
    )


def test_full_coverage_inventory_passes(tmp_path):
    payload = {
        "metric": "dmpnn_ckpt_status",
        "n_checkpoints": 30,
        "per_metal_summary": {m: {"n_ckpts": 5} for m in EXPECTED_METALS},
    }
    inv = tmp_path / "inv.json"
    inv.write_text(json.dumps(payload), encoding="utf-8")
    result = audit_m16(inv, repo_root=REPO_ROOT)
    assert result["status"] == "PASS"
    assert result["metals_missing"] == []


def test_absent_inventory_skips(tmp_path):
    result = audit_m16(tmp_path / "nope.json", repo_root=REPO_ROOT)
    assert result["status"] == "SKIP"


def test_malformed_json_fails_loudly(tmp_path):
    inv = tmp_path / "bad.json"
    inv.write_text("{not json", encoding="utf-8")
    result = audit_m16(inv, repo_root=REPO_ROOT)
    assert result["status"] == "FAIL"
    assert "unreadable" in result["evidence"]


def test_flat_checkpoints_shape_is_understood(tmp_path):
    """The Wf-109 JSON shape (list, not per_metal_summary) still parses."""
    payload = {
        "metric": "dmpnn_ckpt_status",
        "n_checkpoints": 2,
        "checkpoints": [
            {"filename": "dmpnn_pic50_ru_seed0.pt", "metal": "Ru", "seed": 0},
            {"filename": "dmpnn_pic50_pt_seed0.pt", "metal": "Pt", "seed": 0},
        ],
    }
    inv = tmp_path / "flat.json"
    inv.write_text(json.dumps(payload), encoding="utf-8")
    result = audit_m16(
        inv,
        repo_root=REPO_ROOT,
        expected_metals=("Pt", "Ru"),
    )
    assert result["metals_inventoried"] == ["Pt", "Ru"]
    assert result["status"] == "PASS"


def test_gate_registered_on_all_and_skips_by_default(monkeypatch):
    monkeypatch.delenv(OPT_IN_ENV, raising=False)
    pre = pytest.importorskip("molmetal.scripts.pre_flight_check")
    gates = dict(pre.build_gates("all"))
    name = "p3_wf_r43_m16_pt_ckpt_gap"
    assert name in gates, "m16 gate is not registered on the 'all' pipeline"
    assert len(gates) == pre.EXPECTED_GATE_COUNTS["all"], (
        f"gate-count drift: build_gates('all')={len(gates)} vs "
        f"EXPECTED_GATE_COUNTS['all']={pre.EXPECTED_GATE_COUNTS['all']}"
    )
    result = gates[name]()
    assert result.status == "SKIP"
    assert OPT_IN_ENV in result.evidence
