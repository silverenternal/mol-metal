"""Wf-377-A4: GPU launch dashboard final state verification.

Per Wf-360-A3 + Wf-363-B1 + Wf-363-B2 + Wf-377-A4:
  - Dashboard JSON exists at molmetal/reports/gpu_launch_dashboard.json
  - Has per_pair_progress for each launcher
  - Has budget_status field
  - Pre-flight state recorded correctly (89/2/3/94 at Wf-377-A4 timestamp)
  - Zotero + tectonic install status present

Per AGENTS.md §5.1 + kill_then_adapt §0:
  - Dashboard is READ-only aggregator; never launches GPU job
  - Activation remains USER_OK-gated
"""

import json
import os
import pytest

DASHBOARD_PATH = "molmetal/reports/gpu_launch_dashboard.json"
PREFLIGHT_TXT = "molmetal/reports/wf377a4_preflight.txt"


def _load_dashboard():
    """Load the dashboard JSON; skip if file missing."""
    if not os.path.exists(DASHBOARD_PATH):
        pytest.skip(f"{DASHBOARD_PATH} not present (dashboard not yet generated)")
    with open(DASHBOARD_PATH, "r") as f:
        return json.load(f)


class TestWf377A4DashboardFinal:
    """Verify GPU launch dashboard final state per Wf-377-A4 refresh."""

    def test_dashboard_json_exists_with_per_pair_progress(self):
        """Test 1: dashboard JSON exists + has per_pair_progress per launcher."""
        d = _load_dashboard()
        assert "launchers" in d, "dashboard missing launchers"
        assert isinstance(d["launchers"], list)
        assert len(d["launchers"]) == d.get("launcher_count", len(d["launchers"]))
        for launcher in d["launchers"]:
            assert "pair_progress" in launcher, (
                f"launcher {launcher.get('name')} missing pair_progress"
            )
            pp = launcher["pair_progress"]
            assert "total_pairs" in pp
            assert "discovered_pairs" in pp
            assert "ok" in pp
            assert "failed" in pp
            assert "running" in pp
            assert "pending" in pp

    def test_budget_status_field_present(self):
        """Test 2: budget_status field present with expected subfields."""
        d = _load_dashboard()
        assert "budget_status" in d, "dashboard missing budget_status"
        bs = d["budget_status"]
        assert bs.get("session_cap_hr") == 10.0, (
            f"session_cap_hr should be 10.0 (Wf-262 §7.1), got {bs.get('session_cap_hr')}"
        )
        assert "per_launcher" in bs
        assert isinstance(bs["per_launcher"], list)
        assert len(bs["per_launcher"]) == len(d["launchers"])
        assert "total_projected_hr" in bs
        assert "total_remaining_hr" in bs
        assert "honest_framing" in bs
        # Each per_launcher entry must have budget fields
        for entry in bs["per_launcher"]:
            assert "launcher" in entry
            assert "budget_hr" in entry
            assert "remaining_hr" in entry
            assert "status" in entry
            assert "launcher_status" in entry

    def test_preflight_state_recorded_correctly(self):
        """Test 3: pre-flight state 89/2/3/94 recorded correctly per Wf-377-A4."""
        d = _load_dashboard()
        assert "preflight_state" in d, (
            "dashboard missing preflight_state (Wf-377-A4 refresh)"
        )
        ps = d["preflight_state"]
        assert ps.get("pass") == 89, f"expected 89 PASS, got {ps.get('pass')}"
        assert ps.get("fail") == 2, f"expected 2 FAIL, got {ps.get('fail')}"
        assert ps.get("skip") == 3, f"expected 3 SKIP, got {ps.get('skip')}"
        assert ps.get("total") == 94, f"expected 94 TOTAL, got {ps.get('total')}"
        assert ps.get("expected_gate_counts_all") == 94
        assert ps.get("wall_sec") is not None
        assert ps.get("wall_sec") < 300, (
            f"wall_sec too high ({ps.get('wall_sec')}); indicates pre-flight incomplete"
        )
        per_pipeline = ps.get("per_pipeline", {})
        assert per_pipeline.get("all") == 94
        assert per_pipeline.get("chemprop_v2_v3") == 14
        assert per_pipeline.get("cfm") == 6
        assert per_pipeline.get("frontier") == 3

    def test_install_state_zotero_tectonic(self):
        """Test 4 (bonus): Zotero + tectonic install state recorded per Wf-372."""
        d = _load_dashboard()
        assert "install_state" in d, (
            "dashboard missing install_state (Wf-372 Zotero+tectonic)"
        )
        inst = d["install_state"]
        assert inst.get("zotero", {}).get("installed") is True
        assert inst.get("tectonic", {}).get("installed") is True
        assert "140" in inst.get("zotero", {}).get("version", "")
        assert "0.17" in inst.get("tectonic", {}).get("version", "")

    def test_dashboard_activation_gated_user_ok_required(self):
        """Test 5 (bonus): dashboard activation gated + USER_OK required."""
        d = _load_dashboard()
        assert d.get("activation_gated") is True
        assert d.get("user_ok_required") is True
        assert d.get("dashboard_mode") == "READ-ONLY_SCAFFOLD"
        assert d.get("activation_flag_set") is False, (
            "activation_flag_set must remain False — no GPU launch triggered"
        )