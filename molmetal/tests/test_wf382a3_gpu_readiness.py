"""Wf-382-A3 regression-guard test — GPU-blocked TODO items + USER_OK readiness audit.

This test verifies that the readiness checklist from Wf-382-A3 remains
intact for the moment when USER_OK arrives. It does NOT itself launch
any GPU job — it is a CPU-only READ + audit invariant check.

Per `party-twenty-spirit §3.5` + `AGENTS.md §5.1` + `mol-metal-gpu-execution §0`:
- All launch wrappers MUST exist on disk
- Each wrapper MUST enforce USER_OK gate pattern (per Wf-294B + Wf-355 + Wf-366)
- Each wrapper MUST wire pre_flight_check.py (per Wf-347-B1 + Wf-347-B2)
- Per-wave + data-quality gates (wf343c2 + wf343c3) MUST exist (per Wf-356-A1)
- Pre-flight recent report MUST exist with PASS count >= 90
- 3 GPU-blocked TODO items MUST be on disk and audited
- USER_OK_FINAL_CHECKLIST MUST exist with >= 400 LOC

Total: 1 NEW test class × 8 invariants ≈ 10-15 test cases in <1 s CPU-only.
"""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DOCS_DIR = REPO_ROOT / "docs"
TODO_PENDING_DIR = REPO_ROOT / "TODO" / "pending"
TODO_COMPLETED_DIR = REPO_ROOT / "TODO" / "completed"
REPORTS_DIR = REPO_ROOT / "molmetal" / "reports"


# The 7 curated USER_OK-gated launch wrappers per `docs/USER_OK_FINAL_CHECKLIST.md` §1.1
LAUNCH_WRAPPERS = [
    "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh",  # Wf-355-A1
    "launch_wf310e_r20_p4_bindingdb_1.07M.sh",  # Wf-355-A2
    "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh",  # Wf-355-A3 (TODO-376)
    "launch_wf131_p3_deferred_runs.sh",  # Wf-355-B1
    "launch_wf146_deferred_runs.sh",  # Wf-355-B2
    "launch_wf124_p3_g5.sh",  # Wf-366-A1 (TODO-181 G5)
    "recover_g6_failed_tuples.sh",  # Wf-366-A2 (TODO-182 G6)
]


# The 3 GPU-blocked TODO items per Wf-382-A3 §1
GPU_BLOCKED_TODO_ITEMS = [
    "179_wf123_p3_w2_pic50_alignment.md",  # STATUS_HOLD arch infeasible
    "182_wf124_p3_per_innovation_ablation.md",  # G6 launch wrapper READY
    "376a3_m2_r2_gpu_retrain_plan.md",  # Path A READY, Path B WAITING
]


class TestWf382A3GpuReadiness(unittest.TestCase):
    """Wf-382-A3 readiness audit regression-guard tests."""

    def test_all_7_launch_wrappers_exist(self):
        """All 7 USER_OK-gated launch wrappers MUST exist on disk."""
        for wrapper in LAUNCH_WRAPPERS:
            path = SCRIPTS_DIR / wrapper
            self.assertTrue(
                path.exists(),
                f"Launch wrapper missing: {wrapper} (per docs/USER_OK_FINAL_CHECKLIST.md §1.1)",
            )
            self.assertGreater(
                path.stat().st_size,
                5000,
                f"Launch wrapper too small (likely truncated): {wrapper} < 5000 bytes",
            )

    def test_each_wrapper_has_user_ok_gate(self):
        """Each wrapper MUST enforce the USER_OK gate pattern (Wf-294B + Wf-355 + Wf-366)."""
        for wrapper in LAUNCH_WRAPPERS:
            path = SCRIPTS_DIR / wrapper
            content = path.read_text(encoding="utf-8", errors="replace")
            # Search for the canonical USER_OK gate pattern variants:
            # - "USER_OK" / "USER_OK_GATE" in comments or env-var checks
            # - "[ -z \"${USER_OK:-}\" ]" or "[ \"${USER_OK:-0}\" != \"1\" ]"
            has_user_ok_token = "USER_OK" in content
            has_gate_check = (
                "USER_OK:-" in content
                or "USER_OK_GATE" in content
                or "USER_OK=YES" in content
            )
            self.assertTrue(
                has_user_ok_token and has_gate_check,
                f"USER_OK gate pattern missing in {wrapper} (need 'USER_OK' token + env-var check)",
            )

    def test_each_wrapper_wires_pre_flight_check(self):
        """Each wrapper MUST wire pre_flight_check.py (per Wf-347-B1 + Wf-347-B2)."""
        for wrapper in LAUNCH_WRAPPERS:
            path = SCRIPTS_DIR / wrapper
            content = path.read_text(encoding="utf-8", errors="replace")
            self.assertIn(
                "pre_flight_check",
                content,
                f"pre_flight_check.py wiring missing in {wrapper}",
            )

    def test_wf343c2_per_wave_check_script_exists(self):
        """Per Wf-347-B1: wf343c2_per_wave_check.py MUST exist."""
        path = REPO_ROOT / "molmetal" / "scripts" / "wf343c2_per_wave_check.py"
        self.assertTrue(
            path.exists(),
            "wf343c2_per_wave_check.py missing (per Wf-347-B1)",
        )

    def test_wf343c3_data_quality_check_script_exists(self):
        """Per Wf-347-B2 + Wf-356-A1: wf343c3_data_quality_check.py MUST exist."""
        path = REPO_ROOT / "molmetal" / "scripts" / "wf343c3_data_quality_check.py"
        self.assertTrue(
            path.exists(),
            "wf343c3_data_quality_check.py missing (per Wf-347-B2 + Wf-356-A1)",
        )

    def test_pre_flight_recent_report_preserved(self):
        """Pre-flight v3 92/0/2/94 baseline MUST be preserved (per Wf-368-C1)."""
        path = REPORTS_DIR / "wf368c1_preflight.txt"
        self.assertTrue(
            path.exists(),
            "wf368c1_preflight.txt missing — pre-flight cascade FULLY CLOSED state NOT preserved",
        )
        content = path.read_text(encoding="utf-8", errors="replace")
        # Verify PASS count is at least 90 (per Wf-368-C1 92/0/2/94 baseline)
        pass_match = re.search(r'"pass":\s*(\d+)', content)
        self.assertIsNotNone(
            pass_match,
            "wf368c1_preflight.txt does not contain JSON 'pass' count",
        )
        pass_count = int(pass_match.group(1))
        self.assertGreaterEqual(
            pass_count,
            90,
            f"Pre-flight PASS count {pass_count} below 90 (expected 92 per Wf-368-C1 baseline)",
        )

    def test_all_3_gpu_blocked_todo_items_exist(self):
        """All 3 GPU-blocked TODO items MUST exist on disk (per Wf-382-A3 §1)."""
        for todo_item in GPU_BLOCKED_TODO_ITEMS:
            path = TODO_PENDING_DIR / todo_item
            self.assertTrue(
                path.exists(),
                f"GPU-blocked TODO item missing: {todo_item}",
            )

    def test_user_ok_final_checklist_exists(self):
        """docs/USER_OK_FINAL_CHECKLIST.md MUST exist with >= 400 LOC (per Wf-366-C2)."""
        path = DOCS_DIR / "USER_OK_FINAL_CHECKLIST.md"
        self.assertTrue(
            path.exists(),
            "docs/USER_OK_FINAL_CHECKLIST.md missing (per Wf-366-C2 consolidation)",
        )
        line_count = sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
        self.assertGreaterEqual(
            line_count,
            400,
            f"docs/USER_OK_FINAL_CHECKLIST.md too short: {line_count} lines < 400 (expected ~433 per Wf-366-C2)",
        )

    def test_wf382a3_audit_closeout_exists(self):
        """TODO/completed/382a3_gpu_readiness_audit.md MUST exist (this wave)."""
        path = TODO_COMPLETED_DIR / "382a3_gpu_readiness_audit.md"
        self.assertTrue(
            path.exists(),
            "TODO/completed/382a3_gpu_readiness_audit.md missing (Wf-382-A3 closeout)",
        )
        line_count = sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
        self.assertGreaterEqual(
            line_count,
            200,
            f"Wf-382-A3 closeout too short: {line_count} lines < 200 (expected ~280)",
        )


if __name__ == "__main__":
    unittest.main()
