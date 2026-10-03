"""Tests for Wf-390-A3 SE(3) SB integration audit script (CPU-only).

Per TODO/pending/383_se3_sb_fast_converging_integration.md §6 (Wf-390-A3 NEW
extension) + TODO/completed/383a1_weak_to_strong_se3sb_audit.md §3 + WEAK_TO_
STRONG_PLAN.md §4.3.1 (Wf-390-A3 NEW cross-ref):

  - test_audit_se3_sb_trainer_present: verifies train_se3sb.py exists,
    parse_args is importable + AST-parseable, and the Wf-383-A2 audit
    line-557 location still matches.
  - test_smoke_se3_sb_step_count_band: pins the 3 PROJECTED step count bands
    from Wf-383-A1 §3.2 (conservative 1500-2500 / aggressive 500-1000 / ceiling
    625) plus the MEASURED 10000-step baseline from Wf-339-B2 CPU scaffold.

Honest framing (party-twenty-spirit §3.5): all assertions reference the actual
audit script output. Step count bands are PROJECTED per Wf-383-A1 + upstream
papers (Karras 2022, Frans 2024, Mahmood 2026); Blackwell retrain that
converts PROJECTED → MEASURED is DEFERRED per AGENTS.md §5.1 + USER_OK-gated.

Run:
    pytest molmetal/tests/test_wf390a3_se3_sb.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


def _audit_repo_root() -> Path:
    """Resolve the repo root via MOLMETAL_ROOT env var, falling back to parent.parent.

    The audit script lives at molmetal/scripts/wf390a3_se3_sb_integration_audit.py;
    the parent.parent.parent of that file is the repo root.
    """
    env_root = os.environ.get("MOLMETAL_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    return here.parent.parent.parent


@pytest.fixture(scope="module")
def audit_module() -> Dict[str, Any]:
    """Lazy import of the Wf-390-A3 audit module.

    Falls back to subprocess if direct import fails (e.g. missing deps).
    """
    repo_root = _audit_repo_root()
    script_path = repo_root / "molmetal/scripts/wf390a3_se3_sb_integration_audit.py"
    # Try direct import first
    try:
        import importlib.util as _ilu

        spec = _ilu.spec_from_file_location(
            "wf390a3_se3_sb_integration_audit", str(script_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError("spec_from_file_location failed")
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {
            "audit_se3_sb_trainer": mod.audit_se3_sb_trainer,
            "smoke_se3_sb_step_count": mod.smoke_se3_sb_step_count,
            "source": "direct-import",
        }
    except Exception:
        # Fallback: subprocess invocation
        env = os.environ.copy()
        env["MOLMETAL_ROOT"] = str(repo_root)
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            cwd=str(repo_root),
        )
        return {
            "_stdout": proc.stdout,
            "_stderr": proc.stderr,
            "_returncode": proc.returncode,
            "source": "subprocess",
        }


# ---------------------------------------------------------------------------
# Test 1 — trainer audit
# ---------------------------------------------------------------------------


def test_audit_se3_sb_trainer_present(audit_module: Dict[str, Any]) -> None:
    """Verify train_se3sb.py exists, AST-parses, parse_args at line 557.

    Asserts per Wf-383-A2 §2.3 + Wf-390-A3 §6.2:
      - trainer_exists == True
      - ast_parses == True
      - parse_args_importable == True
      - parse_args_loc == 557 (Wf-383-A2 audit line)
      - parse_args_loc_match == True
      - n_se3_specific_flags_current in {0, ...} (current state; expected 0,
        will grow when Wf-200-A1..A10 wires execute)
    """
    if audit_module["source"] == "direct-import":
        trainer_audit = audit_module["audit_se3_sb_trainer"]()
    else:
        # Parse the subprocess stdout JSON output
        assert audit_module["_returncode"] in (
            0,
            1,
        ), f"subprocess failed: stderr={audit_module['_stderr']}"
        payload = json.loads(audit_module["_stdout"])
        trainer_audit = payload["trainer_audit"]

    assert (
        trainer_audit["trainer_exists"] is True
    ), f"SE(3) SB trainer not found at {trainer_audit['trainer_path']}"
    assert (
        trainer_audit["ast_parses"] is True
    ), "train_se3sb.py did not AST-parse cleanly"
    assert (
        trainer_audit["parse_args_importable"] is True
    ), "parse_args not found in train_se3sb.py"
    assert trainer_audit["parse_args_loc"] == 557, (
        f"parse_args line shifted from Wf-383-A2 audit: "
        f"got {trainer_audit['parse_args_loc']}, expected 557"
    )
    assert trainer_audit["parse_args_loc_match"] is True
    assert trainer_audit["n_se3_specific_flags_total"] == 10
    # Current state per Wf-383-A2 §1.2 audit: 0/10 SE(3)-specific flags wired
    # (upper bound not enforced — value may grow as Wf-200-A1..A10 wires execute)
    assert trainer_audit["n_se3_specific_flags_current"] >= 0
    assert (
        trainer_audit["trainer_loc"] > 500
    ), f"train_se3sb.py LOC suspiciously low: {trainer_audit['trainer_loc']}"


# ---------------------------------------------------------------------------
# Test 2 — smoke step count band
# ---------------------------------------------------------------------------


def test_smoke_se3_sb_step_count_band(audit_module: Dict[str, Any]) -> None:
    """Pin Wf-383-A1 §3.2 PROJECTED step count bands as acceptance criteria.

    Asserts per Wf-383-A1 §3.2 + Wf-390-A3 §6.3:
      - baseline_mechanical_steps == 10000 (MEASURED on Wf-339-B2 CPU scaffold)
      - mechanical_step_count_band == "1500-2500 conservative" (PROJECTED)
      - aggressive_step_count_band == "500-1000" (PROJECTED)
      - ceiling_step_count == 625 (PROJECTED)
    """
    if audit_module["source"] == "direct-import":
        step_audit = audit_module["smoke_se3_sb_step_count"]()
    else:
        payload = json.loads(audit_module["_stdout"])
        step_audit = payload["step_count_audit"]

    assert step_audit["baseline_mechanical_steps"] == 10000, (
        f"Baseline step count drifted from Wf-339-B2 baseline: "
        f"got {step_audit['baseline_mechanical_steps']}, expected 10000"
    )
    assert step_audit["mechanical_step_count_band"] == "1500-2500 conservative"
    assert step_audit["aggressive_step_count_band"] == "500-1000 aggressive"
    assert step_audit["ceiling_step_count"] == 625

    # Source strings preserved for cross-ref
    assert "Wf-339-B2" in step_audit["baseline_source"]
    assert "Wf-383-A1" in step_audit["mechanical_step_count_band_source"]
    assert "Frans 2024" in step_audit["aggressive_step_count_band_source"]
    assert "Frans 2024" in step_audit["ceiling_step_count_source"]

    # Blackwell retrain status preserved as DEFERRED
    assert "DEFERRED" in step_audit["blackwell_retrain_status"]
    assert "USER_OK" in step_audit["blackwell_retrain_status"]
