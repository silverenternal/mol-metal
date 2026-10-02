"""Wf-406-A4 — Wf-246-C metal-coordination term retrain SCAFFOLD tests.

================================================================
Background
================================================================
Wf-246-C adds a metal-coordination term to the Wf-131 P3 D-MPNN
multi-task training pipeline. The term is a soft kcal/mol adjustment
on top of the Wf-131 pIC50 regression loss + Wf-318 Wasserstein
regulariser, derived from ``molmetal.molmetal_lam.sbdd_env.metal_coord_term.py``
(Wf-333-V1 sign-fixed kcal adjustment per Vina convention).

Wf-406-A4 (2026-09-30) ships the CPU-only SCAFFOLD surface that
wires ``MetalCoordinationTerm`` into the trainer. These tests
verify:

1. **SCAFFOLD smoke passes**: the scaffold script runs the 5-step
   CPU smoke end-to-end without crashing; import + build +
   kcal_sign_audit all succeed; result.json is emitted.
2. **Launch wrapper syntax + USER_OK gate**: the bash wrapper
   passes ``bash -n`` syntax check; the wrapper REFUSES to run
   when ``USER_OK != 1`` (exit 64); the wrapper runs the SCAFFOLD
   smoke end-to-end when ``USER_OK=1`` is set + mode is
   ``--smoke``.

Honest framing
==============
* Tests are CPU-only and depend on ``rdkit`` being installed
  (transitively, via ``metal_coord_term.py``).
* 0 GPU launches performed by these tests.
* 0 ``paper/*`` touches.
* 0 wet-lab planning (per AGENTS.md §10).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAFFOLD = REPO_ROOT / "molmetal" / "scripts" / "wf406a4_metal_coord_retrain_scaffold.py"
LAUNCH_WRAPPER = REPO_ROOT / "scripts" / "wf406a4_launch_wf246c_metal_coord_retrain.sh"


def test_scaffold_smoke_passes() -> None:
    """Verify the Wf-406-A4 SCAFFOLD script runs the 5-step CPU
    smoke end-to-end without crashing; import + build +
    kcal_sign_audit all succeed; result.json is emitted.

    The smoke writes a ``result.json`` to
    ``molmetal/reports/wf406a4_metal_coord_scaffold/`` (or a tmp
    out_dir if overridden). We use a tmp out_dir to avoid
    polluting the repo state.
    """
    import tempfile

    if not SCAFFOLD.exists():
        pytest.skip(f"SCAFFOLD script missing: {SCAFFOLD}")
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "wf406a4_out"
        cmd = [
            sys.executable,
            str(SCAFFOLD),
            "--metal-coord-term",
            "--smoke",
            "--metals", "Ru", "Ir",
            "--seeds", "0", "1",
            "--epochs", "5",
            "--out-dir", str(out_dir),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"SCAFFOLD smoke failed with rc={proc.returncode}; "
            f"stdout={proc.stdout!r}; stderr={proc.stderr!r}"
        )
        result_path = out_dir / "result.json"
        assert result_path.exists(), (
            f"result.json missing at {result_path}; "
            f"stdout={proc.stdout!r}; stderr={proc.stderr!r}"
        )
        result: Dict[str, Any] = json.loads(result_path.read_text())
        assert result["wf"] == "Wf-406-A4"
        assert result["status"] == "SHIPPED_SCAFFOLD_CPU_ONLY"
        assert result["n_metals"] == 2
        assert result["n_seeds"] == 2
        assert result["kcal_sign_audit"]["n_scores_audited"] == 5
        # The kcal_sign_audit must verify the Wf-333-V1 fix:
        # for c > 0.5 → kcal < 0 (better binding per Vina)
        # for c < 0.5 → kcal > 0 (worse binding per Vina)
        assert result["kcal_sign_audit"]["sign_audit_pass"], (
            "kcal sign audit FAILED; Wf-333-V1 sign fix verification "
            "is broken: " + json.dumps(result["kcal_sign_audit"], indent=2)
        )
        # Honest framing must reflect SCAFFOLD-only + USER_OK gating
        hf = result["honest_framing"]
        assert hf["user_ok_required_for_gpu_launch"] is True
        assert hf["user_ok_env_var"] == "USER_OK=1"
        assert hf["metal_coord_sign_fix_applied"] is True
        assert "BLOCKED" in hf["verdict"]


def test_launch_wrapper_syntax() -> None:
    """Verify the Wf-406-A4 bash launch wrapper:

    (a) ``bash -n`` syntax check passes
    (b) USER_OK gate refuses when USER_OK is not "1" (exit 64)
    (c) Wrapper runs the SCAFFOLD smoke end-to-end when USER_OK=1
        is set + mode is --smoke.

    Uses a tmp out_dir to avoid polluting the repo state.
    """
    import tempfile

    if not LAUNCH_WRAPPER.exists():
        pytest.skip(f"Launch wrapper missing: {LAUNCH_WRAPPER}")
    # (a) bash -n syntax check
    proc = subprocess.run(
        ["bash", "-n", str(LAUNCH_WRAPPER)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"bash -n failed on {LAUNCH_WRAPPER}; "
        f"stdout={proc.stdout!r}; stderr={proc.stderr!r}"
    )
    # (b) USER_OK gate refuses
    env_no_user = os.environ.copy()
    env_no_user.pop("USER_OK", None)
    env_no_user["PYTHONPATH"] = (
        str(REPO_ROOT) + os.pathsep + env_no_user.get("PYTHONPATH", "")
    )
    proc = subprocess.run(
        [str(LAUNCH_WRAPPER), "--dry-run"],
        cwd=str(REPO_ROOT),
        env=env_no_user,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 64, (
        f"USER_OK gate REFUSAL expected rc=64; "
        f"got rc={proc.returncode}; stdout={proc.stdout!r}; "
        f"stderr={proc.stderr!r}"
    )
    assert "USER_OK gate refused" in proc.stderr, (
        f"USER_OK gate refusal message missing from stderr; "
        f"stderr={proc.stderr!r}"
    )
    # (c) USER_OK=1 with --smoke runs SCAFFOLD end-to-end
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "wf406a4_launch_out"
        env_yes_user = os.environ.copy()
        env_yes_user["USER_OK"] = "1"
        env_yes_user["PYTHONPATH"] = (
            str(REPO_ROOT) + os.pathsep + env_yes_user.get("PYTHONPATH", "")
        )
        proc = subprocess.run(
            [
                str(LAUNCH_WRAPPER),
                "--smoke",
                "--metals", "Ru", "Ir", "Rh",
                "--seeds", "0",
                "--epochs", "5",
                "--out-dir", str(out_dir),
            ],
            cwd=str(REPO_ROOT),
            env=env_yes_user,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, (
            f"USER_OK=1 + --smoke failed with rc={proc.returncode}; "
            f"stdout={proc.stdout!r}; stderr={proc.stderr!r}"
        )
        # DRY_RUN.log should be emitted
        log_path = out_dir / "DRY_RUN.log"
        assert log_path.exists(), (
            f"DRY_RUN.log missing at {log_path}; "
            f"stdout={proc.stdout!r}; stderr={proc.stderr!r}"
        )
        # plan.json should be emitted
        plan_path = out_dir / "plan.json"
        assert plan_path.exists(), (
            f"plan.json missing at {plan_path}; "
            f"stdout={proc.stdout!r}; stderr={proc.stderr!r}"
        )
        plan = json.loads(plan_path.read_text())
        assert plan["wf"] == "Wf-406-A4"
        assert plan["mode"] == "R2_smoke"
        assert plan["user_ok_required"] is True
        assert plan["n_pairs"] == 3  # 3 metals × 1 seed
        # SCAFFOLD result.json should also be emitted (smoke runs it)
        result_path = out_dir / "result.json"
        assert result_path.exists(), (
            f"SCAFFOLD result.json missing at {result_path}; "
            f"stdout={proc.stdout!r}; stderr={proc.stderr!r}"
        )