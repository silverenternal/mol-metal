"""Wf-368-A1 — Skip-semantic regression guards for ``wf343c2_per_wave_check.py``.

Per Wf-367-A2 + Wf-368-A1: pre-flight gate
``p3_wf343c2_per_wave_runtime_check`` invokes
``wf343c2_per_wave_check.py --skip-live`` WITHOUT ``--strict``.  The
expected semantic is **Option C**: when ``--skip-live`` is set, the 3
live-probe sub-checks (hyperparams_config_complete + gpu_vram_available
+ disk_free_space) legitimately SKIP, and the overall gate must NOT
downgrade to FAIL — exit code 0 is the correct outcome.

Background (Wf-364-A1 §4.2 + Wf-367-A2 root cause):
  * ``--skip-live`` alone → 3 SKIPs + 4 PASS → ``overall_status=PASS_WITH_SKIP``
    → exit 0 (no ``--strict``).
  * ``--skip-live --strict`` → 3 SKIPs → exit 2 → FAIL cascade.
  * Pre-flight gate's CPU-only context must NEVER use ``--strict``.

This module ships TWO new regression tests that pin the Option C
contract at the script's CLI boundary (subprocess invocation, not just
the in-process ``run_per_wave_checks`` dict):

  1. ``test_wf343c2_skip_live_3_skips_returns_rc_zero`` — fresh
     preflight + 3 SKIPs (--skip-live, no --config, no --strict) →
     exit 0.
  2. ``test_wf343c2_skip_live_zero_fail_returns_rc_zero`` — fresh
     preflight + full YAML + --skip-live + no FAIL → exit 0.

Both tests use ``tmp_path`` for ephemeral preflight + YAML so the
canonical ``molmetal/reports/wf259_run/pre_flight.json`` is NOT
modified.  CPU-only; wall budget < 5 s.

Source: Wf-368-A1 closeout + Wf-367-A2 + Wf-364-A1 §4.2 root-cause.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Make ``wf343c2_per_wave_check`` importable as a module (no torch needed).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / "molmetal" / "scripts"
REPO_ROOT = PROJECT_ROOT

# Allow direct module import for helpers + path constants.
sys.path.insert(0, str(SCRIPT_DIR))

import wf343c2_per_wave_check as wf343c2  # noqa: E402

VENV_PY = "/home/hugo/codes/mol-metal/.venv/bin/python"
WF343C2_SCRIPT = SCRIPT_DIR / "wf343c2_per_wave_check.py"
# The script's --out default lives under REPORTS_DIR, not SCRIPT_DIR.
_WF343C2_REPORT_DIR = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf343c2_per_wave_check"
)


def _write_fresh_preflight(path: Path, *, n_pass: int = 22, n_fail: int = 0) -> None:
    """Write a fresh-looking pre-flight JSON (mtime = now)."""
    obj = {
        "timestamp": "2026-09-29T15:00:00+0000",
        "schema_version": "207.v1",
        "pipeline": "all",
        "summary": {
            "pass": n_pass,
            "fail": n_fail,
            "skip": 0,
            "total": n_pass + n_fail,
            "total_wall_sec": 1.0,
        },
        "gates": [],
    }
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_full_yaml(path: Path) -> None:
    """Write a YAML with all REQUIRED_CONFIG_KEYS."""
    text = (
        "data: {csv: foo.csv}\n"
        "model: {backbone: x}\n"
        "flow: {n_steps: 100}\n"
        "train: {lr: 1e-4}\n"
        "protocol: {epochs: 50, time_budget_hr: 10}\n"
        "eval: {decode_smoke_every: 200}\n"
        "output: {dir: out/}\n"
        "compute: {gpu: cuda:0, yuelbond: true}\n"
        "honest_framing: 'ok'\n"
    )
    path.write_text(text, encoding="utf-8")


def _run_wf343c2_subprocess(
    *,
    wave: str,
    preflight: Path,
    config: Path | None,
    pairs_metals: str,
    pairs_seeds: str,
    budget_hr: float,
    skip_live: bool,
    strict: bool,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Subprocess-run ``wf343c2_per_wave_check.py`` with the given args and
    return the completed process.  CPU-only; wall < 60s."""
    assert WF343C2_SCRIPT.is_file(), (
        f"wf343c2_per_wave_check.py missing: {WF343C2_SCRIPT}"
    )
    args = [
        VENV_PY,
        str(WF343C2_SCRIPT),
        "--wave", wave,
        "--preflight", str(preflight),
        "--pairs-metals", pairs_metals,
        "--pairs-seeds", pairs_seeds,
        "--budget-hr", str(budget_hr),
    ]
    if config is not None:
        args.extend(["--config", str(config)])
    if skip_live:
        args.append("--skip-live")
    if strict:
        args.append("--strict")
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )


# ---------------------------------------------------------------------------
# Test 1 — --skip-live + 3 SKIPs + no FAIL → exit 0 (Option C contract)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path(VENV_PY).exists(), reason="vlm venv not on disk"
)
def test_wf343c2_skip_live_3_skips_returns_rc_zero(tmp_path: Path) -> None:
    """Wf-368-A1 Option C contract: ``wf343c2_per_wave_check.py --skip-live``
    (without ``--config``, without ``--strict``) legitimately produces 3
    SKIPs (hyperparams_config_complete + gpu_vram_available +
    disk_free_space); 0 FAIL; overall=PASS_WITH_SKIP; **exit code 0**.

    Pin the contract end-to-end via subprocess (not just
    ``run_per_wave_checks`` in-process).  The pre-flight gate
    ``gate_wf343c2_per_wave_runtime_check`` invokes the script with
    exactly this argument shape (--skip-live, no --config, no --strict),
    so a regression here will cascade to a FAIL on
    ``p3_wf343c2_per_wave_runtime_check``.
    """
    pf = tmp_path / "preflight.json"
    _write_fresh_preflight(pf, n_pass=22, n_fail=0)
    # No --config (hyperparams_config_complete will SKIP).
    proc = _run_wf343c2_subprocess(
        wave="Wf-368-A1 skip-semantic test",
        preflight=pf,
        config=None,
        pairs_metals="Ru,Ir",
        pairs_seeds="0,1",
        budget_hr=10.0,
        skip_live=True,
        strict=False,
    )
    # Exit code 0 (PASS_WITH_SKIP is a valid outcome per Option C).
    assert proc.returncode == 0, (
        f"Wf-368-A1 Option C contract VIOLATED: "
        f"--skip-live + 3 SKIPs + no FAIL should return rc=0, got "
        f"rc={proc.returncode}. stdout={proc.stdout[-500:]} "
        f"stderr={proc.stderr[-500:]}. This is the regression: someone "
        f"re-introduced --strict into the pre-flight gate invocation, "
        f"causing the legitimate 3 SKIPs to cascade to FAIL exit 2."
    )
    # Verify the JSON output reflects the expected PASS_WITH_SKIP state.
    # The script's --out default is
    # ``molmetal/reports/wf343c2_per_wave_check/per_wave_check.json``
    # (REPORTS_DIR, not SCRIPT_DIR).
    out_path = _WF343C2_REPORT_DIR / "per_wave_check.json"
    assert out_path.is_file(), (
        f"expected JSON output at {out_path} not found after wf343c2 run"
    )
    json_obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert json_obj["n_fail"] == 0, (
        f"expected 0 FAIL, got {json_obj['n_fail']}: "
        f"{[(c['check_id'], c['status']) for c in json_obj['checks']]}"
    )
    assert json_obj["n_skip"] >= 3, (
        f"expected ≥ 3 SKIPs (hyperparams_config_complete + "
        f"gpu_vram_available + disk_free_space), got {json_obj['n_skip']}"
    )
    # The 3 expected SKIP IDs must all be present.
    skip_ids = {c["check_id"] for c in json_obj["checks"] if c["status"] == "SKIP"}
    assert "hyperparams_config_complete" in skip_ids
    assert "gpu_vram_available" in skip_ids
    assert "disk_free_space" in skip_ids
    # overall_status must be PASS_WITH_SKIP (Option C semantic).
    assert json_obj["overall_status"] == "PASS_WITH_SKIP", (
        f"expected overall_status=PASS_WITH_SKIP (Option C), got "
        f"{json_obj['overall_status']!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — --skip-live + 0 FAIL (all PASS or PASS_WITH_SKIP) → exit 0
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path(VENV_PY).exists(), reason="vlm venv not on disk"
)
def test_wf343c2_skip_live_zero_fail_returns_rc_zero(tmp_path: Path) -> None:
    """Wf-368-A1 guard: ``wf343c2_per_wave_check.py --skip-live`` (with
    full --config, without --strict) yields 0 FAIL; exit code 0
    regardless of whether ``n_skip > 0``.

    This is the broader Option C contract: **any time ``n_fail == 0``,
    exit code MUST be 0** (modulo future wave-launcher --strict opt-in).
    Pin this so that future regression (e.g. an "exit 2 on any SKIP"
    added to main()) does not silently break the pre-flight gate's
    CPU-only invocation.
    """
    pf = tmp_path / "preflight.json"
    _write_fresh_preflight(pf, n_pass=22, n_fail=0)
    cfg = tmp_path / "config.yaml"
    _write_full_yaml(cfg)
    proc = _run_wf343c2_subprocess(
        wave="Wf-368-A1 zero-fail guard",
        preflight=pf,
        config=cfg,
        pairs_metals="Ru,Ir,Os",
        pairs_seeds="0,1",
        budget_hr=10.0,
        skip_live=True,
        strict=False,
    )
    # Exit code 0 (no FAIL is FAIL).
    assert proc.returncode == 0, (
        f"Wf-368-A1 zero-fail guard VIOLATED: --skip-live + 0 FAIL "
        f"should return rc=0, got rc={proc.returncode}. "
        f"stdout={proc.stdout[-500:]} stderr={proc.stderr[-500:]}. "
        f"This regression would cascade the pre-flight gate to FAIL "
        f"forever in the CPU-only context."
    )
    # Verify the JSON output reflects the expected state.
    out_path = _WF343C2_REPORT_DIR / "per_wave_check.json"
    assert out_path.is_file(), (
        f"expected JSON output at {out_path} not found after wf343c2 run"
    )
    json_obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert json_obj["n_fail"] == 0, (
        f"expected 0 FAIL, got {json_obj['n_fail']}: "
        f"{[(c['check_id'], c['status']) for c in json_obj['checks']]}"
    )
    # overall_status is either PASS (no SKIP) or PASS_WITH_SKIP (some SKIP).
    # Both are valid outcomes per Option C — exit 0 either way.
    assert json_obj["overall_status"] in {"PASS", "PASS_WITH_SKIP"}, (
        f"expected overall_status in {{PASS, PASS_WITH_SKIP}}, got "
        f"{json_obj['overall_status']!r}"
    )
    # If --config was provided AND --skip-live is set, the only SKIPs
    # should be the 2 live-probe checks (gpu + disk).  hyperparams_config
    # should be PASS in this scenario.
    check_statuses = {
        c["check_id"]: c["status"] for c in json_obj["checks"]
    }
    assert check_statuses["hyperparams_config_complete"] == "PASS", (
        f"--config was provided; hyperparams_config_complete should be "
        f"PASS, got {check_statuses['hyperparams_config_complete']}"
    )
    assert check_statuses["gpu_vram_available"] == "SKIP"
    assert check_statuses["disk_free_space"] == "SKIP"
    # The 4 non-live non-YuelBond checks should all be PASS.
    assert check_statuses["preflight_freshness_24h"] == "PASS"
    assert check_statuses["preflight_no_fail_gates"] == "PASS"
    assert check_statuses["yuelbond_decoder_registered"] == "PASS"
    assert check_statuses["wave_budget_remaining_nonneg"] == "PASS"