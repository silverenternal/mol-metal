"""Wf-310B — TODO-182 G6 final USER_OK-gated launcher tests.

Owner: Hugo Lee
Date: 2026-09-29
Parent: TODO/pending/182_wf124_p3_per_innovation_ablation.md
Per Wf-310B §6: 8 tests covering USER_OK gate, pre-flight ordering,
per-wave checklist, abort_recovery wiring, manifest completeness,
yaml config parse, dry-run idempotency.

These tests are CPU-only — no torch.cuda calls, no actual training.
They inspect the wrapper script's source + the manifest JSON +
the g6 yaml config to validate structural correctness.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = ROOT / "scripts" / "launch_wf310b_g6_final.sh"
MANIFEST = ROOT / "molmetal/reports/wf310b_g6_manifest/60_tuples.json"
G6_YAML = ROOT / "configs/wf124_p3_per_innovation_ablation_g6.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_wrapper() -> str:
    assert WRAPPER.exists(), f"wrapper missing: {WRAPPER}"
    return WRAPPER.read_text()


def _body_start(src: str) -> int:
    """Index of the first executable line in the wrapper (skip header
    comment block + shebang AND the --dry-run block)."""
    marker = "\nset -e\n"
    idx = src.find(marker)
    if idx < 0:
        return 0
    # Skip past the --dry-run block — it contains "echo" lines that
    # mention the tool names without actually invoking them.
    after = idx + len(marker)
    # Find the close of the first dry-run if block by searching for
    # the first standalone `fi\n` after the marker.
    fi_idx = src.find("\nfi\n", after)
    if fi_idx > 0:
        after = fi_idx + len("\nfi\n")
    return after


def _read_manifest() -> dict:
    assert MANIFEST.exists(), f"manifest missing: {MANIFEST}"
    return json.loads(MANIFEST.read_text())


# ---------------------------------------------------------------------------
# Test 1: USER_OK gate blocks real launch
# ---------------------------------------------------------------------------

def test_user_ok_gate_blocks():
    """Without USER_OK env var, the wrapper must exit 1 BEFORE touching GPU."""
    src = _read_wrapper()
    body = src[_body_start(src):]
    # Gate text present
    assert "USER_OK" in src, "USER_OK gate text missing from wrapper"
    assert "kill_then_adapt" in src, "kill_then_adapt reference missing"
    # Gate is checked BEFORE any pre_flight / wave_budget / GPU work
    gate_idx = body.find("if [[ -z \"${USER_OK:-}\" ]]")
    pre_flight_idx = body.find("pre_flight_check.py --pipeline all")
    assert gate_idx > 0, "USER_OK gate if-block not found in body"
    assert pre_flight_idx > 0, "pre_flight_check invocation not found in body"
    assert gate_idx < pre_flight_idx, (
        f"USER_OK gate (idx {gate_idx}) must appear BEFORE pre_flight "
        f"(idx {pre_flight_idx})"
    )
    # Actual end-to-end: invocation without USER_OK exits non-zero
    # (we strip --dry-run so we hit the gate path)
    proc = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "USER_OK"},
        timeout=15,
    )
    assert proc.returncode != 0, (
        f"wrapper exited 0 without USER_OK; expected gate block. "
        f"stdout={proc.stdout[-200:]!r}"
    )
    assert "USER OK REQUIRED" in proc.stdout, (
        f"gate block message missing from stdout; got {proc.stdout[-200:]!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: USER_OK unblocks (dry-run path still exits 0)
# ---------------------------------------------------------------------------

def test_user_ok_unblocks():
    """With USER_OK=1 set, the dry-run path still exits 0 (no GPU exec)."""
    proc = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "USER_OK": "1"},
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"wrapper --dry-run with USER_OK=1 must exit 0; got {proc.returncode}. "
        f"stdout={proc.stdout[-300:]!r} stderr={proc.stderr[-300:]!r}"
    )
    # USER_OK gate acknowledgement appears in the path the script would
    # take IF it weren't dry-run.  The dry-run exits before the gate,
    # so the acknowledgement text only fires for real-launch paths.
    # Verify the gate text exists in the source so the unblock path is
    # structurally present.
    src = _read_wrapper()
    body = src[_body_start(src):]
    assert "USER_OK set — proceeding" in body, (
        "USER_OK unblock acknowledgement line missing from wrapper source"
    )


# ---------------------------------------------------------------------------
# Test 3: pre_flight_check.py called FIRST (before wave_budget + per-wave)
# ---------------------------------------------------------------------------

def test_pre_flight_called_first():
    """pre_flight_check must be invoked before wave_budget_tracker
    and per_wave_checklist (kill_then_adapt ordering invariant)."""
    src = _read_wrapper()
    body = src[_body_start(src):]
    pre_flight_idx = body.find("pre_flight_check.py --pipeline all")
    budget_idx = body.find("wave_budget_tracker.py")
    per_wave_idx = body.find("per_wave_checklist.py --wave")
    assert pre_flight_idx > 0, "pre_flight_check --pipeline all not invoked"
    assert budget_idx > 0, "wave_budget_tracker not invoked"
    assert per_wave_idx > 0, "per_wave_checklist not invoked"
    assert pre_flight_idx < budget_idx, "pre_flight must precede wave_budget"
    assert budget_idx < per_wave_idx, (
        "wave_budget_tracker must precede per_wave_checklist"
    )


# ---------------------------------------------------------------------------
# Test 4: per_wave_checklist emission per sub-wave (a..h)
# ---------------------------------------------------------------------------

def test_per_wave_checklist_emitted():
    """per_wave_checklist.py must be emitted for each of 8 sub-waves a..h."""
    src = _read_wrapper()
    body = src[_body_start(src):]
    for letter in "abcdefgh":
        wave_label = f"Wf-124-P3-G6-{letter}"
        assert wave_label in body, (
            f"per_wave_checklist emission missing for {wave_label}"
        )
    # Count emission occurrences — wrapper uses an emit_wave_checklist
    # shell function; the function itself calls per_wave_checklist.py.
    # So we count BOTH the function-call sites AND the function body call.
    emissions = re.findall(
        r'emit_wave_checklist "Wf-124-P3-G6-([a-h])"', body
    )
    assert len(emissions) >= 8, (
        f"expected 8 per-wave checklist emissions, found {len(emissions)}: {emissions}"
    )
    assert set(emissions) == set("abcdefgh"), (
        f"emission letters do not span a..h: {set(emissions)}"
    )


# ---------------------------------------------------------------------------
# Test 5: abort_recovery.py wired on unexpected abort
# ---------------------------------------------------------------------------

def test_abort_recovery_wired():
    """abort_recovery.py must be invoked when a tuple run fails
    (TODO-207 §7.3 contract)."""
    src = _read_wrapper()
    body = src[_body_start(src):]
    assert "abort_recovery.py" in body, "abort_recovery.py not invoked"
    # Find the actual call site: ${PY} molmetal/scripts/abort_recovery.py \
    abort_call_marker = "${PY} molmetal/scripts/abort_recovery.py"
    abort_idx = body.find(abort_call_marker)
    assert abort_idx > 0, "abort_recovery.py call site not found"
    # The abort_recovery invocation must be guarded by a failure check
    # (set +e ... ${PIPESTATUS[0]} ... if rc != 0)
    rc_check_idx = body.find("PIPESTATUS[0]")
    assert rc_check_idx > 0, "PIPESTATUS capture missing (rc tracking broken)"
    assert rc_check_idx < abort_idx, (
        f"abort_recovery call (idx {abort_idx}) must be guarded by rc check "
        f"(idx {rc_check_idx})"
    )
    # Pattern argument is wired (TODO-207 §7.3)
    after_call = body[abort_idx:]
    assert "--pattern" in after_call, (
        "abort_recovery.py --pattern argument missing"
    )


# ---------------------------------------------------------------------------
# Test 6: 60-tuple manifest complete
# ---------------------------------------------------------------------------

def test_60_tuples_manifest_complete():
    """Manifest must enumerate exactly 60 (config × metal × seed) tuples."""
    m = _read_manifest()
    assert m["totals"]["n_tuples"] == 60, (
        f"manifest totals.n_tuples must be 60, got {m['totals']['n_tuples']}"
    )
    assert len(m["tuples"]) == 60, (
        f"manifest must contain 60 tuple entries, got {len(m['tuples'])}"
    )
    # Coverage: 4 configs × 5 metals × 3 seeds = 60
    configs = {t["config"] for t in m["tuples"]}
    metals = {t["metal"] for t in m["tuples"]}
    seeds = {t["seed"] for t in m["tuples"]}
    assert configs == {"no_multi_task", "no_wass", "no_kendall", "no_dann"}, (
        f"configs mismatch: {configs}"
    )
    assert metals == {"Ru", "Ir", "Os", "Rh", "Re"}, f"metals mismatch: {metals}"
    assert seeds == {0, 1, 2}, f"seeds mismatch: {seeds}"
    # Tuple IDs are 1..60 contiguous
    ids = sorted(t["tuple_id"] for t in m["tuples"])
    assert ids == list(range(1, 61)), f"tuple_ids not contiguous 1..60: {ids}"
    # Every tuple carries a CLI args list
    for t in m["tuples"]:
        assert isinstance(t["cli_args"], list) and len(t["cli_args"]) >= 5, (
            f"tuple {t['tuple_id']} cli_args too short: {t['cli_args']}"
        )
        assert t["expected_wall_min"] > 0, (
            f"tuple {t['tuple_id']} expected_wall_min must be > 0"
        )
        assert 0.0 <= t["expected_test_auc_projected"] <= 1.0, (
            f"tuple {t['tuple_id']} expected_test_auc_projected OOR"
        )


# ---------------------------------------------------------------------------
# Test 7: yaml config parses + required keys present
# ---------------------------------------------------------------------------

def test_yaml_config_parses():
    """g6 yaml config must parse via yaml.safe_load and contain required keys."""
    yaml = pytest.importorskip("yaml")
    assert G6_YAML.exists(), f"g6 yaml missing: {G6_YAML}"
    cfg = yaml.safe_load(G6_YAML.read_text())
    assert isinstance(cfg, dict), "g6 yaml must parse to dict"
    # Required top-level keys (per pre_flight_check.py §2.2 contract)
    for key in ("data", "model", "protocol", "runs", "waves", "compute",
                "ablation_configs", "honest_framing"):
        assert key in cfg, f"g6 yaml missing required top-level key: {key}"
    # 4 ablation configs present
    ab = cfg["ablation_configs"]
    assert set(ab.keys()) >= {
        "no_multi_task", "no_wass", "no_kendall", "no_dann"
    }, f"ablation_configs missing lever(s): {set(ab.keys())}"
    # Total runs field is 60
    assert cfg["protocol"]["total_runs"] == 60, (
        f"protocol.total_runs must be 60, got {cfg['protocol']['total_runs']}"
    )
    assert cfg["ablation_configs"].get("total_runs") == 60, (
        f"ablation_configs.total_runs must be 60"
    )
    # 8 waves decomposition
    assert cfg["waves"]["total_waves"] == 8, (
        f"waves.total_waves must be 8, got {cfg['waves']['total_waves']}"
    )


# ---------------------------------------------------------------------------
# Test 8: dry-run idempotent (twice yields identical tail)
# ---------------------------------------------------------------------------

def test_dry_run_idempotent():
    """--dry-run must be idempotent — invoking twice yields the same
    final summary lines (no GPU exec, no per-tuple mutation)."""
    proc1 = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "USER_OK": "1"},
        timeout=30,
    )
    proc2 = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "USER_OK": "1"},
        timeout=30,
    )
    assert proc1.returncode == 0
    assert proc2.returncode == 0
    # Compare tail (last 5 lines) — must be identical
    tail1 = proc1.stdout.strip().splitlines()[-5:]
    tail2 = proc2.stdout.strip().splitlines()[-5:]
    assert tail1 == tail2, (
        f"dry-run not idempotent:\n  run1 tail: {tail1}\n  run2 tail: {tail2}"
    )
    # Both runs must report exactly 60 tuples planned
    assert "60 (config, metal, seed) tuples PLANNED" in proc1.stdout
    assert "60 (config, metal, seed) tuples PLANNED" in proc2.stdout
