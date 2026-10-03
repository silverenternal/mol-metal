"""Wf-334-U3 — Wf-264 CFM property panel launch script tests.

Owner: Hugo Lee (autonomous Wf-334 ultracode unit)
Date: 2026-09-29
Parent: Wf-298 §1.3 "launch script not yet written" + INDEX.md §22.7
        "Property panel status: NOT STARTED at Wf-265-Collect time".

These tests are CPU-only — they inspect the launch wrapper's source to
validate structural correctness, dry-run behaviour, USER_OK gate
ordering, and the Wf-328 optimization-flag honest-framing contract
(those flags are documented as NA for this RDKit-based panel per
AGENTS.md §5.2, NOT silently wired).

Per AGENTS.md §5.2: the Wf-328 flags (``--torch-compile`` / ``--bf16`` /
``--tf32`` / ``--w2-gradient-mode`` / ``--num-workers``) do NOT apply to
``cfm_property_panel.py`` (SA + QED post-hoc scorer).  The launcher
documents this in its header + the dry-run output; the tests assert
that this honest-negative framing is present (not silently absent).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path("/home/hugo/codes/mol-metal")
WRAPPER = ROOT / "scripts" / "launch_wf264_cfm_property_panel.sh"
SMILES_CSV = ROOT / "molmetal" / "data" / "metalcytotox_rdkit_features.csv"
SCRIPT = ROOT / "molmetal" / "scripts" / "cfm_property_panel.py"


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
    after = idx + len(marker)
    fi_idx = src.find("\nfi\n", after)
    if fi_idx > 0:
        after = fi_idx + len("\nfi\n")
    return after


# ---------------------------------------------------------------------------
# Test 1: launcher script parses (bash -n) + USER_OK gate ordering
# ---------------------------------------------------------------------------


def test_launcher_parses_and_user_ok_gate_blocks_real_launch():
    """Wrapper must:
    - parse via bash -n (no syntax errors)
    - contain a USER_OK gate that fires BEFORE the pre-flight check
      (kill_then_adapt ordering invariant per Wf-300C / Wf-306D)
    - end-to-end invocation without USER_OK must exit non-zero
      before any panel work
    """
    src = _read_wrapper()
    body = src[_body_start(src) :]

    # 1.1 — bash syntax check
    proc_syntax = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (
        proc_syntax.returncode == 0
    ), f"wrapper bash syntax error: stderr={proc_syntax.stderr!r}"

    # 1.2 — USER_OK gate text present in source
    assert "USER_OK" in src, "USER_OK gate text missing from wrapper"
    assert "kill_then_adapt" in src, "kill_then_adapt reference missing"

    # 1.3 — Gate is checked BEFORE pre_flight_check (kill_then_adapt
    # ordering invariant per AGENTS.md §5.1 + Wf-300C pattern).
    gate_idx = body.find('if [[ -z "${USER_OK:-}" ]]')
    pre_flight_idx = body.find("pre_flight_check.py --pipeline all")
    assert gate_idx > 0, "USER_OK gate if-block not found in body"
    assert pre_flight_idx > 0, "pre_flight_check invocation not found in body"
    assert gate_idx < pre_flight_idx, (
        f"USER_OK gate (idx {gate_idx}) must appear BEFORE pre_flight "
        f"(idx {pre_flight_idx})"
    )

    # 1.4 — End-to-end: invocation without USER_OK exits non-zero
    # (we strip --dry-run so we hit the gate path).
    proc = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "USER_OK"},
        timeout=30,
    )
    assert proc.returncode != 0, (
        f"wrapper exited 0 without USER_OK; expected gate block. "
        f"stdout={proc.stdout[-200:]!r}"
    )
    assert (
        "USER OK REQUIRED" in proc.stdout
    ), f"gate block message missing from stdout; got {proc.stdout[-200:]!r}"


# ---------------------------------------------------------------------------
# Test 2: pre-flight integration + Wf-328 honest-framing contract
# ---------------------------------------------------------------------------


def test_preflight_integration_and_wf328_honest_negative():
    """Wrapper must:
    (a) invoke pre_flight_check.py --pipeline all (TODO-207 §1 gate)
    (b) document the Wf-328 optimization flags as NA for the SA/QED
        panel (AGENTS.md §5.2 honest-negative contract) — header text
        + dry-run echo
    (c) reference the correct panel script + SMILES CSV path
    (d) reference cfm_property_panel.py, NOT the D-MPNN trainer
    """
    src = _read_wrapper()
    body = src[_body_start(src) :]

    # (a) pre-flight check wired
    assert (
        "pre_flight_check.py --pipeline all" in body
    ), "pre_flight_check.py --pipeline all invocation missing"

    # (b) Wf-328 honest-negative framing — the Wf-328 flag names must
    # appear in the wrapper header COMMENT + the dry-run echo, each
    # paired with an "NA" / "do NOT apply" marker so future agents can
    # reproduce the honest-negative framing.  Note: the dry-run echo
    # lives INSIDE the --dry-run block which _body_start skips, so we
    # search the full `src` (not `body`) for these strings.
    for flag in (
        "--torch-compile",
        "--bf16",
        "--tf32",
        "--w2-gradient-mode",
        "--num-workers",
    ):
        assert flag in src, (
            f"Wf-328 flag {flag} missing from wrapper (header honest-"
            f"negative contract per AGENTS.md §5.2)"
        )
    # Honest-negative framing language present in header
    assert (
        "do NOT apply" in src or "NA" in src or "do NOT" in src
    ), "Wf-328 NA / do-not-apply framing language missing from wrapper"
    # Section explicitly named in dry-run echo (search src, not body,
    # because the echo is inside the dry-run block that body_start skips)
    assert (
        "Wf-328 optimization flags: NA for this panel" in src
    ), "Wf-328 NA dry-run echo missing from wrapper source"

    # (c) Panel script + SMILES CSV referenced correctly
    assert (
        "cfm_property_panel.py" in src
    ), "cfm_property_panel.py not referenced in wrapper"
    assert (
        str(SMILES_CSV) in src
    ), f"SMILES CSV path {SMILES_CSV} not referenced in wrapper"

    # (d) Wrong trainer (the D-MPNN Wasserstein trainer must NOT be
    # referenced — that is a different ship surface).
    assert "train_dmpnn_multitask_wasserstein" not in src, (
        "Wasserstein D-MPNN trainer should NOT be referenced in Wf-264 "
        "property panel launcher (different ship surface)"
    )

    # (e) Dry-run path works end-to-end (USER_OK env var respected —
    # the dry-run exits before the gate, so this must work without
    # USER_OK too, but we set it for safety).  Note: pre_flight_check
    # --pipeline all can take 60-180s on cold disk; bump timeout to
    # 300s so the test does not flake on slow hosts.
    if SMILES_CSV.exists():
        proc = subprocess.run(
            ["bash", str(WRAPPER), "--dry-run"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "USER_OK": "1"},
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"wrapper --dry-run must exit 0; got {proc.returncode}. "
            f"stdout={proc.stdout[-300:]!r} stderr={proc.stderr[-300:]!r}"
        )
        # Dry-run echo contains honest-negative Wf-328 framing
        assert (
            "Wf-328 optimization flags: NA for this panel" in proc.stdout
        ), "Wf-328 honest-negative framing missing from dry-run stdout"
        # Dry-run exits 0 and CSV sanity check passes
        assert (
            "DRY-RUN OK" in proc.stdout
        ), "DRY-RUN OK marker missing from dry-run stdout"
        assert (
            "OK: SMILES CSV exists" in proc.stdout
        ), "SMILES CSV existence check failed in dry-run stdout"
        # Pre-flight check is invoked and its result is surfaced in
        # dry-run stdout (either PASS marker or the FAIL-ACK marker for
        # the current project regression state — per AGENTS.md §6 pre-
        # flight v3 42 PASS / 15 FAIL / 57 TOTAL HONEST-NEGATIVE).
        # Dry-run does NOT gate on pre-flight FAIL; real-launch path
        # (line ~${PY} pre_flight_check.py || exit 1) does gate.
        preflight_invoked = (
            "Pre-flight check" in proc.stdout or "PREFLIGHT_FAIL_ACK" in proc.stdout
        )
        assert (
            preflight_invoked
        ), "Pre-flight check invocation not surfaced in dry-run stdout"
    else:
        pytest.skip(
            f"SMILES CSV {SMILES_CSV} not present on this host — "
            f"dry-run end-to-end skipped (structural assertions above suffice)"
        )
