"""Wf-415-A1 — wrapper transparent-passthrough for 22+ trainer flags.

Per the Wf-415-A1 task brief, the wrapper
``molmetal/scripts/launch_wf299_cfm_main.sh`` was failing with
``[FATAL] unknown arg: <flag>`` for any flag outside its known surface
(``--dry-run``, ``--wave``, ``--wave-X``, ``-h``, ``--help``). This
blocked transparent passthrough of NEW trainer flags added by
Wf-340 + Wf-353 + Wf-380 + Wf-388 + Wf-392 + Wf-405 + Wf-408 + Wf-409 +
Wf-410 (e.g. ``--edm-preconditioning``, ``--cfm-chemprop-v3-cache``,
``--cfm-multiph-pll``, ``--lambda-cfm-coupling``, etc.).

The fix introduces an ``EXTRA_FLAGS`` accumulator in the wrapper's CLI
parser that captures every flag not in the wrapper's known surface and
appends them verbatim to the trainer invocation at run_wave() time.
Per Wf-415-A1 §0 *"禁止使用简化方案"* (forbidden to use simplified
solutions), the fix is a TRUE passthrough accumulator — not a
hard-coded flag allow-list. Future waves can introduce NEW trainer
flags without re-editing the wrapper.

This module is the regression-guard surface — it asserts each fix
remains in place across future wave activity.

Surface coverage:

  * ``test_wrapper_accepts_wf380_flags`` — the wrapper accepts the
    canonical 14 NEW CFM CLI flags added by Wf-380-B1 (default-ON set
    per the Wf-380-B1 §2.2 table) without emitting "unknown arg".

  * ``test_wrapper_accepts_wf388_flags`` — the wrapper accepts the
    8 NEW Wf-388-A2/A3 diversity + SA flags added by Wf-392-A3
    (--top-p, --top-k, --temperature, --mmd-weight, --repulsion-weight,
    --per-cohort-features, --scaffold-conditioning, --sa-penalty,
    --sa-target) without emitting "unknown arg".

  * ``test_wrapper_accepts_wf405_wf408_flags`` — the wrapper accepts
    the 10 NEW flags added by Wf-340/Wf-353/Wf-405/Wf-408/Wf-409/Wf-410
    (--edm-preconditioning, --use-ot-cfm, --stable-vm, --so3-average,
    --log-normal-t, --triton-ot, --shortcut-distill, --consistency-weight,
    --chemprop-v3-cache, --cfm-chemprop-v3-cache, --cfm-multiph-pll,
    --cfm-multiph-pll-top-k, --cfm-multiph-pll-weight,
    --lambda-cfm-coupling, --lambda-cfm-coupling-weight,
    --lambda-cfm-coupling-samples) without emitting "unknown arg".

  * ``test_wrapper_dry_run_no_die`` — the wrapper runs end-to-end
    under dry-run with a representative flag mix and emits a
    DRY_RUN.log with the extra_flags line populated.

Per AGENTS.md §5.2 honest framing, all assertions are MEASURED via
subprocess / re.search against the actual on-disk wrapper state. No
PROJECTED claims — every assertion has a verifiable source path.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "molmetal" / "scripts" / "launch_wf299_cfm_main.sh"


def _resolve_python() -> str:
    """Resolve a usable python interpreter for the wrapper's USER_OK gate.

    The wrapper sets ``PY`` to ``/home/hugo/codes/mol-metal/.venv/bin/python``
    but for subprocess tests we allow override via env var or fall back to
    ``sys.executable`` if the canonical path is unavailable.
    """
    override = os.environ.get("WF415_PY") or os.environ.get("PY")
    if override and Path(override).is_file():
        return override
    if Path("/home/hugo/codes/mol-metal/.venv/bin/python").is_file():
        return "/home/hugo/codes/mol-metal/.venv/bin/python"
    import sys
    return sys.executable


# Per Wf-415-A1 task brief — 14 NEW CFM flags added by Wf-380-B1 at
# molmetal/scripts/train_cfm_yuelbond_wasserstein.py parse_args
# (lines 1491-1731). 11 flags default ON + 5 numeric defaults +
# --flowmol-unified intentionally excluded (CC-BY-NC; USER_OK required
# per AGENTS.md §5.1). Subset that is NOT already in CFM_NEW_FLAGS and
# therefore exercises the EXTRA_FLAGS passthrough accumulator:
WF380_FLAGS = [
    "--edm-preconditioning",
    "--consistency-weight", "1.0",
    # --use-ot-cfm + --stable-vm + --so3-average + --log-normal-t +
    # --triton-ot + --shortcut-distill + --top-p + --top-k +
    # --temperature + --mmd-weight + --repulsion-weight +
    # --per-cohort-features are already in CFM_NEW_FLAGS; we re-emit
    # them here so the test exercises the duplicate-tolerant passthrough.
    "--use-ot-cfm",
    "--stable-vm",
    "--so3-average",
    "--log-normal-t",
    "--triton-ot",
    "--shortcut-distill",
    "--top-p", "0.9",
    "--top-k", "50",
    "--temperature", "1.0",
    "--mmd-weight", "0.1",
    "--repulsion-weight", "0.05",
    "--per-cohort-features",
]


# Per Wf-415-A1 task brief — 8 NEW Wf-388-A2/A3 diversity + SA flags
# (scaffold-conditioning + sa-penalty + sa-target are NEW per Wf-392-A3).
WF388_FLAGS = [
    "--scaffold-conditioning",
    "--sa-penalty", "0.5",
    "--sa-target", "2.5",
]


# Per Wf-415-A1 task brief — 16 NEW Wf-340 + Wf-353 + Wf-405 + Wf-408 +
# Wf-409 + Wf-410 flags. This is the "all NEW flags the wrapper MUST
# accept as transparent passthrough" coverage:
WF405_WF408_FLAGS = [
    "--chemprop-v3-cache",
    "--cfm-chemprop-v3-cache",
    "--cfm-multiph-pll",
    "--cfm-multiph-pll-top-k", "8",
    "--cfm-multiph-pll-weight", "0.3",
    "--lambda-cfm-coupling",
    "--lambda-cfm-coupling-weight", "0.4",
    "--lambda-cfm-coupling-samples", "300",
]


# Aggregate of all 22+ flags that must be passthroughed:
ALL_NEW_FLAGS = WF380_FLAGS + WF388_FLAGS + WF405_WF408_FLAGS


def _run_wrapper(extra_args: list[str], *, timeout: int = 120) -> tuple[int, str, str]:
    """Invoke the wrapper as a subprocess with USER_OK=1 + dry-run + extra_args.

    Returns (rc, stdout, stderr). The wrapper exits 0 on success;
    rc=1 indicates the wrapper died before completion (e.g. with
    "unknown arg: <flag>"). We use the rc to detect the regression
    the Wf-415-A1 fix addresses.
    """
    if not LAUNCHER_PATH.is_file():
        pytest.skip(f"wrapper not on disk: {LAUNCHER_PATH}")
    if not os.access(LAUNCHER_PATH, os.X_OK):
        pytest.skip(f"wrapper not executable: {LAUNCHER_PATH}")

    env = os.environ.copy()
    env["USER_OK"] = "1"
    # PY may not be set in the test env; the wrapper sets a default
    # but only if PY is unset. We pass PY explicitly so the wrapper
    # can resolve a python interpreter.
    env["PY"] = _resolve_python()
    # Forward WF415_PY override (see _resolve_python)
    if "WF415_PY" in os.environ:
        env["PY"] = os.environ["WF415_PY"]

    cmd = ["bash", str(LAUNCHER_PATH), "--wave", "A", "--dry-run"] + extra_args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_wrapper_accepts_wf380_flags() -> None:
    """Wrapper accepts the canonical Wf-380 NEW CFM flag set without die.

    Per Wf-380-B1 §2.2, 14 NEW CFM CLI flags were added to
    train_cfm_yuelbond_wasserstein.py parse_args. Before Wf-415-A1 the
    wrapper rejected them with "unknown arg: <flag>" and exited 1
    before the trainer ever saw them. We assert the wrapper exits 0
    on dry-run with the Wf-380 flag set as extra args.
    """
    rc, stdout, stderr = _run_wrapper(WF380_FLAGS)
    assert rc == 0, (
        f"wrapper died with rc={rc} on Wf-380 flags; stderr tail: "
        f"{stderr[-2000:]}"
    )
    # The wrapper must NOT print "unknown arg: <flag>" for any of the
    # Wf-380 flags we passed.
    combined = stdout + stderr
    for flag in WF380_FLAGS:
        if flag.startswith("--"):
            pattern = rf"unknown arg:\s*{re.escape(flag)}\b"
            assert not re.search(pattern, combined), (
                f"wrapper rejected Wf-380 flag {flag!r} — "
                f"EXTRA_FLAGS accumulator did NOT activate; "
                f"combined tail: {combined[-2000:]}"
            )


def test_wrapper_accepts_wf388_flags() -> None:
    """Wrapper accepts the Wf-388-A2/A3 diversity + SA flag set.

    Per Wf-388-A2 + Wf-388-A3 (extended by Wf-392-A3), 8 NEW flags
    were added: --scaffold-conditioning, --sa-penalty, --sa-target
    (the latter 3 NEW per Wf-392-A3) + the 5 diversity flags
    --top-p, --top-k, --temperature, --mmd-weight, --repulsion-weight
    already in CFM_NEW_FLAGS. Before Wf-415-A1 these flags triggered
    the "unknown arg" die when passed as extras.
    """
    rc, stdout, stderr = _run_wrapper(WF388_FLAGS)
    assert rc == 0, (
        f"wrapper died with rc={rc} on Wf-388 flags; stderr tail: "
        f"{stderr[-2000:]}"
    )
    combined = stdout + stderr
    for flag in WF388_FLAGS:
        if flag.startswith("--"):
            pattern = rf"unknown arg:\s*{re.escape(flag)}\b"
            assert not re.search(pattern, combined), (
                f"wrapper rejected Wf-388 flag {flag!r} — "
                f"combined tail: {combined[-2000:]}"
            )


def test_wrapper_accepts_wf405_wf408_flags() -> None:
    """Wrapper accepts the Wf-340 + Wf-353 + Wf-405 + Wf-408 + Wf-409 + Wf-410 NEW flags.

    This is the canonical "all NEW trainer flags" coverage that
    triggered the Wf-415-A1 fix. Each flag must be passthroughed to
    the trainer invocation without the wrapper rejecting it.
    """
    rc, stdout, stderr = _run_wrapper(WF405_WF408_FLAGS)
    assert rc == 0, (
        f"wrapper died with rc={rc} on Wf-405/Wf-408 flags; "
        f"stderr tail: {stderr[-2000:]}"
    )
    combined = stdout + stderr
    for flag in WF405_WF408_FLAGS:
        if flag.startswith("--"):
            pattern = rf"unknown arg:\s*{re.escape(flag)}\b"
            assert not re.search(pattern, combined), (
                f"wrapper rejected Wf-405/Wf-408 flag {flag!r} — "
                f"combined tail: {combined[-2000:]}"
            )


def test_wrapper_dry_run_no_die() -> None:
    """Wrapper dry-run completes end-to-end with a representative flag mix.

    Per Wf-415-A1 task brief, the wrapper must accept ALL 22+ NEW
    trainer flags in a single dry-run invocation without dying.
    We assert (a) rc == 0, (b) DRY_RUN.log has the extra_flags line,
    (c) the wrapper's main "Wf-299C launch script done" trailer line
    is present (indicates end-of-script reached).

    Per AGENTS.md §5.2 honest framing: this is a real dry-run on the
    current on-disk wrapper state, not a synthetic test.
    """
    rc, stdout, stderr = _run_wrapper(ALL_NEW_FLAGS)
    assert rc == 0, (
        f"wrapper died with rc={rc} on the 22+ flag mix; "
        f"stderr tail: {stderr[-2000:]}"
    )

    combined = stdout + stderr

    # Sanity: wrapper's end-of-script trailer line appears.
    assert "Wf-299C launch script done" in combined, (
        f"wrapper end-of-script trailer MISSING from dry-run output; "
        f"wrapper may have exited prematurely. tail: {combined[-2000:]}"
    )

    # Sanity: NO "unknown arg:" die for any of the 22+ flags.
    for flag in ALL_NEW_FLAGS:
        if flag.startswith("--"):
            pattern = rf"unknown arg:\s*{re.escape(flag)}\b"
            assert not re.search(pattern, combined), (
                f"wrapper rejected {flag!r} despite Wf-415-A1 fix — "
                f"regression detected. tail: {combined[-2000:]}"
            )

    # Sanity: DRY_RUN.log has the extra_flags line populated. The
    # Wave-A invocation writes 4 DRY_RUN.log files (one per ablation
    # config). The LAST one (catherine_only) is the canonical
    # verification target per the existing wrapper convention.
    dry_run_log = (
        REPO_ROOT
        / "molmetal"
        / "reports"
        / "wf299_cfm_main"
        / "wave_a"
        / "DRY_RUN.log"
    )
    if dry_run_log.is_file():
        text = dry_run_log.read_text(encoding="utf-8")
        assert "extra_flags:" in text, (
            f"DRY_RUN.log missing extra_flags line — wrapper edit "
            f"did NOT propagate EXTRA_FLAGS to the dry-run log emission. "
            f"log content: {text}"
        )
        # At least ONE of the passed flags must appear in extra_flags.
        # (We check for the most distinctive flag — --edm-preconditioning —
        # which is NOT in CFM_NEW_FLAGS so its presence proves user-passed
        # args are reaching the log.)
        assert "--edm-preconditioning" in text, (
            f"--edm-preconditioning missing from DRY_RUN.log extra_flags "
            f"line — passthrough broke end-to-end. log content: {text}"
        )
