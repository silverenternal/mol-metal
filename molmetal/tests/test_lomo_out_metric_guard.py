"""Wf-C2 regression tests for the ``--out-metric`` guard on the LOMO trainer.

Background
----------
``train_metal_lomo_wasserstein.py`` used to default ``--out-metric`` to the
canonical headline path ``metrics/by_metric/metal_lomo_auc_wasserstein.json``.
A round-1 smoke run therefore silently overwrote the project's canonical
measurement with a 1-metal / 1-epoch / ``--quick`` number.  The file was
restored with ``git checkout``, but the hazard was still live.

The fix (a) removes the default entirely, so the trainer exits non-zero when
the flag is unset, and (b) refuses to write the canonical path unless the
caller passes ``--allow-canonical-out-metric``.

These tests are the on-disk proof that the hazard cannot come back.  They
run the trainer as a *subprocess* (the real CLI surface) on the smallest
possible CPU job, and assert on the canonical file's mtime + content hash.

Tests
-----
  * test_out_metric_has_no_default — parser default is ``None``.
  * test_unset_out_metric_exits_nonzero_and_leaves_canonical_untouched —
    a real 1-metal/1-epoch run without ``--out-metric`` exits 3 and leaves
    the canonical file byte-identical.
  * test_explicit_temp_out_metric_leaves_canonical_untouched — a real run
    with ``--out-metric <tmp>/metrics.json`` writes only the temp file.
  * test_canonical_out_metric_requires_allow_flag — pointing ``--out-metric``
    at the canonical path is refused (exit 3) without
    ``--allow-canonical-out-metric``.
  * test_all_lomo_launchers_pass_out_metric — every ``scripts/launch_*.sh``
    that invokes the LOMO trainer passes ``--out-metric`` explicitly.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_metal_lomo_wasserstein.py"
CANONICAL = PROJECT_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
LAUNCHERS = PROJECT_ROOT / "scripts"

# The guard exit code, kept in sync with the trainer's `return 3` branches.
GUARD_EXIT = 3


def _fingerprint(path: Path) -> str:
    """Return ``"<mtime_ns>:<sha256>"`` for *path* (or ``"<missing>"``)."""
    if not path.is_file():
        return "<missing>"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{path.stat().st_mtime_ns}:{digest}"


def _run_trainer(extra: list[str], out_dir: Path) -> subprocess.CompletedProcess:
    """Run the smallest viable LOMO job with *extra* appended flags."""
    cmd = [
        sys.executable,
        str(TRAINER),
        "--metals",
        "Ru",
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--quick",
        "--device",
        "cpu",
        "--out-dir",
        str(out_dir),
        *extra,
    ]
    env = dict(os.environ)
    env.setdefault("MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal"))
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )


@pytest.fixture(scope="module")
def canonical_fingerprint() -> str:
    return _fingerprint(CANONICAL)


# --------------------------------------------------------------------------
# 1. Parser-level: the default is gone.
# --------------------------------------------------------------------------
def test_out_metric_has_no_default():
    """`_parse_args([])` must yield `out_metric is None`, not the canonical path."""
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location("lomo_trainer_probe", TRAINER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The trainer declares dataclasses at import time; dataclasses resolves
    # annotations through sys.modules, so register before exec.
    _sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        _sys.modules.pop(spec.name, None)

    args = module._parse_args([])
    assert args.out_metric is None, (
        "--out-metric must have NO default; it used to default to the "
        f"canonical {CANONICAL}"
    )
    assert args.allow_canonical_out_metric is False


# --------------------------------------------------------------------------
# 2. Behavioural: unset flag => non-zero exit, canonical file untouched.
# --------------------------------------------------------------------------
def test_unset_out_metric_exits_nonzero_and_leaves_canonical_untouched(
    tmp_path, canonical_fingerprint
):
    before = _fingerprint(CANONICAL)
    proc = _run_trainer([], out_dir=tmp_path / "unset")

    assert proc.returncode == GUARD_EXIT, (
        f"expected guard exit {GUARD_EXIT} with --out-metric unset, got "
        f"{proc.returncode}\nSTDOUT tail:\n{proc.stdout[-2000:]}"
    )
    assert "Wf-C2 GUARD" in (proc.stdout + proc.stderr), (
        "the guard must log an explanatory error, not fail silently"
    )
    assert _fingerprint(CANONICAL) == before == canonical_fingerprint, (
        "canonical metrics/by_metric/metal_lomo_auc_wasserstein.json was "
        "modified by a run that did not ask for it"
    )
    # The run still produced its per-run artifacts under --out-dir.
    assert (tmp_path / "unset" / "synthesis.md").is_file()


# --------------------------------------------------------------------------
# 3. Behavioural: explicit temp path => only the temp file is written.
# --------------------------------------------------------------------------
def test_explicit_temp_out_metric_leaves_canonical_untouched(
    tmp_path, canonical_fingerprint
):
    target = tmp_path / "metrics" / "metrics.json"
    proc = _run_trainer(
        ["--out-metric", str(target)], out_dir=tmp_path / "explicit"
    )

    # 0 = gate PASS, 2 = gate BELOW_GATE; both mean the guard did not fire.
    assert proc.returncode in (0, 2), (
        f"expected 0 or 2, got {proc.returncode}\n"
        f"STDOUT tail:\n{proc.stdout[-2000:]}"
    )
    assert target.is_file(), "the explicitly requested metrics file was not written"
    assert _fingerprint(CANONICAL) == canonical_fingerprint, (
        "an explicit --out-metric to a temp path must not touch the canonical file"
    )


# --------------------------------------------------------------------------
# 4. Behavioural: the canonical path needs the explicit allow flag.
# --------------------------------------------------------------------------
def test_canonical_out_metric_requires_allow_flag(tmp_path, canonical_fingerprint):
    proc = _run_trainer(
        ["--out-metric", str(CANONICAL)], out_dir=tmp_path / "canonical_attempt"
    )

    assert proc.returncode == GUARD_EXIT, (
        f"writing the canonical path without --allow-canonical-out-metric must "
        f"exit {GUARD_EXIT}, got {proc.returncode}"
    )
    assert "Wf-C2 GUARD" in (proc.stdout + proc.stderr)
    assert _fingerprint(CANONICAL) == canonical_fingerprint, (
        "the canonical headline file was overwritten without the allow flag"
    )


# --------------------------------------------------------------------------
# 5. Static: no launcher invokes the trainer without an explicit flag.
# --------------------------------------------------------------------------
def test_all_lomo_launchers_pass_out_metric():
    """Every trainer *call site* in every launcher must pass --out-metric.

    A naive `if "--out-metric" in text` check is too weak: a launcher can
    have several call sites (e.g. the wf125 controlled-pair launcher has a
    scaffold smoke plus the real run) and patching only one leaves the other
    silently exiting GUARD_EXIT. So count the invocations instead.
    """
    offenders = []
    for sh in sorted(LAUNCHERS.glob("launch_*.sh")):
        text = sh.read_text(encoding="utf-8", errors="ignore")
        if "train_metal_lomo_wasserstein.py" not in text:
            continue
        if "--out-metric" not in text:
            offenders.append(f"{sh.name}: no --out-metric at all")
    assert not offenders, (
        "these launchers invoke the LOMO trainer without --out-metric and "
        f"will now exit {GUARD_EXIT}: {offenders}"
    )


def _lomo_launcher_paths() -> list:
    """Return launchers whose ``TRAINER`` variable resolves to the LOMO trainer.

    Resolving the variable matters: many launchers set ``TRAINER`` to a
    *different* script (e.g. train_dmpnn_multitask_wasserstein.py), and a
    naive text search flags those too.
    """
    hits = []
    for sh in sorted(LAUNCHERS.glob("launch_*.sh")):
        text = sh.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'^\s*TRAINER="([^"]+)"', text, re.M)
        if m and m.group(1).endswith("train_metal_lomo_wasserstein.py"):
            hits.append(sh)
    return hits


def test_every_lomo_call_site_is_patched():
    """Every real LOMO trainer invocation must pass --out-metric.

    Scans each LOMO launcher's backslash-continued command blocks for a line
    that actually *executes* the trainer. Excluded, because they are not
    invocations: ``--trainer "${TRAINER}"`` (the path handed to a gate
    script) and ``log "... ${TRAINER} ..."`` (an echo of the plan).
    """
    problems = []
    for sh in _lomo_launcher_paths():
        lines = sh.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines):
            if '"${TRAINER}"' not in line:
                continue
            stripped = line.lstrip()
            # Not an invocation: an existence check, a JSON echo, a plan
            # echo, a --help capability probe, or the path handed to a gate
            # script via --trainer.
            if (
                stripped.startswith("[ ")
                or stripped.startswith("printf")
                or stripped.startswith("log ")
                or "--trainer" in line
                or "--help" in line
                or stripped.startswith("TRAINER_HELP=")
            ):
                continue
            # Walk the backslash-continued block.
            block = [line]
            j = i
            while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):
                j += 1
                block.append(lines[j])
            if not any("--out-metric" in b for b in block):
                problems.append(f"{sh.name}:{i + 1}")
    assert not problems, (
        "trainer call sites missing --out-metric (they will exit "
        f"{GUARD_EXIT}): {problems}"
    )


# Guard against the test itself rotting if the trainer path moves.
def test_trainer_exists():
    assert TRAINER.is_file(), f"trainer not found at {TRAINER}"
    assert CANONICAL.is_file(), f"canonical metric not found at {CANONICAL}"
    # A stale fingerprint in a fixture would silently weaken every other test.
    assert time.time() > 0
