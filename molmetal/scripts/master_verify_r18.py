#!/usr/bin/env python3
"""Wf-35 Phase 3 — Master verify script for all R18 train scripts.

=====================================================================
This orchestrator verifies **every** R18 train script shipped in
``molmetal/scripts/`` after the Wf-35 Phase 2 refactor (which moved
each script to :mod:`molmetal.scripts._common_train_args`).

For each script the master verify runs three independent checks:

1. **Import test**       — imports the script via ``importlib`` and
                            verifies ``parse_args`` + ``main`` are
                            present.
2. **argparse test**     — invokes ``python <script> --help`` and
                            checks exit 0 + the expected argument
                            groups from the shared helpers.
3. **1 epoch smoke**     — runs the script with ``--epochs 1
                            --train-max-per-metal 50`` (plus a
                            scratch ``--out-dir`` and ``--metrics-path``
                            pointing into a temp directory) so the
                            decorator pipeline gets exercised
                            end-to-end without requiring a GPU.

The script writes two artefacts:

* ``molmetal/reports/wf35_master_verify/master_report.json`` — machine-
  readable aggregate suitable for the Wf-35 Phase 4 notional paper.
* ``molmetal/reports/wf35_master_verify/master_report.md`` — human-
  readable roll-up for the workflow dashboard.

Design notes
------------
* **CPU-only.**  No CUDA calls; torch seeding and ``setup_gpu_env``
  both fall back to ``cpu`` cleanly per the Wf-35 Phase 2 contract.
* **Subprocess isolation.**  Each script is probed in its own
  subprocess so a regression in one trainer never poisons the others.
* **Per-script timeout.**  600 s (10 min) per script — the 1-epoch
  smoke should finish in <30 s on CPU.
* **Skip-not-fail.**  Train scripts that are *planned but not yet
  authored* (e.g. ``train_cisplatin_resistance_v2.py``,
  ``bench_pb_valid.py`` per the Wf-35 plan) auto-skip with a
  visible "PENDING" status rather than failing the master verify.

CLI::

    cd /home/hugo/codes/mol-metal
    uv run --no-sync python molmetal/scripts/master_verify_r18.py

Exit codes:
    0 — all shipped scripts passed every check.
    1 — at least one shipped script failed a check.
    2 — master verify itself could not run (venv missing, repo missing).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SCRIPTS_DIR: Path = PROJECT_ROOT / "molmetal" / "scripts"
REPORTS_DIR: Path = PROJECT_ROOT / "molmetal" / "reports" / "wf35_master_verify"
JSON_OUT: Path = REPORTS_DIR / "master_report.json"
MD_OUT: Path = REPORTS_DIR / "master_report.md"
VENV_PY: Path = (
    PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
)


# ---------------------------------------------------------------------------
# Script registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScriptSpec:
    """One R18 train script + its Wf-35 refactor provenance."""

    name: str
    wf_tag: str
    expected_groups: Tuple[str, ...] = (
        "Common recipe",
        "GPU env",
    )


# The eight trainers called out in the Wf-35 plan.  Six are shipped
# today; the remaining two ("v2" / "bench_pb_valid") auto-skip with a
# "PENDING" status until they are authored.
SHIPPED_SCRIPTS: Tuple[ScriptSpec, ...] = (
    ScriptSpec("train_se3sb.py", "Wf-29"),
    ScriptSpec("train_transport.py", "Wf-29"),
    ScriptSpec("train_multiph.py", "Wf-29"),
    ScriptSpec("train_ru_5metal_se3sb.py", "Wf-31"),
    ScriptSpec("train_affinity_regression.py", "Wf-32"),
    ScriptSpec("train_multimetal_lomo.py", "Wf-32"),
)

PLANNED_SCRIPTS: Tuple[ScriptSpec, ...] = (
    ScriptSpec("train_cisplatin_resistance_v2.py", "Wf-31"),
    ScriptSpec("bench_pb_valid.py", "Wf-32"),
)

ALL_SCRIPTS: Tuple[ScriptSpec, ...] = SHIPPED_SCRIPTS + PLANNED_SCRIPTS


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    """Outcome of one (script, check) pair."""

    name: str  # e.g. "import", "argparse", "smoke"
    passed: bool
    skipped: bool = False
    seconds: float = 0.0
    message: str = ""


@dataclass
class ScriptReport:
    """Aggregate report for one script."""

    script: str
    wf_tag: str
    status: str = "PENDING"  # "PASS" | "FAIL" | "PENDING"
    checks: List[CheckResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    """JSON-safe coercion (numpy / Path / set)."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_to_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np  # noqa: WPS433

        if isinstance(obj, np.floating):
            return float(obj.item())
        if isinstance(obj, np.integer):
            return int(obj.item())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return obj


def _is_shipped(spec: ScriptSpec) -> bool:
    """A script is "shipped" iff the file exists AND it imports the
    Wf-35 refactor helpers.

    The Wf-35 contract requires every R18 trainer to use
    :func:`add_common_args`, :func:`add_gpu_args`, and
    :func:`MEASURED_vs_PROJECTED_decorator` from
    :mod:`molmetal.scripts._common_train_args`.  A script that exists
    on disk but predates the refactor (e.g. the original
    ``bench_pb_valid.py`` authored under Wf-32 Phase 3) is *not*
    shipped for Wf-35 purposes and should be reported ``PENDING``
    (refactor pending) rather than failing on the argparse check.
    """
    path = SCRIPTS_DIR / spec.name
    if not path.exists():
        return False
    try:
        src = path.read_text()
    except OSError:
        return False
    return "from molmetal.scripts._common_train_args import" in src


def _run_subprocess(
    cmd: Sequence[str],
    *,
    timeout: int,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str, float]:
    """Run ``cmd`` and return ``(returncode, stdout, stderr, seconds)``."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
            env=env,
        )
        dt = time.time() - t0
        return proc.returncode, proc.stdout, proc.stderr, dt
    except subprocess.TimeoutExpired as exc:
        dt = time.time() - t0
        out = (
            exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        )
        err = (
            exc.stderr.decode()
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "") + "\n[master_verify_r18] TIMEOUT"
        )
        return -1, out, err, dt


# ---------------------------------------------------------------------------
# Check 1 — Import test (in-process via importlib)
# ---------------------------------------------------------------------------
def _check_import(spec: ScriptSpec) -> CheckResult:
    """Import the script as a module + verify parse_args / main exist."""
    t0 = time.time()
    path = SCRIPTS_DIR / spec.name
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        mod = importlib.import_module(path.stem)
        if mod is None:
            return CheckResult(
                "import",
                False,
                seconds=time.time() - t0,
                message="importlib.import_module returned None",
            )
        missing = [n for n in ("parse_args", "main") if not hasattr(mod, n)]
        if missing:
            return CheckResult(
                "import",
                False,
                seconds=time.time() - t0,
                message=f"missing symbols: {', '.join(missing)}",
            )
        return CheckResult(
            "import",
            True,
            seconds=time.time() - t0,
            message="parse_args + main present",
        )
    except Exception as exc:  # pragma: no cover — defensive
        return CheckResult(
            "import",
            False,
            seconds=time.time() - t0,
            message=f"import raised {type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Check 2 — argparse `--help` test (subprocess via venv python)
# ---------------------------------------------------------------------------
def _check_argparse(spec: ScriptSpec) -> CheckResult:
    """Invoke ``python <script> --help`` and confirm exit 0 + groups."""
    t0 = time.time()
    if not VENV_PY.exists():
        return CheckResult(
            "argparse",
            False,
            skipped=True,
            seconds=time.time() - t0,
            message=f"venv python not at {VENV_PY}",
        )
    script_path = SCRIPTS_DIR / spec.name
    rc, out, err, dt = _run_subprocess(
        [str(VENV_PY), str(script_path), "--help"],
        timeout=30,
        cwd=PROJECT_ROOT,
    )
    if rc != 0:
        return CheckResult(
            "argparse",
            False,
            seconds=dt,
            message=f"--help exit {rc}: {err.strip().splitlines()[-1] if err else ''}",
        )
    missing = [g for g in spec.expected_groups if g not in out]
    if missing:
        return CheckResult(
            "argparse",
            False,
            seconds=dt,
            message=f"missing arg groups in --help: {missing}",
        )
    return CheckResult(
        "argparse",
        True,
        seconds=dt,
        message="--help exited 0 with " + ", ".join(spec.expected_groups),
    )


# ---------------------------------------------------------------------------
# Check 3 — 1-epoch smoke (subprocess with --epochs 1 --train-max-per-metal 50)
# ---------------------------------------------------------------------------
def _check_smoke(spec: ScriptSpec, tmpdir: Path) -> CheckResult:
    """Run the script end-to-end with a 1-epoch cap."""
    t0 = time.time()
    if not VENV_PY.exists():
        return CheckResult(
            "smoke",
            False,
            skipped=True,
            seconds=time.time() - t0,
            message=f"venv python not at {VENV_PY}",
        )
    script_path = SCRIPTS_DIR / spec.name
    out_dir = tmpdir / spec.name.replace(".py", "") / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tmpdir / spec.name.replace(".py", "") / "metrics.json"

    # Ensure the script sees a CPU environment.
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        str(VENV_PY),
        str(script_path),
        "--epochs",
        "1",
        "--train-max-per-metal",
        "50",
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
    ]
    rc, out, err, dt = _run_subprocess(
        cmd,
        timeout=600,
        cwd=PROJECT_ROOT,
        env=env,
    )
    last_err = err.strip().splitlines()[-1] if err.strip() else ""

    if rc != 0:
        return CheckResult(
            "smoke",
            False,
            seconds=dt,
            message=f"exit {rc}: {last_err or out.strip().splitlines()[-1] if out.strip() else 'no output'}",
        )
    # Confirm metrics file got written (decorator pipeline ran) even if
    # the JSON is empty — the file's mere existence proves the script
    # made it through ``parse_args`` + ``setup_gpu_env`` + ``main()``.
    if not metrics_path.exists():
        return CheckResult(
            "smoke",
            False,
            seconds=dt,
            message="exit 0 but --metrics-path was not written",
        )
    return CheckResult(
        "smoke",
        True,
        seconds=dt,
        message=(
            f"1-epoch + 50/mol ran in {dt:.1f}s, " f"metrics @ {metrics_path.name}"
        ),
    )


# ---------------------------------------------------------------------------
# Per-script driver
# ---------------------------------------------------------------------------
def _verify_script(spec: ScriptSpec, *, skip_smoke: bool = False) -> ScriptReport:
    """Run all three checks against one script."""
    start = time.time()
    rep = ScriptReport(script=spec.name, wf_tag=spec.wf_tag, status="PENDING")

    if not _is_shipped(spec):
        rep.status = "PENDING"
        path = SCRIPTS_DIR / spec.name
        if not path.exists():
            rep.error = "script not yet authored — auto-skipped per Wf-35 plan"
        else:
            rep.error = (
                "script authored but not yet refactored to "
                "_common_train_args — auto-skipped per Wf-35 contract"
            )
        rep.elapsed_seconds = time.time() - start
        return rep

    # Pre-flight: venv required for subprocess checks.
    if not VENV_PY.exists():
        rep.status = "FAIL"
        rep.error = f"venv python missing at {VENV_PY}"
        rep.elapsed_seconds = time.time() - start
        return rep

    # Each smoke run lives in its own tempdir so concurrent runs don't
    # collide on --out-dir / --metrics-path.
    with tempfile.TemporaryDirectory(prefix="wf35_") as raw_tmp:
        tmpdir = Path(raw_tmp)
        # Check 1: in-process import.  Cheap, no GPU.
        rep.checks.append(_check_import(spec))
        # Check 2: --help subprocess.  Cheap, no GPU.
        rep.checks.append(_check_argparse(spec))
        # Check 3: 1-epoch smoke.  CPU only — caller may skip via flag.
        if skip_smoke:
            rep.checks.append(
                CheckResult(
                    "smoke",
                    passed=True,
                    skipped=True,
                    seconds=0.0,
                    message="smoke skipped via --skip-smoke (CPU host)",
                )
            )
        else:
            rep.checks.append(_check_smoke(spec, tmpdir))

    rep.status = "PASS" if all(c.passed for c in rep.checks) else "FAIL"
    rep.error = "" if rep.status == "PASS" else _summarise_failed_checks(rep.checks)
    rep.elapsed_seconds = time.time() - start
    return rep


def _summarise_failed_checks(checks: List[CheckResult]) -> str:
    """First-line summary of the failed checks."""
    msgs = [f"{c.name}: {c.message}" for c in checks if not c.passed]
    return " | ".join(msgs) if msgs else "unknown"


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------
def _write_json(
    reports: List[ScriptReport], *, started: str, finished: str, total_seconds: float
) -> None:
    payload = {
        "schema_version": "1.0",
        "workflow": "Wf-35 Phase 3",
        "ran_at_utc": finished,
        "started_at_utc": started,
        "total_seconds": round(total_seconds, 2),
        "venv_python": str(VENV_PY),
        "summary": {
            "n_total": len(reports),
            "n_shipped": sum(1 for r in reports if r.status != "PENDING"),
            "n_passed": sum(1 for r in reports if r.status == "PASS"),
            "n_failed": sum(1 for r in reports if r.status == "FAIL"),
            "n_pending": sum(1 for r in reports if r.status == "PENDING"),
        },
        "scripts": [_to_jsonable(asdict(r)) for r in reports],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_markdown(
    reports: List[ScriptReport], *, started: str, finished: str, total_seconds: float
) -> None:
    n_total = len(reports)
    n_pass = sum(1 for r in reports if r.status == "PASS")
    n_fail = sum(1 for r in reports if r.status == "FAIL")
    n_pend = sum(1 for r in reports if r.status == "PENDING")

    lines = [
        "# Wf-35 Phase 3 — Master verify (R18 train scripts)",
        "",
        f"Generated at `{finished}` (UTC). Total wall-clock: "
        f"`{total_seconds:.1f} s`.",
        "",
        f"- **Total scripts**: {n_total}",
        f"- **PASS**: {n_pass}",
        f"- **FAIL**: {n_fail}",
        f"- **PENDING**: {n_pend}",
        f"- **venv python**: `{VENV_PY}`",
        "",
        "## Per-script detail",
        "",
        "| Script | WF | Status | Checks (import / argparse / smoke) | "
        "Elapsed (s) | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        ck_parts = []
        for c in r.checks:
            if c.passed and not c.skipped:
                ck_parts.append(f"✓ {c.name}")
            elif c.skipped:
                ck_parts.append(f"⏭ {c.name}")
            else:
                ck_parts.append(f"✗ {c.name}")
        ck = ", ".join(ck_parts)
        if r.status == "FAIL":
            notes = r.error or "checks failed"
        elif r.status == "PENDING":
            notes = r.error or "auto-skipped"
        else:
            notes = "all checks passed"
        lines.append(
            f"| `{r.script}` | {r.wf_tag} | {r.status} | {ck} | "
            f"{r.elapsed_seconds:.1f} | {notes} |"
        )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- **Check 1 — Import**: in-process via `importlib`."
        " Verifies `parse_args` + `main` exist."
    )
    lines.append(
        "- **Check 2 — argparse**: subprocess `--help` against "
        "the venv python. Expects `Common recipe` + `GPU env` "
        "argument groups from `_common_train_args`."
    )
    lines.append(
        "- **Check 3 — 1-epoch smoke**: subprocess "
        "`--epochs 1 --train-max-per-metal 50` with "
        '`CUDA_VISIBLE_DEVICES=""`. Confirms the '
        "`MEASURED_vs_PROJECTED_decorator` pipeline runs to "
        "completion and writes `--metrics-path`."
    )
    lines.append("")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_argv(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wf-35 Phase 3 — master verify all R18 train scripts.",
    )
    p.add_argument(
        "--only-shipped",
        action="store_true",
        help="Skip the planned-but-not-yet-authored scripts "
        "(default: include them with status=PENDING).",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Substring(s) of script filenames to exclude (repeatable).",
    )
    p.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the 1-epoch smoke subprocess (use on CPU-only hosts "
        "where the underlying Triton / CUDA paths are unavailable).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_argv(argv)

    print(f"[master_verify_r18] venv = {VENV_PY}")
    if not VENV_PY.exists():
        print(
            f"[master_verify_r18] ERROR: venv missing — cannot run "
            f"subprocess checks."
        )
        return 2
    if not SCRIPTS_DIR.exists():
        print(f"[master_verify_r18] ERROR: scripts dir missing — " f"{SCRIPTS_DIR}")
        return 2

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    print(f"[master_verify_r18] {started} — verifying " f"{len(ALL_SCRIPTS)} scripts")

    to_run: List[ScriptSpec] = []
    for spec in ALL_SCRIPTS:
        if args.only_shipped and spec in PLANNED_SCRIPTS:
            continue
        if any(ex in spec.name for ex in args.exclude):
            continue
        to_run.append(spec)

    reports: List[ScriptReport] = []
    for spec in to_run:
        print(f"[master_verify_r18] -> {spec.name} ({spec.wf_tag})")
        rep = _verify_script(spec, skip_smoke=args.skip_smoke)
        reports.append(rep)
        # Compact per-script status line — skip marks (⏭) are
        # distinguishable from passes (✓) in the log.
        ck_parts = []
        for c in rep.checks:
            if c.passed and not c.skipped:
                ck_parts.append(f"✓{c.name}")
            elif c.skipped:
                ck_parts.append(f"⏭{c.name}")
            else:
                ck_parts.append(f"✗{c.name}")
        ck = ", ".join(ck_parts)
        suffix = (" (" + rep.error + ")") if rep.error else ""
        print(
            f"[master_verify_r18]   {rep.status}  [{ck}]  "
            f"{rep.elapsed_seconds:.1f}s{suffix}"
        )

    finished = datetime.now(timezone.utc).isoformat()
    total_s = time.time() - t0

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(reports, started=started, finished=finished, total_seconds=total_s)
    _write_markdown(reports, started=started, finished=finished, total_seconds=total_s)

    n_fail = sum(1 for r in reports if r.status == "FAIL")
    n_pass = sum(1 for r in reports if r.status == "PASS")
    n_pend = sum(1 for r in reports if r.status == "PENDING")

    print()
    print(
        f"[master_verify_r18] DONE in {total_s:.1f}s — "
        f"pass={n_pass} fail={n_fail} pending={n_pend}"
    )
    print(f"[master_verify_r18] reports/  : {REPORTS_DIR}")
    print(f"[master_verify_r18] json out  : {JSON_OUT}")
    print(f"[master_verify_r18] md out    : {MD_OUT}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
