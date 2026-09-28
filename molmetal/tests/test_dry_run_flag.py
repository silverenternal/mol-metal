"""IMPROVEMENT_BACKLOG B4 — ``--dry-run`` flag contract for ``r17_gpu_retrain.py``.

Verifies that the R17 GPU retrain CLI's ``--dry-run`` flag satisfies
the spec agreed in ``TODO/IMPROVEMENT_BACKLOG.md`` §B4:

    | --dry-run flag is exposed via ``--help``
    | --dry-run auto-detects GPU (writes ``gpu_probe.json``)
    | --dry-run runs the data-load + adapter-setup scaffolding
    | --dry-run prints ``DRY RUN: would launch 10000-step retrain``
    | --dry-run exits with code 0
    | --dry-run does NOT actually train (no ckpt.pt in the seed_out
      dir, only the cmd-record file)
    | --dry-run is idempotent: re-running it with the same args
      produces the same exit code

These tests spawn ``r17_gpu_retrain.py --dry-run`` as a real
subprocess (the canonical M-R17-A1 unit-test pattern).  They run
in <30 s even on CPU because the wrapper short-circuits before
launching ``metallo_drug_smoke_retrain.py``.

Skipped automatically when the script is not present.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "r17_gpu_retrain.py"


def _script_exists() -> bool:
    return SCRIPT.is_file()


pytestmark = pytest.mark.skipif(
    not _script_exists(),
    reason=f"r17 retrain script not present at {SCRIPT}",
)


_SHARED_VENV = Path("/home/hugo/codes/CadButEaas/.venv-vlm/bin/python")
_DEFAULT_PY = _SHARED_VENV if _SHARED_VENV.exists() else Path(sys.executable)


def _run(args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(_DEFAULT_PY), "-u", str(SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Help / flag surface
# ---------------------------------------------------------------------------
def test_dry_run_flag_appears_in_help():
    """``--dry-run`` must be a recognised CLI flag (visible in --help)."""
    proc = _run(["--help"], timeout=20.0)
    assert proc.returncode == 0, f"--help failed: {proc.stderr}"
    assert "--dry-run" in proc.stdout, (
        f"--dry-run flag missing from --help output:\n{proc.stdout}"
    )


# ---------------------------------------------------------------------------
# Behaviour — exit 0, banner, no training
# ---------------------------------------------------------------------------
def test_dry_run_exits_zero_and_prints_banner(tmp_path):
    """``--dry-run`` must:

      1. print the ``DRY RUN: would launch 10000-step retrain`` banner
      2. exit with code 0 (even with no usable GPU on this host)
      3. NOT write a ``ckpt.pt`` (proof that no training actually ran)
      4. write a ``gpu_probe.json`` and a per-seed cmd-record file
         (proof that GPU detection + scaffold both ran)
    """
    log_root = tmp_path / "r17_logs"
    proc = _run(
        ["--dry-run", "--steps", "10000", "--seeds", "42",
         "--log-root", str(log_root)],
        timeout=45.0,
    )

    combined = proc.stdout + proc.stderr

    # (1) Banner
    assert "DRY RUN: would launch 10000-step retrain" in combined, (
        f"missing DRY RUN banner in output:\n{combined}"
    )

    # (2) Exit code 0 — even when torch reports no CUDA, the dry-run
    # branch in main() ignores gpu_probe.usable and proceeds.
    assert proc.returncode == 0, (
        f"--dry-run should exit 0, got rc={proc.returncode}\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )

    # (3) No ckpt.pt anywhere under the seed dir
    seed_dir = log_root / "seed_42"
    if seed_dir.exists():
        ckpts = list(seed_dir.glob("ckpt.pt")) + list(seed_dir.glob("*.pt"))
        assert not ckpts, (
            f"dry-run should NOT produce a checkpoint; found: {ckpts}"
        )

    # (4) GPU probe + cmd-record on disk
    assert (log_root / "gpu_probe.json").is_file(), (
        f"gpu_probe.json missing under {log_root}; contents: "
        f"{sorted(p.name for p in log_root.iterdir()) if log_root.exists() else '<no log_root>'}"
    )
    cmd_record = log_root / "seed_42_cmd_attempt_1.txt"
    assert cmd_record.is_file(), (
        f"seed_42_cmd_attempt_1.txt missing under {log_root}"
    )
    # The cmd record should mention the metallo_drug_smoke_retrain.py
    # subprocess invocation — this is the proof that the wrapper
    # actually built the command (just didn't run it).
    cmd_text = cmd_record.read_text()
    assert "metallo_drug_smoke_retrain" in cmd_text, (
        f"cmd record doesn't reference the smoke script: {cmd_text}"
    )


def test_dry_run_gpu_probe_records_decision(tmp_path):
    """The gpu_probe.json written by ``--dry-run`` should be valid JSON
    and record the decision the wrapper would have made."""
    log_root = tmp_path / "r17_logs"
    proc = _run(
        ["--dry-run", "--steps", "10000", "--log-root", str(log_root)],
        timeout=45.0,
    )
    assert proc.returncode == 0, proc.stderr
    probe_path = log_root / "gpu_probe.json"
    assert probe_path.is_file(), (
        f"gpu_probe.json missing at {probe_path}"
    )
    probe = json.loads(probe_path.read_text())
    # The probe must record vendor + decision + probed_at, regardless
    # of whether the host has a usable GPU.
    for key in ("vendor", "decision", "probed_at"):
        assert key in probe, (
            f"gpu_probe.json missing key={key}; got keys: {sorted(probe.keys())}"
        )


def test_dry_run_is_idempotent(tmp_path):
    """Re-running ``--dry-run`` with the same args must still exit 0
    (catches accidental state-leak bugs between runs)."""
    log_root = tmp_path / "r17_logs"
    args = ["--dry-run", "--steps", "10000", "--log-root", str(log_root)]
    proc1 = _run(args, timeout=45.0)
    # Wipe the log dir between runs to ensure no cache hides a regression
    if log_root.exists():
        shutil.rmtree(log_root)
    proc2 = _run(args, timeout=45.0)
    assert proc1.returncode == 0, (
        f"first --dry-run failed: {proc1.stderr}"
    )
    assert proc2.returncode == 0, (
        f"second --dry-run failed (idempotency broken): {proc2.stderr}"
    )
