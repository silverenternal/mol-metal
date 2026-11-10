"""Wf-A156 — the GPU safety guard's contract.

THE INVARIANT
=============
No TEST or SMOKE surface in this repo may select a CUDA device
AUTOMATICALLY.  "Automatically" means: the device is chosen because a CUDA
device happened to be visible, with no explicit opt-in.  The idiom that
violates it is::

    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WHY IT MATTERS HERE
===================
The chain runs under the standing user directive 禁止使用gpu, which
OVERRIDES the 2026-09-29 standing GPU grant.  The main thread enforces it
by hand with ``CUDA_VISIBLE_DEVICES=""`` — a HUMAN habit, not a repo
invariant.  It already failed once: a full ``pytest molmetal/tests`` run
was MEASURED burning 1444 MiB + 2218 MiB of VRAM because smoke tests
auto-select CUDA.

WHAT THIS TEST PROVES
=====================
1. The guard FLAGS a synthetic file that auto-selects CUDA (exit 1, the
   site is named with file:line).
2. The guard does NOT flag a synthetic file whose CUDA reference sits
   behind an explicit skip guard (exit 0).
3. The guard exits 0 on an empty tree, and the report says it scanned
   nothing (no false green: the count is in the report).
4. The guard exits 2 on a root that is not a directory.
5. The guard's report names, per site: file, line, rule/pattern,
   classification.
6. The guard MODULE never imports torch — it is safe to run inside a
   CPU-locked session and cannot itself allocate VRAM.
7. THE REPO ITSELF IS CLEAN: scanning the real tree finds no auto-CUDA
   site in any TEST/SMOKE surface.  This is the assertion that would have
   caught the 1444+2218 MiB incident.

CPU-ONLY
========
This module never imports torch at module level and never touches CUDA.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "molmetal" / "scripts" / "wfa156_gpu_safety_guard.py"
DEVICE_HELPER = REPO_ROOT / "molmetal" / "_device.py"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write(root: Path, rel: str, source: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _run_guard(root: Path, extra: tuple = ()) -> subprocess.CompletedProcess:
    """Run the guard as a SUBPROCESS (so its exit code is the contract)."""
    return subprocess.run(
        [
            sys.executable,
            str(GUARD_PATH),
            "--root",
            str(root),
            "--json",
            str(root / "report.json"),
            "--quiet",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )


def _report(root: Path) -> dict:
    return json.loads((root / "report.json").read_text(encoding="utf-8"))


#: A file that AUTO-SELECTS CUDA — the violation.
AUTO_CUDA_SOURCE = (
    "import torch\n"
    "\n"
    "\n"
    "def pick_device():\n"
    '    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n'
)

#: A file whose CUDA reference sits behind an EXPLICIT skip guard.
GUARDED_SOURCE = (
    "import pytest\n"
    "import torch\n"
    "\n"
    "\n"
    "@pytest.mark.skipif(not torch.cuda.is_available(), reason=\"no CUDA\")\n"
    "def test_needs_a_gpu():\n"
    '    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n'
    "    assert device is not None\n"
)


# ---------------------------------------------------------------------------
# (1) a violating file IS flagged
# ---------------------------------------------------------------------------
def test_the_guard_flags_an_auto_cuda_file(tmp_path: Path) -> None:
    """A file that auto-selects CUDA must make the guard exit 1."""
    _write(tmp_path, "tests/test_auto_cuda.py", AUTO_CUDA_SOURCE)
    proc = _run_guard(tmp_path)
    assert proc.returncode == 1, (
        f"the guard did not exit 1 on an auto-CUDA file:\n{proc.stdout}\n{proc.stderr}"
    )
    report = _report(tmp_path)
    assert report["auto_cuda_count"] >= 1
    files = report["auto_cuda_files"]
    assert "tests/test_auto_cuda.py" in files


def test_the_flagged_site_names_file_line_and_pattern(tmp_path: Path) -> None:
    """The report must say WHERE, WHAT and HOW — file, line, pattern."""
    _write(tmp_path, "tests/test_auto_cuda.py", AUTO_CUDA_SOURCE)
    proc = _run_guard(tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = _report(tmp_path)
    sites = report["auto_cuda_sites"]
    assert sites, "no auto_cuda_sites were reported"
    site = sites[0]
    assert site["file"] == "tests/test_auto_cuda.py"
    assert site["line"] == 5, f"wrong line reported: {site}"
    assert site["classification"] == "auto_cuda"
    assert site["surface"] == "test"
    assert site["pattern"], "the report does not name the matching pattern"
    assert site["text"], "the report does not carry the source line"


# ---------------------------------------------------------------------------
# (2) a guarded file is NOT flagged
# ---------------------------------------------------------------------------
def test_the_guard_does_not_flag_a_guarded_file(tmp_path: Path) -> None:
    """A skipif-guarded CUDA reference is an EXPLICIT opt-in, not a violation."""
    _write(tmp_path, "tests/test_guarded.py", GUARDED_SOURCE)
    proc = _run_guard(tmp_path)
    assert proc.returncode == 0, (
        f"the guard flagged an explicitly guarded file:\n{proc.stdout}\n{proc.stderr}"
    )
    report = _report(tmp_path)
    assert report["auto_cuda_count"] == 0
    assert report["auto_cuda_files"] == []


# ---------------------------------------------------------------------------
# (3) an empty tree is clean, and says so
# ---------------------------------------------------------------------------
def test_the_guard_exits_zero_on_an_empty_tree(tmp_path: Path) -> None:
    """Nothing to scan → exit 0, and the report says it scanned nothing."""
    (tmp_path / "tests").mkdir(parents=True)
    proc = _run_guard(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = _report(tmp_path)
    assert report["files_scanned"] == 0
    assert report["scanned_nothing"] is True
    assert report["auto_cuda_count"] == 0


# ---------------------------------------------------------------------------
# (4) an unusable root is a SKIP, not a clean bill of health
# ---------------------------------------------------------------------------
def test_the_guard_exits_two_on_a_missing_root(tmp_path: Path) -> None:
    """A root that is not a directory → exit 2 (skip/error), never 0."""
    proc = _run_guard(tmp_path / "does_not_exist")
    assert proc.returncode == 2, (
        f"the guard did not exit 2 on a missing root:\n{proc.stdout}\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# (5) the guard itself never touches the GPU
# ---------------------------------------------------------------------------
def test_the_guard_module_does_not_import_torch() -> None:
    """The guard must be CPU-only: no module-level torch import."""
    source = GUARD_PATH.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import torch", "from torch")):
            raise AssertionError(
                f"the guard imports torch at module level: {stripped!r}"
            )


def test_the_guard_runs_with_torch_blocked(tmp_path: Path) -> None:
    """Importing the guard must not import torch, even transitively."""
    _write(tmp_path, "tests/test_auto_cuda.py", AUTO_CUDA_SOURCE)
    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch is blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'molmetal' / 'scripts')!r})\n"
        "import wfa156_gpu_safety_guard as g\n"
        "assert 'torch' not in sys.modules, 'importing the guard pulled in torch'\n"
        "code = g.main(['--root', %r, '--quiet'])\n"
        "assert code == 1, code\n"
        "print('OK')\n"
    ) % str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# (6) the repo itself is clean
# ---------------------------------------------------------------------------
def test_the_repo_has_no_auto_cuda_test_surface() -> None:
    """THE INVARIANT: no TEST/SMOKE surface auto-selects CUDA.

    This is the assertion that would have caught the MEASURED
    1444 MiB + 2218 MiB VRAM incident of 2026-10-31.
    """
    proc = _run_guard(REPO_ROOT)
    report = _report(REPO_ROOT)
    offenders = [
        f"{s['file']}:{s['line']} [{s['rule']}] {s['text']}"
        for s in report["auto_cuda_sites"]
        if s["surface"] in ("test", "smoke")
    ]
    assert proc.returncode == 0, (
        "the guard found auto-CUDA sites in a TEST/SMOKE surface:\n"
        + "\n".join(offenders)
    )
    assert report["files_scanned"] > 0, "the guard scanned nothing — a false green"
    assert report["scanned_nothing"] is False


# ---------------------------------------------------------------------------
# (7) the canonical resolver: CPU in a test context
# ---------------------------------------------------------------------------
def test_resolve_device_is_cpu_inside_a_test_context() -> None:
    """``molmetal._device.resolve_device()`` must return CPU from a test."""
    from molmetal._device import resolve_device

    device = resolve_device()
    assert device.type == "cpu", (
        f"resolve_device() returned {device} inside a test surface"
    )


def test_resolve_device_can_be_forced() -> None:
    """``allow_gpu=False`` forces CPU without probing the driver."""
    from molmetal._device import resolve_device

    assert resolve_device(allow_gpu=False).type == "cpu"


def test_cuda_device_indices_is_empty_in_a_test_context() -> None:
    """``fork_rng`` must never be handed a CUDA RNG state from a test."""
    from molmetal._device import cuda_device_indices

    assert cuda_device_indices() == []
