"""Wf-A125 G9 — the GPU-required skips must not be SILENT.

THE GAP (measured 2026-10-31, before this wave).  ``grep -rn
pytest_terminal_summary molmetal/ tests/ pyproject.toml`` returned **0 hits**
repo-wide, and the two GPU-REQUIRED modules skip via ``skipif``.  A CPU
session therefore exited 0 and printed NOTHING about the nine GPU-required
tests that did not run — the skip was invisible.

THE FIX.  The repo-root ``conftest.py`` (created for Wf-A125 G1) marks every
item of the GPU-REQUIRED modules with the already-declared ``gpu`` marker
(``pyproject.toml:151``) and prints EXACTLY ONE machine-readable line per
session::

    mol-metal GPU contract: N GPU-required test(s) did not run (...)

``N`` is the first integer on the line.  The exit code is never touched.

THE PROOF (this file).  (a) a CPU-only pytest run over the GPU-REQUIRED
modules prints the line with a non-zero N and the phrase "did not run";
(b) the N on the line equals the number of gpu-marked skipped items pytest
itself reports; (c) the root conftest's ``GPU_REQUIRED_MODULES`` and the
Wf-A123 contract test's ``GPU_REQUIRED`` list agree, so the two copies of the
list cannot drift apart silently.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFTEST = REPO_ROOT / "conftest.py"
TESTS_DIR = REPO_ROOT / "molmetal" / "tests"

#: The Wf-A123 contract test — READ ONLY for this wave (it carries the same
#: module list; the new test asserts the two agree).
A123_CONTRACT = TESTS_DIR / "test_wfa123_gpu_contract.py"

#: The fixed prefix of the one machine-readable line this wave adds.
CONTRACT_PREFIX = "mol-metal GPU contract:"

#: ``mol-metal GPU contract: 9 GPU-required test(s) did not run (...)``
CONTRACT_LINE_RE = re.compile(
    r"mol-metal GPU contract: (\d+) GPU-required test\(s\) did not run"
)

#: pytest's own tail: ``9 skipped, 1 warning in 0.58s``
PYTEST_SKIPPED_RE = re.compile(r"(\d+) skipped")


def _cpu_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("MOLMETAL_TEST_ALLOW_GPU", None)
    return env


def _pytest(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_cpu_env(),
        timeout=600,
    )


def _load_root_conftest():
    """Import the repo-root conftest as a module (without installing it)."""
    spec = importlib.util.spec_from_file_location("wfa125_root_conftest", ROOT_CONFTEST)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _a123_gpu_required() -> list:
    """AST-extract ``GPU_REQUIRED`` from the Wf-A123 contract test."""
    tree = ast.parse(A123_CONTRACT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GPU_REQUIRED":
                    return [ast.literal_eval(el) for el in node.value.elts]
    raise AssertionError("GPU_REQUIRED is not a module-level list in the A123 file")


# ---------------------------------------------------------------------------
# (a) the line is printed, with a non-zero count and the right phrase
# ---------------------------------------------------------------------------
def test_the_gpu_contract_line_is_printed_with_a_non_zero_count() -> None:
    """A CPU-only session must SAY that the GPU-required tests did not run."""
    proc = _pytest(
        "molmetal/tests/test_wf25_gpu_smoke.py",
        "molmetal/tests/test_rocm_egnn.py",
    )
    out = proc.stdout
    assert proc.returncode == 0, out[-4000:]
    match = CONTRACT_LINE_RE.search(out)
    assert match is not None, (
        "the GPU-contract summary line is missing:\n" + out[-4000:]
    )
    assert int(match.group(1)) > 0, (
        "the GPU-contract line reports 0 GPU-required skips, but the two "
        "GPU-REQUIRED modules skip on a CPU-only session:\n" + out[-4000:]
    )
    assert "did not run" in match.group(0), match.group(0)


# ---------------------------------------------------------------------------
# the line is machine-readable: fixed prefix, count as the FIRST integer
# ---------------------------------------------------------------------------
def test_the_contract_line_is_machine_readable() -> None:
    """The count is the first integer on a line with a fixed prefix."""
    proc = _pytest("molmetal/tests/test_wf25_gpu_smoke.py")
    out = proc.stdout
    lines = [ln for ln in out.splitlines() if ln.startswith(CONTRACT_PREFIX)]
    assert len(lines) == 1, (
        f"expected exactly one GPU-contract line, got {len(lines)}:\n" + out[-4000:]
    )
    line = lines[0]
    numbers = re.findall(r"\d+", line)
    assert numbers, f"the GPU-contract line carries no count: {line!r}"
    assert line.split(":", 1)[1].strip().split()[0] == numbers[0], (
        f"the count is not the first integer on the line: {line!r}"
    )


# ---------------------------------------------------------------------------
# (b) the count equals the number of gpu-marked skips pytest reports
# ---------------------------------------------------------------------------
def test_the_count_matches_pytests_own_gpu_marked_skip_count() -> None:
    """The reported N must equal pytest's own count of gpu-marked skips."""
    proc = _pytest(
        "-m",
        "gpu",
        "molmetal/tests/test_wf25_gpu_smoke.py",
        "molmetal/tests/test_rocm_egnn.py",
    )
    out = proc.stdout
    assert proc.returncode == 0, out[-4000:]
    match = CONTRACT_LINE_RE.search(out)
    assert match is not None, out[-4000:]
    skipped = PYTEST_SKIPPED_RE.search(out)
    assert skipped is not None, (
        "pytest reported no skip count:\n" + out[-4000:]
    )
    assert int(match.group(1)) == int(skipped.group(1)), (
        f"the GPU-contract line says {match.group(1)} but pytest reports "
        f"{skipped.group(1)} gpu-marked skips:\n" + out[-4000:]
    )


# ---------------------------------------------------------------------------
# the gpu marker really is applied (and --strict-markers stays happy)
# ---------------------------------------------------------------------------
def test_the_gpu_marker_selects_the_gpu_required_modules() -> None:
    """Every item of the GPU-REQUIRED modules carries the ``gpu`` marker."""
    proc = _pytest(
        "--collect-only",
        "-m",
        "gpu",
        "molmetal/tests/test_wf25_gpu_smoke.py",
        "molmetal/tests/test_rocm_egnn.py",
    )
    out = proc.stdout
    assert proc.returncode == 0, out[-4000:]
    assert "error" not in out.lower(), (
        "the gpu marker is not registered (--strict-markers):\n" + out[-4000:]
    )
    collected = [
        ln for ln in out.splitlines() if ln.startswith("molmetal/tests/")
    ]
    assert len(collected) == 9, (
        f"expected 9 gpu-marked items, pytest selected {len(collected)}:\n"
        + out[-4000:]
    )


# ---------------------------------------------------------------------------
# (c) the root conftest and the A123 contract test agree on the list
# ---------------------------------------------------------------------------
def test_the_root_conftest_list_matches_the_a123_contract() -> None:
    """The ONE place the list lives vs the Wf-A123 copy — they must agree."""
    if not A123_CONTRACT.is_file():
        pytest.skip("the Wf-A123 contract test is not present in this tree")
    root = _load_root_conftest()
    root_names = sorted(Path(p).name for p in root.GPU_REQUIRED_MODULES)
    a123_names = sorted(_a123_gpu_required())
    assert root_names == a123_names, (
        "the GPU-REQUIRED module list has drifted: root conftest "
        f"{root_names} vs Wf-A123 contract {a123_names}"
    )


def test_every_gpu_required_module_exists() -> None:
    """The list must point at real files."""
    root = _load_root_conftest()
    for rel in root.GPU_REQUIRED_MODULES:
        path = REPO_ROOT / rel
        assert path.is_file(), f"declared GPU-REQUIRED module is missing: {rel}"


# ---------------------------------------------------------------------------
# the line must never change a session's exit code
# ---------------------------------------------------------------------------
def test_the_gpu_contract_line_does_not_change_the_exit_code() -> None:
    """A CPU-only session that skips the GPU tests is a PASSING session."""
    proc = _pytest("molmetal/tests/test_wf25_gpu_smoke.py")
    assert proc.returncode == 0, (
        f"the GPU-contract reporting changed the exit code to {proc.returncode}:\n"
        + proc.stdout[-4000:]
    )
    assert CONTRACT_LINE_RE.search(proc.stdout) is not None
