#!/usr/bin/env python3
"""Tests for the Wf-A165 GPU lock-defeater guard.

The guard is ``molmetal/scripts/wfa165_gpu_lock_guard.py``: a READ-ONLY,
CPU-ONLY static scanner that reports every repo site that can DEFEAT an
ambient ``CUDA_VISIBLE_DEVICES=""`` session lock.

Four obligations are asserted here:

1. NEGATIVE CONTROL — a synthetic file containing each of the four
   mechanisms MUST be detected, and each detection MUST be reported.
2. POSITIVE CONTROL — the lock-preserving forms
   (``${CUDA_VISIBLE_DEVICES-0}``, a child that inherits ``os.environ``)
   MUST NOT be flagged.
3. VACUITY GUARD — the number of findings the guard reports for a
   fixture MUST equal the number it lists.  A guard that reports a count
   but lists fewer is broken.
4. NO GPU — the guard MUST run with ``CUDA_VISIBLE_DEVICES=""`` and MUST
   NOT allocate GPU memory.  This is asserted STATICALLY, by an AST check
   over the guard's own import graph (the same shape as
   ``molmetal/tests/test_wfa162_narrative_self_consistency.py``): if the
   guard never imports torch, it cannot reach a CUDA context.

The guard is loaded by FILE PATH, never as ``molmetal.scripts.<name>``:
importing the ``molmetal`` package would drag in its own import-time
environment bootstrap, which is exactly the coupling this test must not
create.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "molmetal" / "scripts" / "wfa165_gpu_lock_guard.py"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

M1 = "M1_hard_override"
M2 = "M2_empty_default_substitution"
M3 = "M3_env_dropping_spawn"
M4 = "M4_mask_dropping_default"


def _load_guard():
    """Load the guard from its file path, bypassing the molmetal package."""
    spec = importlib.util.spec_from_file_location("_wfa165_gpu_lock_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


# ---------------------------------------------------------------------------
# Fixtures — the synthetic repo used by the controls
# ---------------------------------------------------------------------------

NEGATIVE_SH = """#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=1 /usr/bin/python train.py
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"""

POSITIVE_SH = """#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
echo "CUDA_VISIBLE_DEVICES=0 is only a message"
"""

NEGATIVE_PY = '''import os
import subprocess


def m3a_env_no_inherit():
    subprocess.run(["python", "x.py"], env={"PATH": "/usr/bin"}, check=False)


def m3b_environ_nonblank_literal():
    os.environ["CUDA_VISIBLE_DEVICES"] = "auto"


def m4a_child_env_empty():
    subprocess.run(["x"], env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/bin"})


def m4b_environ_index_literal():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''

POSITIVE_PY = '''import os
import subprocess


def inherits_via_unpack():
    subprocess.run(["x"], env={**os.environ, "OMP_NUM_THREADS": "1"})


def inherits_via_dict():
    subprocess.run(["x"], env=dict(os.environ))


def inherits_via_comprehension():
    subprocess.run(["x"], env={k: v for k, v in os.environ.items() if k != "PY"})


def inherits_via_copy():
    env = os.environ.copy()
    subprocess.run(["x"], env=env)


def env_is_a_parameter(env=None):
    subprocess.run(["x"], env=env)
'''


def _write_repo(root: Path, files: Dict[str, str]) -> Path:
    """Create a tiny git repo with the given files; return its root."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for rel, body in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


@pytest.fixture()
def negative_repo(tmp_path: Path) -> Path:
    return _write_repo(
        tmp_path / "neg",
        {"bad.sh": NEGATIVE_SH, "bad.py": NEGATIVE_PY},
    )


@pytest.fixture()
def positive_repo(tmp_path: Path) -> Path:
    return _write_repo(
        tmp_path / "pos",
        {"good.sh": POSITIVE_SH, "good.py": POSITIVE_PY},
    )


def _run_guard(repo_root: Path, *extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [str(VENV_PYTHON), str(GUARD_PATH), "--repo-root", str(repo_root),
         "--json", *extra],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# 1. Negative control — every mechanism is detected and reported
# ---------------------------------------------------------------------------


def test_negative_control_detects_all_four_mechanisms(negative_repo: Path):
    """Each of the four mechanisms in the fixture must be found AND listed."""
    report = guard.scan_repository(negative_repo)
    counts = {m: report["mechanisms"][m]["count"] for m in (M1, M2, M3, M4)}
    assert counts[M1] >= 2, f"M1 hard override not detected: {counts}"
    assert counts[M2] >= 1, f"M2 empty-default substitution not detected: {counts}"
    assert counts[M3] >= 1, f"M3 env-dropping spawn not detected: {counts}"
    assert counts[M4] >= 2, f"M4 mask-dropping default not detected: {counts}"


def test_negative_control_each_site_is_listed_with_its_mechanism(negative_repo: Path):
    """Every reported finding must carry file, line, source, mechanism, class."""
    report = guard.scan_repository(negative_repo)
    findings = report["findings"]
    assert findings, "the negative fixture produced no findings at all"
    for rec in findings:
        assert rec["file"], "finding has no file"
        assert isinstance(rec["line"], int) and rec["line"] > 0, "finding has no line"
        assert rec["source"], f"finding has no source line: {rec}"
        assert rec["mechanism"] in (M1, M2, M3, M4), rec["mechanism"]
        assert rec["class"], "finding has no class"
    # the four mechanisms must be represented among the listed findings
    seen = {rec["mechanism"] for rec in findings}
    assert seen == {M1, M2, M3, M4}, f"not all mechanisms present: {seen}"


def test_negative_control_cli_reports_the_same_four_mechanisms(negative_repo: Path):
    """The CLI path must report the same mechanisms as the library path."""
    proc = _run_guard(negative_repo)
    assert proc.returncode == 1, f"expected exit 1 with findings: {proc.stderr}"
    report = json.loads(proc.stdout)
    seen = {rec["mechanism"] for rec in report["findings"]}
    assert seen == {M1, M2, M3, M4}, f"not all mechanisms present via CLI: {seen}"


# ---------------------------------------------------------------------------
# 2. Positive control — lock-preserving forms are NOT flagged
# ---------------------------------------------------------------------------


def test_positive_control_preserving_shell_forms_are_not_flagged(positive_repo: Path):
    """`${VAR-0}` (no colon) preserves an explicitly-empty mask: not a defeater."""
    findings = guard.scan_shell_text(POSITIVE_SH)
    assert findings == [], f"lock-preserving shell forms were flagged: {findings}"


def test_positive_control_inheriting_children_are_not_flagged(positive_repo: Path):
    """A child that inherits os.environ keeps the ambient mask."""
    findings = guard.scan_python_text(POSITIVE_PY)
    assert findings == [], f"lock-preserving python forms were flagged: {findings}"


def test_positive_control_repo_reports_zero_findings(positive_repo: Path):
    report = guard.scan_repository(positive_repo)
    assert report["findings_total"] == 0, report["findings"]
    proc = _run_guard(positive_repo)
    assert proc.returncode == 0, "a clean tree must exit 0"


# ---------------------------------------------------------------------------
# 3. Vacuity guard — the reported count equals the number listed
# ---------------------------------------------------------------------------


def test_vacuity_guard_count_equals_number_listed(negative_repo: Path):
    """A guard that reports a count but lists fewer is broken."""
    report = guard.scan_repository(negative_repo)
    for mech in (M1, M2, M3, M4):
        listed = report["findings_by_mechanism"][mech]
        counted = report["mechanisms"][mech]["count"]
        assert counted == len(listed), (
            f"{mech}: reported {counted} but listed {len(listed)}"
        )
    assert report["findings_total"] == len(report["findings"])
    assert report["counts_match_lists"] is True


def test_vacuity_guard_holds_on_the_real_tree():
    """The same invariant must hold on the real repository."""
    report = guard.scan_repository(REPO_ROOT)
    for mech in (M1, M2, M3, M4):
        listed = report["findings_by_mechanism"][mech]
        assert report["mechanisms"][mech]["count"] == len(listed), mech
    assert report["findings_total"] == len(report["findings"])
    assert report["counts_match_lists"] is True


# ---------------------------------------------------------------------------
# 4. No GPU — the guard cannot allocate GPU memory
# ---------------------------------------------------------------------------


def test_the_guard_does_not_import_torch():
    """STATIC: the guard's import graph has no torch, so it cannot touch a GPU."""
    src = GUARD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    assert "torch" not in roots, f"guard imports torch: {sorted(roots)}"


def test_the_guard_imports_only_the_standard_library():
    """Every top-level import of the guard must be stdlib."""
    import sys as _sys

    stdlib = set(_sys.stdlib_module_names)
    src = GUARD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    non_stdlib = {r for r in roots if r not in stdlib and r != "__future__"}
    assert not non_stdlib, f"guard imports non-stdlib modules: {sorted(non_stdlib)}"


def test_the_guard_runs_with_a_blank_cuda_mask_and_no_gpu_allocation():
    """The guard must run under CUDA_VISIBLE_DEVICES='' and allocate nothing.

    The allocation claim is asserted statically (no torch in the import
    graph) plus behaviourally: the child process must report a zero-length
    visible-device list, and the guard must still complete.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [str(VENV_PYTHON), "-c",
         "import os,sys;"
         "print(repr(os.environ.get('CUDA_VISIBLE_DEVICES')));"
         "sys.exit(0)"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "''", proc.stdout

    guard_proc = _run_guard(REPO_ROOT, "--advisory")
    assert guard_proc.returncode == 0, guard_proc.stderr[-2000:]
    report = json.loads(guard_proc.stdout)
    assert report["denominator"]["files_read"] > 0


# ---------------------------------------------------------------------------
# 5. Interface contract
# ---------------------------------------------------------------------------


def test_advisory_flag_forces_exit_zero_while_still_reporting(negative_repo: Path):
    proc = _run_guard(negative_repo, "--advisory")
    assert proc.returncode == 0, "advisory mode must exit 0"
    report = json.loads(proc.stdout)
    assert report["findings_total"] > 0, "advisory mode must still report findings"


def test_the_report_carries_the_denominator(negative_repo: Path):
    """A claim without a denominator is not a measurement."""
    report = guard.scan_repository(negative_repo)
    den = report["denominator"]
    assert den["files_enumerated"] == 2, den
    assert den["files_read"] == 2, den
    assert den["files_unreadable"] == 0, den
    assert den["py_files"] == 1 and den["sh_files"] == 1, den


def test_out_flag_writes_json_atomically_into_the_given_path(negative_repo: Path, tmp_path: Path):
    out = tmp_path / "census.json"
    proc = _run_guard(negative_repo, "--advisory", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.exists(), "the --out path was not written"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tool"] == "wfa165_gpu_lock_guard"
    assert payload["findings_total"] > 0
    # no temp file left behind
    leftovers = [p.name for p in tmp_path.iterdir()
                 if p.is_file() and p.name != "census.json"]
    assert not leftovers, f"atomic write left leftovers: {leftovers}"


def test_the_guard_never_walks_the_filesystem():
    """The guard must use `git ls-files`, never os.walk (this repo is 201 GB)."""
    src = GUARD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    walks = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "walk"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"):
            walks.append(getattr(node, "lineno", 0))
    assert not walks, f"the guard calls os.walk at lines {walks}"
    assert "ls-files" in src, "the guard must enumerate via git ls-files"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
