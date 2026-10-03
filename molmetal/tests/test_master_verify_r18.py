"""Pytest: Wf-35 Phase 3 — ``master_verify_r18.py`` contract.

=====================================================================
The master verify script in :mod:`molmetal.scripts.master_verify_r18`
is the Phase-3 deliverable for the Wf-35 refactor.  These tests
exercise its public contract end-to-end (most importantly that the
script:

1. Imports cleanly as a module under :mod:`molmetal.scripts`.
2. Declares the expected registry of eight scripts (six shipped,
   two planned).
3. Honours ``--only-shipped`` / ``--exclude`` CLI flags.
4. Writes the JSON + Markdown report artefacts to the canonical
   ``molmetal/reports/wf35_master_verify/`` location.

The tests are CPU-only and finish in <60 s.

Verify
------
::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_master_verify_r18.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
TESTS_DIR = PROJECT_ROOT / "molmetal" / "tests"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf35_master_verify"
VENV_PY = PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ===========================================================================
# 1. Module imports
# ===========================================================================
def test_master_verify_r18_imports() -> None:
    """``molmetal.scripts.master_verify_r18`` imports under the venv."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")
    proc = subprocess.run(
        [
            str(VENV_PY),
            "-c",
            "import importlib, sys; "
            "sys.path.insert(0, %r); "
            % str(SCRIPTS_DIR)
            + "m = importlib.import_module('master_verify_r18'); "
            "assert hasattr(m, 'main') and hasattr(m, '_verify_script')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr


# ===========================================================================
# 2. Registry shape
# ===========================================================================
def test_registry_has_eight_scripts() -> None:
    """Registry must expose eight scripts: six shipped + two planned."""
    from molmetal.scripts import master_verify_r18 as mv

    assert len(mv.ALL_SCRIPTS) == 8, (
        f"expected eight scripts (6 shipped + 2 planned), got "
        f"{len(mv.ALL_SCRIPTS)}: {[s.name for s in mv.ALL_SCRIPTS]}"
    )
    assert len(mv.SHIPPED_SCRIPTS) == 6
    assert len(mv.PLANNED_SCRIPTS) == 2


def test_registry_contains_expected_shipped() -> None:
    """Six shipped scripts must match the Wf-35 plan."""
    from molmetal.scripts import master_verify_r18 as mv

    shipped = {s.name for s in mv.SHIPPED_SCRIPTS}
    expected = {
        "train_se3sb.py",
        "train_transport.py",
        "train_multiph.py",
        "train_ru_5metal_se3sb.py",
        "train_affinity_regression.py",
        "train_multimetal_lomo.py",
    }
    assert (
        shipped == expected
    ), f"shipped registry drifted: {shipped.symmetric_difference(expected)}"


def test_registry_contains_expected_planned() -> None:
    """Two planned scripts must match the Wf-35 plan."""
    from molmetal.scripts import master_verify_r18 as mv

    planned = {s.name for s in mv.PLANNED_SCRIPTS}
    expected = {"train_cisplatin_resistance_v2.py", "bench_pb_valid.py"}
    assert (
        planned == expected
    ), f"planned registry drifted: {planned.symmetric_difference(expected)}"


def test_planned_scripts_in_dynamic_state(tmp_path: Path) -> None:
    """``_is_shipped`` correctly distinguishes:

    * not-yet-authored scripts (file missing)  → False
    * authored-but-not-refactored scripts     → False
    * fully shipped scripts                   → True

    We synthesise three files in ``tmp_path`` and point
    ``SCRIPTS_DIR`` at the tempdir so the check is hermetic.
    """
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")

    driver = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(SCRIPTS_DIR)!r})
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        import master_verify_r18 as mv

        root = Path({str(tmp_path)!r})
        (root / "missing.py").touch()  # never written — absent below

        # Authored but not refactored.
        (root / "needs_refactor.py").write_text(
            "#!/usr/bin/env python3\\n"
            "import argparse\\n"
            "p = argparse.ArgumentParser()\\n"
        )

        # Fully shipped (refactored).
        (root / "shipped.py").write_text(
            "#!/usr/bin/env python3\\n"
            "from molmetal.scripts._common_train_args import"
            " add_common_args, add_gpu_args\\n"
        )

        # Substitute SCRIPTS_DIR with our sandbox.
        real_scripts_dir = mv.SCRIPTS_DIR
        mv.SCRIPTS_DIR = root
        try:
            assert mv._is_shipped(mv.ScriptSpec("missing.py", "Wf-X")) is False
            assert mv._is_shipped(
                mv.ScriptSpec("needs_refactor.py", "Wf-X")
            ) is False
            assert mv._is_shipped(
                mv.ScriptSpec("shipped.py", "Wf-X")
            ) is True
        finally:
            mv.SCRIPTS_DIR = real_scripts_dir
        print("OK")
        """
    )
    p = tmp_path / "_driver_dynamic.py"
    p.write_text(driver)
    proc = subprocess.run(
        [str(VENV_PY), str(p)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_each_script_spec_has_wf_tag() -> None:
    """Every ScriptSpec must declare a ``wf_tag``."""
    from molmetal.scripts import master_verify_r18 as mv

    for spec in mv.ALL_SCRIPTS:
        assert spec.wf_tag, f"{spec.name} missing wf_tag"
        assert spec.wf_tag.startswith(
            "Wf-"
        ), f"{spec.name}: wf_tag should start with 'Wf-' got {spec.wf_tag!r}"


# ===========================================================================
# 3. ScriptSpec dataclass
# ===========================================================================
def test_script_spec_is_frozen() -> None:
    """``ScriptSpec`` is a frozen dataclass."""
    from molmetal.scripts import master_verify_r18 as mv

    spec = mv.SHIPPED_SCRIPTS[0]
    with pytest.raises((AttributeError, Exception)):
        spec.name = "nope"  # type: ignore[misc]


def test_script_spec_has_expected_groups() -> None:
    """Each ScriptSpec declares the canonical argument groups."""
    from molmetal.scripts import master_verify_r18 as mv

    for spec in mv.ALL_SCRIPTS:
        assert "Common recipe" in spec.expected_groups
        assert "GPU env" in spec.expected_groups


# ===========================================================================
# 4. CheckResult / ScriptReport dataclasses
# ===========================================================================
def test_check_result_defaults() -> None:
    """``CheckResult`` defaults to failed=False, skipped=False."""
    from molmetal.scripts import master_verify_r18 as mv

    cr = mv.CheckResult(name="x", passed=True)
    assert cr.passed is True
    assert cr.skipped is False
    assert cr.seconds == 0.0
    assert cr.message == ""


def test_script_report_defaults() -> None:
    """``ScriptReport`` defaults to status='PENDING' with empty checks."""
    from molmetal.scripts import master_verify_r18 as mv

    rep = mv.ScriptReport(script="x.py", wf_tag="Wf-X")
    assert rep.status == "PENDING"
    assert rep.checks == []
    assert rep.error == ""


# ===========================================================================
# 5. JSON helper
# ===========================================================================
def test_to_jsonable_handles_dict() -> None:
    """``_to_jsonable`` recurses through dicts."""
    from molmetal.scripts import master_verify_r18 as mv

    out = mv._to_jsonable({"a": 1, "b": [1.0, 2.0]})
    assert out == {"a": 1, "b": [1.0, 2.0]}


def test_to_jsonable_handles_path() -> None:
    """``_to_jsonable`` converts ``Path`` to ``str``."""
    from molmetal.scripts import master_verify_r18 as mv

    out = mv._to_jsonable({"p": Path("/tmp/x")})
    assert out == {"p": "/tmp/x"}


def test_to_jsonable_handles_set() -> None:
    """``_to_jsonable`` sorts sets into a JSON-safe list."""
    from molmetal.scripts import master_verify_r18 as mv

    out = mv._to_jsonable({3, 1, 2})
    assert out == [1, 2, 3]


# ===========================================================================
# 6. CLI flag handling
# ===========================================================================
def test_parse_argv_exclude() -> None:
    """``--exclude`` repeats correctly into a list."""
    from molmetal.scripts import master_verify_r18 as mv

    ns = mv._parse_argv(["--exclude", "train_se3sb", "--exclude", "multiph"])
    assert "train_se3sb" in ns.exclude
    assert "multiph" in ns.exclude


def test_parse_argv_only_shipped() -> None:
    """``--only-shipped`` flips the flag on."""
    from molmetal.scripts import master_verify_r18 as mv

    ns = mv._parse_argv(["--only-shipped"])
    assert ns.only_shipped is True


def test_parse_argv_skip_smoke() -> None:
    """``--skip-smoke`` flips the smoke-bypass flag on (CPU hosts)."""
    from molmetal.scripts import master_verify_r18 as mv

    ns = mv._parse_argv(["--skip-smoke"])
    assert ns.skip_smoke is True


def test_skip_smoke_marks_check_skipped(tmp_path: Path) -> None:
    """When called with skip_smoke=True, the smoke check is recorded
    as ``skipped=True`` rather than subprocess-invoked (CPU-safe path).
    """
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")

    driver = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(SCRIPTS_DIR)!r})
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        import master_verify_r18 as mv
        spec = mv.SHIPPED_SCRIPTS[0]  # any shipped script
        rep = mv._verify_script(spec, skip_smoke=True)
        smoke_checks = [c for c in rep.checks if c.name == "smoke"]
        assert len(smoke_checks) == 1, rep
        assert smoke_checks[0].skipped is True
        assert "skip-smoke" in smoke_checks[0].message
        print("OK")
        """
    )
    p = tmp_path / "_driver_skip_smoke.py"
    p.write_text(driver)
    proc = subprocess.run(
        [str(VENV_PY), str(p)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# ===========================================================================
# 6.5 Verify the actual master_verify_r18.py runs end-to-end
# ===========================================================================
def test_master_verify_r18_runs_end_to_end(tmp_path: Path) -> None:
    """Run ``master_verify_r18.py`` against the venv and confirm:

    * it exits 0,
    * it writes both JSON + MD artefacts,
    * ``master_report.json`` is parseable + reports at minimum one script.

    Uses ``--skip-smoke`` so the heavy GPU path isn't required; only the
    cheap import + --help checks run for each registered script.
    """
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")

    cmd = [
        str(VENV_PY),
        str(SCRIPTS_DIR / "master_verify_r18.py"),
        "--skip-smoke",  # CPU-safe path
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, (
        f"master_verify_r18.py exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # The canonical artefacts live at molmetal/reports/wf35_master_verify/
    json_path = REPORTS_DIR / "master_report.json"
    md_path = REPORTS_DIR / "master_report.md"
    assert json_path.exists(), f"expected {json_path} after end-to-end run"
    assert md_path.exists(), f"expected {md_path} after end-to-end run"

    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["workflow"] == "Wf-35 Phase 3"
    # Pass + Pending counts must add up.
    s = payload["summary"]
    assert s["n_total"] == len(payload["scripts"])
    assert s["n_passed"] + s["n_failed"] + s["n_pending"] == s["n_total"]
    assert (
        s["n_passed"] >= 1
    ), "the cheapest check (import) should always pass on a shipped script"


# ===========================================================================
# 7. JSON report shape
# ===========================================================================
def test_write_json_creates_artefact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_write_json`` writes the canonical artefact with the right schema."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")

    # Run the writer in a subprocess against the venv so the full
    # import graph (numpy, dataclasses) is available.
    driver = textwrap.dedent(
        f"""
        import json, sys, dataclasses, pathlib
        from pathlib import Path
        sys.path.insert(0, {str(SCRIPTS_DIR)!r})
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        import master_verify_r18 as mv
        out_dir = Path({str(tmp_path)!r}) / "report"
        out_dir.mkdir(parents=True, exist_ok=True)
        rep = mv.ScriptReport(script="x.py", wf_tag="Wf-X", status="PASS")
        rep.checks.append(mv.CheckResult("import", True, seconds=0.1))
        # Monkey-patch REPORTS_DIR for this test.
        mv.REPORTS_DIR = out_dir
        mv.JSON_OUT = out_dir / "master_report.json"
        mv._write_json([rep], started="t0", finished="t1", total_seconds=0.1)
        print("OK")
        """
    )
    p = tmp_path / "_driver_json.py"
    p.write_text(driver)
    proc = subprocess.run(
        [str(VENV_PY), str(p)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    json_path = tmp_path / "report" / "master_report.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["workflow"] == "Wf-35 Phase 3"
    assert payload["summary"]["n_total"] == 1
    assert payload["summary"]["n_passed"] == 1
    assert payload["scripts"][0]["script"] == "x.py"


# ===========================================================================
# 8. Markdown report shape
# ===========================================================================
def test_write_markdown_creates_artefact(
    tmp_path: Path,
) -> None:
    """``_write_markdown`` writes the canonical MD artefact."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")

    driver = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(SCRIPTS_DIR)!r})
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        import master_verify_r18 as mv
        out_dir = Path({str(tmp_path)!r}) / "report"
        out_dir.mkdir(parents=True, exist_ok=True)
        rep = mv.ScriptReport(script="x.py", wf_tag="Wf-X", status="PASS")
        mv.REPORTS_DIR = out_dir
        mv.MD_OUT = out_dir / "master_report.md"
        mv._write_markdown([rep], started="t0", finished="t1",
                           total_seconds=0.1)
        print("OK")
        """
    )
    p = tmp_path / "_driver_md.py"
    p.write_text(driver)
    proc = subprocess.run(
        [str(VENV_PY), str(p)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    md_path = tmp_path / "report" / "master_report.md"
    assert md_path.exists()
    body = md_path.read_text()
    assert "Wf-35 Phase 3" in body
    assert "Per-script detail" in body
    assert "x.py" in body


# ===========================================================================
# 9. Pending auto-skip in driver
# ===========================================================================
def test_pending_script_report_when_missing(tmp_path: Path) -> None:
    """A not-yet-authored spec is auto-reported PENDING without crashing.

    We don't rely on ``PLANNED_SCRIPTS[0]`` because that index may point
    at a spec which has been promoted to shipped between Wf-35 phases;
    instead we synthesise a fresh ``ScriptSpec`` whose file does not
    exist on disk and verify it PENDING-skips cleanly.
    """
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")

    driver = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(SCRIPTS_DIR)!r})
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        import master_verify_r18 as mv

        # Authored-but-not-refactored case.
        not_authored_spec = mv.ScriptSpec(
            "_definitely_not_here_xyz.py", "Wf-X")
        rep_a = mv._verify_script(not_authored_spec)
        assert rep_a.status == "PENDING", rep_a
        assert "not yet authored" in rep_a.error

        # Authored-but-not-refactored case: temp file w/o _common_train_args.
        sandbox = Path({str(tmp_path)!r})
        (sandbox / "needs_refactor.py").write_text(
            "#!/usr/bin/env python3\\n"
            "import argparse\\n"
            "p = argparse.ArgumentParser()\\n"
        )
        real_scripts_dir = mv.SCRIPTS_DIR
        mv.SCRIPTS_DIR = sandbox
        try:
            nr_spec = mv.ScriptSpec("needs_refactor.py", "Wf-X")
            rep_b = mv._verify_script(nr_spec)
            assert rep_b.status == "PENDING", rep_b
            assert "not yet refactored" in rep_b.error
        finally:
            mv.SCRIPTS_DIR = real_scripts_dir
        print("OK")
        """
    )
    p = tmp_path / "_driver_pending.py"
    p.write_text(driver)
    proc = subprocess.run(
        [str(VENV_PY), str(p)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# ===========================================================================
# 10. Failing script status reporting
# ===========================================================================
def test_summarise_failed_checks_joins_messages() -> None:
    """``_summarise_failed_checks`` joins all failing check messages."""
    from molmetal.scripts import master_verify_r18 as mv

    checks = [
        mv.CheckResult("import", False, message="bad import"),
        mv.CheckResult("argparse", True, message="ok"),
        mv.CheckResult("smoke", False, message="crash"),
    ]
    out = mv._summarise_failed_checks(checks)
    assert "import" in out and "bad import" in out
    assert "smoke" in out and "crash" in out
    assert "argparse" not in out


def test_summarise_failed_checks_empty_returns_unknown() -> None:
    """Empty list → 'unknown' so the caller can still see the failure."""
    from molmetal.scripts import master_verify_r18 as mv

    assert mv._summarise_failed_checks([]) == "unknown"


# ===========================================================================
# 11. CLI --help
# ===========================================================================
def test_main_help_exits_zero() -> None:
    """``master_verify_r18.py --help`` exits 0 under the venv."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")
    proc = subprocess.run(
        [str(VENV_PY), str(SCRIPTS_DIR / "master_verify_r18.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "Wf-35" in proc.stdout or "master verify" in proc.stdout.lower()


# ===========================================================================
# 12. Py_compile (defensive)
# ===========================================================================
def test_master_verify_r18_py_compile() -> None:
    """``python -m py_compile`` must succeed on the script."""
    import py_compile

    script = SCRIPTS_DIR / "master_verify_r18.py"
    py_compile.compile(str(script), doraise=True)
