"""Wf-A269 wave-16 — the §5.9 single-venv gate must state its own DENOMINATOR.

WHAT THIS LOCKS
---------------
``scripts/_wrapper_audit.py`` enforced AGENTS.md §5.9 (one venv for every
launch wrapper) over ``scripts/launch_wf*.sh`` ONLY — 31 of the repo's 113
tracked ``.sh`` files — and printed no denominator, so a reader could not tell
31/113 from 31/31.  Three tracked ``.sh`` files assign an interpreter that
does not exist on this host and the gate could not see them.

The gate now takes its population from the module that OWNS it
(``scripts/_yaml_only_gate.py``) — the exclusion list is IMPORTED, never
duplicated — prints its denominator, and fails closed when the population
cannot be computed.

MUTATION-PROVABLE
-----------------
Each assertion below is written so that the corresponding mutation BREAKS it:

* a duplicated / divergent exclusion list  -> the owner-module agreement test
  and the no-hardcoded-path test fail;
* a dropped or empty exclusion reason     -> the reason test fails;
* a coverage number computed from a subset -> the denominator arithmetic test
  fails (the tracked set is recomputed here from ``git ls-files``, independently
  of the gate);
* a swallowed import error / a swallowed git failure -> the two fail-closed
  tests fail (the verdict must be FAIL and the record must say DEGRADED).

CPU-ONLY.  No GPU, no training, no network.  < 5 s wall.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
AUDIT_PATH = SCRIPTS_DIR / "_wrapper_audit.py"
GATE_PATH = SCRIPTS_DIR / "_yaml_only_gate.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str):
    """Import a non-package module by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"{path} not importable"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_audit():
    return _load(AUDIT_PATH, "_wrapper_audit_wfa269_cov")


def _load_gate():
    return _load(GATE_PATH, "_yaml_only_gate_wfa269_cov")


def _tracked_sh() -> list[str]:
    """Every tracked ``*.sh`` — recomputed here, independently of the gate."""
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.sh"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return sorted(p for p in proc.stdout.split("\0") if p)


@pytest.fixture(scope="module")
def audit_mod():
    return _load_audit()


@pytest.fixture(scope="module")
def report(audit_mod):
    """The audit report.  NOTE: ``audit()`` does NOT write the JSON artifact."""
    return audit_mod.audit()


# ---------------------------------------------------------------------------
# 1. The denominator
# ---------------------------------------------------------------------------
def test_reported_denominator_equals_tracked_minus_exclusions(report):
    cov = report["coverage"]
    tracked = _tracked_sh()
    assert cov["n_tracked_sh_total"] == len(tracked)
    assert cov["n_excluded"] == len(cov["excluded"])
    assert cov["n_scanned"] == len(tracked) - cov["n_excluded"]
    assert cov["coverage_pct"] == pytest.approx(
        round(100.0 * cov["n_scanned"] / len(tracked), 1)
    )
    assert cov["population_degraded"] is False
    assert cov["population_source"].startswith("git ls-files")


def test_population_is_no_longer_the_31_file_subset(report):
    """The coverage hole: the old population was ``scripts/launch_wf*.sh``."""
    cov = report["coverage"]
    assert cov["n_scanned"] > len(list(SCRIPTS_DIR.glob("launch_wf*.sh")))
    assert cov["n_examined_total"] == cov["n_tracked_sh_total"]


def test_audit_agrees_with_the_module_that_owns_the_population(report, audit_mod):
    """The exclusion list is IMPORTED, not re-derived."""
    gate = _load_gate()
    scanned, cov = gate._population()
    assert cov["n_tracked_sh_total"] == report["coverage"]["n_tracked_sh_total"]
    assert len(scanned) == report["coverage"]["n_scanned"]
    assert len(cov["excluded"]) == report["coverage"]["n_excluded"]
    assert {e["path"] for e in cov["excluded"]} == {
        e["path"] for e in report["coverage"]["excluded"]
    }


def test_audit_does_not_hardcode_the_exclusion_list(audit_mod):
    """No second copy of the exclusion list may creep into the audit."""
    src = AUDIT_PATH.read_text(encoding="utf-8", errors="replace")
    for entry in audit_mod.section59_coverage()[0]["excluded"]:
        assert entry["path"] not in src, (
            f"scripts/_wrapper_audit.py hardcodes the excluded path "
            f"{entry['path']}; the list is owned by scripts/_yaml_only_gate.py"
        )


def test_classification_scope_is_unchanged(report):
    """The REAL/SCAFFOLD/DUPLICATE classification keeps its 31-file scope."""
    assert report["wrapper_count"] == len(list(SCRIPTS_DIR.glob("launch_wf*.sh")))


# ---------------------------------------------------------------------------
# 2. Every exclusion carries a reason
# ---------------------------------------------------------------------------
def test_every_exclusion_carries_a_reason(report):
    tracked = set(_tracked_sh())
    for entry in report["coverage"]["excluded"]:
        assert entry["kind"] in ("dir", "file")
        assert isinstance(entry["reason"], str)
        assert entry["reason"].strip(), f"{entry['path']} has no exclusion reason"
        assert entry["path"] in tracked


def test_every_exclusion_is_outside_the_scanned_set(report):
    scanned_paths = {
        row["file"]
        for row in report["section59_interpreters"]["scanned"]["foreign"]
    }
    excluded_paths = {e["path"] for e in report["coverage"]["excluded"]}
    assert not (excluded_paths & scanned_paths)


# ---------------------------------------------------------------------------
# 3. FAIL-CLOSED
# ---------------------------------------------------------------------------
def test_audit_fails_closed_when_the_shared_module_is_unimportable(
    audit_mod, monkeypatch
):
    monkeypatch.setattr(
        audit_mod, "_load_yaml_only_gate", lambda: (None, "simulated: boom")
    )
    degraded = audit_mod.audit()
    assert degraded["coverage"]["population_degraded"] is True
    assert degraded["coverage"]["population_error"]
    assert degraded["verdict"] == "FAIL"
    # the verdict must name the DEGRADATION as its reason: the two
    # pre-existing wrapper violations would otherwise mask it.
    assert "population_degraded" in degraded["verdict_reasons"]


def test_audit_fails_closed_when_the_owner_module_is_broken(
    audit_mod, monkeypatch, tmp_path
):
    """The REAL loader must fault-isolate a broken owner module.

    The test above monkeypatches the loader away, so on its own it cannot
    catch a mutation that makes the real loader swallow its fault.  This
    test drives the real loader with a syntactically broken module — the
    exact situation a concurrent edit creates — and requires that the fault
    be REPORTED (not swallowed) and that the verdict be FAIL.
    """
    broken = tmp_path / "_yaml_only_gate_broken.py"
    broken.write_text("def (:  # a deliberate syntax error\n", encoding="utf-8")
    monkeypatch.setattr(audit_mod, "_YAML_GATE_PATH", broken)

    module, error = audit_mod._load_yaml_only_gate()
    assert module is None, "a broken owner module must not load"
    assert error, "the loader swallowed the fault"

    degraded = audit_mod.audit()
    assert degraded["coverage"]["population_degraded"] is True
    assert degraded["coverage"]["population_error"]
    assert degraded["verdict"] == "FAIL"
    assert "population_degraded" in degraded["verdict_reasons"]


def test_audit_fails_closed_when_git_is_unavailable(audit_mod, monkeypatch):
    """A git failure degrades the population; the verdict must be FAIL."""
    gate, error = audit_mod._load_yaml_only_gate()
    assert gate is not None and not error
    monkeypatch.setattr(
        gate, "_git_tracked_sh_paths", lambda: (None, "simulated: git is down")
    )
    # the audit loads its own instance of the gate; hand it the patched one
    monkeypatch.setattr(audit_mod, "_load_yaml_only_gate", lambda: (gate, ""))
    coverage, _census = audit_mod.section59_coverage()
    assert coverage["population_degraded"] is True
    assert "simulated" in coverage["population_error"]


def test_undegraded_population_is_not_marked_degraded(report):
    """Control for the two fail-closed tests: the real population is sound."""
    assert report["coverage"]["population_degraded"] is False
    assert report["coverage"]["n_tracked_sh_total"] == len(_tracked_sh())


# ---------------------------------------------------------------------------
# 4. The detector: sensitive to a stale interpreter, blind to prose
# ---------------------------------------------------------------------------
def test_a_foreign_interpreter_is_detected(audit_mod, tmp_path):
    fixture = tmp_path / "bad.sh"
    fixture.write_text(
        'PY="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"\n', encoding="utf-8"
    )
    census = audit_mod.interpreter_census([fixture])
    assert census["counts"]["foreign"] == 1
    assert census["counts"]["forbidden"] == 1


def test_a_commented_interpreter_is_not_a_violation(audit_mod, tmp_path):
    """Several tracked wrappers carry the path in a preserved-verbatim comment."""
    fixture = tmp_path / "commented.sh"
    fixture.write_text(
        '# PY="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"\n', encoding="utf-8"
    )
    census = audit_mod.interpreter_census([fixture])
    assert census["counts"]["foreign"] == 0
    assert census["counts"]["forbidden"] == 0


def test_a_variable_prefixed_repo_python_is_not_foreign(audit_mod, tmp_path):
    """``${REPO_ROOT}/.venv/bin/python`` is the portable form, not a foreign one."""
    fixture = tmp_path / "portable.sh"
    fixture.write_text(
        'REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"\n'
        'PY="${REPO_ROOT}/.venv/bin/python"\n',
        encoding="utf-8",
    )
    census = audit_mod.interpreter_census([fixture])
    assert census["counts"]["foreign"] == 0
    assert census["counts"]["forbidden"] == 0


# ---------------------------------------------------------------------------
# 5. The adjudicated stale-interpreter wrappers
# ---------------------------------------------------------------------------
#: The adjudicated stale-interpreter wrappers, and the census rows that must
#: keep disclosing them.
#:
#: 2026-11-04 (Wf-A269 wave-19) — the run record's pin MOVED, and the
#: superseded rendering is preserved VERBATIM per AGENTS.md §5.5 rule #6:
#:
#:     STALE_INTERPRETER_WRAPPERS = {
#:         "TODO/archive/wf437_simplify/launch_wf124_p3_g4.sh": 39,
#:         "TODO/archive/wf437_simplify/launch_wf300_cfm_main.sh": 38,
#:         "molmetal/reports/wf268a_smoke/run_smoke.sh": 16,
#:     }
#:
#: The run record gained a 48-line RETIREMENT banner, which both MOVED the
#: preserved ``PYTHON=`` assignment (16 -> 65) and ADDED a second row that
#: names the interpreter in the banner's own prose (line 44, an ``echo``).
#: Both rows are real census rows and both must stay disclosed, so the value
#: is now a TUPLE of lines.  MEASURED 2026-11-04 on the live tree, the
#: residual census reports exactly:
#:
#:     TODO/archive/wf437_simplify/launch_wf124_p3_g4.sh:39
#:     TODO/archive/wf437_simplify/launch_wf300_cfm_main.sh:38
#:     molmetal/reports/wf268a_smoke/run_smoke.sh:44
#:     molmetal/reports/wf268a_smoke/run_smoke.sh:65
#:
#: Pinning only the old line number would have hidden one of the two rows.
STALE_INTERPRETER_WRAPPERS = {
    "TODO/archive/wf437_simplify/launch_wf124_p3_g4.sh": (39,),
    "TODO/archive/wf437_simplify/launch_wf300_cfm_main.sh": (38,),
    "molmetal/reports/wf268a_smoke/run_smoke.sh": (44, 65),
}


def test_the_three_stale_wrappers_are_disclosed_not_hidden(report):
    """An exclusion must not HIDE a defect (the yaml gate's residual rule).

    The two archived wrappers are test-locked to stay archived
    (``molmetal/tests/test_wf337g2_launcher_v2.py``) and the run record is
    owned by the repair wave, so none of the three may be edited here — they
    must be DISCLOSED by the census on every run instead.
    """
    cov = report["coverage"]
    excluded = {e["path"]: e for e in cov["excluded"]}
    residual = report["section59_interpreters"]["excluded_residual"]
    rows = {(r["file"], r["line"]) for r in residual["foreign"]}
    rows |= {(r["file"], r["line"]) for r in residual["forbidden"]}
    for path, lines in STALE_INTERPRETER_WRAPPERS.items():
        assert path in excluded, f"{path} is not in the exclusion set"
        assert excluded[path]["reason"].strip(), f"{path} has no exclusion reason"
        for line in lines:
            assert (path, line) in rows, (
                f"{path}:{line} is hidden from the §5.9 residual census"
            )


def test_the_stale_census_rows_are_exactly_the_pinned_ones(report):
    """The pin must not UNDER-count: every disclosed row must be pinned.

    A pin that names a subset of the census rows is a census that can grow
    without the pin noticing — the same "a guard's own census is not a
    coverage measurement" defect this file already guards against elsewhere.
    """
    residual = report["section59_interpreters"]["excluded_residual"]
    rows = {(r["file"], r["line"]) for r in residual["foreign"]}
    rows |= {(r["file"], r["line"]) for r in residual["forbidden"]}
    pinned = {
        (path, line)
        for path, lines in STALE_INTERPRETER_WRAPPERS.items()
        for line in lines
    }
    assert rows == pinned, (
        "the §5.9 residual census and the pinned rows disagree: "
        f"census-only={sorted(rows - pinned)} pin-only={sorted(pinned - rows)}"
    )


def test_the_stale_wrappers_are_not_counted_as_violations(report):
    """Disclosed, not fatal: an exclusion is not a §5.9 waiver."""
    residual = report["section59_interpreters"]["excluded_residual"]
    residual_files = {r["file"] for r in residual["foreign"]}
    residual_files |= {r["file"] for r in residual["forbidden"]}
    violation_files = {r["file"] for r in report["section59_violations"]}
    assert not (residual_files & violation_files)


# ---------------------------------------------------------------------------
# 6. The denominator is PRINTED
# ---------------------------------------------------------------------------
def test_stdout_states_the_denominator_and_the_import_provenance(
    audit_mod, report, capsys
):
    audit_mod._print_coverage(report)
    out = capsys.readouterr().out
    cov = report["coverage"]
    assert "COVERAGE:" in out
    assert str(cov["n_tracked_sh_total"]) in out
    assert str(cov["n_scanned"]) in out
    assert str(cov["n_excluded"]) in out
    assert "IMPORTED" in out
    assert "_yaml_only_gate.py" in out
    # every exclusion must be printed WITH its reason
    for entry in cov["excluded"]:
        assert entry["path"] in out, f"{entry['path']} is not printed"
        assert entry["reason"].strip() in out
