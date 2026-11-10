"""Wf-A269 (2026-11-04) — the COVERAGE half of the AGENTS.md §5.11 YAML-only gate.

The gate used to print ``scanned 86 wrapper(s)`` with NO denominator, so a
reader could not tell 86/113 from 86/86 (§6.269.13's class: a gate that
cannot report its own coverage).  Its population was two NON-RECURSIVE globs
under two roots, which left 27 of the repo's 113 tracked ``*.sh`` files in no
gate population at all.

This file is the REGRESSION LOCK on that repair:

* the denominator the gate reports must equal the tracked ``*.sh`` count
  MINUS the explicit exclusions — recomputed here INDEPENDENTLY, with the
  test's own literal matching, so the check is not circular;
* every excluded path must be ACTUALLY excluded, not merely absent — the
  path must exist in the tracked set and must not be in the scanned set;
* every exclusion must carry a reason;
* the exclusion KEY SETS are FROZEN here, so deleting an entry (or adding an
  unjustified one) fails the suite;
* the exclusion matching must be by EXPLICIT LITERAL PATH — a stray
  path-substring heuristic ("archived" appearing anywhere in the path) is the
  Wf-A167 mis-classification the honest-negative catalogue records, and it
  fails this suite.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ZERO GPU, always (the gate's own trainers import torch).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def yaml_gate():
    return _load(REPO_ROOT / "scripts" / "_yaml_only_gate.py", "_wfa269_cov_gate")


@pytest.fixture(scope="module")
def contract_gate():
    return _load(
        REPO_ROOT / "scripts" / "_launch_contract_gate.py", "_wfa269_cov_contract"
    )


# ---------------------------------------------------------------------------
# The FROZEN exclusion key sets.  Deleting an entry from the gate's constants
# (or adding an unjustified one) fails these two assertions.
# ---------------------------------------------------------------------------
EXPECTED_EXCLUDED_DIRS = {
    "scripts/_retired",
    "TODO/archive/wf437_simplify",
}

#: The frozen exclusion FILE set.
#:
#: 2026-11-04 (Wf-A269 wave-19) — RE-FROZEN to ONE entry.  The superseded
#: 3-entry rendering is preserved VERBATIM below, per AGENTS.md §5.5 rule #6
#: (a correction is additive + dated; the superseded rendering is never
#: silently deleted):
#:
#:     EXPECTED_EXCLUDED_FILES = {
#:         "run_wf187_remaining.sh",
#:         "results/cfm_full_retrain/launch_loop.sh",
#:         "molmetal/reports/wf268a_smoke/run_smoke.sh",
#:     }
#:
#: WHY the first two are gone: both were MIGRATED off the command line
#: (§5.11), so excluding them made the gate BLIND to two live launch
#: surfaces that it now scans and PASSes.  MEASURED 2026-11-04, per-file
#: audit of the live tree:
#:
#:     run_wf187_remaining.sh                  -> PASS (0 violations)
#:     results/cfm_full_retrain/launch_loop.sh -> PASS (0 violations)
#:
#: A frozen expectation that no longer matches the gate is not a guard — it
#: is a red test that hides the two migrations behind it.  The third entry
#: stays excluded because it is RETIRED, not migrated: it is a run record
#: under ``molmetal/reports/`` whose interpreter
#: (/home/hugo/codes/CadButEaas/.venv-vlm/bin/python) no longer exists on
#: this host, and the artifacts it produced sit beside it and must not be
#: overwritten.  Its 10 §5.11 trainer flags are preserved verbatim INSIDE
#: the file as the provenance of those artifacts, and they stay disclosed
#: by the excluded-file residual census on every run.
EXPECTED_EXCLUDED_FILES = {
    "molmetal/reports/wf268a_smoke/run_smoke.sh",
}


def _tracked_sh() -> list[str]:
    """Every tracked ``*.sh``, measured the same way the gate measures it."""
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", "*.sh"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(p for p in proc.stdout.split("\0") if p)


def _expected_exclusions(tracked: list[str]) -> set[str]:
    """The excluded set, recomputed with the TEST's own literal matching.

    Deliberately independent of the gate's matching code: an exact directory
    key (or a path under it) and an exact file key — never a substring.
    """
    out: set[str] = set()
    for rel in tracked:
        if rel in EXPECTED_EXCLUDED_FILES:
            out.add(rel)
            continue
        for directory in EXPECTED_EXCLUDED_DIRS:
            if rel == directory or rel.startswith(directory + "/"):
                out.add(rel)
                break
    return out


# ---------------------------------------------------------------------------
# 1. The denominator
# ---------------------------------------------------------------------------
def test_the_reported_denominator_equals_tracked_minus_the_exclusions(yaml_gate):
    tracked = _tracked_sh()
    paths, cov = yaml_gate._population()

    assert cov["n_tracked_sh_total"] == len(tracked), (
        "the gate's denominator must be the number of tracked *.sh files; "
        f"gate said {cov['n_tracked_sh_total']}, measured {len(tracked)}"
    )
    assert cov["n_excluded"] == len(cov["excluded"])
    assert cov["n_scanned"] == len(paths)
    assert cov["n_scanned"] + cov["n_excluded"] == len(tracked), (
        "scanned + excluded must account for EVERY tracked *.sh"
    )
    assert cov["coverage_pct"] == pytest.approx(
        100.0 * cov["n_scanned"] / len(tracked), abs=0.05
    )
    assert cov["population_degraded"] is False
    assert "git ls-files" in cov["population_source"]


def test_the_excluded_set_is_exactly_the_explicit_exclusions(yaml_gate):
    """The excluded set is recomputed here, not read back from the gate."""
    tracked = _tracked_sh()
    _paths, cov = yaml_gate._population()
    reported = {row["path"] for row in cov["excluded"]}
    assert reported == _expected_exclusions(tracked)


def test_every_excluded_path_is_actually_excluded_not_merely_absent(yaml_gate):
    """An exclusion must EXCLUDE a real file — not name a path that never existed."""
    tracked = set(_tracked_sh())
    paths, cov = yaml_gate._population()
    scanned = {yaml_gate._rel(p) for p in paths}

    assert cov["excluded"], "the exclusion list is empty — nothing is excluded"
    for row in cov["excluded"]:
        rel = row["path"]
        assert rel in tracked, (
            f"{rel} is excluded but is NOT a tracked *.sh — the exclusion is "
            "vacuous (it names a path that was never in the population)"
        )
        assert rel not in scanned, f"{rel} is excluded but was still scanned"
        found = yaml_gate._exclusion_for(REPO_ROOT / rel, rel)
        assert found is not None, f"{rel} is not actually excluded by the matcher"
        kind, reason = found
        assert kind in {"dir", "file"}
        assert reason and reason.strip(), f"{rel} has no exclusion reason"


def test_the_exclusion_list_is_non_empty_and_every_entry_carries_a_reason(yaml_gate):
    for name, table in (
        ("EXCLUDED_WRAPPER_DIRS", yaml_gate.EXCLUDED_WRAPPER_DIRS),
        ("EXCLUDED_WRAPPER_FILES", yaml_gate.EXCLUDED_WRAPPER_FILES),
    ):
        assert table, f"{name} is empty"
        for key, reason in table.items():
            assert isinstance(key, str) and key.strip(), f"{name} has an empty key"
            assert not key.startswith("/"), f"{name} key {key!r} is not repo-relative"
            assert ".." not in key.split("/"), f"{name} key {key!r} escapes the repo"
            assert not any(c in key for c in "*?[]"), (
                f"{name} key {key!r} is a glob, not an explicit literal path"
            )
            assert isinstance(reason, str) and len(reason.strip()) >= 40, (
                f"{name}[{key!r}] has no usable reason"
            )


def test_the_exclusion_key_sets_are_frozen(yaml_gate):
    """MUTATION-PROVABLE: deleting (or adding) an exclusion entry fails here."""
    assert set(yaml_gate.EXCLUDED_WRAPPER_DIRS) == EXPECTED_EXCLUDED_DIRS
    assert set(yaml_gate.EXCLUDED_WRAPPER_FILES) == EXPECTED_EXCLUDED_FILES


def test_the_exclusion_matching_has_no_path_heuristic(yaml_gate):
    """MUTATION-PROVABLE: a stray path-substring heuristic fails here.

    The Wf-A167 guard filed ``results/cfm_full_retrain/launch_loop.sh`` under
    "archived" because of a path-based heuristic and reported a green tree
    while a live retrain loop kept hard-forcing GPU 0 (§6.269 of the
    honest-negative catalogue).  A path that merely LOOKS archival must not
    be excluded.
    """
    lookalikes = [
        "scripts/archived_launcher.sh",
        "scripts/_retired_but_live/launch_x.sh",
        "TODO/archive_other/launch_x.sh",
        "results/cfm_full_retrain/launch_loop.sh.bak",
        "scripts/launch_wf_retired.sh",
    ]
    for rel in lookalikes:
        assert yaml_gate._exclusion_for(REPO_ROOT / rel, rel) is None, (
            f"{rel} was excluded by a path heuristic — exclusions must be "
            "explicit literals"
        )
    # …and the real exclusions are still excluded.
    for rel in EXPECTED_EXCLUDED_FILES:
        assert yaml_gate._exclusion_for(REPO_ROOT / rel, rel) is not None
    for rel in ("scripts/_retired/launch_wf_fix3lines_cfm.sh",):
        assert yaml_gate._exclusion_for(REPO_ROOT / rel, rel) is not None


# ---------------------------------------------------------------------------
# 2. The population did not shrink
# ---------------------------------------------------------------------------
def test_the_previously_scanned_wrappers_are_still_scanned(yaml_gate):
    """The 86 wrappers the gate scanned before must all still be scanned."""
    old = {
        yaml_gate._rel(p)
        for p in yaml_gate._glob_fallback_paths()
        if not yaml_gate._is_excluded_wrapper(p)
    }
    paths, cov = yaml_gate._population()
    new = {yaml_gate._rel(p) for p in paths}
    missing = sorted(old - new)
    assert not missing, f"the population lost wrappers it used to scan: {missing}"
    assert cov["n_scanned"] >= len(old)


def test_the_shipped_tree_satisfies_the_yaml_only_rule(yaml_gate):
    """The widened gate still PASSes over the whole population."""
    report = yaml_gate.audit()
    offenders = [
        f"{v['file']}:{v['line']} [{v['kind']}] {v['detail'][:120]}"
        for v in report["violations"]
    ]
    assert not offenders, "yaml-only violations in the shipped tree:\n" + "\n".join(
        offenders
    )
    assert report["verdict"] == "PASS"
    assert report["n_wrappers_scanned"] == report["coverage"]["n_scanned"]


# ---------------------------------------------------------------------------
# 3. The gate states its coverage on stdout
# ---------------------------------------------------------------------------
def test_the_gate_prints_its_own_coverage(yaml_gate, capsys):
    assert yaml_gate.main([]) == 0
    out = capsys.readouterr().out
    cov = yaml_gate.audit()["coverage"]
    assert "COVERAGE:" in out
    assert str(cov["n_tracked_sh_total"]) in out
    assert "scanned" in out and "excluded" in out
    assert "%" in out
    for row in cov["excluded"]:
        assert row["path"] in out, f"the excluded path {row['path']} was not printed"
    assert out.strip().splitlines()[-1].endswith("PASS")


def test_the_excluded_residual_is_disclosed_not_hidden(yaml_gate):
    """An excluded file's unmigrated §5.11 flags are MEASURED and reported."""
    report = yaml_gate.audit()
    residual = report["excluded_residual"]
    assert residual["n_unmigrated_flags"] > 0, (
        "the excluded files carry unmigrated §5.11 flags; the residual census "
        "must measure them so the exclusion cannot become a silent hole"
    )
    # The residual report must NOT use the word the pre-flight consumer
    # counts to size a failing gate.
    blob = str(residual)
    assert "VIOLATION" not in blob


def test_the_exclusions_are_load_bearing(yaml_gate):
    """Re-admitting an excluded file flips the verdict — the gate can fail.

    2026-11-04 (Wf-A269 wave-19): the target MOVED, and the superseded
    target is preserved VERBATIM per AGENTS.md §5.5 rule #6:

        target = REPO_ROOT / "run_wf187_remaining.sh"

    That file was MIGRATED off the command line, so it no longer carries a
    §5.11 violation and could no longer prove the gate can fail.  The
    surviving target is the RETIRED run record, which still carries its 10
    unmigrated trainer flags (§5.5: the flags are preserved inside the file
    as the provenance of the artifacts beside it).
    """
    target = REPO_ROOT / "molmetal" / "reports" / "wf268a_smoke" / "run_smoke.sh"
    report = yaml_gate.audit([target])
    assert report["n_violations"] > 0
    assert report["verdict"] == "FAIL"


def test_the_migrated_wrappers_are_now_scanned_and_clean(yaml_gate):
    """The two wrappers dropped from the exclusion set are CLEAN, not broken.

    This is the counterpart of the re-freeze: if either file regressed to
    passing hyperparameters on the command line, this test fails and the
    exclusion would have to come back (with a dated reason).
    """
    for rel in ("run_wf187_remaining.sh", "results/cfm_full_retrain/launch_loop.sh"):
        target = REPO_ROOT / rel
        assert target.is_file(), f"{rel} is missing — the population changed"
        report = yaml_gate.audit([target])
        assert report["n_violations"] == 0, (
            f"{rel} carries {report['n_violations']} §5.11 violation(s) again"
        )
        assert report["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# 4. The launch-contract gate shares the SAME population
# ---------------------------------------------------------------------------
def test_the_launch_contract_gate_reports_the_same_coverage(contract_gate, yaml_gate):
    report = contract_gate.audit()
    cov = report["coverage"]
    ref = yaml_gate._population()[1]
    assert cov["n_tracked_sh_total"] == ref["n_tracked_sh_total"]
    assert cov["n_scanned"] == ref["n_scanned"]
    assert cov["n_excluded"] == ref["n_excluded"]
    assert {row["path"] for row in cov["excluded"]} == {
        row["path"] for row in ref["excluded"]
    }
    assert report["verdict"] == "PASS"
    assert report["n_violations"] == 0


def test_the_launch_contract_gate_prints_the_same_coverage_block(
    contract_gate, capsys
):
    assert contract_gate.main([]) == 0
    out = capsys.readouterr().out
    assert "COVERAGE:" in out
    assert "EXCLUDED-FILE RESIDUAL" in out
    assert out.strip().splitlines()[-1].endswith("PASS")
