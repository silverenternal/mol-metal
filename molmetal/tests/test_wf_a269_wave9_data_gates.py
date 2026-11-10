# Licensed under the Apache License, Version 2.0.
"""Wf-A269 wave-9 (2026-11-02) — the launcher data-gate class.

This file is the regression surface for ONE defect class, found while
repairing the Wf-355-A3 R20 §5 launcher
(``scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh``):

  * a "data quality" gate that CANNOT FAIL (a zombie gate) — it measured a
    field the data does not contain, and a shell error silently routed it to
    the PASS branch;
  * the ``grep -c ... || echo 0`` double-count — ``grep -c`` prints "0" AND
    exits 1 on no match, so the fallback appended a SECOND "0" and the
    captured value became "0\\n0";
  * a machine-readable plan record that no JSON reader could parse, guarded
    only by a substring match on its raw text.

The class was NOT confined to the file it was found in: the same census
surfaced a second live instance in ``launch_wf310e_wf135_phase2_gpu.sh``
(a DEAD PASS branch — the mirror image) and a third in an archived script.

Tests (all CPU-only, no GPU, no launches):

  1. test_platinai_check_passes_on_balanced_fixture
  2. test_platinai_check_fails_on_missing_column        (the dead FAIL branch)
  3. test_platinai_check_fails_on_non_numeric_label
  4. test_platinai_check_warns_on_low_active_count       (severity preserved)
  5. test_platinai_check_skips_on_missing_file
  6. test_platinai_check_accepts_rounding_boundary_row  (exact band, not a
     magic tolerance)
  7. test_cohort_balance_check_passes / _fails / _skips
  8. test_wrappers_have_no_live_grep_c_echo_zero         (the class, repo-wide)
  9. test_wf310e_plan_json_is_valid_json                 (the machine-readable
     record must be readable by a JSON reader)
 10. test_wf310e_uses_the_4cell_checker_not_a_cell_grep

Per AGENTS.md §5.2: every assertion is about the CODE's behaviour on data
that is either built here or measured on disk.  No number in this file is
asserted from prose.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
PLATINAI_CHECK = REPO_ROOT / "molmetal" / "scripts" / "wf_platinai_4cell_check.py"
COHORT_CHECK = REPO_ROOT / "molmetal" / "scripts" / "wf_cohort_balance_check.py"
WF310E_WRAPPER = SCRIPTS / "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh"
WF135_WRAPPER = SCRIPTS / "launch_wf310e_wf135_phase2_gpu.sh"

CELLS = ("MCF7", "A549", "A2780", "A2780cis")
COLUMNS = {
    "MCF7": "pIC50_mcf7",
    "A549": "pIC50_a549",
    "A2780": "pIC50_a2780",
    "A2780cis": "pIC50_a2780cis",
}


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _write_platinai_csv(path: Path, n_rows: int = 60, active_value: float = 6.0) -> None:
    """A wide PlatinAI-shaped CSV: one row per ligand, one column per cell."""
    fieldnames = ["SMILES_Ligands", *COLUMNS.values(), "metal"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n_rows):
            row = {"SMILES_Ligands": f"C{i}", "metal": "Pt"}
            for col in COLUMNS.values():
                row[col] = f"{active_value:.4f}"
            w.writerow(row)


def _write_meta(path: Path, n_rows: int, active_per_line: dict[str, int]) -> None:
    path.write_text(
        json.dumps(
            {
                "out": "fixture.csv",
                "n_rows": n_rows,
                "n_target": n_rows,
                "seed": 42,
                "cell_lines": list(CELLS),
                "n_active_per_line": active_per_line,
                "metal": "Pt",
                "schema": ["SMILES_Ligands", *COLUMNS.values(), "metal"],
            },
            indent=2,
        )
    )


def _run_platinai_check(csv_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(PLATINAI_CHECK),
            "--csv",
            str(csv_path),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# 1-6. The PlatinAI 4-cell checker
# ---------------------------------------------------------------------------
def test_platinai_check_passes_on_balanced_fixture(tmp_path: Path) -> None:
    """A CSV carrying all 4 cell-line columns with enough active rows PASSes."""
    csv_path = tmp_path / "plat.csv"
    meta_path = tmp_path / "plat.meta.json"
    _write_platinai_csv(csv_path, n_rows=60)
    _write_meta(meta_path, 60, {c: 60 for c in CELLS})

    proc = _run_platinai_check(csv_path, "--meta", str(meta_path), "--min-active", "50")
    assert proc.returncode == 0, f"expected PASS; rc={proc.returncode}\n{proc.stdout}"
    assert "PASS" in proc.stdout
    # Every cell line must be MEASURED and reported, not silently skipped.
    for cell in CELLS:
        assert cell in proc.stdout, f"cell {cell} missing from the report"
    assert f"active=60" in proc.stdout


def test_platinai_check_fails_on_missing_column(tmp_path: Path) -> None:
    """A declared cell-line column that is ABSENT must FAIL (dead FAIL branch).

    This is the check the old `grep -c "\\"MCF7\\""` could never make: the CSV
    is wide, so a missing column is a real defect.
    """
    csv_path = tmp_path / "plat.csv"
    _write_platinai_csv(csv_path, n_rows=60)
    # Drop the A549 column entirely.
    text = csv_path.read_text().splitlines()
    header = text[0].split(",")
    drop = header.index(COLUMNS["A549"])
    rows = []
    for line in text[1:]:
        parts = line.split(",")
        del parts[drop]
        rows.append(",".join(parts))
    csv_path.write_text("\n".join([",".join(header[:drop] + header[drop + 1 :])] + rows) + "\n")

    proc = _run_platinai_check(csv_path, "--min-active", "50")
    assert proc.returncode == 1, f"expected FAIL; rc={proc.returncode}\n{proc.stdout}"
    assert "FAIL" in proc.stdout
    assert "A549" in proc.stdout


def test_platinai_check_fails_on_non_numeric_label(tmp_path: Path) -> None:
    """A label that is not numeric is a FAIL, not a silent skip."""
    csv_path = tmp_path / "plat.csv"
    _write_platinai_csv(csv_path, n_rows=60)
    text = csv_path.read_text().splitlines()
    header = text[0].split(",")
    idx = header.index(COLUMNS["A549"])
    parts = text[1].split(",")
    parts[idx] = "n/a"
    text[1] = ",".join(parts)
    csv_path.write_text("\n".join(text) + "\n")

    proc = _run_platinai_check(csv_path, "--min-active", "50")
    assert proc.returncode == 1, f"expected FAIL; rc={proc.returncode}\n{proc.stdout}"
    assert "FAIL" in proc.stdout


def test_platinai_check_warns_on_low_active_count(tmp_path: Path) -> None:
    """A low active count is a WARN (rc=0), preserving the original severity.

    The launcher intended a WARNING here; the defect was that the branch was
    UNREACHABLE, not that it was too lenient.
    """
    csv_path = tmp_path / "plat.csv"
    _write_platinai_csv(csv_path, n_rows=60, active_value=4.2)  # numeric, inactive
    proc = _run_platinai_check(csv_path, "--min-active", "50")
    assert proc.returncode == 0, f"expected PASS-with-warning; rc={proc.returncode}\n{proc.stdout}"
    assert "FINDING [WARN]" in proc.stdout
    assert "active=0" in proc.stdout


def test_platinai_check_skips_on_missing_file(tmp_path: Path) -> None:
    """A missing CSV is a SKIP (rc=2), not a crash and not a silent PASS."""
    proc = _run_platinai_check(tmp_path / "absent.csv", "--min-active", "50")
    assert proc.returncode == 2, f"expected SKIP; rc={proc.returncode}\n{proc.stdout}"
    assert "SKIP" in proc.stdout


def test_platinai_check_accepts_rounding_boundary_row(tmp_path: Path) -> None:
    """The meta cross-check uses the EXACT round(x,4) boundary band.

    The builder decided "active" on the UNROUNDED pIC50 and wrote round(x, 4),
    so a row whose unrounded value is in [4.99995, 5.0) is stored as exactly
    5.0 and is counted active by any recompute-from-CSV while the builder did
    not.  The check must accept that delta instead of reporting false drift.
    """
    csv_path = tmp_path / "plat.csv"
    meta_path = tmp_path / "plat.meta.json"
    _write_platinai_csv(csv_path, n_rows=60)
    # Put exactly one boundary row (stored as 5.0) into the A2780 column.
    text = csv_path.read_text().splitlines()
    header = text[0].split(",")
    idx = header.index(COLUMNS["A2780"])
    parts = text[1].split(",")
    parts[idx] = "5.0"
    text[1] = ",".join(parts)
    csv_path.write_text("\n".join(text) + "\n")
    # The builder counted 59 (it excluded that row); the CSV shows 60.
    _write_meta(meta_path, 60, {**{c: 60 for c in CELLS}, "A2780": 59})

    proc = _run_platinai_check(csv_path, "--meta", str(meta_path), "--min-active", "50")
    assert proc.returncode == 0, (
        f"the rounding-boundary row must not be reported as drift; "
        f"rc={proc.returncode}\n{proc.stdout}"
    )
    assert "meta_delta=+1" in proc.stdout


def test_platinai_check_rejects_real_meta_drift(tmp_path: Path) -> None:
    """A drift LARGER than the boundary band IS a FAIL (the meta is stale)."""
    csv_path = tmp_path / "plat.csv"
    meta_path = tmp_path / "plat.meta.json"
    _write_platinai_csv(csv_path, n_rows=60)
    _write_meta(meta_path, 60, {**{c: 60 for c in CELLS}, "A2780": 10})
    proc = _run_platinai_check(csv_path, "--meta", str(meta_path), "--min-active", "50")
    assert proc.returncode == 1, f"expected FAIL; rc={proc.returncode}\n{proc.stdout}"
    assert "FAIL" in proc.stdout


# ---------------------------------------------------------------------------
# 7. The cohort balance checker (the second live instance of the class)
# ---------------------------------------------------------------------------
def test_cohort_balance_check_passes_on_balanced(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "stats": {"n_active": 250, "n_inactive": 250},
                "actives": [f"a{i}" for i in range(250)],
                "inactives": [f"i{i}" for i in range(250)],
            }
        )
    )
    proc = subprocess.run(
        [sys.executable, str(COHORT_CHECK), "--json", str(cohort)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"expected PASS; rc={proc.returncode}\n{proc.stdout}"
    assert "active=250 inactive=250" in proc.stdout


def test_cohort_balance_check_fails_on_unbalanced(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps({"stats": {"n_active": 100, "n_inactive": 250}})
    )
    proc = subprocess.run(
        [sys.executable, str(COHORT_CHECK), "--json", str(cohort)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1, f"expected FAIL; rc={proc.returncode}\n{proc.stdout}"
    assert "FAIL" in proc.stdout


def test_cohort_balance_check_skips_on_missing(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(COHORT_CHECK), "--json", str(tmp_path / "absent.json")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, f"expected SKIP; rc={proc.returncode}\n{proc.stdout}"
    assert "SKIP" in proc.stdout


# ---------------------------------------------------------------------------
# 8. The class, repo-wide: no LIVE `grep -c ... || echo 0`
# ---------------------------------------------------------------------------
def test_wrappers_have_no_live_grep_c_echo_zero() -> None:
    """No LIVE `grep -c` may use `|| echo 0` (the double-count).

    `grep -c` prints "0" AND exits 1 when nothing matches, so `|| echo 0`
    appends a SECOND "0" and the captured value becomes "0\\n0".  Comment
    lines are excluded because the SUPERSEDED RENDERING is preserved
    VERBATIM per AGENTS.md §5.5 rule #6.
    """
    offenders: list[str] = []
    for sh in sorted(SCRIPTS.glob("*.sh")):
        for lineno, line in enumerate(
            sh.read_text(errors="replace").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "grep -c" in line and "|| echo 0" in line:
                offenders.append(f"{sh.name}:{lineno}: {stripped}")
    assert not offenders, (
        "live `grep -c ... || echo 0` double-counts (captured value becomes "
        "'0\\n0'):\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# 9-10. The Wf-310-E launcher itself
# ---------------------------------------------------------------------------
def test_wf310e_uses_the_4cell_checker_not_a_cell_grep() -> None:
    """The 4-cell gate must call the checker, not grep for a cell name."""
    text = WF310E_WRAPPER.read_text()
    live = [
        ln
        for ln in text.splitlines()
        if not ln.strip().startswith("#")
    ]
    live_text = "\n".join(live)
    assert "wf_platinai_4cell_check.py" in live_text, (
        "the wrapper must call molmetal/scripts/wf_platinai_4cell_check.py"
    )
    # The old zombie grep: a quoted UPPERCASE cell name in a wide CSV.
    assert 'grep -c "\\"${CELL}\\""' not in live_text, (
        "the zombie `grep -c \"\\\"${CELL}\\\"\"` gate must not come back: the "
        "CSV is WIDE (pIC50_mcf7 ...), so that pattern can never match"
    )


@pytest.mark.skipif(
    not (REPO_ROOT / ".data" / "molmetal" / "platinai_4cell.csv").is_file(),
    reason="PlatinAI 4-cell CSV not on disk",
)
def test_wf310e_plan_json_is_valid_json() -> None:
    """The plan record must be readable by a JSON reader.

    It used to emit UNQUOTED array elements ("cells": [MCF7,A549,...]), so
    `json.load` raised "Expecting value".  The only guard on this file was a
    substring match on its raw text, which cannot see a syntax error.
    """
    env = os.environ.copy()
    env["USER_OK"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        ["bash", str(WF310E_WRAPPER), "--dry-run", "--smoke-only"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"dry-run failed (rc={proc.returncode}); stderr tail: {proc.stderr[-600:]}"
    )
    plan_path = (
        REPO_ROOT / "molmetal" / "reports" / "wf355a3_r20_p5_tmqm_plat" / "plan.json"
    )
    assert plan_path.is_file(), f"plan.json missing: {plan_path}"
    plan = json.loads(plan_path.read_text())  # raises if invalid
    assert isinstance(plan["cells"], list) and plan["cells"], "cells must be a list"
    assert all(isinstance(c, str) for c in plan["cells"]), "cells must be strings"
    assert isinstance(plan["tmqm_metals"], list)
    assert all(isinstance(m, str) for m in plan["tmqm_metals"])
    assert all(isinstance(s, int) for s in plan["seeds"]), "seeds must be numbers"
    # The 4-cell data verdict must be recorded, not only in stderr.
    assert plan.get("platinai_4cell_check") in {"PASS", "FAIL", "SKIPPED"}, (
        f"plan.json must record the 4-cell check verdict; got "
        f"{plan.get('platinai_4cell_check')!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
