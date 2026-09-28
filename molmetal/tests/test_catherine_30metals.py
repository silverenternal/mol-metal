"""Tests for the Catherine 30-metal closure bench (TODO-96 R20-P5).

================================================================
CPU-only smoke / contract tests for the R20-P5 ship: extend the
Catherine closure benchmark cohort from 9 -> 30 d-block metals.
================================================================
This module exercises:

* :data:`molmetal.scripts.bench_catherine_30metals.DEFAULT_30_METALS` —
  must be 30 metals spanning 3d / 4d / 5d + La.
* :data:`TASK_SPEC_9_METALS` — backward-compat with Wf-37 P2's
  TASK_SPEC_METALS subset.
* :func:`_select_metals` — clamps to [1, 30] and returns the first
  ``n`` metals in periodic-table order.
* The CLI smoke path returns exit code 0 on the smoke cohort (3
  metals).
* The 9-metal run returns exit code 0 (the formal 9->30 expansion
  boundary — the script must still bench 9 metals correctly).
* Backward compat — ``bench_closure_dblock.py`` still works (we
  don't mutate that script).

The tests are CPU-only and complete in <5 s on any CI container.
None of them mutate :mod:`math_skills.catherine_properties` or
:mod:`lam_chem.closure`.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_catherine_30metals.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

try:
    from molmetal.scripts.bench_catherine_30metals import (  # type: ignore
        DEFAULT_30_METALS,
        TASK_SPEC_9_METALS,
        _select_metals,
    )
except Exception:  # pragma: no cover - import path fallback
    import sys as _sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parent.parent
    if str(_here / "scripts") not in _sys.path:
        _sys.path.insert(0, str(_here / "scripts"))
    from bench_catherine_30metals import (  # type: ignore  # noqa: E402
        DEFAULT_30_METALS,
        TASK_SPEC_9_METALS,
        _select_metals,
    )


REPO_ROOT = "/home/hugo/codes/mol-metal"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_default_30_metals_has_30_unique_entries():
    """G2_catherine_30metal — the default cohort must be 30 unique d-block metals."""
    assert len(DEFAULT_30_METALS) == 30, (
        f"DEFAULT_30_METALS must have 30 metals; got {len(DEFAULT_30_METALS)}"
    )
    assert len(set(DEFAULT_30_METALS)) == 30, (
        "DEFAULT_30_METALS contains duplicates"
    )


def test_default_30_metals_covers_3d_4d_5d_plus_la():
    """DEFAULT_30_METALS must cover the 3d, 4d, 5d rows + La."""
    # 3d (10 metals)
    for m in ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"):
        assert m in DEFAULT_30_METALS, f"3d metal {m} missing from DEFAULT_30_METALS"
    # 4d (10 metals)
    for m in ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"):
        assert m in DEFAULT_30_METALS, f"4d metal {m} missing from DEFAULT_30_METALS"
    # 5d + La (10 metals)
    for m in ("La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"):
        assert m in DEFAULT_30_METALS, f"5d/La metal {m} missing from DEFAULT_30_METALS"


def test_default_30_metals_includes_task_spec_9_metals():
    """The 9 task-spec metals (Wf-37 P2) must be a subset of the 30."""
    for m in TASK_SPEC_9_METALS:
        assert m in DEFAULT_30_METALS, (
            f"task-spec metal {m} missing from DEFAULT_30_METALS"
        )


def test_task_spec_9_metals_match_wf37_subset():
    """TASK_SPEC_9_METALS must match Wf-37 P2's canonical 9-metal subset."""
    expected = ("Pt", "Ru", "Ir", "Rh", "Re", "Os", "Au", "Pd", "Ag")
    assert TASK_SPEC_9_METALS == expected


# ---------------------------------------------------------------------------
# _select_metals — bench cohort selector
# ---------------------------------------------------------------------------


def test_select_metals_clamps_to_min_1():
    """n < 1 must clamp to 1 (defensive)."""
    metals = _select_metals(0)
    assert metals == [DEFAULT_30_METALS[0]]
    metals = _select_metals(-10)
    assert metals == [DEFAULT_30_METALS[0]]


def test_select_metals_clamps_to_max_30():
    """n > 30 must clamp to 30 (the full d-block cohort)."""
    metals = _select_metals(100)
    assert len(metals) == 30
    assert metals == list(DEFAULT_30_METALS)


def test_select_metals_returns_first_n_in_order():
    """The first n metals must follow the canonical order."""
    for n in (1, 5, 9, 15, 20, 25, 30):
        metals = _select_metals(n)
        assert metals == list(DEFAULT_30_METALS[:n])
        assert len(metals) == n


def test_select_metals_at_9_returns_task_spec_like_subset():
    """At n=9, returns the first 9 metals (Sc..Zn = 3d row)."""
    metals = _select_metals(9)
    # 3d row only — Sc..Zn
    expected_3d = ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu")
    # Wait — 3d has 10 metals (Sc..Zn); first 9 = Sc..Cu.
    expected_first_9 = (
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    )
    assert metals == list(expected_first_9)


# ---------------------------------------------------------------------------
# CLI smoke (script entry point)
# ---------------------------------------------------------------------------


def test_cli_smoke_runs_3_metals():
    """The CLI smoke path returns exit code 0 on the 3-metal cohort."""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "molmetal.scripts.bench_catherine_30metals",
            "--smoke",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, (
        f"CLI smoke failed: rc={out.returncode}\n"
        f"stdout={out.stdout}\nstderr={out.stderr}"
    )
    assert "n_metals                 : 3" in out.stdout
    assert "[PASS] Catherine 30-metal bench ran on 3 metals" in out.stdout


def test_cli_9_metals_runs_as_backward_compat():
    """The 9-metal run is the formal 9->30 expansion boundary — must PASS."""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "molmetal.scripts.bench_catherine_30metals",
            "--metals", "9",
            "--max-depth", "0",
            "--output-dir",
            "/tmp/wf255_catherine_9metals_test",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, (
        f"CLI 9-metal failed: rc={out.returncode}\n"
        f"stdout={out.stdout}\nstderr={out.stderr}"
    )
    assert "n_metals                 : 9" in out.stdout
    assert "[PASS] Catherine 30-metal bench ran on 9 metals" in out.stdout


def test_cli_30_metals_full_run():
    """The full 30-metal run is the R20-P5 ship target."""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "molmetal.scripts.bench_catherine_30metals",
            "--metals", "30",
            "--max-depth", "0",
            "--output-dir",
            "/tmp/wf255_catherine_30metals_full_test",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, (
        f"CLI 30-metal full failed: rc={out.returncode}\n"
        f"stdout={out.stdout}\nstderr={out.stderr}"
    )
    assert "n_metals                 : 30" in out.stdout
    assert "[PASS] Catherine 30-metal bench ran on 30 metals" in out.stdout
    # Output JSON must contain 30 per_metal entries.
    json_path = Path(
        "/tmp/wf255_catherine_30metals_full_test/per_metal_pass.json"
    )
    assert json_path.exists(), f"per_metal_pass.json not written at {json_path}"
    data = json.loads(json_path.read_text())
    assert data["n_metals"] == 30
    assert len(data["per_metal"]) == 30


# ---------------------------------------------------------------------------
# Backward compat — existing Wf-37 P2 bench still imports cleanly
# ---------------------------------------------------------------------------


def test_bench_closure_dblock_constants_unchanged():
    """G6_no_breaking_changes — the Wf-37 P2 TASK_SPEC_METALS constant still matches."""
    try:
        from molmetal.scripts.bench_closure_dblock import (  # type: ignore
            TASK_SPEC_METALS,
        )
    except Exception:
        from bench_closure_dblock import TASK_SPEC_METALS  # type: ignore
    expected = ("Pt", "Ru", "Ir", "Rh", "Re", "Os", "Au", "Pd", "Ag")
    assert TASK_SPEC_METALS == expected
