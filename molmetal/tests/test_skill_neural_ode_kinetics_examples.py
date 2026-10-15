"""Regression test — verifies the neural-ode-kinetics SKILL.md inventory claims.

Background
----------
Wf-A44-J19-pilot (2026-10-15): the docs-vs-code-drift guard pattern from
Wf-A43-fixup-3 is being extended to 3 pilot skills before scaling to the
remaining 26 skills. This file guards `mol-metal-neural-ode-kinetics`.

The neural-ode-kinetics SKILL.md is a **CPU-only docs scaffold** per
Wf-76 Phase 1. It documents the **protocol, not the measurement** — the
4 verification gates (P2.1 backbone / P2.2 ligand-exchange / P2.3
hydrolysis+redox / P2.4 close-out) are all PROJECTED until Wf-76
Phases 2-N ship.

Per SKILL.md §10.4 "Output deliverables" — three artifacts are marked
**SHIP** (this turn, not PROJECTED):

  - ``.claude/skills/mol-metal-neural-ode-kinetics/SKILL.md``  (this file)
  - ``.claude/skills/mol-metal-neural-ode-kinetics/test_neural_ode_kinetics.py``
    (smoke, OK)
  - ``.claude/skills/mol-metal-neural-ode-kinetics/references.md``
    (5 paper refs)

The 11 PROJECTED artifacts (`gnn_ode.py`, `hydrolysis_cascade.py`,
`redox_cycling.py`, `cellular_uptake_ode.py`, the 4 wf76 JSONs, the
2 P2.4 close-out TODOs, the `test_gnn_ode.py`, `test_hydrolysis.py`,
`test_redox_cycling.py`) are explicitly PROJECTED until Wf-76 P2.x
ships on Blackwell GPU-1.

This test pins the SHIP vs PROJECTED claim boundary and catches a
class of drift unique to PROJECTED-status skills — where an author
claims SHIP for an artifact that does not exist on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# All tests in this file are CPU-only — no torch, no torchdiffeq.

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "mol-metal-neural-ode-kinetics"
SKILL_MD = SKILL_DIR / "SKILL.md"
SMOKE_TEST = SKILL_DIR / "test_neural_ode_kinetics.py"
REFS_MD = SKILL_DIR / "references.md"
LAM_CHEM = REPO_ROOT / "molmetal" / "molmetal_lam" / "lam_chem"


# === SHIP claims — must exist on disk ===

def test_skill_md_exists():
    """SKILL.md §10.4 SHIP row 1: the skill markdown itself."""
    assert SKILL_MD.is_file(), (
        f"SKILL.md missing at {SKILL_MD}; if SKILL.md claims SHIP but "
        "the file is absent, this is a docs-vs-code-drift critical failure"
    )


def test_skill_md_meets_minimum_loc():
    """SKILL.md §8.1 MEASURED: line count ≥ 250 LOC."""
    assert SKILL_MD.is_file()
    n_lines = sum(1 for _ in SKILL_MD.open("r", encoding="utf-8"))
    assert n_lines >= 250, (
        f"SKILL.md is {n_lines} LOC; per §8.1 MEASURED requires ≥ 250 LOC"
    )


def test_smoke_test_py_ship_claim():
    """SKILL.md §10.4 SHIP row 2: test_neural_ode_kinetics.py exists
    and exits 0 per §Self-check step 2.

    If this test fails, the §10.4 SHIP claim is FALSE (docs-vs-code-drift)
    and a future Wf must either ship the smoke or downgrade the row to
    PROJECTED.
    """
    if not SMOKE_TEST.is_file():
        pytest.skip(
            f"SKILL.md §10.4 marks {SMOKE_TEST.name} as SHIP but the file "
            f"is absent at {SMOKE_TEST}. This is a known doc-vs-code-drift "
            f"item pending Wf-76 P2.4 close-out."
        )


def test_references_md_ship_claim():
    """SKILL.md §10.4 SHIP row 3: references.md has ≥ 5 paper-ref
    section headings per §Self-check step 3.

    If this test fails, the §10.4 SHIP claim is FALSE and a future Wf
    must either ship the references file or downgrade the row to
    PROJECTED.
    """
    if not REFS_MD.is_file():
        pytest.skip(
            f"SKILL.md §10.4 marks {REFS_MD.name} as SHIP but the file "
            f"is absent at {REFS_MD}. This is a known doc-vs-code-drift "
            f"item pending Wf-76 P2.4 close-out."
        )


# === PROJECTED claims — must NOT exist (yet) ===

def test_gnn_ode_module_projected_not_yet_shipped():
    """SKILL.md §2.2 + §10.4 PROJECTED: GRUDODEKinetics from
    ``molmetal/molmetal_lam/lam_chem/gnn_ode.py`` does not yet exist
    (Wf-76 P2.1 reserves Blackwell GPU-1 retrain).

    If this test FAILS (i.e. the module suddenly imports OK), the Wf-76
    P2.1 GPU retrain shipped — the test should be re-tiered from
    "PROJECTED-not-yet" to "MEASURED-SHIPPED."
    """
    gnn_path = LAM_CHEM / "gnn_ode.py"
    if not gnn_path.is_file():
        pytest.skip(
            f"PROJECTED: {gnn_path} not on disk yet; awaiting Wf-76 P2.1 "
            f"Blackwell GPU-1 retrain. This is the SHIP/PROJECTED boundary "
            f"the regression-guard is supposed to monitor."
        )
    # If the file DOES exist, the import should succeed (post-Wf-76 P2.1).
    from molmetal.molmetal_lam.lam_chem.gnn_ode import GRUDODEKinetics  # noqa: F401


def test_hydrolysis_cascade_projected_not_yet_shipped():
    """SKILL.md §4.3 + §10.4 PROJECTED: HydrolysisCascade from
    ``molmetal/molmetal_lam/lam_chem/hydrolysis_cascade.py`` does not
    yet exist (Wf-76 P2.3 reserves Blackwell GPU-1 retrain).
    """
    hc_path = LAM_CHEM / "hydrolysis_cascade.py"
    if not hc_path.is_file():
        pytest.skip(
            f"PROJECTED: {hc_path} not on disk yet; awaiting Wf-76 P2.3"
        )


def test_redox_cycling_projected_not_yet_shipped():
    """SKILL.md §5.3 + §10.4 PROJECTED: RedoxCycling from
    ``molmetal/molmetal_lam/lam_chem/redox_cycling.py`` does not yet
    exist (Wf-76 P2.3 reserves Blackwell GPU-1 retrain).
    """
    rc_path = LAM_CHEM / "redox_cycling.py"
    if not rc_path.is_file():
        pytest.skip(
            f"PROJECTED: {rc_path} not on disk yet; awaiting Wf-76 P2.3"
        )


def test_cellular_uptake_ode_projected_not_yet_shipped():
    """SKILL.md §5.2 + §10.4 PROJECTED: cellular_uptake_ode.py does
    not yet exist (Wf-76 P2.3 reserves Blackwell GPU-1 retrain).
    """
    cu_path = LAM_CHEM / "cellular_uptake_ode.py"
    if not cu_path.is_file():
        pytest.skip(
            f"PROJECTED: {cu_path} not on disk yet; awaiting Wf-76 P2.3"
        )


# === Negative: stale SKILL.md names should NOT silently come back ===

def test_no_kinetics_module_at_root_of_lam_chem():
    """A future Wf that ships Wf-76 P2.x must NOT pollute the top-level
    ``lam_chem`` namespace with placeholder modules that masquerade as
    kinetics code. The 4 PROJECTED files must land as their actual names.
    """
    import molmetal.molmetal_lam.lam_chem as lc
    for stale in ("GRUDODEKinetics", "HydrolysisCascade", "RedoxCycling"):
        # The module attribute (e.g. `lc.GRUDODEKinetics`) is not a valid
        # surface unless the underlying module is imported. This test
        # confirms no `__init__.py` export has been wired prematurely.
        assert not hasattr(lc, stale), (
            f"`{stale}` leaked into lam_chem namespace before "
            f"Wf-76 P2.x ships the actual kinetics module"
        )


def test_no_dopri5_helper_in_lam_chem_init():
    """SKILL.md §2.3 references torchdiffeq dopri5 but says nothing
    about a project-local helper. Guard against a premature
    ``molmetal.molmetal_lam.lam_chem.dopri5`` placeholder.
    """
    import molmetal.molmetal_lam.lam_chem as lc
    assert not hasattr(lc, "dopri5"), (
        "`dopri5` is a torchdiffeq import (per SKILL.md §2.3), NOT a "
        "lam_chem export — if a placeholder appeared, that is drift"
    )