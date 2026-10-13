# SPDX-License-Identifier: Apache-2.0
"""The re-run sweep must report the DENOMINATOR and never pool error types.

WHY THIS EXISTS
---------------
The 2026-10-12 sweep died in pocket 2 of 10 and its one surviving line read
``generated 1fmc: n=51 mean=-6.407``. That ``n=51`` is a SURVIVOR count over an
unstated attempted set. The molecules that dropped out were exactly the ones
that failed to become a PDBQT or to parse an atype, so the mean is conditioned
on a molecule property (RDKit embeddability), not a random sample. Quoting it
as the cohort's binding score is the same class of error as the nine other
instances in ``docs/CFM_STATUS_2026_10_12.md``.

Two things have to hold for the re-run to not repeat it:

1. every mean ships beside ``n_attempted``, and is flagged
   ``_survivor_conditioned`` so it cannot be quoted bare;
2. failures are split by exception TYPE, because ``MetalPDBQTUnavailable``
   (a property of the molecule) and ``VinaAtomTypeRejected`` (a harness
   regression) are not the same event and must never be averaged into one
   failure count.

RED CHECK (reported in docs/CFM_STATUS_2026_10_12.md): deleting the
``n_attempted`` key from ``summarise_pocket`` fails
``test_denominator_is_reported_beside_every_mean``; collapsing
``errors_by_type`` into a single ``n_failed`` counter fails
``test_error_types_are_not_pooled``. Both were run and restored.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _mod():
    name = "cfm_vina_generated_sweep"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(REPO / "molmetal" / "scripts" / "cfm_vina_generated_sweep.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _rec(smiles: str, kcal, status: str, error_type=None, atoms=None):
    return {
        "pocket_id": "1fmc",
        "smiles": smiles,
        "kcal_mol": kcal,
        "n_atoms_ligand": atoms,
        "status": status,
        "error_type": error_type,
    }


def _cohort():
    """One docked + two failures of DIFFERENT types.

    Mirrors the real failure mix: a molecule RDKit cannot embed, and a
    harness-level atype rejection.
    """
    return [
        _rec("CC[Re]", -6.4, "MEASURED_GENERIC_ATOM", atoms=41),
        _rec("bad_embed", None, "DOCK_FAILED", "MetalPDBQTUnavailable"),
        _rec("bad_atype", None, "DOCK_FAILED", "VinaAtomTypeRejected"),
    ]


def test_denominator_is_reported_beside_every_mean():
    """The dead sweep's exact defect: a mean with no stated attempted set."""
    s = _mod().summarise_pocket("1fmc", _cohort())
    assert s["n_attempted"] == 3, (
        f"n_attempted={s['n_attempted']!r}; without it the mean is conditioned "
        "on success and the failures are invisible"
    )
    assert s["n_docked"] == 1
    assert s["n_failed"] == 2
    assert s["docked_fraction"] == 1 / 3


def test_survivor_conditioned_mean_is_flagged():
    s = _mod().summarise_pocket("1fmc", _cohort())
    assert s["_survivor_conditioned"] is True, (
        "a mean over docked molecules must announce that it excludes the failures"
    )
    assert s["_mean_excludes"] == 2


def test_error_types_are_not_pooled():
    """Three of today's nine errors were a mismatch between a number and the
    claim about it. Pooling error types hides exactly that."""
    s = _mod().summarise_pocket("1fmc", _cohort())
    by_type = s["errors_by_type"]
    assert by_type.get("MetalPDBQTUnavailable") == 1, (
        f"errors_by_type={by_type!r}; an un-embeddable molecule is a property "
        "of the cohort and must be countable on its own"
    )
    assert by_type.get("VinaAtomTypeRejected") == 1, (
        f"errors_by_type={by_type!r}; a Vina atype rejection is a HARNESS "
        "regression, and pooling it with cohort properties would hide it — "
        "that is how 9 of 10 pockets were lost without anyone noticing"
    )


def test_harness_rejection_is_visible_as_its_own_class():
    """The boron class must be countable, and must be zero-or-investigated."""
    S = _mod()
    exc = RuntimeError(
        "PDBQT parsing error: Atom type B is not a valid AutoDock type"
    )
    assert S.classify_exception(exc) == "VinaAtomTypeRejected"
    assert (
        S.classify_exception(
            type("MetalPDBQTUnavailable", (RuntimeError,), {})("could not embed")
        )
        == "MetalPDBQTUnavailable"
    )


def test_cohort_summary_keeps_the_denominator():
    """The whole-cohort mean must carry the same denominator discipline."""
    S = _mod()
    per_pocket = [S.summarise_pocket("1fmc", _cohort())]
    agg = S.summarise_cohort(_cohort(), per_pocket, "generated")
    assert agg["n_attempted"] == 3
    assert agg["n_docked"] == 1
    assert agg["_survivor_conditioned"] is True
    assert agg["_mean_excludes"] == 2


def test_ligand_size_is_recorded_for_the_control_comparison():
    """Cisplatin is 4 atoms and the cohort ~40; Vina favours larger ligands, so
    the confound has to be visible in the record, not only in prose."""
    s = _mod().summarise_pocket("1fmc", _cohort())
    assert s["mean_n_atoms_ligand_over_docked"] == 41.0
