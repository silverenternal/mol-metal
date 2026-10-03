#!/usr/bin/env python3
"""Smoke / verification script for FG-compat / leave-group veto (TODO-30 P1.2).

Usage
-----
::

    python molmetal/scripts/verify_fg_compat_veto.py --smoke

Outputs (stdout, JSON-friendly lines)::

    n_veto_rules: <int>          # FG vetoed pairs + LG vetoed pairs
    n_fg_vetoed: <int>           # vetoed FG × FG cells
    n_lg_vetoed: <int>           # vetoed FG × LG cells
    veto_rate: <float in [0, 1]> # fraction of SMOKE_CASES vetoed
    n_smoke_cases: <int>
    smoke_results: [...]         # per-case pass/fail (1 = passes veto, 0 = vetoed)

Honest framing
--------------
This is a **smoke** script — it imports the veto, fires the
canonical 15 + 12 = 27 veto rules on a hand-curated set of
canonical med-chem inputs, and reports the rule instance + the
overall veto rate on the smoke cases.  The output is the
rule's *measured* veto rate on the smoke cases; it does NOT
validate that the rule is producing the *correct* veto decision
on novel inputs (that is the unit-test suite's job in
``molmetal/tests/test_fg_compat_veto.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "molmetal")]

from molmetal.molmetal_lam.lam_chem.fg_compat_veto import (
    FGCompatMatrix,
    LeaveGroupVeto,
    VetoAugmentedReducer,
    register_fg_compat_veto,
)


# ---------------------------------------------------------------------------
# Mock reducer for the VetoAugmentedReducer smoke tests
# ---------------------------------------------------------------------------


class _MockReducer:
    """Minimal reducer exposing ``reduce(input) -> str``.

    Used to verify the wrapper's pass-through behaviour and the
    cumulative veto-rate counter.  Echoes the input as
    ``"PASS:<input>"``.
    """

    def __init__(self, prefix: str = "PASS:") -> None:
        self.prefix = prefix
        self.n_calls = 0

    def reduce(self, input_smiles: str) -> str:
        self.n_calls += 1
        return f"{self.prefix}{input_smiles}"


# ---------------------------------------------------------------------------
# Smoke-test fixtures (hand-curated canonical FG + LG pairs).
# ---------------------------------------------------------------------------
# Each tuple is:
#   (description, fg_pair_or_None, leaving_group_or_None,
#    reactant_smiles, expected_veto_bool)
# ``expected_veto_bool`` is True iff the veto MUST fire on the
# supplied inputs.  We aggregate per-case pass/fail and report
# the smoke pass rate.
# ---------------------------------------------------------------------------

SMOKE_CASES: List[Tuple[str, Tuple[str, str] | None, str | None, str, bool]] = [
    # FG vetoes — must fire.
    (
        "aldehyde + aldehyde -> veto",
        ("aldehyde", "aldehyde"),
        None,
        "CC=O",
        True,
    ),
    (
        "carboxylic_acid + carboxylic_acid -> veto",
        ("carboxylic_acid", "carboxylic_acid"),
        None,
        "CC(=O)O",
        True,
    ),
    (
        "primary_amine + primary_amine -> veto",
        ("primary_amine", "primary_amine"),
        None,
        "CN",
        True,
    ),
    (
        "azide + azide -> veto",
        ("azide", "azide"),
        None,
        "CCCN=[N+]=[N-]",
        True,
    ),
    (
        "alkyne + alkyne (Glaser) -> veto",
        ("alkyne", "alkyne"),
        None,
        "C#C",
        True,
    ),
    (
        "boronic_acid + boronic_acid (homocoupling) -> veto",
        ("boronic_acid", "boronic_acid"),
        None,
        "B(O)(O)c1ccccc1",
        True,
    ),
    # FG vetoes — symmetric: same verdict in reverse order.
    (
        "aldehyde + aldehyde (reverse order) -> veto",
        ("aldehyde", "aldehyde"),
        None,
        "O=CC",
        True,
    ),
    # FG passes — must NOT veto.
    (
        "carboxylic_acid + primary_amine (peptide) -> pass",
        ("carboxylic_acid", "primary_amine"),
        None,
        "NCC(=O)O",
        False,
    ),
    (
        "azide + alkyne (CuAAC) -> pass",
        ("azide", "alkyne"),
        None,
        "CCCN=[N+]=[N-].C#CC",
        False,
    ),
    (
        "boronic_acid + aryl_halide (Suzuki) -> pass",
        ("boronic_acid", "aryl_halide"),
        None,
        "B(O)(O)c1ccccc1.Brc1ccccc1",
        False,
    ),
    # LG vetoes — must fire.
    (
        "aryl_halide + F -> veto (F is poor LG)",
        None,
        "F",
        "c1ccc(F)cc1",
        True,
    ),
    (
        "boronic_acid + F -> veto (F on B(OH)2 is poor LG)",
        None,
        "F",
        "B(O)(O)c1ccccc1",
        True,
    ),
    (
        "alkyne + F -> veto (C-F is too strong)",
        None,
        "F",
        "C#C",
        True,
    ),
    (
        "carboxylic_acid + OH -> veto (no -OH as LG on COOH)",
        None,
        "OH",
        "CC(=O)O",
        True,
    ),
    (
        "carboxylic_acid + CO2 -> veto (decarboxylation needs high T)",
        None,
        "CO2",
        "CC(=O)O",
        True,
    ),
    (
        "boronic_acid + N2 -> veto (nonsensical)",
        None,
        "N2",
        "B(O)(O)c1ccccc1",
        True,
    ),
    # LG passes — must NOT veto.
    (
        "azide + N2 -> pass (CuAAC LG)",
        None,
        "N2",
        "CCCN=[N+]=[N-]",
        False,
    ),
    (
        "alkyne + Br -> pass (Sonogashira-style LG)",
        None,
        "Br",
        "BrC#C",
        False,
    ),
    (
        "carboxylic_acid + Cl -> pass (acid chloride pathway)",
        None,
        "Cl",
        "CC(=O)O",
        False,
    ),
    # Combined FG + LG — both must veto to block; if only one
    # fires, the wrapper still blocks.
    (
        "aldehyde + aldehyde + F -> veto (FG veto)",
        ("aldehyde", "aldehyde"),
        "F",
        "CC=O",
        True,
    ),
]


def _smoke_run() -> dict:
    """Run the smoke cases against the canonical veto.

    Returns a dict with::

        {
            "n_smoke_cases": int,
            "smoke_results": [int, int, ...],   # 1 = veto decision correct, 0 = wrong
            "n_passed": int,
            "smoke_pass_rate": float,
        }
    """
    fg_matrix = FGCompatMatrix()
    lg_veto = LeaveGroupVeto()
    n_passed = 0
    smoke_results: List[int] = []
    for (
        _desc,
        fg_pair,
        leaving_group,
        reactant,
        expected_veto,
    ) in SMOKE_CASES:
        # Compute the actual veto decision.
        actual_veto = False
        if fg_pair is not None:
            fg1, fg2 = fg_pair
            if not fg_matrix.is_compatible(fg1, fg2):
                actual_veto = True
        if leaving_group is not None:
            if lg_veto.veto(reactant, leaving_group):
                actual_veto = True
        # Result: 1 iff veto decision matches expectation.
        result = 1 if actual_veto == expected_veto else 0
        smoke_results.append(result)
        n_passed += int(result)
    pass_rate = n_passed / max(1, len(SMOKE_CASES))
    return {
        "n_smoke_cases": len(SMOKE_CASES),
        "smoke_results": smoke_results,
        "n_passed": n_passed,
        "smoke_pass_rate": pass_rate,
    }


def _wrapper_smoke() -> dict:
    """Run the VetoAugmentedReducer smoke tests.

    Returns a dict with::

        {
            "n_veto_calls": int,
            "n_pass_calls": int,
            "veto_rate": float,
        }
    """
    mock = _MockReducer()
    wrapper = VetoAugmentedReducer(base_reducer=mock)
    # 2 vetoes (aldehyde+aldehyde, aryl_halide+F)
    wrapper.reduce("CC=O", fg_pair=("aldehyde", "aldehyde"))
    wrapper.reduce("c1ccc(F)cc1", leaving_group="F")
    # 2 passes (carboxylic_acid + primary_amine, no veto args)
    wrapper.reduce("NCC(=O)O", fg_pair=("carboxylic_acid", "primary_amine"))
    wrapper.reduce("CCO")
    return {
        "n_veto_calls": wrapper.n_veto_calls,
        "n_pass_calls": wrapper.n_pass_calls,
        "veto_rate": wrapper.veto_rate(),
    }


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke-test fixture set and report pass rate",
    )
    args = parser.parse_args(argv)

    n_rules = register_fg_compat_veto()
    fg = FGCompatMatrix()
    lg = LeaveGroupVeto()

    if args.smoke:
        smoke = _smoke_run()
        wrapper = _wrapper_smoke()
        # Reasons breakdown — count by veto type.
        fg_vetoes = sum(
            1 for r in smoke["smoke_results"][:10] if r == 1
        )
        # (The smoke_results array is not bucketed; we just
        # report the per-case pass/fail.)
        report = {
            "n_veto_rules": n_rules,
            "n_fg_vetoed": len(fg.vetoed_pairs),
            "n_lg_vetoed": len(lg.forbidden_pairs),
            "fg_veto_rate_smoke": smoke["smoke_pass_rate"],
            "n_smoke_cases": smoke["n_smoke_cases"],
            "smoke_results": smoke["smoke_results"],
            "n_smoke_passed": smoke["n_passed"],
            "wrapper_smoke": wrapper,
        }
    else:
        report = {
            "n_veto_rules": n_rules,
            "n_fg_vetoed": len(fg.vetoed_pairs),
            "n_lg_vetoed": len(lg.forbidden_pairs),
            "fg_matrix": {
                "n_cells": fg.n_cells,
                "n_vetoed": fg.n_vetoed,
            },
            "lg_matrix": {
                "n_cells": lg.n_cells,
                "n_vetoed": lg.n_vetoed,
            },
        }
    print(json.dumps(report, indent=2))
    # Exit 0 regardless of smoke pass-rate (smoke is informational,
    # the unit tests are the gate).
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
