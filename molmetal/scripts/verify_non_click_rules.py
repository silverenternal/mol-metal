#!/usr/bin/env python3
"""Smoke / verification script for non-click med-chem rules (TODO-30 P1.1).

Usage
-----
::

    python molmetal/scripts/verify_non_click_rules.py --smoke

Outputs (stdout, JSON-friendly lines)::

    n_rules: <int>          # 3 by default
    registry_size: <int>    # size of the populated registry
    smoke_pass_rate: <float in [0, 1]>   # fraction of smoke cases that fired
    n_smoke_cases: <int>
    smoke_results: [...]    # per-rule pass/fail (1/0 each)

Honest framing
--------------
This is a **smoke** script — it imports the rules, fires them on a
hand-curated set of canonical med-chem substrates, and reports the
rule instance + per-rule fire success rate.  The output is the
rule's *measured* pass rate on the smoke cases; it does NOT
validate that the rule is producing the *correct* product (that
is the unit-test suite's job in
``molmetal/tests/test_non_click_medchem_rules.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "molmetal")]

from molmetal.molmetal_lam.lam_chem.non_click_medchem_rules import (
    BocProtectionRule,
    NonClickMedChemRegistry,
    PeptideBondRule,
    ReductiveAminationRule,
    register_non_click_rules,
)


# ---------------------------------------------------------------------------
# Smoke-test fixtures (hand-curated canonical med-chem inputs).
# ---------------------------------------------------------------------------

# (rule_type, callable_apply_args, expected_pass_bool)
SMOKE_CASES: List[Tuple[str, tuple, bool]] = [
    # Peptide-bond: Gly + Gly must produce a product.
    ("peptide_bond", ("NCC(=O)O", "NCC(=O)O"), True),
    # Peptide-bond: Ala + Phe must produce a product.
    ("peptide_bond", ("CC(N)C(=O)O", "NCC(Cc1ccccc1)C(=O)O"), True),
    # Peptide-bond: invalid substrate (no -COOH) must be rejected.
    ("peptide_bond", ("CN", "NCC(=O)O"), False),
    # Peptide-bond: sub-threshold (formic acid) must be rejected.
    ("peptide_bond", ("OC=O", "NCC(=O)O"), False),
    # Boc-protection: methylamine must be protected.
    ("boc_protection", ("CN",), True),
    # Boc-protection: glycine must be protected.
    ("boc_protection", ("NCC(=O)O",), True),
    # Boc-protection: bare benzene (no N) must be rejected.
    ("boc_protection", ("c1ccccc1",), False),
    # Reductive amination: benzaldehyde + methylamine must produce product.
    ("reductive_amination", ("O=Cc1ccccc1", "CN"), True),
    # Reductive amination: acetaldehyde + ethylamine must produce product.
    ("reductive_amination", ("CC=O", "CCN"), True),
    # Reductive amination: ketone + amine must be rejected (aldehyde-only).
    ("reductive_amination", ("CC(=O)C", "CN"), False),
]


def _build_registry() -> NonClickMedChemRegistry:
    """Construct a fresh registry and register the canonical 3 rules."""
    registry = NonClickMedChemRegistry()
    register_non_click_rules(registry)
    return registry


def _smoke_run(registry: NonClickMedChemRegistry) -> dict:
    """Run the smoke cases and aggregate the per-rule pass/fail result.

    Returns a dict with::

        {
            "n_smoke_cases": int,
            "smoke_results": [int, int, ...],   # 1 = expected pass, 0 = expected fail
            "n_passed": int,
            "smoke_pass_rate": float,
        }
    """
    n_passed = 0
    smoke_results: List[int] = []
    for rule_type, args, expected_pass in SMOKE_CASES:
        rule = registry.get_rule(rule_type)
        if rule is None:
            result = 0
        else:
            try:
                outcome = rule.apply(*args)
            except Exception:
                outcome = None
            # When expected_pass is True, the rule must return a non-None SMILES.
            # When expected_pass is False, the rule must return None (rejection).
            actual_pass = outcome is not None
            result = 1 if actual_pass == expected_pass else 0
        smoke_results.append(result)
        n_passed += int(result)
    pass_rate = n_passed / max(1, len(SMOKE_CASES))
    return {
        "n_smoke_cases": len(SMOKE_CASES),
        "smoke_results": smoke_results,
        "n_passed": n_passed,
        "smoke_pass_rate": pass_rate,
    }


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke-test fixture set and report pass rate",
    )
    args = parser.parse_args(argv)

    registry = _build_registry()
    n_rules = len(registry.get_rules())

    if args.smoke:
        smoke = _smoke_run(registry)
        report = {
            "n_rules": n_rules,
            "registry_size": n_rules,
            **smoke,
        }
    else:
        # No --smoke -> just report registry size + rule names.
        report = {
            "n_rules": n_rules,
            "registry_size": n_rules,
            "rules": list(registry.get_rules().keys()),
        }
    print(json.dumps(report, indent=2))
    # Exit 0 regardless of smoke pass-rate (smoke is informational,
    # the unit tests are the gate).
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
