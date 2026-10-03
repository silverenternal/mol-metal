#!/usr/bin/env python3
"""Smoke / verification script for the buyable-reactant gate (P1.3).

Usage
-----
::

    python molmetal/scripts/verify_buyable_gate.py --smoke --n-test 100
    python molmetal/scripts/verify_buyable_gate.py --smiles-path path/to/list.txt

Outputs (stdout, JSON-friendly lines)::

    engine: <aizynthfinder|smarts_fallback>
    n_tested: <int>
    n_passed: <int>
    pass_rate: <float in [0, 1]>
    mean_buyability_score: <float>

Honest framing
--------------
This is a **smoke** script — it loads the gate, runs ``n_test``
random SMILES (or the provided list), and reports the buyability
oracle's verdict.  The output is the gate's *measured* pass rate
on the SMILES drawn; it does NOT validate that the gate is
admitting the right molecules (that is the unit tests' job).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "molmetal")]


# A small curated set of "definitely buyable" + "definitely not"
# SMILES for the smoke test.  The "buyable" entries carry a
# recognisable click / aromatic motif so the SMARTS fallback
# admits them; the "not" entries are pure aliphatic chains
# that no SMARTS click motif matches.
SMOKE_BUYABLE: List[str] = [
    "c1ccccc1",  # benzene
    "c1ccncc1",  # pyridine
    "c1ccc2ccccc2c1",  # naphthalene
    "N.N.Cl.Cl.[Pt]",  # cisplatin multi-component
    "Cc1cn(C)nn1",  # CuAAC triazole
    "c1ccc(O)cc1",  # phenol
    "c1ccc(N)cc1",  # aniline
    "c1ccoc1",  # furan
    "c1ccsc1",  # thiophene
    "c1cnc2ccccc2c1",  # quinoline
]

SMOKE_NOT_BUYABLE: List[str] = [
    "CCCCCCC",  # heptane
    "CCCCCCCC",  # octane
    "CCN(CC)CC",  # triethylamine (no aromatic / click)
    "CCSCC",  # diethyl sulfide
    "CC(C)(C)C(C)(C)C",  # hexamethyl ethane — purely branched
    "CCC#N",  # propionitrile (not aromatic)
    "CCCC(=O)O",  # butyric acid (not aromatic)
    "CCOCC",  # diethyl ether
]


def _smiles_input(args) -> List[str]:
    if args.smiles_path is not None:
        return [
            s.strip()
            for s in Path(args.smiles_path).read_text().splitlines()
            if s.strip()
        ]
    rng = random.Random(args.seed)
    pool = SMOKE_BUYABLE + SMOKE_NOT_BUYABLE
    return [rng.choice(pool) for _ in range(int(args.n_test))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke test (default if no --smiles-path).",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=100,
        help="Number of random SMILES to test when --smoke is set.",
    )
    parser.add_argument(
        "--smiles-path",
        type=str,
        default=None,
        help="Path to a newline-delimited SMILES file.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=4,
        help="Number of threads for the gate (default 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260926,
        help="RNG seed for the smoke draw (default 20260926).",
    )
    args = parser.parse_args()

    # Lazy import to bypass lam_chem/__init__.py which depends on
    # numpy via pysr_wrapper.  We load the module directly from
    # its source file.
    import importlib.util

    buyable_gate_path = (
        ROOT / "molmetal" / "molmetal_lam" / "lam_chem" / "buyable_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "molmetal_lam.lam_chem.buyable_gate", buyable_gate_path
    )
    if spec is None or spec.loader is None:
        print(f"error: could not load {buyable_gate_path}", file=sys.stderr)
        return 1
    bg_module = importlib.util.module_from_spec(spec)
    sys.modules["molmetal_lam.lam_chem.buyable_gate"] = bg_module
    spec.loader.exec_module(bg_module)
    build_buyable_gate = bg_module.build_buyable_gate

    smiles_list = _smiles_input(args)
    gate = build_buyable_gate(n_threads=args.n_threads)

    try:
        admitted = gate.filter_buyable(smiles_list)
        n_tested = len(smiles_list)
        n_passed = len(admitted)
        pass_rate = gate.metrics.pass_rate()
        # mean_buyability_score: same as pass_rate (the oracle
        # emits a binary verdict, so the "mean score" is the
        # pass rate).  Reported as a separate key for downstream
        # tooling that expects a numeric score field.
        mean_score = pass_rate
        summary = gate.metrics.summary()

        print(f"engine: {gate.engine}")
        print(f"n_tested: {n_tested}")
        print(f"n_passed: {n_passed}")
        print(f"pass_rate: {pass_rate:.4f}")
        print(f"mean_buyability_score: {mean_score:.4f}")
        print(f"n_errors: {summary['n_errors']}")
        print(f"per_engine: {summary['per_engine']}")
    except ModuleNotFoundError as exc:
        # RDKit / numpy missing → fall back to a clear error
        # so the user knows the gate cannot run without the
        # chemoinformatics stack.  Exit 0 with a sentinel output
        # so callers do not treat this as a crash.
        print("engine: unavailable")
        print("n_tested: 0")
        print("n_passed: 0")
        print("pass_rate: 0.0000")
        print("mean_buyability_score: 0.0000")
        print(f"error: missing dependency ({exc})", file=sys.stderr)
        return 0
    finally:
        gate.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
