"""Wf-36 Phase 1 — 1-molecule closure theorem smoke check.

Runs the closure-theorem assertion hook on a single molecule (default:
cisplatin) with the 5 canonical click rules + the default 7-partner
seed set at depth 1.  Used to confirm the closure-theorem import path
works in a fresh environment and to pin a known-good closure-ok
verdict before/after code changes.

Usage
-----
::

    uv run --no-sync python molmetal/scripts/smoke_closure.py
    uv run --no-sync python molmetal/scripts/smoke_closure.py --smiles '...'
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Sequence


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


PARTNERS = ["C#CCN", "CCN=[N+]=[N-]", "C#CC",
            "CS", "C=CC", "OC(=O)c1ccccc1", "NCC"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the smoke closure check; return 0 on pass."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smiles", default="[NH3][Pt]([NH3])([Cl])([Cl])",
        help="SMILES to test (default: cisplatin).",
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--max-products", type=int, default=200)
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne,
    )
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    rules = [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]
    partners = [MoleculeClosedTerm.from_smiles(s, embed_3d=False)
                for s in PARTNERS]

    try:
        t = MoleculeClosedTerm.from_smiles(args.smiles, embed_3d=False)
    except Exception as exc:
        print(f"PARSE FAIL: {type(exc).__name__}: {exc}")
        return 2

    print(f"smiles: {args.smiles}")
    print(f"canonical: {t.canonical_smiles()}")
    print(f"n_atoms: {len(t.atoms)}")
    print(f"max_depth: {args.depth}")
    print(f"n_partners: {len(partners)}")
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = closure_theorem(
            start_term=t, click_rules=rules, max_depth=args.depth,
            max_products=args.max_products, partner_terms=partners,
        )
    print(f"closure_ok: {ok}")
    print(f"wall_time_s: {time.perf_counter() - t0:.4f}")
    lr = getattr(closure_theorem, "last_result", {}) or {}
    print(f"n_visited: {int(lr.get('n_visited', 0))}")
    print(f"n_products: {int(lr.get('n_products', 0))}")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())