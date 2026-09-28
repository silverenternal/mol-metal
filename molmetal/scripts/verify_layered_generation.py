#!/usr/bin/env python3
"""Smoke verification for the layered CPU generation skeleton.

This script exercises the three public surfaces of
``molmetal.molmetal_lam.lam_chem.layered_generation`` (per the
TODO-30 P2.3 brief):

  1. :class:`ScaffoldStage` — Bemis-Murcko scaffold extraction.
  2. :class:`AttachStage` — R-group attachment via single-bond joins.
  3. :class:`DecorateStage` — substituent + stereo decoration.
  4. :class:`LayeredGenerationPipeline` — orchestrator that runs the
     three stages in sequence and returns the JSON contract dict.

The script is **CPU-only** — no CUDA, no torch.  It runs in <5 s
on any laptop.

Usage::

    python molmetal/scripts/verify_layered_generation.py --smoke
    python molmetal/scripts/verify_layered_generation.py
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Make ``molmetal`` importable when run as a script (matches the
# pattern in :file:`molmetal/scripts/verify_multi_ph_adapter.py`).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.molmetal_lam.lam_chem.layered_generation import (
    AttachStage,
    DecorateStage,
    LayeredGenerationPipeline,
    ScaffoldStage,
    is_valid_smiles,
)


# ---------------------------------------------------------------------------
# Smoke fixtures — drug-like inputs that span the Bemis-Murcko distribution.
# ---------------------------------------------------------------------------
_DRUG_SEEDS: Tuple[str, ...] = (
    "Cc1ccc(NC(=O)c2ccccc2)cc1",    # paracetamol
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",    # ibuprofen
    "Oc1ccc(C(=O)O)cc1",             # 4-hydroxybenzoic acid
    "Nc1ccc(C(=O)O)cc1",             # 4-aminobenzoic acid
)


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------
def _smoke_scaffold_stage() -> dict:
    """Run :class:`ScaffoldStage` on the smoke seeds."""
    stage = ScaffoldStage(rng=random.Random(0))
    scaffolds: List[str] = []
    valid_count = 0
    for smi in _DRUG_SEEDS:
        sc = stage.generate(smi)
        scaffolds.append(sc)
        if sc and stage.is_valid_scaffold(sc):
            valid_count += 1
    validity = valid_count / len(_DRUG_SEEDS) if _DRUG_SEEDS else 0.0
    return {
        "n_seeds": len(_DRUG_SEEDS),
        "n_scaffold": sum(1 for s in scaffolds if s),
        "n_valid": valid_count,
        "mean_scaffold_validity": validity,
    }


def _smoke_attach_stage() -> dict:
    """Run :class:`AttachStage` on the smoke seeds with [Cl] and [OH]."""
    stage = AttachStage(rng=random.Random(1))
    n_compat = 0
    n_total = 0
    for smi in _DRUG_SEEDS:
        for r in ("[Cl]", "[OH]"):
            if stage.check_compatibility(smi, r):
                n_compat += 1
            n_total += 1
    return {
        "n_checks": n_total,
        "n_compat": n_compat,
        "compat_rate": n_compat / n_total if n_total else 0.0,
    }


def _smoke_decorate_stage() -> dict:
    """Run :class:`DecorateStage` on the smoke seeds."""
    stage = DecorateStage(rng=random.Random(2))
    n_ok = 0
    n_total = len(_DRUG_SEEDS)
    for smi in _DRUG_SEEDS:
        out = stage.decorate_with_report(smi, ["[F]", "[Cl]", "[OH]"])
        if out.valid and out.smiles:
            n_ok += 1
    return {
        "n_seeds": n_total,
        "n_valid": n_ok,
        "valid_rate": n_ok / n_total if n_total else 0.0,
    }


def _smoke_pipeline() -> dict:
    """Run the full :class:`LayeredGenerationPipeline`."""
    pipeline = LayeredGenerationPipeline(
        scaffold_stage=ScaffoldStage(rng=random.Random(10)),
        attach_stage=AttachStage(rng=random.Random(11)),
        decorate_stage=DecorateStage(rng=random.Random(12)),
    )
    n_valid = 0
    n_total = len(_DRUG_SEEDS)
    n_distinct = set()
    for smi in _DRUG_SEEDS:
        out = pipeline.generate(
            smi,
            r_groups=["[Cl]", "[OH]"],
            decorations=["[F]", "[NH2]"],
        )
        if out["valid"]:
            n_valid += 1
            n_distinct.add(out["final_smiles"])
    return {
        "n_seeds": n_total,
        "n_valid": n_valid,
        "n_distinct": len(n_distinct),
        "pass_rate": n_valid / n_total if n_total else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke verification for the layered CPU generation skeleton "
            "(TODO-30 P2.3)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke harness only (default behaviour).",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    print("[1/4] ScaffoldStage")
    sc = _smoke_scaffold_stage()
    print(
        f"      seeds={sc['n_seeds']}, scaffolds={sc['n_scaffold']}, "
        f"valid={sc['n_valid']}, mean_validity={sc['mean_scaffold_validity']:.2%}"
    )
    print("[2/4] AttachStage")
    at = _smoke_attach_stage()
    print(
        f"      checks={at['n_checks']}, compat={at['n_compat']}, "
        f"compat_rate={at['compat_rate']:.2%}"
    )
    print("[3/4] DecorateStage")
    de = _smoke_decorate_stage()
    print(
        f"      seeds={de['n_seeds']}, valid={de['n_valid']}, "
        f"valid_rate={de['valid_rate']:.2%}"
    )
    print("[4/4] LayeredGenerationPipeline")
    pl = _smoke_pipeline()
    print(
        f"      seeds={pl['n_seeds']}, valid={pl['n_valid']}, "
        f"distinct={pl['n_distinct']}, pass_rate={pl['pass_rate']:.2%}"
    )
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.2f}s")

    # Pass criteria: scaffold validity >= 50%, pipeline pass rate >= 50%.
    scaffold_ok = sc["mean_scaffold_validity"] >= 0.5
    pipeline_ok = pl["pass_rate"] >= 0.5
    if scaffold_ok and pipeline_ok:
        print("LAYERED_GENERATION SMOKE PASS")
        return 0
    else:
        print("LAYERED_GENERATION SMOKE FAIL")
        if not scaffold_ok:
            print(
                f"  scaffold mean validity {sc['mean_scaffold_validity']:.2%} < 50%"
            )
        if not pipeline_ok:
            print(f"  pipeline pass rate {pl['pass_rate']:.2%} < 50%")
        return 1


if __name__ == "__main__":
    sys.exit(main())