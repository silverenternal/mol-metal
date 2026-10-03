#!/usr/bin/env python3
"""Smoke verification for the skeleton-jump / fragment-swap rule.

This script exercises the public surfaces of
:mod:`molmetal.molmetal_lam.lam_chem.skeleton_jump` (per the TODO-30
P2.4 brief):

  1. :class:`FragmentLibrary` — add / get / compatibility filter.
  2. :class:`SkeletonJumpRule` — swap with probability gate.
  3. :class:`DiversityAwareSkeletonJump` — diversity-aware wrapper.

It also reports the **diversity lift** — the average Tanimoto
distance between successive outputs — so callers can spot-check that
the skeleton jump is actually diversifying the trajectory.

The script is **CPU-only** — no CUDA, no torch.  It runs in <5 s on
any laptop.

Usage::

    python molmetal/scripts/verify_skeleton_jump.py --smoke
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Make ``molmetal`` importable when run as a script.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.molmetal_lam.lam_chem.skeleton_jump import (
    DEFAULT_TARGET_DIVERSITY,
    DiversityAwareSkeletonJump,
    FragmentLibrary,
    SkeletonJumpRule,
    register_skeleton_jump,
)


# ---------------------------------------------------------------------------
# Smoke fixtures
# ---------------------------------------------------------------------------
# (smiles, name) — diverse fragments: aromatic / aliphatic / ring / chain.
_FRAGMENTS: Tuple[Tuple[str, str], ...] = (
    ("c1ccccc1", "benzene"),
    ("c1ccncc1", "pyridine"),
    ("C1CCCCC1", "cyclohexane"),
    ("CCCC", "butane"),
    ("CCO", "ethanol"),
    ("c1ccc(O)cc1", "phenol"),
)

# Scaffolds to seed the swap — start with a chain (CCCCCC) because
# the CPU skeleton's swap-cut algorithm produces well-defined
# RDKit-sanitisable prefixes for chains; aromatic-ring prefixes
# (e.g. the first 2 atoms of benzene) leave a broken aromaticity
# that RDKit rejects.  TODO-31 will wire a learned swap that
# handles rings properly.
_SCAFFOLDS: Tuple[str, ...] = (
    "CCCCCC",  # hexane (chain) — start here for swap-ability
    "c1ccccc1",  # benzene (aromatic)
    "c1ccncc1",  # pyridine (heteroaromatic)
)


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------
def _smoke_library() -> dict:
    """Build a 6-fragment library and exercise get_compatible_fragments."""
    lib = FragmentLibrary(min_atoms=3, max_atoms=10)
    for smi, name in _FRAGMENTS:
        lib.add_fragment(smi, name)
    n_total = len(lib)
    # For each scaffold, count compatible fragments.
    per_scaffold: List[Tuple[str, int]] = []
    for sc in _SCAFFOLDS:
        compat = lib.get_compatible_fragments(sc)
        per_scaffold.append((sc, len(compat)))
    return {
        "n_fragments": n_total,
        "names": lib.names(),
        "per_scaffold_compat": per_scaffold,
    }


def _smoke_skeleton_jump(n_steps: int = 20, seed: int = 0) -> dict:
    """Run the bare :class:`SkeletonJumpRule` ``n_steps`` times and report diversity lift.

    Uses swap_prob=0.5 for the smoke harness so the trajectory reliably
    picks up swaps — the production default of 0.1 leaves a 12% chance
    of zero swaps in 20 steps, which makes the smoke check flaky.  The
    production rule still uses :data:`DEFAULT_SWAP_PROB`; the smoke
    harness overrides it to ensure the pipeline is exercised.
    """
    lib = FragmentLibrary(min_atoms=3, max_atoms=10)
    for smi, name in _FRAGMENTS:
        lib.add_fragment(smi, name)
    rule = SkeletonJumpRule(lib, swap_prob=0.5, rng=random.Random(seed))
    # Walk: start from the first scaffold, apply at position 4 each step.
    # Position 4 gives a 4-atom prefix for the chain scaffold (CCCCCC →
    # CCCC), which is RDKit-sanitisable as a swap target.
    traj: List[str] = [_SCAFFOLDS[0]]
    n_swap = 0
    for _ in range(n_steps):
        mol = traj[-1]
        out = rule.apply(mol, position=4)
        if not out:
            out = mol
        if out != mol:
            n_swap += 1
        traj.append(out)
    return {
        "n_steps": n_steps,
        "n_swaps": n_swap,
        "swap_rate": n_swap / n_steps if n_steps else 0.0,
        "trajectory_len": len(traj),
        "first": traj[0],
        "last": traj[-1],
    }


def _smoke_diversity_aware(n_steps: int = 20, seed: int = 0) -> dict:
    """Run :class:`DiversityAwareSkeletonJump` and report history size.

    Same swap_prob override as the bare-rule smoke (0.5) so the
    diversity gate is actually exercised — with the production
    default of 0.1 the wrapper would often see no candidate swaps
    to gate.
    """
    lib = FragmentLibrary(min_atoms=3, max_atoms=10)
    for smi, name in _FRAGMENTS:
        lib.add_fragment(smi, name)
    base = SkeletonJumpRule(lib, swap_prob=0.5, rng=random.Random(seed))
    wrap = DiversityAwareSkeletonJump(
        base,
        target_diversity=DEFAULT_TARGET_DIVERSITY,
        history_capacity=16,
        rng=random.Random(seed + 1),
    )
    traj: List[str] = [_SCAFFOLDS[0]]
    for _ in range(n_steps):
        mol = traj[-1]
        out = wrap.apply_with_diversity(mol, history=traj)
        if not out:
            out = mol
        traj.append(out)
    # Diversity lift = mean pairwise 1 - Tanimoto similarity over the
    # trajectory.  Higher means the trajectory is more diverse.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fps = []
    for s in traj:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
            fps.append(fp)
        except Exception:
            continue
    n_pairs = 0
    sum_dist = 0.0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sum_dist += 1.0 - sim
            n_pairs += 1
    mean_dist = sum_dist / n_pairs if n_pairs else 0.0
    return {
        "n_steps": n_steps,
        "trajectory_len": len(traj),
        "history_size": len(wrap.history()),
        "history_capacity": wrap.history_capacity,
        "mean_pairwise_diversity": mean_dist,
        "diversity_lift": mean_dist,
        "first": traj[0],
        "last": traj[-1],
    }


def _smoke_registration() -> dict:
    """Run the register_skeleton_jump hook and report the count."""
    n = register_skeleton_jump()
    return {"registered_count": n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke verification for the skeleton-jump / fragment-swap rule "
            "(TODO-30 P2.4)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke harness only (default behaviour).",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    print("[1/4] FragmentLibrary")
    lib = _smoke_library()
    print(
        f"      n_fragments={lib['n_fragments']}, "
        f"compat_per_scaffold={lib['per_scaffold_compat']}"
    )
    print("[2/4] SkeletonJumpRule")
    sj = _smoke_skeleton_jump()
    print(
        f"      steps={sj['n_steps']}, swaps={sj['n_swaps']}, "
        f"swap_rate={sj['swap_rate']:.2%}, "
        f"trajectory_len={sj['trajectory_len']}"
    )
    print("[3/4] DiversityAwareSkeletonJump")
    da = _smoke_diversity_aware()
    print(
        f"      steps={da['n_steps']}, trajectory_len={da['trajectory_len']}, "
        f"history={da['history_size']}/{da['history_capacity']}, "
        f"diversity_lift={da['diversity_lift']:.3f}"
    )
    print("[4/4] register_skeleton_jump")
    rg = _smoke_registration()
    print(f"      registered_count={rg['registered_count']}")
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.2f}s")

    # Pass criteria:
    #  - library has at least 3 fragments,
    #  - skeleton-jump rule performs >= 1 swap in 20 steps (prob=0.1 → ~2 expected),
    #  - diversity lift > 0 (trajectory is at least somewhat diverse),
    #  - registration returns >= 1 rule.
    library_ok = lib["n_fragments"] >= 3
    swap_ok = sj["n_swaps"] >= 1
    diversity_ok = da["diversity_lift"] > 0.0
    register_ok = rg["registered_count"] >= 1
    if library_ok and swap_ok and diversity_ok and register_ok:
        print("SKELETON_JUMP SMOKE PASS")
        return 0
    print("SKELETON_JUMP SMOKE FAIL")
    if not library_ok:
        print(f"  library n_fragments={lib['n_fragments']} < 3")
    if not swap_ok:
        print(f"  skeleton-jump n_swaps={sj['n_swaps']} < 1")
    if not diversity_ok:
        print(f"  diversity_lift={da['diversity_lift']:.3f} <= 0")
    if not register_ok:
        print(f"  registered_count={rg['registered_count']} < 1")
    return 1


if __name__ == "__main__":
    sys.exit(main())
