"""Verification script for Integration 4: Multi-PH -> PLL signature.

Smoke script that exercises the integration layer end-to-end:

* Spins up a default :class:`MultiPHPLLSignature`.
* Computes augmented signatures on a tiny SMILES list.
* Reports expressivity (``multi_ph_topology_strict``) and wall-time
  per ``compute`` call (in ms).

Usage
-----
    python molmetal/scripts/verify_multi_ph_pll.py --smoke

The script is *not* a unit test — it lives in ``scripts/`` to make the
R18 P3 close-out runnable as a smoke check on a fresh host.  The
CI gate runs the canonical pytest suite; this script is for humans.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SMILES_LIST: List[str] = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "NCCO",
    "CCN",
    "c1ccncc1",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-check Multi-PH -> PLL integration",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick smoke check on a tiny SMILES list.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of repeats for the wall-time benchmark (default 5).",
    )
    args = parser.parse_args()

    # 1. Build a default integration signature.
    from molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration import (
        MultiPHPLLSignature,
        compare_multi_ph_vs_single_pll,
    )

    sig = MultiPHPLLSignature(n_bins=16, k=8)
    print(
        "[verify] signature: n_bins={n_bins} k={k} knn={knn}".format(
            n_bins=sig.n_bins, k=sig.k, knn=sig.knn,
        )
    )

    # 2. Compute augmented signatures on the smoke list (timed).
    t0 = time.perf_counter()
    results = []
    for smi in SMILES_LIST:
        feat, eig = sig.compute(smi)
        results.append(
            {
                "smiles": smi,
                "feat_shape": list(feat.shape),
                "feat_sum": float(feat.sum().item()),
                "eig_shape": list(eig.shape),
                "eig_top1": float(eig[0].item()),
            }
        )
    elapsed = time.perf_counter() - t0
    print(f"[verify] computed {len(SMILES_LIST)} signatures in "
          f"{1000.0 * elapsed / max(1, len(SMILES_LIST)):.2f} ms each")

    # 3. Expressivity + wall-time benchmark via the canonical
    #    helper.
    n_runs = max(1, int(args.n_runs))
    comparison = compare_multi_ph_vs_single_pll(
        SMILES_LIST, n_runs=n_runs, signature=sig,
    )
    print("[verify] comparison summary:")
    for key, val in comparison.items():
        print(f"  - {key}: {val}")

    # 4. Emit the per-molecule results as well.
    out = {
        "signature": {
            "n_bins": sig.n_bins,
            "k": sig.k,
            "knn": sig.knn,
        },
        "per_molecule": results,
        "comparison": comparison,
    }
    print("[verify] JSON:")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
