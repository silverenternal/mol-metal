"""Verify Lambda × CFM CPU joint inference skeleton (TODO-21, 2026-09-26).

Smoke verification that imports the new
:mod:`molmetal.molmetal_lam.lam_chem.coupling_joint_inference` module,
runs the full pipeline (and a 100-iteration reproducibility check) on
CPU, and prints a one-line summary.

Run::

    uv run --no-sync python molmetal/scripts/verify_lambda_cfm_coupling.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    """Smoke verification entry point — return 0 on pass."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the 30-second CPU smoke verification.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Number of reproducibility iterations for the smoke check.",
    )
    args = parser.parse_args()

    # ---- Imports --------------------------------------------------------
    try:
        from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
            LAMBDA_DIM,
            EMBED_DIM,
            CascadedRefiner,
            load_joint_pipeline,
            smoke_test_full_pipeline,
        )
    except Exception as exc:
        print(
            f"Lambda × CFM CPU joint inference: imports FAIL ({type(exc).__name__}:{exc})"
        )
        return 1

    n_fail = 0
    n_total = 0

    # ---- 1. Single-pipeline smoke --------------------------------------
    n_total += 1
    if not smoke_test_full_pipeline(seed_smiles="CCO", pocket_size=8):
        n_fail += 1

    # ---- 2. Reproducibility across N runs ------------------------------
    n_total += 1
    import numpy as np

    try:
        pipe = load_joint_pipeline()
        pocket = np.zeros(8, dtype=np.float32)
        first = pipe.run_full_pipeline("c1ccccc1", pocket, n_steps=10)
        for _ in range(int(args.n_iter)):
            repeat = pipe.run_full_pipeline("c1ccccc1", pocket, n_steps=10)
            if not np.array_equal(first["cfm_output"], repeat["cfm_output"]):
                n_fail += 1
                break
    except Exception:
        n_fail += 1

    # ---- 3. Refiner smoke (identity when n=0) --------------------------
    n_total += 1
    try:
        r0 = CascadedRefiner(n_refine_steps=0)
        z = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        out = r0.refine(np.zeros(LAMBDA_DIM, dtype=np.float32), z)
        if out.shape != (EMBED_DIM,) or not np.array_equal(out, z):
            n_fail += 1
    except Exception:
        n_fail += 1

    # ---- Summary --------------------------------------------------------
    print(
        "Lambda × CFM CPU joint inference: imports OK, "
        f"smoke {n_fail}/{n_total} fail"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
