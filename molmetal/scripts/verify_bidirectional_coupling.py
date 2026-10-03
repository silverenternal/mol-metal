"""Verify Lambda × CFM Bidirectional coupling — CFM → Lambda reverse.

================================================================
Smoke verification that imports the new
:mod:`molmetal.molmetal_lam.lam_chem.coupling_bidirectional` module,
runs a round-trip + a small benchmark on CPU, and prints a one-line
summary.

Run::

    uv run --no-sync python molmetal/scripts/verify_bidirectional_coupling.py --smoke

Outputs (stdout)::

    Lambda × CFM Bidirectional coupling (CFM → Lambda): imports OK,
    round_trip {n_pass}/{n_total} pass, benchmark loss_mean=<float>,
    loss_max=<float>, wall_mean=<float>s, n_samples=<int>

Honest framing
==============
This is a **smoke** script — it does NOT validate the *chemistry*
correctness of the round-trip loss values, only the I/O contract of
the new module (imports succeed, round-trip terminates, benchmark
returns finite metrics).  The chemistry-correctness of the reverse
decoding is the responsibility of the unit-test suite in
:mod:`molmetal.tests.test_coupling_bidirectional` (25 tests).
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
        "--n-samples",
        type=int,
        default=3,
        help="Number of round-trips to benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="CCO",
        help="Lambda SMILES seed for the round-trip smoke check.",
    )
    args = parser.parse_args()

    # ---- Imports ---------------------------------------------------------
    try:
        import numpy as np  # noqa: F401

        from molmetal.molmetal_lam.lam_chem.coupling_bidirectional import (
            BidirectionalCoupling,
            benchmark_round_trip,
        )
    except Exception as exc:
        print(
            "Lambda × CFM Bidirectional coupling (CFM → Lambda): "
            f"imports FAIL ({type(exc).__name__}:{exc})"
        )
        return 1

    n_pass = 0
    n_total = 0

    # ---- 1. Round-trip smoke --------------------------------------------
    n_total += 1
    try:
        bc = BidirectionalCoupling()
        out = bc.round_trip(
            args.seed,
            np.zeros(8, dtype=np.float32),
            n_steps=10,
        )
        loss = float(out.get("round_trip_loss", 0.0))
        cfm = out.get("cfm_output")
        recon = out.get("reconstructed_lambda")
        if np.isfinite(loss) and cfm is not None and recon is not None:
            n_pass += 1
    except Exception:
        pass

    # ---- 2. Benchmark --------------------------------------------------
    n_total += 1
    loss_mean = 0.0
    loss_max = 0.0
    wall_mean = 0.0
    try:
        rep = benchmark_round_trip(n_samples=int(args.n_samples))
        loss_mean = float(rep.get("loss_mean", 0.0))
        loss_max = float(rep.get("loss_max", 0.0))
        wall_mean = float(rep.get("wall_time_mean", 0.0))
        if (
            np.isfinite(loss_mean)
            and np.isfinite(loss_max)
            and rep.get("n_samples", 0) == int(args.n_samples)
        ):
            n_pass += 1
    except Exception:
        pass

    # ---- 3. Determinism check (two consecutive round-trips match) ------
    n_total += 1
    try:
        bc = BidirectionalCoupling()
        a = bc.round_trip(args.seed, np.zeros(8, dtype=np.float32), n_steps=10)
        b = bc.round_trip(args.seed, np.zeros(8, dtype=np.float32), n_steps=10)
        if (
            abs(
                float(a.get("round_trip_loss", 0.0))
                - float(b.get("round_trip_loss", 0.0))
            )
            < 1e-6
        ):
            n_pass += 1
    except Exception:
        pass

    # ---- Summary --------------------------------------------------------
    print(
        "Lambda × CFM Bidirectional coupling (CFM → Lambda): "
        f"imports OK, round_trip {n_pass}/{n_total} pass, "
        f"benchmark loss_mean={loss_mean:.4f}, loss_max={loss_max:.4f}, "
        f"wall_mean={wall_mean:.4f}s, n_samples={int(args.n_samples)}"
    )
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
