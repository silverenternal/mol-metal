"""Verify Wf-236 — Monge-Ampère regularity + Multi-parameter PH integration.

Run::

    python molmetal/scripts/verify_wf236_math_integration.py --smoke

Exercises the Wf-236 wire-in surface end-to-end on a tiny synthetic
OT plan + multi-PH feature batch and prints a one-line PASS / FAIL
summary for each acceptance gate.

Acceptance gates
----------------
* G1 — Monge-Ampère regularity integrated into OT solver
* G2 — Multi-parameter PH integrated into feature extraction
* G3 — Backward compat preserved (upstream OT API re-exported)
* G4 — Existing tests still pass
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    """Smoke verification entry point — return 0 on pass."""
    parser = argparse.ArgumentParser(
        description="Wf-236 — Monge-Ampère regularity + Multi-PH smoke check.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU smoke check.",
    )
    args = parser.parse_args()

    results: dict = {}

    # ------------------------------------------------------------------
    # G1 — Monge-Ampère regularity → OT solver
    # ------------------------------------------------------------------
    try:
        import torch

        from molmetal.adapters.flow_matching_lipman.optimal_transport_v2 import (
            MeanFieldW2Plan,
            compute_ot_plan,
            ma_regularization_factor,
            ma_regularized_ot_plan,
            mean_field_score_smoothed_ot_plan,
            mean_field_step_size,
            mini_batch_ot_coupling,
        )

        # 1a. Well-conditioned input → standard OT.
        torch.manual_seed(0)
        x0 = torch.randn(16, 4)
        x1 = torch.randn(16, 4)
        idx, diag = ma_regularized_ot_plan(
            x0,
            x1,
            lambda_ma=0.01,
            mode="hungarian",
            return_diagnostics=True,
        )
        assert diag["is_regular"] is True
        assert diag["activated"] is False
        assert sorted(idx.tolist()) == list(range(16))

        # 1b. Degenerate input → MA-regularisation activates.
        x0_deg = torch.zeros(8, 3)
        x1_deg = torch.randn(8, 3)
        idx_d, diag_d = ma_regularized_ot_plan(
            x0_deg,
            x1_deg,
            lambda_ma=0.01,
            mode="hungarian",
            return_diagnostics=True,
        )
        assert diag_d["activated"] is True
        assert diag_d["is_regular"] is False
        assert sorted(idx_d.tolist()) == list(range(8))

        # 1c. MeanFieldW2Plan.verify_ma_regularity is exposed.
        plan = MeanFieldW2Plan()
        assert hasattr(plan, "verify_ma_regularity")
        diag_p = plan.verify_ma_regularity(x0)
        assert "Q" in diag_p and "is_regular" in diag_p

        results["G1_monge_ampere_integrated"] = True
        print("G1 Monge-Ampère integrated: PASS")
    except Exception as exc:
        results["G1_monge_ampere_integrated"] = False
        print(f"G1 Monge-Ampère integrated: FAIL ({type(exc).__name__}: {exc})")

    # ------------------------------------------------------------------
    # G2 — Multi-parameter PH → feature extraction
    # ------------------------------------------------------------------
    try:
        import numpy as np

        from molmetal.molmetal_lam.reward.astsr_multiph_hilbert import (
            JointFeatureConcatenator,
            JointHilbertOptimizer,
            MultiPHConcatAdapter,
            StableMultiPHExtractor,
        )

        rng = np.random.default_rng(42)
        B, d_a, n_atoms = 16, 4, 8
        astr = rng.standard_normal((B, d_a))
        coords = rng.standard_normal((B, n_atoms, 3)) * 1.5
        atoms = rng.integers(1, 18, size=(B, n_atoms)).astype(np.int64)
        mask = np.ones((B, n_atoms), dtype=bool)
        y = (astr @ rng.standard_normal(d_a)) + rng.standard_normal(B) * 0.1

        # 2a. Single-pocket compute.
        extractor = StableMultiPHExtractor(n_bins=8)
        feat_single = extractor.compute(
            coords[0],
            atoms[0],
        )
        assert feat_single.shape == (extractor.feature_dim,)

        # 2b. Batched compute.
        feat_batch = extractor.compute_batch(coords, atoms, pocket_mask=mask)
        assert feat_batch.shape == (B, extractor.feature_dim)
        assert extractor.is_stable(feat_batch) is True

        # 2c. Adapter → joint Hilbert fit.
        adapter = MultiPHConcatAdapter(extractor)
        combined = adapter.extract_joint(astr, coords, atoms, pocket_mask=mask)
        assert combined.shape == (B, d_a + extractor.feature_dim)

        joint = JointHilbertOptimizer(
            hilbert_dim=combined.shape[1],
            n_iters=50,
            lr=0.05,
        )
        fit_result = joint.fit(astr, combined[:, d_a:], y)
        assert fit_result["r2"] > -1.0  # sanity: not catastrophically bad

        results["G2_multi_ph_integrated"] = True
        print("G2 Multi-parameter PH integrated: PASS")
    except Exception as exc:
        results["G2_multi_ph_integrated"] = False
        print(f"G2 Multi-parameter PH integrated: FAIL ({type(exc).__name__}: {exc})")

    # ------------------------------------------------------------------
    # G3 — Backward compat
    # ------------------------------------------------------------------
    try:
        import torch

        from molmetal.adapters.flow_matching_lipman.optimal_transport_v2 import (
            PlanMode,
            compute_ot_plan,
            mini_batch_ot_coupling,
        )

        # 3a. Upstream symbols are re-exported.
        assert callable(compute_ot_plan)
        assert callable(mini_batch_ot_coupling)
        assert PlanMode is not None

        # 3b. compute_ot_plan still works as documented.
        torch.manual_seed(0)
        x0 = torch.randn(8, 3)
        x1 = torch.randn(8, 3)
        idx = compute_ot_plan(x0, x1, mode="hungarian")
        assert idx.shape == (8,)
        assert sorted(idx.tolist()) == list(range(8))

        # 3c. New symbols are exported.
        from molmetal.adapters.flow_matching_lipman.optimal_transport_v2 import (
            MeanFieldW2Plan,
            ma_regularized_ot_plan,
            mean_field_score_smoothed_ot_plan,
            mean_field_step_size,
        )

        assert callable(ma_regularized_ot_plan)
        assert callable(mean_field_score_smoothed_ot_plan)
        assert callable(mean_field_step_size)
        assert MeanFieldW2Plan is not None

        # 3d. Existing JointFeatureConcatenator + JointHilbertOptimizer
        # still import from astsr_multiph_hilbert.
        from molmetal.molmetal_lam.reward.astsr_multiph_hilbert import (
            JointFeatureConcatenator,
            JointHilbertOptimizer,
            JointVsSeparateComparator,
        )

        assert callable(JointFeatureConcatenator)
        assert callable(JointHilbertOptimizer)
        assert callable(JointVsSeparateComparator)

        results["G3_backward_compat"] = True
        print("G3 Backward compat: PASS")
    except Exception as exc:
        results["G3_backward_compat"] = False
        print(f"G3 Backward compat: FAIL ({type(exc).__name__}: {exc})")

    # ------------------------------------------------------------------
    # G4 — Existing tests still pass
    # ------------------------------------------------------------------
    try:
        import subprocess

        test_paths = [
            "molmetal/tests/test_optimal_transport_v2.py",
            "molmetal/tests/test_monge_ampere_regularity.py",
            "molmetal/tests/test_multi_parameter_ph.py",
            "molmetal/tests/test_astsr_multiph_hilbert.py",
            "molmetal/tests/test_multi_ph_integration.py",
            "molmetal/tests/test_multi_ph_pll_integration.py",
        ]
        all_pass = True
        for tp in test_paths:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", tp, "-q", "--no-header"],
                capture_output=True,
                text=True,
                cwd=str(_REPO_ROOT),
            )
            if r.returncode != 0:
                all_pass = False
                print(f"  test path FAIL: {tp}")
                # Print last 3 lines for context.
                for line in r.stdout.splitlines()[-3:]:
                    print(f"    {line}")
        results["G4_tests_pass"] = all_pass
        print(f"G4 Existing tests pass: {'PASS' if all_pass else 'FAIL'}")
    except Exception as exc:
        results["G4_tests_pass"] = False
        print(f"G4 Existing tests pass: FAIL ({type(exc).__name__}: {exc})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Wf-236 Verification Summary ---")
    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k}: {status}")

    all_pass = all(results.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
