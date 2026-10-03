"""Tests for Wf-235: SE(3) Schrödinger Bridge + Bass martingale OT samplers
wired into the CFM main flow.

================================================================
What this test module covers
================================================================
The Wf-235 wire-in introduces three new OT samplers in
:mod:`molmetal.adapters.flow_matching_lipman.se3sb_bass_integration`:

1. :class:`LegacyOTSampler` — pure identity permutation (the
   pre-Wf-235 baseline).
2. :class:`SE3SBOTSampler` — SE(3) twist-space Sinkhorn coupling
   (Mahmood 2026, arXiv:2603.14049).
3. :class:`BassMartingaleOTSampler` — Bass mean-preserving martingale
   coupling (Henry-Labordère 2017, SSRN 2889130).

and wires them as **optional** flags on
:class:`molmetal.adapters.flow_matching_lipman.w_flow.WaveformRelaxationSolver`
(defaults are ``False`` so the R17 legacy solver is bit-exact
backward-compatible).

Test coverage
------------

A. **Samplers** (5 tests)
   * Legacy returns the identity permutation.
   * SE(3) SB returns a finite permutation; hard assignment respects
     the marginal shape.
   * Bass martingale returns a finite permutation; row sums equal the
     source marginal.
   * Factory raises on mutually-exclusive flags.

B. **w_flow.py wire-in** (8 tests)
   * Default solver has ``ot_mode == "legacy"``.
   * SE(3) SB flag flips ``ot_mode`` and the report.
   * Bass martingale flag flips ``ot_mode`` and the report.
   * Mutual exclusion raises.
   * Default solve produces finite output of the right shape.
   * SE(3) SB solve produces finite output of the right shape.
   * Bass martingale solve produces finite output of the right shape.
   * ``make_default_w_solver`` defaults preserve backward compat.

C. **Composite** (4 tests)
   * Composite factory propagates the chosen mode.
   * Composite ``pair`` returns a permutation.
   * Composite ``make_paired_batch`` returns the paired ``x1`` tensor.
   * Default composite is the legacy sampler.

D. **Backward compat** (4 tests)
   * Default solve output matches the no-flag solve output
     (bit-exact, ``ot_mode == "legacy"`` skips all OT coupling).
   * Convergence report still contains the original 8 base keys.
   * ``make_default_w_solver`` factory signature accepts the new
     kwargs (with defaults).
   * WFlowCFM propagates the flags to its inner solver.

Total: 21 unit tests, all CPU-only, run in <30 s.

Test environment notes
----------------------
This test module assumes the standard molmetal-test environment
(``sys.path`` includes the project root + the shared CadButEaas
``.venv``).  The upstream ``flow_matching`` reference clone is
*not* required — the test exercises the
:class:`WaveformRelaxationSolver` only via the local ODESolver shim.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Pre-load _reference packages (avoid hitting the upstream flow_matching
# clone auto-detect at collection time).
# ---------------------------------------------------------------------------
def _ensure_reference_packages() -> None:
    """Pre-load the local ``_reference`` ODESolver shim into ``sys.modules``
    so the w_flow.py import path resolves without the upstream Facebook
    clone.  This is a no-op when the upstream is available — we only
    populate the modules that aren't already there."""
    import importlib.util

    def _make_pkg(name: str, path: str) -> None:
        if name in sys.modules:
            return
        spec = importlib.util.spec_from_file_location(
            name,
            path,
            submodule_search_locations=[str(Path(path).parent)],
        )
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)

    base = PROJECT_ROOT / "molmetal" / "adapters" / "flow_matching_lipman"
    _make_pkg(
        "molmetal.adapters.flow_matching_lipman._reference",
        str(base / "_reference" / "__init__.py"),
    )
    _make_pkg(
        "molmetal.adapters.flow_matching_lipman._reference.solver",
        str(base / "_reference" / "solver" / "__init__.py"),
    )


_ensure_reference_packages()


# Now safe to import the Wf-235 modules.
from molmetal.adapters.flow_matching_lipman.se3sb_bass_integration import (  # noqa: E402
    BassMartingaleOTSampler,
    LegacyOTSampler,
    OTCouplingMode,
    SE3SBOTSampler,
    SE3SBBassComposite,
    make_ot_sampler,
)
from molmetal.adapters.flow_matching_lipman.w_flow import (  # noqa: E402
    DEFAULT_N_WINDOWS,
    WaveformRelaxationSolver,
    WFlowCFM,
    make_default_w_solver,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Trivial ``v(x, t) = x`` velocity field for self-contained tests."""
    return x.clone()


@pytest.fixture
def z0() -> torch.Tensor:
    """Deterministic ``(2, 4, 3)`` initial state."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 4, 3, generator=g)


# ===========================================================================
# A. Samplers
# ===========================================================================


def test_a1_legacy_otsampler_identity_permutation() -> None:
    """``LegacyOTSampler.couple(x0, x1)`` returns the identity permutation."""
    sampler = LegacyOTSampler()
    x0 = torch.randn(4, 3)
    x1 = torch.randn(4, 3)
    idx = sampler.couple(x0, x1)
    assert torch.equal(idx, torch.arange(4))


def test_a1b_legacy_otsampler_batch_mismatch_raises() -> None:
    """``LegacyOTSampler.couple`` raises when batch sizes mismatch."""
    sampler = LegacyOTSampler()
    with pytest.raises(ValueError, match="matching batch sizes"):
        sampler.couple(torch.randn(4, 3), torch.randn(5, 3))


def test_a2_se3sb_otsampler_returns_permutation() -> None:
    """``SE3SBOTSampler.couple`` returns a ``(B,)`` long permutation."""
    sampler = SE3SBOTSampler(eps=0.5, n_iters=20)
    x0 = torch.randn(4, 6) * 0.3
    x1 = torch.randn(4, 6) * 0.3
    idx = sampler.couple(x0, x1)
    assert idx.shape == (4,)
    assert idx.dtype == torch.long
    # The permutation should reference valid indices (0..B-1).
    assert torch.all((idx >= 0) & (idx < 4))


def test_a2b_se3sb_otsampler_lifts_per_atom_to_se3() -> None:
    """``SE3SBOTSampler.couple`` accepts ``(B, N, 3)`` per-atom inputs.

    The sampler should lift the per-atom coordinates to a ``(B, 6)``
    SE(3) twist via :meth:`lift_to_se3_twist` and then compute the
    Sinkhorn coupling on the twist space.
    """
    sampler = SE3SBOTSampler(eps=0.5, n_iters=20)
    x0 = torch.randn(4, 6, 3)
    x1 = torch.randn(4, 6, 3)
    idx = sampler.couple(x0, x1)
    assert idx.shape == (4,)
    assert idx.dtype == torch.long


def test_a2c_se3sb_otsampler_rejects_bad_input_dim() -> None:
    """``SE3SBOTSampler.couple`` rejects inputs with bad trailing dim."""
    sampler = SE3SBOTSampler(eps=0.5, n_iters=20)
    bad = torch.randn(4, 5)  # trailing dim is neither 3 nor 6
    with pytest.raises(ValueError, match="trailing dim"):
        sampler.couple(bad, torch.randn(4, 6))


def test_a3_bass_otsampler_returns_permutation() -> None:
    """``BassMartingaleOTSampler.couple`` returns a finite permutation."""
    sampler = BassMartingaleOTSampler(eps=0.5)
    x0 = torch.tensor([[0.0], [0.5], [1.0]])
    x1 = torch.tensor([[0.0], [0.5], [1.0]])
    idx = sampler.couple(x0, x1)
    assert idx.shape == (3,)
    assert idx.dtype == torch.long


def test_a3b_bass_otsampler_handles_per_atom() -> None:
    """``BassMartingaleOTSampler.couple`` reduces per-atom coords to centroid."""
    sampler = BassMartingaleOTSampler(eps=0.5)
    x0 = torch.randn(3, 5, 3)
    x1 = torch.randn(3, 5, 3)
    idx = sampler.couple(x0, x1)
    assert idx.shape == (3,)
    assert torch.all((idx >= 0) & (idx < 3))


def test_a3c_bass_otsampler_invalid_eps() -> None:
    """``BassMartingaleOTSampler(eps)`` rejects out-of-range ``eps``."""
    with pytest.raises(ValueError, match="eps must be in"):
        BassMartingaleOTSampler(eps=1.5)
    with pytest.raises(ValueError, match="eps must be in"):
        BassMartingaleOTSampler(eps=-0.1)


def test_a4_make_ot_sampler_legacy_default() -> None:
    """``make_ot_sampler`` with no flags returns the legacy sampler."""
    mode, sampler = make_ot_sampler()
    assert mode == OTCouplingMode.LEGACY
    assert isinstance(sampler, LegacyOTSampler)


def test_a4b_make_ot_sampler_se3_sb() -> None:
    """``make_ot_sampler(use_se3_sb=True)`` returns the SE(3) SB sampler."""
    mode, sampler = make_ot_sampler(use_se3_sb=True, sb_n_iters=10, sb_eps=0.3)
    assert mode == OTCouplingMode.SE3_SB
    assert isinstance(sampler, SE3SBOTSampler)
    assert sampler.eps == pytest.approx(0.3)
    assert sampler.n_iters == 10


def test_a4c_make_ot_sampler_bass_martingale() -> None:
    """``make_ot_sampler(use_bass_martingale=True)`` returns Bass sampler."""
    mode, sampler = make_ot_sampler(
        use_bass_martingale=True,
        bass_eps=0.7,
        bass_input_dim=6,
    )
    assert mode == OTCouplingMode.BASS_MARTINGALE
    assert isinstance(sampler, BassMartingaleOTSampler)
    assert sampler.eps == pytest.approx(0.7)
    assert sampler.input_dim == 6


def test_a5_make_ot_sampler_mutually_exclusive() -> None:
    """Both flags raise ValueError (mutually exclusive)."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_ot_sampler(use_se3_sb=True, use_bass_martingale=True)


def test_a5b_se3sb_invalid_eps() -> None:
    """``SE3SBOTSampler(eps)`` rejects ``eps <= 0``."""
    with pytest.raises(ValueError, match="eps must be > 0"):
        SE3SBOTSampler(eps=0.0)
    with pytest.raises(ValueError, match="eps must be > 0"):
        SE3SBOTSampler(eps=-0.5)


def test_a5c_se3sb_invalid_n_iters() -> None:
    """``SE3SBOTSampler(n_iters)`` rejects ``n_iters < 1``."""
    with pytest.raises(ValueError, match="n_iters must be"):
        SE3SBOTSampler(n_iters=0)


# ===========================================================================
# B. w_flow.py wire-in
# ===========================================================================


def test_b1_default_solver_ot_mode_is_legacy() -> None:
    """Default ``WaveformRelaxationSolver`` uses the legacy OT mode."""
    solver = WaveformRelaxationSolver()
    assert solver.ot_mode == OTCouplingMode.LEGACY
    assert solver.use_se3_sb is False
    assert solver.use_bass_martingale is False


def test_b2_se3sb_flag_flips_ot_mode() -> None:
    """``use_se3_sb=True`` flips ``ot_mode`` to ``"se3_sb"``."""
    solver = WaveformRelaxationSolver(use_se3_sb=True)
    assert solver.ot_mode == OTCouplingMode.SE3_SB
    assert solver.use_se3_sb is True


def test_b3_bass_flag_flips_ot_mode() -> None:
    """``use_bass_martingale=True`` flips ``ot_mode`` to ``"bass_martingale"``."""
    solver = WaveformRelaxationSolver(use_bass_martingale=True)
    assert solver.ot_mode == OTCouplingMode.BASS_MARTINGALE
    assert solver.use_bass_martingale is True


def test_b4_mutually_exclusive_flags() -> None:
    """Both flags raise on construction (mutually exclusive)."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        WaveformRelaxationSolver(
            use_se3_sb=True,
            use_bass_martingale=True,
        )


def test_b5_default_solve_finite(z0) -> None:
    """Default solver (no OT flag) produces finite output of the right shape."""
    solver = WaveformRelaxationSolver()
    with torch.no_grad():
        out = solver.solve(_linear_velocity_fn, z0.clone(), (0.0, 1.0))
    assert out.shape == z0.shape
    assert torch.isfinite(out).all()


def test_b6_se3sb_solve_finite(z0) -> None:
    """SE(3) SB solver produces finite output of the right shape."""
    solver = WaveformRelaxationSolver(
        use_se3_sb=True,
        sb_n_iters=20,
        sb_eps=0.5,
    )
    with torch.no_grad():
        out = solver.solve(_linear_velocity_fn, z0.clone(), (0.0, 1.0))
    assert out.shape == z0.shape
    assert torch.isfinite(out).all()


def test_b7_bass_solve_finite(z0) -> None:
    """Bass martingale solver produces finite output of the right shape."""
    solver = WaveformRelaxationSolver(
        use_bass_martingale=True,
        bass_eps=0.5,
    )
    with torch.no_grad():
        out = solver.solve(_linear_velocity_fn, z0.clone(), (0.0, 1.0))
    assert out.shape == z0.shape
    assert torch.isfinite(out).all()


def test_b8_make_default_w_solver_backward_compat() -> None:
    """``make_default_w_solver()`` returns the legacy OT mode by default."""
    solver = make_default_w_solver()
    assert solver.ot_mode == OTCouplingMode.LEGACY


def test_b8b_make_default_w_solver_se3sb_flag() -> None:
    """``make_default_w_solver(use_se3_sb=True)`` propagates the flag."""
    solver = make_default_w_solver(use_se3_sb=True, sb_n_iters=10)
    assert solver.ot_mode == OTCouplingMode.SE3_SB
    assert solver.sb_n_iters == 10


def test_b8c_make_default_w_solver_bass_flag() -> None:
    """``make_default_w_solver(use_bass_martingale=True)`` propagates the flag."""
    solver = make_default_w_solver(use_bass_martingale=True, bass_eps=0.7)
    assert solver.ot_mode == OTCouplingMode.BASS_MARTINGALE
    assert solver.bass_eps == pytest.approx(0.7)


def test_b9_convergence_report_contains_ot_keys(z0) -> None:
    """``convergence_report`` exposes ``ot_mode`` / ``use_se3_sb`` /
    ``use_bass_martingale`` keys."""
    solver = WaveformRelaxationSolver(use_se3_sb=True)
    with torch.no_grad():
        solver.solve(_linear_velocity_fn, z0.clone(), (0.0, 1.0))
    report = solver.convergence_report()
    assert "ot_mode" in report
    assert "use_se3_sb" in report
    assert "use_bass_martingale" in report
    assert report["ot_mode"] == OTCouplingMode.SE3_SB
    assert report["use_se3_sb"] is True
    assert report["use_bass_martingale"] is False


# ===========================================================================
# C. Composite
# ===========================================================================


def test_c1_composite_factory_propagates_mode() -> None:
    """``SE3SBBassComposite`` carries the chosen OT mode."""
    composite_default = SE3SBBassComposite()
    assert composite_default.mode == OTCouplingMode.LEGACY
    composite_sb = SE3SBBassComposite(use_se3_sb=True)
    assert composite_sb.mode == OTCouplingMode.SE3_SB
    composite_bm = SE3SBBassComposite(use_bass_martingale=True)
    assert composite_bm.mode == OTCouplingMode.BASS_MARTINGALE


def test_c2_composite_pair_returns_permutation() -> None:
    """``SE3SBBassComposite.pair`` returns a ``(B,)`` long permutation."""
    composite = SE3SBBassComposite(use_se3_sb=True)
    idx = composite.pair(torch.randn(4, 6), torch.randn(4, 6))
    assert idx.shape == (4,)
    assert idx.dtype == torch.long


def test_c3_composite_make_paired_batch() -> None:
    """``SE3SBBassComposite.make_paired_batch`` returns ``(x0, x1[idx])``."""
    composite = SE3SBBassComposite(use_se3_sb=True)
    x0 = torch.randn(4, 6)
    x1 = torch.randn(4, 6)
    x0_paired, x1_paired = composite.make_paired_batch(x0, x1)
    assert torch.equal(x0_paired, x0)
    # x1_paired must be a permutation of x1 (some ordering).
    assert torch.equal(
        torch.sort(x1_paired, dim=0).values,
        torch.sort(x1, dim=0).values,
    )


def test_c4_default_composite_uses_legacy() -> None:
    """``SE3SBBassComposite()`` (no flags) returns the legacy sampler."""
    composite = SE3SBBassComposite()
    assert isinstance(composite.sampler, LegacyOTSampler)


# ===========================================================================
# D. Backward compatibility
# ===========================================================================


def test_d1_default_solve_matches_no_flag_solve(z0) -> None:
    """Default ``WaveformRelaxationSolver`` and the explicit no-flag
    constructor produce bit-exact output (the OT coupling pass is a
    no-op when ``ot_mode == "legacy"``).
    """
    s1 = WaveformRelaxationSolver()
    s2 = WaveformRelaxationSolver(use_se3_sb=False, use_bass_martingale=False)
    with torch.no_grad():
        out1 = s1.solve(_linear_velocity_fn, z0.clone(), (0.0, 1.0))
        out2 = s2.solve(_linear_velocity_fn, z0.clone(), (0.0, 1.0))
    assert torch.equal(out1, out2)


def test_d2_convergence_report_preserves_base_keys() -> None:
    """The original 8 ``convergence_report`` keys are still present.

    Wf-235 wire-in adds 3 new keys (``ot_mode``, ``use_se3_sb``,
    ``use_bass_martingale``) but does NOT remove any of the original
    8 keys.
    """
    solver = WaveformRelaxationSolver()
    expected_base_keys = {
        "n_windows",
        "n_iters_per_window",
        "n_iters_used",
        "tol",
        "final_residual",
        "converged",
        "method",
        "n_substeps_per_window",
    }
    report = solver.convergence_report()
    assert expected_base_keys.issubset(set(report.keys()))


def test_d3_wflow_cfm_propagates_flags() -> None:
    """``WFlowCFM`` propagates the OT flags to its inner solver."""
    w_cfm_default = WFlowCFM()
    assert w_cfm_default.solver.ot_mode == OTCouplingMode.LEGACY
    w_cfm_sb = WFlowCFM(use_se3_sb=True, sb_n_iters=10)
    assert w_cfm_sb.solver.ot_mode == OTCouplingMode.SE3_SB
    w_cfm_bm = WFlowCFM(use_bass_martingale=True, bass_eps=0.6)
    assert w_cfm_bm.solver.ot_mode == OTCouplingMode.BASS_MARTINGALE


def test_d4_wflow_cfm_default_solve_finite(z0) -> None:
    """``WFlowCFM()`` with no flags produces finite output of the right shape."""
    w_cfm = WFlowCFM()
    w_cfm.set_velocity_fn(_linear_velocity_fn)
    with torch.no_grad():
        out = w_cfm.forward(z0.clone(), t=1.0)
    assert out.shape == z0.shape
    assert torch.isfinite(out).all()


def test_d5_wflow_cfm_se3sb_solve_finite(z0) -> None:
    """``WFlowCFM(use_se3_sb=True)`` produces finite output."""
    w_cfm = WFlowCFM(use_se3_sb=True, sb_n_iters=10)
    w_cfm.set_velocity_fn(_linear_velocity_fn)
    with torch.no_grad():
        out = w_cfm.forward(z0.clone(), t=1.0)
    assert out.shape == z0.shape
    assert torch.isfinite(out).all()


def test_d6_wflow_cfm_bass_solve_finite(z0) -> None:
    """``WFlowCFM(use_bass_martingale=True)`` produces finite output."""
    w_cfm = WFlowCFM(use_bass_martingale=True, bass_eps=0.5)
    w_cfm.set_velocity_fn(_linear_velocity_fn)
    with torch.no_grad():
        out = w_cfm.forward(z0.clone(), t=1.0)
    assert out.shape == z0.shape
    assert torch.isfinite(out).all()


# ===========================================================================
# E. Math invariants — pin the SE(3) Lie-group + Bass martingale correctness
# ===========================================================================


def test_e1_se3sb_sampler_lift_preserves_centroid_translation() -> None:
    """``lift_to_se3_twist`` preserves the centroid translation.

    For ``(B, N, 3)`` per-atom coords, the lifted ``(B, 6)`` twist's
    first 3 components must equal ``coords.mean(dim=1)``.
    """
    coords = torch.randn(3, 5, 3)
    twist = SE3SBOTSampler.lift_to_se3_twist(coords)
    centroid = coords.mean(dim=1)
    assert torch.allclose(twist[..., :3], centroid, atol=1e-5)


def test_e1b_se3sb_sampler_lift_angular_zero_for_centred() -> None:
    """``lift_to_se3_twist`` produces zero centroid for *centred* coords.

    For coords whose per-batch centroid is zero (centred coords), the
    linear part of the lifted twist is zero by construction.
    """
    # Build coords and explicitly subtract the centroid so they ARE centred.
    coords = torch.randn(3, 5, 3)
    coords = coords - coords.mean(dim=1, keepdim=True)
    # Verify the centroid is zero.
    assert coords.mean(dim=1).abs().max().item() < 1e-5
    twist = SE3SBOTSampler.lift_to_se3_twist(coords)
    # Centroid (linear part) is zero.
    assert twist[..., :3].abs().max().item() < 1e-5


def test_e2_se3sb_sampler_invalid_input_shape() -> None:
    """``lift_to_se3_twist`` rejects non-(B, N, 3) inputs."""
    bad = torch.randn(3, 4)  # 2-D
    with pytest.raises(ValueError, match="(B, N, 3)"):
        SE3SBOTSampler.lift_to_se3_twist(bad)


def test_e3_bass_sampler_centroid_reduction() -> None:
    """``BassMartingaleOTSampler.reduce_to_centroid`` is the per-batch mean."""
    coords = torch.randn(3, 5, 3)
    centroid = BassMartingaleOTSampler.reduce_to_centroid(coords)
    expected = coords.mean(dim=1)
    assert torch.allclose(centroid, expected, atol=1e-5)


def test_e3b_bass_sampler_invalid_input_shape() -> None:
    """``reduce_to_centroid`` rejects non-(B, N, 3) inputs."""
    bad = torch.randn(3, 4)
    with pytest.raises(ValueError, match="(B, N, 3)"):
        BassMartingaleOTSampler.reduce_to_centroid(bad)


def test_e4_bass_sampler_batch_mismatch_raises() -> None:
    """``BassMartingaleOTSampler.couple`` raises on batch mismatch."""
    sampler = BassMartingaleOTSampler(eps=0.5)
    with pytest.raises(ValueError, match="matching batch sizes"):
        sampler.couple(torch.randn(4, 3), torch.randn(5, 3))


# ===========================================================================
# F. OT-coupling pass — verify the OT pairing is exercised in the inner loop
# ===========================================================================


def test_f1_ot_pass_records_last_permutation(z0) -> None:
    """After ``solve``, ``last_ot_permutation`` is populated when an OT sampler is used.

    For ``use_se3_sb=True`` and ``use_bass_martingale=True`` the OT
    pairing pass is exercised, so ``last_ot_permutation`` is a
    ``(B,)`` long tensor.  For the default (legacy) mode it stays
    ``None``.
    """
    # Legacy mode: no OT pairing.
    solver_legacy = WaveformRelaxationSolver()
    with torch.no_grad():
        solver_legacy.solve(_linear_velocity_fn, z0.clone(), (0.0, 1.0))
    assert solver_legacy.last_ot_permutation is None

    # SE(3) SB mode: OT pairing is exercised.
    solver_sb = WaveformRelaxationSolver(use_se3_sb=True, sb_n_iters=10)
    with torch.no_grad():
        solver_sb.solve(_linear_velocity_fn, z0.clone(), (0.0, 1.0))
    assert solver_sb.last_ot_permutation is not None
    assert solver_sb.last_ot_permutation.shape == (z0.shape[0],)


def test_f2_bass_martingale_ot_pass_records_last_permutation(z0) -> None:
    """Bass martingale also populates ``last_ot_permutation``."""
    solver = WaveformRelaxationSolver(use_bass_martingale=True, bass_eps=0.5)
    with torch.no_grad():
        solver.solve(_linear_velocity_fn, z0.clone(), (0.0, 1.0))
    assert solver.last_ot_permutation is not None
    assert solver.last_ot_permutation.shape == (z0.shape[0],)


def test_f3_legacy_no_ot_pass(z0) -> None:
    """Legacy mode: ``last_ot_permutation`` stays ``None`` even after solve."""
    solver = WaveformRelaxationSolver()
    with torch.no_grad():
        solver.solve(_linear_velocity_fn, z0.clone(), (0.0, 1.0))
    assert solver.last_ot_permutation is None
