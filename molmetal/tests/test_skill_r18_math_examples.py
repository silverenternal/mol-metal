"""Regression test — verifies the r18-math SKILL.md API references match on-disk code.

Background
----------
Wf-A44-J19-pilot (2026-10-15): the docs-vs-code-drift guard pattern from
Wf-A43-fixup-3 (which fixed the click-chemistry skill) is being extended
to 3 pilot skills before scaling to the remaining 26 skills. This file
guards `mol-metal-r18-math`.

The r18-math SKILL.md (§14 "Worked example" + §R18 ship reference)
claims the SE(3) Schrödinger Bridge adapter exposes a function called
``integrate_with_cfm(samples, field, n_steps, method=...)`` that
"duck-type compatible with `flow_matching.solver.ode_solver.ODESolver`".

On-disk reality (verified 2026-10-15):

  - ``molmetal/math_skills/se3_schrodinger_bridge.py`` exposes the
    primitive classes and SO(3) helpers — ``SchrodingerBridgeSE3``,
    ``SE3Geodesic``, ``LogSinkhornLoss``, ``so3_slerp``, ``so3_exp``,
    ``so3_log``, ``se3_exp``, ``se3_log``, ``check_preconditions``.
  - ``molmetal/math_skills/transport_relaxation.py`` exposes
    ``RelaxedOT``, ``BoltzmannEndpoint``, ``TransportRelaxationTerm``.
  - ``molmetal/math_skills/multi_parameter_ph.py`` exposes
    ``MultiPHSummarizer``, ``TwoParameterFiltration``,
    ``TwoParameterDiagram``, ``MultiPHKernel``, ``sliced_wasserstein_kernel``.

There is NO ``integrate_with_cfm`` on ``se3_schrodinger_bridge`` — it is
a SKILL.md prose claim that escaped the docs-vs-code-drift audit. This
test file pins the real API and asserts the prose name does NOT come
back without a corresponding adapter implementation.

Per AGENTS.md §5.5 filesystem-truth: every claim about the SKILL.md is
verified against the on-disk module, not the prose.
"""

from __future__ import annotations

# All tests in this file are CPU-only.
import pytest

from molmetal.math_skills.se3_schrodinger_bridge import (
    SE3Geodesic,
    SchrodingerBridgeSE3,
    LogSinkhornLoss,
    so3_slerp,
    so3_exp,
    so3_log,
    se3_exp,
    se3_log,
    check_preconditions,
)
from molmetal.math_skills.transport_relaxation import (
    RelaxedOT,
    BoltzmannEndpoint,
    TransportRelaxationTerm,
)
from molmetal.math_skills.multi_parameter_ph import (
    MultiPHSummarizer,
    TwoParameterFiltration,
    TwoParameterDiagram,
    MultiPHKernel,
    sliced_wasserstein_kernel,
    two_param_filtration,
)


# === T1 SE(3) Schrödinger Bridge — class instantiation + smoke ===

def test_se3_geodesic_instantiates():
    """SKILL.md §14 R18 ship: SE3Geodesic is a real class."""
    g = SE3Geodesic()
    assert g is not None


def test_schrodinger_bridge_se3_with_config():
    """SKILL.md §14: SchrodingerBridgeSE3(sigma, eps, n_iters, seed)."""
    sb = SchrodingerBridgeSE3(sigma=0.1, eps=0.05, n_iters=2, seed=0)
    assert sb is not None
    # Sigma / eps / n_iters should be queryable
    assert sb.sigma == pytest.approx(0.1)
    assert sb.eps == pytest.approx(0.05)


def test_so3_slerp_midpoint_shape():
    """SKILL.md §14: so3_slerp returns a (B, 3, 3) rotation tensor."""
    import torch
    r1 = torch.eye(3).unsqueeze(0)
    r2 = torch.tensor([[[0.0, -1.0, 0.0],
                         [1.0,  0.0, 0.0],
                         [0.0,  0.0, 1.0]]]).float()
    mid = so3_slerp(r1, r2, 0.5)
    assert tuple(mid.shape) == (1, 3, 3)


def test_so3_log_so3_exp_roundtrip():
    """SKILL.md §1.1: so3_log / so3_exp are inverses on SO(3)."""
    import torch
    r = torch.tensor([[[0.0, -1.0, 0.0],
                        [1.0,  0.0, 0.0],
                        [0.0,  0.0, 1.0]]]).float()
    omega = so3_log(r)
    r_back = so3_exp(omega)
    assert torch.allclose(r, r_back, atol=1e-5)


def test_se3_log_se3_exp_roundtrip():
    """SKILL.md §1.1: se3_log / se3_exp are inverses on SE(3)."""
    import torch
    # A rigid transform: 90-deg Z-rotation + small translation
    T = torch.eye(4).unsqueeze(0)
    T[0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0],
                                   [1.0,  0.0, 0.0],
                                   [0.0,  0.0, 1.0]])
    T[0, :3, 3] = torch.tensor([0.1, 0.0, 0.0])
    xi = se3_log(T)
    T_back = se3_exp(xi)
    assert torch.allclose(T, T_back, atol=1e-4)


# === T2 Transport Relaxation — class instantiation ===

def test_relaxed_ot_instantiates():
    """SKILL.md §14: RelaxedOT class instantiates with sensible defaults."""
    rot = RelaxedOT()
    assert rot is not None


def test_boltzmann_endpoint_instantiates():
    """SKILL.md §14: BoltzmannEndpoint class is exposed."""
    be = BoltzmannEndpoint()
    assert be is not None


def test_transport_relaxation_term_instantiates():
    """SKILL.md §14: TransportRelaxationTerm class is exposed."""
    trt = TransportRelaxationTerm()
    assert trt is not None


# === T5 Multi-Parameter Persistence — class instantiation ===

def test_multi_ph_summarizer_instantiates():
    """SKILL.md §14: MultiPHSummarizer class instantiates."""
    mps = MultiPHSummarizer()
    assert mps is not None


def test_two_parameter_filtration_instantiates():
    """SKILL.md §14: TwoParameterFiltration class is exposed."""
    tpf = TwoParameterFiltration()
    assert tpf is not None


def test_two_param_filtration_helper_exposed():
    """SKILL.md §14: two_param_filtration helper function is exposed."""
    # Just check callable — non-blocking for non-standard arg shapes
    assert callable(two_param_filtration)


def test_sliced_wasserstein_kernel_exposed():
    """SKILL.md §14: sliced_wasserstein_kernel helper is exposed."""
    assert callable(sliced_wasserstein_kernel)


# === Negative: stale SKILL.md names should NOT silently come back ===

def test_integrate_with_cfm_does_not_exist_on_se3():
    """SKILL.md §14 worked-example claims ``integrate_with_cfm`` on the
    SE(3) SB adapter; on-disk reality is that this name does not exist.
    If a future agent re-adds it without wiring the duck body, this guard
    forces a re-read of the SKILL.md drift.
    """
    import molmetal.math_skills.se3_schrodinger_bridge as m
    assert not hasattr(m, "integrate_with_cfm"), (
        "stale name `integrate_with_cfm` returned to "
        "se3_schrodinger_bridge.py — re-check the Wf-A43-style "
        "docs-vs-code-drift audit and the §14 worked example"
    )


def test_no_duck_typed_odesolver_compat():
    """SKILL.md §14 also claims ``integrate_with_cfm`` is duck-typed
    compatible with ``flow_matching.solver.ode_solver.ODESolver``. The
    real adapter path is through ``molmetal/adapters/se3_sb_integration.py``
    (per §R18 mechanism table), NOT through the math module. Make sure
    no fake ``odesolver_compat`` shim has been added to the math module.
    """
    import molmetal.math_skills.se3_schrodinger_bridge as m
    for stale in ("odesolver_compat", "to_odesolver", "as_odesolver"):
        assert not hasattr(m, stale), (
            f"stale name `{stale}` should not be on se3_schrodinger_bridge.py"
        )


def test_no_top_level_log_sinkhorn_helper():
    """A stale name sometimes appears in SKILL.md drafts as
    ``log_sinkhorn``; on disk the class is ``LogSinkhornLoss``. Guard
    against the bare ``log_sinkhorn`` import.
    """
    import molmetal.math_skills.se3_schrodinger_bridge as m
    # The class IS real; the bare function is not.
    assert hasattr(m, "LogSinkhornLoss")
    assert not hasattr(m, "log_sinkhorn"), (
        "stale name `log_sinkhorn` (function) appeared on "
        "se3_schrodinger_bridge.py — only `LogSinkhornLoss` (class) is real"
    )