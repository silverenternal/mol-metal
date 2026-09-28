"""Pytest: Integration 1 — SE(3) SB adapter → Lipman CFM ODE solver.

================================================================
Why this test module exists
================================================================
This module pins the Integration 1 contract end-to-end:

1. :class:`SE3LipmanSolver` integrates a Lipman-CFM-shaped
   ``velocity_fn(x, t) -> (B, N, 3)`` callable on the SE(3)
   manifold and returns a ``(B, N, 3)`` trajectory with the same
   shape as the input.
2. :meth:`SE3LipmanSolver.verify_se3_orthogonality` returns a max
   orthogonality error ``< 1e-3`` (the brief's hard contract).
3. :func:`compare_se3_lipman_vs_base` returns a JSON-serialisable
   dict with wall-time + accuracy + orthogonality diagnostics.
4. :class:`SE3CFMComposite` wraps a base CFM callable and the
   SE(3) solver, and :meth:`forward` returns a tensor with the
   same shape as ``z0``.
5. The integration layer does NOT modify the locked
   ``molmetal.adapters.flow_matching_lipman`` (the R17 close-out
   invariant).

All tests run on CPU in <30 s — no GPU / external dependencies
required.  Tests instantiate a small deterministic ``nn.Module``
that mimics the Lipman-CFM EGNN velocity-field contract.

Test coverage (25 tests):

 1. test_se3_lipman_solver_shape: (B, N, 3) preserved on solve
 2. test_se3_lipman_solver_dtype: dtype preserved on solve
 3. test_se3_lipman_solver_finite: trajectory finite on solve
 4. test_se3_lipman_solver_orthogonality: max orth error < 1e-3
 5. test_se3_lipman_solver_invalid_n_steps: rejects n_steps < 1
 6. test_se3_lipman_solver_invalid_method: rejects unknown method
 7. test_se3_lipman_solver_invalid_x0_shape: rejects non-(B, N, 3)
 8. test_se3_lipman_solver_invalid_t_span: rejects t0 >= t1
 9. test_se3_lipman_solver_invalid_ortho_tol: rejects non-positive tol
10. test_se3_lipman_solver_rejects_wrong_se3_adapter_type
11. test_compare_se3_vs_base_returns_dict
12. test_compare_se3_vs_base_keys_complete
13. test_compare_se3_vs_base_wall_time_finite
14. test_compare_se3_vs_base_speedup_factor_positive
15. test_compare_se3_vs_base_shape_preserved
16. test_compare_se3_vs_base_rejects_n_runs_lt_1
17. test_se3_cfm_composite_forward_shape_preserved
18. test_se3_cfm_composite_forward_dtype_preserved
19. test_se3_cfm_composite_forward_finite
20. test_se3_cfm_composite_deterministic: same seed → same output
21. test_se3_cfm_composite_callable: __call__ == forward
22. test_se3_cfm_composite_rejects_non_callable_base_cfm
23. test_se3_cfm_composite_rejects_wrong_se3_solver_type
24. test_se3_cfm_composite_rejects_t_le_zero
25. test_integration_layer_does_not_modify_flow_matching_lipman

All tests are deterministic (fixed seeds) so the suite is
reproducible across hosts.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers — a deterministic dummy EGNN velocity field for testing
# ---------------------------------------------------------------------------
class _DummyLipmanEGNN(nn.Module):
    """Minimal Lipman CFM velocity net that returns ``(B, N, 3)``.

    The output is a deterministic function of the input pose and
    the flow time::

        v(x, t) = sin(t) * x.mean(dim=1, keepdim=True)

    This is the same dummy used by
    :file:`molmetal/tests/test_se3_sb_integration.py`.  It is *not*
    SE(3)-equivariant — the integration layer's equivariance
    contract is verified separately via the orthogonality audit.
    """

    def forward(
        self,
        x: torch.Tensor,           # (B, N, 3)
        t: torch.Tensor,           # (B,) or scalar
        pocket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t_scalar = float(t.item())
        else:
            t_scalar = float(t.mean().item())
        return math.sin(t_scalar) * x.mean(dim=1, keepdim=True)


class _IdentityLipmanEGNN(nn.Module):
    """Self-map velocity net: ``v(x, t) = x``.

    The :meth:`SE3LipmanSolver.verify_se3_orthogonality` audit
    uses this velocity because under the self-map the SE(3)
    trajectory walks a non-trivial rotation (the rotation block
    changes by a non-trivial SO(3) element on every step).
    """

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pocket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x.clone()


def _make_solver(
    *,
    n_steps: int = 10,
    method: str = "midpoint",
    ortho_tol: float = 1e-3,
    velocity_net: Optional[nn.Module] = None,
) -> "SE3LipmanSolver":
    """Helper — construct an :class:`SE3LipmanSolver` for tests."""
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField
    from molmetal.adapters.se3_lipman_integration import SE3LipmanSolver

    if velocity_net is None:
        velocity_net = _DummyLipmanEGNN()
    se3_adapter = SE3SBVelocityField(base_velocity_net=velocity_net)
    return SE3LipmanSolver(
        base_lipman_solver=None,
        se3_adapter=se3_adapter,
        n_steps=n_steps,
        method=method,
        ortho_tol=ortho_tol,
    )


def _make_composite(
    *,
    n_steps: int = 10,
    method: str = "midpoint",
    base_cfm: Optional[Callable] = None,
) -> "SE3CFMComposite":
    """Helper — construct an :class:`SE3CFMComposite` for tests."""
    from molmetal.adapters.se3_lipman_integration import (
        SE3CFMComposite,
        SE3LipmanSolver,
    )

    if base_cfm is None:
        base_cfm = _DummyLipmanEGNN()
    se3_solver = _make_solver(n_steps=n_steps, method=method)
    return SE3CFMComposite(
        base_cfm=base_cfm,
        se3_solver=se3_solver,
        n_steps=n_steps,
        method=method,
    )


# ---------------------------------------------------------------------------
# Test 1: SE3LipmanSolver.solve preserves (B, N, 3) shape
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_shape():
    """``solve`` returns a tensor of shape ``(B, N, 3)`` matching ``x0``.

    The Integration 1 brief requires the solver to be a *drop-in*
    for the R17 ODESolver — same shape contract, same dtype, same
    finite-output guarantee.
    """
    solver = _make_solver(n_steps=10)
    x0 = torch.randn(2, 5, 3)
    out = solver.solve(
        velocity_fn=_DummyLipmanEGNN(),
        x0=x0,
        t_span=(0.0, 1.0),
    )
    assert out.shape == x0.shape, (
        f"SE3LipmanSolver.solve must preserve (B, N, 3); got "
        f"{tuple(out.shape)} for input {tuple(x0.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 2: SE3LipmanSolver.solve preserves dtype
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_dtype():
    """``solve`` preserves the input dtype (default float32)."""
    solver = _make_solver(n_steps=5)
    x0 = torch.randn(2, 5, 3).double()  # float64
    out = solver.solve(
        velocity_fn=_DummyLipmanEGNN(),
        x0=x0,
        t_span=(0.0, 1.0),
    )
    assert out.dtype == x0.dtype, (
        f"SE3LipmanSolver.solve must preserve dtype; got "
        f"{out.dtype} for input {x0.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 3: SE3LipmanSolver.solve produces finite output
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_finite():
    """``solve`` produces a finite trajectory for ``n_steps >= 1``."""
    solver = _make_solver(n_steps=20)
    torch.manual_seed(42)
    x0 = torch.randn(4, 6, 3)
    out = solver.solve(
        velocity_fn=_DummyLipmanEGNN(),
        x0=x0,
        t_span=(0.0, 1.0),
    )
    assert torch.isfinite(out).all(), (
        f"SE3LipmanSolver.solve output must be finite; got NaN/Inf in {out}"
    )


# ---------------------------------------------------------------------------
# Test 4: SE3LipmanSolver.verify_se3_orthogonality: max orth error < 1e-3
# ---------------------------------------------------------------------------
def test_se3_lipman_orthogonality():
    """The orthogonality audit returns ``max orth error < 1e-3``.

    The Integration 1 brief hard contract is
    ``max orth error < 1e-3``.  We assert the value returned by
    :meth:`verify_se3_orthogonality` is below this floor.
    """
    solver = _make_solver(
        n_steps=20,
        velocity_net=_IdentityLipmanEGNN(),
    )
    err = solver.verify_se3_orthogonality(n_runs=3)
    assert err < 1e-3, (
        f"SE3LipmanSolver.verify_se3_orthogonality must return error "
        f"< 1e-3; got {err}"
    )


# ---------------------------------------------------------------------------
# Test 5: SE3LipmanSolver rejects n_steps < 1
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_invalid_n_steps():
    """``n_steps < 1`` raises :class:`ValueError` at construction."""
    with pytest.raises(ValueError):
        _make_solver(n_steps=0)


# ---------------------------------------------------------------------------
# Test 6: SE3LipmanSolver rejects unknown method
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_invalid_method():
    """Unknown ``method`` strings raise :class:`ValueError`."""
    with pytest.raises(ValueError):
        _make_solver(method="rk4")


# ---------------------------------------------------------------------------
# Test 7: SE3LipmanSolver rejects non-(B, N, 3) x0
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_invalid_x0_shape():
    """``x0`` with wrong rank raises :class:`ValueError` on solve."""
    solver = _make_solver(n_steps=5)
    bad = torch.randn(3, 4)  # 2-D, not 3-D
    with pytest.raises(ValueError):
        solver.solve(velocity_fn=_DummyLipmanEGNN(), x0=bad, t_span=(0.0, 1.0))


# ---------------------------------------------------------------------------
# Test 8: SE3LipmanSolver rejects inverted t_span
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_invalid_t_span():
    """``t_span[0] >= t_span[1]`` raises :class:`ValueError` on solve."""
    solver = _make_solver(n_steps=5)
    x0 = torch.randn(2, 4, 3)
    with pytest.raises(ValueError):
        solver.solve(
            velocity_fn=_DummyLipmanEGNN(),
            x0=x0,
            t_span=(1.0, 0.0),  # inverted
        )


# ---------------------------------------------------------------------------
# Test 9: SE3LipmanSolver rejects non-positive ortho_tol
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_invalid_ortho_tol():
    """``ortho_tol <= 0`` raises :class:`ValueError` at construction."""
    with pytest.raises(ValueError):
        _make_solver(ortho_tol=0.0)
    with pytest.raises(ValueError):
        _make_solver(ortho_tol=-1.0)


# ---------------------------------------------------------------------------
# Test 10: SE3LipmanSolver rejects wrong se3_adapter type
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_rejects_wrong_se3_adapter_type():
    """Non-:class:`SE3SBVelocityField` ``se3_adapter`` raises TypeError."""
    from molmetal.adapters.se3_lipman_integration import SE3LipmanSolver

    with pytest.raises(TypeError):
        SE3LipmanSolver(
            base_lipman_solver=None,
            se3_adapter=torch.nn.Module(),  # wrong type
            n_steps=10,
        )


# ---------------------------------------------------------------------------
# Test 11: compare_se3_lipman_vs_base returns a dict
# ---------------------------------------------------------------------------
def test_compare_se3_vs_base_returns_dict():
    """``compare_se3_lipman_vs_base`` returns a JSON-serialisable dict."""
    from molmetal.adapters.flow_matching_lipman._reference.solver import ODESolver
    from molmetal.adapters.se3_lipman_integration import compare_se3_lipman_vs_base

    base = ODESolver(method="midpoint", n_steps=10)
    se3_solver = _make_solver(n_steps=10)
    report = compare_se3_lipman_vs_base(
        lipman_solver=base,
        se3_solver=se3_solver,
        n_runs=2,
    )
    assert isinstance(report, dict), (
        f"compare_se3_lipman_vs_base must return a dict; got {type(report).__name__}"
    )


# ---------------------------------------------------------------------------
# Test 12: compare_se3_lipman_vs_base keys complete
# ---------------------------------------------------------------------------
def test_compare_se3_vs_base_keys_complete():
    """The comparison dict contains every required key."""
    from molmetal.adapters.flow_matching_lipman._reference.solver import ODESolver
    from molmetal.adapters.se3_lipman_integration import compare_se3_lipman_vs_base

    base = ODESolver(method="midpoint", n_steps=10)
    se3_solver = _make_solver(n_steps=10)
    report = compare_se3_lipman_vs_base(
        lipman_solver=base,
        se3_solver=se3_solver,
        n_runs=2,
    )
    required_keys = {
        "se3_wall_time_s",
        "base_wall_time_s",
        "speedup_factor",
        "accuracy_diff_norm",
        "max_orthogonality_err",
        "n_runs",
        "n_steps_base",
        "method_base",
        "shape_preserved",
    }
    assert required_keys.issubset(report.keys()), (
        f"compare_se3_lipman_vs_base must include keys {required_keys}; "
        f"got {set(report.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 13: compare_se3_lipman_vs_base wall-time is finite
# ---------------------------------------------------------------------------
def test_compare_se3_vs_base_wall_time_finite():
    """``se3_wall_time_s`` is a finite positive float."""
    from molmetal.adapters.flow_matching_lipman._reference.solver import ODESolver
    from molmetal.adapters.se3_lipman_integration import compare_se3_lipman_vs_base

    base = ODESolver(method="midpoint", n_steps=10)
    se3_solver = _make_solver(n_steps=10)
    report = compare_se3_lipman_vs_base(
        lipman_solver=base,
        se3_solver=se3_solver,
        n_runs=2,
    )
    assert math.isfinite(report["se3_wall_time_s"]), (
        f"se3_wall_time_s must be finite; got {report['se3_wall_time_s']}"
    )
    assert report["se3_wall_time_s"] > 0.0, (
        f"se3_wall_time_s must be positive; got {report['se3_wall_time_s']}"
    )


# ---------------------------------------------------------------------------
# Test 14: compare_se3_lipman_vs_base speedup_factor is positive
# ---------------------------------------------------------------------------
def test_compare_se3_vs_base_speedup_factor_positive():
    """When the base solver succeeds, ``speedup_factor`` is positive."""
    from molmetal.adapters.flow_matching_lipman._reference.solver import ODESolver
    from molmetal.adapters.se3_lipman_integration import compare_se3_lipman_vs_base

    base = ODESolver(method="midpoint", n_steps=10)
    se3_solver = _make_solver(n_steps=10)
    report = compare_se3_lipman_vs_base(
        lipman_solver=base,
        se3_solver=se3_solver,
        n_runs=2,
    )
    sf = report["speedup_factor"]
    if sf is not None:
        assert sf > 0.0, (
            f"speedup_factor must be positive; got {sf}"
        )


# ---------------------------------------------------------------------------
# Test 15: compare_se3_lipman_vs_base shape_preserved
# ---------------------------------------------------------------------------
def test_compare_se3_vs_base_shape_preserved():
    """``shape_preserved`` is ``True`` when both solvers return finite output."""
    from molmetal.adapters.flow_matching_lipman._reference.solver import ODESolver
    from molmetal.adapters.se3_lipman_integration import compare_se3_lipman_vs_base

    # ODESolver requires a velocity_model at construction; pass a
    # deterministic ``(x, t) -> v`` closure so sample() can integrate.
    base = ODESolver(
        velocity_model=lambda x, t: x.clone(),
        method="midpoint",
        n_steps=10,
    )
    se3_solver = _make_solver(n_steps=10)
    report = compare_se3_lipman_vs_base(
        lipman_solver=base,
        se3_solver=se3_solver,
        n_runs=2,
    )
    assert report["shape_preserved"], (
        f"shape_preserved must be True when both solvers succeed; got "
        f"report={report}"
    )


# ---------------------------------------------------------------------------
# Test 16: compare_se3_lipman_vs_base rejects n_runs < 1
# ---------------------------------------------------------------------------
def test_compare_se3_vs_base_rejects_n_runs_lt_1():
    """``n_runs < 1`` raises :class:`ValueError`."""
    from molmetal.adapters.se3_lipman_integration import compare_se3_lipman_vs_base

    se3_solver = _make_solver(n_steps=10)
    with pytest.raises(ValueError):
        compare_se3_lipman_vs_base(
            lipman_solver=lambda x: x,
            se3_solver=se3_solver,
            n_runs=0,
        )


# ---------------------------------------------------------------------------
# Test 17: SE3CFMComposite.forward preserves shape
# ---------------------------------------------------------------------------
def test_se3_cfm_composite_forward_shape_preserved():
    """``SE3CFMComposite.forward`` returns a tensor of shape ``(B, N, 3)``."""
    composite = _make_composite(n_steps=10)
    torch.manual_seed(0)
    z0 = torch.randn(2, 5, 3)
    out = composite.forward(z0, t=1.0)
    assert out.shape == z0.shape, (
        f"SE3CFMComposite.forward must preserve (B, N, 3); got "
        f"{tuple(out.shape)} for input {tuple(z0.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 18: SE3CFMComposite.forward preserves dtype
# ---------------------------------------------------------------------------
def test_se3_cfm_composite_forward_dtype_preserved():
    """``SE3CFMComposite.forward`` preserves the input dtype."""
    composite = _make_composite(n_steps=10)
    z0 = torch.randn(2, 5, 3).double()
    out = composite.forward(z0, t=1.0)
    assert out.dtype == z0.dtype, (
        f"SE3CFMComposite.forward must preserve dtype; got "
        f"{out.dtype} for input {z0.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 19: SE3CFMComposite.forward produces finite output
# ---------------------------------------------------------------------------
def test_se3_cfm_composite_forward_finite():
    """``SE3CFMComposite.forward`` produces a finite trajectory."""
    composite = _make_composite(n_steps=15)
    torch.manual_seed(7)
    z0 = torch.randn(3, 6, 3)
    out = composite.forward(z0, t=1.0)
    assert torch.isfinite(out).all(), (
        f"SE3CFMComposite.forward output must be finite; got NaN/Inf in {out}"
    )


# ---------------------------------------------------------------------------
# Test 20: SE3CFMComposite is deterministic under same seed
# ---------------------------------------------------------------------------
def test_se3_cfm_composite_deterministic():
    """Same seed → same composite output (no random ops in the path)."""
    composite = _make_composite(n_steps=10)
    torch.manual_seed(13)
    z0_a = torch.randn(2, 5, 3)
    out_a = composite.forward(z0_a, t=1.0)
    torch.manual_seed(13)
    z0_b = torch.randn(2, 5, 3)
    out_b = composite.forward(z0_b, t=1.0)
    assert torch.equal(out_a, out_b), (
        f"SE3CFMComposite must be deterministic under same seed; got "
        f"max diff = {(out_a - out_b).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 21: SE3CFMComposite.__call__ matches forward
# ---------------------------------------------------------------------------
def test_se3_cfm_composite_callable():
    """``composite(z0, t) == composite.forward(z0, t)`` (callable alias)."""
    composite = _make_composite(n_steps=10)
    torch.manual_seed(17)
    z0 = torch.randn(2, 5, 3)
    out_forward = composite.forward(z0, t=0.7)
    out_call = composite(z0, t=0.7)
    assert torch.equal(out_forward, out_call), (
        f"SE3CFMComposite.__call__ must match .forward; got "
        f"max diff = {(out_forward - out_call).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 22: SE3CFMComposite rejects non-callable base_cfm
# ---------------------------------------------------------------------------
def test_se3_cfm_composite_rejects_non_callable_base_cfm():
    """Non-callable ``base_cfm`` raises :class:`TypeError` at construction."""
    from molmetal.adapters.se3_lipman_integration import (
        SE3CFMComposite,
        SE3LipmanSolver,
    )

    se3_solver = _make_solver(n_steps=10)
    with pytest.raises(TypeError):
        SE3CFMComposite(
            base_cfm="not_callable",  # type: ignore[arg-type]
            se3_solver=se3_solver,
            n_steps=10,
        )


# ---------------------------------------------------------------------------
# Test 23: SE3CFMComposite rejects wrong se3_solver type
# ---------------------------------------------------------------------------
def test_se3_cfm_composite_rejects_wrong_se3_solver_type():
    """Non-:class:`SE3LipmanSolver`` raises :class:`TypeError`."""
    from molmetal.adapters.se3_lipman_integration import SE3CFMComposite

    with pytest.raises(TypeError):
        SE3CFMComposite(
            base_cfm=_DummyLipmanEGNN(),
            se3_solver=torch.nn.Module(),  # wrong type
            n_steps=10,
        )


# ---------------------------------------------------------------------------
# Test 24: SE3CFMComposite rejects t <= 0
# ---------------------------------------------------------------------------
def test_se3_cfm_composite_rejects_t_le_zero():
    """``t <= 0`` raises :class:`ValueError` on forward."""
    composite = _make_composite(n_steps=5)
    z0 = torch.randn(2, 4, 3)
    with pytest.raises(ValueError):
        composite.forward(z0, t=0.0)
    with pytest.raises(ValueError):
        composite.forward(z0, t=-0.5)


# ---------------------------------------------------------------------------
# Test 25: Integration layer does NOT modify flow_matching_lipman
# ---------------------------------------------------------------------------
def test_integration_layer_does_not_modify_flow_matching_lipman():
    """Importing the integration layer must not mutate ``flow_matching_lipman``.

    The R17 close-out invariant is that the locked
    :mod:`molmetal.adapters.flow_matching_lipman` package is not
    modified by downstream integrations.  We verify this by
    snapshotting the public attribute names of the
    ``flow_matching_lipman`` module before and after the import.
    """
    from molmetal.adapters import flow_matching_lipman

    pre = set(dir(flow_matching_lipman))
    # Force re-import to exercise the import path.
    import importlib
    from molmetal.adapters import se3_lipman_integration
    importlib.reload(se3_lipman_integration)
    post = set(dir(flow_matching_lipman))
    removed = pre - post
    added = post - pre
    assert removed == set(), (
        f"Importing se3_lipman_integration must not remove any "
        f"flow_matching_lipman attributes; removed={removed}"
    )
    assert added == set(), (
        f"Importing se3_lipman_integration must not add any "
        f"flow_matching_lipman attributes; added={added}"
    )
    # The integration layer must expose the three required symbols.
    for name in ("SE3LipmanSolver", "SE3CFMComposite", "compare_se3_lipman_vs_base"):
        assert hasattr(se3_lipman_integration, name), (
            f"se3_lipman_integration must expose {name!r}"
        )
