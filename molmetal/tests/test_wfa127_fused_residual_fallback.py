"""Wf-A127 (2026-10-31) — the fused-residual-add FALLBACKS computed the wrong function.

THE DEFECT
----------
``triton_kernels.fused_residual_add`` documents the contract
``out = alpha * x + beta * residual`` (module docstring line 4,
:func:`fused_residual_add` docstring, and the Triton kernel body
``out = alpha * x + beta * res``).  The Triton kernel was CORRECT; the
pure-PyTorch fallbacks were not.

PyTorch's ``torch.add(input, other, alpha=a)`` computes
``input + a * other`` — ``alpha`` scales the **SECOND** operand.  Every
fallback that used that form therefore computed ``x + alpha * residual``:

* the ``alpha`` on ``x`` was silently dropped, and
* ``beta`` was dropped entirely.

The CPU branch tried to patch the missing ``beta`` with a second
in-place ``add_(residual, alpha=beta - 1)``, which yields
``x + (alpha + beta - 1) * residual`` — still not the contract.

MEASURED (2026-10-31, CPU-only) before the fix:
``tests/test_new_kernels.py::test_fused_residual_add_parity[0.5-1.0]``
FAILED with ``max diff 1.6904566287994385``.  It passed on GPU because
the Triton kernel — the one path that was already right — is what runs
there.

WHY IT MATTERS
--------------
Every production call site of the two gated wrappers
(:func:`molmetal.adapters.egnn_rocm._maybe_fused_residual_add` at
lines 653 / 740, and :func:`models.velocity_net._maybe_fused_residual_add`
at lines 381 / 551) currently uses the default ``alpha == beta == 1.0``,
so the numbers coincide today.  But the dispatch gate
``triton_config.use_fused_residual_add(x.numel())`` is SIZE-dependent,
and the size-thresholded branches are exactly the ones that route
through the fallbacks — a latent wrong-function in a model's residual
path is the defect class this project is trying to eliminate.

WHAT THIS FILE COVERS
---------------------
1. The CPU path of :func:`triton_kernels.fused_residual_add` matches the
   contract for alpha/beta combinations that EXPOSE the old bug.
2. The pure-PyTorch fallback helper
   (:func:`triton_kernels.fused_residual_add.residual_add_fallback`)
   matches the contract for the geometries of the CUDA-only fallback
   branches (0-dim residual, general broadcast, other broadcast shapes).
3. A structural (AST) guard that every fallback ``return`` inside
   :func:`triton_kernels.fused_residual_add` delegates to that helper and
   that no ``torch.add(`` survives in the function — this extends the
   guarantee to the CUDA-only branches WITHOUT a GPU.
4. The two gated wrappers' gate-OFF paths match the contract.
5. The alpha == beta == 1.0 identity: the fix is BIT-IDENTICAL to the old
   behaviour for the values every production call site uses.
6. A GPU-REQUIRED test that exercises the real CUDA fallback branches —
   SKIPPED (never faked) on a CPU-only session.

Run with::

    CUDA_VISIBLE_DEVICES="" uv run pytest -q \
        molmetal/tests/test_wfa127_fused_residual_fallback.py --tb=short
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from triton_kernels import fused_residual_add as _fused_residual_add
from triton_kernels.config import triton_config
from triton_kernels.fused_residual_add import residual_add_fallback


# ---------------------------------------------------------------------------
# The contract — one place, so every assertion below compares against the
# same thing.
# ---------------------------------------------------------------------------
def _contract(x: torch.Tensor, res: torch.Tensor, alpha: float, beta: float):
    """``out = alpha * x + beta * residual`` — the documented contract."""
    return alpha * x + beta * res


#: The alpha/beta combinations that EXPOSE the old CPU-branch bug
#: (``x + (alpha + beta - 1) * residual``).
#:
#: Algebraically, ``old_cpu - contract = (1 - alpha) * (x - residual)``,
#: so the CPU branch was wrong exactly when ``alpha != 1`` — ``beta``
#: cancels out of the difference, which is why the pre-existing parity
#: test's ``(1.0, -0.3)`` case passed while ``(0.5, 1.0)`` failed.
#: Every entry is *verified* to expose it inside
#: :func:`test_cpu_branch_exposed_the_bug` rather than asserted from memory.
EXPOSING_COMBOS = [
    (0.5, 1.0),
    (0.5, 2.0),
    (2.0, 0.5),
    (-0.25, 0.75),
    (0.0, 1.0),
    (1.5, -0.5),
]

#: The combinations that EXPOSE the old CUDA-shape bug (``x + alpha * res``).
#: Algebraically ``old_cuda - contract = (1 - alpha) * x + (alpha - beta) * res``,
#: so that branch was wrong for every combination except ``alpha == beta == 1``.
CUDA_SHAPE_EXPOSING_COMBOS = EXPOSING_COMBOS + [
    (1.0, 0.0),
    (1.0, -0.3),
]

#: The combinations the pre-existing parity test already covers (one of
#: which — ``(1.0, -0.3)`` — coincidentally did NOT expose the CPU bug:
#: the old formula gives ``x - 0.3 * res``, which equals the contract).
PARITY_COMBOS = [(1.0, 1.0), (0.5, 1.0), (1.0, -0.3)]


def _old_cpu_formula(x, res, alpha, beta):
    """The pre-fix CPU fallback, verbatim — used only as the counter-example."""
    return torch.add(x, res, alpha=alpha).add_(res, alpha=(beta - 1.0))


def _old_cuda_shape_formula(x, res, alpha):
    """The pre-fix CUDA-shape fallback, verbatim — used only as the counter-example."""
    return torch.add(x, res, alpha=alpha)


# ---------------------------------------------------------------------------
# 1. The CPU path of the public wrapper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,beta", PARITY_COMBOS + EXPOSING_COMBOS)
def test_cpu_path_matches_contract(alpha, beta):
    """``fused_residual_add`` on CPU computes ``alpha * x + beta * residual``."""
    torch.manual_seed(0)
    x = torch.randn(8, 16, dtype=torch.float32)
    res = torch.randn(8, 16, dtype=torch.float32)

    out = _fused_residual_add(x, res, alpha=alpha, beta=beta)
    expected = _contract(x, res, alpha, beta)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6), (
        f"CPU fused_residual_add(alpha={alpha}, beta={beta}) must equal "
        f"alpha*x + beta*residual; max diff {(out - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("alpha,beta", EXPOSING_COMBOS)
def test_cpu_branch_exposed_the_bug(alpha, beta):
    """Guard that these combos really discriminate old-vs-new.

    If the OLD CPU formula were restored, this test's first assertion
    (old != contract) would still hold, but
    :func:`test_cpu_path_matches_contract` would fail — so the pair is a
    true regression guard rather than a vacuous one.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 32, dtype=torch.float32)
    res = torch.randn(64, 32, dtype=torch.float32)

    old = _old_cpu_formula(x, res, alpha, beta)
    expected = _contract(x, res, alpha, beta)
    assert not torch.allclose(old, expected, atol=1e-6, rtol=1e-6), (
        f"alpha={alpha}, beta={beta} does NOT expose the old CPU bug — "
        f"the old formula already equals the contract here"
    )


# ---------------------------------------------------------------------------
# 2. The pure-PyTorch fallback helper — the arithmetic of the CUDA-only
#    fallback branches, tested WITHOUT a GPU.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,beta", PARITY_COMBOS + EXPOSING_COMBOS)
def test_fallback_helper_matches_contract_same_shape(alpha, beta):
    """The helper computes the contract for the same-shape geometry."""
    torch.manual_seed(1)
    x = torch.randn(6, 10, dtype=torch.float32)
    res = torch.randn(6, 10, dtype=torch.float32)

    out = residual_add_fallback(x, res, alpha=alpha, beta=beta)
    expected = _contract(x, res, alpha, beta)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6), (
        f"residual_add_fallback(alpha={alpha}, beta={beta}) mismatch; "
        f"max diff {(out - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("alpha,beta", PARITY_COMBOS + EXPOSING_COMBOS)
def test_fallback_helper_zero_dim_residual(alpha, beta):
    """0-dim residual (the ``residual.dim() == 0`` CUDA branch geometry).

    On CUDA this geometry is routed to a fallback branch; on CPU the
    same arithmetic is what the CPU path runs.  The helper is the shared
    implementation, so testing it here covers the branch's arithmetic
    without a GPU.
    """
    torch.manual_seed(2)
    x = torch.randn(4, 5, 6, dtype=torch.float32)
    res = torch.tensor(2.75, dtype=torch.float32)  # 0-dim

    out = residual_add_fallback(x, res, alpha=alpha, beta=beta)
    expected = _contract(x, res, alpha, beta)
    assert torch.allclose(
        out, expected, atol=1e-6, rtol=1e-6
    ), f"0-dim residual mismatch; max diff {(out - expected).abs().max().item()}"


@pytest.mark.parametrize("alpha,beta", PARITY_COMBOS + EXPOSING_COMBOS)
def test_fallback_helper_general_broadcast(alpha, beta):
    """General (non-trailing-axis) broadcast — the other CUDA branch geometry.

    ``x`` is ``(3, 4, 5)`` and ``residual`` is ``(1, 4, 5)``: different
    rank, so the kernel cannot express it and the wrapper falls back.
    """
    torch.manual_seed(3)
    x = torch.randn(3, 4, 5, dtype=torch.float32)
    res = torch.randn(1, 4, 5, dtype=torch.float32)

    out = residual_add_fallback(x, res, alpha=alpha, beta=beta)
    expected = _contract(x, res, alpha, beta)
    assert out.shape == x.shape
    assert torch.allclose(
        out, expected, atol=1e-6, rtol=1e-6
    ), f"general-broadcast mismatch; max diff {(out - expected).abs().max().item()}"


@pytest.mark.parametrize("alpha,beta", PARITY_COMBOS + EXPOSING_COMBOS)
def test_fallback_helper_other_broadcast_shape(alpha, beta):
    """A broadcast shape the kernel does not cover at all (``(3, 1, 1)``).

    Same rank as ``x`` but not trailing-axis — the wrapper's last
    fallback branch.
    """
    torch.manual_seed(4)
    x = torch.randn(3, 4, 5, dtype=torch.float32)
    res = torch.randn(3, 1, 1, dtype=torch.float32)

    out = residual_add_fallback(x, res, alpha=alpha, beta=beta)
    expected = _contract(x, res, alpha, beta)
    assert out.shape == x.shape
    assert torch.allclose(
        out, expected, atol=1e-6, rtol=1e-6
    ), f"other-broadcast mismatch; max diff {(out - expected).abs().max().item()}"


@pytest.mark.parametrize("alpha,beta", CUDA_SHAPE_EXPOSING_COMBOS)
def test_old_cuda_shape_formula_was_wrong(alpha, beta):
    """The pre-fix CUDA-shape fallback (``x + alpha * res``) was wrong too.

    Computed here so the claim is not asserted from memory: the old
    CUDA-shape formula drops the ``alpha`` on ``x`` and the whole
    ``beta``, so it differs from the contract for every combo except
    ``alpha == beta == 1.0``.
    """
    torch.manual_seed(5)
    x = torch.randn(32, 8, dtype=torch.float32)
    res = torch.randn(32, 8, dtype=torch.float32)

    old = _old_cuda_shape_formula(x, res, alpha)
    expected = _contract(x, res, alpha, beta)
    assert not torch.allclose(
        old, expected, atol=1e-6, rtol=1e-6
    ), f"alpha={alpha}, beta={beta} does NOT expose the old CUDA-shape bug"


# ---------------------------------------------------------------------------
# 3. Structural guard — extends the guarantee to the CUDA-only branches
#    WITHOUT a GPU.
# ---------------------------------------------------------------------------
def _return_expressions(func) -> list[str]:
    """The source text of every ``return`` expression inside ``func``."""
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    fn_node = tree.body[0]
    assert isinstance(
        fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)
    ), f"expected a function def, got {type(fn_node).__name__}"
    exprs = []
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Return) and node.value is not None:
            exprs.append(ast.unparse(node.value))
    return exprs


def test_all_fallback_returns_delegate_to_the_reference_helper():
    """Every fallback ``return`` in ``fused_residual_add`` calls the helper.

    This is what makes the CUDA-only branches (0-dim residual, general
    broadcast, other broadcast shapes) covered on a CPU-only machine:
    they all delegate to :func:`residual_add_fallback`, whose arithmetic
    is asserted above.
    """
    exprs = _return_expressions(_fused_residual_add)
    fallback_exprs = [e for e in exprs if "_FusedResidualAddFunction.apply" not in e]
    assert len(fallback_exprs) == 4, (
        f"expected 4 fallback returns in fused_residual_add, got "
        f"{len(fallback_exprs)}: {fallback_exprs}"
    )
    for expr in fallback_exprs:
        assert expr.startswith("residual_add_fallback("), (
            f"every fallback return must call residual_add_fallback(...); "
            f"got {expr!r}"
        )


def _torch_add_call_count(func) -> int:
    """Number of REAL ``torch.add(...)`` CALL nodes inside ``func``.

    AST-based (not a text grep) so that a docstring or a comment that
    *mentions* ``torch.add`` — e.g. the warning comment explaining why
    the call is wrong — is not counted.
    """
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "add"
                and isinstance(f.value, ast.Name)
                and f.value.id == "torch"
            ):
                count += 1
    return count


def test_no_torch_add_survives_in_the_wrapper():
    """No ``torch.add(...)`` CALL remains inside ``fused_residual_add``.

    ``torch.add(input, other, alpha=a)`` is the trap that carried the
    defect (``input + a * other``).  Its absence from the wrapper is
    the structural guarantee that the bug cannot come back.
    """
    assert _torch_add_call_count(_fused_residual_add) == 0, (
        "fused_residual_add must not call torch.add( — its `alpha` scales "
        "the second operand, which is the wrong function"
    )


def test_no_torch_add_survives_in_the_gated_wrappers():
    """Same structural guard for the two gated wrappers."""
    from molmetal.adapters import egnn_rocm
    from models import velocity_net

    for name, mod in (("egnn_rocm", egnn_rocm), ("velocity_net", velocity_net)):
        assert _torch_add_call_count(mod._maybe_fused_residual_add) == 0, (
            f"{name}._maybe_fused_residual_add must not call torch.add( — "
            f"its `alpha` scales the second operand, which is the wrong function"
        )


# ---------------------------------------------------------------------------
# 4. The gated wrappers' gate-OFF paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,beta", EXPOSING_COMBOS)
def test_egnn_rocm_gate_off_matches_contract(monkeypatch, alpha, beta):
    """The EGNN adapter's gate-OFF fallback computes the contract."""
    from molmetal.adapters import egnn_rocm

    monkeypatch.setattr(triton_config, "_enabled", False)
    torch.manual_seed(6)
    x = torch.randn(16, 32, dtype=torch.float32)
    res = torch.randn(16, 32, dtype=torch.float32)

    out = egnn_rocm._maybe_fused_residual_add(x, res, alpha=alpha, beta=beta)
    expected = _contract(x, res, alpha, beta)
    assert torch.allclose(
        out, expected, atol=1e-6, rtol=1e-6
    ), f"egnn_rocm gate-OFF mismatch; max diff {(out - expected).abs().max().item()}"


@pytest.mark.parametrize("alpha,beta", EXPOSING_COMBOS)
def test_velocity_net_gate_off_matches_contract(monkeypatch, alpha, beta):
    """The CFM velocity net's gate-OFF fallback computes the contract."""
    from models import velocity_net

    monkeypatch.setattr(triton_config, "_enabled", False)
    torch.manual_seed(7)
    x = torch.randn(2, 8, 16, dtype=torch.float32)
    res = torch.randn(2, 8, 16, dtype=torch.float32)

    out = velocity_net._maybe_fused_residual_add(x, res, alpha=alpha, beta=beta)
    expected = _contract(x, res, alpha, beta)
    assert torch.allclose(
        out, expected, atol=1e-6, rtol=1e-6
    ), f"velocity_net gate-OFF mismatch; max diff {(out - expected).abs().max().item()}"


# ---------------------------------------------------------------------------
# 5. The alpha == beta == 1.0 identity — the fix changes NO published number
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_alpha_beta_one_is_bit_identical_to_the_old_behaviour(dtype):
    """At ``alpha == beta == 1.0`` the new formula is BIT-IDENTICAL to the old.

    Every production call site (egnn_rocm :653 / :740, velocity_net :381 /
    :551) uses the default ``alpha == beta == 1.0``, so this is the
    statement that the fix does not move any published metric.

    Argument: IEEE-754 multiplication by ``1.0`` is exact, so the old
    ``x + 1.0 * residual`` equals ``x + residual``; and the old extra
    ``+ 0.0 * residual`` term is an exact identity for finite inputs.
    MEASURED below rather than assumed.
    """
    torch.manual_seed(8)
    x = (torch.randn(37, 64) * 100).to(dtype)
    res = (torch.randn(37, 64) * 100).to(dtype)

    old = _old_cpu_formula(x, res, 1.0, 1.0)
    new = residual_add_fallback(x, res, 1.0, 1.0)
    old_gate = _old_cuda_shape_formula(x, res, 1.0)
    assert torch.equal(old, new), (
        f"alpha=beta=1: old CPU fallback and the new reference must be "
        f"bit-identical for {dtype}"
    )
    assert torch.equal(old_gate, new), (
        f"alpha=beta=1: old CUDA-shape fallback and the new reference must "
        f"be bit-identical for {dtype}"
    )


def test_alpha_beta_one_identity_against_the_plain_sum():
    """``alpha == beta == 1`` reduces to ``x + residual`` exactly."""
    torch.manual_seed(9)
    x = torch.randn(128, 16, dtype=torch.float32)
    res = torch.randn(128, 16, dtype=torch.float32)
    assert torch.equal(residual_add_fallback(x, res, 1.0, 1.0), x + res)


def test_gradients_flow_through_the_fallback():
    """The fallback is differentiable and its gradients match the contract."""
    torch.manual_seed(10)
    x = torch.randn(8, 8, dtype=torch.float32, requires_grad=True)
    res = torch.randn(8, 8, dtype=torch.float32, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    res2 = res.detach().clone().requires_grad_(True)

    residual_add_fallback(x, res, 0.5, 2.0).sum().backward()
    _contract(x2, res2, 0.5, 2.0).sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(res.grad, res2.grad, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# 6. GPU-REQUIRED — the REAL CUDA fallback branches.
#    SKIPPED on a CPU-only session; never faked.
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU-REQUIRED: the CUDA fallback branches need a CUDA tensor; "
    "this session is CPU-only (CUDA_VISIBLE_DEVICES='') so the branches "
    "cannot be reached without faking CUDA, which would be a false green",
)
@pytest.mark.parametrize("alpha,beta", EXPOSING_COMBOS)
def test_cuda_fallback_branches_match_contract(alpha, beta):
    """On CUDA the broadcast-shape fallbacks also compute the contract.

    Covers the three CUDA-only branches: 0-dim residual, general
    (non-trailing-axis) broadcast, and "any other broadcast shape".
    """
    dev = torch.device("cuda")
    torch.manual_seed(11)

    cases = [
        (torch.randn(4, 5, 6, device=dev), torch.tensor(2.75, device=dev)),
        (torch.randn(3, 4, 5, device=dev), torch.randn(1, 4, 5, device=dev)),
        (torch.randn(3, 4, 5, device=dev), torch.randn(3, 1, 1, device=dev)),
    ]
    for x, res in cases:
        out = _fused_residual_add(x, res, alpha=alpha, beta=beta)
        expected = _contract(x, res, alpha, beta)
        assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), (
            f"CUDA fallback mismatch (x={tuple(x.shape)}, "
            f"res={tuple(res.shape)}); max diff "
            f"{(out - expected).abs().max().item()}"
        )
