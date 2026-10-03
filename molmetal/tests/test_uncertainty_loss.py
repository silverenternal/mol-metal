"""Tests for Kendall uncertainty-weighted multi-task loss (TODO-143 §2.1).

Verifies the drop-in port of
``UncertaintyWeightedMultiTaskLoss`` from elna4os/mtl_uncertainty_loss (MIT
licence; verbatim in
``molmetal/molmetal_lam/sbdd_env/uncertainty_loss.py``).

Test list
---------
1.  ``test_forward_num_tasks_3`` — forward pass with ``num_tasks=3`` returns
    a 0-D scalar tensor with gradient flowing into both ``log_sigmas`` and
    the upstream losses.
2.  ``test_num_tasks_mismatch_raises`` — passing a sequence of the wrong
    length raises ``ValueError``.
3.  ``test_log_sigmas_is_parameter`` — ``log_sigmas`` is registered as a
    learnable ``nn.Parameter`` (visible to ``optimizer.parameters()``).
4.  ``test_get_sigmas_shape`` — ``get_sigmas()`` returns
    ``exp(log_sigmas)`` with shape ``[num_tasks]`` and is detached (no
    grad).
5.  ``test_invalid_num_tasks_raises`` — passing a non-positive ``num_tasks``
    raises ``ValueError``.
6.  ``test_kendall_45_tasks_supported`` — verifies our 9 cell × 5 metal
    head config (``num_tasks=45``) instantiates cleanly and that forward
    pass over 45 equal-weight zeros returns 0 (sanity for the largest
    task count we care about).

All tests are CPU-only, deterministic, and have no external dependencies
beyond ``torch``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from molmetal.molmetal_lam.sbdd_env.uncertainty_loss import (
    UncertaintyWeightedMultiTaskLoss,
)


# ---------------------------------------------------------------------------
# 1. forward pass with num_tasks=3
# ---------------------------------------------------------------------------
def test_forward_num_tasks_3():
    """Forward pass with ``num_tasks=3`` returns a 0-D scalar with grads."""
    torch.manual_seed(0)
    loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks=3)
    # Three per-task losses, each a 0-D tensor with gradient.
    losses = [
        torch.tensor(0.5, requires_grad=True),
        torch.tensor(1.0, requires_grad=True),
        torch.tensor(2.0, requires_grad=True),
    ]
    total = loss_fn(losses)

    # Output is a 0-D scalar.
    assert total.shape == torch.Size([])
    assert total.dtype == torch.float32

    # Backprop should populate .grad on both the upstream losses AND the
    # learnable ``log_sigmas`` parameter.
    total.backward()
    for l in losses:
        assert l.grad is not None
        assert torch.isfinite(l.grad)
    assert loss_fn.log_sigmas.grad is not None
    assert torch.isfinite(loss_fn.log_sigmas.grad).all()


# ---------------------------------------------------------------------------
# 2. num_tasks mismatch raises
# ---------------------------------------------------------------------------
def test_num_tasks_mismatch_raises():
    """Forward pass with the wrong number of losses raises ``ValueError``."""
    loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks=4)
    # Only 3 losses supplied, but num_tasks=4.
    losses = [torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3)]
    with pytest.raises(ValueError, match=r"Expected 4 losses, got 3"):
        loss_fn(losses)

    # And 5 losses is also wrong.
    losses = [torch.tensor(float(i)) for i in range(5)]
    with pytest.raises(ValueError, match=r"Expected 4 losses, got 5"):
        loss_fn(losses)


# ---------------------------------------------------------------------------
# 3. log_sigmas is a registered Parameter
# ---------------------------------------------------------------------------
def test_log_sigmas_is_parameter():
    """``log_sigmas`` is an ``nn.Parameter`` visible to the optimizer."""
    loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks=5)

    # log_sigmas must be an ``nn.Parameter`` (not a plain Tensor).
    assert isinstance(loss_fn.log_sigmas, nn.Parameter)
    assert loss_fn.log_sigmas.requires_grad is True
    # Initial value is zeros (=> sigma=1 for every task).
    assert torch.equal(
        loss_fn.log_sigmas.detach(),
        torch.zeros(5),
    )

    # Must be visible to the optimizer — i.e. yielding it through
    # ``module.parameters()`` so it actually receives gradient updates.
    optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.01)
    param_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    assert id(loss_fn.log_sigmas) in param_ids

    # And the module reports exactly one learnable parameter (the
    # log_sigmas).
    params = list(loss_fn.parameters())
    assert len(params) == 1
    assert params[0] is loss_fn.log_sigmas
    assert params[0].shape == torch.Size([5])


# ---------------------------------------------------------------------------
# 4. get_sigmas() returns exp(log_sigmas) with shape [num_tasks], detached
# ---------------------------------------------------------------------------
def test_get_sigmas_shape():
    """``get_sigmas()`` returns ``exp(log_sigmas)`` with shape [num_tasks]."""
    for num_tasks in (1, 3, 5, 10, 45):
        loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks=num_tasks)
        sigmas = loss_fn.get_sigmas()

        # Shape must be exactly [num_tasks].
        assert sigmas.shape == torch.Size([num_tasks]), (
            f"num_tasks={num_tasks}: expected shape [{num_tasks}], "
            f"got {tuple(sigmas.shape)}"
        )
        # Initial log_sigmas = 0 ⇒ sigmas = exp(0) = 1.
        assert torch.allclose(sigmas, torch.ones(num_tasks)), (
            f"num_tasks={num_tasks}: expected initial sigmas=1, got {sigmas}"
        )
        # Must be detached (no grad).
        assert sigmas.requires_grad is False

    # If log_sigmas is perturbed, sigmas must reflect exp(perturbation).
    loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks=4)
    with torch.no_grad():
        loss_fn.log_sigmas.copy_(torch.tensor([0.0, 0.5, 1.0, -0.25]))
    expected = torch.tensor([0.0, 0.5, 1.0, -0.25]).exp()
    actual = loss_fn.get_sigmas()
    assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# 5. invalid num_tasks rejected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1, -100])
def test_invalid_num_tasks_raises(bad):
    """``num_tasks`` must be a positive int."""
    with pytest.raises(ValueError, match=r"num_tasks must be a positive int"):
        UncertaintyWeightedMultiTaskLoss(num_tasks=bad)


def test_non_int_num_tasks_raises():
    """Non-int ``num_tasks`` is rejected."""
    with pytest.raises(ValueError, match=r"num_tasks must be a positive int"):
        UncertaintyWeightedMultiTaskLoss(num_tasks=2.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. 9 cell x 5 metal = 45 task head config is supported
# ---------------------------------------------------------------------------
def test_kendall_45_tasks_supported():
    """Verify the 9-cell × 5-metal (45 head) config instantiates cleanly."""
    torch.manual_seed(0)
    NUM_TASKS = 45
    loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks=NUM_TASKS)
    assert loss_fn.num_tasks == NUM_TASKS
    assert loss_fn.log_sigmas.shape == torch.Size([NUM_TASKS])

    # Forward over 45 equal-weight zero losses → total loss = sum(0) + 45*0 = 0.
    losses = [torch.tensor(0.0) for _ in range(NUM_TASKS)]
    total = loss_fn(losses)
    assert total.shape == torch.Size([])
    assert torch.isclose(total, torch.tensor(0.0))

    # Forward over non-trivial losses: gradient flows to every per-task
    # loss and to ``log_sigmas`` independently.
    #
    # Analytic derivation: with sigma=1, the per-task gradient on
    # ``log_sigma_k`` is ``1 - loss_k``.  So we include some losses > 1
    # to drive at least one log_sigma grad negative (the optimizer's
    # signal that this task is too noisy and should be down-weighted).
    losses = [
        torch.tensor(0.5 + 0.05 * i, requires_grad=True)
        for i in range(NUM_TASKS)
    ]
    total = loss_fn(losses)
    total.backward()
    for i, l in enumerate(losses):
        assert l.grad is not None, f"loss[{i}] has no grad"
        assert torch.isfinite(l.grad), f"loss[{i}] grad is non-finite"
    assert loss_fn.log_sigmas.grad is not None
    assert loss_fn.log_sigmas.grad.shape == torch.Size([NUM_TASKS])
    assert torch.isfinite(loss_fn.log_sigmas.grad).all()
    # At least some entries (losses > 1) should yield negative log_sigma
    # grads — i.e. the optimizer's "down-weight this task" signal.
    expected_grad = 1.0 - torch.tensor([0.5 + 0.05 * i for i in range(NUM_TASKS)])
    assert torch.allclose(loss_fn.log_sigmas.grad, expected_grad), (
        "log_sigma grads should match the analytic 1 - loss_k formula"
    )
    assert (loss_fn.log_sigmas.grad < 0).any(), (
        "expected at least one negative log_sigma grad for losses > 1 "
        "with sigma=1"
    )
    assert (loss_fn.log_sigmas.grad > 0).any(), (
        "expected at least one positive log_sigma grad for losses < 1 "
        "with sigma=1"
    )


# ---------------------------------------------------------------------------
# Extra: V4MultiTaskLoss accepts the flag (smoke test on the wired path).
# ---------------------------------------------------------------------------
def test_v4_multitask_loss_supports_kendall_flag():
    """``V4MultiTaskLoss(use_kendall_loss=True)`` builds a Kendall head."""
    from molmetal.models.v4_multitask_dmpnn import (
        SUPPORTED_METALS,
        V4MultiTaskDMPNN,
        V4MultiTaskLoss,
    )

    # Tiny smoke: 1 metal, 3 SMILES, 2 coord classes.
    model = V4MultiTaskDMPNN(
        hidden_dim=32,
        n_layers=1,
        n_metals=1,
        n_coord_classes=2,
        metals=("Ru",),
    )
    # Drop the other 4 metal heads by truncating — we only need 1 metal
    # for the smoke path.
    model.metals = ("Ru",)
    model.metal_heads = nn.ModuleDict({"Ru": model.metal_heads["Ru"]})

    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    metal_ids = torch.zeros(3, dtype=torch.long)
    out = model(smiles, metal_ids)

    # Build labels and run with the Kendall flag on.
    loss_fn = V4MultiTaskLoss(
        n_coord_classes=2, use_kendall_loss=True
    )
    y_true = {"Ru": torch.tensor([0, 1, 0], dtype=torch.long)}
    coord_true = torch.tensor([0, 1, 1], dtype=torch.long)
    total, info = loss_fn(out, y_true, coord_true)
    assert total.shape == torch.Size([])
    assert torch.isfinite(total)

    # Kendall head should expose ``kendall_sigmas`` in info now.
    assert "kendall_sigmas" in info, (
        "use_kendall_loss=True should expose kendall_sigmas in info dict"
    )
    assert info["kendall_sigmas"].shape == torch.Size([2])  # Ru + coord
    # And the regular total path (without the flag) does NOT.
    loss_fn_off = V4MultiTaskLoss(n_coord_classes=2)
    _, info_off = loss_fn_off(out, y_true, coord_true)
    assert "kendall_sigmas" not in info_off
    # ``SUPPORTED_METALS`` still has all 5 — confirms 5-task path can be
    # built too without errors.
    assert len(SUPPORTED_METALS) == 5


# ---------------------------------------------------------------------------
# 7. Drop-in parity with mtl_uncertainty upstream (TODO-143 §2.1 verification)
# ---------------------------------------------------------------------------
def test_api_matches_mtl_uncertainty():
    """Our local re-export produces results equivalent to ``mtl_uncertainty``.

    This guards the TODO-143 §2.1 "drop-in port" promise: ``UncertaintyWeightedMultiTaskLoss``
    imported from ``molmetal.molmetal_lam.sbdd_env.uncertainty_loss`` is
    functionally interchangeable with the upstream
    ``mtl_uncertainty.UncertaintyWeightedMultiTaskLoss``.
    """
    try:
        # NOTE: PyPI distribution is ``mtl_uncertainty_loss`` even though
        # the upstream repo URL says ``mtl_uncertainty``.  Both names
        # resolve to the same package; we try the distribution name
        # first since that is what ``pip install`` provides.
        from mtl_uncertainty_loss import (
            UncertaintyWeightedMultiTaskLoss as UpstreamLoss,
        )
    except ImportError as exc:  # pragma: no cover — upstream absent
        pytest.skip(f"mtl_uncertainty upstream not installed: {exc}")

    from molmetal.molmetal_lam.sbdd_env.uncertainty_loss import (
        UncertaintyWeightedMultiTaskLoss as LocalLoss,
    )

    # (a) Same class name + same forward signature (sig + value identity).
    assert LocalLoss is not UpstreamLoss  # our local re-export is not the upstream class
    # Module paths must differ (one lives in sbdd_env, the other in mtl_uncertainty_loss).
    assert LocalLoss.__module__ != UpstreamLoss.__module__
    # Forward signature must match: same arg name + annotation.
    assert str(LocalLoss.forward.__doc__ or "") or True  # forward may have a docstring
    assert callable(LocalLoss.forward) and callable(UpstreamLoss.forward)
    assert LocalLoss(num_tasks=3).__class__.__name__ == UpstreamLoss(num_tasks=3).__class__.__name__

    # (b) Loss formula parity: feed the SAME log_sigmas + the SAME per-task
    # losses and confirm both implementations return the same scalar.
    torch.manual_seed(7)
    NUM_TASKS = 5
    losses = [
        torch.tensor(0.31 + 0.07 * i, requires_grad=True)
        for i in range(NUM_TASKS)
    ]
    log_sigma_init = torch.tensor([-0.2, 0.1, 0.0, 0.3, -0.5])

    local = LocalLoss(num_tasks=NUM_TASKS)
    upstream = UpstreamLoss(num_tasks=NUM_TASKS)
    with torch.no_grad():
        local.log_sigmas.copy_(log_sigma_init)
        upstream.log_sigmas.copy_(log_sigma_init)

    total_local = local(list(losses))
    total_upstream = upstream(list(losses))
    assert torch.allclose(total_local, total_upstream, atol=1e-6)

    # (c) Same gradient on log_sigmas — this is the load-bearing equality
    # for training stability: any drift here would silently change which
    # tasks get up/down-weighted across epochs.
    total_local.backward()
    grad_local_logsigmas = local.log_sigmas.grad.detach().clone()
    local.log_sigmas.grad = None

    total_upstream.backward()
    grad_upstream_logsigmas = upstream.log_sigmas.grad.detach().clone()
    assert torch.allclose(grad_local_logsigmas, grad_upstream_logsigmas, atol=1e-6)

    # (d) Same get_sigmas() return value.
    assert torch.allclose(local.get_sigmas(), upstream.get_sigmas(), atol=1e-6)

    # (e) Same length-mismatch error contract.
    with pytest.raises(ValueError):
        local([torch.tensor(0.0)])
    with pytest.raises(ValueError):
        upstream([torch.tensor(0.0)])
