"""Exercise real POT solvers so a silent Hungarian fallback cannot pass."""

import sys

import pytest
import torch

from flow_matching.optimal_transport import mini_batch_ot_coupling


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires ROCm/CUDA",
))]


@pytest.mark.parametrize("device", DEVICES)
def test_real_sinkhorn_uses_torch_backend_and_retains_soft_plan(monkeypatch, device):
    ot = pytest.importorskip("ot")
    real_sinkhorn = ot.sinkhorn
    observations = []

    def observe(a, b, cost, **kwargs):
        # This spy executes real POT; no synthetic plan is substituted.
        plan = real_sinkhorn(a, b, cost, **kwargs)
        observations.append((a, b, cost, plan))
        return plan

    monkeypatch.setattr(ot, "sinkhorn", observe)
    x0 = torch.tensor([[0.0], [1.0], [2.0]], device=device)
    x1 = x0.flip(0)
    records = []
    idx = mini_batch_ot_coupling(x1, x0, torch.zeros(3, device=device),
                                reg=1.0, max_iter=200, seed=42, diagnostics=records)
    assert len(observations) == len(records) == 1
    a, b, cost, plan = observations[0]
    assert all(value.device == torch.device(device) for value in (a, b, cost, plan, idx))
    assert records[0]["effective_backend"] == "pot_sinkhorn_log"
    assert records[0]["fallback_reason"] is None
    assert records[0]["rounding"] == "greedy_plan_permutation"
    assert records[0]["marginal_max_abs_error"] < 1e-5
    torch.testing.assert_close(plan.sum(0), b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(plan.sum(1), a, atol=1e-5, rtol=1e-5)
    # Positive off-diagonal transport verifies that the solver was entropic,
    # while the API still returns the discrete greedy projection.
    assert (plan > 1e-4).sum().item() > len(x0)
    assert idx.tolist() == [2, 1, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_log_domain_handles_large_cost_without_hungarian_fallback(device):
    pytest.importorskip("ot")
    x0 = torch.tensor([[0.0, 0.0], [100.0, 100.0]], device=device)
    x1 = torch.tensor([[90.0, 90.0], [10.0, 10.0]], device=device)
    records = []
    idx = mini_batch_ot_coupling(x1, x0, torch.zeros(2, device=device),
                                reg=0.05, diagnostics=records)
    assert idx.tolist() == [1, 0]
    assert records[0]["effective_backend"] == "pot_sinkhorn_log"
    assert records[0]["fallback_reason"] is None
    assert records[0]["marginal_max_abs_error"] < 1e-3


def test_missing_pot_fallback_is_reported_and_warns(monkeypatch):
    pytest.importorskip("scipy")
    monkeypatch.setitem(sys.modules, "ot", None)
    x0 = torch.tensor([[0.0], [1.0]])
    records = []
    with pytest.warns(RuntimeWarning, match="SciPy Hungarian on CPU"):
        idx = mini_batch_ot_coupling(x0.flip(0), x0, torch.zeros(2), diagnostics=records)
    assert idx.tolist() == [1, 0]
    assert records[0]["effective_backend"] == "scipy_hungarian"
    assert records[0]["fallback_reason"] == "POT unavailable"
    assert records[0]["plan_device"] == "cpu"


def test_requested_hungarian_is_distinct_from_sinkhorn():
    pytest.importorskip("scipy")
    x0 = torch.tensor([[0.0], [1.0]])
    records = []
    idx = mini_batch_ot_coupling(x0.flip(0), x0, torch.zeros(2),
                                method="hungarian", diagnostics=records)
    assert idx.tolist() == [1, 0]
    assert records[0]["effective_backend"] == "scipy_hungarian"
    assert records[0]["fallback_reason"] is None


def test_optional_loss_diagnostics_preserve_pairing_and_loss():
    pytest.importorskip("ot")
    from flow_matching.loss import ConditionalFlowMatchingLoss
    from models.velocity_net import VelocityNet

    torch.manual_seed(3)
    model = VelocityNet(hidden_dim=8, n_layers=1, cond_dim=0)
    objective = ConditionalFlowMatchingLoss(model, use_minibatch_ot=True)
    x0, x1 = torch.randn(2, 3, 3), torch.randn(2, 3, 3)
    inputs = dict(
        h_node=torch.randn(2, 3, 8), batch_idx=torch.zeros(2, dtype=torch.long),
        edge_index=torch.tensor([[[0, 1, 2], [1, 2, 0]]]).expand(2, -1, -1).contiguous(),
    )
    torch.manual_seed(7)
    baseline = objective(x0, x1, **inputs)
    diagnostics = []
    torch.manual_seed(7)
    observed = objective(x0, x1, ot_diagnostics=diagnostics, **inputs)
    torch.testing.assert_close(observed.loss, baseline.loss, atol=0, rtol=0)
    torch.testing.assert_close(observed.target_velocity_, baseline.target_velocity_, atol=0, rtol=0)
    assert len(diagnostics) == 1
    assert diagnostics[0]["effective_backend"] == "pot_sinkhorn_log"
