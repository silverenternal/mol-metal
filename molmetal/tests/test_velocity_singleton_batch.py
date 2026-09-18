"""Single-graph flow matching must keep the model's explicit batch axis."""
import pytest
import torch

from models._scatter import scatter_sum
from models.velocity_net import VelocityNet


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_singleton_loss_forward_and_backward(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU is unavailable")
    from flow_matching.loss import ConditionalFlowMatchingLoss

    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        torch.manual_seed(12)
        model = VelocityNet(hidden_dim=8, n_layers=2).to(device)
        objective = ConditionalFlowMatchingLoss(model)
        x0 = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], device=device)
        x1 = x0 + torch.tensor([[[.1, .2, .3], [.3, -.2, .1], [-.1, .2, -.3]]], device=device)
        edges = torch.tensor([[[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]], device=device)
        result = objective(x0, x1, h_node=torch.randn(1, 3, 8, device=device), edge_index=edges)
        assert result.pred_velocity.shape == x1.shape
        assert torch.isfinite(result.loss)
        result.loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g).item() for g in gradients)


@pytest.mark.parametrize("batch_size", [None, 1])
def test_legacy_single_graph_scatter_shape_values_and_gradients(batch_size):
    features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
    dst = torch.tensor([0, 2, 2])
    result = scatter_sum(torch.tensor([1, 0, 1]), dst, features, 3,
                         mask=torch.tensor([True, False, True]), batch_size=batch_size)
    torch.testing.assert_close(result, torch.tensor([[1., 2.], [0., 0.], [5., 6.]]))
    weights = torch.arange(6.).reshape(3, 2)
    (result * weights).sum().backward()
    torch.testing.assert_close(features.grad, torch.tensor([[0., 1.], [0., 0.], [4., 5.]]))
