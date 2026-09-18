"""Batched graph encoding/velocity must equal independent graph evaluation."""
import copy

import pytest
import torch

from models.encoder import MolEncoder
from models.velocity_net import VelocityNet


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encoder_velocity_batch_values_and_gradients_equal_separate_graphs(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU is unavailable")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        torch.manual_seed(7)
        encoder = MolEncoder(hidden_dim=8, n_layers=2).to(device)
        velocity = VelocityNet(hidden_dim=8, n_layers=2).to(device)
        enc_reference, vel_reference = copy.deepcopy(encoder), copy.deepcopy(velocity)
        atoms = torch.tensor([[6, 7, 8, 1], [9, 6, 1, 0]], device=device)
        positions = torch.randn(2, 4, 3, device=device, requires_grad=True)
        independent_positions = positions.detach().clone().requires_grad_(True)
        # Different topologies, atom types, masks and coordinates expose
        # accidental flattening of the index-channel axis and cross-talk.
        edges = torch.tensor([[[0, 1, 0, 2, 3, 1], [1, 0, 2, 0, 1, 3]],
                              [[2, 0, 1, 1, 3, 2], [0, 2, 2, 0, 0, 1]]], device=device)
        edge_mask = torch.tensor([[True] * 6, [True, True, True, True, False, True]], device=device)
        node_mask = torch.tensor([[True] * 4, [True, True, True, False]], device=device)
        times = torch.tensor([.2, .8], device=device)

        def evaluate(enc, vel, xyz, index):
            node, edge = enc(atoms[index], xyz, edges[index], edge_mask=edge_mask[index], node_mask=node_mask[index])
            output = vel(node, xyz, times[index], edges[index], edge_mask=edge_mask[index], node_mask=node_mask[index])
            return node, edge, output

        actual = evaluate(encoder, velocity, positions, slice(None))
        separate = [evaluate(enc_reference, vel_reference, independent_positions[i:i+1], slice(i, i+1))
                    for i in range(2)]
        expected = tuple(torch.cat([rows[k] for rows in separate]) for k in range(3))
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, atol=3e-5, rtol=3e-5)
        weights = [torch.randn_like(value) for value in actual]
        sum((value * weight).sum() for value, weight in zip(actual, weights)).backward()
        sum((value * weight).sum() for value, weight in zip(expected, weights)).backward()
        torch.testing.assert_close(positions.grad, independent_positions.grad, atol=5e-5, rtol=5e-5)
        for model, reference in ((encoder, enc_reference), (velocity, vel_reference)):
            for param, ref_param in zip(model.parameters(), reference.parameters()):
                assert (param.grad is None) == (ref_param.grad is None)
                if param.grad is not None:
                    torch.testing.assert_close(param.grad, ref_param.grad, atol=5e-5, rtol=5e-5)

        # Changing only the other molecule must leave this molecule's output
        # unchanged, even though the shared network is evaluated jointly.
        changed = positions.detach().clone()
        changed[1] *= 9
        changed_output = evaluate(encoder, velocity, changed, slice(None))
        for value, changed_value in zip(actual, changed_output):
            torch.testing.assert_close(value[0], changed_value[0], atol=3e-5, rtol=3e-5)
