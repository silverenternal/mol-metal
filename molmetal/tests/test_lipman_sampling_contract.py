"""Sampling honors zero initialization, conditioning scale, seed and eval mode."""
import pytest
import torch

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Molecule, Pocket
from molmetal.ports import GenerationConfig


@pytest.fixture(autouse=True)
def rng_isolation():
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        yield


def fixed_inputs():
    g = torch.Generator().manual_seed(0)
    xyz = torch.randn(12, 3, generator=g) * 2
    pocket = Pocket('toy', xyz, torch.randint(1, 18, (12,), generator=g),
                    torch.zeros(12, dtype=torch.long), torch.zeros(12, dtype=torch.long),
                    torch.ones(12, dtype=torch.bool), xyz.mean(0), 6.)
    molecules = []
    for i in range(3):
        g = torch.Generator().manual_seed(100 + i)
        molecules.append(Molecule(coords=torch.randn(6, 3, generator=g),
                                  atom_types=torch.randint(1, 10, (6,), generator=g),
                                  bonds=torch.zeros(2, 0, dtype=torch.long), bond_types=torch.zeros(0, dtype=torch.long),
                                  formal_charges=torch.zeros(6, dtype=torch.long)))
    return pocket, molecules


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_two_update_pocket_sampling_is_finite_and_seeded(seed):
    torch.manual_seed(seed)
    adapter = LipmanFlowMatchingAdapter(hidden_dim=32, n_layers=2, lr=5e-3)
    adapter.setup()
    pocket, molecules = fixed_inputs()
    x = torch.stack([m.coords for m in molecules]).to(adapter.device)
    atoms = torch.stack([m.atom_types for m in molecules]).to(adapter.device)
    edges = adapter._make_dummy_edge_index(3, 6, adapter.device)
    embed = adapter._encode_pocket(pocket, 3, 6, adapter.device)
    initial = adapter.velocity_field(x, atoms, edges, torch.zeros(3, device=adapter.device), pocket_embed=embed)
    assert torch.count_nonzero(initial['vel']) == 0
    adapter.train_step(None, molecules)
    adapter.train_step(pocket, molecules)
    # Retain heterogeneous submodule mode, not just the root flag.
    adapter.velocity_field.layers[0].eval()
    states = [(m, m.training) for root in (adapter.velocity_field, adapter.pocket_encoder) for m in root.modules()]
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(adapter.device).clone() if adapter.device.type == 'cuda' else None
    config = GenerationConfig(n_samples=2, n_steps=4, seed=seed)
    first = adapter.generate(pocket, config)
    second = adapter.generate(pocket, config)
    assert all(m.training == state for m, state in states)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(adapter.device))
    for a, b in zip(first, second):
        assert torch.isfinite(a.coords).all()
        # ROCm atomic scatter accumulation can differ by a few float32 ULPs
        # despite identical random draws; this is not bitwise determinism.
        torch.testing.assert_close(a.coords, b.coords, rtol=1e-6, atol=1e-6)
        assert torch.equal(a.atom_types, b.atom_types)
        assert torch.all(a.atom_types > 0)
    different = adapter.generate(pocket, GenerationConfig(n_samples=2, n_steps=4, seed=seed + 1))
    assert any(not torch.equal(a.coords, b.coords) for a, b in zip(first, different))


def test_declared_pocket_scale_changes_encoding_linearly():
    torch.manual_seed(42)
    adapter = LipmanFlowMatchingAdapter(hidden_dim=8, n_layers=1, pocket_embed_scale=.25)
    adapter.setup()
    pocket, _ = fixed_inputs()
    scaled = adapter._encode_pocket(pocket, 1, 6, adapter.device)
    raw = adapter.pocket_encoder(pocket.coords.to(adapter.device).unsqueeze(0),
                                 pocket.atom_types.to(adapter.device).unsqueeze(0),
                                 pocket.mask.to(adapter.device).unsqueeze(0))
    torch.testing.assert_close(scaled, raw * .25)
    assert adapter.get_metadata()['pocket_embed_scale'] == .25


def test_sampling_restores_mode_on_failure(monkeypatch):
    adapter = LipmanFlowMatchingAdapter(hidden_dim=8, n_layers=1)
    adapter.setup()
    adapter.pocket_encoder.eval()
    def failed(*args):
        assert not adapter.velocity_field.training and not adapter.pocket_encoder.training
        raise RuntimeError('solver failed')
    monkeypatch.setattr(adapter, '_generate_impl', failed)
    with pytest.raises(RuntimeError, match='solver failed'):
        adapter.generate(None, GenerationConfig(n_samples=1))
    assert adapter.velocity_field.training
    assert not adapter.pocket_encoder.training


@pytest.mark.parametrize('scale', [-1., float('nan'), float('inf')])
def test_invalid_pocket_scale_is_rejected(scale):
    with pytest.raises(ValueError, match='pocket_embed_scale'):
        LipmanFlowMatchingAdapter(pocket_embed_scale=scale)
