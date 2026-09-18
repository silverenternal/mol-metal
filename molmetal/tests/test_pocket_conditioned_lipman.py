"""Pytest: T5 pocket-conditioned LipmanFlowMatchingAdapter.

Two tests (T5 spec):
1. ``test_pocket_conditioning_round_trip`` — calling ``train_step`` and
   ``generate`` with a pocket must NOT break the existing API
   contract: ``train_step`` returns a finite float, ``generate`` returns
   a list of ``Molecule`` with the requested ``n_samples`` length and
   ``(n_atoms, 3)`` coords; ``pocket=None`` and ``pocket=Pocket(...)``
   produce comparable loss magnitudes.
2. ``test_pocket_conditioning_loss_decreases`` — over 30 training
   steps on synthetic data the loss strictly decreases (later mean <
     earlier mean).

Both tests use a tiny model (hidden_dim=32, n_layers=2) so they run in
seconds on CPU or ROCm.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman import (
    LipmanFlowMatchingAdapter,
    PocketEncoder,
)
from molmetal.domain import Molecule, Pocket
from molmetal.ports import GenerationConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_pocket():
    """A deterministic toy pocket — 12 atoms around origin."""
    g = torch.Generator().manual_seed(0)
    coords = torch.randn(12, 3, generator=g) * 2.0
    atom_types = torch.randint(1, 18, (12,), generator=g)
    return Pocket(
        pdb_id="toy_pocket",
        coords=coords,
        atom_types=atom_types,
        residue_ids=torch.zeros(12, dtype=torch.long),
        chain_ids=torch.zeros(12, dtype=torch.long),
        mask=torch.ones(12, dtype=torch.bool),
        center=coords.mean(0),
        radius=6.0,
    )


@pytest.fixture
def toy_mols():
    g = torch.Generator().manual_seed(1)
    out = []
    for i in range(3):
        g_i = torch.Generator().manual_seed(100 + i)
        coords = torch.randn(6, 3, generator=g_i)
        atom_types = torch.randint(1, 10, (6,), generator=g_i)
        out.append(Molecule(
            coords=coords, atom_types=atom_types,
            bonds=torch.zeros(2, 0, dtype=torch.long),
            bond_types=torch.zeros(0, dtype=torch.long),
            formal_charges=torch.zeros(6, dtype=torch.long),
        ))
    return out


@pytest.fixture
def small_adapter():
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        adapter = LipmanFlowMatchingAdapter(hidden_dim=32, n_layers=2, lr=5e-3)
        adapter.setup()
        yield adapter


# ---------------------------------------------------------------------------
# Test 1: round-trip doesn't break with pocket conditioning
# ---------------------------------------------------------------------------
def test_pocket_conditioning_round_trip(small_adapter, toy_pocket, toy_mols):
    """``train_step`` and ``generate`` work end-to-end with pocket=None
    AND pocket=Pocket(...).  Both code paths produce finite outputs and
    comparable loss magnitudes."""
    # Path 1: pocket=None (legacy / unconditioned).
    loss_none = small_adapter.train_step(pocket=None, mols=toy_mols)
    assert isinstance(loss_none, float)
    assert math.isfinite(loss_none)
    assert loss_none > 0.0  # CFM + atom-CE are positive
    assert "cfm" in small_adapter.last_losses
    assert "atom" in small_adapter.last_losses

    # Path 2: pocket=Pocket(...) (T5 conditioning).
    loss_pocket = small_adapter.train_step(pocket=toy_pocket, mols=toy_mols)
    assert isinstance(loss_pocket, float)
    assert math.isfinite(loss_pocket)
    assert loss_pocket > 0.0

    # Loss magnitudes should be in the same order of magnitude — pocket
    # conditioning is additive-bias only, it should not blow up the
    # training signal.  Allow 50x slack for the first step (vel head is
    # zero-init but the pocket encoder adds a non-trivial initial bias).
    assert 0.02 * loss_none < loss_pocket < 50 * loss_none, (
        f"pocket conditioning broke loss magnitude: "
        f"loss_none={loss_none:.3f}, loss_pocket={loss_pocket:.3f}"
    )

    # PocketEncoder shape contract.
    enc = small_adapter.pocket_encoder
    assert isinstance(enc, PocketEncoder)
    assert enc.hidden_dim == 32
    b, p = 2, 5
    device = small_adapter.device
    embed = enc(
        torch.randn(b, p, 3, device=device),
        torch.randint(1, 10, (b, p), device=device),
        torch.ones(b, p, dtype=torch.bool, device=device),
    )
    assert embed.shape == (b, enc.hidden_dim)

    # Generate with pocket conditioning must return n_samples Molecules
    # with correct (n_atoms, 3) coords.  n_atoms=8 is the adapter default
    # when GenerationConfig has no ``n_atoms`` field.
    cfg = GenerationConfig(n_samples=2, n_steps=4)
    mols_out = small_adapter.generate(pocket=toy_pocket, config=cfg)
    assert len(mols_out) == 2
    for m in mols_out:
        assert isinstance(m, Molecule)
        assert m.coords.shape == (8, 3)
        assert m.atom_types.shape == (8,)
        assert torch.isfinite(m.coords).all()
    # Generate with pocket=None must also work.
    mols_none = small_adapter.generate(pocket=None, config=cfg)
    assert len(mols_none) == 2


# ---------------------------------------------------------------------------
# Test 2: loss decreases over training (pocket conditioning is learnable)
# ---------------------------------------------------------------------------
def test_pocket_conditioning_loss_decreases(small_adapter, toy_pocket, toy_mols):
    """Keep the 5% decrease gate on a common data/noise evaluation bank.

    Comparing first/last five *different* noisy batches could fail even when
    optimization helped. This is a fixed tiny-data optimization smoke; it
    makes no held-out molecular generalization claim. Training noise changes
    at every update; only evaluation uses identical targets/noise/times.
    """
    adapter = small_adapter
    device = adapter.device
    x1 = torch.stack([m.coords for m in toy_mols]).to(device).repeat(16, 1, 1)
    atoms = torch.stack([m.atom_types for m in toy_mols]).to(device).repeat(16, 1)
    generator = torch.Generator(device=device).manual_seed(2026)
    x0 = torch.randn(x1.shape, device=device, generator=generator)
    t = torch.linspace(.05, .95, 16, device=device).repeat_interleave(len(toy_mols))
    edges = adapter._make_dummy_edge_index(len(x1), x1.shape[1], device)

    def common_loss():
        adapter.velocity_field.eval()
        adapter.pocket_encoder.eval()
        try:
            with torch.no_grad():
                embed = adapter._encode_pocket(toy_pocket, len(x1), x1.shape[1], device)
                sample = adapter.path.sample(x_0=x0, x_1=x1, t=t)
                output = adapter.velocity_field(sample.x_t, torch.zeros_like(atoms), edges, t, pocket_embed=embed)
                cfm = (output['vel'] - sample.dx_t).square().sum(-1).mean()
                atom = torch.nn.functional.cross_entropy(output['atom_logits'].flatten(0, 1), atoms.flatten())
                return float(cfm + adapter._atom_loss_weight * atom)
        finally:
            adapter.velocity_field.train()
            adapter.pocket_encoder.train()

    before = common_loss()
    pocket_before = [param.detach().clone() for param in adapter.pocket_encoder.parameters()]
    # Identity prediction now receives masked inputs instead of copying its
    # labels from the embedding; retain the 5% criterion with 100 updates.
    for step in range(100):
        torch.manual_seed(1000 + step)
        assert math.isfinite(adapter.train_step(pocket=toy_pocket, mols=toy_mols))
    after = common_loss()
    assert after < .95 * before, f"common evaluation loss: {before:.4f} -> {after:.4f}"
    assert any(not torch.equal(before_param, after_param)
               for before_param, after_param in zip(pocket_before, adapter.pocket_encoder.parameters()))
