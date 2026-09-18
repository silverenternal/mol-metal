"""Tests for the joint atom-type + coord FM ``generate()``.

These tests exercise the ``LipmanFlowMatchingAdapter`` end-to-end but stay
small enough to run as part of the standard test suite:

* ``test_no_z0_padding``          — generated mol has no Z=0 atoms (padding masked)
* ``test_atoms_in_realistic_range`` — all Z in [1, 18]
* ``test_atom_loss_decreases``     — 10 train_steps, atom_loss must drop > 0.5
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest
import torch

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Molecule


def _make_gen_config(n_samples: int = 16, n_steps: int = 4,
                     n_atoms: int = 8, seed: int = 0) -> SimpleNamespace:
    """Build a GenerationConfig-compatible namespace with ``n_atoms``.

    ``GenerationConfig`` is a frozen dataclass that does not include an
    ``n_atoms`` field — the adapter uses ``getattr(config, "n_atoms", 8)``
    so a SimpleNamespace with the same attributes satisfies the duck-typed
    contract without touching the port schema.
    """
    return SimpleNamespace(
        n_samples=n_samples,
        n_steps=n_steps,
        temperature=1.0,
        seed=seed,
        conditioning={},
        n_atoms=n_atoms,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_random_molecule(n_atoms: int = 8, seed: int = 0,
                          z_max: int = 10) -> Molecule:
    """Build a small random Molecule on CPU; the adapter moves tensors.

    ``z_max`` defaults to 10 because the velocity field's ``atom_head``
    outputs ``max_atomic_number=100`` logits but the test only needs a
    handful of realistic elements to exercise the masked-CF path.
    """
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(n_atoms, 3, generator=g, dtype=torch.float32)
    atom_types = torch.randint(1, z_max, (n_atoms,), generator=g, dtype=torch.long)
    return Molecule(
        coords=coords,
        atom_types=atom_types,
        bonds=torch.zeros(2, 0, dtype=torch.long),
        bond_types=torch.zeros(0, dtype=torch.long),
        formal_charges=torch.zeros(n_atoms, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
class TestGenerateAtomTypes:
    """End-to-end smoke tests for the new joint ``generate()``."""

    def test_no_z0_padding(self):
        """The atom_logits[..., 0] = -inf mask must be honoured at sample time."""
        torch.manual_seed(0)
        adapter = LipmanFlowMatchingAdapter(hidden_dim=32, n_layers=2, lr=5e-3)
        adapter.setup()

        # Minimal training: 3 steps so the atom_head leaves its zero init
        # and produces non-uniform logits.  Without this every atom would
        # get Z=0 purely by chance (all logits equal → argmax over zeros).
        mols = [_make_random_molecule(n_atoms=8, seed=i) for i in range(4)]
        for s in range(3):
            torch.manual_seed(100 + s)
            adapter.train_step(pocket=None, mols=mols)

        # Now generate a batch and assert no padding leakage.
        torch.manual_seed(200)
        cfg = _make_gen_config(n_samples=16, n_steps=4, n_atoms=8, seed=0)
        out: List[Molecule] = adapter.generate(pocket=None, config=cfg)
        assert len(out) == 16
        for i, m in enumerate(out):
            assert m.n_atoms == 8, f"mol {i} has {m.n_atoms} atoms, expected 8"
            assert (m.atom_types != 0).all(), (
                f"mol {i} has Z=0 padding atom — the -inf mask in generate() "
                f"is not being applied correctly.  atom_types={m.atom_types.tolist()}"
            )

    def test_atoms_in_realistic_range(self):
        """All generated atoms must lie in [1, 18] — no noble gases or heavy metals."""
        torch.manual_seed(1)
        adapter = LipmanFlowMatchingAdapter(hidden_dim=32, n_layers=2, lr=5e-3)
        adapter.setup()

        # Train with molecules whose atom_types are restricted to Z ∈ [1, 18].
        # This is the realistic regime: the atom_head must learn a sharp
        # distribution over organic elements.  Without this constraint the
        # zero-init atom_head samples uniformly over all 100 elements
        # (including Z > 18) — that's correct behaviour, but not what we
        # want to assert here.  We use a heavily repeated element pattern
        # (mostly C / N / O) so that even 30 train steps collapse the
        # softmax onto the support.
        g = torch.Generator().manual_seed(7)
        mols = []
        # Organic-element priors for small drug-like fragments
        priors = [6, 6, 6, 6, 7, 7, 8, 8, 16, 1]
        for i in range(8):
            coords = torch.randn(8, 3, generator=g, dtype=torch.float32) * 0.1
            atom_types = torch.tensor(
                [priors[int(idx)] for idx in torch.randint(0, len(priors), (8,), generator=g)],
                dtype=torch.long,
            )
            mols.append(Molecule(
                coords=coords,
                atom_types=atom_types,
                bonds=torch.zeros(2, 0, dtype=torch.long),
                bond_types=torch.zeros(0, dtype=torch.long),
                formal_charges=torch.zeros(8, dtype=torch.long),
            ))

        # Train longer (50 steps) on a fixed batch so the atom_head really
        # concentrates mass on the small subset of classes seen above.
        for s in range(50):
            torch.manual_seed(300 + s)
            adapter.train_step(pocket=None, mols=mols)

        torch.manual_seed(400)
        cfg = _make_gen_config(n_samples=16, n_steps=4, n_atoms=8, seed=0)
        out = adapter.generate(pocket=None, config=cfg)
        assert len(out) == 16
        # concat all atom_types for a single check
        all_z = torch.cat([m.atom_types for m in out])
        assert int(all_z.min()) >= 1, (
            f"min Z = {int(all_z.min())} — should be ≥ 1 (padding is masked)")
        assert int(all_z.max()) <= 18, (
            f"max Z = {int(all_z.max())} — should be ≤ 18 (no noble gases / "
            f"heavy metals in realistic range)"
        )
        # Sanity: Z must be a positive integer
        assert all_z.dtype == torch.long
        assert (all_z > 0).all()

    def test_atom_loss_decreases(self):
        """10 train_steps must drop atom_loss by > 0.5 (i.e. ~half of its initial value).

        Atom-loss is bounded above by ``ln(max_atomic_number) ≈ 4.6``; a
        uniform distribution over 100 classes gives ``atom_loss ≈ 4.6``,
        and a perfect prediction gives 0.  The zero-initialised atom_head
        starts with all logits = 0 → softmax = 1/100 → ``atom_loss = ln(100) ≈ 4.605``.
        A drop of 0.5 (≈ 10% of init) is the bar set by the task spec.
        """
        torch.manual_seed(42)
        adapter = LipmanFlowMatchingAdapter(hidden_dim=32, n_layers=2, lr=5e-3)
        adapter.setup()

        # Tiny fixed batch — deterministic across the 10-step run.
        mols = [_make_random_molecule(n_atoms=8, seed=i) for i in range(4)]

        torch.manual_seed(0)
        first_total = adapter.train_step(pocket=None, mols=mols)
        first_atom = adapter.last_losses["atom"]

        atom_losses = [first_atom]
        for s in range(9):
            torch.manual_seed(1 + s)
            adapter.train_step(pocket=None, mols=mols)
            atom_losses.append(adapter.last_losses["atom"])

        last_atom = atom_losses[-1]
        drop = first_atom - last_atom

        print(f"atom_loss[0]  = {first_atom:.4f}")
        print(f"atom_loss[-1] = {last_atom:.4f}")
        print(f"atom_loss drop = {drop:.4f} (must be > 0.5)")
        print(f"full series    = {[round(x, 3) for x in atom_losses]}")

        # Task spec: drop > 0.5 over 10 steps
        assert drop > 0.5, (
            f"atom_loss did not drop > 0.5 over 10 steps: "
            f"first={first_atom:.4f}, last={last_atom:.4f}, drop={drop:.4f}.  "
            f"Full series: {atom_losses}"
        )
        # Also sanity-check total loss is finite.
        assert torch.isfinite(torch.tensor(first_total)), "initial loss is NaN/Inf"
        assert torch.isfinite(torch.tensor(adapter.last_losses["total"])), \
            "final loss is NaN/Inf"
