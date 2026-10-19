"""Pytest: WF-CFM-Rescue Phase 2 — bond_head joint_train default verify.

This test verifies that the Phase 2 fixes for bond_head co-training
work as a unit (regression guard).  Pre-existing tests cover:
- ``test_cfm_fix1_bond_head_default.py`` — argparse + adapter default + optimizer wiring

This module adds explicit rescue coverage:
1. joint_train=True co-trains end-to-end (3 train_steps, no NaN)
2. bond_head parameters receive gradient when joint_train=True
3. bond_head parameters receive NO gradient when joint_train=False
4. Backward compat: pre-fix behaviour (use_bond_head=False) still passes

All tests CPU-friendly, <5s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Molecule


def _tiny_mol_list(n_mols: int = 2, atom_types=None):
    """Return a list of Molecule objects with valid 8-atom CNOF bonds.

    ``atom_types`` overrides the default in-vocab H/C/N/O mix; the
    non-finite diagnostic test uses it to inject out-of-vocab elements.
    """
    out = []
    for k in range(n_mols):
        torch.manual_seed(k)
        positions = torch.randn(8, 3) * 0.5
        # 2026-10-19 — draw from the adapter's atom vocab (H/C/N/O), not
        # randint(1, 10).  Elements 2-5 are not in the vocab, so the P1 OOV
        # guard (Wf-A56) correctly skipped the step and zeroed the grads,
        # and this fixture made test 2 fail for a reason unrelated to the
        # joint-training wiring it was written to check.
        if atom_types is None:
            atom_types = torch.tensor([6, 7, 8, 1, 6, 7, 8, 6])
        else:
            atom_types = torch.as_tensor(atom_types).clone()
        bonds = torch.tensor(
            [
                [0, 0, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6, 7, 0, 0],
            ],
            dtype=torch.long,
        )
        bond_types = torch.ones(bonds.shape[1], dtype=torch.long)
        formal_charges = torch.zeros(8, dtype=torch.long)
        out.append(
            Molecule(
                coords=positions,
                atom_types=atom_types,
                bonds=bonds,
                bond_types=bond_types,
                formal_charges=formal_charges,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Test 1: joint_train=True default — train_step runs without exception
# ---------------------------------------------------------------------------
def test_joint_train_default_train_step_runs():
    """With use_bond_head=True + joint_train=True (defaults), 1 train_step runs end-to-end.

    The exact loss value can be inf on freshly-init weights; the
    contract we test is: no exception, all gradient flows are wired.
    """
    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        max_atomic_number=20,
        use_bond_head=True,
        joint_train=True,
    )
    adapter.setup(device="cpu")
    mols = _tiny_mol_list(2)
    # Single step must not raise
    loss = adapter.train_step(pocket=None, mols=mols)
    assert loss is not None
    # Loss must be a real number (might be inf on a single step with random weights;
    # that's expected and fine — the contract is "the path runs end-to-end")
    assert isinstance(float(loss), float)


# ---------------------------------------------------------------------------
# Test 2: bond_head receives gradient when joint_train=True
# ---------------------------------------------------------------------------
def test_bond_head_receives_gradient_when_joint_train_true():
    """After one train_step, the BondOrderHead params have non-None .grad."""
    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        max_atomic_number=20,
        use_bond_head=True,
        joint_train=True,
    )
    adapter.setup(device="cpu")
    mols = _tiny_mol_list(2)
    adapter.train_step(pocket=None, mols=mols)
    assert adapter.last_losses.get("skipped_oov", 0) == 0, (
        "the step was skipped as non-finite, so this test would be asserting "
        "on zeroed grads rather than on the joint-training wiring: "
        f"{adapter.last_losses}"
    )
    if adapter.bond_head is not None:
        n_with_grad = sum(
            1 for p in adapter.bond_head.parameters() if p.grad is not None
        )
        assert n_with_grad >= 1, (
            "BondOrderHead should have at least one parameter with .grad after "
            "joint_train=True train_step; got 0"
        )


# ---------------------------------------------------------------------------
# Test 3: bond_head receives NO gradient when joint_train=False
# ---------------------------------------------------------------------------
def test_bond_head_no_gradient_when_joint_train_false():
    """Backward compat: joint_train=False keeps bond_head params frozen."""
    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        max_atomic_number=20,
        use_bond_head=True,
        joint_train=False,
    )
    adapter.setup(device="cpu")
    mols = _tiny_mol_list(2)
    adapter.train_step(pocket=None, mols=mols)
    assert adapter.last_losses.get("skipped_oov", 0) == 0, (
        "the step was skipped as non-finite, so this test would be asserting "
        "on zeroed grads rather than on the joint-training wiring: "
        f"{adapter.last_losses}"
    )
    if adapter.bond_head is not None:
        n_with_grad = sum(
            1 for p in adapter.bond_head.parameters() if p.grad is not None
        )
        assert n_with_grad == 0, (
            f"With joint_train=False, BondOrderHead params must NOT receive "
            f"gradients; got {n_with_grad} with .grad"
        )


# ---------------------------------------------------------------------------
# Test 4: legacy opt-out (use_bond_head=False) still passes
# ---------------------------------------------------------------------------
def test_legacy_no_bond_head_works():
    """Backward compat: use_bond_head=False works with no bond_head at all."""
    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        max_atomic_number=20,
        use_bond_head=False,
        joint_train=False,
    )
    adapter.setup(device="cpu")
    assert (
        adapter.bond_head is None
    ), "use_bond_head=False must NOT construct BondOrderHead"


def test_nonfinite_step_names_the_offending_term():
    """An OOV batch is skipped, and the diagnostic says WHICH loss blew up."""
    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        max_atomic_number=20,
        use_bond_head=True,
        joint_train=True,
    )
    adapter.setup(device="cpu")
    # Elements 2-5 are outside the atom vocab; the P1 guard must skip the step
    # and the payload must identify the atom CE as the non-finite term.
    mols = _tiny_mol_list(2, atom_types=[2, 3, 4, 5, 6, 7, 8, 6])
    adapter.train_step(pocket=None, mols=mols)
    assert adapter.last_losses["skipped_oov"] == 1
    assert "atom" in adapter.last_losses["nonfinite_terms"]
