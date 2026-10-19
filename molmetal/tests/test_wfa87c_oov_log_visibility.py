# SPDX-License-Identifier: Apache-2.0
"""Wf-A87c — the OOV counters must be visible in the training log.

WHY THIS EXISTS
---------------
Wf-A87/Wf-A87b made `train_step` exclude masked-target rows and count them
(`oov_target_atom_rows`, `oov_target_bond_rows`, `skipped_oov`).  A counter
that only lives in `adapter.last_losses` is invisible to anyone reading the
training log: a run whose atoms were silently dropped looks identical to a
clean one.  These tests pin the two properties that make it visible:

* the log suffix is EMPTY when nothing was excluded (normal runs stay
  byte-identical), and
* the counters ACCUMULATE across steps (the end-of-step merge must not reset
  them), so the printed value is a run total, not a per-step 1.
"""

from __future__ import annotations

import torch

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Molecule
from molmetal.scripts.train_pocket_flow import _oov_log_suffix


def test_suffix_is_empty_on_a_clean_step() -> None:
    assert _oov_log_suffix({}) == ""
    assert _oov_log_suffix({"cfm": 1.0, "atom": 2.0, "bond": 0.5}) == ""


def test_suffix_renders_every_counter() -> None:
    suffix = _oov_log_suffix(
        {"oov_target_atom_rows": 10, "oov_target_bond_rows": 2, "skipped_oov": 1}
    )
    assert "oov_atom_rows=10" in suffix
    assert "oov_bond_rows=2" in suffix
    assert "skipped_oov=1" in suffix


def _adapter() -> LipmanFlowMatchingAdapter:
    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        max_atomic_number=20,
        use_bond_head=True,
        joint_train=True,
    )
    adapter.setup(device="cpu")
    return adapter


def _oov_mols(n_mols: int = 2):
    """8-atom molecules whose first 4 atoms are outside the atom vocab."""
    out = []
    for k in range(n_mols):
        torch.manual_seed(k)
        bonds = torch.tensor(
            [[0, 0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0, 0]],
            dtype=torch.long,
        )
        out.append(
            Molecule(
                coords=torch.randn(8, 3) * 0.5,
                atom_types=torch.tensor([2, 3, 4, 5, 6, 7, 8, 6]),
                bonds=bonds,
                bond_types=torch.ones(bonds.shape[1], dtype=torch.long),
                formal_charges=torch.zeros(8, dtype=torch.long),
            )
        )
    return out


def test_counters_accumulate_across_steps() -> None:
    adapter = _adapter()
    mols = _oov_mols(2)
    adapter.train_step(pocket=None, mols=mols)
    first = int(adapter.last_losses["oov_target_atom_rows"])
    adapter.train_step(pocket=None, mols=mols)
    second = int(adapter.last_losses["oov_target_atom_rows"])
    assert first == 8, f"4 OOV atoms x 2 molecules expected, got {first}"
    assert second == 2 * first, (
        "counters must be cumulative; a reset to 1 means the end-of-step "
        f"assignment dropped them (first={first}, second={second})"
    )


def test_clean_batch_keeps_the_log_suffix_empty() -> None:
    adapter = _adapter()
    mols = _oov_mols(2)
    for m in mols:
        object.__setattr__(m, "atom_types", torch.tensor([6, 7, 8, 1, 6, 7, 8, 6]))
    adapter.train_step(pocket=None, mols=mols)
    assert _oov_log_suffix(adapter.last_losses) == ""
