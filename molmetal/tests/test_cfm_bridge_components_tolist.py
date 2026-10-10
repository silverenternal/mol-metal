# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The bridging loop must read its index rows once, not once per candidate.

WHAT CHANGED (2026-10-10)
-------------------------
`YuelBondBondDecoder._bridge_components` walked the head's candidate pairs with

    for k in range(edge_index.shape[1]):
        i = int(edge_index[0][k])
        j = int(edge_index[1][k])

Indexing a CUDA tensor with a Python ``int`` is a device-to-host transfer, so
that was one sync PER CANDIDATE PAIR — hundreds per generated molecule, times
every molecule of the per-epoch gen eval. ``_select_bonds``, in the same file,
already read the same two rows with a single ``.tolist()`` each.

The fix hoists those two reads out of the loop. It returns the same ints, so
the candidate list — and therefore which edges get bridged — is unchanged.

WHY THE TEST IS DEVICE-PARITY, NOT A GOLDEN VALUE
------------------------------------------------
The hoisted read and the per-pair read are the same numbers by construction,
so a golden-value test would only pin the decoder's chemistry, which
``test_cfm_d2_connectivity_prior.py`` already owns. What the change could
plausibly have broken is device-dependence: ``.tolist()`` moves the tensor to
host, so anything that ended up depending on where the call ran would show up
as CPU and CUDA disagreeing. Both are run and required to match.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.models.bond_head import BOND_SINGLE, AtomCloud
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import YuelBondBondDecoder

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="device-parity check needs CUDA"
)


class _FakeResult:
    """`_bridge_components` reads only `.logits` and `.edge_index`."""

    def __init__(self, logits: torch.Tensor, edge_index: torch.Tensor) -> None:
        self.logits = logits
        self.edge_index = edge_index


def _two_fragment_case(device: torch.device, seed: int = 0):
    """Two carbon fragments 2.9 A apart, with the head calling the
    cross-fragment pair a confident SINGLE bond.

    The bridging prior returns early when the graph is already one component,
    so a connected fixture would skip the loop entirely. 2.9 A is inside the
    head's 3.5 A cutoff, which is why the head emits the pair at all.
    """
    torch.manual_seed(seed)
    positions = torch.tensor(
        [[0.00, 0.0, 0.0], [1.50, 0.0, 0.0], [4.40, 0.0, 0.0], [5.90, 0.0, 0.0]],
        dtype=torch.float32,
    )
    cloud = AtomCloud(
        positions=positions.to(device),
        atomic_numbers=torch.tensor([6, 6, 6, 6], dtype=torch.long).to(device),
    )
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long).to(device)
    logits = torch.full((edge_index.shape[1], 5), -4.0).to(device)
    logits[:, BOND_SINGLE] = 2.0
    logits[1, BOND_SINGLE] = 3.0  # the (1, 2) cross-fragment pair, strongest

    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    selected = [(0, 1, BOND_SINGLE), (2, 3, BOND_SINGLE)]
    return decoder, cloud, _FakeResult(logits, edge_index), selected, []


@requires_cuda
def test_cpu_and_cuda_bridge_the_same_edges():
    for seed in (0, 1, 2):
        results = []
        for dev in (torch.device("cpu"), torch.device("cuda:0")):
            decoder, cloud, yres, selected, dative_in = _two_fragment_case(dev, seed)
            bond_orders, dative, n_added = decoder._bridge_components(
                cloud, yres, selected, dative_in
            )
            results.append((
                [tuple(int(v) for v in b) for b in bond_orders],
                [tuple(int(v) for v in b) for b in dative],
                int(n_added),
            ))
        assert results[0] == results[1], (
            f"seed {seed}: the hoisted .tolist() read made the result "
            f"device-dependent — CPU {results[0]} vs CUDA {results[1]}"
        )


@requires_cuda
def test_the_prior_actually_bridges_on_this_fixture():
    """Guards the test above from passing vacuously: if the fixture produced
    one component the prior returns early and no read happens at all."""
    device = torch.device("cuda:0")
    decoder, cloud, yres, selected, dative_in = _two_fragment_case(device, 0)
    # `_bridge_components` appends to the list it is handed and returns that
    # same object, so `selected` has to be copied before the call or the
    # difference below is empty by aliasing rather than by chemistry.
    before = list(selected)
    bond_orders, _dative, n_added = decoder._bridge_components(
        cloud, yres, selected, dative_in
    )
    assert n_added == 1, f"expected one bridging edge, added {n_added}"
    bridged = {tuple(b[:2]) for b in bond_orders} - {tuple(b[:2]) for b in before}
    assert bridged == {(1, 2)}, (
        f"expected the (1,2) cross-fragment candidate to be bridged, got {bridged}"
    )


@requires_cuda
def test_an_empty_candidate_list_is_still_a_no_op():
    """The hoisted reads must not turn the zero-edge early return into an
    attempt to index an empty row."""
    device = torch.device("cuda:0")
    decoder = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    cloud = AtomCloud(
        positions=torch.zeros((3, 3)).to(device),
        atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.long).to(device),
    )
    empty = torch.zeros((2, 0), dtype=torch.long, device=device)
    logits = torch.zeros((0, 5), device=device)
    orders, dative, n_added = decoder._bridge_components(
        cloud, _FakeResult(logits, empty), [], []
    )
    assert (orders, dative, n_added) == ([], [], 0)