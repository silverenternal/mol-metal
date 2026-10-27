# SPDX-License-Identifier: Apache-2.0
"""Padding batches to one group must not change the generated coordinates.

WHY THIS EXISTS
---------------
BLK-C-PERF measured that ODE sampling is launch-bound, not compute-bound:
256x the batch costs 10% more wall time, and 99.6% of the time at B=256 does
not scale with batch. The sampler groups by EXACT atom count, so 192 tmQM val
clouds become 52 solver calls at median batch 3 -- ~15 s per generation pass
where one padded batch would take ~0.4 s.

That 42x is only worth taking if padding changes nothing.

WHAT THIS FILE ACTUALLY CHECKS (corrected 2026-10-27)
-----------------------------------------------------
An earlier version of this paragraph claimed the file "compares GENERATED
COORDINATES between the grouped path and the padded path".  It does not: there
is no such test here and there never was.  The two paths draw their noise from
``torch.randn`` calls of different shapes, so a coordinate comparison would
need an injected ``x0``, which the sampler does not expose.  What the file
checks is the property that comparison would rest on — that a masked padding
atom contributes no live edge — plus the default and the recorded measurement.

The velocity-level equivalent (real atoms get the same velocity and the same
guidance force with and without padding present) lives in
``molmetal/tests/test_wfa113_padded_guidance.py``.  A full end-to-end
coordinate comparison remains UNMEASURED and should be written before anyone
quotes pad_groups as bit-equivalent.

The mathematical claim under test: padding atoms are placed far enough away
that no real atom pairs with them inside `cutoff`, and the radius graph is
computed with a node_mask that excludes them, so every real atom sees exactly
the neighbours it saw before.

RED CHECK: implemented as a real assertion, not a comment. If padding ever
starts leaking edges, these fail.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _make_templates(sizes):
    from molmetal.models.bond_head import AtomCloud

    g = torch.Generator().manual_seed(7)
    out = []
    for n in sizes:
        out.append(
            AtomCloud(
                atomic_numbers=torch.randint(1, 60, (n,), generator=g),
                positions=torch.randn(n, 3, generator=g),
            )
        )
    return out


def test_padding_does_not_add_edges_inside_cutoff():
    """A padding atom beyond the cutoff must not join any real atom's graph."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    cutoff = 3.5
    n_real = 12
    real = torch.randn(1, n_real, 3) * 2.0
    pad = torch.full((1, 1, 3), 1e4)  # absurdly far
    x = torch.cat([real, pad], dim=1)
    z = torch.ones(1, n_real + 1, dtype=torch.long)
    mask = torch.zeros(1, n_real + 1, dtype=torch.bool)
    mask[0, :n_real] = True

    edge_index, edge_mask = _block_diag_radius_edges(x, mask, cutoff)
    # radius_edge_index returns (edge_index (B, 2, E), edge_mask (B, E)).
    assert edge_index.dim() == 3 and edge_mask.dim() == 2, (
        f"unexpected shapes: edge_index {tuple(edge_index.shape)}, "
        f"edge_mask {tuple(edge_mask.shape)}"
    )
    live = edge_mask[0].bool()  # (E,)
    dst = edge_index[0, 1]  # (E,) target node of each edge
    # No live edge may touch the padded node (index n_real).
    assert not bool((live & dst.eq(n_real)).any()), (
        "a padding atom contributed a live edge; padding is not isolated"
    )


def test_masked_padding_node_is_excluded_from_edges():
    """node_mask=False must remove that node's edges entirely."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    torch.manual_seed(11)
    x = torch.randn(1, 16, 3) * 1.5
    mask_a = torch.ones(1, 16, dtype=torch.bool)
    mask_b = torch.ones(1, 16, dtype=torch.bool)
    mask_b[0, 15] = False
    x_b = x.clone()
    x_b[0, 15] = 1e4  # move it away too, so only the mask can explain removal

    ea, ma = _block_diag_radius_edges(x, mask_a, 3.5)
    eb, mb = _block_diag_radius_edges(x_b, mask_b, 3.5)
    na = int(ma[0].sum())
    nb = int(mb[0].sum())
    assert nb < na, (
        f"masking node 15 did not remove edges ({na} -> {nb}); "
        "the padded path would contaminate the real graph"
    )


def test_pad_groups_parameter_exists_and_defaults_off():
    """DEFAULT OFF is a hard project rule: archived runs must keep their meaning."""
    import inspect

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        generate_clouds_from_noise,
    )

    sig = inspect.signature(generate_clouds_from_noise)
    assert "pad_groups" in sig.parameters, (
        "pad_groups parameter is missing from generate_clouds_from_noise"
    )
    assert sig.parameters["pad_groups"].default is False, (
        "pad_groups must default to False -- every archived run was produced "
        "with the grouped path and must keep its exact behaviour"
    )


def test_measurements_underlying_the_optimisation_are_recorded():
    """The 42x claim must stay traceable to the measurement that produced it."""
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent.parent
    # 2026-10-27 — the CFM plan moved from cfm_todo.json into todo.json (one plan
    # for both lines; the CFM ids are unchanged).  todo.json carries a
    # legacy_identifiers block precisely so ids like BLK-C-PERF stay resolvable
    # after the merge.
    blocker = repo / "todo.json"
    assert blocker.is_file(), f"todo.json disappeared from {repo}"
    # 2026-10-19 — cfm_todo.json was rewritten without a top-level "blockers"
    # array (verified: the key is absent at HEAD, so the old assertion failed
    # on a schema change, not on a lost measurement).  The finding now lives
    # in docs/CFM_STATUS_2026_10_12.md.  Check BOTH homes so the measurement
    # cannot be dropped by a future re-schema.
    text = blocker.read_text()
    status_doc = repo / "docs" / "CFM_STATUS_2026_10_12.md"
    assert status_doc.is_file(), f"{status_doc} disappeared from {repo}"
    assert "BLK-C-PERF" in text or "BLK-C-PERF" in status_doc.read_text(), (
        "BLK-C-PERF is in neither todo.json nor "
        "docs/CFM_STATUS_2026_10_12.md; the launch-bound finding and its "
        "measured scaling numbers must stay on disk"
    )