"""Wf-A108g — size-homogeneous grouping of the FM batch.

The memory cost of the FM forward is set by how many atoms one padded
batch carries, not by how many parameters the field has.  MEASURED on 64
real MetalCytoToxDB clouds (2026-10-23): atom counts run 16 / 38 / 225
(min / median / max) with mean 46.2, so padding every cloud to the batch
maximum puts 2 957 real atoms into 14 400 slots — 79.5% of the batch is
zeros, and 83.6% of the ``(B, 2, E)`` edge slots too.

Splitting the batch into groups whose sizes are within a factor of
``ratio`` cuts that pad width per group.  The split is EXACT, not an
approximation: the radius graph is block-diagonal and ``node_mask`` drops
every padded endpoint, so clouds never see each other inside one forward.
The one exception is the legacy radial head, whose centroid is a mean
over the padded width; :meth:`CFMJointModel._fm_padding_invariant` gates
on that and leaves chunking off.

These tests are CPU-only and need no checkpoint: they exercise the plan
and the merge with a per-atom stub field, which is enough to catch the
two merge bugs this code had (flattening before slicing, and slicing the
already-flattened chunk at the wrong offset).
"""

from __future__ import annotations

import pytest
import torch

from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel


class _PerAtomField(torch.nn.Module):
    """Stand-in for the EGNN velocity field.

    Every output is a per-atom function of ``x`` and the cloud's ``t``, so
    the padded result is exactly the real result on the real rows.  That
    is the property the grouping relies on; the real field satisfies it
    whenever :meth:`CFMJointModel._fm_padding_invariant` is True.
    """

    def __init__(self, velocity_head: str = "vector_channels", masked: bool = False):
        super().__init__()
        self.velocity_head = velocity_head
        self.radial_centroid_masked = masked
        self.prediction_target = "velocity"
        self.calls: list[int] = []

    def forward(self, x, z, edge_index, t_b, edge_mask=None, pocket_embed=None):
        self.calls.append(int(x.shape[0]))
        return {
            "v": x * 2.0 + t_b[:, None, None],
            "x1_hat": x + 1.0,
        }


def _model(field: _PerAtomField, ratio: float, max_clouds: int = 64) -> CFMJointModel:
    """A CFMJointModel with only the attributes the FM forward reads."""
    model = CFMJointModel.__new__(CFMJointModel)
    torch.nn.Module.__init__(model)
    model.vel_field = field
    model.cutoff = 5.0
    model.fm_batch_chunk_ratio = ratio
    model.fm_batch_chunk_max_clouds = max_clouds
    return model


def _batch(sizes):
    x = torch.randn(sum(sizes), 3)
    z = torch.randint(1, 8, (sum(sizes),))
    t = torch.rand(len(sizes))
    t_atom = t.repeat_interleave(torch.tensor(sizes))
    return x, z, t_atom


# ---------------------------------------------------------------------------
# The plan itself
# ---------------------------------------------------------------------------
def test_plan_partitions_every_cloud_exactly_once() -> None:
    sizes = [16, 225, 38, 40, 17, 90, 16, 225, 46, 60]
    groups = CFMJointModel._fm_chunk_plan(sizes, 1.5, 64)
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(len(sizes)))
    assert len(flat) == len(set(flat)), "a cloud appears in two groups"


def test_plan_bounds_the_size_spread_within_a_group() -> None:
    sizes = [16, 17, 20, 38, 46, 60, 90, 225]
    for ratio in (1.0, 1.25, 1.5, 2.0):
        for group in CFMJointModel._fm_chunk_plan(sizes, ratio, 64):
            lo, hi = min(sizes[i] for i in group), max(sizes[i] for i in group)
            assert hi <= lo * ratio, f"ratio={ratio} group={group} spans {lo}..{hi}"


def test_plan_respects_max_clouds() -> None:
    sizes = [50] * 20
    groups = CFMJointModel._fm_chunk_plan(sizes, 2.0, 6)
    assert all(len(g) <= 6 for g in groups)
    assert sum(len(g) for g in groups) == 20


def test_plan_is_deterministic() -> None:
    sizes = [225, 16, 38, 225, 17, 40]
    first = CFMJointModel._fm_chunk_plan(sizes, 1.5, 64)
    assert first == CFMJointModel._fm_chunk_plan(sizes, 1.5, 64)


def test_plan_cuts_the_padding_of_a_skewed_batch() -> None:
    """One 225-atom outlier must not pad the 63 clouds around it."""
    sizes = [225] + [16] * 21 + [38] * 20 + [46] * 22
    padded = len(sizes) * max(sizes)
    grouped = sum(
        len(g) * max(sizes[i] for i in g)
        for g in CFMJointModel._fm_chunk_plan(sizes, 1.5, 64)
    )
    real = sum(sizes)
    # Single-group padding is 64*225 = 14 400 slots for 2 333 real atoms.
    assert (padded, real) == (14400, 2333)
    assert grouped < padded / 3, f"grouped slots {grouped} vs padded {padded}"


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_padding_invariance_gate() -> None:
    assert _model(_PerAtomField("vector_channels"), 1.5)._fm_padding_invariant()
    assert _model(_PerAtomField("radial", masked=True), 1.5)._fm_padding_invariant()
    # Legacy radial centroid is x_t.mean(dim=1) over the padded width.
    assert not _model(_PerAtomField("radial", masked=False), 1.5)._fm_padding_invariant()


# ---------------------------------------------------------------------------
# The forward
# ---------------------------------------------------------------------------
def test_chunked_forward_matches_the_single_padded_forward() -> None:
    sizes = [16, 38, 225, 17, 40, 20]
    x, z, t_atom = _batch(sizes)
    field = _PerAtomField()

    single = _model(field, 0.0)
    out_pad, node_mask = single._fm_field_forward(x, t_atom, z, cloud_sizes=sizes)
    assert tuple(out_pad["v"].shape) == (len(sizes), max(sizes), 3)
    ref = out_pad["v"][node_mask]

    chunked = _model(field, 1.5)
    out_flat, flat_mask = chunked._fm_field_forward(x, t_atom, z, cloud_sizes=sizes)
    assert tuple(out_flat["v"].shape) == (sum(sizes), 3)
    assert bool(flat_mask.all()) and flat_mask.shape == (sum(sizes),)

    assert torch.allclose(out_flat["v"], ref, atol=1e-6, rtol=0)
    assert torch.allclose(out_flat["x1_hat"], out_pad["x1_hat"][node_mask], atol=1e-6)


def test_chunked_forward_actually_splits_a_skewed_batch() -> None:
    sizes = [16, 38, 225, 17, 40, 20]
    x, z, t_atom = _batch(sizes)
    field = _PerAtomField()
    _model(field, 1.5)._fm_field_forward(x, t_atom, z, cloud_sizes=sizes)
    assert len(field.calls) > 1, "a 14x size spread must not stay in one group"
    assert sum(field.calls) == len(sizes)


def test_chunking_stays_off_for_a_padding_dependent_field() -> None:
    sizes = [16, 225]
    x, z, t_atom = _batch(sizes)
    field = _PerAtomField("radial", masked=False)
    out, node_mask = _model(field, 1.5)._fm_field_forward(x, t_atom, z, cloud_sizes=sizes)
    assert field.calls == [2], "one padded forward, as before Wf-A108g"
    assert tuple(out["v"].shape) == (2, 225, 3)
    assert node_mask.shape == (2, 225)


def test_cloud_sizes_must_cover_every_row() -> None:
    x, z, t_atom = _batch([10, 12])
    with pytest.raises(ValueError, match="cloud boundaries"):
        _model(_PerAtomField(), 1.5)._fm_field_forward(x, t_atom, z, cloud_sizes=[10, 11])


def test_ratio_zero_is_the_default_and_is_bit_identical() -> None:
    sizes = [16, 38, 225]
    x, z, t_atom = _batch(sizes)
    field = _PerAtomField()
    _model(field, 0.0)._fm_field_forward(x, t_atom, z, cloud_sizes=sizes)
    assert field.calls == [3]
