"""Tests for the BLK-C-L3 sampling-time guidance probe.

The force is a repulsive core: U = strength * sum_{i<j} (d0_ij / d_ij)^3 with
d0 = 0.85 * (rcov_i + rcov_j). These tests pin the two things that are easy to
get silently wrong and that would make the whole A/B meaningless:

  1. the SIGN -- a sign flip turns a repulsion into an attraction, which would
     make the sampled geometry collapse further and could be mistaken for
     "guidance does not work";
  2. the DIRECTION of the asymmetry -- if the force is symmetric under i<->j
     exchange in the wrong way, every pair is double-counted and the magnitude
     silently doubles.

RED CHECK REPORTED (see the module docstring of
molmetal/scripts/wf_blkc_l3_guidance_probe.py and the wave notes): the force was
FIRST written returning ``-f``, which autograd shows is an attraction. Reverting
to that form fails ``test_force_matches_autograd_on_a_two_atom_system``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "molmetal" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.wf_blkc_l3_guidance_probe import (  # noqa: E402
    guidance_force,
)


def _autograd_force(x: torch.Tensor, rc: torch.Tensor, strength: float) -> torch.Tensor:
    """-dU/dx computed by autograd, counting each unordered pair exactly once.

    The mask must be strictly UPPER triangular. `~torch.eye(n)` looks right but
    is the full off-diagonal, which contains BOTH (i,j) and (j,i) and silently
    doubles every pair.
    """
    xg = x.clone().requires_grad_(True)
    n = x.shape[1]
    d = torch.cdist(xg[0], xg[0])  # (n, n)
    d0 = 0.85 * (rc[0].unsqueeze(0) + rc[0].unsqueeze(1))  # (n, n)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    u = strength * ((d0[mask] / d[mask]) ** 3).sum()
    u.backward()
    return -xg.grad[0].detach()


def test_force_matches_autograd_on_a_two_atom_system():
    x = torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]])
    rc = torch.tensor([[1.5, 1.5]])
    got = guidance_force(x, None, rc, 0.05)[0]  # (N, 3)
    want = _autograd_force(x, rc, 0.05)
    assert got.shape == want.shape, f"{tuple(got.shape)} != {tuple(want.shape)}"
    assert torch.allclose(got, want, atol=1e-4), f"{got} != {want}"


def test_force_is_repulsive_not_attractive():
    """The atom at the origin must be pushed AWAY from the atom at +x."""
    x = torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]])
    rc = torch.tensor([[1.5, 1.5]])
    f = guidance_force(x, None, rc, 0.05)[0]  # (N, 3)
    # atom0 sits at x=0, atom1 at x=+1.2, so atom0 must be pushed toward -x.
    assert f[0][0] < 0, f"atom0 pushed toward +x ({f[0][0]}): this is an ATTRACTION"
    assert f[1][0] > 0, f"atom1 pushed toward -x ({f[1][0]}): this is an ATTRACTION"


def test_force_shape_matches_positions():
    """A shape slip here broadcasts the velocity wrong and silently corrupts
    every guided sample."""
    x = torch.rand(3, 7, 3) * 2.0
    rc = torch.full((3, 7), 1.3)
    f = guidance_force(x, None, rc, 0.05)
    assert f.shape == x.shape, f"{tuple(f.shape)} != {tuple(x.shape)}"


def test_force_is_equal_and_opposite():
    x = torch.rand(1, 4, 3) * 2.0
    rc = torch.full((1, 4), 1.3)
    f = guidance_force(x, None, rc, 0.05)[0]
    assert torch.allclose(f.sum(dim=0), torch.zeros(3), atol=1e-4), f"net force {f.sum(0)}"


def test_force_vanishes_when_all_pairs_are_far_apart():
    """No overlap means no potential gradient -- the term must switch off."""
    x = torch.tensor([[[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]])
    rc = torch.tensor([[1.5, 1.5]])
    f = guidance_force(x, None, rc, 0.05)[0]
    assert f.abs().max() < 1e-6, f"non-zero force at 50 A separation: {f}"


def test_force_scales_with_strength():
    x = torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]])
    rc = torch.tensor([[1.5, 1.5]])
    a = guidance_force(x, None, rc, 0.05)[0]
    b = guidance_force(x, None, rc, 0.10)[0]
    assert torch.allclose(b, 2.0 * a, atol=1e-5), "force does not scale linearly in strength"


def test_guidance_increases_separation_under_plain_euler():
    """End-to-end sanity: a velocity field of ONLY this force must increase the
    interatomic separation, not decrease it."""
    for strength, expect_increase in ((0.05, True), (-0.05, False)):
        x = torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]])
        rc = torch.tensor([[1.5, 1.5]])
        d0_sep = (x[0, 0] - x[0, 1]).norm().item()
        # One small Euler step: enough to expose the sign, not enough for an
        # attractive run to tunnel through the pair and land back on d0.
        x = x + 0.01 * guidance_force(x, None, rc, strength)
        d1_sep = (x[0, 0] - x[0, 1]).norm().item()
        if expect_increase:
            assert d1_sep > d0_sep, f"repulsion did not push apart: {d0_sep} -> {d1_sep}"
        else:
            assert d1_sep < d0_sep, f"negative strength did not pull together: {d0_sep} -> {d1_sep}"