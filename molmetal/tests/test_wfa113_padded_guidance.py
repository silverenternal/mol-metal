# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 / T44 + T37 — the padded sampling branch must not silently differ.

TWO DEFECTS, ONE BRANCH
-----------------------
SA-24 / D44.  ``generate_clouds_from_noise`` has two paths.  The grouped path
wraps the velocity closure with ``_BondLengthGuidance`` when
``bond_length_guidance_weight > 0``.  The padded path (``pad_groups: on``) went
straight from ``_conditioned_velocity_fn`` to ``ODESolver`` and never read the
weight at all.  A run with ``pad_groups: on`` and weight 0.1 therefore sampled
WITHOUT the term while the same config with ``pad_groups: off`` sampled WITH
it, and nothing in the log or the artifact said which had happened.  Three
configs on disk set a non-zero weight and none of them sets ``gen_pad_groups``,
which is the only reason the two numbers were never compared.

SA-25-adjacent / T37.  The padded loop ran without ``torch.no_grad()`` while
the grouped loop had it, so every ODE step built an autograd graph over the
whole padded batch.

WHY THE FIX IS NOT JUST "ADD THE WRAPPER"
-----------------------------------------
``_BondLengthGuidance`` builds its OWN radius graph with
``radius_edge_index(x, None, cutoff)`` — no node mask.  In a padded batch that
graph contains padding atoms, so (a) a real atom could receive a force from a
padding atom and (b) padding atoms (Z=0, table entry 1.50 A) would be pulled
toward the molecule.  That is the opposite of the padded path's invariant.
So the wrapper gained a ``node_mask`` parameter and the trainer passes the same
mask the velocity field gets.

WHAT THIS FILE MEASURES
-----------------------
Velocity equivalence on the real atoms (not a coordinate comparison through
the sampler, which would be confounded by the two paths drawing their noise
from differently shaped ``torch.randn`` calls), plus the two branch-level
facts: the wrapper is constructed, and the loop runs without grad.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from molmetal.models.bond_head import AtomCloud
from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
    _conditioned_velocity_fn,
    _with_bond_length_guidance,
    generate_clouds_from_noise,
)


class _StubVelocity(torch.nn.Module):
    """Deterministic velocity that depends on x and on the radius graph.

    It reads the graph so that a leak of padding edges would show up in the
    real atoms' velocity, and it records whether grad was enabled when it ran.
    """

    cutoff = 3.5

    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.grad_enabled: list[bool] = []

    def vel_field(self, x, z, edge_index, t, edge_mask=None, cond_cn=None):
        self.grad_enabled.append(torch.is_grad_enabled())
        v = -0.25 * x
        if edge_mask is not None and edge_index.shape[-1] > 0:
            src = edge_index[:, 0, :]
            dst = edge_index[:, 1, :]
            m = edge_mask.float().unsqueeze(-1)
            disp = x.gather(1, dst.unsqueeze(-1).expand(-1, -1, 3)) - x.gather(
                1, src.unsqueeze(-1).expand(-1, -1, 3)
            )
            agg = torch.zeros_like(x)
            agg.scatter_add_(
                1, dst.unsqueeze(-1).expand(-1, -1, 3), disp * m * 0.01
            )
            v = v + agg
        return {"vel": v}


def _cloud(n: int, seed: int) -> AtomCloud:
    g = torch.Generator().manual_seed(seed)
    z = torch.randint(1, 20, (n,), generator=g)
    z[0] = 78
    return AtomCloud(
        atomic_numbers=z,
        positions=torch.randn(n, 3, generator=g),
        metal_atom_index=torch.tensor(0),
    )


# ---------------------------------------------------------------------------
# 1. The wrapper is one definition, and weight 0 is an identity
# ---------------------------------------------------------------------------


def test_weight_zero_returns_the_same_callable():
    def vf(x, t):
        return x

    assert _with_bond_length_guidance(vf, torch.zeros(1, 3), 3.5, 0.0) is vf


# ---------------------------------------------------------------------------
# 2. Padding must not contribute a guidance force
# ---------------------------------------------------------------------------


def test_guidance_does_not_leak_padding():
    """Real atoms must feel the same force with and without padding present.

    Before the fix the guidance built ``radius_edge_index(x, None, cutoff)``,
    so padding atoms entered the graph.  They are placed NEAR the real atoms
    here on purpose: the mask, not the distance, is what has to exclude them.
    """
    torch.manual_seed(3)
    n = 6
    x_real = torch.randn(1, n, 3) * 1.2
    z_real = torch.tensor([[78, 6, 7, 6, 8, 6]])

    # Padding: three atoms placed INSIDE the cutoff of the real ones.
    pad = x_real[:, :3, :] + 0.4
    x_pad = torch.cat([x_real, pad], dim=1)
    z_pad = torch.cat(
        [z_real, torch.zeros(1, 3, dtype=torch.long)], dim=1
    )
    mask_real = torch.ones(1, n, dtype=torch.bool)
    mask_pad = torch.cat(
        [mask_real, torch.zeros(1, 3, dtype=torch.bool)], dim=1
    )

    v_real = _with_bond_length_guidance(
        lambda x, t: torch.zeros_like(x), z_real, 3.5, 0.1, mask_real
    )(x_real, torch.zeros(1))
    v_pad = _with_bond_length_guidance(
        lambda x, t: torch.zeros_like(x), z_pad, 3.5, 0.1, mask_pad
    )(x_pad, torch.zeros(1))

    assert torch.allclose(v_real, v_pad[:, :n, :], atol=1e-6), (
        "padding atoms changed the guidance force on the real atoms; the "
        "node_mask is not reaching radius_edge_index"
    )
    assert torch.allclose(v_pad[:, n:, :], torch.zeros(1, 3, 3), atol=1e-6), (
        "padding atoms received a guidance force of their own"
    )


# ---------------------------------------------------------------------------
# 3. The padded branch actually applies the term (D44)
# ---------------------------------------------------------------------------


def test_padded_branch_constructs_the_guidance(monkeypatch):
    """``pad_groups=True`` + weight > 0 must build the wrapper.

    Counted by replacing the class in the adapter module, which is where the
    helper imports it from.  Before the fix this count was 0 for the padded
    branch and >= 1 for the grouped branch.
    """
    import molmetal.adapters.flow_matching_lipman as fml

    real_cls = fml._BondLengthGuidance
    made: list[float] = []

    class _Spy(real_cls):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):
            made.append(float(kw.get("guidance_weight", -1.0)))
            super().__init__(*a, **kw)

    monkeypatch.setattr(fml, "_BondLengthGuidance", _Spy)

    model = _StubVelocity()
    templates = [_cloud(6, 1), _cloud(9, 2)]
    out = generate_clouds_from_noise(
        model,
        templates,
        torch.device("cpu"),
        method="euler",
        n_steps=2,
        seed=0,
        noise_sigma=1.0,
        pad_groups=True,
        pad_group_bucket=8,
        bond_length_guidance_weight=0.1,
    )
    assert len(out) == 2
    assert made, "pad_groups=True never constructed _BondLengthGuidance"
    assert all(w == pytest.approx(0.1) for w in made)


def test_grouped_branch_still_constructs_the_guidance(monkeypatch):
    """The control: the branch that always worked must keep working."""
    import molmetal.adapters.flow_matching_lipman as fml

    real_cls = fml._BondLengthGuidance
    made: list[float] = []

    class _Spy(real_cls):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):
            made.append(float(kw.get("guidance_weight", -1.0)))
            super().__init__(*a, **kw)

    monkeypatch.setattr(fml, "_BondLengthGuidance", _Spy)
    model = _StubVelocity()
    generate_clouds_from_noise(
        model,
        [_cloud(6, 1), _cloud(9, 2)],
        torch.device("cpu"),
        method="euler",
        n_steps=2,
        seed=0,
        noise_sigma=1.0,
        pad_groups=False,
        bond_length_guidance_weight=0.1,
    )
    assert made, "the grouped branch stopped constructing the guidance"


# ---------------------------------------------------------------------------
# 4. The padded branch runs without grad (T37)
# ---------------------------------------------------------------------------


def test_padded_branch_runs_without_grad():
    model = _StubVelocity()
    generate_clouds_from_noise(
        model,
        [_cloud(6, 1), _cloud(9, 2)],
        torch.device("cpu"),
        method="euler",
        n_steps=2,
        seed=0,
        noise_sigma=1.0,
        pad_groups=True,
        pad_group_bucket=8,
        bond_length_guidance_weight=0.0,
    )
    assert model.grad_enabled, "the stub velocity was never called"
    assert not any(model.grad_enabled), (
        "the padded branch sampled with grad enabled; every ODE step builds "
        "an autograd graph over the whole padded batch"
    )


def test_grouped_branch_runs_without_grad():
    model = _StubVelocity()
    generate_clouds_from_noise(
        model,
        [_cloud(6, 1), _cloud(9, 2)],
        torch.device("cpu"),
        method="euler",
        n_steps=2,
        seed=0,
        noise_sigma=1.0,
        pad_groups=False,
        bond_length_guidance_weight=0.0,
    )
    assert model.grad_enabled and not any(model.grad_enabled)
