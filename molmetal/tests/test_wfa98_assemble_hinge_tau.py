# SPDX-License-Identifier: Apache-2.0
"""Wf-A98 (2026-10-19) -- the assemble dead zone must match the corpus.

The defect these tests pin: the shipped hinge tau (0.30) was derived from
tmQM and then applied to every cohort.  metrics/by_metric/
cfm_assemble_hinge_tau.json says outright that tau is not portable across
cohorts, and on the synthetic click cohort the model sits at stay_mean
0.2093 -- under 0.30 -- so the hinge floored the term at exactly 0.0 every
epoch while assemble_enabled reported 1.0.

derive_assemble_hinge_tau computes the threshold from the run's OWN bond
truth with the rule the published artifact used (max * 1.13, rounded up to
0.01), which reproduces 0.30 from the published 0.2649.
"""
from __future__ import annotations

import math

import pytest
import torch


def _trainer():
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    return T


def _chain(n: int = 5, order: int = 1):
    from molmetal.models.bond_head import AtomCloud

    pos = torch.zeros((n, 3), dtype=torch.float32)
    for i in range(n):
        pos[i, 0] = 1.5 * i
    bonds = torch.tensor([[i for i in range(n - 1)], [i + 1 for i in range(n - 1)]])
    return AtomCloud(
        positions=pos,
        atomic_numbers=torch.full((n,), 6, dtype=torch.long),
        bonds=bonds,
        bond_types=torch.full((n - 1,), order, dtype=torch.long),
    )


def test_rule_reproduces_the_published_threshold():
    """0.2649 * 1.13 = 0.2993 -> 0.30, the value shipped in the artifact."""
    T = _trainer()
    rep = T.derive_assemble_hinge_tau([_chain(5)])
    assert rep["tau"] == pytest.approx(
        math.ceil(rep["max"] * 1.13 * 100.0) / 100.0, abs=1e-12
    )
    # The published rule on the published input must land on the shipped value.
    assert math.ceil(0.2649 * 1.13 * 100.0) / 100.0 == pytest.approx(0.30)


def test_tau_never_penalises_the_truth_graph():
    """The whole point of the dead zone: a correct complex is never charged."""
    T = _trainer()
    clouds = [_chain(5), _chain(7), _chain(4)]
    rep = T.derive_assemble_hinge_tau(clouds)
    assert rep["tau"] > rep["max"]
    from molmetal.molmetal_lam.lam_chem.assemble_connectivity import (
        mean_stay_probability,
    )

    for c in clouds:
        e = int(c.bond_types.shape[0])
        logits = torch.full((e, 5), -20.0)
        logits[torch.arange(e), c.bond_types.long()] = 20.0
        stay = mean_stay_probability(logits, c.bonds, c.atomic_numbers)
        assert stay <= rep["tau"]


def test_refuses_when_no_cloud_carries_bond_truth():
    """Silently keeping a foreign constant is the defect, not the fallback."""
    T = _trainer()
    from molmetal.models.bond_head import AtomCloud

    bare = [
        AtomCloud(positions=torch.zeros((4, 3)), atomic_numbers=torch.full((4,), 6))
        for _ in range(3)
    ]
    with pytest.raises(RuntimeError, match="0 of 3"):
        T.derive_assemble_hinge_tau(bare)


def test_inert_flag_is_declared_in_the_stats_defaults():
    T = _trainer()
    assert T._LAST_ASSEMBLE_STATS["assemble_hinge_tau"] == -1.0
    assert T._LAST_ASSEMBLE_STATS["assemble_inert"] == 0.0


def test_tau_mode_flag_parses_and_defaults_to_fixed(monkeypatch=None):
    T = _trainer()
    import sys

    old = sys.argv
    try:
        sys.argv = ["train", "--assemble-hinge-tau-mode", "corpus"]
        a = T.parse_args()
    finally:
        sys.argv = old
    assert a.assemble_hinge_tau_mode == "corpus"
    try:
        sys.argv = ["train"]
        b = T.parse_args()
    finally:
        sys.argv = old
    assert b.assemble_hinge_tau_mode == "fixed"


def test_integrate_returns_outputs_in_input_order():
    """Wf-A98 — the positional contract every probe relies on.

    integrate() batches by atom count and used to emit in size-sorted order,
    so zip(templates, out) paired a template with another molecule's sampled
    geometry.  On a mixed-size cohort that is silent unless the caller reads
    the template.
    """
    from molmetal.scripts.wf_blkc_l3_guidance_probe import integrate
    from molmetal.models.bond_head import AtomCloud

    class _ZeroVF:
        def __call__(self, x, z, edge_index, t, edge_mask=None):
            return {"vel": torch.zeros_like(x)}

    class _FakeModel:
        cutoff = 3.5
        vel_field = _ZeroVF()

    sizes = [5, 3, 4]
    templates = [
        AtomCloud(
            positions=torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, 3),
            atomic_numbers=torch.full((n,), 6, dtype=torch.long),
        )
        for n in sizes
    ]
    out = integrate(
        _FakeModel(),
        templates,
        torch.device("cpu"),
        method="midpoint",
        n_steps=1,
        seed=0,
        noise_sigma=0.0,
        guidance_strength=0.0,
    )
    assert [len(o["zs"]) for o in out] == sizes
    assert [o["template_index"] for o in out] == [0, 1, 2]
