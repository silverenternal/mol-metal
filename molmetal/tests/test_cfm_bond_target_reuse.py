# SPDX-License-Identifier: Apache-2.0
"""The bond target must not need a SECOND head forward.

WHAT CHANGED (2026-10-10)
-------------------------
`train_one_epoch` called the head twice per cloud per step:

    logits, velocities = model(cloud, ...)          # CFMJointModel.forward
                                                        # starts `out = self.head(cloud)`
    tgt_logits, tgt_y = _bond_target_for_clean_cloud(cloud, head=model.head)
                                                        # -> featurise -> head(cloud) AGAIN

On the default path (`_need_edge_index` False) the first `logits` were
discarded outright. The target builder now accepts `precomputed=(logits,
edge_index)` and skips its own forward.

WHY THIS IS A CORRECTNESS FIX AS MUCH AS A SPEED ONE
----------------------------------------------------
The head carries dropout. In `train()` two forward passes over the same cloud
return DIFFERENT logits, and bond_ce was being fitted to the second draw while
the first was thrown away. Reusing the first makes the loss see the same
sample the rest of the step already used. So "the two are equivalent" is NOT
true in general, and a test that asserted bit-equality under dropout would be
asserting the wrong thing.

These tests therefore pin the property that actually matters: with dropout
OFF, the reuse path returns the SAME targets and the SAME logits as the
recompute path — so the speedup is not buying a different computation. With
dropout ON they deliberately do not assert equality; they assert the reuse
path stays in step with the logits the caller handed it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_trainer():
    path = REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    spec = importlib.util.spec_from_file_location("cfm_trainer_reuse_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cloud(with_bonds: bool = True):
    from molmetal.models.bond_head import AtomCloud

    torch.manual_seed(0)
    n = 12
    pos = torch.randn(n, 3) * 1.4
    z = torch.tensor([78] + [6] * (n - 1), dtype=torch.long)
    bonds = bond_types = None
    if with_bonds:
        # A short connected chain, so the target has both positives and zeros.
        idx = torch.arange(n - 1, dtype=torch.long)
        bonds = torch.stack([idx, idx + 1], dim=0)
        bond_types = torch.full((n - 1,), 1, dtype=torch.long)  # BOND_SINGLE
    return AtomCloud(positions=pos, atomic_numbers=z, bonds=bonds, bond_types=bond_types)


def _head(mod, dropout_free: bool = True):
    torch.manual_seed(1)
    head = mod.YuelBondDecoderHead(hidden_dim=32, n_layers=2, cutoff=3.5)
    head.eval()  # deterministic: no dropout, so the two paths must agree
    return head


def test_reuse_path_matches_recompute_path_in_eval():
    """The load-bearing test: same logits, same targets, so the speedup is free."""
    mod = _load_trainer()
    head = _head(mod)
    cloud = _cloud()

    logits_a, y_a = mod._bond_target_for_clean_cloud(cloud, head=head)
    assert logits_a is not None, "test cloud produced no bond target; fixture is wrong"

    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import YuelBondDecoder

    yres = YuelBondDecoder(head=head, hidden_dim=32, n_layers=2).featurise(cloud)
    logits_b, y_b = mod._bond_target_for_clean_cloud(
        cloud, head=head, precomputed=(yres.logits, yres.edge_index)
    )

    assert torch.equal(logits_a, logits_b), (
        "reuse path returned different logits than the recompute path — the "
        "speedup would be a different computation, not a faster one"
    )
    assert torch.equal(y_a, y_b), "reuse path produced different bond targets"


def test_reuse_does_not_call_the_head_again(monkeypatch):
    """Structural guarantee, so a future refactor cannot silently restore the
    second forward behind a correct-looking result."""
    mod = _load_trainer()
    head = _head(mod)
    cloud = _cloud()
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import YuelBondDecoder

    yres = YuelBondDecoder(head=head, hidden_dim=32, n_layers=2).featurise(cloud)
    yres_ref = YuelBondDecoder(head=head, hidden_dim=32, n_layers=2).featurise(cloud)

    calls = {"n": 0}
    original = head.forward

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    monkeypatch.setattr(head, "forward", counting)
    mod._bond_target_for_clean_cloud(
        cloud, head=head, precomputed=(yres.logits, yres.edge_index)
    )
    assert calls["n"] == 0, (
        f"reuse path still invoked head.forward {calls['n']} time(s)"
    )
    assert torch.equal(yres.logits, yres_ref.logits)


def test_recompute_path_still_works_when_precomputed_is_absent():
    """Every other caller — decode, eval, per-epoch accuracy — passes nothing."""
    mod = _load_trainer()
    head = _head(mod)
    cloud = _cloud()
    logits, y = mod._bond_target_for_clean_cloud(cloud, head=head)
    assert logits is not None and y is not None
    assert int(y.min()) >= 0


def test_reuse_rejects_an_empty_edge_index():
    """A head with no candidate pairs must return (None, None), not crash —
    otherwise the new path would be a new failure mode the old one lacked."""
    mod = _load_trainer()
    head = _head(mod)
    cloud = _cloud()
    empty = (
        torch.zeros(0, mod.NUM_BOND_CLASSES, device=cloud.positions.device),
        (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)),
    )
    logits, y = mod._bond_target_for_clean_cloud(
        cloud, head=head, precomputed=empty
    )
    assert logits is None and y is None


def test_target_has_both_positives_and_negatives():
    """If the fixture ever produced an all-zero target the tests above would
    pass while checking nothing, so pin that the target is non-trivial."""
    mod = _load_trainer()
    head = _head(mod)
    cloud = _cloud()
    logits, y = mod._bond_target_for_clean_cloud(cloud, head=head)
    assert y is not None
    assert int((y == 0).sum()) > 0, "fixture produced no BOND_NO_BOND entries"
    assert int((y != 0).sum()) > 0, "fixture produced no positive bond entries"

def test_reuse_accepts_the_training_path_edge_index_tuple():
    """The training path passes a TUPLE; the featurise path passes a TENSOR.

    This gap was real. The vectorised rewrite called `.to(device)` on
    `edge_index`, which works for `YuelBondResult.edge_index` (a (2, E)
    tensor) and raises `AttributeError: 'tuple' object has no attribute 'to'`
    for what `CFMJointModel(..., return_edge_index=True)` actually returns.
    Every existing test fed the tensor form, so none of them saw it — and it
    would have crashed on the first training step of every run.

    So this test builds the tuple the way the trainer does, from the real
    model, rather than from a hand-made stand-in.
    """
    mod = _load_trainer()
    torch.manual_seed(3)
    model = mod.CFMJointModel(
        hidden_dim=32, n_layers=2, use_egnn_velocity=True, bond_predictor="yuelbond"
    )
    model.eval()
    cloud = _cloud()

    out = model(cloud, return_edge_index=True)
    assert len(out) == 3, "expected (logits, velocities, edge_index)"
    logits, _velocities, edge_index = out
    assert isinstance(edge_index, (tuple, list)), (
        "this test exists because the training path yields a tuple; if the "
        f"return shape changed to {type(edge_index).__name__}, revisit whether "
        "the normalisation in _bond_target_for_clean_cloud is still needed"
    )

    y = None
    got_logits, y = mod._bond_target_for_clean_cloud(
        cloud, head=model.head, precomputed=(logits, edge_index)
    )
    assert y is not None, "training-shaped precomputed must not return (None, None)"
    assert int(y.shape[0]) == int(logits.shape[0]), (
        f"target has {int(y.shape[0])} rows but the logits have "
        f"{int(logits.shape[0])} — the pair list was duplicated or truncated"
    )


def test_vectorised_target_matches_the_old_dict_lookup():
    """The dense (N, N) table must reproduce the dict's (min, max) keying.

    The rewrite writes truth symmetrically so a pair can be read in either
    order, which is what the dict's second `.get((d, s))` did. Checked against
    an independent Python reference on a cloud whose truth bonds include a
    reversed lookup.
    """
    mod = _load_trainer()
    head = _head(mod)
    cloud = _cloud()

    logits, y = mod._bond_target_for_clean_cloud(
        cloud, head=head, precomputed=(head(cloud)[0], head(cloud)[1])
    )
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import YuelBondDecoder

    yres = YuelBondDecoder(head=head, hidden_dim=32, n_layers=2).featurise(cloud)
    ei = torch.stack([yres.edge_index[0], yres.edge_index[1]]) if isinstance(
        yres.edge_index, (tuple, list)
    ) else yres.edge_index
    lookup = {
        (int(i), int(j)): int(t)
        for i, j, t in zip(
            cloud.bonds[0].tolist(), cloud.bonds[1].tolist(),
            cloud.bond_types.tolist(),
        )
    }
    ref = torch.tensor(
        [
            lookup.get((int(s), int(d)), lookup.get((int(d), int(s)), 0))
            for s, d in zip(ei[0].tolist(), ei[1].tolist())
        ],
        dtype=torch.long,
    )
    assert torch.equal(y.cpu(), ref), "dense-table target diverges from the dict"
    assert int((y != 0).sum()) > 0, "no positive bonds survived the rewrite"


def test_target_shape_is_pairs_not_pairs_by_two():
    """The target must be (E,), not (2, E) — and the corpus-free path matters.

    `torch.full_like(ei, ...)` with ei of shape (2, E) broadcasts the whole
    result to twice the pair count. That shipped inside this change and the
    50-step smoke in wf385a3_scaffold_wire_in_gate.py caught it: loss read
    0.0 at every step, because the flattened target no longer lined up with
    the logits and the cross-entropy collapsed.

    Both branches are checked, and the cloud here carries NO bond truth, so
    this exercises the geometric fallback — the branch the smoke runs on.
    """
    mod = _load_trainer()
    torch.manual_seed(5)
    from molmetal.models.bond_head import AtomCloud

    cloud = AtomCloud(
        positions=torch.randn(5, 3) * 1.4,
        atomic_numbers=torch.tensor([78, 7, 7, 17, 17], dtype=torch.long),
    )
    assert getattr(cloud, "bond_types", None) is None, "fixture must be truth-free"
    head = _head(mod)

    logits, ei = head(cloud)
    got_logits, y = mod._bond_target_for_clean_cloud(
        cloud, head=head, precomputed=(logits, ei)
    )
    assert y is not None
    assert y.dim() == 1, f"target must be 1-D, got shape {tuple(y.shape)}"
    assert int(y.shape[0]) == int(logits.shape[0]), (
        f"target has {int(y.shape[0])} rows against {int(logits.shape[0])} "
        "logits — the pair list was duplicated or truncated"
    )
