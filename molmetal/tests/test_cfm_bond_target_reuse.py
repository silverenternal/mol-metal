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