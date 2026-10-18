"""Wf-A82 (2026-10-18) — one construction site for CFMJointModel.

The inference side used to re-derive the architecture from a state_dict and
MEASURED it wrong on a real v2 checkpoint (`head.atom_encoder.weight` 13 vs 4,
`head.pair_head.0.weight` 131 vs 259).  `build_joint_model` is now the only
place the architecture is decided, and `_rebuild_v2_model` loads with
`strict=True` so any drift between the factory and an artifact is an error
rather than a silently half-initialised model.
"""

import argparse
import unittest

import torch

from molmetal.scripts.cfm_property_panel import _rebuild_v2_model
from molmetal.scripts.train_cfm_yuelbond_wasserstein import build_joint_model


def _args(**kw):
    base = dict(
        hidden_dim=16,
        n_layers=1,
        edge_cutoff=3.5,
        use_egnn_velocity=True,
        use_fm_objective=False,
        yuelbond_vocab="pt",
        bond_head_kind="single",
        bond_predictor="yuelbond",
        egnn_max_atomic_number=100,
        egnn_context_dropout=0.1,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class FactoryTest(unittest.TestCase):
    def test_is_deterministic(self) -> None:
        a = build_joint_model(_args())
        b = build_joint_model(_args())
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(sorted(sa), sorted(sb))
        for k in sa:
            self.assertEqual(tuple(sa[k].shape), tuple(sb[k].shape), msg=k)

    def test_reads_hidden_dim_from_args(self) -> None:
        a = build_joint_model(_args(hidden_dim=16))
        b = build_joint_model(_args(hidden_dim=32))
        self.assertNotEqual(
            tuple(a.state_dict()["head.atom_encoder.weight"].shape),
            tuple(b.state_dict()["head.atom_encoder.weight"].shape),
        )

    def test_defaults_fill_missing_keys(self) -> None:
        # A checkpoint written before a kwarg existed must still rebuild.
        thin = argparse.Namespace(
            hidden_dim=16, n_layers=1, edge_cutoff=3.5,
            use_egnn_velocity=True, yuelbond_vocab="pt",
        )
        m = build_joint_model(thin)
        self.assertTrue(any(k.startswith("vel_field.") for k in m.state_dict()))


class RebuildTest(unittest.TestCase):
    def test_strict_load_of_a_factory_state(self) -> None:
        args = _args()
        state = build_joint_model(args).state_dict()
        ck = {"args": vars(args), "model_state_dict": state}
        m = _rebuild_v2_model(ck, state)
        self.assertIsNotNone(m)
        self.assertTrue(any(k.startswith("vel_field.") for k in m.state_dict()))

    def test_missing_args_is_refused(self) -> None:
        args = _args()
        state = build_joint_model(args).state_dict()
        self.assertIsNone(_rebuild_v2_model({"model_state_dict": state}, state))

    def test_shape_mismatch_is_refused_not_half_loaded(self) -> None:
        args = _args()
        state = dict(build_joint_model(args).state_dict())
        k = "head.atom_encoder.weight"
        state[k] = torch.zeros(
            state[k].shape[0], state[k].shape[1] + 1
        )
        self.assertIsNone(
            _rebuild_v2_model(
                {"args": vars(args), "model_state_dict": state}, state
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
