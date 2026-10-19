# SPDX-License-Identifier: Apache-2.0
"""Wf-A96 (2026-10-19) — legacy-checkpoint compatibility.

Two defects motivated these tests, both MEASURED on
molmetal/reports/cfm_fullcorpus_2026_10_10/checkpoints/
cfm_yuelbond_wasserstein_seed0_last.pt:

1. The checkpoint carries vel_field.time_embed_proj.{weight,bias} while the
   current field names the projection vel_field.t_to_h and builds it with
   bias=False.  Strict loading failed, so the L3 Phase-3 re-measure could not
   run at all.  migrate_legacy_vel_state_dict renames the keys AND rebuilds
   the module with a bias, so the trained bias is not silently dropped.

2. molmetal/eval/_pocket_encoder.try_build_full_cfm_model built
   CFMJointModel(hidden_dim, n_layers) and loaded with strict=False, which
   reported success while leaving every velocity-field tensor at random init.
   It now refuses that partial build.
"""
from __future__ import annotations

import pytest
import torch


def _trainer():
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    return T


def _tiny_egnn(t_embedding_dim: int = 8):
    T = _trainer()
    return T.CFMJointModel(
        hidden_dim=16,
        n_layers=1,
        use_egnn_velocity=True,
        t_embedding_dim=t_embedding_dim,
    )


def test_migrate_renames_and_keeps_the_trained_bias():
    T = _trainer()
    src = _tiny_egnn(8)
    sd = src.state_dict()
    # Simulate the legacy layout: same tensors, old name, plus a bias the
    # current module does not have.
    legacy = {}
    for k, v in sd.items():
        if k == "vel_field.t_to_h.weight":
            legacy["vel_field.time_embed_proj.weight"] = v.clone()
        else:
            legacy[k] = v.clone()
    bias = torch.arange(16, dtype=torch.float32)
    legacy["vel_field.time_embed_proj.bias"] = bias

    dst = _tiny_egnn(8)
    migrated, report = T.migrate_legacy_vel_state_dict(dst, legacy, verbose=False)

    assert report["renamed"] == 2
    assert report["rebuilt_with_bias"] is True
    assert report["t_embedding_dim"] == 8
    assert "vel_field.time_embed_proj.weight" not in migrated
    assert "vel_field.t_to_h.weight" in migrated

    missing, unexpected = dst.load_state_dict(migrated, strict=False)
    assert missing == [] and unexpected == []
    assert dst.vel_field.t_to_h.bias is not None
    assert torch.equal(dst.vel_field.t_to_h.bias, bias)


def test_migrate_is_a_noop_without_legacy_keys():
    T = _trainer()
    src = _tiny_egnn(8)
    sd = src.state_dict()
    dst = _tiny_egnn(8)
    out, report = T.migrate_legacy_vel_state_dict(dst, sd, verbose=False)
    assert report == {}
    assert set(out) == set(sd)


def test_migrate_refuses_when_the_model_has_no_projection():
    T = _trainer()
    src = _tiny_egnn(8)
    legacy = {
        ("vel_field.time_embed_proj." + k.split("vel_field.t_to_h.")[1]): v
        for k, v in src.state_dict().items()
        if k.startswith("vel_field.t_to_h.")
    }
    plain = T.CFMJointModel(hidden_dim=16, n_layers=1, use_egnn_velocity=True)
    with pytest.raises(RuntimeError, match="no vel_field.t_to_h"):
        T.migrate_legacy_vel_state_dict(plain, legacy, verbose=False)


def test_pocket_encoder_refuses_a_partial_velocity_build():
    from molmetal.eval._pocket_encoder import try_build_full_cfm_model

    T = _trainer()
    model = T.CFMJointModel(hidden_dim=16, n_layers=2)
    state = model.state_dict()
    state["vel_field.time_embed_proj.weight"] = torch.zeros(16, 8)

    built, out = try_build_full_cfm_model(state)
    assert built is False
    assert out is None
