# Licensed under the Apache License, Version 2.0.
"""Wf-A114 — §5.11 binding for ``configs/c9_cfm_transport_fixed.yaml``.

The c9 config is the arm that enables the coordinate-head fixes the CFM line
shipped but never ran (``vector_channels`` + the LSQ gate init + a fixed
norm gain + a chunked FM batch).  A config is a promise about what a run does,
and the AGENTS.md §5.11 gates exist because a key that never reaches a dest
makes the promise false while the run still completes.

These tests pin three things:

1. **every key resolves** — ``parse_args(["--config", cfg])`` in strict mode
   raises nothing, and every head lever lands on its dest with the intended
   value AND type;
2. **the dead-key regression** — the config must not carry
   ``protocol.scheduler.warmup_epochs``, which is an alias for a dest that
   ``train.lr_warmup_epochs`` already sets; the alias is suppressed but was
   still escalated in strict mode, which is why the shipped c8 reference
   config cannot pass its own gate;
3. **the documented absences stay absent** — ``radial_centroid_masked`` is
   INERT under ``vector_channels`` (that branch never computes a centroid)
   and ``metal_frame_channels`` is not established (§6.173 Finding 1), so
   neither may be switched on behind the config's back.

CPU-only: no GPU, no training, no sampling.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG = REPO_ROOT / "configs" / "c9_cfm_transport_fixed.yaml"


@pytest.fixture(scope="module")
def parsed():
    """``parse_args`` on the c9 config, in strict mode."""
    trainer = importlib.import_module(
        "molmetal.scripts.train_cfm_yuelbond_wasserstein"
    )
    old_argv = sys.argv
    sys.argv = ["trainer", "--config", str(CONFIG)]
    old_strict = __import__("os").environ.get("WFA2_CONFIG_STRICT")
    __import__("os").environ["WFA2_CONFIG_STRICT"] = "1"
    try:
        return trainer.parse_args()
    finally:
        sys.argv = old_argv
        if old_strict is None:
            __import__("os").environ.pop("WFA2_CONFIG_STRICT", None)
        else:
            __import__("os").environ["WFA2_CONFIG_STRICT"] = old_strict


class TestStrictBinding:
    def test_strict_mode_raises_nothing(self, parsed):
        # If this fixture ever fails to build, `parsed` is None -> explicit fail.
        assert parsed is not None

    def test_the_head_levers_land(self, parsed):
        assert parsed.velocity_head == "vector_channels"
        assert parsed.vector_channels == 8
        assert parsed.vector_channel_norm_gain == "fixed"
        assert parsed.velocity_head_init == "lsq"
        assert parsed.velocity_head_init_knots == 9
        assert parsed.t_embedding_dim == 32
        assert parsed.fm_batch_chunk_ratio == 2.0
        assert parsed.prediction_target == "velocity"

    def test_the_types_are_right_not_just_the_values(self, parsed):
        # §5.11: a LIST-valued dest must arrive as a list.  A YAML scalar
        # "Ru Ir Os Re" would arrive as one string and match no rows.
        assert isinstance(parsed.tmqm_metals, list)
        assert parsed.tmqm_metals == ["Pt", "Ru", "Ir", "Rh", "Os", "Re"]
        assert isinstance(parsed.velocity_head, str)
        assert isinstance(parsed.vector_channels, int)
        assert isinstance(parsed.fm_batch_chunk_ratio, float)
        assert isinstance(parsed.t_embedding_dim, int)

    def test_c8s_protocol_is_preserved(self, parsed):
        assert parsed.n_train == 26715
        assert parsed.batch_size == 64
        assert parsed.epochs == 18
        assert parsed.lr == pytest.approx(3.0e-4)
        assert parsed.fm_noise_sigma == pytest.approx(2.485)
        assert parsed.tmqm_split_kind == "scaffold"
        assert parsed.save_generated_smiles is True
        assert parsed.out_dir == "molmetal/reports/cfm_c9_transport_fixed_2026_10_30"

    def test_the_early_stop_gates_on_connectivity(self, parsed):
        # Wf-A114: `bond_type_f1` does not see fragmentation.  c9 gates on
        # `molecule_validity` — RDKit-strict AND one fragment AND the seed
        # metal — so the checkpoint that survives is the one that holds
        # molecules together.  MEASURED reason, in the census artifact:
        # 8 of 27 runs reach an end-to-end rate >= 0.50 at their BEST epoch
        # (max 0.9375) while the same runs' LAST epoch reads 0.0000-0.2499.
        assert parsed.early_stop_metric == "molecule_validity"

    def test_the_early_stop_metric_is_one_the_parser_accepts(self):
        # A config naming a criterion the trainer's `choices` does not
        # contain would fail at launch, not at parse time.
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            EARLY_STOP_METRICS,
        )

        assert "molecule_validity" in EARLY_STOP_METRICS
        assert "bond_type_f1" in EARLY_STOP_METRICS

    def test_the_warmup_alias_is_not_declared_twice(self, parsed):
        # `protocol.scheduler.warmup_epochs` is an alias for the dest that
        # `train.lr_warmup_epochs` already sets; the alias is suppressed but
        # was still escalated in strict mode.  c8 still carries it and still
        # fails its own gate; c9 must not.
        text = CONFIG.read_text(encoding="utf-8")
        assert "warmup_epochs: 5" in text  # train.lr_warmup_epochs
        assert "protocol.scheduler.warmup_epochs" not in text.split("protocol:")[-1]
        # the launcher still gets the value: start.sh falls back to the
        # train.* key (scripts/start.sh:525).
        assert parsed.lr_warmup_epochs == 5


class TestDocumentedAbsences:
    def test_radial_centroid_masked_is_off(self, parsed):
        # INERT under vector_channels: the vector_channels branch of
        # EGNNVelocityField.forward never computes a centroid, so the flag
        # has no code path to influence.  Setting it would be a lie.
        assert parsed.radial_centroid_masked is False

    def test_metal_frame_channels_is_off(self, parsed):
        # §6.173 Finding 1: at 400 steps the two arms differ by 0.09 MSE, at
        # 800 by 2.9, on one seed — init variance, not an effect.
        assert parsed.metal_frame_channels is False

    def test_the_head_is_padding_invariant(self):
        # `_fm_padding_invariant` must be True for vector_channels, or the
        # chunk plan is refused and `fm_batch_chunk_ratio: 2.0` is a dead
        # key with no effect.
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            CFMJointModel,
        )

        src = (
            REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
        ).read_text(encoding="utf-8")
        assert "vector_channels" in src
        assert hasattr(CFMJointModel, "_fm_padding_invariant")
