# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 / T52 — the pair head's +2.0 NO_BOND bias was never a prior.

THE DEFECT
----------
Every YuelBond pair head has initialised its output layer as

    pair_head[-1].bias.zero_()
    pair_head[-1].bias[BOND_NO_BOND] = +2.0

The intent, stated in the source, is that the head starts at the class prior.
But only the BIAS was set.  ``nn.Linear``'s default init leaves the WEIGHT at
U(-1/sqrt(H), +1/sqrt(H)), so the logit is

    logit = W h + b,   |W h| ~ |h| / sqrt(H) * O(1)

and on the 5-metal cohort |h| is large enough that |W h| reaches 32 — the
hand-set +2.0 is a 6% perturbation of a random number.  MEASURED at init on
8 clouds (seed 0, metalcytotox_5metal_multitask_v3.csv): CE 16.0048,
|logit|max 32.64, p(NO_BOND) 0.037.  With the weight zeroed and the bias kept:
CE 0.5645, |logit|max 2.00, p(NO_BOND) 0.649.

This is what T42's kill clause found.  T42 set out to prove the epoch-0
bond_ce transient (15.9 -> 1.6) was the class-weight denominator; the
unweighted CE turned out to be just as large (16.60 vs 13.51 at ep0), so the
denominator was exonerated and the kill clause pointed here.

WHAT THIS FILE CHECKS
---------------------
1. The head honours ``zero_init_pair_out=True``: output weight all zeros,
   bias untouched.
2. The default is False, so every direct constructor call and every archived
   checkpoint stays bit-exact.
3. The trainer flag exists, defaults ON, and its dest is what
   ``build_joint_model`` reads.
4. ``build_joint_model`` forwards the flag to the head, end to end, for all
   three YuelBond heads.
5. The measured init statistics are on disk and say what the code comments
   say they say.

WHAT THIS FILE DOES NOT CLAIM
-----------------------------
It does not claim the zeroed init trains better.  That needs a run, and the
standing directive forbids one.  It claims only that the head now starts where
its own source says it starts.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
ARTIFACT = REPO / "metrics" / "by_metric" / "wfa113_t52_bond_head_init.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dec():
    return _load("_t52_decoder", REPO / "molmetal" / "molmetal_lam" / "lam_chem" / "yuelbond_decoder.py")


@pytest.fixture(scope="module")
def trainer():
    sys.path.insert(0, str(TRAINER.parent))
    return _load("_t52_trainer", TRAINER)


def _head(dec, cls_name: str, **kw):
    cls = getattr(dec, cls_name)
    torch.manual_seed(0)
    if "EdgeMPNN" in cls_name:
        return cls(hidden_dim=32, n_edge_layers=2, **kw)
    return cls(hidden_dim=32, n_layers=2, **kw)


# ---------------------------------------------------------------- the head


@pytest.mark.parametrize(
    "cls_name",
    ["YuelBondDecoderHead", "YuelBondEdgeMPNNHead", "YuelBondEdgeMPNNHeadResidual"],
)
def test_zero_init_pair_out_zeroes_only_the_weight(dec, cls_name):
    head = _head(dec, cls_name, zero_init_pair_out=True)
    out = head.pair_head[-1]
    assert float(out.weight.detach().abs().sum()) == 0.0
    # The bias is the whole point of the flag: it must survive.
    assert float(out.bias.detach()[dec.BOND_NO_BOND]) == 2.0
    assert float(out.bias.detach().abs().sum()) == 2.0


@pytest.mark.parametrize(
    "cls_name",
    ["YuelBondDecoderHead", "YuelBondEdgeMPNNHead", "YuelBondEdgeMPNNHeadResidual"],
)
def test_default_leaves_the_weight_at_linear_init(dec, cls_name):
    """Default False => archived checkpoints and direct callers unchanged."""
    head = _head(dec, cls_name)
    out = head.pair_head[-1]
    assert float(out.weight.detach().abs().sum()) > 0.0


def test_zero_init_makes_the_bias_the_prior(dec):
    """With W = 0 the logits are the bias, so the head predicts the prior."""
    head = _head(dec, "YuelBondDecoderHead", zero_init_pair_out=True)
    out = head.pair_head[-1]
    p = torch.softmax(out.bias.detach(), dim=0)
    # softmax([2,0,0,0,0]) = 7.389 / (7.389 + 4)
    assert p[dec.BOND_NO_BOND] == pytest.approx(0.6489, abs=5e-4)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------- the trainer


def test_trainer_flag_defaults_on(trainer, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["probe"])
    assert trainer.parse_args().bond_head_zero_out is True


def test_trainer_flag_can_be_turned_off(trainer, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["probe", "--no-bond-head-zero-out"])
    assert trainer.parse_args().bond_head_zero_out is False


def test_build_joint_model_reads_the_flag():
    src = TRAINER.read_text()
    assert 'bond_head_zero_out=bool(getattr(args, "bond_head_zero_out", True))' in src


def test_build_joint_model_forwards_to_every_yuelbond_head():
    src = TRAINER.read_text()
    # One forward per YuelBond head construction.  The dmpnn branch has no
    # such bias, so exactly three call sites carry the kwarg.
    assert src.count("zero_init_pair_out=bool(bond_head_zero_out)") == 3


def test_launch_banner_states_the_resolved_value():
    src = TRAINER.read_text()
    assert "pair_head_out=" in src
    assert "DEFAULT-INIT (bias swamped by weight)" in src


# ------------------------------------------------------------- the artifact


def test_artifact_is_on_disk_and_says_what_the_comments_say():
    assert ARTIFACT.exists(), f"missing artifact {ARTIFACT}"
    d = json.loads(ARTIFACT.read_text())
    on, off = d["rows"]
    assert on["bond_head_zero_out_flag"] is True
    assert off["bond_head_zero_out_flag"] is False
    assert on["pair_head_out_weight_abs_sum"] == 0.0
    assert off["pair_head_out_weight_abs_sum"] > 0.0
    # The numbers quoted in the decoder comment and the --help text.
    assert off["bond_ce"] == pytest.approx(16.0048, abs=1e-3)
    assert on["bond_ce"] == pytest.approx(0.5645, abs=1e-3)
    assert off["logit_abs_max"] == pytest.approx(32.64, abs=1e-2)
    assert on["logit_abs_max"] == pytest.approx(2.00, abs=1e-2)
    # Same cohort both times, so the comparison is paired.
    assert on["n_pairs"] == off["n_pairs"]
    assert on["n_clouds"] == off["n_clouds"]
