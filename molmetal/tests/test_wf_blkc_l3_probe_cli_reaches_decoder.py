# SPDX-License-Identifier: Apache-2.0
"""Run-time flags must reach the DECODER, not just the report.

THE DEFECT THIS PINS
--------------------
``wf_blkc_l3_guidance_probe.py::decode(model, clouds, args_dict)`` was called
with ``a`` -- the namespace stored INSIDE the checkpoint (``state["args"]``,
the training-time configuration) -- rather than the run-time one. Every flag
supplied on the command line or in the YAML was therefore invisible to the
decoder while still being recorded in the report's ``cohort`` block.

MEASURED, not inferred. The aromatic-bias sweep launched 2026-10-12 with
``--bond-aromatic-logit-bias 0.0`` and ``0.5`` produced byte-identical
SMILES on all 64 clouds: 0 of 64 rows differed. That was not a null result,
it was the sweep having run the same decode four times. The report said the
arms were run at 0.0 and 0.5; the decoder had been handed no offset at all.

This is the same shape as the six other failures in
``docs/CFM_STATUS_2026_10_12.md``: the harness reports success because the
path that would have failed was never taken.

TWO NAMESPACES, TWO QUESTIONS
-----------------------------
``a``     -- how the checkpoint was TRAINED (architecture, vocab table, cutoff)
``cli``   -- how it is being RUN NOW (offset, guidance strength, arm label)

Architecture must come from ``a``; the operating point must come from
``cli``. Run-time wins where they overlap, since that is what a flag is for.

RED CHECK: restore the bare ``decode(model, sampled, a)`` call and
``test_run_time_bias_reaches_the_decoder`` goes red.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PROBE = REPO / "molmetal" / "scripts" / "wf_blkc_l3_guidance_probe.py"
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _probe():
    if "_blkc_l3_probe_test" in sys.modules:
        return sys.modules["_blkc_l3_probe_test"]
    spec = importlib.util.spec_from_file_location("_blkc_l3_probe_test", str(PROBE))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_blkc_l3_probe_test"] = m
    spec.loader.exec_module(m)
    return m


def _trainer():
    if "_blkc_l3_trainer_test" in sys.modules:
        return sys.modules["_blkc_l3_trainer_test"]
    spec = importlib.util.spec_from_file_location("_blkc_l3_trainer_test", str(TRAINER))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_blkc_l3_trainer_test"] = m
    spec.loader.exec_module(m)
    return m


def _main_source() -> str:
    return inspect.getsource(_probe().main)


def test_decode_is_not_called_with_the_checkpoint_namespace_alone():
    """The literal ``decode(model, sampled, a)`` is the defect."""
    src = _main_source()
    assert "decode(model, sampled, a)" not in src, (
        "the decoder is being handed the CHECKPOINT's args namespace, so every "
        "run-time flag is invisible to it while still being recorded in the "
        "report. The 2026-10-12 bias sweep produced byte-identical SMILES on "
        "all 64 clouds at bias 0.0 and 0.5 because of exactly this."
    )


def test_run_args_merge_exists_and_carries_the_bias():
    src = _main_source()
    assert "run_args" in src, "no merge of run-time flags into the decode args"
    assert "bond_aromatic_logit_bias" in src, (
        "the aromatic offset must be in the list of keys copied from the "
        "run-time namespace; it is the flag the whole sweep turns on"
    )


def test_architecture_still_comes_from_the_checkpoint():
    """Merging must not replace architecture with run-time defaults."""
    src = _main_source()
    assert "dict(a)" in src, (
        "the merge must start from the checkpoint args so hidden_dim, "
        "n_layers, cutoff and the vocab table still describe how the "
        "checkpoint was trained"
    )


def test_helper_turns_a_non_zero_bias_into_a_vector():
    """The end-to-end intent: a run-time 0.5 becomes a real offset."""
    vec = _trainer()._aromatic_bias_vector(
        type("A", (), {"bond_aromatic_logit_bias": 0.5})()
    )
    assert vec == [0.0, 0.0, 0.0, 0.0, 0.5], (
        f"a run-time bias of 0.5 must reach the decoder as an aromatic-class "
        f"offset; got {vec}"
    )


def test_checkpoint_namespace_alone_produces_no_offset():
    """This is the exact condition that made the first sweep a no-op."""
    ckpt_like = type("A", (), {})()  # no bond_aromatic_logit_bias attribute
    assert _trainer()._aromatic_bias_vector(ckpt_like) is None, (
        "a namespace without the flag must yield None (no offset) -- this is "
        "what the checkpoint args gave, and why both arms matched"
    )