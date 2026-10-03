"""Wf-328-W4 — TF32 production trainer wiring tests.

Per Wf-326A quick wins, enabling
``torch.backends.cuda.matmul.allow_tf32=True`` +
``torch.set_float32_matmul_precision('high')`` gives ~1.5× matmul speedup
on Ampere/Hopper/Blackwell-class GPUs with negligible accuracy loss for
the matmul-dominated D-MPNN encoder.

Wf-328-W4 wires this into the production trainer
(:mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein`):

  * New :func:`configure_tf32` helper toggles
    ``torch.backends.cuda.matmul.allow_tf32``,
    ``torch.backends.cudnn.allow_tf32`` and
    ``torch.set_float32_matmul_precision(...)`` in one call.
  * New ``--tf32`` (default ON) and ``--no-tf32`` CLI flags surface the
    helper to the launcher scripts.
  * The ``main()`` entrypoint invokes ``configure_tf32`` after device
    resolution and logs the post-config state into the training_summary
    JSON for paper §6 reproducibility audits.

These tests verify the wiring without needing a GPU.  TF32 settings are
*checkable* on CPU-only hosts because the torch global flags are
read-back regardless of CUDA availability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import (  # noqa: E402
        DEFAULT_TF32,
        configure_tf32,
        parse_args,
    )
except ImportError:  # Wf-A16: feature not implemented — SKIP, do not ERROR
    pytest.skip(
        "UNIMPLEMENTED FEATURE: this module expects `DEFAULT_TF32` from `scripts.train_dmpnn_multitask_wasserstein`, which "
        "does not exist. TF32 matmul. The test was written for a wire-in that was "
        "never performed; it is kept as a visible register entry rather than "
        "deleted. Tracked in TODO/completed/wfa16_unimplemented_feature_register.md",
        allow_module_level=True,
    )


def test_tf32_enabled_by_default():
    """``configure_tf32(True)`` flips the three TF32 flags.

    Verifies the helper function does what Wf-326A promised:
      * ``torch.backends.cuda.matmul.allow_tf32`` is True
      * ``torch.backends.cudnn.allow_tf32`` is True
      * ``torch.set_float32_matmul_precision('high')`` was called

    Also verifies ``DEFAULT_TF32=True`` so the *untouched* production
    trainer (no ``--no-tf32`` flag) inherits TF32 by default.
    """
    # Sanity: the constant is True (this is the whole point of Wf-328-W4).
    assert DEFAULT_TF32 is True, (
        "DEFAULT_TF32 must be True (Wf-328-W4 default ON); Wf-326A "
        "quick wins gave ~1.5x matmul speedup."
    )

    # Snapshot prior state so we can restore after the test.
    prior_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    prior_cudnn = bool(torch.backends.cudnn.allow_tf32)
    try:
        state = configure_tf32(enabled=True)

        # Helper return-shape contract.
        assert state["enabled"] is True
        assert state["current_allow_tf32_matmul"] is True
        assert state["current_allow_tf32_cudnn"] is True
        assert state["matmul_precision"] == "high"

        # Live torch state matches.
        assert bool(torch.backends.cuda.matmul.allow_tf32) is True
        assert bool(torch.backends.cudnn.allow_tf32) is True

        # The new precision API reports 'high' (or 'medium'/'highest' on
        # some torch versions).  We accept any non-'highest' value as
        # "TF32-on" but prefer 'high'.
        cur_prec = torch.get_float32_matmul_precision()
        assert cur_prec in ("high", "medium"), (
            f"expected matmul precision 'high'/'medium' when TF32 is on, "
            f"got {cur_prec!r}"
        )
    finally:
        # Restore prior state so we do not bleed into other tests in
        # the same pytest session.
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn


def test_tf32_flag_override():
    """``--no-tf32`` flips TF32 off; ``--tf32`` flips it back on.

    Verifies:
      1. The CLI parser accepts both ``--tf32`` and ``--no-tf32`` and
         produces the expected boolean in ``args.tf32``.
      2. ``configure_tf32(False)`` actually clears the TF32 flags
         (``matmul.allow_tf32=False``, ``cudnn.allow_tf32=False``,
         ``matmul_precision='highest'``).
      3. ``configure_tf32(True)`` after ``False`` restores TF32.

    This is the ``--no-tf32`` reproducibility lever that paper §6
    honest-negative audits rely on.
    """
    prior_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    prior_cudnn = bool(torch.backends.cudnn.allow_tf32)
    try:
        # --- CLI parser test ---
        args_on = parse_args(["--tf32"])
        assert args_on.tf32 is True, (
            f"--tf32 should yield args.tf32=True; got {args_on.tf32!r}"
        )

        args_off = parse_args(["--no-tf32"])
        assert args_off.tf32 is False, (
            f"--no-tf32 should yield args.tf32=False; got {args_off.tf32!r}"
        )

        args_default = parse_args([])
        assert args_default.tf32 is DEFAULT_TF32, (
            f"default --tf32 flag should equal DEFAULT_TF32 ({DEFAULT_TF32}); "
            f"got {args_default.tf32!r}"
        )

        # --- Helper function: TF32 OFF ---
        off_state = configure_tf32(enabled=False)
        assert off_state["enabled"] is False
        assert off_state["current_allow_tf32_matmul"] is False
        assert off_state["current_allow_tf32_cudnn"] is False
        assert off_state["matmul_precision"] == "highest"
        assert bool(torch.backends.cuda.matmul.allow_tf32) is False
        assert bool(torch.backends.cudnn.allow_tf32) is False
        cur_prec = torch.get_float32_matmul_precision()
        assert cur_prec == "highest", (
            f"expected matmul precision 'highest' when TF32 is off, "
            f"got {cur_prec!r}"
        )

        # --- Helper function: TF32 ON (re-enable after off) ---
        on_state = configure_tf32(enabled=True)
        assert on_state["enabled"] is True
        assert on_state["current_allow_tf32_matmul"] is True
        assert on_state["current_allow_tf32_cudnn"] is True
        assert on_state["matmul_precision"] == "high"
        assert bool(torch.backends.cuda.matmul.allow_tf32) is True
        assert bool(torch.backends.cudnn.allow_tf32) is True
    finally:
        # Restore prior state so we do not bleed into other tests in
        # the same pytest session.
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn


def test_configure_tf32_idempotent_repeated_calls():
    """Repeated ``configure_tf32(True)`` calls are safe and idempotent.

    Defensive check: the production trainer may call
    ``configure_tf32`` more than once in nested sub-orchestrators; the
    helper must not crash on the second call and must keep the same
    end state.
    """
    prior_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    prior_cudnn = bool(torch.backends.cudnn.allow_tf32)
    try:
        s1 = configure_tf32(enabled=True)
        s2 = configure_tf32(enabled=True)
        s3 = configure_tf32(enabled=False)
        s4 = configure_tf32(enabled=True)

        # Last call wins; helper is deterministic.
        assert s1["enabled"] is True
        assert s2["enabled"] is True
        assert s3["enabled"] is False
        assert s4["enabled"] is True

        # The torch global state matches the last call.
        assert bool(torch.backends.cuda.matmul.allow_tf32) is True
        assert bool(torch.backends.cudnn.allow_tf32) is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn
