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
  * New ``--tf32`` (store_true, default OFF) CLI flag surfaces the
    helper to the launcher scripts.  **af84a545 removed the companion
    ``--no-tf32`` flag** (and reverted the ``--tf32`` default to OFF);
    only the helper + ``DEFAULT_TF32`` constant survived, so the tests
    below assert the shipped ``--tf32``-only surface.
  * The ``main()`` entrypoint invokes ``configure_tf32`` after device
    resolution and logs the post-config state into the training_summary
    JSON for paper §6 reproducibility audits.

These tests verify the wiring without needing a GPU.  TF32 settings are
*checkable* on CPU-only hosts because the torch global flags are
read-back regardless of CUDA availability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.train_dmpnn_multitask_wasserstein import (  # noqa: E402
    DEFAULT_TF32,
    configure_tf32,
    parse_args,
)


def test_tf32_enabled_by_default():
    """``configure_tf32(True)`` flips the three TF32 flags.

    Verifies the helper function does what Wf-326A promised:
      * ``torch.backends.cuda.matmul.allow_tf32`` is True
      * ``torch.backends.cudnn.allow_tf32`` is True
      * ``torch.set_float32_matmul_precision('high')`` was called

    Also verifies ``DEFAULT_TF32=True`` — the *helper constant* default
    (the CLI no longer inherits it: af84a545 shipped ``--tf32`` as a
    store_true default OFF; see ``test_tf32_flag_override``).
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
    """``--tf32`` parses ON; the removed ``--no-tf32`` is rejected.

    Re-scoped to the SHIPPED surface (af84a545 removed ``--no-tf32`` and
    shipped ``--tf32`` as a store_true default OFF).  Verifies:
      1. ``parse_args(["--tf32"]).tf32 is True``; ``parse_args([])`` is
         False — the shipped CLI default, which is *independent* of the
         ``DEFAULT_TF32`` helper constant (still True).  The removed
         ``--no-tf32`` raises ``SystemExit``.
      2. ``configure_tf32(False)`` clears the TF32 flags
         (``matmul.allow_tf32=False``, ``cudnn.allow_tf32=False``,
         ``matmul_precision='highest'``); ``configure_tf32(True)``
         restores them.  The helper is unchanged by the revert.

    Shipped-default-vs-constant mismatch: ``DEFAULT_TF32 is True`` (the
    helper's default argument) but the CLI's ``--tf32`` default is False.
    The pre-revert trainer bound the two together; af84a545 did not.
    """
    prior_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    prior_cudnn = bool(torch.backends.cudnn.allow_tf32)
    try:
        # --- CLI parser test ---
        args_on = parse_args(["--tf32"])
        assert (
            args_on.tf32 is True
        ), f"--tf32 should yield args.tf32=True; got {args_on.tf32!r}"

        # Shipped store_true default is OFF (NOT DEFAULT_TF32).
        args_default = parse_args([])
        assert args_default.tf32 is False, (
            f"shipped --tf32 default should be False (store_true); "
            f"got {args_default.tf32!r}"
        )

        # --no-tf32 was removed by af84a545 → argparse rejects it.
        with pytest.raises(SystemExit):
            parse_args(["--no-tf32"])

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
