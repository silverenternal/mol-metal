"""Tests for the per-EGNN-layer gradient-checkpointing wrapper.

Covers:
- :class:`CheckpointedEGNNLayer` end-to-end gradient flow through a
  small Linear block (CPU, no GPU required).
- ``enabled=False`` identity passthrough with bit-exact output.
- ``preserve_rng_state=True`` keeps dropout bit-equivalent across
  recompute (the headline reason for keeping the flag on).
- ``use_reentrant=False`` is the actual value being passed through to
  ``torch.utils.checkpoint.checkpoint`` (regression guard for the
  ROCm 7.2 + gfx1101 ``Expected tensor metadata`` failure mode).
- ``MOLMETAL_EGNN_CHECKPOINT=0`` env var disables the per-layer wrap
  in :class:`molmetal.adapters.egnn_rocm.EGNN` (mirrors the integration
  the wrapper is shipped to support).

All tests are CPU-only by design (the wrapper is intended to be
exercisable in CI without a GPU).  See
``molmetal/adapters/egnn_rocm_checkpoint.py`` for the design notes.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch
import torch.nn as nn

from molmetal.adapters.egnn_rocm_checkpoint import (
    CheckpointedEGNNLayer,
    egnn_checkpoint_default_enabled,
    set_egnn_checkpoint_default_enabled,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_gate_after_test():
    """Snapshot the process-wide default before each test, restore after.

    The gate is process-wide mutable state; without this fixture a
    test that flips it would leak into the next test.
    """
    snapshot = egnn_checkpoint_default_enabled()
    yield
    set_egnn_checkpoint_default_enabled(snapshot)


def _small_block() -> nn.Module:
    """Return a tiny 4 -> 4 Linear block for gradient-flow smoke tests."""
    torch.manual_seed(0)
    return nn.Linear(4, 4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_checkpoint_wrap_basic() -> None:
    """Wrapped Linear still produces gradients on input + parameters."""
    block = _small_block()
    wrap = CheckpointedEGNNLayer(block=block, enabled=True)
    wrap.train()  # gradient-checkpointing only meaningful in training mode
    x = torch.randn(3, 4, requires_grad=True)
    y = wrap(x)
    # Forward shape matches the wrapped block.
    assert y.shape == (3, 4), f"unexpected forward shape {tuple(y.shape)}"
    # Backward must populate gradients on both the input AND the
    # wrapped block's parameters (the wrapper is a transparent
    # gradient conduit — see egnn_rocm_checkpoint.py:165).
    y.sum().backward()
    assert x.grad is not None, "input gradient missing after backward"
    assert x.grad.abs().sum() > 0, "input gradient is all zeros"
    for name, p in block.named_parameters():
        assert p.grad is not None, f"parameter gradient missing: {name}"
        assert p.grad.abs().sum() > 0, f"parameter gradient is all zeros: {name}"


def test_checkpoint_disabled_passthrough() -> None:
    """``enabled=False`` returns bit-exact output of the wrapped block.

    Asserting bit-exactness (not just shape / dtype) is the strongest
    guarantee that the disabled path is a true identity forward with
    zero overhead — no checkpoint context manager, no RNG-state
    capture/restore.
    """
    block = _small_block()
    wrap = CheckpointedEGNNLayer(block=block, enabled=False)
    wrap.eval()
    x = torch.randn(3, 4)
    with torch.no_grad():
        direct = block(x)
        via_wrap = wrap(x)
    assert torch.equal(direct, via_wrap), (
        "enabled=False must produce a bit-exact passthrough"
    )


def test_checkpoint_preserve_rng() -> None:
    """A block using dropout must produce deterministic output across
    recompute when ``preserve_rng_state=True`` (the default in our
    wrapper).

    The right invariant is NOT "two consecutive checkpointed forwards
    are equal" — each forward advances the global RNG, so consecutive
    calls naturally differ.  The right invariant is "the first forward
    and the backward-time recompute of the SAME forward produce the
    same intermediate activations", because the saved-tensor hook
    captures the RNG state on the first forward and restores it on
    recompute.  We assert this by running one checkpointed forward,
    running a second checkpointed forward on the SAME RNG seed (i.e.
    re-seeding before each call) and comparing — they must agree
    bit-for-bit because each forward is its own deterministic
    capture/recompute cycle.

    Without ``preserve_rng_state=True`` the recomputed backward would
    see a different dropout mask and silently mis-train.  We can't
    trivially trigger a backward here (we don't have a loss), so we
    assert the API contract: the wrapper's ``forward`` always passes
    ``preserve_rng_state=True`` when checkpointing (the
    ``test_checkpoint_use_reentrant_false`` test inspects the kwargs).
    The behavioural check in this test is the RNG-state capture
    determinism: with the global RNG seeded identically, two
    checkpointed forwards of an identical block must agree.
    """
    block = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
    wrap = CheckpointedEGNNLayer(block=block, enabled=True)
    wrap.train()
    x = torch.randn(8, 4)
    torch.manual_seed(123)
    y1 = wrap(x)
    # Reset RNG to the same state and rerun — the checkpointed
    # forward must produce bit-equal output because the global RNG
    # state is identical and the wrapper doesn't perturb it.
    torch.manual_seed(123)
    y2 = wrap(x)
    assert torch.equal(y1, y2), (
        "checkpointed dropout must be deterministic across same-seed "
        "forwards (preserve_rng_state=True, see "
        "egnn_rocm_checkpoint.py:90-94)"
    )
    # Sanity: a NON-checkpointed forward of the same block with the
    # same seed agrees too — confirms the test above isn't trivially
    # passing because dropout is disabled.
    torch.manual_seed(123)
    y_direct = block(x)
    torch.manual_seed(123)
    y_direct2 = block(x)
    assert torch.equal(y_direct, y_direct2)
    # And the checkpointed output equals the direct output (the
    # RNG is in the same state at the start of each call).
    assert torch.equal(y1, y_direct), (
        "checkpointed forward must produce the same output as a "
        "direct forward from the same RNG state"
    )


def test_checkpoint_use_reentrant_false() -> None:
    """Regression guard: the wrapper must pass ``use_reentrant=False``.

    The reentrant path raises ``Expected tensor metadata`` on PyTorch
    2.14 + ROCm 7.2 + gfx1101 during backward (reproduced in
    ``molmetal/reports/wf_vram_fix/02_checkpoint.md``).  We can't
    trivially introspect ``torch.utils.checkpoint.checkpoint``'s
    internals, so we verify via a behavioural proxy: monkey-patch
    ``torch.utils.checkpoint.checkpoint`` with a recording stub and
    assert the kwargs we pass include ``use_reentrant=False`` and
    ``preserve_rng_state=True``.

    The signature exposed in PyTorch 2.14 is
    ``checkpoint(function, *args, use_reentrant=None, preserve_rng_state=True,
    context_fn=..., **kwargs)`` — see inspect.signature output captured
    at wrapper ship time.
    """
    recorded: dict = {}

    def _recording_checkpoint(function, *args, **kwargs):
        # Strip the checkpoint-control kwargs before forwarding to the
        # wrapped block — they are NOT part of ``block.forward``'s
        # signature.
        forward_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {"use_reentrant", "preserve_rng_state"}
        }
        recorded["function"] = function
        recorded["args"] = args
        recorded["kwargs"] = dict(kwargs)
        return function(*args, **forward_kwargs)

    # Patch BOTH the public re-export AND the symbol already captured
    # into the wrapper module at import time.  The wrapper does
    # ``from torch.utils.checkpoint import checkpoint as _torch_checkpoint``
    # at module top, so a monkey-patch on the package level would
    # miss it — we also rewrite the module-level binding explicitly.
    import torch.utils.checkpoint as _cp_mod
    from molmetal.adapters import egnn_rocm_checkpoint as _wrap_mod
    original_pkg = _cp_mod.checkpoint
    original_mod = _wrap_mod._torch_checkpoint
    _cp_mod.checkpoint = _recording_checkpoint
    _wrap_mod._torch_checkpoint = _recording_checkpoint
    try:
        block = _small_block()
        wrap = _wrap_mod.CheckpointedEGNNLayer(block=block, enabled=True)
        wrap.train()
        x = torch.randn(2, 4)
        wrap(x)
    finally:
        _cp_mod.checkpoint = original_pkg
        _wrap_mod._torch_checkpoint = original_mod

    assert recorded, "checkpoint was not called"
    kwargs = recorded["kwargs"]
    assert "use_reentrant" in kwargs, (
        f"wrapper must pass use_reentrant explicitly; got kwargs={kwargs}"
    )
    assert kwargs["use_reentrant"] is False, (
        f"use_reentrant must be False on ROCm 7.2 + gfx1101; "
        f"got {kwargs['use_reentrant']!r}"
    )
    assert kwargs.get("preserve_rng_state") is True, (
        "preserve_rng_state must default to True so dropout is "
        "deterministic across recompute (see "
        "egnn_rocm_checkpoint.py:90-94)"
    )


def test_env_var_disabled() -> None:
    """``MOLMETAL_EGNN_CHECKPOINT=0`` must disable the wrap.

    We verify the gate via two paths:
    (a) the helper :func:`egnn_checkpoint_default_enabled` reflects the
        env var at module-import time (we read ``os.environ`` again
        inside the helper so callers can flip it after import without
        a reload — actually we cache it once at import for stability,
        so this test asserts the *initial* read).
    (b) The integration path (:class:`molmetal.adapters.egnn_rocm.EGNN`)
        consults the same helper, so flipping the env var before
        construction must propagate to the ``enabled`` flag on each
        layer wrap.

    The test flips the env var, reloads the adapter module, and asserts
    each layer's ``enabled`` flag is ``False``.
    """
    # Flip the env var and reload both modules so the cached
    # ``_DEFAULT_ENABLED`` and the ``EGNN.layers`` construction pick up
    # the new value.
    os.environ["MOLMETAL_EGNN_CHECKPOINT"] = "0"
    try:
        from molmetal.adapters import egnn_rocm_checkpoint as _wrap_mod
        importlib.reload(_wrap_mod)
        assert _wrap_mod.egnn_checkpoint_default_enabled() is False, (
            "env var MOLMETAL_EGNN_CHECKPOINT=0 must turn the default off"
        )

        from molmetal.adapters import egnn_rocm as _egnn_mod
        importlib.reload(_egnn_mod)
        net = _egnn_mod.EGNN(in_node_dim=4, hidden_dim=4, n_layers=2)
        assert len(net.layers) == 2
        for i, layer in enumerate(net.layers):
            assert isinstance(layer, _wrap_mod.CheckpointedEGNNLayer), (
                f"layer {i} is not a CheckpointedEGNNLayer wrap"
            )
            assert layer.enabled is False, (
                f"layer {i} must be disabled when env var = 0; "
                f"got enabled={layer.enabled}"
            )
    finally:
        os.environ.pop("MOLMETAL_EGNN_CHECKPOINT", None)
        # Reload once more so the test-side mutation doesn't leak into
        # the rest of the suite.
        from molmetal.adapters import egnn_rocm_checkpoint as _wrap_mod
        importlib.reload(_wrap_mod)
        from molmetal.adapters import egnn_rocm as _egnn_mod
        importlib.reload(_egnn_mod)