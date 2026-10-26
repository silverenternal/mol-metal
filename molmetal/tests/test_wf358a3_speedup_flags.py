"""Wf-358-A3 — the three speedup flags are gone from the shipped trainer.

=====================================================================
Wf-358-A3 (2026-09-30) added three opt-in speedup switches to
``train_dmpnn_multitask_wasserstein.py``:

  - ``--gradient-checkpointing`` — wraps the per-mol encoder forward in
    :func:`torch.utils.checkpoint.checkpoint` (~O(sqrt(depth)) memory at
    ~30% more FLOPs).
  - ``--channels-last`` — NHWC memory format on CUDA, small throughput
    lift on the per-metal head Linear ops.
  - ``--torch-compile-dynamic`` — sets
    ``torch._dynamo.config.dynamic_shapes=True`` before any
    ``torch.compile`` call.

MEASURED 2026-10-27: none of the three exists in the trainer any more.
``af84a545`` ("m21 selective Wf-A17 revert") restored the file from a
snapshot that predates them, to undo a model-performance regression
(A2780cis 0.7278 → 0.3356, A2780 0.6931 → 0.5104 in the same commit).
The revert was about the model, not about these flags; they went with
the file.  ``DEFAULT_CHANNELS_LAST`` and ``DEFAULT_TORCH_COMPILE_DYNAMIC``
are gone too.

This file used to assert all three parsed, applied, and defaulted off,
which required importing constants that no longer exist.  Asserting the
shipped surface now means asserting their absence: if a future wave
re-ports them, these tests fail, and whoever re-ports them has to say so
and re-validate on GPU (they are GPU-throughput features; a CPU test can
only prove the wiring, not the speedup).

Per AGENTS.md §5.2: this is a record of a measured gap, not a plan to
close it.  Restoring the flags means re-porting code that the m21 revert
removed for a measured AUC reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

TRAINER_PATH = (
    _PROJECT_ROOT
    / "molmetal"
    / "scripts"
    / "train_dmpnn_multitask_wasserstein.py"
)


@pytest.fixture(scope="module")
def trainer_source() -> str:
    assert TRAINER_PATH.is_file(), f"trainer missing: {TRAINER_PATH}"
    return TRAINER_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "symbol",
    [
        "--gradient-checkpointing",
        "--torch-compile-dynamic",
        "DEFAULT_CHANNELS_LAST",
        "DEFAULT_TORCH_COMPILE_DYNAMIC",
    ],
)
def test_speedup_symbol_is_absent(trainer_source: str, symbol: str) -> None:
    """The symbol is absent — the m21 revert removed it.

    Absence is the shipped state, not an aspiration.  If this fails, the
    symbol came back: move it to a positive test and record the GPU
    revalidation that the re-port needs.

    ``--channels-last`` is deliberately not in this list: MEASURED
    2026-10-27, it survived the revert (argparse at
    ``train_dmpnn_multitask_wasserstein.py:3894``, threaded as the
    ``channels_last`` kwarg).  Only the ``DEFAULT_CHANNELS_LAST`` constant
    is gone.
    """
    assert symbol not in trainer_source, (
        f"{symbol!r} is back in train_dmpnn_multitask_wasserstein.py. "
        f"The m21 revert (af84a545) removed it along with the rest of the "
        f"Wf-358-A3 speedup surface. Re-adding it is a capability "
        f"restoration: assert it positively here and validate the speedup "
        f"on GPU, rather than letting this file silently pass."
    )


def test_channels_last_survived_and_stays_off_by_default() -> None:
    """``--channels-last`` still parses, defaults off, and is threaded.

    The flag survived the revert even though the ``DEFAULT_CHANNELS_LAST``
    constant did not.  This is the one positive half of Wf-358-A3 that is
    still on the shipped surface.
    """
    from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args

    args = parse_args(["--metals", "Ru", "--seeds", "0", "--epochs", "1"])
    assert args.channels_last is False, (
        "channels_last must default OFF; a D-MPNN has no spatial axes, so "
        "ON is a no-op that only risks a memory-format copy"
    )
    args_on = parse_args(
        ["--metals", "Ru", "--seeds", "0", "--epochs", "1", "--channels-last"]
    )
    assert args_on.channels_last is True, "--channels-last did not parse"


def test_compile_path_still_guards_dynamic_shapes() -> None:
    """``dynamic_shapes`` is set before every compile, as Wf-358-A3 required.

    That part of the wave survived the revert — not as a flag, but as an
    unconditional guard inside the ``--torch-compile`` branch, which is the
    only compile path left.
    """
    src = TRAINER_PATH.read_text(encoding="utf-8")
    compile_idx = src.find(
        "model = torch.compile(model, mode=DEFAULT_TORCH_COMPILE_MODE)"
    )
    guard_idx = src.find("torch._dynamo.config.dynamic_shapes = True")
    assert compile_idx > 0, "the compile call site moved; update this test"
    assert 0 < guard_idx < compile_idx, (
        "dynamic_shapes must be set BEFORE torch.compile, otherwise the "
        "compiled artifact rejects variable batch sizes (the last batch of "
        "an epoch is short whenever n_train % batch_size != 0)"
    )
