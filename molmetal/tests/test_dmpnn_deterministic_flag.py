"""Regression tests: the D-MPNN trainer's ``--deterministic`` flag is wired.

Background (PA18, 2026-10-12): the D-MPNN trainer was measured to be
run-to-run non-deterministic on CUDA even with every RNG seeded
(``torch.manual_seed`` / ``torch.cuda.manual_seed_all`` / ``np.random.seed`` /
``random.seed``).  MEASURED, three FRESH Re fold-0 runs gave 0.6658 / 0.8242 /
0.6206, and ``PYTHONHASHSEED=0`` did not change that.  The residual variation
is cuBLAS / cuDNN kernel selection.

``--deterministic`` closes that gap by pinning
``torch.use_deterministic_algorithms(True)`` + ``cudnn.deterministic`` +
``CUBLAS_WORKSPACE_CONFIG``.  MEASURED, two fresh Re fold-0 runs then agree to
the last digit (0.7397154345808845).

These tests assert the flag exists and that the three determinism levers are
actually set — a flag declared but never read would pass a `--help` probe and
change nothing (AGENTS.md §5.7 FLAG_ONLY).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MTW_TRAINER = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
)


def test_trainer_declares_deterministic_flag():
    text = MTW_TRAINER.read_text()
    assert re.search(r'"--deterministic"', text), (
        "train_dmpnn_multitask_wasserstein.py does not declare a "
        "--deterministic flag; a single-seed run cannot be reproduced."
    )


def test_trainer_calls_use_deterministic_algorithms():
    text = MTW_TRAINER.read_text()
    assert re.search(
        r"torch\.use_deterministic_algorithms\s*\(\s*True", text
    ), (
        "the --deterministic path does not call "
        "torch.use_deterministic_algorithms(True); cuBLAS/cuDNN kernels "
        "remain free to vary between runs."
    )


def test_trainer_sets_cublas_workspace_config():
    text = MTW_TRAINER.read_text()
    assert re.search(r'CUBLAS_WORKSPACE_CONFIG"\s*\]\s*=\s*":4096:8"', text), (
        "the --deterministic path does not set "
        'CUBLAS_WORKSPACE_CONFIG=":4096:8"; cuBLAS will fall back to a '
        "non-deterministic workspace and the flag becomes a no-op."
    )


def test_trainer_sets_cudnn_deterministic():
    text = MTW_TRAINER.read_text()
    assert re.search(
        r"torch\.backends\.cudnn\.deterministic\s*=\s*True", text
    ), (
        "the --deterministic path does not set cudnn.deterministic=True; "
        "cuDNN convolution/matmul algorithm choice remains run-dependent."
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
