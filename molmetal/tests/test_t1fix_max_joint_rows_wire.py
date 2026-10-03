"""T1-FIX — ``--max-joint-rows`` FLAG_ONLY wire-in regression guard (§5.7).

CONFIRMED defect (review agent, 2026-10-03): the flag was declared in
``parse_args`` but ``max_joint_rows`` occurred exactly once in the file —
zero live reads, so the cap never applied.  T1-FIX wired it into the
joint-pool assembly of ``train_one_metal_one_seed()``.

CPU-only, deterministic, <5 s.  The end-to-end proof (``n_joint_train``
3444 -> 200 on a real 2-epoch CPU smoke) is in the T1-FIX closeout;
these tests guard the wiring so it cannot silently regress to FLAG_ONLY.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"


def test_flag_has_live_reads_not_flag_only():
    src = TRAINER.read_text(encoding="utf-8")
    # declaration + signature kwarg + live guard + main() threading
    assert src.count("max_joint_rows") >= 4, "flag regressed to FLAG_ONLY"
    assert "if max_joint_rows and 0 < max_joint_rows < len(train_pool_idx):" in src
    assert "max_joint_rows=getattr(args," in src
    # seeded permutation, NOT head-truncation (metal-blocked pool)
    assert "np.random.default_rng(seed)" in src


def test_parse_args_accepts_flag_roundtrip():
    try:
        from molmetal.scripts.train_dmpnn_multitask_wasserstein import parse_args
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"parse_args unavailable: {exc}")
    saved = sys.argv
    try:
        sys.argv = ["trainer.py"]
        default = parse_args()
        sys.argv = ["trainer.py", "--max-joint-rows", "200"]
        capped = parse_args()
    finally:
        sys.argv = saved
    assert default.max_joint_rows is None
    assert capped.max_joint_rows == 200


def test_trainer_function_exposes_kwarg_default_none():
    try:
        from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
            train_one_metal_one_seed,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"trainer import unavailable: {exc}")
    sig = inspect.signature(train_one_metal_one_seed)
    assert "max_joint_rows" in sig.parameters
    assert sig.parameters["max_joint_rows"].default is None
