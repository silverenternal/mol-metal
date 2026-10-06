"""Regression tests: D-MPNN trainers must seed the Python builtin ``random``.

Background (Wf ultracode audit finding): both
``train_dmpnn_multitask_wasserstein.py`` and ``train_dmpnn_pic50_real.py``
set ``torch.manual_seed`` / ``np.random.seed`` / ``torch.cuda.manual_seed_all``
but skip the Python ``random`` module. Any code path that calls
``random.shuffle``, ``random.choice``, ``random.sample`` (e.g. split
shuffles, scaffold selection) is then non-deterministic across runs even
when the trainer's torch / numpy seeds are pinned.

These tests assert the source contains the missing ``random.seed(seed)``
call. They fail when the seed line is absent (proving the gap), and pass
once it is added.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MTW_TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
PIC50_TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_pic50_real.py"


def _imports_random(text: str) -> bool:
    """Return True iff the source imports the Python builtin ``random``."""
    return bool(
        re.search(r"^\s*import\s+random\b", text, re.MULTILINE)
        or re.search(r"^\s*from\s+random\s+import\b", text, re.MULTILINE)
    )


def _calls_random_seed(text: str) -> bool:
    """Return True iff the source calls ``random.seed(`` (the Python builtin
    module, NOT ``np.random.seed`` or ``torch.cuda.manual_seed_all``).

    The token must be at a word boundary — i.e. preceded by whitespace, a
    comma, paren, or start-of-line, NOT by ``.`` (which would mean it is an
    attribute on something else).
    """
    return bool(
        re.search(
            r"(?:^|[\s,(\[])\s*random\.seed\s*\(",
            text,
            re.MULTILINE,
        )
    )


def test_mtw_trainer_imports_random():
    text = MTW_TRAINER.read_text()
    assert _imports_random(text), (
        "train_dmpnn_multitask_wasserstein.py does not import the Python "
        "builtin ``random``; downstream code calling random.shuffle / "
        "random.choice / random.sample will be non-deterministic across runs."
    )


def test_mtw_trainer_calls_random_seed():
    text = MTW_TRAINER.read_text()
    assert _calls_random_seed(text), (
        "train_dmpnn_multitask_wasserstein.py does not call ``random.seed(seed)`` "
        "alongside torch.manual_seed / np.random.seed; cross-run determinism is "
        "not established for Python builtin RNG paths."
    )


def test_pic50_trainer_imports_random():
    text = PIC50_TRAINER.read_text()
    assert _imports_random(text), (
        "train_dmpnn_pic50_real.py does not import the Python builtin ``random``; "
        "downstream code calling random.shuffle / random.choice / random.sample "
        "will be non-deterministic across runs."
    )


def test_pic50_trainer_calls_random_seed():
    text = PIC50_TRAINER.read_text()
    assert _calls_random_seed(text), (
        "train_dmpnn_pic50_real.py does not call ``random.seed(seed)`` alongside "
        "torch.manual_seed / np.random.seed; cross-run determinism is not "
        "established for Python builtin RNG paths."
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
