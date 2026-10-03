"""Regression test: Wf-50 pre-condition patch, MOLMETAL_DATA_DIR honored."""

from __future__ import annotations
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TARGETS = [
    "molmetal/scripts/train_metal_cytotox.py",
    "molmetal/scripts/train_platinai_4cell.py",
    "molmetal/scripts/train_metal_lomo.py",
]
PRE_AMBLE_MARKER = "Wf-50 pre-condition: ensure MOLMETAL_DATA_DIR defaults"


@pytest.mark.parametrize("rel_path", TARGETS)
def test_wf50_pre_amble_present(rel_path):
    p = PROJECT_ROOT / rel_path
    assert p.exists(), f"{p} not found"
    txt = p.read_text()
    assert PRE_AMBLE_MARKER in txt, f"{rel_path}: Wf-50 pre-amble missing"


@pytest.mark.parametrize("rel_path", TARGETS)
def test_wf50_no_hard_mnt_storage(rel_path):
    import re as _re

    p = PROJECT_ROOT / rel_path
    txt = p.read_text()
    hard_default = _re.search(r'default=[`\'"]/mnt/storage', txt)
    assert not hard_default, f"{rel_path}: hard-coded /mnt/storage default found"
