"""Wf-A6 — anchor walker (``molmetal/scripts/walk_anchors.py``) contract.

The walker exists so ``p3_wf343c4_anchor_freshness_24h`` FAILs are
actionable: an operator can enumerate the stale LIVE anchors instead of
re-running the producing waves blind.  These tests pin the two semantics
that make it correct: the 24h budget, and the frozen-path exemption the
gate already applies.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from walk_anchors import stale_live_anchors  # noqa: E402


def _anchor(path: Path, age_hours: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    old = time.time() - age_hours * 3600.0
    os.utime(path, (old, old))
    return path


def test_stale_live_anchor_is_flagged(tmp_path: Path):
    p = _anchor(tmp_path / "reports" / "x" / "wf269c_report.json", 190.0)
    stale = stale_live_anchors(
        max_age_sec=24 * 3600,
        glob_patterns=(str(tmp_path / "reports" / "**" / "wf269c_report.json"),),
        frozen_paths=(),
    )
    assert len(stale) == 1, stale
    assert stale[0][0] == str(p)
    assert stale[0][1] > 24


def test_fresh_anchor_not_flagged(tmp_path: Path):
    _anchor(tmp_path / "reports" / "x" / "wf269c_report.json", 1.0)
    stale = stale_live_anchors(
        max_age_sec=24 * 3600,
        glob_patterns=(str(tmp_path / "reports" / "**" / "wf269c_report.json"),),
        frozen_paths=(),
    )
    assert stale == []


def test_frozen_anchor_is_exempt(tmp_path: Path):
    p = _anchor(tmp_path / "reports" / "x" / "wf269c_report.json", 190.0)
    stale = stale_live_anchors(
        max_age_sec=24 * 3600,
        glob_patterns=(str(tmp_path / "reports" / "**" / "wf269c_report.json"),),
        frozen_paths=((str(p), "historical"),),
    )
    assert stale == [], "frozen historical anchor must not be flagged"


def test_include_frozen_lists_exempt_anchor(tmp_path: Path):
    p = _anchor(tmp_path / "reports" / "x" / "wf269c_report.json", 190.0)
    stale = stale_live_anchors(
        max_age_sec=24 * 3600,
        glob_patterns=(str(tmp_path / "reports" / "**" / "wf269c_report.json"),),
        frozen_paths=((str(p), "historical"),),
        include_frozen=True,
    )
    assert len(stale) == 1 and stale[0][0] == str(p)
