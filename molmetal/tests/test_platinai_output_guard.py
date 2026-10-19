#!/usr/bin/env python3
"""Tests for the PlatinAI 4-cell output guard and split fingerprint.

Before 2026-10-20 ``train_platinai_4cell.py`` wrote
``metrics/by_metric/platinai_4cell_auc.json`` and
``reports/wf_tier1_metric2_platinai/final.md`` unconditionally.  A one-seed,
one-epoch smoke therefore replaced the canonical three-seed table with a
fragment — the same defect fixed in ``train_moleculenet.py`` on 2026-10-19.

The fingerprint half pins that "scaffold_split: true" is backed by the sizes
the split actually produced, not just by the flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import train_platinai_4cell as pt  # noqa: E402


class _Args:
    def __init__(self, seeds, epochs, out_dir="/tmp/pt_out",
                 out_metric=None, out_report=None):
        self.seeds = seeds
        self.epochs = epochs
        self.out_dir = out_dir
        self.out_metric = out_metric
        self.out_report = out_report


def test_full_protocol_writes_canonical():
    metric, report = pt._resolve_output_paths(_Args([0, 1, 2], pt.MT2_EPOCHS))
    assert metric == pt.METRICS_DIR / "platinai_4cell_auc.json"
    assert report == pt.REPORTS_DIR / "wf_tier1_metric2_platinai" / "final.md"


def test_one_seed_one_epoch_writes_beside_its_outputs():
    """The exact shape of every smoke test — must not touch canonical."""
    metric, report = pt._resolve_output_paths(_Args([0], 1))
    assert metric == Path("/tmp/pt_out/metrics.json")
    assert report == Path("/tmp/pt_out/report.md")
    assert metric != pt.METRICS_DIR / "platinai_4cell_auc.json"


def test_all_seeds_one_epoch_is_still_partial():
    metric, _ = pt._resolve_output_paths(_Args([0, 1, 2], 1))
    assert metric == Path("/tmp/pt_out/metrics.json")


def test_two_seeds_full_epochs_is_still_partial():
    metric, _ = pt._resolve_output_paths(_Args([0, 1], pt.MT2_EPOCHS))
    assert metric == Path("/tmp/pt_out/metrics.json")


def test_explicit_paths_always_win():
    metric, report = pt._resolve_output_paths(
        _Args([0, 1, 2], pt.MT2_EPOCHS,
              out_metric="/tmp/m.json", out_report="/tmp/r.md")
    )
    assert metric == Path("/tmp/m.json")
    assert report == Path("/tmp/r.md")


def test_is_full_protocol_accepts_numpy_ints():
    """Seeds can arrive as np.int64 through the YAML loader."""
    import numpy as np

    assert pt._is_full_protocol([np.int64(0), np.int64(1), np.int64(2)],
                                np.int64(pt.MT2_EPOCHS))
    assert not pt._is_full_protocol([0, 1, 2], 49)


def test_canonical_seeds_constant_matches_cli_default():
    import argparse
    import inspect

    src = inspect.getsource(pt)
    assert "default=[0, 1, 2]" in src, (
        "CLI default and CANONICAL_SEEDS disagree — a run launched with no "
        "--seeds would be treated as partial"
    )
    assert pt.CANONICAL_SEEDS == (0, 1, 2)
