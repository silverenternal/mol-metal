"""A partial MoleculeNet run must not replace the canonical artifact.

Wf-A53 (2026-10-19) — ``--out-metric`` / ``--out-report`` defaulted to
``metrics/by_metric/moleculenet_auc.json`` and
``molmetal/reports/wf_tier2_molnet/final.md``.  Any bare invocation therefore
overwrote the project's headline MoleculeNet table, including a one-task,
one-seed debugging run.  MEASURED: on 2026-10-19 a ``bace``/seed-0 run
replaced the canonical JSON with a one-task fragment (n_test 302 -> 299) and
nothing downstream could tell.

The rule now: explicit path > canonical (full four-task, >=5-seed protocol
only) > ``<out-dir>``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_moleculenet_output_guard_under_test",
        PROJECT_ROOT / "molmetal" / "scripts" / "train_moleculenet.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer()


def _args(tasks, seeds, out_metric=None, out_report=None):
    return argparse.Namespace(
        tasks=tasks, seeds=seeds, out_metric=out_metric, out_report=out_report
    )


OUT_DIR = PROJECT_ROOT / "logs" / "some_run"


def test_partial_run_writes_beside_its_outputs(trainer):
    """One task, one seed — the exact shape of the 2026-10-19 clobber."""
    out_metric, out_report = trainer._resolve_output_paths(
        _args(["bace"], [0]), OUT_DIR
    )
    assert out_metric == OUT_DIR / "metrics.json"
    assert out_report == OUT_DIR / "report.md"
    assert out_metric != trainer.METRICS_DIR / "moleculenet_auc.json"


def test_all_tasks_one_seed_is_still_partial(trainer):
    out_metric, _ = trainer._resolve_output_paths(
        _args(list(trainer.FULL_PROTOCOL_TASKS), [0]), OUT_DIR
    )
    assert out_metric == OUT_DIR / "metrics.json"


def test_four_tasks_four_seeds_is_still_partial(trainer):
    out_metric, _ = trainer._resolve_output_paths(
        _args(list(trainer.FULL_PROTOCOL_TASKS), [0, 1, 2, 3]), OUT_DIR
    )
    assert out_metric == OUT_DIR / "metrics.json"


def test_full_protocol_writes_canonical(trainer):
    out_metric, out_report = trainer._resolve_output_paths(
        _args(list(trainer.FULL_PROTOCOL_TASKS), [0, 1, 2, 3, 4]), OUT_DIR
    )
    assert out_metric == trainer.METRICS_DIR / "moleculenet_auc.json"
    assert out_report == trainer.REPORTS_DIR / "wf_tier2_molnet" / "final.md"


def test_task_order_does_not_matter(trainer):
    out_metric, _ = trainer._resolve_output_paths(
        _args(["hiv", "clintox", "bace", "bbbp"], [0, 1, 2, 3, 4]), OUT_DIR
    )
    assert out_metric == trainer.METRICS_DIR / "moleculenet_auc.json"


def test_explicit_path_always_wins(trainer):
    """A config that names its own destination is honoured, partial or not."""
    out_metric, out_report = trainer._resolve_output_paths(
        _args(
            ["bace"],
            [0],
            out_metric="/tmp/explicit_metrics.json",
            out_report="/tmp/explicit_report.md",
        ),
        OUT_DIR,
    )
    assert out_metric == Path("/tmp/explicit_metrics.json")
    assert out_report == Path("/tmp/explicit_report.md")


# ---------------------------------------------------------------------------
# Stale per-seed files must not be back-filled into a new aggregate
# ---------------------------------------------------------------------------
def test_per_seed_outputs_separates_stale_from_fresh(trainer, tmp_path):
    """A reused out-dir must not mix a previous run's seed into this one.

    Before the fix the report stage read back any existing
    ``<task>_<backbone>_seed<s>.json``.  A seed that crashed therefore
    contributed the previous run's metrics, under the previous protocol.
    """
    import os
    import time

    run_started = time.time()
    fresh = tmp_path / "bace_dmpnn_seed0.json"
    fresh.write_text("{}")
    stale = tmp_path / "bace_dmpnn_seed1.json"
    stale.write_text("{}")
    old = run_started - 3600
    os.utime(stale, (old, old))

    got_fresh, got_stale = trainer._per_seed_outputs(
        tmp_path, "bace", "dmpnn", [0, 1, 2], run_started
    )
    assert [p.name for p in got_fresh] == ["bace_dmpnn_seed0.json"]
    # seed 1 is stale, seed 2 was never written — both are unusable
    assert sorted(p.name for p in got_stale) == [
        "bace_dmpnn_seed1.json",
        "bace_dmpnn_seed2.json",
    ]
