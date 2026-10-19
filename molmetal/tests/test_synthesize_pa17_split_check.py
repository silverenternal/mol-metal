#!/usr/bin/env python3
"""Tests for the PA17 same-protocol split check.

The comparison block in ``moleculenet_auc_chemprop_v2_same_protocol.json`` is
only meaningful if both arms were split identically.  Before 2026-10-20 the
script asserted that in prose (``same_protocol_requires``) and never checked
it, which is how a 2026-10-12 chemprop run on the isomeric key stayed citable
as "same protocol" after the 2026-10-19 key alignment.

These tests pin the check itself: a size mismatch must stop the write, and an
unverifiable comparison must not silently pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import synthesize_pa17_metric as syn  # noqa: E402

TASKS = syn.TASKS


def _task_payload(sizes, auc=0.80):
    """A task block with an explicit per-seed size vector."""
    return {
        "test_auc_mean": auc,
        "test_auc_std": 0.01,
        "per_seed_n_train": [a for a, _ in sizes],
        "per_seed_n_test": [b for _, b in sizes],
        "n_train": sizes[0][0],
        "n_test": sizes[0][1],
        "chemprop_dmpnn_literature_baseline": 0.90,
    }


def _write_inputs(tmp_path, chemprop_sizes, ours_sizes, *, ours_run_dir=None):
    rep = {
        "date": "2026-10-20",
        "seeds": [0, 1],
        "tasks": {t: _task_payload(chemprop_sizes[t]) for t in TASKS},
    }
    ours = {
        "split_key": "bemis_murcko_stereo_blind (includeChirality=False)",
        "backbone": "dmpnn",
        "seeds": [0, 1],
        "tasks": {t: _task_payload(ours_sizes[t], auc=0.85) for t in TASKS},
    }
    lit = {"tasks": {t: {"chemprop_dmpnn_baseline": 0.90} for t in TASKS}}

    paths = {}
    for name, payload in (("rep", rep), ("ours", ours), ("lit", lit)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = p
    return paths


SAME = {t: [(1000, 300), (1001, 301)] for t in TASKS}


def _argv(paths, out, *extra):
    return [
        "--report", str(paths["rep"]),
        "--ours", str(paths["ours"]),
        "--literature", str(paths["lit"]),
        "--out", str(out),
        *extra,
    ]


def test_matching_sizes_write_and_record_verified(tmp_path):
    paths = _write_inputs(tmp_path, SAME, SAME)
    out = tmp_path / "out.json"
    assert syn.main(_argv(paths, out)) == 0
    doc = json.loads(out.read_text())
    assert doc["split_check"]["verified"] is True
    assert doc["split_check"]["seeds_compared"] == 2
    assert doc["report_date"] == "2026-10-20"


def test_size_mismatch_refuses_to_write(tmp_path):
    """The exact shape of the 2026-10-19 defect: 299 vs 302 on bace."""
    ours = dict(SAME)
    ours["bace"] = [(1066, 299), (1058, 305)]
    chemprop = dict(SAME)
    chemprop["bace"] = [(1061, 302), (1061, 302)]
    paths = _write_inputs(tmp_path, chemprop, ours)
    out = tmp_path / "out.json"

    assert syn.main(_argv(paths, out)) == 2
    assert not out.exists(), "a mismatched comparison must not reach disk"


def test_mismatch_can_be_overridden_but_is_marked_unverified(tmp_path):
    ours = dict(SAME)
    ours["bace"] = [(1066, 299)]
    chemprop = dict(SAME)
    chemprop["bace"] = [(1061, 302)]
    paths = _write_inputs(tmp_path, chemprop, ours)
    out = tmp_path / "out.json"

    rc = syn.main(_argv(paths, out, "--allow-unverified-split"))
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["split_check"]["verified"] is False


def test_no_sizes_at_all_refuses_by_default(tmp_path):
    """Silence is not agreement: an absent fingerprint must not pass."""
    paths = _write_inputs(tmp_path, SAME, SAME)
    rep = json.loads(paths["rep"].read_text())
    for t in TASKS:
        rep["tasks"][t].pop("per_seed_n_test")
        rep["tasks"][t].pop("per_seed_n_train")
        rep["tasks"][t].pop("n_test")
        rep["tasks"][t].pop("n_train")
    paths["rep"].write_text(json.dumps(rep), encoding="utf-8")

    ours = json.loads(paths["ours"].read_text())
    for t in TASKS:
        ours["tasks"][t].pop("per_seed_n_test")
        ours["tasks"][t].pop("per_seed_n_train")
        ours["tasks"][t].pop("n_test")
        ours["tasks"][t].pop("n_train")
    paths["ours"].write_text(json.dumps(ours), encoding="utf-8")

    out = tmp_path / "out.json"
    empty_dir = tmp_path / "no_runs"
    empty_dir.mkdir()
    assert syn.main(_argv(paths, out, "--ours-run-dir", str(empty_dir))) == 2
    assert not out.exists()


def test_scalar_only_report_is_compared_on_seed_zero(tmp_path):
    """A pre-2026-10-20 report carries only seed-0 sizes; that still counts."""
    paths = _write_inputs(tmp_path, SAME, SAME)
    rep = json.loads(paths["rep"].read_text())
    for t in TASKS:
        rep["tasks"][t].pop("per_seed_n_test")
        rep["tasks"][t].pop("per_seed_n_train")
    paths["rep"].write_text(json.dumps(rep), encoding="utf-8")

    out = tmp_path / "out.json"
    assert syn.main(_argv(paths, out)) == 0
    doc = json.loads(out.read_text())
    assert doc["split_check"]["verified"] is True
    assert doc["split_check"]["seeds_compared"] == 1


def test_per_seed_files_are_used_when_aggregate_lacks_a_vector(tmp_path):
    """Ground truth fallback: the per-seed run files the aggregate came from."""
    paths = _write_inputs(tmp_path, SAME, SAME)
    ours = json.loads(paths["ours"].read_text())
    for t in TASKS:
        ours["tasks"][t].pop("per_seed_n_test")
        ours["tasks"][t].pop("per_seed_n_train")
        ours["tasks"][t].pop("n_test")
        ours["tasks"][t].pop("n_train")
    paths["ours"].write_text(json.dumps(ours), encoding="utf-8")

    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    for t in TASKS:
        for seed, (ntr, nte) in enumerate(SAME[t]):
            (run_dir / f"{t}_dmpnn_seed{seed}.json").write_text(
                json.dumps({"n_train": ntr, "n_test": nte}), encoding="utf-8"
            )

    out = tmp_path / "out.json"
    assert syn.main(_argv(paths, out, "--ours-run-dir", str(run_dir))) == 0
    doc = json.loads(out.read_text())
    assert doc["split_check"]["verified"] is True
    assert doc["split_check"]["seeds_compared"] == 2
