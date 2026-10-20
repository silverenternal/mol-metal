#!/usr/bin/env python3
"""Tests for the C7 training-scale audit.

The audit's whole claim is "this pool came from a model trained on N complexes",
so the tier rule and the run-config lookup are the parts that must not drift.
A silently wrong n_train would re-attribute a 1%-scale artefact to the
architecture -- which is exactly the error being corrected.
"""

from __future__ import annotations

import json

from molmetal.scripts.c7_scale_matched_pools import (
    FULL_CORPUS_N_TRAIN,
    SMALL_TRAIN_CUTOFF,
    run_config,
    tier_of,
)


def test_full_corpus_tier_is_the_whole_corpus():
    assert tier_of(FULL_CORPUS_N_TRAIN) == "full_corpus"
    assert tier_of(FULL_CORPUS_N_TRAIN + 1) == "full_corpus"


def test_one_below_full_corpus_is_not_full_corpus():
    assert tier_of(FULL_CORPUS_N_TRAIN - 1) != "full_corpus"


def test_the_published_pools_are_small_tier():
    # the three pools cfm_sa_yield_audit.py scored
    for n in (64, 256):
        assert tier_of(n) == "small"


def test_unknown_n_train_does_not_land_in_a_real_tier():
    assert tier_of(None) == "unknown"


def test_mid_tier_is_between_the_two_cutoffs():
    assert tier_of(SMALL_TRAIN_CUTOFF) == "mid"
    assert tier_of(2048) == "mid"


def test_run_config_reads_the_runs_own_metrics(tmp_path):
    (tmp_path / "per_epoch_metrics.json").write_text(
        json.dumps(
            {
                "config": {"n_train": 26715, "epochs": 50},
                "best": {"bond_exact_type_rate": 0.42, "bond_type_macro_f1": 0.5},
            }
        )
    )
    cfg = run_config(tmp_path, {})
    assert cfg["n_train"] == 26715
    assert cfg["bond_exact_type_rate"] == 0.42


def test_run_config_falls_back_to_the_census(tmp_path):
    census = {"runs": [{"run": tmp_path.name, "config": {"n_train": 256, "epochs": 30}}]}
    cfg = run_config(tmp_path, census)
    assert cfg["n_train"] == 256


def test_run_config_prefers_the_run_over_the_census(tmp_path):
    (tmp_path / "per_epoch_metrics.json").write_text(json.dumps({"config": {"n_train": 64}}))
    census = {"runs": [{"run": tmp_path.name, "config": {"n_train": 26715}}]}
    assert run_config(tmp_path, census)["n_train"] == 64


def test_run_config_survives_a_corrupt_metrics_file(tmp_path):
    (tmp_path / "per_epoch_metrics.json").write_text("{not json")
    census = {"runs": [{"run": tmp_path.name, "config": {"n_train": 256}}]}
    assert run_config(tmp_path, census)["n_train"] == 256
