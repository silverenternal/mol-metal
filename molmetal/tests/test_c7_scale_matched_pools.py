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
    generation_validity,
    per_epoch_stats,
    pipeline_of,
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


# --- pipeline classification -------------------------------------------------
# v1.0 compared a full-corpus reconstruction against a 256-sample generation
# pool and called the gap "training scale". The header is the only reliable
# discriminator: two reconstruction runs are named cfm_fullcorpus_*.


def _write_csv(path, header):
    path.write_text(",".join(header) + "\n")


def test_metal_z_header_is_a_reconstruction(tmp_path):
    path = tmp_path / "generated_smiles.csv"
    _write_csv(path, ["epoch", "seed", "smiles", "sanitized", "n_fragments", "largest_fragment_frac", "metal_z"])
    assert pipeline_of(path) == "reconstruction"


def test_n_frags_raw_header_is_generation(tmp_path):
    path = tmp_path / "generated_smiles.csv"
    _write_csv(path, ["epoch", "smiles", "n_frags_raw", "largest_frag_frac"])
    assert pipeline_of(path) == "generation"


def test_unknown_header_is_not_silently_generation(tmp_path):
    path = tmp_path / "generated_smiles.csv"
    _write_csv(path, ["epoch", "smiles"])
    assert pipeline_of(path) == "unknown"


def test_missing_file_is_unknown(tmp_path):
    assert pipeline_of(tmp_path / "nope.csv") == "unknown"


def test_generation_validity_takes_the_last_logged_epoch(tmp_path):
    (tmp_path / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"epoch": 1, "gen_validity": 0.1}),
                json.dumps({"epoch": 2, "gen_validity": 0.25, "gen_validity_n": 768}),
                json.dumps({"epoch": 2, "event": "done"}),
            ]
        )
    )
    got = generation_validity(tmp_path)
    assert got["value"] == 0.25
    assert got["n"] == 768
    assert got["epoch"] == 2


def test_generation_validity_absent_is_none(tmp_path):
    (tmp_path / "events.jsonl").write_text(json.dumps({"epoch": 1, "train_loss": 3.0}))
    assert generation_validity(tmp_path) is None


def test_generation_validity_survives_a_truncated_line(tmp_path):
    (tmp_path / "events.jsonl").write_text('{"epoch": 1, "gen_validity": 0.4}\n{"epoch": 2, "gen_')
    assert generation_validity(tmp_path)["value"] == 0.4


# --- per-epoch breakdown -----------------------------------------------------
# A 30-epoch pool mixes a model that cannot emit bonds with one that can. The
# pooled rate is not the rate anyone would ship, so the last epoch is reported
# separately.


def _rows(pairs):
    return [{"epoch": str(e), "smiles": s} for e, s in pairs]


def test_per_epoch_splits_by_epoch():
    rows = _rows([(0, "a"), (0, "b"), (1, "a")])
    sa = {"a": 4.0, "b": 6.0}
    got = per_epoch_stats(rows, sa, {})
    assert got["0"]["n"] == 2
    assert got["0"]["n_sa_le_gate"] == 1
    assert got["1"]["n"] == 1
    assert got["1"]["n_sa_le_gate"] == 1


def test_per_epoch_counts_joint_only_when_both_hold():
    rows = _rows([(0, "a"), (0, "b")])
    sa = {"a": 4.0, "b": 4.0}
    nov = {"a": 0.95, "b": 0.50}
    got = per_epoch_stats(rows, sa, nov)
    assert got["0"]["n_sa_le_gate"] == 2
    assert got["0"]["n_joint"] == 1


def test_per_epoch_skips_unparseable_smiles():
    rows = _rows([(0, "a"), (0, "unparseable")])
    got = per_epoch_stats(rows, {"a": 4.0}, {})
    assert got["0"]["n"] == 1


def test_per_epoch_skips_blank_epoch():
    rows = [{"epoch": "", "smiles": "a"}]
    assert per_epoch_stats(rows, {"a": 4.0}, {}) == {}


def test_per_epoch_reports_wilson_for_small_n():
    rows = _rows([(0, "a")])
    got = per_epoch_stats(rows, {"a": 4.0}, {})
    lo, hi = got["0"]["wilson_ci95"]
    # 1 of 1 is not 1.0: the Wilson interval still carries the sample size.
    assert 0.0 < lo < 1.0
    assert lo < hi <= 1.0
