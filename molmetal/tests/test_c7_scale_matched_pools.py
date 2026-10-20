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
    dedupe_pools,
    generation_validity,
    per_epoch_stats,
    pipeline_of,
    run_config,
    tier_of,
    verdict_reading,
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


# --- the crashed-run fallback ------------------------------------------------
# A run that dies before its first per_epoch_metrics.json write has no census
# row either, so without a third source its pool would be tier "unknown" and
# drop out of every cross-scale table. MEASURED 2026-10-20: the C7 full-corpus
# launch died inside epoch 4 with 2,816 generated SMILES and no metrics file.


def _write_ckpt(run_dir, args, *, use_torch=False, name="cfm_yuelbond_wasserstein_seed0_last.pt"):
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / name
    blob = {"args": args, "epoch": 3, "model_state_dict": {}}
    if use_torch:
        import torch

        torch.save(blob, path)
    else:
        import pickle

        path.write_bytes(pickle.dumps(blob))
    return path


def test_run_config_reads_the_checkpoint_when_metrics_and_census_are_absent(tmp_path):
    _write_ckpt(tmp_path, {"n_train": 26715, "epochs": 18, "batch_size": 64})
    cfg = run_config(tmp_path, {})
    assert cfg["n_train"] == 26715
    assert tier_of(cfg["n_train"]) == "full_corpus"


def test_run_config_reads_a_torch_save_checkpoint_too(tmp_path):
    # The trainer writes raw pickle through atomic_write_file, but falls back
    # to torch.save if that fails; both must be readable.
    _write_ckpt(tmp_path, {"n_train": 4096, "epochs": 2}, use_torch=True)
    assert run_config(tmp_path, {})["n_train"] == 4096


def test_checkpoint_fallback_never_overrides_a_real_metrics_file(tmp_path):
    (tmp_path / "per_epoch_metrics.json").write_text(json.dumps({"config": {"n_train": 64}}))
    _write_ckpt(tmp_path, {"n_train": 26715})
    assert run_config(tmp_path, {})["n_train"] == 64


def test_checkpoint_fallback_returns_empty_without_a_checkpoints_dir(tmp_path):
    assert run_config(tmp_path, {}) == {}


def test_checkpoint_fallback_ignores_a_checkpoint_without_args(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    import pickle

    (ckpt_dir / "cfm_yuelbond_wasserstein_seed0_last.pt").write_bytes(
        pickle.dumps({"epoch": 0, "model_state_dict": {}})
    )
    assert run_config(tmp_path, {}) == {}


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


# --- the verdict sentence is derived, not written ----------------------------
# The sentence used to be a literal. When the first full-corpus free-generation
# pool landed, `full_corpus_generation_pool_exists` (computed from the data)
# flipped to True while the prose still said "There is no full-corpus
# free-generation pool". Nothing compared the two, so nothing failed.


def _tier(n, rate, lo, hi):
    return {"n_scored": n, "rate_sa_le_gate": rate, "wilson_ci95": [lo, hi]}


def test_reading_says_a_full_corpus_pool_exists_when_it_does():
    reading = verdict_reading({"by_tier": {"full_corpus": _tier(2816, 0.0036, 0.0019, 0.0065)}})
    assert "full-corpus free-generation pool exists" in reading
    assert "There is no full-corpus" not in reading
    assert "2816" in reading
    assert "0.0036" in reading


def test_reading_says_absent_when_the_tier_is_absent():
    reading = verdict_reading({"by_tier": {"small": _tier(2330, 0.0069, 0.0042, 0.0111)}})
    assert "There is no full-corpus free-generation pool" in reading


def test_reading_compares_full_corpus_against_small_in_the_right_direction():
    worse = verdict_reading(
        {
            "by_tier": {
                "full_corpus": _tier(2816, 0.0036, 0.0019, 0.0065),
                "small": _tier(2330, 0.0069, 0.0042, 0.0111),
            }
        }
    )
    assert "not better" in worse

    better = verdict_reading(
        {
            "by_tier": {
                "full_corpus": _tier(2816, 0.0200, 0.015, 0.026),
                "small": _tier(2330, 0.0069, 0.0042, 0.0111),
            }
        }
    )
    assert "better on this gate" in better
    assert "not better" not in better


def test_reading_never_contradicts_the_pool_exists_flag():
    for tiers in (
        {"full_corpus": _tier(10, 0.5, 0.2, 0.8)},
        {"small": _tier(10, 0.5, 0.2, 0.8)},
        {},
    ):
        generation = {"by_tier": tiers}
        exists = "full_corpus" in tiers
        reading = verdict_reading(generation)
        assert exists == ("full-corpus free-generation pool exists" in reading)
        assert exists != ("There is no full-corpus free-generation pool" in reading)


# --- duplicate pools are one pool --------------------------------------------
# MEASURED 2026-10-21: a second session copied the crashed full-corpus run's
# directory to cfm_c7_fullgen_ep4_partial_2026_10_20, and the aggregates counted
# the same 2,816 molecules twice (full-corpus tier 5,715 instead of 2,899).


def test_a_byte_identical_pool_is_dropped_and_reported():
    pools = [
        {"label": "a", "sha256": "h1", "n_rows": 10, "n_scored": 10},
        {"label": "b", "sha256": "h1", "n_rows": 10, "n_scored": 10},
        {"label": "c", "sha256": "h2", "n_rows": 5, "n_scored": 5},
    ]
    kept, dropped = dedupe_pools(pools)
    assert [p["label"] for p in kept] == ["a", "c"]
    assert len(dropped) == 1
    assert dropped[0]["label"] == "b"
    assert dropped[0]["same_content_as"] == "a"


def test_a_pool_without_a_hash_is_kept():
    # An unhashed pool cannot be proven a duplicate; dropping it would be the
    # worse error, so it survives and is counted.
    kept, dropped = dedupe_pools([{"label": "a"}, {"label": "b"}])
    assert len(kept) == 2
    assert dropped == []


def test_dedupe_does_not_depend_on_directory_names():
    # Same content under unrelated names is still one pool.
    kept, _ = dedupe_pools(
        [
            {"label": "cfm_c7_fullgen_2026_10_20", "sha256": "x"},
            {"label": "cfm_c7_fullgen_ep4_partial_2026_10_20", "sha256": "x"},
        ]
    )
    assert len(kept) == 1
