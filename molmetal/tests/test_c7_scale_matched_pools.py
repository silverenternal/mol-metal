#!/usr/bin/env python3
"""Tests for the C7 training-scale audit.

The audit's whole claim is "this pool came from a model trained on N complexes",
so the tier rule and the run-config lookup are the parts that must not drift.
A silently wrong n_train would re-attribute a 1%-scale artefact to the
architecture -- which is exactly the error being corrected.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from molmetal.scripts.c7_scale_matched_pools import (
    FULL_CORPUS_N_TRAIN,
    SMALL_TRAIN_CUTOFF,
    dedupe_pools,
    file_is_run_of_record,
    generation_validity,
    per_epoch_stats,
    pipeline_of,
    pool_own_draw,
    resolve_cohort,
    resolve_population,
    run_config,
    tier_of,
    verdict_reading,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / "molmetal" / "reports"
ARTIFACT = PROJECT_ROOT / "metrics" / "by_metric" / "c7_scale_matched_pools.json"
CENSUS_PATH = PROJECT_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"


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
#
# Wf-A114D (2026-10-30): the header ALONE is no longer the discriminator.
# ``wf_cfm_decode_dump_smiles.py`` wrote ONE header in TWO eras -- before
# 2026-10-12 it ran only the bond head on real tmQM geometries, from
# 2026-10-12 it called the trainer's ODE sampler.  So a ``metal_z`` header is
# decided by the CONTENT of its first data row (the rule owner:
# molmetal/scripts/cfm_pool_kind.py).


def _write_csv(path, header, rows=()):
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n")


METAL_Z_HEADER = [
    "epoch",
    "seed",
    "smiles",
    "sanitized",
    "n_fragments",
    "largest_fragment_frac",
    "metal_z",
]


def test_a_metal_z_header_alone_does_not_decide_the_kind(tmp_path):
    # No data row -> no content -> the kind is NOT established.  Unknown, not
    # a silent bucket.
    path = tmp_path / "generated_smiles.csv"
    _write_csv(path, METAL_Z_HEADER)
    assert pipeline_of(path) == "unknown"


def test_metal_z_header_with_era_a_content_is_a_reconstruction(tmp_path):
    # era A: the bond head on a real tmQM geometry -> sanitized == 1 and a
    # populated metal_z.
    path = tmp_path / "generated_smiles.csv"
    _write_csv(
        path,
        METAL_Z_HEADER,
        [
            {
                "epoch": 0,
                "seed": 0,
                "smiles": "CCO",
                "sanitized": 1,
                "n_fragments": 1,
                "largest_fragment_frac": 1.0,
                "metal_z": "77",
            }
        ],
    )
    assert pipeline_of(path) == "reconstruction"


def test_metal_z_header_with_era_b_content_is_generation(tmp_path):
    # era B: the ODE sampler -> sanitized is structurally 0 and metal_z is
    # empty.  This is the pool of record (cfm_fullcorpus_2026_10_10).
    path = tmp_path / "generated_smiles.csv"
    _write_csv(
        path,
        METAL_Z_HEADER,
        [
            {
                "epoch": 0,
                "seed": 0,
                "smiles": "CCO",
                "sanitized": 0,
                "n_fragments": 1,
                "largest_fragment_frac": 1.0,
                "metal_z": "",
            }
        ],
    )
    assert pipeline_of(path) == "generation"


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


def test_a_strict_prefix_pool_is_dropped_even_with_a_different_hash():
    # MEASURED 2026-10-21: after the full-corpus run resumed past epoch 4, the
    # 4-epoch partial stopped being a byte copy and became a prefix. The sha256
    # rule went quiet and the tier counted 2,816 molecules twice.
    full = {
        "label": "full",
        "sha256": "h-full",
        "n_rows": 8,
        "_row_keys": {(str(e), "C") for e in range(8)},
    }
    partial = {
        "label": "partial",
        "sha256": "h-partial",
        "n_rows": 4,
        "_row_keys": {(str(e), "C") for e in range(4)},
    }
    kept, dropped = dedupe_pools([partial, full])
    assert [p["label"] for p in kept] == ["full"]
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "strict_prefix_of"
    assert dropped[0]["same_content_as"] == "full"


def test_a_partially_overlapping_pool_is_kept():
    # Overlap is not containment: the pool holds molecules the larger one does
    # not, so dropping it would delete real samples.
    big = {"label": "big", "sha256": "h1", "n_rows": 6, "_row_keys": {(str(e), "C") for e in range(6)}}
    other = {"label": "other", "sha256": "h2", "n_rows": 3, "_row_keys": {("0", "C"), ("99", "N"), ("98", "O")}}
    kept, dropped = dedupe_pools([big, other])
    assert [p["label"] for p in kept] == ["big", "other"]
    assert dropped == []


def test_prefix_dedupe_is_order_independent():
    # The larger pool must win regardless of which file the glob returned first.
    full = {"label": "full", "sha256": "h-full", "n_rows": 8, "_row_keys": {(str(e), "C") for e in range(8)}}
    partial = {"label": "partial", "sha256": "h-p", "n_rows": 2, "_row_keys": {(str(e), "C") for e in range(2)}}
    for order in ([full, partial], [partial, full]):
        kept, dropped = dedupe_pools(order)
        assert [p["label"] for p in kept] == ["full"], order
        assert len(dropped) == 1


def test_equal_size_identical_row_sets_are_not_treated_as_a_prefix():
    # Same size + same rows but different hashes should not be silently merged;
    # "strict" subset is the only safe rule.
    a = {"label": "a", "sha256": "h1", "n_rows": 3, "_row_keys": {("0", "C"), ("1", "N"), ("2", "O")}}
    b = {"label": "b", "sha256": "h2", "n_rows": 3, "_row_keys": {("0", "C"), ("1", "N"), ("2", "O")}}
    kept, dropped = dedupe_pools([a, b])
    assert len(kept) == 2
    assert dropped == []


# --- v1.6 (Wf-A115 P2): THE COHORT AND THE EPOCH -----------------------------
#
# TWO defects, both measured on disk:
#
#   1. The cohort was read from a CLOBBERED record.  A run's
#      ``per_epoch_metrics.json`` is written ONCE at the END of training to a
#      FIXED path, so a later run launched into the same ``--out-dir``
#      OVERWRITES it.  For ``cfm_fullcorpus_2026_10_10`` it holds a 1-epoch
#      CPU smoke's config (n_train=64), NOT the run of record's 26,715.
#
#   2. ``gen_validity`` was taken at the WRONG epoch.  The old rule kept the
#      LAST row of the append-only ``events.jsonl`` that carried a
#      ``gen_validity``.  The stream holds SEVERAL populations (a real run,
#      then a smoke that wrote into the same directory), so for that pool it
#      published the SMOKE's epoch 0 (117/128 = 0.9140625) beside a pool
#      whose own epoch is 10.
#
# THE RULE: the ``gen_validity`` published on a pool row is the resolved
# population's value AT THE POOL'S OWN LAST EPOCH, from the AGGREGATE row
# (the one carrying ``gen_validity_n``).


def _artifact():
    return json.loads(ARTIFACT.read_text())


def _pool(label):
    for pool in _artifact()["pools"]:
        if pool["label"] == label:
            return pool
    raise AssertionError(f"pool {label} is not in the artifact")


def _pool_max_epoch(label):
    """The pool CSV's own max ``epoch`` column value -- the pool's last epoch."""
    path = REPORTS / label / "generated_smiles.csv"
    with path.open(newline="") as handle:
        epochs = [
            int(row["epoch"])
            for row in csv.DictReader(handle)
            if row.get("epoch") not in (None, "")
        ]
    return max(epochs)


# (a) the 66-row full-corpus free-generation pool's cohort is the run of
#     record's, so it belongs to the full_corpus tier and NOT to the small one.
def test_the_66_row_pool_carries_the_run_of_record_cohort():
    pool = _pool("cfm_fullcorpus_2026_10_10")
    assert pool["n_rows"] == 66
    assert pool["n_train"] == 26715
    assert pool["tier"] == "full_corpus"
    assert pool["train_fraction_of_corpus"] == 1.0


# (b) its published validity is the run of record's epoch-10 aggregate -- not
#     the smoke's epoch-0 value the old rule published.
def test_the_66_row_pool_publishes_the_epoch_10_validity():
    pool = _pool("cfm_fullcorpus_2026_10_10")
    assert pool["gen_validity"] == {"epoch": 10, "n": 768, "value": 0.3255208333333333}
    assert pool["gen_validity"]["value"] != 0.9140625


# (c) THE UNIFORM RULE: every published validity is at its own pool's last
#     epoch.  This is the check the two reconstruction pools failed before
#     (they published the run's LAST epoch, 8 and 11, instead of the pool's
#     own 5 and 8).
def test_every_published_validity_is_at_its_own_pool_epoch():
    artifact = _artifact()
    assert artifact["pools"], "the artifact carries no pools"
    for pool in artifact["pools"]:
        own_epoch = _pool_max_epoch(pool["label"])
        assert pool["gen_validity"]["epoch"] == own_epoch, pool["label"]
        assert pool["pool_epoch"] == own_epoch, pool["label"]


def test_the_two_reconstruction_pools_moved_to_their_own_epochs():
    # MEASURED: cfm_objective_terms_2026_10_11's pool carries epoch 5 on
    # 43/43 rows and its run's epoch-5 aggregate is 0.234375; the run's
    # epoch-8 value (0.07421875) described a DIFFERENT model.
    terms = _pool("cfm_objective_terms_2026_10_11")
    assert terms["gen_validity"] == {"epoch": 5, "n": 768, "value": 0.234375}
    b64 = _pool("cfm_fullcorpus_b64_2026_10_11")
    assert b64["gen_validity"] == {"epoch": 8, "n": 768, "value": 0.3763020833333333}


def test_the_unchanged_pools_keep_their_published_validity():
    # The uniform rule must REPRODUCE the five pools whose published epoch
    # already equalled their pool's own last epoch.
    expected = {
        "cfm_c7_fullgen_2026_10_20": {"epoch": 17, "n": 704, "value": 1.0},
        "cfm_c8_fullgen_2026_10_24": {"epoch": 2, "n": 704, "value": 1.0},
        "cfm_pcgrad_off_2026_10_15": {"epoch": 29, "n": 48, "value": 0.08333333333333333},
        "cfm_pcgrad_on_2026_10_14": {"epoch": 29, "n": 48, "value": 0.2916666666666667},
        "cfm_ode_link_probe": {"epoch": 1, "n": 768, "value": 0.013020833333333334},
    }
    for label, want in expected.items():
        assert _pool(label)["gen_validity"] == want, label


def test_the_pool_own_draw_is_published_separately_from_the_validity():
    # The pool's OWN draw (66 survivors of 192 attempts) is a DIFFERENT
    # measurement from the run of record's validity, so it has its own field
    # and its own source string.
    pool = _pool("cfm_fullcorpus_2026_10_10")
    own = pool["gen_validity_pool"]
    assert own["n"] == 66
    assert own["n_attempted"] == 192
    assert own["value"] == 66 / 192 == 0.34375
    assert "cfm_free_generation_panel_seed0.json" in own["source"]
    assert pool["gen_validity"]["value"] != own["value"]


def test_pool_own_draw_ignores_a_panel_for_another_run():
    panel = {"checkpoint": "molmetal/reports/cfm_other/checkpoints/x.pt (epoch 10, n_train 1)"}
    assert pool_own_draw(panel, "cfm_fullcorpus_2026_10_10", 10) is None


def test_pool_own_draw_ignores_a_panel_for_another_epoch():
    panel = {
        "checkpoint": "molmetal/reports/cfm_fullcorpus_2026_10_10/checkpoints/x.pt (epoch 10)",
        "generation": {"n_valid": 66, "n_attempted": 192},
    }
    assert pool_own_draw(panel, "cfm_fullcorpus_2026_10_10", 5) is None


# (d) the reconstruction SA cohort is UNTOUCHED -- the rule moved the
#     reconstruction pools' VALIDITY, not their synthesisability numbers.
def test_the_reconstruction_sa_cohort_is_untouched():
    artifact = _artifact()
    for view in (
        artifact["by_tier"]["reconstruction"]["full_corpus"],
        artifact["by_tier_by_kind"]["reconstruction"]["full_corpus"],
        artifact["by_pipeline"]["reconstruction"]["by_tier"]["full_corpus"],
    ):
        assert view["n_pools"] == 2
        assert view["n_scored"] == 83
        assert view["n_sa_le_gate"] == 28
        assert view["rate_sa_le_gate"] == 0.3373
        assert view["wilson_ci95"] == [0.2448, 0.4442]
        assert view["pool_mean_sa"] == 5.0327
        assert view["pool_mean_qed"] == 0.2158
    # ... while the reconstruction VALIDITY did move, because the rule is
    # now uniform across both pipelines.
    assert artifact["by_pipeline"]["reconstruction"]["gen_validity_mean"] == 0.3053
    assert artifact["by_pipeline"]["generation"]["gen_validity_mean"] == 0.4523


# (e) a CLOBBERED directory must not yield the clobbered cohort.
def test_a_clobbered_metrics_file_does_not_win_the_cohort(tmp_path):
    run_dir = tmp_path / "cfm_clobbered"
    run_dir.mkdir()
    # The clobbered record: a 1-epoch CPU smoke that wrote into the same
    # --out-dir AFTER the run of record finished.
    (run_dir / "per_epoch_metrics.json").write_text(
        json.dumps({"config": {"n_train": 64, "n_val": 32, "epochs": 1}})
    )
    # The census attests the RUN OF RECORD's cohort.
    census = {
        "runs": [
            {
                "run": run_dir.name,
                "n_train": 26715,
                "n_val": 192,
                "cohort_source": "run_log:ts_matched",
                "cohort_reason": "the run log echoes every epoch of the run of record",
                "integrity": {
                    "run_of_record_segment_index": 0,
                    "population_matched_segment_index": 2,
                },
            }
        ]
    }
    cohort = resolve_cohort(run_dir, census, [])
    assert cohort["n_train"] == 26715
    assert cohort["n_val"] == 192
    assert cohort["source"] == "census:run_log:ts_matched"


def test_a_clobbered_metrics_file_is_not_used_even_without_a_census_row(tmp_path):
    # No census row at all, but the event stream carries the EVIDENCE: two
    # populations, and the per-epoch file is the LATER one's record (its
    # epoch-0 loss is the smoke's 13.68, not the run of record's 9.9).  The
    # census's own cross-check calls that class_a_other_population, so the
    # file may not attest a cohort -- the cohort is unattested, never 64.
    run_dir = tmp_path / "cfm_clobbered"
    run_dir.mkdir()
    events = [
        {"event": "train_epoch", "epoch": 0, "train_loss": 9.9, "ts": 1.0},
        {"event": "train_epoch", "epoch": 1, "train_loss": 8.8, "ts": 2.0},
        {"event": "done", "epoch": 1, "n_epochs_run": 2, "ts": 3.0},
        {"event": "train_epoch", "epoch": 0, "train_loss": 13.68, "ts": 4.0},
        {"event": "done", "epoch": 0, "n_epochs_run": 1, "ts": 5.0},
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in events) + "\n"
    )
    (run_dir / "per_epoch_metrics.json").write_text(
        json.dumps(
            {
                "config": {"n_train": 64, "n_val": 32},
                "per_epoch_metrics": [{"epoch": 0, "train_loss": 13.68}],
            }
        )
    )
    assert file_is_run_of_record(run_dir) is False
    cohort = resolve_cohort(run_dir, {"runs": []}, [])
    assert cohort["n_train"] is None
    assert cohort["source"] == "unattested"


def test_the_metrics_file_may_attest_the_cohort_when_it_is_the_run_of_record(tmp_path):
    # One population, and the file agrees with it: the file IS the run of
    # record's own record, so its cohort is admissible.
    run_dir = tmp_path / "cfm_clean"
    run_dir.mkdir()
    events = [
        {"event": "train_epoch", "epoch": 0, "train_loss": 9.9, "ts": 1.0},
        {"event": "done", "epoch": 0, "n_epochs_run": 1, "ts": 2.0},
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in events) + "\n"
    )
    (run_dir / "per_epoch_metrics.json").write_text(
        json.dumps(
            {
                "config": {"n_train": 256, "n_val": 192},
                "per_epoch_metrics": [{"epoch": 0, "train_loss": 9.9}],
            }
        )
    )
    assert file_is_run_of_record(run_dir) is True
    cohort = resolve_cohort(run_dir, {"runs": []}, [])
    assert cohort["n_train"] == 256
    assert cohort["source"] == "per_epoch_metrics.json:config"


def test_an_unattested_cohort_is_reported_as_unattested(tmp_path):
    run_dir = tmp_path / "cfm_bare"
    run_dir.mkdir()
    census = {
        "runs": [
            {
                "run": run_dir.name,
                "n_train": None,
                "n_val": None,
                "cohort_source": "unattested",
                "integrity": {"run_of_record_segment_index": 0},
            }
        ]
    }
    cohort = resolve_cohort(run_dir, census, [])
    assert cohort["n_train"] is None
    assert cohort["source"] == "unattested"


# the epoch rule, exercised on a synthetic event stream: the LAST row is a
# LATER population, and the aggregate row at the pool's epoch is the answer.
def test_the_epoch_rule_ignores_a_later_population(tmp_path):
    run_dir = tmp_path / "cfm_two_populations"
    run_dir.mkdir()
    rows = [
        {"event": "train_epoch", "epoch": 9, "gen_validity": 0.5, "gen_validity_n": 768, "ts": 1.0},
        {"event": "train_epoch", "epoch": 10, "gen_validity": 0.25, "gen_validity_n": 768, "ts": 2.0},
        # a per-seed row of the SAME epoch carries no gen_validity_n
        {"event": "gen_eval", "epoch": 10, "gen_validity": 0.9, "ts": 2.0},
        # a LATER population (a 1-epoch smoke) wrote into the same directory
        {"event": "train_epoch", "epoch": 0, "gen_validity": 0.9140625, "gen_validity_n": 128, "ts": 3.0},
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = generation_validity(run_dir, epoch=10)
    assert got == {"epoch": 10, "n": 768, "value": 0.25}


def test_the_epoch_rule_is_none_when_the_epoch_is_absent(tmp_path):
    run_dir = tmp_path / "cfm_one_epoch"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(
        json.dumps({"event": "train_epoch", "epoch": 3, "gen_validity": 0.5, "gen_validity_n": 8})
        + "\n"
    )
    assert generation_validity(run_dir, epoch=10) is None


def test_the_population_falls_back_to_the_segment_containing_the_pool_epoch(tmp_path):
    run_dir = tmp_path / "cfm_segments"
    run_dir.mkdir()
    rows = [
        {"event": "train_epoch", "epoch": 0, "gen_validity": 0.1, "gen_validity_n": 8, "ts": 1.0},
        {"event": "train_epoch", "epoch": 1, "gen_validity": 0.2, "gen_validity_n": 8, "ts": 2.0},
        {"event": "done", "epoch": 1, "n_epochs_run": 2, "ts": 3.0},
        {"event": "train_epoch", "epoch": 0, "gen_validity": 0.99, "gen_validity_n": 8, "ts": 4.0},
        {"event": "done", "epoch": 0, "n_epochs_run": 1, "ts": 5.0},
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    population = resolve_population(run_dir, {"runs": []}, 1)
    assert population["source"] == "fallback:segment_contains_pool_epoch"
    assert population["segment_index"] == 0
    got = generation_validity(run_dir, epoch=1, rows=population["rows"])
    assert got == {"epoch": 1, "n": 8, "value": 0.2}


def test_the_population_prefers_the_census_run_of_record(tmp_path):
    run_dir = tmp_path / "cfm_segments"
    run_dir.mkdir()
    rows = [
        {"event": "train_epoch", "epoch": 0, "gen_validity": 0.99, "gen_validity_n": 8, "ts": 1.0},
        {"event": "done", "epoch": 0, "n_epochs_run": 1, "ts": 2.0},
        {"event": "train_epoch", "epoch": 0, "gen_validity": 0.2, "gen_validity_n": 8, "ts": 3.0},
        {"event": "done", "epoch": 0, "n_epochs_run": 1, "ts": 4.0},
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    census = {
        "runs": [
            {
                "run": run_dir.name,
                "integrity": {
                    "run_of_record_segment_index": 1,
                    "run_of_record_epoch_indices": [0],
                },
            }
        ]
    }
    population = resolve_population(run_dir, census, 0)
    assert population["source"] == "census:run_of_record_segment_index"
    assert population["segment_index"] == 1
    got = generation_validity(run_dir, epoch=0, rows=population["rows"])
    assert got == {"epoch": 0, "n": 8, "value": 0.2}


# (f) the tier cells moved with the pool.
def test_the_generation_tiers_moved_with_the_pool():
    artifact = _artifact()
    for view in (
        artifact["by_tier"]["generation"],
        artifact["by_tier_by_kind"]["generation"],
        artifact["by_pipeline"]["generation"]["by_tier"],
    ):
        full = view["full_corpus"]
        assert full["n_pools"] == 3
        assert full["n_scored"] == 14846
        assert full["n_sa_le_gate"] == 60
        assert full["rate_sa_le_gate"] == 0.004
        assert full["wilson_ci95"] == [0.0031, 0.0052]
        assert full["pool_mean_sa"] == 6.7781
        assert full["pool_mean_qed"] == 0.1056
        small = view["small"]
        assert small["n_pools"] == 2
        assert small["n_scored"] == 2330
        assert small["n_sa_le_gate"] == 16
        assert small["rate_sa_le_gate"] == 0.0069
        assert small["wilson_ci95"] == [0.0042, 0.0111]
        assert small["pool_mean_sa"] == 7.5109
        assert small["pool_mean_qed"] == 0.1991


def test_the_small_tier_is_exactly_the_two_n_train_256_pools():
    artifact = _artifact()
    small = artifact["by_tier"]["generation"]["small"]
    assert small["n_pools"] == 2
    labels = sorted(
        pool["label"]
        for pool in artifact["pools"]
        if pool["pipeline"] == "generation" and pool["tier"] == "small"
    )
    assert labels == ["cfm_pcgrad_off_2026_10_15", "cfm_pcgrad_on_2026_10_14"]
    for label in labels:
        assert _pool(label)["n_train"] == 256


def test_the_verdict_reading_quotes_the_new_numbers():
    reading = _artifact()["verdict"]["reading"]
    assert "14846 molecules" in reading
    assert "0.0040 [0.0031, 0.0052]" in reading
    assert "0.0069 [0.0042, 0.0111]" in reading
    # the qualitative claim still holds against the NEW numbers
    assert "not better on this gate" in reading


def test_the_verdict_scalars_did_not_move():
    verdict = _artifact()["verdict"]
    assert verdict["full_corpus_generation_pool_exists"] is True
    assert verdict["generation_molecules"] == 17188
    assert verdict["generation_pools"] == 6
    assert verdict["generation_tiers_missing"] == []
    assert verdict["reconstruction_molecules"] == 83


def test_the_selection_gap_note_names_the_run_of_record_cohort():
    note = _artifact()["selection_gap"]["note"]
    assert "26715/256/256" in note
    assert "n_train 64/256/256" not in note
    # the rest of that block is unchanged
    assert "2,396" in note


# (g) Wf-A121 / I3 (R7) -- the MOLECULE-weighted companion to the pool mean.
#
# `pool_mean_sa` / `pool_mean_qed` are the UNWEIGHTED mean over the tier's
# pools; `n_scored` is a SUM over molecules.  Both views are published, and
# the weighted one is the molecule-level mean recomputed from the pool rows.
def test_the_by_tier_block_carries_both_the_unweighted_and_the_weighted_mean():
    artifact = _artifact()
    for view in (
        artifact["by_tier"]["generation"],
        artifact["by_tier_by_kind"]["generation"],
        artifact["by_pipeline"]["generation"]["by_tier"],
    ):
        full = view["full_corpus"]
        assert full["pool_mean_sa"] == 6.7781
        assert full["pool_mean_qed"] == 0.1056
        assert full["molecule_weighted_mean_sa"] == 6.9754
        assert full["molecule_weighted_mean_qed"] == 0.1201
        small = view["small"]
        assert small["pool_mean_sa"] == 7.5109
        assert small["pool_mean_qed"] == 0.1991
        assert small["molecule_weighted_mean_sa"] == 7.5106
        assert small["molecule_weighted_mean_qed"] == 0.1991


def test_the_weighting_note_says_which_is_which():
    artifact = _artifact()
    note = artifact["by_tier"]["generation"]["full_corpus"]["pool_mean_sa_weighting"]
    assert "UNWEIGHTED mean over the tier's pools" in note
    assert "n_scored is a SUM over molecules" in note
    assert "molecule_weighted_mean_*" in note


def test_the_weighted_mean_is_the_molecule_weighted_value_of_the_pool_rows():
    artifact = _artifact()
    for tier in ("full_corpus", "mid", "small"):
        rows = [
            p
            for p in artifact["pools"]
            if p["pipeline"] == "generation" and p["tier"] == tier
        ]
        if not rows:
            continue
        den = sum(p["n_scored"] for p in rows)
        sa = sum(p["sa_mean"] * p["n_scored"] for p in rows) / den
        qed = sum(p["qed_mean"] * p["n_scored"] for p in rows) / den
        view = artifact["by_tier"]["generation"][tier]
        assert view["molecule_weighted_mean_sa"] == round(sa, 4)
        assert view["molecule_weighted_mean_qed"] == round(qed, 4)
        # ... and the UNWEIGHTED mean is still the unweighted one.
        unweighted = sum(p["sa_mean"] for p in rows) / len(rows)
        assert view["pool_mean_sa"] == round(unweighted, 4)


def test_the_weighted_companion_is_also_on_the_reconstruction_tier():
    artifact = _artifact()
    view = artifact["by_tier"]["reconstruction"]["full_corpus"]
    assert view["pool_mean_sa"] == 5.0327
    assert view["pool_mean_qed"] == 0.2158
    assert view["molecule_weighted_mean_sa"] == 5.0327
    # the two reconstruction pools are not the same size, so the weighted
    # QED is NOT the unweighted one
    assert view["molecule_weighted_mean_qed"] == 0.2154
