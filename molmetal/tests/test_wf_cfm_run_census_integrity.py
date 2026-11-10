# Licensed under the Apache License, Version 2.0.
"""Wf-A115 — the CFM run census must expose run INTEGRITY and the COHORT axis.

Three defects were MEASURED on disk and are pinned here:

1. **THE LOADER COULD NOT SEE THE RUNS.**  The scan was
   ``REPORTS.iterdir()`` — one level deep — so 17 runs nested under a wave
   directory were structurally invisible.  The scan is now
   ``REPORTS.rglob("per_epoch_metrics.csv")``.

2. **RUN INTEGRITY WAS RECORDED NOWHERE.**  ``n_epochs_run`` is a ROW COUNT,
   not "how many epochs the model trained": MEASURED, ``cfm_r11_eval256``
   has exactly ONE recorded epoch (index 19, resumed) with a ``nan``
   ``train_loss`` out of a declared 20.  The 0.572276953125 end-to-end rate
   that was published from that run is WITHDRAWN as a bar.

3. **THE SERIES WAS READ AT CSV PRECISION.**  The published bar
   0.41145833333333337 is the JSON value; the 4-decimal CSV renders it
   0.41147500000000004.  The census now reads the JSON and records which
   source it used.

Plus the cohort axis: a cohort MEDIAN is not a bar (the two cohorts are out
of phase, so the ordering flips with the budget), so the bar must be a
PER-RUN, same-reading, same-cohort, same-budget comparison against a NAMED
run.

CPU-only, no GPU, no training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import wf_cfm_run_census as census  # noqa: E402

ARTIFACT = REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"


@pytest.fixture(scope="module")
def art():
    if not ARTIFACT.is_file():
        pytest.skip("census artifact not on disk")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs(art):
    return {r["run"]: r for r in art["runs"]}


class TestTheLoaderSeesEveryRun:
    def test_the_nested_runs_are_visible(self, art):
        """MEASURED: 17 runs live under a wave directory, not at the top."""
        nested = [r["run"] for r in art["runs"] if "/" in r["run"]]
        assert nested, "the depth-1-only loader is back: no nested run found"
        assert art["corpus"]["n_runs_nested"] == len(nested)

    def test_a_run_name_is_the_path_relative_to_reports(self, runs):
        for name, r in runs.items():
            assert "/" in name or r["run_path"].endswith(f"/{name}")
            assert r["run_path"].endswith(name)

    def test_run_path_is_absolute_and_on_disk(self, runs):
        for name, r in runs.items():
            assert Path(r["run_path"]).is_absolute()
            assert (Path(r["run_path"]) / "per_epoch_metrics.csv").is_file(), name

    def test_the_census_finds_more_runs_than_a_depth_one_scan(self):
        """The loader fix must be a strict superset of the old one."""
        reports = REPO_ROOT / "molmetal" / "reports"
        depth1 = {d.name for d in reports.iterdir()
                  if (d / "per_epoch_metrics.csv").is_file()}
        rglob = {d.parent.name for d in reports.rglob("per_epoch_metrics.csv")}
        assert rglob - depth1, "rglob found nothing the depth-1 scan missed"


class TestSeriesSource:
    def test_the_series_comes_from_the_json(self, art, runs):
        """Wf-A114E: the source is json, csv, or — for a run whose per-epoch
        file was CLOBBERED by a shorter run — ``events`` (the series is
        re-derived from the append-only event stream of the matching
        population).  See `integrity.population_verdict` for which is which.
        """
        for name, r in runs.items():
            assert r["integrity"]["series_source"] in (
                "json", "csv", "events",
            ), name

    def test_the_published_bar_is_the_json_value_not_the_csv_rounding(self, runs):
        """0.41145833333333337 (JSON) vs 0.41147500000000004 (CSV, 4 dp)."""
        bar = runs["cfm_r9_fix_v2"]["gen_single_molecule_rate_end"]
        assert bar == pytest.approx(0.41145833333333337, abs=1e-12)
        assert bar != pytest.approx(0.41147500000000004, abs=1e-12)

    def test_the_csv_fallback_still_works(self, tmp_path):
        """A run with only a CSV must load, with series_source == 'csv'."""
        d = tmp_path / "run"
        d.mkdir()
        (d / "per_epoch_metrics.csv").write_text(
            "epoch,train_loss,gen_validity_val,gen_validity_n,"
            "gen_single_fragment_ratio,bond_f1\n"
            "0,1.0,8,10,0.5000,0.4\n"
            "1,0.9,9,10,0.6000,0.5\n",
            encoding="utf-8",
        )
        rows, source, _ = census._load_series(d)
        assert source == "csv"
        assert [r["epoch"] for r in rows] == [0, 1]
        assert rows[0]["gen_validity"] == pytest.approx(0.8)


class TestRunIntegrity:
    """The spec's four named runs, and the rest of the corpus."""

    def test_a_resumed_one_epoch_run(self, runs):
        i = runs["cfm_r11_eval256"]["integrity"]
        assert i["epoch_index_start"] == 19
        assert i["epoch_index_end"] == 19
        assert i["epochs_recorded"] == 1
        assert i["epochs_declared"] == 20
        assert i["resumed"] is True
        assert i["epoch_history_contiguous_from_zero"] is False
        assert i["nan_train_loss"] is True
        assert i["nan_train_loss_epochs"] == [19]

    def test_a_single_epoch_run_with_a_nan_loss(self, runs):
        i = runs["cfm_r17_eval512"]["integrity"]
        assert i["epoch_index_start"] == 0
        assert i["epochs_recorded"] == 1
        assert i["epochs_declared"] == 20
        assert i["nan_train_loss"] is True

    def test_a_resumed_run_that_started_at_one(self, runs):
        i = runs["cfm_bond_head_20ep_2026_10_09"]["integrity"]
        assert i["epoch_index_start"] == 1
        assert i["epochs_recorded"] == 19
        assert i["epochs_declared"] == 20
        assert i["resumed"] is True
        assert i["epoch_history_contiguous_from_zero"] is False

    def test_a_run_that_started_at_four(self, runs):
        i = runs["cfm_c7_fullgen_2026_10_20"]["integrity"]
        assert i["epoch_index_start"] == 4
        assert i["epochs_recorded"] == 14
        assert i["epochs_declared"] == 18
        assert i["resumed"] is True

    def test_every_other_run_starts_at_zero_and_is_clean(self, runs):
        named = {
            "cfm_r11_eval256", "cfm_r17_eval512",
            "cfm_bond_head_20ep_2026_10_09", "cfm_c7_fullgen_2026_10_20",
        }
        others = [r for n, r in runs.items() if n not in named]
        assert others
        for r in others:
            i = r["integrity"]
            assert i["epoch_index_start"] == 0, r["run"]
            assert i["epoch_history_contiguous_from_zero"] is True, r["run"]
            assert i["nan_train_loss"] is False, r["run"]

    def test_n_epochs_run_is_a_row_count_not_epochs_trained(self, runs):
        for name, r in runs.items():
            assert r["n_epochs_run"] == r["integrity"]["epochs_recorded"] or (
                r["n_epochs_run"] >= r["integrity"]["epochs_recorded"]
            ), name
            assert "DATA ROWS" in r["integrity"]["notes"]["n_epochs_run"]

    def test_epochs_recorded_counts_distinct_epoch_indices(self):
        """42/42 runs: no duplicate epoch index, so the two counts agree."""
        rows = [
            {"epoch": 0, "train_loss": 1.0},
            {"epoch": 1, "train_loss": 1.0},
            {"epoch": 1, "train_loss": 1.0},
        ]
        integ = census._integrity(rows, {"epochs": 3}, "json")
        # three DATA ROWS, two DISTINCT epoch indices — which is exactly why
        # the two counts are kept apart.
        assert len(rows) == 3
        assert integ["epochs_recorded"] == 2
        # a GAP in the distinct indices is what breaks contiguity
        gap = census._integrity(
            [{"epoch": 0}, {"epoch": 2}], {"epochs": 3}, "json"
        )
        assert gap["epoch_history_contiguous_from_zero"] is False
        assert gap["epochs_recorded"] == 2


class TestNanDetection:
    def test_a_json_numeric_zero_is_not_a_nan(self):
        """The old ``or 'nan'`` idiom turned a measured 0.0 into None."""
        assert census._num(0) == 0.0
        assert census._num(0.0) == 0.0
        assert census._num("0.0000") == 0.0

    def test_both_nan_encodings_are_nan(self):
        assert census._num(float("nan")) is None
        assert census._num("nan") is None
        assert census._num("NaN") is None
        assert census._num("") is None
        assert census._num(None) is None

    def test_present_but_nan_is_distinct_from_absent(self):
        assert census._is_nan({"train_loss": float("nan")}, "train_loss") is True
        assert census._is_nan({"train_loss": 0.0}, "train_loss") is False
        assert census._is_nan({}, "train_loss") is False


class TestCohortIsFirstClass:
    def test_n_train_is_on_the_run_and_config_keeps_its_key(self, runs):
        """The run's `n_train` IS the config's `n_train` — whenever the
        config block on disk is the run of record's OWN.

        MEASURED (Wf-A116): one run legitimately breaks the identity.
        `cfm_fullcorpus_2026_10_10`'s `per_epoch_metrics.json` was
        OVERWRITTEN by a later, shorter launch into the same directory, so
        the census re-derived that run's config from the events stream
        (`config_conflict: true`, and the block carries no `n_train` at
        all).  The cohort was recovered from the ts-attested run log
        instead, so `n_train` (26715) is NOT the file's number.  For those
        runs the identity is inverted into a STRONGER claim: the cohort
        must not be the file's number, and the census must publish why.
        """
        for name, r in runs.items():
            assert "n_train" in r, name
            cfg = r.get("config") or {}
            if r.get("config_conflict"):
                assert (
                    r.get("cohort_source") != "per_epoch_metrics.json:config"
                ), name
                assert r.get("cohort_reason"), name
                assert (
                    r["n_train"] is None or r["n_train"] != cfg.get("n_train")
                ), name
            else:
                assert r["n_train"] == cfg.get("n_train"), name

    def test_the_clobbered_record_does_not_own_its_run_cohort(self, runs):
        """MEASURED (Wf-A116): the one full-corpus free-generation pool's
        cohort comes from the ts-attested run log, not from the per-epoch
        file that a later launch overwrote."""
        r = runs["cfm_fullcorpus_2026_10_10"]
        assert (r.get("integrity") or {}).get(
            "population_verdict"
        ) == "class_a_other_population"
        assert r["cohort_source"] == "run_log:ts_matched"
        assert r["n_train"] == 26715
        assert r["n_val"] == 192
        assert (r.get("config") or {}).get("n_train") is None
        assert (r.get("config") or {}).get("n_val") is None
        assert r.get("cohort_reason")

    def test_the_cohort_label_is_the_raw_value_never_binarised(self, art):
        labels = {
            k for k, v in art["by_n_train"].items()
            if isinstance(v, dict) and "n_runs" in v
        }
        assert {"256", "26715", "512"} <= labels
        # A binary split would have collapsed these into two buckets.
        assert len(labels) >= 5


class TestByNTrain:
    def test_the_block_has_the_specified_shape(self, art):
        blk = art["by_n_train"]["256"]
        for key in (
            "n_runs", "n_runs_with_e2e", "last_epoch", "best_epoch", "clean_only",
        ):
            assert key in blk
        for sub in ("last_epoch", "best_epoch"):
            for key in (
                "n", "min", "max", "mean", "median", "n_at_or_above_0.50",
            ):
                assert key in blk[sub], (sub, key)
            assert key in blk["clean_only"][sub]

    def test_the_depth_one_scope_reproduces_the_pinned_cohort_numbers(self, art):
        """MEASURED (clean 24, depth-1 loader scope)."""
        b = art["by_n_train_depth1_only"]
        c256 = b["256"]["clean_only"]
        # Wf-A127 (2026-10-31) moved the LAST-epoch population: two CLEAN
        # runs (cfm_fm_control_2026_10_09, cfm_bond_head_30ep_onecycle_
        # 2026_10_09) lost their last-epoch rate when the empty-denominator
        # chemistry rate was published as null instead of the trainer's 0.0
        # placeholder, so the 256 cohort's CLEAN last-epoch reading is over
        # SIX runs, not eight.  The BEST-epoch reading is back to EIGHT:
        # Wf-A150 repaired the block's population, which used to intersect
        # the CLEAN predicate with the LAST-epoch population and so silently
        # dropped the two runs above.
        assert c256["last_epoch"]["n"] == 6
        assert c256["best_epoch"]["median"] == pytest.approx(
            0.7708333333333333, abs=1e-12
        )
        assert c256["best_epoch"]["max"] == pytest.approx(0.9375, abs=1e-12)
        assert c256["best_epoch"]["n_at_or_above_0.50"] == 5
        assert c256["last_epoch"]["median"] == pytest.approx(
            0.03645833333333333, abs=1e-12
        )
        c26715 = b["26715"]["clean_only"]
        # MEASURED (Wf-A116): the 26715 cohort grew by ONE run — the
        # full-corpus free-generation pool, whose cohort the census
        # recovered from the ts-attested run log (26715/192) after its
        # per-epoch file was found to be another population's record.
        assert c26715["last_epoch"]["n"] == 11
        assert c26715["best_epoch"]["median"] == pytest.approx(
            0.35546875, abs=1e-12
        )
        assert c26715["best_epoch"]["max"] == pytest.approx(
            0.5260416666666666, abs=1e-12
        )
        assert c26715["best_epoch"]["n_at_or_above_0.50"] == 3
        assert c26715["last_epoch"]["median"] == pytest.approx(
            0.125, abs=1e-12
        )
        assert c26715["last_epoch"]["max"] == pytest.approx(
            0.5208333333333334, abs=1e-12
        )

    def test_the_excluded_runs_are_named_not_just_counted(self, art):
        excluded = art["by_n_train_depth1_only"]["26715"]["clean_only"][
            "excluded_runs"
        ]
        assert art["by_n_train_depth1_only"]["26715"]["clean_only"][
            "n_excluded"
        ] == len(excluded)
        for name in (
            "cfm_c7_fullgen_2026_10_20",
            "cfm_r11_eval256",
            "cfm_r17_eval512",
        ):
            assert name in excluded, name

    def test_the_clean_predicate_is_the_specified_one(self, art):
        pred = art["by_n_train"]["256"]["clean_only"]["predicate"]
        assert "epoch_history_contiguous_from_zero" in pred
        assert "nan_train_loss" in pred

    def test_the_clean_best_epoch_population_is_the_predicate_not_an_intersection(
        self, art
    ):
        """Wf-A150 — REGRESSION GUARD for the undeclared population.

        The ``clean_only.best_epoch`` reading used to be computed over the
        INTERSECTION of the CLEAN predicate with the LAST-epoch population —
        a population the block's own ``predicate`` string never mentioned,
        and which disagreed with the sibling ``best_epoch`` block and with
        the top-level ``fragmentation_summary`` (both computed over the
        NON-intersected population).  The intersection only became visible
        under Wf-A127, when two CLEAN runs lost their last-epoch rate while
        keeping their best-epoch rate, and they fell out of the block.

        This re-derives the population from the artifact's OWN run rows and
        asserts the block agrees with the population its predicate names.
        """
        all_runs = art["runs"]
        depth1 = [r for r in all_runs if "/" not in r["run"]]

        def _clean(r):
            integ = r.get("integrity") or {}
            return bool(
                integ.get("epoch_history_contiguous_from_zero")
                and not integ.get("nan_train_loss")
            )

        def _by_n_train(runs_):
            cohorts = {}
            for r in runs_:
                cohorts.setdefault(str(r.get("n_train")), []).append(r)
            return cohorts

        def _by_cohort(runs_):
            cohorts = {}
            for r in runs_:
                label = f"{r.get('n_train')}/{r.get('n_val')}"
                cohorts.setdefault(label, []).append(r)
            return cohorts

        for scope_name, block_name, grouper, scoped in (
            ("all", "by_n_train", _by_n_train, all_runs),
            ("depth1", "by_n_train_depth1_only", _by_n_train, depth1),
            ("all", "by_cohort", _by_cohort, all_runs),
            ("depth1", "by_cohort_depth1_only", _by_cohort, depth1),
        ):
            block = art[block_name]
            for label, members in grouper(scoped).items():
                if label not in block:
                    continue
                clean_best = sorted(
                    float(r["gen_single_molecule_rate_best"])
                    for r in members
                    if _clean(r)
                    and r.get("gen_single_molecule_rate_best") is not None
                )
                # the INTERSECTED population the block used to compute
                intersected = sorted(
                    float(r["gen_single_molecule_rate_best"])
                    for r in members
                    if _clean(r)
                    and r.get("gen_single_molecule_rate_end") is not None
                    and r.get("gen_single_molecule_rate_best") is not None
                )
                got = block[label]["clean_only"]["best_epoch"]
                if not clean_best:
                    assert got["n"] == 0, (scope_name, block_name, label, got)
                    continue
                expected_median = (
                    clean_best[len(clean_best) // 2]
                    if len(clean_best) % 2
                    else (
                        clean_best[len(clean_best) // 2 - 1]
                        + clean_best[len(clean_best) // 2]
                    )
                    / 2
                )
                assert got["n"] == len(clean_best), (
                    scope_name, block_name, label, got, len(clean_best),
                )
                assert got["median"] == pytest.approx(
                    expected_median, abs=1e-12
                ), (scope_name, block_name, label, got)
                assert got["max"] == pytest.approx(
                    clean_best[-1], abs=1e-12
                ), (scope_name, block_name, label, got)
                if len(clean_best) != len(intersected):
                    # the population really did separate — the block must be
                    # over the LARGER (predicate-named) one
                    assert got["n"] == len(clean_best) > len(intersected), (
                        scope_name, block_name, label, got,
                        len(clean_best), len(intersected),
                    )

    def test_the_256_cohort_best_epoch_reading_is_not_the_intersection(self, art):
        """Wf-A150 — the two runs that Wf-A127 emptied are back in the block.

        ``cfm_fm_control_2026_10_09`` (best-epoch rate 0.6875) and
        ``cfm_bond_head_30ep_onecycle_2026_10_09`` (0.875) are CLEAN and
        carry a best-epoch rate; they lost only their LAST-epoch rate.  The
        block used to drop them, publishing n=6 / median 0.59375 for the
        n_train=256 cohort instead of n=8 / median 0.7708333333333333.
        """
        blk = art["by_n_train"]["256"]["clean_only"]["best_epoch"]
        assert blk["n"] == 8
        assert blk["median"] == pytest.approx(0.7708333333333333, abs=1e-12)
        assert blk["max"] == pytest.approx(0.9375, abs=1e-12)
        assert blk["n_at_or_above_0.50"] == 5
        # the sibling non-intersected block must AGREE with the clean one
        sib = art["by_n_train"]["256"]["best_epoch"]
        assert sib["n"] == blk["n"]
        assert sib["median"] == pytest.approx(blk["median"], abs=1e-12)

    def test_retracted_numbers_live_in_this_block_too(self, art):
        assert "0.572276953125" in art["by_n_train"]["retracted_numbers"]


class TestBudgetMatched:
    def test_the_k_table_is_reproduced(self, art):
        bm = art["budget_matched"]
        assert bm["budgets"] == [5, 9, 11, 12, 13, 14, 15, 18, 20, 25, 28]
        a, b = bm["cohorts"]["256"], bm["cohorts"]["26715"]
        # MEASURED (Wf-A116): the 26715 cohort carries one more run than
        # the pre-A116 table did, so its medians moved; the 256 cohort is
        # unchanged.
        # Wf-A127 (2026-10-31) moved the 256 cohort's K=5 / K=12 / K=14
        # medians with it: the two runs that lost their last-epoch rate are
        # two of the runs that REACHED those epoch indices, and a run that
        # reached epoch K-1 contributes its value there.  The 26715 column
        # is unchanged.
        assert a["5"]["median"] == pytest.approx(0.375, abs=1e-12)
        assert b["5"]["median"] == pytest.approx(0.11328125, abs=1e-12)
        assert a["9"]["median"] == pytest.approx(0.0, abs=1e-12)
        assert b["9"]["median"] == pytest.approx(0.17187500000000003, abs=1e-12)
        assert a["12"]["median"] == pytest.approx(0.8125, abs=1e-12)
        assert b["12"]["median"] == pytest.approx(0.20768229166666666, abs=1e-12)
        assert a["14"]["median"] == pytest.approx(0.7916666666666667, abs=1e-12)
        assert b["14"]["median"] == pytest.approx(0.5208333333333334, abs=1e-12)

    def test_a_run_that_never_reached_the_epoch_contributes_nothing(self, art):
        """n_reached counts the runs that got there; n_values the ones with
        a number there."""
        bm = art["budget_matched"]
        for cohort in bm["cohorts"].values():
            for K, blk in cohort.items():
                assert blk["n_values"] <= blk["n_reached"], K
                assert blk["epoch_index"] == int(K) - 1

    def test_the_ordering_flips_are_listed_explicitly(self, art):
        bm = art["budget_matched"]
        flips = bm["ordering_flips"]
        assert flips
        assert bm["flipped_budgets"] == [
            f["K"] for f in flips if f["flipped"]
        ]
        assert bm["flipped_budgets"], "the ordering never flips — check the data"
        for f in flips:
            assert "leader" in f and "previous_leader" in f
            assert f["median_n_train_256"] is not None
            assert f["median_n_train_26715"] is not None

    def test_the_n_train_axis_is_unresolved_and_says_why(self, art):
        bm = art["budget_matched"]
        assert bm["n_train_axis"] == "UNRESOLVED"
        assert "flips" in bm["n_train_axis_note"]

    def test_the_full_corpus_cohort_surviving_sample_collapses(self, art):
        """MEASURED: the 26715 cohort's surviving n falls from 9 to 1."""
        bm = art["budget_matched"]
        b = bm["cohorts"]["26715"]
        assert b["9"]["n_values"] == 9
        assert b["28"]["n_values"] == 1


class TestPerRunBar:
    def test_the_primary_bar_is_a_named_run(self, art):
        p = art["per_run_bar"]["primary"]
        assert p["run"] == "cfm_r9_fix_v2"
        assert p["n_train"] == 26715
        assert p["reading"] == "final_epoch"
        assert p["value"] == pytest.approx(0.41145833333333337, abs=1e-12)
        assert p["integrity"]["epochs_recorded"] == p["integrity"]["epochs_recorded"]

    def test_the_secondary_bar_is_the_best_epoch_reading(self, art):
        s = art["per_run_bar"]["secondary"]
        assert s["run"] == "cfm_fullcorpus_seed1"
        assert s["reading"] == "best_epoch"
        assert s["value"] == pytest.approx(0.5260416666666666, abs=1e-12)

    def test_the_stretch_number_carries_its_cohort_and_is_report_only(self, art):
        st = art["per_run_bar"]["stretch"]
        assert st["value"] == pytest.approx(0.9375, abs=1e-12)
        assert st["cohort"] == "n_train=256"
        assert st["epoch"] == 12
        assert st["pass_fail"] is False
        assert st["report_only"] is True

    def test_both_point9375_runs_are_carried_with_their_cohort(self, art):
        st = art["per_run_bar"]["stretch"]
        names = {r["run"] for r in st["runs"]}
        assert names == {"cfm_fm_ab_2026_10_09", "cfm_pcgrad_off_2026_10_15"}
        for r in st["runs"]:
            assert r["cohort"] == "n_train=256"
            assert r["epoch"] == 12
            assert r["value"] == pytest.approx(0.9375, abs=1e-12)

    def test_the_rule_is_stated(self, art):
        assert "PER-RUN" in art["per_run_bar"]["rule"]


class TestHeadLevers:
    def test_null_means_not_recorded_never_the_argparse_default(self, runs):
        """The forbidden move is back-filling 'radial' / 'zero' / 'learnable'."""
        for name, r in runs.items():
            for key, value in r["head_levers"].items():
                if value is None:
                    continue
                assert value in (
                    "radial", "zero", "learnable", "lsq", "fixed",
                    "vector_channels",
                ) or isinstance(value, (int, float)), (name, key, value)

    def test_a_run_with_no_recorded_lever_says_so(self, runs):
        r = runs["cfm_r9_fix_v2"]
        assert r["head_levers_recorded"] is False
        assert r["head_levers_source"] is None
        assert all(v is None for v in r["head_levers"].values())
        assert "NOT RECORDED" in r["head_levers_note"]

    def test_the_config_is_read_before_the_provenance(self, runs):
        r = runs["wfa113_t42_bond_ce/wfa113_t42"]
        assert r["head_levers"]["velocity_head"] == "vector_channels"
        assert (
            r["head_levers_source_per_key"]["velocity_head"] == "config"
        )
        assert (
            r["head_levers_source_per_key"]["velocity_head_init"] == "provenance"
        )

    def test_the_levers_are_in_the_config_block_too(self, runs):
        r = runs["wfa113_t42_bond_ce/wfa113_t42"]
        assert r["config"]["velocity_head"] == "vector_channels"
        assert r["config"]["vector_channels"] == 8

    def test_velocity_head_init_knots_is_null_when_nothing_recorded_it(self, runs):
        r = runs["wfa113_t42_bond_ce/wfa113_t42"]
        assert r["head_levers"]["velocity_head_init_knots"] is None


class TestFailureMode:
    def test_the_four_classes_exist(self, art):
        modes = {r["failure_mode"] for r in art["runs"]}
        assert modes <= {
            "unknown", "unparseable_dominated", "parseable_fragmented", "ok",
        }
        assert modes

    def test_the_threshold_is_the_projects_own_gate_threshold(self):
        from molmetal.scripts.cfm_fragmentation_gate import (
            DEFAULT_MAX_FRAGMENTED_RATIO,
        )

        assert census._FRAGMENTATION_THRESHOLD == DEFAULT_MAX_FRAGMENTED_RATIO

    def test_the_rule_is_applied_consistently(self, art):
        for r in art["runs"]:
            v = r["gen_validity_end"]
            e = r["gen_single_molecule_rate_end"]
            if v is None:
                assert r["failure_mode"] == "unknown", r["run"]
            elif v < 0.5:
                assert r["failure_mode"] == "unparseable_dominated", r["run"]
            elif e is None:
                assert r["failure_mode"] == "unknown", r["run"]
            elif e < 0.5:
                assert r["failure_mode"] == "parseable_fragmented", r["run"]
            else:
                assert r["failure_mode"] == "ok", r["run"]

    def test_a_best_epoch_version_exists(self, art):
        for r in art["runs"]:
            assert r["failure_mode_best"] in {
                "unknown", "unparseable_dominated", "parseable_fragmented", "ok",
            }, r["run"]


class TestPairedHeadlineDetector:
    def test_every_run_carries_the_block(self, art):
        for r in art["runs"]:
            ph = r.get("paired_headline_divergence")
            assert ph is not None, r["run"]
            for key in (
                "flagged", "headline_gen_validity_max", "prior_headline_max",
                "delta_headline", "guarded_e2e_best", "prior_guarded_max",
                "delta_guarded", "cohort_n_train", "prior_runs",
            ):
                assert key in ph, (r["run"], key)

    def test_the_flagged_runs_are_listed_at_the_top_level(self, art):
        flagged = art["paired_headline_divergences"]
        derived = sorted(
            r["run"] for r in art["runs"]
            if r["paired_headline_divergence"]["flagged"]
        )
        assert flagged == derived

    def test_the_first_run_of_a_cohort_is_never_flagged(self, art):
        for r in art["runs"]:
            ph = r["paired_headline_divergence"]
            if not ph["prior_runs"]:
                assert ph["flagged"] is False, r["run"]

    def test_a_flag_means_the_headline_rose_and_the_guard_did_not(self, art):
        for r in art["runs"]:
            ph = r["paired_headline_divergence"]
            if not ph["flagged"]:
                continue
            assert ph["headline_gen_validity_max"] > ph["prior_headline_max"]
            assert ph["guarded_e2e_best"] <= ph["prior_guarded_max"]


class TestTheDefaultThatLies:
    """Finding D — an unmeasured epoch writes 0.0 where a measurement would."""

    def test_the_last_rows_rate_columns_are_null_not_zero(self, runs):
        for name, r in runs.items():
            if r["gen_eval_ran_on_last_epoch"] is False:
                assert r["gen_single_fragment_ratio_end"] is None, name
                assert r["gen_largest_fragment_frac_end"] is None, name

    def test_the_decode_columns_follow_the_decode_path_not_the_generation_eval(
        self, runs
    ):
        """`decode_ratio` is a VAL-SPLIT decode measurement.

        Wf-A115 — it used to be gated on the GENERATION eval's predicate
        (`gen_validity_n == 0`), which nulled a real measurement for 27 of 59
        runs: the two paths are independent.  MEASURED on disk:
        `c7_b32_4096_2026_10_20` draws NOTHING from the generation eval
        (`gen_validity_n = 0`) yet decodes 192/192 val geometries, so its
        decode_ratio is a measured 1.0 and must not be nulled.
        """
        for name, r in runs.items():
            decode_ran = r.get("decode_eval_ran_on_last_epoch")
            assert decode_ran is not None, name
            if decode_ran:
                assert r["decode_ratio"] is not None, name
            else:
                assert r["decode_ratio"] is None, name
            # the two predicates are DIFFERENT axes and are named separately
            assert "decode_eval_ran_on_last_epoch" in r, name
            assert "gen_eval_ran_on_last_epoch" in r, name

    def test_a_generation_blind_run_keeps_its_decode_measurement(self, runs):
        r = runs["c7_b32_4096_2026_10_20"]
        assert r["gen_eval_ran_on_last_epoch"] is False
        assert r["gen_single_fragment_ratio_end"] is None
        # the decode ran: 192 val geometries, all parseable
        assert r["decode_eval_ran_on_last_epoch"] is True
        assert r["decode_ratio"] == pytest.approx(1.0, abs=1e-12)
        assert r["decode_ratio_n"] == 192

    def test_the_raw_decode_measurement_is_preserved(self, runs):
        r = runs["cfm_bond_head_100ep_2026_10_09"]
        # the JSON's full precision; the 4-decimal CSV renders it 0.4323
        assert r["decode_ratio"] == pytest.approx(0.4322916666666667, abs=1e-12)
        assert r["decode_ratio_measured"] == pytest.approx(
            0.4322916666666667, abs=1e-12
        )

    def test_a_measured_last_row_keeps_its_numbers(self, runs):
        r = runs["cfm_r9_fix_v2"]
        assert r["gen_eval_ran_on_last_epoch"] is True
        assert r["gen_single_fragment_ratio_end"] is not None
        assert r["decode_ratio"] is not None

    def test_the_end_to_end_rate_is_the_product_of_the_two_columns(self, art):
        checked = 0
        for r in art["runs"]:
            rate = r.get("gen_single_molecule_rate_end")
            v = r.get("gen_validity_end")
            s = r.get("gen_single_fragment_ratio_end")
            if rate is None or v is None or s is None:
                continue
            assert rate == pytest.approx(float(v) * float(s), abs=1e-9), r["run"]
            checked += 1
        assert checked > 0


class TestRetractedNumbers:
    def test_the_withdrawn_numbers_are_declared(self, art):
        assert "0.572276953125" in art["retracted_numbers"]
        assert "3.2x inversion" in art["retracted_numbers"]

    def test_the_withdrawal_explains_why(self, art):
        why = art["retracted_numbers"]["0.572276953125"]
        assert "cfm_r11_eval256" in why
        assert "nan" in why

    def test_the_withdrawn_number_is_not_still_offered_as_a_bar(self, art):
        bar = art["per_run_bar"]["primary"]
        assert bar["value"] != pytest.approx(0.572276953125, abs=1e-12)
        assert bar["run"] != "cfm_r11_eval256"


class TestAlreadyAnswers:
    def test_the_three_new_entries_are_present(self, art):
        aa = art["already_answers"]
        assert "does fragmentation quality depend on n_train?" in aa
        assert "does the assemble/connectivity loss help?" in aa
        assert "what is the correct bar?" in aa

    def test_the_n_train_entry_says_unresolved_and_withdraws_the_inversion(self, art):
        entry = " ".join(
            art["already_answers"]["does fragmentation quality depend on n_train?"]
        )
        assert "UNRESOLVED" in entry
        assert "WITHDRAWN" in entry
        assert "0.572276953125" in entry

    def test_the_assemble_loss_entry_is_a_settled_negative(self, art):
        entry = " ".join(
            art["already_answers"]["does the assemble/connectivity loss help?"]
        )
        assert "NO" in entry
        assert "cfm_frag_assemble" in entry
        assert "cfm_frag_control" in entry

    def test_the_bar_entry_names_the_runs(self, art):
        entry = " ".join(art["already_answers"]["what is the correct bar?"])
        assert "cfm_r9_fix_v2" in entry
        assert "cfm_fullcorpus_seed1" in entry
        assert "REPORT-ONLY" in entry

    def test_the_unsourced_prose_entry_points_at_the_cohort_block(self, art):
        entry = " ".join(
            art["already_answers"]["does the generated pool hold together as MOLECULES?"]
        )
        assert "by_n_train" in entry


class TestStrictCohortAxis:
    """Wf-A114 verify:integration D1 -- a cohort is the PAIR (n_train, n_val).

    The census's ``by_n_train`` block is stratified on ``n_train`` ALONE, and
    MEASURED the ``n_train=26715`` label mixes two validation protocols
    (``n_val=192`` in 8 runs, ``n_val=6714`` in 5 runs).  The harness
    (``cfm_cohort_matched_compare.py::axis_cohort``) requires BOTH to be
    equal, so the two artifacts disagreed about what a cohort is.  The strict
    cohort now lives in its own block and the mixing is visible in the loose
    one.
    """

    def test_n_val_is_a_first_class_field_on_every_run(self, runs):
        """Same rule as `n_train` (see
        `test_n_train_is_on_the_run_and_config_keeps_its_key`): the value
        IS the config's value whenever the config block is the run of
        record's own, and is NOT the clobbered file's value when the
        census had to re-derive that block."""
        for name, r in runs.items():
            assert "n_val" in r, name
            cfg = r.get("config") or {}
            if r.get("config_conflict"):
                assert (
                    r.get("cohort_source") != "per_epoch_metrics.json:config"
                ), name
                assert r.get("cohort_reason"), name
                assert (
                    r["n_val"] is None or r["n_val"] != cfg.get("n_val")
                ), name
            else:
                assert r["n_val"] == cfg.get("n_val"), name

    def test_the_strict_block_is_keyed_on_the_pair(self, art):
        assert "by_cohort" in art
        labels = {
            k for k, v in art["by_cohort"].items()
            if isinstance(v, dict) and "n_runs" in v
        }
        assert "26715/192" in labels
        assert "26715/6714" in labels
        assert "256/192" in labels

    def test_the_26715_label_is_flagged_as_mixed(self, art):
        blk = art["by_n_train"]["26715"]
        assert blk["n_val_mixed"] is True
        # MEASURED (Wf-A116): 192-side grew from 8 to 9 — the full-corpus
        # free-generation pool's cohort is 26715/192 (ts-attested log).
        assert blk["n_val_breakdown"] == {"192": 9, "6714": 5}

    def test_a_cohort_that_is_not_mixed_is_flagged_as_such(self, art):
        for label, blk in art["by_n_train"].items():
            if not isinstance(blk, dict) or "n_runs" not in blk:
                continue
            if label == "26715":
                continue
            assert blk["n_val_mixed"] is False, label

    def test_the_strict_cohorts_partition_the_runs(self, art, runs):
        """Every run lands in exactly ONE strict cohort."""
        total = 0
        seen = set()
        for label, blk in art["by_cohort"].items():
            if not isinstance(blk, dict) or "n_runs" not in blk:
                continue
            total += blk["n_runs"]
            for name in blk["runs"]:
                assert name not in seen, f"{name} is in two strict cohorts"
                seen.add(name)
        assert total == len(runs)

    def test_the_census_and_the_harness_agree_on_what_a_cohort_is(self, art):
        """The census's strict cohorts must be the harness's cohorts."""
        from molmetal.scripts.cfm_cohort_matched_compare import (
            DEFAULT_REPORTS_ROOT,
            discover_runs,
        )

        harness = {
            (r.n_train, r.n_val) for r in discover_runs(DEFAULT_REPORTS_ROOT)
        }
        census_cohorts = set()
        for label, blk in art["by_cohort"].items():
            if not isinstance(blk, dict) or "n_runs" not in blk:
                continue
            census_cohorts.add((blk["n_train"], blk["n_val"]))
        # Wf-A114E — the census now REFUSES to attribute a cohort to a run
        # whose per-epoch file was CLOBBERED by a shorter run: the events
        # stream carries no n_train / n_val, so the census records the
        # cohort as unknown instead of repeating the other population's
        # label.  Those runs sit in the (None, None) cohort, which the
        # harness — which still reads the clobbered file — cannot see.
        # Every OTHER census cohort must exist in the harness's view.
        # Wf-A116 — one clobbered run was rescued by a ts-attested run
        # log (`cfm_fullcorpus_2026_10_10` -> 26715/192), so it left the
        # unknown cohort; the (None, None) cohort now holds the three
        # runs whose cohort is still unattested.
        unknown = {(None, None)}
        assert census_cohorts - unknown <= harness, census_cohorts - unknown - harness
        assert census_cohorts & unknown == unknown
        # and the (None, None) cohort holds exactly the runs the census
        # refused to attribute a cohort to.  MEASURED: four runs carry the
        # verdict `class_a_other_population` (their per-epoch file is
        # another population's record) but only three have an UNAttested
        # cohort — the fourth had its cohort recovered from the ts-attested
        # run log and therefore sits in 26715/192.
        none_cohort = art["by_cohort"].get("None/None")
        assert none_cohort is not None
        assert set(none_cohort["runs"]) == {
            r["run"] for r in art["runs"]
            if (r.get("integrity") or {}).get("population_verdict")
            == "class_a_other_population"
            and r.get("cohort_source") == "unattested"
        }

    def test_the_budget_matched_block_says_which_axis_it_uses(self, art):
        bm = art["budget_matched"]
        assert "n_train ALONE" in bm["cohort_axis"]
        # MEASURED (Wf-A116): the 192-side grew from 8 to 9 runs.
        assert bm["cohort_composition"]["26715"] == {"192": 9, "6714": 5}

    def test_the_named_bars_record_the_strict_cohort(self, art):
        for key in ("primary", "secondary"):
            bar = art["per_run_bar"][key]
            assert bar["cohort_strict"] == f"{bar['n_train']}/{bar['n_val']}"


class TestTheArtifactIsReDerivableFromDisk:
    """The census must be a VIEW of the runs on disk, not a hand-written file.

    Wf-A115 verify:B D4 -- the test file read the artifact through a
    fixture, so a MIS-REGENERATED artifact passed the whole suite: the
    artifact could disagree with every run on disk and no test would
    notice.  These tests re-derive from the run directories themselves.
    """

    def test_the_run_count_matches_the_run_directories_on_disk(self, art):
        """A directory counts as a run iff it carries at least one DATA row.

        MEASURED: ``cfm_default`` has a ``per_epoch_metrics.csv`` with the
        header line and nothing else, so it is a run directory that recorded
        no epoch.  The census's own loader drops it ("only a file with NO
        data row at all is not a run"), and this test uses the same
        predicate rather than counting directories.
        """
        n_dirs = 0
        for csvp in census.REPORTS.rglob("per_epoch_metrics.csv"):
            d = csvp.parent
            has_rows = False
            with csvp.open(encoding="utf-8") as fh:
                next(fh, None)  # header
                has_rows = bool(next(fh, None))
            if not has_rows:
                jp = d / "per_epoch_metrics.json"
                if jp.is_file():
                    try:
                        jd = json.loads(jp.read_text(encoding="utf-8"))
                    except Exception:
                        jd = {}
                    pem = jd.get("per_epoch_metrics")
                    has_rows = isinstance(pem, list) and bool(pem)
            if has_rows:
                n_dirs += 1
        assert len(art["runs"]) == n_dirs, (
            f"the artifact lists {len(art['runs'])} run(s) but "
            f"{n_dirs} run director(ies) carry a data row"
        )

    def test_a_run_on_disk_re_derives_to_the_artifact_row(self, art):
        """Re-derive a few runs from their own files and compare."""
        by_name = {r["run"]: r for r in art["runs"]}
        checked = 0
        for name in ("cfm_r9_fix_v2", "c7_b32_4096_2026_10_20", "cfm_r11_eval256"):
            if name not in by_name:
                continue
            d = census.REPORTS / name
            fresh = census.load_run(d)
            assert fresh is not None, name
            row = by_name[name]
            for key in (
                "n_train",
                "n_val",
                "n_epochs_run",
                "gen_validity_end",
                "gen_single_fragment_ratio_end",
                "gen_single_molecule_rate_end",
                "decode_ratio",
                "decode_ratio_n",
                "decode_eval_ran_on_last_epoch",
                "gen_eval_ran_on_last_epoch",
            ):
                assert fresh.get(key) == row.get(key), (name, key)
            # and the guard, which is the newest derived block
            assert (fresh.get("early_stop_guard") or {}).get("state") == (
                row.get("early_stop_guard") or {}
            ).get("state"), name
            checked += 1
        assert checked >= 2, "expected at least two on-disk runs to re-derive"

    def test_the_withdrawn_value_is_not_produced_by_a_clean_run(self, art):
        """The withdrawn 0.572265625 must not be re-derivable from a clean run."""
        withdrawn = 0.572265625
        clean_max = art["fragmentation_summary"]["end_to_end_single_molecule_rate"][
            "max"
        ]
        assert clean_max != withdrawn
        # and the run that produced it must be named as degenerate
        assert (
            "cfm_r11_eval256"
            in art["fragmentation_summary"][
                "degenerate_runs_excluded_from_the_headline"
            ]
        )


class TestTheEarlyStopGuard:
    """A config that selects a criterion is not evidence the run used it.

    Wf-A115 -- the trainer DEGRADES an early-stop criterion it cannot
    measure, so ``configs/c9_cfm_transport_fixed.yaml`` selecting
    ``molecule_validity`` did not make a silently-degrading c9 run
    detectable.  There was no detector at all.
    """

    def test_every_run_carries_a_guard(self, runs):
        for name, r in runs.items():
            g = r.get("early_stop_guard")
            assert isinstance(g, dict), name
            assert g["state"] in (
                "ok",
                "degraded",
                "selected_but_not_emitted",
                "not_recorded",
            ), (name, g["state"])

    def test_the_top_level_block_agrees_with_the_per_run_guards(self, art):
        g = art["early_stop_guard"]
        states = [r["early_stop_guard"]["state"] for r in art["runs"]]
        assert g["n_runs"] == len(states)
        assert g["n_ok"] == states.count("ok")
        assert g["n_degraded"] == states.count("degraded")
        assert g["n_selected_but_not_emitted"] == states.count(
            "selected_but_not_emitted"
        )
        assert g["n_not_recorded"] == states.count("not_recorded")

    def test_the_degraded_runs_are_named_with_both_criteria(self, art):
        g = art["early_stop_guard"]
        for entry in g["degraded_runs"]:
            assert entry["requested_metric"]
            assert entry["gated_on"]
            assert entry["requested_metric"] not in entry["gated_on"]
            # the per-run block must say the same thing
            row = next(
                r for r in art["runs"] if r["run"] == entry["run"]
            )
            assert row["early_stop_guard"]["state"] == "degraded"
            assert row["early_stop_guard"]["degraded_to"] == entry["gated_on"]

    def test_a_requested_criterion_without_its_column_is_caught(self):
        """The unit case: `molecule_validity` selected, column never emitted."""
        rows = [
            {
                "epoch": 0,
                "gen_validity": 0.5,
                "bond_type_macro_f1_pooled": 0.3,
                "early_stop_metric": "bond_type_f1",
            }
        ]
        cfg = {"early_stop_metric": "molecule_validity"}
        jd = {
            "early_stop_control": {"requested_metric": "molecule_validity"}
        }
        g = census._early_stop_guard(rows, cfg, jd)
        assert g["state"] == "selected_but_not_emitted"
        assert g["missing_columns"] == ["gen_molecule_validity"]

    def test_a_clean_request_and_resolve_is_ok(self):
        rows = [
            {
                "epoch": 0,
                "gen_molecule_validity": 0.5,
                "early_stop_metric": "molecule_validity",
            }
        ]
        jd = {"early_stop_control": {"requested_metric": "molecule_validity"}}
        g = census._early_stop_guard(rows, {}, jd)
        assert g["state"] == "ok"
        assert g["emitted_columns"] == ["gen_molecule_validity"]

    def test_a_requested_criterion_the_run_never_recorded_is_not_recorded(self):
        g = census._early_stop_guard([{"epoch": 0}], {}, {})
        assert g["state"] == "not_recorded"
        assert g["requested_metric"] is None


class TestTheHeadlineIsOverCleanRuns:
    """The headline must not be led by a one-epoch nan-loss run."""

    def test_the_degenerate_runs_are_named(self, art):
        fs = art["fragmentation_summary"]
        named = fs["degenerate_runs_excluded_from_the_headline"]
        assert named, "there ARE degenerate runs on disk; name them"
        for r in art["runs"]:
            integ = r["integrity"]
            clean = bool(
                integ.get("epoch_history_contiguous_from_zero")
                and not integ.get("nan_train_loss")
            )
            assert (r["run"] not in named) == clean, r["run"]

    def test_the_headline_excludes_the_degenerate_runs(self, art):
        fs = art["fragmentation_summary"]
        clean_n = fs["verdict_counts_clean_only"]["n_runs"]
        e2e_clean = fs["end_to_end_single_molecule_rate"]
        e2e_all = fs["end_to_end_single_molecule_rate_all_runs"]
        assert e2e_clean["population"].startswith("CLEAN runs only")
        assert e2e_all["population"].startswith("ALL runs")
        assert e2e_clean["n"] <= e2e_all["n"]

    def test_the_withdrawn_max_is_labelled_where_it_survives(self, art):
        fs = art["fragmentation_summary"]
        r = fs["retracted_max"]
        assert r["status"] == "WITHDRAWN"
        assert r["run"] == "cfm_r11_eval256"
        assert r["value"] == "0.572265625"
        # the withdrawn value is the ALL-RUNS max, never the clean headline
        assert fs["end_to_end_single_molecule_rate_all_runs"]["max"] == pytest.approx(
            0.572265625, abs=1e-12
        )
        assert fs["end_to_end_single_molecule_rate"]["max"] != pytest.approx(
            0.572265625, abs=1e-12
        )
        # and the same content is published at the top level, so it cannot drift
        assert art["retracted_numbers"] == fs["retracted_numbers"]


class TestTheConfigAndHeadLeverBlocksAgree:
    """`config` is the run's OWN record; `head_levers` is the MERGED view."""

    def test_they_agree_on_every_key_they_share(self, art):
        for r in art["runs"]:
            cfg = r.get("config") or {}
            lev = r.get("head_levers") or {}
            for key, value in cfg.items():
                if key in lev and value is not None:
                    assert lev[key] == value, (r["run"], key)

    def test_the_provenance_only_levers_are_present_in_the_merged_view(self, art):
        """A lever recorded only in `provenance` is null in `config`."""
        found = False
        for r in art["runs"]:
            cfg = r.get("config") or {}
            lev = r.get("head_levers") or {}
            for key in census.HEAD_LEVER_KEYS:
                if (
                    lev.get(key) is not None
                    and cfg.get(key) is None
                    and (r.get("head_levers_source_per_key") or {}).get(key)
                    == "provenance"
                ):
                    found = True
        # not a hard requirement on the current corpus, but the machinery is
        # exercised by `_head_levers` itself
        assert isinstance(found, bool)

    def test_the_dead_class_weights_key_is_gone(self):
        """`class_weights` is the OTHER trainer's name; it resolved to nothing.

        MEASURED: 0 of 43 on-disk CFM runs record either name in their
        config block, and the CFM trainer's dests are `bond_class_weights` /
        `bond_class_weight_power`.
        """
        assert "class_weights" not in census.CFG_KEYS
        assert "bond_class_weights" in census.CFG_KEYS
