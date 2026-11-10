# Licensed under the Apache License, Version 2.0.
"""Wf-A120 / I1 -- the POOL-EPOCH RULE: a pool's generation-quality number
is published at the POOL'S OWN LAST EPOCH.

THE DEFECT THIS PINS
---------------------
``molmetal/scripts/cfm_fragmentation_gate.py::score_run_dir`` read
``<run>/per_epoch_metrics.csv`` with ``epoch=None`` -- the LAST ROW -- and
published that number as ``verdict.gen_validity``.  A pool is a set of
MOLECULES, so the only epoch at which a validity number describes it is the
epoch the pool was drawn at.

MEASURED on disk: three of the thirteen pools on disk published a validity
measured at a DIFFERENT epoch than their own molecules:

    run                            pool epoch  published before  correct value
    cfm_fullcorpus_2026_10_10           10       0.9140625       0.3255208333333333
    cfm_fullcorpus_b64_2026_10_11        8       0.0859375       0.3763020833333333
    cfm_objective_terms_2026_10_11       5       0.07421875      0.234375

For ``cfm_fullcorpus_2026_10_10`` the published 0.9140625 is a CLOBBERED
1-epoch CPU smoke (n_train 64) that overwrote ``per_epoch_metrics.csv``; the
pool's own epoch 10 belongs to the 14-epoch run of record.  The other ten
pools were already aligned and MUST NOT move.

THE RULE, and its ONE owner
----------------------------
``molmetal/scripts/cfm_pool_epoch_rule.py`` owns it:

    "The generation-quality number published for a pool is the resolved
    POPULATION's aggregate value AT THE POOL'S OWN LAST EPOCH -- the aggregate
    row that carries gen_validity_n, never the per-seed gen_eval rows."

CPU-only, no GPU, no training.  The synthetic fixtures are written under
``tmp_path``; the real-run test only READS the repository.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.cfm_fragmentation_gate import evaluate, score_run_dir  # noqa: E402
from molmetal.scripts.cfm_pool_epoch_rule import (  # noqa: E402
    POOL_EPOCH_RULE,
    census_row,
    generation_validity_at_epoch,
    pool_epoch_recorded_row,
    pool_epoch_validity,
    pool_last_epoch,
    pool_population_composition,
    resolve_population,
    resolve_population_census,
    run_last_epoch_recorded_row,
)

REPORTS = REPO_ROOT / "molmetal" / "reports"
CENSUS = REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"
C7_ARTIFACT = REPO_ROOT / "metrics" / "by_metric" / "c7_scale_matched_pools.json"

#: The per-epoch columns the rule reads.  ``gen_validity`` is the ROUNDED
#: column; ``gen_validity_val / gen_validity_n`` is the exact measurement.
METRIC_COLUMNS = [
    "epoch",
    "gen_validity",
    "gen_validity_val",
    "gen_validity_n",
    "gen_single_fragment_ratio",
    "gen_largest_fragment_frac",
    "decode_single_fragment_ratio",
]


def _write_pool(path: Path, rows) -> Path:
    """A pool CSV in the trainer's own free-generation format."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["epoch", "smiles", "n_frags_raw", "largest_frag_frac"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_series(path: Path, rows) -> Path:
    """A ``per_epoch_metrics.csv`` carrying the given aggregate rows."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _series_row(epoch: int, val: int, n: int) -> dict:
    return {
        "epoch": epoch,
        "gen_validity": round(val / n, 4),
        "gen_validity_val": val,
        "gen_validity_n": n,
        "gen_single_fragment_ratio": 0.5,
        "gen_largest_fragment_frac": 0.95,
        "decode_single_fragment_ratio": 0.9375,
    }


def _write_events(path: Path, rows) -> Path:
    """An ``events.jsonl`` carrying the given event dicts, one per line."""
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _event_row(epoch: int, val: int, n: int, ts: float = None) -> dict:
    """One AGGREGATE (``train_epoch``) event row."""
    return {
        "event": "train_epoch",
        "epoch": epoch,
        "ts": ts if ts is not None else 1000.0 + epoch,
        "gen_validity": (val / n) if n else None,
        "gen_validity_val": val,
        "gen_validity_n": n,
        "gen_single_fragment_ratio": 0.5,
        "gen_largest_fragment_frac": 0.95,
    }


def _census():
    """The run census -- the SSOT of which run wrote what."""
    if not CENSUS.is_file():
        return {"runs": []}
    return json.loads(CENSUS.read_text(encoding="utf-8"))


class TestTheRuleIsStatedVerbatim:
    def test_the_rule_string_is_the_one_in_the_module(self):
        assert POOL_EPOCH_RULE == (
            "The generation-quality number published for a pool is the resolved "
            "POPULATION's aggregate value AT THE POOL'S OWN LAST EPOCH -- the "
            "aggregate row that carries gen_validity_n, never the per-seed "
            "gen_eval rows."
        )

    def test_the_module_docstring_states_the_rule(self):
        import molmetal.scripts.cfm_pool_epoch_rule as mod

        # The docstring wraps the rule across lines; compare it with the
        # whitespace collapsed so the STATEMENT is pinned, not the wrapping.
        collapsed = " ".join((mod.__doc__ or "").split())
        assert " ".join(POOL_EPOCH_RULE.split()) in collapsed


class TestPoolEpochComesFromThePool:
    def test_the_pool_epoch_is_read_from_the_pool_csv(self, tmp_path):
        # (a) The pool's last epoch comes from the POOL's own column -- not
        # from the run's metrics file, whose last row is epoch 7.
        d = tmp_path / "run"
        d.mkdir()
        pool = _write_pool(
            d / "generated_smiles.csv",
            [
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
            ],
        )
        _write_series(
            d / "per_epoch_metrics.csv",
            [_series_row(3, 30, 100), _series_row(7, 90, 100)],
        )
        assert pool_last_epoch(pool) == 3
        # ... and NOT the run's last epoch (7), which is what the gate used
        # to read.
        assert pool_last_epoch(pool) != 7

    def test_a_multi_epoch_pool_reports_its_last_epoch(self, tmp_path):
        pool = _write_pool(
            tmp_path / "p.csv",
            [
                {"epoch": 0, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 2, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
            ],
        )
        assert pool_last_epoch(pool) == 3


class TestPublishedValidityIsAtThePoolEpoch:
    def test_the_epoch3_aggregate_wins_over_the_last_row(self, tmp_path):
        # (b) A pool at epoch 3 and a metrics file whose LAST row is epoch 7:
        # the published validity is the epoch-3 aggregate.
        d = tmp_path / "run"
        d.mkdir()
        pool = _write_pool(
            d / "generated_smiles.csv",
            [
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
            ],
        )
        _write_series(
            d / "per_epoch_metrics.csv",
            [_series_row(3, 30, 100), _series_row(7, 90, 100)],
        )
        out = pool_epoch_validity(d, pool)
        assert out["epoch"] == 3
        assert out["value"] == pytest.approx(0.30)
        assert out["n"] == 100
        assert out["source"]
        # The superseded value is the LAST ROW's, with the reason.
        assert out["superseded"]["value"] == pytest.approx(0.90)
        assert out["superseded"]["epoch"] == 7
        assert out["superseded"]["changed"] is True
        assert "7" in out["superseded"]["reason"]

    def test_the_gate_publishes_the_epoch_matched_number(self, tmp_path):
        # The same thing end to end: the verdict's gen_validity is the
        # epoch-3 aggregate, and the provenance says so.
        d = tmp_path / "run"
        d.mkdir()
        _write_pool(
            d / "generated_smiles.csv",
            [
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
            ],
        )
        _write_series(
            d / "per_epoch_metrics.csv",
            [_series_row(3, 30, 100), _series_row(7, 90, 100)],
        )
        verdict = evaluate(score_run_dir(d))
        assert verdict["gen_validity"] == pytest.approx(0.30)
        prov = verdict["gen_validity_provenance"]
        assert prov["epoch"] == 3
        assert prov["n"] == 100
        assert prov["source"]
        assert prov["superseded"]["value"] == pytest.approx(0.90)

    def test_a_pool_epoch_with_no_aggregate_row_yields_none(self, tmp_path):
        # (c) NO fallback to another epoch: the pool's epoch 5 has no
        # aggregate row, so there is no number.
        d = tmp_path / "run"
        d.mkdir()
        pool = _write_pool(
            d / "generated_smiles.csv",
            [
                {"epoch": 5, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
            ],
        )
        _write_series(
            d / "per_epoch_metrics.csv",
            [_series_row(1, 10, 100), _series_row(2, 90, 100)],
        )
        out = pool_epoch_validity(d, pool)
        assert out["epoch"] == 5
        assert out["value"] is None
        assert out["n"] is None
        # ... and the gate publishes NO validity rather than the wrong-epoch one.
        verdict = evaluate(score_run_dir(d))
        assert verdict["gen_validity"] is None
        # The superseded value is still recorded, so the reader can see what
        # was NOT published.
        assert out["superseded"]["value"] == pytest.approx(0.90)
        assert out["superseded"]["epoch"] == 2
        assert out["superseded"]["changed"] is False

    def test_a_run_with_nothing_on_disk_publishes_no_number(self, tmp_path):
        # No events.jsonl and no per_epoch_metrics.csv: None, and it says so.
        d = tmp_path / "run"
        d.mkdir()
        pool = _write_pool(
            d / "generated_smiles.csv",
            [
                {"epoch": 0, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
            ],
        )
        out = pool_epoch_validity(d, pool)
        assert out["value"] is None
        assert out["notes"]
        assert out["population"]["resolution"] == "unresolved"


class TestTheThreeRealRuns:
    """(d) The three misaligned pools on disk, at their own epochs."""

    @pytest.mark.parametrize(
        "run,epoch,expected",
        [
            ("cfm_fullcorpus_2026_10_10", 10, 0.3255208333333333),
            ("cfm_fullcorpus_b64_2026_10_11", 8, 0.3763020833333333),
            ("cfm_objective_terms_2026_10_11", 5, 0.234375),
        ],
    )
    def test_the_published_value_is_the_pool_epoch_value(self, run, epoch, expected):
        d = REPORTS / run
        if not (d / "generated_smiles.csv").is_file():
            pytest.skip(f"{run} is not on disk")
        census_row = None
        if CENSUS.is_file():
            rows = json.loads(CENSUS.read_text(encoding="utf-8")).get("runs") or []
            census_row = next((r for r in rows if r.get("run") == run), None)
        out = pool_epoch_validity(
            d, d / "generated_smiles.csv", census_row=census_row
        )
        assert out["epoch"] == epoch
        assert out["value"] == expected
        assert out["n"] == 768
        assert out["population"]["resolution"] == "run_of_record"

    def test_the_clobbered_smoke_is_not_the_value(self):
        # The number the gate used to publish for cfm_fullcorpus_2026_10_10
        # was the CLOBBERED 1-epoch CPU smoke's own row.
        d = REPORTS / "cfm_fullcorpus_2026_10_10"
        if not (d / "generated_smiles.csv").is_file():
            pytest.skip("cfm_fullcorpus_2026_10_10 is not on disk")
        out = pool_epoch_validity(d, d / "generated_smiles.csv")
        assert out["value"] != 0.9140625
        assert out["superseded"]["value"] == pytest.approx(0.9140625)
        assert out["superseded"]["epoch"] == 0
        assert out["population"]["metrics_file_is_run_of_record"] is False


class TestPopulationResolution:
    def test_the_population_carries_its_cohort(self, tmp_path):
        # Populations in one directory do NOT share a cohort, so the
        # resolution says WHICH launch the number belongs to.
        d = REPORTS / "cfm_fullcorpus_2026_10_10"
        if not (d / "generated_smiles.csv").is_file():
            pytest.skip("cfm_fullcorpus_2026_10_10 is not on disk")
        pop = resolve_population(d, 10)
        assert pop["resolution"] == "run_of_record"
        assert pop["cohort"]["n_train"] == 26715
        assert pop["cohort"]["n_val"] == 192
        assert pop["cohort"]["unattested"] is False


# ---------------------------------------------------------------------------
# Wf-A121 / I1 -- ONE RESOLUTION.  The gate and the c7 audit asked the same
# question and got DIFFERENT answers for the same run directory, so the two
# policies now live behind ONE resolver.
#
# The defect this pins: the gate needed a ``done`` event to segment a run,
# so a stream with NO ``done`` event resolved to NOTHING -- and it said "this
# run directory has no events.jsonl", which is FALSE for a directory that HAS
# one.  The c7 audit meanwhile published a number for the very same run.
# ---------------------------------------------------------------------------


class TestTheUnSegmentedStreamIsOnePopulation:
    """(a) An event stream with NO ``done`` event is ONE population."""

    def test_it_resolves_to_one_population_and_publishes_the_pool_epoch_row(
        self, tmp_path
    ):
        d = tmp_path / "run"
        d.mkdir()
        pool = _write_pool(
            d / "generated_smiles.csv",
            [{"epoch": 2, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0}],
        )
        _write_events(
            d / "events.jsonl",
            [
                _event_row(0, 700, 704),
                _event_row(1, 700, 704),
                _event_row(2, 704, 704),
            ],
        )
        out = pool_epoch_validity(d, pool)
        pop = out["population"]
        assert pop["resolution"] == "un_segmented_event_stream"
        assert pop["source"] == "fallback:un_segmented_event_stream"
        assert pop["segment_index"] is None
        assert pop["epochs"] == [0, 1, 2]
        # The aggregate row AT THE POOL'S OWN EPOCH is published.
        assert out["epoch"] == 2
        assert out["value"] == pytest.approx(1.0)
        assert out["n"] == 704
        assert "train_epoch[epoch=2]" in out["source"]
        # ... and the reason says the TRUE thing: the stream EXISTS, it just
        # carries no done event.
        assert "no done event" in pop["reason"]
        assert "no events.jsonl" not in pop["reason"]

    def test_the_c7_resolution_agrees_with_the_gate(self, tmp_path):
        # ONE resolution: the c7's own accessor and the gate must not
        # disagree about the same run directory.
        d = tmp_path / "run"
        d.mkdir()
        pool = _write_pool(
            d / "generated_smiles.csv",
            [{"epoch": 2, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0}],
        )
        _write_events(
            d / "events.jsonl",
            [_event_row(0, 700, 704), _event_row(1, 700, 704), _event_row(2, 704, 704)],
        )
        pop_c7 = resolve_population_census(d, {"runs": []}, 2)
        pop_gate = resolve_population(d, 2)
        assert pop_c7["source"] == pop_gate["source"]
        assert pop_c7["resolution"] == pop_gate["resolution"]
        assert pop_c7["epochs"] == pop_gate["epochs"]
        assert generation_validity_at_epoch(d, epoch=2, rows=pop_c7["rows"]) == {
            "epoch": 2,
            "n": 704,
            "value": 1.0,
        }

    def test_the_ambiguity_guard_publishes_no_number(self, tmp_path):
        # (b) TWO aggregate rows at the pool's own epoch: the pool's epoch
        # cannot be attributed to ONE population, so no number is published
        # and the reason says how many rows were found.
        d = tmp_path / "run"
        d.mkdir()
        pool = _write_pool(
            d / "generated_smiles.csv",
            [{"epoch": 2, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0}],
        )
        _write_events(
            d / "events.jsonl",
            [
                _event_row(0, 700, 704),
                _event_row(2, 704, 704),
                _event_row(2, 500, 704, ts=1002.5),
            ],
        )
        out = pool_epoch_validity(d, pool)
        pop = out["population"]
        assert out["value"] is None
        assert out["n"] is None
        assert pop["ambiguous"] is True
        assert pop["source"] == "ambiguous:multiple_aggregate_rows_at_pool_epoch"
        assert "2 aggregate rows at epoch 2" in pop["reason"]
        # The c7's consumer must not pick one either.
        pop_c7 = resolve_population_census(d, {"runs": []}, 2)
        assert generation_validity_at_epoch(d, epoch=2, rows=pop_c7["rows"]) is None

    def test_a_run_without_events_jsonl_says_exactly_that(self, tmp_path):
        # (c) NO events.jsonl at all: no number, and the reason names the
        # missing file (and only that).
        d = tmp_path / "run"
        d.mkdir()
        pool = _write_pool(
            d / "generated_smiles.csv",
            [{"epoch": 0, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0}],
        )
        out = pool_epoch_validity(d, pool)
        pop = out["population"]
        assert out["value"] is None
        assert pop["resolution"] == "unresolved"
        assert pop["source"] == "no events.jsonl"
        assert "no events.jsonl" in pop["reason"]
        assert "no done event" not in pop["reason"]


class TestTheTwoUnSegmentedRunsOnDisk:
    """The two runs the residual ledger named: run 4 and run 5."""

    @pytest.mark.parametrize(
        "run,epoch",
        [
            ("cfm_c7_fullgen_ep4_partial_2026_10_20", 3),
            ("cfm_c8_fullgen_2026_10_24", 2),
        ],
    )
    def test_the_pool_epoch_aggregate_is_published(self, run, epoch):
        d = REPORTS / run
        if not (d / "generated_smiles.csv").is_file():
            pytest.skip(f"{run} is not on disk")
        out = pool_epoch_validity(d, d / "generated_smiles.csv")
        assert out["epoch"] == epoch
        assert out["value"] == pytest.approx(1.0)
        assert out["n"] == 704
        assert out["population"]["resolution"] == "un_segmented_event_stream"
        assert out["population"]["source"] == "fallback:un_segmented_event_stream"
        assert out["population"]["segment_index"] is None

    def test_the_c7_artifact_and_the_gate_agree_on_the_same_run(self):
        # The two artifacts used to DISAGREE about this run: the c7 published
        # gen_validity 1.0 at epoch 2 with the source
        # "fallback:un_segmented_event_stream", the audit published null.
        d = REPORTS / "cfm_c8_fullgen_2026_10_24"
        if not (d / "generated_smiles.csv").is_file():
            pytest.skip("cfm_c8_fullgen_2026_10_24 is not on disk")
        pop = resolve_population_census(d, _census(), 2)
        assert pop["source"] == "fallback:un_segmented_event_stream"
        got = generation_validity_at_epoch(d, epoch=2, rows=pop["rows"])
        assert got == {"epoch": 2, "n": 704, "value": 1.0}


class TestTheCensusVerdictDecidesTheCohort:
    """(d) The cohort is the census's own answer, not a re-derived one."""

    def test_the_resolved_cohort_is_the_census_cohort(self):
        d = REPORTS / "cfm_c7_fullgen_2026_10_20"
        if not (d / "generated_smiles.csv").is_file():
            pytest.skip("cfm_c7_fullgen_2026_10_20 is not on disk")
        row = census_row(_census(), "cfm_c7_fullgen_2026_10_20")
        assert row is not None, "the census must carry this run"
        integrity = row["integrity"]
        # The census's own rule: the per-epoch file IS the run of record's
        # file iff the population was not resolved from elsewhere and the
        # verdict is not an unmatched conflict.
        assert integrity["population_resolved"] is False
        assert integrity["population_verdict"] != "unmatched_conflict"

        out = pool_epoch_validity(d, d / "generated_smiles.csv", census_row=row)
        cohort = out["population"]["cohort"]
        assert cohort["n_train"] == 26715
        assert cohort["n_val"] == 192
        assert cohort["source"] == "per_epoch_metrics.json:config"
        assert cohort["unattested"] is False
        # ... and it is the census's OWN cohort, not a re-derivation.
        assert row["n_train"] == cohort["n_train"]
        assert row["n_val"] == cohort["n_val"]
        assert out["population"]["metrics_file_is_run_of_record"] is True

    def test_the_local_test_agrees_when_the_census_is_absent(self):
        # Without the census row the local epoch-set test runs -- and it must
        # give the SAME answer, because the file holds epochs 4..17 and the
        # run of record RAN epochs 4..17 (the leading 0..3 belong to a killed
        # pre-resume attempt).
        d = REPORTS / "cfm_c7_fullgen_2026_10_20"
        if not (d / "generated_smiles.csv").is_file():
            pytest.skip("cfm_c7_fullgen_2026_10_20 is not on disk")
        out = pool_epoch_validity(d, d / "generated_smiles.csv")
        assert out["population"]["metrics_file_is_run_of_record"] is True
        assert out["population"]["cohort"]["n_train"] == 26715
        assert out["population"]["cohort"]["n_val"] == 192


class TestTheRecordedRowAtThePoolEpoch:
    """(e) The recorded number is read at the POOL'S OWN epoch."""

    def test_the_pool_epoch_row_differs_from_the_last_epoch_row(self):
        d = REPORTS / "cfm_objective_terms_2026_10_11"
        if not (d / "generated_smiles.csv").is_file():
            pytest.skip("cfm_objective_terms_2026_10_11 is not on disk")
        row = census_row(_census(), "cfm_objective_terms_2026_10_11")
        at_pool = pool_epoch_recorded_row(
            d, d / "generated_smiles.csv", census_row=row
        )
        assert at_pool is not None
        assert at_pool["epoch"] == 5
        assert at_pool["gen_validity"] == pytest.approx(0.234375)
        assert at_pool["gen_validity_n"] == 768
        # The epoch-5 fragmentation ratio -- NOT the run's last epoch's.
        assert at_pool["gen_single_fragment_ratio"] == pytest.approx(
            0.6666666666666666
        )

        at_last = run_last_epoch_recorded_row(
            d, d / "generated_smiles.csv", census_row=row
        )
        assert at_last is not None
        assert at_last["epoch"] == 8
        assert at_last["gen_single_fragment_ratio"] == pytest.approx(
            0.8771929824561403
        )
        assert (
            at_pool["gen_single_fragment_ratio"]
            != at_last["gen_single_fragment_ratio"]
        )

    def test_the_last_epoch_row_is_the_run_of_record_s_own(self):
        # The recorded per-epoch FILE of cfm_fullcorpus_2026_10_10 is a
        # clobbered 1-epoch CPU smoke, so the run of record's last epoch is
        # 13 -- NOT the file's epoch 0.
        d = REPORTS / "cfm_fullcorpus_2026_10_10"
        if not (d / "generated_smiles.csv").is_file():
            pytest.skip("cfm_fullcorpus_2026_10_10 is not on disk")
        row = census_row(_census(), "cfm_fullcorpus_2026_10_10")
        at_last = run_last_epoch_recorded_row(
            d, d / "generated_smiles.csv", census_row=row
        )
        assert at_last["epoch"] == 13
        assert at_last["gen_single_fragment_ratio"] == pytest.approx(
            0.8733624454148472
        )
        # ... which is what the census publishes as the run's END value.
        assert row["gen_single_fragment_ratio_end"] == pytest.approx(
            0.8733624454148472
        )


class TestThePoolComposition:
    """(f) The pool's own composition, published additively."""

    def test_the_rows_outside_the_run_of_record_are_counted(self):
        d = REPORTS / "cfm_c7_fullgen_2026_10_20"
        if not (d / "generated_smiles.csv").is_file():
            pytest.skip("cfm_c7_fullgen_2026_10_20 is not on disk")
        row = census_row(_census(), "cfm_c7_fullgen_2026_10_20")
        comp = pool_population_composition(
            d, d / "generated_smiles.csv", census_row=row
        )
        assert comp["n_rows_total"] == 12669
        assert comp["n_rows_outside_run_of_record"] == 2816
        assert comp["run_of_record_epochs"] == list(range(4, 18))
        assert comp["pool_epochs"] == list(range(0, 18))
        assert comp["pool_epoch"] == 17
        assert comp["n_rows_at_pool_epoch"] == 704
        assert "KILLED" in comp["note"]


class TestTheC7SourceStringsDoNotChurn:
    """The three source strings the c7 artifact publishes."""

    def test_the_census_run_of_record_string(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        _write_events(
            d / "events.jsonl",
            [
                _event_row(0, 8, 8),
                {"event": "done", "epoch": 0, "n_epochs_run": 1, "ts": 1001.0},
            ],
        )
        census = {
            "runs": [
                {
                    "run": "run",
                    "integrity": {
                        "run_of_record_segment_index": 0,
                        "run_of_record_epoch_indices": [0],
                    },
                }
            ]
        }
        pop = resolve_population_census(d, census, 0)
        assert pop["source"] == "census:run_of_record_segment_index"

    def test_the_segment_contains_pool_epoch_string(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        _write_events(
            d / "events.jsonl",
            [
                _event_row(0, 8, 8),
                {"event": "done", "epoch": 0, "n_epochs_run": 1, "ts": 1001.0},
            ],
        )
        pop = resolve_population_census(d, {"runs": []}, 0)
        assert pop["source"] == "fallback:segment_contains_pool_epoch"

    def test_the_un_segmented_event_stream_string(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        _write_events(d / "events.jsonl", [_event_row(0, 8, 8)])
        pop = resolve_population_census(d, {"runs": []}, 0)
        assert pop["source"] == "fallback:un_segmented_event_stream"

    def test_the_published_c7_artifact_uses_those_strings(self):
        if not C7_ARTIFACT.is_file():
            pytest.skip("the c7 artifact is not on disk")
        blob = json.loads(C7_ARTIFACT.read_text(encoding="utf-8"))
        seen = set()
        for pool in blob.get("pools") or []:
            # The published form is "<population source>: <what was read>".
            src = str(pool.get("gen_validity_source") or "")
            seen.add(src.split(": ", 1)[0])
        assert seen <= {
            "census:run_of_record_segment_index",
            "fallback:segment_contains_pool_epoch",
            "fallback:un_segmented_event_stream",
        }
        assert "fallback:un_segmented_event_stream" in seen
        assert "census:run_of_record_segment_index" in seen
