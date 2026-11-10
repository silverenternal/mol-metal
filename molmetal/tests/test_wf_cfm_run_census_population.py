# SPDX-License-Identifier: Apache-2.0
"""Wf-A114E — the census must score the POPULATION it names, not a clobbered one.

The trainer writes ``per_epoch_metrics.json`` ONCE, at the end of training,
to a FIXED path, so two runs launched into the same ``--out-dir`` overwrite
each other's record and the survivor is whichever finished LAST.  The census
read that one file and scored it under the directory's name, with no warning.

``events.jsonl`` is append-only and keeps EVERY population that ever wrote
into the directory, each with its own ``done`` event, so it is the cross-check
the census needed.  Four classes are pinned here, because they are NOT
interchangeable:

  * **A — the census file is a TRUNCATED PREFIX of a longer population.**
    The defect.  The series must be re-derived from the events rows of the
    matching population; the prefix must never be scored.
  * **B — the run RESUMED.**  The events stream carries the pre-resume
    epochs.  Preferring the longer series would double-count epochs the run
    never trained, so these runs stay excluded from the CLEAN population.
  * **C — the events stream is the SHORTER record.**  Preferring it would
    lose epochs, so the per-epoch file stays.
  * **D — no events stream at all.**  Nothing to check; the file stands.

Wf-A116 — THE CROSS-CHECK IS VALUE-AWARE
----------------------------------------
The classes above were decided by comparing EPOCH INDEX SETS only, so the
census could not tell a truncated record of the run of record from ANOTHER
launch's record that happened to occupy the same directory.  MEASURED:
``wf299_cfm_main/wave_a/full_stack``'s file (train_loss 2.07847 / 1.84889 /
1.65890) and its events rows (1.40822 / 1.22302) DISAGREE at every shared
epoch, and the census called that "the events stream is a PARTIAL record" —
MEASURED FALSE.

The cross-check now matches the per-epoch file against the events rows of
each population:

  * the file IS the run of record -> today's length-based classes stay;
  * the file is ANOTHER population's record -> ``class_a_other_population``:
    the run of record is scored and re-derived from its events rows, and
    the reason names WHICH population and WHICH epochs disagree;
  * the file matches NO population -> ``class_d_foreign_launch``: the events
    stream is a FOREIGN launch that reused the directory.  The file is the
    run's own record and is KEPT, and the run is flagged UNCORROBORATED.

Two traps, both measured: one epoch can carry rows from more than one
population (``cfm_bond_head_20ep_2026_10_09`` holds epoch 1 twice), so the
file must match SOME row at that epoch; and ``nan == nan`` is the SAME
population (``cfm_r11_eval256``, ``cfm_r17_eval512``).

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
REPORTS = REPO_ROOT / "molmetal" / "reports"

#: The four REAL runs whose per-epoch file is ANOTHER population's record,
#: with the end-to-end rate each one reads under the census BEFORE the fix
#: (the truncated prefix) and the rate it reads AFTER (the events-derived
#: series): last epoch, best epoch, and the epoch the best sits on.
#: MEASURED from disk.
#:
#: Wf-A116 — the verdict NAME changed for these four: they are NOT a
#: truncated prefix of the run of record (their values disagree with it at
#: every shared epoch), they are another population's record.  Their
#: re-derived numbers did NOT change.
OTHER_POPULATION = {
    "cfm_fullcorpus_2026_10_10": (
        0.0078125, 0.5208333333333334, 0.5208333333333334, 13,
    ),
    "cfm_round3_retrain/run": (
        0.0, 0.29874213836477986, 0.47395833333333326, 17,
    ),
    "cfm_round7_f5f6_retrain/run": (
        0.02734375, 0.5572916666666666, 0.5572916666666666, 19,
    ),
    "cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529": (
        0.0078125, 0.5572916666666666, 0.5572916666666666, 19,
    ),
}

#: Wf-A116 — WHICH population each of those four files turned out to be.
#: MEASURED from the events streams on disk.
OTHER_POPULATION_SEGMENT = {
    "cfm_fullcorpus_2026_10_10": 2,
    "cfm_round3_retrain/run": 2,
    "cfm_round7_f5f6_retrain/run": 3,
    "cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529": 1,
}

#: The two REAL class-B runs (resumed).  They MUST stay out of the CLEAN
#: population — readmitting them would be a false improvement.
CLASS_B = ("cfm_c7_fullgen_2026_10_20", "cfm_bond_head_20ep_2026_10_09")

#: The one REAL class-C run: the events stream is the shorter record.
CLASS_C = "wf299_cfm_main/wave_a/full_stack"


@pytest.fixture(scope="module")
def art():
    if not ARTIFACT.is_file():
        pytest.skip("census artifact not on disk")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs(art):
    return {r["run"]: r for r in art["runs"]}


# ----------------------------------------------------------------------
# synthetic run directories
# ----------------------------------------------------------------------
def _row(epoch, *, gv_val=50, gv_n=100, sfr=0.4, lff=0.95, dec=0.5):
    """One per-epoch row, in the shape the JSON metrics file uses."""
    return {
        "epoch": epoch,
        "train_loss": 1.0 / (epoch + 1),
        "gen_validity_val": gv_val,
        "gen_validity_n": gv_n,
        "gen_single_fragment_ratio": sfr,
        "gen_largest_fragment_frac": lff,
        "decode_ratio_rdkit_strict": dec,
        "decode_ratio_n": gv_n,
        "bond_f1": 0.5,
    }


def _event_row(epoch, ts, **kw):
    row = {
        "event": "train_epoch",
        "epoch": epoch,
        "ts": ts,
        "train_loss": 1.0 / (epoch + 1),
        "gen_validity_val": 50,
        "gen_validity_n": 100,
        "gen_single_fragment_ratio": 0.4,
        "gen_largest_fragment_frac": 0.95,
        "decode_ratio": 0.5,
        "decode_ratio_n": 100,
        "lr": 0.0003,
    }
    row.update(kw)
    return row


def _gen_eval_row(epoch, ts, n_total, n_per_seed):
    """One `gen_eval` event row: the generation sampling budget of a seed."""
    return {
        "event": "gen_eval",
        "epoch": epoch,
        "ts": ts,
        "seed": 0,
        "gen_validity": 0.5,
        "n_total": n_total,
        "n_per_seed": n_per_seed,
        "n_seeds": 1,
    }


def _write_run(d: Path, *, json_rows, config, events, csv_rows=None):
    """Write one synthetic run directory (metrics file + events stream)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "per_epoch_metrics.json").write_text(
        json.dumps({"config": config, "per_epoch_metrics": json_rows}),
        encoding="utf-8",
    )
    if csv_rows is None:
        csv_rows = json_rows
    header = "epoch,train_loss,gen_validity_val,gen_validity_n," \
             "gen_single_fragment_ratio,gen_largest_fragment_frac," \
             "decode_ratio,bond_f1"
    lines = [header]
    for r in csv_rows:
        lines.append(
            f"{r['epoch']},{r.get('train_loss')},{r.get('gen_validity_val')},"
            f"{r.get('gen_validity_n')},{r.get('gen_single_fragment_ratio')},"
            f"{r.get('gen_largest_fragment_frac')},"
            f"{r.get('decode_ratio_rdkit_strict', 0.0)},{r.get('bond_f1')}"
        )
    (d / "per_epoch_metrics.csv").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    with (d / "events.jsonl").open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    return d


# ----------------------------------------------------------------------
# class A — the defect
# ----------------------------------------------------------------------
class TestTheTruncatedPrefix:
    """A per-epoch file that is a PREFIX of a longer population."""

    def _synthetic(self, tmp_path):
        # Ten epochs in the events stream, the run of record.
        events = [
            _event_row(i, 100.0 + i, gen_single_fragment_ratio=0.8 if i == 9 else 0.4)
            for i in range(10)
        ]
        events.append({
            "event": "done",
            "ts": 200.0,
            "n_epochs": 20,
            "n_epochs_run": 10,
            "best_epoch": 9,
            "gate_metric": "combined",
            "gate_value": 0.4,
        })
        # The metrics file holds only the FIRST THREE epochs, and its own
        # config belongs to the 1-epoch run that overwrote it.
        prefix = [
            _row(i, gv_val=100, gv_n=100, sfr=0.01) for i in range(3)
        ]
        d = _write_run(
            tmp_path / "clobbered",
            json_rows=prefix,
            config={"epochs": 1, "n_train": 64, "n_val": 32},
            events=events,
        )
        return census.load_run(d)

    def test_the_conflict_is_detected(self, tmp_path):
        run = self._synthetic(tmp_path)
        integ = run["integrity"]
        assert integ["population_conflict"] is True
        assert integ["population_verdict"] == "class_a_truncated_prefix"
        assert integ["population_resolved"] is True
        assert "OVERWRITTEN" in integ["population_reason"]

    def test_the_truncated_prefix_is_not_scored(self, tmp_path):
        run = self._synthetic(tmp_path)
        # The events-derived series: ten epochs, ending at epoch 9 with
        # gv 0.5 x sfr 0.8 = 0.4.  The prefix read 3 epochs at 0.01.
        assert run["integrity"]["epochs_recorded"] == 10
        assert run["integrity"]["epoch_index_end"] == 9
        assert run["gen_single_molecule_rate_end"] == pytest.approx(0.4)
        assert run["gen_single_molecule_rate_end"] != pytest.approx(0.01)
        assert run["gen_single_molecule_rate_best"] == pytest.approx(0.4)
        assert run["gen_single_molecule_rate_best_epoch"] == 9

    def test_the_source_is_recorded(self, tmp_path):
        run = self._synthetic(tmp_path)
        assert run["integrity"]["series_source"] == "events"
        assert run["integrity"]["population_source"] == "events.jsonl"
        assert run["config_source"] == "events.jsonl"
        assert run["config_complete"] is False
        assert run["config_conflict"] is True

    def test_the_events_facts_are_recorded(self, tmp_path):
        run = self._synthetic(tmp_path)
        integ = run["integrity"]
        assert integ["events_present"] is True
        assert integ["events_done_count"] == 1
        assert integ["events_done_n_epochs_run"] == [10]
        assert integ["events_max_n_epochs_run"] == 10
        assert integ["events_train_epoch_row_count"] == 10
        assert integ["events_epoch_indices"] == list(range(10))
        assert integ["events_epoch_index_start"] == 0
        assert integ["events_epoch_index_end"] == 9
        assert integ["events_segment_count"] == 1
        assert integ["run_of_record_segment_index"] == 0
        assert integ["run_of_record_epoch_indices"] == list(range(10))

    def test_the_run_of_records_sampling_budget_is_recorded(self, tmp_path):
        """The `gen_eval` denominator of the RUN OF RECORD, not the smoke's."""
        events = [
            _event_row(i, 100.0 + i) for i in range(10)
        ] + [
            _gen_eval_row(i, 150.0 + i, n_total=192, n_per_seed=256)
            for i in range(10)
        ] + [
            {
                "event": "done",
                "ts": 200.0,
                "n_epochs": 20,
                "n_epochs_run": 10,
                "best_epoch": 9,
            },
            _event_row(0, 300.0),
            _gen_eval_row(0, 350.0, n_total=32, n_per_seed=32),
            {
                "event": "done",
                "ts": 400.0,
                "n_epochs": 1,
                "n_epochs_run": 1,
                "best_epoch": 0,
            },
        ]
        prefix = [_row(i, gv_val=100, gv_n=100, sfr=0.01) for i in range(3)]
        d = _write_run(
            tmp_path / "clobbered",
            json_rows=prefix,
            config={"epochs": 1, "n_train": 64, "n_val": 32},
            events=events,
        )
        run = census.load_run(d)
        integ = run["integrity"]
        # The run of record sampled 192 molecules per seed; the smoke that
        # overwrote the file sampled 32.  The flat field names the RECORD's.
        assert integ["events_gen_eval_n_total"] == [192]
        assert integ["events_gen_eval_n_per_seed"] == [256]
        # Both populations stay visible, each with its own budget.
        segs = {s["index"]: s for s in integ["events_segments"]}
        assert segs[0]["gen_eval_n_total"] == [192]
        assert segs[1]["gen_eval_n_total"] == [32]


# ----------------------------------------------------------------------
# class B — the resume
# ----------------------------------------------------------------------
class TestTheResumedRun:
    """A run whose events stream ALSO carries the pre-resume epochs."""

    def _synthetic(self, tmp_path):
        events = [
            _event_row(i, 100.0 + i) for i in range(4)          # killed attempt
        ] + [
            _event_row(i, 200.0 + i) for i in range(4, 10)      # the resume
        ]
        events.append({
            "event": "done",
            "ts": 300.0,
            "n_epochs": 20,
            "n_epochs_run": 6,
            "best_epoch": 9,
        })
        rows = [_row(i) for i in range(4, 10)]
        d = _write_run(
            tmp_path / "resumed",
            json_rows=rows,
            config={"epochs": 20, "n_train": 256, "n_val": 192},
            events=events,
        )
        return census.load_run(d)

    def test_it_is_not_a_conflict(self, tmp_path):
        run = self._synthetic(tmp_path)
        integ = run["integrity"]
        assert integ["population_conflict"] is False
        assert integ["population_verdict"] == "class_b_resume_prehistory"
        assert "pre-resume" in integ["population_reason"]

    def test_the_pre_resume_epochs_are_not_imported(self, tmp_path):
        run = self._synthetic(tmp_path)
        # The census still reads the run's OWN file: six epochs, 4..9.
        assert run["integrity"]["epochs_recorded"] == 6
        assert run["integrity"]["epoch_index_start"] == 4
        assert run["integrity"]["series_source"] == "json"

    def test_it_stays_out_of_the_clean_population(self, tmp_path):
        run = self._synthetic(tmp_path)
        assert run["integrity"]["resumed"] is True
        assert census._is_clean(run) is False


# ----------------------------------------------------------------------
# class C — the events stream is the shorter record
# ----------------------------------------------------------------------
class TestTheShorterEventsStream:
    """The events stream carries FEWER epochs than the metrics file."""

    def _synthetic(self, tmp_path):
        events = [_event_row(i, 100.0 + i) for i in range(2)]
        rows = [_row(i) for i in range(3)]
        d = _write_run(
            tmp_path / "partial",
            json_rows=rows,
            config={"epochs": 3, "n_train": 256, "n_val": 192},
            events=events,
        )
        return census.load_run(d)

    def test_it_is_not_a_conflict(self, tmp_path):
        run = self._synthetic(tmp_path)
        integ = run["integrity"]
        assert integ["population_conflict"] is False
        assert integ["population_verdict"] == "class_c_events_shorter"

    def test_no_epoch_is_lost(self, tmp_path):
        run = self._synthetic(tmp_path)
        assert run["integrity"]["epochs_recorded"] == 3
        assert run["integrity"]["epoch_index_end"] == 2
        assert run["integrity"]["series_source"] == "json"
        assert run["gen_single_molecule_rate_end"] == pytest.approx(0.2)


# ----------------------------------------------------------------------
# Wf-A116 — the VALUE-AWARE cross-check
# ----------------------------------------------------------------------
class TestTheValueAwareCrossCheck:
    """The cross-check compares VALUES, not epoch index sets."""

    def test_a_file_that_is_another_populations_record(self, tmp_path):
        """The file matches population 1; the run of record is population 0."""
        events = [
            _event_row(i, 100.0 + i, gen_single_fragment_ratio=0.8 if i == 9 else 0.4)
            for i in range(10)
        ]
        events.append({
            "event": "done",
            "ts": 200.0,
            "n_epochs": 20,
            "n_epochs_run": 10,
            "best_epoch": 9,
        })
        # A SECOND population wrote into the same directory, with its own
        # (completely different) loss values.
        events += [
            {**_event_row(0, 300.0), "train_loss": 5.0},
            {**_event_row(1, 301.0), "train_loss": 4.0},
            {**_event_row(2, 302.0), "train_loss": 3.0},
            {
                "event": "done",
                "ts": 400.0,
                "n_epochs": 3,
                "n_epochs_run": 3,
                "best_epoch": 2,
            },
        ]
        # The per-epoch file holds the SECOND population's rows.
        rows = [
            {**_row(0, gv_val=100, gv_n=100, sfr=0.01), "train_loss": 5.0},
            {**_row(1, gv_val=100, gv_n=100, sfr=0.01), "train_loss": 4.0},
            {**_row(2, gv_val=100, gv_n=100, sfr=0.01), "train_loss": 3.0},
        ]
        d = _write_run(
            tmp_path / "other_population",
            json_rows=rows,
            config={"epochs": 3, "n_train": 64, "n_val": 32},
            events=events,
        )
        run = census.load_run(d)
        integ = run["integrity"]
        assert integ["population_verdict"] == "class_a_other_population"
        assert integ["population_conflict"] is True
        assert integ["population_resolved"] is True
        assert integ["population_matched_segment_index"] == 1
        assert integ["population_disagreed_epochs"] == [0, 1, 2]
        assert "population 1" in integ["population_reason"]
        # the RUN OF RECORD is scored, not the other population's file
        assert integ["series_source"] == "events"
        assert integ["epochs_recorded"] == 10
        assert run["gen_single_molecule_rate_end"] == pytest.approx(0.4)

    def test_the_file_must_match_SOME_row_at_an_epoch(self, tmp_path):
        """One epoch can carry rows from more than one population.

        MEASURED: ``cfm_bond_head_20ep_2026_10_09``'s population holds
        epoch 1 twice — 0.82975046 from a killed pre-resume attempt and
        0.83484674 from the run of record — and the file's epoch 1 row is
        0.83485 (the 5-decimal rendering).  Comparing against the FIRST
        row at that epoch would report a mismatch that is not there.
        """
        events = [
            {**_event_row(0, 100.0), "train_loss": 0.99},
            {**_event_row(1, 101.0), "train_loss": 0.82975046},
            {**_event_row(1, 200.0), "train_loss": 0.83484674},
            {**_event_row(2, 201.0), "train_loss": 0.79261572},
            {
                "event": "done",
                "ts": 300.0,
                "n_epochs": 20,
                "n_epochs_run": 2,
                "best_epoch": 2,
            },
        ]
        rows = [
            {**_row(1), "train_loss": 0.83485},
            {**_row(2), "train_loss": 0.79262},
        ]
        d = _write_run(
            tmp_path / "double_epoch",
            json_rows=rows,
            config={"epochs": 20, "n_train": 256, "n_val": 192},
            events=events,
        )
        run = census.load_run(d)
        integ = run["integrity"]
        assert integ["population_verdict"] == "class_b_resume_prehistory"
        assert integ["population_conflict"] is False
        assert integ["population_disagreed_epochs"] == []

    def test_nan_equals_nan(self, tmp_path):
        """A nan loss on both sides is the SAME population."""
        events = [
            {**_event_row(0, 100.0), "train_loss": float("nan")},
            {
                "event": "done",
                "ts": 200.0,
                "n_epochs": 20,
                "n_epochs_run": 1,
                "best_epoch": 0,
            },
        ]
        rows = [{**_row(0), "train_loss": float("nan")}]
        d = _write_run(
            tmp_path / "nan_loss",
            json_rows=rows,
            config={"epochs": 20, "n_train": 256, "n_val": 192},
            events=events,
        )
        run = census.load_run(d)
        integ = run["integrity"]
        assert integ["population_verdict"] == "no_conflict"
        assert integ["population_conflict"] is False
        assert integ["population_uncorroborated"] is False
        assert integ["population_disagreed_epochs"] == []

    def test_a_genuine_truncation_keeps_the_class_a_name(self, tmp_path):
        """The class-A name is reserved for a SAME-population truncation."""
        events = [_event_row(i, 100.0 + i) for i in range(10)] + [{
            "event": "done",
            "ts": 200.0,
            "n_epochs": 20,
            "n_epochs_run": 10,
            "best_epoch": 9,
        }]
        rows = [_row(i, gv_val=100, gv_n=100, sfr=0.01) for i in range(3)]
        d = _write_run(
            tmp_path / "truncated",
            json_rows=rows,
            config={"epochs": 1, "n_train": 64, "n_val": 32},
            events=events,
        )
        run = census.load_run(d)
        integ = run["integrity"]
        assert integ["population_verdict"] == "class_a_truncated_prefix"
        assert integ["epochs_recorded"] == 10

    def test_a_foreign_stream_is_kept_but_flagged(self, tmp_path):
        """The events stream disagrees on the VALUES and matches nothing."""
        events = [
            {**_event_row(0, 100.0), "train_loss": 1.4082169085741043},
            {**_event_row(1, 101.0), "train_loss": 1.2230224758386612},
        ]
        rows = [
            {**_row(0), "train_loss": 2.078470081090927},
            {**_row(1), "train_loss": 1.8488879799842834},
            {**_row(2), "train_loss": 1.6588959693908691},
        ]
        d = _write_run(
            tmp_path / "foreign",
            json_rows=rows,
            config={"epochs": 3, "n_train": 256, "n_val": 192},
            events=events,
        )
        run = census.load_run(d)
        integ = run["integrity"]
        assert integ["population_verdict"] == "class_d_foreign_launch"
        assert integ["population_uncorroborated"] is True
        assert integ["population_disagreed_epochs"] == [0, 1]
        # the FILE is kept: no epoch is lost
        assert integ["epochs_recorded"] == 3
        assert integ["series_source"] == "json"
        assert "PARTIAL" not in integ["population_reason"]
        assert "FOREIGN" in integ["population_reason"]


# ----------------------------------------------------------------------
# class D — no events stream
# ----------------------------------------------------------------------
class TestNoEventsStream:
    def test_the_file_stands_unchanged(self, tmp_path):
        d = tmp_path / "lonely"
        d.mkdir()
        rows = [_row(i) for i in range(2)]
        (d / "per_epoch_metrics.json").write_text(
            json.dumps({"config": {"epochs": 2, "n_train": 64}, "per_epoch_metrics": rows}),
            encoding="utf-8",
        )
        (d / "per_epoch_metrics.csv").write_text(
            "epoch,train_loss,gen_validity_val,gen_validity_n,"
            "gen_single_fragment_ratio,decode_ratio\n"
            "0,1.0,50,100,0.4,0.5\n1,0.5,50,100,0.4,0.5\n",
            encoding="utf-8",
        )
        run = census.load_run(d)
        integ = run["integrity"]
        assert integ["events_present"] is False
        assert integ["population_verdict"] == "events_absent"
        assert integ["population_conflict"] is False
        assert integ["series_source"] == "json"
        assert integ["epochs_recorded"] == 2


# ----------------------------------------------------------------------
# the REAL runs on disk
# ----------------------------------------------------------------------
class TestTheRealOtherPopulationRuns:
    """The four runs the diagnosis measured, on disk.

    Wf-A116 — their verdict NAME changed: the per-epoch file is ANOTHER
    population's record, not a truncated prefix of the run of record.  The
    re-derived numbers did NOT change.
    """

    def test_all_four_are_detected(self, runs):
        for name in OTHER_POPULATION:
            integ = runs[name]["integrity"]
            assert integ["population_conflict"] is True, name
            assert integ["population_verdict"] == "class_a_other_population", name
            assert integ["population_resolved"] is True, name
            assert integ["series_source"] == "events", name

    def test_the_reason_names_the_population_and_the_epochs(self, runs):
        for name, seg in OTHER_POPULATION_SEGMENT.items():
            integ = runs[name]["integrity"]
            assert integ["population_matched_segment_index"] == seg, name
            reason = integ["population_reason"]
            assert f"population {seg}" in reason, name
            assert "DISAGREE" in reason, name
            assert integ["population_disagreed_epochs"], name

    def test_the_reported_rate_is_the_events_derived_one(self, runs):
        for name, (old, new_last, new_best, epoch) in OTHER_POPULATION.items():
            run = runs[name]
            assert run["gen_single_molecule_rate_end"] == pytest.approx(new_last), name
            assert run["gen_single_molecule_rate_end"] != pytest.approx(old), name
            assert run["gen_single_molecule_rate_best"] == pytest.approx(new_best), name
            assert run["gen_single_molecule_rate_best_epoch"] == epoch, name

    def test_the_two_flipped_runs_are_above_the_gate(self, runs):
        """MEASURED: two of the four cross the 0.50 gate once re-derived."""
        assert runs["cfm_fullcorpus_2026_10_10"][
            "gen_single_molecule_rate_end"
        ] >= 0.50
        assert runs["cfm_round7_f5f6_retrain/run"][
            "gen_single_molecule_rate_end"
        ] >= 0.50

    def test_the_prefix_is_not_scored(self, runs):
        for name, (old, _last, _best, _epoch) in OTHER_POPULATION.items():
            assert runs[name]["gen_single_molecule_rate_end"] != old, name

    def test_the_run_of_record_is_the_longest_population(self, runs):
        assert runs["cfm_fullcorpus_2026_10_10"]["integrity"][
            "run_of_record_epoch_indices"
        ] == list(range(14))
        for name in ("cfm_round3_retrain/run", "cfm_round7_f5f6_retrain/run"):
            assert runs[name]["integrity"]["run_of_record_epoch_indices"] == list(
                range(20)
            ), name

    def test_the_config_is_flagged_as_partial(self, runs):
        for name in OTHER_POPULATION:
            assert runs[name]["config_source"] == "events.jsonl", name
            assert runs[name]["config_complete"] is False, name
            assert runs[name]["config_conflict"] is True, name
        # n_train / n_val are NOT recoverable from the events stream.  Only
        # the cohort recovery (Wf-A115) can supply them, and only for a
        # population a run log attests — see the cohort test below.
        for name in (
            "cfm_round3_retrain/run",
            "cfm_round7_f5f6_retrain/run",
            "cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529",
        ):
            assert runs[name]["n_train"] is None, name
            assert runs[name]["n_val"] is None, name

    def test_the_cohort_wiring_survives_the_rename(self, runs):
        """Wf-A115 — the cohort of the RUN OF RECORD is recovered from a log."""
        full = runs["cfm_fullcorpus_2026_10_10"]
        assert full["n_train"] == 26715
        assert full["n_val"] == 192
        assert full["cohort_source"] == "run_log:ts_matched"
        for name in (
            "cfm_round3_retrain/run",
            "cfm_round7_f5f6_retrain/run",
            "cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529",
        ):
            assert runs[name]["n_train"] is None, name
            assert runs[name]["n_val"] is None, name
            assert runs[name]["cohort_source"] == "unattested", name


class TestTheRealResumeRunsStayExcluded:
    def test_the_class_b_runs_are_not_clean(self, runs):
        for name in CLASS_B:
            integ = runs[name]["integrity"]
            assert integ["population_verdict"] == "class_b_resume_prehistory", name
            assert integ["population_conflict"] is False, name
            assert integ["resumed"] is True, name
            assert census._is_clean(runs[name]) is False, name

    def test_the_class_b_runs_keep_their_own_series(self, runs):
        for name in CLASS_B:
            assert runs[name]["integrity"]["series_source"] == "json", name
            assert runs[name]["integrity"]["epochs_recorded"] == (
                14 if name == "cfm_c7_fullgen_2026_10_20" else 19
            ), name

    def test_they_are_still_named_as_degenerate(self, art):
        degenerate = art["fragmentation_summary"][
            "degenerate_runs_excluded_from_the_headline"
        ]
        for name in CLASS_B:
            assert name in degenerate, name


class TestTheRealForeignLaunchRun:
    """Wf-A116 — the one run whose events stream is a FOREIGN launch.

    MEASURED: ``wf299_cfm_main/wave_a/full_stack``'s per-epoch file holds
    train_loss 2.07847 / 1.84889 / 1.65890 (epochs 0/1/2) while its
    ``events.jsonl`` holds 2 train_epoch rows, epoch 0 = 1.4082169085741043
    and epoch 1 = 1.2230224758386612, and NO ``done`` event.  The two
    records DISAGREE at every shared epoch, so the events stream is not "a
    partial record" of this run.
    """

    def test_it_keeps_its_third_epoch(self, runs):
        run = runs[CLASS_C]
        integ = run["integrity"]
        assert integ["population_verdict"] == "class_d_foreign_launch"
        assert integ["epochs_recorded"] == 3
        assert integ["series_source"] == "json"

    def test_it_is_flagged_uncorroborated(self, runs):
        integ = runs[CLASS_C]["integrity"]
        assert integ["population_uncorroborated"] is True
        assert integ["population_disagreed_epochs"] == [0, 1]

    def test_the_reason_does_not_claim_a_partial_record(self, runs):
        reason = runs[CLASS_C]["integrity"]["population_reason"]
        assert "PARTIAL" not in reason
        assert "FOREIGN" in reason
        assert "UNCORROBORATED" in reason
        # the disagreeing epochs and their values are NAMED
        assert "epoch 0" in reason and "epoch 1" in reason
        assert "2.07847" in reason and "1.40822" in reason

    def test_it_is_not_a_class_c(self, runs):
        """The class-C reading is reserved for an AGREEING shorter stream."""
        integ = runs[CLASS_C]["integrity"]
        assert integ["population_verdict"] != "class_c_events_shorter"
        assert integ["population_conflict"] is False


# ----------------------------------------------------------------------
# the artifact as a whole
# ----------------------------------------------------------------------
class TestTheArtifact:
    def test_every_run_carries_the_population_fields(self, runs):
        for name, r in runs.items():
            integ = r["integrity"]
            for field in (
                "events_present",
                "events_done_count",
                "events_done_n_epochs_run",
                "events_max_n_epochs_run",
                "events_train_epoch_row_count",
                "events_epoch_indices",
                "events_epoch_index_start",
                "events_epoch_index_end",
                "events_gen_eval_epoch_indices",
                "events_segment_count",
                "events_segments",
                "run_of_record_segment_index",
                "run_of_record_epoch_indices",
                "population_conflict",
                "population_verdict",
                "population_reason",
                "population_source",
            ):
                assert field in integ, f"{name} has no integrity.{field}"
            assert integ["population_verdict"], name
            assert integ["population_reason"], name
            assert r["config_source"] in (
                "per_epoch_metrics.json",
                "per_epoch_metrics.csv",
                "events.jsonl",
            ), name

    def test_the_source_count_block_has_an_events_bucket(self, art):
        counts = art["corpus"]["series_source"]
        assert "events" in counts
        assert counts["events"] == len(OTHER_POPULATION)

    def test_the_conflict_block_lists_the_four_runs(self, art):
        blk = art["population_conflicts"]
        assert blk["n_conflicted"] == 4
        assert blk["n_conflicted_resolved"] == 4
        assert blk["n_conflicted_refused"] == 0
        assert sorted(c["run"] for c in blk["conflicted_runs"]) == sorted(
            OTHER_POPULATION
        )

    def test_the_verdict_distribution_is_the_measured_one(self, art):
        counts = art["population_conflicts"]["verdict_counts"]
        assert counts["no_conflict"] == 43
        assert counts["class_b_resume_prehistory"] == 2
        assert counts["events_absent"] == 9
        # Wf-A116 — the four class-A runs are ANOTHER population's record
        # and the one class-C run is a FOREIGN launch, so the class-A
        # truncation class and the class-C class are both empty on this
        # corpus: no run's file is a truncated prefix of its own run.
        assert counts["class_a_other_population"] == 4
        assert counts["class_d_foreign_launch"] == 1
        assert "class_a_truncated_prefix" not in counts
        assert "class_c_events_shorter" not in counts
        assert sum(counts.values()) == art["n_runs"]

    def test_the_uncorroborated_run_is_listed(self, art):
        blk = art["population_conflicts"]
        assert blk["n_uncorroborated"] == 1
        run = blk["uncorroborated_runs"][0]
        assert run["run"] == CLASS_C
        assert run["verdict"] == "class_d_foreign_launch"
        assert run["disagreed_epochs"] == [0, 1]

    def test_no_run_is_refused_on_this_corpus(self, art):
        """No run on disk falls into the unmatched class."""
        blk = art["population_conflicts"]
        assert blk["n_conflicted_refused"] == 0

    def test_the_not_counted_run_is_published(self, art):
        """Wf-A116 — a chimaera directory is PUBLISHED, not dropped."""
        assert art["n_not_counted_runs"] == 1
        assert art["corpus"]["n_run_directories_with_per_epoch_csv"] == 60
        assert art["corpus"]["n_runs_not_counted"] == 1
        entry = art["not_counted_runs"][0]
        assert entry["run"] == "cfm_default"
        ev = entry["evidence"]
        assert ev["per_epoch_metrics_csv_rows"] == 0
        assert ev["per_epoch_metrics_json_rows"] == 0
        assert ev["events_present"] is True
        assert ev["events_done_count"] == 3
        assert ev["events_done_n_epochs_run"] == [1, 4, 0]
        assert ev["events_train_epoch_row_count"] == 5
        # and the prose names it
        assert "cfm_default" in art["not_counted"]

    def test_the_unmatched_class_refuses_the_run(self, tmp_path):
        """A run whose two records disagree irreconcilably is REFUSED."""
        events = [_event_row(i, 100.0 + i) for i in range(4)]
        events.append({
            "event": "done", "ts": 200.0, "n_epochs": 20, "n_epochs_run": 4,
        })
        # The file holds epochs 8..11: neither equal to, nor a prefix of,
        # nor a subset of the run of record's 0..3.
        rows = [_row(i) for i in range(8, 12)]
        d = _write_run(
            tmp_path / "unmatched",
            json_rows=rows,
            config={"epochs": 20, "n_train": 256, "n_val": 192},
            events=events,
        )
        run = census.load_run(d)
        integ = run["integrity"]
        assert integ["population_verdict"] == "unmatched_conflict"
        assert integ["population_conflict"] is True
        assert integ["population_resolved"] is False
        assert "REFUSED" in integ["population_reason"]
        # Refused => excluded from every headline population.
        assert census._is_clean(run) is False
