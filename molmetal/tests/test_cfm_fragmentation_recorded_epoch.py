# Licensed under the Apache License, Version 2.0.
"""Wf-A121 / I2 -- the AUDIT side of the pool-epoch rule: the cross-check at
the pool's own epoch, the pool's own composition, and the labelled no-pool
fallback.

THE DEFECTS THIS PINS
---------------------
R2  ``cfm_fragmentation_audit.py``'s ``recorded_vs_rederived`` block put the
    census's ``gen_single_fragment_ratio_end`` -- the run's LAST-epoch
    aggregate -- beside the re-derivation, which is over the pool's OWN rows.
    Two different epochs were compared.  MEASURED: for
    ``cfm_objective_terms_2026_10_11`` the recorded side read 0.8772 (its
    epoch-8 aggregate) while the pool's own epoch 5 reads 0.6667.

R1  The gate and the c7 audit disagreed about the SAME run directory: a
    stream with NO ``done`` event resolved to NOTHING in the gate (with the
    FALSE reason "this run directory has no events.jsonl") while the c7
    treated the whole stream as ONE population.  MEASURED:
    ``cfm_c7_fullgen_ep4_partial_2026_10_20`` and
    ``cfm_c8_fullgen_2026_10_24`` both hold an ``events.jsonl``.

R8  The gate's no-pool fallback published a validity with NO label saying it
    was the recorded series' LAST-epoch value rather than a pool-epoch value.

CPU-only, no GPU, no training.  The synthetic fixtures are written under
``tmp_path``; the real-run tests only READ the repository.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.cfm_fragmentation_audit import audit_run  # noqa: E402
from molmetal.scripts.cfm_fragmentation_gate import evaluate, score_run_dir  # noqa: E402

ARTIFACT = REPO_ROOT / "metrics" / "by_metric" / "cfm_fragmentation_audit.json"
AUDIT_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "cfm_fragmentation_audit.py"

#: The two runs the residual ledger named: their event stream carries NO
#: ``done`` event, so the WHOLE STREAM is one population.
UNSEGMENTED_RUNS = [
    ("cfm_c7_fullgen_ep4_partial_2026_10_20", 3),
    ("cfm_c8_fullgen_2026_10_24", 2),
]

#: The one run directory with NO ``events.jsonl`` at all.
NO_EVENTS_RUN = "wf124_p2_cfm_full"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


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
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "epoch",
                "gen_validity",
                "gen_validity_val",
                "gen_validity_n",
                "gen_single_fragment_ratio",
                "gen_largest_fragment_frac",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_events(path: Path, rows) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _event_row(epoch: int, ratio: float, ts: float) -> dict:
    """One AGGREGATE (``train_epoch``) event row."""
    return {
        "event": "train_epoch",
        "epoch": epoch,
        "ts": ts,
        "gen_validity": 1.0,
        "gen_validity_val": 4,
        "gen_validity_n": 4,
        "gen_single_fragment_ratio": ratio,
        "gen_largest_fragment_frac": 0.95,
    }


def _artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip("the fragmentation-audit artifact is not on disk")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _run_row(blob: dict, run: str) -> dict:
    for row in blob.get("runs") or []:
        if row.get("run") == run:
            return row
    raise AssertionError(f"the artifact carries no run row for {run}")


# ---------------------------------------------------------------------------
# (a) the synthetic run: the recorded side is read at the POOL'S epoch
# ---------------------------------------------------------------------------


class TestTheCrossCheckIsAtThePoolEpoch:
    def test_the_recorded_side_is_the_pool_epoch_value(self, tmp_path):
        # A synthetic run whose recorded series' LAST row is at epoch 4 while
        # its pool is epoch 3.  The cross-check must compare the pool's OWN
        # epoch, and keep the last-epoch value beside it.
        d = tmp_path / "run"
        d.mkdir()
        _write_pool(
            d / "generated_smiles.csv",
            [
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
            ],
        )
        _write_events(
            d / "events.jsonl",
            [
                _event_row(3, 0.5, 1003.0),
                _event_row(4, 0.75, 1004.0),
            ],
        )
        # The census's own row: the run's LAST-epoch aggregate.
        census_row = {
            "run": "run",
            "gen_single_fragment_ratio_end": 0.75,
            "integrity": {"epoch_index_end": 4},
        }
        out = audit_run(d, census_row)
        rvr = out["recorded_vs_rederived"]
        assert rvr is not None
        # The RECORDED side is the pool's OWN epoch (3), not the run's last
        # epoch (4).
        assert rvr["recorded_single_fragment_ratio"] == pytest.approx(0.5)
        assert rvr["recorded_at_epoch"] == 3
        assert "epoch=3" in rvr["recorded_source"]
        # The OLD value is kept beside it, labelled with ITS epoch.
        assert rvr["recorded_last_epoch_single_fragment_ratio"] == pytest.approx(0.75)
        assert rvr["recorded_last_epoch"] == 4
        assert "gen_single_fragment_ratio_end" in rvr["recorded_last_epoch_source"]
        # The re-derived side is UNCHANGED: over the pool's own rows.
        assert rvr["rederived_single_fragment_ratio"] == pytest.approx(1.0)
        assert rvr["abs_diff"] == pytest.approx(0.5)
        # The note says what the reader needs to know.
        assert "POOL'S OWN EPOCH" in rvr["note"]
        assert "RECONSTRUCTION" in rvr["note"]

    def test_the_pool_epoch_aggregate_is_the_published_validity(self, tmp_path):
        # The same synthetic run: the validity published for the pool is the
        # epoch-3 aggregate, and the epoch-4 row is only the superseded one.
        d = tmp_path / "run"
        d.mkdir()
        _write_pool(
            d / "generated_smiles.csv",
            [{"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0}],
        )
        _write_events(
            d / "events.jsonl",
            [_event_row(3, 0.5, 1003.0), _event_row(4, 0.75, 1004.0)],
        )
        out = audit_run(d, None)
        pev = out["pool_epoch_validity"]
        assert out["pool_epoch"] == 3
        assert pev["value"] == pytest.approx(1.0)
        assert pev["n"] == 4
        assert out["verdict"]["gen_validity"] == pytest.approx(1.0)

    def test_the_pool_population_composition_is_published(self, tmp_path):
        # The composition is ADDITIVE: it says how much of the pool lies
        # outside the run of record, and it changes no other number.
        d = tmp_path / "run"
        d.mkdir()
        _write_pool(
            d / "generated_smiles.csv",
            [
                {"epoch": 0, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
            ],
        )
        _write_events(
            d / "events.jsonl",
            [_event_row(0, 0.5, 1000.0), _event_row(3, 0.5, 1003.0)],
        )
        census_row = {
            "run": "run",
            "integrity": {"run_of_record_segment_index": 0},
        }
        out = audit_run(d, census_row)
        comp = out["pool_population_composition"]
        assert comp["pool_epochs"] == [0, 3]
        assert comp["n_rows_total"] == 2
        assert comp["population_resolution"]


# ---------------------------------------------------------------------------
# (b)(c)(d) the REAL artifact, read from disk
# ---------------------------------------------------------------------------


class TestTheRealArtifact:
    def test_the_artifact_is_not_older_than_the_audit_script(self):
        # A stale artifact would publish null for the two un-segmented runs
        # and 4 for the blind count; this pins the staleness itself.
        if not ARTIFACT.is_file():
            pytest.skip("the fragmentation-audit artifact is not on disk")
        assert ARTIFACT.stat().st_mtime >= AUDIT_SCRIPT.stat().st_mtime, (
            "the fragmentation-audit artifact is STALE: regenerate it with "
            ".venv/bin/python molmetal/scripts/cfm_fragmentation_audit.py"
        )

    @pytest.mark.parametrize("run,epoch", UNSEGMENTED_RUNS)
    def test_the_un_segmented_runs_publish_one(self, run, epoch):
        # (b) The two runs whose event stream carries NO ``done`` event: the
        # whole stream is ONE population, so the pool's own epoch's aggregate
        # is published (1.0, n=704) instead of null.
        row = _run_row(_artifact(), run)
        assert row["verdict"]["gen_validity"] == pytest.approx(1.0)
        pev = row["pool_epoch_validity"]
        assert pev["value"] == pytest.approx(1.0)
        assert pev["n"] == 704
        assert row["pool_epoch"] == epoch
        assert "train_epoch[epoch=%d]" % epoch in pev["source"]
        # The reason says the TRUE thing: the stream EXISTS and carries no
        # done event.
        population = pev["population"]
        assert population["resolution"] == "un_segmented_event_stream"
        assert population["source"] == "fallback:un_segmented_event_stream"
        assert "no done event" in population["reason"]
        assert "no events.jsonl" not in population["reason"]

    def test_the_blind_validity_signature_count_is_six(self):
        # (c) The two newly-resolved runs are fragmented while their validity
        # is saturated, so they join the blind-metric signature.
        blob = _artifact()
        assert blob["summary"]["n_blind_validity_signature"] == 6
        blind = {r["run"] for r in blob["blind_validity_signature"]}
        for run, _epoch in UNSEGMENTED_RUNS:
            assert run in blind

    def test_the_run_without_events_jsonl_still_publishes_null(self):
        # (d) wf124_p2_cfm_full has NO events.jsonl at all, so no population
        # can be resolved and its null is CORRECT -- and its reason says so.
        row = _run_row(_artifact(), NO_EVENTS_RUN)
        assert row["verdict"]["gen_validity"] is None
        assert row["pool_epoch_validity"]["value"] is None
        population = row["pool_epoch_validity"]["population"]
        assert population["resolution"] == "unresolved"
        assert population["source"] == "no events.jsonl"
        assert "no events.jsonl" in population["reason"]

    def test_the_summary_counts_that_must_not_move(self):
        # The blind count is the ONLY summary number this wave moves.
        blob = _artifact()
        summary = blob["summary"]
        assert summary["n_runs_considered"] == 13
        assert summary["n_runs_with_a_pool"] == 13
        assert summary["n_generation_pools"] == 10
        assert summary["n_reconstruction_pools"] == 2
        assert summary["n_unknown_pools"] == 1
        assert summary["n_pools_with_csv_cross_check"] == 9
        assert summary["n_pools_with_csv_mismatch"] == 0
        assert summary["generation_side"]["gate_pass"] == 0
        assert summary["generation_side"]["gate_fail"] == 10
        assert summary["reconstruction_side"]["gate_pass"] == 1
        assert summary["reconstruction_side"]["gate_fail"] == 1
        assert summary["unknown_side"]["gate_fail"] == 1

    def test_every_run_keeps_its_pass_fail_verdict(self):
        # The verdicts do NOT move: the two newly-resolved runs were already
        # FAIL and stay FAIL.
        blob = _artifact()
        statuses = {
            r["run"]: (r.get("verdict") or {}).get("status")
            for r in blob["runs"]
        }
        assert statuses["cfm_c7_fullgen_ep4_partial_2026_10_20"] == "FAIL"
        assert statuses["cfm_c8_fullgen_2026_10_24"] == "FAIL"
        assert statuses["cfm_fullcorpus_b64_2026_10_11"] == "PASS"


# ---------------------------------------------------------------------------
# (e) the LABELLED no-pool fallback (the gate's ``elif met:`` branch)
# ---------------------------------------------------------------------------


class TestTheLabelledNoPoolFallback:
    def test_the_last_row_value_says_it_is_a_last_row_value(self, tmp_path):
        # No pool at all: the published validity is the recorded series' LAST
        # row, and the provenance SAYS so.
        d = tmp_path / "run"
        d.mkdir()
        _write_series(
            d / "per_epoch_metrics.csv",
            [
                {
                    "epoch": 1,
                    "gen_validity": 0.30,
                    "gen_validity_val": 30,
                    "gen_validity_n": 100,
                    "gen_single_fragment_ratio": 0.5,
                    "gen_largest_fragment_frac": 0.95,
                },
                {
                    "epoch": 7,
                    "gen_validity": 0.90,
                    "gen_validity_val": 90,
                    "gen_validity_n": 100,
                    "gen_single_fragment_ratio": 0.5,
                    "gen_largest_fragment_frac": 0.95,
                },
            ],
        )
        verdict = evaluate(score_run_dir(d))
        prov = verdict["gen_validity_provenance"]
        assert verdict["gen_validity"] == pytest.approx(0.90)
        assert prov["epoch"] == 7.0
        assert prov["epoch_role"] == "recorded_series_last_row"
        joined = " ".join(prov["notes"])
        assert "LAST row" in joined
        assert "NOT a pool-epoch value" in joined

    def test_an_explicit_epoch_is_labelled_as_such(self, tmp_path):
        # An explicit epoch is the caller's own choice: the value is THAT
        # epoch's row, and the provenance says so.
        d = tmp_path / "run"
        d.mkdir()
        _write_series(
            d / "per_epoch_metrics.csv",
            [
                {
                    "epoch": 1,
                    "gen_validity": 0.30,
                    "gen_validity_val": 30,
                    "gen_validity_n": 100,
                    "gen_single_fragment_ratio": 0.5,
                    "gen_largest_fragment_frac": 0.95,
                },
                {
                    "epoch": 7,
                    "gen_validity": 0.90,
                    "gen_validity_val": 90,
                    "gen_validity_n": 100,
                    "gen_single_fragment_ratio": 0.5,
                    "gen_largest_fragment_frac": 0.95,
                },
            ],
        )
        verdict = evaluate(score_run_dir(d, epoch=1))
        prov = verdict["gen_validity_provenance"]
        assert verdict["gen_validity"] == pytest.approx(0.30)
        assert prov["epoch"] == 1.0
        assert prov["epoch_role"] == "the_requested_epoch"
        joined = " ".join(prov["notes"])
        assert "requested epoch 1" in joined


# ---------------------------------------------------------------------------
# (e) Wf-A121C -- the FIELDS are honest
#
#   FIX 3  the ``recorded_vs_rederived`` block's PRESENCE follows the
#          RECORDED SIDE AT THE POOL'S OWN EPOCH, not the census carrying a
#          last-epoch value.
#   FIX 4  the blind list's ``single_molecule_rate`` carries the run's OWN
#          published end-to-end number (the verdict's) plus the three
#          LABELLED rates beside it.
#   FIX 2  the blind list carries the pool-epoch validity (value, n, epoch)
#          under its own keys.
#   FIX 5  the module docstring's false ``0.0085`` attribution is gone.
#   FIX 6  the smoke script's device resolution honours a CPU override.
# ---------------------------------------------------------------------------


B64_RUN = "cfm_fullcorpus_b64_2026_10_11"
B64_EVENTS = REPO_ROOT / "molmetal" / "reports" / B64_RUN / "events.jsonl"

#: The two runs the census does NOT carry: their recorded side at the pool's
#: own epoch is on disk, so their cross-check exists.
CENSUS_LESS_RUNS = [
    ("cfm_c7_fullgen_ep4_partial_2026_10_20", 3),
    ("cfm_c8_fullgen_2026_10_24", 2),
]


def _numbers(obj) -> set:
    """Every number that appears anywhere inside ``obj``."""
    out = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _numbers(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(float(obj))
    return out


def _events_rows(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _docstring() -> str:
    from molmetal.scripts import cfm_fragmentation_audit as mod

    return mod.__doc__ or ""


def _b64_contrast_block() -> str:
    """The docstring paragraph that carries the b64 contrast."""
    text = _docstring()
    start = text.find("MEASURED contrast on")
    assert start >= 0, "the docstring no longer carries a b64 contrast"
    end = text.find("Read it with the direction of the metric", start)
    assert end > start, "the b64 contrast paragraph could not be located"
    return text[start:end]


class TestTheBlockPresenceFollowsThePoolEpoch:
    """FIX 3."""

    def test_the_two_census_less_runs_now_carry_the_block(self):
        blob = _artifact()
        for run, epoch in CENSUS_LESS_RUNS:
            row = _run_row(blob, run)
            rvr = row.get("recorded_vs_rederived")
            assert rvr is not None, f"{run} has no cross-check"
            # The recorded side is at the POOL'S OWN epoch.
            assert rvr["recorded_at_epoch"] == epoch
            assert rvr["recorded_single_fragment_ratio"] is not None
            assert rvr["recorded_source"]
            # Every OTHER key stays populated.
            assert rvr["rederived_single_fragment_ratio"] is not None
            assert rvr["abs_diff"] is not None
            assert "POOL'S OWN EPOCH" in rvr["note"]

    def test_the_last_epoch_side_is_read_from_the_runs_own_events(self):
        # Wf-A123 FIX 6 (DEFECT B): this used to assert that the last-epoch
        # side was NULL with the reason "the census carries no row for this
        # run" -- a reason that asserted an ABSENCE disk contradicts: both
        # census-less runs carry a last-epoch aggregate in their OWN
        # events.jsonl.  The last-epoch side is now read from there.
        blob = _artifact()
        for run, epoch in CENSUS_LESS_RUNS:
            rvr = _run_row(blob, run).get("recorded_vs_rederived")
            assert rvr["recorded_last_epoch_single_fragment_ratio"] is not None, run
            assert rvr["recorded_last_epoch"] == epoch, run
            source = rvr["recorded_last_epoch_source"]
            assert "events.jsonl" in source, source
            # The reason says what the CENSUS lacks -- not that the value
            # does not exist anywhere.
            assert "census" in source, source
            assert "census carries no row for this run, so no last-epoch" not in source
            # The two numbers stay DISTINCTLY labelled.
            assert rvr["recorded_last_epoch_equals_pool_epoch"] is True, run

    def test_no_run_that_had_the_block_lost_it(self):
        # The 10 runs that carried a block before the fix must still carry
        # one, with the SAME recorded side at the pool's own epoch.
        blob = _artifact()
        carried = [
            r["run"]
            for r in blob["runs"]
            if (r.get("recorded_vs_rederived") or {}).get("recorded_at_epoch")
            is not None
        ]
        for run, _epoch in CENSUS_LESS_RUNS:
            carried.remove(run)
        assert len(carried) == 10, carried
        for run in carried:
            rvr = _run_row(blob, run).get("recorded_vs_rederived")
            assert rvr is not None, run
            assert rvr["recorded_single_fragment_ratio"] is not None, run

    def test_a_census_less_run_gets_the_block_synthetically(self, tmp_path):
        # A synthetic run with NO census row: the recorded side at the pool's
        # own epoch is on disk, so the cross-check must exist.  Wf-A123 FIX 6
        # (DEFECT B): its last-epoch side is read from the run's OWN
        # events.jsonl (epoch 4, 0.75) instead of being published as a null.
        d = tmp_path / "run"
        d.mkdir()
        _write_pool(
            d / "generated_smiles.csv",
            [
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 3, "smiles": "C.C", "n_frags_raw": 2, "largest_frag_frac": 0.5},
            ],
        )
        _write_events(
            d / "events.jsonl",
            [_event_row(3, 0.5, 1003.0), _event_row(4, 0.75, 1004.0)],
        )
        out = audit_run(d, None)
        rvr = out.get("recorded_vs_rederived")
        assert rvr is not None
        assert rvr["recorded_single_fragment_ratio"] == pytest.approx(0.5)
        assert rvr["recorded_at_epoch"] == 3
        # The run's OWN last aggregate row (epoch 4), not a null.
        assert rvr["recorded_last_epoch_single_fragment_ratio"] == pytest.approx(0.75)
        assert rvr["recorded_last_epoch"] == 4
        assert "events.jsonl" in rvr["recorded_last_epoch_source"]
        # The pool's epoch (3) and the run's last epoch (4) DIFFER here, and
        # the artifact says so rather than implying one was copied.
        assert rvr["recorded_last_epoch_equals_pool_epoch"] is False


class TestTheBlindListCarriesTheRunOwnRates:
    """FIX 2 + FIX 4."""

    def test_every_blind_entry_carries_the_verdicts_own_rate(self):
        blob = _artifact()
        blind = blob["blind_validity_signature"]
        assert blind, "the artifact carries no blind-validity signature"
        for entry in blind:
            run = entry["run"]
            verdict = _run_row(blob, run).get("verdict") or {}
            assert entry["single_molecule_rate"] is not None, run
            assert entry["single_molecule_rate"] == pytest.approx(
                verdict["single_molecule_rate"]
            ), run
            # The three LABELLED rates are published beside it.
            row = _run_row(blob, run)
            for key in (
                "single_molecule_rate_over_pool_rows",
                "single_molecule_rate_over_declared_denominator",
                "single_molecule_rate_end_to_end",
            ):
                assert key in entry, f"{run} has no {key}"
                assert entry[key] == pytest.approx(row[key]), f"{run}.{key}"

    def test_the_blind_entry_carries_the_pool_epoch_validity(self):
        blob = _artifact()
        for entry in blob["blind_validity_signature"]:
            run = entry["run"]
            row = _run_row(blob, run)
            pev = row["pool_epoch_validity"]
            assert entry["gen_validity_at_pool_epoch"] == pytest.approx(pev["value"])
            assert entry["gen_validity_n"] == pev["n"]
            assert entry["pool_epoch"] == row["pool_epoch"]
            # ``gen_validity_end`` is CENSUS-sourced and untouched.
            assert entry["gen_validity_end"] == (
                row.get("census") or {}
            ).get("gen_validity_end")

    def test_the_four_census_carried_runs_agree_with_their_census_value(self):
        blob = _artifact()
        census_carried = [
            e
            for e in blob["blind_validity_signature"]
            if e["gen_validity_end"] is not None
        ]
        assert len(census_carried) == 4, census_carried
        for entry in census_carried:
            assert entry["gen_validity_at_pool_epoch"] == pytest.approx(
                entry["gen_validity_end"]
            ), entry["run"]

    def test_the_two_new_runs_read_their_own_pool_epoch(self):
        blob = _artifact()
        blind = {e["run"]: e for e in blob["blind_validity_signature"]}
        for run, epoch in CENSUS_LESS_RUNS:
            entry = blind[run]
            assert entry["gen_validity_end"] is None
            assert entry["gen_validity_at_pool_epoch"] == pytest.approx(1.0)
            assert entry["gen_validity_n"] == 704
            assert entry["pool_epoch"] == epoch

    def test_a_synthetic_blind_entry_carries_the_rates(self, tmp_path, monkeypatch):
        # The blind builder, end to end, on a synthetic run directory.
        from molmetal.scripts import cfm_fragmentation_audit as mod

        reports = tmp_path / "reports"
        d = reports / "synthetic_run"
        d.mkdir(parents=True)
        _write_pool(
            d / "generated_smiles.csv",
            [
                {"epoch": 3, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0},
                {"epoch": 3, "smiles": "C.C", "n_frags_raw": 2, "largest_frag_frac": 0.5},
                {"epoch": 3, "smiles": "C.C", "n_frags_raw": 2, "largest_frag_frac": 0.5},
                {"epoch": 3, "smiles": "C.C", "n_frags_raw": 2, "largest_frag_frac": 0.5},
            ],
        )
        _write_events(d / "events.jsonl", [_event_row(3, 0.5, 1003.0)])
        out = tmp_path / "artifact.json"
        monkeypatch.setattr(mod, "REPORTS", reports)
        monkeypatch.setattr(mod, "CENSUS", tmp_path / "no_census.json")
        assert mod.main(["--out", str(out)]) == 0
        blob = json.loads(out.read_text(encoding="utf-8"))
        blind = blob["blind_validity_signature"]
        assert len(blind) == 1, blind
        entry = blind[0]
        row = _run_row(blob, "synthetic_run")
        verdict = row["verdict"]
        assert verdict["blind_validity_signature"] is True
        assert entry["single_molecule_rate"] == pytest.approx(
            verdict["single_molecule_rate"]
        )
        assert entry["single_molecule_rate"] is not None
        for key in (
            "single_molecule_rate_over_pool_rows",
            "single_molecule_rate_over_declared_denominator",
            "single_molecule_rate_end_to_end",
        ):
            assert key in entry
        assert entry["gen_validity_at_pool_epoch"] == pytest.approx(1.0)
        assert entry["gen_validity_n"] == 4
        assert entry["pool_epoch"] == 3


class TestTheDocstringContrast:
    """FIX 5."""

    def test_the_false_attribution_is_gone(self):
        assert "0.0085" not in _docstring()

    def test_the_b64_contrast_numbers_are_on_disk(self):
        block = _b64_contrast_block()
        quoted = {
            float(m)
            for m in re.findall(r"\*\*([0-9]+(?:\.[0-9]+)?)\*\*", block)
        }
        assert quoted, "the b64 contrast quotes no number"
        events = _numbers(_events_rows(B64_EVENTS))
        audit = _numbers(_run_row(_artifact(), B64_RUN))
        for number in quoted:
            assert number in events or number in audit, (
                f"{number} is on neither the raw events rows nor the audit row"
            )

    def test_the_trainer_side_numbers_are_in_the_raw_events_rows(self):
        block = _b64_contrast_block()
        events = _numbers(_events_rows(B64_EVENTS))
        for number, quoted in (
            (0.5986159169550173, "**0.5986159169550173**"),
            (0.3763020833333333, "**0.3763020833333333**"),
            (768.0, "**768**"),
            (8.0, "**8**"),
        ):
            assert number in events, f"{number} is not in the raw events rows"
            assert quoted in block, f"{quoted} is not quoted in the contrast"

    def test_the_reconstruction_number_is_the_published_one(self):
        block = _b64_contrast_block()
        assert "**0.75**" in block
        row = _run_row(_artifact(), B64_RUN)
        assert row["single_fragment_ratio"] == pytest.approx(0.75)
        assert row["n_rows_in_csv"] == 40
        assert row["n_single"] == 30


class TestTheSmokeDeviceResolution:
    """FIX 6 (a): the CPU path is REACHABLE, not an accident of the machine."""

    def test_the_cpu_override_wins_even_when_cuda_is_available(self, monkeypatch):
        from molmetal.scripts import metallo_drug_smoke_retrain as smoke

        monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: True)
        assert smoke._resolve_device({"MOLMETAL_SMOKE_DEVICE": "cpu"}) == "cpu"

    def test_the_default_is_unchanged(self, monkeypatch):
        from molmetal.scripts import metallo_drug_smoke_retrain as smoke

        monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: True)
        assert smoke._resolve_device({}) == "cuda"
        monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: False)
        assert smoke._resolve_device({}) == "cpu"

    def test_an_unknown_override_falls_back_to_the_default(self, monkeypatch):
        from molmetal.scripts import metallo_drug_smoke_retrain as smoke

        monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: False)
        assert smoke._resolve_device({"MOLMETAL_SMOKE_DEVICE": "tpu"}) == "cpu"

    def test_the_on_cpu_test_carries_the_cpu_forcing_env(self):
        src = (REPO_ROOT / "molmetal" / "tests" / "test_smoke_flag.py").read_text(
            encoding="utf-8"
        )
        assert 'env["CUDA_VISIBLE_DEVICES"] = ""' in src
        assert 'env["MOLMETAL_SMOKE_DEVICE"] = "cpu"' in src
        # The subprocess must actually RECEIVE that env.
        assert "env=env" in src
        # And the run must be asserted to have been CPU.
        assert 'metrics["config"]["device"] == "cpu"' in src
        assert 'metrics["config"]["cuda_available"] is False' in src


class TestTheCohortReasonStatesOnlyWhatIsOnDisk:
    """FIX 1 -- ``cfm_cohort_recovery.py``'s reason may not name a file that
    is not there."""

    def _run_dir(self, tmp_path, *, with_metrics_file, file_epochs):
        d = tmp_path / "run"
        d.mkdir()
        _write_events(
            d / "events.jsonl",
            [_event_row(0, 0.5, 1000.0), _event_row(1, 0.5, 1001.0)],
        )
        if with_metrics_file:
            (d / "per_epoch_metrics.json").write_text(
                json.dumps(
                    {
                        "config": {"n_train": 512, "n_val": 192},
                        "per_epoch_metrics": [{"epoch": e} for e in file_epochs],
                    }
                ),
                encoding="utf-8",
            )
        return d

    def test_no_metrics_file_means_no_overwrite_clause(self, tmp_path):
        from molmetal.scripts.cfm_cohort_recovery import recover_cohort

        d = self._run_dir(tmp_path, with_metrics_file=False, file_epochs=[])
        out = recover_cohort(d, {1000.0, 1001.0})
        assert out["source"] == "unattested"
        assert "overwritten by a later run" not in out["reason"]
        assert "no per-epoch metrics file" in out["reason"]

    def test_a_clobbered_metrics_file_keeps_the_clause(self, tmp_path):
        from molmetal.scripts.cfm_cohort_recovery import recover_cohort

        d = self._run_dir(tmp_path, with_metrics_file=True, file_epochs=[0])
        out = recover_cohort(d, {1000.0, 1001.0})
        assert out["source"] == "unattested"
        assert "overwritten by a later run" in out["reason"]

    def test_a_matching_metrics_file_does_not_claim_an_overwrite(self, tmp_path):
        from molmetal.scripts.cfm_cohort_recovery import recover_cohort

        d = self._run_dir(tmp_path, with_metrics_file=True, file_epochs=[0, 1])
        out = recover_cohort(d, {1000.0, 1001.0})
        assert out["source"] == "unattested"
        assert "overwritten by a later run" not in out["reason"]

    def test_the_real_run_directory_has_no_metrics_file(self):
        # The run whose reason named a file that is not there.
        d = REPO_ROOT / "molmetal" / "reports" / CENSUS_LESS_RUNS[0][0]
        assert not (d / "per_epoch_metrics.json").is_file()
        blob = _artifact()
        cohort = (
            _run_row(blob, CENSUS_LESS_RUNS[0][0])["pool_epoch_validity"]["population"]
        )["cohort"]
        assert "overwritten by a later run" not in cohort["reason"]
        assert "no per-epoch metrics file" in cohort["reason"]
