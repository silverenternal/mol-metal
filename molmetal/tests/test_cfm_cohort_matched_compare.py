# Licensed under the Apache License, Version 2.0.
"""Wf-A114 CLUSTER D -- the cohort-matched comparison harness, and its refusals.

The CFM line's generation quality was never measured on the right denominator.
``gen_validity`` is an RDKit PARSE rate and ``gen_single_fragment_ratio`` is
conditioned on the PARSED ROWS ONLY, so the honest end-to-end rate is

    e2e = (gen_validity_val / gen_validity_n) * gen_single_fragment_ratio

and a number quoted from one cohort, at one epoch, at one budget was being
compared against a number from another cohort, at another epoch, at another
budget.  These tests pin the five axes that make a comparison legitimate, and
-- just as important -- pin the REFUSALS.  A refusal is a success.

The axes:

1. ``cohort``       -- ``config.n_train`` and ``config.n_val`` must be equal;
2. ``epoch_budget`` -- both runs must have a RECORDED row at the compared
   epoch index (a run that stopped at epoch 4 never contributes an epoch-10
   value);
3. ``not_measured`` -- the contributing row must satisfy ``gen_validity_n > 0``
   (the trainer writes ``0.0`` into the gen columns when the generation eval
   did NOT run, which is indistinguishable from a measured 0);
4. ``reading``      -- the ``best`` reading needs both runs to cover the SAME
   epochs;
5. ``history``      -- each run's epoch set must be contiguous AND start at 0,
   and the contributing row must not carry a NaN ``train_loss``;
6. ``provenance``   -- the GENERATION pool (never the reconstruction pool),
   matching pool construction, and -- via the ONE shared rule in
   ``molmetal/scripts/cfm_pool_kind.py`` -- an on-disk pool that is a
   GENERATION pool and does not contradict the per-epoch row.

CPU-only, standard library only, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.cfm_cohort_matched_compare import (  # noqa: E402
    AXIS_COHORT,
    AXIS_EPOCH_BUDGET,
    AXIS_HISTORY,
    AXIS_NOT_MEASURED,
    AXIS_POPULATION,
    AXIS_PROVENANCE,
    AXIS_READING,
    compare_runs,
    e2e_generation,
    find_suggested_pair,
    load_run,
    main,
)
from molmetal.scripts.cfm_pool_kind import (  # noqa: E402
    KIND_GENERATION,
    KIND_RECONSTRUCTION,
    classify_pool,
)

REPORTS_ROOT = REPO_ROOT / "molmetal" / "reports"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _row(
    epoch: int,
    *,
    n: int = 100,
    valid: int = 90,
    sfr: float = 0.5,
    solver: str = "midpoint/50",
    eval_n: int = 256,
    seeds: str = "0,1,2,3",
    loss: float = 1.0,
) -> dict:
    """One synthetic per-epoch row, shaped like the trainer's."""
    return {
        "epoch": epoch,
        "gen_validity_n": n,
        "gen_validity_val": valid,
        "gen_validity": valid / n,
        "gen_single_fragment_ratio": sfr,
        "gen_solver": solver,
        "gen_eval_n": eval_n,
        "gen_eval_seeds": seeds,
        "train_loss": loss,
    }


def _write_run(
    root: Path,
    name: str,
    rows,
    *,
    n_train: int = 256,
    n_val: int = 192,
    declared_epochs=None,
) -> Path:
    """Write a synthetic run directory with a per_epoch_metrics.json."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    payload = {
        "config": {
            "n_train": n_train,
            "n_val": n_val,
            "epochs": declared_epochs
            if declared_epochs is not None
            else (max(r["epoch"] for r in rows) + 1 if rows else 0),
        },
        "per_epoch_metrics": rows,
    }
    (directory / "per_epoch_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return directory


def _write_pool(
    directory: Path, header: str, *, epoch: int = 0, n_rows: int = 3, n_frags: int = 1
) -> Path:
    """Write an on-disk pool CSV with the given header.

    The rows carry the header's own columns, so a ``n_frags_raw`` pool reports
    the single-fragment fraction the harness corroborates against.
    """
    pool = directory / "generated_smiles.csv"
    columns = header.split(",")
    lines = [header]
    for _ in range(n_rows):
        cells = []
        for column in columns:
            if column == "epoch":
                cells.append(str(epoch))
            elif column in ("n_frags_raw", "n_fragments"):
                cells.append(str(n_frags))
            elif column == "metal_z":
                cells.append("77")
            elif column == "sanitized":
                # The era-A content of a reconstruction pool: the bond head's
                # own sanitised flag on a real geometry.  A metal_z header
                # alone is NOT decidable (one script wrote one header in two
                # eras), so the reconstruction pool must carry this too.
                cells.append("1")
            else:
                cells.append("x")
        lines.append(",".join(cells))
    pool.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pool


def _contiguous_rows(first: int, last: int, **kwargs) -> list:
    return [_row(e, **kwargs) for e in range(first, last + 1)]


#: The cohort of the RUN OF RECORD in the synthetic clobbered directory.
CLOBBERED_ROR_COHORT = (26715, 192)
#: The cohort in the SURVIVING (clobbered) per-epoch file's config block.
CLOBBERED_SMOKE_COHORT = (64, 32)
#: The published artifact's three named runs and their cohorts.
BAR_RUNS = (
    ("cfm_r9_fix_v2", (26715, 6714)),
    ("cfm_fullcorpus_seed1", (26715, 192)),
    ("cfm_fm_ab_2026_10_09", (256, 192)),
)


def _write_clobbered_run(
    root: Path,
    name: str = "clobbered",
    *,
    ror_epochs: int = 4,
    ror_loss_base: float = 1.0,
) -> Path:
    """A run directory whose per-epoch file is a LATER SMOKE's record.

    This reproduces the defect MEASURED on disk in
    ``molmetal/reports/cfm_fullcorpus_2026_10_10``: a 4-epoch run of record
    at cohort 26715/192, then a 1-epoch CPU smoke at 64/32 whose ``done``
    event is LATER.  ``per_epoch_metrics.json`` is written ONCE at the END of
    training to a FIXED path, so the SURVIVOR is the smoke's: its config
    block, its epoch series and everything derived from them.

    The directory therefore holds three artifacts that disagree on purpose:

    * ``events.jsonl`` -- append-only, so it keeps BOTH populations (the
      run of record's 4 epoch rows and the smoke's 1);
    * ``master.log`` -- the run of record's log: it echoes EVERY ``ts`` of
      that population and states the cohort it built;
    * ``per_epoch_metrics.json`` -- the SMOKE's record (config 64/32, one
      epoch, a different ``train_loss``).
    """
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)

    ror_events = [
        {
            "event": "train_epoch",
            "epoch": e,
            "ts": 1000.0 + e,
            "train_loss": ror_loss_base + 0.1 * e,
        }
        for e in range(ror_epochs)
    ]
    smoke_event = {
        "event": "train_epoch",
        "epoch": 0,
        "ts": 2000.0,
        "train_loss": 13.68301,
    }
    events = list(ror_events) + [
        {"event": "done", "ts": 1010.0, "n_epochs_run": ror_epochs, "n_epochs": ror_epochs},
        smoke_event,
        {"event": "done", "ts": 2010.0, "n_epochs_run": 1, "n_epochs": 1},
    ]
    (directory / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )

    # The run of record's log: the trainer's cohort line and the event rows
    # it printed (each one carries the `ts` that ties the log to the
    # population).
    log_lines = [
        f"[Wf-124 P2] building cohorts: train={CLOBBERED_ROR_COHORT[0]}, "
        f"val={CLOBBERED_ROR_COHORT[1]}"
    ]
    log_lines.extend(json.dumps(row) for row in ror_events)
    (directory / "master.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    # The SMOKE's record -- the file that survived at the FIXED path.
    payload = {
        "config": {
            "n_train": CLOBBERED_SMOKE_COHORT[0],
            "n_val": CLOBBERED_SMOKE_COHORT[1],
            "epochs": 1,
        },
        "per_epoch_metrics": [_row(0, n=32, valid=32, sfr=1.0, loss=13.68301)],
    }
    (directory / "per_epoch_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return directory


# --------------------------------------------------------------------------
# 1. axis 1 -- cohort
# --------------------------------------------------------------------------
class TestCohortAxis:
    def test_refuses_when_n_train_differs(self, tmp_path):
        """Two runs on different cohorts must not be compared."""
        _write_run(tmp_path, "a", _contiguous_rows(0, 4), n_train=256, n_val=192)
        _write_run(tmp_path, "b", _contiguous_rows(0, 4), n_train=26715, n_val=192)
        a = load_run(tmp_path, "a")
        b = load_run(tmp_path, "b")

        result = compare_runs(a, b)

        assert result.ok is False
        assert AXIS_COHORT in result.unmatched_axes
        detail = result.axes[AXIS_COHORT].detail
        assert "n_train" in detail
        assert "256" in detail and "26715" in detail

    def test_refuses_when_n_val_differs(self, tmp_path):
        """A different validation cohort is a different cohort."""
        _write_run(tmp_path, "a", _contiguous_rows(0, 4), n_train=256, n_val=192)
        _write_run(tmp_path, "b", _contiguous_rows(0, 4), n_train=256, n_val=6714)
        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"))

        assert result.ok is False
        assert AXIS_COHORT in result.unmatched_axes
        assert "n_val" in result.axes[AXIS_COHORT].detail


# --------------------------------------------------------------------------
# 2. axis 2 -- epoch budget
# --------------------------------------------------------------------------
class TestEpochBudgetAxis:
    def test_refuses_when_the_epoch_index_is_not_reached_by_both_runs(self, tmp_path):
        """A run that stopped at epoch 2 never contributes an epoch-4 value."""
        _write_run(tmp_path, "a", _contiguous_rows(0, 4))
        _write_run(tmp_path, "b", _contiguous_rows(0, 2))
        a = load_run(tmp_path, "a")
        b = load_run(tmp_path, "b")

        result = compare_runs(a, b, epoch_index=4)

        assert result.ok is False
        assert AXIS_EPOCH_BUDGET in result.unmatched_axes
        detail = result.axes[AXIS_EPOCH_BUDGET].detail
        assert "b" in detail
        assert "no recorded row at epoch 4" in detail

    def test_the_budget_rule_compares_at_the_earlier_of_the_two_last_epochs(self, tmp_path):
        """K = min(last_A, last_B) + 1, so the index is min(last_A, last_B)."""
        _write_run(tmp_path, "a", _contiguous_rows(0, 9))
        _write_run(tmp_path, "b", _contiguous_rows(0, 4))
        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"))

        assert result.epoch_index == 4
        assert result.axes[AXIS_EPOCH_BUDGET].ok is True


# --------------------------------------------------------------------------
# 3. axis 2b -- the contributing row must carry a measurement
# --------------------------------------------------------------------------
class TestNotMeasuredAxis:
    def test_refuses_when_the_contributing_row_has_no_generation_measurement(self, tmp_path):
        """gen_validity_n == 0 means the eval did not run -- never a measured 0."""
        rows = _contiguous_rows(0, 4)
        rows[4]["gen_validity_n"] = 0
        rows[4]["gen_validity_val"] = 0
        rows[4]["gen_validity"] = 0.0
        rows[4]["gen_single_fragment_ratio"] = 0.0
        _write_run(tmp_path, "a", rows)
        _write_run(tmp_path, "b", _contiguous_rows(0, 4))

        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"))

        assert result.ok is False
        assert AXIS_NOT_MEASURED in result.unmatched_axes
        detail = result.axes[AXIS_NOT_MEASURED].detail
        assert "gen_validity_n=0" in detail
        assert "DEFAULT" in detail
        # the unmeasured row must NOT become a 0.0 in the payload
        assert result.values["a"] is None

    def test_gen_eval_n_is_not_the_measurement_flag(self, tmp_path):
        """11 rows on disk carry gen_eval_n == 0 while gen_validity_n > 0."""
        rows = _contiguous_rows(0, 4)
        rows[4]["gen_eval_n"] = 0
        _write_run(tmp_path, "a", rows)
        _write_run(tmp_path, "b", _contiguous_rows(0, 4))

        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"))

        assert result.axes[AXIS_NOT_MEASURED].ok is True
        assert result.values["a"] == pytest.approx(0.9 * 0.5)


# --------------------------------------------------------------------------
# 4. axis 3 -- reading
# --------------------------------------------------------------------------
class TestReadingAxis:
    def test_refuses_when_the_epoch_sets_differ_for_the_best_reading(self, tmp_path):
        """A max over a noisy series is only comparable over the same range."""
        _write_run(tmp_path, "a", _contiguous_rows(0, 4))
        _write_run(tmp_path, "b", _contiguous_rows(0, 9))
        a = load_run(tmp_path, "a")
        b = load_run(tmp_path, "b")

        result = compare_runs(a, b, reading="best")

        assert result.ok is False
        assert AXIS_READING in result.unmatched_axes
        assert "SAME epochs" in result.axes[AXIS_READING].detail

    def test_the_best_reading_is_allowed_when_the_epoch_sets_match(self, tmp_path):
        _write_run(tmp_path, "a", _contiguous_rows(0, 4))
        _write_run(tmp_path, "b", _contiguous_rows(0, 4))
        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"), reading="best")

        assert result.axes[AXIS_READING].ok is True


# --------------------------------------------------------------------------
# 5. axis 4 -- history
# --------------------------------------------------------------------------
class TestHistoryAxis:
    def test_refuses_when_the_history_does_not_start_at_epoch_zero(self, tmp_path):
        """19 invisible epochs means the epoch index is not the same budget."""
        _write_run(tmp_path, "a", _contiguous_rows(0, 4))
        _write_run(tmp_path, "b", _contiguous_rows(3, 7))
        a = load_run(tmp_path, "a")
        b = load_run(tmp_path, "b")

        result = compare_runs(a, b)

        assert result.ok is False
        assert AXIS_HISTORY in result.unmatched_axes
        detail = result.axes[AXIS_HISTORY].detail
        assert "start at 3, not 0" in detail
        assert "invisible" in detail

    def test_refuses_when_the_contributing_row_has_a_nan_train_loss(self, tmp_path):
        """cfm_r11_eval256's single row carries train_loss = nan."""
        rows = _contiguous_rows(0, 4)
        rows[4]["train_loss"] = float("nan")
        _write_run(tmp_path, "a", rows)
        _write_run(tmp_path, "b", _contiguous_rows(0, 4))

        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"))

        assert result.ok is False
        assert AXIS_HISTORY in result.unmatched_axes
        assert "train_loss = nan" in result.axes[AXIS_HISTORY].detail

    def test_refuses_when_a_non_contributing_row_has_a_nan_train_loss(self, tmp_path):
        """Wf-A114 verify:D D3 -- the WHOLE history is scanned, not one row.

        The compared row is clean here; the break is at epoch 1.  A run whose
        loss went nan mid-training has a broken history whatever row is
        compared, so the comparison must be refused.
        """
        rows = _contiguous_rows(0, 4)
        rows[1]["train_loss"] = float("nan")
        _write_run(tmp_path, "a", rows)
        _write_run(tmp_path, "b", _contiguous_rows(0, 4))

        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"))

        assert result.ok is False
        assert AXIS_HISTORY in result.unmatched_axes
        detail = result.axes[AXIS_HISTORY].detail
        assert "train_loss = nan" in detail
        assert "[1]" in detail
        # the compared row itself is clean -- the refusal is NOT from it
        assert result.rows["a"]["train_loss"] == 1.0

    def test_refuses_when_a_non_contributing_row_has_no_train_loss(self, tmp_path):
        """An absent train_loss anywhere in the history breaks the integrity."""
        rows = _contiguous_rows(0, 4)
        rows[2].pop("train_loss")
        _write_run(tmp_path, "a", rows)
        _write_run(tmp_path, "b", _contiguous_rows(0, 4))

        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"))

        assert result.ok is False
        assert AXIS_HISTORY in result.unmatched_axes
        assert "no train_loss" in result.axes[AXIS_HISTORY].detail


# --------------------------------------------------------------------------
# 6. axis 5 -- provenance
# --------------------------------------------------------------------------
class TestProvenanceAxis:
    def test_refuses_when_the_generation_solver_differs(self, tmp_path):
        """Two pools built by different solvers are not the same measurement."""
        _write_run(tmp_path, "a", _contiguous_rows(0, 4, solver="midpoint/50"))
        _write_run(tmp_path, "b", _contiguous_rows(0, 4, solver="rk4/100"))
        a = load_run(tmp_path, "a")
        b = load_run(tmp_path, "b")

        result = compare_runs(a, b)

        assert result.ok is False
        assert AXIS_PROVENANCE in result.unmatched_axes
        detail = result.axes[AXIS_PROVENANCE].detail
        assert "gen_solver" in detail
        assert "midpoint/50" in detail and "rk4/100" in detail

    def test_refuses_when_the_pool_is_a_reconstruction_not_a_generation(self, tmp_path):
        """A metal_z pool is decoder fidelity, never the generator's quality."""
        a_dir = _write_run(tmp_path, "a", _contiguous_rows(0, 4))
        _write_run(tmp_path, "b", _contiguous_rows(0, 4))
        _write_pool(a_dir, "epoch,seed,smiles,sanitized,n_fragments,metal_z")

        # the shared rule must agree with the harness about the kind
        kind = classify_pool(a_dir / "generated_smiles.csv")
        assert kind.kind == KIND_RECONSTRUCTION

        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"))

        assert result.ok is False
        assert AXIS_PROVENANCE in result.unmatched_axes
        detail = result.axes[AXIS_PROVENANCE].detail
        assert "RECONSTRUCTION" in detail
        assert "metal_z" in detail

    def test_accepts_a_generation_pool_that_agrees_with_the_row(self, tmp_path):
        """A n_frags_raw pool is a generation pool, and it must agree."""
        a_dir = _write_run(tmp_path, "a", _contiguous_rows(0, 4, n=4, valid=4, sfr=1.0))
        _write_run(tmp_path, "b", _contiguous_rows(0, 4, n=4, valid=4, sfr=1.0))
        _write_pool(
            a_dir,
            "epoch,smiles,n_frags_raw,largest_frag_frac",
            epoch=4,
            n_frags=1,
        )

        kind = classify_pool(a_dir / "generated_smiles.csv")
        assert kind.kind == KIND_GENERATION

        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"))

        assert result.axes[AXIS_PROVENANCE].ok is True
        corroboration = result.axes[AXIS_PROVENANCE].extra["pool_kind"]["a"][
            "corroboration"
        ]
        assert corroboration["status"] == "AGREES"

    def test_the_reconstruction_family_is_never_read_for_the_verdict(self, tmp_path):
        """decode_ratio_val / decode_single_fragment_ratio are never the verdict."""
        rows = _contiguous_rows(0, 4)
        for row in rows:
            row["decode_ratio_val"] = 0.99
            row["decode_single_fragment_ratio"] = 1.0
        _write_run(tmp_path, "a", rows)
        _write_run(tmp_path, "b", rows)

        result = compare_runs(load_run(tmp_path, "a"), load_run(tmp_path, "b"))

        # 0.9 * 0.5 = 0.45 -- the GENERATION number, not the reconstruction's 0.99
        assert result.values["a"] == pytest.approx(0.45)


# --------------------------------------------------------------------------
# 7. the happy path
# --------------------------------------------------------------------------
class TestVerdict:
    def test_computes_the_paired_delta_at_the_matched_epoch(self, tmp_path):
        """The verdict is a paired delta on one epoch index, both arms."""
        _write_run(
            tmp_path,
            "a",
            _contiguous_rows(0, 4, n=100, valid=50, sfr=0.25),
        )
        _write_run(
            tmp_path,
            "b",
            _contiguous_rows(0, 4, n=100, valid=80, sfr=0.25),
        )
        a = load_run(tmp_path, "a")
        b = load_run(tmp_path, "b")

        result = compare_runs(a, b)

        assert result.ok is True
        assert result.unmatched_axes == []
        assert result.epoch_index == 4
        assert result.values["a"] == pytest.approx(0.5 * 0.25)
        assert result.values["b"] == pytest.approx(0.8 * 0.25)
        assert result.values["b"] - result.values["a"] == pytest.approx(0.3 * 0.25)

    def test_reads_the_full_precision_json_and_never_the_rounded_csv(self):
        """The CSV's 4 dp product and the JSON's differ by 1.13e-5 on disk."""
        json_path = REPORTS_ROOT / "cfm_r11_eval256" / "per_epoch_metrics.json"
        csv_path = REPORTS_ROOT / "cfm_r11_eval256" / "per_epoch_metrics.csv"
        if not json_path.is_file() or not csv_path.is_file():
            pytest.skip("not on disk")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        row = [r for r in payload["per_epoch_metrics"] if r.get("epoch") == 19][0]

        measured = e2e_generation(row)
        assert measured == pytest.approx(0.572265625, abs=1e-12)

        # the CSV's rounded gen_single_fragment_ratio gives a DIFFERENT product
        csv_line = [
            line
            for line in csv_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("19,")
        ][0]
        csv_ratio = float(csv_line.split(",")[csv_line.split(",").index("0.8277")])
        csv_product = 0.69140625 * csv_ratio
        assert csv_product == pytest.approx(0.572276953125, abs=1e-9)
        assert abs(csv_product - measured) > 1e-5
        # the harness must not reproduce the CSV's product
        assert measured != pytest.approx(csv_product, abs=1e-9)


# --------------------------------------------------------------------------
# 8. the CLI, the artifacts and the refusal payload
# --------------------------------------------------------------------------
class TestCli:
    def test_writes_the_verdict_and_the_refusal_payloads_to_disk(self, tmp_path):
        _write_run(tmp_path, "a", _contiguous_rows(0, 4, n=100, valid=50, sfr=0.25))
        _write_run(tmp_path, "b", _contiguous_rows(0, 4, n=100, valid=80, sfr=0.25))
        out_dir = tmp_path / "out"

        code = main(
            [
                "--run-a",
                "a",
                "--run-b",
                "b",
                "--reports-root",
                str(tmp_path),
                "--out-dir",
                str(out_dir),
            ]
        )

        assert code == 0
        payload = json.loads((out_dir / "compare.json").read_text(encoding="utf-8"))
        assert payload["verdict"] == "VERDICT"
        assert payload["unmatched_axes"] == []
        assert payload["e2e"]["a"]["label"] == "MEASURED"
        assert "per_epoch_metrics.json" in payload["e2e"]["a"]["source"]
        assert payload["e2e"]["delta_b_minus_a"]["value"] == pytest.approx(0.075)
        assert "one run is one run" in payload["caveat"]
        assert (out_dir / "compare.md").is_file()

    def test_the_refusal_payload_names_every_unmatched_axis(self, tmp_path):
        _write_run(tmp_path, "a", _contiguous_rows(0, 4), n_train=256)
        _write_run(tmp_path, "b", _contiguous_rows(3, 7), n_train=26715)
        out_dir = tmp_path / "out"

        code = main(
            [
                "--run-a",
                "a",
                "--run-b",
                "b",
                "--reports-root",
                str(tmp_path),
                "--out-dir",
                str(out_dir),
            ]
        )

        assert code == 1
        payload = json.loads((out_dir / "compare.json").read_text(encoding="utf-8"))
        assert payload["verdict"] == "REFUSED"
        assert AXIS_COHORT in payload["unmatched_axes"]
        assert AXIS_HISTORY in payload["unmatched_axes"]
        assert payload["e2e"] is None
        assert set(payload["refusal"]["axes"]) == set(payload["unmatched_axes"])

    def test_a_missing_run_is_a_skip(self, tmp_path, capsys):
        code = main(
            [
                "--run-a",
                "does_not_exist",
                "--run-b",
                "also_missing",
                "--reports-root",
                str(tmp_path),
            ]
        )

        assert code == 2
        assert "SKIP" in capsys.readouterr().out


# --------------------------------------------------------------------------
# 9. the suggestion on a refusal
# --------------------------------------------------------------------------
class TestSuggestion:
    def test_suggests_the_nearest_matched_pair_on_disk(self, tmp_path):
        """A refusal names the nearest valid pair in the other run's cohort."""
        _write_run(tmp_path, "a", _contiguous_rows(0, 4), n_train=26715, n_val=6714)
        _write_run(tmp_path, "b", _contiguous_rows(0, 4), n_train=256, n_val=192)
        _write_run(tmp_path, "partner", _contiguous_rows(0, 7), n_train=256, n_val=192)
        a = load_run(tmp_path, "a")
        b = load_run(tmp_path, "b")

        suggestion = find_suggested_pair(tmp_path, a, b, reading="epoch")

        assert suggestion is not None
        assert suggestion["run_b"] == "b"
        assert suggestion["run_a"] == "partner"
        assert suggestion["epoch_index"] == 4


# --------------------------------------------------------------------------
# 10. the bar is INTEGRITY-GATED (Wf-A114 verify:D D1)
# --------------------------------------------------------------------------
class TestBarTraceIsIntegrityGated:
    """A bar that is ON DISK is not automatically a MEASUREMENT.

    Wf-A114 verify:D D1 -- ``trace_bar_to_disk`` used to scan every run x
    every epoch and label the first hit ``MEASURED``.  That re-minted the
    WITHDRAWN number: ``cfm_r11_eval256`` records exactly ONE epoch (index
    19) out of a declared 20, its history does not start at zero and its
    ``train_loss`` is ``nan`` -- and its epoch-19 end-to-end rate is
    ``0.572265625``, the value the census withdrew.
    """

    def test_a_bar_on_a_broken_run_is_rejected_not_measured(self, tmp_path):
        # the run reproduces cfm_r11_eval256: ONE epoch (19), nan loss
        row = _row(19, n=256, valid=232, sfr=1.0)
        row["train_loss"] = float("nan")
        _write_run(tmp_path, "cfm_r11_eval256", [row], n_train=26715, n_val=6714)
        bar = e2e_generation(row)
        assert bar == pytest.approx(232 / 256)

        out_dir = tmp_path / "out"
        code = main(
            [
                "--run-a",
                "cfm_r11_eval256",
                "--run-b",
                "cfm_r11_eval256",
                "--reports-root",
                str(tmp_path),
                "--bar",
                repr(bar),
                "--out-dir",
                str(out_dir),
            ]
        )
        assert code in (0, 1)

        payload = json.loads((out_dir / "compare.json").read_text(encoding="utf-8"))
        assert payload["bar"]["label"] == "REJECTED"
        assert payload["bar"]["value"] == pytest.approx(bar)
        assert "start at 19, not 0" in payload["bar"]["rejection_reason"]
        assert "train_loss = nan" in payload["bar"]["rejection_reason"]
        assert "traced_to" not in payload["bar"]

    def test_a_bar_on_an_intact_run_is_measured(self, tmp_path):
        # every epoch but one carries a distinct rate, so the bar pins ONE
        # epoch (the trace scans epochs in order and reports the first hit)
        rows = [_row(e, n=4, valid=4, sfr=0.5) for e in range(5)]
        rows[2] = _row(2, n=4, valid=4, sfr=1.0)
        _write_run(tmp_path, "clean", rows, n_train=256, n_val=192)
        bar = e2e_generation(rows[2])
        assert bar == pytest.approx(1.0)

        out_dir = tmp_path / "out"
        main(
            [
                "--run-a",
                "clean",
                "--run-b",
                "clean",
                "--reports-root",
                str(tmp_path),
                "--bar",
                repr(bar),
                "--out-dir",
                str(out_dir),
            ]
        )

        payload = json.loads((out_dir / "compare.json").read_text(encoding="utf-8"))
        assert payload["bar"]["label"] == "MEASURED"
        assert payload["bar"]["traced_to"]["run"] == "clean"
        assert payload["bar"]["traced_to"]["epoch"] == 2

    def test_a_bar_no_run_carries_is_projected(self, tmp_path):
        _write_run(tmp_path, "clean", _contiguous_rows(0, 4))
        out_dir = tmp_path / "out"
        main(
            [
                "--run-a",
                "clean",
                "--run-b",
                "clean",
                "--reports-root",
                str(tmp_path),
                "--bar",
                "0.123456789",
                "--out-dir",
                str(out_dir),
            ]
        )

        payload = json.loads((out_dir / "compare.json").read_text(encoding="utf-8"))
        assert payload["bar"]["label"] == "PROJECTED"
        assert "no run on disk carries" in payload["bar"]["source"]


# --------------------------------------------------------------------------
# 11. the best reading records EACH ARM'S OWN epoch (Wf-A114 verify:D D2)
# --------------------------------------------------------------------------
class TestBestReadingRecordsPerArmEpoch:
    """The payload used to claim both compared numbers came from the budget
    index while each arm's maximum actually sat at its own epoch."""

    def test_the_payload_names_the_epoch_of_each_arm(self, tmp_path):
        # A's max is at epoch 1, B's max is at epoch 3, both over 0..4
        rows_a = [
            _row(0, n=4, valid=4, sfr=0.5),
            _row(1, n=4, valid=4, sfr=1.0),
            _row(2, n=4, valid=4, sfr=0.25),
            _row(3, n=4, valid=4, sfr=0.25),
            _row(4, n=4, valid=4, sfr=0.25),
        ]
        rows_b = [
            _row(0, n=4, valid=4, sfr=0.25),
            _row(1, n=4, valid=4, sfr=0.25),
            _row(2, n=4, valid=4, sfr=0.25),
            _row(3, n=4, valid=4, sfr=1.0),
            _row(4, n=4, valid=4, sfr=0.25),
        ]
        _write_run(tmp_path, "a", rows_a)
        _write_run(tmp_path, "b", rows_b)

        out_dir = tmp_path / "out"
        code = main(
            [
                "--run-a",
                "a",
                "--run-b",
                "b",
                "--reports-root",
                str(tmp_path),
                "--reading",
                "best",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert code == 0

        payload = json.loads((out_dir / "compare.json").read_text(encoding="utf-8"))
        assert payload["compared_epochs"] == {"a": 1, "b": 3}
        assert payload["e2e"]["a"]["epoch"] == 1
        assert payload["e2e"]["b"]["epoch"] == 3
        # the budget index is the SEARCHED WINDOW's last index, and the
        # payload says so -- it never claims both numbers came from it
        assert payload["epoch_index"] == 4
        assert payload["searched_epoch_range"] == [0, 4]
        assert "SEARCH WINDOW" in payload["epoch_index_meaning"]
        assert "DIFFERENT epochs" in payload["axes"]["reading"]["detail"]

    def test_the_best_search_never_leaves_the_common_window(self, tmp_path):
        """Both arms search the SAME window, so the max is budget-matched."""
        rows_a = [_row(0, n=4, valid=4, sfr=0.5), _row(1, n=4, valid=4, sfr=1.0)]
        rows_b = [
            _row(0, n=4, valid=4, sfr=0.25),
            _row(1, n=4, valid=4, sfr=0.25),
            _row(2, n=4, valid=4, sfr=1.0),
        ]
        _write_run(tmp_path, "a", rows_a)
        _write_run(tmp_path, "b", rows_b)

        result = compare_runs(
            load_run(tmp_path, "a"), load_run(tmp_path, "b"), reading="best"
        )

        # the budget rule compares at min(last_A, last_B) = 1
        assert result.epoch_index == 1
        # B's maximum is at epoch 2, OUTSIDE the common window -> not used
        assert result.compared_epochs["b"] == 0
        assert result.values["b"] == pytest.approx(0.25)
        # the reading axis refuses the mismatched epoch sets anyway
        assert AXIS_READING in result.unmatched_axes


# --------------------------------------------------------------------------
# 12. the COHORT of the RUN OF RECORD (Wf-A116b) -- a clobbered directory
# --------------------------------------------------------------------------
class TestCohortComesFromTheRunOfRecord:
    """The cohort is the RUN OF RECORD's, not the clobbered file's config.

    ``per_epoch_metrics.json`` is written ONCE at the END of training to a
    FIXED path, so a later run launched into the same ``--out-dir``
    OVERWRITES it with a DIFFERENT run's config block and epoch series.
    MEASURED on disk: ``molmetal/reports/cfm_fullcorpus_2026_10_10`` holds a
    14-epoch run of record at cohort 26715/192 and, in the SAME directory,
    two 1-epoch CPU smokes at 64/32 whose ``done`` events are LATER; the
    surviving file is the SMOKE's.
    """

    def test_a_clobbered_directory_does_not_yield_the_clobbered_cohort(self, tmp_path):
        """The 64/32 in the surviving config block is the SMOKE's cohort."""
        _write_clobbered_run(tmp_path)

        run = load_run(tmp_path, "clobbered")

        # the surviving file's own config block says 64/32 ...
        assert (run.config_n_train, run.config_n_val) == CLOBBERED_SMOKE_COHORT
        # ... and the cohort of the RUN OF RECORD is the ts-attested one
        assert (run.n_train, run.n_val) == CLOBBERED_ROR_COHORT
        assert run.cohort_source == "run_log:ts_matched"
        assert "ts_matched" in run.cohort_source
        assert run.config_cohort_agrees_with_resolved is False

    def test_the_flag_fires_for_a_clobbered_directory(self, tmp_path):
        """The flag + the reason string -- never a silent comparison."""
        _write_clobbered_run(tmp_path)

        run = load_run(tmp_path, "clobbered")

        assert run.per_epoch_file_is_run_of_record is False
        assert run.population_verdict == "class_a_other_population"
        assert "NOT of the run of record" in run.population_reason

    def test_a_clobbered_directory_is_refused_and_publishes_no_number(self, tmp_path):
        """The clobbered series must not be published as the run's."""
        _write_clobbered_run(tmp_path)
        _write_run(
            tmp_path, "peer", _contiguous_rows(0, 3), n_train=26715, n_val=192
        )
        out_dir = tmp_path / "out"

        code = main(
            [
                "--run-a",
                "clobbered",
                "--run-b",
                "peer",
                "--reports-root",
                str(tmp_path),
                "--out-dir",
                str(out_dir),
            ]
        )

        assert code == 1
        payload = json.loads((out_dir / "compare.json").read_text(encoding="utf-8"))
        assert payload["verdict"] == "REFUSED"
        assert AXIS_POPULATION in payload["unmatched_axes"]
        # no epoch series is published for a run whose file is not its own
        assert payload["e2e"] is None
        # the flag and the reason travel with the run
        assert payload["runs"]["a"]["per_epoch_file_is_run_of_record"] is False
        assert (
            payload["runs"]["a"]["population_verdict"] == "class_a_other_population"
        )
        assert "NOT of the run of record" in payload["runs"]["a"]["population_reason"]
        # and the cohort it published is the RUN OF RECORD's
        assert payload["runs"]["a"]["n_train"] == CLOBBERED_ROR_COHORT[0]
        assert payload["runs"]["a"]["n_val"] == CLOBBERED_ROR_COHORT[1]
        assert payload["runs"]["a"]["config_n_train"] == CLOBBERED_SMOKE_COHORT[0]
        # the refusal names the axis and its reason
        detail = payload["axes"][AXIS_POPULATION]["detail"]
        assert "clobbered" in detail
        assert "class_a_other_population" in detail

    def test_a_run_without_an_events_stream_keeps_its_config_block_cohort(self, tmp_path):
        """No events stream -> nothing to cross-check -> the config block stands."""
        _write_run(tmp_path, "plain", _contiguous_rows(0, 3), n_train=256, n_val=192)

        run = load_run(tmp_path, "plain")

        assert (run.n_train, run.n_val) == (256, 192)
        assert run.cohort_source == "per_epoch_metrics.json:config"
        assert run.per_epoch_file_is_run_of_record is True
        assert run.population_verdict == "events_absent"

    def test_the_three_named_bar_runs_are_unaffected(self):
        """The published artifact's three runs must resolve as they did.

        The artifact ``metrics/by_metric/cfm_cohort_matched_bar.json`` is
        generated over these three NAMED runs, so their cohorts -- and the
        bar values derived from them -- must not move.
        """
        for name, cohort in BAR_RUNS:
            if not (REPORTS_ROOT / name / "per_epoch_metrics.json").is_file():
                pytest.skip(f"{name} is not on disk")
            run = load_run(REPORTS_ROOT, name)
            assert (run.n_train, run.n_val) == cohort, name
            assert (run.config_n_train, run.config_n_val) == cohort, name
            assert run.per_epoch_file_is_run_of_record is True, name
            assert run.population_verdict == "no_conflict", name

    def test_the_published_bar_artifact_agrees_with_the_resolved_cohorts(self):
        """The artifact's bars are the runs' cohorts -- they must still agree."""
        artifact = REPO_ROOT / "metrics" / "by_metric" / "cfm_cohort_matched_bar.json"
        if not artifact.is_file():
            pytest.skip("the bar artifact is not on disk")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        for name, bar in payload["bars"].items():
            run = load_run(REPORTS_ROOT, bar["traced_to"]["run"])
            assert bar["cohort"]["n_train"] == run.n_train, name
            assert bar["cohort"]["n_val"] == run.n_val, name
            row = run.row_at(int(bar["traced_to"]["epoch"]))
            assert e2e_generation(row) == pytest.approx(bar["value"], abs=1e-12), name
