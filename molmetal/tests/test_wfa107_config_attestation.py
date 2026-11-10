"""The three UNGUARDED config reads must go through the run's owner.

Wf-A121 / I3 (R5).  ``per_epoch_metrics.json`` is written ONCE, at the END of
training, to a FIXED path, so two runs launched into the same ``--out-dir``
clobber each other's record and the survivor is whichever finished LAST.
Reading ``n_train`` / ``n_val`` (or the A/B's knobs) straight out of that
file therefore publishes a SIBLING POPULATION's config.

These tests build the two shapes on disk -- a run whose record IS the run of
record, and a run whose record was clobbered by a later smoke -- and assert
that the first still publishes its numbers and the second publishes null plus
the reason.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from molmetal.scripts import post_hoc_reselect_ckpt as phrc
from molmetal.scripts import wfa107_ab_analysis as wfa107
from molmetal.scripts import wfa107e_ab_analysis as wfa107e
from molmetal.scripts.cfm_cohort_recovery import (
    COHORT_SOURCE_CONFIG,
    COHORT_SOURCE_UNATTESTED,
)


def _write_run(
    run_dir: Path,
    *,
    n_epochs: int = 5,
    recorded_epochs: tuple[int, ...] | None = None,
    config: dict | None = None,
    extra_knobs: dict | None = None,
) -> Path:
    """A run directory: an event stream, a metrics file and a recorded series.

    ``recorded_epochs`` is the epoch set the RECORDED per-epoch file holds.
    The default is the whole run (the record IS the run of record); a shorter
    set is the clobbered case -- a later smoke overwrote the file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(n_epochs))
    with (run_dir / "events.jsonl").open("w") as handle:
        for epoch in epochs:
            handle.write(
                json.dumps({"event": "train_epoch", "epoch": epoch, "ts": 1000.0 + epoch})
                + "\n"
            )
        handle.write(
            json.dumps(
                {
                    "event": "done",
                    "ts": 2000.0,
                    "n_epochs_run": n_epochs,
                    "best_epoch": epochs[-1],
                }
            )
            + "\n"
        )

    cfg = {"n_train": 200, "n_val": 64, "epochs": n_epochs, "seed": 0}
    cfg.update(extra_knobs or {})
    (run_dir / "per_epoch_metrics.json").write_text(
        json.dumps(
            {
                "config": cfg,
                "best": {"epoch": epochs[-1], "gate_metric": "combined"},
                "per_epoch_metrics": [
                    {"epoch": e, "train_loss": 1.0, "bond_f1": 0.5} for e in epochs
                ],
            }
        )
    )
    recorded = epochs if recorded_epochs is None else list(recorded_epochs)
    with (run_dir / "per_epoch_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        for epoch in recorded:
            writer.writerow({"epoch": epoch, "train_loss": 1.0})
    return run_dir


# ---------------------------------------------------------------------------
# wfa107_ab_analysis.py -- the cohort comes from the owner
# ---------------------------------------------------------------------------


def test_the_cohort_comes_from_the_owner_when_the_record_is_the_run_of_record(tmp_path):
    run_dir = _write_run(tmp_path / "wfa107_control")
    cohort = wfa107._cohort_from_owner(run_dir)
    assert cohort["n_train"] == 200
    assert cohort["n_val"] == 64
    assert cohort["source"] == COHORT_SOURCE_CONFIG
    assert not cohort["unattested"]


def test_a_clobbered_record_publishes_no_cohort(tmp_path):
    # the run of record ran 5 epochs; the surviving file holds ONE epoch --
    # a later smoke overwrote it.
    run_dir = _write_run(tmp_path / "wfa107_control", recorded_epochs=(0,))
    cohort = wfa107._cohort_from_owner(run_dir)
    assert cohort["n_train"] is None
    assert cohort["n_val"] is None
    assert cohort["source"] == COHORT_SOURCE_UNATTESTED
    assert cohort["unattested"]
    assert cohort["reason"]


def test_the_arm_publishes_null_and_the_reason_for_a_clobbered_record(tmp_path):
    root = tmp_path
    _write_run(root / "wfa107_control", recorded_epochs=(0,))
    arm = wfa107._load_arm(root, "control")
    assert arm is not None
    assert arm["config"]["n_train"] is None
    assert arm["config"]["n_val"] is None
    assert arm["config"]["cohort_source"] == COHORT_SOURCE_UNATTESTED
    assert arm["config"]["cohort_reason"]
    # the rest of the config block is untouched
    assert arm["config"]["epochs"] == 5
    assert arm["config"]["seed"] == 0


def test_an_attested_arm_publishes_the_owners_numbers_and_no_extra_keys(tmp_path):
    root = tmp_path
    _write_run(root / "wfa107_control")
    arm = wfa107._load_arm(root, "control")
    assert arm is not None
    assert arm["config"]["n_train"] == 200
    assert arm["config"]["n_val"] == 64
    # an attested cohort needs no note: the artifact stays byte-identical
    assert "cohort_source" not in arm["config"]
    assert "cohort_reason" not in arm["config"]


# ---------------------------------------------------------------------------
# wfa107e_ab_analysis.py -- the knobs come from the owner
# ---------------------------------------------------------------------------


def test_the_wfa107e_arm_is_attested_when_the_record_is_the_run_of_record(tmp_path):
    run_dir = _write_run(
        tmp_path / "run",
        extra_knobs={"grad_clip_scope": "per_module", "max_norm": 1.0},
    )
    attestation = wfa107e._attest_arm(run_dir)
    assert attestation["attested"] is True
    arm = wfa107e._load_arm(str(run_dir))
    assert arm["grad_clip_scope"] == "per_module"
    assert arm["max_norm"] == 1.0
    assert arm["attested"] is True


def test_a_clobbered_wfa107e_record_publishes_no_knob(tmp_path):
    run_dir = _write_run(
        tmp_path / "run",
        recorded_epochs=(0,),
        extra_knobs={"grad_clip_scope": "per_module", "max_norm": 1.0},
    )
    attestation = wfa107e._attest_arm(run_dir)
    assert attestation["attested"] is False
    assert attestation["reason"]
    arm = wfa107e._load_arm(str(run_dir))
    for knob in wfa107e._KNOBS:
        assert arm[knob] is None
    assert arm["attested"] is False


# ---------------------------------------------------------------------------
# post_hoc_reselect_ckpt.py -- a record that is not the run of record's is
# REFUSED, because its best.epoch belongs to another population
# ---------------------------------------------------------------------------


def test_the_run_of_record_is_attested(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    attested, why = phrc.attest_record(run_dir)
    assert attested is True
    assert why


def test_a_clobbered_record_is_refused(tmp_path):
    run_dir = _write_run(tmp_path / "run", recorded_epochs=(0,))
    attested, why = phrc.attest_record(run_dir)
    assert attested is False
    assert "run of record" in why


def test_the_reselect_refuses_a_clobbered_record(tmp_path):
    run_dir = _write_run(tmp_path / "run", recorded_epochs=(0,))
    (run_dir / "checkpoints").mkdir()
    assert phrc.reselect(run_dir) == 1
