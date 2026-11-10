# SPDX-License-Identifier: Apache-2.0
"""Wf-A115 — a clobbered run's COHORT must come from an artifact tied to the
RUN OF RECORD, or be reported as unattested.

THE DEFECT THIS PINS
--------------------
``molmetal/scripts/wf_cfm_run_census.py`` reads a run's cohort out of
``<run_dir>/per_epoch_metrics.json``'s ``config`` block.  That file is written
ONCE, at the END of training, to a FIXED path, so a later run launched into
the same ``--out-dir`` CLOBBERS it.  Wf-A114E fixed the epoch SERIES by
re-deriving it from the append-only ``events.jsonl``, but the cohort was left
behind: the four clobbered runs came out with ``n_train = None`` and dropped
out of every cohort axis -- including the one they belong to.

The fix is ``molmetal/scripts/cfm_cohort_recovery.py``: a run log attests the
run of record iff it echoes EVERY ``ts`` of that population, and a sibling
population's log is NOT evidence about the run of record.

THE ON-DISK REGRESSION
----------------------
MEASURED, from the real artifacts:

* ``molmetal/reports/cfm_fullcorpus_2026_10_10`` holds THREE populations --
  a 14-epoch run at cohort ``26715/192``, then two 1-epoch CPU smokes at
  ``64/32``.  Its surviving ``master.log`` echoes the 14-epoch population's
  epochs, so the cohort IS recoverable: ``26715/192``.
* ``cfm_round3_retrain/run``, ``cfm_round7_f5f6_retrain/run`` and
  ``cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529`` hold a
  20-epoch run of record whose log did NOT survive (the surviving
  ``train.log`` echoes a 4-epoch population with a different evaluation
  width, ``gen_validity_n`` 768 vs 192).  Their cohort is NOT recoverable
  and must be reported as UNATTESTED -- the sibling's ``256/192`` is not
  evidence about the run of record.

CPU-only.  Reads files; runs no training, no sampling, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.cfm_cohort_recovery import (  # noqa: E402
    COHORT_SOURCE_LOG,
    COHORT_SOURCE_UNATTESTED,
    recover_cohort,
    scan_run_logs,
)

REPORTS = REPO_ROOT / "molmetal" / "reports"


# --- helpers ----------------------------------------------------------------


def _run_of_record_ts(run_dir: Path) -> Set[float]:
    """The ``ts`` of the RUN OF RECORD's ``train_epoch`` rows.

    Re-derived HERE, independently of the census, so this test is a genuine
    cross-check of the rule rather than a restatement of it: the stream is
    segmented on its ``done`` events and the longest population wins, with
    the earliest ``done`` breaking a tie.
    """
    events: List[Dict[str, Any]] = []
    with (run_dir / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    populations: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for obj in events:
        current.append(obj)
        if obj.get("event") == "done":
            populations.append(current)
            current = []
    if current:
        populations.append(current)
    assert populations, "the events stream holds no population"

    def _key(pop: List[Dict[str, Any]]) -> Tuple[int, float]:
        done = [o for o in pop if o.get("event") == "done"]
        n = int(done[-1].get("n_epochs_run") or 0) if done else 0
        ts = float(done[-1].get("ts") or 0.0) if done else 0.0
        return (n, -ts)

    pop = max(populations, key=_key)
    return {
        float(o["ts"])
        for o in pop
        if o.get("event") == "train_epoch" and isinstance(o.get("ts"), (int, float))
    }


def _write_log(path: Path, cohort: Tuple[int, int], epochs: List[int]) -> Path:
    """A log that states a cohort and echoes the given epochs' timestamps."""
    lines = [
        "[Wf-124 P2] device=cuda:0",
        f"[Wf-124 P2] building cohorts: train={cohort[0]}, val={cohort[1]}",
        f"[tmqm-clouds] metals=['Pt']: {cohort[0]} cloud(s) from 1280 candidate(s)",
        f"[tmqm-clouds] metals=['Pt']: {cohort[1]} cloud(s) from 960 candidate(s)",
        f"[Wf-124 P2] cohort built: train={cohort[0]} val={cohort[1]}",
    ]
    for i, ep in enumerate(epochs):
        row = {
            "epoch": ep,
            "event": "train_epoch",
            "train_loss": 1.0 + i,
            "gen_validity": 0.5,
            "gen_validity_n": 192,
            "gen_validity_val": 96,
            "gen_single_fragment_ratio": 0.5,
            "ts": 1000.0 + i,
        }
        lines.append(json.dumps(row, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- the rule, on synthetic artifacts ---------------------------------------


def test_a_log_that_echoes_every_epoch_attests_the_cohort(tmp_path):
    _write_log(tmp_path / "master.log", (26715, 192), [0, 1, 2])
    res = recover_cohort(tmp_path, {1000.0, 1001.0, 1002.0})
    assert res["source"] == COHORT_SOURCE_LOG
    assert (res["n_train"], res["n_val"]) == (26715, 192)


def test_a_log_that_echoes_only_some_epochs_does_not_attest(tmp_path):
    """A SIBLING population's log is not evidence about the run of record."""
    _write_log(tmp_path / "train.log", (256, 192), [0, 1, 2, 3])
    res = recover_cohort(tmp_path, {1000.0, 1001.0, 1002.0, 1003.0, 9000.0})
    assert res["source"] == COHORT_SOURCE_UNATTESTED
    assert res["n_train"] is None and res["n_val"] is None
    assert "does not attest" in res["reason"] or "NOT" in res["reason"]


def test_a_log_with_no_epoch_echo_does_not_attest(tmp_path):
    _write_log(tmp_path / "master.log", (256, 192), [])
    res = recover_cohort(tmp_path, {1000.0, 1001.0})
    assert res["source"] == COHORT_SOURCE_UNATTESTED


def test_two_cohort_statements_that_disagree_are_refused(tmp_path):
    path = _write_log(tmp_path / "master.log", (256, 192), [0, 1])
    text = path.read_text(encoding="utf-8").replace(
        "cohort built: train=256 val=192", "cohort built: train=64 val=32"
    )
    path.write_text(text, encoding="utf-8")
    res = recover_cohort(tmp_path, {1000.0, 1001.0})
    assert res["source"] == COHORT_SOURCE_UNATTESTED
    assert "DIFFERENT" in res["reason"]


def test_a_directory_with_no_log_is_unattested(tmp_path):
    res = recover_cohort(tmp_path, {1000.0})
    assert res["source"] == COHORT_SOURCE_UNATTESTED
    assert res["n_train"] is None


def test_the_config_block_is_used_only_as_a_declared_fallback(tmp_path):
    """With no population known, the caller may still hand over a cohort."""
    res = recover_cohort(tmp_path, set(), config_cohort=(64, 32))
    assert (res["n_train"], res["n_val"]) == (64, 32)
    assert res["source"] == "per_epoch_metrics.json:config"


# --- the rule, on the REAL artifacts ----------------------------------------


def test_the_real_full_corpus_run_recovers_its_cohort_from_its_log():
    """MEASURED: the run of record is the 14-epoch population at 26715/192."""
    d = REPORTS / "cfm_fullcorpus_2026_10_10"
    if not (d / "events.jsonl").is_file():
        pytest.skip("the run directory is not on disk")
    ts = _run_of_record_ts(d)
    assert len(ts) == 14, "the run of record should be the 14-epoch population"
    res = recover_cohort(d, ts)
    assert res["source"] == COHORT_SOURCE_LOG
    assert (res["n_train"], res["n_val"]) == (26715, 192)


@pytest.mark.parametrize(
    "name",
    [
        "cfm_round3_retrain/run",
        "cfm_round7_f5f6_retrain/run",
        "cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529",
    ],
)
def test_the_real_runs_whose_log_died_are_unattested(name):
    """MEASURED: the surviving log belongs to a 4-epoch sibling population."""
    d = REPORTS / name
    if not (d / "events.jsonl").is_file():
        pytest.skip("the run directory is not on disk")
    ts = _run_of_record_ts(d)
    assert len(ts) == 20, "the run of record should be the 20-epoch population"
    res = recover_cohort(d, ts)
    assert res["source"] == COHORT_SOURCE_UNATTESTED
    assert res["n_train"] is None and res["n_val"] is None


def test_the_survivor_of_the_clobber_is_not_the_run_of_record():
    """MEASURED: populations in ONE directory do not share a cohort.

    This is the counter-example that makes the sibling-log rule necessary:
    ``cfm_fullcorpus_2026_10_10``'s per-epoch file carries the 1-epoch CPU
    smoke's cohort (``64/32``), while its run of record is ``26715/192``.
    """
    d = REPORTS / "cfm_fullcorpus_2026_10_10"
    payload = json.loads((d / "per_epoch_metrics.json").read_text(encoding="utf-8"))
    cfg = payload.get("config") or {}
    assert (cfg.get("n_train"), cfg.get("n_val")) == (64, 32)
    ts = _run_of_record_ts(d)
    res = recover_cohort(d, ts)
    assert (res["n_train"], res["n_val"]) == (26715, 192)
    assert (res["n_train"], res["n_val"]) != (cfg.get("n_train"), cfg.get("n_val"))


def test_scan_run_logs_reports_the_cohort_line_it_read():
    d = REPORTS / "cfm_fullcorpus_2026_10_10"
    if not (d / "master.log").is_file():
        pytest.skip("the run directory is not on disk")
    records = scan_run_logs(d)
    assert records, "the run directory holds a log"
    master = [r for r in records if r["file"] == "master.log"]
    assert master, "master.log is scanned"
    assert master[0]["cohort"] == [26715, 192]
    assert master[0]["n_epoch_rows"] == 14


# --- the rule, AS THE CENSUS USES IT ---------------------------------------


def test_the_census_recovers_the_full_corpus_run_cohort_from_its_log():
    """MEASURED: the census row for the clobbered run carries 26715/192."""
    from molmetal.scripts.wf_cfm_run_census import load_run

    d = REPORTS / "cfm_fullcorpus_2026_10_10"
    if not (d / "events.jsonl").is_file():
        pytest.skip("the run directory is not on disk")
    row = load_run(d)
    assert row is not None
    assert row["cohort_source"] == COHORT_SOURCE_LOG
    assert (row["n_train"], row["n_val"]) == (26715, 192)
    assert "ts_matched" in row["cohort_source"]


@pytest.mark.parametrize(
    "name",
    [
        "cfm_round3_retrain/run",
        "cfm_round7_f5f6_retrain/run",
        "cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529",
    ],
)
def test_the_census_leaves_the_unattested_runs_without_a_cohort(name):
    """MEASURED: the census must NOT give these runs their sibling's 256/192."""
    from molmetal.scripts.wf_cfm_run_census import load_run

    d = REPORTS / name
    if not (d / "events.jsonl").is_file():
        pytest.skip("the run directory is not on disk")
    row = load_run(d)
    assert row is not None
    assert row["cohort_source"] == COHORT_SOURCE_UNATTESTED
    assert row["n_train"] is None and row["n_val"] is None
    assert row["cohort_reason"], "an unattested cohort must say why"


def test_a_clobbered_run_does_not_inherit_a_sibling_population_cohort():
    """MEASURED: the census's own cohort axes drop the unattested runs.

    They must land in the ``None/None`` bucket, NOT in ``256/192`` -- which is
    the cohort their 4-epoch sibling population used.
    """
    from molmetal.scripts.wf_cfm_run_census import load_run

    d = REPORTS / "cfm_round3_retrain/run"
    if not (d / "events.jsonl").is_file():
        pytest.skip("the run directory is not on disk")
    row = load_run(d)
    assert row is not None
    assert (row["n_train"], row["n_val"]) != (256, 192)
    assert f"{row['n_train']}/{row['n_val']}" == "None/None"
