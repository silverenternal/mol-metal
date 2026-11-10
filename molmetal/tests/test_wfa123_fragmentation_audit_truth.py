"""Wf-A123 FIX 6 -- the CFM fragmentation audit must not publish a null that
disk contradicts, and must not swallow a broken census.

DEFECT B: the audit published ``recorded_last_epoch_single_fragment_ratio:
null`` with the reason "the census carries no row for this run, so no
last-epoch aggregate value is recorded for it".  The reason asserted an
ABSENCE that disk contradicts -- MEASURED 2026-10-31, the two census-less
runs DO carry a last-epoch aggregate in their own ``events.jsonl``:

  cfm_c7_fullgen_ep4_partial_2026_10_20  epoch 3  0.08380681818181818
  cfm_c8_fullgen_2026_10_24               epoch 2  0.34801136363636365

DEFECT C: ``_census_by_run()`` caught EVERY exception and returned ``{}``, so
an unreadable census was indistinguishable from a census that simply had no
row for a run.

Nothing here touches the GPU and nothing here writes a published artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts import cfm_fragmentation_audit as mod  # noqa: E402

ARTIFACT = PROJECT_ROOT / "metrics" / "by_metric" / "cfm_fragmentation_audit.json"

#: The runs the census does not carry, with the last-epoch aggregate MEASURED
#: from their own events.jsonl (see the module docstring).
CENSUS_LESS = {
    "cfm_c7_fullgen_ep4_partial_2026_10_20": (3, 0.08380681818181818),
    "cfm_c8_fullgen_2026_10_24": (2, 0.34801136363636365),
}


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _run_row(blob: dict, run: str) -> dict:
    for row in blob.get("runs") or []:
        if row.get("run") == run:
            return row
    raise AssertionError(f"run {run!r} is not in the artifact")


def test_the_census_read_outcome_is_published() -> None:
    """DEFECT C: the artifact says how the census was READ."""
    blob = _artifact()
    status = blob.get("census_read")
    assert status is not None, "the artifact does not publish the census read outcome"
    assert status["census_read"] in ("ok", "missing", "parse_error"), status
    assert status["census_present"] is True, status
    assert status["census_rows_read"] > 0, status
    assert status["census_error"] is None, status


def test_every_run_row_carries_the_census_read_outcome() -> None:
    """A reader of ONE run can tell 'no row' from 'could not be read'."""
    blob = _artifact()
    for row in blob.get("runs") or []:
        census = row.get("census") or {}
        assert census.get("census_read") == "ok", row.get("run")
        assert "census_row_present" in census, row.get("run")
        assert census.get("census_rows_read") == blob["census_read"]["census_rows_read"]


def test_a_broken_census_is_reported_not_swallowed(tmp_path: Path) -> None:
    """DEFECT C: an unreadable census yields a parse_error, never a silent {}."""
    broken = tmp_path / "broken_census.json"
    broken.write_text("{not json at all", encoding="utf-8")
    old = mod.CENSUS
    try:
        mod.CENSUS = broken
        rows, status = mod._census_by_run()
    finally:
        mod.CENSUS = old
    assert rows == {}
    assert status["census_read"] == "parse_error", status
    assert status["census_error"], status
    assert "JSONDecodeError" in status["census_error"] or "Error" in status["census_error"]


def test_a_missing_census_is_reported_not_swallowed(tmp_path: Path) -> None:
    """A census that is not there says so."""
    absent = tmp_path / "no_such_census.json"
    old = mod.CENSUS
    try:
        mod.CENSUS = absent
        rows, status = mod._census_by_run()
    finally:
        mod.CENSUS = old
    assert rows == {}
    assert status["census_read"] == "missing", status
    assert status["census_present"] is False, status


@pytest.mark.parametrize("run,expected", sorted(CENSUS_LESS.items()))
def test_the_census_less_run_publishes_its_own_last_epoch_value(
    run: str, expected: tuple
) -> None:
    """DEFECT B: the null is gone; the run's OWN value is published."""
    epoch, value = expected
    blob = _artifact()
    row = _run_row(blob, run)
    agree = row.get("recorded_vs_rederived") or {}
    assert agree.get("recorded_last_epoch_single_fragment_ratio") == pytest.approx(value), (
        f"{run} still publishes a null / wrong last-epoch value: "
        f"{agree.get('recorded_last_epoch_single_fragment_ratio')!r}"
    )
    assert agree.get("recorded_last_epoch") == epoch, agree
    source = agree.get("recorded_last_epoch_source") or ""
    assert "events.jsonl" in source, source
    assert "census" in source, source  # it says WHY the census was not used
    # The value the run's own events carry is the value published.
    events = (
        PROJECT_ROOT
        / "molmetal"
        / "reports"
        / run
        / "events.jsonl"
    )
    last = None
    for line in events.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "train_epoch" or rec.get("type") == "train_epoch":
            last = rec
    assert last is not None, f"no train_epoch row in {events}"
    assert last.get("epoch") == epoch, last
    assert last.get("gen_single_fragment_ratio") == pytest.approx(value), last


@pytest.mark.parametrize("run,expected", sorted(CENSUS_LESS.items()))
def test_the_pool_epoch_and_the_last_epoch_are_labelled_distinctly(
    run: str, expected: tuple
) -> None:
    """The two numbers are separate fields, each with its own source."""
    epoch, _ = expected
    blob = _artifact()
    row = _run_row(blob, run)
    agree = row.get("recorded_vs_rederived") or {}
    assert agree.get("recorded_at_epoch") == epoch, agree
    assert agree.get("recorded_last_epoch") == epoch, agree
    # They coincide BY CONSTRUCTION here, and the artifact says so rather
    # than implying one was copied from the other.
    assert agree.get("recorded_last_epoch_equals_pool_epoch") is True, agree
    assert agree.get("recorded_source") != agree.get("recorded_last_epoch_source"), agree


def test_no_run_publishes_a_null_last_epoch_value() -> None:
    """No run's last-epoch side is a null any more."""
    blob = _artifact()
    for row in blob.get("runs") or []:
        agree = row.get("recorded_vs_rederived") or {}
        if not agree:
            continue
        assert agree.get("recorded_last_epoch_single_fragment_ratio") is not None, (
            f"{row.get('run')} still publishes a null last-epoch value"
        )
