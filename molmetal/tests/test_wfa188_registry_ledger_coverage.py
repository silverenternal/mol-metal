# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wf-A188 -- registry / ledger / disk coverage ratchet (CPU-only).

WHAT THIS FILE GUARDS
---------------------
Three coverage properties that no pre-flight gate can see:

  (i)   every metric file on disk has a registry row
        (``uncovered`` must be 0);
  (ii)  every ledger LATEST row's ``source_json_sha256`` equals the sha256 of
        that file on disk (``drift`` must be 0);
  (iii) a RATCHET on the number of registry rows that have NO ledger row
        (``no_run``): the count must not INCREASE above the number measured
        when this test was written.

RATCHET BASELINE (measured 2026-11-01, Wf-A188)
-----------------------------------------------
``no_run`` = **183** registry rows with no ledger row.

That number is NOT a fixed target.  183 registry rows legitimately have no
ledger row: their artefacts predate the ledger, or they are untracked files
that were never run.  The ratchet therefore asserts a NON-INCREASE, not an
equality -- otherwise the next wave that publishes an artefact without
running it would fail the build for something the ledger does not promise.

History of the number, all MEASURED on this working tree:

  * 2026-11-01 BEFORE the Wf-A188 repair: **186**
    (run_ledger.staleness_summary() -> ``no_run: 186``).
  * 2026-11-01 AFTER the Wf-A188 (B) repair, which appended ONE ledger row
    for each of the three uncovered ``wfa155_*`` artefacts: **183**.
    This file was written against the 183 figure.

WHAT THE RATCHET DOES NOT COVER
-------------------------------
The strict git-provenance rule (``wfa166_run_ledger_provenance.audit_ledger``)
is NOT re-implemented here.  It is registered in NO pre-flight pipeline, so
its verdict is invisible to a launch; wiring it in changes
``pre_flight_check.EXPECTED_GATE_COUNTS``, which is a USER decision.  This
test file therefore covers the FILE-level coverage only, and says so.

PROVING THE TEST BITES
----------------------
``test_drift_check_bites_on_a_broken_ledger`` and
``test_ratchet_bites_when_a_ledger_row_disappears`` both operate on a COPY of
the ledger in a pytest ``tmp_path``.  The real
``molmetal/reports/run_ledger.jsonl`` is never mutated by this suite.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in str(sys.path):
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import run_ledger as rl  # noqa: E402
from molmetal.scripts import wfa188_ledger_disk_audit as wfa188  # noqa: E402

LEDGER = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"
REGISTRY = REPO_ROOT / "registry" / "metrics.yaml"

#: MEASURED 2026-11-01 (Wf-A188), see the module docstring.
NO_RUN_RATCHET_BASELINE = 183
NO_RUN_RATCHET_MEASURED_ON = "2026-11-01"

#: MEASURED 2026-11-01 (Wf-A188): the registry held 582 metric rows at the
#: moment the baseline above was taken.  Asserted as a NON-DECREASE only.
REGISTRY_ROWS_AT_BASELINE = 582


def _audit(**kwargs):
    """Run the audit against the repo's registry, with optional overrides."""
    kwargs.setdefault("registry_metrics_path", REGISTRY)
    kwargs.setdefault("repo_root", REPO_ROOT)
    return wfa188.audit(**kwargs)


def _copy_ledger(tmp_path: Path) -> Path:
    """Copy the real ledger into ``tmp_path`` so the suite never mutates it."""
    dest = tmp_path / "run_ledger.jsonl"
    shutil.copy2(LEDGER, dest)
    return dest


# ---------------------------------------------------------------------------
# (i) every metric file on disk has a registry row
# ---------------------------------------------------------------------------
def test_every_metric_file_on_disk_has_a_registry_row():
    report = _audit()
    counts = report["counts"]
    assert counts["uncovered"] == 0, (
        "metric files on disk with NO registry row: {n} -> {list}".format(
            n=counts["uncovered"], list=report["uncovered"]
        )
    )


# ---------------------------------------------------------------------------
# (ii) every ledger LATEST row's sha equals the file's sha on disk
# ---------------------------------------------------------------------------
def test_every_ledger_latest_row_sha_matches_disk():
    report = _audit()
    counts = report["counts"]
    assert counts["drift"] == 0, (
        "ledger LATEST rows whose source_json_sha256 differs from the file "
        "on disk: {n} -> {list}".format(
            n=counts["drift"],
            list=[
                (d["metric_id"], d["ledger_sha256"], d["disk_sha256"])
                for d in report["drift"]
            ],
        )
    )


def test_ledger_latest_rows_cover_every_ledger_metric():
    """The audit must see every metric the ledger mentions (sanity on itself)."""
    report = _audit()
    rows, corrupt = wfa188._read_ledger(LEDGER)
    assert corrupt == [], "corrupt ledger lines: {c}".format(c=corrupt)
    latest = wfa188.latest_by_metric(rows)
    assert report["counts"]["ledger_latest_rows"] == len(latest)
    assert report["counts"]["ledger_rows"] == len(rows)


# ---------------------------------------------------------------------------
# (iii) the ratchet: registry rows with no ledger row must not INCREASE
# ---------------------------------------------------------------------------
def test_ratchet_registry_rows_without_ledger_row_does_not_increase():
    report = _audit()
    no_run = report["counts"]["no_run"]
    assert no_run <= NO_RUN_RATCHET_BASELINE, (
        "registry rows with NO ledger row rose from the {d} baseline of {b} to {n}. "
        "Baseline measured {d} (Wf-A188). The no-run ids are: {ids}".format(
            d=NO_RUN_RATCHET_MEASURED_ON,
            b=NO_RUN_RATCHET_BASELINE,
            n=no_run,
            ids=report["no_run"][:20],
        )
    )


def test_ratchet_baseline_is_reproducible_from_disk():
    """The pinned baseline must match what the disk actually says today.

    The registry row count is asserted as a NON-DECREASE, not an equality:
    the registry legitimately grows every time a wave publishes an artefact,
    and a test that fails on growth is a gate people stop reading.  Only the
    ``no_run`` COUNT is pinned, and it is pinned as a non-increase.
    """
    report = _audit()
    counts = report["counts"]
    assert counts["registry_metric_rows"] >= REGISTRY_ROWS_AT_BASELINE, (
        "registry metric rows SHRANK below the {n} measured at the baseline "
        "(now {m}) -- a shrinking registry means rows were deleted, which is "
        "not a growth event".format(
            n=REGISTRY_ROWS_AT_BASELINE, m=counts["registry_metric_rows"]
        )
    )
    assert counts["no_run"] <= NO_RUN_RATCHET_BASELINE


# ---------------------------------------------------------------------------
# PROOF THAT THE TEST BITES -- both run against a COPY of the ledger
# ---------------------------------------------------------------------------
def test_drift_check_bites_on_a_broken_ledger(tmp_path):
    """Append a row with a WRONG sha to a COPY of the ledger -> drift > 0."""
    broken = _copy_ledger(tmp_path)
    rows, _ = wfa188._read_ledger(broken)
    latest = wfa188.latest_by_metric(rows)
    victim = "m7_diversity"
    assert victim in latest, "fixture metric missing from the ledger"
    src = latest[victim]["source_json"]

    rl.record_run(
        victim,
        latest[victim].get("value"),
        source_json=src,
        git_sha=latest[victim].get("git_sha"),
        host="ZhaoLab",
        source_json_sha256="0" * 64,  # a sha no file on disk can carry
        timestamp="2026-11-01T00:00:00+00:00",
        ledger_path=broken,
    )

    report = _audit(ledger_path=broken)
    drift_ids = [d["metric_id"] for d in report["drift"]]
    assert report["counts"]["drift"] >= 1, "the drift check did NOT fire"
    assert victim in drift_ids, (
        "the broken sha for {v} was not reported; drift ids = {ids}".format(
            v=victim, ids=drift_ids
        )
    )


def test_ratchet_bites_when_a_ledger_row_disappears(tmp_path):
    """Remove a ledger row from a COPY -> the no_run count RISES -> ratchet fires."""
    broken = _copy_ledger(tmp_path)
    rows, _ = wfa188._read_ledger(broken)
    victim = "m7_diversity"
    kept = [r for r in rows if r.get("metric_id") != victim]
    assert len(kept) < len(rows), "fixture metric missing from the ledger"
    with broken.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    report = _audit(ledger_path=broken)
    no_run = report["counts"]["no_run"]
    assert no_run > NO_RUN_RATCHET_BASELINE, (
        "removing one ledger row did not raise the no_run count above the "
        "baseline ({b}); got {n}".format(b=NO_RUN_RATCHET_BASELINE, n=no_run)
    )
    assert victim in report["no_run"]


def test_uncovered_check_bites_when_a_registry_row_is_removed(tmp_path):
    """Drop a registry row from a COPY of the registry -> uncovered > 0."""
    import yaml

    broken = tmp_path / "metrics.yaml"
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    victim = "wfa155_ligand_connected_rate"
    assert victim in data["metrics"], "fixture metric missing from the registry"
    del data["metrics"][victim]
    broken.write_text(
        yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )

    report = _audit(registry_metrics_path=broken)
    assert report["counts"]["uncovered"] >= 1, "the uncovered check did NOT fire"
    assert "wfa155_ligand_connected_rate.json" in report["uncovered"]


# ---------------------------------------------------------------------------
# the audit module itself
# ---------------------------------------------------------------------------
def test_audit_is_read_only_on_the_real_ledger():
    """Running the audit must not change a single byte of the real ledger."""
    before = LEDGER.read_bytes()
    _audit()
    assert LEDGER.read_bytes() == before


def test_audit_reports_the_expected_shape():
    report = _audit()
    for key in ("counts", "drift", "no_run", "uncovered"):
        assert key in report
    assert report["counts"]["corrupt_ledger_lines"] == 0
    text = wfa188.render_summary(report)
    assert "Wf-A188 ledger-vs-disk vs registry coverage audit" in text
    assert "DRIFT" in text and "NO_RUN" in text and "UNCOVERED" in text


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
