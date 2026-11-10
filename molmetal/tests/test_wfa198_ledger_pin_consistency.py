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

"""Wf-A198 -- tests for the ledger version-pin auditor and the CLI fix.

WHAT IS UNDER TEST
------------------
``molmetal/scripts/wfa198_ledger_pin_consistency.py`` (the auditor): it
measures whether the run ledger's VERSION PIN -- the PAIR
``(git_sha, source_json_sha256)`` -- is self-consistent.  The repository's
pre-flight gate ``p3_run_ledger_provenance`` validates only the sha half.

``molmetal/scripts/run_ledger.py`` (the CLI): the DEFAULT mode of
``run_ledger.py provenance`` used to print a NUMERIC superseded count for a
quantity it never computed.

MEASURED 2026-11-02 (CPU-only, 0 GPU) on this working tree:

* ledger = 432 rows, every row carrying a complete pin;
* 400 rows pin-consistent, **32 inconsistent** (27 sha-mismatch -- the file
  exists at the pinned commit with DIFFERENT content -- and 5 absent-at-
  commit -- the file does not exist at the pinned commit at all);
* the gate's own classification of the 27 sha-mismatch rows leaves
  **19** of them looking REPRODUCIBLE, so the gate cannot see them;
* the CLI's default mode printed ``superseded_never_committed=0`` for a
  quantity it did not measure; ``--all`` printed
  ``superseded_never_committed=9``.

Every number below is a MEASURED value, not a target.  The snapshot
assertions are DATED: if the ledger grows, they fail, and the correct
response is to re-measure and re-publish the report -- not to relax them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import run_ledger as rl  # noqa: E402
from molmetal.scripts import wfa198_ledger_pin_consistency as wfa198  # noqa: E402

LEDGER = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"

#: MEASURED 2026-11-02 (Wf-A198) on this working tree -- see the module
#: docstring.  These are a DATED SNAPSHOT of the ledger as it stood, not a
#: target: the ledger is append-only, so any append invalidates them and the
#: report must be re-measured.
MEASURED_ON = "2026-11-02"
MEASURED_LEDGER_ROWS = 432
MEASURED_PIN_CONSISTENT = 400
MEASURED_PIN_INCONSISTENT = 32
MEASURED_SHA_MISMATCH = 27
MEASURED_ABSENT_AT_COMMIT = 5
MEASURED_GATE_REPRODUCIBLE_OF_SHA_MISMATCH = 19
MEASURED_SUPERSEDED_COUNT_ALL_MODE = 9

#: MEASURED 2026-11-02 (Wf-A198): the five rows whose ``source_json`` does
#: not exist AT ALL at the commit their row pins.
MEASURED_ABSENT_PATHS = (
    "metrics/by_metric/chembl29_metal_scan.json",
    "metrics/by_metric/wfa155_ligand_connected_rate.json",
    "metrics/by_metric/wfa155_metal_coordination_sphere.json",
    "metrics/by_metric/wfa155_posebusters_validity.json",
    "molmetal/reports/wf67_p4_2_closure_r20/reachability_overall.json",
)

#: The PRE-FIX default-mode rendering, PRESERVED VERBATIM (AGENTS.md 5.5
#: rule #6).  The CLI printed this numeric count for a quantity it never
#: computed.
PRE_FIX_DEFAULT_RENDERING = (
    "ledger provenance: rows=432 metrics=399 distinct_source_paths=401 "
    "OPEN_never_committed=2 superseded_never_committed=0"
)

_SUPERSEDED_RE = re.compile(r"superseded_never_committed=(\S+)")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ledger_rows_and_shas(path: Path) -> List[Dict[str, Any]]:
    """Every row of a ledger, with the sha256 of its raw line."""
    out: List[Dict[str, Any]] = []
    with path.open("rb") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            out.append(
                {
                    "line_sha256": _sha256_bytes(raw),
                    "row": json.loads(stripped.decode("utf-8")),
                }
            )
    return out


def _content_sha256_at(commit: str, path: str) -> Optional[str]:
    """sha256 of a file's content at a commit -- INDEPENDENT of the auditor.

    Deliberately NOT the auditor's own helper: the bite tests must not use
    the code under test to build their own fixtures.
    """
    proc = subprocess.run(
        ["git", "cat-file", "-p", f"{commit}:{path}"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return _sha256_bytes(proc.stdout)


def _copy_ledger(tmp_path: Path) -> Path:
    dest = tmp_path / "run_ledger.jsonl"
    shutil.copy2(LEDGER, dest)
    return dest


def _rewrite_row(ledger_copy: Path, index: int, **fields: Any) -> None:
    """Rewrite ONE row of a ledger COPY, in place, keeping the order."""
    lines = ledger_copy.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[index])
    row.update(fields)
    lines[index] = json.dumps(row, sort_keys=True, ensure_ascii=False)
    ledger_copy.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _numeric_superseded(stdout: str) -> Optional[int]:
    """The numeric superseded count in a CLI output, or None if not measured.

    This is the guard under test: it must return None for a rendering that
    prints a TOKEN and an int for one that prints a NUMBER.
    """
    m = _SUPERSEDED_RE.search(stdout)
    if not m:
        return None
    token = m.group(1)
    return int(token) if token.isdigit() else None


# ---------------------------------------------------------------------------
# (1) the auditor MEASURES the pin inconsistency
# ---------------------------------------------------------------------------
def test_auditor_pin_counts_are_internally_consistent():
    report = wfa198.audit()
    c = report["counts"]
    assert c["ledger_corrupt_lines"] == 0
    assert c["rows_with_a_complete_pin"] == c["ledger_rows"], (
        "every ledger row carries both a git_sha and a source_json_sha256; "
        "if this changes, the AUDIT's denominator must change too"
    )
    assert (
        c["pin_consistent"] + c["pin_inconsistent"] == c["rows_with_a_complete_pin"]
    )
    assert (
        c["sha_mismatch_exists_at_commit"] + c["absent_at_commit"]
        == c["pin_inconsistent"]
    )
    assert c["pin_inconsistent"] > 0, (
        "the defect class must never silently vanish: if this is 0, the "
        "ledger changed and the report must be re-measured"
    )


def test_auditor_measures_the_dated_pin_snapshot():
    """The DATED MEASURED snapshot (2026-11-02).

    If this fails, the ledger changed (it is append-only) -- re-measure and
    re-publish the report; do NOT relax the assertion.
    """
    c = wfa198.audit()["counts"]
    assert c["ledger_rows"] == MEASURED_LEDGER_ROWS, (
            "ledger row count moved off the {d} measured on {m}: now {n}".format(
                d=MEASURED_LEDGER_ROWS, m=MEASURED_ON, n=c["ledger_rows"]
            )
        )
    assert c["pin_consistent"] == MEASURED_PIN_CONSISTENT
    assert c["pin_inconsistent"] == MEASURED_PIN_INCONSISTENT
    assert c["sha_mismatch_exists_at_commit"] == MEASURED_SHA_MISMATCH
    assert c["absent_at_commit"] == MEASURED_ABSENT_AT_COMMIT
    assert (
        c["gate_says_reproducible__sha_mismatch"]
        == MEASURED_GATE_REPRODUCIBLE_OF_SHA_MISMATCH
    ), (
        "the number of sha-mismatched rows the gate classifies as "
        "reproducible moved off the {n} measured on {d}: now {m}".format(
            n=MEASURED_GATE_REPRODUCIBLE_OF_SHA_MISMATCH,
            d=MEASURED_ON,
            m=c["gate_says_reproducible__sha_mismatch"],
        )
    )


def test_auditor_absent_at_commit_paths_are_the_measured_five():
    report = wfa198.audit()
    absent = sorted(
        v["source_json"] for v in report["offending_rows"] if not v["exists_at_commit"]
    )
    assert absent == sorted(MEASURED_ABSENT_PATHS), (
        "the set of paths absent at their pinned commit moved off the {d} "
        "measured on {m}".format(d=len(MEASURED_ABSENT_PATHS), m=MEASURED_ON)
    )


def test_auditor_gate_classification_matches_the_real_gate():
    """The auditor's replicated verdict must equal the REAL gate's verdict.

    ``run_ledger.uncommitted_provenance`` reports a verdict for the metric's
    latest row and for its superseded rows.  Where it reports one, the
    auditor must agree on that row -- otherwise the "the gate cannot see
    them" comparison is meaningless.
    """
    from molmetal.scripts.run_ledger import uncommitted_provenance

    report = wfa198.audit()
    by_key = {
        (v["metric_id"], v["source_json_sha256"]): v for v in report["row_verdicts"]
    }
    gate = uncommitted_provenance(include_superseded=True)
    checked = 0
    for d in gate.get("detail") or []:
        rows = [d["latest_row"]] + list(d.get("superseded_rows") or [])
        for row in rows:
            # the gate's row dicts carry no metric_id -- it lives on the
            # enclosing detail entry.
            key = (d.get("metric_id"), row.get("source_json_sha256"))
            mine = by_key.get(key)
            assert mine is not None, (
                "the gate reported a row the auditor does not know about: "
                "{k}".format(k=key)
            )
            assert mine["gate_classification"] == row["verdict"], (
                "verdict mismatch for {k}: auditor says {a}, the gate says "
                "{g}".format(
                    k=key, a=mine["gate_classification"], g=row["verdict"]
                )
            )
            checked += 1
    assert checked > 0, "the gate reported nothing -- the comparison is vacuous"


# ---------------------------------------------------------------------------
# (2) the auditor is READ-ONLY
# ---------------------------------------------------------------------------
def test_auditor_is_read_only_on_the_real_ledger():
    before_bytes = LEDGER.read_bytes()
    before_stat = LEDGER.stat()
    before_rows = _ledger_rows_and_shas(LEDGER)

    wfa198.audit()

    after_bytes = LEDGER.read_bytes()
    after_stat = LEDGER.stat()
    after_rows = _ledger_rows_and_shas(LEDGER)

    assert after_bytes == before_bytes, (
        "the auditor MUTATED the append-only ledger -- its sha256 changed"
    )
    assert len(after_rows) == len(before_rows)
    for b, a in zip(before_rows, after_rows):
        assert b["line_sha256"] == a["line_sha256"], (
            "the auditor MUTATED a ledger row: {m}".format(m=a["row"].get("metric_id"))
        )
    assert after_stat.st_mtime == before_stat.st_mtime, (
        "the ledger's mtime moved -- something wrote to it"
    )


def test_auditor_source_contains_no_write_operations():
    """STATIC check: the auditor's own source opens nothing for writing."""
    src = (REPO_ROOT / "molmetal" / "scripts" / "wfa198_ledger_pin_consistency.py")
    text = src.read_text(encoding="utf-8")
    for banned in (
        'open("w"',
        "open('w'",
        '"a"',
        "'a'",
        ".write_text(",
        ".write_bytes(",
        ".unlink(",
        ".mkdir(",
        "shutil.copy",
        "os.remove(",
        "os.rename(",
    ):
        assert banned not in text, (
            "the auditor contains a write operation {b!r} -- it must be "
            "read-only by construction".format(b=banned)
        )


def test_auditor_writes_nothing_under_metrics():
    """No file under ``metrics/by_metric/`` may change while it runs."""
    root = REPO_ROOT / "metrics" / "by_metric"
    before = {
        p.name: (p.stat().st_size, _sha256_bytes(p.read_bytes()))
        for p in sorted(root.glob("*.json"))
    }
    wfa198.audit()
    after = {
        p.name: (p.stat().st_size, _sha256_bytes(p.read_bytes()))
        for p in sorted(root.glob("*.json"))
    }
    assert before == after, "the auditor changed a published metric file"


# ---------------------------------------------------------------------------
# (3) THE TEST MUST BITE -- a deliberately inconsistent pair
# ---------------------------------------------------------------------------
def test_auditor_bites_on_a_deliberately_inconsistent_pair(tmp_path):
    """Re-point a CONSISTENT row's git_sha at a commit whose content differs.

    The sha half stays right and the commit half stops pinning it -- exactly
    the defect class.  The auditor MUST flag that row, and MUST NOT flag the
    untouched copy.
    """
    base = wfa198.audit()
    consistent = [v for v in base["row_verdicts"] if v["pin_consistent"]]
    assert consistent, "fixture: the real ledger has no consistent row"

    candidates = sorted({v["git_sha"] for v in base["row_verdicts"]})
    victim = None
    for v in consistent[:50]:
        for c in candidates:
            if c == v["git_sha"]:
                continue
            other = _content_sha256_at(c, v["repo_relative"])
            if other is not None and other != v["source_json_sha256"]:
                victim = (v, c, other)
                break
        if victim:
            break
    assert victim is not None, (
        "fixture: no consistent row could be re-pointed at a commit whose "
        "content differs"
    )
    row, other_commit, other_sha = victim

    broken = _copy_ledger(tmp_path)
    _rewrite_row(broken, row["index"], git_sha=other_commit)

    # the doctored row must now be flagged ...
    rep = wfa198.audit(broken, repo_root=REPO_ROOT)
    flagged = {v["index"]: v for v in rep["offending_rows"]}
    assert row["index"] in flagged, (
        "the auditor did NOT flag the re-pointed row {i} ({m}) -- the "
        "checker does not discriminate".format(i=row["index"], m=row["metric_id"])
    )
    v = flagged[row["index"]]
    assert v["pin_consistent"] is False
    assert v["exists_at_commit"] is True
    assert v["sha256_at_commit"] == other_sha
    assert v["sha256_at_commit"] != v["source_json_sha256"]

    # ... and the UNTOUCHED copy must flag exactly ONE row more than the
    # real ledger (the doctored one) -- a checker that flags everything is
    # just as broken as one that flags nothing.
    untouched_dir = tmp_path / "untouched"
    untouched_dir.mkdir(parents=True, exist_ok=True)
    untouched = wfa198.audit(_copy_ledger(untouched_dir), repo_root=REPO_ROOT)
    assert len(untouched["offending_rows"]) == len(base["offending_rows"]), (
        "the auditor's verdict on the UNTOUCHED ledger differs from its "
        "verdict on the real ledger"
    )


def test_auditor_bites_on_a_path_absent_at_the_pinned_commit(tmp_path):
    """Re-point a row at a commit where the path does not exist AT ALL."""
    base = wfa198.audit()
    consistent = [v for v in base["row_verdicts"] if v["pin_consistent"]]
    proc = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    root_commit = proc.stdout.strip()
    assert root_commit, "fixture: no root commit"

    victim = None
    for v in consistent:
        if _content_sha256_at(root_commit, v["repo_relative"]) is None:
            victim = v
            break
    assert victim is not None, (
        "fixture: no consistent row whose path is absent at the root commit"
    )

    broken = _copy_ledger(tmp_path)
    _rewrite_row(broken, victim["index"], git_sha=root_commit)

    rep = wfa198.audit(broken, repo_root=REPO_ROOT)
    flagged = {v["index"]: v for v in rep["offending_rows"]}
    assert victim["index"] in flagged
    v = flagged[victim["index"]]
    assert v["exists_at_commit"] is False
    assert v["sha256_at_commit"] is None


# ---------------------------------------------------------------------------
# (4) the CLI reporting fix
# ---------------------------------------------------------------------------
def test_cli_default_mode_does_not_print_a_numeric_superseded_count(capsys):
    rc = rl.main(["provenance"])
    out = capsys.readouterr()
    assert "superseded_never_committed=NOT_MEASURED" in out.out, (
        "the default mode must print the token NOT_MEASURED for a quantity "
        "it did not measure; got: {o}".format(o=out.out.splitlines()[:1])
    )
    assert _numeric_superseded(out.out) is None, (
        "the default mode still prints a NUMBER for an unmeasured quantity"
    )
    assert rc in (0, 1), "the provenance command returns 0 (PASS) or 1 (WARN)"
    assert "NOT MEASURED" in out.err, (
        "the default mode must say on stderr that it did not measure the "
        "superseded count; stderr was: {e!r}".format(e=out.err)
    )


def test_cli_all_mode_prints_the_measured_superseded_count(capsys):
    from molmetal.scripts.run_ledger import uncommitted_provenance

    rl.main(["provenance", "--all"])
    out = capsys.readouterr()
    n = _numeric_superseded(out.out)
    assert n is not None, (
        "--all DID measure the superseded count, so it must print a number; "
        "got: {o}".format(o=out.out.splitlines()[:1])
    )
    truth = uncommitted_provenance(include_superseded=True)["counts"][
        "superseded_uncommitted_rows"
    ]
    assert n == truth, (
        "the CLI printed {n} but the function measured {t}".format(n=n, t=truth)
    )
    assert n == MEASURED_SUPERSEDED_COUNT_ALL_MODE, (
        "the superseded count moved off the {n} measured on {d}: now {m}".format(
            n=MEASURED_SUPERSEDED_COUNT_ALL_MODE, d=MEASURED_ON, m=n
        )
    )


def test_cli_guard_bites_on_the_preserved_pre_fix_rendering(capsys):
    """The guard must REJECT the old numeric rendering.

    This is the bite proof for the CLI half: the preserved pre-fix line is
    a real rendering the guard has to reject.
    """
    assert _numeric_superseded(PRE_FIX_DEFAULT_RENDERING) == 0, (
        "fixture: the preserved pre-fix rendering must parse as a number"
    )
    rl.main(["provenance"])
    out = capsys.readouterr()
    assert _numeric_superseded(out.out) is None, (
        "the post-fix default rendering must NOT parse as a number"
    )


def test_preflight_gate_still_reports_the_superseded_count():
    """The CLI fix must NOT change the gate: it calls the FUNCTION."""
    from molmetal.scripts.pre_flight_check import gate_run_ledger_provenance

    r = gate_run_ledger_provenance("p3_run_ledger_provenance", "P3", "run_ledger")
    assert "superseded_never_committed={n}".format(
        n=MEASURED_SUPERSEDED_COUNT_ALL_MODE
    ) in r.evidence, (
        "the gate must still report the measured superseded count; got: "
        "{e}".format(e=r.evidence)
    )


# ---------------------------------------------------------------------------
# (5) SCOPE: this wave added NO gate and changed NO count pin
# ---------------------------------------------------------------------------
def test_this_wave_added_no_preflight_gate():
    from molmetal.scripts.pre_flight_check import build_gates

    for pipeline in ("all", "launch"):
        names = [g[0] for g in build_gates(pipeline)]
        assert not [n for n in names if "wfa198" in n or "pin_consist" in n], (
            "Wf-A198 must NOT register a pre-flight gate (registering a gate "
            "is a USER decision)"
        )


def test_expected_gate_counts_are_self_consistent_and_unchanged():
    from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS, build_gates

    # If this wave had added a gate without updating the pin, these two
    # would disagree.  The PIN VALUES are deliberately not asserted:
    # another wave may legitimately add a gate, and a test that fails on
    # that is a test people stop reading.
    assert len(build_gates("all")) == EXPECTED_GATE_COUNTS["all"]
    assert len(build_gates("launch")) == EXPECTED_GATE_COUNTS["launch"]


def test_report_describes_the_ledger_on_disk():
    """The published report must describe the ledger that is on disk NOW.

    The ledger is append-only, so any append invalidates the report.  The
    failure message says so -- the correct response is to re-measure and
    re-publish the report, not to relax the assertion.
    """
    report_json = REPO_ROOT / "molmetal" / "reports" / "wfa198_ledger_pin_consistency" / "report.json"
    if not report_json.is_file():
        pytest.skip("the Wf-A198 report has not been published")
    published = json.loads(report_json.read_text(encoding="utf-8"))
    digest = _sha256_bytes(LEDGER.read_bytes())
    assert published.get("ledger_sha256") == digest, (
        "the published report describes ledger sha256 {p} but the ledger on "
        "disk is {d} -- the ledger is append-only, so the report must be "
        "re-measured and re-published".format(p=published.get("ledger_sha256"), d=digest)
    )
