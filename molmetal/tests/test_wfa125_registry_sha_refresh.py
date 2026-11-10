# SPDX-License-Identifier: Apache-2.0
"""Wf-A125 — the registry sha refresh (G6) + the two artifact repairs (G7, G8).

G6 — THE SHA RE-POINT COMMAND
------------------------------
``registry/metrics.yaml`` is the SSOT and its ``source_json_sha256`` is the
currency of ``registry.py check``.  Until Wf-A125 there was NO way to
re-point ONE row's sha: the only writer was ``registry.py generate``, which
rewrites the whole file (a yaml round-trip churns the narrative fields and
re-wraps 578 rows).  ``registry.py repoint <metric_id> [--apply]`` is the
surgical alternative: it edits ONE ``source_json_sha256:`` line (and the
matching ``last_refreshed:`` line) and leaves every other byte alone.

G7 — THE wf32_pb_valid ARTIFACT
--------------------------------
The artifact measured 1 pocket x 1 candidate x 2 pose modes = 2 evaluations
and BOTH were SKIPPED ("posebusters not installed"), so PoseBusters validity
was NOT MEASURED.  The file also carried no ``spec_target``, so a reader
could not see that the DECLARED sweep was 10 pockets x 5 candidates x 2 modes
= 100 cells and was never run.

G8 — THE CFM CENSUS FRAGMENTATION SPLIT
---------------------------------------
A ``None`` fragmentation ratio has TWO causes and the census conflated them
into one reason string that was FALSE for one of them.  ``field_absent`` (the
column was never written) and ``denominator_empty`` (the column was written
but the generation eval drew nothing, so the denominator is empty) are now
distinguished.

CPU-ONLY.  Nothing here trains, samples, or touches CUDA.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "registry" / "metrics.yaml"
CENSUS = REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"
WF32 = REPO_ROOT / "metrics" / "by_metric" / "wf32_pb_valid.json"
M8_SUMMARY = REPO_ROOT / "molmetal" / "reports" / "wf32_m8_summary.json"

REPOINTED_BY_THIS_WAVE = ("wf32_pb_valid", "cfm_run_census")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry_rows() -> dict:
    return (yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {})["metrics"]


def _synthetic_registry(tmp_path: Path, sha: str) -> tuple:
    """A one-row registry + its source JSON under a tmp repo root."""
    root = tmp_path / "repo"
    src = root / "metrics" / "by_metric" / "m_synthetic.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text('{"value": 0.5}\n', encoding="utf-8")
    reg = tmp_path / "metrics.yaml"
    reg.write_text(
        "metrics:\n"
        "  m_synthetic:\n"
        "    value: 0.5\n"
        "    source_json: metrics/by_metric/m_synthetic.json\n"
        f"    source_json_sha256: {sha}\n"
        "    last_refreshed: '2026-01-01T00:00:00Z'\n",
        encoding="utf-8",
    )
    return reg, root, src


# ---------------------------------------------------------------------------
# G6 — the surgical sha re-point
# ---------------------------------------------------------------------------
def test_repoint_is_a_dry_run_by_default(tmp_path) -> None:
    """(G6-a) Nothing is written unless ``--apply`` is passed."""
    from molmetal.scripts.registry import repoint

    reg, root, src = _synthetic_registry(tmp_path, "ab" * 32)
    before = reg.read_bytes()
    report = repoint(
        "m_synthetic", apply=False, metrics_path=reg, repo_root=root
    )
    assert report["changed"] is True, report
    assert report["applied"] is False, report
    assert reg.read_bytes() == before, "the dry run wrote to disk"
    assert report["new_sha256"] == _sha256(src), report


def test_repoint_changes_only_the_sha_line(tmp_path) -> None:
    """(G6-b) The edit is surgical: every other byte is IDENTICAL."""
    from molmetal.scripts.registry import repoint

    reg, root, src = _synthetic_registry(tmp_path, "ab" * 32)
    before_lines = reg.read_text(encoding="utf-8").splitlines(keepends=True)
    report = repoint("m_synthetic", apply=True, metrics_path=reg, repo_root=root)
    assert report["applied"] is True, report
    after_lines = reg.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(before_lines) == len(after_lines)
    changed = [
        (i, b, a)
        for i, (b, a) in enumerate(zip(before_lines, after_lines))
        if b != a
    ]
    # exactly two lines change: the sha and its last_refreshed stamp
    assert len(changed) == 2, changed
    assert "source_json_sha256:" in changed[0][1], changed
    assert changed[0][2].strip() == f"source_json_sha256: {_sha256(src)}", changed
    assert "last_refreshed:" in changed[1][1], changed
    # the row still parses and the sha is now the disk sha
    doc = yaml.safe_load(reg.read_text(encoding="utf-8"))
    row = doc["metrics"]["m_synthetic"]
    assert row["source_json_sha256"] == _sha256(src)
    assert row["value"] == 0.5


def test_repoint_does_not_touch_an_already_current_row(tmp_path) -> None:
    """(G6-c) A needless rewrite is exactly the churn G6 exists to remove."""
    from molmetal.scripts.registry import repoint

    reg, root, src = _synthetic_registry(tmp_path, "ab" * 32)
    repoint("m_synthetic", apply=True, metrics_path=reg, repo_root=root)
    settled = reg.read_bytes()
    report = repoint("m_synthetic", apply=True, metrics_path=reg, repo_root=root)
    assert report["changed"] is False, report
    assert report["applied"] is False, report
    assert reg.read_bytes() == settled, "a current row was rewritten anyway"


def test_repoint_refuses_an_unknown_metric(tmp_path) -> None:
    """(G6-d) A typo must be an error, not a silent no-op."""
    from molmetal.scripts.registry import repoint

    reg, root, _src = _synthetic_registry(tmp_path, "ab" * 32)
    with pytest.raises(KeyError):
        repoint("m_no_such_metric", apply=True, metrics_path=reg, repo_root=root)


def test_repoint_refuses_a_missing_source_file(tmp_path) -> None:
    """(G6-d) It must not invent a sha for a file that is not there."""
    from molmetal.scripts.registry import repoint

    reg, root, src = _synthetic_registry(tmp_path, "ab" * 32)
    src.unlink()
    with pytest.raises(ValueError):
        repoint("m_synthetic", apply=True, metrics_path=reg, repo_root=root)


def test_repoint_refuses_a_row_with_two_sha_lines(tmp_path) -> None:
    """(G6-d) An ambiguous row is refused, not guessed at."""
    from molmetal.scripts.registry import repoint

    reg, root, _src = _synthetic_registry(tmp_path, "ab" * 32)
    reg.write_text(
        reg.read_text(encoding="utf-8")
        + f"    source_json_sha256: {'cd' * 32}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        repoint("m_synthetic", apply=True, metrics_path=reg, repo_root=root)


def test_the_real_registry_is_untouched_by_a_dry_run() -> None:
    """(G6-e) A dry run against the REAL registry must not write."""
    from molmetal.scripts.registry import repoint

    before = REGISTRY.read_bytes()
    report = repoint("wf32_pb_valid", apply=False)
    assert report["dry_run"] is True, report
    assert REGISTRY.read_bytes() == before, "the dry run wrote to the real registry"
    assert report["changed"] is False, (
        "the wf32_pb_valid row is already re-pointed by this wave, so a "
        "dry run must report 'already current'"
    )


# ---------------------------------------------------------------------------
# G6 — the re-pointed rows are CURRENT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("metric_id", REPOINTED_BY_THIS_WAVE)
def test_the_repointed_row_matches_the_file_on_disk(metric_id) -> None:
    """(G6-f) The registry row's sha == sha256 of the artifact on disk."""
    rows = _registry_rows()
    assert metric_id in rows, metric_id
    src = REPO_ROOT / rows[metric_id]["source_json"]
    assert src.is_file(), src
    assert rows[metric_id]["source_json_sha256"] == _sha256(src), (
        metric_id,
        rows[metric_id]["source_json_sha256"],
        _sha256(src),
    )


def test_registry_check_still_passes() -> None:
    """(G6-g) The SSOT is still consistent after the re-point."""
    proc = subprocess.run(
        [sys.executable, "molmetal/scripts/registry.py", "check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "REGISTRY: PASS" in proc.stdout, proc.stdout
    assert "0 stale" in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# G7 — the wf32_pb_valid artifact
# ---------------------------------------------------------------------------
def test_the_wf32_artifact_declares_the_spec_target_verbatim() -> None:
    """(G7-a) ``spec_target`` is copied from the declaration of record.

    It is NOT re-derived from the artifact: re-deriving it would re-create
    the very drift this gap is about.
    """
    artifact = json.loads(WF32.read_text(encoding="utf-8"))
    summary = json.loads(M8_SUMMARY.read_text(encoding="utf-8"))
    assert "spec_target" in artifact, sorted(artifact)
    assert artifact["spec_target"] == summary["spec_target"], (
        artifact["spec_target"],
        summary["spec_target"],
    )
    assert artifact["spec_target"]["n_cells"] == 100, artifact["spec_target"]


def test_the_wf32_artifact_does_not_claim_a_measured_validity() -> None:
    """(G7-b) The honest framing must say the validity was NOT MEASURED."""
    artifact = json.loads(WF32.read_text(encoding="utf-8"))
    framing = artifact["wf432a1_pb_validity_deferred_block"]["honest_framing"]
    assert "NOT MEASURED" in framing, framing
    assert "NEVER RUN" in framing, framing
    assert "do not cite this artifact as a PoseBusters validity result" in framing
    # The measured scope is 1 pocket x 1 candidate x 2 modes = 2 evaluations
    assert artifact["n_pockets"] == 1, artifact["n_pockets"]
    assert artifact["n_candidates"] == 1, artifact["n_candidates"]
    assert artifact["n_evaluations"] == 2, artifact["n_evaluations"]
    assert all(
        ev["status"] == "skipped" for ev in artifact["per_evaluation"]
    ), artifact["per_evaluation"]
    assert all(ev["pb_valid"] is None for ev in artifact["per_evaluation"])


def test_the_wf32_artifact_makes_no_multi_pocket_pass_claim() -> None:
    """(G7-c) No part of the artifact may read as "N pockets PASS"."""
    artifact = json.loads(WF32.read_text(encoding="utf-8"))
    framing = artifact["wf432a1_pb_validity_deferred_block"]["honest_framing"]
    # Everything EXCEPT the honest framing must be free of a pass claim.
    # (The framing itself NAMES the SOTA claim in order to disclaim it.)
    elsewhere = json.dumps(
        {
            k: v
            for k, v in artifact.items()
            if k != "wf432a1_pb_validity_deferred_block"
        }
    ).lower()
    for forbidden in ("pockets pass", "sota", "passes the gate"):
        assert forbidden not in elsewhere, forbidden
    # ... and the framing disclaims exactly those claims.
    assert "Any SOTA-gate claim about a multi-pocket set" in framing, framing
    assert "is NOT reproduced here" in framing, framing
    # every recorded verdict is a SKIP with no measured validity
    for ev in artifact["per_evaluation"]:
        assert ev["status"] == "skipped", ev
        assert ev["pb_valid"] is None, ev
        assert ev["n_checks"] == 0, ev
        assert ev["n_passed"] == 0, ev
        assert ev["pass_rate"] == 0.0, ev
    # the only pass_rate figures are the skipped 0.0 placeholders
    assert artifact["overall_pb_pass_rate"]["mol"] == 0.0
    assert artifact["overall_pb_pass_rate"]["dock"] == 0.0
    assert artifact["n_pb_checks_total"] == 26


# ---------------------------------------------------------------------------
# G8 — the two distinct SKIP causes
# ---------------------------------------------------------------------------
def test_the_two_skip_causes_are_distinguished() -> None:
    """(G8-a) ``field_absent`` != ``denominator_empty``."""
    from molmetal.scripts.wf_cfm_run_census import _fragmentation_verdict

    absent = _fragmentation_verdict(
        None, None, gen_validity=None, field_present=False
    )
    empty = _fragmentation_verdict(None, None, gen_validity=None, field_present=True)
    assert absent["status"] == "SKIP", absent
    assert empty["status"] == "SKIP", empty
    assert absent["basis"] == "field_absent", absent
    assert empty["basis"] == "denominator_empty", empty
    assert absent["basis"] != empty["basis"]
    assert absent["reason"] != empty["reason"]


def test_the_published_census_carries_both_skip_causes() -> None:
    """(G8-b) The census carries BOTH reasons, with the measured split."""
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    bases = {}
    for run in census["runs"]:
        gate = run.get("fragmentation_gate") or {}
        bases.setdefault(gate.get("basis"), 0)
        bases[gate["basis"]] += 1
    assert bases.get("field_absent", 0) > 0, bases
    assert bases.get("denominator_empty", 0) > 0, bases
    # the SKIP count is the sum of the two causes
    n_skip = census["fragmentation_summary"]["n_skip"]
    assert n_skip == bases.get("field_absent", 0) + bases.get(
        "denominator_empty", 0
    ), (n_skip, bases)


def test_the_census_changelog_records_the_split() -> None:
    """(G8-c) The meaning change is documented IN the artifact."""
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    entry = census["changelog"]["wf_a125"]
    joined = " ".join(entry)
    assert "field_absent" in joined, joined
    assert "denominator_empty" in joined, joined
    assert "11 field_absent and 16 denominator_empty" in joined, joined
    # the rule that decides the split is stated
    assert "census's OWN normalised per-epoch row" in joined, joined
