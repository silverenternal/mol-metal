# SPDX-License-Identifier: Apache-2.0
"""Wf-A125 — the pre-flight WARN contract (G5, G11).

WHAT WAS BROKEN
---------------
**G5 — a dead gate.**  ``molmetal/scripts/run_ledger.py::uncommitted_provenance``
existed with exactly two callers (its own CLI and one test) and **NO
pre-flight caller**.  So the defect it detects — "a ledger row names a file
version this repository has never held, so no commit can reproduce it" —
was invisible to every launch.  Wf-A125 registers it as
``p3_run_ledger_provenance``.

**G11a — WARN was invisible.**  ``main()`` counted ``pass``/``fail``/``skip``
but not ``warn``, so ``pass + fail + skip`` was LESS than ``total``: an
internally inconsistent summary that nothing checked.  The roll-up now counts
``warn`` and the identity ``pass + fail + skip + warn == total`` holds.

**G11b — a gate that claimed to enforce but could not.**  ``
gate_run_ledger_staleness`` used to return ``WARN`` for a stale ledger while
its own docstring said it "emits SKIP unless WF_RUN_LEDGER=1".  Neither was
true: the gate has enforced by DEFAULT since 2026-10-08, ``WF_RUN_LEDGER=0``
is a DOWNGRADE to SKIP, and ``WARN`` could not fail a launch.  It now returns
``FAIL`` for a stale ledger.

**G11d — unbounded evidence.**  The gate enumerated every non-current metric
id inline: MEASURED 6400 characters on one line (182 ``no_run`` metrics).  The
evidence is now capped while still naming the stale ids — the only actionable
ones — in full.

CPU-ONLY.  No gate invoked here touches CUDA.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molmetal.scripts.pre_flight_check import (
    EXPECTED_GATE_COUNTS,
    RUN_LEDGER_ENFORCE_ENV,
    build_gates,
    gate_run_ledger_provenance,
    gate_run_ledger_staleness,
    main,
)

PROVENANCE_GATE = "p3_run_ledger_provenance"
STALENESS_GATE = "p3_run_ledger_staleness"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ledger_rows(rows):
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"


def _row(metric_id, sha, value, source_json=None):
    return {
        "metric_id": metric_id,
        "timestamp": "2026-10-31T00:00:00+00:00",
        "value": value,
        "source_json": source_json or f"metrics/by_metric/{metric_id}.json",
        "source_json_sha256": sha,
        "git_sha": "0" * 40,
        "data_sha": None,
        "config_sha": None,
    }


def _registry(path: Path, metrics):
    lines = ["metrics:"]
    for mid, (sha, value) in metrics.items():
        lines.append(f"  {mid}:")
        lines.append(f"    value: {value}")
        lines.append(f"    source_json: metrics/by_metric/{mid}.json")
        lines.append(f"    source_json_sha256: {sha}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _ledger_enforcement_on(monkeypatch):
    """Never let an ambient env var downgrade the gate under test."""
    monkeypatch.delenv(RUN_LEDGER_ENFORCE_ENV, raising=False)


# ---------------------------------------------------------------------------
# G5 — the provenance gate is REGISTERED
# ---------------------------------------------------------------------------
def test_the_provenance_gate_is_registered_in_every_pipeline() -> None:
    """(G5-a) A gate with no pre-flight caller was a dead gate.

    The plan puts the gate in the shared ``if pipeline == "all":`` block, so
    it is present in ``"all"`` AND in ``"launch"`` (which normalises to
    ``"all"``).  It is deliberately NOT in the narrow ``frontier`` / ``cfm``
    / ``chemprop_v2_v3`` pipelines — those are subsets of "all", and a
    ledger-provenance check is a launch-level invariant, not a frontier one.
    """
    for pipeline in ("all", "launch"):
        names = {name for name, _ in build_gates(pipeline)}
        assert PROVENANCE_GATE in names, (pipeline, sorted(names))
        assert STALENESS_GATE in names, (pipeline, sorted(names))


def test_the_provenance_gate_sits_beside_the_staleness_gate() -> None:
    """(G5-a) The two ledger gates must have the SAME pipeline membership."""
    membership = {}
    for pipeline in ("all", "launch", "frontier", "cfm", "chemprop_v2_v3"):
        names = {name for name, _ in build_gates(pipeline)}
        membership[pipeline] = (
            PROVENANCE_GATE in names,
            STALENESS_GATE in names,
        )
    assert set(membership.values()) == {(True, True), (False, False)}, membership


def test_the_gate_counts_are_updated_for_the_new_gate() -> None:
    """(G5-a) Registering a gate must move the expected totals with it."""
    all_names = {name for name, _ in build_gates("all")}
    launch_names = {name for name, _ in build_gates("launch")}
    assert EXPECTED_GATE_COUNTS["all"] == len(all_names), (
        EXPECTED_GATE_COUNTS["all"],
        len(all_names),
    )
    assert EXPECTED_GATE_COUNTS["launch"] == len(launch_names), (
        EXPECTED_GATE_COUNTS["launch"],
        len(launch_names),
    )


# ---------------------------------------------------------------------------
# G5 — the provenance gate DETECTS the defect
# ---------------------------------------------------------------------------
def test_the_provenance_gate_fails_on_a_never_committed_row(tmp_path) -> None:
    """(G5-b) A row naming a version the repo has never held is a FAIL."""
    repo_root = tmp_path / "repo"
    (repo_root / "metrics" / "by_metric").mkdir(parents=True)
    src = repo_root / "metrics" / "by_metric" / "m_synthetic.json"
    src.write_text('{"value": 0.5}\n', encoding="utf-8")

    ledger = tmp_path / "ledger.jsonl"
    # The sha names a version of the file that exists NOWHERE: not on disk
    # (the disk file hashes differently) and not in any commit (the tmp
    # repo is not a git repository).
    ledger.write_text(
        _ledger_rows([_row("m_synthetic", "de" * 32, 0.5)]), encoding="utf-8"
    )

    result = gate_run_ledger_provenance(
        PROVENANCE_GATE, "P3", "run_ledger", ledger_path=ledger, repo_root=repo_root
    )
    assert result.status == "FAIL", (result.status, result.evidence)
    assert "OPEN_never_committed=1" in result.evidence, result.evidence
    assert "m_synthetic" in result.evidence, result.evidence


def test_the_provenance_gate_passes_on_the_real_ledger() -> None:
    """(G5-c) The real ledger has no OPEN uncommitted row (MEASURED 2026-10-31).

    The 9 SUPERSEDED rows are reported but do NOT block: the ledger is
    append-only, so a superseded bad row stays in the history forever.
    """
    result = gate_run_ledger_provenance(PROVENANCE_GATE, "P3", "run_ledger")
    assert result.status == "PASS", (result.status, result.evidence)
    assert "OPEN_never_committed=0" in result.evidence, result.evidence


def test_the_provenance_gate_fails_on_a_broken_ledger(tmp_path) -> None:
    """A gate that cannot read its input is a FAIL, never a silent PASS."""
    broken = tmp_path / "broken.jsonl"
    broken.write_text("this is not json\n", encoding="utf-8")
    result = gate_run_ledger_provenance(
        PROVENANCE_GATE,
        "P3",
        "run_ledger",
        ledger_path=broken,
        repo_root=tmp_path,
    )
    assert result.status == "FAIL", (result.status, result.evidence)


# ---------------------------------------------------------------------------
# G11b — the staleness gate FAILs, and never WARNs
# ---------------------------------------------------------------------------
def test_the_staleness_gate_never_returns_warn(tmp_path) -> None:
    """(G11b) WARN could not fail a launch while the gate claimed to enforce."""
    registry = _registry(
        tmp_path / "metrics.yaml", {"m_current": ("aa" * 32, 0.5)}
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        _ledger_rows([_row("m_current", "aa" * 32, 0.5)]), encoding="utf-8"
    )
    result = gate_run_ledger_staleness(
        STALENESS_GATE, "P3", "run_ledger", ledger_path=ledger,
        registry_metrics_path=registry,
    )
    assert result.status == "PASS", (result.status, result.evidence)
    assert result.status != "WARN", result.status


def test_the_staleness_gate_fails_on_a_stale_row(tmp_path) -> None:
    """(G11b) A registry row the ledger does not back is a FAIL."""
    registry = _registry(tmp_path / "metrics.yaml", {"m_stale": ("aa" * 32, 0.9)})
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        _ledger_rows([_row("m_stale", "bb" * 32, 0.1)]), encoding="utf-8"
    )
    result = gate_run_ledger_staleness(
        STALENESS_GATE, "P3", "run_ledger", ledger_path=ledger,
        registry_metrics_path=registry,
    )
    assert result.status == "FAIL", (result.status, result.evidence)
    assert "stale=1" in result.evidence, result.evidence
    assert "m_stale" in result.evidence, result.evidence


def test_the_staleness_gate_passes_on_the_real_ledger() -> None:
    """(G11b) The real ledger has 0 stale rows (MEASURED 2026-10-31)."""
    result = gate_run_ledger_staleness(STALENESS_GATE, "P3", "run_ledger")
    assert result.status == "PASS", (result.status, result.evidence)
    assert "stale=0" in result.evidence, result.evidence


# ---------------------------------------------------------------------------
# G11d — bounded evidence
# ---------------------------------------------------------------------------
def test_the_staleness_evidence_is_bounded_but_names_the_stale_id(
    tmp_path,
) -> None:
    """(G11d) The evidence was 6400 chars on one line (182 no_run metrics)."""
    metrics = {}
    ledger_rows = []
    for i in range(200):
        mid = f"m_no_run_{i:03d}"
        metrics[mid] = ("cc" * 32, 0.1 * i)
    metrics["m_the_stale_one"] = ("dd" * 32, 0.77)
    ledger_rows.append(_row("m_the_stale_one", "ee" * 32, 0.42))
    registry = _registry(tmp_path / "metrics.yaml", metrics)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(_ledger_rows(ledger_rows), encoding="utf-8")

    result = gate_run_ledger_staleness(
        STALENESS_GATE, "P3", "run_ledger", ledger_path=ledger,
        registry_metrics_path=registry,
    )
    assert result.status == "FAIL", (result.status, result.evidence)
    assert len(result.evidence) < 2000, len(result.evidence)
    assert "m_the_stale_one" in result.evidence, result.evidence
    assert "... and 175 more" in result.evidence, result.evidence
    assert "(full list: run_ledger.py status)" in result.evidence, result.evidence


# ---------------------------------------------------------------------------
# G11a — the summary identity
# ---------------------------------------------------------------------------
def test_the_summary_counts_warn_and_the_identity_holds(tmp_path) -> None:
    """(G11a) ``pass + fail + skip + warn == total`` must hold."""
    out = tmp_path / "preflight.json"
    rc = main(
        [
            "--gates",
            f"{STALENESS_GATE},{PROVENANCE_GATE}",
            "--json",
            str(out),
        ]
    )
    assert rc == 0, rc
    report = json.loads(out.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert "warn" in summary, summary
    assert (
        summary["pass"] + summary["fail"] + summary["skip"] + summary["warn"]
        == summary["total"]
    ), summary
    assert summary["total"] == 2, summary
    assert summary["warn"] == 0, summary


def test_the_json_report_carries_the_warn_counter(tmp_path) -> None:
    """The report schema is what a downstream consumer reads."""
    out = tmp_path / "preflight.json"
    rc = main(["--gates", STALENESS_GATE, "--json", str(out)])
    assert rc == 0, rc
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["summary"]) >= {
        "pass",
        "fail",
        "skip",
        "warn",
        "total",
    }, sorted(report["summary"])
    for gate in report["gates"]:
        assert gate["status"] in ("PASS", "FAIL", "SKIP", "WARN"), gate
