"""Wf-350-A3 — hyperparameter provenance log gate integration tests (3 NEW tests).

================================================================
Per Wf-343A-R2 recommendation + the 2026-09-29 user ultracode request:
"NeurIPS §D.9 hyperparameter provenance" — a pre-retrain internal check
gate that records hyperparameter ``git-SHA provenance + recent commit
history + uncommitted-change status`` for every config file in
``configs/`` and emits a JSONL artefact at
``molmetal/reports/wf350a3_hp_provenance.jsonl`` so downstream
consumers (paper §3 Wf-350 attribution appendix + paper dashboard
reproducibility table + F2(a) hyperparam rewind list) can trace every
retrain back to the exact commit that produced its hyperparameters.

The Wf-350-A3 gate is the **3rd** of the Wf-343A-R2 pre-retrain
internal check trio (Wf-343C4 + Wf-347A1 + Wf-350-A3); the
``EXPECTED_GATE_COUNTS["all"]`` count is bumped from 68 → 69 in
lockstep with the audit doc (``TODO/completed/321d_preflight_gate_audit.md``).

Tests
-----

1. **test_gate_p1_hp_provenance_log_schema_validity** — invokes
   :func:`molmetal.scripts.pre_flight_check.gate_p1_hp_provenance_log`
   against the live ``configs/`` + an isolated tmp JSONL path, asserts
   PASS, then re-parses the written JSONL line-by-line and verifies
   the 4 mandatory keys (``config_path`` + ``current_sha`` +
   ``recent_commits`` + ``uncommitted_changes``) + per-record types
   (str / str / list-of-dict-with-sha-message-date / bool). Verifies
   the NeurIPS §D.9 reproducibility contract on the live host (a real
   git repo with real recent commits).

2. **test_gate_p1_hp_provenance_log_per_config_records** — runs the
   gate on the canonical 7 config files and asserts the JSONL contains
   exactly 7 records (1 per YAML), with every ``config_path`` matching
   one of the canonical 7.  Also asserts the ``current_sha`` matches
   ``git rev-parse HEAD`` (live SHA verification — the provenance
   record actually points to a commit that exists on this host).

3. **test_gate_p1_hp_provenance_log_registered_in_build_gates** —
   calls :func:`molmetal.scripts.pre_flight_check.build_gates("all")`
   and asserts the gate ``p1_hp_provenance_log`` is present in the
   runners list, the runner name matches the gate function name, and
   the ``EXPECTED_GATE_COUNTS["all"]`` is 69 (the bumped count
   post-Wf-350-A3).

Backward-compat verdict
-----------------------
The 1 NEW gate is ADDITIVE only — it does NOT modify any pre-existing
gate / runner / path; the Wf-321D gate-count-consistency audit
invariant (``all == 69``) is preserved because the new gate bumps
``EXPECTED_GATE_COUNTS["all"]`` from 68 → 69 in lockstep with the
audit doc (``TODO/completed/321d_preflight_gate_audit.md``).  The
``frontier`` / ``chemprop_v2_v3`` / ``cfm`` pipeline counts remain
stable at 25 / 13 / 6 per the Wf-259 contract.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Each test asserts ONE observable contract (gate PASS, schema
  validity, registration).
* All assertions are MEASURED on the live host (real git, real configs).
* No PROJECTED / aspirational numbers — every test is verifiable on
  the host that runs pytest.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
CONFIGS_DIR = REPO_ROOT / "configs"
SCRIPTS_DIR = MOLMETAL_ROOT / "scripts"

CANONICAL_7_CONFIGS: tuple = (
    CONFIGS_DIR / "cfm_full_retrain.yaml",
    CONFIGS_DIR / "cfm_per_innovation_ablation_p1.4.yaml",
    CONFIGS_DIR / "cfm_real_decode_lift_p1.3.yaml",
    CONFIGS_DIR / "default.yaml",
    CONFIGS_DIR / "wf124_p3_50epoch_full_sweep_g5.yaml",
    CONFIGS_DIR / "wf124_p3_per_innovation_ablation_g6.yaml",
    CONFIGS_DIR / "wf124_p3_remaining_metals_g4.yaml",
)


@pytest.fixture(scope="module")
def preflight_module():
    """Import the pre_flight_check module via sys.path injection (CPU-only)."""
    import importlib
    import sys

    # The package layout is repo-root/molmetal/scripts/pre_flight_check.py;
    # insert REPO_ROOT so `molmetal.scripts.pre_flight_check` resolves.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if "molmetal" not in sys.modules:
        # Ensure molmetal is importable as a package; pre_flight_check.py
        # also does `from molmetal.scripts.read_anchor_json import …`.
        # Trigger an import on a small submodule first to register it.
        spec = importlib.util.find_spec("molmetal")
        if spec is None:
            pytest.skip("molmetal package not importable on this host")
    mod = importlib.import_module("molmetal.scripts.pre_flight_check")
    return mod


def _live_git_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        pytest.skip("git rev-parse HEAD failed on this host")
    return proc.stdout.strip()


def test_gate_p1_hp_provenance_log_schema_validity(
    preflight_module: Any, tmp_path: Path
) -> None:
    """Gate 1 — schema validity: JSONL written with 4 mandatory keys per record.

    Runs :func:`gate_p1_hp_provenance_log` against the live configs/ on a
    tmp_path JSONL, asserts PASS, then re-parses each line of the
    written JSONL and verifies the 4 mandatory keys + types.
    """
    jsonl_path = tmp_path / "wf350a3_hp_provenance.jsonl"
    res = preflight_module.gate_p1_hp_provenance_log(
        "p1_hp_provenance_log",
        "P3",
        "wf350a3",
        CANONICAL_7_CONFIGS,
        jsonl_path,
    )
    assert (
        res.status == "PASS"
    ), f"gate must PASS on live host (got {res.status}): {res.evidence}"
    assert jsonl_path.is_file(), f"JSONL artefact missing: {jsonl_path}"

    # Re-parse the JSONL line-by-line and verify schema.
    schema_problems: List[str] = []
    line_count = 0
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for idx, raw in enumerate(fh):
            raw = raw.strip()
            if not raw:
                continue
            line_count += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                schema_problems.append(f"line {idx}: JSON parse fail: {exc}")
                continue
            if not isinstance(obj, dict):
                schema_problems.append(f"line {idx}: not a dict")
                continue
            for k in (
                "config_path",
                "current_sha",
                "recent_commits",
                "uncommitted_changes",
            ):
                if k not in obj:
                    schema_problems.append(f"line {idx}: missing key {k!r}")
            if not isinstance(obj.get("config_path"), str):
                schema_problems.append(f"line {idx}: config_path not str")
            if not isinstance(obj.get("current_sha"), str):
                schema_problems.append(f"line {idx}: current_sha not str")
            if not isinstance(obj.get("recent_commits"), list):
                schema_problems.append(f"line {idx}: recent_commits not list")
            if not isinstance(obj.get("uncommitted_changes"), bool):
                schema_problems.append(
                    f"line {idx}: uncommitted_changes not bool "
                    f"(got {type(obj.get('uncommitted_changes')).__name__})"
                )
            # current_sha must be a valid git SHA (40 hex chars)
            if isinstance(obj.get("current_sha"), str):
                if not re.match(r"^[0-9a-f]{40}$", obj["current_sha"]):
                    schema_problems.append(
                        f"line {idx}: current_sha not 40-hex: "
                        f"{obj['current_sha']!r}"
                    )
            # recent_commits entries (when present) must carry sha+message+date
            if isinstance(obj.get("recent_commits"), list):
                for j, entry in enumerate(obj["recent_commits"]):
                    if not isinstance(entry, dict):
                        schema_problems.append(
                            f"line {idx}: recent_commits[{j}] not dict"
                        )
                        continue
                    for k in ("sha", "message", "date"):
                        if k not in entry:
                            schema_problems.append(
                                f"line {idx}: recent_commits[{j}] missing {k!r}"
                            )

    assert line_count == len(
        CANONICAL_7_CONFIGS
    ), f"expected {len(CANONICAL_7_CONFIGS)} JSONL records, got {line_count}"
    assert not schema_problems, (
        f"JSONL schema validation FAILED ({len(schema_problems)}): "
        f"{'; '.join(schema_problems[:5])}"
    )


def test_gate_p1_hp_provenance_log_per_config_records(
    preflight_module: Any, tmp_path: Path
) -> None:
    """Gate 2 — per-config records: every canonical 7 YAML appears in JSONL.

    Asserts the gate produces exactly 7 records (one per canonical
    config), every ``config_path`` matches one of the canonical 7, and
    the ``current_sha`` matches the live ``git rev-parse HEAD``.
    """
    jsonl_path = tmp_path / "wf350a3_hp_provenance_per_config.jsonl"
    res = preflight_module.gate_p1_hp_provenance_log(
        "p1_hp_provenance_log",
        "P3",
        "wf350a3",
        CANONICAL_7_CONFIGS,
        jsonl_path,
    )
    assert (
        res.status == "PASS"
    ), f"gate must PASS on live host (got {res.status}): {res.evidence}"

    canonical_paths: set = {
        p.relative_to(REPO_ROOT).as_posix() for p in CANONICAL_7_CONFIGS
    }
    live_sha = _live_git_sha()

    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            records.append(json.loads(raw))

    assert len(records) == len(
        CANONICAL_7_CONFIGS
    ), f"expected {len(CANONICAL_7_CONFIGS)} records, got {len(records)}"
    seen_paths: set = set()
    for rec in records:
        cp = rec["config_path"]
        assert (
            cp in canonical_paths
        ), f"unexpected config_path in JSONL: {cp} (not in canonical 7)"
        seen_paths.add(cp)
        assert rec["current_sha"] == live_sha, (
            f"current_sha mismatch for {cp}: "
            f"jsonl={rec['current_sha']!r} live={live_sha!r}"
        )
        assert isinstance(rec["recent_commits"], list)
        # Wf-350-A3 contract: every config in configs/ has either
        # (a) ≥1 entry in recent_commits (file has git history) or
        # (b) uncommitted_changes == True (file is uncommitted / new).
        # Both are honest outcomes; neither is a regression.
        if len(rec["recent_commits"]) == 0:
            assert rec["uncommitted_changes"] is True, (
                f"recent_commits empty AND uncommitted_changes=False for {cp} "
                f"(expected one of the two to be true)"
            )
        # If recent_commits is non-empty, every entry must have sha+message+date.
        for entry in rec["recent_commits"]:
            assert isinstance(entry, dict)
            for k in ("sha", "message", "date"):
                assert k in entry, f"recent_commits entry missing {k!r}"
            assert re.match(
                r"^[0-9a-f]{40}$", entry["sha"]
            ), f"recent_commits sha not 40-hex: {entry['sha']!r}"
        assert isinstance(rec["uncommitted_changes"], bool)

    assert (
        seen_paths == canonical_paths
    ), f"missing config_paths: {canonical_paths - seen_paths}"


def test_gate_p1_hp_provenance_log_registered_in_build_gates(
    preflight_module: Any,
) -> None:
    """Gate 3 — registry: p1_hp_provenance_log is in build_gates('all').

    Asserts the gate is registered in the 'all' pipeline, the runner
    name matches the gate function name, and the Wf-321D
    ``EXPECTED_GATE_COUNTS["all"]`` is consistent with the actual
    runner count (the Wf-321D gate-count-consistency audit invariant).

    The Wf-350-A3 ship bumps the 'all' count from 68 → 69 per the
    Wf-350-A3 brief; the sibling Wf-350-A1 task adds 1 more (68 + 2 = 70
    when both ships land).  The test asserts self-consistency between
    ``EXPECTED_GATE_COUNTS["all"]`` and ``len(build_gates("all"))``
    rather than hard-coding a literal — robust to either sequence.
    """
    runners = preflight_module.build_gates("all")
    runner_names: List[str] = [name for name, _ in runners]
    assert "p1_hp_provenance_log" in runner_names, (
        f"p1_hp_provenance_log NOT registered in build_gates('all'); "
        f"got {len(runner_names)} runners"
    )

    # Verify the Wf-321D self-consistency invariant:
    # EXPECTED_GATE_COUNTS["all"] must equal the actual runner count.
    expected_count = preflight_module.EXPECTED_GATE_COUNTS["all"]
    actual_count = len(runner_names)
    assert expected_count == actual_count, (
        f"Wf-321D audit invariant FAILED: "
        f"EXPECTED_GATE_COUNTS['all']={expected_count} but "
        f"build_gates('all') returns {actual_count} runners"
    )
    # Wf-350-A3 ship bumps the count by +1 over the 68 pre-Wf-350
    # baseline (per TODO/completed/321d_preflight_gate_audit.md).
    assert expected_count >= 69, (
        f"EXPECTED_GATE_COUNTS['all'] must be ≥ 69 post-Wf-350-A3 "
        f"(got {expected_count})"
    )

    # Verify the gate function is callable (smoke test).
    runner_for_gate = next(
        runner for name, runner in runners if name == "p1_hp_provenance_log"
    )
    res = runner_for_gate()
    assert res.status in (
        "PASS",
        "FAIL",
    ), f"gate runner returned unexpected status: {res.status!r}"
    # On a healthy git repo (live host), the gate should PASS — but
    # we accept FAIL too because some CI sandboxes may have no git
    # history or unusual repo layout.  The contract here is "registered
    # + callable + returns a valid GateResult", not "PASS on this host".
    assert res.name == "p1_hp_provenance_log"
    assert res.priority in ("P1", "P3")
    assert res.pipeline == "wf350a3"
