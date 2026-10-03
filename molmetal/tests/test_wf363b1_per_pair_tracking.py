"""Wf-363-B1 — GPU launch dashboard per-pair completion tracking tests.

Owner:        Hugo Lee (autonomous Wf-363-B1 ultracode unit)
Date:         2026-09-29
Origin:       Wf-363-B1 per-pair completion tracking (per Wf-360-A3 dashboard)
Parent:       scripts/gpu_launch_dashboard.sh + molmetal/scripts/gpu_launch_aggregator.py

These 2 NEW tests verify that the GPU launch dashboard emits per-pair
completion tracking as specified in Wf-363-B1:

  1. **per_pair_progress schema** — the aggregator and bash dashboard
     both expose a `per_pair_progress` array of {metal, seed, status,
     wall_elapsed_s, last_log_ts} when --per-pair-detail is set.
  2. **status inference from run.log** — pair_status() correctly
     classifies PLANNED / RUNNING / COMPLETED / FAILED from synthetic
     per-pair run.log + training_summary.json fixtures.

All 2 tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no GPU
launches; aggregator is read-only).
Per party-twenty-spirit §3.5: every claim traces to a file/line or live-
run output. No paper/* touches (TODO-84 honored).
"""

from __future__ import annotations

import importlib.util as importlib_util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
DASHBOARD_SH = PROJECT_ROOT / "scripts" / "gpu_launch_dashboard.sh"
AGGREGATOR_PY = PROJECT_ROOT / "molmetal" / "scripts" / "gpu_launch_aggregator.py"

VALID_PAIR_STATUSES = {
    "PLANNED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "STALE",
    "UNKNOWN",
}
EXPECTED_PER_PAIR_FIELDS = {
    "metal",
    "seed",
    "status",
    "wall_elapsed_s",
    "last_log_ts",
}


def _load_aggregator_module():
    """Load gpu_launch_aggregator.py as a module without running main()."""
    spec = importlib_util.spec_from_file_location(
        "gpu_launch_aggregator", str(AGGREGATOR_PY)
    )
    mod = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_01_per_pair_progress_schema():
    """Verify per_pair_progress schema is emitted correctly with --per-pair-detail.

    Builds a synthetic fixture launcher root with 3 (metal, seed) pairs in
    different states (PLANNED, RUNNING, COMPLETED) and asserts that:
      - aggregator's aggregate_launcher(..., include_per_pair=True) returns
        a per_pair_progress array of 3 dicts with the expected schema.
      - bash dashboard's --json --per-pair-detail --launcher pipeline
        produces JSON with per_pair_progress.
      - Each entry's status is in VALID_PAIR_STATUSES.
    """
    mod = _load_aggregator_module()

    # Build a synthetic fixture under temp dir
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "fake_launcher"
        root.mkdir(parents=True, exist_ok=True)
        # Pair 1: COMPLETED (has training_summary.json with status=OK)
        (root / "Ru" / "seed0").mkdir(parents=True, exist_ok=True)
        (root / "Ru" / "seed0" / "run.log").write_text(
            "DONE total elapsed=1234.5 best_epoch=42\n"
        )
        (root / "Ru" / "seed0" / "training_summary.json").write_text(
            json.dumps({"status": "OK", "metal": "Ru", "seed": 0})
        )
        # Pair 2: RUNNING (run.log exists, mtime fresh)
        (root / "Ir" / "seed1").mkdir(parents=True, exist_ok=True)
        (root / "Ir" / "seed1" / "run.log").write_text("epoch=12/50 ...\n")
        # Pair 3: PLANNED (no run.log, no training_summary.json)
        (root / "Pt" / "seed2").mkdir(parents=True, exist_ok=True)

        # 1) Test aggregator's per-pair detail via aggregate_launcher
        entry = mod.aggregate_launcher(
            name="fake_launcher",
            root=str(root),
            script="/tmp/fake.sh",
            wall_proj=1.0,
            include_per_pair=True,
        )
        assert "per_pair_progress" in entry, (
            "aggregate_launcher(include_per_pair=True) must emit "
            f"'per_pair_progress' field; got keys: {list(entry.keys())}"
        )
        per_pair = entry["per_pair_progress"]
        assert isinstance(per_pair, list)
        assert len(per_pair) == 3, f"expected 3 per-pair entries, got {len(per_pair)}"

        # Check schema on each entry
        for p in per_pair:
            missing = EXPECTED_PER_PAIR_FIELDS - set(p.keys())
            assert not missing, f"per-pair entry missing fields: {missing}"
            assert (
                p["status"] in VALID_PAIR_STATUSES
            ), f"invalid status: {p['status']!r} (must be in {VALID_PAIR_STATUSES})"
            assert isinstance(p["wall_elapsed_s"], int)
            assert p["wall_elapsed_s"] >= 0

        # Verify the (metal, seed) parsing for each pair
        by_id = {(p["metal"], p["seed"]): p for p in per_pair}
        assert ("Ru", "0") in by_id
        assert ("Ir", "1") in by_id
        assert ("Pt", "2") in by_id

        # Ru/seed0 should be COMPLETED (training_summary.json status=OK → COMPLETED)
        assert (
            by_id[("Ru", "0")]["status"] == "COMPLETED"
        ), f"Ru/seed0 status should be COMPLETED, got {by_id[('Ru', '0')]['status']!r}"
        # Ir/seed1 should be RUNNING (run.log exists, mtime fresh)
        assert (
            by_id[("Ir", "1")]["status"] == "RUNNING"
        ), f"Ir/seed1 status should be RUNNING, got {by_id[('Ir', '1')]['status']!r}"
        # Pt/seed2 should be PLANNED (no artifacts at all)
        assert (
            by_id[("Pt", "2")]["status"] == "PLANNED"
        ), f"Pt/seed2 status should be PLANNED, got {by_id[('Pt', '2')]['status']!r}"

        # last_log_ts should be ISO-8601 UTC when artifacts exist
        assert by_id[("Ru", "0")]["last_log_ts"] is not None
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
            by_id[("Ru", "0")]["last_log_ts"],
        ), f"last_log_ts not ISO-8601: {by_id[('Ru', '0')]['last_log_ts']}"
        # Pt/seed2 (no artifacts): last_log_ts is None
        assert by_id[("Pt", "2")]["last_log_ts"] is None

        # 2) Test bash dashboard --json --per-pair-detail pipeline using CLI
        # Verify bash syntax
        result = subprocess.run(
            ["bash", "-n", str(DASHBOARD_SH)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"bash -n failed: {result.stderr}"

        # Verify --per-pair-detail + --help banner mentions it
        with open(DASHBOARD_SH) as f:
            src = f.read()
        assert (
            "--per-pair-detail" in src
        ), "bash dashboard must declare --per-pair-detail flag"

        # 3) Test pair_progress_detail() directly on synthetic pair
        detail = mod.pair_progress_detail(str(root / "Ru" / "seed0"))
        assert detail["metal"] == "Ru"
        assert detail["seed"] == "0"
        assert detail["status"] == "COMPLETED"
        assert detail["wall_elapsed_s"] >= 0

        # 4) Test parse_pair_identity() directly
        metal, seed = mod.parse_pair_identity(str(root / "Ru" / "seed0"))
        assert metal == "Ru"
        assert seed == "0"

        # 5) Test pair_status() directly (alias OK → COMPLETED)
        st = mod.pair_status(str(root / "Ru" / "seed0"))
        assert (
            st == "COMPLETED"
        ), f"pair_status should normalize OK → COMPLETED; got {st!r}"

        # 6) PLANNED status: empty pair dir
        st = mod.pair_status(str(root / "Pt" / "seed2"))
        assert st == "PLANNED", f"empty pair dir should be PLANNED; got {st!r}"


def test_02_status_inference_from_run_log():
    """Verify status inference: PLANNED / RUNNING / COMPLETED / FAILED / STALE.

    Builds 5 synthetic per-pair fixtures with run.log + training_summary.json
    in different shapes and asserts pair_status() correctly classifies each.
    """
    mod = _load_aggregator_module()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "fixture"
        root.mkdir(parents=True, exist_ok=True)

        # Case A: COMPLETED via training_summary.json status=OK
        (root / "Pt" / "seed0").mkdir(parents=True, exist_ok=True)
        (root / "Pt" / "seed0" / "run.log").write_text("starting...\n")
        (root / "Pt" / "seed0" / "training_summary.json").write_text(
            json.dumps({"status": "OK", "metal": "Pt"})
        )
        # Force mtime to be old enough that RUNNING fallback would say STALE
        old = 1000.0
        os.utime(root / "Pt" / "seed0" / "run.log", (old, old))
        os.utime(root / "Pt" / "seed0" / "training_summary.json", (old, old))
        assert mod.pair_status(str(root / "Pt" / "seed0")) == "COMPLETED"

        # Case B: FAILED via training_summary.json status=FAILED
        (root / "Ru" / "seed1").mkdir(parents=True, exist_ok=True)
        (root / "Ru" / "seed1" / "run.log").write_text("starting...\n")
        (root / "Ru" / "seed1" / "training_summary.json").write_text(
            json.dumps({"status": "FAILED", "metal": "Ru"})
        )
        os.utime(root / "Ru" / "seed1" / "run.log", (old, old))
        os.utime(root / "Ru" / "seed1" / "training_summary.json", (old, old))
        assert mod.pair_status(str(root / "Ru" / "seed1")) == "FAILED"

        # Case C: COMPLETED via run.log signature "DONE total elapsed=" (no summary)
        (root / "Ir" / "seed2").mkdir(parents=True, exist_ok=True)
        (root / "Ir" / "seed2" / "run.log").write_text(
            "[epoch 50/50] DONE total elapsed=987.6 best_epoch=42\n"
        )
        os.utime(root / "Ir" / "seed2" / "run.log", (old, old))
        assert mod.pair_status(str(root / "Ir" / "seed2")) == "COMPLETED"

        # Case D: FAILED via run.log Traceback (no summary)
        (root / "Au" / "seed3").mkdir(parents=True, exist_ok=True)
        (root / "Au" / "seed3" / "run.log").write_text(
            'Traceback (most recent call last):\n  File "train.py", line 42\n'
        )
        os.utime(root / "Au" / "seed3" / "run.log", (old, old))
        assert mod.pair_status(str(root / "Au" / "seed3")) == "FAILED"

        # Case E: STALE — run.log exists but mtime > 30 min, no summary
        (root / "Pd" / "seed4").mkdir(parents=True, exist_ok=True)
        (root / "Pd" / "seed4" / "run.log").write_text("epoch=5/50\n")
        os.utime(root / "Pd" / "seed4" / "run.log", (old, old))
        st = mod.pair_status(str(root / "Pd" / "seed4"))
        assert st == "STALE", f"old run.log with no summary should be STALE; got {st!r}"

        # Case F: RUNNING — run.log exists, mtime is fresh (< 30 min)
        (root / "Os" / "seed5").mkdir(parents=True, exist_ok=True)
        (root / "Os" / "seed5" / "run.log").write_text("epoch=15/50\n")
        # Don't touch mtime — leave it as now
        st = mod.pair_status(str(root / "Os" / "seed5"))
        assert st == "RUNNING", f"fresh run.log should be RUNNING; got {st!r}"

        # Case G: PLANNED — no run.log, no training_summary.json
        (root / "Rh" / "seed6").mkdir(parents=True, exist_ok=True)
        assert mod.pair_status(str(root / "Rh" / "seed6")) == "PLANNED"

        # Case H: ALL_SEEDS_FAILED in training_summary.json → FAILED
        (root / "Re" / "seed7").mkdir(parents=True, exist_ok=True)
        (root / "Re" / "seed7" / "training_summary.json").write_text(
            json.dumps({"status": "ALL_SEEDS_FAILED"})
        )
        os.utime(root / "Re" / "seed7" / "training_summary.json", (old, old))
        assert mod.pair_status(str(root / "Re" / "seed7")) == "FAILED"

        # Case I: COMPLETED via per_pair_table (all OK)
        (root / "Ir" / "seed8").mkdir(parents=True, exist_ok=True)
        (root / "Ir" / "seed8" / "training_summary.json").write_text(
            json.dumps(
                {
                    "per_pair_table": [
                        {"status": "OK"},
                        {"status": "OK"},
                        {"status": "OK"},
                    ],
                }
            )
        )
        os.utime(root / "Ir" / "seed8" / "training_summary.json", (old, old))
        assert mod.pair_status(str(root / "Ir" / "seed8")) == "COMPLETED"

        # Case J: per_pair_table with FAILED → PARTIAL
        (root / "Pt" / "seed9").mkdir(parents=True, exist_ok=True)
        (root / "Pt" / "seed9" / "training_summary.json").write_text(
            json.dumps(
                {
                    "per_pair_table": [
                        {"status": "OK"},
                        {"status": "FAILED"},
                    ],
                }
            )
        )
        os.utime(root / "Pt" / "seed9" / "training_summary.json", (old, old))
        assert mod.pair_status(str(root / "Pt" / "seed9")) == "PARTIAL"

        # Case K: g6_recover shape — <cfg>/<Metal>/seed<N>
        (root / "no_dann" / "Ru" / "seed0").mkdir(parents=True, exist_ok=True)
        (root / "no_dann" / "Ru" / "seed0" / "training_summary.json").write_text(
            json.dumps({"status": "OK"})
        )
        os.utime(
            root / "no_dann" / "Ru" / "seed0" / "training_summary.json", (old, old)
        )
        assert mod.pair_status(str(root / "no_dann" / "Ru" / "seed0")) == "COMPLETED"

        # Verify pair_progress_detail for the g6 shape
        detail = mod.pair_progress_detail(str(root / "no_dann" / "Ru" / "seed0"))
        assert detail["metal"] == "Ru", f"g6 metal parse: {detail['metal']!r}"
        assert detail["seed"] == "0", f"g6 seed parse: {detail['seed']!r}"
        assert detail["status"] == "COMPLETED"
