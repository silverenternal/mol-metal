"""Tests for Wf-330-C1: Catherine 6/6 properties × 25 d-block metals bench.

==================================================================
Wf-330-C1: R20-P5 closure theorem + Catherine 6/6 properties.

This file pins the MEASURED bench output for the Wf-57 Phase-3
:func:`molmetal.scripts.bench_catherine_6_props_25metals` ship surface,
on a 25-metal × 5-seed × ``--max-products 1000`` sweep.

Two MEASURED verdicts are pinned:

* **REAL closure BFS path** (default; molmetal_lam + RDKit importable):
  ``0 / 25`` metals × 6/6 PASS — P2_transitivity (witness_rate=0.0)
  and P3_symmetry (well_typed_rate=0.0) FAIL on every metal.  This is
  the honest Wf-330-C1 negative: the real closure-theorem BFS does not
  satisfy both transitivity (no finite witness for visited terms) and
  symmetry (non-idempotent second BFS pass).  Categorical closure
  axioms are NOT yet theorem-grade; only operational invariants
  (P1, P4, P5, P6) hold per-metal.  See TODO/completed/330c1_*.md for
  the HONEST_NEGATIVE_R20_P5 ledger.

* **SYNTHETIC fallback path** (``--synthetic-only``; deterministic
  CPU-only stub): ``25 / 25`` metals × 6/6 PASS — useful for slim CI
  environments and as a comparator to the real BFS.  This is the
  Wf-57 ship-target fallback (Wf-57 §2.2) and is NOT a real
  measurement; the JSON explicitly carries ``"synthetic_only": true``.

The PROJECTED target per skill §8.5.2 was ``6/6 PASS on ≥ 25 metals``
— partially met by the synthetic fallback (PASS 25/25) and NOT met by
the real BFS path (PASS 0/25).  The gap on the real path is the
R20-P5 honest-negative ledger.

MEASURED anchors:

  * REAL BFS path
    ``molmetal/reports/wf330_closure_c25/per_metal_pass.json``
    — 25 metals × 1 seed, max_products=1000, max_depth=1 (2026-09-29)
    — n_metals=25, n_pass_6_of_6=0, wall=2.47s
  * SYNTHETIC path
    ``molmetal/reports/wf330_closure_c25_synth/per_metal_pass.json``
    — 25 metals × 5 seeds, max_products=1000, max_depth=1 (2026-09-29)
    — n_metals=25, n_pass_6_of_6=25, wall=1.10s

Per `party-twenty-spirit` §3.5 zero-tolerance discipline: any future
agent claiming 6/6 PASS on the real closure BFS must first close the
P2 + P3 failure ledger; ``synthetic_only=true`` JSON is NOT a real
measurement and MUST NOT be cited as such.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"

# ---------------------------------------------------------------------------
# MEASURED anchors — DO NOT change without re-running the bench on disk.
# ---------------------------------------------------------------------------

REAL_BFS_REPORT = REPORTS_DIR / "wf330_closure_c25" / "per_metal_pass.json"
SYNTHETIC_REPORT = REPORTS_DIR / "wf330_closure_c25_synth" / "per_metal_pass.json"

# Wf-330-C1 real-BFS ship surface MEASURED verdict (2026-09-29).
REAL_BFS_N_METALS_MEASURED = 25
REAL_BFS_6_OF_6_PASS_MEASURED = 0  # HONEST_NEGATIVE_R20_P5
REAL_BFS_FOUR_OF_6_PASS_MEASURED = 25  # P1+P4+P5+P6 pass on every metal
REAL_BFS_P2_PASS_MEASURED = 0  # witness_rate=0.0 on every metal
REAL_BFS_P3_PASS_MEASURED = 0  # well_typed_rate=0.0 on every metal
REAL_BFS_P1_PASS_MEASURED = 25
REAL_BFS_P4_PASS_MEASURED = 25
REAL_BFS_P5_PASS_MEASURED = 25
REAL_BFS_P6_PASS_MEASURED = 25
REAL_BFS_REACHABILITY_MEASURED = 0.018  # 18 / 1000 = 0.018 per metal
REAL_BFS_N_VISITED_MEASURED = 18
REAL_BFS_N_PRODUCTS_MEASURED = 10

# Wf-330-C1 synthetic fallback MEASURED verdict (2026-09-29).
SYNTHETIC_N_METALS_MEASURED = 25
SYNTHETIC_6_OF_6_PASS_MEASURED = 25  # CPU-only stub passes 6/6 by construction

# PROJECTED target per skill §8.5.2.
PROJECTED_6_OF_6_PASS_GATE = 25  # R20-P5 closure + Catherine ship target
PROJECTED_REACHABILITY_GATE = 0.95  # per-metal reachability ≥ 0.95

# Tolerance for JSON-file float comparison (3 dp).
RTOL = 1e-3


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file; return empty dict if missing."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _python_executable() -> str:
    """Locate the project's primary venv binary used for CLI bench runs.

    The bench is CPU-only; the project's ``/home/hugo/codes/mol-metal/.venv`` Python
    (the same binary the Wf-330 ship surface uses) is preferred, but the
    bare ``python3`` fallback keeps the test runnable in a fresh shell.
    """
    candidates = [
        Path("/home/hugo/codes/mol-metal/.venv/bin/python"),
        Path("/home/hugo/.venvs/.venv-flowmol311/bin/python"),
        Path("/home/hugo/codes/mol-metal/.venv/bin/python"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return sys.executable


# ---------------------------------------------------------------------------
# Test 1: bench_catherine_6_props_25metals runs on 25 metals (smoke)
# ---------------------------------------------------------------------------


def test_bench_catherine_runs_on_25_metals() -> None:
    """Bench script imports and accepts --metals 25 + --max-products 1000.

    Runs the smoke path (3 metals, max_products=32) under a 90-second
    timeout so the test stays fast even when the heavy 25-metal bench
    has already been emitted to disk.
    """
    bench_script = SCRIPTS_DIR / "bench_catherine_6_props_25metals.py"
    assert bench_script.is_file(), (
        f"bench script missing: {bench_script}"
    )

    smoke_out = REPORTS_DIR / "wf330_closure_c25_smoke"
    smoke_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        _python_executable(),
        str(bench_script),
        "--metals", "3",
        "--seeds", "0",
        "--max-depth", "1",
        "--max-products", "32",
        "--out-dir", str(smoke_out),
    ]
    env = {"PYTHONPATH": str(PROJECT_ROOT), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=90,
    )
    # The bench returns 0 on success; warnings + RDKit noise are tolerated.
    assert proc.returncode == 0, (
        f"bench returned {proc.returncode}\nSTDOUT:\n{proc.stdout[-1500:]}\n"
        f"STDERR:\n{proc.stderr[-1500:]}"
    )

    smoke_json = smoke_out / "per_metal_pass.json"
    assert smoke_json.is_file(), (
        f"bench did not write per_metal_pass.json to {smoke_out}"
    )
    payload = _load_json(smoke_json)
    assert payload.get("metadata", {}).get("n_metals") == 3
    assert len(payload.get("metals", {})) == 3


# ---------------------------------------------------------------------------
# Test 2: output schema matches catherine_6_props.json contract
# ---------------------------------------------------------------------------


def test_catherine_output_schema_contract() -> None:
    """The shipped per_metal_pass.json matches the Wf-57 contract schema.

    Schema (Wf-57 Phase 3):

      ``metadata``: dict with required keys
        ``wf`` (== 'Wf-57'), ``phase`` (== 'Phase 3'),
        ``n_metals`` (int), ``n_seeds`` (int),
        ``max_depth`` (int), ``max_products`` (int),
        ``use_gpu`` (bool), ``real_bfs_available`` (bool),
        ``synthetic_only`` (bool), ``timestamp`` (str),
        ``wall_time_s_total`` (float)

      ``metals``: dict[metal_symbol, dict] where each verdict has
        ``six_of_six_pass`` (bool), ``per_property`` (dict of 6 entries)

      ``per_property[pid]``: ``passed`` (bool), ``n_samples`` (int),
        ``n_violations`` (int), ``wall_time_s`` (float), ``notes`` (str)

      ``summaries``: dict[metal_symbol, dict] with reachability_rate etc.
    """
    required_metadata = {
        "wf", "phase", "n_metals", "n_seeds", "max_depth", "max_products",
        "use_gpu", "real_bfs_available", "synthetic_only", "timestamp",
        "wall_time_s_total",
    }
    required_pids = {
        "P1_reflexivity", "P2_transitivity", "P3_symmetry",
        "P4_dblock", "P5_reachability", "P6_composability",
    }
    required_per_property_keys = {
        "passed", "n_samples", "n_violations", "wall_time_s", "notes",
    }

    for path, expected_synthetic in [
        (REAL_BFS_REPORT, False),
        (SYNTHETIC_REPORT, True),
    ]:
        payload = _load_json(path)
        if not payload:
            pytest.skip(f"report missing on disk: {path}")
        metadata = payload.get("metadata", {})
        assert required_metadata.issubset(metadata.keys()), (
            f"missing metadata keys in {path}: "
            f"{required_metadata - metadata.keys()}"
        )
        assert metadata.get("wf") == "Wf-57"
        assert metadata.get("phase") == "Phase 3"
        assert metadata.get("synthetic_only") is expected_synthetic
        assert isinstance(metadata.get("n_metals"), int)
        assert isinstance(metadata.get("n_seeds"), int)
        assert isinstance(metadata.get("wall_time_s_total"), (int, float))

        metals = payload.get("metals", {})
        assert len(metals) == metadata["n_metals"], (
            f"metals count mismatch: {len(metals)} vs "
            f"{metadata['n_metals']}"
        )
        for metal_symbol, verdict in metals.items():
            assert "six_of_six_pass" in verdict, (
                f"{metal_symbol} verdict missing six_of_six_pass"
            )
            assert isinstance(verdict["six_of_six_pass"], bool)
            per_property = verdict.get("per_property", {})
            assert required_pids.issubset(per_property.keys()), (
                f"{metal_symbol} per_property missing PIDs: "
                f"{required_pids - per_property.keys()}"
            )
            for pid, entry in per_property.items():
                assert required_per_property_keys.issubset(entry.keys()), (
                    f"{metal_symbol}/{pid} entry missing keys: "
                    f"{required_per_property_keys - entry.keys()}"
                )
                assert isinstance(entry["passed"], bool)
                assert isinstance(entry["n_samples"], int)
                assert isinstance(entry["n_violations"], int)

        summaries = payload.get("summaries", {})
        assert len(summaries) == metadata["n_metals"]
        for metal_symbol, summary in summaries.items():
            assert summary.get("metal") == metal_symbol
            assert "reachability_rate" in summary


# ---------------------------------------------------------------------------
# Test 3: verdict schema (six_of_six_pass + per_property consistency)
# ---------------------------------------------------------------------------


def test_catherine_verdict_schema() -> None:
    """The per-metal verdict's ``six_of_six_pass`` equals (P1..P6 all PASS).

    Six_of_six_pass MUST equal (P1 & P2 & P3 & P4 & P5 & P6).  On the
    real BFS path this evaluates to False (P2/P3 fail); on the
    synthetic path it evaluates to True (all six PASS by construction).
    """
    for path in (REAL_BFS_REPORT, SYNTHETIC_REPORT):
        payload = _load_json(path)
        if not payload:
            pytest.skip(f"report missing on disk: {path}")
        pids = [
            "P1_reflexivity", "P2_transitivity", "P3_symmetry",
            "P4_dblock", "P5_reachability", "P6_composability",
        ]
        for metal_symbol, verdict in payload["metals"].items():
            per_property = verdict["per_property"]
            expected_six = all(
                per_property[pid]["passed"] for pid in pids
            )
            assert verdict["six_of_six_pass"] == expected_six, (
                f"{metal_symbol} verdict inconsistent: "
                f"6/6={verdict['six_of_six_pass']} vs "
                f"P1..P6 all-pass={expected_six}"
            )


# ---------------------------------------------------------------------------
# Test 4: per-metal 6/6 PASS verification (HONEST_NEGATIVE_R20_P5 pinning)
# ---------------------------------------------------------------------------


def test_per_metal_6_of_6_pass_pins_honest_negative() -> None:
    """Pin the Wf-330-C1 HONEST_NEGATIVE_R20_P5 ledger.

    Real BFS path MEASURED (2026-09-29):
      * n_metals=25, n_6_of_6_pass=0
      * P2_transitivity: 0/25 PASS (witness_rate=0.0 on every metal)
      * P3_symmetry: 0/25 PASS (well_typed_rate=0.0 on every metal)
      * P1+P4+P5+P6: 25/25 PASS each (operational invariants hold)
      * Per-metal reachability_rate=0.018 (n_visited=18 / max_products=1000)

    Synthetic fallback MEASURED (2026-09-29):
      * n_metals=25, n_6_of_6_pass=25 (CPU-only stub)

    The PROJECTED gate (≥ 25 metals × 6/6 PASS) is met by the
    synthetic fallback (25/25) but NOT by the real BFS path (0/25).
    The gap on the real path is the R20-P5 honest-negative ledger:
    the closure-theorem BFS does not yet satisfy transitivity (no
    finite witness path for visited terms) or symmetry (non-idempotent
    on second BFS pass).
    """
    # --- Real BFS path MEASURED ledger ---
    real = _load_json(REAL_BFS_REPORT)
    if not real:
        pytest.skip(f"real BFS report missing on disk: {REAL_BFS_REPORT}")

    real_metadata = real["metadata"]
    assert real_metadata["n_metals"] == REAL_BFS_N_METALS_MEASURED
    assert real_metadata["max_products"] == 1000
    assert real_metadata["synthetic_only"] is False
    assert real_metadata["real_bfs_available"] is True

    real_metals = real["metals"]
    n_6_of_6 = sum(1 for v in real_metals.values() if v["six_of_six_pass"])
    assert n_6_of_6 == REAL_BFS_6_OF_6_PASS_MEASURED, (
        f"Real BFS 6/6 count drifted: {n_6_of_6} vs "
        f"MEASURED {REAL_BFS_6_OF_6_PASS_MEASURED}"
    )

    # Per-property ledger (real BFS: 4/6 pass; P2 + P3 fail uniformly).
    pids = [
        "P1_reflexivity", "P2_transitivity", "P3_symmetry",
        "P4_dblock", "P5_reachability", "P6_composability",
    ]
    expected = {
        "P1_reflexivity": REAL_BFS_P1_PASS_MEASURED,
        "P2_transitivity": REAL_BFS_P2_PASS_MEASURED,
        "P3_symmetry": REAL_BFS_P3_PASS_MEASURED,
        "P4_dblock": REAL_BFS_P4_PASS_MEASURED,
        "P5_reachability": REAL_BFS_P5_PASS_MEASURED,
        "P6_composability": REAL_BFS_P6_PASS_MEASURED,
    }
    for pid in pids:
        n_pass = sum(
            1 for v in real_metals.values()
            if v["per_property"][pid]["passed"]
        )
        assert n_pass == expected[pid], (
            f"Real BFS {pid} count drifted: {n_pass} vs "
            f"MEASURED {expected[pid]}"
        )

    # Per-metal reachability ledger (n_visited=18 / 1000 = 0.018).
    for metal_symbol, summary in real["summaries"].items():
        assert summary["n_visited"] == REAL_BFS_N_VISITED_MEASURED, (
            f"{metal_symbol} n_visited drifted: "
            f"{summary['n_visited']} vs "
            f"MEASURED {REAL_BFS_N_VISITED_MEASURED}"
        )
        assert summary["n_products"] == REAL_BFS_N_PRODUCTS_MEASURED
        assert math.isclose(
            summary["reachability_rate"],
            REAL_BFS_REACHABILITY_MEASURED,
            rel_tol=RTOL,
        ), (
            f"{metal_symbol} reachability_rate drifted: "
            f"{summary['reachability_rate']} vs "
            f"MEASURED {REAL_BFS_REACHABILITY_MEASURED}"
        )

    # --- Synthetic fallback MEASURED ledger ---
    synth = _load_json(SYNTHETIC_REPORT)
    if not synth:
        pytest.skip(f"synthetic report missing on disk: {SYNTHETIC_REPORT}")

    synth_metadata = synth["metadata"]
    assert synth_metadata["n_metals"] == SYNTHETIC_N_METALS_MEASURED
    assert synth_metadata["synthetic_only"] is True

    synth_n_6_of_6 = sum(
        1 for v in synth["metals"].values() if v["six_of_six_pass"]
    )
    assert synth_n_6_of_6 == SYNTHETIC_6_OF_6_PASS_MEASURED, (
        f"Synthetic 6/6 count drifted: {synth_n_6_of_6} vs "
        f"MEASURED {SYNTHETIC_6_OF_6_PASS_MEASURED}"
    )

    # --- Gap to PROJECTED gate ---
    real_gap = PROJECTED_6_OF_6_PASS_GATE - n_6_of_6
    synth_gap = PROJECTED_6_OF_6_PASS_GATE - synth_n_6_of_6
    assert real_gap == PROJECTED_6_OF_6_PASS_GATE, (
        f"Real BFS gap to PROJECTED gate = {real_gap} "
        f"(HONEST_NEGATIVE_R20_P5)"
    )
    assert synth_gap == 0, (
        f"Synthetic gap to PROJECTED gate = {synth_gap} "
        f"(should be 0; CPU-only stub passes by construction)"
    )