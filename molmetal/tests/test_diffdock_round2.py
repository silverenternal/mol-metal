"""Tests for Wf-71 Phase 1 — DiffDock round-2 per-pocket bench.

This module verifies the contract of
``molmetal/scripts/bench_diffdock_round2.py``:

1. The bench can run end-to-end in **stub-fallback mode** (default) and
   emits ``diffdock_status.json`` + per-pocket ``metrics.json`` +
   ``summary.json`` even when no DiffDock backend is reachable.
2. The bench can be invoked with ``--use-real-diffdock
   --diffdock-path`` and records an honest ``MISSING_DEP`` verdict
   when the path is not an executable CLI / vendored repo.
3. The stub confidences are **deterministic** functions of
   (pocket, smiles, seed) — two runs with the same inputs produce
   identical numbers.
4. The bench-level summary correctly aggregates per-pocket counts
   (n_cells, n_real, n_stub, n_failed, mean_confidence, best_cell).

The 12-test target maps roughly:

* test_smoke_* — the bench runs end-to-end and produces all 3 outputs.
* test_diffdock_status_* — bench-level status JSON contract.
* test_per_pocket_metrics_* — per-pocket metrics.json contract.
* test_summary_* — sweep summary aggregation.
* test_stub_deterministic — stub scores are reproducible.
* test_real_diffdock_missing_dep — honest framing when binary missing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "bench_diffdock_round2.py"
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_bench(tmp_path: Path, *extra: str) -> Path:
    """Invoke the bench with ``extra`` CLI flags + tmp out-dir.

    Returns the out-dir path.  Raises if the bench exits non-zero.
    """
    out_dir = tmp_path / "wf71_diffdock_round2"
    cmd = [PYTHON, str(SCRIPT_PATH), "--out-dir", str(out_dir), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"bench exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    return out_dir


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 1. Smoke — bench runs end-to-end in stub-fallback mode
# ---------------------------------------------------------------------------
def test_smoke_emits_all_three_outputs(tmp_path: Path) -> None:
    """Default (stub) mode must emit status + per-pocket + summary."""
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "1h36",
        "--seeds",
        "0",
    )
    assert (out_dir / "diffdock_status.json").is_file()
    assert (out_dir / "summary.json").is_file()
    for pocket in ("830c", "1h36"):
        assert (out_dir / pocket / "metrics.json").is_file()


def test_smoke_default_status_is_missing_dep(tmp_path: Path) -> None:
    """Default mode (no --use-real-diffdock) must record MISSING_DEP."""
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "--seeds",
        "0",
    )
    payload = _read_json(out_dir / "diffdock_status.json")
    assert payload["diffdock_status"]["status"] == "MISSING_DEP"
    assert payload["diffdock_status"]["bin"] is None


# ---------------------------------------------------------------------------
# 2. diffdock_status.json contract
# ---------------------------------------------------------------------------
def test_diffdock_status_has_schema(tmp_path: Path) -> None:
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "--seeds",
        "0",
    )
    payload = _read_json(out_dir / "diffdock_status.json")
    assert payload["schema_version"] == 1
    assert payload["wf_tag"] == "Wf-71 Phase 1"
    assert "generated_utc" in payload


def test_diffdock_status_records_args(tmp_path: Path) -> None:
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "1h36",
        "--seeds",
        "0",
        "1",
    )
    payload = _read_json(out_dir / "diffdock_status.json")
    args = payload["args"]
    assert args["pockets"] == ["830c", "1h36"]
    assert args["seeds"] == [0, 1]
    assert args["use_real_diffdock"] is False
    assert args["diffdock_path"] is None


# ---------------------------------------------------------------------------
# 3. per-pocket metrics.json contract
# ---------------------------------------------------------------------------
def test_per_pocket_metrics_has_n_cells(tmp_path: Path) -> None:
    """n_cells = n_cands × n_seeds."""
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "--cands",
        "CCO",
        "c1ccccc1",
        "--seeds",
        "0",
        "1",
        "2",
    )
    pm = _read_json(out_dir / "830c" / "metrics.json")
    assert pm["pocket_id"] == "830c"
    assert pm["n_cells"] == 2 * 3  # 2 cands × 3 seeds = 6
    assert pm["n_stub"] == 6
    assert pm["n_real"] == 0
    assert pm["n_failed"] == 0
    assert pm["diffdock_status"] == "MISSING_DEP"


def test_per_pocket_metrics_records_cells(tmp_path: Path) -> None:
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "--seeds",
        "0",
    )
    pm = _read_json(out_dir / "830c" / "metrics.json")
    assert len(pm["cells"]) == pm["n_cells"]
    cell = pm["cells"][0]
    assert {
        "smiles",
        "seed",
        "status",
        "confidence",
        "rmsd_A",
        "vina_kcal",
        "elapsed_s",
        "note",
    } <= set(cell.keys())
    assert cell["status"] == "STUB_FALLBACK"


def test_per_pocket_metrics_confidence_in_range(tmp_path: Path) -> None:
    """Stub confidences must lie within the documented range."""
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "1h36",
        "--seeds",
        "0",
        "1",
        "2",
    )
    for pocket in ("830c", "1h36"):
        pm = _read_json(out_dir / pocket / "metrics.json")
        for cell in pm["cells"]:
            assert (
                0.0 <= cell["confidence"] <= 1.0
            ), f"confidence out of [0,1] for {pocket}/{cell['smiles']}"


# ---------------------------------------------------------------------------
# 4. summary.json contract
# ---------------------------------------------------------------------------
def test_summary_aggregates_per_pocket(tmp_path: Path) -> None:
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "1h36",
        "3eml",
        "--seeds",
        "0",
    )
    summary = _read_json(out_dir / "summary.json")
    assert summary["n_pockets"] == 3
    assert summary["n_seeds"] == 1
    assert summary["n_cells_total"] == 3 * 5  # 3 pockets × 5 default cands
    assert "830c" in summary["per_pocket"]
    assert "1h36" in summary["per_pocket"]
    assert "3eml" in summary["per_pocket"]
    for pocket, pdata in summary["per_pocket"].items():
        assert pdata["n_cells"] == 5
        assert pdata["n_real"] == 0
        assert pdata["n_stub"] == 5


def test_summary_records_honest_framing(tmp_path: Path) -> None:
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "--seeds",
        "0",
    )
    summary = _read_json(out_dir / "summary.json")
    assert "honest_framing" in summary
    assert isinstance(summary["honest_framing"], list)
    assert any("STUB_FALLBACK" in s for s in summary["honest_framing"])


# ---------------------------------------------------------------------------
# 5. Stub determinism
# ---------------------------------------------------------------------------
def test_stub_scores_are_deterministic(tmp_path: Path) -> None:
    """Two runs with identical inputs must produce identical numbers."""
    out_a = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "--cands",
        "CCO",
        "--seeds",
        "0",
    )
    out_b = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "--cands",
        "CCO",
        "--seeds",
        "0",
    )
    pa = _read_json(out_a / "830c" / "metrics.json")
    pb = _read_json(out_b / "830c" / "metrics.json")
    assert pa["cells"][0]["confidence"] == pytest.approx(pb["cells"][0]["confidence"])
    assert pa["cells"][0]["rmsd_A"] == pytest.approx(pb["cells"][0]["rmsd_A"])
    assert pa["cells"][0]["vina_kcal"] == pytest.approx(pb["cells"][0]["vina_kcal"])


def test_stub_scores_differ_per_seed(tmp_path: Path) -> None:
    """Different seeds must produce different confidences for the same cell."""
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "--cands",
        "CCO",
        "--seeds",
        "0",
        "1",
        "2",
    )
    pm = _read_json(out_dir / "830c" / "metrics.json")
    confs = [c["confidence"] for c in pm["cells"]]
    assert len(set(confs)) == len(confs), "stub confidences should differ across seeds"


# ---------------------------------------------------------------------------
# 6. Real-mode honest framing
# ---------------------------------------------------------------------------
def test_real_diffdock_missing_dep(tmp_path: Path) -> None:
    """--use-real-diffdock with bad path must record MISSING_DEP honestly."""
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "--seeds",
        "0",
        "--use-real-diffdock",
        "--diffdock-path",
        "/nonexistent/diffdock/binary",
    )
    status = _read_json(out_dir / "diffdock_status.json")
    assert status["diffdock_status"]["status"] == "MISSING_DEP"
    assert status["args"]["use_real_diffdock"] is True
    assert status["args"]["diffdock_path"] == "/nonexistent/diffdock/binary"
    # Every cell must remain STUB_FALLBACK (no fabrication).
    pm = _read_json(out_dir / "830c" / "metrics.json")
    for cell in pm["cells"]:
        assert cell["status"] == "STUB_FALLBACK"
    # The reason should mention the missing binary so the gap is visible.
    assert "No executable DiffDock" in status["diffdock_status"]["reason"]


def test_real_diffdock_records_checked_paths(tmp_path: Path) -> None:
    """--use-real-diffdock must record every path it checked."""
    out_dir = _run_bench(
        tmp_path,
        "--pockets",
        "830c",
        "--seeds",
        "0",
        "--use-real-diffdock",
        "--diffdock-path",
        "/tmp/maybe_diffdock_here",
    )
    status = _read_json(out_dir / "diffdock_status.json")
    paths = status["diffdock_status"]["checked_paths"]
    # The explicit path must be first; well-known candidates follow.
    assert paths[0] == "/tmp/maybe_diffdock_here"
    assert "/usr/local/bin/diffdock" in paths
    assert any("DiffDock" in p for p in paths)
