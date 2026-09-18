"""Smoke tests for ``molmetal/scripts/r4_c_full_sweep.py``.

Two checks only — both are non-mutating:

1. ``--help`` exits 0 and prints the new CLI surface
   (``--config`` / ``--dry-run``, no ``--sota-aligned``).
2. ``--dry-run`` exits 0, loads the canonical SOTA-aligned YAML,
   prints a summary, and DOES NOT invoke the per-pocket runner.

The actual sweep is intentionally NOT exercised here — that's gated
on Vina + DiffDock/FlowDock oracle availability (TODO/pending/
decisions.md D4) and is a future test target.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "r4_c_full_sweep.py"
POCKETS_DIR = Path(
    "/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10"
)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run the script with ``args`` and capture stdout/stderr/return code."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )


def _combined(proc: subprocess.CompletedProcess) -> str:
    """Combine stdout + stderr so logging.info output is searchable too."""
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------
def test_help_exits_zero_and_lists_new_flags() -> None:
    """``--help`` exits 0 and surfaces the SOTA-aligned CLI surface."""
    proc = _run(["--help"])
    assert proc.returncode == 0, proc.stderr
    assert "--config" in proc.stdout
    assert "--dry-run" in proc.stdout
    assert "--n-pockets" in proc.stdout
    # Old flag removed:
    assert "--sota-aligned" not in proc.stdout


# ---------------------------------------------------------------------------
# --dry-run (only when the pocket dir is reachable)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not POCKETS_DIR.is_dir(),
    reason=f"CrossDocked pocket dir missing: {POCKETS_DIR}",
)
def test_dry_run_loads_config_and_exits_zero_without_sweeping(tmp_path: Path) -> None:
    """``--dry-run`` loads the YAML, prints summary, exits 0, writes no files."""
    output_prefix = tmp_path / "r4_c_dryrun"
    proc = _run(
        [
            "--pockets", str(POCKETS_DIR),
            "--output-prefix", str(output_prefix),
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr

    combined = _combined(proc)
    # Summary block markers — every section line printed by --dry-run.
    assert "[dry-run] SOTA-aligned config OK:" in proc.stdout
    assert "[dry-run] OK — sweep NOT executed." in proc.stdout
    # The canonical YAML fields appear in the summary:
    assert "sota_aligned_targetdiff.yaml" in combined
    assert "test_set          : crossdocked_pocket10" in combined
    assert "n_test_pockets    : 100" in combined
    # MCTS knobs from the YAML surface (stdout + logging.info combined):
    assert "top_k=100" in combined
    assert "early_stop=True" in combined
    assert "patience=50" in combined
    assert "'max_depth': 3" in combined
    assert "'n_simulations': 1000" in combined
    assert "'branching_target': 1020" in combined
    # Lambda-only levers:
    assert "extended_204" in combined
    assert "all_5" in combined

    # Critically: no CSV / JSON / MD written — proves no sweep ran.
    assert not (tmp_path / "r4_c_dryrun.csv").exists()
    assert not (tmp_path / "r4_c_dryrun.json").exists()
    assert not (tmp_path / "r4_c_dryrun.md").exists()


@pytest.mark.skipif(
    not POCKETS_DIR.is_dir(),
    reason=f"CrossDocked pocket dir missing: {POCKETS_DIR}",
)
def test_dry_run_respects_n_pockets_override(tmp_path: Path) -> None:
    """``--n-pockets 3`` overrides the argparse default of 5 in dry-run."""
    output_prefix = tmp_path / "r4_c_dryrun_override"
    proc = _run(
        [
            "--pockets", str(POCKETS_DIR),
            "--output-prefix", str(output_prefix),
            "--n-pockets", "3",
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    combined = _combined(proc)
    assert "n_pockets_request : 3" in combined
    assert "n_pockets_found   : 3" in combined


@pytest.mark.skipif(
    not POCKETS_DIR.is_dir(),
    reason=f"CrossDocked pocket dir missing: {POCKETS_DIR}",
)
def test_dry_run_caps_n_pockets_at_yaml_n_test_pockets(tmp_path: Path) -> None:
    """``--n-pockets`` is capped at the YAML ``n_test_pockets`` (100)."""
    output_prefix = tmp_path / "r4_c_dryrun_cap"
    proc = _run(
        [
            "--pockets", str(POCKETS_DIR),
            "--output-prefix", str(output_prefix),
            "--n-pockets", "500",
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    # 500 is capped to 100 by min(args.n_pockets, n_test_pockets=100).
    # The orchestrator logs the effective n_pockets on stderr.
    combined = _combined(proc)
    assert "Effective run plan: n_pockets=100" in combined
    assert "config cap = 100" in combined
    # The dry-run summary line must reflect the CAP (not the CLI override).
    assert "n_pockets_request : 100" in combined
    assert "n_pockets_found   : 100" in combined


@pytest.mark.skipif(
    not POCKETS_DIR.is_dir(),
    reason=f"CrossDocked pocket dir missing: {POCKETS_DIR}",
)
def test_dry_run_respects_n_pockets_override(tmp_path: Path) -> None:
    """``--n-pockets 3`` overrides the argparse default of 5 in dry-run."""
    output_prefix = tmp_path / "r4_c_dryrun_override"
    proc = _run(
        [
            "--pockets", str(POCKETS_DIR),
            "--output-prefix", str(output_prefix),
            "--n-pockets", "3",
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    combined = _combined(proc)
    assert "n_pockets_request : 3" in combined
    assert "n_pockets_found   : 3" in combined


@pytest.mark.skipif(
    not POCKETS_DIR.is_dir(),
    reason=f"CrossDocked pocket dir missing: {POCKETS_DIR}",
)
def test_dry_run_caps_n_pockets_at_yaml_n_test_pockets(tmp_path: Path) -> None:
    """``--n-pockets`` is capped at the YAML ``n_test_pockets`` (100)."""
    output_prefix = tmp_path / "r4_c_dryrun_cap"
    proc = _run(
        [
            "--pockets", str(POCKETS_DIR),
            "--output-prefix", str(output_prefix),
            "--n-pockets", "500",
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    # 500 should be capped to 100 by min(args.n_pockets, n_test_pockets=100).
    # But the actual pocket count found is whatever CrossDocked_pocket10
    # has (much more than 100 in this env).  What we assert: the CLI
    # value of 500 is NOT respected, and the cap holds.
    # The orchestrator logs the effective n_pockets on stderr (via
    # logging.info) and the cap surfaces on stdout too.
    combined = _combined(proc)
    assert "Effective run plan: n_pockets=100" in combined
    assert "config cap = 100" in combined
    # The dry-run summary line must reflect the CAP (not the CLI override):
    assert "n_pockets_request : 100" in combined
    assert "n_pockets_found   : 100" in combined
