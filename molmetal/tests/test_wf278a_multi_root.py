"""Tests for Wf-298B: wf278a_g4_aggregate.py multi-root patch.

Adds multi-root --input-dirs support to wf278a_g4_aggregate.py for
cascading G4 (30-epoch) + G5 (50-epoch) sweeps with last-wins-by-mtime
dedup. Verifies:

  1. test_multi_root_merge   — multi-root merge produces union of (metal, seed) keys.
  2. test_last_wins_dedup    — when same (metal, seed) appears under both roots,
                              the file with the latest mtime wins.
  3. test_expected_pairs_15  — default expected_pairs is 15 (5 metals × 3 seeds).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "molmetal" / "scripts" / "wf278a_g4_aggregate.py"


def _seed_payload(metal: str, seed: int, epochs: int = 30) -> dict:
    """Build a minimal per_seed_metrics_seed{N}.json payload."""
    return {
        "metal": metal,
        "seed": seed,
        "epochs": epochs,
        "alpha": 0.5,
        "wasserstein_weight": 0.05,
        "use_l2_guard": True,
        "ckpt_path": f"results/_test/{metal}/seed{seed}/ckpts/dmpnn.pt",
        "hidden": 300,
        "depth": 3,
        "n_train": 100,
        "n_val": 50,
        "n_test": 50,
        "elapsed_seconds": 60.0,
        "val_metrics": {"roc_auc": 0.8, "n": 50, "n_pos": 10, "n_neg": 40},
        "test_metrics": {"roc_auc": 0.75, "n": 50, "n_pos": 10, "n_neg": 40},
        "val_pic50_metrics": {"mae": 0.6, "rmse": 0.8, "pearson": 0.3},
        "test_pic50_metrics": {"mae": 0.65, "rmse": 0.85, "pearson": 0.25},
        "wasserstein_history": [1.0, 0.9, 0.8, 0.7],
        "status": "OK",
    }


def _write_seed_file(
    root: Path,
    wave: str,
    metal: str,
    seed: int,
    payload: dict,
    touch_mtime: float | None = None,
) -> Path:
    """Write a nested per_seed_metrics_seed{N}.json under (root/wave/Metal/seed{N}/metal/).

    If touch_mtime is given, set the file's mtime to that absolute timestamp
    (used by test_last_wins_dedup to force a deterministic dedup ordering).
    """
    metal_lower = metal.lower()
    out = (
        root
        / wave
        / metal
        / f"seed{seed}"
        / metal_lower
        / f"per_seed_metrics_seed{seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    if touch_mtime is not None:
        import os

        os.utime(out, (touch_mtime, touch_mtime))
    return out


def _import_module():
    """Import wf278a_g4_aggregate as a module via importlib."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "wf278a_g4_aggregate", str(SCRIPT_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_multi_root_merge(tmp_path: Path) -> None:
    """Multi-root merge: union of (metal, seed) keys across both roots.

    G4 has Ru/seed0, Ir/seed1; G5 has Ru/seed0 (overlap) + Os/seed2 (new).
    After merge we expect 3 unique (metal, seed) pairs, NOT 4 (no double-count).
    """
    mod = _import_module()

    g4 = tmp_path / "g4"
    g5 = tmp_path / "g5"

    # G4 (older roots) — set explicit older mtime
    _write_seed_file(g4, "A", "Ru", 0, _seed_payload("Ru", 0), touch_mtime=1_000_000.0)
    _write_seed_file(g4, "A", "Ir", 1, _seed_payload("Ir", 1), touch_mtime=1_000_000.0)
    # G5 — newer mtime on overlap, plus a brand-new Os/seed2
    _write_seed_file(
        g5, "B", "Ru", 0, _seed_payload("Ru", 0, epochs=50), touch_mtime=2_000_000.0
    )
    _write_seed_file(
        g5, "B", "Os", 2, _seed_payload("Os", 2, epochs=50), touch_mtime=2_000_000.0
    )

    out = mod.collect_seed_metrics_multi_root([g4, g5])
    keys = {(s["metal"], s["seed"]) for s in out}

    assert len(out) == 3, f"expected 3 unique pairs after dedup, got {len(out)}: {keys}"
    assert ("Ru", 0) in keys
    assert ("Ir", 1) in keys
    assert ("Os", 2) in keys


def test_last_wins_dedup(tmp_path: Path) -> None:
    """Last-wins-by-mtime: when same (metal, seed) appears in both roots, the
    file with the later mtime (G5 50-ep) must be chosen over the G4 30-ep file.
    """
    mod = _import_module()

    g4 = tmp_path / "g4"
    g5 = tmp_path / "g5"

    g4_payload = _seed_payload("Ru", 0, epochs=30)
    g5_payload = _seed_payload("Ru", 0, epochs=50)
    g5_payload["test_metrics"]["roc_auc"] = 0.9999  # Distinctive value

    g4_path = _write_seed_file(g4, "A", "Ru", 0, g4_payload, touch_mtime=1_000_000.0)
    g5_path = _write_seed_file(g5, "B", "Ru", 0, g5_payload, touch_mtime=2_000_000.0)

    out = mod.collect_seed_metrics_multi_root([g4, g5])
    assert len(out) == 1, f"expected 1 deduped pair, got {len(out)}"
    picked = out[0]
    # The newer-mtime G5 payload (epochs=50, roc_auc=0.9999) must win
    assert (
        picked["epochs"] == 50
    ), f"expected epochs=50 (G5 wins), got {picked['epochs']}"
    assert picked["test_metrics"]["roc_auc"] == pytest.approx(0.9999)
    # And the older G4 file must NOT appear
    assert (
        picked is not g4_payload
    )  # at least proves the merge materialised a fresh dict


def test_expected_pairs_15() -> None:
    """Default expected_pairs is 15 (5 metals × 3 seeds), per Wf-298B patch."""
    import subprocess

    # Invoke the CLI with --help to inspect the default. argparse prints the
    # default in the help text when default= is set; we instead parse the
    # source to assert the default value, which is robust against argparse
    # help formatting.
    src = SCRIPT_PATH.read_text()
    assert "expected_pairs" in src
    # The default is set to 15 in the argparse.add_argument(...) call.
    # Accept either form: default=15 or "--expected-pairs", default=15
    assert "default=15" in src, (
        "wf278a_g4_aggregate.py must set --expected-pairs default=15 "
        "(5 metals × 3 seeds)"
    )
    # And the legacy hard-coded "expected_pairs": 14 must be gone.
    assert (
        '"expected_pairs": 14' not in src
    ), "Legacy hard-coded expected_pairs=14 must be removed (replaced by CLI default=15)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
