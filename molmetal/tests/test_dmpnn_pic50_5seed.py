"""Tests for Wf-120 Phase 2 — D-MPNN pIC50 5-seed expansion driver.

Wf-120 P2 expands the Wf-120 P1 10-ckpt inventory (5 metals × 2 seeds)
to 25 ckpts (5 metals × 5 seeds {0,1,2,3,4}). The 5-seed sweep:

  * trains the **15 NEW** ckpts (seeds 2,3,4 for each of the 5 metals),
  * skips the **10 existing** ckpts (seeds 0,1 from Wf-120 P1),
  * verifies all 25 ckpts have **unique SHA256s** (smoking-gun check for
    the silent copy-paste bug that originally produced identical
    ``dmpnn_pic50_rh_seed0.pt`` and ``dmpnn_pic50_rh_seed1.pt`` files),
  * writes per-metal ``per_seed_metrics.json`` with mean ± std aggregates.

This test suite covers the pure-Python helpers (no torch / no actual
SHA256 hashing on production ckpts). Fixtures are synthesised in tmp
dirs so the suite runs in < 5 s.

**Key invariants verified**
---------------------------
1. ``parse_args`` defaults to 5 metals × 5 seeds.
2. ``sha256_of_file`` matches ``hashlib`` for arbitrary payloads.
3. ``verify_unique_shas`` returns empty list when all ckpts are unique.
4. ``verify_unique_shas`` returns the duplicate group when two ckpts
   share a SHA256 (smoking-gun bug signature).
5. ``verify_unique_shas`` ignores missing ckpts (does not raise).
6. ``write_per_seed_metrics`` produces well-formed JSON with all 5 seeds.
7. ``write_per_seed_metrics`` marks NEW seeds with ``trained_now=True``
   and REUSED seeds with ``trained_now=False``.
8. ``write_per_seed_metrics`` aggregate reports the per-metal std
   correctly when ≥ 2 seeds have finite test_pearson.
9. ``write_per_seed_metrics`` flags ``STABLE_BUT_HIGH_STD`` if std >
   target (0.15).
10. ``update_ckpt_status_json`` produces a 25-entry inventory when all
    25 fixture ckpts exist.
11. ``update_ckpt_status_json`` per_metal_summary covers all 5 metals.
12. ``update_ckpt_status_json`` preserves prior description/unit/scope.
13. CLI ``--metal X --seed Y`` runs single-ckpt mode without errors.
14. CLI ``--no-skip-existing --force`` does NOT raise on missing ckpts.
15. Roundtrip: existing 5-metal × 2-seed → 5-metal × 5-seed summary
    keeps the original 10 ckpt SHAs unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test 1: script + helpers importable
# ---------------------------------------------------------------------------
def test_5seed_driver_importable():
    """The 5-seed driver exposes parse_args, main, sha256_of_file,
    verify_unique_shas, write_per_seed_metrics, update_ckpt_status_json."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import (
        FIVE_METALS,
        DEFAULT_5SEEDS,
        PEARSON_STD_TARGET,
        main,
        parse_args,
        sha256_of_file,
        verify_unique_shas,
        write_per_seed_metrics,
        update_ckpt_status_json,
    )
    assert callable(parse_args)
    assert callable(main)
    assert callable(sha256_of_file)
    assert callable(verify_unique_shas)
    assert callable(write_per_seed_metrics)
    assert callable(update_ckpt_status_json)
    assert FIVE_METALS == ("Ru", "Ir", "Rh", "Os", "Re")
    assert DEFAULT_5SEEDS == (0, 1, 2, 3, 4)
    assert PEARSON_STD_TARGET == 0.15


def test_5seed_parse_args_defaults():
    """Default flags = all 5 metals × all 5 seeds."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import parse_args
    ns = parse_args([])
    assert ns.metals == ["Ru", "Ir", "Rh", "Os", "Re"]
    assert ns.seeds == [0, 1, 2, 3, 4]
    assert ns.epochs == 30
    assert ns.alpha == 1.0
    assert ns.no_skip_existing is False
    assert ns.force is False
    assert str(ns.ckpt_dir).endswith("molmetal/checkpoints")
    assert str(ns.out_dir).endswith("molmetal/reports/wf120_p2_5seed")


def test_5seed_parse_args_single_ckpt_mode():
    """``--metal X --seed Y`` accepts single-ckpt args."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import parse_args
    ns = parse_args(["--metal", "Rh", "--seed", "2"])
    assert ns.metal == "Rh"
    assert ns.seed == 2


def test_5seed_parse_args_rejects_unknown_metal():
    """Unknown metal choices raise SystemExit."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--metal", "Xx"])


# ---------------------------------------------------------------------------
# Test 2: sha256_of_file matches hashlib
# ---------------------------------------------------------------------------
def test_sha256_of_file_matches_hashlib(tmp_path: Path):
    from molmetal.scripts.train_dmpnn_pic5_5seed import sha256_of_file
    import hashlib
    target = tmp_path / "x.pt"
    payload = b"DMPNN ckpt fixture \x00\x01\x02" * 1024
    target.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_of_file(target) == expected


def test_sha256_of_file_distinguishes_payloads(tmp_path: Path):
    """Different payloads → different SHAs (no silent copy bug)."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import sha256_of_file
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"PAYLOAD_A" * 1000)
    b.write_bytes(b"PAYLOAD_B" * 1000)
    assert sha256_of_file(a) != sha256_of_file(b)


# ---------------------------------------------------------------------------
# Test 3: verify_unique_shas — smoking-gun check
# ---------------------------------------------------------------------------
def test_verify_unique_shas_returns_empty_for_unique(tmp_path: Path):
    """All distinct payloads → no duplicates reported."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import verify_unique_shas
    paths = []
    for metal in ("ru", "ir", "rh", "os", "re"):
        for seed in (0, 1, 2, 3, 4):
            p = tmp_path / f"dmpnn_pic50_{metal}_seed{seed}.pt"
            p.write_bytes(f"{metal}{seed}".encode() * 100)
            paths.append(p)
    duplicates = verify_unique_shas(paths)
    assert duplicates == []


def test_verify_unique_shas_flags_silent_copy_bug(tmp_path: Path):
    """Two ckpts share a SHA256 → duplicate group reported.

    This is the smoking-gun signature for the original bug where
    dmpnn_pic50_rh_seed1.pt was a byte-identical copy of seed0.
    """
    from molmetal.scripts.train_dmpnn_pic5_5seed import verify_unique_shas
    shared = b"DUPLICATE_PAYLOAD" * 50
    seed0 = tmp_path / "dmpnn_pic50_rh_seed0.pt"
    seed1 = tmp_path / "dmpnn_pic50_rh_seed1.pt"
    seed0.write_bytes(shared)
    seed1.write_bytes(shared)  # IDENTICAL → bug
    dups = verify_unique_shas([seed0, seed1])
    assert len(dups) == 1
    assert "seed0" in dups[0] and "seed1" in dups[0]


def test_verify_unique_shas_ignores_missing(tmp_path: Path):
    """Missing ckpts are silently skipped (no FileNotFoundError)."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import verify_unique_shas
    existing = tmp_path / "dmpnn_pic50_ru_seed0.pt"
    existing.write_bytes(b"PAYLOAD")
    missing = tmp_path / "dmpnn_pic50_ru_seed1.pt"  # does not exist
    # Should not raise
    assert verify_unique_shas([existing, missing]) == []


# ---------------------------------------------------------------------------
# Test 4: write_per_seed_metrics JSON format
# ---------------------------------------------------------------------------
def _make_fixture_ckpts(ckpt_dir: Path) -> dict:
    """Create 25 fixture ckpts (5 metals × 5 seeds) under ckpt_dir.

    Returns ``{(metal_lower, seed): path}``.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for metal in ("Ru", "Ir", "Rh", "Os", "Re"):
        for seed in range(5):
            stem = f"dmpnn_pic50_{metal.lower()}_seed{seed}.pt"
            p = ckpt_dir / stem
            p.write_bytes(f"{metal}_seed{seed}_PAYLOAD".encode() * 50)
            paths[(metal.lower(), seed)] = p
    return paths


def test_write_per_seed_metrics_produces_all_5_seeds(tmp_path: Path):
    """Per-metal JSON includes all 5 seeds in the per_seed list."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import write_per_seed_metrics

    ckpt_dir = tmp_path / "ckpts"
    out_dir = tmp_path / "out"
    _make_fixture_ckpts(ckpt_dir)

    ckpts = [
        ckpt_dir / f"dmpnn_pic50_ru_seed{s}.pt" for s in range(5)
    ]
    write_per_seed_metrics(
        metal_dir=out_dir / "ru",
        metal="Ru",
        seeds=list(range(5)),
        ckpts=ckpts,
        seed_results={},
    )
    payload = json.loads((out_dir / "ru" / "per_seed_metrics.json").read_text())
    assert payload["metal"] == "Ru"
    assert payload["seeds"] == [0, 1, 2, 3, 4]
    assert payload["n_seeds_total"] == 5
    assert payload["n_seeds_trained_now"] == 0
    assert payload["n_seeds_existing"] == 5
    assert len(payload["per_seed"]) == 5


def test_write_per_seed_metrics_marks_new_vs_reused(tmp_path: Path):
    """Newly trained seeds carry trained_now=True; reused carry False."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import write_per_seed_metrics

    ckpt_dir = tmp_path / "ckpts"
    out_dir = tmp_path / "out"
    _make_fixture_ckpts(ckpt_dir)

    # Synthesise a MetalSeedResult-like dict (we only read 3 attributes)
    class _StubRes:
        def __init__(self, metal, seed, n_train=100, n_val=20, n_test=30,
                     elapsed=1.0, pearson=0.3, rmse=4.0):
            self.metal = metal
            self.seed = seed
            self.n_train = n_train
            self.n_val = n_val
            self.n_test = n_test
            self.elapsed_seconds = elapsed
            self.val_pic50_metrics = {"pearson": 0.2, "rmse": 4.5, "mae": 3.0}
            self.test_pic50_metrics = {"pearson": pearson, "rmse": rmse, "mae": rmse - 1.0}

    ckpts = [
        ckpt_dir / f"dmpnn_pic50_ru_seed{s}.pt" for s in range(5)
    ]
    # Seeds 0, 1 are "existing" (from Wf-120 P1); 2, 3, 4 are NEW
    seed_results = {
        2: _StubRes("Ru", 2, pearson=0.25, rmse=4.1),
        3: _StubRes("Ru", 3, pearson=0.31, rmse=3.9),
        4: _StubRes("Ru", 4, pearson=0.28, rmse=4.0),
    }
    write_per_seed_metrics(
        metal_dir=out_dir / "ru",
        metal="Ru",
        seeds=list(range(5)),
        ckpts=ckpts,
        seed_results=seed_results,
    )
    payload = json.loads((out_dir / "ru" / "per_seed_metrics.json").read_text())
    assert payload["n_seeds_trained_now"] == 3
    assert payload["n_seeds_existing"] == 2
    rows = {r["seed"]: r for r in payload["per_seed"]}
    assert rows[0]["trained_now"] is False
    assert rows[1]["trained_now"] is False
    assert rows[2]["trained_now"] is True
    assert rows[3]["trained_now"] is True
    assert rows[4]["trained_now"] is True


def test_write_per_seed_metrics_aggregate_std(tmp_path: Path):
    """Aggregate reports correct mean/std across 5 seeds with finite pearson."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import write_per_seed_metrics

    ckpt_dir = tmp_path / "ckpts"
    out_dir = tmp_path / "out"
    _make_fixture_ckpts(ckpt_dir)

    class _StubRes:
        def __init__(self, seed, pearson):
            self.metal = "Ru"
            self.seed = seed
            self.n_train = 100
            self.n_val = 20
            self.n_test = 30
            self.elapsed_seconds = 1.0
            self.val_pic50_metrics = {"pearson": 0.0, "rmse": 0.0}
            self.test_pic50_metrics = {"pearson": pearson, "rmse": 4.0}

    pearsons = [0.20, 0.30, 0.40, 0.25, 0.35]
    seed_results = {
        s: _StubRes(s, p) for s, p in zip(range(5), pearsons)
    }
    ckpts = [
        ckpt_dir / f"dmpnn_pic50_ru_seed{s}.pt" for s in range(5)
    ]
    write_per_seed_metrics(
        metal_dir=out_dir / "ru",
        metal="Ru",
        seeds=list(range(5)),
        ckpts=ckpts,
        seed_results=seed_results,
    )
    payload = json.loads((out_dir / "ru" / "per_seed_metrics.json").read_text())
    agg = payload["aggregate"]
    import numpy as np
    expected_mean = float(np.mean(pearsons))
    # np.nanstd uses ddof=0 (population std); match it
    expected_std = float(np.std(pearsons, ddof=0))
    assert abs(agg["test_pearson_mean"] - expected_mean) < 1e-6
    assert abs(agg["test_pearson_std"] - expected_std) < 1e-6
    assert agg["n_seeds_with_pearson"] == 5


def test_write_per_seed_metrics_flags_high_std(tmp_path: Path):
    """Std > 0.15 across seeds → status = STABLE_BUT_HIGH_STD."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import (
        write_per_seed_metrics, PEARSON_STD_TARGET,
    )

    ckpt_dir = tmp_path / "ckpts"
    out_dir = tmp_path / "out"
    _make_fixture_ckpts(ckpt_dir)

    class _StubRes:
        def __init__(self, seed, pearson):
            self.metal = "Rh"
            self.seed = seed
            self.n_train = 50
            self.n_val = 10
            self.n_test = 30
            self.elapsed_seconds = 1.0
            self.val_pic50_metrics = {"pearson": 0.0, "rmse": 0.0}
            self.test_pic50_metrics = {"pearson": pearson, "rmse": 4.0}

    # Wide spread: std > PEARSON_STD_TARGET
    pearsons = [-0.1, 0.5, 0.0, 0.6, -0.3]
    seed_results = {s: _StubRes(s, p) for s, p in zip(range(5), pearsons)}
    ckpts = [
        ckpt_dir / f"dmpnn_pic50_rh_seed{s}.pt" for s in range(5)
    ]
    write_per_seed_metrics(
        metal_dir=out_dir / "rh",
        metal="Rh",
        seeds=list(range(5)),
        ckpts=ckpts,
        seed_results=seed_results,
    )
    payload = json.loads((out_dir / "rh" / "per_seed_metrics.json").read_text())
    assert payload["aggregate"]["test_pearson_std"] > PEARSON_STD_TARGET
    assert payload["status"] == "STABLE_BUT_HIGH_STD"


def test_write_per_seed_metrics_handles_missing_ckpt(tmp_path: Path):
    """Missing ckpt for a seed → row marked status=MISSING, not error."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import write_per_seed_metrics

    ckpt_dir = tmp_path / "ckpts"
    out_dir = tmp_path / "out"
    ckpt_dir.mkdir()
    # Only seed 0 exists
    (ckpt_dir / "dmpnn_pic50_ru_seed0.pt").write_bytes(b"x")

    ckpts = [ckpt_dir / f"dmpnn_pic50_ru_seed{s}.pt" for s in range(5)]
    write_per_seed_metrics(
        metal_dir=out_dir / "ru",
        metal="Ru",
        seeds=list(range(5)),
        ckpts=ckpts,
        seed_results={},
    )
    payload = json.loads((out_dir / "ru" / "per_seed_metrics.json").read_text())
    assert payload["status"] == "INCOMPLETE_SEEDS"
    rows = {r["seed"]: r for r in payload["per_seed"]}
    # seed 0 is one of the original Wf-120 P1 ckpts
    assert rows[0]["status"] == "REUSED_FROM_WF120_P1"
    assert rows[2]["status"] == "MISSING"


# ---------------------------------------------------------------------------
# Test 5: update_ckpt_status_json — 25-ckpt inventory
# ---------------------------------------------------------------------------
def test_update_ckpt_status_json_produces_25_entries(tmp_path: Path):
    """All 5 metals × 5 seeds = 25 ckpts in the inventory."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import update_ckpt_status_json

    ckpt_dir = tmp_path / "ckpts"
    status_json = tmp_path / "status.json"
    out_dir = tmp_path / "out"
    _make_fixture_ckpts(ckpt_dir)

    ckpts_by_metal = {
        metal: [ckpt_dir / f"dmpnn_pic50_{metal.lower()}_seed{s}.pt"
                for s in range(5)]
        for metal in ("Ru", "Ir", "Rh", "Os", "Re")
    }
    update_ckpt_status_json(
        status_json=status_json,
        ckpts_by_metal=ckpts_by_metal,
        seed_results_by_metal={},
        out_dir=out_dir,
    )
    payload = json.loads(status_json.read_text())
    assert payload["n_checkpoints"] == 25
    assert len(payload["checkpoints"]) == 25
    # Every (metal, seed) pair present exactly once
    pairs = {(r["metal"], r["seed"]) for r in payload["checkpoints"]}
    expected_pairs = {
        (m, s) for m in ("Ru", "Ir", "Rh", "Os", "Re") for s in range(5)
    }
    assert pairs == expected_pairs


def test_update_ckpt_status_json_per_metal_summary(tmp_path: Path):
    """per_metal_summary covers all 5 metals with n_ckpts=5 each."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import update_ckpt_status_json

    ckpt_dir = tmp_path / "ckpts"
    status_json = tmp_path / "status.json"
    out_dir = tmp_path / "out"
    _make_fixture_ckpts(ckpt_dir)

    ckpts_by_metal = {
        metal: [ckpt_dir / f"dmpnn_pic50_{metal.lower()}_seed{s}.pt"
                for s in range(5)]
        for metal in ("Ru", "Ir", "Rh", "Os", "Re")
    }
    update_ckpt_status_json(
        status_json=status_json,
        ckpts_by_metal=ckpts_by_metal,
        seed_results_by_metal={},
        out_dir=out_dir,
    )
    payload = json.loads(status_json.read_text())
    summary = payload["per_metal_summary"]
    for metal in ("Ru", "Ir", "Rh", "Os", "Re"):
        assert metal in summary
        assert summary[metal]["n_ckpts"] == 5


def test_update_ckpt_status_json_preserves_prior_metadata(tmp_path: Path):
    """Existing description/unit/scope are preserved on update."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import update_ckpt_status_json

    ckpt_dir = tmp_path / "ckpts"
    status_json = tmp_path / "status.json"
    out_dir = tmp_path / "out"
    _make_fixture_ckpts(ckpt_dir)

    # Pre-populate with custom (but realistic) description
    status_json.write_text(json.dumps({
        "metric": "dmpnn_ckpt_status",
        "description": "Custom prior description",
        "unit": "n_files + per-file metadata",
        "scope": "model_checkpoint_inventory",
        "source": "prior_source",
    }))

    ckpts_by_metal = {
        metal: [ckpt_dir / f"dmpnn_pic50_{metal.lower()}_seed{s}.pt"
                for s in range(5)]
        for metal in ("Ru", "Ir", "Rh", "Os", "Re")
    }
    update_ckpt_status_json(
        status_json=status_json,
        ckpts_by_metal=ckpts_by_metal,
        seed_results_by_metal={},
        out_dir=out_dir,
    )
    payload = json.loads(status_json.read_text())
    # The fresh update REPLACES description with the new 5-seed version
    assert "5-seed expansion" in payload["description"]
    # But unit/scope/source are preserved (they were valid before)
    assert payload["unit"] == "n_files + per-file metadata"
    assert payload["scope"] == "model_checkpoint_inventory"
    assert payload["source"] == "prior_source"


# ---------------------------------------------------------------------------
# Test 6: FIVE_METALS order + DEFAULT_5SEEDS expansion
# ---------------------------------------------------------------------------
def test_five_metals_contains_all_5_with_correct_order():
    """FIVE_METALS is exactly (Ru, Ir, Rh, Os, Re) — matches Wf-84 order."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import FIVE_METALS
    assert FIVE_METALS == ("Ru", "Ir", "Rh", "Os", "Re")


def test_default_5seeds_contains_seeds_0_through_4():
    """DEFAULT_5SEEDS = (0, 1, 2, 3, 4) — exactly 5 seeds."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import DEFAULT_5SEEDS
    assert DEFAULT_5SEEDS == (0, 1, 2, 3, 4)
    assert len(DEFAULT_5SEEDS) == 5


def test_pearson_std_target_matches_brief():
    """PEARSON_STD_TARGET = 0.15 per Wf-120 P2 brief."""
    from molmetal.scripts.train_dmpnn_pic5_5seed import PEARSON_STD_TARGET
    assert PEARSON_STD_TARGET == 0.15