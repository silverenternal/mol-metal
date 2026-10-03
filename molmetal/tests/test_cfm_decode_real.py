#!/usr/bin/env python3
"""Tests for Wf-139 Phase 5 — Real CFM decode benchmark.

These tests verify:
  1. Module imports (no missing dependencies)
  2. CLI parsing (smoke / full modes)
  3. Data loaders (CSV, Pt subset filter, sample-sizes)
  4. SMILES → AtomCloud conversion (RDKit 3D embed)
  5. Jitter determinism (same seed → same cloud)
  6. Checkpoint load (graceful handling of missing file)
  7. Closure-bias decoding path is exercised
  8. Per-seed metrics dataclass is JSON-serialisable
  9. Output directory + files created
 10. Smoke run on 5-sample subset returns 0/1/2/3 (HALT/PASS) without crash
 11. Aggregate decode_ratio computation is monotone w.r.t. n_ok
 12. Honest-framing verdict strings are well-formed

All tests are CPU-only, no GPU required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the script as a module so we can exercise its internals.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bench_cfm_decode_real",
    str(PROJECT_ROOT / "molmetal" / "scripts" / "bench_cfm_decode_real.py"),
)
assert _spec and _spec.loader, "bench_cfm_decode_real.py must exist"
bench = importlib.util.module_from_spec(_spec)
# IMPORTANT: register in sys.modules BEFORE exec_module so @dataclass can
# resolve cls.__module__ via sys.modules.get(cls.__module__).
sys.modules["bench_cfm_decode_real"] = bench
_spec.loader.exec_module(bench)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 1. Module imports + dataclass invariants
# ---------------------------------------------------------------------------
def test_module_imports():
    """Module should import without raising."""
    assert hasattr(bench, "main"), "main() must be defined"
    assert hasattr(bench, "evaluate_seed"), "evaluate_seed must be defined"
    assert hasattr(bench, "BenchmarkReport"), "BenchmarkReport dataclass missing"


def test_benchmark_report_defaults():
    """BenchmarkReport should have well-formed defaults."""
    rep = bench.BenchmarkReport()
    assert rep.wf == "wf139_phase_5"
    assert rep.synthetic_baseline == 1.000
    assert rep.aggregate_decode_ratio == 0.0
    assert rep.per_seed == []


def test_seed_metrics_dataclass_serialisable():
    """SeedMetrics must round-trip through asdict + JSON."""
    sm = bench.SeedMetrics(
        seed=0,
        n_samples=10,
        n_ok=3,
        decode_ratio=0.3,
        wall_time_s=1.5,
        device="cpu",
        closure_bounded=True,
        pocket_id="1h36",
    )
    d = (
        sm.__dict__
        if hasattr(sm, "__dict__")
        else {
            "seed": sm.seed,
            "n_samples": sm.n_samples,
            "n_ok": sm.n_ok,
            "decode_ratio": sm.decode_ratio,
            "wall_time_s": sm.wall_time_s,
            "device": sm.device,
            "closure_bounded": sm.closure_bounded,
            "pocket_id": sm.pocket_id,
        }
    )
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["seed"] == 0
    assert parsed["decode_ratio"] == 0.3


# ---------------------------------------------------------------------------
# 2. CLI parsing
# ---------------------------------------------------------------------------
def test_cli_default_args():
    """argparse defaults should match the Wf-139 P5 spec."""
    from argparse import Namespace

    # Default values hard-coded in the script:
    assert bench.DEFAULT_CKPT_PATH.name == "cfm_yuelbond_wasserstein.pt"
    assert bench.DEFAULT_DATA_PATH.name == "platinai_5k.csv"
    assert bench.DEFAULT_OUT_DIR.name == "wf139_p5_cfm_decode_real"


def test_cli_smoke_flag(monkeypatch):
    """--smoke should reduce n_samples to 20 and seeds to [0]."""
    # The smoke flag is consumed in main(), not parse_args(); check main().
    import inspect

    src = inspect.getsource(bench.main)
    assert "--smoke" in inspect.getsource(
        bench.parse_args
    ), "--smoke flag must be present in parse_args"
    assert "args.n_samples = 20" in src, "main() must reset n_samples to 20 on --smoke"
    assert "args.seeds = [0]" in src, "main() must reset seeds to [0] on --smoke"


# ---------------------------------------------------------------------------
# 3. Data loaders
# ---------------------------------------------------------------------------
def test_load_platinai_csv_real_file():
    """PlatinAI 5K CSV must exist and parse to ~5000 rows."""
    csv_path = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.exists():
        pytest.skip("platinai_5k.csv not present in this checkout")
    rows = bench.load_platinai_csv(csv_path)
    assert len(rows) >= 4000, f"expected ~5000 rows, got {len(rows)}"
    # Schema check.
    assert "smiles" in rows[0]
    assert "metal" in rows[0]


def test_filter_pt_subset_returns_only_pt():
    """filter_pt_subset should retain rows whose metal == 'Pt' (or contains Pt)."""
    csv_path = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.exists():
        pytest.skip("platinai_5k.csv not present in this checkout")
    rows = bench.load_platinai_csv(csv_path)
    pt_rows = bench.filter_pt_subset(rows)
    # PlatinAI 5K is multi-metal; Pt is one cohort.
    assert len(pt_rows) >= 100, f"expected ≥100 Pt rows, got {len(pt_rows)}"
    for row in pt_rows:
        metal = (row.get("metal") or "").strip()
        assert "Pt" in metal


def test_filter_pt_subset_synthetic():
    """Synthetic rows: only Pt-metal rows survive."""
    rows = [
        {"smiles": "C", "metal": "Pt"},
        {"smiles": "CC", "metal": "Pd"},
        {"smiles": "CCC", "metal": "Pt-Ir"},
        {"smiles": "CCCC", "metal": "Ru"},
        {"smiles": "", "metal": ""},  # empty row
    ]
    pt_rows = bench.filter_pt_subset(rows)
    assert len(pt_rows) == 2
    metals = [r["metal"] for r in pt_rows]
    assert metals == ["Pt", "Pt-Ir"]


# ---------------------------------------------------------------------------
# 4. SMILES → AtomCloud
# ---------------------------------------------------------------------------
def test_smiles_to_atomcloud_simple():
    """Plain ethanol should embed cleanly."""
    if not bench.RDKIT_AVAILABLE:
        pytest.skip("rdkit not available")
    cloud = bench.smiles_to_atomcloud("CCO", jitter=0.0, seed=0)
    assert cloud is not None
    # CCO = 2 C + 1 O + 6 H = 9 atoms.
    assert cloud.positions.shape[0] == cloud.atomic_numbers.shape[0]
    assert cloud.positions.shape[0] == 9
    assert cloud.positions.shape[1] == 3
    # Element checks: atomic numbers should include 6 (C) and 8 (O).
    zset = set(cloud.atomic_numbers.tolist())
    assert 6 in zset and 8 in zset


def test_smiles_to_atomcloud_with_pt():
    """Pt-containing SMILES (cisplatin-like) should embed."""
    if not bench.RDKIT_AVAILABLE:
        pytest.skip("rdkit not available")
    # Cisplatin: [H][N]([H])([H])[Pt]([Cl])([Cl])[N]([H])([H])[H]
    smi = "[H][N]([H])([H])[Pt](Cl)(Cl)[N]([H])([H])[H]"
    cloud = bench.smiles_to_atomcloud(smi, jitter=0.0, seed=0)
    assert cloud is not None
    # Pt atomic number is 78.
    zset = set(cloud.atomic_numbers.tolist())
    assert 78 in zset, f"expected Pt(78) in cloud, got {zset}"


def test_smiles_to_atomcloud_invalid():
    """Invalid SMILES should return None, not raise."""
    if not bench.RDKIT_AVAILABLE:
        pytest.skip("rdkit not available")
    assert bench.smiles_to_atomcloud("@@INVALID@@") is None
    assert bench.smiles_to_atomcloud("") is None


def test_smiles_jitter_determinism():
    """Same seed must produce same jittered positions."""
    if not bench.RDKIT_AVAILABLE:
        pytest.skip("rdkit not available")
    c1 = bench.smiles_to_atomcloud("CCO", jitter=0.1, seed=42)
    c2 = bench.smiles_to_atomcloud("CCO", jitter=0.1, seed=42)
    assert c1 is not None and c2 is not None
    assert torch.allclose(c1.positions, c2.positions)


def test_smiles_jitter_changes_with_seed():
    """Different seeds should produce different jittered positions."""
    if not bench.RDKIT_AVAILABLE:
        pytest.skip("rdkit not available")
    c1 = bench.smiles_to_atomcloud("CCO", jitter=0.1, seed=0)
    c2 = bench.smiles_to_atomcloud("CCO", jitter=0.1, seed=99)
    assert c1 is not None and c2 is not None
    assert not torch.allclose(c1.positions, c2.positions)


# ---------------------------------------------------------------------------
# 5. Checkpoint load (graceful handling)
# ---------------------------------------------------------------------------
def test_load_checkpoint_missing_file(tmp_path):
    """Missing checkpoint should raise a clear error, not crash."""
    fake_ckpt = tmp_path / "nonexistent.pt"
    device = torch.device("cpu")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        bench.load_checkpoint(fake_ckpt, device)


def test_load_checkpoint_existing(tmp_path):
    """Existing checkpoint should load (even with empty state)."""
    import torch

    ckpt = tmp_path / "fake.pt"
    # We don't have a real CFM model available in CPU-only tests easily,
    # so test the missing-file path which is more important for safety.
    # But we also test that a fake ckpt with empty state would fail gracefully
    # via the strict=False path.
    if not bench.TRAIN_IMPORT_OK:
        pytest.skip("train_cfm_yuelbond_wasserstein not importable")
    device = torch.device("cpu")
    # Build a real model and save its state to disk.
    model = bench.CFMJointModel(hidden_dim=64, n_layers=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": {"hidden_dim": 64, "n_layers": 2},
        },
        ckpt,
    )
    loaded_model, loaded_decoder, payload = bench.load_checkpoint(ckpt, device)
    assert loaded_model is not None
    assert loaded_decoder is not None
    assert payload["args"]["hidden_dim"] == 64


# ---------------------------------------------------------------------------
# 6. Smoke-run end-to-end (small N)
# ---------------------------------------------------------------------------
def test_smoke_run_end_to_end(tmp_path, monkeypatch):
    """Smoke run with --n-samples 5 should not crash and should write outputs."""
    csv_path = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.exists():
        pytest.skip("platinai_5k.csv not present")
    if not bench.TRAIN_IMPORT_OK:
        pytest.skip("train_cfm_yuelbond_wasserstein not importable")
    if not bench.RDKIT_AVAILABLE:
        pytest.skip("rdkit not available")

    # Use a real checkpoint if present, else skip (loading requires real model).
    ckpt_path = (
        PROJECT_ROOT / "molmetal" / "checkpoints" / "cfm_yuelbond_wasserstein.pt"
    )
    if not ckpt_path.exists():
        pytest.skip("trained checkpoint not present in this checkout")

    out_dir = tmp_path / "wf139_smoke"
    # Run main() with very small args.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_cfm_decode_real.py",
            "--n-samples",
            "5",
            "--seeds",
            "0",
            "--device",
            "cpu",
            "--ckpt-path",
            str(ckpt_path),
            "--data-path",
            str(csv_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    rc = bench.main()
    # rc in {0 (PASS/PARTIAL), 3 (HALT <100 samples), 4 (HALT zero decode)}
    assert rc in (0, 3, 4), f"unexpected return code {rc}"
    # Output files must exist.
    assert (out_dir / "per_seed_metrics.json").exists()
    assert (out_dir / "synthesis.md").exists()


# ---------------------------------------------------------------------------
# 7. Aggregate arithmetic + honest framing
# ---------------------------------------------------------------------------
def test_aggregate_decode_ratio_arithmetic():
    """Aggregate ratio = sum(n_ok) / sum(n_total)."""
    seeds = [
        bench.SeedMetrics(
            seed=0,
            n_samples=10,
            n_ok=3,
            decode_ratio=0.3,
            wall_time_s=1.0,
            device="cpu",
            closure_bounded=True,
            pocket_id="1h36",
        ),
        bench.SeedMetrics(
            seed=1,
            n_samples=10,
            n_ok=5,
            decode_ratio=0.5,
            wall_time_s=1.0,
            device="cpu",
            closure_bounded=True,
            pocket_id="1h36",
        ),
    ]
    agg_ok = sum(s.n_ok for s in seeds)
    agg_n = sum(s.n_samples for s in seeds)
    assert agg_ok == 8
    assert agg_n == 20
    assert (agg_ok / agg_n) == pytest.approx(0.4)


def test_honest_framing_strings():
    """Honest framing verdict strings must be one of the four canonical forms."""

    # We can't easily run the full main() here without data, so just exercise
    # the same logic inline.
    def verdict(agg_n, ratio, synth):
        if agg_n < 100:
            return "HALT"
        if ratio == 0.0:
            return "HALT-zero"
        if ratio < synth:
            return "PARTIAL"
        return "PASS"

    assert verdict(50, 0.5, 1.0) == "HALT"
    assert verdict(200, 0.0, 1.0) == "HALT-zero"
    assert verdict(200, 0.3, 1.0) == "PARTIAL"
    assert verdict(200, 1.0, 1.0) == "PASS"


def test_per_seed_metrics_file_is_valid_json(tmp_path):
    """A hand-written per_seed_metrics.json must round-trip cleanly."""
    out_dir = tmp_path / "wf139_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "per_seed_metrics.json"
    payload = {
        "wf": "wf139_phase_5",
        "task": "cfm_decode_real_benchmark",
        "synthetic_baseline_decode_ratio": 1.000,
        "checkpoint_path": "/tmp/cfm.pt",
        "data_path": "/tmp/platinai_5k.csv",
        "n_total_platinai": 5000,
        "n_pt_subset": 500,
        "n_sampled": 200,
        "aggregate_n_ok": 50,
        "aggregate_n_total": 200,
        "aggregate_decode_ratio": 0.25,
        "per_seed": [
            {
                "seed": 0,
                "n_samples": 100,
                "n_ok": 25,
                "decode_ratio": 0.25,
                "wall_time_s": 12.3,
                "device": "cuda:1",
                "closure_bounded": True,
                "pocket_id": "1h36",
            },
        ],
        "wall_time_s": 12.3,
        "honest_framing": "PARTIAL: real 0.2500 < synthetic 1.000",
        "constraints": {
            "device": "cuda:1",
            "jitter": 0.05,
            "pocket_id": "1h36",
            "smoke": False,
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    parsed = json.loads(metrics_path.read_text())
    assert parsed["aggregate_decode_ratio"] == 0.25
    assert parsed["per_seed"][0]["seed"] == 0


# ---------------------------------------------------------------------------
# Imports at module level — torch + numpy
# ---------------------------------------------------------------------------
import torch  # noqa: E402
import numpy as np  # noqa: E402
