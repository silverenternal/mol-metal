#!/usr/bin/env python3
"""Regression tests for the ex1 harness flags added for the PA3/PA15 migration.

Covers AGENTS.md §5.7 (every new flag wire-in gets a CPU smoke) for the four
flags added 2026-10-15/16:

  * ``--cohort-spec {paper_funnel,24h}``
  * ``--dump-test-scores``
  * ``--no-adapt``
  * ``--config`` (YAML round-trip, §5.11)

The §5.7 shape is adapted to a classifier harness: instead of ``decode_ratio``
(CFM-only), the checks are that the source training actually reduces the loss
over 50 optimizer steps, and that ``no_adapt`` yields the documented zero-shot
contract (no support draw, full held-out query, ``adapted_auc`` NaN).

Data-dependent tests skip cleanly when MetalCytoToxDB.csv is absent.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DATA_DIR = Path(
    os.environ.get("MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal"))
)
_HAS_DB = (_DATA_DIR / "MetalCytoToxDB.csv").exists()
_needs_db = pytest.mark.skipif(not _HAS_DB, reason="MetalCytoToxDB.csv not present")


# ---------------------------------------------------------------------------
# 1. parse_args accepts the new flags
# ---------------------------------------------------------------------------
def test_parse_args_accepts_new_flags():
    from molmetal.scripts.ex1_metal_injection_contrast import parse_args

    a = parse_args(["--cohort-spec", "24h", "--no-adapt"])
    assert a.cohort_spec == "24h"
    assert a.no_adapt is True

    b = parse_args(["--cohort-spec", "paper_funnel", "--dump-test-scores", "/tmp/x.jsonl"])
    assert b.cohort_spec == "paper_funnel"
    assert str(b.dump_test_scores).endswith("x.jsonl")
    assert b.no_adapt is False


def test_parse_args_rejects_unknown_cohort_spec():
    from molmetal.scripts.ex1_metal_injection_contrast import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--cohort-spec", "not_a_cohort"])


# ---------------------------------------------------------------------------
# 2. §5.11 YAML round-trip — by TYPE, not just value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cfg,expect_arms,expect_rich,expect_chem",
    [
        ("configs/ex1_lomo_24h.yaml", ["none"], False, False),
        ("configs/ex1_pa15_24h_token.yaml", ["readout"], False, False),
        ("configs/ex1_pa15_24h_desc.yaml", ["readout"], True, True),
        ("configs/ex1_pa15_24h_film.yaml", ["film"], False, False),
    ],
)
def test_yaml_roundtrip_by_type(cfg, expect_arms, expect_rich, expect_chem):
    from molmetal.scripts.ex1_metal_injection_contrast import parse_args

    path = PROJECT_ROOT / cfg
    assert path.exists(), f"{cfg} missing"
    a = parse_args(["--config", str(path)])

    assert isinstance(a.metals, list) and a.metals == ["Ru", "Ir", "Rh", "Os", "Re"]
    assert isinstance(a.seeds, list) and a.seeds == [42, 43, 44, 49, 123]
    assert isinstance(a.k, list)
    assert isinstance(a.epochs, int) and a.epochs == 50
    assert isinstance(a.lr, float)
    assert isinstance(a.hidden, int)
    assert a.cohort_spec == "24h"
    assert a.arms == expect_arms
    assert a.no_adapt is True
    assert a.rich_metal is expect_rich
    assert a.chem_metal is expect_chem


def test_yaml_explicit_cli_wins_over_config():
    """§5.11 precedence: explicit CLI flag > YAML > argparse default."""
    from molmetal.scripts.ex1_metal_injection_contrast import parse_args

    path = PROJECT_ROOT / "configs/ex1_lomo_24h.yaml"
    a = parse_args(["--config", str(path), "--seeds", "7"])
    # argparse type=int is applied to CLI tokens; the YAML cohort must NOT win.
    assert [int(s) for s in a.seeds] == [7]


# ---------------------------------------------------------------------------
# 3. §5.7 smoke — the source training reduces the loss over ~50 steps
# ---------------------------------------------------------------------------
@_needs_db
def test_source_training_reduces_loss_50_steps():
    import torch

    from molmetal.models.metal_aware_dmpnn import MetalAwareDMPNN
    from molmetal.scripts.ex1_metal_injection_contrast import (
        featurize,
        load_cohort,
        lomo_split,
    )

    cohort = load_cohort(("Ru", "Ir", "Rh", "Os", "Re"), cohort_spec="24h")
    tr, _te, _n = lomo_split(cohort, "Re")
    graphs = featurize(tr, None, True, False, False)
    y = np.asarray(tr["y"], dtype=np.float32)

    torch.manual_seed(42)
    model = MetalAwareDMPNN(
        metal_injection="none", metal_feat_dim=18, hidden=32, dropout=0.0
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()

    valid = [(g, yi) for g, yi in zip(graphs, y) if g is not None]
    gs = [g for g, _ in valid]
    ys = torch.from_numpy(np.asarray([v for _, v in valid], dtype=np.float32))

    def _loss() -> float:
        model.eval()
        with torch.no_grad():
            return float(lossf(model.forward_graphs(gs, device=torch.device("cpu")), ys))

    l0 = _loss()
    model.train()
    n = len(gs)
    bs = 64
    steps_per_pass = (n + bs - 1) // bs
    passes = max(2, -(-50 // steps_per_pass))  # ceil to reach >= 50 steps
    rng = np.random.default_rng(0)
    steps = 0
    for _ in range(passes):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            idx = order[s : s + bs]
            logits = model.forward_graphs([gs[i] for i in idx], device=torch.device("cpu"))
            loss = lossf(logits, ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
    l1 = _loss()
    assert steps >= 50, f"only {steps} optimizer steps ran"
    assert np.isfinite(l0) and np.isfinite(l1)
    assert l1 < l0, f"loss did not decrease: {l0:.4f} -> {l1:.4f}"


# ---------------------------------------------------------------------------
# 4. no_adapt contract on the real cohort
# ---------------------------------------------------------------------------
@_needs_db
def test_no_adapt_contract_and_score_dump(tmp_path):
    from molmetal.scripts.ex1_metal_injection_contrast import load_cohort, run_one
    import torch

    cohort = load_cohort(("Ru", "Ir", "Rh", "Os", "Re"), cohort_spec="24h")
    r = run_one(
        cohort, "Re", "none", 10, 42, epochs=1, adapt_steps=5, lr=1e-3,
        device=torch.device("cpu"), no_adapt=True, return_scores=True,
    )
    # no support draw, full held-out query
    assert r["n_support"] == 0
    assert r["n_query"] == 74  # Re 24 h cohort n_unique
    assert r["no_adapt"] is True
    assert np.isnan(r["adapted_auc"])
    assert np.isfinite(r["zero_shot_auc"])
    # score dump alignment
    s = r["_scores"]
    assert len(s["y"]) == r["n_query"]
    assert len(s["p_zero"]) == r["n_query"]
    assert len(s["p_adapt"]) == 0


@_needs_db
def test_kshot_path_unaffected_by_no_adapt_flag(tmp_path):
    """Regression: the default (adapting) path still adapts and fills scores."""
    from molmetal.scripts.ex1_metal_injection_contrast import load_cohort, run_one
    import torch

    cohort = load_cohort(("Ru", "Ir", "Rh", "Os", "Re"), cohort_spec="24h")
    r = run_one(
        cohort, "Re", "none", 10, 42, epochs=1, adapt_steps=5, lr=1e-3,
        device=torch.device("cpu"), no_adapt=False, return_scores=True,
    )
    assert r["n_support"] == 10
    assert r["n_query"] == 74 - 10
    assert r["no_adapt"] is False
    assert np.isfinite(r["adapted_auc"])
    assert len(r["_scores"]["p_adapt"]) == r["n_query"]


# ---------------------------------------------------------------------------
# 5. cohort loader sizes (24 h) — documented constants
# ---------------------------------------------------------------------------
@_needs_db
@pytest.mark.parametrize("metal,n_unique", [("Ru", 1255), ("Ir", 534), ("Rh", 51), ("Os", 36), ("Re", 74)])
def test_24h_cohort_sizes(metal, n_unique):
    from molmetal.scripts.ex1_metal_injection_contrast import load_cohort

    cohort = load_cohort((metal,), cohort_spec="24h")
    assert len(cohort[metal]["smiles"]) == n_unique
    assert len(cohort[metal]["y"]) == n_unique
    assert len(cohort[metal]["os"]) == n_unique
    assert len(cohort[metal]["chg"]) == n_unique


def test_unknown_cohort_spec_raises():
    from molmetal.scripts.ex1_metal_injection_contrast import load_cohort

    with pytest.raises(ValueError):
        load_cohort(("Ru",), cohort_spec="nope")
