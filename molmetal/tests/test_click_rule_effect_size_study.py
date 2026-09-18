"""Tests for ``molmetal/scripts/click_rule_effect_size_study.py``.

Six tests cover:
1. CLI help works (parser wiring)
2. Smoke: 1 pocket x 1 seed per rule produces valid JSON output
3. Cohen's d correctness against a known-sample reference
4. All 5 click rules are iterated
5. JSON output schema contains rules / metrics / effect_sizes
6. CSV output schema has rule + metric columns

The smoke test bypasses the real MCTS budget by monkey-patching
``r4_lambda_only_run.run_one_cell`` with a deterministic stub that
returns a fixed ``CellResult``.  This keeps the test CPU-only and
under 5 s.  The real Lambda stack is NOT exercised here — that's
what the production run does.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from molmetal.scripts import click_rule_effect_size_study as ces


PROJECT_ROOT = Path("/home/hugo/codes/try_triton_on_rocm")
SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "click_rule_effect_size_study.py"
MANIFEST = PROJECT_ROOT / "molmetal" / "data" / "crossdocked100_manifest.csv"


# ---------------------------------------------------------------------------
# Fixture: deterministic CellResult stub.
# ---------------------------------------------------------------------------
class _StubCell:
    """Drop-in replacement for r4_lambda_only_run.CellResult with
    only the four metrics we test, plus the diagnostic counts."""

    def __init__(self, pocket_id, seed, rule):
        # Use rule-dependent values so the 5x4 panel is non-trivial
        # and Cohen's d can be computed (even if degenerate n=1).
        seed_offset = (seed % 5) * 0.01
        rule_offset = {
            "CuAAC":         0.30,
            "SPAAC":         0.20,
            "ThiolEne":      0.10,
            "Suzuki":        0.05,
            "AmideCoupling": 0.01,
        }.get(rule, 0.0)
        self.pocket_id = pocket_id
        self.seed = seed
        self.n_simulations = 200
        self.n_top_k = 20
        self.reference_smiles = "CCO"
        self.candidates = ["CCO", "CCN"]
        self.n_candidates = 2
        self.n_distinct = 2
        # Diversity: CuAAC > SPAAC > ThiolEne > Suzuki > AmideCoupling
        self.diversity_tanimoto = 0.05 + rule_offset + seed_offset
        # Validity: similar pattern
        self.validity_rate = 0.50 + rule_offset * 0.5 + seed_offset
        # Synthesizability: monotonically increasing
        self.synthesizability_rate = 0.40 + rule_offset * 0.4 + seed_offset
        # Metal compliance: zero when no metal-seed is passed (default)
        self.metal_compliance_rate = 0.0
        self.elapsed_s = 0.1


# ---------------------------------------------------------------------------
# 1) --help exits zero and lists the expected flags
# ---------------------------------------------------------------------------
def test_study_cli_help():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    out = (proc.stdout or "") + (proc.stderr or "")
    for needle in ("--pockets", "--seeds", "--n-simulations",
                   "--output-dir", "--manifest", "--metal-seed"):
        assert needle in out, f"--help missing flag {needle}"


# ---------------------------------------------------------------------------
# 2) 1x1 smoke per rule, JSON output is well-formed
# ---------------------------------------------------------------------------
def test_study_smoke_1x1_per_rule(tmp_path, monkeypatch):
    """Run 1 pocket x 1 seed per rule (5 cells total) and verify the
    JSON output has the expected schema."""
    # Bypass real MCTS search with the deterministic stub.
    from molmetal.scripts import r4_lambda_only_run

    def fake_run_one_cell(*, pocket_id, reference_smiles, seed,
                          n_simulations, n_top_k, click_rules=None,
                          **kwargs):
        # click_rules is a list like ['CuAAC']
        rule = click_rules[0] if click_rules else "CuAAC"
        return _StubCell(pocket_id, seed, rule)

    monkeypatch.setattr(r4_lambda_only_run, "run_one_cell", fake_run_one_cell)

    # Build a tiny 1-row manifest in tmp_path to keep the test
    # hermetic from the production manifest.
    tiny_manifest = tmp_path / "manifest.csv"
    tiny_manifest.write_text(
        "pocket_id,receptor_path,ligand_path,ref_path,metal_atoms,n_atoms,n_residues,source\n"
        "test_pocket_tiny,/tmp/r.pdb,/tmp/l.sdf,,,5,5,Smoke\n"
    )

    # Point load_reference_smiles at a deterministic stub too.
    def fake_load_reference_smiles(ligand_path):
        return "CCO"
    monkeypatch.setattr(r4_lambda_only_run, "load_reference_smiles",
                        fake_load_reference_smiles)

    output_dir = tmp_path / "study_out"
    payload = ces.run_study(
        manifest_path=tiny_manifest,
        n_pockets=1,
        seeds=[42],
        n_simulations=20,
        n_top_k=10,
        output_dir=output_dir,
    )

    # The script writes 3 files: study.json, study.csv, summary.md
    assert (output_dir / "study.json").is_file()
    assert (output_dir / "study.csv").is_file()
    assert (output_dir / "summary.md").is_file()

    # JSON schema
    with open(output_dir / "study.json") as f:
        loaded = json.load(f)
    assert set(loaded["rules"]) == set(ces.STUDY_RULES)
    assert set(loaded["metrics"]) == set(ces.STUDY_METRICS)
    # n_cells_total = 5 rules * 1 cell = 5
    assert loaded["n_cells_total"] == 5


# ---------------------------------------------------------------------------
# 3) Cohen's d correctness against a known-sample reference
# ---------------------------------------------------------------------------
def test_effect_size_cohen_d_correctness():
    """Cohen's d matches the textbook pooled-variance formula.

    Reference: Cohen 1988, eq. 2.3.5, p. 67.  ``statistics.variance``
    in Python uses the *sample* (n-1) denominator — both samples have
    mean - 2 == 1.0 and sample var = 0.7 -> pooled = 0.7
    -> d = (4.8 - 1.8) / sqrt(0.7) ~= 3.5857.

    Hand-computed example:
        treatment = [4, 5, 6, 5, 4] -> mu_t = 4.8, sample_var_t = 0.7
        baseline  = [1, 2, 3, 2, 1] -> mu_b = 1.8, sample_var_b = 0.7
        pooled = sqrt(((5-1)*0.7 + (5-1)*0.7) / (5+5-2)) = sqrt(0.7)
        d = (4.8 - 1.8) / sqrt(0.7) ~= 4.0 / 0.83666 ~= 3.5857
    """
    treatment = [4, 5, 6, 5, 4]
    baseline  = [1, 2, 3, 2, 1]
    d = ces.cohens_d(treatment, baseline)
    assert d is not None
    # 0.01 tolerance for finite-precision rounding
    assert abs(d - 3.5857) < 0.01, f"expected ~3.5857, got {d}"

    # Degenerate (n=1) -> None
    assert ces.cohens_d([1.0], [1.0, 2.0]) is None
    assert ces.cohens_d([1.0, 2.0], [1.0]) is None

    # BOTH samples zero-variance -> None (pooled = 0 => division by zero)
    assert ces.cohens_d([5.0, 5.0, 5.0], [1.0, 1.0, 1.0]) is None

    # One-sample zero-variance IS well-defined (Cohen 1988 §2.3.4):
    # d is finite when pooled_var > 0 even if one sample has s=0.
    # treatment = [5,5,5] -> mu=5, sample_var=0
    # baseline  = [1,2,3] -> mu=2, sample_var=1 (n-1 denom)
    # pooled = sqrt(((2)*0 + (2)*1) / (3+3-2)) = sqrt(2/4) = sqrt(0.5)
    # d = (5-2)/sqrt(0.5) = 3/0.7071 ~= 4.2426
    d_zv = ces.cohens_d([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
    assert d_zv is not None
    assert abs(d_zv - 4.2426) < 0.01, f"expected ~4.2426, got {d_zv}"

    # Symmetry: d(t, b) = -d(b, t)
    d_tb = ces.cohens_d(treatment, baseline)
    d_bt = ces.cohens_d(baseline, treatment)
    assert d_tb is not None and d_bt is not None
    assert abs(d_tb + d_bt) < 1e-9


# ---------------------------------------------------------------------------
# 4) STUDY_RULES iterates all 5 click rules
# ---------------------------------------------------------------------------
def test_5_rules_iteration():
    assert len(ces.STUDY_RULES) == 5
    assert set(ces.STUDY_RULES) == {
        "CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling",
    }


# ---------------------------------------------------------------------------
# 5) JSON output schema (rules, metrics, effect_sizes)
# ---------------------------------------------------------------------------
def test_output_json_schema(tmp_path, monkeypatch):
    """Run a minimal study and check JSON schema completeness."""
    from molmetal.scripts import r4_lambda_only_run

    def fake_run_one_cell(*, pocket_id, reference_smiles, seed,
                          n_simulations, n_top_k, click_rules=None,
                          **kwargs):
        rule = click_rules[0] if click_rules else "CuAAC"
        return _StubCell(pocket_id, seed, rule)

    def fake_load_reference_smiles(ligand_path):
        return "CCO"

    monkeypatch.setattr(r4_lambda_only_run, "run_one_cell", fake_run_one_cell)
    monkeypatch.setattr(r4_lambda_only_run, "load_reference_smiles",
                        fake_load_reference_smiles)

    tiny_manifest = tmp_path / "manifest.csv"
    tiny_manifest.write_text(
        "pocket_id,receptor_path,ligand_path,ref_path,metal_atoms,n_atoms,n_residues,source\n"
        "test_pocket_tiny,/tmp/r.pdb,/tmp/l.sdf,,,5,5,Smoke\n"
    )

    output_dir = tmp_path / "study_out"
    payload = ces.run_study(
        manifest_path=tiny_manifest,
        n_pockets=1,
        seeds=[42],
        n_simulations=20,
        n_top_k=10,
        output_dir=output_dir,
    )

    # Top-level keys
    for key in ("rules", "metrics", "panel_mean", "panel_std",
                "panel_n", "effect_sizes", "rule_citations",
                "config", "cells"):
        assert key in payload, f"missing top-level key: {key}"

    # rules + metrics dimension check
    assert len(payload["rules"]) == 5
    assert len(payload["metrics"]) == 4

    # panel_mean: 5 rules x 4 metrics
    for rule in ces.STUDY_RULES:
        assert rule in payload["panel_mean"]
        for metric in ces.STUDY_METRICS:
            assert metric in payload["panel_mean"][rule]
            # mean is in [0, 1] for the four metrics we care about
            v = payload["panel_mean"][rule][metric]
            assert 0.0 <= v <= 1.0, (
                f"{rule}/{metric} mean {v} out of [0,1]"
            )

    # effect_sizes: 5 rules x 4 metrics, each value is float or None
    for rule in ces.STUDY_RULES:
        assert rule in payload["effect_sizes"]
        for metric in ces.STUDY_METRICS:
            d = payload["effect_sizes"][rule][metric]
            assert d is None or isinstance(d, float)


# ---------------------------------------------------------------------------
# 6) CSV output schema (rule + metric columns)
# ---------------------------------------------------------------------------
def test_output_csv_schema(tmp_path, monkeypatch):
    """CSV must have the rule + metric + 4 numeric columns."""
    from molmetal.scripts import r4_lambda_only_run

    def fake_run_one_cell(*, pocket_id, reference_smiles, seed,
                          n_simulations, n_top_k, click_rules=None,
                          **kwargs):
        rule = click_rules[0] if click_rules else "CuAAC"
        return _StubCell(pocket_id, seed, rule)

    def fake_load_reference_smiles(ligand_path):
        return "CCO"

    monkeypatch.setattr(r4_lambda_only_run, "run_one_cell", fake_run_one_cell)
    monkeypatch.setattr(r4_lambda_only_run, "load_reference_smiles",
                        fake_load_reference_smiles)

    tiny_manifest = tmp_path / "manifest.csv"
    tiny_manifest.write_text(
        "pocket_id,receptor_path,ligand_path,ref_path,metal_atoms,n_atoms,n_residues,source\n"
        "test_pocket_tiny,/tmp/r.pdb,/tmp/l.sdf,,,5,5,Smoke\n"
    )

    output_dir = tmp_path / "study_out"
    ces.run_study(
        manifest_path=tiny_manifest,
        n_pockets=1,
        seeds=[42],
        n_simulations=20,
        n_top_k=10,
        output_dir=output_dir,
    )

    csv_path = output_dir / "study.csv"
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # CSV must have exactly 5 * 4 = 20 rows
    assert len(rows) == 5 * 4

    # Header has rule + metric columns
    assert "rule" in reader.fieldnames
    assert "metric" in reader.fieldnames
    for col in ("mean", "std", "n", "cohens_d_vs_others"):
        assert col in reader.fieldnames, f"missing column {col}"

    # Spot-check: every (rule, metric) combination appears exactly once
    seen = set()
    for r in rows:
        key = (r["rule"], r["metric"])
        assert key not in seen, f"duplicate row {key}"
        seen.add(key)
    assert seen == {(rule, metric)
                    for rule in ces.STUDY_RULES
                    for metric in ces.STUDY_METRICS}