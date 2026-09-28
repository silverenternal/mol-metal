"""Tests for ``molmetal.scripts.bench_weak_real_multi_seed`` — Wf-60 Phase 1.

Wf-60 ships a single orchestrator that wraps the three R17-close-out
trainers (m1, m2, m19) and emits a per-seed ledger.  These tests assert
the contract: dry-run emits commands, aggregation rounds per-seed values
to mean/std, synthesis.md is emitted, and pass criteria match the Wf-60
gate targets (m1 ≥ 0.74, m2 ≥ 0.60, m19 ≥ 0.55).

Run via::

    cd /home/hugo/codes/mol-metal && source env.sh && molmetal/.venvs/mol-metal-venv/bin/python -m pytest \\
        molmetal/tests/test_weak_real_multi_seed.py -v

Each test is light enough to run on CPU (no GPU required) — they exercise
the wrapper structure (re-export, dry-run, aggregation, markdown) without
touching the heavy 5-seed × 50-epoch real runs.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from molmetal.scripts.bench_weak_real_multi_seed import (  # noqa: E402
    RECIPES,
    MetricRecipe,
    _aggregate,
    _parse_args,
    _run_one_metric,
    _write_synthesis_md,
)


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------
EXPECTED_METRICS = {"m1", "m2", "m19"}


def test_recipes_catalogue_has_all_three_metrics() -> None:
    """Wf-60 spec: m1 + m2 + m19 must all be present."""
    assert set(RECIPES.keys()) >= EXPECTED_METRICS


def test_recipes_have_paths_that_resolve(tmp_path: Path) -> None:
    """Each recipe's trainer_path must point to an actual .py file."""
    for recipe in RECIPES.values():
        assert recipe.trainer_path.exists(), f"{recipe.metric} -> {recipe.trainer_path} missing"
        assert recipe.trainer_path.suffix == ".py"


def test_recipes_use_strict_scaffold_and_seed_param() -> None:
    """Strict-scaffold split must be enforced for m1 + m2 (m19 is LOMO-only).

    ``train_metal_lomo.py`` does not accept a ``--scaffold-split`` flag because
    LOMO is defined by leaving one metal out at a time; the per-fold split is
    the multi-metal held-out grouping itself.  Therefore the strict-scaffold
    contract applies to m1 + m2.
    """
    for metric in ("m1", "m2"):
        assert "--scaffold-split" in RECIPES[metric].extra_args, (
            f"{metric} missing --scaffold-split in extra_args"
        )
    # The wrapper itself passes --seeds; ensure recipes do not pre-declare seeds.
    for recipe in RECIPES.values():
        assert "--seeds" not in recipe.extra_args


def test_recipes_gate_targets_match_wf60_spec() -> None:
    """Wf-60 spec § Verify: m1 ≥ 0.74, m2 ≥ 0.60, m19 ≥ 0.55."""
    assert RECIPES["m1"].gate_target == pytest.approx(0.74)
    assert RECIPES["m2"].gate_target == pytest.approx(0.60)
    assert RECIPES["m19"].gate_target == pytest.approx(0.55)


def test_recipes_baselines_match_r17_canonical() -> None:
    """R17 baselines used for delta-vs-baseline comparison."""
    assert RECIPES["m1"].gate_baseline == pytest.approx(0.7123, rel=1e-3)
    assert RECIPES["m2"].gate_baseline == pytest.approx(0.5532, rel=1e-3)
    assert RECIPES["m19"].gate_baseline == pytest.approx(0.5346, rel=1e-3)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def test_parse_args_default_metrics_all_three() -> None:
    args = _parse_args([])
    assert set(args.metrics) == EXPECTED_METRICS


def test_parse_args_custom_seeds() -> None:
    args = _parse_args(["--seeds", "7", "8", "9"])
    assert args.seeds == [7, 8, 9]


def test_parse_args_dry_run_flag() -> None:
    args = _parse_args(["--dry-run"])
    assert args.dry_run is True


def test_parse_args_rejects_unknown_metric() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--metrics", "bogus"])


def test_parse_args_cuda_visible_devices_default_is_gpu1() -> None:
    """Spec: GPU-1 RTX 5090 only — leave Blackwell sm_120 free."""
    args = _parse_args([])
    assert args.cuda_visible_devices == "1"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _make_runinfo(metric: str, rc: int = 0, values: List[float] | None = None) -> Dict[str, object]:
    out_dir = Path(f"/tmp/wf60_fake_{metric}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "value_per_seed" if metric == "m1" else (
            "value_overall_mean_auc_per_seed" if metric == "m2" else "per_run_auc"
        ): values or [],
        "value": 0.0,
    }
    json_path = out_dir / "per_seed_metrics.json"
    json_path.write_text(json.dumps(payload))
    return {
        "recipe": metric,
        "returncode": rc,
        "wall_s": 1.0,
        "log_path": str(out_dir / "log"),
        "per_seed_json": str(json_path),
        "report_md": str(out_dir / "report.md"),
        "cmd": ["echo", metric],
        "timestamp": "2026-09-26T00:00:00Z",
    }


def test_aggregate_mean_matches_statistics_fmean() -> None:
    """Aggregated mean must equal ``statistics.fmean`` for the same input."""
    runinfo = _make_runinfo("m1", values=[0.7, 0.72, 0.74, 0.71, 0.73])
    summary = _aggregate({"m1": runinfo}, {"m1": [0, 1, 2, 3, 4]}, RECIPES)
    entry = summary["metrics"]["m1"]
    expected = statistics.fmean([0.7, 0.72, 0.74, 0.71, 0.73])
    assert entry["mean_across_seeds"] == pytest.approx(expected, rel=1e-9)
    assert entry["n_seeds_observed"] == 5


def test_aggregate_gate_pass_when_target_met() -> None:
    """m1 with mean ≥ 0.74 → gate_pass True."""
    runinfo = _make_runinfo("m1", values=[0.74] * 5)
    summary = _aggregate({"m1": runinfo}, {"m1": [0, 1, 2, 3, 4]}, RECIPES)
    assert summary["metrics"]["m1"]["gate_pass"] is True


def test_aggregate_gate_fail_when_below_target() -> None:
    """m1 with mean 0.71 → gate_pass False."""
    runinfo = _make_runinfo("m1", values=[0.70, 0.71, 0.72, 0.71, 0.71])
    summary = _aggregate({"m1": runinfo}, {"m1": [0, 1, 2, 3, 4]}, RECIPES)
    assert summary["metrics"]["m1"]["gate_pass"] is False


def test_aggregate_marks_failures_with_status_field() -> None:
    """Subprocess exit-code != 0 should set status='FAILED' on entry."""
    bad = _make_runinfo("m2", rc=2)
    bad["per_seed_json"] = "/tmp/nonexistent.json"
    summary = _aggregate({"m2": bad}, {"m2": [0]}, RECIPES)
    assert summary["metrics"]["m2"].get("status") in {"FAILED", "NO_OUTPUT_JSON"}


def test_aggregate_delta_versus_r17_baseline() -> None:
    """Δ vs R17 baseline must be (mean - baseline) — not absolute."""
    runinfo = _make_runinfo("m19", values=[0.6, 0.62, 0.58, 0.61, 0.59])
    summary = _aggregate({"m19": runinfo}, {"m19": list(range(5))}, RECIPES)
    delta = summary["metrics"]["m19"]["delta_vs_r17_baseline"]
    baseline = RECIPES["m19"].gate_baseline
    mean = statistics.fmean([0.6, 0.62, 0.58, 0.61, 0.59])
    assert delta == pytest.approx(mean - baseline, rel=1e-9)


# ---------------------------------------------------------------------------
# Synthesis markdown
# ---------------------------------------------------------------------------
def test_write_synthesis_md_emits_required_sections(tmp_path: Path) -> None:
    """Synthesis markdown must contain run ledger + per-metric verdict."""
    summary = {
        "generated_at": "2026-09-26T00:00:00Z",
        "metrics": {
            "m1": {
                "label": "test",
                "n_seeds_requested": 5,
                "n_seeds_observed": 5,
                "per_seed_values": [0.74] * 5,
                "mean_across_seeds": 0.74,
                "std_across_seeds": 0.0,
                "r17_baseline": RECIPES["m1"].gate_baseline,
                "gate_target": RECIPES["m1"].gate_target,
                "gate_pass": True,
                "delta_vs_r17_baseline": 0.0277,
                "raw_payload_keys": [],
            }
        },
    }
    runinfo = _make_runinfo("m1", values=[0.74] * 5)
    md_path = _write_synthesis_md(tmp_path, summary, RECIPES, {"m1": runinfo})
    text = md_path.read_text()
    assert "Wf-60 Phase 1" in text
    assert "Run ledger" in text
    assert "Per-metric verdict" in text
    assert "Pass criteria" in text
    assert "0.74" in text  # mean
    assert "PASS" in text  # verdict


def test_synthesis_md_handles_failed_metrics_gracefully(tmp_path: Path) -> None:
    """A failed metric must render as FAILED, not crash the synthesis."""
    summary = {
        "generated_at": "2026-09-26T00:00:00Z",
        "metrics": {
            "m2": {"status": "FAILED", "returncode": 1, "log_path": "/tmp/x"}
        },
    }
    runinfo = {
        "recipe": "m2",
        "returncode": 1,
        "wall_s": 1.0,
        "log_path": "/tmp/x",
    }
    md_path = _write_synthesis_md(tmp_path, summary, RECIPES, {"m2": runinfo})
    text = md_path.read_text()
    assert "FAILED" in text


# ---------------------------------------------------------------------------
# subprocess invocation (mocked — no GPU required)
# ---------------------------------------------------------------------------
def test_dry_run_prints_commands_without_invoking(capsys: pytest.CaptureFixture[str]) -> None:
    """dry-run should build cmd strings and exit without subprocess.run."""
    fake_results: Dict[str, Dict[str, object]] = {}
    # Skip the run loop — call main with argv programmatically.
    argv = ["--metrics", "m1", "--dry-run", "--out-dir", "/tmp/wf60_dryrun"]
    # Reset sys.argv because _parse_args uses parse_args() which reads sys.argv.
    saved = sys.argv
    try:
        sys.argv = ["bench_weak_real_multi_seed.py", *argv]
        from molmetal.scripts.bench_weak_real_multi_seed import main as wf60_main
        rc = wf60_main(argv)
    finally:
        sys.argv = saved
    captured = capsys.readouterr()
    assert rc == 0
    assert "train_metal_cytotox" in captured.out


def test_run_one_metric_returns_runinfo_dict(tmp_path: Path) -> None:
    """_run_one_metric should return the runinfo dict (even if subprocess fails)."""
    recipe = RECIPES["m1"]
    # Use a syntactically valid python invocation that exits non-zero on missing args.
    # Easier: directly call with a non-existent metric to trigger SystemExit.
    with mock.patch("subprocess.run") as mock_run:
        mock_proc = mock.Mock(returncode=0)
        mock_run.return_value = mock_proc
        out_dir = tmp_path / "wf60_unit"
        # Build a fake trainer_path so the command list is well-formed
        fake_path = tmp_path / "fake.py"
        fake_path.write_text("print('hello')\n")
        recipe_overridden = MetricRecipe(
            metric=recipe.metric,
            label=recipe.label,
            trainer_module=recipe.trainer_module,
            trainer_path=fake_path,
            extra_args=recipe.extra_args,
            per_seed_pattern=recipe.per_seed_pattern,
            out_metric_relpath=recipe.out_metric_relpath,
            gate_baseline=recipe.gate_baseline,
            gate_target=recipe.gate_target,
        )
        result = _run_one_metric(
            recipe_overridden,
            seeds=[0, 1, 2],
            out_dir=out_dir,
            cuda_visible_devices="1",
            timeout_s=10,
        )
        assert isinstance(result, dict)
        assert "returncode" in result
        assert "wall_s" in result
        assert "log_path" in result
        assert "per_seed_json" in result
        assert mock_run.called


# ---------------------------------------------------------------------------
# Sanity: ensure the wrapper does not directly import torch/rdkit.
# ---------------------------------------------------------------------------
def test_wrapper_does_not_import_torch_or_rdkit() -> None:
    """The wrapper is a thin orchestrator — heavy ML imports must stay inside subprocesses."""
    # Read the script source and assert no torch / rdkit imports leak through.
    src = Path(PROJECT_ROOT / "molmetal" / "scripts" / "bench_weak_real_multi_seed.py").read_text()
    assert "import torch" not in src
    assert "from torch" not in src
    assert "from rdkit" not in src
    assert "import rdkit" not in src
