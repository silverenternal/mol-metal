"""Wf-A171 D3 — tests for the L5 protocol-validation hook THRESHOLD module.

CPU-only.  No GPU, no training, no metric promotion.  Run with::

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
        molmetal/tests/test_wfa171_l5_threshold.py -q

What is covered (the six areas the Wf-A171 D3 brief asks for):

1. all nine L5 hooks are covered;
2. a MISSING source is reported as MISSING, never as 0;
3. a count below the threshold is flagged (SHORTFALL);
4. a count at/above the threshold is NOT flagged (MEETS);
5. the module is importable with NO torch import;
6. the STAGED patch does not change the ``build_gates`` counts (live import
   in a subprocess, patched copy vs. the working tree).
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE_REL = "molmetal/scripts/wfa171_l5_threshold.py"
MODULE_PATH = REPO / MODULE_REL
PATCH_REL = "molmetal/reports/wfa171_d3_threshold/patch_d3_threshold.diff"
PATCH_PATH = REPO / PATCH_REL
PLAN_REL = "TODO/03_plans/PRE_TRAINING_GATE_v2.md"

EXPECTED_HOOK_IDS = [
    "krasnov_m1_per_seed_per_split_5fold_5metal",
    "krasnov_m21_per_cell_line_per_seed",
    "krasnov_protocol_alignment_diffs_documented",
    "rusanov_m2_per_cell_line_per_cohort",
    "rusanov_m2_4cell_baseline_comparison",
    "rusanov_protocol_alignment_diffs_documented",
    "drugood_m17_assay_split_auc",
    "drugood_m17_scaffold_split_auc",
    "drugood_m17_size_split_auc",
]

#: hook_id -> (plan-doc line of the Verdict-gate line, expected threshold,
#: expected threshold kind).  The line numbers are the ones quoted in
#: molmetal/reports/wfa171_d3_threshold/report.md.
EXPECTED_THRESHOLDS = {
    "krasnov_m1_per_seed_per_split_5fold_5metal": (272, 125, "record_count"),
    "krasnov_m21_per_cell_line_per_seed": (280, 45, "record_count"),
    "krasnov_protocol_alignment_diffs_documented": (288, 2, "doc_presence"),
    "rusanov_m2_per_cell_line_per_cohort": (298, 12, "record_count"),
    "rusanov_m2_4cell_baseline_comparison": (306, 1, "count_above_value"),
    "rusanov_protocol_alignment_diffs_documented": (314, 1, "doc_presence"),
    "drugood_m17_assay_split_auc": (324, -0.02, "value_gt"),
    "drugood_m17_scaffold_split_auc": (332, -0.02, "value_gt"),
    "drugood_m17_size_split_auc": (340, -0.02, "value_gt"),
}

sys.path.insert(0, str(REPO))
from molmetal.scripts import wfa171_l5_threshold as M  # noqa: E402


# ---------------------------------------------------------------- area 1 ---


def test_all_nine_hooks_are_covered():
    specs = M.default_specs()
    assert len(specs) == 9
    assert [s.hook_id for s in specs] == EXPECTED_HOOK_IDS


def test_hook_ids_appear_in_the_plan_doc():
    text = (REPO / PLAN_REL).read_text(encoding="utf-8")
    for hook_id in EXPECTED_HOOK_IDS:
        assert f"`{hook_id}`" in text, hook_id


def test_thresholds_are_parsed_from_disk_not_hardcoded():
    """The threshold must come from the plan doc's Verdict-gate line."""
    specs = {s.hook_id: s for s in M.default_specs()}
    for hook_id, (line, value, kind) in EXPECTED_THRESHOLDS.items():
        thr = specs[hook_id].threshold
        assert thr.kind == kind, hook_id
        assert thr.spec_line == line, (hook_id, thr.spec_line, line)
        assert thr.value == value, (hook_id, thr.value, value)
        # the verdict line quoted back is the doc's own line
        plan_lines = (REPO / PLAN_REL).read_text(encoding="utf-8").split("\n")
        assert plan_lines[line - 1].startswith("- **Verdict gate**:"), hook_id
        assert thr.quote in plan_lines[line - 1], hook_id


def test_record_count_thresholds_are_the_documented_ones():
    specs = {s.hook_id: s for s in M.default_specs()}
    assert specs["krasnov_m1_per_seed_per_split_5fold_5metal"].threshold.value == 125
    assert specs["krasnov_m21_per_cell_line_per_seed"].threshold.value == 45
    assert specs["rusanov_m2_per_cell_line_per_cohort"].threshold.value == 12


# ---------------------------------------------------------------- area 2 ---


def test_missing_source_is_reported_missing_not_zero(tmp_path):
    spec = M.HookSpec(
        hook_id="synthetic_missing",
        label="synthetic",
        threshold=M.Threshold(kind="record_count", value=12),
        source_path="metrics/by_metric/__definitely_not_here__.json",
        counting_key="per_cohort",
    )
    rec = M.evaluate_hook(spec, repo_root=tmp_path)
    assert rec.status == "MISSING_SOURCE"
    assert rec.measured_count is None, "a missing source must not read as 0"
    assert rec.meets_threshold is None
    assert rec.shortfall_n is None
    assert rec.source_present is False


def test_a_real_hook_whose_source_is_absent_reads_missing():
    rec = M.evaluate_hook_by_id("rusanov_m2_4cell_baseline_comparison")
    assert rec is not None
    assert rec.status == "MISSING_SOURCE"
    assert rec.measured_count is None
    assert rec.meets_threshold is None


def test_primary_source_without_the_counting_key_reads_missing_key():
    """The DrugOOD spec source exists but carries no split/delta keys."""
    rec = M.evaluate_hook_by_id("drugood_m17_assay_split_auc")
    assert rec is not None
    assert rec.primary_source_present is True
    assert rec.primary_counting_key_present is False
    assert rec.primary_status == "MISSING_KEY"
    assert rec.primary_measured_count is None


# ---------------------------------------------------------------- area 3 ---


def test_count_below_threshold_is_flagged(tmp_path):
    rel = "metrics/by_metric/synthetic.json"
    (tmp_path / "metrics" / "by_metric").mkdir(parents=True)
    (tmp_path / rel).write_text(
        json.dumps({"per_cohort": [1, 2, 3]}), encoding="utf-8"
    )
    spec = M.HookSpec(
        hook_id="synthetic_low",
        label="synthetic",
        threshold=M.Threshold(kind="record_count", value=12),
        source_path=rel,
        counting_key="per_cohort",
    )
    rec = M.evaluate_hook(spec, repo_root=tmp_path)
    assert rec.status == "SHORTFALL"
    assert rec.meets_threshold is False
    assert rec.measured_count == 3
    assert rec.shortfall_n == 9
    assert "SHORTFALL" in M.format_evidence(rec)


def test_measured_shortfalls_on_disk():
    """MEASURED record counts vs the spec thresholds (Wf-A171 D3 report)."""
    expected = {
        "krasnov_m1_per_seed_per_split_5fold_5metal": (25, 100),
        "krasnov_m21_per_cell_line_per_seed": (9, 36),
        "rusanov_m2_per_cell_line_per_cohort": (3, 9),
    }
    for hook_id, (count, shortfall) in expected.items():
        rec = M.evaluate_hook_by_id(hook_id)
        assert rec is not None
        assert rec.status == "SHORTFALL", hook_id
        assert rec.measured_count == count, (hook_id, rec.measured_count, count)
        assert rec.shortfall_n == shortfall, (hook_id, rec.shortfall_n, shortfall)
        assert rec.meets_threshold is False


# ---------------------------------------------------------------- area 4 ---


def test_count_at_threshold_is_not_flagged(tmp_path):
    rel = "metrics/by_metric/synthetic_exact.json"
    (tmp_path / "metrics" / "by_metric").mkdir(parents=True)
    (tmp_path / rel).write_text(
        json.dumps({"per_cohort": list(range(12))}), encoding="utf-8"
    )
    spec = M.HookSpec(
        hook_id="synthetic_exact",
        label="synthetic",
        threshold=M.Threshold(kind="record_count", value=12),
        source_path=rel,
        counting_key="per_cohort",
    )
    rec = M.evaluate_hook(spec, repo_root=tmp_path)
    assert rec.status == "MEETS"
    assert rec.meets_threshold is True
    assert rec.measured_count == 12
    assert rec.shortfall_n is None
    assert "SHORTFALL" not in M.format_evidence(rec)


def test_count_above_threshold_is_not_flagged(tmp_path):
    rel = "metrics/by_metric/synthetic_over.json"
    (tmp_path / "metrics" / "by_metric").mkdir(parents=True)
    (tmp_path / rel).write_text(
        json.dumps({"per_cohort": list(range(20))}), encoding="utf-8"
    )
    spec = M.HookSpec(
        hook_id="synthetic_over",
        label="synthetic",
        threshold=M.Threshold(kind="record_count", value=12),
        source_path=rel,
        counting_key="per_cohort",
    )
    rec = M.evaluate_hook(spec, repo_root=tmp_path)
    assert rec.status == "MEETS"
    assert rec.meets_threshold is True
    assert rec.measured_count == 20


def test_value_threshold_meet_and_shortfall(tmp_path):
    rel = "metrics/by_metric/synthetic_value.json"
    (tmp_path / "metrics" / "by_metric").mkdir(parents=True)
    (tmp_path / rel).write_text(
        json.dumps({"delta": -0.05}), encoding="utf-8"
    )
    thr = M.Threshold(kind="value_gt", value=-0.02)
    spec = M.HookSpec(
        hook_id="synthetic_value",
        label="synthetic",
        threshold=thr,
        source_path=rel,
        counting_key="delta",
    )
    rec = M.evaluate_hook(spec, repo_root=tmp_path)
    assert rec.status == "SHORTFALL"
    assert rec.meets_threshold is False
    assert rec.measured_value == pytest.approx(-0.05)
    assert rec.shortfall_pp == pytest.approx(3.0)


def test_four_tuple_spec_api_is_supported(tmp_path):
    """The brief's (hook_id, spec_threshold, source_path, counting_key) API."""
    rel = "metrics/by_metric/synthetic_tuple.json"
    (tmp_path / "metrics" / "by_metric").mkdir(parents=True)
    (tmp_path / rel).write_text(
        json.dumps({"per_run_auc": [1, 2, 3, 4]}), encoding="utf-8"
    )
    rec = M.evaluate_hook(
        ("synthetic_tuple", 5, rel, "per_run_auc"), repo_root=tmp_path
    )
    assert rec.status == "SHORTFALL"
    assert rec.measured_count == 4
    assert rec.shortfall_n == 1


# ---------------------------------------------------------------- area 5 ---


def test_module_is_importable_with_no_torch_import():
    """Fresh interpreter, module loaded by path: torch must not be loaded."""
    code = (
        "import importlib.util, sys\n"
        "name = 'wfa171_l5_threshold_isolated'\n"
        f"spec = importlib.util.spec_from_file_location(name, {str(MODULE_PATH)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules[name] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "print('TORCH_LOADED' if 'torch' in sys.modules else 'NO_TORCH')\n"
        "print(mod.STRICT_ENV)\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    assert "NO_TORCH" in out.stdout, out.stdout
    assert M.STRICT_ENV in out.stdout


def test_module_source_has_no_torch_import_statement():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "torch"
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "torch"


def test_strict_mode_is_opt_in_and_never_produces_fail():
    assert M.strict_enabled({}) is False
    assert M.strict_enabled({"WFA171_L5_STRICT": "0"}) is False
    assert M.strict_enabled({"WFA171_L5_STRICT": "1"}) is True
    # The staged patch's strict branch must WARN, never FAIL.
    patch = PATCH_PATH.read_text(encoding="utf-8")
    assert '"WARN",' in patch
    strict_hunk = patch[patch.index("strict mode") - 1200 : patch.index("strict mode") + 900]
    assert '"FAIL"' not in strict_hunk


# ---------------------------------------------------------------- area 6 ---


def _build_farm(tmp_path: Path) -> Path:
    """A symlink farm whose ``molmetal/scripts/pre_flight_check.py`` is the
    patched copy.  The farm resolves to the real repo contents, so the
    patched module sees the same data files."""
    farm = tmp_path / "farm"
    farm.mkdir()
    for entry in REPO.iterdir():
        if entry.name in (".git", "molmetal"):
            continue
        os.symlink(entry, farm / entry.name)
    mm = farm / "molmetal"
    mm.mkdir()
    for entry in (REPO / "molmetal").iterdir():
        if entry.name == "scripts":
            continue
        os.symlink(entry, mm / entry.name)
    sc = mm / "scripts"
    sc.mkdir()
    for entry in (REPO / "molmetal" / "scripts").iterdir():
        if entry.name == "pre_flight_check.py":
            continue
        os.symlink(entry, sc / entry.name)
    shutil.copy2(
        REPO / "molmetal" / "scripts" / "pre_flight_check.py",
        sc / "pre_flight_check.py",
    )
    return farm


def _gate_counts(cwd: Path) -> dict:
    code = (
        "import sys, json; sys.path.insert(0, '.')\n"
        "import molmetal.scripts.pre_flight_check as m\n"
        "print(json.dumps({p: len(m.build_gates(p)) for p in "
        "('all', 'frontier', 'chemprop_v2_v3', 'cfm')}))\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    line = [x for x in out.stdout.split("\n") if x.strip().startswith("{")][-1]
    return json.loads(line)


def test_staged_patch_applies_cleanly_to_a_scratch_copy(tmp_path):
    scratch = tmp_path / "scratch"
    (scratch / "molmetal" / "scripts").mkdir(parents=True)
    shutil.copy2(
        REPO / "molmetal" / "scripts" / "pre_flight_check.py",
        scratch / "molmetal" / "scripts" / "pre_flight_check.py",
    )
    res = subprocess.run(
        ["git", "apply", "--check", "-p1", str(PATCH_PATH)],
        cwd=str(scratch),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)


def test_staged_patch_does_not_change_build_gates_counts(tmp_path):
    farm = _build_farm(tmp_path)
    applied = subprocess.run(
        ["git", "apply", "-p1", str(PATCH_PATH)],
        cwd=str(farm),
        capture_output=True,
        text=True,
    )
    assert applied.returncode == 0, (applied.stdout, applied.stderr)

    patched_counts = _gate_counts(farm)
    live_counts = _gate_counts(REPO)
    assert patched_counts == live_counts, (patched_counts, live_counts)
    # and the counts still match the module's own declared expectations
    from molmetal.scripts import pre_flight_check as live  # noqa: PLC0415

    assert patched_counts["all"] == live.EXPECTED_GATE_COUNTS["all"]
    assert patched_counts["frontier"] == live.EXPECTED_GATE_COUNTS["frontier"]


def test_staged_patch_does_not_add_or_remove_gates():
    """The patch touches no ``runners +=`` site, so the gate NAMES are equal."""
    original = (REPO / "molmetal" / "scripts" / "pre_flight_check.py").read_text(
        encoding="utf-8"
    )
    patch = PATCH_PATH.read_text(encoding="utf-8")
    added = [
        ln[1:]
        for ln in patch.split("\n")
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    removed = [
        ln[1:]
        for ln in patch.split("\n")
        if ln.startswith("-") and not ln.startswith("---")
    ]
    assert not any("runners +=" in ln for ln in added)
    assert not any("runners +=" in ln for ln in removed)
    assert not any("EXPECTED_GATE_COUNTS" in ln for ln in added)
    assert not any("EXPECTED_GATE_COUNTS" in ln for ln in removed)
    # every removed line is re-added with one suffix appended
    for line in removed:
        assert line in original


# ------------------------------------------------------------ extra guards ---


def test_evaluate_hooks_returns_nine_records():
    recs = M.evaluate_hooks()
    assert len(recs) == 9
    assert [r.hook_id for r in recs] == EXPECTED_HOOK_IDS


def test_every_record_carries_the_required_fields():
    for rec in M.evaluate_hooks():
        assert rec.threshold is not None
        assert rec.source_present in (True, False)
        assert rec.status in (
            "MEETS",
            "SHORTFALL",
            "MISSING_SOURCE",
            "MISSING_KEY",
        )
        if rec.status == "SHORTFALL":
            assert rec.meets_threshold is False
            assert rec.shortfall_n is not None or rec.shortfall_pp is not None
        if rec.status == "MEETS":
            assert rec.meets_threshold is True


def test_no_metric_file_is_written_by_this_module():
    """The module is read-only: it must not touch metrics/ or registry/."""
    before = {
        p: p.stat().st_mtime_ns
        for p in (REPO / "metrics" / "by_metric").iterdir()
        if p.is_file()
    }
    M.evaluate_hooks()
    after = {
        p: p.stat().st_mtime_ns
        for p in (REPO / "metrics" / "by_metric").iterdir()
        if p.is_file()
    }
    assert before == after
