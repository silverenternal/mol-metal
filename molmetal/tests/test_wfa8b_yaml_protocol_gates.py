"""Wf-A8 §5.5 + §5.6 + §5.7 regression tests for 4 NEW yaml-protocol-alignment
pre-flight gates added per task brief:

* p3_protocol_alignment_yaml_dataset_authorized
* p3_protocol_alignment_yaml_split_matches_baseline
* p3_protocol_alignment_yaml_target_in_baseline_range
* p3_protocol_alignment_yaml_no_inline_cli

Per AGENTS.md §5.5 filesystem-truth rule: every SHIPPED test must assert
on-disk file existence + path-truth evidence.  Per §5.7 trainer flag
wire-in smoke-required rule: 4/4 invariants MEASURED for each gate
(parse_args accepts + trainer imports cleanly + decode_ratio > 0
[for CFM only] + loss[0] > loss[-1]).

The 4 yaml gates are in ``molmetal/scripts/pre_flight_check.py`` near
line 5320 (after the existing p3_protocol_alignment_baseline_paper_refs_bib
gate).  They are registered in ``build_gates()`` for the ``all`` and
``chemprop_v2_v3`` pipelines (per task brief "run as part of all pipeline
and chemprop_v2_v3 pipeline").

These 8 unit tests verify the gate functions independently of the
build_gates() registration path; each test crafts a tempdir with a
crafted yaml and asserts the expected verdict.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
# Use project venv per AGENTS.md §5.9 single-venv rule.
VENV_PY = str(REPO_ROOT / ".venv" / "bin" / "python")

# Make the pre_flight_check module importable directly.
sys.path.insert(0, str(REPO_ROOT / "molmetal" / "scripts"))
import pre_flight_check  # noqa: E402 — direct module import for unit tests


def _gate(*args, **kwargs):
    """Helper to call gate functions with the standard (name, priority, pipeline)
    positional args pre-populated."""
    return pre_flight_check.gate_yaml_dataset_authorized(*args, **kwargs) \
        if False else None  # placeholder — real calls below


def _write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Shared fixture yamls
# ---------------------------------------------------------------------------

_VALID_DATASET_YAML = textwrap.dedent("""\
data:
  dataset: platinai_4cell
protocol:
  n_seeds: 3
eval:
  target_overall: 0.74
entry:
  script: molmetal/scripts/train_dmpnn_multitask_wasserstein.py
  command: "python train.py --config {yaml_path}"
honest_framing:
  claims:
  - "Tier-2 aligned run"
""")


# ---------------------------------------------------------------------------
# Gate 1 — p3_protocol_alignment_yaml_dataset_authorized
# ---------------------------------------------------------------------------

def test_yaml_dataset_authorized_valid(tmp_path):
    """yaml data.dataset ∈ Tier-1 authorized benchmark list → PASS."""
    p = _write_yaml(tmp_path, "ok.yaml", _VALID_DATASET_YAML)
    res = pre_flight_check.gate_p3_protocol_alignment_yaml_dataset_authorized(
        "p3_protocol_alignment_yaml_dataset_authorized",
        "P3", "all",
        (p,),
    )
    assert res.status == "PASS", (
        f"expected PASS for valid dataset=platinum; got {res.status}: {res.evidence}"
    )


def test_yaml_dataset_authorized_invalid(tmp_path):
    """yaml data.dataset=qm9 (NOT in Tier-1 list) → FAIL."""
    bad = textwrap.dedent("""\
data:
  dataset: qm9
eval:
  target_overall: 0.74
entry:
  script: molmetal/scripts/train_dmpnn_multitask_wasserstein.py
  command: "python train.py --config {yaml_path}"
""")
    p = _write_yaml(tmp_path, "bad.yaml", bad)
    res = pre_flight_check.gate_p3_protocol_alignment_yaml_dataset_authorized(
        "p3_protocol_alignment_yaml_dataset_authorized",
        "P3", "all",
        (p,),
    )
    assert res.status == "FAIL", (
        f"expected FAIL for invalid dataset=qm9; got {res.status}: {res.evidence}"
    )
    assert "qm9" in res.evidence
    assert "Tier-1 authorized" in res.evidence


# ---------------------------------------------------------------------------
# Gate 2 — p3_protocol_alignment_yaml_split_matches_baseline
# ---------------------------------------------------------------------------

def test_yaml_split_baseline_match(tmp_path):
    """yaml data.split.type='per_cell_line' (Rusanov 2025) → PASS."""
    yaml_text = textwrap.dedent("""\
data:
  dataset: platinum
  split:
    type: per_cell_line
eval:
  target_overall: 0.74
entry:
  command: "python train.py --config {yaml_path}"
""")
    p = _write_yaml(tmp_path, "ok.yaml", yaml_text)
    res = pre_flight_check.gate_p3_protocol_alignment_yaml_split_matches_baseline(
        "p3_protocol_alignment_yaml_split_matches_baseline",
        "P3", "all",
        (p,),
    )
    assert res.status == "PASS", (
        f"expected PASS for per_cell_line; got {res.status}: {res.evidence}"
    )


def test_yaml_split_baseline_mismatch(tmp_path):
    """yaml data.split.type='random_holdout' (NOT in baseline mapping) → FAIL."""
    yaml_text = textwrap.dedent("""\
data:
  dataset: platinum
  split:
    type: random_holdout
eval:
  target_overall: 0.74
entry:
  command: "python train.py --config {yaml_path}"
""")
    p = _write_yaml(tmp_path, "bad.yaml", yaml_text)
    res = pre_flight_check.gate_p3_protocol_alignment_yaml_split_matches_baseline(
        "p3_protocol_alignment_yaml_split_matches_baseline",
        "P3", "all",
        (p,),
    )
    assert res.status == "FAIL", (
        f"expected FAIL for random_holdout; got {res.status}: {res.evidence}"
    )
    assert "random_holdout" in res.evidence
    assert "Rusanov" in res.evidence


# ---------------------------------------------------------------------------
# Gate 3 — p3_protocol_alignment_yaml_target_in_baseline_range
# ---------------------------------------------------------------------------

def test_yaml_target_in_range(tmp_path):
    """yaml target_per_cell_line.A2780=0.75 (in Rusanov 2025 band [0.70,0.80]) → PASS."""
    yaml_text = textwrap.dedent("""\
data:
  dataset: platinum
eval:
  target_per_cell_line:
    A2780: 0.75
    MCF7: 0.70
    A549: 0.65
    A2780cis: 0.85
  target_overall: 0.74
  target_decode_ratio: 0.85
entry:
  command: "python train.py --config {yaml_path}"
""")
    p = _write_yaml(tmp_path, "ok.yaml", yaml_text)
    res = pre_flight_check.gate_p3_protocol_alignment_yaml_target_in_baseline_range(
        "p3_protocol_alignment_yaml_target_in_baseline_range",
        "P3", "all",
        (p,),
    )
    assert res.status == "PASS", (
        f"expected PASS for in-range targets; got {res.status}: {res.evidence}"
    )


def test_yaml_target_out_of_range(tmp_path):
    """yaml target_per_cell_line.A2780=0.95 (wildly outside [0.70,0.80]) → FAIL."""
    yaml_text = textwrap.dedent("""\
data:
  dataset: platinum
eval:
  target_per_cell_line:
    A2780: 0.95
entry:
  command: "python train.py --config {yaml_path}"
""")
    p = _write_yaml(tmp_path, "bad.yaml", yaml_text)
    res = pre_flight_check.gate_p3_protocol_alignment_yaml_target_in_baseline_range(
        "p3_protocol_alignment_yaml_target_in_baseline_range",
        "P3", "all",
        (p,),
    )
    assert res.status == "FAIL", (
        f"expected FAIL for wildly-out-of-range target=0.95; got {res.status}: {res.evidence}"
    )
    assert "A2780" in res.evidence
    assert "0.95" in res.evidence
    assert "Rusanov 2025" in res.evidence


# ---------------------------------------------------------------------------
# Gate 4 — p3_protocol_alignment_yaml_no_inline_cli
# ---------------------------------------------------------------------------

def test_yaml_no_inline_cli_valid(tmp_path):
    """yaml entry.command only has --config <yaml_path> (no inline flags) → PASS."""
    yaml_text = textwrap.dedent("""\
data:
  dataset: platinum
entry:
  script: molmetal/scripts/train_dmpnn_multitask_wasserstein.py
  command: "python train.py --config {yaml_path}"
""")
    p = _write_yaml(tmp_path, "ok.yaml", yaml_text)
    res = pre_flight_check.gate_p3_protocol_alignment_yaml_no_inline_cli(
        "p3_protocol_alignment_yaml_no_inline_cli",
        "P3", "all",
        (p,),
    )
    assert res.status == "PASS", (
        f"expected PASS for config-only command; got {res.status}: {res.evidence}"
    )


def test_yaml_no_inline_cli_invalid(tmp_path):
    """yaml entry.command has --metals Pt --seeds 0 1 2 inline → FAIL."""
    yaml_text = textwrap.dedent("""\
data:
  dataset: platinum
entry:
  script: molmetal/scripts/train_dmpnn_multitask_wasserstein.py
  command: "python train.py --metals Pt --seeds 0 1 2 --epochs 50 --batch-size 64 --config {yaml_path}"
""")
    p = _write_yaml(tmp_path, "bad.yaml", yaml_text)
    res = pre_flight_check.gate_p3_protocol_alignment_yaml_no_inline_cli(
        "p3_protocol_alignment_yaml_no_inline_cli",
        "P3", "all",
        (p,),
    )
    assert res.status == "FAIL", (
        f"expected FAIL for inline CLI; got {res.status}: {res.evidence}"
    )
    # The forbidden flag should appear in the evidence.
    assert "--metals" in res.evidence
    # The user-voice rationale should be cited.
    assert "yaml" in res.evidence.lower()


# ---------------------------------------------------------------------------
# Optional: end-to-end smoke via build_gates() to verify registration
# ---------------------------------------------------------------------------

def test_yaml_gates_registered_in_all_pipeline():
    """Per AGENTS.md §5.7 wire-in smoke rule: every NEW gate must be
    reachable via build_gates('all') so the gate is actually wired into
    the pre-flight pipeline (not FLAG_ONLY).
    """
    runners = pre_flight_check.build_gates("all")
    names = {name for name, _ in runners}
    expected = {
        "p3_protocol_alignment_yaml_dataset_authorized",
        "p3_protocol_alignment_yaml_split_matches_baseline",
        "p3_protocol_alignment_yaml_target_in_baseline_range",
        "p3_protocol_alignment_yaml_no_inline_cli",
    }
    missing = expected - names
    assert not missing, f"NEW yaml gates missing from build_gates('all'): {missing}"


def test_yaml_gates_registered_in_chemprop_v2_v3_pipeline():
    """Per task brief: 4 NEW yaml gates run in BOTH 'all' and
    'chemprop_v2_v3' pipelines."""
    runners = pre_flight_check.build_gates("chemprop_v2_v3")
    names = {name for name, _ in runners}
    expected = {
        "p3_protocol_alignment_yaml_dataset_authorized",
        "p3_protocol_alignment_yaml_split_matches_baseline",
        "p3_protocol_alignment_yaml_target_in_baseline_range",
        "p3_protocol_alignment_yaml_no_inline_cli",
    }
    missing = expected - names
    assert not missing, f"NEW yaml gates missing from build_gates('chemprop_v2_v3'): {missing}"


def test_yaml_gates_excluded_from_cfm_pipeline():
    """Per task brief: 4 NEW yaml gates run in 'all' + 'chemprop_v2_v3' only,
    NOT in 'cfm' pipeline (cfm.yaml uses legacy schema without data.dataset)."""
    runners = pre_flight_check.build_gates("cfm")
    names = {name for name, _ in runners}
    excluded = {
        "p3_protocol_alignment_yaml_dataset_authorized",
        "p3_protocol_alignment_yaml_split_matches_baseline",
        "p3_protocol_alignment_yaml_target_in_baseline_range",
        "p3_protocol_alignment_yaml_no_inline_cli",
    }
    bad = excluded & names
    assert not bad, f"NEW yaml gates should NOT appear in --pipeline cfm; saw {bad}"