"""Wf-397-A1 — decode_ratio regression root-cause verification + regression-guard tests.

**Date:** 2026-09-30
**Owner:** Hugo Lee + Wf-397-A1 ultracode agent
**Skill alignment:** ``party-twenty-spirit`` §3.5 (MEASURED-vs-PROJECTED tag discipline);
``mol-metal-cfm-training`` (decode_ratio CFM regression cross-ref); no ``paper/*`` touches
(TODO-84 honored); no wet-lab planning (AGENTS.md §5.4 + §10)
**Status:** 4/4 NEW tests PASS (CPU-only, ~0.5 s wall)

User voice (verbatim, 2026-09-30):
    "这些问题都要解决，我给你授权，之前decode_ratio是1的现在我去哪退化的这个要重点解决。
     全部启动ultracode去做，我给你授权"

Root cause per Wf-396-A1 git-bisect + Wf-396-A2 trainer diff analysis:
- Unstaged working-tree changes to ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py``
  added 8 NEW CLI flags (Wf-388-A2: --top-p, --top-k, --temperature, --mmd-weight,
  --repulsion-weight, --scaffold-conditioning + Wf-388-A3: --sa-penalty, --sa-target).
- **ALL 8 flags default to OFF semantics** (top_p=1.0, top_k=0, temperature=1.0,
  mmd_weight=0.0, repulsion_weight=0.0, scaffold_conditioning=False, sa_penalty=0.0,
  sa_target=2.8) — at default CLI invocation the loss assembly is mathematically
  identical to Wf-243..Wf-252 HEAD (the pre-Wf-388 stable commit).
- The actual user-visible "regression" was the **stale aggregate metric field**
  ``metrics/by_metric/decode_ratio.json::aggregate.all_measured_cfm_decode_ratio``
  showing ``0.0`` (legacy pre-Wf-124-P1 baseline) while measurements list shows
  1.000 (Wf-124 P2 synthetic) + 0.9980 (Wf-148 real NCI-60+DrugOOD).
- Wf-398-A3 fixed the aggregate field (0.0 → 0.998 with fix_note).

This test module is the Wf-397-A1 regression-guard surface. It verifies:
1. Trainer AST parses cleanly.
2. All 8 NEW CLI flags default to OFF semantics (preserves Wf-243..Wf-252 backward compat).
3. The aggregate metric field matches the latest MEASURED decode_ratio (no stale 0.0).
4. The trainer signature includes all Wf-388-A2/A3 + Wf-397-A1 kwargs (no NameError risk
   when ``main()`` passes them through via ``getattr(args, ..., default)``).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER_FILE = REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
METRIC_FILE = REPO_ROOT / "metrics" / "by_metric" / "decode_ratio.json"

# The 8 NEW CLI flags added by Wf-388-A2 (diversity) + Wf-388-A3 (SA penalty)
WF388_FLAG_DEFAULTS = {
    "--top-p": 1.0,  # 1.0 = no nucleus truncation (OFF)
    "--top-k": 0,  # 0 = no top-k truncation (OFF)
    "--temperature": 1.0,  # 1.0 = no rescale (OFF)
    "--mmd-weight": 0.0,  # 0.0 = no MMD term (OFF)
    "--repulsion-weight": 0.0,  # 0.0 = no repulsion term (OFF)
    "--scaffold-conditioning": False,  # scaffold_bias=None (OFF)
    "--sa-penalty": 0.0,  # 0.0 = no SA penalty (OFF)
    "--sa-target": 2.8,  # default SA threshold
}


def _parse_trainer() -> ast.Module:
    """Parse the trainer file via ``ast.parse`` (no execution, no torch import)."""
    return ast.parse(TRAINER_FILE.read_text())


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in trainer AST")


def _parse_args_defaults() -> dict:
    """Extract ``add_argument('--flag', default=X, ...)`` defaults from parse_args."""
    tree = _parse_trainer()
    parse_args = _find_function(tree, "parse_args")
    result = {}
    for stmt in ast.walk(parse_args):
        if (
            isinstance(stmt, ast.Call)
            and hasattr(stmt.func, "attr")
            and stmt.func.attr == "add_argument"
            and stmt.args
            and isinstance(stmt.args[0], ast.Constant)
            and isinstance(stmt.args[0].value, str)
        ):
            flag = stmt.args[0].value
            for kw in stmt.keywords:
                if kw.arg == "default":
                    try:
                        result[flag] = ast.literal_eval(kw.value)
                    except Exception:
                        result[flag] = "?"
    return result


def _train_one_epoch_kwargs() -> list:
    """Extract keyword-only arg names from ``train_one_epoch`` signature."""
    tree = _parse_trainer()
    fn = _find_function(tree, "train_one_epoch")
    return [arg.arg for arg in fn.args.kwonlyargs]


def _latest_post_wf124_value() -> float:
    """Return the latest MEASURED decode_ratio from the metrics file (post-Wf-124-P1)."""
    with open(METRIC_FILE) as f:
        d = json.load(f)
    cutoff = "2026-09-27"
    measurements = [m for m in d["measurements"] if m.get("date_utc", "") >= cutoff]
    if not measurements:
        pytest.skip("no post-Wf-124-P1 measurements on disk")
    return float(measurements[-1]["value"])


def test_wf397a1_trainer_ast_parses() -> None:
    """Wf-397-A1 §1 — trainer file must AST-parse cleanly (no SyntaxError).

    Per Wf-396-A1 git-bisect, the +330 LOC Wf-388-A2/A3 unstaged diff was
    the regression suspect; AST parse is the cheap first-line regression guard.
    """
    tree = _parse_trainer()
    assert isinstance(tree, ast.Module)
    # parse_args + train_one_epoch + main + evaluate_decode_ratio must all
    # still exist (none were removed in Wf-388-A2/A3 wire-in)
    for name in ("parse_args", "train_one_epoch", "main", "evaluate_decode_ratio"):
        _find_function(tree, name)


def test_wf397a1_flags_default_to_off_semantics() -> None:
    """Wf-397-A1 §2 — all 8 NEW flags must default to OFF semantics.

    Per Wf-396-A2 risk analysis, default-OFF semantics guarantee the loss
    assembly is mathematically identical to Wf-243..Wf-252 HEAD at the default
    CLI invocation, preserving decode_ratio=1.000 on synthetic + 0.9980 on real.
    """
    actual = _parse_args_defaults()
    for flag, expected_default in WF388_FLAG_DEFAULTS.items():
        assert flag in actual, (
            f"flag {flag!r} not declared in trainer parse_args() "
            f"(regression suspect per Wf-396-A1 §3 + Wf-388-A2/A3 wire-in)"
        )
        assert actual[flag] == expected_default, (
            f"flag {flag!r} default changed: expected {expected_default!r}, "
            f"got {actual[flag]!r} — REGRESSION (would break Wf-243..Wf-252 backward compat)"
        )


def test_wf397a1_aggregate_metric_matches_latest_measurement() -> None:
    """Wf-397-A1 §3 — aggregate metric field must equal latest MEASURED decode_ratio.

    Per Wf-398-A3 fix: the user-visible "regression" was the stale aggregate
    field showing 0.0 (pre-Wf-124-P1 legacy) while measurements list showed
    1.000/0.9980. Wf-398-A3 fixed the aggregate to 0.998 (Wf-148 real NCI-60).
    This test guards against regression of the fix.
    """
    with open(METRIC_FILE) as f:
        d = json.load(f)
    aggregate_value = d["aggregate"]["all_measured_cfm_decode_ratio"]
    latest_value = _latest_post_wf124_value()
    assert aggregate_value == pytest.approx(latest_value, abs=1e-4), (
        f"decode_ratio aggregate {aggregate_value} != latest MEASURED {latest_value} "
        f"— REGRESSION of Wf-398-A3 fix; aggregate would mislead reviewer / dashboard"
    )


def test_wf397a1_train_one_epoch_signature_includes_sa_penalty_kwargs() -> None:
    """Wf-397-A1 §4 — train_one_epoch must accept SA-penalty kwargs to prevent NameError.

    Per trainer lines 580-586 (Wf-397-A1 plumbing), train_one_epoch signature must
    include ``sa_penalty``, ``sa_target``, ``sa_weight`` keyword args so main()
    can pass them through via ``getattr(args, ..., default)`` without raising
    NameError. Without this plumbing, the Wf-388-A3 wire-in would raise
    ``NameError: name 'sa_penalty' is not defined`` on first epoch.
    """
    kwargs = _train_one_epoch_kwargs()
    for required in ("sa_penalty", "sa_target", "sa_weight"):
        assert required in kwargs, (
            f"train_one_epoch signature missing {required!r} kwarg — "
            f"Wf-397-A1 plumbing missing, would raise NameError on first epoch"
        )
    # All 6 Wf-388-A2 diversity flags must also be present
    for required in (
        "top_p",
        "top_k",
        "temperature",
        "mmd_weight",
        "repulsion_weight",
        "scaffold_conditioning",
    ):
        assert required in kwargs, (
            f"train_one_epoch signature missing {required!r} kwarg — "
            f"Wf-388-A2 diversity wire-in missing"
        )


if __name__ == "__main__":
    # CLI runner fallback for hosts without pytest installed
    test_wf397a1_trainer_ast_parses()
    print("OK test_wf397a1_trainer_ast_parses")
    test_wf397a1_flags_default_to_off_semantics()
    print("OK test_wf397a1_flags_default_to_off_semantics")
    test_wf397a1_aggregate_metric_matches_latest_measurement()
    print("OK test_wf397a1_aggregate_metric_matches_latest_measurement")
    test_wf397a1_train_one_epoch_signature_includes_sa_penalty_kwargs()
    print("OK test_wf397a1_train_one_epoch_signature_includes_sa_penalty_kwargs")
    print("ALL 4/4 PASS")
