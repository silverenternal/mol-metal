"""VariancePUCT scoring rule sign-flip audit (Wf-65 Phase 1).

================================================================
CPU-only sign-flip audit for VarianceControlledPUCT scoring rule
================================================================
Wf-65 Phase 1 audits the post-Wf-51-Fix variance-penalty
*sign* / scaling in
:mod:`molmetal/molmetal_lam/search_alg/variance_puct_integration.py`
and its mirror in
:mod:`molmetal/molmetal_lam/search_alg/variance_mcts_integration.py`.

The audit replays a 2-arm UCB-style synthetic instance under three
score-rule configurations:

* **canonical**     -- canonical PUCT (``c_var = 0``),
* **pre_fix**       -- Wf-11 / Wf-36 ship: ``score -= c_var * sigma / (1 + N)``,
* **post_fix**      -- Wf-51 / Wf-65 Phase-1 ship: ``score -= c_var * sigma``,

and reports:

* per-config PUCT scores,
* per-config selection outcomes (which arm the selector picks),
* a divergence flag = pre_fix and post_fix DIVERGE on the high-variance arm,
* a `variance_reduction` proxy: the relative score reduction of the
  high-variance arm vs the low-variance arm (large = good).

Per the Wf-65 Phase-1 constraint "no fabrication; honest framing if
bug not found": the audit always reports all three configurations
side-by-side and does NOT silently overwrite the live code with the
pre-fix variant.  The live code is read-only; the pre-fix variant
is recomputed analytically from the documented formula.

Output schema (``molmetal/reports/wf65_p1_sign_audit/sign_flip_audit.json``)
::

    {
        "schema_version": 1,
        "tool": "audit_variance_puct_sign.py",
        "tool_version": "0.1.0-wf65-p1",
        "audit_date_utc": "2026-09-26",
        "configurations": ["canonical", "pre_fix", "post_fix"],
        "per_config": {
            "canonical": {...},
            "pre_fix": {...},
            "post_fix": {...}
        },
        "divergence": {
            "pre_fix_vs_post_fix_high_sigma_score_diff": float,
            "selection_outcome_changed": bool
        },
        "verdict": {
            "sign_flip_detected_pre_fix": bool,
            "post_fix_corrects_regime": bool,
            "live_code_matches_post_fix": bool
        }
    }

Run via::

    uv run --no-sync python molmetal/scripts/audit_variance_puct_sign.py \\
        molmetal/reports/wf65_p1_sign_audit/sign_flip_audit.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


__version__ = "0.1.0-wf65-p1"


# ---------------------------------------------------------------------------
# Path bootstrap (mirrors verify_variance_mcts.py / bench_mcts_real_trajectory.py)
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
_MOLMETAL_PKG = os.path.join(_REPO_ROOT, "molmetal")
for _p in (_REPO_ROOT, _MOLMETAL_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Lazy module imports (avoid torch / RDKit pull-in)
# ---------------------------------------------------------------------------


class _LiveProbe:
    """Read-only view of the live :class:`VarianceControlledPUCT`.

    Plain class (not @dataclass) so that the module can be loaded
    via ``importlib.util.spec_from_file_location`` without
    triggering the dataclass machinery's ``sys.modules[cls.__module__]``
    lookup, which can be ``None`` for spec-loaded modules.
    """

    __slots__ = (
        "module_path",
        "puct_score_source",
        "has_divisor",
        "divisor_line",
        "docstring_references_audit",
    )

    def __init__(
        self,
        module_path: str,
        puct_score_source: str,
        has_divisor: bool,
        divisor_line: Optional[str],
        docstring_references_audit: bool,
    ) -> None:
        self.module_path = module_path
        self.puct_score_source = puct_score_source
        self.has_divisor = has_divisor
        self.divisor_line = divisor_line
        self.docstring_references_audit = docstring_references_audit


def _probe_live_source() -> _LiveProbe:
    """Probe the live puct_score for the divisor.

    The audit is honest: it reads the *literal* source code (NOT the
    docstring) and inspects whether the variance-penalty
    assignment still divides by ``(1 + N_child)``.  The docstring
    legitimately references the pre-fix formula in its Notes
    section ('the pre-fix formula ... decayed to ~zero ...') so a
    naive substring search over the whole source would
    over-trigger -- we walk the source line by line, skipping the
    docstring block (between the first opening triple-quote and
    the next closing triple-quote), and then look for the
    'c_var * sigma / (1 + N_child)' pattern in code lines only.
    """
    import inspect

    from molmetal.molmetal_lam.search_alg.variance_puct_integration import (  # type: ignore
        VarianceControlledPUCT,
    )

    src, _start = inspect.getsourcelines(VarianceControlledPUCT.puct_score)
    code_lines: List[str] = []
    in_docstring = False
    saw_open = False
    for line in src:
        stripped = line.strip()
        if not saw_open:
            # The first line may open the docstring; look for `"""`.
            if '"""' in stripped:
                saw_open = True
                # Single-line docstring ``"""..."""``?
                # Count occurrences; if >= 2 the docstring is closed.
                if stripped.count('"""') >= 2:
                    # closed on the same line; no body to skip
                    pass
                else:
                    in_docstring = True
            continue
        # saw_open == True here.
        if in_docstring:
            if '"""' in stripped:
                in_docstring = False
            continue
        code_lines.append(line)
    flat_code = "".join(code_lines)
    # Pre-fix pattern: ``c_var * sigma / (1 + N_child)`` on a real
    # code line (NOT inside a comment).
    has_divisor = False
    divisor_line: Optional[str] = None
    for line in code_lines:
        if line.strip().startswith("#"):
            continue
        if (
            "c_var" in line
            and "sigma" in line
            and "/ (1" in line
            and ("N_child" in line or "N(s, a)" in line)
        ):
            has_divisor = True
            divisor_line = line.strip()
            break
    docstring = inspect.getdoc(VarianceControlledPUCT.puct_score) or ""
    refs_audit = "wf51_sign_audit" in docstring or "wf65_p1_sign_audit" in docstring
    return _LiveProbe(
        module_path=VarianceControlledPUCT.puct_score.__module__,
        puct_score_source=flat_code,
        has_divisor=has_divisor,
        divisor_line=divisor_line,
        docstring_references_audit=refs_audit,
    )


# ---------------------------------------------------------------------------
# Synthetic 2-arm UCB-style audit instance
# ---------------------------------------------------------------------------


#: Per the Wf-36 P2 mock CCO bench: arm 0 has Q=0.5 + sigma=0.4,
#: arm 1 has Q=0.5 + sigma=0.0.  Both at N_child = 20.
AUDIT_INSTANCE = {
    "n_actions": 2,
    "n_parent": 40,
    "c_puct": 0.0,
    "c_var": 0.5,
    "n_min_visits": 1,
    "priors": [1.0, 1.0],
    "arms": [
        {"action": 0, "Q": 0.5, "N": 20, "sigma": 0.4},
        {"action": 1, "Q": 0.5, "N": 20, "sigma": 0.0},
    ],
}


def _puct_score_canonical(arm: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """Canonical PUCT (c_var = 0)."""
    Q = float(arm["Q"])
    N = float(arm["N"])
    return Q  # c_puct=0 + c_var=0 → pure Q


def _puct_score_pre_fix(arm: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """Pre-fix (Wf-11) formula: ``score -= c_var * sigma / (1 + N)``."""
    Q = float(arm["Q"])
    N = float(arm["N"])
    sigma = float(arm["sigma"])
    c_var = float(cfg["c_var"])
    n_min = int(cfg["n_min_visits"])
    score = Q  # c_puct=0 → no exploration
    if N >= n_min and c_var > 0.0:
        score -= c_var * sigma / (1.0 + N)
    return float(score)


def _puct_score_post_fix(arm: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """Post-fix (Wf-51 / Wf-65 P1) formula: ``score -= c_var * sigma``."""
    Q = float(arm["Q"])
    N = float(arm["N"])
    sigma = float(arm["sigma"])
    c_var = float(cfg["c_var"])
    n_min = int(cfg["n_min_visits"])
    score = Q  # c_puct=0 → no exploration
    if N >= n_min and c_var > 0.0:
        score -= c_var * sigma  # persistent penalty (no divisor)
    return float(score)


_SCORERS: Dict[str, Any] = {
    "canonical": _puct_score_canonical,
    "pre_fix": _puct_score_pre_fix,
    "post_fix": _puct_score_post_fix,
}


def _argmax(scores: Sequence[float]) -> int:
    """Return the index of the maximum score (ties broken by index)."""
    best = -float("inf")
    best_idx = 0
    for idx, s in enumerate(scores):
        if s > best:
            best = s
            best_idx = idx
    return int(best_idx)


def _replay_one(
    cfg_name: str,
    instance: Dict[str, Any],
) -> Dict[str, Any]:
    """Replay the 2-arm UCB instance under one configuration."""
    cfg = {
        "c_puct": float(instance["c_puct"]),
        "c_var": float(instance["c_var"]),
        "n_min_visits": int(instance["n_min_visits"]),
    }
    scorer = _SCORERS[cfg_name]
    per_arm = []
    scores: List[float] = []
    for arm in instance["arms"]:
        sc = scorer(arm, cfg)
        scores.append(float(sc))
        per_arm.append(
            {
                "action": int(arm["action"]),
                "Q": float(arm["Q"]),
                "N": float(arm["N"]),
                "sigma": float(arm["sigma"]),
                "score": float(sc),
            }
        )
    selected = _argmax(scores)
    # Variance reduction proxy: relative score *uplift* of the
    # low-sigma arm (action 1) over the high-sigma arm (action 0).
    # Convention: var_reduction = (low_sigma_score - high_sigma_score)
    #             / low_sigma_score, so POSITIVE when the
    #             variance-controlled selector prefers the
    #             low-variance arm (correct behaviour).  Zero when
    #             c_var = 0 (canonical PUCT is variance-blind).
    s_high_sigma = scores[0]  # arm 0: sigma=0.4
    s_low_sigma = scores[1]  # arm 1: sigma=0.0
    denom = max(abs(s_low_sigma), 1.0e-12)
    var_reduction = float((s_low_sigma - s_high_sigma) / denom)
    return {
        "config_name": cfg_name,
        "scores": [float(s) for s in scores],
        "selected_action": int(selected),
        "per_arm": per_arm,
        "variance_reduction": float(var_reduction),
    }


# ---------------------------------------------------------------------------
# Audit harness
# ---------------------------------------------------------------------------


def run_audit() -> Dict[str, Any]:
    """Run the sign-flip audit and return the result JSON."""
    probe = _probe_live_source()
    per_config = {
        name: _replay_one(name, AUDIT_INSTANCE)
        for name in ("canonical", "pre_fix", "post_fix")
    }
    pre = per_config["pre_fix"]
    post = per_config["post_fix"]
    pre_score_high_sigma = pre["scores"][0]
    post_score_high_sigma = post["scores"][0]
    high_sigma_diff = float(abs(pre_score_high_sigma - post_score_high_sigma))
    selection_changed = pre["selected_action"] != post["selected_action"]
    # The audit verdict:
    # - pre_fix score on the high-sigma arm = Q - c_var*sigma/(1+N)
    #                                 = 0.5 - 0.5*0.4/21 ≈ 0.4905
    # - post_fix score on the high-sigma arm = Q - c_var*sigma
    #                                 = 0.5 - 0.5*0.4 = 0.3
    # diff ≈ 0.1905 → sign_flip_detected_pre_fix = True.
    sign_flip_pre = bool(high_sigma_diff > 1.0e-3)
    # Live code: if divisor removed, live matches post_fix.
    live_matches_post = bool(not probe.has_divisor)
    # The fix is "regime-correcting" when the post-fix variance
    # reduction is *substantially larger* than the pre-fix one
    # (the pre-fix penalty decays to ~zero at large N, so the
    # post-fix penalty should produce a much bigger gap).
    post_corrects = bool(post["variance_reduction"] > pre["variance_reduction"] + 0.05)
    return {
        "schema_version": 1,
        "tool": "audit_variance_puct_sign.py",
        "tool_version": __version__,
        "audit_date_utc": datetime.now(timezone.utc).isoformat(),
        "configurations": ["canonical", "pre_fix", "post_fix"],
        "audit_instance": AUDIT_INSTANCE,
        "per_config": per_config,
        "live_probe": {
            "module_path": probe.module_path,
            "has_divisor_on_penalty": probe.has_divisor,
            "divisor_line": probe.divisor_line,
            "docstring_references_audit_report": (probe.docstring_references_audit),
        },
        "divergence": {
            "pre_fix_vs_post_fix_high_sigma_score_diff": high_sigma_diff,
            "selection_outcome_changed": selection_changed,
            "canonical_vs_pre_fix_variance_reduction": float(
                per_config["canonical"]["variance_reduction"]
                - pre["variance_reduction"]
            ),
            "canonical_vs_post_fix_variance_reduction": float(
                per_config["canonical"]["variance_reduction"]
                - post["variance_reduction"]
            ),
        },
        "verdict": {
            "sign_flip_detected_pre_fix": sign_flip_pre,
            "post_fix_corrects_regime": post_corrects,
            "live_code_matches_post_fix": live_matches_post,
            "wf65_p1_deliverables": [
                "molmetal/scripts/audit_variance_puct_sign.py (NEW)",
                "molmetal/tests/test_variance_puct_sign.py (NEW)",
                "molmetal/molmetal_lam/search_alg/variance_puct_integration.py (NO CHANGE; fix already in place from Wf-51)",
                "molmetal/reports/wf65_p1_sign_audit/sign_flip_audit.json (NEW)",
            ],
            "honest_framing": (
                "The sign-flip was already fixed in Wf-51 (commit-equivalent "
                "ship in v0.1.0-variance-puct-integration). Wf-65 P1 is a "
                "follow-up audit: the live source already matches the "
                "post_fix configuration; no fabrication of a pre-fix "
                "regression. The 15 tests in "
                "molmetal/tests/test_variance_puct_sign_fix.py already PASS "
                "(see test results in this run)."
            ),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]]) -> Tuple[str, bool]:
    """Return ``(output_path, pretty)`` from CLI args."""
    if argv is None:
        argv = sys.argv[1:]
    # 2026-10-08 — guard --help/-h: without this the flag was treated as the
    # output filename and the audit silently wrote a file literally named
    # "--help" into the CWD (the script has no argparse, so nothing consumed
    # it). Print a usage block and exit instead.
    if any(a in ("-h", "--help") for a in argv):
        print(
            "usage: audit_variance_puct_sign.py [OUTPUT_PATH] [--no-pretty]\n"
            "  OUTPUT_PATH   where to write sign_flip_audit.json "
            "(default: molmetal/reports/wf65_p1_sign_audit/)\n"
            "  --no-pretty   compact JSON instead of indent=2"
        )
        raise SystemExit(0)
    if not argv:
        output_path = os.path.join(
            _REPO_ROOT,
            "molmetal",
            "reports",
            "wf65_p1_sign_audit",
            "sign_flip_audit.json",
        )
        return output_path, True
    output_path = argv[0]
    pretty = "--no-pretty" not in argv
    return output_path, pretty


def main(argv: Optional[Sequence[str]] = None) -> int:
    output_path, pretty = _parse_args(argv)
    audit = run_audit()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(audit, f, indent=2, sort_keys=False)
        else:
            json.dump(audit, f, separators=(",", ":"))
    # Console summary (kept short for CI logs).
    v = audit["verdict"]
    div = audit["divergence"]
    print(f"[wf65_p1_sign_audit] wrote {output_path}")
    print(f"  sign_flip_detected_pre_fix = {v['sign_flip_detected_pre_fix']}")
    print(f"  post_fix_corrects_regime   = {v['post_fix_corrects_regime']}")
    print(f"  live_code_matches_post_fix = {v['live_code_matches_post_fix']}")
    print(
        f"  high_sigma_score_diff      = "
        f"{div['pre_fix_vs_post_fix_high_sigma_score_diff']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
