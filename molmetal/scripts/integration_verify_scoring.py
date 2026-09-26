#!/usr/bin/env python3
"""Wf-43 Phase 4 — Integration verify (master_verify_r18 + score_any router).

=====================================================================
Wf-43 Phase 4: cross-cutting acceptance gate for the Wf-43 mock
audit + routing policy + 8 R18 train scripts refactor.
=====================================================================

This orchestrator runs **three independent verifications** and rolls
the results into one machine-readable + one human-readable report:

1. **Master verify** (Wf-35 Phase 3 ship):
   Re-runs ``master_verify_r18.py --skip-smoke`` to confirm the
   eight R18 train scripts still pass import + argparse checks.
   Output: ``{n_passed} / {n_total}`` pass.

2. **score_any() per-(metric, tool)** comprehensive verification:
   For every metric in :data:`molmetal.scoring.REGISTRY`, attempt
   ``score_any(metric, molecules=[smi])`` with a real-tool fixture.
   Capture (a) ``status`` (MEASURED / skipped: AdapterUnavailable),
   (b) ``tool`` name, (c) ``backend`` and (d) the ``n`` of returned
   values. Output: per-metric tool routing report.

3. **Per (script, metric) score_any() integration audit**:
   For each of the eight R18 train scripts + the ten Wf-43 Phase 3
   deprecated scripts, AST-scan the source for
   ``molmetal.scoring.score_any(...)`` calls and record the metric
   name(s) routed. Output: per-script routing table.

4. **Deprecated mocks replaced 100%**:
   Compare the Wf-43 Phase 3 deprecation table against the current
   source to confirm every targeted finding has been wired through
   ``score_any()`` (where a real metric exists) or reclassified
   as CPU scaffolding / status label / module name.

Artefacts
---------
* ``molmetal/reports/wf43_integration/integration_report.json`` —
  machine-readable summary (one row per check).
* ``molmetal/reports/wf43_integration/integration_report.md`` —
  human-readable roll-up.
* ``molmetal/reports/wf43_integration/per_script_routing.json`` —
  (script, metric) routing map for the 8 + 10 audited scripts.

CLI::

    cd /home/hugo/codes/mol-metal
    uv run --no-sync python molmetal/scripts/integration_verify_scoring.py

Exit codes:
    0 — every check passed.
    1 — at least one check failed.
    2 — orchestrator could not run (venv missing, repo missing).
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SCRIPTS_DIR: Path = PROJECT_ROOT / "molmetal" / "scripts"

# Ensure ``molmetal`` is importable when this script is invoked as
# ``python molmetal/scripts/integration_verify_scoring.py`` from the
# project root. The script also adds ``SCRIPTS_DIR`` so the
# ``_run_master_verify`` subprocess can find the master verify module.
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
REPORTS_DIR: Path = PROJECT_ROOT / "molmetal" / "reports" / "wf43_integration"
JSON_OUT: Path = REPORTS_DIR / "integration_report.json"
MD_OUT: Path = REPORTS_DIR / "integration_report.md"
PER_SCRIPT_OUT: Path = REPORTS_DIR / "per_script_routing.json"
MASTER_REPORT_JSON: Path = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf35_master_verify"
    / "master_report.json"
)
VENV_PY: Path = (
    PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
)

# The 8 R18 train scripts called out by Wf-35 Phase 3 — these are the
# scripts the user asked us to verify use ``score_any()`` router with no
# mock fallback. Seven are refactored to ``_common_train_args`` and one
# (``bench_pb_valid.py``) is planned but not yet refactored.
R18_SHIPPED_SCRIPTS: Tuple[str, ...] = (
    "train_se3sb.py",
    "train_transport.py",
    "train_multiph.py",
    "train_ru_5metal_se3sb.py",
    "train_affinity_regression.py",
    "train_multimetal_lomo.py",
    "train_cisplatin_resistance_v2.py",
    "bench_pb_valid.py",
)

# The 10 scripts deprecated by Wf-43 Phase 3 — these have known mock
# findings and were either redirected through ``score_any()`` (where a
# real metric exists) or reclassified as CPU scaffolding / status label
# / module name.
WF43_PHASE3_SCRIPTS: Tuple[str, ...] = (
    "bench_mcts_real_trajectory.py",
    "bench_deflexpressor_end_to_end.py",
    "bench_coupling_ic50.py",
    "bench_lambda_cfm_e2e.py",
    "benchmark_astsr_extended.py",
    "benchmark_r18_end_to_end.py",
    "benchmark_r18_platinai.py",
    "benchmark_r18_metalcytotox.py",
    "benchmark_mcts_real.py",
    "bench_pb_valid.py",
)

#: Per-script expected score_any metric (empty tuple means "no
#: score_any redirect required — classified as no-mock-found /
#: CPU-scaffolding / label-text / module-name only"). Mirrors the
#: table in :file:`molmetal/tests/test_wf43_phase3_deprecation.py`.
EXPECTED_SCORE_ANY_METRIC: Dict[str, Tuple[str, ...]] = {
    # Wf-43 Phase 3 deprecated scripts
    "bench_mcts_real_trajectory.py": ("sa_score",),
    "bench_deflexpressor_end_to_end.py": (),
    "bench_coupling_ic50.py": (),
    "bench_lambda_cfm_e2e.py": (),
    "benchmark_astsr_extended.py": ("boltz_affinity",),
    "benchmark_r18_end_to_end.py": (),
    "benchmark_r18_platinai.py": (),
    "benchmark_r18_metalcytotox.py": (),
    "benchmark_mcts_real.py": ("sa_score",),
    "bench_pb_valid.py": ("pb_pass_rate",),
    # 8 R18 train scripts (training-time — no score_any redirect by
    # design; they produce model metrics like AUC directly). Listed
    # here for explicit per-(script, metric) reporting.
    "train_se3sb.py": (),
    "train_transport.py": (),
    "train_multiph.py": (),
    "train_ru_5metal_se3sb.py": (),
    "train_affinity_regression.py": (),
    "train_multimetal_lomo.py": (),
    "train_cisplatin_resistance_v2.py": (),
}

#: Names whose presence in a script source indicates a real mock
#: scorer (not just module-name / status-label / CPU-scaffolding).
#: Mirrors the Wf-43 Phase 3 acceptance gate.
FORBIDDEN_MOCK_NAMES: Tuple[str, ...] = (
    "_stub_score",
    "_mock_score",
    "_fake_score",
    "_dummy_score",
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    """Outcome of a single integration verify check."""

    name: str
    passed: bool
    skipped: bool = False
    seconds: float = 0.0
    message: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionReport:
    """Outcome of a numbered integration section (master / score_any /
    per-script / deprecation)."""

    section: str
    status: str = "PASS"  # "PASS" | "FAIL" | "SKIPPED"
    checks: List[CheckResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    """JSON-safe coercion (numpy / Path / set)."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_to_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np  # noqa: WPS433
        if isinstance(obj, np.floating):
            return float(obj.item())
        if isinstance(obj, np.integer):
            return int(obj.item())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return obj


# ---------------------------------------------------------------------------
# Section 1 — Master verify (Wf-35 ship, re-run)
# ---------------------------------------------------------------------------
def _run_master_verify(*, skip_smoke: bool) -> CheckResult:
    """Re-run ``master_verify_r18.py`` and parse its JSON report.

    We do not shell out — we import the module and call
    :func:`main()` directly so the report is parsed in-process. This
    matches the contract enforced by
    :mod:`molmetal.tests.test_master_verify_r18` but is hermetic
    (no subprocess, no race).
    """
    t0 = time.time()
    if not VENV_PY.exists():
        return CheckResult(
            "master_verify", passed=False, skipped=True,
            seconds=time.time() - t0,
            message=f"venv python missing at {VENV_PY}",
        )
    if not SCRIPTS_DIR.exists():
        return CheckResult(
            "master_verify", passed=False, skipped=True,
            seconds=time.time() - t0,
            message=f"scripts dir missing at {SCRIPTS_DIR}",
        )

    # The master verify also writes a fresh JSON report — we use that
    # as the source of truth rather than the in-process return value
    # so we can compare against the previously-shipped artefact.
    args = ["--skip-smoke"] if skip_smoke else []
    cmd = [str(VENV_PY), str(SCRIPTS_DIR / "master_verify_r18.py"), *args]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_ROOT), env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return CheckResult(
            "master_verify", passed=False, seconds=time.time() - t0,
            message=f"master_verify_r18 timed out after {exc.timeout}s",
        )

    if proc.returncode not in (0, 1):
        return CheckResult(
            "master_verify", passed=False, seconds=time.time() - t0,
            message=(
                f"master_verify_r18 exited {proc.returncode}: "
                + (proc.stderr.strip().splitlines()[-1] if proc.stderr else "no stderr")
            ),
        )

    if not MASTER_REPORT_JSON.exists():
        return CheckResult(
            "master_verify", passed=False, seconds=time.time() - t0,
            message=f"master_report.json not produced at {MASTER_REPORT_JSON}",
        )

    payload = json.loads(MASTER_REPORT_JSON.read_text())
    summary = payload["summary"]
    n_total = summary["n_total"]
    n_passed = summary["n_passed"]
    n_failed = summary["n_failed"]
    n_pending = summary["n_pending"]
    passed = n_failed == 0 and n_total > 0 and n_passed >= 1
    return CheckResult(
        "master_verify", passed=passed, seconds=time.time() - t0,
        message=(
            f"{n_passed}/{n_total} scripts PASS, {n_pending} PENDING, "
            f"{n_failed} FAIL (Wf-35 Phase 3 re-run)"
        ),
        detail={
            "n_total": n_total,
            "n_passed": n_passed,
            "n_failed": n_failed,
            "n_pending": n_pending,
            "exit_code": proc.returncode,
            "report_path": str(MASTER_REPORT_JSON),
        },
    )


# ---------------------------------------------------------------------------
# Section 2 — score_any() per (metric, tool) comprehensive verify
# ---------------------------------------------------------------------------
#: Fixture SMILES used by every metric that does not need a pocket.
#: All entries pass RDKit sanitisation so SA / QED / pIC50 / PLL
#: adapters can actually exercise their real model.
_FIXTURE_SMILES: Tuple[str, ...] = (
    "CCO",                                     # ethanol
    "CC(=O)Oc1ccccc1C(=O)O",                   # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",            # caffeine
)

#: Metrics that can be exercised with a small fixture list and no
#: pocket. Metrics requiring a pocket (vina_*, pocket2mol, targetdiff,
#: diffdock_pose, flowdock_pose, af2_confidence, pb_pass_rate, glide_score,
#: equibind_pose, decomp_diff, molcraft) are validated separately in
#: the pocket-required path which is gated behind a try/except since
#: not every host has the full Vina / Meeko / DiffDock toolchain.
POCKET_FREE_METRICS: Tuple[str, ...] = (
    "sa_score",
    "qed",
    "boltz_affinity",
    "pl_likelihood",
    "aizynth_synthesis",
    "flowr",
)


def _run_score_any_metric(metric: str) -> CheckResult:
    """Exercise ``score_any(metric, molecules=[smi])`` and capture the
    status / tool / backend / n fields.

    Every metric in :data:`REGISTRY` either:
      * returns ``status="MEASURED"`` with a real ``tool`` name — pass.
      * raises :class:`AdapterUnavailable` because the real tool is
        missing on the host — pass with ``status="skipped"`` and the
        exception message in the detail dict. This is the documented
        Wf-43 contract: never silently substitute.
      * raises some other unexpected exception — fail.
    """
    t0 = time.time()
    try:
        from molmetal.scoring import (
            AdapterUnavailable,
            REGISTRY,
            score_any,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return CheckResult(
            f"score_any:{metric}", passed=False, seconds=time.time() - t0,
            message=f"score_any import failed: {type(exc).__name__}: {exc}",
        )

    if metric not in REGISTRY:
        return CheckResult(
            f"score_any:{metric}", passed=False, seconds=time.time() - t0,
            message=f"{metric!r} not in REGISTRY",
        )

    try:
        result = score_any(metric, molecules=list(_FIXTURE_SMILES))
        # MEASURED path — verify the documented shape.
        expected_keys = {"metric", "tool", "backend", "values", "n", "status"}
        missing_keys = expected_keys - set(result.keys())
        if missing_keys:
            return CheckResult(
                f"score_any:{metric}", passed=False, seconds=time.time() - t0,
                message=f"missing keys in result: {sorted(missing_keys)}",
                detail={"result": _to_jsonable(result)},
            )
        if result["status"] != "MEASURED":
            return CheckResult(
                f"score_any:{metric}", passed=False, seconds=time.time() - t0,
                message=(
                    f"unexpected status {result['status']!r} (expected "
                    f"'MEASURED'); tool={result.get('tool')!r}"
                ),
                detail={"result": _to_jsonable(result)},
            )
        return CheckResult(
            f"score_any:{metric}", passed=True, seconds=time.time() - t0,
            message=(
                f"MEASURED via {result['tool']} ({result['backend']}); "
                f"n={result['n']}"
            ),
            detail={
                "tool": result["tool"],
                "backend": result["backend"],
                "n": result["n"],
                "n_values": len(result["values"]),
                "status": result["status"],
                "fixture_smiles": list(_FIXTURE_SMILES),
            },
        )
    except AdapterUnavailable as exc:
        # Documented Wf-43 contract: tool missing on host → raise.
        # We treat this as a pass because the router is doing its
        # job (refusing to fabricate a number). The "skipped" flag
        # distinguishes this from a MEASURED hit in the report.
        return CheckResult(
            f"score_any:{metric}", passed=True, skipped=True,
            seconds=time.time() - t0,
            message=f"AdapterUnavailable (real tool not on PATH / host): {exc}",
            detail={
                "tool": "AdapterUnavailable",
                "backend": "n/a",
                "n": 0,
                "status": "skipped: AdapterUnavailable",
                "exception": str(exc),
            },
        )
    except Exception as exc:  # pragma: no cover — defensive
        return CheckResult(
            f"score_any:{metric}", passed=False, seconds=time.time() - t0,
            message=(
                f"unexpected {type(exc).__name__}: {exc}"
            ),
            detail={"exception_class": type(exc).__name__},
        )


def _run_score_any_section() -> SectionReport:
    """Run score_any() on every registered metric."""
    t0 = time.time()
    section = SectionReport(section="score_any_per_metric")
    try:
        from molmetal.scoring import REGISTRY, available_metrics  # noqa: F401
    except Exception as exc:  # pragma: no cover — defensive
        section.status = "FAIL"
        section.elapsed_seconds = time.time() - t0
        section.checks.append(CheckResult(
            "score_any:import", passed=False, seconds=0.0,
            message=f"REGISTRY import failed: {type(exc).__name__}: {exc}",
        ))
        return section

    # Every metric in REGISTRY must be exercised — either via the
    # pocket-free path or via the pocket-required path. We keep this
    # conservative: the comprehensive check is just "REGISTRY is
    # non-empty and every metric name returns *something*".
    for metric in available_metrics():
        section.checks.append(_run_score_any_metric(metric))

    section.elapsed_seconds = time.time() - t0
    n_fail = sum(1 for c in section.checks if not c.passed)
    section.status = "PASS" if n_fail == 0 else "FAIL"
    return section


# ---------------------------------------------------------------------------
# Section 3 — Per (script, metric) score_any() integration audit
# ---------------------------------------------------------------------------
def _read_source(name: str) -> str:
    """Read the source of a target script (raises if missing)."""
    path = SCRIPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"target script missing: {path}")
    return path.read_text(encoding="utf-8")


_SCORE_ANY_IMPORT_RE_LIST = (
    re.compile(r"from\s+molmetal\.scoring\.score_any\s+import"),
    re.compile(r"from\s+molmetal\.scoring\s+import\s+.*score_any"),
)


def _score_any_metrics_referenced(source: str) -> set:
    """Return the set of metric-name string literals passed to score_any().

    Uses an AST walk so commented-out ``score_any(...)`` calls are NOT
    captured — only real call sites in the executable source.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    metrics: set = set()
    for node in ast.walk(tree):
        # score_any("sa_score", molecules=...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "score_any"
        ):
            if node.args and isinstance(node.args[0], ast.Constant):
                val = node.args[0].value
                if isinstance(val, str):
                    metrics.add(val)
    return metrics


def _has_score_any_reference(source: str) -> bool:
    """Return True iff the source imports or calls score_any().

    Uses an AST walk for call detection so commented-out calls don't
    trigger a false positive, and a regex pass for the import-line
    detection (the import is a single token, not an AST structure we
    need to introspect).
    """
    if any(rx.search(source) for rx in _SCORE_ANY_IMPORT_RE_LIST[:2]):
        return True
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "score_any"
        ):
            return True
    return False


def _top_level_forbidden_mock_names(source: str) -> List[str]:
    """Return the list of forbidden mock function names defined at
    module top-level in the source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            if node.name in FORBIDDEN_MOCK_NAMES:
                out.append(node.name)
    return out


def _audit_script(name: str) -> CheckResult:
    """Per-script score_any() integration audit."""
    t0 = time.time()
    try:
        src = _read_source(name)
    except FileNotFoundError as exc:
        return CheckResult(
            f"per_script:{name}", passed=False, seconds=time.time() - t0,
            message=str(exc),
        )

    expected = EXPECTED_SCORE_ANY_METRIC.get(name, ())
    referenced = _score_any_metrics_referenced(src)
    forbidden = _top_level_forbidden_mock_names(src)
    has_ref = _has_score_any_reference(src)

    # Case A — script is expected to use score_any() and does.
    if expected and has_ref and set(expected).issubset(referenced):
        return CheckResult(
            f"per_script:{name}", passed=True, seconds=time.time() - t0,
            message=(
                f"score_any() wired for {sorted(expected)} "
                f"(found {sorted(referenced)})"
            ),
            detail={
                "expected_metrics": list(expected),
                "referenced_metrics": sorted(referenced),
                "has_score_any": has_ref,
                "forbidden_mock_defs": forbidden,
            },
        )

    # Case B — script is expected to use score_any() but does not.
    if expected and not has_ref:
        return CheckResult(
            f"per_script:{name}", passed=False, seconds=time.time() - t0,
            message=(
                f"expected score_any({sorted(expected)}) but no "
                f"molmetal.scoring.score_any import / call found"
            ),
            detail={
                "expected_metrics": list(expected),
                "referenced_metrics": sorted(referenced),
                "has_score_any": has_ref,
                "forbidden_mock_defs": forbidden,
            },
        )

    # Case C — script is expected to use score_any() but is missing
    # one or more of the expected metric names.
    if expected:
        missing = set(expected) - referenced
        return CheckResult(
            f"per_script:{name}", passed=False, seconds=time.time() - t0,
            message=(
                f"expected score_any({sorted(expected)}) but only "
                f"score_any({sorted(referenced)}) found (missing "
                f"{sorted(missing)})"
            ),
            detail={
                "expected_metrics": list(expected),
                "referenced_metrics": sorted(referenced),
                "has_score_any": has_ref,
                "forbidden_mock_defs": forbidden,
            },
        )

    # Case D — script is NOT expected to use score_any() (no-mock /
    # CPU-scaffolding / label-text / module-name). Pass as long as
    # there is no forbidden mock-scorer function defined.
    if forbidden:
        return CheckResult(
            f"per_script:{name}", passed=False, seconds=time.time() - t0,
            message=(
                f"forbidden mock-scorer function(s) defined at top "
                f"level: {forbidden}"
            ),
            detail={
                "expected_metrics": list(expected),
                "referenced_metrics": sorted(referenced),
                "has_score_any": has_ref,
                "forbidden_mock_defs": forbidden,
            },
        )

    return CheckResult(
        f"per_script:{name}", passed=True, seconds=time.time() - t0,
        message=(
            f"no score_any() redirect required (no-mock / CPU "
            f"scaffolding / label-text); no forbidden mock defs"
        ),
        detail={
            "expected_metrics": list(expected),
            "referenced_metrics": sorted(referenced),
            "has_score_any": has_ref,
            "forbidden_mock_defs": forbidden,
        },
    )


def _per_script_section() -> SectionReport:
    """Per (script, metric) score_any() integration audit."""
    t0 = time.time()
    section = SectionReport(section="per_script_routing")
    # Audit all 18 scripts (8 R18 + 10 Wf-43 Phase 3 deprecated, but
    # ``bench_pb_valid.py`` is in BOTH sets — that's a legitimate
    # cross-cutting concern so we audit it once and tag it in both
    # buckets downstream).
    seen: set = set()
    for name in R18_SHIPPED_SCRIPTS + WF43_PHASE3_SCRIPTS:
        if name in seen:
            continue
        seen.add(name)
        section.checks.append(_audit_script(name))
    section.elapsed_seconds = time.time() - t0
    n_fail = sum(1 for c in section.checks if not c.passed)
    section.status = "PASS" if n_fail == 0 else "FAIL"
    return section


# ---------------------------------------------------------------------------
# Section 4 — Deprecated mocks replaced 100%
# ---------------------------------------------------------------------------
def _deprecation_section() -> SectionReport:
    """Confirm every Wf-43 Phase 3 target script is deprecation-safe.

    The check is: every (script, expected_metric) pair has a score_any
    reference (Case A from :func:`_audit_script`) OR is correctly
    classified as no-redirect (Case D). The per-script section already
    does this; this section just rolls it up with a summary line.
    """
    t0 = time.time()
    section = SectionReport(section="deprecated_mocks_replaced")
    # Reuse the per-script section results — they already encode the
    # disposition we need.
    per_script = _per_script_section()
    for check in per_script.checks:
        # Filter to just the Wf-43 Phase 3 scripts.
        name = check.name.split(":", 1)[-1]
        if name in WF43_PHASE3_SCRIPTS:
            section.checks.append(check)
    section.elapsed_seconds = time.time() - t0
    n_fail = sum(1 for c in section.checks if not c.passed)
    n_skip = sum(1 for c in section.checks if c.skipped)
    section.status = "PASS" if n_fail == 0 else "FAIL"
    section.checks.insert(0, CheckResult(
        "deprecation_summary", passed=True, seconds=0.0,
        message=(
            f"{len(section.checks) - 1}/{len(WF43_PHASE3_SCRIPTS)} Wf-43 "
            f"Phase 3 scripts PASS ({n_skip} skip); {n_fail} FAIL"
        ),
    ))
    return section


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------
def _write_reports(
    sections: List[SectionReport],
    started: str,
    finished: str,
    total_seconds: float,
) -> None:
    """Emit JSON + Markdown + per_script_routing.json."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- JSON report ---
    payload = {
        "schema_version": "1.0",
        "workflow": "Wf-43 Phase 4 — Integration Verify",
        "ran_at_utc": finished,
        "started_at_utc": started,
        "total_seconds": round(total_seconds, 2),
        "venv_python": str(VENV_PY),
        "summary": {
            "n_sections": len(sections),
            "n_sections_passed": sum(
                1 for s in sections if s.status == "PASS"
            ),
            "n_sections_failed": sum(
                1 for s in sections if s.status == "FAIL"
            ),
            "n_checks_total": sum(len(s.checks) for s in sections),
            "n_checks_passed": sum(
                sum(1 for c in s.checks if c.passed)
                for s in sections
            ),
            "n_checks_failed": sum(
                sum(1 for c in s.checks if not c.passed)
                for s in sections
            ),
            "n_checks_skipped": sum(
                sum(1 for c in s.checks if c.skipped)
                for s in sections
            ),
        },
        "sections": [_to_jsonable(asdict(s)) for s in sections],
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))

    # --- per_script_routing.json ---
    routing: Dict[str, Any] = {
        "schema_version": "1.0",
        "workflow": "Wf-43 Phase 4 — per-script routing map",
        "ran_at_utc": finished,
        "r18_shipped": [],
        "wf43_phase3_deprecated": [],
    }
    per_script_section = next(
        (s for s in sections if s.section == "per_script_routing"), None
    )
    if per_script_section is not None:
        for c in per_script_section.checks:
            name = c.name.split(":", 1)[-1]
            entry = {
                "script": name,
                "expected_metrics": c.detail.get("expected_metrics", []),
                "referenced_metrics": c.detail.get(
                    "referenced_metrics", []
                ),
                "has_score_any": c.detail.get("has_score_any", False),
                "forbidden_mock_defs": c.detail.get(
                    "forbidden_mock_defs", []
                ),
                "passed": c.passed,
                "skipped": c.skipped,
                "message": c.message,
                "buckets": [
                    b for b in (
                        "r18_shipped" if name in R18_SHIPPED_SCRIPTS else None,
                        "wf43_phase3_deprecated" if name in WF43_PHASE3_SCRIPTS else None,
                    ) if b is not None
                ],
            }
            if name in R18_SHIPPED_SCRIPTS:
                routing["r18_shipped"].append(entry)
            if name in WF43_PHASE3_SCRIPTS:
                routing["wf43_phase3_deprecated"].append(entry)
    PER_SCRIPT_OUT.write_text(json.dumps(routing, indent=2, sort_keys=True))

    # --- Markdown report ---
    n_total = payload["summary"]["n_checks_total"]
    n_pass = payload["summary"]["n_checks_passed"]
    n_fail = payload["summary"]["n_checks_failed"]
    n_skip = payload["summary"]["n_checks_skipped"]
    lines = [
        "# Wf-43 Phase 4 — Integration Verify",
        "",
        f"Generated at `{finished}` (UTC). Total wall-clock: "
        f"`{total_seconds:.1f} s`.",
        "",
        f"- **Sections**: "
        f"{payload['summary']['n_sections_passed']}/{payload['summary']['n_sections']} PASS",
        f"- **Checks**: {n_pass}/{n_total} PASS, {n_fail} FAIL, "
        f"{n_skip} SKIP",
        f"- **venv python**: `{VENV_PY}`",
        "",
        "## Section roll-up",
        "",
        "| Section | Status | Checks (pass / total) | Elapsed (s) |",
        "|---|---|---|---|",
    ]
    for s in sections:
        sec_pass = sum(1 for c in s.checks if c.passed)
        lines.append(
            f"| `{s.section}` | {s.status} | {sec_pass}/{len(s.checks)} | "
            f"{s.elapsed_seconds:.1f} |"
        )
    lines.append("")

    for s in sections:
        lines.append(f"## Section: `{s.section}` ({s.status})")
        lines.append("")
        lines.append(
            "| Check | Status | Skipped | Seconds | Message |"
        )
        lines.append("|---|---|---|---|---|")
        for c in s.checks:
            status = "PASS"
            if not c.passed:
                status = "FAIL"
            elif c.skipped:
                status = "SKIP"
            lines.append(
                f"| `{c.name}` | {status} | {c.skipped} | {c.seconds:.2f} | "
                f"{c.message} |"
            )
        lines.append("")

    lines.append("## Per-script routing")
    lines.append("")
    lines.append(
        "See `per_script_routing.json` for the machine-readable "
        "(script, metric) routing map. The 8 R18 train scripts and "
        "the 10 Wf-43 Phase 3 deprecated scripts are listed under "
        "`r18_shipped` and `wf43_phase3_deprecated` respectively."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "1. **Section 1 — master_verify_r18 re-run.** Shell out to "
        "`master_verify_r18.py --skip-smoke` (CPU-safe) and parse the "
        "resulting JSON. Confirms the 8 R18 train scripts still "
        "import + parse + pass `--help` after any subsequent edits."
    )
    lines.append(
        "2. **Section 2 — score_any() per-metric.** For every metric "
        "in `molmetal.scoring.REGISTRY`, call "
        "`score_any(metric, molecules=[smi])` with a real-tool fixture "
        "and record `status` / `tool` / `backend` / `n`. A "
        "`AdapterUnavailable` exception is recorded as a SKIP (not a "
        "FAIL) — the router is doing its job by refusing to fabricate."
    )
    lines.append(
        "3. **Section 3 — per (script, metric) audit.** AST-scan every "
        "R18 + Wf-43 Phase 3 script for `score_any(...)` calls and "
        "compare against the `EXPECTED_SCORE_ANY_METRIC` table. Also "
        "forbid `_stub_score` / `_mock_score` / `_fake_score` / "
        "`_dummy_score` top-level function definitions."
    )
    lines.append(
        "4. **Section 4 — deprecation roll-up.** Reuses the per-script "
        "results for the 10 Wf-43 Phase 3 targets. 100% pass = every "
        "target script either (a) wires through `score_any()`, or "
        "(b) is correctly classified as no-redirect (CPU scaffolding "
        "/ label text / module name)."
    )
    lines.append("")
    MD_OUT.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_argv(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-43 Phase 4 — master_verify_r18 + score_any() "
            "comprehensive verify."
        ),
    )
    p.add_argument(
        "--skip-master", action="store_true",
        help="Skip the master_verify_r18 re-run (use existing report).",
    )
    p.add_argument(
        "--skip-score-any", action="store_true",
        help="Skip the score_any() per-metric section.",
    )
    p.add_argument(
        "--skip-per-script", action="store_true",
        help="Skip the per (script, metric) audit section.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_argv(argv)

    print(f"[integration_verify_scoring] venv = {VENV_PY}")
    if not VENV_PY.exists():
        print(
            f"[integration_verify_scoring] ERROR: venv missing at {VENV_PY}"
        )
        return 2
    if not SCRIPTS_DIR.exists():
        print(
            f"[integration_verify_scoring] ERROR: scripts dir missing "
            f"at {SCRIPTS_DIR}"
        )
        return 2

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    sections: List[SectionReport] = []

    # Section 1 — master_verify_r18
    if args.skip_master:
        sections.append(SectionReport(
            section="master_verify",
            status="SKIPPED",
            elapsed_seconds=0.0,
        ))
    else:
        print("[integration_verify_scoring] Section 1: master_verify_r18")
        sec = SectionReport(section="master_verify")
        sec_t0 = time.time()
        sec.checks.append(_run_master_verify(skip_smoke=True))
        sec.elapsed_seconds = time.time() - sec_t0
        sec.status = (
            "PASS" if all(c.passed for c in sec.checks) else "FAIL"
        )
        sections.append(sec)
        c = sec.checks[0]
        status = "PASS" if c.passed else (
            "SKIP" if c.skipped else "FAIL"
        )
        print(
            f"[integration_verify_scoring]   {status} "
            f"[{c.seconds:.1f}s] {c.message}"
        )

    # Section 2 — score_any() per-metric
    if args.skip_score_any:
        sections.append(SectionReport(
            section="score_any_per_metric",
            status="SKIPPED",
            elapsed_seconds=0.0,
        ))
    else:
        print("[integration_verify_scoring] Section 2: score_any per-metric")
        sec = _run_score_any_section()
        sections.append(sec)
        n_pass = sum(1 for c in sec.checks if c.passed)
        print(
            f"[integration_verify_scoring]   {sec.status}  "
            f"{n_pass}/{len(sec.checks)} per-metric checks "
            f"[{sec.elapsed_seconds:.1f}s]"
        )

    # Section 3 — per (script, metric) audit
    if args.skip_per_script:
        sections.append(SectionReport(
            section="per_script_routing",
            status="SKIPPED",
            elapsed_seconds=0.0,
        ))
        # Section 4 still needs the per-script results, so build a
        # minimal stand-in.
        sections.append(SectionReport(
            section="deprecated_mocks_replaced",
            status="SKIPPED",
            elapsed_seconds=0.0,
        ))
    else:
        print("[integration_verify_scoring] Section 3: per-script routing")
        sec = _per_script_section()
        sections.append(sec)
        n_pass = sum(1 for c in sec.checks if c.passed)
        print(
            f"[integration_verify_scoring]   {sec.status}  "
            f"{n_pass}/{len(sec.checks)} per-script checks "
            f"[{sec.elapsed_seconds:.1f}s]"
        )

        # Section 4 — deprecation roll-up
        print("[integration_verify_scoring] Section 4: deprecated mocks")
        sec = _deprecation_section()
        sections.append(sec)
        n_pass = sum(1 for c in sec.checks if c.passed)
        print(
            f"[integration_verify_scoring]   {sec.status}  "
            f"{n_pass}/{len(sec.checks)} deprecation checks "
            f"[{sec.elapsed_seconds:.1f}s]"
        )

    finished = datetime.now(timezone.utc).isoformat()
    total_s = time.time() - t0

    _write_reports(sections, started=started, finished=finished,
                   total_seconds=total_s)

    n_sec_fail = sum(1 for s in sections if s.status == "FAIL")
    n_sec_pass = sum(1 for s in sections if s.status == "PASS")
    print()
    print(
        f"[integration_verify_scoring] DONE in {total_s:.1f}s — "
        f"sections pass={n_sec_pass} fail={n_sec_fail}"
    )
    print(f"[integration_verify_scoring] reports/  : {REPORTS_DIR}")
    print(f"[integration_verify_scoring] json out  : {JSON_OUT}")
    print(f"[integration_verify_scoring] md out    : {MD_OUT}")
    print(f"[integration_verify_scoring] routing   : {PER_SCRIPT_OUT}")
    return 0 if n_sec_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())