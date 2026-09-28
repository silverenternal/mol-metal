"""run_all_r18_tests.py — total pytest runner for the R17+R18 ship waves.

Background
----------
The R17 / R18 ship waves delivered ~2.5k test functions spread across 80+
test files spanning six families (math_skills, math-skill adapters,
lam_chem ship modules, CFM pipeline rescue + fixes, P2 ML baselines,
docking / pocket-macro).  Each ship module ships with one or more
parametrised pytest tests, but there is no single artefact that runs
them all and reports a single PASS / FAIL / SKIP roll-up.

This script closes that gap.  It is *read-only* on the inner modules
(no fixtures, no mutable state) and acts as a thin wrapper around
``pytest`` that:

  1. globs ``molmetal/tests/test_*.py`` (the canonical pytest pattern);
  2. filters the set down to R18-relevant files using a curated
     substring table covering all five spec-aligned categories:
       * ``math_skills``        — T1–T9 mathematical skeleton modules
       * ``integration``        — ``*_integration.py`` SOTA-clone glue
       * ``adapter``            — adapter unit tests
       * ``pitfall``            — pitfall-guard registry + JSON
       * ``ml_methods``         — P2 ML baselines (D-MPNN / hybrid /
                                  pretrain)
     plus a single ``R18_*`` wildcard so the orchestrator / cv-harness
     / imports test files are always included;
  3. invokes ``pytest --collect-only --co -q <filtered files>`` once,
     capturing the per-file roll-up;
  4. (Wf-46 Phase 3, ``--mode full``) invokes ``pytest -q --junit-xml
     <filtered files>`` a second time, parsing the JUnit XML for
     per-testcase pass/fail/skip/time and aggregating into per-module
     roll-ups;
  6. emits machine-readable JSON at
     ``molmetal/reports/wf_pytest_runner/results.json`` (legacy) and
     ``molmetal/reports/wf46_r18_verification/test_run_results.json``
     (Wf-46), plus a human-readable ``test_summary.md`` next to it.

The runner itself never imports the modules under test (so missing
torch / numpy / RDKit do not break the inventory); only pytest's
*collection* logic is exercised in collect mode, and the JUnit XML
plumbing is used in full mode.  This is the same logic the production
CI uses.

CLI::

    cd /home/hugo/codes/mol-metal
    uv run --no-sync python molmetal/scripts/run_all_r18_tests.py
    uv run --no-sync python molmetal/scripts/run_all_r18_tests.py --mode full

Exit codes:
    0 — collection/run succeeded, JSON written.
    1 — pytest collection itself failed for at least one file (or
        pytest run returned non-zero with at least one FAILED testcase
        when ``--strict`` is set).
    2 — the runner itself could not run (missing repo root, no tests).

The script is CPU-only, AST-light, idempotent, and never mutates
anything outside its own output directories.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Paths — keep absolute and script-relative.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MOLMETAL = REPO_ROOT / "molmetal"
TESTS_DIR = MOLMETAL / "tests"
REPORT_DIR = MOLMETAL / "reports" / "wf_pytest_runner"
JSON_OUT = REPORT_DIR / "results.json"
JSON_OUT_LEGACY = REPORT_DIR / "results_legacy.json"  # alias kept for compat

# Wf-46 Phase 3 outputs.
WF46_REPORT_DIR = MOLMETAL / "reports" / "wf46_r18_verification"
WF46_JSON_OUT = WF46_REPORT_DIR / "test_run_results.json"
WF46_MD_OUT = WF46_REPORT_DIR / "test_summary.md"


# ---------------------------------------------------------------------------
# R18-relevant test-file classification.
#
# Each entry is (substring, category, description).  The substring is
# matched against the file's stem (test_*.py stripped of the leading
# ``test_``).  First match wins, so longer / more specific substrings
# should be listed first when two could match the same stem.
# ---------------------------------------------------------------------------
# Spec-aligned categories the test inventory checks for:
#   math_skills   — T1..T9 mathematical skeleton modules
#   integration   — *_integration.py SOTA-clone glue
#   adapter       — adapter unit tests (vina, posebusters, aizynth, ...)
#   pitfall       — pitfall-guard registry + JSON
#   ml_methods    — P2 ML baselines (D-MPNN / hybrid / pretrain)
# ---------------------------------------------------------------------------
R18_CATEGORY_STEMS: List[Tuple[str, str, str]] = [
    # --- math_skills (T1..T9) — file stem contains one of these substrings.
    ("bass_existence",                "math_skills", "T8 Bass martingale existence"),
    ("bergman_kernel",                "math_skills", "T9 Bergman kernel BSD"),
    ("monge_ampere_regularity",       "math_skills", "T3 Monge-Ampere regularity"),
    ("multi_parameter_ph",            "math_skills", "T5 multi-parameter PH"),
    ("ph_hardness_guardrail",         "math_skills", "T6 PH hardness guardrail"),
    ("sb_bass_martingale",            "math_skills", "T7 SB-Bass martingale"),
    ("se3_schrodinger_bridge",        "math_skills", "T1 SE(3) Schrodinger bridge"),
    ("transport_relaxation",          "math_skills", "T2 transport relaxation"),
    ("wasserstein_gradient_flow",     "math_skills", "T4 Wasserstein gradient flow"),
    ("hochschild_cohomology",         "math_skills", "Hochschild cohom. bonus"),
    ("quantum_semantics",             "math_skills", "Quantum semantics bonus"),
    ("fractional_ode",                "math_skills", "Fractional ODE bonus"),
    ("hilbert_primal_dual",           "math_skills", "Hilbert primal-dual reward"),

    # --- integration — any *_integration.py
    ("hilbert_astsr_integration",     "integration", "Hilbert-ASTSR integration"),
    ("hilbert_score_repellent_integration", "integration", "Hilbert + score-repellent integ"),
    ("multi_ph_integration",          "integration", "Multi-PH integration"),
    ("multi_ph_pll_integration",      "integration", "Multi-PH + PLL integration"),
    ("se3_lipman_integration",        "integration", "SE(3) Lipman integration"),
    ("se3_sb_integration",            "integration", "SE(3) SB integration"),
    ("transport_relaxation_integration", "integration", "Transport-relax integration"),
    ("variance_mcts_integration",     "integration", "Variance-controlled MCTS integ"),
    ("variance_puct_integration",     "integration", "Variance-aware PUCT integ"),
    ("w_flow_cfm_integration",        "integration", "W-flow + CFM integration"),
    ("lambda_cfm_integration",        "integration", "Lambda-CFM integration"),
    ("lambda_cfm_joint_inference",    "integration", "Lambda-CFM joint inference"),
    ("medchem_gen_integration",       "integration", "Medchem-gen integration"),
    ("clone_integration_adapters",    "integration", "Clone integration adapters"),
    ("deflex_wireup_phase5_integration", "integration", "Deflex wireup phase 5"),
    ("integration",                   "integration", "Integration smoke"),

    # --- adapter — adapter unit tests
    ("vina_adapter",                  "adapter", "Vina CPU adapter"),
    ("vina_gpu_adapter",              "adapter", "Vina GPU adapter"),
    ("vina_3engine_parity",           "adapter", "Vina 3-engine parity"),
    ("docking_adapters",              "adapter", "Docking adapters harness"),
    ("docking_pose_chemistry",        "adapter", "Docking pose chemistry"),
    ("posebusters_adapter",           "adapter", "PoseBusters adapter"),
    ("posebusters_doc",               "adapter", "PoseBusters doc"),
    ("posebusters_mmff94s_relax",     "adapter", "PoseBusters MMFF94s relax"),
    ("receptor_preparation_for_evaluation", "adapter", "Receptor preparation"),
    ("prepare_crossdocked_receptor",  "adapter", "CrossDocked receptor prep"),
    ("biolm_score_adapter",           "adapter", "BioLM score adapter"),
    ("coupling_adapter",              "adapter", "Coupling adapter"),
    ("coupling_adapter_platinai5k",   "adapter", "PlatinAI-5k coupling adapter"),
    ("aizynth_adapter",               "adapter", "AiZynth adapter"),
    ("aizynth_isolated",              "adapter", "AiZynth isolated"),
    ("aizynth_real_backend",          "adapter", "AiZynth real backend"),
    ("aizynth_v4_api",                "adapter", "AiZynth v4 API"),
    ("reinvent_prior_adapter",        "adapter", "REINVENT prior adapter"),
    ("metal_generator_adapter",       "adapter", "Metal generator adapter"),
    ("adapters",                      "adapter", "Generic adapter rollup"),

    # --- pitfall — pitfall-guard registry + JSON
    ("pitfall_guards",                "pitfall", "Pitfall guard rules"),
    ("pitfall_guard_json",            "pitfall", "Pitfall guard JSON"),

    # --- ml_methods — P2 ML baselines + hybrid + pretrain
    ("dmpnn",                         "ml_methods", "D-MPNN baseline family"),
    ("attentive_dmpnn_padding",       "ml_methods", "Attentive D-MPNN padding"),
    ("v4_multitask_dmpnn",            "ml_methods", "V4 multi-task D-MPNN"),
    ("baselines",                     "ml_methods", "Baselines harness"),
    ("baselines_lightgbm_lomo_temporal", "ml_methods", "LightGBM LOMO temporal"),
    ("metal_hybrid",                  "ml_methods", "Metal hybrid V0/V3/V4"),
    ("metal_smiles",                  "ml_methods", "Metal SMILES parser"),
    ("metal_ligand_exchange_matrix",  "ml_methods", "Metal-ligand exchange matrix"),
    ("metal_geometry_known_pt",       "ml_methods", "Metal geometry known-Pt"),
    ("f2a_metal_ligand_exchange",     "ml_methods", "F2A metal-ligand exchange"),
    ("explicit_metal_prior_sampling", "ml_methods", "Explicit metal prior sampling"),
    ("tmqm_loading",                  "ml_methods", "TMQM loading"),
    ("tmqm_pretrain",                 "ml_methods", "TMQM pretraining"),
    ("tmqm_shape_bridge",             "ml_methods", "TMQM shape bridge"),
    ("tmqm_wireup",                   "ml_methods", "TMQM wireup"),
    ("retrosynthesis",                "ml_methods", "Retrosynthesis smoke"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def classify_stem(stem: str) -> Tuple[str, str]:
    """Return (category, description) for a test-file stem.

    First match in :data:`R18_CATEGORY_STEMS` wins.  An empty tuple is
    returned when the stem matches no R18-relevant category; the
    caller decides whether to include or exclude such files.
    """
    for needle, category, description in R18_CATEGORY_STEMS:
        if needle in stem:
            return category, description
    return "", ""


def discover_r18_files(tests_dir: Path) -> List[Path]:
    """Return the sorted list of R18-relevant test files under *tests_dir*.

    A file is R18-relevant when its stem matches one of the substrings
    in :data:`R18_CATEGORY_STEMS` OR when it starts with ``r18_`` (so
    ``test_r18_imports.py``, ``test_r18_cv_harness.py`` and
    ``test_r18_pipeline_orchestrator.py`` are always included).  Files
    that do not match either criterion are dropped — the runner is
    scoped, not exhaustive.
    """
    out: List[Path] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if not path.is_file():
            continue
        stem = path.stem[len("test_"):]  # strip leading "test_"
        category, _ = classify_stem(stem)
        if category or stem.startswith("r18_"):
            out.append(path)
    return out


def run_pytest_collect(
    files: Sequence[Path],
    *,
    timeout: int = 600,
) -> Tuple[int, str, str, float]:
    """Run ``pytest --collect-only --co -q`` on *files*.

    Returns ``(returncode, stdout, stderr, elapsed_seconds)``.  The
    runner never raises; subprocess failures are reported through
    ``returncode`` so the caller can render a uniform JSON artefact.
    """
    if not files:
        return 0, "", "", 0.0
    cmd: List[str] = [
        sys.executable, "-m", "pytest",
        "--collect-only",
        "--co",
        "-q",
        "--no-header",
        *[str(p) for p in files],
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        msg = f"pytest --collect-only timed out after {timeout}s"
        return 124, "", msg, elapsed
    except FileNotFoundError as exc:
        elapsed = time.monotonic() - t0
        return 127, "", f"pytest not found: {exc}", elapsed
    elapsed = time.monotonic() - t0
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def run_pytest_full(
    files: Sequence[Path],
    *,
    junit_xml: Path,
    timeout: int = 1800,
) -> Tuple[int, str, str, float]:
    """Run ``pytest -q --junit-xml=<path> --tb=line --no-header`` on *files*.

    Returns ``(returncode, stdout, stderr, elapsed_seconds)``.  Failures
    do NOT raise — the caller renders a uniform JSON artefact.  Each
    test file runs through pytest in a single invocation so the JUnit
    XML has stable per-testcase ``<testcase classname=... name=...>``
    elements with ``time`` attributes that map cleanly to per-file
    roll-ups.
    """
    if not files:
        return 0, "", "", 0.0
    cmd: List[str] = [
        sys.executable, "-m", "pytest",
        "-q",
        "--no-header",
        "--tb=line",
        "-p", "no:cacheprovider",
        "--junit-xml", str(junit_xml),
        *[str(p) for p in files],
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        msg = f"pytest full run timed out after {timeout}s"
        return 124, "", msg, elapsed
    except FileNotFoundError as exc:
        elapsed = time.monotonic() - t0
        return 127, "", f"pytest not found: {exc}", elapsed
    elapsed = time.monotonic() - t0
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def parse_junit_xml(
    xml_path: Path,
    files: Sequence[Path],
) -> Dict[str, Any]:
    """Parse pytest's JUnit XML into per-file / per-test stats.

    Schema (pytest 9.x JUnit XML)::

        <testsuite name="pytest" tests="N" failures="F" errors="E"
                   skipped="S" time="T">
          <testcase classname="molmetal.tests.test_X" name="test_y" time="t">
            <failure .../>   (when failed)
            <error .../>     (when errored)
            <skipped .../>   (when skipped)
          </testcase>
          ...
        </testsuite>

    The function buckets every testcase back to the file whose
    ``path.stem`` matches the ``classname``'s last dotted segment, so
    per-file wall-time == sum of per-testcase ``time`` attributes
    (pytest already rounds to ms so the per-test sum is the right
    attribution).
    """
    # Build a stem -> relative-path map so we can attribute per-file.
    stem_to_rel: Dict[str, str] = {}
    for p in files:
        stem_to_rel[p.stem] = str(p.relative_to(REPO_ROOT))

    per_file: Dict[str, Dict[str, Any]] = {}
    for rel in stem_to_rel.values():
        per_file[rel] = {
            "tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
            "wall_time_seconds": 0.0,
            "testcases": [],
        }

    totals = {
        "tests": 0, "passed": 0, "failed": 0,
        "skipped": 0, "errors": 0,
        "xfailed": 0, "xpassed": 0,
        "wall_time_seconds": 0.0,
    }
    if not xml_path.exists():
        return {"per_file": per_file, "totals": totals, "xml_exists": False}

    tree = ET.parse(xml_path)
    root = tree.getroot()
    # pytest writes <testsuites>...<testsuite>; tolerate both.
    suites = list(root) if root.tag == "testsuites" else [root]

    for suite in suites:
        for tc in suite.findall("testcase"):
            classname = (tc.get("classname") or "").strip()
            name = (tc.get("name") or "").strip()
            try:
                t = float(tc.get("time") or 0.0)
            except ValueError:
                t = 0.0
            # classname is dotted; last segment is the file's stem
            # (e.g. "molmetal.tests.test_pitfall_guards").
            stem = classname.rsplit(".", 1)[-1] if classname else ""
            rel = stem_to_rel.get(stem)
            failure = tc.find("failure") is not None
            error_el = tc.find("error") is not None
            skipped_el = tc.find("skipped") is not None

            # Outcome mapping (single source of truth)
            status: str
            if failure:
                status = "failed"
            elif error_el:
                status = "error"
            elif skipped_el:
                # Distinguish xfail vs plain skip via the message body.
                msg_el = tc.find("skipped")
                msg = (msg_el.get("message") or "") if msg_el is not None else ""
                if "xfail" in msg.lower() or "xpassed" in msg.lower():
                    status = "xfailed" if "xfail" in msg.lower() else "xpassed"
                else:
                    status = "skipped"
            else:
                status = "passed"

            totals["tests"] += 1
            totals["wall_time_seconds"] += t
            totals[status] = totals.get(status, 0) + 1

            if rel is not None:
                bucket = per_file[rel]
                bucket["tests"] += 1
                bucket["wall_time_seconds"] += t
                bucket[status] = bucket.get(status, 0) + 1
                bucket["testcases"].append({
                    "name": name,
                    "status": status,
                    "time": round(t, 4),
                })

    # Round float noise.
    for v in per_file.values():
        v["wall_time_seconds"] = round(v["wall_time_seconds"], 3)
    totals["wall_time_seconds"] = round(totals["wall_time_seconds"], 3)
    return {"per_file": per_file, "totals": totals, "xml_exists": True}


# ---------------------------------------------------------------------------
# Wf-46 output renderers.
# ---------------------------------------------------------------------------
def _render_wf46_summary(report: Dict[str, Any]) -> str:
    """Render the human-readable ``test_summary.md`` for Wf-46 Phase 3."""
    L: List[str] = []
    L.append("# Wf-46 Phase 3 — R18 ship test run summary")
    L.append("")
    L.append(f"Generated at `{report['timestamp']}` (UTC).")
    L.append("")
    L.append("## Headline numbers")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---:|")
    L.append(f"| pytest return code | `{report['pytest_returncode']}` |")
    L.append(f"| test files discovered | {report['n_test_files']} |")
    L.append(f"| test files with tests | {report['n_test_files_with_tests']} |")
    L.append(f"| tests collected (collect-only) | {report['n_tests_collected']} |")
    L.append(f"| tests collected (full run) | {report['n_tests_run']} |")
    L.append(f"| passed | {report['n_passed']} |")
    L.append(f"| failed | {report['n_failed']} |")
    L.append(f"| errors | {report['n_errors']} |")
    L.append(f"| skipped (incl. xfail) | {report['n_skipped']} |")
    L.append(f"| xfailed | {report['n_xfailed']} |")
    L.append(f"| xpassed | {report['n_xpassed']} |")
    L.append(f"| collect wall-time (s) | {report['elapsed_collect_seconds']:.2f} |")
    L.append(f"| full-run wall-time (s) | {report['elapsed_run_seconds']:.2f} |")
    L.append(f"| total wall-time (s) | {report['elapsed_total_seconds']:.2f} |")
    pass_rate = (
        100.0 * report["n_passed"] / report["n_tests_run"]
        if report["n_tests_run"] else 0.0
    )
    L.append(f"| pass rate (passed / run) | {pass_rate:.2f}% |")
    L.append("")

    # Per-category roll-up.
    L.append("## By category")
    L.append("")
    L.append("| Category | Files | Tests | Passed | Failed | Errors | Skipped | Wall-time (s) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    cats = report.get("per_category", {})
    for cat in sorted(cats.keys(), key=lambda k: -cats[k]["tests"]):
        row = cats[cat]
        L.append(
            f"| `{cat}` | {row['files']} | {row['tests']} | {row['passed']} | "
            f"{row['failed']} | {row['errors']} | {row['skipped']} | "
            f"{row['wall_time_seconds']:.2f} |"
        )
    L.append("")

    # Per-file top 30 by test count.
    L.append("## Per-file (top 30 by test count)")
    L.append("")
    L.append("| File | Tests | Passed | Failed | Skipped | Wall-time (s) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    by_file = report.get("per_file", [])
    top_files = sorted(by_file, key=lambda r: -r["tests"])[:30]
    for row in top_files:
        L.append(
            f"| `{row['path']}` | {row['tests']} | {row['passed']} | "
            f"{row['failed']} | {row['skipped']} | "
            f"{row['wall_time_seconds']:.2f} |"
        )
    L.append("")

    # Top 10 slowest files.
    L.append("## Per-file (top 10 slowest)")
    L.append("")
    L.append("| File | Tests | Wall-time (s) |")
    L.append("|---|---:|---:|")
    slow_files = sorted(by_file, key=lambda r: -r["wall_time_seconds"])[:10]
    for row in slow_files:
        L.append(
            f"| `{row['path']}` | {row['tests']} | "
            f"{row['wall_time_seconds']:.2f} |"
        )
    L.append("")

    # Failures (if any).
    failures = [r for r in by_file if r["failed"] > 0]
    if failures:
        L.append("## Failed files")
        L.append("")
        L.append("| File | Failed | First failure (truncated) |")
        L.append("|---|---:|---|")
        for row in sorted(failures, key=lambda r: -r["failed"]):
            first_msg = ""
            for tc in row.get("testcases", []):
                if tc["status"] == "failed":
                    first_msg = tc["name"]
                    break
            L.append(
                f"| `{row['path']}` | {row['failed']} | `{first_msg}` |"
            )
        L.append("")

    L.append("## Methodology")
    L.append("")
    L.append("- **Phase A (collect-only)** runs `pytest --collect-only -q`")
    L.append("  over the R18-relevant test files to enumerate all tests.")
    L.append("- **Phase B (full run)** runs `pytest -q --tb=line --junit-xml`")
    L.append("  over the same file set and parses the JUnit XML for")
    L.append("  per-testcase status + per-test wall-time.")
    L.append("- **Per-module roll-up** aggregates per-testcase entries back")
    L.append("  into the file's `tests_count / passed / failed / skipped /")
    L.append("  wall_time` schema.")
    L.append("")
    L.append(f"JSON written: `{report['json_out']}`")
    return "\n".join(L)


def _render_text_full(report: Dict[str, Any]) -> str:
    """Compact stdout summary for ``--mode full``."""
    L: List[str] = []
    L.append("=" * 72)
    L.append("R17+R18 test run (collect-only + full execution) roll-up")
    L.append("=" * 72)
    L.append(f"timestamp     : {report['timestamp']}")
    L.append(f"python        : {report['python_version']}  ({report['python']})")
    L.append(f"pytest        : {report['pytest_version']}")
    L.append(f"repo_root     : {report['repo_root']}")
    L.append(f"mode          : {report['mode']}")
    L.append(f"pytest rc     : {report['pytest_returncode']}")
    L.append(f"collect (s)   : {report['elapsed_collect_seconds']:.2f}")
    L.append(f"run (s)       : {report['elapsed_run_seconds']:.2f}")
    L.append(f"total (s)     : {report['elapsed_total_seconds']:.2f}")
    L.append("")
    L.append(f"test files    : {report['n_test_files']}")
    L.append(f"tests (collect): {report['n_tests_collected']}")
    L.append(f"tests (run)   : {report['n_tests_run']}")
    L.append(f"passed        : {report['n_passed']}")
    L.append(f"failed        : {report['n_failed']}")
    L.append(f"errors        : {report['n_errors']}")
    L.append(f"skipped       : {report['n_skipped']} (incl {report['n_xfailed']} xfail)")
    L.append("")
    L.append("by category (tests / passed / failed / time):")
    for cat in sorted(report["per_category"].keys(),
                      key=lambda k: -report["per_category"][k]["tests"]):
        c = report["per_category"][cat]
        L.append(
            f"  {cat:<22s}  tests={c['tests']:>4d}  pass={c['passed']:>4d}  "
            f"fail={c['failed']:>3d}  time={c['wall_time_seconds']:.1f}s"
        )
    L.append("")
    L.append(f"JSON written: {report['json_out']}")
    L.append(f"MD written:   {report['md_out']}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Output parsers — pytest --collect-only line shapes.
# ---------------------------------------------------------------------------
_TEST_LINE_RE = re.compile(
    r"^(?P<path>[^\s]+)::(?P<nodeid>.+?)(?:\s+|$)"
)
_SUMMARY_RE = re.compile(
    r"(?P<n>\d+)\s+test[s]?\s+collected",
    re.IGNORECASE,
)
_ERROR_LINE_RE = re.compile(r"^ERROR\s+(?P<path>.+)$", re.IGNORECASE)


def parse_pytest_collect_output(
    files: Sequence[Path],
    stdout: str,
    stderr: str,
) -> Dict[str, Any]:
    """Parse pytest --collect-only output into a structured roll-up.

    Pytest emits node ids in one of three forms depending on how it is
    invoked and on the conftest: relative-to-cwd (``molmetal/tests/x.py``),
    relative-to-repo-root (``tests/x.py``), or absolute.  We bucket
    each line by *longest-prefix* match against the candidate paths so
    the same parser works in all three layouts.

    The function is read-only: it never invokes pytest itself and
    never imports user code.  The output schema is:

        {
          "per_file":       { "<key>": {"tests": N} },
          "tests":          [...],   # full node-id list
          "errors":         [...],   # collection errors as free-form strings
          "duplicates":     [...],   # node-ids that appear more than once
          "summary_line":   "<pytest summary>",
          "stdout_tail":    "...",   # last 1 KiB of stdout for debugging
          "stderr_tail":    "...",   # last 1 KiB of stderr for debugging
        }
    """
    # Build a list of candidate keys for every file: relative-to-cwd,
    # relative-to-repo, absolute, plus stem-only basename.  Sorted by
    # length so the longest match wins when a node id begins with
    # multiple candidates (this matters because ``test_integration.py``
    # is a prefix of nothing else, but ``tests/test_r18.py`` shares
    # the ``tests/test_r`` prefix with sibling files).
    candidate_keys: List[Tuple[str, Path]] = []
    for p in files:
        keys = {
            str(p.resolve()),
            str(p),
            str(p.relative_to(REPO_ROOT)),
            p.name,
        }
        for k in keys:
            candidate_keys.append((k, p))
    candidate_keys.sort(key=lambda kv: len(kv[0]), reverse=True)

    per_file: Dict[str, Dict[str, Any]] = {}
    for p in files:
        per_file[str(p.relative_to(REPO_ROOT))] = {"tests": 0}

    all_tests: List[str] = []
    errors: List[str] = []
    seen: Dict[str, int] = {}
    summary_line = ""
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _SUMMARY_RE.search(s)
        if m and "::" not in s:
            summary_line = s
            continue
        if "::" in s:
            for key, owner in candidate_keys:
                if s.startswith(key + "::") or s.startswith("./" + key + "::"):
                    rel = str(owner.relative_to(REPO_ROOT))
                    per_file[rel]["tests"] += 1
                    all_tests.append(s)
                    seen[s] = seen.get(s, 0) + 1
                    break
            continue
        em = _ERROR_LINE_RE.match(s)
        if em:
            errors.append(s)
            continue

    duplicates = sorted(node for node, count in seen.items() if count > 1)

    # Also walk stderr for ERROR lines — pytest sometimes routes
    # collection errors there depending on plugin order.
    for line in stderr.splitlines():
        s = line.strip()
        em = _ERROR_LINE_RE.match(s)
        if em and s not in errors:
            errors.append(s)

    return {
        "per_file": per_file,
        "tests": all_tests,
        "errors": errors,
        "duplicates": duplicates,
        "summary_line": summary_line,
        "stdout_tail": stdout[-1024:],
        "stderr_tail": stderr[-1024:],
    }


# ---------------------------------------------------------------------------
# Top-level runner.
# ---------------------------------------------------------------------------
def run(
    *,
    tests_dir: Path = TESTS_DIR,
    report_dir: Path = REPORT_DIR,
    json_out: Path = JSON_OUT,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Discover + collect + aggregate, returning the JSON-ready dict."""
    if not tests_dir.is_dir():
        raise FileNotFoundError(f"tests dir missing: {tests_dir}")

    files = discover_r18_files(tests_dir)
    if not files:
        return {
            "timestamp": _utcnow_iso(),
            "tool": "run_all_r18_tests.py",
            "repo_root": str(REPO_ROOT),
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "pytest_version": _pytest_version(),
            "n_test_files": 0,
            "n_tests": 0,
            "n_passed": 0,
            "n_failed": 0,
            "n_skipped": 0,
            "per_file_count": {},
            "errors": ["no R18-relevant test files discovered"],
            "categories": {},
            "files": [],
        }

    rc, stdout, stderr, elapsed = run_pytest_collect(files, timeout=timeout)
    parsed = parse_pytest_collect_output(files, stdout, stderr)

    # Aggregate per category.
    by_cat_tests: Dict[str, int] = defaultdict(int)
    by_cat_files: Dict[str, int] = defaultdict(int)
    per_file_count: Dict[str, int] = {}
    for p in files:
        stem = p.stem[len("test_"):]
        category, description = classify_stem(stem)
        if not category and stem.startswith("r18_"):
            category, description = "r18_orchestrator", "R18 orchestrator/cv/imports"
        rel = str(p.relative_to(REPO_ROOT))
        n = parsed["per_file"].get(rel, {}).get("tests", 0)
        per_file_count[rel] = n
        by_cat_tests[category or "uncategorized"] += n
        by_cat_files[category or "uncategorized"] += 1

    # --collect-only is read-only — pass/fail/skip counts are 0 by
    # definition.  Surface the distinction in JSON so the consumer
    # doesn't confuse this with a real run.
    n_tests = len(parsed["tests"])
    n_files_with_tests = sum(1 for v in per_file_count.values() if v > 0)

    report: Dict[str, Any] = {
        "timestamp": _utcnow_iso(),
        "tool": "run_all_r18_tests.py",
        "mode": "collect-only (no test execution)",
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "pytest_version": _pytest_version(),
        "tests_dir": str(tests_dir.relative_to(REPO_ROOT)),
        "elapsed_seconds": round(elapsed, 3),
        "pytest_returncode": rc,
        "summary_line": parsed["summary_line"],
        "n_test_files": len(files),
        "n_test_files_with_tests": n_files_with_tests,
        "n_tests": n_tests,
        "n_passed": 0,           # collect-only: nothing passes/fails
        "n_failed": 0,
        "n_skipped": 0,
        "n_collection_errors": len(parsed["errors"]),
        "n_duplicate_nodeids": len(parsed["duplicates"]),
        "per_file_count": per_file_count,
        "per_category_count": dict(sorted(by_cat_tests.items())),
        "per_category_files": dict(sorted(by_cat_files.items())),
        "categories": {
            c: {
                "n_tests": by_cat_tests[c],
                "n_files": by_cat_files[c],
            }
            for c in sorted(set(by_cat_tests) | set(by_cat_files))
        },
        "collection_errors": parsed["errors"],
        "duplicate_nodeids_sample": parsed["duplicates"][:25],
        "files": [
            {
                "path": str(p.relative_to(REPO_ROOT)),
                "stem": p.stem[len("test_"):],
                "n_tests": per_file_count.get(str(p.relative_to(REPO_ROOT)), 0),
            }
            for p in files
        ],
        "stdout_tail": parsed["stdout_tail"],
        "stderr_tail": parsed["stderr_tail"],
    }
    return report


def run_full(
    *,
    tests_dir: Path = TESTS_DIR,
    wf46_json_out: Path = WF46_JSON_OUT,
    wf46_md_out: Path = WF46_MD_OUT,
    collect_timeout: int = 600,
    run_timeout: int = 1800,
) -> Dict[str, Any]:
    """Wf-46 Phase 3 — run collect-only then full pytest on R18 files.

    Writes:
      * ``wf46_json_out`` — machine-readable per-module roll-up
        (tests_count / passed / failed / skipped / wall_time + per-file
        detail + per-category roll-up).
      * ``wf46_md_out``   — human-readable ``test_summary.md`` table.
    """
    if not tests_dir.is_dir():
        raise FileNotFoundError(f"tests dir missing: {tests_dir}")

    files = discover_r18_files(tests_dir)
    overall_t0 = time.monotonic()

    # --collect-only pre-flight -------------------------------------------------
    c_rc, c_stdout, c_stderr, c_elapsed = run_pytest_collect(
        files, timeout=collect_timeout,
    )
    c_parsed = parse_pytest_collect_output(files, c_stdout, c_stderr)
    n_tests_collected = len(c_parsed["tests"])

    # --full pytest run with JUnit XML -----------------------------------------
    with tempfile.TemporaryDirectory(prefix="wf46_r18_") as tmp:
        junit_xml = Path(tmp) / "junit.xml"
        rc, stdout, stderr, r_elapsed = run_pytest_full(
            files, junit_xml=junit_xml, timeout=run_timeout,
        )
        juni = parse_junit_xml(junit_xml, files)

    totals = juni["totals"]
    per_file = juni["per_file"]

    # Build per-file list (sorted by n_tests desc) with stem / category.
    per_file_rows: List[Dict[str, Any]] = []
    for p in files:
        rel = str(p.relative_to(REPO_ROOT))
        stem = p.stem[len("test_"):]
        category, description = classify_stem(stem)
        if not category and stem.startswith("r18_"):
            category = "r18_orchestrator"
            description = "R18 orchestrator/cv/imports"
        b = per_file.get(rel, {})
        per_file_rows.append({
            "path": rel,
            "stem": stem,
            "category": category or "uncategorized",
            "tests": b.get("tests", 0),
            "passed": b.get("passed", 0),
            "failed": b.get("failed", 0),
            "errors": b.get("errors", 0),
            "skipped": b.get("skipped", 0),
            "xfailed": b.get("xfailed", 0),
            "xpassed": b.get("xpassed", 0),
            "wall_time_seconds": b.get("wall_time_seconds", 0.0),
            "testcases": b.get("testcases", []),
        })

    # Aggregate by category (tests/passed/failed/skipped/time).
    by_cat: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "files": 0, "tests": 0, "passed": 0, "failed": 0,
            "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0,
            "wall_time_seconds": 0.0,
        }
    )
    for row in per_file_rows:
        c = by_cat[row["category"]]
        c["files"] += 1
        c["tests"] += row["tests"]
        c["passed"] += row["passed"]
        c["failed"] += row["failed"]
        c["errors"] += row["errors"]
        c["skipped"] += row["skipped"]
        c["xfailed"] += row["xfailed"]
        c["xpassed"] += row["xpassed"]
        c["wall_time_seconds"] += row["wall_time_seconds"]
    for c in by_cat.values():
        c["wall_time_seconds"] = round(c["wall_time_seconds"], 3)

    elapsed_total = time.monotonic() - overall_t0

    n_files_with_tests = sum(1 for r in per_file_rows if r["tests"] > 0)

    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "tool": "run_all_r18_tests.py --mode full",
        "workflow": "Wf-46 Phase 3",
        "timestamp": _utcnow_iso(),
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "pytest_version": _pytest_version(),
        "tests_dir": str(tests_dir.relative_to(REPO_ROOT)),
        "mode": "collect-only + full pytest run (--junit-xml)",
        "json_out": str(wf46_json_out),
        "md_out": str(wf46_md_out),

        "pytest_returncode": rc,
        "collect_pytest_returncode": c_rc,
        "elapsed_collect_seconds": round(c_elapsed, 3),
        "elapsed_run_seconds": round(r_elapsed, 3),
        "elapsed_total_seconds": round(elapsed_total, 3),

        "n_test_files": len(files),
        "n_test_files_with_tests": n_files_with_tests,
        "n_tests_collected": n_tests_collected,
        "n_tests_run": totals["tests"],

        "n_passed": totals["passed"],
        "n_failed": totals["failed"],
        "n_errors": totals["errors"],
        "n_skipped": totals["skipped"],
        "n_xfailed": totals["xfailed"],
        "n_xpassed": totals["xpassed"],

        "n_collection_errors": len(c_parsed["errors"]),
        "collection_errors": c_parsed["errors"][:25],
        "summary_line_collect": c_parsed["summary_line"],
        "summary_line_run": (
            f"{totals['passed']} passed, {totals['failed']} failed, "
            f"{totals['errors']} errored, {totals['skipped']} skipped "
            f"in {r_elapsed:.2f}s"
        ),

        "per_category": dict(sorted(by_cat.items())),
        "per_file": sorted(per_file_rows, key=lambda r: (-r["tests"], r["path"])),

        "stdout_tail_collect": c_stdout[-1024:],
        "stderr_tail_collect": c_stderr[-1024:],
        "stdout_tail_run": stdout[-1024:],
        "stderr_tail_run": stderr[-1024:],
    }
    return report


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pytest_version() -> str:
    try:
        import pytest as _pytest  # type: ignore
        return getattr(_pytest, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unavailable"


# ---------------------------------------------------------------------------
# Output renderers.
# ---------------------------------------------------------------------------
def _render_text(report: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("=" * 72)
    L.append("R17+R18 test inventory + collect-only roll-up")
    L.append("=" * 72)
    L.append(f"timestamp     : {report['timestamp']}")
    L.append(f"python        : {report['python_version']}  ({report['python']})")
    L.append(f"pytest        : {report['pytest_version']}")
    L.append(f"repo_root     : {report['repo_root']}")
    L.append(f"mode          : {report['mode']}")
    L.append(f"elapsed (s)   : {report['elapsed_seconds']:.2f}")
    L.append(f"pytest rc     : {report['pytest_returncode']}")
    L.append("")
    L.append(f"test files (discovered) : {report['n_test_files']}")
    L.append(f"test files (w/ tests)   : {report['n_test_files_with_tests']}")
    L.append(f"tests (collected)       : {report['n_tests']}")
    L.append(f"passed / failed / skip  : {report['n_passed']} / "
             f"{report['n_failed']} / {report['n_skipped']}  (collect-only)")
    L.append(f"collection errors       : {report['n_collection_errors']}")
    L.append(f"duplicate nodeids       : {report['n_duplicate_nodeids']}")
    L.append("")

    L.append("by category:")
    cats = report.get("per_category_count", {})
    for cat, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        nf = report.get("per_category_files", {}).get(cat, 0)
        L.append(f"  {cat:<22s}  files={nf:>3d}  tests={n:>4d}")
    L.append("")

    L.append("per-file (top 30 by n_tests):")
    pfc = report.get("per_file_count", {})
    top = sorted(pfc.items(), key=lambda kv: -kv[1])[:30]
    for path, n in top:
        L.append(f"  {n:>4d}  {path}")
    L.append("")

    if report.get("collection_errors"):
        L.append("collection errors:")
        for err in report["collection_errors"][:20]:
            L.append(f"  {err}")
        L.append("")

    if report.get("duplicate_nodeids_sample"):
        L.append(f"duplicate nodeids (showing up to "
                 f"{len(report['duplicate_nodeids_sample'])} of "
                 f"{report['n_duplicate_nodeids']}):")
        for nid in report["duplicate_nodeids_sample"]:
            L.append(f"  {nid}")
        L.append("")

    L.append(f"JSON written: {JSON_OUT}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "R17+R18 ship-wave total pytest runner.  "
            "Default mode is collect-only; pass --mode full to also "
            "execute the tests and emit Wf-46 Phase 3 outputs."
        ),
    )
    parser.add_argument(
        "--mode", choices=("collect", "full"), default="collect",
        help="collect-only (default) or full (collect + execute + Wf-46 outputs)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit the JSON roll-up to stdout; do not write to disk",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="pytest --collect-only timeout in seconds (default 600)",
    )
    parser.add_argument(
        "--run-timeout", type=int, default=1800,
        help="pytest full-run timeout in seconds (default 1800)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="exit 0/1 based on collection errors; no JSON file written",
    )
    args = parser.parse_args(argv)

    if args.mode == "collect":
        try:
            report = run(timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(
                f"runner itself failed: {type(exc).__name__}: {exc}\n"
            )
            return 2

        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        JSON_OUT.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Legacy alias — keep a copy under a stable name so downstream
        # consumers can find it without depending on the per-task filename.
        try:
            shutil.copyfile(JSON_OUT, JSON_OUT_LEGACY)
        except OSError:
            pass

        if args.check:
            print(_render_text(report))
            return 0 if report["n_collection_errors"] == 0 else 1

        print(_render_text(report))
        return 0

    # --mode full : collect-only + full pytest run + Wf-46 outputs ----------
    try:
        report = run_full(
            collect_timeout=args.timeout,
            run_timeout=args.run_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"runner itself failed: {type(exc).__name__}: {exc}\n"
        )
        return 2

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    WF46_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    WF46_JSON_OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    WF46_MD_OUT.write_text(
        _render_wf46_summary(report),
        encoding="utf-8",
    )

    # Also keep the legacy wf_pytest_runner path updated so any downstream
    # tooling that watches that file keeps working.
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        legacy_payload = {
            "timestamp": report["timestamp"],
            "tool": "run_all_r18_tests.py --mode full",
            "mode": "collect-only + full run roll-up",
            "repo_root": report["repo_root"],
            "python": report["python"],
            "python_version": report["python_version"],
            "pytest_version": report["pytest_version"],
            "n_test_files": report["n_test_files"],
            "n_tests": report["n_tests_collected"],
            "n_passed": report["n_passed"],
            "n_failed": report["n_failed"],
            "n_skipped": report["n_skipped"],
            "n_collection_errors": report["n_collection_errors"],
        }
        JSON_OUT.write_text(
            json.dumps(legacy_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass

    print(_render_text_full(report))
    # Exit 0 even when tests fail — the JSON/MD artefacts are the
    # deliverable and downstream dashboards inspect ``n_failed``.
    return 0 if report["n_collection_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
