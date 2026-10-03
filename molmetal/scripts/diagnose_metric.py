"""D1 — Auto-diagnostic for metric regressions.

Given a metric ID (``mN``, e.g. ``m14``), parse
``TODO/MECHANISM_METRIC_MAPPING.md`` §4 (failure-mode table), extract
the list of mechanisms to check, derive per-mechanism pytest
invocations, run them sequentially, and recommend the most likely
root cause + suggested fix.

Honest framing
==============

* The mapping table is hand-curated in ``MECHANISM_METRIC_MAPPING.md``;
  this script only *parses* it and *runs pytest*.  It does NOT mutate
  any source file.
* The pitfall-to-pytest mapping is hard-coded below (see
  ``MECHANISM_PYTEST``) and aligned with ``test_pitfall_guards.py`` /
  ``wf_pitfall_guard/check_all.py``.  Adding a new mechanism requires
  updating both the markdown table AND this dictionary.

Usage
=====

::

    cd /home/hugo/codes/mol-metal
    source env.sh
    python -u molmetal/scripts/diagnose_metric.py --metric m14

    # Run only the symptom's first failure-mode row (default: all rows
    # that mention the metric ID)
    python -u molmetal/scripts/diagnose_metric.py --metric m14 \\
        --symptom "m14 decode_ratio = 0"

    # JSON output for CI consumption
    python -u molmetal/scripts/diagnose_metric.py --metric m14 --json

Exit code is 0 when every mechanism pytest PASSes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
# The mapping table moved under TODO/02_metrics/ during the A2
# subdirectory reorg.  Search both locations so the script remains
# robust to whichever path the maintainers settle on.
MAPPING_MD_CANDIDATES = (
    REPO_ROOT / "TODO" / "MECHANISM_METRIC_MAPPING.md",
    REPO_ROOT / "TODO" / "02_metrics" / "MECHANISM_METRIC_MAPPING.md",
)
TEST_DIR = REPO_ROOT / "molmetal" / "tests"


def _resolve_mapping_md() -> Path:
    for cand in MAPPING_MD_CANDIDATES:
        if cand.is_file():
            return cand
    return MAPPING_MD_CANDIDATES[0]  # for the FATAL message below

# ---------------------------------------------------------------------------
# Static mechanism → pytest mapping.
#
# Each entry: (pytest keyword selector, suggested fix file:line, brief).
# The selectors are matched against the test file via ``pytest -k``.
# When a mechanism has no pitfall-guard test (e.g. M5 / M6 EGNN / CFM),
# we point at the canonical test file path with a broad -k that
# pytest will still execute.
# ---------------------------------------------------------------------------

MECHANISM_PYTEST: Dict[str, Tuple[str, str, str]] = {
    # ID     pytest -k selector              suggested fix path                       brief
    "M1":  ("synflownet or atom_training",     "molmetal/molmetal_lam/lam_chem/lambda_combinators.py", "Molecular Lambda Calculus (atoms-as-combinators)"),
    "M2":  ("metal_ligand_exchange or pt_click_compat", "molmetal/molmetal_lam/lam_chem/pt_click_compat.py", "β-reduction rules (84 + 18 + 4 NEW SMARTS)"),
    "M3":  ("proof_search or mcts",            "molmetal/molmetal_lam/search_alg/proof_search.py",     "KataGo-MCTS proof search"),
    "M4":  ("pll_signature or pll_heuristic",  "molmetal/molmetal_lam/lam_chem/pll_signature.py",      "PLL heuristic (Persistent Local Laplacian)"),
    "M5":  ("egnn_velocity or egnn_coord",     "molmetal/adapters/flow_matching_lipman/__init__.py",   "SE(3)-equivariant EGNN"),
    "M6":  ("cfm_decoder or cfm_rescue",       "molmetal/adapters/flow_matching_lipman/__init__.py",   "Lipman 2023 CFM (CondOT path + midpoint ODE)"),
    "M7":  ("bond_head or yuelbond",           "molmetal/models/bond_head.py",                          "YuelBond bond head (learning-based)"),
    "M8":  ("lambda_astsr",                    "molmetal/molmetal_lam/reward/lambda_astsr.py",          "LambdaASTSR (Deflexpressor-style AST SR)"),
    "M9":  ("coupling_adapter or coupling_platinai5k", "molmetal/molmetal_lam/reward/coupling_adapter.py", "Coupling adapter (lambda × CFM bridge)"),
    "M10": ("metal_ligand_exchange or pt_metal_ligand_exchange", "molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py", "84-rule MLE matrix"),
    "M11": ("pt_click_compat or metal_ligand_exchange",          "molmetal/molmetal_lam/lam_chem/pt_click_compat.py",          "18 + 4 NEW SMARTS"),
    "M12": ("p1_oov or p1_skips",              "molmetal/adapters/flow_matching_lipman/__init__.py:1825", "20-element atom vocab"),
    "M13": ("p4_clamped or p4_amp",            "molmetal/adapters/flow_matching_lipman/__init__.py:2170", "Logits clamp (±50)"),
    "M14": ("p5_safety_max or p5_r4",          "molmetal/scripts/metallo_drug_smoke_retrain.py",         "LR warmup (linear 200 steps)"),
    "M15": ("p2_synflownet or p2_swap_smarts", "molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py:152", "Shim TypeError fix"),
    "M16": ("p7_model_wrapper",                "molmetal/adapters/flow_matching_lipman/_reference/utils/__init__.py",     "ModelWrapper kw-time fallback"),
    "M17": ("p6_configure_reference",          "molmetal/adapters/flow_matching_lipman/_reference_loader.py:83",         "Reference loader flat-clone auto-detect"),
    "M18": ("sa_score or erl",                 "molmetal/molmetal_lam/sbdd_env/sa_score.py",            "Ertl-Schuffenhauer SA score"),
    "M19": ("posebusters or pb_valid",         "molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py", "PoseBusters geometric filter"),
    "M20": ("biopython or plip",               "BioPython built-in",                                     "BioPython PDB NeighborSearch (PLIP replacement)"),
    "M21": ("scaffold or leakage",             "molmetal/data/leakage_utils.py",                         "Scaffold split (data leakage prevention)"),
}


# ---------------------------------------------------------------------------
# Parsing §4 of MECHANISM_METRIC_MAPPING.md
# ---------------------------------------------------------------------------

# A §4 row has the shape:
#   | m14 decode_ratio = 0 | M15 → M16 → M17 → M13 → M14 | `pytest ...` |
# We tokenise on the leading metric (m\d+) and the arrow chain in the
# middle column.
_ROW_RE = re.compile(
    r"^\|\s*(?P<symptom>m\d+[^|]*?)\s*\|\s*"
    r"(?P<mechs>(?:M\d+\s*(?:→\s*)?)+)\s*\|\s*"
    r"(?P<cmd>`[^`]+`|[^|]+?)\s*\|\s*$"
)


def parse_failure_modes(md_text: str) -> List[Dict[str, object]]:
    """Extract §4 failure-mode rows.

    Returns a list of dicts with keys ``symptom``, ``metric``,
    ``mechanisms`` (ordered list of M-IDs), ``command`` (raw pytest
    string).  Only rows that mention a metric ID are kept.
    """
    in_section = False
    rows: List[Dict[str, object]] = []
    for raw_line in md_text.splitlines():
        if raw_line.startswith("## 4."):
            in_section = True
            continue
        if in_section and raw_line.startswith("## 5."):
            break
        if not in_section or not raw_line.startswith("|"):
            continue
        m = _ROW_RE.match(raw_line)
        if not m:
            continue
        symptom = m.group("symptom").strip()
        metric_match = re.match(r"^(m\d+)\b", symptom)
        if not metric_match:
            continue
        mechs_raw = m.group("mechs").strip()
        mechanisms = [tok for tok in re.split(r"→|\s+", mechs_raw) if re.fullmatch(r"M\d+", tok)]
        cmd_raw = m.group("cmd").strip()
        if cmd_raw.startswith("`") and cmd_raw.endswith("`"):
            cmd_raw = cmd_raw[1:-1].strip()
        rows.append(
            {
                "metric": metric_match.group(1),
                "symptom": symptom,
                "mechanisms": mechanisms,
                "command": cmd_raw,
            }
        )
    return rows


def filter_rows(rows: List[Dict[str, object]], metric: str, symptom: Optional[str]) -> List[Dict[str, object]]:
    """Pick the failure-mode rows relevant to ``metric`` (and optionally ``symptom``)."""
    metric_rows = [r for r in rows if r["metric"] == metric]
    if symptom is not None:
        specific = [r for r in metric_rows if symptom.lower() in str(r["symptom"]).lower()]
        if specific:
            return specific
    return metric_rows


# ---------------------------------------------------------------------------
# Pytest invocation
# ---------------------------------------------------------------------------

def run_pytest(keyword: str, timeout: int = 120) -> Tuple[int, str, str, float]:
    """Run ``pytest -k '<keyword>'`` and return (rc, stdout, stderr, wall_s).

    Runs against the canonical pitfall-guard file first (which is where
    the P1–P7 regressions live).  If no tests are collected there (the
    mechanism has no dedicated guard, e.g. M5 EGNN / M6 CFM / M7
    YuelBond), fall back to the broader ``molmetal/tests`` tree so the
    user's keyword can match its canonical home (e.g.
    ``test_egnn_velocity_cfg.py``, ``test_cfm_rescue_decode_smoke.py``).

    Invokes pytest via ``sys.executable -m pytest`` so that the
    *current* interpreter's pytest is used — important when the host
    has both a system pytest (no torch) and a venv pytest (with torch)
    on PATH.  ``pyproject.toml`` still controls collection from the
    REPO_ROOT.
    """
    candidates: List[str] = [
        "molmetal/tests/test_pitfall_guards.py",
        "molmetal/tests",
    ]
    last_rc = 1
    last_stdout = ""
    last_stderr = ""
    t0 = time.time()
    for rel in candidates:
        cmd: List[str] = [sys.executable, "-m", "pytest", rel, "-k", keyword, "--tb=line", "--no-header", "-q"]
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "", f"pytest -k {keyword!r} timed out after {timeout}s", time.time() - t0
        last_rc, last_stdout, last_stderr = proc.returncode, proc.stdout, proc.stderr
        # pytest returncode 5 = "no tests collected".  Skip to next candidate.
        if proc.returncode == 5:
            continue
        return proc.returncode, proc.stdout, proc.stderr, time.time() - t0
    return last_rc, last_stdout, last_stderr, time.time() - t0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def short_status(stdout: str) -> str:
    """Return a one-line summary (e.g. ``1 passed`` / ``3 failed``)."""
    for line in stdout.splitlines():
        if " passed" in line or " failed" in line:
            return line.strip()
    return "no summary line"


def run_diagnosis(rows: List[Dict[str, object]], timeout: int) -> Dict[str, object]:
    """Run pytest for every mechanism in the filtered failure-mode rows."""
    results: List[Dict[str, object]] = []
    for row in rows:
        for mech in row["mechanisms"]:  # type: ignore[index]
            kw = MECHANISM_PYTEST.get(mech, (None, "", ""))[0]
            if kw is None:
                results.append(
                    {
                        "mechanism": mech,
                        "status": "SKIP",
                        "note": "no pytest mapping registered",
                        "symptom": row["symptom"],
                    }
                )
                continue
            rc, out, _err, wall = run_pytest(kw, timeout=timeout)
            if rc == 0:
                status = "PASS"
            elif rc == 5:  # pytest "no tests collected"
                status = "MISS"
            else:
                status = "FAIL"
            results.append(
                {
                    "mechanism": mech,
                    "status": status,
                    "returncode": rc,
                    "wall_seconds": round(wall, 2),
                    "summary": short_status(out) if status != "MISS" else "no test collected",
                    "keyword": kw,
                    "symptom": row["symptom"],
                }
            )
    likely = next((r for r in results if r["status"] == "FAIL"), None)
    return {"results": results, "likely_root_cause": likely}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_human(payload: Dict[str, object], metric: str) -> None:
    rows = payload.get("results", [])  # type: ignore[assignment]
    by_symptom: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:  # type: ignore[union-attr]
        by_symptom.setdefault(r["symptom"], []).append(r)  # type: ignore[arg-type]

    print(f"Metric {metric} — diagnostic:")
    for symptom, items in by_symptom.items():
        print(f"  Symptom: {symptom}")
        for item in items:
            mech = item["mechanism"]
            _, _, brief = MECHANISM_PYTEST.get(mech, ("", "", mech))
            kw = item.get("keyword", "")
            print(f"    - check {mech} ({brief}): pytest -k '{kw}'")
    print("\nRunning diagnostics...")
    fail_count = pass_count = miss_count = 0
    for item in rows:  # type: ignore[union-attr]
        mech = item["mechanism"]
        status = item["status"]
        summary = item.get("summary", "")
        if status == "PASS":
            print(f"    {mech}: PASS")
            pass_count += 1
        elif status == "FAIL":
            print(f"    {mech}: FAIL — {summary}")
            fail_count += 1
        elif status == "MISS":
            print(f"    {mech}: MISS — {summary}")
            miss_count += 1
        else:
            print(f"    {mech}: {status} — {summary}")
    likely = payload.get("likely_root_cause")
    if likely is None:
        print(f"\nAll {pass_count} checks PASS (no regression detected).")
    else:
        mech = likely["mechanism"]
        _, fix_path, brief = MECHANISM_PYTEST.get(mech, ("", "molmetal/", ""))
        print(f"\nLikely root cause: {mech} ({brief})")
        print(f"Suggested fix: see {fix_path}")
    print(f"\nSummary: PASS={pass_count} FAIL={fail_count} MISS={miss_count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metric", help="Metric ID, e.g. m14")
    parser.add_argument(
        "--symptom",
        default=None,
        help="Optional substring filter on the symptom (e.g. 'm14 decode_ratio = 0').",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Per-pytest timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--list", action="store_true", help="List metrics covered by §4 and exit.")
    args = parser.parse_args(argv)

    if not MAPPING_MD_CANDIDATES[0].is_file() and not MAPPING_MD_CANDIDATES[1].is_file():
        print(
            f"FATAL: mapping file not found at {MAPPING_MD_CANDIDATES[0]} or {MAPPING_MD_CANDIDATES[1]}",
            file=sys.stderr,
        )
        return 2

    md_text = _resolve_mapping_md().read_text(encoding="utf-8")
    rows = parse_failure_modes(md_text)

    if args.list:
        seen = sorted({r["metric"] for r in rows})  # type: ignore[arg-type]
        print("Metrics with failure-mode entries in §4:", ", ".join(seen))
        return 0

    if not args.metric:
        parser.error("--metric is required (or pass --list to enumerate metrics).")

    metric = args.metric.lower()
    if not re.fullmatch(r"m\d+", metric):
        parser.error("--metric must look like m1, m14, ...")

    selected = filter_rows(rows, metric, args.symptom)
    if not selected:
        print(f"No failure-mode row in §4 mentions {metric}.", file=sys.stderr)
        print("Available metrics:", ", ".join(sorted({r['metric'] for r in rows})), file=sys.stderr)
        return 1

    payload = run_diagnosis(selected, timeout=args.timeout)
    if args.json:
        out = {
            "metric": metric,
            "symptom_filter": args.symptom,
            "rows": selected,
            "diagnostics": payload["results"],
            "likely_root_cause": payload["likely_root_cause"],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print_human(
            {
                "results": payload["results"],
                "likely_root_cause": payload["likely_root_cause"],
            },
            metric,
        )

    has_fail = any(r["status"] == "FAIL" for r in payload["results"])  # type: ignore[union-attr]
    return 1 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
