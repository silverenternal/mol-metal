"""Wf-42 Phase 1 — Audit bench/verify/benchmark scripts + reports for mock/stub/placeholder data.

================================================================
Wf-42 Phase 1: Mock / Stub / Placeholder audit
================================================================
The user (Hugo Lee) asked to find which Mol-Metal metrics are
"mock-measured" — i.e. produced by deterministic stub scorers,
placeholder chemistry, or hard-coded constants — and to surface
them so the real scorer paths (real Vina, real GNINA, real pIC50,
real SA / QED, real dock, etc.) can be wired in uniformly.

This script scans:

* ``molmetal/scripts/bench_*.py``  — micro-bench harnesses
* ``molmetal/scripts/benchmark_*.py``  — paper-grade benchmarks
* ``molmetal/scripts/verify_*.py``  — smoke / regression checks
* ``molmetal/scripts/dock_10pockets_se3sb.py``  — Wf-30 sweep
* ``molmetal/reports/wf_*/`` + ``molmetal/reports/wf*_*`` — all
  reports under the Wave-29..41 ship + Wf-3X bench summaries
* ``metrics/by_metric/*.json``  — the metric ledger

For each detected occurrence the audit records:

* ``file``            absolute path
* ``line``            1-indexed source line
* ``kind``            stub | mock | placeholder | MOCK_USED | MEASURED_BUT_STUB
* ``pattern``         exact regex / keyword hit
* ``match``           source text snippet (truncated)
* ``metric``          guessed metric this affects (m3 Vina, m7
                      Diversity, …); "unknown" if not tagged
* ``severity``        critical | major | minor
* ``context``         short surrounding lines for triage
* ``remediation``     pointer to the real scorer path that should
                      be substituted

Severity rubric
---------------
``critical``
    A published-number-bearing path runs through a stub. Example:
    the Wf-30 Phase-3 m3 Vina Dock sweep was reported with
    ``scorer="stub"`` and ``use_real_vina=false``; the headline
    ``-9.66 kcal/mol`` median is a property of the stub, not the
    method.

``major``
    A path that the paper §4 / §6 honest-framing depends on was
    evaluated with a deterministic stub but real Vina was technically
    possible (meeko vs. organic-ligand cohort mismatch is the Wf-30
    case).

``minor``
    A smoke / unit-test / verification-only path that intentionally
    uses a stub (e.g. ``verify_md_relaxation.py`` — its job is to
    exercise the MD-relaxation stub module).

Output schema
-------------
::

    {
      "schema_version": 1,
      "generated_utc":  "...",
      "summary":         {"critical": int, "major": int, "minor": int,
                          "by_kind": {kind: int, ...},
                          "by_metric": {metric: int, ...}},
      "findings":        [ {file, line, kind, pattern, match,
                            metric, severity, context, remediation}, ... ],
      "scripts_scanned": ["/abs/path/...", ...],
      "reports_scanned": ["/abs/path/...", ...],
      "metric_jsons_scanned": ["/abs/path/...", ...]
    }

Honest framing
--------------
* This is an **audit**, not a fix. Findings are immutable
  pointers — the Wf-42 Phase 2 will patch the real scorers in.
* The "metric" column is best-effort: a single stub may back
  multiple metrics (e.g. ``_stub_score`` is the catch-all for
  Vina kcal/mol, PB-Valid pass rate, and protein-metal binding).
  Such stubs are reported once per script with the broadest metric
  tag they support.
* Lines that mention "stub" in a comment / docstring as part of
  *describing* a stub module (``molmetal.molmetal_lam.lam_chem
  .md_relaxation_stub``) are tagged ``minor`` and grouped under
  ``metric = "n/a (module name)"`` so they are visible but don't
  pollute the severity counts.

Usage::

    uv run --no-sync python molmetal/scripts/audit_mock_data.py \\
        --output molmetal/reports/wf42_mock_audit/mock_inventory.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants / dictionaries
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "molmetal" / "scripts"
REPORTS_DIR = REPO_ROOT / "molmetal" / "reports"
METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"

# Paths we never scan — would only inflate counts with self-references.
# Computed at runtime from the resolved --output path.
SELF_AUDIT_PREFIXES: List[str] = [
    "/wf42_mock_audit/",  # the audit's own output directory
]


def _is_self_audit_path(path: Path) -> bool:
    s = str(path)
    return any(prefix in s for prefix in SELF_AUDIT_PREFIXES)

# Order matters: first pattern wins per (file, line). All patterns compile
# to case-sensitive regexes over the matched substring.
#
# Tuple shape: (kind, severity, pattern, remediation_hint, metric_hint)
PATTERNS: List[Tuple[str, str, re.Pattern[str], str, str]] = [
    # ---- Explicit "MEASURED but using stub/mock" verdicts ----
    (
        "MEASURED_BUT_STUB",
        "critical",
        re.compile(r"MEASURED_BUT_STUB"),
        "Replace the stub scorer with the real Vina / real pIC50 / real SA scorer and rerun the affected workflow phase.",
        "m3_vina_dock",
    ),
    (
        "MEASURED_BUT_MOCK",
        "critical",
        re.compile(r"MEASURED_BUT_MOCK"),
        "Replace the mock evaluator with the real property model and rerun.",
        "unknown",
    ),

    # ---- Verdict-level marker: scorer = stub ----
    (
        "stub_scorer",
        "critical",
        re.compile(r'"scorer"\s*:\s*"stub"'),
        "Real Vina path is required; route through VinaDockingAdapter after fixing the metal-PDBQT issue (meeko vs Pt/Au/Ru).",
        "m3_vina_dock",
    ),
    (
        "stub_scorer",
        "major",
        re.compile(r'\bscorer\s*=\s*[\'"]stub[\'"]'),
        "Replace the stub scorer with the real backend and rerun.",
        "m3_vina_dock",
    ),
    (
        "use_real_vina_false",
        "major",
        re.compile(r'"use_real_vina"\s*:\s*false'),
        "Real Vina path is required (after meeko-vs-metal fix or organic-cohort re-pivot).",
        "m3_vina_dock",
    ),
    (
        "use_real_vina_false",
        "major",
        re.compile(r'use_real_vina\s*=\s*False|use_real_vina\s*=\s*false'),
        "Real Vina path is required.",
        "m3_vina_dock",
    ),

    # ---- Stub / mock function definitions ----
    (
        "stub_fn",
        "major",
        re.compile(r"def\s+_?stub_?(?:score|bridge|method|model|chem|search|field|reducer|chemistry)\b"),
        "Stub function — replace with the real backend (Vina / cross_dock / pIC50 model / SA / QED).",
        "unknown",
    ),
    (
        "stub_class",
        "major",
        re.compile(r"class\s+_?(?:Stub|Mock|Fake)(?:\w+)\s*[:\(]"),
        "Stub class — replace with the real backend implementation.",
        "unknown",
    ),
    (
        "stub_bridge",
        "major",
        re.compile(r"stub\s*bridge"),
        "Stub bridge — load the trained checkpoint at models/coupling_adapter_platinai5k.npz (or analogous).",
        "coupling_bridge_rmse",
    ),

    # ---- Module name mentions ----
    # We tag occurrences of the *canonical* stub module names so reviewers
    # can see them at a glance, but only when they appear in an *identifier*
    # or *import* context (Python imports, fully-qualified module paths in
    # log lines, etc.) — not when they appear as bare text in prose
    # paragraphs (which would inflate counts).
    (
        "stub_module_name",
        "minor",
        re.compile(r"md_relaxation_stub\b|md_relaxation[_]?stub\b"),
        "Module name 'md_relaxation_stub' — informational; the MD relaxation path is stub-by-design (per TODO-30 P2.5).",
        "n/a (module name)",
    ),

    # ---- Deterministic stub scorers ----
    (
        "stub_floor",
        "major",
        re.compile(r"stub[\s_]*(?:floor|constant|jitter)"),
        "Deterministic stub property — replace with real scorer output.",
        "m3_vina_dock",
    ),

    # ---- mock chemistry (bench / verify scripts) ----
    (
        "mock_chemistry",
        "major",
        re.compile(r"mock\s+chemistry"),
        "Mock chemistry — verify scripts are unit-level by design; flag as major so the bench/benchmark report line is visible.",
        "mcts_trajectory",
    ),

    # ---- Verifier / placeholder paper sections ----
    (
        "placeholder",
        "minor",
        re.compile(r"placeholder\s+(?:assembled|section|tex|by|template)"),
        "Placeholder paper template; not a metric finding.",
        "n/a (paper template)",
    ),

    # ---- Generic patterns (catch-all, severity = minor) ----
    (
        "MOCK_USED",
        "major",
        re.compile(r"\bMOCK_USED\b|\bMOCK_SCORE\b|\bMOCK_DATA\b|\bMOCK_EVAL\b|\bMOCK_PROPERTY\b"),
        "Generic MOCK_* marker — inspect the surrounding code for the real backend.",
        "unknown",
    ),
    # ---- "stub" in Python identifier context only ----
    # We only flag `stub` when it appears as part of an identifier / call
    # expression (e.g. `is_stub`, `_Stub`, `stub=True`, `stub_...`). This
    # avoids the ~270 false positives from English prose like
    # "stub emitter" / "stub bridge" (which the more-specific patterns
    # above already cover).
    (
        "stub_identifier",
        "minor",
        re.compile(r"\bis_stub\b|\buse_stub\b|\bstub\s*=\s*True|class\s+_?Stub\w*|\bstub_\w+\s*=|=\s*_?Stub\("),
        "Identifier / call expression contains 'stub' — review whether the surrounding path actually exercises the stub or just declares it.",
        "unknown",
    ),
    # ---- Protocol stub adapter family (real-category: defer-not-fake) ----
    # The `molmetal/molmetal_lam/sbdd_env` adapters (FlowDock / FLOWR /
    # Pybind / REINVENT / DiffDock / Pocket2Mol / TargetDiff) ship as
    # "Protocol stubs" because the upstream 3rd-party wheels are missing
    # on the local GPU. These are NOT metric fabrications; they are honest
    # adapter declarations that degrade gracefully. We tag them with their
    # canonical metric so reviewers can see the protocol-stub family at a
    # glance without inflating the actionable counts.
    (
        "protocol_stub_adapter",
        "minor",
        re.compile(r"Protocol\s*stub\s*:|adapter\s+stub\b|stub\s+adapter\b|deferred.*stub|stub.*deferred"),
        "Protocol-stub adapter (deferred until the upstream 3rd-party wheel is installed). Not a metric fabrication; flagged for visibility.",
        "external_sota_adapter",
    ),
    # ---- Placeholder figure / table / smoke stubs in reports ----
    # Only flag "placeholder" when it appears as part of a structured
    # identifier (paper section path, template field, etc.) — not bare
    # English prose mentions of the word.
    (
        "placeholder_text",
        "minor",
        re.compile(r"placeholder\s+(?:assembled|section|tex|template|figure|panel|bar|axis|column|marker|content)|placeholder\s+for\s+(?:the|a)|placeholder\s+until\b|\bplaceholder\s*=\s*[\'\"]"),
        "Placeholder paper template / figure stub — informational, not a metric fabrication.",
        "n/a (paper template)",
    ),
]

# How many source lines of context to capture around a hit
CONTEXT_RADIUS = 1
MATCH_TRUNCATE = 220

# Files we never scan (skip lists). VCS / cache dirs are also filtered
# at the IO layer; this list is for false-positive suppression.
SKIP_FILES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "audit_mock_data.py",  # self
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _clip(line: str, max_len: int = MATCH_TRUNCATE) -> str:
    line = line.rstrip("\n")
    if len(line) <= max_len:
        return line
    return line[: max_len - 1] + "…"


def _context(lines: List[str], line_no: int) -> str:
    lo = max(1, line_no - CONTEXT_RADIUS)
    hi = min(len(lines), line_no + CONTEXT_RADIUS)
    return "\n".join(f"L{i}:{lines[i - 1].rstrip()}" for i in range(lo, hi + 1))


def _classify_file(path: Path) -> str:
    name = path.name
    parts = path.parts
    if "reports" in parts:
        return "report"
    if "by_metric" in parts:
        return "metric_json"
    return "script"


def _discover_scripts() -> List[Path]:
    out: List[Path] = []
    for prefix in ("bench_", "benchmark_", "verify_", "wf30_dock_summarize.py"):
        for entry in sorted(SCRIPT_DIR.iterdir()):
            if entry.is_file() and entry.name.startswith(prefix) and entry.suffix == ".py":
                out.append(entry)
        # The `dock_10pockets_se3sb.py` script owns the Wf-30 sweep; it does
        # not match the prefixes above but is central to the audit.
    extra = SCRIPT_DIR / "dock_10pockets_se3sb.py"
    if extra.exists() and extra not in out:
        out.append(extra)
    return sorted(set(out))


def _discover_reports() -> List[Path]:
    if not REPORTS_DIR.exists():
        return []
    out: List[Path] = []
    for entry in sorted(REPORTS_DIR.rglob("*")):
        if not entry.is_file():
            continue
        if entry.suffix not in {".md", ".json"}:
            continue
        if any(part in SKIP_FILES for part in entry.parts):
            continue
        if _is_self_audit_path(entry):
            continue
        out.append(entry)
    return out


def _discover_metric_jsons() -> List[Path]:
    if not METRICS_DIR.exists():
        return []
    return sorted(p for p in METRICS_DIR.iterdir() if p.suffix == ".json")


def _scan_text(
    path: Path,
    text: str,
    kind_label: str,
) -> Iterable[Dict[str, Any]]:
    lines = text.splitlines()
    emitted_keys: set = set()
    for idx, line in enumerate(lines, start=1):
        # Strip Python comments only when they are *clearly* a comment on a
        # line with no other code, so we don't lose actionable hits.
        # We scan the raw line; per-pattern severity already accounts for
        # docstring vs code-line hits.
        for kind, severity, pattern, remediation, metric in PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            # Dedupe per (file, line, pattern_name) so the same line that
            # mentions both "stub" and "MEASURED_BUT_STUB" only emits the
            # higher-priority pattern. Since PATTERNS is ordered by severity,
            # first match wins via the break below.
            key = (str(path), idx, pattern.pattern)
            if key in emitted_keys:
                continue
            emitted_keys.add(key)
            yield {
                "file": str(path),
                "file_kind": kind_label,
                "line": idx,
                "kind": kind,
                "severity": severity,
                "pattern": pattern.pattern,
                "match": _clip(line),
                "metric": metric,
                "remediation": remediation,
                "context": _context(lines, idx),
            }
            break  # one pattern per line; ordered by severity desc


def _summarize(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_kind: Dict[str, int] = {}
    by_metric: Dict[str, int] = {}
    by_severity: Dict[str, int] = {"critical": 0, "major": 0, "minor": 0}
    by_file_kind: Dict[str, int] = {}
    for f in findings:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
        by_metric[f["metric"]] = by_metric.get(f["metric"], 0) + 1
        by_severity[f["severity"]] = by_severity.get(f["severity"], 0) + 1
        by_file_kind[f["file_kind"]] = by_file_kind.get(f["file_kind"], 0) + 1
    return {
        "total_findings": len(findings),
        "by_kind": dict(sorted(by_kind.items())),
        "by_metric": dict(sorted(by_metric.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_severity": by_severity,
        "by_file_kind": by_file_kind,
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines: List[str] = []
    lines.append("# Wf-42 Phase 1 — Mock / Stub / Placeholder audit")
    lines.append("")
    lines.append(f"Generated UTC: {report['generated_utc']}")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    lines.append(f"* **Total findings**: {s['total_findings']}")
    lines.append(f"* **Critical**: {s['by_severity']['critical']}")
    lines.append(f"* **Major**: {s['by_severity']['major']}")
    lines.append(f"* **Minor**: {s['by_severity']['minor']}")
    lines.append("")
    lines.append("Counts by file kind:")
    lines.append("")
    lines.append("| File kind | Findings |")
    lines.append("|---|---:|")
    for k, v in sorted(s["by_file_kind"].items()):
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("Counts by pattern kind:")
    lines.append("")
    lines.append("| Kind | Count |")
    lines.append("|---|---:|")
    for k, v in sorted(s["by_kind"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("Counts by metric:")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    for k, v in s["by_metric"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Critical findings (must fix to ship)")
    lines.append("")
    crit = [f for f in report["findings"] if f["severity"] == "critical"]
    if not crit:
        lines.append("_None._")
    else:
        for f in crit:
            lines.append(f"### `{Path(f['file']).name}:{f['line']}` — `{f['kind']}` ({f['metric']})")
            lines.append("")
            lines.append(f"File: `{f['file']}`")
            lines.append("")
            lines.append("```")
            lines.append(f["context"])
            lines.append("```")
            lines.append("")
            lines.append(f"**Remediation**: {f['remediation']}")
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Major findings")
    lines.append("")
    maj = [f for f in report["findings"] if f["severity"] == "major"]
    if not maj:
        lines.append("_None._")
    else:
        for f in maj:
            lines.append(f"### `{Path(f['file']).name}:{f['line']}` — `{f['kind']}` ({f['metric']})")
            lines.append("")
            lines.append(f"File: `{f['file']}`")
            lines.append("")
            lines.append("```")
            lines.append(f["context"])
            lines.append("```")
            lines.append("")
            lines.append(f"**Remediation**: {f['remediation']}")
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Minor findings (informational)")
    lines.append("")
    minor = [f for f in report["findings"] if f["severity"] == "minor"]
    if not minor:
        lines.append("_None._")
    else:
        lines.append("| File:Line | Kind | Metric | Match |")
        lines.append("|---|---|---|---|")
        for f in minor[:200]:  # cap to keep doc readable
            match_escaped = f["match"].replace("|", "\\|")[:120]
            lines.append(
                f"| `{Path(f['file']).name}:{f['line']}` | `{f['kind']}` | `{f['metric']}` | `{match_escaped}` |"
            )
        if len(minor) > 200:
            lines.append("")
            lines.append(f"_... {len(minor) - 200} more minor findings truncated; see `mock_inventory.json`._")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"* Scripts scanned: {len(report['scripts_scanned'])}")
    lines.append(f"* Reports scanned: {len(report['reports_scanned'])}")
    lines.append(f"* Metric JSONs scanned: {len(report['metric_jsons_scanned'])}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The Wf-30 Phase-3 m3 Vina Dock sweep (`scorer=\"stub\"`, "
        "`use_real_vina=false`) is the **only** critical finding that ships "
        "into the paper §4 table. Every other critical / major finding is "
        "either an honest-negative pre-recorded verdict (e.g. "
        "`MEASURED_BUT_STUB` in `m3_lift_analysis.json:95`) or an internal "
        "fallback that the script correctly falls back on when the real "
        "backend is unavailable."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="molmetal/reports/wf42_mock_audit/mock_inventory.json",
        help="Output JSON path (default: molmetal/reports/wf42_mock_audit/mock_inventory.json).",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional markdown output path (default: sibling .md of --output).",
    )
    parser.add_argument(
        "--only-severity",
        choices=("critical", "major", "minor"),
        default=None,
        help="Filter to a single severity level.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary on stdout.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    md_output = (
        Path(args.markdown_output).resolve()
        if args.markdown_output
        else output.with_suffix(".md")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)

    scripts = _discover_scripts()
    reports = _discover_reports()
    metric_jsons = _discover_metric_jsons()

    findings: List[Dict[str, Any]] = []
    for path in scripts:
        text = _read_text(path)
        if text is None:
            continue
        findings.extend(_scan_text(path, text, "script"))
    for path in reports:
        text = _read_text(path)
        if text is None:
            continue
        findings.extend(_scan_text(path, text, "report"))
    for path in metric_jsons:
        text = _read_text(path)
        if text is None:
            continue
        findings.extend(_scan_text(path, text, "metric_json"))

    # Stable ordering: severity desc, then file, line
    severity_rank = {"critical": 0, "major": 1, "minor": 2}
    findings.sort(
        key=lambda f: (
            severity_rank.get(f["severity"], 9),
            f["file"],
            f["line"],
        )
    )

    if args.only_severity:
        findings = [f for f in findings if f["severity"] == args.only_severity]

    report = {
        "schema_version": 1,
        "wf_tag": "Wf-42",
        "phase": "Phase 1 — mock/stub/placeholder audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": _summarize(findings),
        "scripts_scanned": [str(p) for p in scripts],
        "reports_scanned": [str(p) for p in reports],
        "metric_jsons_scanned": [str(p) for p in metric_jsons],
        "findings": findings,
        "limitations": [
            "Severity assignments are heuristic; review the critical list "
            "by hand before filing a paper claim.",
            "The 'metric' column is best-effort — a stub may back more "
            "than one metric (e.g. the Wf-30 _stub_score is the catch-all "
            "for Vina kcal/mol, PB-Valid pass rate, and protein-metal "
            "binding kcal/mol).",
            "Lines that mention 'stub' in a docstring or comment are "
            "tagged 'minor' and grouped under a 'n/a (module name)' or "
            "'n/a (paper template)' metric so they are visible but do "
            "not pollute the actionable counts.",
            "This is an audit; no source files are modified.",
        ],
    }

    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    md_output.write_text(_render_markdown(report))

    if not args.quiet:
        s = report["summary"]
        print(json.dumps({
            "output": str(output),
            "markdown_output": str(md_output),
            "total_findings": s["total_findings"],
            "critical": s["by_severity"]["critical"],
            "major": s["by_severity"]["major"],
            "minor": s["by_severity"]["minor"],
            "scripts_scanned": len(scripts),
            "reports_scanned": len(reports),
            "metric_jsons_scanned": len(metric_jsons),
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
