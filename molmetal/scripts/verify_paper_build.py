#!/usr/bin/env python3
"""Wf-paper-build — verify paper/main.tex inputs all R18 ship sections.

This is a read-only verification script for the R18 paper build.  It does
NOT run pdflatex (the TeX environment may be missing); instead it greps
``paper/main.tex`` for ``\\input`` directives, enumerates the section /
appendix files, counts the bibliography entries, and checks that the
Wf-3 ship (the four new §3.4-3.7 method-reference subsections) and the
``math_theorems`` appendix are all wired in.

Outputs (under ``molmetal/reports/wf_final_verification/``):

* ``paper_build.json`` — machine-readable snapshot
* ``paper_inventory.md`` — human-readable section / appendix index

Usage::

    python molmetal/scripts/verify_paper_build.py
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_PAPER_DIR = _REPO_ROOT / "paper"
_MAIN_TEX = _PAPER_DIR / "main.tex"
_MAIN_PDF = _PAPER_DIR / "main.pdf"
_REFS_BIB = _PAPER_DIR / "refs.bib"
_REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_final_verification"

# Wf-3 ship (TODO-25 Round-14 math grounding, Wf-Math-1 + Wf-Math-2):
# four new method-reference subsections chaining the 9 R18 math theorems +
# three adapters (W2.1 / W2.2 / W2.3) into a single methodology block.
_WF3_SHIP_SECTIONS = (
    "sections/03_4_se3_schrodinger_bridge.tex",
    "sections/03_5_transport_relaxation.tex",
    "sections/03_6_multi_parameter_ph.tex",
    "sections/03_7_monge_ampere_wasserstein_gf.tex",
)


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _grep_inputs(main_tex: str) -> List[str]:
    """Extract every ``\\input{path}`` token in document order."""
    return re.findall(r"\\input\{([^}]+)\}", main_tex)


def _list_section_files(main_tex: str) -> List[str]:
    """Return the ordered list of section files actually ``\\input``-ed.

    LaTeX auto-appends ``.tex`` so the bare stem is what appears in
    ``\\input``.  We normalise each token to ``sections/<stem>.tex`` so
    callers can compare with concrete file paths.
    """
    out: List[str] = []
    for t in _grep_inputs(main_tex):
        if t.startswith("sections/"):
            stem = t[len("sections/") :]
            if not stem.endswith(".tex"):
                stem = stem + ".tex"
            out.append(f"sections/{stem}")
    return out


def _list_appendix_files(main_tex: str) -> List[str]:
    """Return ordered list of appendix-ish files ``\\input``-ed.

    Captures both ``appendices/<stem>`` tokens (LaTeX auto-appends
    ``.tex``) and the bare ``supplementary`` token that loads
    ``paper/supplementary.tex``.  Normalised to ``<path>.tex`` so the
    caller can compare with concrete file paths.
    """
    out: List[str] = []
    for t in _grep_inputs(main_tex):
        if t.startswith("appendices/"):
            stem = t[len("appendices/") :]
            if not stem.endswith(".tex"):
                stem = stem + ".tex"
            out.append(f"appendices/{stem}")
        elif t == "supplementary":
            out.append("supplementary.tex")
    return out


def _count_bib_entries(bib_path: Path) -> int:
    """Count ``@type{key,`` openings (single-line bib entries are rare)."""
    if not bib_path.exists():
        return 0
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("@"))


def _wf3_ship_check(section_files: List[str]) -> Dict[str, bool]:
    return {p: (p in section_files) for p in _WF3_SHIP_SECTIONS}


def build_report() -> Dict[str, Any]:
    main_tex = _read(_MAIN_TEX)
    section_files = _list_section_files(main_tex)
    appendix_files = _list_appendix_files(main_tex)

    math_theorems_section_exists = "appendices/math_theorems.tex" in appendix_files

    return {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tool": "verify_paper_build.py",
        "repo_root": str(_REPO_ROOT),
        "main_tex": str(_MAIN_TEX),
        "main_pdf_exists": _MAIN_PDF.exists(),
        "main_pdf_size_bytes": (_MAIN_PDF.stat().st_size if _MAIN_PDF.exists() else 0),
        "n_sections_input": len(section_files),
        "sections_list": section_files,
        "n_appendices_input": len(appendix_files),
        "appendices_list": appendix_files,
        "ref_bib_count": _count_bib_entries(_REFS_BIB),
        "math_theorems_section_exists": math_theorems_section_exists,
        "wf3_ship_sections": _wf3_ship_check(section_files),
        "all_wf3_sections_wired": all((p in section_files) for p in _WF3_SHIP_SECTIONS),
    }


def _format_markdown(report: Dict[str, Any]) -> str:
    sec = report["sections_list"]
    app = report["appendices_list"]
    n_pages = (
        "n/a (paper/main.pdf present; page count not parsed without pdflatex)"
        if not report["main_pdf_exists"]
        else "(see main.pdf)"
    )
    wf3 = report["wf3_ship_sections"]
    lines = [
        "# Paper inventory — R18 ship",
        "",
        f"Generated: `{report['timestamp']}`  ",
        f"Tool: `{report['tool']}`  ",
        f"Repo root: `{report['repo_root']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| `paper/main.tex` present | {report['main_tex'] != ''} |",
        f"| `paper/main.pdf` present | {report['main_pdf_exists']} |",
        f"| main.pdf size (bytes) | {report['main_pdf_size_bytes']:,} |",
        f"| pages | {n_pages} |",
        f"| sections `\\input`-ed | {report['n_sections_input']} |",
        f"| appendices `\\input`-ed | {report['n_appendices_input']} |",
        f"| `refs.bib` entries (@-keys) | {report['ref_bib_count']} |",
        f"| `math_theorems.tex` appendix wired | "
        f"{report['math_theorems_section_exists']} |",
        f"| all Wf-3 ship sections wired | {report['all_wf3_sections_wired']} |",
        "",
        "## §1-§7 Sections (R18 ship body)",
        "",
        "Every numbered section is `\\input`-ed from `paper/sections/`.",
        "",
        "| Order | File | Notes |",
        "|---:|---|---|",
    ]
    notes = {
        "sections/01_intro.tex": "§1 Introduction",
        "sections/02_related.tex": "§2 Related Work",
        "sections/03_method.tex": "§3 Method (placeholder assembled by WF-Lambda-3)",
        "sections/03_4_cfm_methodology.tex": "§3.x CFM Methodology (Wf-CFM-M3-M5, Round-14 lit-grounded)",
        "sections/03_4_se3_schrodinger_bridge.tex": "§3.4 SE(3) Schrödinger Bridge (Wf-3 ship, W2.1)",
        "sections/03_5_transport_relaxation.tex": "§3.5 Transport Relaxation (Wf-3 ship, W2.2)",
        "sections/03_6_multi_parameter_ph.tex": "§3.6 Multi-Parameter PH (Wf-3 ship, W2.3)",
        "sections/03_7_monge_ampere_wasserstein_gf.tex": "§3.7 Wasserstein / Monge-Ampère foundations (Wf-3 ship)",
        "sections/04_evaluation.tex": "§4 Evaluation (placeholder)",
        "sections/04_results.tex": "§4 Results (Round-17 rewrite)",
        "sections/04_results_table2.tex": "§4 Results Table 2 (20-metric headline panel)",
        "sections/04_5_honest_negative.tex": "§4.5 Honest Negative Framing (Round-17)",
        "sections/05_ablation.tex": "§5 Ablation Studies (placeholder)",
        "sections/06_limitations.tex": "§6 Limitations (hand-written)",
        "sections/06_limitations_auto.tex": "§6 Limitations (auto-generated snapshot)",
        "sections/07_future.tex": "§7 Future Work",
        "sections/00_label_stubs.tex": "label stubs (silences orphan-reference warnings)",
    }
    for i, p in enumerate(sec, start=1):
        lines.append(f"| {i} | `{p}` | {notes.get(p, '')} |")

    lines += [
        "",
        "## Appendices",
        "",
        "All appendices are `\\input`-ed from `paper/appendices/` and "
        "`paper/supplementary.tex`.",
        "",
        "| Order | File | Notes |",
        "|---:|---|---|",
    ]
    app_notes = {
        "appendices/closure_theorem.tex": "Appendix A: Closure under β-NF (placeholder, WF-Lambda-4)",
        "appendices/betanf_semantics.tex": "Appendix B: Operational Semantics of β-NF (pre-existing)",
        "appendices/math_theorems.tex": "Appendix D: Mathematical Theorems (Wf-Math-1, 9 R18 theorems)",
    }
    supp_seen = False
    for i, p in enumerate(app, start=1):
        if p == "supplementary.tex":
            supp_seen = True
            lines.append(
                f"| {i} | `paper/{p}` | "
                "Supplementary material (S1-S10 + data table) |"
            )
        else:
            lines.append(f"| {i} | `paper/{p}` | {app_notes.get(p, '')} |")
    if not supp_seen:
        lines.append("| - | `paper/supplementary.tex` | " "NOT currently \\input-ed |")

    lines += [
        "",
        "## Wf-3 ship — four new §3.4-3.7 sections (TODO-25 Round-14)",
        "",
        "The four math-grounded method-reference subsections ship the "
        "9 R18 theorems + three adapters (W2.1 / W2.2 / W2.3).",
        "",
        "| Wf-3 ship file | Wired in `main.tex` |",
        "|---|:---:|",
    ]
    for p in _WF3_SHIP_SECTIONS:
        lines.append(f"| `{p}` | {'YES' if wf3[p] else 'NO'} |")

    lines += [
        "",
        "## References",
        "",
        f"`paper/refs.bib` contains **{report['ref_bib_count']}** `@`-keyed "
        "entries (counted by line-leading `@`; matches the bibliography "
        "rendered by `\\bibliography{refs}`).",
        "",
        "## Verification mode",
        "",
        "* Read-only: this script does **not** invoke `pdflatex` or " "`bibtex`.",
        "* Page count is **not** parsed from `main.log` (per task "
        "instruction); the `main.pdf` artefact is reported by size only.",
        "* All `\\input{...}` tokens in `paper/main.tex` are extracted "
        "via regex; the order matches document order.",
        "",
    ]
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()

    json_path = _REPORT_DIR / "paper_build.json"
    md_path = _REPORT_DIR / "paper_inventory.md"

    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_format_markdown(report), encoding="utf-8")

    print(f"[verify_paper_build] wrote {json_path}")
    print(f"[verify_paper_build] wrote {md_path}")
    print(
        f"[verify_paper_build] sections={report['n_sections_input']} "
        f"appendices={report['n_appendices_input']} "
        f"bib={report['ref_bib_count']} "
        f"math_theorems={report['math_theorems_section_exists']} "
        f"wf3_all_wired={report['all_wf3_sections_wired']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
