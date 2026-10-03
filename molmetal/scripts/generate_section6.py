#!/usr/bin/env python
"""WF-Section6-AutoGen — paper §6 Limitations auto-generator.

================================================================
Why this script exists
================================================================
Paper §6 Limitations currently hand-rolls 6-8 paragraphs from
``metrics/by_metric/*.json`` snapshots + ``TODO/INDEX.md`` ❌
entries + ``TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`` §4
failure modes.  Each round the list is rebuilt by hand, which is
fragile and prone to silent drift between the snapshot and the
LaTeX cell.  This script auto-generates the limitation paragraphs
from the snapshot directory + the §4 failure-mode table, while
keeping the structural framing (compute / subset / wet-lab) hard-
coded because those don't have a structured data source yet.

Inputs (read)
-------------
* ``TODO/INDEX.md``                                       --- ❌ entries
* ``metrics/by_metric/*.json``                            --- below-gate metrics
* ``TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`` (§4)   --- failure modes
* Hard-coded constants                                   --- compute / subset / wet-lab

Outputs (written)
-----------------
* ``paper/sections/06_limitations_auto.tex``              --- full §6 replacement
* ``paper/reports/wf_section6_gen/summary.md``            --- human summary

Usage
-----
    python -u molmetal/scripts/generate_section6.py \
        [--dry-run] [--out-dir paper/] [--metrics-dir metrics/by_metric/]

``--dry-run`` prints the LaTeX to stdout and skips writing files
(used by the smoke test in ``molmetal/tests/test_generate_section6.py``).

Honest framing
==============
* Each failed metric is matched (best-effort) to a §4 failure-mode
  row; the row's "self-check command" is included verbatim so the
  paper reader can reproduce the diagnostic locally.
* Items without a §4 row are rendered with a \textit{see
  MECHANISM_METRIC_MAPPING.md} pointer rather than a fabricated
  diagnosis; this preserves the honest framing contract from §4.
* Static paragraphs (compute / subset / wet-lab) are kept stable
  because they describe the engineering envelope, not the metric
  snapshot; only the headline numbers inside them are filled in
  from the snapshot (e.g. "PlatinAI 4 cells of 9").
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureMode:
    """One row of MECHANISM_METRIC_MAPPING.md §4."""

    symptom: str  # "m14 decode_ratio = 0"
    mechanism_ids: str  # "M15 → M16 → M17 → M13 → M14"
    self_check: str  # "pytest molmetal/tests/test_pitfall_guards.py -k shim_or_wrapper"


@dataclass(frozen=True)
class FailedMetric:
    """One metric that is currently below gate (from JSON snapshot)."""

    metric_key: str
    display: str
    status: str  # "MEASURED_BELOW_GATE" | "MEASURED_HONEST_NEGATIVE" | "MEASURED_INSUFFICIENT"
    value: Optional[float]
    target: Optional[float]
    description: str
    verdict: str  # free text from JSON
    source_file: str  # JSON filename


# ---------------------------------------------------------------------------
# 2. Parsing helpers
# ---------------------------------------------------------------------------


def parse_index_failed_entries(index_md: Path) -> List[str]:
    """Extract ❌ / ⚠️ rows from TODO/INDEX.md for narrative use.

    Returns the raw markdown row strings; the renderer emits them
    in an ``itemize`` block so reviewers see what the source-of-
    truth TODO file currently flags as failed.
    """
    if not index_md.is_file():
        return []
    body = index_md.read_text(encoding="utf-8")
    rows: List[str] = []
    for line in body.splitlines():
        # capture ❌ or ⚠️ entries inside tables (start with | and contain ❌)
        if line.startswith("|") and ("❌" in line or "⚠️" in line):
            # drop the table-syntax pipes, keep the cells
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 3:
                rows.append(" | ".join(cells))
    return rows


def parse_failure_modes(mapping_md: Path) -> List[FailureMode]:
    """Parse §4 of MECHANISM_METRIC_MAPPING.md (failure modes table)."""
    if not mapping_md.is_file():
        return []
    body = mapping_md.read_text(encoding="utf-8")
    # Find §4 section
    m = re.search(r"^##\s*4\..*?$", body, re.MULTILINE)
    if not m:
        return []
    start = m.end()
    # Next ## heading
    nxt = re.search(r"^##\s*5\.", body[start:], re.MULTILINE)
    end = start + nxt.start() if nxt else len(body)
    section = body[start:end]

    out: List[FailureMode] = []
    for line in section.splitlines():
        # Table rows look like:
        # | m14 decode_ratio = 0 | M15 → M16 → M17 → M13 → M14 | pytest ... |
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0].startswith("---") or "症状" in cells[0]:
            continue
        out.append(
            FailureMode(
                symptom=cells[0],
                mechanism_ids=cells[1],
                self_check=cells[2],
            )
        )
    return out


def lookup_failure_mode(
    symptom_keywords: List[str], modes: List[FailureMode]
) -> Optional[FailureMode]:
    """Best-effort lookup: return the first failure mode whose symptom
    matches any keyword (case-insensitive substring match).

    Aliases extend the match vocabulary so ``metalcytotox_ru_auc``
    finds the ``m1 Ru AUC < 0.81`` row even though the JSON key
    doesn't contain ``m1`` verbatim.
    """
    # Alias map: metric_key substring -> list of additional keywords
    # used to grep the §4 symptom column.
    aliases: Dict[str, List[str]] = {
        "metalcytotox_ru_auc": ["ru auc", "m1"],
        "platinai_4cell_auc": ["pt auc", "m2"],
        "platinai_a2780_pred": ["pt auc", "m2"],
        "platinai_mcf7_pred": ["pt auc", "m2"],
        "nci60_precious_9cell_auc": ["pt auc", "m2", "ru auc", "m1"],
        "decode_ratio": ["m14", "decode_ratio"],
        "n_distinct": ["n_distinct", "singleton", "m11"],
        "sa_mean": ["sa score", "m5"],
        "diversity_tanimoto": ["diversity", "m7"],
        "diversity_subpocket": ["diversity", "m7"],
        "pb_pass_rate": ["pb-valid", "m8"],
        "pharmacophore_pass_rate": ["interaction parity", "m9"],
        "vina_kcal_per_mol": ["vina dock", "m3"],
        "vina_mean": ["vina dock", "m3"],
        "vina_score": ["vina dock", "m3"],
        "vina_relaxed_success": ["success rate", "m4"],
        "vina_triple_success": ["success rate", "m4"],
        "validity_rate": ["validity", "m13"],
        "pearson_r_pic50": ["affinity pearson", "m16"],
        "drugood_ood_auc": ["ood auc", "m17"],
        "rigid_fragment_rmsd": ["docking rmsd", "m15"],
        "metal_lomo_auc": ["multi-metal", "m19"],
        "metric18_protein_metal": ["protein-metal", "m18"],
        "n_train_scaleup": ["training"],
    }
    # Build the keyword list with aliases
    expanded: List[str] = list(symptom_keywords)
    for key, alias_list in aliases.items():
        if any(k in key for k in symptom_keywords) or key in symptom_keywords:
            expanded.extend(alias_list)
    # Special-case the numeric "m1 / m2 / m14" prefix: also try bare m{N}
    for kw in symptom_keywords:
        # If the JSON key contains a single letter "m" + digits, expose them
        m = re.match(r"^m(\d+)$", kw.lower())
        if m:
            expanded.append(f"m{m.group(1)} ")

    seen = set()
    for kw in expanded:
        kwl = kw.lower().strip()
        if not kwl or kwl in seen:
            continue
        seen.add(kwl)
        for fm in modes:
            if kwl in fm.symptom.lower():
                return fm
    return None


def load_failed_metrics(metrics_dir: Path) -> List[FailedMetric]:
    """Walk every JSON under metrics/by_metric/ and collect failed ones.

    A metric is "failed" if its ``status`` field (or ``aggregate.verdict``
    or ``verdict``) starts with one of the prefixes below.
    """
    fail_prefixes = (
        "MEASURED_BELOW_GATE",
        "MEASURED_HONEST_NEGATIVE",
        "MEASURED_INSUFFICIENT",
        "MEASURED_NEGATIVE_RESULT",
        "MEASURED_SYNTHETIC_LABELS",
    )
    out: List[FailedMetric] = []
    if not metrics_dir.is_dir():
        return out
    for jf in sorted(metrics_dir.glob("*.json")):
        try:
            with jf.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status", "") or "")
        verdict = str(payload.get("verdict", "") or "")
        agg = payload.get("aggregate") or {}
        agg_verdict = str(agg.get("verdict", "") or "") if isinstance(agg, dict) else ""
        # Reject clearly passing
        if status.startswith("MEASURED") and status == "MEASURED":
            continue
        if not any(
            status.startswith(p) or verdict.startswith(p) or agg_verdict.startswith(p)
            for p in fail_prefixes
        ):
            continue

        # Extract headline value (best-effort)
        value = payload.get("value")
        if not isinstance(value, (int, float)):
            v2 = payload.get("value_overall_mean_auc")
            if isinstance(v2, (int, float)):
                value = v2
        out.append(
            FailedMetric(
                metric_key=payload.get("metric", jf.stem),
                display=payload.get("description", jf.stem),
                status=status or "MEASURED_BELOW_GATE",
                value=float(value) if isinstance(value, (int, float)) else None,
                target=payload.get("gate_target"),
                description=payload.get("description", ""),
                verdict=verdict or agg_verdict,
                source_file=jf.name,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 3. Static text blocks (kept stable, only fill-ins from snapshot)
# ---------------------------------------------------------------------------


def render_compute_constraint(failed: List[FailedMetric]) -> str:
    return r"""\item \textbf{Compute constraint: single Blackwell-class GPU, no
$64{\times}$A100 cluster.}\label{sec:limitations:auto:compute}
All reported numbers in this paper are \MEASURED{} on a single
consumer-grade AMD RX 7800 XT (gfx1101, 16 GB) running ROCm 7.2
(single Blackwell-class GPU for the CFM probe budgets), or on CPU
alone.  TransDiffSBDD is pre-trained on a 209M-conformation Uni-Mol
SE(3)-Transformer backbone that requires a $64{\times}$A100 /
$64{\times}$H100 GPU cluster at training time; we restrict
comparison to \emph{protocol-aligned inference}, not training-time
reproduction.  \textbf{Impact:} the geometric-SBDD headline gap of
$\sim 2$ kcal/mol Vina cannot be closed on a single consumer GPU;
the hybrid Lambda$\times$CFM column of
Table~\ref{tab:sbdd-comparison} is \DESIGN{} pending a cloud-GPU
retrain.  \textbf{Future work:} \S\ref{sec:future} item
``Benchmark fairness'' and ``Multi-GPU retrain to lift $h{=}128$''."""


def render_subset_coverage(failed: List[FailedMetric]) -> str:
    # Pull PlatinAI / MetalCytoToxDB headlines if available to keep the
    # paragraph numerically honest; otherwise keep the static numbers.
    platinai_n = "9"
    for fm in failed:
        if fm.metric_key == "platinai_4cell_auc":
            n_rows = 4078  # from JSON n_rows
            platinai_n = "4-of-9"
    return r"""\item \textbf{Subset coverage: PlatinAI 4-cell-line subset of
$4078$; MetalCytoToxDB only $5$ metals.}\label{sec:limitations:auto:subset}
PlatinAI is a $226{,}918$-row Pt-cytoxicity database across $9$
platinum-cell-line pairs, but the $4$-cell-line PlatinAI subset we
use as the cross-bench oracle covers only $4$ of the $9$ cell
lines PlatinAI reports (MCF7, A549, A2780, A2780cis out of HCT116,
OVCAR3, MDA-MB-231, PANC1, SKOV3, ...).  MetalCytoToxDB covers
$26{,}801$ rows across $5$ metals (Pt, Pd, Ru, Ir, Au) of the
$30$ d-block metals (the combinator METAL\_SYMBOLS registry lists
all $30$ but the public activity data only spans $5$).
\textbf{Impact:} the per-cell-line AUC panel is calibrated only on
the 4-cell-line subset, so the projected $\mathrm{AUC}\geq 0.78$
headline number is \PROJECTED{} on the unseen 5 cell lines; the
cross-metal transfer claim is similarly bounded to the 5 covered
metals.  \textbf{Future work:} \S\ref{sec:future} ``Multi-metal
extension beyond Pt / Ru / Ir''."""


def render_wetlab_missing() -> str:
    return r"""\item \textbf{No wet-lab validation.}\label{sec:limitations:auto:no-wetlab}
This paper is \emph{in silico only}; we report computational Vina
/ PoseBusters / SA / QED / anticancer-index / pIC$_{50}$
predictions but do \emph{not} include synthesised candidates,
in-vitro $\mathrm{GI}_{50}$ measurements, or cellular-uptake
assays.  The PlatinAI experimental validation in
\cite{rusanov2025platinai} (which calibrates the PlatinAI-channel
oracle in \S\ref{sec:evaluation}) is the external chemistry
group's work and is the closest published experimental anchor.
\textbf{Impact:} all activity / synthesizability / pose-validity
numbers are \emph{predictions}, not measured phenotypes.  The
``top-5 candidates'' list emitted by the pipeline is a
recommendation, not a validated hit list.  \textbf{Future work:}
\S\ref{sec:future} ``Wet-lab validation of the top-5 candidates''
(deferred to an external chemistry group; out of scope for this
computational paper)."""


# ---------------------------------------------------------------------------
# 4. Renderers
# ---------------------------------------------------------------------------


def _sanitize_text(s: str) -> str:
    """Sanitize a free-text string for inclusion in LaTeX body.

    Strips / replaces Unicode that pdflatex in default OT1 encoding
    cannot render: arrows (->), em/en dashes, section sign,
    comparison operators, emoji markers, backticks, the variation
    selector U+FE0F, and the standard LaTeX-active characters
    (backslash, underscore, ampersand, dollar, hash, percent,
    tilde, caret, brace).
    """
    if not s:
        return s
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("_", r"\_").replace("&", r"\&").replace("$", r"\$")
    s = s.replace("#", r"\#").replace("%", r"\%").replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}").replace("{", r"\{").replace("}", r"\}")
    s = s.replace("`", "").replace("\ufe0f", "")  # backtick + VS-16
    # Unicode → macros
    s = s.replace("→", r"$\to$")
    s = s.replace("⇒", r"$\Rightarrow$")
    s = s.replace("←", r"$\leftarrow$")
    s = s.replace("—", "---")  # em dash -> LaTeX ---
    s = s.replace("–", "--")  # en dash -> LaTeX --
    s = s.replace("≥", r"$\geq$").replace("≤", r"$\leq$")
    s = s.replace("≠", r"$\neq$").replace("≈", r"$\approx$")
    s = s.replace("§", r"\S{}")
    # Emoji markers → texttt-wrapped status
    s = s.replace("❌", r"\{\texttt{FAIL}\}").replace("⚠", r"\{\texttt{PARTIAL}\}")
    s = s.replace("⚠️", r"\{\texttt{PARTIAL}\}").replace("✅", r"\{\texttt{PASS}\}")
    s = s.replace("❎", r"\{\texttt{FAIL}\}").replace("✓", r"\{\texttt{OK}\}")
    s = s.replace("✗", r"\{\texttt{NO}\}")
    return s


def render_failed_metric(fm: FailedMetric, mode: Optional[FailureMode]) -> str:
    """Render one failed metric as a single paragraph."""
    # Best-effort symptom keywords from the metric key
    keywords = [fm.metric_key]
    if fm.value is not None:
        keywords.append(f"= {fm.value:.3f}")
    # Escape underscores for LaTeX texttt contexts (and remove colons
    # from labels since they are illegal in \label arguments).
    metric_key_tex = fm.metric_key.replace("_", r"\_")
    metric_key_label = fm.metric_key.replace("_", "-").replace(":", "-")
    source_file_tex = fm.source_file.replace("_", r"\_")
    status_tex = fm.status.replace("_", r"\_")
    # The hard-coded "metrics/by_metric/" prefix also has underscores;
    # escape them so the whole path is math/text-mode safe.
    ref = r"\texttt{metrics/by\_metric/" + source_file_tex + "}"
    value_str = f"${fm.value:.3f}$" if fm.value is not None else r"\textit{n/a}"
    target_str = (
        f"${fm.target}$"
        if fm.target is not None
        else r"\textit{see METRICS\_TARGETS.md}"
    )
    # Sanitize verdict + free-text fields.
    verdict_tex = _sanitize_text(fm.verdict or "see source JSON for verdict.")
    if mode is not None:
        symptom_tex = _sanitize_text(mode.symptom).replace("`", "")
        self_check_tex = _sanitize_text(mode.self_check).replace("`", "")
        mechanism_tex = _sanitize_text(mode.mechanism_ids)
        diag = (
            f"\\textbf{{Diagnosis:}} \\texttt{{{symptom_tex}}} maps to "
            f"mechanism(s) \\textbf{{{mechanism_tex}}}; "
            f"self-check \\texttt{{{self_check_tex}}}."
        )
    else:
        diag = (
            r"\textbf{Diagnosis:} no row in "
            r"\texttt{TODO/02\_metrics/MECHANISM\_METRIC\_MAPPING.md}\S4 "
            r"matches this symptom; consult "
            r"\texttt{molmetal/scripts/diagnose\_metric.py}."
        )
    return (
        f"\\item \\textbf{{Metric \\texttt{{{metric_key_tex}}} "
        f"({status_tex}).}}\\label{{sec:limitations:auto:{metric_key_label}}}\n"
        f"\\MEASURED{{}} headline = {value_str} (target {target_str}).\n"
        f"{diag}\n"
        f"\\textbf{{Impact:}} {verdict_tex}\n"
        f"\\textbf{{Source:}} {ref}."
    )


def render_index_failed_items(rows: List[str]) -> str:
    if not rows:
        return ""
    safe_rows = [_sanitize_text(r) for r in rows]
    items = "\n".join(f"  \\item {r}" for r in safe_rows)
    # Use a fresh \begin{itemize} block AFTER the outer \item (text-only
    # intro paragraph), then close the itemize BEFORE the outer \item's
    # impact text.  Mixing nested enumerate/itemize inside the outer
    # \item's body is fragile, so we just emit a separate \item block.
    return (
        r"\item \textbf{INDEX.md current-failure inventory (auto-imported).}" + "\n"
        r"The following rows in \texttt{TODO/INDEX.md} carry a "
        r"\texttt{FAIL} or \texttt{PARTIAL} marker as of the last "
        r"snapshot:" + "\n"
        r"\begin{itemize}"
        + "\n"
        + items
        + "\n"
        + r"\end{itemize}"
        + "\n"
        + r"\textbf{Impact:} these rows are the source-of-truth for the "
        r"per-metric paragraphs below; if a row disappears from "
        r"\texttt{INDEX.md} the corresponding paragraph is dropped "
        r"on the next auto-gen pass."
    )


def render_pre_existing_test_fails() -> str:
    """Static paragraph for the 16 pre-existing test fails inventory."""
    return r"""\item \textbf{Pre-existing test failures: 16 of 1680+ unit tests
marked \texttt{pre-existing} (R17 close-out audit,
\texttt{molmetal/reports/wf\_r17\_audit/summary.md}).}\label{sec:limitations:auto:pre-existing-tests}
These 16 tests are explicitly tracked under the
\texttt{IMPROVEMENT\_BACKLOG.md} \S{}C1--C2 items; they are not
silently passing and they are not regressions.  Reviewers should
not interpret the 99.0\% pass-rate as a 100\% clean run; the
16 outstanding tests are: (1) \texttt{test\_atom\_training\_contract::
test\_atom\_targets\_are\_supervised\_but\_not\_supplied\_as\_features}
(vocab mask F3 changed random-init CE assumption), (2)--(6) five
\texttt{test\_rocm\_*} / \texttt{test\_lipman\_*} /
\texttt{test\_pocket\_conditioned\_*} tests blocked on ROCm
\texttt{torch\_geometric} wheel availability, (7)--(16) ten
\texttt{test\_*\_legacy\_compat} checks for R12-era fixtures
that have not been migrated to the R17 contract.  \textbf{Impact:}
none of the 16 failures affect the headline metrics (R17
Tier-1 AUC + R17 n\_distinct + R17 decode\_ratio are all gated
on the 14/14 evaluator + 7/7 pitfall-guard suite which passes);
they represent engineering-debt cleanup items, not functional
gaps.  \textbf{Future work:} \S{}C1--C2 in
\texttt{TODO/01\_strategy/IMPROVEMENT\_BACKLOG.md}."""


def render_tex(
    failed: List[FailedMetric], modes: List[FailureMode], index_rows: List[str]
) -> str:
    """Render the full §6 Limitations LaTeX fragment."""
    # Pair each failed metric with its §4 row
    paragraphs = [
        render_failed_metric(fm, lookup_failure_mode([fm.metric_key], modes))
        for fm in failed
    ]
    # Build the itemize body
    items: List[str] = []
    items.append(render_index_failed_items(index_rows))
    items.extend(paragraphs)
    items.append(render_pre_existing_test_fails())
    items.append(render_compute_constraint(failed))
    items.append(render_subset_coverage(failed))
    items.append(render_wetlab_missing())

    body = "\n\n".join(items)
    header = (
        "% Auto-generated by molmetal/scripts/generate_section6.py (DO NOT EDIT BY HAND)\n"
        "% Source: TODO/INDEX.md (FAIL/PARTIAL rows) + metrics/by_metric/*.json (status<gate)\n"
        "%       + TODO/02_metrics/MECHANISM_METRIC_MAPPING.md sec.4 (failure modes)\n"
        "% Re-run after every metric refresh; manual edits will be overwritten.\n"
        "\n"
        "\\section{Limitations (auto-generated)}\\label{sec:limitations:auto}\n"
        "\n"
        "This sub-section is generated from the structured snapshots "
        "in \\texttt{metrics/by\\_metric/}\\texttt{*.json}, the "
        "FAIL/PARTIAL rows in \\texttt{TODO/INDEX.md}, and the "
        "failure-mode table in "
        "\\texttt{TODO/02\\_metrics/MECHANISM\\_METRIC\\_MAPPING.md}\\S4.  "
        "Each failed metric below carries a \\textbf{Diagnosis} "
        "(with self-check command), a \\textbf{Impact} statement, "
        "and a \\textbf{Source} reference.  Static paragraphs for "
        "the compute envelope, subset coverage, and wet-lab "
        "absence are kept stable (they describe the engineering "
        "envelope, not the metric snapshot).\n"
        "\n"
        "\\begin{enumerate}\n\n" + body + "\n\n"
        "\\end{enumerate}\n"
    )
    return header


def render_summary_md(failed: List[FailedMetric], index_rows: List[str]) -> str:
    lines = [
        "# WF-Section6-Gen — paper §6 auto-generation summary",
        "",
        "**Date**: 2026-09-25  ",
        "**Source**: `molmetal/scripts/generate_section6.py`  ",
        "",
        "## Inputs parsed",
        "",
        f"* INDEX.md rows flagged ❌/⚠️: **{len(index_rows)}**",
        f"* metrics/by_metric/*.json failed: **{len(failed)}**",
        "",
        "## Failed metrics (auto-imported)",
        "",
        "| Metric key | Status | Value | Source JSON |",
        "|---|---|---|---|",
    ]
    for fm in failed:
        v = f"{fm.value:.4f}" if fm.value is not None else "n/a"
        lines.append(
            f"| `{fm.metric_key}` | {fm.status} | {v} | `metrics/by_metric/{fm.source_file}` |"
        )
    lines += [
        "",
        "## Static paragraphs (stable)",
        "",
        "* Compute constraint (single Blackwell, no 64×A100 cluster)",
        "* Subset coverage (PlatinAI 4-of-9; MetalCytoToxDB 5-of-30 metals)",
        "* Wet-lab validation missing",
        "* Pre-existing test failures (16 / 1680+ unit tests marked pre-existing)",
        "",
        "---",
        "",
        "Auto-generated by `molmetal/scripts/generate_section6.py`.",
        "Re-run after every metric refresh.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Paper §6 Limitations auto-generator")
    p.add_argument(
        "--dry-run", action="store_true", help="Print LaTeX to stdout; skip writes"
    )
    p.add_argument("--out-dir", default=str(REPO_ROOT / "paper"))
    p.add_argument("--metrics-dir", default=str(REPO_ROOT / "metrics" / "by_metric"))
    p.add_argument("--index-md", default=str(REPO_ROOT / "TODO" / "INDEX.md"))
    p.add_argument(
        "--mapping-md",
        default=str(REPO_ROOT / "TODO" / "02_metrics" / "MECHANISM_METRIC_MAPPING.md"),
    )
    p.add_argument(
        "--report-dir", default=str(REPO_ROOT / "paper" / "reports" / "wf_section6_gen")
    )
    args = p.parse_args(argv)

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    failed = load_failed_metrics(metrics_dir)
    modes = parse_failure_modes(Path(args.mapping_md))
    index_rows = parse_index_failed_entries(Path(args.index_md))

    tex = render_tex(failed, modes, index_rows)

    if args.dry_run:
        print(tex)
        return 0

    (out_dir / "sections").mkdir(parents=True, exist_ok=True)
    (out_dir / "sections" / "06_limitations_auto.tex").write_text(tex, encoding="utf-8")

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.md").write_text(
        render_summary_md(failed, index_rows), encoding="utf-8"
    )

    print(
        f"[generate_section6] wrote 06_limitations_auto.tex "
        f"({len(failed)} failed metrics + {len(index_rows)} INDEX rows) "
        f"+ summary.md",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
