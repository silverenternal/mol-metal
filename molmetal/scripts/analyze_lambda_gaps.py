"""Wf-72 Phase 1 — L13-L16 Lambda gap signal extractor.

CPU-only docs analysis. Parses R20/R21 close-out files + Wf-44 root-cause catalog
+ Wf-42 mock inventory + INDEX.md §17 to extract *gap signals* — recurring
honest-negative findings, structural ceilings, deferred GPU phases, and unmet
gates that no current L1-L12 row in LAMBDA_ROADMAP §4.2 addresses directly.

Outputs:
  - molmetal/reports/wf72_l13_l16/gap_signals.json
  - molmetal/reports/wf72_l13_l16/synthesis.md  (machine-readable summary)

Per party-twenty-spirit §3.5: 0-tolerance honest framing — no fabricated MEASURED
numbers, no PROJECTED numbers labeled as MEASURED, no GPU/wet-lab phases
silently claimed ship.

Usage (CPU-only):
    python molmetal/scripts/analyze_lambda_gaps.py \\
        --repo-root /home/hugo/codes/mol-metal \\
        --out-dir molmetal/reports/wf72_l13_l16

Author: Wf-72 ultracode agent (Phase 1, 2026-09-26).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------- Constants (per TODO/03_plans/LAMBDA_ROADMAP.md §4.2) ----------

# L1-L12 already documented in LAMBDA_ROADMAP §4.2.
EXISTING_LAMBDAS = list(range(1, 13))

# Phrases that signal an honest-negative or unresolved gate. Each pattern
# appears in ≥ 1 R20 close-out file and signals a gap the L1-L12 grid does
# not address directly.
# pattern -> (signal_kind, target_candidate)
# signal_kind ∈ {"structural_ceiling", "deferred_phase", "unmet_gate",
#                "design_gap", "tier1_metric_undefined"}
# target_candidate ∈ {"L13", "L14", "L15", "L16"} (explicit, not regex match)
GAP_SIGNAL_PATTERNS: Dict[str, List[Tuple[str, str, str]]] = {
    "structural_ceiling": [
        # Reachability ~0.51-0.59 vs gate 0.95 -> L13 (multi-metal coord)
        (r"0\.5[0-9]{2,3}", "reachability_structural_ceiling", "L13"),
        # valence-BNF / dative-bond rejection -> L13 (coord chemistry-aware)
        (r"valence-BNF|dative.bond", "dative_bond_parser", "L13"),
        # closure depth=1 ceiling -> L13 (hierarchical relaxation)
        (r"max_depth\s*=\s*1", "closure_relaxation_depth2", "L13"),
    ],
    "deferred_phase": [
        # GPU retrain deferred / wet-lab deferred -> L14 (GNN-ODE kinetics
        # would replace static bench with dynamic)
        (r"GPU retrain.*deferred|retrain.*deferred", "gpu_retrain_deferred", "L14"),
        # in-flight R20-P1 -> L15 (Pareto-front replaces single-scalar opt)
        (r"in\s*flight", "in_flight_r20_p1", "L15"),
        # OOD Phase 2 NOT re-measured -> L16 (wet-lab-in-loop would close)
        (r"NOT\s+RE-MEASURED", "ood_phase2_unmeasured", "L16"),
        # Wet-lab cooperation deferred 6 months -> L16 (bayes-opt closes loop)
        (r"wet-lab.*deferred|deferred.*wet-lab", "wetlab_deferred", "L16"),
    ],
    "unmet_gate": [
        # gate NOT MET (general) -> L13 (structural)
        (r"NOT\s+MET", "gate_unmet", "L13"),
        # reachability gate 0.95 -> L13 (coord geometry structural)
        (r"gate\s+0\.95", "reachability_gate_unmet", "L13"),
        # MetalCytoToxDB 0.0300 floor -> L16 (need wet-lab to validate)
        (r"0\.0300", "metalcyt_ood_floor", "L16"),
    ],
    "design_gap": [
        # m18 multi-modal binding UNDEFINED -> L15 (Pareto multi-obj)
        (
            r"m18.*UNDEFINED|UNDEFINED.*m18|protein.metal binding",
            "m18_multimodal_undefined",
            "L15",
        ),
        # m19 LOMO UNDEFINED -> L15 (Pareto over toxicity/efficacy/ADMET)
        (
            r"m19.*UNDEFINED|UNDEFINED.*m19|multi.metal LOMO",
            "m19_lomo_undefined",
            "L15",
        ),
        # m20 tumor UNDEFINED -> L15 (Pareto tumor-specific)
        (r"m20.*UNDEFINED|UNDEFINED.*m20|tumor.specific", "m20_tumor_undefined", "L15"),
        # multi-objective Pareto references -> L15
        (r"Pareto|multi.objective", "pareto_design_gap", "L15"),
        # kinetic / GNN-ODE references -> L14
        (r"kinetic|GNN.ODE|graph neural ODE", "kinetics_design_gap", "L14"),
        # hierarchical / multi-metal coordination -> L13
        (r"hierarchical|multi.metal coordination", "hierarchical_design_gap", "L13"),
        # wet-lab-in-the-loop / Bayesian opt -> L16
        (
            r"wet-lab-in-the-loop|Bayesian optimization|in.silico.vs.wet",
            "bayes_opt_design_gap",
            "L16",
        ),
    ],
    "tier1_metric_undefined": [
        # All m18/m19/m20 UNDEFINED cluster -> L15
        (r"m18.*UNDEFINED|UNDEFINED.*m18", "m18_metric_undefined", "L15"),
        (r"m19.*UNDEFINED|UNDEFINED.*m19", "m19_metric_undefined", "L15"),
        (r"m20.*UNDEFINED|UNDEFINED.*m20", "m20_metric_undefined", "L15"),
    ],
}

# Gap candidates L13-L16 — derived from R20/R21 ship honest-negative signals.
GAP_CANDIDATES: Dict[str, Dict] = {
    "L13": {
        "title": "Hierarchical multi-metal coordination geometry",
        "signal_kind": "design_gap",
        "evidence_files": [
            "TODO/completed/91_wf41_3way_integration.md",
            "TODO/completed/95_r20_p4_3way_100pocket.md",
        ],
        "evidence_summary": (
            "L10 (Deflexpressor + Multi-PH + ASTSR 3-way) closes at synthetic "
            "R²=-0.36 (Wf-41 P1) and Wf-46 T5 triangle R²=+0.58; R20-P4 "
            "real pIC50 R²≥+0.30 NOT executed. No current gap addresses "
            "hierarchical coordination geometry (4d/5d orbitals + multi-metal "
            "cluster centers) — single-metal-center closure only."
        ),
        "priority": "HIGH",
        "r_first_close": "R22-P4",
        "depends_on": ["L1", "L6", "L10"],
    },
    "L14": {
        "title": "Graph Neural ODE for metallodrug kinetics / dynamic bond lability",
        "signal_kind": "design_gap",
        "evidence_files": [
            "TODO/completed/93_r20_p2_hilbert_astsr_gpu.md",
            "TODO/completed/96_r20_p5_closure_catherine.md",
        ],
        "evidence_summary": (
            "L5 (Hilbert-ASTSR) measures static R² on pIC50 but does not "
            "model kinetic lability (ligand-exchange rates, hydrolysis t1/2). "
            "Wf-67 P2 reachability ceiling 0.5962 reveals static rule library "
            "insufficient for dynamic coordination chemistry. Graph Neural "
            "ODE adds continuous-time dynamics to ASTSR signature."
        ),
        "priority": "MEDIUM",
        "r_first_close": "R23-P2",
        "depends_on": ["L5", "L7"],
    },
    "L15": {
        "title": "Multi-objective Pareto front over toxicity / efficacy / ADMET",
        "signal_kind": "design_gap",
        "evidence_files": [
            "TODO/INDEX.md",
            "TODO/completed/85_mock_audit.md",
        ],
        "evidence_summary": (
            "Wf-42 mock audit catalogues 8 critical + 23 major + 98 minor mock "
            "sites; m18/m19/m20 tier-1 metrics all WEAK_REAL or UNDEFINED per "
            "INDEX §17.1. No current gap jointly optimizes toxicity + efficacy "
            "+ ADMET — each is treated as a single scalar. Pareto-front "
            "extension would unify 22 metrics × 5 categories (7 STRONG_REAL "
            "+ 11 WEAK_REAL + 5+ MOCK + ~6 UNDEFINED) into a single "
            "dominance-aware search."
        ),
        "priority": "MEDIUM",
        "r_first_close": "R22-P3",
        "depends_on": ["L3", "L8"],
    },
    "L16": {
        "title": "Wet-lab-in-the-loop Bayesian optimization (m18/m19/m20 validation)",
        "signal_kind": "design_gap",
        "evidence_files": [
            "TODO/pending/97_r20_p6_wetlab_ic50.md",
            "TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md",
        ],
        "evidence_summary": (
            "R20-P6 (TODO-97) wet-lab IC50 cooperation with ORNL/PKU/协和 "
            "deferred 6 months; in-silico-vs-wet-lab agreement Pearson r ≥ 0.7 "
            "target not started. No current gap closes the in-silico → "
            "wet-lab feedback loop. Bayesian optimization over D-MPNN "
            "predictions would close m18/m19/m20 tier-1 metric ceiling."
        ),
        "priority": "HIGH",
        "r_first_close": "R23-P3",
        "depends_on": ["L13", "L15"],
    },
}


@dataclass
class GapSignal:
    """A single signal extracted from a close-out file."""

    source_file: str
    signal_kind: str
    matched_pattern: str
    line_excerpt: str
    candidate_L: Optional[str] = None


@dataclass
class GapAnalysisResult:
    """Aggregated result for one Lambda gap candidate."""

    candidate: str
    title: str
    priority: str
    r_first_close: str
    signal_count: int = 0
    supporting_files: List[str] = field(default_factory=list)
    matched_kinds: Dict[str, int] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _extract_signals_from_file(
    file_path: Path, max_excerpt_chars: int = 160
) -> List[GapSignal]:
    """Walk a close-out file line-by-line, emit GapSignals per pattern hit."""
    if not file_path.is_file():
        return []
    text = _safe_read_text(file_path)
    out: List[GapSignal] = []
    rel = str(file_path)
    for line_no, raw in enumerate(text.splitlines(), start=1):
        for kind, pattern_list in GAP_SIGNAL_PATTERNS.items():
            for pattern, _, target_candidate in pattern_list:
                if re.search(pattern, raw, flags=re.IGNORECASE):
                    excerpt = raw.strip()
                    if len(excerpt) > max_excerpt_chars:
                        excerpt = excerpt[: max_excerpt_chars - 3] + "..."
                    out.append(
                        GapSignal(
                            source_file=f"{rel}:L{line_no}",
                            signal_kind=kind,
                            matched_pattern=pattern,
                            line_excerpt=excerpt,
                            candidate_L=target_candidate,
                        )
                    )
    return out


def _scan_r20_closeouts(repo_root: Path) -> List[Path]:
    """Collect R20 close-out files + Wf-44 root-cause catalog."""
    candidates: List[Path] = []
    completed = repo_root / "TODO" / "completed"
    if not completed.is_dir():
        return candidates
    for name in [
        "92_r20_p1_variance_puct_lazy_rdkit.md",
        "93_r20_p2_hilbert_astsr_gpu.md",
        "94_r20_p3_multiph_pll_feature.md",
        "95_r20_p4_3way_100pocket.md",
        "96_r20_p5_closure_catherine.md",
        "97_r20_p6_wetlab_ic50_dryrun.md",
        "98_wf44_closure_ood_phase2.md",
        "99_r20_skill_ecosystem.md",
        "100_r20_r21_consolidation.md",
        "100_wf71_p3_wf49_cross_round2.md",
        "87j_wf44_root_cause_fixes.md",
        "85_mock_audit.md",
    ]:
        path = completed / name
        if path.is_file():
            candidates.append(path)
    # Wf-70 R20 status ledger (Phase 1 of Wf-70)
    ledger = completed / "r20_status_ledger_2026_09_26.md"
    if ledger.is_file():
        candidates.append(ledger)
    return candidates


def _scan_index_section17(repo_root: Path) -> List[Path]:
    """INDEX.md is also a gap-signal source (22 metrics × 5 categories)."""
    path = repo_root / "TODO" / "INDEX.md"
    return [path] if path.is_file() else []


def _scan_lambda_roadmap(repo_root: Path) -> List[Path]:
    """LAMBDA_ROADMAP §4.2 L1-L12 grid also emits design-gap signals."""
    path = repo_root / "TODO" / "03_plans" / "LAMBDA_ROADMAP.md"
    return [path] if path.is_file() else []


def aggregate_signals(
    repo_root: Path,
) -> Tuple[Dict[str, GapAnalysisResult], dict]:
    """Walk all R20 close-out files; bucket GapSignals by candidate_L."""
    files = (
        _scan_r20_closeouts(repo_root)
        + _scan_index_section17(repo_root)
        + _scan_lambda_roadmap(repo_root)
    )
    all_signals: List[GapSignal] = []
    for f in files:
        all_signals.extend(_extract_signals_from_file(f))

    # Initialize results for L13-L16.
    results: Dict[str, GapAnalysisResult] = {
        c: GapAnalysisResult(
            candidate=c,
            title=GAP_CANDIDATES[c]["title"],
            priority=GAP_CANDIDATES[c]["priority"],
            r_first_close=GAP_CANDIDATES[c]["r_first_close"],
            summary=GAP_CANDIDATES[c]["evidence_summary"],
        )
        for c in GAP_CANDIDATES
    }

    # Bucket signals by candidate_L (now explicit per pattern).
    for sig in all_signals:
        cand = sig.candidate_L
        if cand is None:
            continue
        if cand not in results:
            continue
        r = results[cand]
        r.signal_count += 1
        r.supporting_files.append(sig.source_file)
        r.matched_kinds[sig.signal_kind] = r.matched_kinds.get(sig.signal_kind, 0) + 1

    # Dedupe supporting_files (preserve order).
    for r in results.values():
        seen = set()
        deduped: List[str] = []
        for f in r.supporting_files:
            if f in seen:
                continue
            seen.add(f)
            deduped.append(f)
        r.supporting_files = deduped

    meta = {
        "files_scanned": [str(p.relative_to(repo_root)) for p in files],
        "n_files_scanned": len(files),
        "n_signals_total": len(all_signals),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lambda_grid_existing": EXISTING_LAMBDAS,
        "lambda_grid_after_extension": EXISTING_LAMBDAS + [13, 14, 15, 16],
    }
    return results, meta


def render_signals_json(results: Dict[str, GapAnalysisResult], meta: dict) -> dict:
    """Render the canonical gap_signals.json output."""
    payload: dict = {
        "$schema": "wf72_l13_l16_gap_signals/v1",
        "generated_at_utc": meta["generated_at_utc"],
        "files_scanned": meta["files_scanned"],
        "n_files_scanned": meta["n_files_scanned"],
        "n_signals_total": meta["n_signals_total"],
        "lambda_grid_existing": meta["lambda_grid_existing"],
        "lambda_grid_after_extension": meta["lambda_grid_after_extension"],
        "gap_candidates": {},
    }
    for cand, r in results.items():
        meta_c = GAP_CANDIDATES[cand]
        payload["gap_candidates"][cand] = {
            "title": r.title,
            "priority": r.priority,
            "r_first_close": r.r_first_close,
            "depends_on": meta_c["depends_on"],
            "evidence_summary": r.summary,
            "evidence_files_static": meta_c["evidence_files"],
            "signal_count": r.signal_count,
            "supporting_files_from_scan": r.supporting_files,
            "matched_kinds": r.matched_kinds,
            "honest_framing": (
                "CANDIDATE — based on R20/R21 ship honest-negative findings. "
                "MEASURED signal counts reflect pattern-match hits in close-out "
                "files; they are NOT a guarantee the candidate is the right "
                "shape for the gap. R22-R25 plan owners should re-evaluate."
            ),
        }
    return payload


def render_synthesis_md(results: Dict[str, GapAnalysisResult], meta: dict) -> str:
    """Render a short machine-readable synthesis.md for human review."""
    lines: List[str] = []
    lines.append("# Wf-72 L13-L16 Gap Signals — synthesis\n")
    lines.append(f"Generated UTC: {meta['generated_at_utc']}\n")
    lines.append(f"Files scanned: {meta['n_files_scanned']}\n")
    lines.append(f"Total signals: {meta['n_signals_total']}\n")
    lines.append(
        f"Existing grid: L1-L{meta['lambda_grid_existing'][-1]} "
        f"({len(meta['lambda_grid_existing'])} rows)\n"
    )
    lines.append(
        f"After extension: L1-L{meta['lambda_grid_after_extension'][-1]} "
        f"({len(meta['lambda_grid_after_extension'])} rows)\n\n"
    )

    lines.append(
        "| Candidate | Title | Priority | R first close | Signals | Top kinds |\n"
    )
    lines.append("|---|---|---|---|---:|---|\n")
    for cand, r in results.items():
        kinds = ", ".join(
            f"{k}={v}"
            for k, v in sorted(r.matched_kinds.items(), key=lambda kv: -kv[1])[:3]
        )
        lines.append(
            f"| **{cand}** | {r.title} | {r.priority} | {r.r_first_close} | "
            f"{r.signal_count} | {kinds or '(no scan hits - evidence by file)'} |\n"
        )
    lines.append("\n## Per-candidate supporting files (top 5 each)\n\n")
    for cand, r in results.items():
        lines.append(f"### {cand} — {r.title}\n\n")
        for f in r.supporting_files[:5]:
            lines.append(f"- `{f}`\n")
        if not r.supporting_files:
            lines.append("- (no scan hits — see static evidence_files in JSON)\n")
        lines.append("\n")
    return "".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default="/home/hugo/codes/mol-metal",
        help="path to mol-metal repo root",
    )
    parser.add_argument(
        "--out-dir",
        default="molmetal/reports/wf72_l13_l16",
        help="output dir (relative to repo-root)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results, meta = aggregate_signals(repo_root)
    payload = render_signals_json(results, meta)
    (out_dir / "gap_signals.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "synthesis.md").write_text(
        render_synthesis_md(results, meta), encoding="utf-8"
    )

    # Stdout summary (handy for shell piping).
    print(
        f"[wf72] scanned {meta['n_files_scanned']} files; "
        f"{meta['n_signals_total']} signals; "
        f"{len(results)} candidates."
    )
    for cand, r in results.items():
        print(
            f"[wf72] {cand} priority={r.priority} "
            f"signals={r.signal_count} r_close={r.r_first_close}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
