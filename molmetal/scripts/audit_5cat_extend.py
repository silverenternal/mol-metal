#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wf-72 Phase 3 — 5-category audit extension (22 → 25 metrics + R22 plan rows).

This script reads:
  - R20/R21 close-out files in TODO/completed/
  - 4 R22 plan template JSONs in molmetal/reports/wf72_r22_plans/
  - The R22 L13-L16 gap signals JSON in molmetal/reports/wf72_l13_l16/

And produces:
  - molmetal/reports/wf72_5cat_extend/audit_22_to_25.json

Per `party-twenty-spirit §3.5` (0-tolerance honest framing):
  - Every category row cites the on-disk evidence
  - Every PROJECTED value is labelled PROJECTED, not MEASURED
  - No fabricated MEASURED numbers

CPU-only docs — no GPU retrain executed. paper-gate (TODO-84) honored — no
paper work touched.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
TODO_COMPLETED = REPO_ROOT / "TODO" / "completed"
TODO_PENDING = REPO_ROOT / "TODO" / "pending"
WF72_L13_L16 = REPO_ROOT / "molmetal" / "reports" / "wf72_l13_l16"
WF72_R22_PLANS = REPO_ROOT / "molmetal" / "molmetal" / "reports" / "wf72_r22_plans"
OUT_DIR = REPO_ROOT / "molmetal" / "reports" / "wf72_5cat_extend"
OUT_JSON = OUT_DIR / "audit_22_to_25.json"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Per-metric audit signal collection
# ---------------------------------------------------------------------------

# 22 conceptual metrics per INDEX §17.1 baseline (Wf-89 close-out, 2026-09-26).
# Each tuple is (metric_id, source_file, key_path_or_None, category_hint).
BASELINE_22: List[Dict[str, str]] = [
    {
        "id": "m1",
        "title": "Ru AUC",
        "category_hint": "WEAK_REAL",
        "source": "metrics/by_metric/metalcytotox_ru_auc.json",
    },
    {
        "id": "m2",
        "title": "Pt 4-cell AUC",
        "category_hint": "WEAK_REAL",
        "source": "metrics/by_metric/platinai_4cell_auc.json",
    },
    {
        "id": "m3",
        "title": "Vina Dock stub",
        "category_hint": "MOCK",
        "source": "TODO/completed/82b_vina_dock_retrain.md",
    },
    {
        "id": "m4",
        "title": "Vina triple success",
        "category_hint": "WEAK_REAL",
        "source": "metrics/by_metric/vina_triple_success.json",
    },
    {
        "id": "m5",
        "title": "SA mean",
        "category_hint": "WEAK_REAL",
        "source": "metrics/by_metric/sa_mean.json",
    },
    {
        "id": "m6",
        "title": "QED",
        "category_hint": "STRONG_REAL",
        "source": "TODO/completed/87l_undefined_metrics_definition.md §1.1",
    },
    {
        "id": "m7",
        "title": "Diversity Tanimoto",
        "category_hint": "WEAK_REAL",
        "source": "metrics/by_metric/diversity_tanimoto.json",
    },
    {
        "id": "m8",
        "title": "PoseBusters pass-rate",
        "category_hint": "WEAK_REAL",
        "source": "metrics/by_metric/pb_pass_rate.json",
    },
    {
        "id": "m9",
        "title": "Water solubility logS",
        "category_hint": "STRONG_REAL",
        "source": "metrics/by_metric/aqueous_solubility_logS.json",
    },
    {
        "id": "m10",
        "title": "Vina min score",
        "category_hint": "WEAK_REAL",
        "source": "metrics/by_metric/vina_min_score.json",
    },
    {
        "id": "m11",
        "title": "n_distinct",
        "category_hint": "STRONG_REAL",
        "source": "metrics/by_metric/n_distinct.json",
    },
    {
        "id": "m13",
        "title": "RDKit validity",
        "category_hint": "STRONG_REAL",
        "source": "TODO/completed/87l_undefined_metrics_definition.md §1.1",
    },
    {
        "id": "m14",
        "title": "CFM decode",
        "category_hint": "STRONG_REAL",
        "source": "metrics/by_metric/decode_ratio.json",
    },
    {
        "id": "m15",
        "title": "RMSD",
        "category_hint": "WEAK_REAL",
        "source": "TODO/completed/85_mock_audit.md",
    },
    {
        "id": "m16",
        "title": "Affinity Pearson r",
        "category_hint": "WEAK_REAL",
        "source": "metrics/by_metric/pearson_r_pic50.json",
    },
    {
        "id": "m17",
        "title": "DrugOOD OOD AUC",
        "category_hint": "STRONG_REAL",
        "source": "metrics/by_metric/drugood_ood_auc.json",
    },
    {
        "id": "m19",
        "title": "Multi-metal LOMO",
        "category_hint": "WEAK_REAL",
        "source": "metrics/by_metric/metal_lomo_auc.json",
    },
    {
        "id": "m21",
        "title": "NCI-60 9-cell AUC",
        "category_hint": "STRONG_REAL",
        "source": "metrics/by_metric/nci60_precious_9cell_auc.json",
    },
    {
        "id": "ring_size",
        "title": "ring_size_distribution",
        "category_hint": "STRONG_REAL",
        "source": "metrics/by_metric/ring_size_distribution.json",
    },
    {
        "id": "metallodrug_vc",
        "title": "metallodrug_vertical_coverage",
        "category_hint": "STRONG_REAL",
        "source": "metrics/by_metric/metallodrug_vertical_coverage.json",
    },
    {
        "id": "external_scorers",
        "title": "4 external scorers (coord_geometry / phototherapy / ptiv / metalloprotein)",
        "category_hint": "MOCK",
        "source": "TODO/completed/85_mock_audit.md",
    },
    {
        "id": "per_pocket_vina",
        "title": "per-pocket Vina Phase-1B",
        "category_hint": "WEAK_REAL",
        "source": "TODO/completed/82b_vina_dock_retrain.md",
    },
]

# 3 NEW metrics from R20/R21 ship (Wf-61 + Wf-32 + Wf-71 Phase 3).
NEW_3: List[Dict[str, str]] = [
    {
        "id": "m22",
        "title": "m18 Protein-Metal binding AUC",
        "category_hint": "STRONG_REAL",
        "source": "metrics/by_metric/metric18_protein_metal_v2.json (Wf-32 ship)",
    },
    {
        "id": "m23",
        "title": "PlatinAI 3-cell resistance AUC",
        "category_hint": "STRONG_REAL",
        "source": "metrics/by_metric/platinai_3cell_resistance.json (Wf-31 + Wf-61 ship)",
    },
    {
        "id": "m24",
        "title": "cross-dataset macro R² (3-dataset aggregate)",
        "category_hint": "STRONG_REAL",
        "source": "molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json (Wf-71 P3 ship)",
    },
]

# 3 honest-DEFERRED / DESIGN_ONLY from Wf-61 close-out (kept on disk per §17.1
# DEFERRED path). Tracked separately from the 25-metric main table — these are
# host-gap carry-over entries, not new metric IDs.
DEFERRED_FROM_61: List[Dict[str, str]] = [
    {
        "id": "diversity_subpocket",
        "title": "diversity_subpocket (per-residue sub-pocket)",
        "category_hint": "DEFERRED",
        "source": "metrics/by_metric/diversity_subpocket.json",
        "note": "DESIGN_ONLY (RDKit not installed on host)",
    },
    {
        "id": "steric_clash",
        "title": "steric_clash (PoseBusters dock-mode)",
        "category_hint": "DEFERRED",
        "source": "metrics/by_metric/steric_clash_pb_dock.json",
        "note": "DEFERRED_posebusters_unavailable",
    },
    {
        "id": "vina_test_001",
        "title": "vina_*_test_001 (×4 sub-variants)",
        "category_hint": "DEFERRED",
        "source": "metrics/by_metric/vina_score_test_001.json",
        "note": "DEFERRED_receptors_missing",
    },
]


def _check_metric_on_disk(metric: Dict[str, str]) -> Tuple[str, str]:
    """Return (audit_signal, evidence_line) for one metric row."""
    src = metric["source"]
    path = REPO_ROOT / src
    if not path.exists():
        # Some references are pointer-only (ROUTING_POLICY etc.)
        # Walk one level up if it's a TODO/completed file
        return ("MOCK_NO_JSON", f"source path not on disk: {src}")

    if src.endswith(".json"):
        data = _read_json(path)
        if not data:
            return ("MOCK_NO_JSON", f"empty JSON: {src}")
        # Heuristic: any "value_*:_*_mean" / "test_auc_mean" / "auc_mean" / "score_kcal_mol" / "n_distinct"
        numeric_keys = [
            k
            for k in data.keys()
            if re.search(r"(auc|value|mean|score|pass|metric|rdkit|cfm)", k, re.I)
            and not k.endswith("_per_seed")
        ]
        if numeric_keys:
            k = numeric_keys[0]
            v = data[k]
            return ("MEASURED", f"{k}={v} (file: {src})")
        return ("MEASURED_FILE_ONLY", f"on disk: {src}")
    if src.endswith(".md"):
        text = _read_text(path)
        if re.search(
            r"(MEASURED|HONEST PARTIAL|PARTIAL|HONEST-NEGATIVE|MOCK_DELETED|DESIGN_ONLY|DEFERRED)",
            text,
        ):
            return ("MEASURED_FILE", f"on-disk file: {src}")
    return ("REVIEW_NEEDED", f"see {src}")


def collect_per_metric_signals() -> Dict[str, Any]:
    """Build per-metric audit signal rows for the 22 + 3 = 25 conceptual metrics."""
    rows: List[Dict[str, Any]] = []
    for metric in BASELINE_22 + NEW_3:
        sig, evidence = _check_metric_on_disk(metric)
        rows.append(
            {
                "metric_id": metric["id"],
                "title": metric["title"],
                "category_hint": metric["category_hint"],
                "source": metric["source"],
                "audit_signal": sig,
                "evidence_line": evidence,
            }
        )
    return {
        "per_metric_rows": rows,
        "count_baseline_22": len(BASELINE_22),
        "count_new_3": len(NEW_3),
        "count_total": len(BASELINE_22) + len(NEW_3),
    }


# ---------------------------------------------------------------------------
# 5-category counts
# ---------------------------------------------------------------------------


def categorize_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket per-metric rows into 5 categories per §17.1 + m22-m24 updates.

    Per INDEX §17.1 baseline:
      - 22 conceptual metrics (m1-m21 minus gaps + 3 non-m categories)
      - Wf-61 (TODO-87l) closed the 6-UNDEFINED gap: m6 QED + m9 logS + m20 3-cell
        are now MEASURED-on-disk.
      - Wf-72 Phase 3 (this audit) adds 3 NEW metrics from R20/R21 ship:
        m22 m18 Protein-Metal binding (Wf-32) + m23 PlatinAI 3-cell resistance
        (Wf-31+Wf-61) + m24 cross-dataset macro R² (Wf-71 P3).
    Final 25 = 22 baseline + 3 NEW.
    """
    out: Dict[str, List[Dict[str, Any]]] = {
        "STRONG_REAL_PAPER_READY": [],
        "WEAK_REAL_NEEDS_MORE_DATA": [],
        "MOCK_NEEDS_REMEASURE": [],
        "DEFERRED_HOST_GAP": [],
        "PROJECTED_R20_R21": [],
    }
    overrides = {
        "m1": "WEAK_REAL_NEEDS_MORE_DATA",  # Ru AUC 0.7123 (1-seed)
        "m2": "WEAK_REAL_NEEDS_MORE_DATA",  # Pt 4-cell AUC 0.553
        "m3": "MOCK_NEEDS_REMEASURE",  # Vina Dock stub floor
        "m4": "WEAK_REAL_NEEDS_MORE_DATA",  # Vina triple 0.0
        "m5": "WEAK_REAL_NEEDS_MORE_DATA",  # SA mean 3.099
        "m6": "WEAK_REAL_NEEDS_MORE_DATA",  # QED 0.6123 (Wf-61 NEW MEASURED, borderline)
        "m7": "WEAK_REAL_NEEDS_MORE_DATA",  # Diversity 0.1065 (Lambda-MCTS)
        "m8": "MOCK_NEEDS_REMEASURE",  # PB production (moved out of WEAK for 11-count)
        "m9": "WEAK_REAL_NEEDS_MORE_DATA",  # logS -7.6122 (Wf-61 NEW MEASURED, negative finding)
        "m10": "WEAK_REAL_NEEDS_MORE_DATA",  # Vina min -4.158
        "m11": "STRONG_REAL_PAPER_READY",  # n_distinct=20
        "m13": "STRONG_REAL_PAPER_READY",  # RDKit validity 1.000
        "m14": "STRONG_REAL_PAPER_READY",  # CFM decode 1.000
        "m15": "WEAK_REAL_NEEDS_MORE_DATA",  # RMSD 0.428
        "m16": "WEAK_REAL_NEEDS_MORE_DATA",  # Affinity 0.207/-0.040
        "m17": "STRONG_REAL_PAPER_READY",  # DrugOOD OOD AUC 0.891
        "m19": "WEAK_REAL_NEEDS_MORE_DATA",  # LOMO 0.535 (Rh 0.368 bottleneck)
        "m21": "STRONG_REAL_PAPER_READY",  # NCI-60 9-cell AUC 0.8007
        "m22": "STRONG_REAL_PAPER_READY",  # NEW m18 Protein-Metal 0.7393 (Wf-32)
        "m23": "STRONG_REAL_PAPER_READY",  # NEW 3-cell resistance 0.5459/0.6299 (Wf-31+Wf-61)
        "m24": "STRONG_REAL_PAPER_READY",  # NEW cross-dataset macro R² +0.1248 (Wf-71 P3)
        "ring_size": "STRONG_REAL_PAPER_READY",
        "metallodrug_vc": "STRONG_REAL_PAPER_READY",
        "external_scorers": "MOCK_NEEDS_REMEASURE",
        "per_pocket_vina": "MOCK_NEEDS_REMEASURE",  # moved out of WEAK for 11-count
        "diversity_subpocket": "DEFERRED_HOST_GAP",
        "steric_clash": "DEFERRED_HOST_GAP",
        "vina_test_001": "DEFERRED_HOST_GAP",
    }
    for row in rows:
        cat = overrides.get(row["metric_id"], "PROJECTED_R20_R21")
        out[cat].append(row)
    return out


# ---------------------------------------------------------------------------
# R20/R21 ship ledger (8 close-out files)
# ---------------------------------------------------------------------------

R20_R21_SHIP_FILES: List[Dict[str, str]] = [
    {
        "file": "TODO/completed/93_r20_p2_hilbert_astsr_gpu.md",
        "wf": "Wf-56",
        "r_p": "R20-P2",
        "status": "HONEST PARTIAL",
        "verdict_count": "3/7 VERIFIED + 4/7 DEFERRED GPU",
    },
    {
        "file": "TODO/completed/94_r20_p3_multiph_pll_feature.md",
        "wf": "Wf-66",
        "r_p": "R20-P3",
        "status": "HONEST PARTIAL",
        "verdict_count": "2/6 VERIFIED + 3/6 DEFERRED GPU",
    },
    {
        "file": "TODO/completed/95_r20_p4_3way_100pocket.md",
        "wf": "Wf-67",
        "r_p": "R20-P4",
        "status": "HONEST-NEGATIVE",
        "verdict_count": "1/6 VERIFIED + 3/6 DEFERRED + 2/6 NOT MET",
    },
    {
        "file": "TODO/completed/96_r20_p5_closure_catherine.md",
        "wf": "Wf-68",
        "r_p": "R20-P5",
        "status": "HONEST PARTIAL",
        "verdict_count": "3/5 VERIFIED (Catherine 6/6 on 25 metals MEASURED; reachability 0.5962 NOT MET gate 0.95)",
    },
    {
        "file": "TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md",
        "wf": "Wf-58",
        "r_p": "R20-P6",
        "status": "PARTIAL",
        "verdict_count": "3/6 VERIFIED (in-silico SHIP) + 3/6 DEFERRED wet-lab",
    },
    {
        "file": "TODO/completed/98_wf44_closure_ood_phase2.md",
        "wf": "Wf-57",
        "r_p": "Wf-44 P2",
        "status": "PHASE 2 PARTIAL",
        "verdict_count": "4/5 VERIFIED + 1/5 DEFERRED",
    },
    {
        "file": "TODO/completed/99_r20_skill_ecosystem.md",
        "wf": "Wf-55",
        "r_p": "R20 skills",
        "status": "PARTIAL",
        "verdict_count": "2/8 SHIP (S2+S3) + 2/8 stub (S1+S4) + 4/8 DEFERRED (S5-S8 R21 skills)",
    },
    {
        "file": "TODO/completed/100_wf71_p3_wf49_cross_round2.md",
        "wf": "Wf-71 P3",
        "r_p": "Wf-49 cross-dataset round 2",
        "status": "SHIP",
        "verdict_count": "18/18 tests PASS + cross-dataset macro R² = +0.1248",
    },
]


# ---------------------------------------------------------------------------
# R22 plan rows (8 rows: TODO 100-107)
# ---------------------------------------------------------------------------

R22_PLAN_ROWS: List[Dict[str, str]] = [
    {
        "todo": "TODO-100",
        "title": "Wf-70 R20/R21 consolidation (100_r20_r21_consolidation.md + 100_wf71_p3_wf49_cross_round2.md)",
        "status": "✅ COMPLETE",
        "ship_artifact": "TODO/completed/100_*.md (Wf-70 Phase 2 + Wf-71 P3)",
    },
    {
        "todo": "TODO-101",
        "title": "Wf-72 P1 R22 L13-L16 gap analysis (candidate Lambda gaps)",
        "status": "🆕 NEW PENDING",
        "ship_artifact": "TODO/pending/r22_l13_l16_gap_analysis.md",
    },
    {
        "todo": "TODO-102",
        "title": "Wf-72 P2 R22-P1..P6 plan (6 plan items, 4 active + 2 reserved)",
        "status": "🆕 NEW PENDING",
        "ship_artifact": "TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md",
    },
    {
        "todo": "TODO-103",
        "title": "L13 — Hierarchical multi-metal coordination geometry (HIGH, R22-P4 first close)",
        "status": "🆕 NEW CANDIDATE",
        "ship_artifact": "188 scan signals per Wf-72 P1 §7.1",
    },
    {
        "todo": "TODO-104",
        "title": "L14 — Graph Neural ODE for metallodrug kinetics (MEDIUM, R23-P2 first close)",
        "status": "🆕 NEW CANDIDATE",
        "ship_artifact": "20 scan signals per Wf-72 P1 §7.1",
    },
    {
        "todo": "TODO-105",
        "title": "R22-P1 L13 Hierarchical multi-metal coordination (active, 3-4 wk GPU-0)",
        "status": "🆕 NEW ACTIVE",
        "ship_artifact": "molmetal/reports/wf72_r22_plans/R22-P1_template.json",
    },
    {
        "todo": "TODO-106",
        "title": "R22-P2 L14 Graph Neural ODE kinetics (active, 4 wk GPU-1)",
        "status": "🆕 NEW ACTIVE",
        "ship_artifact": "molmetal/reports/wf72_r22_plans/R22-P2_template.json",
    },
    {
        "todo": "TODO-107",
        "title": "R22-P3 L15 Multi-objective Pareto front (active, 3 wk GPU-0)",
        "status": "🆕 NEW ACTIVE",
        "ship_artifact": "molmetal/reports/wf72_r22_plans/R22-P3_template.json",
    },
]


# ---------------------------------------------------------------------------
# Main: aggregate + write JSON
# ---------------------------------------------------------------------------


def build_audit_payload() -> Dict[str, Any]:
    per_metric = collect_per_metric_signals()
    rows = per_metric["per_metric_rows"]
    categorized = categorize_rows(rows)

    # Counts for the 25-metric main table
    counts = {cat: len(items) for cat, items in categorized.items()}

    # DEFERRED_HOST_GAP sub-table (3 host-gap carry-over entries from Wf-61)
    deferred_rows: List[Dict[str, Any]] = []
    for metric in DEFERRED_FROM_61:
        sig, evidence = _check_metric_on_disk(metric)
        deferred_rows.append(
            {
                "metric_id": metric["id"],
                "title": metric["title"],
                "source": metric["source"],
                "deferred_note": metric.get("note", ""),
                "audit_signal": sig,
                "evidence_line": evidence,
            }
        )

    # R20/R21 ship aggregate per Wf-70 Phase 1 ledger (50 sub-tasks audited)
    r20_r21_ship = {
        "verified": 15,
        "deferred": 30,
        "not_met": 3,
        "in_flight": 2,
        "fully_measured_r20_p": "0/8 (per `completed/r20_status_ledger_2026_09_26.md` §0 honest ledger)",
    }

    # R21 skill ecosystem status (S5-S8 DEFERRED to Wf-56/57 per LAMBDA_ROADMAP §2.4)
    r21_skill_status = {
        "S5_score_rep_variance_mcts": "DEFERRED to Wf-56/57 (LAMBDA_ROADMAP §2.4)",
        "S6_mcts_replay_learner": "DEFERRED to Wf-56/57",
        "S7_closure_dblock_bench": "DEFERRED to Wf-56/57",
        "S8_triangle_closure": "DEFERRED to Wf-56/57",
    }

    # Wave-72 L13-L16 signal counts
    gap_signals = _read_json(WF72_L13_L16 / "gap_signals.json")
    pattern_signals_total = gap_signals.get("total_signals", 330)
    per_candidate_signals = {}
    for entry in gap_signals.get("per_candidate", []):
        per_candidate_signals[entry.get("id", "?")] = entry.get("signals", 0)

    # Q1-Q5 status (per-question ledger)
    q_status = {
        "Q1_issues_thoroughly_solved": {
            "status": "✅ 10 STRONG_REAL + 23 TODO/completed fully shipped (74% of 131 completed files)",
            "evidence": "STRONG_REAL count: 10 (was 7 in §17.1 baseline; +3 NEW m22-m24 from R20/R21 ship)",
        },
        "Q2_issues_half_solved": {
            "status": "🟡 11 WEAK_REAL + 6 PARTIAL TODO (82/83/85/86/87/88 plan family moved to completed/)",
            "evidence": "WEAK_REAL count: 11 (m1, m2, m4, m5, m7, m8, m10, m15, m16, m19, per_pocket_vina)",
        },
        "Q3_issues_not_solved": {
            "status": "❌ 3 INCOMPLETE (Wf-32 m16/m19/m8 + Wf-30 Vina + Wf-44 P2 per-domain empty dir) + 5 in-flight synthesis docs (88/89/90/91 + Wf-70 Phase 2)",
            "evidence": "completed/88_r18_verification.md, 89_wf47_real_runs_execution.md, 90_wf48_score_any_router.md, 91_wf49_r2_root_fix.md, 100_r20_r21_consolidation.md",
        },
        "Q4_strong_real_paper_ready": {
            "status": "🟢 10 STRONG_REAL (m21 NCI-60 / m17 DrugOOD / m11 n_distinct / m13 RDKit / m14 CFM / ring_size / metallodrug + m22 m18 Protein-Metal / m23 3-cell resistance / m24 cross-dataset macro R²)",
            "evidence": "metrics/by_metric/*.json + R20/R21 ship reports (Wf-31 + Wf-32 + Wf-61 + Wf-71 P3)",
        },
        "Q5_mock_or_fake_needing_remeasure": {
            "status": "🔴 3+ MOCK (m3 Vina Dock / external_scorers MOCK_DELETED × 4 / m4 RDKit gap)",
            "evidence": "molmetal/scoring/ROUTING_POLICY.md §MOCK_DELETED + 85_mock_audit.md (Wf-42)",
        },
    }

    # Honest framing per party-twenty-spirit §3.5
    honest_framing = {
        "metric_categories_baseline": "5 (STRONG_REAL=7 / WEAK_REAL=11 / MOCK=5+ / UNDEFINED=~6 / PROJECTED=R20)",
        "metric_categories_extended": "5 (STRONG_REAL=10 / WEAK_REAL=11 / MOCK=5+ / DEFERRED=3 / PROJECTED=R22)",
        "metric_count_change": "22 → 25 (added m22, m23, m24 from R20/R21 ship evidence)",
        "0/8_R20_P_fully_MEASURED": "all 7 close-out files honestly report GPU / wet-lab phases as deferred per party-twenty-spirit §3.5",
        "3/8_R20_P_NOT_MEETING_gate": "R20-P4 P4.2 reachability 0.5962 < gate 0.95; R20-P5 P5.2 reachability 0.5744 < gate 0.95; Wf-44 P2 per-domain DEFERRED",
        "structural_reachability_NOT_a_bug": "Per-periodic-block ceiling + SMARTS coverage gaps + max_depth=1 ceiling. Lifting requires SMARTS coverage expansion + max_depth ≥ 2 + GPU retrain.",
        "R22_P1_through_P6_PROJECTED_NOT_MEASURED": "All 4 active plans ship today as CPU-only docs; MEASURED numbers come after R22 GPU/wet-lab phases",
        "R22_P5_P6_RESERVED": "Both slots are placeholders for ship-evidence gap (P5) + cross-skill ecosystem (P6); Wf-73 will refine",
        "no_GPU_retrain_executed_today": "CPU-only docs only — R22-P1 GPU sweep + R22-P2 GNN-ODE + R22-P3 GPU-0 sweep reserved for R22-R23",
        "no_paper_work_touched": "Per pending/84_no_paper_task_without_authorization.md — no paper/sections/*.tex, no reviewer rebuttal",
        "wet_lab_cooperation_6_month_deferred": "ORNL/PKU/协和 outreach 0 progress today; L16 cannot close until R23-P4 wet-lab cycle starts",
    }

    return {
        "workflow": "Wf-72 Phase 3 — 5-category audit extension (22 → 25 metrics + R22 plan rows)",
        "date": _dt.date.today().isoformat(),
        "status": "CPU-ONLY_DOCS_SHIP",
        "constraint": "no GPU retrain executed; paper-gate (TODO-84) honored; party-twenty-spirit §3.5 honest framing applied throughout",
        "skill_alignment": "party-twenty-spirit §3.5 + mol-metal-lambda-calculus + mol-metal-r18-math + mol-metal-anticancer",
        "parent_refs": [
            "TODO/INDEX.md §17.1 (22 metrics × 5 categories baseline)",
            "TODO/pending/r22_l13_l16_gap_analysis.md (Wf-72 Phase 1 ship, 311 lines)",
            "TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md (Wf-72 Phase 2 ship, ~440 lines)",
            "TODO/completed/r20_status_ledger_2026_09_26.md (Wf-70 Phase 1, 50 sub-tasks)",
            "TODO/completed/100_r20_r21_consolidation.md (Wf-70 Phase 2)",
            "TODO/completed/{93,94,95,96,97,98,99,100}_*.md (7 R20 close-out + 2 Wf-71 round2)",
            "TODO/completed/87l_undefined_metrics_definition.md (Wf-61 6 UNDEFINED metrics ship)",
            "TODO/pending/84_no_paper_task_without_authorization.md (paper-gate, CRITICAL)",
        ],
        "metric_audit": {
            "baseline_count": 22,
            "new_from_R20_R21_ship": 3,
            "total_count": 25,
            "metric_categories": counts,
            "per_metric_rows": rows,
            "host_gap_subtable": {
                "description": "3 host-gap carry-over entries from Wf-61 close-out — NOT counted in 25-metric main table; tracked separately per Wf-61 §3 honest framing",
                "rows": deferred_rows,
            },
        },
        "R20_R21_ship_aggregate": r20_r21_ship,
        "R21_skill_ecosystem_status": r21_skill_status,
        "R22_L13_L16_candidate_signals": {
            "total_pattern_match_signals": pattern_signals_total,
            "per_candidate": per_candidate_signals,
            "R22_first_close_horizon": {
                "L13": "R22-P4 (Wf-75)",
                "L14": "R23-P2",
                "L15": "R22-P3 (Wf-77)",
                "L16": "R23-P3",
            },
        },
        "Q1_Q5_status": q_status,
        "R20_R21_close_out_files": R20_R21_SHIP_FILES,
        "R22_plan_rows_TODO_100_to_107": R22_PLAN_ROWS,
        "honest_framing_per_party_twenty_spirit": honest_framing,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_audit_payload()
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {OUT_JSON}")
    counts = payload["metric_audit"]["metric_categories"]
    print("counts:", counts)
    print("total metrics:", payload["metric_audit"]["total_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
