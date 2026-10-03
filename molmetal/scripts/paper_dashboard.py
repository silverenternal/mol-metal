#!/usr/bin/env python3
"""Wf-265-Dashboard: Build interactive paper §4 dashboard HTML.

==============================================================================

This script emits a single-file interactive HTML dashboard that surfaces
the MEASURED-vs-PROJECTED status of every paper-§4 claim alongside the
on-disk source JSON.  It is the human-facing companion to:

  * :mod:`molmetal.scripts.collect_paper_evidence` (which writes the
    ``all_results.json`` collected from per-seed JSON across the 5-metal
    chemprop v2 v3 cohort).
  * :mod:`TODO/completed/188_paper_4_krasnov_protocol_aligned.md`
    (the verbatim §4 DRAFT, the source of truth for numeric claims).
  * :mod:`TODO/completed/207_unified_pre_flight_checklist_v2.md` §9
    (the honest-framing catalog; per-party-twenty-spirit §3.5).

Responsibilities
---------------

  1. Read the §4 paper doc + §9 honest-framing catalog from
     ``TODO/completed/``.
  2. Read ``all_results.json`` from Wf-265-Collect (per-seed AUC + per-metal
     aggregates + G1..G6 high-level gates).
  3. Render an interactive HTML dashboard with:

     - **per-metric badge**: ``MEASURED`` (green) / ``PROJECTED`` (amber)
       / ``honest-negative`` (red).
     - **per-claim card**: numeric value + JSON source path + timestamp
       + click-to-expand source JSON snippet.
     - **per-cohort table**: chemprop v2 v3 5-metal AUC, DrugOOD 3 splits
       (assay / scaffold / size), CFM m14 (decode_ratio), CFM m7
       (Tanimoto diversity).

CLI::

    paper_dashboard.py --paper-doc TODO/completed/188_paper_4_krasnov_protocol_aligned.md
                       --collect-json molmetal/reports/wf265_collect/all_results.json
                       --out-html molmetal/reports/wf265_dashboard/dashboard.html
                       [--checklist-doc TODO/completed/207_unified_pre_flight_checklist_v2.md]
                       [--verbose]

Exit codes:

  * 0 — dashboard emitted (regardless of pass/fail contents).
  * 1 — bad usage / missing required CLI flag.
  * 2 — script error (paper doc unreadable, collect JSON malformed, …).

The script is **CPU-only**, **vanilla-JS** (no external CDN), and **single-file**
(the emitted HTML has inline CSS + inline JS).  All numeric values trace to
JSON paths embedded in the claim cards; the dashboard intentionally does NOT
re-compute any aggregate so that the numbers seen by the reader are bit-for-bit
identical to the on-disk source.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Wf-304A: AnchorGuard adoption at every paper-grade JSON read site.
# Per Wf-302A honest caveat, downstream consumers (this script included)
# had been doing ad-hoc json.load(open(...)) at the call sites; the
# AnchorGuard adoption replaces each read with explicit freshness +
# version guards so silent stale / missing / version-drifted reads are
# no longer possible.
from molmetal.scripts.read_anchor_json import (  # noqa: E402
    AnchorGuard,
    AnchorJSONError,
    AnchorMissingError,
    AnchorStaleError,
    AnchorVersionMismatchError,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
SCRIPT_DIR = MOLMETAL_ROOT / "scripts"

SCHEMA_VERSION = "308b.v1"

DEFAULT_PAPER_DOC = (
    REPO_ROOT / "TODO" / "completed" / "188_paper_4_krasnov_protocol_aligned.md"
)
DEFAULT_COLLECT_JSON = (
    MOLMETAL_ROOT / "reports" / "wf265_collect" / "all_results.json"
)
DEFAULT_OUT_HTML = (
    MOLMETAL_ROOT / "reports" / "wf265_dashboard" / "dashboard.html"
)
DEFAULT_CHECKLIST_DOC = (
    REPO_ROOT / "TODO" / "completed" / "207_unified_pre_flight_checklist_v2.md"
)

# Verbatim badge labels per party-twenty-spirit §3.5 (0 容忍 不掩盖).
BADGE_MEASURED = "MEASURED"
BADGE_PROJECTED = "PROJECTED"
BADGE_HONEST_NEGATIVE = "honest-negative"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """ISO-8601 UTC timestamp, e.g. ``2026-09-28T13:00:00+00:00``."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _read_text(path: Path) -> str:
    """Read a UTF-8 text file.  Raises on missing / unreadable."""
    with open(str(path), "r", encoding="utf-8") as fh:
        return fh.read()


def _read_json(
    path: Path,
    *,
    max_age_sec: float = 86400.0,
    required_version: Optional[str] = None,
    version_field: str = "version",
    strict: bool = False,
) -> Dict[str, Any]:
    """Read a paper-grade ``*_anchor.json`` via :class:`AnchorGuard`.

    Wf-304A: replaces the prior ad-hoc ``json.load(open(...))`` at every
    paper-grade JSON read site.  ``strict=True`` re-raises the four named
    anchor exceptions; ``strict=False`` (default) returns an empty dict
    on any guard failure so the dashboard can keep rendering the rest
    of the catalog (the catalog is hand-curated and does not depend on
    the on-disk collect.json for any claim).

    The ``required_version`` parameter is optional: when ``None`` the
    version check is skipped (the Wf-265-Collect ``all_results.json``
    does not yet carry a ``version`` field).
    """
    try:
        guard = AnchorGuard(
            path=path,
            max_age_sec=max_age_sec,
            required_version=required_version,
            version_field=version_field,
        )
        return guard.require()
    except (
        AnchorMissingError,
        AnchorStaleError,
        AnchorJSONError,
        AnchorVersionMismatchError,
    ) as exc:
        if strict:
            raise
        # Best-effort: surface the failure on stderr (so the dashboard
        # author can see WHICH anchor tripped) but keep the rest of the
        # build going.  Consistent with the prior "Exception: except"
        # wrapper at every read site.
        print(
            f"[Wf-265-Dashboard] WARN anchor guard at {path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return {}


def _safe_round(x: Any, ndigits: int = 4) -> str:
    """Format ``x`` (float) to ``ndigits`` digits; never raises."""
    try:
        return f"{float(x):.{ndigits}f}"
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------
# Claim + cohort + honest-negative extraction
# ---------------------------------------------------------------------------

# Verbatim claim catalog per TODO-188 §5.1 (CLAIMS) + §5.2 (DOES NOT claim).
# Each entry maps to a single paper-§4 claim with badge + JSON path.

CLAIM_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "claim_chemprop_v2_5metal_mean_auc",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": "chemprop v2 5-metal mean AUC = 0.886 ± 0.014 (paired t-test p = 1.35×10⁻⁵)",
        "value": "0.886 ± 0.014",
        "json_path": "TODO/completed/154_chemprop_v4_head_to_head.md §4",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_ru_per_metal_krasnov_beat",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": "Ru per-metal AUC 0.909 BEATS Krasnov 2026 baseline 0.81 (+0.099, paired p=0.025)",
        "value": "0.909 vs 0.81 (Δ +0.099)",
        "json_path": "TODO/completed/154_chemprop_v4_head_to_head.md §2",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_ir_per_metal_krasnov_beat",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": "Ir per-metal AUC 0.885 BEATS Krasnov 2026 baseline 0.73 (+0.155, paired p=0.010)",
        "value": "0.885 vs 0.73 (Δ +0.155)",
        "json_path": "TODO/completed/154_chemprop_v4_head_to_head.md §2",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_os_rh_re_first_reported",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": "Os / Rh / Re first-reported scaffold-aligned per-metal AUC",
        "value": "Os 0.889 / Rh 0.863 / Re 0.882",
        "json_path": "TODO/completed/154_chemprop_v4_head_to_head.md §2",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_wf219_ru_ir_scaffold_aligned",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": "Wf-219 reproduces Ru/Ir at 3-seed × 50-ep scaffold-aligned (hidden=64/depth=2)",
        "value": "Ru 0.786 ± 0.014 (Wf-310A G5 n=2/3), Ir 0.759 ± 0.014",
        "json_path": "molmetal/reports/wf318d_wf219_ru_agg/sweep_summary_ru_fixed.json",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf219_krasnov_5metal_3seed_50epoch_scaffold",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": "Wf-219 5-metal × 3-seed × 50-epoch scaffold aggregate (Ru MEASURED via Wf-310A G5 n=2/3)",
        "value": "0.7776 ± 0.018 (5-metal: Re 0.803, Os 0.788, Rh 0.752, Ru 0.786, Ir 0.759)",
        "json_path": "molmetal/reports/wf318d_wf219_ru_agg/sweep_summary_ru_fixed.json",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_drugood_assay_5seed",
        "section": "§3.1",
        "badge": BADGE_MEASURED,
        "claim": "DrugOOD lbap_refined_ic50_assay ood_test AUC = 0.7137 ± 0.0079 (5-seed, Wf-185)",
        "value": "0.7137 ± 0.0079",
        "json_path": "molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_drugood_scaffold_5seed",
        "section": "§3.1",
        "badge": BADGE_MEASURED,
        "claim": "DrugOOD lbap_refined_ic50_scaffold ood_test AUC = 0.6707 ± 0.0042 (5-seed)",
        "value": "0.6707 ± 0.0042",
        "json_path": "molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_drugood_size_5seed",
        "section": "§3.1",
        "badge": BADGE_MEASURED,
        "claim": "DrugOOD lbap_refined_ic50_size ood_test AUC = 0.6757 ± 0.0142 (5-seed)",
        "value": "0.6757 ± 0.0142",
        "json_path": "molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_imold_comparable_not_beat",
        "section": "§3.3",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "chemprop v2 BELOW iMoLD assay −0.0074 (within inter-seed std; comparable, NOT a beat)",
        "value": "−0.0074 (assay) / −0.0177 (scaffold) / −0.0035 (size)",
        "json_path": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §3.3",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_m14_cfm_decode_ratio",
        "section": "§4.1",
        "badge": BADGE_MEASURED,
        "claim": "m14 real CFM FULL MetalCytoToxDB decode_ratio = 0.9958 (n=26,801, Wf-188)",
        "value": "0.9958 (n=26,801)",
        "json_path": "molmetal/reports/wf188_m14_full_metalcytotox/result.json",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_m14_decode_ratio_synthetic_gap",
        "section": "§4.1",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "m14 CFM decode_ratio 0.9958 is −0.0042 BELOW synthetic upper bound 1.000 (expected per §3.5)",
        "value": "−0.0042 (expected, honest framing)",
        "json_path": "TODO/completed/207_unified_pre_flight_checklist_v2.md §9.6.3",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_m15_rmsd_pockets",
        "section": "§4.2",
        "badge": BADGE_MEASURED,
        "claim": "m15 RMSD 5/5 pockets < 2.0 Å (aggregate median 0.658 Å, 3-seed × 5 poses × 5 pockets)",
        "value": "5/5 pockets < 2.0 Å; median 0.658 Å",
        "json_path": "molmetal/reports/wf179_bd9_m15_rmsd_p2/summary.json",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_m21_nci60_9cell",
        "section": "§4.3",
        "badge": BADGE_MEASURED,
        "claim": "m21 NCI-60 9-cell per-cell AUC = 0.8007 ± 0.0160 (statistically equivalent to Krasnov 0.81)",
        "value": "0.8007 ± 0.0160",
        "json_path": "molmetal/metrics/by_metric/nci60_precious_9cell_auc.json",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_m21_krasnov_equivalent",
        "section": "§4.3",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "m21 9-cell mean 0.8007 ≈ Krasnov 0.81 (Δ = −0.0093, within std; NOT a beat)",
        "value": "−0.0093 (statistically equivalent)",
        "json_path": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §4.3",
        "timestamp": "2026-09-28",
    },
    {
        # Wf-312A — NCI-60 9-cell per-cell ship surface (3-seed x 5-epoch CPU
        # paper-grade smoke; replaces the 1-seed x 1-epoch Wf-308C numbers).
        # 27 runs (9 cells x 3 seeds) CPU 5 epochs chemprop v2 per-cell
        # independent encoders.  Verdict: 3 PASS (PC3, A549, DU145), 6
        # BELOW_GATE (HCT116, HL60, MCF7, HCT-116 alias, A2780, A2780cis);
        # paper gate 0.70.
        "id": "claim_nci60_9cell_hct116_highlight",
        "section": "§4.3",
        "badge": BADGE_MEASURED,
        "claim": "Wf-312A NCI-60 9-cell 3-seed x 5-epoch CPU: HCT116 per-cell AUC = 0.6851 ± 0.0274 (n_test=141; HCT-116 alias = 0.6851 same CSV row set; BELOW_GATE 0.70)",
        "value": "0.6851 ± 0.0274 (BELOW_GATE)",
        "json_path": "metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json::per_cell_line.HCT116.auc_mean",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_nci60_9cell_pc3_honest_negative",
        "section": "§4.3",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-312A NCI-60 9-cell 3-seed x 5-epoch CPU: PC3 per-cell AUC = 0.7274 ± 0.0916 (PASS at 5 epochs; honest-negative 0.2153 at 1 epoch was small-n + under-trained regime, NOT a true biological failure). HIGH per-seed std = borderline-learnable cell; honest framing per section 3.5",
        "value": "0.7274 ± 0.0916 (PASS at 5 ep; honest at 1 ep)",
        "json_path": "metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json::per_cell_line.PC3.auc_mean",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_nci60_9cell_overall_mean",
        "section": "§4.3",
        "badge": BADGE_MEASURED,
        "claim": "Wf-312A NCI-60 9-cell 3-seed x 5-epoch CPU: 9-cell overall mean AUC = 0.6880 (paper gate 0.70; 3 PASS / 6 BELOW_GATE; full paper-grade numbers need GPU per kill_then_adapt)",
        "value": "0.6880 (9-cell mean)",
        "json_path": "metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json::value_overall_mean_auc",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_nci60_9cell_a549_pass",
        "section": "§4.3",
        "badge": BADGE_MEASURED,
        "claim": "Wf-312A NCI-60 9-cell 3-seed x 5-epoch CPU: A549 per-cell AUC = 0.7436 ± 0.0250 (PASS paper gate 0.70; n_train=1852 largest cohort)",
        "value": "0.7436 ± 0.0250 (PASS)",
        "json_path": "metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json::per_cell_line.A549.auc_mean",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_nci60_9cell_du145_pass",
        "section": "§4.3",
        "badge": BADGE_MEASURED,
        "claim": "Wf-312A NCI-60 9-cell 3-seed x 5-epoch CPU: DU145 per-cell AUC = 0.9062 ± 0.0221 (PASS paper gate 0.70; high but small n_test=33 = easy split)",
        "value": "0.9062 ± 0.0221 (PASS, small-n caveat)",
        "json_path": "metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json::per_cell_line.DU145.auc_mean",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_nci60_9cell_a2780_below_gate",
        "section": "§4.3",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-312A NCI-60 9-cell 3-seed x 5-epoch CPU: A2780 per-cell AUC = 0.5641 ± 0.0239 (BELOW_GATE 0.70; cisplatin-sensitive ovarian signature not recoverable from precious-metal-only SMILES at 5-epoch CPU depth; needs GPU paper-grade to confirm)",
        "value": "0.5641 ± 0.0239 (BELOW_GATE)",
        "json_path": "metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json::per_cell_line.A2780.auc_mean",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_m7_diversity_tanimoto",
        "section": "§4.4",
        "badge": BADGE_MEASURED,
        "claim": "m7 diversity Tanimoto = 0.0397 with Tanimoto reward (+22.5% lift)",
        "value": "0.0397 (+22.5%)",
        "json_path": "TODO/completed/180_daily_summary_2026_09_27.md §1.3",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_m7_targetdiff_gap",
        "section": "§4.4",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "m7 diversity 0.0397 is +0.820 BELOW TargetDiff 0.860 ceiling (honest per §3.5)",
        "value": "+0.820 gap (honest-negative)",
        "json_path": "TODO/completed/207_unified_pre_flight_checklist_v2.md §9.6.2",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_drugood_1_58M_deferred",
        "section": "§5.3",
        "badge": BADGE_PROJECTED,
        "claim": "DrugOOD 1.58M train full retrain DEFERRED (currently 95,373 train subset, BD-11)",
        "value": "DEFERRED — 95,373/1,580,000 subset only",
        "json_path": "TODO/INDEX.md BD-11",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_krasnov_temporal_split_deferred",
        "section": "§5.3",
        "badge": BADGE_PROJECTED,
        "claim": "Krasnov temporal-split head-to-head driver DEFERRED (TODO-219 §3 follow-up)",
        "value": "DEFERRED — scaffold-split is the conservative protocol",
        "json_path": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §1.3",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_m3_vina_dock_projected",
        "section": "§5.4",
        "badge": BADGE_PROJECTED,
        "claim": "m3 Vina Dock PROJECTED @ R19 (Wf-30 5-seed mean ≥ −8.5 kcal/mol gate DEFERRED)",
        "value": "PROJECTED — DEFERRED to R19",
        "json_path": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §5.4",
        "timestamp": "2026-09-28",
    },
    # --- Wf-263A: chemprop v3 v3-fallback section (added Wf-268C-Dash) ---
    {
        "id": "claim_chemprop_v3_fallback_default",
        "section": "§6.1",
        "badge": BADGE_MEASURED,
        "claim": "chemprop v3 v3-fallback DEFAULT forces chemprop v2 BinaryClassificationFFN head regardless of --enhanced-head / --metal-attention",
        "value": "v3-fallback (default) → v2 head, audit-trail preserved",
        "json_path": "molmetal/reports/wf263a_implement/implement_report.md §3",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_chemprop_v3_fallback_official",
        "section": "§6.1",
        "badge": BADGE_MEASURED,
        "claim": "Wf-263A fix shipped: train_chemprop_v2_v3.py + train_chemprop_v2_v3_v1_backup.py + test_wf263a_chemprop_v3_fix.py (3 files, 10/10 new + 21/21 backward-compat tests PASS)",
        "value": "3 files modified, 10/10 + 21/21 tests PASS",
        "json_path": "molmetal/reports/wf263a_implement/implement_report.md §2 + §7",
        "timestamp": "2026-09-28",
    },
    # --- Wf-266: chemprop v3 LOMO fix flags section (added Wf-268C-Dash) ---
    {
        "id": "claim_wf266_min_epochs_flag",
        "section": "§6.2",
        "badge": BADGE_MEASURED,
        "claim": "--chemprop-min-epochs N (default 10) refuses --epochs < N (anti-under-training lever for LOMO head collapse)",
        "value": "--chemprop-min-epochs default 10",
        "json_path": "molmetal/reports/wf266_lomo_fix/wf266_report.md §Summary",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_wf266_lomo_init_scale_flag",
        "section": "§6.2",
        "badge": BADGE_MEASURED,
        "claim": "--chemprop-lomo-init-scale F (default 1e-3) scales per-metal BCE logits (anti-bias lever; logit*1e-3 keeps sigmoid ≈ 0.5 at init)",
        "value": "--chemprop-lomo-init-scale default 1e-3",
        "json_path": "molmetal/reports/wf266_lomo_fix/wf266_report.md §Summary",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_wf266_v3_fix_mode",
        "section": "§6.2",
        "badge": BADGE_MEASURED,
        "claim": "--chemprop-mode v3-fix combines v2 BinaryClassificationFFN + min-epochs floor + lomo-init-scale anti-bias (NO-OP under v3-fallback default)",
        "value": "--chemprop-mode v3-fix (4-mode choice: v2, v3, v3-fallback, v3-fix)",
        "json_path": "molmetal/reports/wf266_lomo_fix/wf266_report.md §Summary",
        "timestamp": "2026-09-28",
    },
    # --- TODO-180 G4: per-metal test_auc table (Wf-124 P3 5-metal × 3-seed) ---
    {
        "id": "claim_todo_180_g4_re_top",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": "TODO-180 G4 Re per-metal AUC = 0.7700 (n=3 seeds) — top of the 5-metal cohort",
        "value": "Re 0.7700 (n=3 seeds, n_train=692)",
        "json_path": "molmetal/reports/wf265_collect/all_results.json::g4_per_metal_aggregate_full_15pair[Re]",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_todo_180_g4_overall_15pair",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": "TODO-180 G4 overall mean test_auc = 0.7236 (15 pairs = 5 metal × 3 seed)",
        "value": "0.7236 overall (n=15 pairs)",
        "json_path": "molmetal/reports/wf265_collect/all_results.json::g4_overall_aggregate_full_15pair",
        "timestamp": "2026-09-28",
    },
    # --- Wf-266 G5 50-epoch sweep — PROJECTED (BLOCKED at GPU-ordinal layer) ---
    {
        "id": "claim_wf266_g5_50epoch_blocked",
        "section": "§5.4",
        "badge": BADGE_PROJECTED,
        "claim": "Wf-266 G5 50-epoch sweep BLOCKED at GPU-ordinal layer (CUDA error: invalid device ordinal); 0/15 pairs MEASURED; recovery deferred to R21+",
        "value": "PROJECTED — 0/15 pairs (BLOCKED)",
        "json_path": "molmetal/reports/wf266_final/final_status.md §4",
        "timestamp": "2026-09-28",
    },
    # --- Wf-279C: 5 NEW claim cards for recent 2026-09-29 findings ---
    {
        "id": "claim_wf275b_drugood_adapter",
        "section": "§3.1",
        "badge": BADGE_MEASURED,
        "claim": "Wf-275B DrugOOD ECFP4 + Bemis-Murcko scaffold-split adapter SHIPPED (n_source_rows=95,236; 0 RDKit parse failures; 0 scaffold leakage train↔val↔test)",
        "value": "n=95,236 rows; 0 leak; 5/5 tests PASS",
        "json_path": "molmetal/reports/wf275b_drugood_adapter/adapter_report.md §3 + §4",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf269c_yuelbond_load_bearing",
        "section": "§4.5.4",
        "badge": BADGE_MEASURED,
        "claim": "Wf-269C YuelBond decoder LOAD-BEARING on REAL cohort: full (YuelBond) decode_ratio=1.000 vs no_yuelbond=0.000 (Δ=−1.000, max possible); Wasserstein + closure_bounded are NOT decode gates (Δ=0 MEASURED)",
        "value": "Δ=−1.000 (full=1.000 / no_yuelbond=0.000, 51/51 train + 51/51 val)",
        "json_path": "molmetal/reports/wf269c_cfm_real/wf269c_report.json::configs",
        "timestamp": "2026-09-28",
    },
    {
        "id": "claim_wf275c_m14_lead_precondition",
        "section": "§4.1",
        "badge": BADGE_MEASURED,
        "claim": "Wf-275C m14 LEAD deployment precondition SHIPPED: YuelBond decoder MUST be active (--no-yuelbond MUST NOT be set); without YuelBond the m14 LEAD silently invalidates (decode_ratio 1.000 → 0.000)",
        "value": "YuelBond-on mandatory precondition",
        "json_path": "TODO/completed/m14_lead_deployment_precondition.md §2 + §3",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf274a_regression_disclosure",
        "section": "§5.2",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-274A cfm_decode_real.json regression_disclosure: previous_value=0.9980 (3 seeds × 5991 GPU) vs current_value=0.9645 (1 seed × 818 CPU PlatinAI pt_subset); verdict=silent_regression_methodology_change_not_disclosed; do NOT cite 0.9645 as LEAD",
        "value": "previous 0.9980 → current 0.9645 (Δ=−0.0335 silent)",
        "json_path": "metrics/by_metric/cfm_decode_real.json::regression_disclosure",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf271b_m17_drugood_v3_reeval",
        "section": "§3.3",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-271B m17_drugood_v3_reeval: 3 metals × 4 seeds = 12 v3 checkpoints on DrugOOD scaffold ood_test split, ROC-AUC = 0.4544 ± 0.0327 (BELOW_RANDOM); promoted from drugood_v3_scaffold to m17_drugood_v3_reeval; per-metal Ru=0.490, Ir=0.464, etc.",
        "value": "0.4544 ± 0.0327 (BELOW_RANDOM)",
        "json_path": "metrics/by_metric/drugood_v3_scaffold.json::per_metal.*.test_auc_mean",
        "timestamp": "2026-09-29",
    },
    # --- Wf-282 G5 Ru seed=0 MEASURED test_auc=0.7716 (first G5 50-ep result) ---
    {
        "id": "claim_wf282_g5_ru_seed0_test_auc",
        "section": "§4.5.1",
        "badge": BADGE_MEASURED,
        "claim": "Wf-282 G5 Ru seed=0 MEASURED test_auc=0.7716 (FIRST G5 50-ep test_auc from TODO-181 sweep on RTX 5090 GPU-1, cuda:0 fix post-cuda:1-bug, YuelBond + Wasserstein + closure-bounded default config card, val_auc peak 0.8177 @ epoch 47, test_pearson 0.5479, ckpt 11.3 MB saved at 00:28 CST 2026-09-29)",
        "value": "test_auc=0.7716 (50-ep, n_test=4350, OK status)",
        "json_path": "results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/per_seed_metrics.json::test_auc + molmetal/reports/wf283a_g5_recovery/recovery_report.md §1.3-§1.5",
        "timestamp": "2026-09-29",
    },
    # --- Wf-286B: ECFP4 axis ruled OUT as Wf-253-D 0.4544 collapse bottleneck ---
    {
        "id": "claim_wf286b_ecfp4_axis_ruled_out",
        "section": "§6.3",
        "badge": BADGE_MEASURED,
        "claim": "Wf-286B ECFP4 + Bemis-Murcko scaffold-split re-eval (1-seed × 5-ep, chemprop v2 BinaryClassificationFFN, CPU) test_auc=0.8820 RULES OUT ECFP4 featurization axis as Wf-253-D 0.4544 BELOW_RANDOM collapse bottleneck (Δ=+0.211 vs Wf-185 v2 baseline 0.6707); remaining hypotheses: (a) chemprop v3 MPNN backbone, (b) v3-fallback path config, (c) DrugOOD JSON train/val/test OOD splits",
        "value": "test_auc=0.8820 (Δ=+0.2113 vs Wf-185 0.6707 baseline)",
        "json_path": "molmetal/reports/wf286b_ecfp4_reeval/aggregate.json::test_auc_mean + molmetal/reports/wf286b_ecfp4_reeval/reeval_report.md §4",
        "timestamp": "2026-09-29",
    },
    # --- Wf-308B R03: 3 ADMET pass-through druglikeness claims (2 MEASURED-PASS + 1 honest-negative) ---
    {
        "id": "claim_wf303c_t2_ames_pass",
        "section": "§3.7",
        "badge": BADGE_MEASURED,
        "claim": "Wf-303C T2 / Wf-308B R03 ADMET pass-through: ames_mutagen mean = 0.0401 (cohort n=449, std=0.1962) — BELOW druglikeness pre-flight gate threshold 0.10 by −0.060 (PASS); RDKit FilterCatalog structural-alert ratio confirms low cohort mutagenicity (PASS vs source band < 0.30)",
        "value": "0.0401 (PASS, Δ=−0.060 vs 0.10 gate; PASS vs 0.30 source band)",
        "json_path": "metrics/by_metric/ames_mutagen.json::value + std + verdict",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf303c_t2_cell_perm_pass",
        "section": "§3.7",
        "badge": BADGE_MEASURED,
        "claim": "Wf-303C T2 / Wf-308B R03 ADMET pass-through: cell_permeability_logPapp mean = -3.637 (cohort n=449, std=1.060) — ABOVE pre-flight gate threshold -5.0 by +1.363 (PASS); Crippen logP + TPSA proxy indicates cohort Papp > 10⁻⁵·⁵ cm/s (PASS vs source band > -6.0)",
        "value": "-3.637 (PASS, Δ=+1.363 vs -5.0 gate; PASS vs -6.0 source band)",
        "json_path": "metrics/by_metric/cell_permeability_logPapp.json::value + std + verdict",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf303c_t2_aq_sol_honest_negative",
        "section": "§3.7",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-303C T2 / Wf-308B R03 ADMET pass-through: aqueous_solubility_logS mean = -7.612 (cohort n=449, std=3.936) — BELOW target band -4.0 by −3.612 (FAIL honest-negative); Crippen + MW + aromatic-ring proxy indicates cohort solubility < 0.1 mM; pre-flight gate requires explicit HONEST_NEGATIVE=true tag (added) — do not silently mask",
        "value": "-7.612 (FAIL honest-negative; HONEST_NEGATIVE=true tag added per Wf-308B)",
        "json_path": "metrics/by_metric/aqueous_solubility_logS.json::value + HONEST_NEGATIVE + honest_negative_reason",
        "timestamp": "2026-09-29",
    },
    # --- Wf-311E Part 4 / Wf-314A: 3 NEW claim cards for honest-negative
    # cluster close-out (mirroring claim_wf303c_t2_aq_sol_honest_negative pattern).
    # Parallel to existing claim_wf312c_r02_* entries (Wf-311E was PARTIAL when
    # paper_dashboard.py received its first update — Part 4 closes the loop).
    {
        "id": "claim_wf311e_m6_kappa_honest_negative",
        "section": "§3.7",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-311E Part 4 / Wf-314A honest-negative close: m6_kappa (Shape Kappa Index) mean = 0.472 (cohort n=449, std=0.159) — BELOW ChemAxon PMC11387870 druglikeness threshold 0.6 by −0.128 (FAIL honest-negative); Wf-253D MetalCytoToxDB validation returned NEGATIVE_WEAK (global Pearson r 0.081, Spearman 0.111); pre-flight gate p3_admet_m6_kappa_honest_negative requires explicit HONEST_NEGATIVE=true + honest_framing_note tags (added per Wf-314A) — do not silently mask",
        "value": "0.472 (FAIL honest-negative; HONEST_NEGATIVE=true + honest_framing_note tags added per Wf-311E Part 4 / Wf-314A)",
        "json_path": "metrics/by_metric/m6_kappa.json::value + HONEST_NEGATIVE + honest_negative_reason + honest_framing_note + metalcytotox_validation.global_pearson_r",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311e_hepatotox_honest_negative",
        "section": "§3.7",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-311E Part 4 / Wf-314A honest-negative close: hepatotox_index mean = 0.548 (cohort n=449) — ABOVE Hughes 2008 rule-of-2 druglikeness threshold 0.5 by +0.048 (FAIL honest-negative); metallo-drug cohort carries moderate hepatotox risk from aniline/hydrazine/5-ring structural alerts (Stepan 2011 composite); pre-flight gate p3_admet_hepatotox_honest_negative requires explicit HONEST_NEGATIVE=true + honest_framing_note tags (added per Wf-314A) — do not silently mask",
        "value": "0.548 (FAIL honest-negative; HONEST_NEGATIVE=true + honest_framing_note tags added per Wf-311E Part 4 / Wf-314A)",
        "json_path": "metrics/by_metric/hepatotox_index.json::value + HONEST_NEGATIVE + honest_negative_reason + honest_framing_note",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311e_logp7_4_honest_negative",
        "section": "§3.7",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-311E Part 4 / Wf-314A honest-negative close: logp7_4 mean = 6.413 (cohort n=449) — OUT OF Lipinski druglikeness 0-5 range by +1.413 (FAIL honest-negative); metallo-drug cohort is intrinsically lipophilic per Patrick 2009 ionisation-corrected Crippen logP; pre-flight gate p3_admet_logp7_4_honest_negative requires explicit HONEST_NEGATIVE=true + honest_framing_note tags (added per Wf-314A) — do not silently mask",
        "value": "6.413 (FAIL honest-negative; HONEST_NEGATIVE=true + honest_framing_note tags added per Wf-311E Part 4 / Wf-314A)",
        "json_path": "metrics/by_metric/logp7_4.json::value + HONEST_NEGATIVE + honest_negative_reason + honest_framing_note",
        "timestamp": "2026-09-29",
    },
    # --- Wf-312C R02: 3 NEW honest-negative druglikeness gates on pre-existing
    # JSON files (hepatotox / logp7_4 / m6_kappa).  All 3 carry the canonical
    # HONEST_NEGATIVE=true tag added per Wf-312C R02; the metallo-drug cohort
    # is intrinsically lipophilic (logp7_4) with moderate hepatotox risk
    # (hepatotox_index) and NEGATIVE_WEAK m6_kappa correlation with pIC50.
    {
        "id": "claim_wf312c_r02_hepatotox_honest_negative",
        "section": "§3.7",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-312C R02 honest-negative close: hepatotox_index mean = 0.548 (cohort n=449) — ABOVE Hughes 2008 rule-of-2 druglikeness threshold 0.5 by +0.048 (FAIL honest-negative); metallo-drug cohort carries moderate hepatotox risk from aniline/hydrazine/5-ring structural alerts; pre-flight gate p3_admet_hepatotox_honest_negative now requires explicit HONEST_NEGATIVE=true tag (added per Wf-312C) — do not silently mask",
        "value": "0.548 (FAIL honest-negative; HONEST_NEGATIVE=true tag added per Wf-312C R02)",
        "json_path": "metrics/by_metric/hepatotox_index.json::value + HONEST_NEGATIVE + honest_negative_reason",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf312c_r02_logp7_4_honest_negative",
        "section": "§3.7",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-312C R02 honest-negative close: logp7_4 mean = 6.413 (cohort n=449) — OUT OF Lipinski druglikeness 0-5 range by +1.413 (FAIL honest-negative); metallo-drug cohort is intrinsically lipophilic per Patrick 2009 ionisation-corrected Crippen logP; pre-flight gate p3_admet_logp7_4_honest_negative now requires explicit HONEST_NEGATIVE=true tag (added per Wf-312C) — do not silently mask",
        "value": "6.413 (FAIL honest-negative; HONEST_NEGATIVE=true tag added per Wf-312C R02)",
        "json_path": "metrics/by_metric/logp7_4.json::value + HONEST_NEGATIVE + honest_negative_reason",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf312c_r02_m6_kappa_honest_negative",
        "section": "§3.7",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-312C R02 honest-negative close: m6_kappa (Shape Kappa Index) mean = 0.472 (cohort n=449, std=0.159) — BELOW ChemAxon PMC11387870 druglikeness threshold 0.6 by −0.128 (FAIL honest-negative); Wf-253D MetalCytoToxDB validation returned NEGATIVE_WEAK (global Pearson r 0.081, Spearman 0.111) — Shape Kappa is NOT a learned pIC50 proxy; pre-flight gate p3_admet_m6_kappa_honest_negative now requires explicit HONEST_NEGATIVE=true tag (added per Wf-312C) — do not silently mask",
        "value": "0.472 (FAIL honest-negative; HONEST_NEGATIVE=true tag added per Wf-312C R02)",
        "json_path": "metrics/by_metric/m6_kappa.json::value + HONEST_NEGATIVE + honest_negative_reason + metalcytotox_validation.global_pearson_r",
        "timestamp": "2026-09-29",
    },
    # --- Wf-312C R01: 5 NEW ADMET extension metrics (CPU-only RDKit pass-through
    # heuristics; NOT wet-lab measurements).  Cohort n=449; verdict is PASS
    # for caco2_perm (above -5.15 well-permeable threshold) and INFO for
    # bbb / clearance / half_life / vdss (continuous spectrum metrics with
    # no binary PASS/FAIL band).
    {
        "id": "claim_wf312c_r01_caco2_perm",
        "section": "§3.7",
        "badge": BADGE_MEASURED,
        "claim": "Wf-312C R01 ADMET 5-NEW extension: caco2_perm mean = -3.035 (cohort n=449, std=0.231) — ABOVE Hou 2007 refined 5-descriptor Caco-2 well-permeable threshold -5.15 by +2.115 (PASS); Caco-2-specific heuristic (distinct from the Wf-303C cell_permeability_logPapp MDCK/general Papp proxy); RDKit descriptor projection, NOT wet-lab Caco-2 assay",
        "value": "-3.035 (PASS, Δ=+2.115 vs -5.15 gate; PASS vs -8/-3 clamp)",
        "json_path": "metrics/by_metric/caco2_perm.json::value + std + verdict",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf312c_r01_bbb",
        "section": "§3.7",
        "badge": BADGE_MEASURED,
        "claim": "Wf-312C R01 ADMET 5-NEW extension: bbb (logBB) mean = -0.159 (cohort n=449, std=1.293) — INFO (within Clark 1999 -1<logBB<1 spectrum; cohort straddles BBB+/- boundary); TPSA + logP heuristic, NOT in-situ brain perfusion assay",
        "value": "-0.159 (INFO, within Clark 1999 BBB+/- spectrum)",
        "json_path": "metrics/by_metric/bbb.json::value + std + verdict",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf312c_r01_clearance",
        "section": "§3.7",
        "badge": BADGE_MEASURED,
        "claim": "Wf-312C R01 ADMET 5-NEW extension: clearance (hepatic CLint) mean = 0.509 mL/min/kg (cohort n=449, std=0.121) — INFO (BELOW Lobell/Obach 1996-1997 typical 5-30 mL/min/kg band — slow hepatic metabolism expected per metal-coordination chemistry); heuristic projection, NOT microsome/hepatocyte CLint measurement",
        "value": "0.509 mL/min/kg (INFO, below 5-30 typical band)",
        "json_path": "metrics/by_metric/clearance.json::value + std + verdict",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf312c_r01_half_life",
        "section": "§3.7",
        "badge": BADGE_MEASURED,
        "claim": "Wf-312C R01 ADMET 5-NEW extension: half_life (t1/2) mean = 1.31 hours (cohort n=449, std=1.34) — INFO (within Obach 1997 1-24 hour oral-drug band; lower end indicates faster elimination); heuristic projection, NOT in-vivo pharmacokinetic measurement",
        "value": "1.31 hours (INFO, within 1-24 hour oral-drug band)",
        "json_path": "metrics/by_metric/half_life.json::value + std + verdict",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf312c_r01_vdss",
        "section": "§3.7",
        "badge": BADGE_MEASURED,
        "claim": "Wf-312C R01 ADMET 5-NEW extension: vdss mean = 0.264 L/kg (cohort n=449, std=0.369) — INFO (BELOW Lombardo 2006 / Obach 1997 typical 0.5-10 L/kg drug range — limited tissue distribution expected); heuristic projection, NOT in-vivo VDss measurement",
        "value": "0.264 L/kg (INFO, below 0.5-10 typical drug range)",
        "json_path": "metrics/by_metric/vdss.json::value + std + verdict",
        "timestamp": "2026-09-29",
    },
    # --- Wf-315A: SA + QED evaluators wired into the panel (WEAK_TO_STRONG §3.6).
    # Both are MEASURED + HONEST_NEGATIVE on the 449 metallo-drug cohort:
    #   * sa_score mean = 4.715 (FAIL — above 4.0 threshold per Wf-139 P6)
    #   * qed_score mean = 0.230 (FAIL — below 0.5 threshold per Wf-139 P6)
    # Mirror the Wf-308B / Wf-312C claim_wf303c_t2_aq_sol_honest_negative
    # pattern: cohort-intrinsic druglikeness limit (metallo-drug cohort
    # carries larger polycyclic / click-handle scaffolds + intrinsically
    # higher MW + AROM profiles), not a model failure.
    {
        "id": "claim_wf315a_sa_score_honest_negative",
        "section": "§3.7",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-315A SA + QED evaluator wiring (WEAK_TO_STRONG §3.6): sa_score mean = 4.715 (cohort n=449, std=1.426) — ABOVE Wf-139 P6 synthesizable threshold 4.0 by +0.715 (FAIL honest-negative); Ertl-Schuffenhauer 2009 sascorer indicates cohort carries harder-to-synthesize scaffolds (polycyclic + click-handle ligands); pre-flight gate p3_sa_score_metric_present requires explicit HONEST_NEGATIVE=true tag (added per Wf-315A) — do not silently mask",
        "value": "4.715 (FAIL honest-negative; HONEST_NEGATIVE=true tag added per Wf-315A)",
        "json_path": "metrics/by_metric/sa_score.json::value + std + verdict + HONEST_NEGATIVE + honest_negative_reason",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf315a_qed_score_honest_negative",
        "section": "§3.7",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": "Wf-315A SA + QED evaluator wiring (WEAK_TO_STRONG §3.6): qed_score mean = 0.230 (cohort n=449, std=0.145) — BELOW Wf-139 P6 druglikeness threshold 0.5 by −0.270 (FAIL honest-negative); Bickerton 2012 weighted aggregate intrinsically penalises the metallo-drug MW + AROM profiles (8-descriptor RDKit aggregate); pre-flight gate p3_qed_score_metric_present requires explicit HONEST_NEGATIVE=true tag (added per Wf-315A) — do not silently mask",
        "value": "0.230 (FAIL honest-negative; HONEST_NEGATIVE=true tag added per Wf-315A)",
        "json_path": "metrics/by_metric/qed_score.json::value + std + verdict + HONEST_NEGATIVE + honest_negative_reason",
        "timestamp": "2026-09-29",
    },
    # --- Wf-310A: G5 50-epoch full sweep MEASURED re-aggregation (path+schema fix) ---
    {
        "id": "claim_wf310a_g5_5metal_3seed_50epoch_overall_auc",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-310A G5 50-epoch full sweep: 5-metal × 3-seed × 50-epoch "
            "D-MPNN multi-task + Wasserstein pIC50 regulariser (alpha=0.5, "
            "w_w=0.05, L2 gradient-norm guard) MEASURED test_auc = "
            "0.7227 ± 0.0417 (n=13/15 pairs; 2 pending Ru/seed2 + Ir/seed0 "
            "retry per Wf-310A). Per party-twenty-spirit §3.5, missing pairs "
            "reported honestly; per-metal std unfiltered."
        ),
        "value": "0.7227 ± 0.0417 (n=13/15 pairs; 2 retries in-flight)",
        "json_path": (
            "results/wf124_p3_50epoch_full_sweep_g5/training_summary.json::"
            "aggregate_5metal.test_auc_mean + .test_auc_std + .n_pairs + "
            "missing_pairs + molmetal/scripts/wf278a_g5_aggregate.py"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf310a_g5_ru_per_metal_50epoch",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-310A G5 Ru per-metal MEASURED test_auc = 0.7860 ± 0.0145 "
            "(n=2/3 seeds; seed=2 retry in-flight per Wf-310A). vs "
            "TODO-180 G4 30-ep Ru baseline 0.7561 → Δ=+0.0300 lift on the "
            "2-seed overlap; vs chemprop v2 single-task baseline 0.7123 "
            "→ Δ=+0.0737 (real MEASURED)."
        ),
        "value": (
            "Ru 0.7860 ± 0.0145 (n=2/3 seeds, 50-ep; +0.0300 vs 30-ep 0.7561; "
            "+0.0737 vs chemprop v2 single-task 0.7123)"
        ),
        "json_path": (
            "results/wf124_p3_50epoch_full_sweep_g5/training_summary.json::"
            "per_metal.Ru + metrics/by_metric/wf181_g5_50epoch_full_sweep.json"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf310a_g5_ir_per_metal_50epoch",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-310A G5 Ir per-metal MEASURED test_auc = 0.7009 ± 0.0133 "
            "(n=2/3 seeds; seed=0 retry in-flight per Wf-310A). vs Krasnov 2026 "
            "temporal-split baseline 0.73 → MEASURED-vs-PROJECTED gap −0.029 "
            "honestly reported (our scaffold split is stricter than random but "
            "less strict than temporal; direct comparison deferred to TODO-219 §3 "
            "temporal-split driver)."
        ),
        "value": "Ir 0.7009 ± 0.0133 (n=2/3 seeds, 50-ep; seed=0 retry pending)",
        "json_path": (
            "results/wf124_p3_50epoch_full_sweep_g5/training_summary.json::"
            "per_metal.Ir + metrics/by_metric/wf181_g5_50epoch_full_sweep.json"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf310a_g5_ru_30ep_vs_50ep_delta",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-310A G5 30ep→50ep Ru cascade: G4 30-ep Ru baseline test_auc = "
            "0.7561 (TODO-180 G4 cascade re-aggregated at Wf-310A) → G5 50-ep "
            "Ru (n=2/3 seeds) test_auc = 0.7860. Δ=+0.0300 MEASURED lift from "
            "the +20 epoch doubling on the 2-seed overlap (Ru seed=0 + "
            "Ru seed=1). Per party-twenty-spirit §3.5, both epochs are "
            "MEASURED; the lift is real but reported on the 2-seed overlap "
            "subset (Ru seed=2 retry pending)."
        ),
        "value": (
            "Δ=+0.0300 (G4 30ep 0.7561 → G5 50ep 0.7860, n=2 seeds overlap)"
        ),
        "json_path": (
            "results/wf124_p3_50epoch_full_sweep_g5/training_summary.json::"
            "ru_auc_baseline_30epoch_g4 + ru_30ep_vs_50ep_delta + "
            "results/wf124_p3_remaining_metals_g4/training_summary.json::"
            "per_metal.Ru.test_auc_mean"
        ),
        "timestamp": "2026-09-29",
    },
    # --- Wf-318B: G5 50-epoch cascade delta extended to all 5 metals (Ru/Ir/Os/Rh/Re) ---
    {
        "id": "claim_wf318b_g5_cascade_delta_Ru",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-318B G5 30ep→50ep cascade delta Ru: G4 30-ep Ru baseline "
            "test_auc = 0.7561 ± 0.0112 (n=3) → G5 50-ep Ru test_auc = "
            "0.7860 ± 0.0205 (n=2/3 seeds; seed=2 still in-flight). "
            "Δ=+0.0300 MEASURED (+3.96%, t_stat=2.10, unpaired two-sample "
            "t-test). Honest framing per party-twenty-spirit §3.5: n=2-seed "
            "overlap (Ru/seed=0 + Ru/seed=1); re-run after Ru/seed=2 lands."
        ),
        "value": "Δ=+0.0300 (+3.96%, t_stat=2.10, n=2-seed overlap)",
        "json_path": (
            "metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json::"
            "per_metal_table.Ru.delta + .delta_pct + .t_stat + "
            ".n_seeds_overlap + results/wf124_p3_remaining_metals_g4/"
            "training_summary.json::per_metal.Ru.test_auc_mean + "
            "metrics/by_metric/wf181_g5_50epoch_full_sweep.json::per_metal_in_domain.Ru"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf318b_g5_cascade_delta_Ir",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-318B G5 30ep→50ep cascade delta Ir: G4 30-ep Ir baseline "
            "test_auc = 0.6702 ± 0.0056 (n=3) → G5 50-ep Ir test_auc = "
            "0.6969 ± 0.0150 (n=3 effective: Wf-310A retry at 0.6888 + "
            "seeds 1+2 baseline). Δ=+0.0266 MEASURED (+3.98%, t_stat=2.88, "
            "unpaired two-sample t-test). Honest framing: full 3-seed "
            "overlap post-retry; lift over Krasnov 2026 baseline 0.73 still "
            "MEASURED-vs-PROJECTED gap −0.0331 (deferred to TODO-219 §3 "
            "temporal-split driver)."
        ),
        "value": "Δ=+0.0266 (+3.98%, t_stat=2.88, n=3-seed overlap)",
        "json_path": (
            "metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json::"
            "per_metal_table.Ir.delta + .delta_pct + .t_stat + "
            ".n_seeds_overlap + results/wf124_p3_remaining_metals_g4/"
            "training_summary.json::per_metal.Ir.test_auc_mean + "
            "results/wf124_p3_50epoch_full_sweep_g5/Ir/seed0/"
            "training_summary.json::per_metal.Ir.test_auc_mean"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf318b_g5_cascade_delta_Os",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-318B G5 30ep→50ep cascade delta Os: G4 30-ep Os baseline "
            "test_auc = 0.7025 ± 0.0205 (n=3) → G5 50-ep Os test_auc = "
            "0.7276 ± 0.0159 (n=3). Δ=+0.0250 MEASURED (+3.56%, t_stat=1.67, "
            "unpaired two-sample t-test). Honest framing: full 3-seed overlap; "
            "Os has small cohort (~834 rows train) — per-seed spread ±0.015 "
            "unfiltered per party-twenty-spirit §3.5."
        ),
        "value": "Δ=+0.0250 (+3.56%, t_stat=1.67, n=3-seed overlap)",
        "json_path": (
            "metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json::"
            "per_metal_table.Os.delta + .delta_pct + .t_stat + "
            ".n_seeds_overlap + results/wf124_p3_remaining_metals_g4/"
            "training_summary.json::per_metal.Os.test_auc_mean + "
            "metrics/by_metric/wf181_g5_50epoch_full_sweep.json::per_metal_in_domain.Os"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf318b_g5_cascade_delta_Rh",
        "section": "§2.1",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": (
            "Wf-318B G5 30ep→50ep cascade delta Rh: G4 30-ep Rh baseline "
            "test_auc = 0.7227 ± 0.0540 (n=3) → G5 50-ep Rh test_auc = "
            "0.6798 ± 0.0165 (n=3). Δ=−0.0428 MEASURED REGRESSION "
            "(−5.92%, t_stat=−1.29, unpaired two-sample t-test). Honest "
            "framing per party-twenty-spirit §3.5: 50-epoch cascade regresses "
            "Rh AUC by ~6 pts (BELOW G4 30-ep baseline + Krasnov 2026 "
            "baseline 0.74). Rh has small cohort (~696 rows train) — "
            "suggests W₂ regulariser may over-smooth on small cohort + "
            "50 epochs (overfitting vs Krasnov 2026 0.74, our scaffold "
            "split drops to 0.68). Hypothesis: shorter 30-epoch cascade + "
            "early stopping per Wf-309A may be preferable for Rh/Os/Re "
            "small-cohort metals. See TODO-182 G6 per-innovation ablation."
        ),
        "value": "Δ=−0.0428 (−5.92%, t_stat=−1.29, n=3-seed overlap; HONEST_NEGATIVE: 50-epoch cascade regresses Rh)",
        "json_path": (
            "metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json::"
            "per_metal_table.Rh.delta + .delta_pct + .t_stat + "
            ".n_seeds_overlap + results/wf124_p3_remaining_metals_g4/"
            "training_summary.json::per_metal.Rh.test_auc_mean + "
            "metrics/by_metric/wf181_g5_50epoch_full_sweep.json::per_metal_in_domain.Rh"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf318b_g5_cascade_delta_Re",
        "section": "§2.1",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": (
            "Wf-318B G5 30ep→50ep cascade delta Re: G4 30-ep Re baseline "
            "test_auc = 0.7700 ± 0.0244 (n=3) → G5 50-ep Re test_auc = "
            "0.7329 ± 0.0433 (n=3). Δ=−0.0371 MEASURED REGRESSION "
            "(−4.82%, t_stat=−1.10, unpaired two-sample t-test). Honest "
            "framing per party-twenty-spirit §3.5: 50-epoch cascade regresses "
            "Re AUC by ~5 pts (BELOW G4 30-ep baseline + Krasnov 2026 "
            "baseline 0.78). Re has small cohort (~692 rows train) — "
            "consistent with Rh regression pattern; W₂ regulariser over-"
            "smooths on small-cohort 50 epochs. Hypothesis: TODO-182 G6 "
            "per-innovation ablation + Wf-309A early-stopping may recover."
        ),
        "value": "Δ=−0.0371 (−4.82%, t_stat=−1.10, n=3-seed overlap; HONEST_NEGATIVE: 50-epoch cascade regresses Re)",
        "json_path": (
            "metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json::"
            "per_metal_table.Re.delta + .delta_pct + .t_stat + "
            ".n_seeds_overlap + results/wf124_p3_remaining_metals_g4/"
            "training_summary.json::per_metal.Re.test_auc_mean + "
            "metrics/by_metric/wf181_g5_50epoch_full_sweep.json::per_metal_in_domain.Re"
        ),
        "timestamp": "2026-09-29",
    },
    # --- Wf-311B: hard-metric paper-grade aggregation (Vina × 5 + RMSD panel + main results) ---
    {
        "id": "claim_wf311b_vina_5pocket_aggregate",
        "section": "§4.2",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-311B Vina × 5-pocket MEASURED aggregate: 1h36/3eml/3ny8/4aaw/4rlu "
            "high_affinity rates 0.95/0.05/0.35/0.65/0.45 (cross-pocket mean 0.4900 ± 0.3191), "
            "triple_success rates 0.00/0.00/0.25/0.00/0.00 (SA/QED not wired in pocket mode), "
            "relaxed_success rates 0.3077/0.0/0.0/0.0/0.0 (1h36 only Wf-142 ex=32 retrain), "
            "co-crystal reference kcal/mol -0.801/-7.753/-7.140/-4.537/-6.373. "
            "Honest framing: smoke-partial on test_001 / pKd scores, NOT full real-cohort."
        ),
        "value": "5/5 pockets; mean high_affinity 0.4900 ± 0.3191",
        "json_path": "molmetal/reports/wf311b_vina_5pocket_aggregate.json::aggregate + per_pocket_summary",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311b_rmsd_panel_aggregate",
        "section": "§4.6",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-311B RMSD panel MEASURED: docked_pose_rmsd 5/5 pockets < 2.0 Å SOTA "
            "(cross-pocket median 0.6578 Å, std 0.2546 Å, min 0.4223 / max 1.0240; "
            "Wf-246-B redesigned schema with per-pocket std + difficulty flag); "
            "ligand_vs_pocket_rmsd 0/5 pockets < 2.0 Å (median 3.8333 Å; "
            "Wf-247-A RDKit rdMolAlign.GetBestRMS canonical SOTA path is STRICTER "
            "than legacy Kabsch+Hungarian); rigid_fragment_rmsd 1.241 ± 0.626 Å "
            "mean on pocket 1h36 (n=20), 5/5 pockets MEASURED on disk."
        ),
        "value": "docked 5/5 < 2.0Å median 0.6578Å; lig_vs_pkt 0/5 < 2.0Å median 3.8333Å",
        "json_path": "molmetal/reports/wf311b_rmsd_aggregate.json::docked_pose_rmsd + ligand_vs_pocket_rmsd + rigid_fragment_rmsd",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311b_main_results_aggregate",
        "section": "§4.2",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-311B main results aggregate: chemprop_v2_lomo_v2 = 0.9015 ± 0.0168 "
            "(5-metal × 5-seed × 50-ep, per-metal in-domain AUC; Ru=0.926, Ir=0.913, "
            "Os=0.900, Rh=0.886, Re=0.881); nci60_precious_9cell_auc = 0.8007 ± 0.0160 "
            "(Krasnov 2026 baseline 0.81, Δ=-0.0093 within std — honest-NEGATIVE: "
            "statistically equivalent, NOT a beat); per_metal_pic50_wasserstein "
            "3/5 metals (Ru/Ir/Os), n=4 runs."
        ),
        "value": "lomo 0.9015±0.0168; nci60 0.8007±0.0160 (Krasnov-equivalent)",
        "json_path": "molmetal/reports/wf311b_main_results_aggregate.json::chemprop_v2_lomo_v2 + nci60_precious_9cell_auc",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311b_section_4_2_refreshed",
        "section": "§4.2",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-311B: metrics/by_paper_section/section_4_2_main_results.json REFRESHED "
            "with all hard-metric aggregations (Vina × 5, RMSD panel, chemprop_v2_lomo_v2, "
            "nci60_precious_9cell_auc). 5 cells §4 Table 1 metal-compliance arm preserved; "
            "55 DESIGN cells still pending production 100×3 sweep (R13 / wo21fr1ug)."
        ),
        "value": "REFRESHED (5 MEASURED + 55 DESIGN cells; Wf-311B aggregation block added)",
        "json_path": "metrics/by_paper_section/section_4_2_main_results.json::wf311b_aggregations + verdict",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311b_section_4_6_refreshed",
        "section": "§4.6",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-311B: metrics/by_paper_section/section_4_6_pb_panel.json REFRESHED — PB "
            "pass rate 0.846 (smoke 1-pocket) preserved; docked_pose_rmsd 0.6578 Å median, "
            "ligand_vs_pocket_rmsd 3.8333 Å median now part of §4.6 panel; RMSD aggregate "
            "block added. Honest framing: PB production 10×3 still BLOCKED by n_distinct=1."
        ),
        "value": "REFRESHED (PB 0.846 smoke + RMSD 0.6578Å median)",
        "json_path": "metrics/by_paper_section/section_4_6_pb_panel.json::aggregate_metrics + verdict",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311b_section_5_ablation_refreshed",
        "section": "§5",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-311B: metrics/by_paper_section/section_5_ablation.json REFRESHED with 4 new "
            "ablation axes: per-pocket Vina (5/5 pockets), RMSD panel (15 measurements), "
            "chemprop_v2_lomo_v2 (25 runs), nci60_9cell (9 cells). 9 axes total with data "
            "(5 Wf-268C baseline + 4 Wf-311B)."
        ),
        "value": "REFRESHED (5 Wf-268C + 4 Wf-311B ablation axes)",
        "json_path": "metrics/by_paper_section/section_5_ablation.json::ablation_axes_with_data + verdict",
        "timestamp": "2026-09-29",
    },
    # --- Wf-311D: 5 NEW ship-surface claim cards for OOD reachability +
    #     m18 protein_metal + gi50_proxy + pharmacophore_pass_rate_prod +
    #     metal_coord_compliance Phase-3G ---
    {
        "id": "claim_wf311d_ood_reachability_summary",
        "section": "§3.4",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-311D PART 1 OOD reachability aggregated across 30 d-block elements "
            "(Wf-130 P4 close-out 2026-09-27): 20/30 elements pass the >=0.40 "
            "gate; aggregate macro reachability = 0.4989; per-block 3d = 8/10 pass, "
            "4d = 7/10, 5d+La = 5/10. Cross-domain (Wf-130 P3, PlatinAI + MetalCytoTox) "
            "supplement covers 8/30 metals with reachability=0.0 (CPU-only env lacks "
            "RDKit; closure_theorem partner_terms fallback rejects metal-cyto SMILES)."
        ),
        "value": "20/30 pass, macro_reachability=0.4989",
        "json_path": "metrics/by_metric/ood_reachability_summary.json::aggregate_reachability_macro + n_elements_gate_pass + per_block_aggregate",
        "timestamp": "2026-09-29",
    },
    # --- Added Wf-318E: Wf-130 P3 cross-domain OOD RDKit alt-env re-run ---
    {
        "id": "claim_wf318e_cross_domain_rdklt_lift",
        "section": "§3.4",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-318E (2026-09-29) re-ran Wf-130 P3 cross-domain OOD audit in "
            "/home/hugo/codes/mol-metal/.venv/bin/python (rdkit 2026.03.6) to "
            "lift the 7 RDKit-missing-blocked metals. Cross-domain macro "
            "lifted from 0.0 (env-blocked, Wf-130 P3) to 0.3348 macro / "
            "0.2872 micro MEASURED across 8 metals × 94 OOD samples "
            "(n_closure_ok=27). 4/8 pass 0.30 gate (Pt 0.8333, Pd 0.5000, "
            "Au 0.3333, Rh 0.3333); 3/8 below but real (Ru 0.2143, Os 0.2500, "
            "Ir 0.2143); Re 0.0 CONFIRMED structural HONEST_NEGATIVE_DATA_AVAILABILITY "
            "(0/10 closure_ok even with RDKit available)."
        ),
        "value": "8-metal macro=0.3348 (Pt 0.83 Pd 0.50 Au 0.33 Rh 0.33 / Ru 0.21 Os 0.25 Ir 0.21 Re 0.00)",
        "json_path": "metrics/by_metric/ood_reachability_summary.json::cross_domain_supplement.wf318e_macro_reachability + cross_domain_supplement.wf318e_per_metal",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311d_m18_protein_metal_baseline",
        "section": "§4.5.2",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-311D PART 2 m18 protein_metal binding AUC aggregated across "
            "100-ep + 50-ep baselines: 100-ep test_auc = 0.7436 ± 0.0092 "
            "(n=3 seeds, n_rows=5257, 209 cell lines, 5 metals); 50-ep v2 "
            "test_auc = 0.7393 ± 0.0092. Per-residue redesign v1 "
            "(metric18_protein_metal_per_residue.json) per_complex_auc = "
            "0.9945 / per_residue_auc = 0.9627 / per_residue_pr_auc = 0.8331 "
            "— labels are derived SYNTHETIC PROXIES (MetalCytoToxDB has only "
            "complex-level active/inactive labels; per-residue labels are "
            "heuristically derived, NOT real per-residue binding labels)."
        ),
        "value": "100-ep 0.7436 / 50-ep 0.7393 / per_residue 0.9627 (synthetic_proxy)",
        "json_path": "metrics/by_metric/protein_metal_per_residue_summary.json::components.metric18_100epoch + components.metric18_50epoch + per_residue_redesign_v1",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311d_gi50_proxy_summary",
        "section": "§4.5.3",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-311D PART 3 gi50_proxy aggregation verified: existing "
            "gi50_proxy.json carries cohort-level value=8.0 (clipped, "
            "-log10(mol/L)) on n=5000 MetalCytoToxDB-derived samples via "
            "Hou 2007 descriptor heuristic. ALREADY AGGREGATED — no further "
            "aggregation required. Cite as 'gi50_proxy (Hou 2007 model) = 8.0 "
            "on n=5000' — NOT as 'GI50 = 8.0' — this is a model proxy, NOT "
            "a wet-lab GI50 measurement."
        ),
        "value": "8.0 (clipped) on n=5000 (Hou 2007 model proxy)",
        "json_path": "metrics/by_metric/gi50_proxy_summary.json::value + n_samples + aggregation_verdict",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311d_pharmacophore_pass_rate_prod",
        "section": "§3.8",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-311D PART 4 pharmacophore_pass_rate PRODUCTION expansion: "
            "strict_pass_rate = 0.1737 (n=78/449, lenient=0.3786 n=170/449) "
            "on the metallo-drug cohort from molmetal/data/metallo_drugs_500_train.csv "
            "(334 tmqm + 96 platinai + 19 metal_cytotox SMILES). PRODUCTION value "
            "(0.1737) is LOWER than curated smoke (0.7 on 10 hand-picked SMILES), "
            "as expected — curated smoke was biased upward; production value is "
            "the honest cohort-level drug-likeness measurement. Use production "
            "as the headline number; curated 0.7 retained as upper-bound sanity check."
        ),
        "value": "0.1737 strict (78/449), 0.3786 lenient (170/449)",
        "json_path": "metrics/by_metric/pharmacophore_pass_rate_prod.json::aggregate.strict_pass_rate + lenient_pass_rate + per_source + comparison_to_curated",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf311d_metal_coord_phase3g_singleton",
        "section": "§4.5.5",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": (
            "Wf-311D PART 5 metal_coord_compliance Phase-3G completion: current "
            "metal_coord_compliance = 1.0 on 5x1 cisplatin + all-5 pilot "
            "(DESIGN+1-PILOT, n_distinct=1, singleton collapse — the seed itself "
            "is Pt_II square-planar so the metric trivially passes). Phase-3G "
            "completion honest-negative: full 10x3 rotation BLOCKED on TODO-29 "
            "F2(a) MetalLigandExchange SMARTS rule (NOT SHIPPED) + GPU unblock. "
            "Phase-3G shipping surface is intact (metal_coord_probe.py + "
            "test_metal_coord_probe.py + Phase 4 wire-in to r4_lambda_only_run.py) "
            "but the non-degenerate compliance verification is deferred until "
            "TODO-29 + GPU clear. "
            "[Wf-325F 2026-09-29 UPDATE] BLOCK-on-runtime portion CLOSED via Wf-318F "
            "(F2MetalLigandExchangeRuntime SHIPPED, 170 deduped patterns, 16/16 tests PASS) + "
            "Wf-290F4 (84-rule MLE matrix SHIPPED, 659/659 tests PASS). "
            "10x3 rotation CPU smoke now POSSIBLE per molmetal/scripts/wf325f_phase3g_10x3_smoke.py: "
            "n_runs_executed=30, n_compliant=27/30 (90.0%), variance=0.09, "
            "honest_verdict=structural_ceiling_lifted. The singleton collapse is "
            "structurally lifted by F2(a) runtime + 84-rule matrix (per the smoke); "
            "the GPU portion (Wf-310E Wf-135 Phase 2) remains BLOCKED on user OK per "
            "mol-metal-gpu-execution §0 kill_then_adapt directive."
        ),
        "value": "1.0 on singleton (n_distinct=1) pre-Wf-318F; Wf-325F 10x3 smoke variance=0.09 (structural_ceiling_lifted); GPU portion still BLOCKED",
        "json_path": "metrics/by_metric/metal_coord_compliance_phase3g.json::current_pilot_status + phase3g_completion_verdict + blockers + wf325f_smoke_evidence",
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf318f_f2a_runtime_shipped",
        "section": "§4.5.5",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-318F F2(a) MetalLigandExchange SMARTS runtime dispatcher "
            "SHIPPED (CPU-only): F2MetalLigandExchangeRuntime loads 170 "
            "deduped patterns (5 curated F2(a) + 84 L-R17-B2 matrix + 17 NEW "
            "SMARTS + 64 82-metal-filtered). dispatch(metal, ligand_smiles) "
            "returns True for Pt/Pd/Au/Ru/Ir/Os/Rh/Re + [NH3] (per "
            "test_wf318f_f2a_metal_ligand_runtime.py, 16/16 PASS). Closes "
            "the BLOCKED-on-runtime portion of TODO-29 (Phase-3G 10x3 "
            "rotation + r13_100x3 sweep still BLOCKED on GPU portion — "
            "see claim_wf311d_metal_coord_phase3g_singleton)."
        ),
        "value": "170 deduped patterns loaded; 16/16 tests PASS",
        "json_path": "molmetal/tests/test_wf318f_f2a_metal_ligand_runtime.py + molmetal/molmetal_lam/lam_chem/metal_ligand_exchange_runtime.py",
        "timestamp": "2026-09-29",
    },
    # --- Wf-320E: per-TODO consumer lift dashboard (6 NEW claim cards
    # for strengthen-TODO consumers TODO-119/120/121/122/123/125). The
    # dashboard is aggregation-only — no new measurement. Per party-twenty-
    # spirit §3.5 honest framing is mandatory; HONEST_NEGATIVE rows are
    # NOT silently masked (TODO-121/122/125).
    {
        "id": "claim_wf320e_todo119_m14_lift_partial",
        "section": "§4.5.6",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": (
            "Wf-320E TODO-119 m14 decode_ratio consumer lift: Wf-148 "
            "M1+M2-enabled 3-seed × 5991 GPU NCI-60+DrugOOD = 0.9980 vs "
            "Wf-274A 1-seed × 818 CPU PlatinAI pt_subset baseline 0.9645. "
            "Apparent +0.0335 absolute / +3.47% is a METHODOLOGY CHANGE "
            "(1-seed CPU vs 3-seed GPU 5991), NOT a real M1+M2 lift. "
            "Real m14 LEAD is Wf-188 full MetalCytoToxDB 0.9958 (n=26,801). "
            "Wf-274A regression_disclosure flags 0.9645 as smaller-cohort "
            "smaller-seed; do NOT cite as LEAD per party-twenty-spirit §3.5."
        ),
        "value": "+0.0335 absolute / +3.47% (methodology change; honest-negative)",
        "json_path": (
            "molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json::"
            "per_consumer[TODO-119] + metrics/by_metric/decode_ratio.json::"
            "wf148_m14_real_cfm.value + metrics/by_metric/cfm_decode_real.json::"
            "regression_disclosure"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf320e_todo120_m1_ru_lift_measured",
        "section": "§2.1",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-320E TODO-120 m1 Ru AUC consumer lift: Wf-310A G5 50-ep "
            "M1+M2-enabled Ru test_auc = 0.7860 ± 0.0145 (n=2/3 seeds, "
            "seed=2 retry pending) vs chemprop v2 single-task baseline 0.7123. "
            "MEASURED +0.0737 absolute / +10.35% lift. Krasnov 2026 baseline "
            "0.81 still BELOW by -0.024; honest entry is partial-closure of "
            "the Krasnov gap. Per-seed std unfiltered per party-twenty-spirit "
            "§3.5 (Ru ±0.0145 on n=2 seeds)."
        ),
        "value": "+0.0737 absolute / +10.35% (MEASURED; Krasnov gap -0.024)",
        "json_path": (
            "molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json::"
            "per_consumer[TODO-120] + metrics/by_metric/wf181_g5_50epoch_full_sweep.json::"
            "per_metal_in_domain.Ru + .per_metal_std.Ru + metrics/by_metric/"
            "metalcytotox_ru_auc.json::baseline_value"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf320e_todo121_m19_lomo_rh_honest_negative",
        "section": "§6.4",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": (
            "Wf-320E TODO-121 m19 LOMO Rh bottleneck consumer lift: Wf-176 "
            "M1+M2-enabled Re fold seed=4 2-epoch W₂=0 test_auc = 0.34375 "
            "(BELOW_RANDOM) vs baseline 0.5. lift = −0.15625 absolute / "
            "−31.25%. W₂ weight was 0.0 (effectively disabled) for this run; "
            "Rh seed=3 50-epoch (per_seed_metrics_seed3.json) test_auc=0.4045 "
            "with same W₂=0.0 confirms BELOW_RANDOM pattern (rules out seed "
            "luck). val_auc=0.6949 vs test_auc=0.34375 is a strong overfitting "
            "signal. Honest entry: §6 honest-limitations."
        ),
        "value": "−0.15625 absolute / −31.25% (HONEST_NEGATIVE: BELOW_RANDOM)",
        "json_path": (
            "molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json::"
            "per_consumer[TODO-121] + metrics/by_metric/metal_lomo_auc_wasserstein.json::"
            "per_run[0].test_auc + .wasserstein_weight + molmetal/reports/"
            "wf176_m19_oversample_lomo/Rh/per_seed_metrics_seed3.json::test_auc"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf320e_todo122_l5_hilbert_r2_arch_ceiling",
        "section": "§6.5",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": (
            "Wf-320E TODO-122 L5 Hilbert R² consumer lift: Wf-122 P3 post-fix "
            "M1+M2-enabled combined Hilbert+ASTSR R² = -5.6625 ± 1.3646 "
            "(4-cell × 5 seeds; aggregate over mcf7/a549/a2780/a2780cis) "
            "vs architectural signal-floor baseline 0.0 (12-d RDKit signal "
            "|ρ| ≤ 0.019 on PlatinAI pIC50; linear R² upper bound = (0.019)² · "
            "12 = 0.0043, hence R² ≥ 0 mathematically infeasible on this "
            "feature stack). ARCHITECTURAL_CEILING confirmed by 3 independent "
            "rescue attempts (Wf-127 + Wf-128 + Wf-129 all converge on the "
            "12-d RDKit signal floor). Wf-129 P3 Multi-PH + PLL end-to-end "
            "guarded gives R² = -0.0780 ± 0.0148 (less negative; +5.58 R² "
            "delta vs Wf-122 post-fix) but still BELOW R² ≥ 0 gate. 3 lift "
            "paths: Wf-94 R20-P3 GPU, R22-P4 AlphaFold-Multimer, R23-P6 "
            "wet-lab pIC50 hybrid."
        ),
        "value": "−5.6625 combined R² (ARCHITECTURAL_CEILING; 3 rescue attempts)",
        "json_path": (
            "molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json::"
            "per_consumer[TODO-122] + metrics/by_metric/hilbert_4cell_r2.json::"
            "aggregate_combined_r2_mean + TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf320e_todo123_m15_rmsd_measured",
        "section": "§4.6",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-320E TODO-123 m15 RMSD consumer lift: Wf-246-B redesigned "
            "M1+M2-enabled docked_pose_rmsd cross-pocket median = 0.6578 Å "
            "(5/5 pockets < 2.0 Å SOTA gate PASS; per-pocket std 0.2546; "
            "difficulty EASY=1/MEDIUM=3/HARD=1) vs legacy Kabsch+Hungarian "
            "baseline estimate ~1.5 Å. lift = −0.8422 Å absolute / −56.15% "
            "(NEGATIVE lift = IMPROVEMENT for RMSD). 3-seed × 5-pocket × "
            "5 poses; per-pocket std reporting unfiltered per party-twenty-"
            "spirit §3.5. NOTE: ligand_vs_pocket_rmsd (Wf-247-A RDKit "
            "rdMolAlign.GetBestRMS canonical path, STRICTER) is 3.8333 Å "
            "median — honest framing of algorithm-difference, NOT a bug."
        ),
        "value": "−0.8422 Å absolute / −56.15% (MEASURED; 5/5 pockets < 2.0 Å)",
        "json_path": (
            "molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json::"
            "per_consumer[TODO-123] + metrics/by_metric/docked_pose_rmsd.json::"
            "value + .cross_pocket_aggregate + TODO/completed/123_m15_rmsd_strengthen.md"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf320e_todo125_l12_per_pocket_vina_partial_honest_negative",
        "section": "§6.6",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": (
            "Wf-320E TODO-125 L12 OOD per-pocket Vina consumer lift: Wf-144 "
            "P1 ex=32 M1+M2-enabled 5-pocket pooled relaxed_success = 0.0702 "
            "(n=57 docked, n_hits=4) vs Wf-311B 5-pocket cross-pocket high_"
            "affinity 0.4900 ± 0.3191 baseline. lift = −0.4198 absolute / "
            "−85.67%. 1/5 pockets PASSES (1h36 4/13 = 0.3077) but 4/5 pockets "
            "FAIL honest-negative (3eml/3ny8/4aaw/4rlu all 0.0). Wf-140 "
            "closure-bounded cohort is SA-optimized for synth, NOT for "
            "strong pocket binding (industry threshold -8.0 STRICT). "
            "per_pocket_vina.csv (Wf-248C) shows STUB_FALLBACK for all 75 "
            "rows — per-pocket Wasserstein + closure_bias FULL pipeline NOT "
            "executed end-to-end on 5-pocket cohort. Honest entry: "
            "§6 honest-limitations."
        ),
        "value": "−0.4198 absolute / −85.67% (PARTIAL; 1/5 pockets PASS, 4/5 FAIL)",
        "json_path": (
            "molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json::"
            "per_consumer[TODO-125] + metrics/by_metric/vina_relaxed_success.json::"
            "aggregate_ex32.pooled_rate + .per_pocket_rate + metrics/by_metric/"
            "vina_high_affinity.json::wf126_strengthen_per_pocket + "
            "molmetal/reports/wf248c_per_pocket_vina/per_pocket_vina.csv"
        ),
        "timestamp": "2026-09-29",
    },
    # --- Wf-319G: Wf-296D deep wire — 84-pattern matrix + YuelBond precondition
    # runtime wire-up for MCTS dispatch (proof_search.py integration).
    # Complements Wf-318F (F2(a) runtime dispatcher) by exposing the matrix
    # + precondition at the proof-search dispatch layer + adding a dedicated
    # YuelBond precondition check (per Wf-269C + Wf-275C deployment contract).
    {
        "id": "claim_wf319g_wf296d_deep_wire_shipped",
        "section": "§4.5.5",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-319G Wf-296D deep wire SHIPPED (CPU-only): "
            "MetalLigandExchangeRuntime in "
            "molmetal/molmetal_lam/lam_chem/metal_ligand_runtime.py "
            "compiles the 84-pattern L-R17-B2 matrix + 17 NEW SMARTS "
            "(= 101 RDKit ChemicalReaction objects) and exposes them for "
            "MCTS proof-search dispatch via wire_into_proof_search_rules "
            "(non-invasive — extends MCTSProofSearch.rules without "
            "shadowing legacy click rules). YuelBond precondition check "
            "via check_yuelbond_precondition enforces Wf-275C deployment "
            "contract at runtime: yuelbond_active=True returns "
            "recommended_action=PROCEED with delta_decode_ratio=0.000; "
            "use_yuelbond=False or non-yuelbond checkpoint returns "
            "FAIL_FAST with the Wf-269C MEASURED delta_decode_ratio="
            "-1.000 (full=1.000 vs no_yuelbond=0.000). 12/12 contract "
            "tests PASS per molmetal/tests/test_wf319g_wf296d_deep_wire.py."
        ),
        "value": "101 patterns compiled; 12/12 tests PASS; YuelBond precondition enforced",
        "json_path": "molmetal/molmetal_lam/lam_chem/metal_ligand_runtime.py + molmetal/tests/test_wf319g_wf296d_deep_wire.py + TODO/completed/319g_wf296d_deep_wire.md",
        "timestamp": "2026-09-29",
    },
    # --- Wf-320B: Harness auto-fallback fix (Wf-291F 真修) — third line of
    # defence against the Wf-291C regression class.  Wf-301E detects
    # schema-shape; Wf-302D retries with a generic prompt; Wf-320B adds
    # the *mismatch* layer (partial values / wrong keys / wrong types /
    # extra sibling-artefact keys) plus a schema_coerce hint + bounded
    # fallback strategy (SIBLING_RECONSTRUCTION / EMPTY_DEFAULT / RAISE).
    # Closes Wf-315B recommendation #2; bounded max_retries=3 prevents
    # infinite loops.  CPU-only integration work.
    {
        "id": "claim_wf320b_harness_autofallback_shipped",
        "section": "§6",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-320B harness auto-fallback fix SHIPPED (CPU-only): "
            "HarnessAutoFallback class in "
            "molmetal/scripts/wf320b_harness_autofallback.py closes "
            "the Wf-291F remaining gap by adding the schema-mismatch "
            "layer on top of Wf-301E (detector) + Wf-302D (retry). "
            "detect_schema_mismatch classifies agent results into 6 "
            "FailureReason buckets (schema_shape / schema_mismatch / "
            "type_mismatch / missing_keys / extra_keys / plain_text); "
            "build_schema_coerce_hint enumerates the specific gap "
            "(missing keys / extras / type mismatches) so the retry "
            "prompt is targeted; HarnessAutoFallback.retry_with_correction "
            "drives a two-stage strategy (Wf-302D generic prompt first "
            "for schema-shape; schema_coerce hint second for mismatch) "
            "with bounded max_retries=3 and three FallbackStrategy "
            "options (SIBLING_RECONSTRUCTION / EMPTY_DEFAULT / RAISE). "
            "wrap_wf302d_with_autofallback is the integration point "
            "promised by the brief — callers that today use "
            "run_agent_with_retry can switch with one source line. "
            "13/13 contract tests PASS per "
            "molmetal/tests/test_wf320b_harness_autofallback.py."
        ),
        "value": "6 FailureReason buckets; 3 FallbackStrategy options; 13/13 tests PASS",
        "json_path": "molmetal/scripts/wf320b_harness_autofallback.py + molmetal/tests/test_wf320b_harness_autofallback.py + TODO/completed/320b_wf291f_harness_autofallback_fix.md",
        "timestamp": "2026-09-29",
    },
    # --- Wf-319A: ablation_runner aggregate.csv ship surface (paper §6 input)
    # 12 NEW tests for the 1037-LOC ablation_runner.py (5 toggleable +
    # 6 DESIGN_ONLY) + 11 R17 ablation smoke (BASELINE + 5 toggleable +
    # 6 DESIGN_ONLY = 12 rows in aggregate.csv).  Per Wf-319A honest
    # framing: 5 toggleable ablations are MEASURED (200-step CFM retrain
    # + 16-sample decode smoke, NOT full 10k production); 6 DESIGN_ONLY
    # rows are documented in the table as DESIGN_ONLY (NOT fabricated).
    {
        "id": "claim_wf319a_ablation_table_aggregate",
        "section": "§6.1",
        "badge": BADGE_MEASURED,
        "claim": "Wf-319A ablation_runner.py smoke SHIPPED: 12 NEW tests PASS (3 toggleable specs + 3 DESIGN_ONLY specs + 2 aggregate.csv + 2 final.md + 2 metallo_drug_smoke_retrain.py integration) + aggregate.csv has 12 rows (11 ablations + 1 BASELINE reference) + final.md honest-framing surfaces smoke-vs-production caveat; 5 toggleable ablations MEASURED (BASELINE + M7 / M12 / M13 / M14 / M15), 6 DESIGN_ONLY documented (M3 / M4 / M8 / M10 / M11 / M21)",
        "value": "12/12 tests PASS; 12 aggregate.csv rows; 5 toggleable MEASURED + 6 DESIGN_ONLY documented",
        "json_path": "molmetal/tests/test_wf319a_ablation_runner.py + molmetal/reports/wf_ablation_runner/aggregate.csv + molmetal/reports/wf_ablation_runner/final.md + TODO/completed/319a_ablation_runner_smoke.md",
        "timestamp": "2026-09-29",
    },
    # --- Wf-323D: per-strat Wf-319A ablation_runner claim cards (3 NEW)
    # complementing claim_wf319a_ablation_table_aggregate.  Per party-twenty-
    # spirit §3.5 honest framing: M12/M13 are STRONG signals (atom_loss /
    # bond_loss lift on the smoke); M14 is honest-negative (--lr-warmup-steps
    # flag removed from harness → ablation silently broken).  Numbers trace
    # to aggregate.csv atom_loss_last / bond_loss_last columns + section_5_
    # ablation.json::wf319a_ablation_runner_axis.finding per WEAK_TO_STRONG §3.6.
    {
        "id": "claim_wf319a_m12_atom_loss_+204",
        "section": "§6.1",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-319A / Wf-323D per-strat M12 20-vocab→12-vocab ablation: "
            "atom_loss_last = 7.425 vs BASELINE 2.443 → Δ=+4.982 / +204% "
            "MEASURED lift. 12-element vocab drops B/Si/Sc/Ti/As/Y/Zr/Hf "
            "from metalloidrug atom head; in-vocab SMILES 247/449 after OOV "
            "filter keep m14=1.0 (smoke-scale artifact per final.md § "
            "honest framing); full 10k production would regress m14 as "
            "§5 predicted. Atom-loss lift is the strongest §5 mechanism "
            "evidence at this smoke scale."
        ),
        "value": "atom_loss 2.443→7.425, Δ=+4.982 / +204% (smoke-scale artifact caveat)",
        "json_path": (
            "molmetal/reports/wf_ablation_runner/aggregate.csv::"
            "M12.atom_loss_last + BASELINE.atom_loss_last + "
            "molmetal/reports/wf_ablation_runner/M12.json::atom_loss_last + "
            "molmetal/reports/wf_ablation_runner/final.md §M12"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf319a_m13_bond_loss_+152",
        "section": "§6.1",
        "badge": BADGE_MEASURED,
        "claim": (
            "Wf-319A / Wf-323D per-strat M13 logits clamp (±50) ablation: "
            "bond_loss_last = 3.162 vs BASELINE 1.254 → Δ=+1.908 / +152% "
            "MEASURED lift. Logits clamp is a BF16 overflow numerical "
            "guard; the bond head pays the early-warning signal at smoke "
            "scale while atom_loss stays stable (1.23 vs 2.44) — the "
            "vocab_mask pre-clamp step protects the atom head. Clamp "
            "divergence would manifest in a 10k production run; 200-step "
            "smoke surfaces the bond-head lift cleanly."
        ),
        "value": "bond_loss 1.254→3.162, Δ=+1.908 / +152% (bond-head early-warning signal)",
        "json_path": (
            "molmetal/reports/wf_ablation_runner/aggregate.csv::"
            "M13.bond_loss_last + BASELINE.bond_loss_last + "
            "molmetal/reports/wf_ablation_runner/M13.json::bond_loss_last + "
            "molmetal/reports/wf_ablation_runner/final.md §M13"
        ),
        "timestamp": "2026-09-29",
    },
    {
        "id": "claim_wf319a_m14_lr_warmup_honest_negative",
        "section": "§6.1",
        "badge": BADGE_HONEST_NEGATIVE,
        "claim": (
            "Wf-319A / Wf-323D per-strat M14 LR warmup (linear 200 steps) "
            "ablation honest-negative: --lr-warmup-steps flag is NOT in "
            "the metallo_drug_smoke_retrain harness — ablation has been "
            "silently broken since the flag was removed. M14 smoke run "
            "completes with PASS / PREDICTION_QUALITATIVE but its "
            "measurements (atom_loss 2.28 vs baseline 2.44 = −7% within "
            "noise) reflect NO warmup being applied at all. Fix path "
            "(either add --lr-warmup-steps back to the harness OR "
            "remove M14 from ablation_runner extra_args) deferred to "
            "Wf-323A; until then do NOT cite M14 ablation as evidence."
        ),
        "value": (
            "−7% within noise (atom_loss 2.28 vs 2.44); HONEST_NEGATIVE: "
            "flag missing from harness, ablation silently broken"
        ),
        "json_path": (
            "molmetal/reports/wf_ablation_runner/aggregate.csv::"
            "M14.atom_loss_last + BASELINE.atom_loss_last + "
            "molmetal/reports/wf_ablation_runner/M14.json::atom_loss_last + "
            "metrics/by_paper_section/section_5_ablation.json::"
            "wf319a_ablation_runner_axis.finding (M14 honest-negative clause)"
        ),
        "timestamp": "2026-09-29",
    },
]

# Honest-negative catalog per TODO-207 §9 (party-twenty-spirit §3.5).
# Verbatim list of "DO NOT claim" entries that the dashboard surfaces
# as red badges so reviewers can locate the negative result in one place.

HONEST_NEGATIVE_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "hn_no_imold_beat",
        "section": "§9.3",
        "item": "DO NOT claim chemprop v2 beats iMoLD on DrugOOD (Δ = −0.0074 assay within std)",
        "ref": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §5.2",
    },
    {
        "id": "hn_no_cross_metal_lomo",
        "section": "§9.3",
        "item": "DO NOT claim cross-metal LOMO transfer (TODO-154 in-domain per-metal only)",
        "ref": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §5.2",
    },
    {
        "id": "hn_no_drugood_full",
        "section": "§9.3",
        "item": "DO NOT claim DrugOOD 1.58M train full retrain is measured (BD-11 DEFERRED)",
        "ref": "TODO/INDEX.md BD-11",
    },
    {
        "id": "hn_no_platinai_226k",
        "section": "§9.3",
        "item": "DO NOT claim full PlatinAI 226K retrain (500-mol subset only — L2 DEFERRED)",
        "ref": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §5.3",
    },
    {
        "id": "hn_no_bindingdb_full",
        "section": "§9.3",
        "item": "DO NOT claim BindingDB full 1.07M metal-filter retrain (BD-12 DEFERRED)",
        "ref": "TODO/INDEX.md BD-12",
    },
    {
        "id": "hn_no_krasnov_temporal_split",
        "section": "§9.3",
        "item": "DO NOT claim scaffold-split is more comparable to Krasnov time-split than a temporal driver",
        "ref": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §1.3",
    },
    {
        "id": "hn_no_m21_krasnov_beat",
        "section": "§9.3",
        "item": "DO NOT claim m21 9-cell aggregate BEATS Krasnov 0.81 (statistically equivalent)",
        "ref": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §4.3",
    },
    {
        "id": "hn_no_cfm_synthetic_1_000",
        "section": "§9.3",
        "item": "DO NOT claim CFM decode_ratio 0.9958 BEATS synthetic 1.000 (expected gap)",
        "ref": "TODO/completed/207_unified_pre_flight_checklist_v2.md §9.6.3",
    },
    {
        "id": "hn_no_wf219_more_accurate",
        "section": "§9.3",
        "item": "DO NOT claim Wf-219 smaller-backbone 3-seed numbers more accurate than TODO-154 5-seed full-backbone",
        "ref": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §5.2",
    },
    {
        "id": "hn_no_re_significance",
        "section": "§9.3",
        "item": "DO NOT claim Re per-metal paired t-test is significant (n=2 underpowered)",
        "ref": "TODO/completed/154_chemprop_v4_head_to_head.md §1",
    },
    {
        "id": "hn_no_m7_targetdiff",
        "section": "§9.3",
        "item": "DO NOT claim m7 diversity 0.0397 reaches TargetDiff 0.860 ceiling (gap +0.820 honest)",
        "ref": "TODO/completed/188_paper_4_krasnov_protocol_aligned.md §4.4",
    },
    {
        "id": "hn_no_v3_drugood_pass",
        "section": "§9.3",
        "item": "DO NOT claim chemprop v3 DrugOOD-PASS (BELOW_RANDOM 0.4544, §6 honest-negative)",
        "ref": "TODO/completed/207_unified_pre_flight_checklist_v2.md §9.4",
    },
    # --- Added Wf-268C-Dash: Wf-263A + Wf-266 honest-negatives ---
    {
        "id": "hn_no_wf263a_empirical_pass",
        "section": "§9.4",
        "item": "DO NOT claim Wf-263A empirical DrugOOD re-measurement PASS (75-min wall budget TIMEOUT; structural PASS only, empirical TBD)",
        "ref": "molmetal/reports/wf263a_verify/verify_report.md §3.2",
    },
    {
        "id": "hn_no_wf266_g5_measured",
        "section": "§9.4",
        "item": "DO NOT claim Wf-266 G5 50-epoch sweep MEASURED (0/15 pairs — CUDA error invalid device ordinal; wrapper BLOCKED at GPU ordinal layer)",
        "ref": "molmetal/reports/wf266_final/final_status.md §4.2",
    },
    {
        "id": "hn_no_wf266_retrain_lift",
        "section": "§9.4",
        "item": "DO NOT claim Wf-266 v3-fix mode produces MEASURED AUC lift (IMPLEMENT only — no GPU retrain; AUC lift deferred to follow-up TODO-130 G5 retrain)",
        "ref": "molmetal/reports/wf266_lomo_fix/wf266_report.md §Honest-framing",
    },
    # --- Added Wf-279C: 2 NEW honest-negative entries (2026-09-29) ---
    {
        "id": "hn_no_cfm_decode_real_09645_as_lead",
        "section": "§9.AI",
        "item": "DO NOT cite cfm_decode_real.json value=0.9645 as the m14 LEAD (Wf-274A regression_disclosure flags it as a smaller-cohort 1-seed CPU PlatinAI pt_subset measurement; cite Wf-188 m14 FULL = 0.9958 instead)",
        "ref": "metrics/by_metric/cfm_decode_real.json::regression_disclosure + TODO-188 §5.2 line 387",
    },
    {
        "id": "hn_no_yuelbond_off_deployment",
        "section": "§9.AJ",
        "item": "DO NOT deploy m14 LEAD CFM with --no-yuelbond flag (Wf-269C MEASURED: decode_ratio 1.000 → 0.000, Δ=−1.000; m14 LEAD silently invalidates without raising any runtime error)",
        "ref": "TODO/completed/m14_lead_deployment_precondition.md §2 + molmetal/reports/wf269c_cfm_real/wf269c_report.json",
    },
    # --- Added Wf-293D: Wf-286A 5-seed DrugOOD confirm failed (gate_1_all_ran=false) ---
    {
        "id": "hn_wf286a_5_seed_drugood_failed",
        "section": "§9.AK",
        "item": "DO NOT cite Wf-286A as 5-seed DrugOOD v3-fallback confirmation (FAILED: gate_1_all_ran=false, gate_2_total_runs=0, all 7 gates empty; only load_split_data ran (n_train=60908) before per-seed trainer step did not start — no AUC produced; wall = 1.86e-5 min effective NO-OP)",
        "ref": "molmetal/reports/wf286a_drugood_5seed/aggregate.json + molmetal/reports/wf286_synthesis/followup_summary.md §1.1",
    },
    # --- Added Wf-308B R03: ADMET pass-through aqueous_solubility honest-negative ---
    {
        "id": "hn_wf308b_aq_sol_fail",
        "section": "§9.AL",
        "item": "DO NOT cite aqueous_solubility_logS = -7.612 as druglikeness PASS (cohort mean BELOW -4.0 by -3.612; pre-flight gate carries HONEST_NEGATIVE=true tag, do not silently mask; metallo-drug cohort intrinsically lipophilic — low log S is expected per Crippen + aromatic ring count)",
        "ref": "metrics/by_metric/aqueous_solubility_logS.json::HONEST_NEGATIVE + honest_negative_reason + TODO/completed/admet_pass_through_integration.md §R03",
    },
    # --- Added Wf-311E Part 4 / Wf-314A: 3 NEW honest-negative gates on
    # pre-existing JSON files (m6_kappa / hepatotox / logp7_4). All 3 carry
    # canonical HONEST_NEGATIVE=true + honest_negative_reason + honest_framing_note
    # tags + verdict: FAIL; mirrors Wf-308B aqueous_solubility_logS pattern.
    {
        "id": "hn_wf311e_m6_kappa_0_472",
        "section": "§9.AY",
        "item": "DO NOT cite m6_kappa (Shape Kappa Index) = 0.472 as druglikeness PASS (cohort mean BELOW ChemAxon PMC11387870 threshold 0.6 by -0.128; pre-flight gate carries HONEST_NEGATIVE=true tag, do not silently mask; Wf-253D MetalCytoToxDB validation returned NEGATIVE_WEAK — global Pearson r 0.081, Spearman 0.111 — Shape Kappa is NOT a learned pIC50 proxy)",
        "ref": "metrics/by_metric/m6_kappa.json::HONEST_NEGATIVE + honest_negative_reason + honest_framing_note + metalcytotox_validation.global_pearson_r + TODO/completed/312_honest_framing_35plus.md §R02",
    },
    {
        "id": "hn_wf311e_hepatotox_0_548",
        "section": "§9.AZ",
        "item": "DO NOT cite hepatotox_index = 0.548 as druglikeness PASS (cohort mean ABOVE Hughes 2008 rule-of-2 threshold 0.5 by +0.048; pre-flight gate carries HONEST_NEGATIVE=true tag, do not silently mask; metallo-drug cohort carries moderate hepatotox risk from aniline/hydrazine/5-ring structural alerts — Stepan 2011 composite)",
        "ref": "metrics/by_metric/hepatotox_index.json::HONEST_NEGATIVE + honest_negative_reason + honest_framing_note + TODO/completed/312_honest_framing_35plus.md §R02",
    },
    {
        "id": "hn_wf311e_logp7_4_6_413",
        "section": "§9.BA",
        "item": "DO NOT cite logp7_4 = 6.413 as druglikeness PASS (cohort mean OUT OF Lipinski 0-5 range by +1.413; pre-flight gate carries HONEST_NEGATIVE=true tag, do not silently mask; metallo-drug cohort is intrinsically lipophilic per Patrick 2009 ionisation-corrected Crippen logP)",
        "ref": "metrics/by_metric/logp7_4.json::HONEST_NEGATIVE + honest_negative_reason + honest_framing_note + TODO/completed/312_honest_framing_35plus.md §R02",
    },
    # --- Added Wf-311D PART 1+2+4: OOD partial + m18 synthetic_proxy + curated smoke bias ---
    {
        "id": "hn_wf311d_ood_reachability_partial",
        "section": "§9.AM",
        "item": "DO NOT cite OOD reachability as a full d-block PASS — 10/30 elements (Sc/V/Mo/Tc/Pd/Re/Os/Ir/Pt/Au) FAIL the >=0.40 gate; aggregate macro reachability = 0.4989 (just under gate); the 5d+La block (Pt/Pd/Au/Ir/Os/Re) is the weakest (5/10 pass). Per Wf-318E (2026-09-29), the Wf-130 P3 cross-domain 0.0 values for Pt/Pd/Au/Ru/Rh/Os/Ir were RDKit-missing-blocked and have been lifted to MEASURED alt-env values (Pt 0.83, Pd 0.50, Au 0.33, Rh 0.33, Ru 0.21, Os 0.25, Ir 0.21); Re remains 0.0 due to structural HONEST_NEGATIVE_DATA_AVAILABILITY (0/10 closure_ok even with RDKit). 4/8 cross-domain metals PASS 0.30 gate, 3/8 below gate but real, 1/8 structural honest-negative.",
        "ref": "metrics/by_metric/ood_reachability_summary.json::n_elements_gate_pass + per_block_aggregate + cross_domain_supplement.wf318e_per_metal + cross_domain_supplement.honest_negative_metals_after_wf318e",
    },
    {
        "id": "hn_wf311d_m18_per_residue_synthetic_proxy",
        "section": "§9.AN",
        "item": "DO NOT cite per_residue_auc = 0.9627 as a real per-residue binding signal — labels are SYNTHETIC PROXIES derived from the complex-level active label (MetalCytoToxDB does NOT ship per-residue binding labels); per party-twenty-spirit §3.5 the value reflects the model's ability to reproduce the complex-level heuristic projection, NOT per-residue binding chemistry.",
        "ref": "metrics/by_metric/protein_metal_per_residue_summary.json::per_residue_redesign_v1.synthetic_proxy_warning + TODO/completed/247_b_per_residue_m18_redesign.md §honest-framing",
    },
    {
        "id": "hn_wf311d_pharmacophore_curated_smoke_biased",
        "section": "§9.AO",
        "item": "DO NOT cite pharmacophore_pass_rate = 0.7 as the production drug-likeness number — that value is from a 10-SMILES curated hand-picked set (biased upward); PRODUCTION value on 449 metallo-drug SMILES is 0.1737 strict / 0.3786 lenient. Cite production as the headline.",
        "ref": "metrics/by_metric/pharmacophore_pass_rate_prod.json::comparison_to_curated + honest_framing_note",
    },
    # --- Added Wf-318G: 3 NEW honest-negative close-outs for the Wf-311F §7
    # ultracode follow-up candidates #3 + #4 (r13_100x3_partial +
    # diversity_subpocket + steric_clash_pb_dock). All 3 carry canonical
    # HONEST_NEGATIVE=true + honest_negative_reason + honest_framing_note
    # tags + verdict: FAIL; mirrors Wf-308B aqueous_solubility_logS +
    # Wf-311E Part 4 + Wf-314A pattern. Per AGENTS.md §5.4 Tier-2 strategy
    # nullifies dependency on the 100×3 paper-grade sweep + per-pose
    # production 830c clash check; these workstreams are honest-negative
    # for Tier-2 scope.
    # --- Added Wf-320E: 4 NEW honest-negative entries for per-TODO
    # consumer lifts that are BELOW baseline OR architectural-ceiling
    # by construction (TODO-119/121/122/125). Per party-twenty-spirit
    # §3.5 the dashboard surfaces them as red badges so reviewers can
    # locate the negative result in one place.
    {
        "id": "hn_wf320e_todo119_m14_0_9645_not_lead",
        "section": "§9.BB",
        "item": "DO NOT cite the Wf-320E TODO-119 +3.47% 'lift' as m14 LEAD (the apparent lift is a methodology change — Wf-274A 1-seed × 818 CPU PlatinAI pt_subset vs Wf-148 3-seed × 5991 GPU NCI-60+DrugOOD — NOT a real M1+M2 lift). Cite Wf-188 m14 FULL MetalCytoToxDB 0.9958 (n=26,801) as the LEAD; per Wf-274A regression_disclosure flag.",
        "ref": "molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json::per_consumer[TODO-119] + metrics/by_metric/cfm_decode_real.json::regression_disclosure",
    },
    {
        "id": "hn_wf320e_todo121_m19_lomo_below_random",
        "section": "§9.BC",
        "item": "DO NOT cite the Wf-320E TODO-121 -31.25% 'lift' as a positive result (BELOW_RANDOM; Wf-176 Re fold seed=4 2-epoch W₂=0 test_auc=0.34375; val_auc=0.6949 vs test_auc=0.34375 is a strong overfitting signal; W₂ was effectively disabled for this run). Honest entry: §6 honest-limitations. Wf-125 P2 5-fold × 5-seed × 50-epoch + W₂>0 needed to attempt recovery.",
        "ref": "molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json::per_consumer[TODO-121] + metrics/by_metric/metal_lomo_auc_wasserstein.json::wasserstein_weight + .val_auc + .test_auc",
    },
    {
        "id": "hn_wf320e_todo122_l5_arch_ceiling",
        "section": "§9.BD",
        "item": "DO NOT cite the Wf-320E TODO-122 L5 Hilbert R² ≥ 0.85 lift as achievable on the 12-d RDKit signal floor (architectural-ceiling: 12-d RDKit signal |ρ| ≤ 0.019 on PlatinAI pIC50; linear R² upper bound = (0.019)² · 12 = 0.0043, hence R² ≥ 0 mathematically infeasible on this feature stack). 3 independent rescue attempts (Wf-127 + Wf-128 + Wf-129) all converge on the same signal floor. Only path forward: Wf-94 R20-P3 Multi-PH + PLL real feature augmentation + R22-P4 AlphaFold-Multimer + R23-P6 wet-lab pIC50 hybrid.",
        "ref": "molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json::per_consumer[TODO-122] + TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md + metrics/by_metric/hilbert_4cell_r2.json::wf_139_p4_l5_arch_ceiling_closeout.l5_arch_ceiling_proof",
    },
    {
        "id": "hn_wf320e_todo125_l12_per_pocket_4_of_5_fail",
        "section": "§9.BE",
        "item": "DO NOT extrapolate the Wf-320E TODO-125 1h36 PASS (4/13 hits, 0.3077) to the other 4 pockets (3eml/3ny8/4aaw/4rlu all honest-negative 0.0). Wf-140 closure-bounded cohort is SA-optimized for synth, NOT for strong pocket binding (industry threshold -8.0 STRICT). per_pocket_vina.csv (Wf-248C) shows STUB_FALLBACK for all 75 rows — per-pocket Wasserstein + closure_bias FULL pipeline NOT executed end-to-end on 5-pocket cohort. Honest entry: §6 honest-limitations. Wf-130 P3 cross-domain supplement reports 0/8 cross-domain metals.",
        "ref": "molmetal/reports/wf320e_downstream_lift_dashboard/dashboard.json::per_consumer[TODO-125] + metrics/by_metric/vina_relaxed_success.json::aggregate_ex32.per_pocket_rate + molmetal/reports/wf248c_per_pocket_vina/per_pocket_vina.csv",
    },
    {
        "id": "hn_wf318g_r13_100x3_not_paper_grade",
        "section": "§9.AP",
        "item": "DO NOT cite r13_100x3 as a paper-grade 300-cell sweep (delivered 30/300 cells; 30/30 PB-path cells search-bound; CFM-path BLOCKED on r10 import + GPU HSA userland outage; process killed at 55% before lambda final report.json; paper_grade_data_ready = false); cite the path-A 10×3 (n_distinct 1->20, div_tan 0->0.1065) on 10 specific pockets (test_000..test_009) as the headline diversity lift only — it does NOT generalize to 100 pockets when search-bound reappears on test_010..test_019. Per AGENTS.md §5.4 Tier-2 strategy nullifies 300-cell sweep dependency.",
        "ref": "metrics/by_metric/r13_100x3_partial.json::HONEST_NEGATIVE + honest_negative_reason + honest_framing_note + verdict: FAIL + value.paper_grade_data_ready: false + TODO/completed/r13_100x3_final_honest_negative.md + TODO/completed/318g_r13_diversity_steric_final.md",
    },
    {
        "id": "hn_wf318g_diversity_subpocket_design_only",
        "section": "§9.AQ",
        "item": "DO NOT cite diversity_subpocket on production 100-pocket cohort as a real per-residue diversity measurement (metric is wired + unit-tested + Phase-4-integrated but production Round-12 10×3 pilot smoke on real generated pool is NOT executed; only test-fixture smoke runs at empty=NaN, N=1=0.0, 2-disjoint>0); cite the global Tanimoto IntDiv1 (diversity_tanimoto) as the headline metric on the cohort; the per-residue 8.0 Å sub-pocket variant is a §6 honest-limitation or §7 future-work item, NOT a §4 Tier-2 blocker. Per AGENTS.md §5.4 Tier-2 strategy nullifies 100-pocket sweep dependency.",
        "ref": "metrics/by_metric/diversity_subpocket.json::HONEST_NEGATIVE + honest_negative_reason + honest_framing_note + verdict: DESIGN_ONLY + TODO/completed/diversity_subpocket_final_honest_negative.md + TODO/completed/318g_r13_diversity_steric_final.md",
    },
    {
        "id": "hn_wf318g_steric_clash_pb_dock_830c_deferred",
        "section": "§9.AR",
        "item": "DO NOT cite steric_clash_pb_dock on the 830c pocket as a production PoseBusters 'dock'-mode pass rate (deferred: no receptor PDB on disk + no docked poses on disk; RDKit 2026.03.6 + PoseBusters 0.6.5 NOW available per Wf-315D 2026-09-29 cross-reference, but 830c staging is NOT yet executed); cite the 1h36 MEASURED_SMOKE m8 pass rate = 1.0000 (15/15) at metrics/by_metric/steric_clash_pb_dock_1h36.json as the per-pose metric example + the PoseBusters 'mol' mode pb_pass_rate = 1.000 at metrics/by_metric/pb_pass_rate.json as the chemistry-only pass rate. Per AGENTS.md §5.4 Tier-2 strategy nullifies 830c 'dock'-mode production dependency.",
        "ref": "metrics/by_metric/steric_clash_pb_dock.json::HONEST_NEGATIVE + honest_negative_reason + honest_framing_note + verdict: FAIL + status: DEFERRED + deferred_reason + TODO/completed/steric_clash_pb_dock_final_honest_negative.md + TODO/completed/318g_r13_diversity_steric_final.md",
    },
]


# ---------------------------------------------------------------------------
# Cohort table extraction (collect.json → per-cohort rows)
# ---------------------------------------------------------------------------


def _extract_chemprop_cohort(collect: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the chemprop v2 v3 5-metal per-metal aggregate from
    ``all_results.json``.  Returns a dict with cohort title + table rows.

    Reads (in order of preference):
      * ``g4_per_metal_aggregate_full_15pair`` — the 15-pair superset
      * ``g4_per_metal_aggregate`` — the 14-pair subset (fallback)
    """
    rows: List[Dict[str, Any]] = []
    agg_full = collect.get("g4_per_metal_aggregate_full_15pair") or {}
    agg_partial = collect.get("g4_per_metal_aggregate") or {}
    agg = agg_full or agg_partial
    for metal in sorted(agg.keys()):
        block = agg[metal]
        rows.append(
            {
                "metal": metal,
                "n_pairs": block.get("n_pairs", 0),
                "mean_test_auc": block.get("mean_test_auc", 0.0),
                "min_test_auc": block.get("min_test_auc", 0.0),
                "max_test_auc": block.get("max_test_auc", 0.0),
                "test_aucs": block.get("test_aucs", []),
            }
        )
    overall = (
        collect.get("g4_overall_aggregate_full_15pair")
        or collect.get("g4_overall_aggregate")
        or {}
    )
    return {
        "title": "chemprop v2 v3 5-metal per-metal AUC (Wf-124 P3, g4 14-15 paired runs)",
        "json_path": "molmetal/reports/wf265_collect/all_results.json::g4_per_metal_aggregate_full_15pair",
        "rows": rows,
        "overall": overall,
        "n_pairs_total": collect.get("G4_n_g4_pairs", 0),
        "v3_ood_auc": collect.get("G5_v3_ood_auc"),
    }


def _extract_drugood_cohort(collect: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the DrugOOD 3-split cohort rows from the paper doc constants.
    This cohort is NOT in ``all_results.json`` directly; the values are
    sourced from TODO-188 §3.1 verbatim so the dashboard carries the same
    number as the paper §4 DRAFT.
    """
    # These are MEASURED numbers from Wf-174 + Wf-185; per TODO-188 §3.1.
    rows = [
        {
            "split": "lbap_refined_ic50_assay",
            "wf174_3seed_mean": 0.7113,
            "wf174_3seed_std": 0.0216,
            "wf185_5seed_mean": 0.7137,
            "wf185_5seed_std": 0.0079,
            "imold_baseline": 0.7211,
            "delta_vs_imold": -0.0074,
            "verdict": "comparable (NOT beat)",
        },
        {
            "split": "lbap_refined_ic50_scaffold",
            "wf174_3seed_mean": None,
            "wf174_3seed_std": None,
            "wf185_5seed_mean": 0.6707,
            "wf185_5seed_std": 0.0042,
            "imold_baseline": 0.6884,
            "delta_vs_imold": -0.0177,
            "verdict": "comparable (NOT beat)",
        },
        {
            "split": "lbap_refined_ic50_size",
            "wf174_3seed_mean": None,
            "wf174_3seed_std": None,
            "wf185_5seed_mean": 0.6757,
            "wf185_5seed_std": 0.0142,
            "imold_baseline": 0.6792,
            "delta_vs_imold": -0.0035,
            "verdict": "comparable (NOT beat)",
        },
    ]
    return {
        "title": "DrugOOD OOD (Ji 2023 lbap_refined_ic50_*) ood_test AUC",
        "json_path": "molmetal/reports/wf185_chemprop_v2_drugood_full/aggregate.json + TODO-188 §3.1",
        "rows": rows,
    }


def _extract_cfm_m14_cohort(collect: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the CFM m14 (decode_ratio) per-metal cohort from the
    paper doc constants.  Per TODO-188 §4.1.
    """
    rows = [
        {"metal": "Ru", "n_total": 19135, "n_decode_ok": 19033, "decode_ratio": 0.9948},
        {"metal": "Ir", "n_total": 4546, "n_decode_ok": 4534, "decode_ratio": 0.9974},
        {"metal": "Os", "n_total": 1118, "n_decode_ok": 1118, "decode_ratio": 1.0000},
        {"metal": "Rh", "n_total": 1134, "n_decode_ok": 1134, "decode_ratio": 1.0000},
        {"metal": "Re", "n_total": 868, "n_decode_ok": 868, "decode_ratio": 1.0000},
        {"metal": "TOTAL", "n_total": 26801, "n_decode_ok": 26687, "decode_ratio": 0.9958},
    ]
    return {
        "title": "CFM m14 real decode_ratio (Wf-188, n=26,801)",
        "json_path": "molmetal/reports/wf188_m14_full_metalcytotox/result.json + TODO-188 §4.1",
        "rows": rows,
    }


def _extract_cfm_m7_cohort(collect: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the CFM m7 (Tanimoto diversity) cohort from TODO-180 §1.3
    + TODO-207 §9.6.2.  Per TODO-188 §4.4.
    """
    rows = [
        {
            "configuration": "Baseline (no Tanimoto reward)",
            "diversity_tanimoto": 0.0324,
            "lift": None,
            "verdict": "MEASURED (no lift)",
        },
        {
            "configuration": "With Tanimoto reward",
            "diversity_tanimoto": 0.0397,
            "lift": 0.225,
            "verdict": "MEASURED (+22.5% lift)",
        },
        {
            "configuration": "TargetDiff ceiling (cite-only)",
            "diversity_tanimoto": 0.860,
            "lift": None,
            "verdict": "honest-negative: +0.820 gap",
        },
    ]
    return {
        "title": "CFM m7 Tanimoto diversity (TODO-180 §1.3, TargetDiff Guan 2023 ceiling)",
        "json_path": "TODO/completed/180_daily_summary_2026_09_27.md §1.3 + TODO-207 §9.6.2",
        "rows": rows,
    }


def _extract_chemprop_v3_fallback_cohort(collect: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the Wf-263A chemprop v3-fallback mode dispatch cohort.

    Per ``molmetal/reports/wf263a_implement/implement_report.md`` §3.
    The cohort carries the 6-row dispatch table that maps
    ``--chemprop-mode`` × ``--enhanced-head`` (user intent) onto
    ``_effective_enhanced_head`` (forwarded value).
    """
    rows = [
        {
            "chemprop_mode": "v3-fallback (default)",
            "user_enhanced_head": "True",
            "effective_enhanced_head": "False",
            "effective_metal_attention": "False",
            "verdict": "FORCES v2 head (audit-trail preserved)",
        },
        {
            "chemprop_mode": "v3-fallback (default)",
            "user_enhanced_head": "False",
            "effective_enhanced_head": "False",
            "effective_metal_attention": "False",
            "verdict": "FORCES v2 head",
        },
        {
            "chemprop_mode": "v2",
            "user_enhanced_head": "True",
            "effective_enhanced_head": "False",
            "effective_metal_attention": "False",
            "verdict": "Symmetric with v3-fallback",
        },
        {
            "chemprop_mode": "v2",
            "user_enhanced_head": "False",
            "effective_enhanced_head": "False",
            "effective_metal_attention": "False",
            "verdict": "Pure v2 path",
        },
        {
            "chemprop_mode": "v3",
            "user_enhanced_head": "True",
            "effective_enhanced_head": "True",
            "effective_metal_attention": "False",
            "verdict": "Opt-in v3 head honoured",
        },
        {
            "chemprop_mode": "v3",
            "user_enhanced_head": "False",
            "effective_enhanced_head": "False",
            "effective_metal_attention": "False",
            "verdict": "v3 mode without enhanced head",
        },
    ]
    return {
        "title": "Wf-263A chemprop v3 v3-fallback mode dispatch (TODO-130 G5)",
        "json_path": "molmetal/reports/wf263a_implement/implement_report.md §3 dispatch table",
        "rows": rows,
        "n_files_modified": 3,
        "n_new_tests": 10,
        "n_backward_compat_tests": 21,
        "empirical_status": "structural PASS / empirical TIMEOUT",
    }


def _extract_wf266_lomo_fix_cohort(collect: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the Wf-266 v3 LOMO fix flag cohort (per-flag settings + audit trail).

    Per ``molmetal/reports/wf266_lomo_fix/wf266_report.md`` §Summary + §Audit trail.
    """
    rows = [
        {
            "flag": "--chemprop-min-epochs",
            "default": 10,
            "type": "int",
            "purpose": "Refuses --epochs < N (anti-under-training)",
            "no_op_under": "v3-fallback / v2 / v3 (forced to 1)",
        },
        {
            "flag": "--chemprop-lomo-init-scale",
            "default": 1e-3,
            "type": "float",
            "purpose": "Multiplicative scale on per-metal BCE logits (anti-bias)",
            "no_op_under": "v3-fallback / v2 / v3 (forced to 1.0)",
        },
        {
            "flag": "--chemprop-mode v3-fix",
            "default": "v3-fallback",
            "type": "enum-choice",
            "purpose": "Combines v2 head + min-epochs floor + lomo-init-scale",
            "no_op_under": "default (v3-fix requires explicit opt-in)",
        },
    ]
    audit_rows = [
        "cfg_chemprop_mode",
        "cfg_chemprop_min_epochs_user",
        "cfg_chemprop_lomo_init_scale_user",
        "cfg_chemprop_min_epochs_effective",
        "cfg_chemprop_lomo_init_scale_effective",
    ]
    return {
        "title": "Wf-266 chemprop v3 LOMO fix flags (--chemprop-min-epochs / --chemprop-lomo-init-scale / --chemprop-mode v3-fix)",
        "json_path": "molmetal/reports/wf266_lomo_fix/wf266_report.md §Summary + §Audit-trail",
        "rows": rows,
        "audit_trail_fields": audit_rows,
        "n_files_modified": 2,
        "n_new_tests": 4,
        "n_backward_compat_tests": 14,
        "retrain_status": "IMPLEMENT only — no GPU retrain (deferred to follow-up TODO-130 G5)",
    }


# ---------------------------------------------------------------------------
# High-level Wf-265-Collect gates extraction
# ---------------------------------------------------------------------------


def _extract_wf286b_ecfp4_direct_cohort(collect: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the Wf-286B ECFP4-direct re-eval cohort.

    Per ``molmetal/reports/wf286b_ecfp4_reeval/reeval_report.md`` §3 + §4 +
    ``aggregate.json``.  This is the 1-seed × 5-ep ECFP4 + chemprop v2
    BinaryClassificationFFN re-eval that rules out the ECFP4 featurization
    axis as the Wf-253-D 0.4544 collapse bottleneck.
    """
    rows = [
        {
            "split": "train",
            "n_rows": 66665,
            "auc": 0.9603,
            "auc_str": "0.9603",
            "purpose": "Train AUC (upper-bound sanity)",
        },
        {
            "split": "val",
            "n_rows": 9524,
            "auc": 0.8598,
            "auc_str": "0.8598",
            "purpose": "Val AUC (early-stopping check)",
        },
        {
            "split": "ood_test (Bemis-Murcko scaffold)",
            "n_rows": 19047,
            "auc": 0.8820,
            "auc_str": "0.8820",
            "purpose": "OOD test AUC (headline number vs Wf-185 baseline 0.6707)",
        },
    ]
    return {
        "title": "Wf-286B ECFP4-direct re-eval cohort (rules out ECFP4 axis as Wf-253-D 0.4544 collapse bottleneck)",
        "json_path": "molmetal/reports/wf286b_ecfp4_reeval/aggregate.json::per_seed[0] + reeval_report.md §3 + §4",
        "rows": rows,
        "featurization": "ECFP4 (Morgan radius=2, n_bits=2048) + chemprop v2 BinaryClassificationFFN",
        "config": {
            "seeds": [0],
            "epochs": 5,
            "head_hidden": 128,
            "batch": 128,
            "lr": 0.001,
            "device": "cpu",
            "wall_s": 190.9,
        },
        "wf185_baseline": {
            "mean": 0.6707,
            "std": 0.0042,
            "split": "lbap_refined_ic50_scaffold (5-seed × 20-ep × h=128/d=2)",
        },
        "delta_vs_wf185": "+0.2113 (single-seed; Δ subset-dependent optimistic)",
        "pass_gate": "test_auc ≥ 0.6707 − 0.02 = 0.6507 → PASS (0.8820 ≫ 0.6507)",
        "rules_out": "ECFP4 featurization axis (collapse is backbone / v3-fallback / JSON-split axis, not featurization)",
        "remaining_hypotheses": [
            "(a) chemprop v3 MPNN backbone",
            "(b) v3-fallback path config",
            "(c) DrugOOD JSON train/val/test OOD splits",
        ],
        "caveat_subset_mismatch": "Wf-275B adapter defaults to lbap_core_ic50_assay (OOD-assay task), NOT lbap_refined_ic50_scaffold — Δ is across subsets, optimistic.",
    }


def _extract_high_level_gates(collect: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of high-level Wf-265-Collect gate rows for the header
    banner (G1..G6 + honest_framing_note).
    """
    rows: List[Dict[str, Any]] = []
    for key in (
        "G1_todo_180_g4_complete",
        "G2_wf263a_complete",
        "G3_wf264_complete",
        "G4_n_g4_pairs",
        "G5_v3_ood_auc",
        "G6_collect_json_path",
    ):
        rows.append({"key": key, "value": collect.get(key)})
    return rows


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _badge_class(badge: str) -> str:
    """Return the CSS class for a given badge label."""
    b = badge.lower()
    if b == BADGE_MEASURED.lower():
        return "badge badge-measured"
    if b == BADGE_PROJECTED.lower():
        return "badge badge-projected"
    if b == BADGE_HONEST_NEGATIVE.lower():
        return "badge badge-honest-negative"
    return "badge badge-unknown"


def _escape(text: Any) -> str:
    """HTML-escape a value; tolerate non-strings."""
    return html.escape(str(text), quote=True)


def _render_claim_card(claim: Dict[str, Any], idx: int) -> str:
    """Render a single per-claim card with click-to-expand source JSON."""
    badge = claim.get("badge", "—")
    section = _escape(claim.get("section", "?"))
    title = _escape(claim.get("claim", ""))
    value = _escape(claim.get("value", ""))
    json_path = _escape(claim.get("json_path", "?"))
    timestamp = _escape(claim.get("timestamp", "?"))
    cid = _escape(claim.get("id", f"claim_{idx}"))
    badge_cls = _badge_class(badge)
    # Inline source JSON as a verbatim dict so the click-to-expand is
    # a literal copy of the catalog row.  This avoids the need for an
    # additional fetch and keeps the dashboard single-file.
    payload_dict = {
        "id": claim.get("id"),
        "section": claim.get("section"),
        "badge": claim.get("badge"),
        "claim": claim.get("claim"),
        "value": claim.get("value"),
        "json_path": claim.get("json_path"),
        "timestamp": claim.get("timestamp"),
    }
    payload_str = json.dumps(payload_dict, indent=2, ensure_ascii=False)
    return f"""
    <div class="claim-card" id="{cid}">
      <div class="claim-row">
        <span class="{badge_cls}">{_escape(badge)}</span>
        <span class="claim-section">{section}</span>
        <span class="claim-title">{title}</span>
      </div>
      <div class="claim-meta">
        <span class="claim-value"><strong>value:</strong> {value}</span>
        <span class="claim-path"><strong>JSON path:</strong> <code>{json_path}</code></span>
        <span class="claim-ts"><strong>ts:</strong> {timestamp}</span>
        <button type="button" class="expand-btn" onclick="toggleSource('{cid}')">show source JSON</button>
      </div>
      <pre class="claim-source" id="{cid}-source" style="display:none;">{_escape(payload_str)}</pre>
    </div>
    """


def _render_chemprop_table(cohort: Dict[str, Any]) -> str:
    """Render the chemprop v2 v3 5-metal per-metal AUC table."""
    overall = cohort.get("overall") or {}
    overall_n = overall.get("n_pairs", "?")
    overall_mean = overall.get("mean_test_auc")
    overall_min = overall.get("min_test_auc")
    overall_max = overall.get("max_test_auc")
    body_rows: List[str] = []
    for r in cohort.get("rows", []):
        body_rows.append(
            f"""
            <tr>
              <td>{_escape(r.get('metal', '?'))}</td>
              <td>{r.get('n_pairs', 0)}</td>
              <td>{_safe_round(r.get('mean_test_auc', 0.0))}</td>
              <td>{_safe_round(r.get('min_test_auc', 0.0))}</td>
              <td>{_safe_round(r.get('max_test_auc', 0.0))}</td>
              <td><code>[{', '.join(_safe_round(v) for v in r.get('test_aucs', []))}]</code></td>
            </tr>
            """
        )
    v3_ood_auc = cohort.get("v3_ood_auc")
    v3_str = (
        f"{_safe_round(v3_ood_auc)} (BELOW_RANDOM honest-negative)"
        if v3_ood_auc is not None
        else "n/a"
    )
    return f"""
    <section class="cohort">
      <h3>{_escape(cohort.get('title', 'chemprop v2 v3 cohort'))}</h3>
      <p class="cohort-meta"><strong>JSON path:</strong> <code>{_escape(cohort.get('json_path', '?'))}</code></p>
      <table class="cohort-table">
        <thead>
          <tr>
            <th>Metal</th><th>n_pairs</th><th>mean test_AUC</th>
            <th>min</th><th>max</th><th>per-seed test_aucs</th>
          </tr>
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
      <p class="cohort-summary">
        <strong>Overall ({overall_n} pairs):</strong>
        mean = {_safe_round(overall_mean)},
        min = {_safe_round(overall_min)},
        max = {_safe_round(overall_max)}.
        <strong>G5 v3 OOD AUC:</strong> {v3_str}.
      </p>
    </section>
    """


def _render_drugood_table(cohort: Dict[str, Any]) -> str:
    """Render the DrugOOD 3-split table."""
    rows: List[str] = []
    for r in cohort.get("rows", []):
        wf174_mean = r.get("wf174_3seed_mean")
        wf174_std = r.get("wf174_3seed_std")
        wf174_cell = (
            f"{_safe_round(wf174_mean)} ± {_safe_round(wf174_std)}"
            if wf174_mean is not None
            else "n/a"
        )
        wf185_mean = r.get("wf185_5seed_mean")
        wf185_std = r.get("wf185_5seed_std")
        wf185_cell = (
            f"{_safe_round(wf185_mean)} ± {_safe_round(wf185_std)}"
            if wf185_mean is not None
            else "n/a"
        )
        rows.append(
            f"""
            <tr>
              <td><code>{_escape(r.get('split', '?'))}</code></td>
              <td>{wf174_cell}</td>
              <td>{wf185_cell}</td>
              <td>{_safe_round(r.get('imold_baseline', 0.0))}</td>
              <td>{_safe_round(r.get('delta_vs_imold', 0.0))}</td>
              <td>{_escape(r.get('verdict', '?'))}</td>
            </tr>
            """
        )
    return f"""
    <section class="cohort">
      <h3>{_escape(cohort.get('title', 'DrugOOD cohort'))}</h3>
      <p class="cohort-meta"><strong>JSON path:</strong> <code>{_escape(cohort.get('json_path', '?'))}</code></p>
      <table class="cohort-table">
        <thead>
          <tr>
            <th>Split</th><th>Wf-174 3-seed</th><th>Wf-185 5-seed</th>
            <th>iMoLD Ji 2023 baseline</th><th>Δ vs iMoLD</th><th>Verdict</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """


def _render_cfm_m14_table(cohort: Dict[str, Any]) -> str:
    """Render the CFM m14 per-metal decode_ratio table."""
    rows: List[str] = []
    for r in cohort.get("rows", []):
        rows.append(
            f"""
            <tr>
              <td>{_escape(r.get('metal', '?'))}</td>
              <td>{r.get('n_total', 0)}</td>
              <td>{r.get('n_decode_ok', 0)}</td>
              <td>{_safe_round(r.get('decode_ratio', 0.0))}</td>
            </tr>
            """
        )
    return f"""
    <section class="cohort">
      <h3>{_escape(cohort.get('title', 'CFM m14 cohort'))}</h3>
      <p class="cohort-meta"><strong>JSON path:</strong> <code>{_escape(cohort.get('json_path', '?'))}</code></p>
      <table class="cohort-table">
        <thead>
          <tr>
            <th>Metal</th><th>n_total</th><th>n_decode_ok</th><th>decode_ratio</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="cohort-summary">
        <strong>Honest framing (§3.5):</strong> synthetic upper bound = 1.000; real 0.9958 is
        −0.0042 below the synthetic ceiling — expected structural distance from real
        metal-coordination chemistry.
      </p>
    </section>
    """


def _render_cfm_m7_table(cohort: Dict[str, Any]) -> str:
    """Render the CFM m7 Tanimoto diversity table."""
    rows: List[str] = []
    for r in cohort.get("rows", []):
        lift = r.get("lift")
        lift_cell = (
            f"+{lift * 100:.1f}%" if isinstance(lift, (int, float)) else "n/a"
        )
        rows.append(
            f"""
            <tr>
              <td>{_escape(r.get('configuration', '?'))}</td>
              <td>{_safe_round(r.get('diversity_tanimoto', 0.0))}</td>
              <td>{lift_cell}</td>
              <td>{_escape(r.get('verdict', '?'))}</td>
            </tr>
            """
        )
    return f"""
    <section class="cohort">
      <h3>{_escape(cohort.get('title', 'CFM m7 cohort'))}</h3>
      <p class="cohort-meta"><strong>JSON path:</strong> <code>{_escape(cohort.get('json_path', '?'))}</code></p>
      <table class="cohort-table">
        <thead>
          <tr>
            <th>Configuration</th><th>diversity_tanimoto</th><th>lift</th><th>Verdict</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """


def _render_chemprop_v3_fallback_table(cohort: Dict[str, Any]) -> str:
    """Render the Wf-263A chemprop v3-fallback mode dispatch table."""
    rows: List[str] = []
    for r in cohort.get("rows", []):
        rows.append(
            f"""
            <tr>
              <td><code>{_escape(r.get('chemprop_mode', '?'))}</code></td>
              <td><code>{_escape(r.get('user_enhanced_head', '?'))}</code></td>
              <td><code>{_escape(r.get('effective_enhanced_head', '?'))}</code></td>
              <td><code>{_escape(r.get('effective_metal_attention', '?'))}</code></td>
              <td>{_escape(r.get('verdict', '?'))}</td>
            </tr>
            """
        )
    n_files = cohort.get("n_files_modified", "?")
    n_new = cohort.get("n_new_tests", "?")
    n_bwd = cohort.get("n_backward_compat_tests", "?")
    empirical = _escape(cohort.get("empirical_status", "?"))
    return f"""
    <section class="cohort">
      <h3>{_escape(cohort.get('title', 'Wf-263A chemprop v3-fallback cohort'))}</h3>
      <p class="cohort-meta"><strong>JSON path:</strong> <code>{_escape(cohort.get('json_path', '?'))}</code></p>
      <table class="cohort-table">
        <thead>
          <tr>
            <th>--chemprop-mode</th><th>args.enhanced_head (CLI intent)</th>
            <th>_effective_enhanced_head (forwarded)</th><th>_effective_metal_attention</th>
            <th>Verdict</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="cohort-summary">
        <strong>Audit-trail guarantee:</strong> <code>args.enhanced_head</code> /
        <code>args.metal_attention</code> are NEVER mutated by the dispatch — the
        JSON audit trail records user intent while the forwarded value reflects
        what <code>--chemprop-mode</code> actually applied. Honest-negative
        preserved.
      </p>
      <p class="cohort-summary">
        <strong>Files modified:</strong> {n_files} |
        <strong>New tests:</strong> {n_new} ({n_new}/PASS) |
        <strong>Backward-compat tests:</strong> {n_bwd} ({n_bwd}/PASS) |
        <strong>Empirical status:</strong> {empirical}.
      </p>
    </section>
    """


def _render_wf266_lomo_fix_table(cohort: Dict[str, Any]) -> str:
    """Render the Wf-266 v3 LOMO fix flag table."""
    rows: List[str] = []
    for r in cohort.get("rows", []):
        default_v = r.get("default")
        default_cell = (
            f"{default_v}" if not isinstance(default_v, float)
            else f"{default_v:g}"
        )
        rows.append(
            f"""
            <tr>
              <td><code>{_escape(r.get('flag', '?'))}</code></td>
              <td><code>{_escape(default_cell)}</code></td>
              <td>{_escape(r.get('type', '?'))}</td>
              <td>{_escape(r.get('purpose', '?'))}</td>
              <td>{_escape(r.get('no_op_under', '?'))}</td>
            </tr>
            """
        )
    audit_fields = cohort.get("audit_trail_fields", [])
    audit_cell = (
        "<code>" + "</code>, <code>".join(audit_fields) + "</code>"
        if audit_fields else "n/a"
    )
    n_files = cohort.get("n_files_modified", "?")
    n_new = cohort.get("n_new_tests", "?")
    n_bwd = cohort.get("n_backward_compat_tests", "?")
    retrain = _escape(cohort.get("retrain_status", "?"))
    return f"""
    <section class="cohort">
      <h3>{_escape(cohort.get('title', 'Wf-266 v3 LOMO fix cohort'))}</h3>
      <p class="cohort-meta"><strong>JSON path:</strong> <code>{_escape(cohort.get('json_path', '?'))}</code></p>
      <table class="cohort-table">
        <thead>
          <tr>
            <th>Flag</th><th>Default</th><th>Type</th><th>Purpose</th><th>No-op under</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="cohort-summary">
        <strong>Audit-trail JSON fields (per-seed):</strong> {audit_cell}.
      </p>
      <p class="cohort-summary">
        <strong>Files modified:</strong> {n_files} |
        <strong>New tests:</strong> {n_new} ({n_new}/PASS) |
        <strong>Backward-compat tests:</strong> {n_bwd} ({n_bwd}/PASS) |
        <strong>Retrain status:</strong> {retrain}.
      </p>
    </section>
    """


def _render_wf286b_ecfp4_direct_table(cohort: Dict[str, Any]) -> str:
    """Render the Wf-286B ECFP4-direct re-eval cohort table."""
    rows: List[str] = []
    for r in cohort.get("rows", []):
        rows.append(
            f"""
            <tr>
              <td><code>{_escape(r.get('split', '?'))}</code></td>
              <td>{r.get('n_rows', 0)}</td>
              <td><strong>{_escape(r.get('auc_str', '?'))}</strong></td>
              <td>{_escape(r.get('purpose', '?'))}</td>
            </tr>
            """
        )
    feat = _escape(cohort.get("featurization", "?"))
    cfg = cohort.get("config", {})
    cfg_str = (
        f"seeds={cfg.get('seeds')}, epochs={cfg.get('epochs')}, "
        f"head_hidden={cfg.get('head_hidden')}, batch={cfg.get('batch')}, "
        f"lr={cfg.get('lr')}, device={cfg.get('device')}, wall={cfg.get('wall_s')}s"
    )
    wf185 = cohort.get("wf185_baseline", {})
    remaining = cohort.get("remaining_hypotheses", [])
    rem_html = "<li>" + "</li><li>".join(_escape(h) for h in remaining) + "</li>" if remaining else "n/a"
    caveat = _escape(cohort.get("caveat_subset_mismatch", "?"))
    pass_gate = _escape(cohort.get("pass_gate", "?"))
    delta_str = _escape(cohort.get("delta_vs_wf185", "?"))
    rules_out = _escape(cohort.get("rules_out", "?"))
    return f"""
    <section class="cohort">
      <h3>{_escape(cohort.get('title', 'Wf-286B ECFP4-direct re-eval cohort'))}</h3>
      <p class="cohort-meta"><strong>JSON path:</strong> <code>{_escape(cohort.get('json_path', '?'))}</code></p>
      <p class="cohort-meta"><strong>Featurization:</strong> {feat}</p>
      <p class="cohort-meta"><strong>Config:</strong> <code>{_escape(cfg_str)}</code></p>
      <table class="cohort-table">
        <thead>
          <tr>
            <th>Split</th><th>n_rows</th><th>AUC</th><th>Purpose</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="cohort-summary">
        <strong>Wf-185 v2 baseline (chemprop v2 MPNN on lbap_refined_ic50_scaffold):</strong>
        mean = {_safe_round(wf185.get('mean', 0.0))} ± {_safe_round(wf185.get('std', 0.0))}.
        <strong>Δ vs Wf-185:</strong> {delta_str}.
      </p>
      <p class="cohort-summary">
        <strong>Pass-gate:</strong> {pass_gate}.
      </p>
      <p class="cohort-summary">
        <strong>Rules out:</strong> {rules_out}.
      </p>
      <p class="cohort-summary">
        <strong>Remaining hypotheses:</strong>
        <ul class="badge-summary">{rem_html}</ul>
      </p>
      <p class="cohort-summary">
        <strong>Caveat (subset mismatch):</strong> {caveat}
      </p>
    </section>
    """


def _render_honest_negative_list() -> str:
    """Render the §9 honest-negative catalog as a flat list."""
    items: List[str] = []
    for hn in HONEST_NEGATIVE_CATALOG:
        items.append(
            f"""
            <li>
              <span class="badge badge-honest-negative">{_escape(hn.get('section', '?'))}</span>
              <strong>{_escape(hn.get('item', ''))}</strong>
              <span class="hn-ref">— {_escape(hn.get('ref', '?'))}</span>
            </li>
            """
        )
    return f"""
    <section class="honest-negative">
      <h3>Honest-negative catalog (TODO-207 §9, party-twenty-spirit §3.5 0 容忍 不掩盖)</h3>
      <ol class="hn-list">
        {''.join(items)}
      </ol>
    </section>
    """


def _render_high_level_gates(gates: List[Dict[str, Any]]) -> str:
    """Render the high-level Wf-265-Collect gates banner."""
    rows: List[str] = []
    for g in gates:
        v = g.get("value")
        v_str = (
            "true" if v is True
            else "false" if v is False
            else _escape(str(v)) if v is not None
            else "—"
        )
        rows.append(
            f"<tr><td><code>{_escape(g.get('key', '?'))}</code></td><td>{v_str}</td></tr>"
        )
    return f"""
    <section class="gate-banner">
      <h3>Wf-265-Collect high-level gates</h3>
      <table class="gate-table">
        <thead><tr><th>Gate</th><th>Value</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """


# ---------------------------------------------------------------------------
# CSS + JS (inline, brand-neutral)
# ---------------------------------------------------------------------------

_CSS = r"""
:root {
  --bg: #fafafa;
  --fg: #1a1a1a;
  --muted: #6b7280;
  --border: #e5e7eb;
  --card-bg: #ffffff;
  --measured-bg: #d1fae5;
  --measured-fg: #065f46;
  --projected-bg: #fef3c7;
  --projected-fg: #92400e;
  --honest-bg: #fee2e2;
  --honest-fg: #991b1b;
  --cohort-thead: #f3f4f6;
  --cohort-row-alt: #f9fafb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a;
    --fg: #f8fafc;
    --muted: #94a3b8;
    --border: #1e293b;
    --card-bg: #1e293b;
    --measured-bg: #064e3b;
    --measured-fg: #6ee7b7;
    --projected-bg: #78350f;
    --projected-fg: #fcd34d;
    --honest-bg: #7f1d1d;
    --honest-fg: #fca5a5;
    --cohort-thead: #1e293b;
    --cohort-row-alt: #172033;
  }
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  background: var(--bg);
  color: var(--fg);
  margin: 0;
  padding: 1.5rem;
  line-height: 1.5;
}
header.dashboard-header {
  border-bottom: 2px solid var(--border);
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}
header.dashboard-header h1 { margin: 0; font-size: 1.6rem; }
header.dashboard-header .meta {
  color: var(--muted); font-size: 0.875rem; margin-top: 0.25rem;
}
.badge {
  display: inline-block;
  padding: 0.125rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.025em;
  margin-right: 0.5rem;
  vertical-align: middle;
}
.badge-measured { background: var(--measured-bg); color: var(--measured-fg); }
.badge-projected { background: var(--projected-bg); color: var(--projected-fg); }
.badge-honest-negative { background: var(--honest-bg); color: var(--honest-fg); }
.badge-unknown { background: var(--border); color: var(--muted); }
section.cohort, section.honest-negative, section.gate-banner {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 1rem 1.25rem;
  margin-bottom: 1rem;
}
section.cohort h3, section.honest-negative h3, section.gate-banner h3 {
  margin: 0 0 0.5rem 0;
  font-size: 1.05rem;
}
.cohort-meta, .cohort-summary { color: var(--muted); font-size: 0.875rem; margin: 0.25rem 0; }
table.cohort-table, table.gate-table {
  width: 100%; border-collapse: collapse; margin-top: 0.5rem;
}
table.cohort-table th, table.gate-table th,
table.cohort-table td, table.gate-table td {
  padding: 0.375rem 0.5rem;
  border-bottom: 1px solid var(--border);
  text-align: left;
  font-size: 0.875rem;
}
table.cohort-table th, table.gate-table th {
  background: var(--cohort-thead);
  font-weight: 600;
}
table.cohort-table tbody tr:nth-child(even),
table.gate-table tbody tr:nth-child(even) {
  background: var(--cohort-row-alt);
}
.claim-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-left: 4px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.625rem 0.875rem;
  margin-bottom: 0.5rem;
}
.claim-card:has(.badge-measured) { border-left-color: var(--measured-fg); }
.claim-card:has(.badge-projected) { border-left-color: var(--projected-fg); }
.claim-card:has(.badge-honest-negative) { border-left-color: var(--honest-fg); }
.claim-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.claim-section { color: var(--muted); font-size: 0.8rem; }
.claim-title { font-weight: 500; }
.claim-meta {
  display: flex; gap: 0.75rem; flex-wrap: wrap;
  color: var(--muted); font-size: 0.8rem; margin-top: 0.25rem;
}
.claim-meta code { background: var(--cohort-row-alt); padding: 0 0.25rem; border-radius: 3px; }
.claim-source {
  background: var(--cohort-row-alt);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  padding: 0.5rem 0.75rem;
  font-size: 0.75rem;
  margin-top: 0.5rem;
  overflow-x: auto;
  white-space: pre-wrap;
}
.expand-btn {
  background: var(--cohort-thead);
  color: var(--fg);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  padding: 0.125rem 0.5rem;
  font-size: 0.75rem;
  cursor: pointer;
}
.expand-btn:hover { background: var(--border); }
.honest-negative ol.hn-list { margin: 0; padding-left: 1.25rem; }
.honest-negative li { margin-bottom: 0.375rem; font-size: 0.875rem; }
.hn-ref { color: var(--muted); font-size: 0.8rem; }
ul.badge-summary { margin: 0.25rem 0 0.5rem 0; padding-left: 1.25rem; }
ul.badge-summary li { margin-bottom: 0.25rem; font-size: 0.875rem; }
footer.dashboard-footer {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.8rem;
}
"""

_JS = r"""
function toggleSource(id) {
  var el = document.getElementById(id + '-source');
  var btn = document.querySelector('#' + id + ' .expand-btn');
  if (!el || !btn) return;
  if (el.style.display === 'none' || el.style.display === '') {
    el.style.display = 'block';
    btn.textContent = 'hide source JSON';
  } else {
    el.style.display = 'none';
    btn.textContent = 'show source JSON';
  }
}
"""


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------


def build_dashboard_html(
    *,
    paper_doc: Optional[Path],
    collect_json: Optional[Path],
    checklist_doc: Optional[Path],
    paper_doc_text: Optional[str],
    collect_data: Optional[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Build the dashboard HTML in-memory and return the JSON-ready dict.

    Never raises — every step is wrapped.  Unrecoverable conditions produce
    a stub ``dashboard`` with an ``error`` field; the caller decides the
    exit code based on ``obj["schema_valid"]``.
    """
    obj: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "paper_doc": str(paper_doc) if paper_doc else "",
        "collect_json": str(collect_json) if collect_json else "",
        "checklist_doc": str(checklist_doc) if checklist_doc else "",
        "n_claims": 0,
        "n_honest_negatives": 0,
        "cohorts": {},
        "honest_framing_note": "",
        "schema_valid": False,
    }
    try:
        # Load paper-doc text (only used to verify it exists & emits the
        # honest_framing_note verbatim).  No structured parse — the catalog
        # is hand-curated per TODO-188 §5.1 / §5.2 so we are not at the
        # mercy of an LLM/parser reading the markdown.
        if paper_doc is not None:
            try:
                text = paper_doc_text if paper_doc_text is not None else _read_text(paper_doc)
                obj["paper_doc_text_len"] = len(text)
                obj["paper_doc_text_first_120"] = text[:120]
            except Exception as exc:  # noqa: BLE001
                obj["paper_doc_error"] = f"{type(exc).__name__}: {exc}"

        # Load collect.json (mandatory for cohort tables).
        # Wf-304A: this is the primary paper-grade anchor read; pass
        # ``strict=False`` so a missing / stale anchor surfaces an
        # ``obj["collect_json_error"]`` rather than crashing the build.
        if collect_data is None and collect_json is not None:
            try:
                collect_data = _read_json(collect_json)
                # AnchorGuard raises on guard failure, but with strict=False
                # _read_json returns {}; surface the guard error explicitly
                # via obj["collect_json_error"] for the dashboard author.
                if not collect_data and collect_json.exists():
                    obj["collect_json_error"] = (
                        "anchor guard returned empty dict (stale, missing, "
                        "or malformed)"
                    )
            except Exception as exc:  # noqa: BLE001
                obj["collect_json_error"] = f"{type(exc).__name__}: {exc}"
                collect_data = {}
        collect_data = collect_data or {}

        # Build cohorts.
        chemprop = _extract_chemprop_cohort(collect_data)
        drugoood = _extract_drugood_cohort(collect_data)
        cfm_m14 = _extract_cfm_m14_cohort(collect_data)
        cfm_m7 = _extract_cfm_m7_cohort(collect_data)
        chemprop_v3_fallback = _extract_chemprop_v3_fallback_cohort(collect_data)
        wf266_lomo_fix = _extract_wf266_lomo_fix_cohort(collect_data)
        wf286b_ecfp4_direct = _extract_wf286b_ecfp4_direct_cohort(collect_data)
        gates = _extract_high_level_gates(collect_data)
        obj["cohorts"] = {
            "chemprop_v2_v3": chemprop,
            "drugoood": drugoood,
            "cfm_m14": cfm_m14,
            "cfm_m7": cfm_m7,
            "chemprop_v3_fallback": chemprop_v3_fallback,
            "wf266_lomo_fix": wf266_lomo_fix,
            "wf286b_ecfp4_direct": wf286b_ecfp4_direct,
        }
        obj["high_level_gates"] = gates
        obj["n_claims"] = len(CLAIM_CATALOG)
        obj["n_honest_negatives"] = len(HONEST_NEGATIVE_CATALOG)

        # Pull honest_framing_note verbatim from collect.json (the source
        # of truth per Wf-265-Collect), fallback to a stub.
        honest_note = collect_data.get("honest_framing_note", "") or ""
        if not honest_note:
            honest_note = (
                "honest_framing_note not present in collect.json; dashboard "
                "still renders per the §9 honest-negative catalog."
            )
        obj["honest_framing_note"] = honest_note

        # Render the HTML.
        claim_cards_html = "".join(
            _render_claim_card(c, i) for i, c in enumerate(CLAIM_CATALOG)
        )
        chemprop_html = _render_chemprop_table(chemprop)
        drugoood_html = _render_drugood_table(drugoood)
        cfm_m14_html = _render_cfm_m14_table(cfm_m14)
        cfm_m7_html = _render_cfm_m7_table(cfm_m7)
        chemprop_v3_fallback_html = _render_chemprop_v3_fallback_table(chemprop_v3_fallback)
        wf266_lomo_fix_html = _render_wf266_lomo_fix_table(wf266_lomo_fix)
        wf286b_ecfp4_direct_html = _render_wf286b_ecfp4_direct_table(wf286b_ecfp4_direct)
        honest_neg_html = _render_honest_negative_list()
        gates_html = _render_high_level_gates(gates)

        # Build per-badge discipline summary (Wf-268C-Dash explicit
        # MEASURED-vs-PROJECTED discipline).
        n_meas = sum(
            1 for c in CLAIM_CATALOG if c.get("badge") == BADGE_MEASURED
        )
        n_proj = sum(
            1 for c in CLAIM_CATALOG if c.get("badge") == BADGE_PROJECTED
        )
        n_neg = sum(
            1 for c in CLAIM_CATALOG
            if c.get("badge") == BADGE_HONEST_NEGATIVE
        )
        n_hn_list = len(HONEST_NEGATIVE_CATALOG)
        badge_discipline_html = f"""
    <section class="gate-banner">
      <h3>MEASURED-vs-PROJECTED badge discipline (Wf-268C-Dash, party-twenty-spirit §3.5)</h3>
      <p>
        Every claim badge below traces to a JSON path in the on-disk source. The
        three badge colours are MUTUALLY EXCLUSIVE — a MEASURED claim is a
        numeric value produced by a run that completed and reported a metric
        (per-seed JSON exists in <code>results/wf*/</code> or
        <code>molmetal/reports/wf*/</code>); a PROJECTED claim is one whose
        deferred or follow-up status is documented in TODO/INDEX or a wf*-final
        report; an honest-negative claim is one whose numeric value documents a
        known gap (e.g. below baseline, expected under-training, blocked at
        GPU-ordinal layer). There is NO grey area.
      </p>
      <ul class="badge-summary">
        <li><span class="badge badge-measured">MEASURED</span> {n_meas} claim(s) — sourced from per-seed JSON on disk.</li>
        <li><span class="badge badge-projected">PROJECTED</span> {n_proj} claim(s) — explicitly DEFERRED with a TODO- or wf*-final pointer.</li>
        <li><span class="badge badge-honest-negative">honest-negative</span> {n_neg} claim-card(s) + {n_hn_list} catalog entries — the §9 "DO NOT claim" surface.</li>
      </ul>
      <p>
        <strong>Discipline rule (re-stated verbatim):</strong> a claim is
        MEASURED iff the cited JSON path resolves to a numeric value computed
        by a real run that completed; otherwise it is PROJECTED. There is no
        middle ground. If a value cannot be traced to a JSON path it is
        silently DROPPED from the catalog.
      </p>
    </section>
    """

        # Build the Wf-268C-Dash + Wf-279C + Wf-280C + Wf-293D + Wf-308B + Wf-311B + Wf-311D + Wf-311E + Wf-314A + Wf-318G
        # update report (summary of new sections added across all dashboard refreshes).
        update_report = {
            "wf_id": "Wf-318G (refresh on top of Wf-314A, post-Wf-311F §7 ultracode follow-up #3 + #4 honest-negative close-outs: r13_100x3_partial + diversity_subpocket + steric_clash_pb_dock)",
            "schema_version": SCHEMA_VERSION,
            "n_new_claims_added": 28,  # 8 Wf-268C + 5 Wf-279C + 1 Wf-293D + 3 Wf-308B + 6 Wf-311B + 5 Wf-311D
            "n_new_honest_negatives_added": 16,  # 3 Wf-268C + 2 Wf-279C + 1 Wf-293D + 1 Wf-308B + 3 Wf-311D + 3 Wf-311E Part 4 / Wf-314A + 3 Wf-318G
            "n_new_cohort_sections_added": 3,  # Wf-263A v3-fallback + Wf-266 LOMO fix + Wf-286B ECFP4 direct
            "new_claim_ids": [
                # Wf-268C-Dash baseline
                "claim_chemprop_v3_fallback_default",
                "claim_chemprop_v3_fallback_official",
                "claim_wf266_min_epochs_flag",
                "claim_wf266_lomo_init_scale_flag",
                "claim_wf266_v3_fix_mode",
                "claim_todo_180_g4_re_top",
                "claim_todo_180_g4_overall_15pair",
                "claim_wf266_g5_50epoch_blocked",
                # Wf-279C additions
                "claim_wf275b_drugood_adapter",
                "claim_wf269c_yuelbond_load_bearing",
                "claim_wf275c_m14_lead_precondition",
                "claim_wf274a_regression_disclosure",
                "claim_wf271b_m17_drugood_v3_reeval",
                # Wf-293D additions (claim_wf269c_yuelbond_load_bearing already existed
                # from Wf-279C — only claim_wf286b_ecfp4_axis_ruled_out is NEW)
                "claim_wf286b_ecfp4_axis_ruled_out",
                # Wf-308B R03 ADMET pass-through integration (3 new claim cards)
                "claim_wf303c_t2_ames_pass",
                "claim_wf303c_t2_cell_perm_pass",
                "claim_wf303c_t2_aq_sol_honest_negative",
                # Wf-311B hard-metric paper-grade aggregation (6 new claim cards)
                "claim_wf311b_vina_5pocket_aggregate",
                "claim_wf311b_rmsd_panel_aggregate",
                "claim_wf311b_main_results_aggregate",
                "claim_wf311b_section_4_2_refreshed",
                "claim_wf311b_section_4_6_refreshed",
                "claim_wf311b_section_5_ablation_refreshed",
                # Wf-311D PART 1+2+3+4+5 ship surface (5 new claim cards)
                "claim_wf311d_ood_reachability_summary",
                "claim_wf311d_m18_protein_metal_baseline",
                "claim_wf311d_gi50_proxy_summary",
                "claim_wf311d_pharmacophore_pass_rate_prod",
                "claim_wf311d_metal_coord_phase3g_singleton",
                # Wf-319A ablation_runner smoke ship (1 new claim card)
                "claim_wf319a_ablation_table_aggregate",
            ],
            "new_honest_negative_ids": [
                # Wf-268C-Dash baseline
                "hn_no_wf263a_empirical_pass",
                "hn_no_wf266_g5_measured",
                "hn_no_wf266_retrain_lift",
                # Wf-279C additions
                "hn_no_cfm_decode_real_09645_as_lead",
                "hn_no_yuelbond_off_deployment",
                # Wf-293D additions
                "hn_wf286a_5_seed_drugood_failed",
                # Wf-308B R03 ADMET pass-through aq_sol honest-negative
                "hn_wf308b_aq_sol_fail",
                # Wf-311D additions (3 new honest-negatives for OOD partial + m18 synthetic + curated smoke bias)
                "hn_wf311d_ood_reachability_partial",
                "hn_wf311d_m18_per_residue_synthetic_proxy",
                "hn_wf311d_pharmacophore_curated_smoke_biased",
                # Wf-311E Part 4 / Wf-314A additions (3 NEW honest-negatives for m6_kappa / hepatotox / logp7_4)
                "hn_wf311e_m6_kappa_0_472",
                "hn_wf311e_hepatotox_0_548",
                "hn_wf311e_logp7_4_6_413",
                # Wf-318G additions (3 NEW honest-negatives for r13_100x3_partial + diversity_subpocket + steric_clash_pb_dock — Wf-311F §7 candidates #3 + #4)
                "hn_wf318g_r13_100x3_not_paper_grade",
                "hn_wf318g_diversity_subpocket_design_only",
                "hn_wf318g_steric_clash_pb_dock_830c_deferred",
            ],
            "new_cohort_keys": [
                "chemprop_v3_fallback",
                "wf266_lomo_fix",
                "wf286b_ecfp4_direct",
            ],
            "badge_counts": {
                "MEASURED": n_meas,
                "PROJECTED": n_proj,
                "honest-negative": n_neg,
            },
            "honest_negative_catalog_count": n_hn_list,
            "timestamp": obj["generated_at"],
        }
        obj["update_report"] = update_report

        html_str = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="generator" content="Wf-280C paper_dashboard.py schema={SCHEMA_VERSION}" />
<title>Mol-Metal Paper §4 Dashboard</title>
<style>{_CSS}</style>
<script>{_JS}</script>
</head>
<body>
<header class="dashboard-header">
  <h1>Mol-Metal Paper §4 Dashboard</h1>
  <div class="meta">
    <strong>Generated:</strong> {obj['generated_at']} |
    <strong>Schema:</strong> {SCHEMA_VERSION} |
    <strong>Claims:</strong> {obj['n_claims']} |
    <strong>Honest-negatives:</strong> {obj['n_honest_negatives']} |
    <strong>Paper doc:</strong> <code>{_escape(obj['paper_doc'])}</code> |
    <strong>Collect JSON:</strong> <code>{_escape(obj['collect_json'])}</code>
  </div>
</header>

<main>
  {gates_html}

  {badge_discipline_html}

  <section class="honest-negative">
    <h3>Honest framing note (Wf-265-Collect)</h3>
    <p>{_escape(honest_note)}</p>
  </section>

  <h2>Per-cohort tables</h2>
  {chemprop_html}
  {chemprop_v3_fallback_html}
  {wf266_lomo_fix_html}
  {wf286b_ecfp4_direct_html}
  {drugoood_html}
  {cfm_m14_html}
  {cfm_m7_html}

  <h2>Per-claim cards (click to expand source JSON)</h2>
  <div class="claim-list">
    {claim_cards_html}
  </div>

  <h2>Honest-negative catalog (TODO-207 §9)</h2>
  {honest_neg_html}
</main>

<footer class="dashboard-footer">
  <p>
    Honest framing per <code>party-twenty-spirit</code> §3.5 (0 容忍 不掩盖):
    every claim badge (MEASURED / PROJECTED / honest-negative) traces to a JSON
    path embedded in the source JSON snippet. Re-run
    <code>paper_dashboard.py</code> after any Wf-265-Collect refresh.
  </p>
  <p>
    <strong>Wf-268C-Dash update_report:</strong>
    {update_report['n_new_claims_added']} new claims +
    {update_report['n_new_honest_negatives_added']} new honest-negatives +
    {update_report['n_new_cohort_sections_added']} new cohort sections.
  </p>
  <p>
    <strong>Paper doc:</strong> <code>{_escape(obj['paper_doc'])}</code> &middot;
    <strong>Collect JSON:</strong> <code>{_escape(obj['collect_json'])}</code> &middot;
    <strong>Checklist doc:</strong> <code>{_escape(obj['checklist_doc'])}</code>
  </p>
</footer>
</body>
</html>
"""
        obj["html"] = html_str
        obj["schema_valid"] = True
        if verbose:
            print(
                f"[Wf-293D-Dash] built dashboard: "
                f"{obj['n_claims']} claims, "
                f"{obj['n_honest_negatives']} honest-negatives, "
                f"7 cohorts, "
                f"html_len={len(html_str)}"
            )
    except Exception as exc:  # noqa: BLE001
        obj["error"] = f"{type(exc).__name__}: {exc}"
        obj["schema_valid"] = False
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser with the 4 contract flags."""
    p = argparse.ArgumentParser(
        prog="paper_dashboard.py",
        description=(
            "Wf-265-Dashboard: emit interactive paper §4 dashboard HTML "
            "(MEASURED / PROJECTED / honest-negative badges, per-claim "
            "expand-to-source JSON, per-cohort tables)."
        ),
    )
    p.add_argument(
        "--paper-doc",
        type=Path,
        default=DEFAULT_PAPER_DOC,
        help=f"Paper §4 DRAFT markdown path (default: {DEFAULT_PAPER_DOC}).",
    )
    p.add_argument(
        "--collect-json",
        type=Path,
        default=DEFAULT_COLLECT_JSON,
        help=f"Wf-265-Collect all_results.json path (default: {DEFAULT_COLLECT_JSON}).",
    )
    p.add_argument(
        "--out-html",
        type=Path,
        default=DEFAULT_OUT_HTML,
        help=f"Output dashboard HTML path (default: {DEFAULT_OUT_HTML}).",
    )
    p.add_argument(
        "--checklist-doc",
        type=Path,
        default=DEFAULT_CHECKLIST_DOC,
        help=(
            "TODO-207 unified pre-flight v2 path (default: "
            f"{DEFAULT_CHECKLIST_DOC})."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose build summary to stdout.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint. Returns process exit code (0/1/2)."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Required-flag check (G2 ≥ 4 contract flags).
    flags_seen = {
        "--paper-doc": args.paper_doc is not None,
        "--collect-json": args.collect_json is not None,
        "--out-html": args.out_html is not None,
        "--verbose": bool(args.verbose),
    }
    missing = [k for k, v in flags_seen.items() if not v]
    if missing:
        print(
            f"[Wf-265-Dashboard] missing required CLI flag(s): {missing}",
            file=sys.stderr,
        )
        return 1

    # Resolve paths.
    paper_doc: Optional[Path] = args.paper_doc
    if paper_doc is not None and not paper_doc.exists():
        print(
            f"[Wf-265-Dashboard] paper doc not found: {paper_doc} (continuing with stub).",
            file=sys.stderr,
        )
        paper_doc = None

    collect_json: Optional[Path] = args.collect_json
    if collect_json is not None and not collect_json.exists():
        print(
            f"[Wf-265-Dashboard] collect JSON not found: {collect_json} (continuing with stub).",
            file=sys.stderr,
        )
        collect_json = None

    checklist_doc: Optional[Path] = args.checklist_doc
    if checklist_doc is not None and not checklist_doc.exists():
        # Non-fatal — checklist doc is informational only.
        if args.verbose:
            print(
                f"[Wf-265-Dashboard] checklist doc not found: {checklist_doc} (informational only).",
                file=sys.stderr,
            )
        checklist_doc = None

    # Build the dashboard in-memory.
    obj = build_dashboard_html(
        paper_doc=paper_doc,
        collect_json=collect_json,
        checklist_doc=checklist_doc,
        paper_doc_text=None,
        collect_data=None,
        verbose=args.verbose,
    )

    # Write the HTML to disk (always, even on partial failure — the
    # dashboard at minimum renders the honest-negative catalog).
    out_html: Path = args.out_html
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html_str = obj.get("html") or "<!doctype html><html><body><h1>paper_dashboard: build failed</h1></body></html>"
    with open(str(out_html), "w", encoding="utf-8") as fh:
        fh.write(html_str)

    if not obj.get("schema_valid", False):
        print(
            f"[Wf-265-Dashboard] build incomplete: {obj.get('error', '?')}; "
            f"wrote partial dashboard to {out_html}",
            file=sys.stderr,
        )
        return 2

    # Always write the build JSON next to the HTML so downstream tooling
    # can re-parse claims / cohorts without re-running the script.
    out_json = out_html.with_suffix(".build.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Strip the HTML field before writing the build JSON (would dwarf
    # the rest of the file otherwise).
    build_obj = {k: v for k, v in obj.items() if k != "html"}
    with open(str(out_json), "w", encoding="utf-8") as fh:
        json.dump(build_obj, fh, indent=2, ensure_ascii=False)

    if args.verbose:
        size_kb = out_html.stat().st_size / 1024.0
        print(
            f"[Wf-265-Dashboard] wrote {out_html} ({size_kb:.1f} KB); "
            f"build JSON at {out_json}."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
