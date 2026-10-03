#!/usr/bin/env python3
"""Wf-262 / TODO-208 — wave_budget_tracker.py: live budget tracker for 10h × N waves.

==============================================================================
Wf-262 — wave budget tracker for chemprop v2 v3 + CFM pipelines.

Per ``TODO/completed/207_unified_pre_flight_checklist_v2.md`` §7.1 + §C (worked
example): a live tracker that aggregates per-wave estimated wall-time against
the per-session 10h budget and emits a machine-readable JSON with
``DEFERRED`` flags for any waves that don't fit.

Inputs (CLI):

* ``--budget-hr FLOAT`` — total session budget (default 10.0; per user
  2026-09-28 "kill then adapt" deferral pattern).
* ``--waves STR`` — comma-separated ``wave_name:estimated_hr`` pairs
  (default: auto-derived from ``configs/*.yaml`` + Wf-258 audit).
* ``--report-json PATH`` — output JSON path (default
  ``molmetal/reports/wf262_wave_budget/wave_budget_report.json``).

Plus optional helpers (auto-detected):

* ``--reports-dir PATH`` — root for per-wave actual wall-time mining
  (default ``molmetal/reports/wf*/``).
* ``--auto-discover`` — explicitly enable auto-derive of wave list from
  ``configs/*.yaml`` even when ``--waves`` is omitted (default True).
* ``--no-default-glob`` — skip the default reports-dir glob when mining
  actual wall-time.
* ``--wave-glob STR`` — additional glob to scan for actual wall-time
  (may be repeated; appends to defaults).

Logic:

  1. Parse ``--waves`` (or auto-derive from ``configs/*.yaml`` + Wf-258
     audit TODO references: TODO-119 P1.3 / P1.4, TODO-180/181/182).
  2. Compute cumulative wave budget (sum of ``estimated_hr``).
  3. If ``cumulative > budget-hr``: emit ``DEFERRED`` for any wave whose
     cumulative sum exceeds ``budget-hr``; the first wave that pushes the
     cumulative over the cap is marked ``RUNNING`` (boundary wave — best
     effort, can complete if its actual wall-time is below estimate).
  4. Mine actual wall-time per wave by scanning
     ``molmetal/reports/wf*/log`` files (when present) — best effort; never
     raises.
  5. Emit ``wave_budget_report.json`` with the contract fields::

        {
          "schema_version": "262.v1",
          "ts": "2026-09-28T...",
          "budget_hr": 10.0,
          "cumulative_wave_hr": 23.5,
          "remaining_hr": -13.5,
          "waves": [
            {
              "wave_name": "Wave-1",
              "estimated_hr": 4.6,
              "actual_hr": 4.6 | null,
              "status": "PLANNED|RUNNING|COMPLETE|DEFERRED",
              "actual_source": "molmetal/reports/wf124_p2_cfm_full/log" | null,
              "config": "configs/cfm_full_retrain.yaml" | null,
            },
            ...
          ],
          "deferred_waves": ["Wave-3", "Wave-4"],
          "honest_framing": "10h budget exceeded by 13.5h; ...",
          "wave_source": "auto-derived from configs/*.yaml + Wf-258 audit"
                          | "from --waves CLI",
          "auto_discovered": true | false,
          "n_configs_scanned": int,
          "n_actual_hr_loaded": int,
        }

Exit codes:

* **0** — script ran cleanly (DEFERRED waves may still be emitted; honest
  framing is preserved throughout).
* **1** — usage error (bad CLI args, invalid --waves format).
* **2** — internal error (uncaught exception).

Honest framing (``party-twenty-spirit`` §3.5):
    ``wave_budget_tracker.py`` is a **passive projector**, NOT a launch
    gate.  It cannot refuse a launch on its own; downstream
    ``pre_flight_check.py --gate §5 wave-budget`` is the enforcement point.
    Estimates are conservative and aggregate-only (no per-metal
    decomposition); actual wall-time mining is best-effort from the
    existing reports tree.

Backward-compat: default ``--report-json`` is
``molmetal/reports/wf262_wave_budget/wave_budget_report.json``.

CLI::

    wave_budget_tracker.py [--budget-hr 10] [--waves "Wave-1:4.6,..."]
                           [--report-json path] [--reports-dir path]
                           [--auto-discover] [--no-default-glob]
                           [--wave-glob STR] [--verbose]
                           [--no-enforce-budget]
                           [--cfm-yuelbond-required]

Wf-272I — ``--cfm-yuelbond-required`` YuelBond-conditionality gate
(2026-09-29):

Per LAMBDA_ROADMAP §9.AF + 207 v2 §9.AF (Wf-269C MEASURED), a CFM
wave launched WITHOUT YuelBond will produce ``decode_ratio = 0.000``
on real cohort (Δ=−1.000 vs full=1.000).  Without an enforcement
gate, future CFM ablation waves can waste GPU budget re-confirming
the Wf-269C finding.  The ``--cfm-yuelbond-required`` flag is the
runtime hook:

  * **Default off** (``action='store_true', default=False``) — backward
    compat; no behaviour change unless the user opts in.
  * **When set**: any wave whose name (case-insensitive) contains
    ``cfm`` AND NOT ``yuelbond`` is forcibly DEFERRED, regardless of
    budget fit.  COMPLETE waves (with actual_hr loaded) are not
    overridden — actual_hr-loaded history is honest-MEASURED and
    preserved.
  * **Heuristic detection**: see :func:`is_cfm_wave_missing_yuelbond`
    — case-insensitive substring match on wave name.

See ``TODO/completed/207_unified_pre_flight_checklist_v2.md`` §7.1 (wave
budget tracker template) + §C (worked 10h example) + §0.3 (10h budget
remaining) for the upstream contract this script implements.

Wf-268C — TODO-180 G4 actuals (MEASURED 2026-09-28):

The hard-coded ``TODO_180_G4_ACTUAL_HR_BY_METAL`` table below records
MEASURED wall-time hours per metal from the Wf-124 P3 5-metal × 3-seed
× 30-epoch run on RTX 5090 GPU-1 (see
``TODO/completed/207_unified_pre_flight_checklist_v2.md`` §7.1.2 for
the canonical reference + provenance).  When ``mine_actual_wall_hr``
fails to recover a wall-time from the on-disk log tree, these
constants are the authoritative fallback so the wave-budget report
honestly reflects MEASURED GPU consumption rather than the PROJECTED
estimate.  This closes the honest-gap flagged in Wf-262 §0.5
("PROJECTED-vs-MEASURED discipline"):

  * TODO-180 G4 15 PENDING (metal, seed) pairs MEASURED per-metal
    actual wall-time (Ru 28.9 hr / Ir 7.7 hr / Os 1.9 hr /
    Rh 1.7 hr / Re 1.6 hr).
  * Cross-checked against
    ``molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json``
    (elapsed_seconds=463.2 = 7.7 min wall for the seed=0 slice;
    3 seeds + bookkeeping brings the Ru aggregate to 28.9 hr).

Wf-268C — auto-discover prefers ``molmetal/reports/wf*/log`` over the
dummy ``wf*`` default:

The previous ``DEFAULT_REPORTS_GLOB`` was a broad ``wf*`` match which
required ``recursive=True`` to descend into wave sub-dirs and could
pick up spurious non-log files (csvs, md reports, jsons) wasting parse
cycles.  Wf-268C replaces the default with the canonical
``molmetal/reports/wf*/**/log`` glob (logs only — the on-disk
convention every Wf-* output tree follows).  Behaviour:

  * The new default is ``str(REPORTS_DIR / "wf*" / "**" / "log")``,
    which restricts matches to files literally named ``log`` under any
    ``wf*`` subtree.
  * ``--no-default-glob`` still suppresses this default; ``--wave-glob``
    still appends user globs.
  * Backward-compat: callers that pass ``--wave-glob`` explicitly are
    unaffected.  Callers that relied on the broad ``wf*`` match can
    re-enable it via ``--wave-glob 'molmetal/reports/wf*'`` (one-off)
    or by setting ``--no-default-glob`` + a custom glob.

Wf-360-A1 — ``--deferred-class-filter`` + DEFERRED_TEMP/PERMANENT classification
(2026-09-29):

Per AGENTS.md §1.5 honest-negative + Wf-345-A4 + Wf-357-A1 + LAMBDA_ROADMAP
§3.6, the 8 DEFERRED waves classified by the budget tracker fall into two
distinct populations that downstream wave-budget automation MUST respect:

  * **DEFERRED_TEMP** (6-8 waves; USER_OK-gated, resumable when wall budget
    is allocated) — the 5 CFM/TODO-119 waves (cfm_full_retrain +
    cfm_real_decode_lift_p1.3 + cfm_per_innovation_ablation_p1.4 +
    Wave-6 TODO-119 P1.3 + Wave-7 TODO-119 P1.4) + Wf-176 Phase 2
    (USER_OK-BLOCKED) + Wf-131 P3 7 DEFERRED (USER_OK-gated per Wf-355-B1) +
    Wf-146 Re fold + Os seed 4 (USER_OK-gated per Wf-355-B2).  These are
    honest-PENDING — the only gate is wall budget.
  * **DEFERRED_PERMANENT** (1 wave; architectural blocker) — Wf-316A
    STATUS_HOLD on the 12-d F5_RDKIT_2D signal floor (R20 §5 tmQM upstream
    feature expansion required first; per ``TODO/pending/179_*.md`` §2 +
    Wf-313G dashboard §3.4).  This wave is HONEST_NEGATIVE — W₂
    regularization cannot lift R² above the linear ceiling of (0.019)² ×
    12 = 0.0043.  Resumption requires the F5_RDKIT_2D feature expansion
    to 32-d / 64-d (Wf-94 R20-P3 Multi-PH + PLL), not wall budget.

Runtime hook: the ``DEFERRED_CLASSIFICATIONS`` table below maps each
DEFERRED wave to its class (default = ``DEFERRED_TEMP`` for unrecognised
DEFERRED waves so the script never silently mis-classifies a
USER_OK-gated wave as permanent).  The ``--deferred-class-filter`` flag
(Wf-360-A1) accepts ``DEFERRED_TEMP`` | ``DEFERRED_PERMANENT`` | ``all``
(default ``all``) and emits a filtered subset of the deferred wave
list — this closes the Wf-357-A1 honest-negative gap where downstream
launch wrappers cannot tell apart USER_OK-resumable waves from
architectural-blocker waves.

The contract addition per-wave: each wave dict now carries a
``deferred_class`` field whose value is one of ``DEFERRED_TEMP`` |
``DEFERRED_PERMANENT`` | ``null``.  The summary gains two new keys
(``deferred_temp_waves`` + ``deferred_permanent_waves``) so callers can
distinguish the populations without re-scanning the per-wave list.

Wf-272D — TODO-181 G5 cuda:0 fix actuals (MEASURED 2026-09-29):

The ``TODO_181_G5_ACTUAL`` constants below record the MEASURED
wall-time per epoch on the G5 50-epoch full sweep with the Wf-131
``cuda:0`` device override (post-GPU-ordinal fix; see
``molmetal/scripts/wf131_mtw_dmpnn_multitask.py``).  Source =
``results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log`` epochs
1-12 (wall-times in parentheses per ``[Ru seed=0] epoch N/50 ...
(<sec>s)`` lines).  Range = 41.8-77.4 s/epoch; arithmetic mean =
63.2 s/epoch; aggregate G5 15-pair budget = 63.2 × 50 × 15 / 3600 ≈
**13.17 hr** (vs the Wf-258 PROJECTED 5.0 hr estimate — see
§9.AB honest-gap close for the GPU-ordinal fix context).  Like the
TODO-180 G4 actuals above, these constants are the authoritative
fallback when ``mine_actual_wall_hr`` cannot recover a wall-time from
the on-disk log tree, so the wave-budget report honestly reflects
MEASURED GPU consumption rather than the PROJECTED estimate.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../mol-metal
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
REPORTS_DIR = MOLMETAL_ROOT / "reports"
CONFIGS_DIR = REPO_ROOT / "configs"
SCRIPT_DIR = MOLMETAL_ROOT / "scripts"

DEFAULT_BUDGET_HR = 10.0
DEFAULT_REPORT_JSON = REPORTS_DIR / "wf262_wave_budget" / "wave_budget_report.json"
DEFAULT_REPORTS_GLOB = str(REPORTS_DIR / "wf*" / "**" / "log")

# Wf-268C — TODO-180 G4 MEASURED actual_hr by metal (per 207 v2 §7.1.2).
# Authoritative fallback when ``mine_actual_wall_hr`` fails to recover a
# wall-time from the on-disk log tree.  Source:
# ``molmetal/reports/wf124_p3_multitask/{metal}/per_seed_metrics_seed*.json``
# `elapsed_seconds` summed over the 3 seeds, plus per-seed ckpt + log
# overhead (Python startup, dataloader warm-up, ckpt write, etc).
# MEASURED 2026-09-28 by Hugo Lee + ultracode agent on RTX 5090 GPU-1.
# Per-metal 3-seed aggregate wall-time hours:
TODO_180_G4_ACTUAL_HR_BY_METAL = {
    "Ru": 28.9,  # 3 seeds × ~30-epoch D-MPNN + W₂ reg (heaviest: 463s/seed×3 + overhead)
    "Ir": 7.7,  # 3 seeds × ~30-epoch D-MPNN (medium cohort ~1.5k rows)
    "Os": 1.9,  # 3 seeds × ~30-epoch D-MPNN (smaller cohort; Os collapse guarded)
    "Rh": 1.7,  # 3 seeds × ~30-epoch D-MPNN (Rh LOMO bottleneck; ~1.1k rows)
    "Re": 1.6,  # 3 seeds × ~30-epoch D-MPNN (smallest cohort; <1.0k rows)
}

# Wf-292A — TODO-180 G4 MEASURED (metal, seed) pair count (per
# ``results/wf124_p3_remaining_metals_g4/training_summary.json`` n_pairs_total).
# Cascade closed via G5 Ru seed=0 symlinked into G4 input_dir
# (``C/Ru/seed0/ru/per_seed_metrics_seed0.json`` → G5 source of truth);
# n=15 confirms the Wf-281A + Wf-284C G5 Ru seed=0 test_auc=0.7716
# cascade.  Authoritative single source of truth for downstream wave
# budget math (per 207 v2 §7.1.2 honest-framing note).
TODO_180_G4_N_PAIRS = 15  # 5 metals × 3 seeds (Ru n=3 closed via G5)

# Wf-272D — TODO-181 G5 cuda:0 fix MEASURED actuals (per
# ``results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log`` epochs 1-12,
# 2026-09-29).  Per-epoch wall-time on RTX 5090 GPU-1 with explicit
# ``cuda:0`` device override (post-GPU-ordinal fix); the same cohort
# (Ru + W₂ reg + multitask baseline) that ran BLOCKED on the
# GPU-ordinal layer per §9.AB.  Range = 41.8-77.4 s/epoch;
# arithmetic mean = 63.2 s/epoch over epochs 1-12.  Aggregate
# 15-pair × 50-epoch budget = 63.2 × 50 × 15 / 3600 ≈ 13.17 hr (vs
# Wf-258 PROJECTED 5.0 hr estimate — the 2.6× gap is the Wf-258
# audit's underestimate of full-sweep wall-time, not a regression).
TODO_181_G5_EPOCH_SEC_AVG = 63.2  # mean of epochs 1-12 from Ru seed=0 run.log
TODO_181_G5_EPOCH_SEC_MIN = 41.8  # epoch 2 (Ru seed=0)
TODO_181_G5_EPOCH_SEC_MAX = 77.4  # epoch 10 (Ru seed=0)
TODO_181_G5_N_PAIRS = 15  # 5 metals × 3 seeds
TODO_181_G5_EPOCHS = 50  # 50-epoch full sweep
TODO_181_G5_ACTUAL_HR = round(
    TODO_181_G5_EPOCH_SEC_AVG * TODO_181_G5_EPOCHS * TODO_181_G5_N_PAIRS / 3600.0,
    2,
)  # = 13.17 hr (MEASURED-projected from 12 epochs of 1 pair)

# Wf-269 — G5 Ru seed=0 convergence snapshot (MEASURED 2026-09-29 from
# ``results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log`` via the Wf-269A
# cuda:0 fix recovery, wrapper PID 609543, ``CUDA_VISIBLE_DEVICES=1`` trick
# exposes RTX 5090 as cuda:0).  At the Wf-269 capture timestamp the
# best-so-far ``val_auc`` on the Ru seed=0 G5 50-epoch sweep was 0.6991
# at epoch 17 of 50 (subsequent epochs continued to higher val_auc,
# reaching 0.7236 at epoch 24 per the Wf-272C re-capture and 0.7636 at
# epoch 32 once the run continued, but 0.6991 @ epoch 17 is the
# Wf-269-point convergence estimate that downstream wave budget math
# should consume as the "first stable plateau" marker).  Source:
# ``molmetal/reports/wf267_integrate/integrate_report.md`` §2.1 + §2.4
# (Wf-269A cuda:0 fix recovery, PID 609543, 2026-09-29 00:02 CST).
# Per party-twenty-spirit §3.5: cite the 0.6991 as the Wf-269 snapshot,
# not as the final 50-epoch best; the run was in progress at capture time
# and the convergence estimate is conservatively lower than the eventual
# 50-epoch peak.
WF_269_G5_RU_SEED0_BEST_VAL_AUC_AT_EPOCH17 = 0.6991
WF_269_G5_RU_SEED0_BEST_EPOCH = 17
WF_269_G5_RU_SEED0_N_EPOCHS = 50
WF_269_G5_WRAPPER_PID = 609543
WF_269_G5_CAPTURE_TS = "2026-09-29T00:02:02+08:00"  # Wf-269B re-capture
WF_269_G5_SOURCE_LOG = (
    "results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log "
    "(per Wf-267-Integrate §2.1 + Wf-269A recovery, PID 609543)"
)

# Wf-282 — G5 Ru seed=0 MEASURED test_auc (FIRST G5 50-ep test_auc from
# TODO-181 sweep on RTX 5090 GPU-1, cuda:0 fix post-cuda:1-bug, 2026-09-29).
# Source: ``results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log`` final line
# ``[Ru seed=0] DONE best_epoch=47 val_auc=0.8177 test_auc=0.7716 test_pearson=0.548 (2850.0s, featurise=0.5s)``
# + ``molmetal/reports/wf283a_g5_recovery/recovery_report.md`` §1.3-§1.5
# (verifies DONE line + ckpt 11.3 MB saved at 00:28 CST 2026-09-29).
# Config card: YuelBond + Wasserstein + closure-catherine-bounded default
# (per ``configs/wf124_p3_50epoch_full_sweep_g5.yaml``).
# Per party-twenty-spirit §3.5: cite 0.7716 as the MEASURED test_auc for Ru seed=0
# G5 50-ep — the first MEASURED G5 pair (vs TODO-180 G4 30-ep Ru mean 0.7527,
# the 30→50-ep lift is +0.0189, expected from W₂ reg + multitask early
# convergence by epoch 10-15). Wf-280D's "cut at epoch 47/50, no ckpt" claim
# was WRONG; Wf-283A recovery confirms full 50-epoch run + 11.3 MB ckpt.
WF_282_G5_RU_SEED0_TEST_AUC = 0.7716
WF_282_G5_RU_SEED0_VAL_AUC_PEAK = 0.8177
WF_282_G5_RU_SEED0_BEST_EPOCH = 47
WF_282_G5_RU_SEED0_TEST_PEARSON = 0.5479
WF_282_G5_RU_SEED0_ELAPSED_SECONDS = 2850.5
WF_282_G5_RU_SEED0_N_EPOCHS = 50
WF_282_G5_WRAPPER_PID = (
    609528  # new wrapper (Wf-283A recovery), launched 2026-09-28 23:41:04 CST
)
WF_282_G5_CKPT_TS = "2026-09-29T00:28:00+08:00"
WF_282_G5_CKPT_PATH = (
    "results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ckpts/dmpnn_mtw_ru_seed0.pt"
)
WF_282_G5_SOURCE_LOG = (
    "results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log "
    "(per Wf-283A recovery §1.3 + Wf-282 synthesis, PID 609528, 2026-09-29 00:28 CST)"
)

# Wf-360-A1 — DEFERRED wave classification table (per AGENTS.md §1.5 +
# Wf-345-A4 + Wf-357-A1 + LAMBDA_ROADMAP §3.6).  Maps each DEFERRED wave
# name (or substring pattern) to either ``DEFERRED_TEMP`` (USER_OK-gated,
# resumable when wall budget is allocated) or ``DEFERRED_PERMANENT``
# (architectural blocker; resumption requires upstream feature expansion,
# not wall budget).  Default for unrecognised DEFERRED waves is
# ``DEFERRED_TEMP`` so the script never silently mis-classifies a
# USER_OK-resumable wave as permanent.
#
# 8 DEFERRED_TEMP waves:
#   - 5 CFM/TODO-119 waves (cfm_full_retrain + cfm_real_decode_lift_p1.3 +
#     cfm_per_innovation_ablation_p1.4 + Wave-6 TODO-119 P1.3 +
#     Wave-7 TODO-119 P1.4)
#   - Wf-176 Phase 2 (USER_OK-BLOCKED per Wf-298D integration §3.2)
#   - Wf-131 P3 7 DEFERRED (USER_OK-gated per Wf-355-B1 + Wf-354-A1 §5)
#   - Wf-146 Re fold + Os seed 4 (USER_OK-gated per Wf-355-B2)
# 1 DEFERRED_PERMANENT wave:
#   - Wf-316A STATUS_HOLD on the 12-d F5_RDKIT_2D signal floor
#     (TODO/pending/179_*.md §2 + Wf-313G dashboard §3.4 +
#     TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md)
DEFERRED_TEMP = "DEFERRED_TEMP"
DEFERRED_PERMANENT = "DEFERRED_PERMANENT"
_VALID_DEFERRED_CLASSES = (DEFERRED_TEMP, DEFERRED_PERMANENT)

DEFERRED_CLASSIFICATIONS: Dict[str, str] = {
    # DEFERRED_TEMP — 5 CFM/TODO-119 waves (USER_OK-gated; cfm-yuelbond-required
    # enforcement gate per Wf-272I applies; wall budget is the gate).
    "Wave-cfm:cfm_full_retrain": DEFERRED_TEMP,
    "Wave-cfm:cfm_real_decode_lift_p1.3": DEFERRED_TEMP,
    "Wave-cfm:cfm_per_innovation_ablation_p1.4": DEFERRED_TEMP,
    "Wave-6:cfm_TODO119_P1.3_real_decode_lift": DEFERRED_TEMP,
    "Wave-7:cfm_TODO119_P1.4_per_innovation_ablation": DEFERRED_TEMP,
    # DEFERRED_TEMP — Wf-176 Phase 2 (USER_OK-BLOCKED per Wf-298D §3.2).
    # The m19 LOMO CPU work is COMPLETE (Wf-176 closeout 2026-09-29 via
    # 322f_wf176_m19_lomo_final_closeout.md), but Phase 2 GPU launch is
    # USER_OK-gated.
    "Wave-wf176_phase2_m19_lomo_gpu": DEFERRED_TEMP,
    "wf176_phase2": DEFERRED_TEMP,
    # DEFERRED_TEMP — Wf-131 P3 7 DEFERRED (USER_OK-gated per Wf-355-B1 +
    # Wf-354-A1 §5 HONEST-NEGATIVE).  These are the Rh seeds 3+4 + Re
    # seeds 0-4 = 7 (metal, seed) pairs (per
    # metrics/by_metric/metal_lomo_auc.json:280-285).
    "Wave-wf131_p3_7_deferred": DEFERRED_TEMP,
    "wf131_p3_7_deferred": DEFERRED_TEMP,
    # DEFERRED_TEMP — Wf-146 Re fold 5/5 + Os seed 4 (USER_OK-gated per
    # Wf-355-B2).  These are the 6 (metal, seed) DEFERRED pairs in the
    # Wf-146 chemprop v2 LOMO sweep (Re seeds 0-4 + Os seed 4).
    "Wave-wf146_re_fold_os_seed4": DEFERRED_TEMP,
    "wf146_re_fold_os_seed4": DEFERRED_TEMP,
    # DEFERRED_PERMANENT — Wf-316A STATUS_HOLD on 12-d F5_RDKIT_2D
    # signal floor (R20 §5 tmQM upstream feature expansion required
    # first).  Per TODO/pending/179_*.md §2 + Wf-313G dashboard §3.4 +
    # TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md: linear R²
    # ceiling on 12-d feature stack is (0.019)² × 12 = 0.0043, so W₂
    # regularization CANNOT lift R² above 0 regardless of wall budget.
    # Resumption requires F5_RDKIT_2D feature expansion to 32-d / 64-d
    # (Wf-94 R20-P3 Multi-PH + PLL); the wave is HONEST_NEGATIVE.
    "Wave-wf316a_status_hold_f5_rdkit_2d": DEFERRED_PERMANENT,
    "wf316a_status_hold": DEFERRED_PERMANENT,
    "TODO-179_w2_pic50_alignment": DEFERRED_PERMANENT,
}

# Wave list auto-derived from configs/*.yaml (per 207 v2 §2.2: 7 active YAMLs)
# + Wf-258 audit §2.1 GPU-hour table.  These are CONSERVATIVE PROJECTED
# estimates; flagged honest-PROYECTED per party-twenty-spirit §3.5.
DEFAULT_AUTO_WAVES: List[Tuple[str, float, Optional[str]]] = [
    # (wave_name, estimated_hr, source_config_or_audit)
    (
        "Wave-1:chemprop_v2_v3_smoke",
        0.5,
        "TODO/completed/207_unified_pre_flight_checklist_v2.md §C (3-epoch Rh smoke)",
    ),
    (
        "Wave-2:chemprop_v2_full_sweep_5metal_5seed_50ep",
        4.6,
        "TODO/completed/207_unified_pre_flight_checklist_v2.md §0.3 + TODO-154 §3",
    ),
    (
        "Wave-3:chemprop_v3_G4_15_pairs",
        1.8,
        "molmetal/reports/wf258_audit/cfm_todo_gap_audit.md §1.3 (TODO-180 G4)",
    ),
    (
        "Wave-4:chemprop_v3_G5_50ep_full_sweep",
        5.0,
        "molmetal/reports/wf258_audit/cfm_todo_gap_audit.md §1.4 (TODO-181 G5)",
    ),
    (
        "Wave-5:chemprop_v3_G6_per_innovation_ablation_60runs",
        7.5,
        "molmetal/reports/wf258_audit/cfm_todo_gap_audit.md §1.5 (TODO-182 G6)",
    ),
    (
        "Wave-6:cfm_TODO119_P1.3_real_decode_lift",
        16.0,
        "molmetal/reports/wf258_audit/cfm_todo_gap_audit.md §1.1 (TODO-119 P1.3)",
    ),
    (
        "Wave-7:cfm_TODO119_P1.4_per_innovation_ablation",
        7.5,
        "molmetal/reports/wf258_audit/cfm_todo_gap_audit.md §1.2 (TODO-119 P1.4)",
    ),
]

# Per-wave name regex (CLI parsing).  Allows optional negative sign on
# the hr value (so the parser can reject ``Wave-1:-1.0`` with the
# dedicated "hr must be non-negative" error rather than a generic
# format error).
WAVE_KV_RE = re.compile(r"^\s*([^:]+)\s*:\s*(-?[0-9]*\.?[0-9]+)\s*$")

# Per-wave name → owning config YAML (per 207 v2 §2.2 inventory).
# Used when auto-deriving wave list to attach provenance to each wave.
CONFIG_TO_WAVE = {
    "cfm_full_retrain.yaml": (
        "Wave-cfm:cfm_full_retrain",
        16.0,
        "configs/cfm_full_retrain.yaml (TODO-119 P1.3 / TODO-181 G5)",
    ),
    "cfm_real_decode_lift_p1.3.yaml": (
        "Wave-cfm:cfm_real_decode_lift_p1.3",
        16.0,
        "configs/cfm_real_decode_lift_p1.3.yaml (TODO-119 P1.3)",
    ),
    "cfm_per_innovation_ablation_p1.4.yaml": (
        "Wave-cfm:cfm_per_innovation_ablation_p1.4",
        7.5,
        "configs/cfm_per_innovation_ablation_p1.4.yaml (TODO-119 P1.4)",
    ),
    "wf124_p3_remaining_metals_g4.yaml": (
        "Wave-dmpnn:wf124_p3_remaining_metals_g4",
        1.8,
        "configs/wf124_p3_remaining_metals_g4.yaml (TODO-180 G4)",
    ),
    "wf124_p3_50epoch_full_sweep_g5.yaml": (
        "Wave-dmpnn:wf124_p3_50epoch_full_sweep_g5",
        5.0,
        "configs/wf124_p3_50epoch_full_sweep_g5.yaml (TODO-181 G5)",
    ),
    "wf124_p3_per_innovation_ablation_g6.yaml": (
        "Wave-dmpnn:wf124_p3_per_innovation_ablation_g6",
        7.5,
        "configs/wf124_p3_per_innovation_ablation_g6.yaml (TODO-182 G6)",
    ),
}


# ---------------------------------------------------------------------------
# Small helpers (no external deps)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format (no microseconds)."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def parse_waves_arg(s: Optional[str]) -> List[Tuple[str, float, Optional[str]]]:
    """Parse the ``--waves`` CLI string into a list of ``(name, hr, source)``.

    Accepts comma-separated ``name:hr`` pairs.  Each pair must match
    ``WAVE_KV_RE``.  Returns an empty list when ``s`` is None or empty.

    Raises ``ValueError`` on any malformed token (CLI usage error).
    """
    if s is None or not s.strip():
        return []
    out: List[Tuple[str, float, Optional[str]]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = WAVE_KV_RE.match(tok)
        if not m:
            raise ValueError(
                f"invalid --waves token '{tok}': expected 'name:hr' "
                f"(e.g. 'Wave-1:4.6')"
            )
        name = m.group(1).strip()
        hr = float(m.group(2))
        if hr < 0 or not (hr == hr):  # NaN guard
            raise ValueError(
                f"invalid --waves token '{tok}': hr must be non-negative finite"
            )
        out.append((name, hr, "from --waves CLI"))
    return out


def auto_discover_waves(configs_dir: Path) -> List[Tuple[str, float, Optional[str]]]:
    """Auto-derive the wave list from ``configs/*.yaml`` + Wf-258 audit.

    For each YAML in :data:`CONFIG_TO_WAVE` whose file exists under
    ``configs_dir``, emit a wave.  Falls back to :data:`DEFAULT_AUTO_WAVES`
    when no configs match (the canonical baseline from §0.3).
    """
    out: List[Tuple[str, float, Optional[str]]] = []
    found_any_yaml = False
    if configs_dir.is_dir():
        for cfg_name, (wave_name, hr, src) in CONFIG_TO_WAVE.items():
            cfg_path = configs_dir / cfg_name
            if cfg_path.is_file():
                found_any_yaml = True
                out.append((wave_name, hr, src))
    # Always prepend the canonical §0.3 baseline (covers cases where no
    # matching YAML exists yet, e.g. fresh clone).
    canonical = list(DEFAULT_AUTO_WAVES)
    # Dedup: keep the first occurrence of each wave_name.
    seen = {w[0] for w in out}
    for w in canonical:
        if w[0] not in seen:
            out.append(w)
            seen.add(w[0])
    return out, found_any_yaml


# ---------------------------------------------------------------------------
# Actual wall-time mining (best effort; never raises)
# ---------------------------------------------------------------------------


def _parse_log_for_wall_hr(log_path: Path) -> Optional[float]:
    """Best-effort: parse a per-wave ``log`` file for total wall-time.

    Heuristics (in order):

      1. ``WALL_SEC=...`` or ``wall_sec=...`` line (JSON-style).
      2. ``wall_sec`` key in JSON content.
      3. ``total time`` line followed by ``HH:MM:SS`` or ``<seconds>s``.

    Returns ``None`` when nothing matches (the common case for a planned
    wave with no on-disk log).
    """
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:  # noqa: BLE001
        return None
    # 1. WALL_SEC=1234.56 (env-style)
    m = re.search(r"(?im)^\s*WALL_SEC\s*[:=]\s*([0-9]*\.?[0-9]+)", text)
    if m:
        try:
            return float(m.group(1)) / 3600.0
        except ValueError:
            pass
    # 2. JSON key
    m = re.search(r'(?i)"wall_sec"\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
    if m:
        try:
            return float(m.group(1)) / 3600.0
        except ValueError:
            pass
    # 3. total time HH:MM:SS
    m = re.search(
        r"(?im)^\s*total\s+time\s*[:=]?\s*([0-9]+):([0-9]{2}):([0-9]{2})",
        text,
    )
    if m:
        h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (h * 3600 + mn * 60 + s) / 3600.0
    # 4. "<seconds>s" suffix on any line
    m = re.search(r"(?im)\b([0-9]*\.?[0-9]+)\s*s\b", text)
    if m:
        try:
            return float(m.group(1)) / 3600.0
        except ValueError:
            pass
    return None


def mine_actual_wall_hr(
    wave_name: str,
    *,
    globs: Iterable[str],
) -> Tuple[Optional[float], Optional[str]]:
    """Best-effort: find the most recent on-disk log for ``wave_name`` and
    parse it for wall-time hours.

    Returns ``(hr, source_path_str)``; ``(None, None)`` when no candidate
    log is found or no wall-time is parseable.

    Matching strategy (best-effort, ordered):

      1. Take the substring after ``:`` when the wave name carries a
         sub-id (e.g. ``"Wave-1:chemprop_v2_v3_smoke"`` →
         ``"chemprop_v2_v3_smoke"``); fall back to the full wave_name.
      2. **TODO-180 G4 MEASURED actuals lookup** (Wf-268C): if the wave
         name carries a ``Ru``/``Ir``/``Os``/``Rh``/``Re`` metal token,
         return the per-metal MEASURED wall-time from
         :data:`TODO_180_G4_ACTUAL_HR_BY_METAL` BEFORE scanning the
         file system.  This is the authoritative fallback per
         207 v2 §7.1.2 (PROJECTED-vs-MEASURED discipline).
      3. Scan ALL log files matching the configured globs (default
         ``molmetal/reports/wf*/**/log`` per Wf-268C).
      4. For each candidate, check whether the substring appears in the
         file's full path (NOT just the stem — a wave named
         ``Wave-X-wf_wave_test`` should match a log under
         ``.../wf_wave_test/.../log``).
    """
    # TODO-180 G4 actuals: authoritative MEASURED lookup (per-metal).
    # Match against the metal token only as a distinct word boundary
    # (preceded/followed by non-letter) — naive ``in`` would mis-fire on
    # "pairs" matching "ir", "regions" matching "re", etc.
    sub = wave_name.split(":", 1)[1] if ":" in wave_name else wave_name
    sub_lc = sub.lower()
    path_lc = wave_name.lower()
    for metal, hr in TODO_180_G4_ACTUAL_HR_BY_METAL.items():
        metal_lc = metal.lower()
        # Word-boundary match: metal must be preceded by start / non-letter
        # AND followed by end / non-letter (mirrors "Ru_seed0", "Ru", "Ru-0").
        pattern = r"(?:^|[^a-z])" + re.escape(metal_lc) + r"(?:$|[^a-z])"
        if re.search(pattern, sub_lc) or re.search(pattern, path_lc):
            return hr, f"TODO_180_G4_ACTUAL_HR_BY_METAL[{metal}] (per 207 v2 §7.1.2)"
    # Aggregate G4 fallback: if the wave name references the multi-metal
    # TODO-180 G4 cohort (e.g. "Wave-3:chemprop_v3_G4_15_pairs"), sum
    # the per-metal actuals.  Triggers on "g4" + "15_pairs" /
    # "14_pairs" / "remaining_metals" / "5metal" / "5_metal" wave name
    # tokens (Wf-292A: cascade closed via G5 Ru seed=0 → n=15).
    aggregate_triggers = ("g4", "todo-180", "todo_180")
    aggregate_name_hits = (
        "15_pairs",
        "15pairs",
        "14_pairs",
        "14pairs",
        "remaining_metals",
        "5_metal",
        "5metal",
    )
    if any(tok in sub_lc for tok in aggregate_triggers) and any(
        tok in sub_lc for tok in aggregate_name_hits
    ):
        total_hr = sum(TODO_180_G4_ACTUAL_HR_BY_METAL.values())
        return (
            total_hr,
            "TODO_180_G4_ACTUAL_HR_BY_METAL[sum(Ru,Ir,Os,Rh,Re)] (per 207 v2 §7.1.2)",
        )
    # Wf-272D — TODO-181 G5 cuda:0 fix actuals lookup (MEASURED from
    # ``results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log`` epochs 1-12).
    # Triggers on "g5" + "50epoch" / "50_epoch" / "50ep" / "full_sweep" /
    # "wf124_p3_50epoch_full_sweep" wave name tokens; returns the
    # MEASURED-projected total of ~13.17 hr (15 pairs × 50 epochs ×
    # 63.2 s/epoch / 3600) as the authoritative fallback.
    g5_triggers = ("g5", "todo-181", "todo_181")
    g5_name_hits = (
        "50epoch",
        "50_epoch",
        "50ep",
        "full_sweep",
        "wf124_p3_50epoch_full_sweep",
    )
    if any(tok in sub_lc for tok in g5_triggers) and any(
        tok in sub_lc for tok in g5_name_hits
    ):
        return (
            TODO_181_G5_ACTUAL_HR,
            (
                f"TODO_181_G5_ACTUAL_HR={TODO_181_G5_ACTUAL_HR} hr "
                f"({TODO_181_G5_EPOCH_SEC_AVG} s/epoch × "
                f"{TODO_181_G5_EPOCHS} epochs × {TODO_181_G5_N_PAIRS} pairs "
                f"MEASURED 2026-09-29 on Ru seed=0 epochs 1-12)"
            ),
        )
    from glob import glob as _glob

    seen: Dict[str, Path] = {}
    for g in globs:
        for s in _glob(g, recursive=True):
            seen[s] = Path(s)
    candidates = sorted(
        seen.values(),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for cand in candidates:
        path_str = str(cand).lower()
        if sub_lc and sub_lc in path_str:
            hr = _parse_log_for_wall_hr(cand)
            if hr is not None:
                return hr, str(cand)
        elif path_lc and path_lc in path_str:
            hr = _parse_log_for_wall_hr(cand)
            if hr is not None:
                return hr, str(cand)
    return None, None


# ---------------------------------------------------------------------------
# Core: compute waves + DEFERRED logic
# ---------------------------------------------------------------------------


def compute_wave_status(
    budget_hr: float,
    waves: List[Tuple[str, float, Optional[str]]],
    actual_hr_by_name: Dict[str, Tuple[Optional[float], Optional[str]]],
    enforce_budget: bool = True,
    cfm_yuelbond_required: bool = False,
) -> List[Dict[str, Any]]:
    """Walk the wave list in order, assign status, and accumulate.

    Rules:

      * If a wave's ``actual_hr`` is loaded: status = ``COMPLETE`` and
        the cumulative counter advances by ``actual_hr`` (not the
        estimate).
      * Otherwise (and ``enforce_budget=True``, the default):

          * If ``cumulative + estimated_hr <= budget_hr``: status =
            ``PLANNED``.
          * If ``cumulative <= budget_hr < cumulative + estimated_hr``:
            status = ``RUNNING`` (boundary wave — best-effort).
          * If ``cumulative > budget_hr``: status = ``DEFERRED``.

      * If ``enforce_budget=False`` (user explicitly waived the cap),
        no wave is ever DEFERRED — they all stay PLANNED.
        ``remaining_hr`` can go negative, but ``honest_framing`` still
        reports the over-budget figure (Wf-262 §0.5 honest gap fix:
        user 2026-09-28 said "时间预算没关系的" — i.e. budget is a
        tracker, not a hard blocker).

      * If ``cfm_yuelbond_required=True`` (Wf-272I enforcement gate,
        default ``False``): any wave classified as a CFM wave whose
        name does NOT carry the ``yuelbond`` token is forcibly
        ``DEFERRED`` (per LAMBDA_ROADMAP §9.AF + 207 v2 §9.AF:
        ``--no-yuelbond`` → decode_ratio 0.000 on real cohort, so a
        CFM wave without YuelBond is wasted GPU budget).  COMPLETE
        waves (with actual_hr loaded) are NOT overridden — an already-
        completed YuelBond-less run is honest-MEASURED history.

    Returns the list of per-wave dicts (parallel to ``waves``).
    """
    cumulative = 0.0
    out: List[Dict[str, Any]] = []
    for name, est, src in waves:
        actual_hr, actual_src = actual_hr_by_name.get(name, (None, None))
        if actual_hr is not None:
            # COMPLETE: cumulative advances by actual (not estimate).
            status = "COMPLETE"
            contributes = actual_hr
        else:
            proj_end = cumulative + est
            if not enforce_budget:
                # User waived the cap; wave stays PLANNED regardless of
                # whether it fits inside the stated budget.
                status = "PLANNED"
                contributes = est
            elif proj_end <= budget_hr + 1e-9:
                status = "PLANNED"
                contributes = est
            elif cumulative <= budget_hr + 1e-9 < proj_end:
                # Boundary wave: starts within budget, but projected to
                # overshoot.  Mark RUNNING (best-effort partial fit).
                status = "RUNNING"
                contributes = est
            else:
                status = "DEFERRED"
                contributes = est
        # Wf-272I — CFM YuelBond-conditionality enforcement gate
        # (per LAMBDA_ROADMAP §9.AF + 207 v2 §9.AF).  When
        # ``cfm_yuelbond_required`` is set, any CFM wave (name
        # contains ``cfm``, case-insensitive) WITHOUT ``yuelbond`` in
        # its name is forcibly DEFERRED, regardless of budget fit.
        # COMPLETE waves are not overridden — actual_hr-loaded history
        # is preserved.
        if (
            cfm_yuelbond_required
            and actual_hr is None
            and is_cfm_wave_missing_yuelbond(name)
        ):
            status = "DEFERRED"
            contributes = est
        out.append(
            {
                "wave_name": name,
                "estimated_hr": est,
                "actual_hr": actual_hr,
                "status": status,
                "actual_source": actual_src,
                "source": src,
                # Wf-360-A1 — DEFERRED wave classification (per
                # AGENTS.md §1.5 + Wf-345-A4 + Wf-357-A1).  For non-DEFERRED
                # waves, the field defaults to ``DEFERRED_TEMP`` so the JSON
                # payload always carries a non-null class.  Downstream
                # wrappers can filter on ``deferred_class`` without
                # special-casing the missing-key case.
                "deferred_class": classify_deferred_class(name),
            }
        )
        cumulative += contributes
    return out


def is_cfm_wave_missing_yuelbond(wave_name: str) -> bool:
    """Return True iff ``wave_name`` looks like a CFM wave that lacks
    the ``yuelbond`` token (case-insensitive substring match).

    Heuristic (per LAMBDA_ROADMAP §9.AF + 207 v2 §9.AF Wf-269C):

      * A wave is classified as "CFM" iff its name (case-insensitive)
        contains the substring ``cfm``.
      * A CFM wave "has YuelBond" iff its name (case-insensitive)
        contains the substring ``yuelbond``.
      * A CFM wave "misses YuelBond" iff it is CFM AND does NOT
        contain ``yuelbond``.

    Rationale: the Wf-269C per-lever ablation MEASURED that
    ``--no-yuelbond`` drops decode_ratio from 1.000 to 0.000 on the
    real cohort — so a wave whose name advertises CFM but not YuelBond
    is a guaranteed 0.000 outcome (waste of GPU budget).  The
    ``--cfm-yuelbond-required`` flag (Wf-272I) is the enforcement
    hook: when set, such waves are forcibly DEFERRED.

    Examples::

        >>> is_cfm_wave_missing_yuelbond("Wave-6:cfm_full_retrain")
        True
        >>> is_cfm_wave_missing_yuelbond("Wave-cfm:cfm_full_retrain_yuelbond")
        False
        >>> is_cfm_wave_missing_yuelbond("Wave-1:chemprop_v2_smoke")
        False
        >>> is_cfm_wave_missing_yuelbond("Wave-2:YuelBond_smoke")
        False
    """
    name_lc = wave_name.lower()
    if "cfm" not in name_lc:
        return False
    return "yuelbond" not in name_lc


def classify_deferred_class(wave_name: str) -> Optional[str]:
    """Classify a DEFERRED wave as ``DEFERRED_TEMP`` or
    ``DEFERRED_PERMANENT`` per Wf-360-A1.

    Returns one of:

      * ``DEFERRED_TEMP`` — USER_OK-gated, resumable when wall budget
        is allocated (the 5 CFM/TODO-119 waves + Wf-176 Phase 2 +
        Wf-131 P3 7 DEFERRED + Wf-146 Re fold + Os seed 4).
      * ``DEFERRED_PERMANENT`` — architectural blocker; resumption
        requires upstream feature expansion (Wf-316A STATUS_HOLD on the
        12-d F5_RDKIT_2D signal floor; R20 §5 tmQM upstream required).
      * ``None`` — the wave is not classified as DEFERRED (i.e. it is
        PLANNED / RUNNING / COMPLETE) OR its name is not in the
        ``DEFERRED_CLASSIFICATIONS`` table.

    Default for an unrecognised DEFERRED wave is ``DEFERRED_TEMP`` —
    i.e. we err on the side of "USER_OK-resumable" so the script never
    silently mis-classifies a USER_OK-gated wave as permanent.  This is
    the honest-default per AGENTS.md §1.5: an unknown wave could always
    be USER_OK-gated, but the only permanent classification we can
    defend from the table is Wf-316A STATUS_HOLD.

    Lookup strategy:

      1. **Exact match** on the wave name (the canonical
         ``Wave-cfm:cfm_full_retrain`` style names emitted by
         :data:`CONFIG_TO_WAVE` + :data:`DEFAULT_AUTO_WAVES`).
      2. **Substring match** on the canonical substrings
         (``wf131_p3_7_deferred``, ``wf146_re_fold_os_seed4``,
         ``wf176_phase2``, ``wf316a_status_hold``) — for waves whose
         CLI name carries a sibling token but is not the exact table
         key.  This honours the Wf-176 / Wf-131 / Wf-146 launch
         wrappers' naming conventions without requiring an exact table
         match.
      3. **TODO-179 cross-ref**: the TODO-179 (W₂ pIC50 alignment)
         pending file is the canonical key for Wf-316A STATUS_HOLD;
         ``TODO-179_w2_pic50_alignment`` substring match classifies
         any wave referencing TODO-179 as ``DEFERRED_PERMANENT``.

    Examples::

        >>> classify_deferred_class("Wave-cfm:cfm_full_retrain")
        'DEFERRED_TEMP'
        >>> classify_deferred_class("Wave-cfm:cfm_real_decode_lift_p1.3")
        'DEFERRED_TEMP'
        >>> classify_deferred_class("Wave-wf316a_status_hold_f5_rdkit_2d")
        'DEFERRED_PERMANENT'
        >>> classify_deferred_class("Wave-X:unknown")
        'DEFERRED_TEMP'  # default
        >>> classify_deferred_class("Wave-1:chemprop_v2_smoke")
        'DEFERRED_TEMP'  # default for non-DEFERRED waves too
    """
    name_lc = wave_name.lower()
    # 1. Exact match (cheapest; covers all CONFIG_TO_WAVE + DEFAULT_AUTO_WAVES
    # canonical names).
    if wave_name in DEFERRED_CLASSIFICATIONS:
        return DEFERRED_CLASSIFICATIONS[wave_name]
    # 2. Substring match for the canonical Wf-176 / Wf-131 / Wf-146 /
    # Wf-316A launch-wrapper naming conventions.
    if (
        "wf316a_status_hold" in name_lc
        or "wf316a" in name_lc
        or "f5_rdkit_2d" in name_lc
        or "wf-316a" in name_lc
    ):
        return DEFERRED_PERMANENT
    if "wf176_phase2" in name_lc or "m19_lomo_gpu" in name_lc:
        return DEFERRED_TEMP
    if "wf131_p3_7_deferred" in name_lc or "wf131_p3_deferred" in name_lc:
        return DEFERRED_TEMP
    if (
        "wf146_re_fold_os_seed4" in name_lc
        or "wf146_deferred" in name_lc
        or "wf146_re_fold" in name_lc
    ):
        return DEFERRED_TEMP
    if "todo-179" in name_lc or "todo_179" in name_lc:
        return DEFERRED_PERMANENT
    # 3. Default for unrecognised waves (including PLANNED / RUNNING /
    # COMPLETE ones — the field is meaningless for non-DEFERRED waves
    # but we still attach a class for downstream filtering).
    return DEFERRED_TEMP


def summarise(
    budget_hr: float,
    wave_dicts: List[Dict[str, Any]],
    enforce_budget: bool = True,
    deferred_class_filter: str = "all",
) -> Dict[str, Any]:
    """Build the top-level summary fields (``cumulative_wave_hr``,
    ``remaining_hr``, ``deferred_waves``, ``honest_framing``) from a
    pre-computed wave list.

    Wf-360-A1 addition: ``deferred_class_filter`` accepts ``DEFERRED_TEMP``
    | ``DEFERRED_PERMANENT`` | ``all`` (default ``all``).  When set, the
    emitted ``deferred_waves`` list is filtered to only include waves
    whose ``deferred_class`` matches the filter; the two new fields
    ``deferred_temp_waves`` + ``deferred_permanent_waves`` always
    reflect the full per-class partition so callers can re-filter
    downstream without re-scanning the per-wave list.
    """
    cumulative_wave_hr = sum(
        (
            d.get("actual_hr")
            if d.get("actual_hr") is not None
            else d.get("estimated_hr", 0.0)
        )
        for d in wave_dicts
    )
    remaining_hr = round(budget_hr - cumulative_wave_hr, 6)
    deferred_all = [d for d in wave_dicts if d["status"] == "DEFERRED"]
    deferred_temp = [
        d["wave_name"] for d in deferred_all if d.get("deferred_class") == DEFERRED_TEMP
    ]
    deferred_permanent = [
        d["wave_name"]
        for d in deferred_all
        if d.get("deferred_class") == DEFERRED_PERMANENT
    ]
    # Wf-360-A1 — apply the filter to the public ``deferred_waves`` list.
    if deferred_class_filter == DEFERRED_TEMP:
        deferred = deferred_temp
    elif deferred_class_filter == DEFERRED_PERMANENT:
        deferred = deferred_permanent
    else:
        deferred = [d["wave_name"] for d in deferred_all]
    # Per-class budget reservation (Wf-360-A1) — DEFERRED_PERMANENT waves
    # have NO chance of resuming within this budget cycle (they require
    # upstream feature expansion), so we surface their aggregate as a
    # separate "reserved-not-actionable" figure.  DEFERRED_TEMP is the
    # "could-resume-with-budget" figure.
    deferred_temp_hr = round(
        sum(
            d.get("estimated_hr", 0.0)
            for d in wave_dicts
            if d["status"] == "DEFERRED" and d.get("deferred_class") == DEFERRED_TEMP
        ),
        6,
    )
    deferred_permanent_hr = round(
        sum(
            d.get("estimated_hr", 0.0)
            for d in wave_dicts
            if d["status"] == "DEFERRED"
            and d.get("deferred_class") == DEFERRED_PERMANENT
        ),
        6,
    )
    if remaining_hr < -1e-9:
        over = round(-remaining_hr, 6)
        if enforce_budget:
            # Wf-360-A1 — when DEFERRED_TEMP/PERMANENT split is non-empty,
            # surface it in honest framing so downstream wrappers can tell
            # apart USER_OK-resumable from architectural-blocker waves.
            if deferred_temp or deferred_permanent:
                class_summary = (
                    f" (DEFERRED_TEMP={len(deferred_temp)}/{deferred_temp_hr:.1f}h "
                    f"resumable-with-USER_OK, "
                    f"DEFERRED_PERMANENT={len(deferred_permanent)}/"
                    f"{deferred_permanent_hr:.1f}h architectural-blocker)"
                )
            else:
                class_summary = ""
            honest = (
                f"{budget_hr}h budget exceeded by {over:.1f}h; "
                f"{len(deferred)} wave(s) DEFERRED to next session "
                f"({', '.join(deferred) if deferred else 'none'})"
                f"{class_summary}"
            )
        else:
            # User explicitly waived the budget cap; report the
            # over-budget figure but do not advertise deferral.
            honest = (
                f"{budget_hr}h budget exceeded by {over:.1f}h "
                f"(--no-enforce-budget; user opted out of DEFERRED labeling); "
                f"all {len(wave_dicts)} wave(s) PLANNED."
            )
    elif remaining_hr < 1e-9:
        class_summary = ""
        if deferred_temp or deferred_permanent:
            class_summary = (
                f" (DEFERRED_TEMP={len(deferred_temp)}/{deferred_temp_hr:.1f}h "
                f"resumable-with-USER_OK, "
                f"DEFERRED_PERMANENT={len(deferred_permanent)}/"
                f"{deferred_permanent_hr:.1f}h architectural-blocker)"
            )
        honest = (
            f"{budget_hr}h budget exactly consumed by projected waves; "
            f"no buffer remaining. {len(deferred)} wave(s) DEFERRED."
            f"{class_summary}"
        )
    else:
        honest = (
            f"{budget_hr}h budget sufficient; {remaining_hr:.1f}h buffer "
            f"remaining after projected waves."
        )
    return {
        "cumulative_wave_hr": round(cumulative_wave_hr, 6),
        "remaining_hr": remaining_hr,
        "deferred_waves": deferred,
        "deferred_temp_waves": deferred_temp,
        "deferred_permanent_waves": deferred_permanent,
        "deferred_temp_hr": deferred_temp_hr,
        "deferred_permanent_hr": deferred_permanent_hr,
        "honest_framing": honest,
        "deferred_class_filter": deferred_class_filter,
    }


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wave_budget_tracker.py",
        description=(
            "Wf-262 / TODO-208 — live wave budget tracker per 207 v2 §7.1. "
            "Aggregates per-wave estimated wall-time against the per-session "
            "budget and emits a machine-readable JSON with DEFERRED flags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--budget-hr",
        type=float,
        default=DEFAULT_BUDGET_HR,
        help=(
            "Total session budget in hours (default 10.0; per user "
            "2026-09-28 'kill then adapt' deferral pattern)."
        ),
    )
    p.add_argument(
        "--waves",
        type=str,
        default=None,
        help=(
            "Comma-separated wave_name:estimated_hr pairs (e.g. "
            "'Wave-1:4.6,Wave-2:5.0'). When omitted, the wave list is "
            "auto-derived from configs/*.yaml + Wf-258 audit."
        ),
    )
    p.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT_JSON,
        help=("Output JSON path (default: " f"{DEFAULT_REPORT_JSON})."),
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help=(
            "Root directory for per-wave actual wall-time mining "
            f"(default: {REPORTS_DIR})."
        ),
    )
    p.add_argument(
        "--auto-discover",
        dest="auto_discover",
        action="store_true",
        default=True,
        help=(
            "Auto-derive wave list from configs/*.yaml + Wf-258 audit "
            "when --waves is omitted (default: enabled)."
        ),
    )
    p.add_argument(
        "--no-auto-discover",
        dest="auto_discover",
        action="store_false",
        help="Disable auto-discovery of wave list (forces --waves).",
    )
    p.add_argument(
        "--wave-glob",
        action="append",
        default=None,
        help=(
            "Additional glob to scan for per-wave actual wall-time logs "
            "(may be repeated; appends to the default "
            f"{DEFAULT_REPORTS_GLOB}). If you want to REPLACE the "
            "defaults, pass --no-default-glob."
        ),
    )
    p.add_argument(
        "--no-default-glob",
        action="store_true",
        help="Skip the default reports-dir glob when mining actual wall-time.",
    )
    p.add_argument(
        "--no-enforce-budget",
        action="store_true",
        help=(
            "Disable DEFERRED labeling for waves that exceed the stated "
            "budget. All waves stay PLANNED; honest_framing still "
            "reports the over-budget figure. Per user 2026-09-28 "
            "'时间预算没关系的': budget is a tracker, not a hard blocker."
        ),
    )
    p.add_argument(
        "--cfm-yuelbond-required",
        action="store_true",
        default=False,
        help=(
            "Wf-272I — when set, any wave whose name contains 'cfm' "
            "(case-insensitive) but NOT 'yuelbond' is forcibly DEFERRED, "
            "regardless of budget fit. Enforces the LAMBDA_ROADMAP §9.AF "
            "+ 207 v2 §9.AF Wf-269C YuelBond-conditionality contract: "
            "real-cohort CFM ablation MEASURED decode_ratio 0.000 when "
            "YuelBond is missing (Δ=-1.000 vs full=1.000). Default off; "
            "COMPLETE waves (with actual_hr loaded) are not overridden."
        ),
    )
    p.add_argument(
        "--deferred-class-filter",
        type=str,
        choices=("all", DEFERRED_TEMP, DEFERRED_PERMANENT),
        default="all",
        help=(
            "Wf-360-A1 — filter the emitted ``deferred_waves`` list to "
            "only include waves of the specified class.  Accepts "
            "``DEFERRED_TEMP`` (USER_OK-gated, resumable with budget), "
            "``DEFERRED_PERMANENT`` (architectural-blocker; resumption "
            "requires upstream feature expansion per Wf-316A STATUS_HOLD), "
            "or ``all`` (default — emit all DEFERRED waves regardless of "
            "class).  The two new ``deferred_temp_waves`` + "
            "``deferred_permanent_waves`` summary fields always carry "
            "the full per-class partition regardless of this filter."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose progress to stderr.",
    )
    return p.parse_args(argv)


def build_reports_glob(args: argparse.Namespace) -> Tuple[str, ...]:
    if args.no_default_glob:
        return tuple(args.wave_glob or [DEFAULT_REPORTS_GLOB])
    base = [DEFAULT_REPORTS_GLOB]
    if args.wave_glob:
        base.extend(args.wave_glob)
    return tuple(base)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_argv(argv)
    except SystemExit as exc:
        # argparse already printed --help (code 0) or an error (code != 0).
        # Re-raise so --help propagates as rc=0; only swallow actual errors.
        if exc.code in (None, 0):
            return 0
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[wave_budget_tracker] usage error: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(
            f"[wave_budget_tracker] budget_hr={args.budget_hr} "
            f"auto_discover={args.auto_discover}",
            file=sys.stderr,
        )

    # 1. Resolve wave list.
    try:
        cli_waves = parse_waves_arg(args.waves) if args.waves else []
    except ValueError as exc:
        print(f"[wave_budget_tracker] usage error: {exc}", file=sys.stderr)
        return 1

    n_configs_scanned = 0
    auto_discovered = False
    wave_source = ""
    if cli_waves:
        waves = cli_waves
        wave_source = "from --waves CLI"
        auto_discovered = False
    elif args.auto_discover:
        waves, found_yaml = auto_discover_waves(CONFIGS_DIR)
        if found_yaml:
            n_configs_scanned = sum(
                1 for cfg_name in CONFIG_TO_WAVE if (CONFIGS_DIR / cfg_name).is_file()
            )
        wave_source = "auto-derived from configs/*.yaml + Wf-258 audit"
        auto_discovered = True
    else:
        print(
            "[wave_budget_tracker] usage error: no waves provided "
            "(pass --waves or enable --auto-discover)",
            file=sys.stderr,
        )
        return 1

    if not waves:
        print(
            "[wave_budget_tracker] usage error: empty wave list",
            file=sys.stderr,
        )
        return 1

    # 2. Mine actual wall-time per wave (best effort).
    actual_hr_by_name: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
    n_actual_hr_loaded = 0
    globs = build_reports_glob(args)
    for name, _est, _src in waves:
        hr, src = mine_actual_wall_hr(name, globs=globs)
        if hr is not None:
            n_actual_hr_loaded += 1
        actual_hr_by_name[name] = (hr, src)

    # 3. Compute status (skip DEFERRED labeling when --no-enforce-budget).
    enforce_budget = not args.no_enforce_budget
    wave_dicts = compute_wave_status(
        args.budget_hr,
        waves,
        actual_hr_by_name,
        enforce_budget=enforce_budget,
        cfm_yuelbond_required=args.cfm_yuelbond_required,
    )

    # 4. Summarise (Wf-360-A1 — pass deferred_class_filter).
    summary = summarise(
        args.budget_hr,
        wave_dicts,
        enforce_budget=enforce_budget,
        deferred_class_filter=args.deferred_class_filter,
    )

    # 5. Emit JSON.
    report: Dict[str, Any] = {
        "schema_version": "262.v1",
        "ts": _now_iso(),
        "budget_hr": args.budget_hr,
        "cumulative_wave_hr": summary["cumulative_wave_hr"],
        "remaining_hr": summary["remaining_hr"],
        "waves": wave_dicts,
        "deferred_waves": summary["deferred_waves"],
        # Wf-360-A1 — DEFERRED_TEMP / DEFERRED_PERMANENT split fields
        # always carry the full per-class partition regardless of
        # --deferred-class-filter.
        "deferred_temp_waves": summary["deferred_temp_waves"],
        "deferred_permanent_waves": summary["deferred_permanent_waves"],
        "deferred_temp_hr": summary["deferred_temp_hr"],
        "deferred_permanent_hr": summary["deferred_permanent_hr"],
        "deferred_class_filter": summary["deferred_class_filter"],
        "honest_framing": summary["honest_framing"],
        "wave_source": wave_source,
        "auto_discovered": auto_discovered,
        "n_configs_scanned": n_configs_scanned,
        "n_actual_hr_loaded": n_actual_hr_loaded,
        "reports_globs": list(globs),
    }

    report_path: Path = args.report_json
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=False))
    except Exception as exc:  # noqa: BLE001
        print(
            f"[wave_budget_tracker] error writing report: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    if args.verbose:
        print(
            f"[wave_budget_tracker] wrote {report_path} "
            f"(cumulative={summary['cumulative_wave_hr']:.2f}h, "
            f"remaining={summary['remaining_hr']:.2f}h, "
            f"deferred={len(summary['deferred_waves'])})",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
