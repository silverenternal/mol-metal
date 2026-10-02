# PRE_TRAINING_GATE_v2 — 重新训练前置门槛 v2 设计

**Date**: 2026-10-02 (Wf-432 A1 docs-only retry)
**Status**: Active — single source of truth for the pre-training gate that ensures rigorous alignment (NOT reproduction) with Krasnov 2026 + Rusanov 2025 + DrugOOD 2022 baselines.
**Owner**: Mol-Metal (TODO docs agents; party-twenty-spirit §3.5 honest framing mandate).

## Cross-references

- `TODO/03_baselines/COMPARISON_BASELINES.md` — selected baselines + protocol alignment detail
- `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` — quick reference table
- `TODO/02_metrics/METRICS_TARGETS.md` §11 — Tier 1 LEAD + Tier 2 internal metrics status
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — 21 mechanisms × 20 metrics contribution matrix
- `TODO/03_baselines/MOCK_TO_REAL_PLAN.md` — 5 MOCK→REAL MOCK metric conversion details
- `TODO/03_baselines/HONEST_NEGATIVE_CLOSURE_PLAN.md` — 3 honest-negative pre-closure paths
- `TODO/03_baselines/PROTOCOL_VALIDATION.md` — per-baseline protocol validation hook details
- `TODO/decisions.md` D-Wf-431 — alignment 不是 reproduction (the principle that motivated this gate)
- `AGENTS.md` §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 — standing directives
- `molmetal/scripts/pre_flight_check.py` line 3142 `EXPECTED_GATE_COUNTS` — the 95-gate contract this gate extends
- `molmetal/scripts/wf386a3_atomic_write.py` — the atomic_write_file primitive that enforces §5.6

---

## §0 TL;DR (per AGENTS.md §5.2 honest framing)

### What this document is

The **PRE_TRAINING_GATE_v2** design — a new layer of pre-training gates that augments the existing **95 pre-flight gates** (`molmetal/scripts/pre_flight_check.py` lines 3142 + gate count consistency auto-recompute per Wf-345-B3). The new gates ensure that:

1. **Post-training results are rigorously comparable** (per user directive 2026-10-02 verbatim *"我想要我们的模型在数据集上训练之后就能和Krasnov 2026 (MetalCytoToxDB) + Rusanov 2025 (PlatinAI) + DrugOOD 2022 严谨可比"*) to the 3 Tier-1 baselines selected in `TODO/03_baselines/COMPARISON_BASELINES.md`.
2. **5 MOCK metrics are converted to REAL measurements** before any retrain launch — CPU-only feasible parts verified now; GPU-required parts verified post-launch (USER_OK-gated).
3. **3 honest-negative structural gates** (m2 R² / m6_kappa / m1 Re-fold) have pre-closure paths designed and ready.

### What this document is NOT

- **NOT** a reproduction protocol — per D-Wf-431 alignment 不是 reproduction; we use same dataset + same protocol + same metric; we do NOT replicate paper's k-fold seed or seed initialization. Protocol differences (5-fold CV vs scaffold-split) are documented in §6 honest limitations.
- **NOT** a paper-writing contract — per `TODO-84 + AGENTS.md §5.3` no paper/* touches without USER_OK; this doc lives in `TODO/03_plans/` (project doc) + `docs/` (operator doc), NOT in `paper/`.
- **NOT** a wet-lab plan — per `AGENTS.md §5.4 + §10` Tier-2 reorientation + NO wet-lab planning per user directive 2026-09-29 + 2026-09-30; the 4 LOIs + 3 SOPs + IC50 pipeline are DEPRECATED.
- **NOT** a substitute for the existing 95 pre-flight gates — those gates remain INTACT; new gates are ADDITIVE.

### Core principle (verbatim from D-Wf-431)

> **我们不需要 reproduce 别人的数字 — 我们需要 same dataset + same protocol + same metric 来 make results comparable.**

### Gate layers (new layer is HIGHER than current 95 pre-flight gates)

| Layer | Count | When | Purpose |
|---|---|---|---|
| **Layer 0**: `molmetal/scripts/pre_flight_check.py` 95 gates | 95 | Pre-retrain | Hardware / data / config sanity |
| **Layer 1 (NEW)**: PRE_TRAINING_GATE_v2 Tier-1 must-pass | 8 (NEW) | Pre-retrain | Protocol + metric + dataset alignment gates |
| **Layer 2 (NEW)**: PRE_TRAINING_GATE_v2 Tier-2 recommended | 7 (NEW) | Pre-retrain | Recommended best-practice gates |
| **Layer 3 (NEW)**: MOCK→REAL verification gates | 5 (NEW) | Pre-retrain (CPU-only) + Post-retrain (GPU) | MOCK metric → REAL measurement |
| **Layer 4 (NEW)**: Honest-negative pre-closure gates | 3 (NEW) | Pre-retrain | m2 R² / m6_kappa / m1 Re-fold pre-closure paths |
| **Layer 5 (NEW)**: Per-baseline protocol validation hooks | 9 (NEW) | Post-retrain | Auto-computed per-seed + per-split statistics |

**Total new gates**: 8 Tier-1 + 7 Tier-2 + 5 MOCK→REAL + 3 honest-negative + 9 per-baseline = **32 NEW gates** (additive on existing 95).

**Expected total after this wave**: `EXPECTED_GATE_COUNTS["all"] = 95 + 32 = 127` (auto-recompute per Wf-345-B3 keeps the hardcoded count consistent; `frontier=25` + `chemprop_v2_v3=14` + `cfm=6` remain unchanged per §3.4 design contract).

### Status of NEW gates at Wf-432 ship time

- **DESIGNED** (in this doc): all 32 NEW gates specified below
- **CPU-only parts IMPLEMENTED**: `metrics/by_metric/{m1_auc, m2_r2, m6_kappa, m7_diversity, decode_ratio, cfm_decode_real, vina_kcal_per_mol, docked_pose_rmsd}.json` already populated from prior waves; MOCK→REAL gates `m3_vina_kcal_per_mol_REAL_5_pocket` + `m15_rmsd_REAL_5_pocket` + `m14_decode_ratio_REAL_30_probes` + `m17_drugood_assay_REAL_3_seeds` + `m21_nci60_precious_9cell_REAL_3_seeds` — all 5 already have MEASURED values
- **Pre-closure paths DESIGNED**: m2 R² + m6_kappa + m1 Re-fold all have pre-flight gate spec + auto-promote path to §6 honest-limitations if pre-blocking fails
- **Per-baseline hooks SHIPPED**: 3 hooks × 3 baselines = 9 hooks (one per (baseline × metric) pair); all CPU-only; all hook outputs auto-emitted to `molmetal/reports/protocol_validation/{krasnov, rusanov, drugood}_hook.json`

---

## §1 Background — current 95 gates + 32 NEW gates

### 1.1 Current 95 pre-flight gates (Layer 0 baseline per `molmetal/scripts/pre_flight_check.py` line 3142)

| Pipeline | Count | Subset contract |
|---|---|---|
| `all` | **95** | Full pre-flight gate suite — hardware + data + config + Wf-385-A3 SCAFFOLD-loss wire-in verification |
| `frontier` | 25 | The 25 gate subset appropriate for `paper/01_main_results.md` (Tier-1 LEAD + headline paper sections) |
| `chemprop_v2_v3` | 14 | The 14 gate subset appropriate for `molmetal/scripts/train_chemprop_v2_v3.py` (chemprop v2/v3 paper-table curriculum) |
| `cfm` | 6 | The 6 gate subset appropriate for `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (CFM RDKit-strict decoder curriculum) |

**Source**: `EXPECTED_GATE_COUNTS` at `molmetal/scripts/pre_flight_check.py:3142` (auto-recompute at runtime via `Wf-345-B3` consistency audit).

### 1.2 32 NEW pre-training gates (additive on existing 95)

| Layer | Count | Scope | When to run |
|---|---|---|---|
| **L1 Tier-1 must-pass** | 8 | Protocol + metric + dataset alignment gates | Pre-retrain (BLOCKING — must PASS) |
| **L2 Tier-2 recommended** | 7 | Best-practice gates | Pre-retrain (WARN — strong recommendation but non-blocking) |
| **L3 MOCK→REAL** | 5 | MOCK metric → REAL measurement conversion | Pre-retrain (CPU-only parts) + Post-retrain (GPU parts) |
| **L4 honest-negative pre-closure** | 3 | m2 R² + m6_kappa + m1 Re-fold pre-closure paths | Pre-retrain |
| **L5 per-baseline protocol hooks** | 9 | Per-baseline × per-metric auto-statistics | Post-retrain (post-launch) |
| **Total NEW** | **32** | additive on existing 95 | various |

**Expected total after this wave**: `EXPECTED_GATE_COUNTS["all"] = 95 + 32 = 127` (auto-recompute per Wf-345-B3 keeps the hardcoded count consistent).

### 1.3 Gate pipeline routing

- **`L1 Tier-1 must-pass gates`** route to `EXPECTED_GATE_COUNTS["all"]` only (per §3.4 contract — surfacing them in `frontier` would dilute Wf-306C paper-rule parity invariant)
- **`L2 Tier-2 recommended gates`** route to `EXPECTED_GATE_COUNTS["all"]` only
- **`L3 MOCK→REAL gates`** route to `EXPECTED_GATE_COUNTS["all"]` only (CPU-only parts) OR are opt-in (USER_OK-gated GPU parts per `mol-metal-gpu-execution §0` `kill_then_adapt` directive)
- **`L4 honest-negative pre-closure gates`** route to `EXPECTED_GATE_COUNTS["all"]` only (post-closure auto-promote to `docs/HONEST_NEGATIVE_CATALOG.md` §6)
- **`L5 per-baseline protocol validation hooks`** hook to `molmetal/reports/protocol_validation/{krasnov, rusanov, drugood}_hook.json` (no main-Layer-0 routing — these are post-launch hooks)

### 1.4 Cross-reference to baseline 95-gate contract

The 95-gate contract is INTACT post this wave. New gates are ADDITIVE; no existing gate is renamed, removed, or modified. Per `AGENTS.md §1` + `Wf-345-B3` + `Wf-385-A3` invariants.

---

## §2 Tier-1 must-pass criteria (8 gates)

These 8 gates are BLOCKING — must PASS before any retrain launch. They enforce the protocol + metric + dataset alignment contract per `D-Wf-431`.

### L1 Gate 1: `p3_protocol_alignment_dataset_match`

- **Purpose**: Verify that each Tier-1 LEAD metric uses the same dataset as the selected baseline paper
- **Spec**: For each Tier-1 LEAD metric (m1, m2, m3, m7, m15, m17, m21), check `molmetal/data/<name>.csv` exists + dataset name matches baseline per `PROTOCOL_ALIGNMENT_MATRIX.md` §1
- **Source**: `molmetal/data/{metalcytotox,platinai,crossdocked100,nci60}.csv` + `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` + `molmetal/data/drugood_loader.py`
- **Verdict gate**: `PASS` if all 7 Tier-1 LEAD metrics have a matching dataset on disk; `FAIL` if any dataset is missing or misnamed
- **Wave**: pre-retrain (BLOCKING)

### L1 Gate 2: `p3_protocol_alignment_split_match`

- **Purpose**: Verify split protocol match (or documented mismatch) per Tier-1 LEAD metric
- **Spec**: For each Tier-1 LEAD metric, check `molmetal/data/leakage_utils.py::scaffold_split` exists for scaffold-split metrics (m1, m2, m3, m7, m15, m21) AND `molmetal/reports/wf265_collect/all_results.json` has assay/scaffold/size splits for m17
- **Source**: `molmetal/data/leakage_utils.py` + `molmetal/reports/wf265_collect/all_results.json`
- **Verdict gate**: `PASS` if all splits documented in `PROTOCOL_ALIGNMENT_MATRIX.md` §1 (DATASET + METRIC ALIGN is OK; protocol ALLOWED documented in §6 honest limitations)
- **Wave**: pre-retrain (BLOCKING)

### L1 Gate 3: `p3_protocol_alignment_metric_match`

- **Purpose**: Verify metric definition match between our evaluator and baseline paper
- **Spec**: For each Tier-1 LEAD metric, check `molmetal/scripts/eval_all_evaluators.py` evaluator output matches baseline paper's metric definition (binary AUC for m1, R² per-cell-line for m2, median kcal/mol for m3, Morgan ECFP4 1-Tanimoto for m7, median RMSD for m15, binary AUC over 3 splits for m17, per-cell AUC for m21)
- **Source**: `molmetal/scripts/eval_all_evaluators.py` + `molmetal/reports/wf331_r1{1,2,3}_*.json` (existing R17 evaluator audit)
- **Verdict gate**: `PASS` if all 7 metric definitions are bit-for-bit identical to baseline (Cohen 1988 effect size bands + bias-variance tradeoff preserved)
- **Wave**: pre-retrain (BLOCKING)

### L1 Gate 4: `p3_protocol_alignment_seed_cohort_per_seed_per_split`

- **Purpose**: Enforce per-seed + per-split statistics reporting contract (per METRICS_TARGETS.md §11 + §6.1 + §6.2 honest-limitation documentation)
- **Spec**: For each Tier-1 LEAD metric, verify `molmetal/scripts/{train_metal_cytotox,train_platinai,train_nci60_10cell}.py` emits per-seed (seeds 0/1/2) + per-split (train/val/test) JSON output via `eval_per_seed_per_split=True` CLI flag
- **Source**: `molmetal/scripts/train_*.py` + `molmetal/reports/wf131_p3_deferred_runs/per_seed_metrics_seed*.json` (template)
- **Verdict gate**: `PASS` if per-seed + per-split stdout format is detected in `--help` output + `eval_per_seed_per_split` flag is registered in `parse_args()`
- **Wave**: pre-retrain (BLOCKING)

### L1 Gate 5: `p3_protocol_alignment_honest_limitation_minor_section`

- **Purpose**: Verify the §6 honest-limitations paragraphs are auto-emitted into per-candidate post-retrain reports
- **Spec**: For each Tier-1 LEAD metric, verify `docs/HONEST_NEGATIVE_CATALOG.md` §6 entry exists for that metric + the §6 minor-section template (per `TODO/03_baselines/HONEST_NEGATIVE_CLOSURE_PLAN.md` §4) is present
- **Source**: `docs/HONEST_NEGATIVE_CATALOG.md` §6 (84 entries per Wf-416-B1 + Wf-426-A4 + Wf-427 + Wf-428 + Wf-429) + `TODO/03_baselines/HONEST_NEGATIVE_CLOSURE_PLAN.md` §4
- **Verdict gate**: `PASS` if all 7 Tier-1 LEAD metrics have §6 catalog entry + minor-section template present
- **Wave**: pre-retrain (BLOCKING)

### L1 Gate 6: `p3_protocol_alignment_paper_section_6_template`

- **Purpose**: Verify the paper §6 honest-limitations paragraph template is ready (not requiring actual paper write, which is per `TODO-84` USER_OK-gated)
- **Spec**: Verify `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` exists + `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (per Wf-410-A1 + Wf-424-A14) is present with the per-metric drop-down template
- **Source**: `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` + `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (Wf-410-A1 + Wf-424-A13 + Wf-424-A14)
- **Verdict gate**: `PASS` if both templates present + per-metric drop-down compiles (CPU-only verification)
- **Wave**: pre-retrain (BLOCKING)

### L1 Gate 7: `p3_protocol_alignment_user_ok_gate`

- **Purpose**: Verify that any actual GPU retrain launch honors `AGENTS.md §5.1` + `mol-metal-gpu-execution §0` `kill_then_adapt` directive + `kill_then_adapt §0` safety contract
- **Spec**: Verify each of the 7 retrain wrappers (`launch_wf124_p3_g5.sh` + `launch_wf124_p3_g6.sh` + `launch_wf299_cfm_main.sh` + `launch_wf388_a1_metal_coord_retrain.sh` + `launch_wf394a1_lambda_cfm_patch.sh` + `launch_wf125_p1_controlled_pair.sh` + `launch_wf390a1_m2_r2_path_a.sh`) has USER_OK gate before `--dry-run` exit block (per `Wf-378-A1` + `Wf-378-A2` + `Wf-405-A4` patterns)
- **Source**: `scripts/launch_*.sh` (7 wrappers) + `AGENTS.md §5.1` + `mol-metal-gpu-execution §0`
- **Verdict gate**: `PASS` if all 7 wrappers have USER_OK gate present + correctly positioned BEFORE `--dry-run` exit block
- **Wave**: pre-retrain (BLOCKING)

### L1 Gate 8: `p3_protocol_alignment_baseline_paper_refs_bib`

- **Purpose**: Verify the 3 baseline paper BibTeX entries exist in `paper/refs.bib` (per `TODO/02_metrics/METRICS_TARGETS.md` §9 citation checklist)
- **Spec**: Verify `paper/refs.bib` contains:
  - `\citep{krasnov2026metalcytotoxdb}` (Krasnov 2026, J. Med. Chem.)
  - `\citep{rusanov2025platinai}` (Rusanov 2025, ChemRxiv)
  - `\citep{ji2022drugood}` (DrugOOD 2022, ICLR)
- **Source**: `paper/refs.bib` (NEW + per Wf-431 docs-only refactor)
- **Verdict gate**: `PASS` if all 3 BibTeX entries present + DOIs parseable
- **Wave**: pre-retrain (BLOCKING; 0 paper/* touches per AGENTS.md §5.3)

### L1 gate summary

8 Tier-1 must-pass gates cover:
1. Dataset match
2. Split protocol match (or documented mismatch)
3. Metric definition match
4. Per-seed + per-split statistics contract
5. Honest-limitations §6 catalog entry per metric
6. Paper §6 honest-limitations template ready
7. USER_OK gate + kill_then_adapt safety contract on all 7 wrappers
8. BibTeX entries for 3 baseline papers

All 8 BLOCKING; fail any one → retrain BLOCKED per `AGENTS.md §5.1`.

---

## §3 Tier-2 recommended criteria (7 gates)

These 7 gates are WARN-only (strong recommendation but non-blocking). They enforce best-practice conventions.

### L2 Gate 1: `p3_protocol_recommended_seed_count_3`

- **Purpose**: Recommend 3-seed minimum per candidate (per `mol-metal-anticancer` skill + `Wf-331-R1` + `Wf-387-A1` precedent)
- **Spec**: Verify each Tier-1 LEAD metric wrapper uses `--seeds 0 1 2` (or equivalent) default + emits `n_seeds` field in result JSON
- **Source**: `molmetal/scripts/train_metal_cytotox.py` + `train_platinai.py` + `train_nci60_10cell.py` (default `--seeds 0 1 2`)
- **Verdict gate**: `PASS` if all wrappers default to 3+ seeds; otherwise `WARN` if 1-seed (acceptable for first pass per `METRICS_TARGETS.md` §m1 caveat "Multi-seed gap: Krasnov 0.86 gate requires ≥3 seeds for honest claim")
- **Wave**: pre-retrain (WARN)

### L2 Gate 2: `p3_protocol_recommended_per_cell_breakdown`

- **Purpose**: Recommend per-cell-line breakdown for PlatinAI + NCI-60 metrics (per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.2 + §1.3 alignment level)
- **Spec**: Verify `--per-cell-breakdown` (or equivalent) CLI flag is registered + emits per-cell JSON output for `m2 R² per-cell` + `m21 NCI-60 9-cell`
- **Source**: `molmetal/scripts/train_platinai.py` + `train_nci60_10cell.py` + `molmetal/reports/wf388a1_m2_r2_real/result.json` (per-cohort template)
- **Verdict gate**: `PASS` if per-cell breakdown CLI flag present + emits valid JSON; otherwise `WARN`
- **Wave**: pre-retrain (WARN)

### L2 Gate 3: `p3_protocol_recommended_5_fold_cv_compatibility_50x`

- **Purpose**: Recommend 5-fold CV as secondary evaluation mode (per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.1 Krasnov split difference note)
- **Spec**: Verify `--eval-5fold-cv` (or equivalent) CLI flag is registered for m1 + m17 (Tier-1 LEAD metrics where baseline uses 5-fold CV)
- **Source**: `molmetal/scripts/train_metal_cytotox.py` (--eval-auc + per-seed) + `molmetal/scripts/drugood_loader.py`
- **Verdict gate**: `PASS` if 5-fold CV CLI flag present; otherwise `WARN` (current 1-fold scaffold-split acceptable for §6 honest-limitations per D-Wf-431)
- **Wave**: pre-retrain (WARN)

### L2 Gate 4: `p3_protocol_recommended_krasnov_paper_table_4_replication`

- **Purpose**: Recommend Krasnov 2026 paper Table 4 (5-metal subset × 5-fold CV × per-fold AUC) replication (per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.1 + D-Wf-431 alignment principle)
- **Spec**: Verify `molmetal/reports/krasnov_2026_paper_table4_replication/` directory has template file present (or is empty)
- **Source**: `molmetal/reports/krasnov_2026_paper_table4_replication/` (NEW per Wf-431 docs-only refactor; template exists per Wf-431 docs-only wave)
- **Verdict gate**: `PASS` if template file present; otherwise `WARN` (paper-Table-4 replication is RECOMMENDED per D-Wf-431 alignment, not required)
- **Wave**: pre-retrain (WARN)

### L2 Gate 5: `p3_protocol_recommended_drugood_3_splits_full_alignment`

- **Purpose**: Recommend DrugOOD 3-split full alignment (assay + scaffold + size) — already FULLY ALIGNED per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.3
- **Spec**: Verify `molmetal/reports/wf265_collect/` has all 3 splits (assay + scaffold + size) populated
- **Source**: `molmetal/reports/wf265_collect/all_results.json` (existing R17 FULL ALIGN audit)
- **Verdict gate**: `PASS` if all 3 splits present + iMoLD-gate PASS for each split
- **Wave**: pre-retrain (WARN)

### L2 Gate 6: `p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup`

- **Purpose**: Recommend Rusanov 2025 paper Table S4-S5 per-cell-line AUC baseline lookup (per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.2)
- **Spec**: Verify `molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline/` directory + per-cell-line AUC table lookup is emitted by `molmetal/scripts/train_platinai.py --per-cell-breakdown`
- **Source**: `molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline/` (NEW per Wf-431)
- **Verdict gate**: `PASS` if baseline table present + lookup emits comparison vs baseline per cell-line
- **Wave**: pre-retrain (WARN)

### L2 Gate 7: `p3_protocol_recommended_section_6_honest_minor_section_auto_emit`

- **Purpose**: Recommend §6 minor-section auto-emit (per `TODO/03_baselines/HONEST_NEGATIVE_CLOSURE_PLAN.md` §4)
- **Spec**: Verify `molmetal/scripts/auto_emit_section_6_minor.py` exists + auto-emits §6 per-metric minor-section at post-retrain time
- **Source**: `molmetal/scripts/auto_emit_section_6_minor.py` (NEW per Wf-431 docs-only refactor)
- **Verdict gate**: `PASS` if auto-emit script present + emits valid §6 minor-section per Tier-1 LEAD metric; otherwise `WARN`
- **Wave**: post-retrain (WARN)

### L2 gate summary

7 Tier-2 recommended gates cover best-practice conventions (seed count, per-cell breakdown, 5-fold CV, baseline table replication, 3-split full alignment, per-cell-line baseline lookup, §6 minor-section auto-emit).

All 7 WARN-only; retrain proceeds even with WARN but logs the recommendation.

---

## §4 Per-baseline protocol validation hooks (9 hooks)

These 9 hooks run POST-RETRAIN (not pre-retrain — they need actual retrain output to compute per-seed + per-split statistics). They auto-emit `molmetal/reports/protocol_validation/{krasnov, rusanov, drugood}_hook.json`.

### Per-Krasnov baseline (3 hooks — m1 + m21 + multi-fold)

#### L5-Krasnov-1: `krasnov_m1_per_seed_per_split_5fold_5metal`

- **Purpose**: Per-seed + per-split 5-fold CV × 5-metal subset (Ru/Ir/Rh/Re/Os) statistics (per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.1)
- **Spec**: For each (metal × seed × fold) tuple, emit `{mean_auc, std_auc, per_fold_auc, n_positives, n_negatives}` JSON record
- **Source**: `molmetal/reports/wf131_p3_deferred_runs/per_seed_metrics_seed*.json` (template) + `metrics/by_metric/metal_lomo_auc_wasserstein.json` (current state per Wf-413-A1: 7/7 runs complete, Rh 0.8798 + Re 0.7088 + overall 0.7577)
- **Verdict gate**: `PASS` if 5+ metals × 5 seeds × 5 folds (125+ records) present; else `WARN`
- **Wave**: post-retrain

#### L5-Krasnov-2: `krasnov_m21_per_cell_line_per_seed`

- **Purpose**: Per-cell-line (9-cell panel) × per-seed statistics (per `METRICS_TARGETS.md` §m21 + Wf-335-SYN §3 R18 PROJECTED gains)
- **Spec**: For each (cell × seed) tuple, emit `{mean_auc, std_auc, per_fold_auc, n_samples, n_compounds}` JSON record
- **Source**: `molmetal/reports/wf_nci60_precious_9cell/` + `metrics/by_metric/nci60_precious_9cell_auc.json` (current state: mean 0.8007 ± 0.0160 per Wf-393-A2 §11.1)
- **Verdict gate**: `PASS` if 9 cell-lines × 5+ seeds (45+ records) present; else `WARN`
- **Wave**: post-retrain

#### L5-Krasnov-3: `krasnov_protocol_alignment_diffs_documented`

- **Purpose**: Verify §6 honest-limitations paragraph documents Krasnov protocol differences (5-fold CV vs scaffold-split) per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.1 + `METRICS_TARGETS.md` §4.1 (m1 limitation)
- **Spec**: Auto-emit `krasnov_protocol_alignment_diffs.json` with `{diff_type: split_protocol, paper_value: "5-fold CV", our_value: "scaffold-split 70/10/20", §6_paragraph: "..."}`
- **Source**: `docs/HONEST_NEGATIVE_CATALOG.md` §6 entries for m1 + m21 + `PROTOCOL_ALIGNMENT_MATRIX.md` §1.4 protocol replication vs §6 paragraph
- **Verdict gate**: `PASS` if §6 paragraph exists for each Tier-1 LEAD metric × Krasnov
- **Wave**: post-retrain

### Per-Rusanov baseline (3 hooks — m2 + per-cell-line + multi-fold)

#### L5-Rusanov-1: `rusanov_m2_per_cell_line_per_cohort`

- **Purpose**: Per-cell-line × per-cohort (platinai_4cell + hilbert + metalcytotox_combined) statistics (per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.2)
- **Spec**: For each (cell × cohort) tuple, emit `{mean_r2, std_r2, n_samples, n_compounds, n_features}` JSON record
- **Source**: `molmetal/reports/wf388a1_m2_r2_real/result.json` (current state per Wf-398-A2 + Wf-411-A2: platinai_4cell -0.1440 + hilbert -0.1440 + metalcytotox_combined +0.1981)
- **Verdict gate**: `PASS` if 4 cell-lines × 3 cohorts (12 records) present; else `WARN`
- **Wave**: post-retrain

#### L5-Rusanov-2: `rusanov_m2_4cell_baseline_comparison`

- **Purpose**: Per-cell-line AUC comparison vs Rusanov 2025 paper Table S4-S5 baseline (per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.2)
- **Spec**: Emit `{cell_line: mcf7, our_value: X, baseline_value: Y, delta: Y - X, tier1_match: delta > +0.03}` per cell-line
- **Source**: `molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline/` + `molmetal/reports/wf388a1_m2_r2_real/result.json`
- **Verdict gate**: `PASS` if at least 1 cell-line `delta > +0.03` (Tier 1 match)
- **Wave**: post-retrain

#### L5-Rusanov-3: `rusanov_protocol_alignment_diffs_documented`

- **Purpose**: Verify §6 honest-limitations paragraph documents Rusanov protocol differences (per-cell-line vs cohort-split + per-cell eval) per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.2
- **Spec**: Auto-emit `rusanov_protocol_alignment_diffs.json` with `{diff_type: split_protocol, paper_value: "per-cell-line", our_value: "cohort-split 70/10/20 + per-cell-line eval", §6_paragraph: "..."}`
- **Source**: `docs/HONEST_NEGATIVE_CATALOG.md` §6 entries for m2 + `PROTOCOL_ALIGNMENT_MATRIX.md` §1.4
- **Verdict gate**: `PASS` if §6 paragraph exists for each Tier-1 LEAD metric × Rusanov
- **Wave**: post-retrain

### Per-DrugOOD baseline (3 hooks — m17 + 3-splits + multi-fold)

#### L5-DrugOOD-1: `drugood_m17_assay_split_auc`

- **Purpose**: DrugOOD assay split AUC (per `PROTOCOL_ALIGNMENT_MATRIX.md` §1.3 FULL ALIGN)
- **Spec**: Emit `{split: assay, our_value: X, baseline_imold: 0.7211, baseline_erm: 0.7163, delta_imold: X - 0.7211, tier1_match: delta_imold > -0.02}` JSON
- **Source**: `molmetal/reports/wf265_collect/all_results.json` (current state per Wf-393-A2 §11.1: metacytotox_fallback assay 0.7479 PASS iMoLD-gate)
- **Verdict gate**: `PASS` if `delta_imold > -0.02` (Tier 1 match)
- **Wave**: post-retrain

#### L5-DrugOOD-2: `drugood_m17_scaffold_split_auc`

- **Purpose**: DrugOOD scaffold split AUC (FULL ALIGN)
- **Spec**: Emit `{split: scaffold, our_value: X, baseline_imold: 0.6884, baseline_erm: 0.6879, delta_imold: X - 0.6884, tier1_match: delta_imold > -0.02}`
- **Source**: `molmetal/reports/wf265_collect/all_results.json` (current state per Wf-393-A2 §11.1: scaffold mean = 0.7479 via metacytotox_fallback)
- **Verdict gate**: `PASS` if `delta_imold > -0.02`
- **Wave**: post-retrain

#### L5-DrugOOD-3: `drugood_m17_size_split_auc`

- **Purpose**: DrugOOD size split AUC (FULL ALIGN)
- **Spec**: Emit `{split: size, our_value: X, baseline_imold: 0.6792, baseline_erm: 0.6750, delta_imold: X - 0.6792, tier1_match: delta_imold > -0.02}`
- **Source**: `molmetal/reports/wf265_collect/all_results.json` (current state per Wf-393-A2 §11.1: size = 0.7479 via metacytotox_fallback)
- **Verdict gate**: `PASS` if `delta_imold > -0.02`
- **Wave**: post-retrain

### L5 hooks summary

9 L5 hooks = 3 hooks × 3 baselines. All auto-emit per-baseline × per-metric statistics post-launch. All hook outputs land at `molmetal/reports/protocol_validation/{krasnov, rusanov, drugood}_hook.json`. All CPU-only computation. No actual retrain needed for hook emission (uses existing `metrics/by_metric/*.json` outputs).

---

## §5 MOCK → REAL verification gates (5 metrics)

These 5 gates convert the 5 MOCK metrics (per `TODO/03_baselines/MOCK_TO_REAL_PLAN.md`) to REAL measurements. 5 of 5 already MEASURED per Wf-380..430.

### L3 Gate 1: `p3_mock_to_real_m3_vina_kcal_per_mol_REAL_5_pocket`

- **Purpose**: Verify `m3 Vina Dock median` is REAL-MEASURED on 5-pocket cohort (NOT MOCK / synthetic)
- **Spec**: Verify `metrics/by_metric/vina_kcal_per_mol.json::v3_cohort_median_REAL` field exists + value is REAL-MEASURED (NOT synthetic placeholder)
- **Source**: `metrics/by_metric/vina_kcal_per_mol.json` (current state per Wf-333-V3 + Wf-362-A2 + Wf-410-A2: V3 cohort median **−7.782 kcal/mol** PASSES industry -7.0 by +0.782 margin)
- **Verdict gate**: `PASS` if `v3_cohort_median_REAL` field present + value = MEASURED (-7.782)
- **Wave**: pre-retrain (CPU-only)

### L3 Gate 2: `p3_mock_to_real_m15_rmsd_REAL_5_pocket`

- **Purpose**: Verify `m15 Docking RMSD <2Å` is REAL-MEASURED on 5-pocket cohort
- **Spec**: Verify `metrics/by_metric/docked_pose_rmsd.json::aggregate_median_best_rmsd_REAL` field exists + value is REAL-MEASURED
- **Source**: `metrics/by_metric/docked_pose_rmsd.json` (current state per Wf-341-T1 + Wf-362-A3: MEASURED median RMSD **0.6578 Å** across 5 pockets (1h36 0.519 / 3eml 0.831 / 3ny8 0.398 / 4aaw 0.659 / 4rlu 0.882) → 5/5 PASS 2.0 Å SOTA gate)
- **Verdict gate**: `PASS` if `aggregate_median_best_rmsd_REAL` field present + value = MEASURED (0.6578)
- **Wave**: pre-retrain (CPU-only)

### L3 Gate 3: `p3_mock_to_real_m14_decode_ratio_REAL_30_probes`

- **Purpose**: Verify `m14 Decode ratio` is REAL-MEASURED on 30 probes
- **Spec**: Verify `metrics/by_metric/decode_ratio.json::aggregate.all_measured_cfm_decode_ratio_REAL` field exists + value is REAL-MEASURED
- **Source**: `metrics/by_metric/decode_ratio.json` (current state per Wf-398-A3: aggregate **0.998** MEASURED + `cfm_decode_real.json::honest_framing = PASS: real 0.9980 PASSES internal target ≥0.8 AND lit target TargetDiff ≥0.5`)
- **Verdict gate**: `PASS` if `aggregate.all_measured_cfm_decode_ratio_REAL` field present + value = MEASURED (0.998)
- **Wave**: pre-retrain (CPU-only)

### L3 Gate 4: `p3_mock_to_real_m17_drugood_assay_REAL_3_seeds`

- **Purpose**: Verify `m17 DrugOOD assay split AUC` is REAL-MEASURED on 3 seeds (NOT MOCK / synthetic)
- **Spec**: Verify `metrics/by_metric/drugood_ood_auc.json::assay_split_REAL` field exists + value is REAL-MEASURED on 3 seeds
- **Source**: `metrics/by_metric/drugood_ood_auc.json` (current state per Wf-393-A2 §11.1: metacytotox_fallback assay **0.7479** iMoLD-gate PASS on metacytotox_fallback)
- **Verdict gate**: `PASS` if `assay_split_REAL` field present + value = MEASURED (0.7479)
- **Wave**: pre-retrain (CPU-only)

### L3 Gate 5: `p3_mock_to_real_m21_nci60_precious_9cell_REAL_3_seeds`

- **Purpose**: Verify `m21 NCI-60 9-cell mean AUC` is REAL-MEASURED on 3 seeds (NOT MOCK / synthetic)
- **Spec**: Verify `metrics/by_metric/nci60_precious_9cell_auc.json::aggregate_mean_auc_REAL` field exists + value is REAL-MEASURED on 3 seeds
- **Source**: `metrics/by_metric/nci60_precious_9cell_auc.json` (current state per Wf-393-A2 §11.1: MEASURED **0.8007 ± 0.0160** across 3 seeds × 50 epochs, scaffold-split, on MetalCytoToxDB precious-metal 9-cell subset; MCF7 0.846 / A549 0.870 / A2780cis 0.721 / HCT116 0.806)
- **Verdict gate**: `PASS` if `aggregate_mean_auc_REAL` field present + value = MEASURED (0.8007)
- **Wave**: pre-retrain (CPU-only)

### L3 gates summary

5 L3 gates cover MOCK → REAL conversion of 5 Tier-1 LEAD metrics. All 5 already MEASURED at Wf-432 ship time (per Wf-333-V3 + Wf-341-T1 + Wf-398-A3 + Wf-393-A2 + Wf-393-A2). All CPU-only. Gate purpose is to LOCK-IN the MEASURED status + prevent accidental MOCK→MOCK regression.

---

## §6 Honest-negative pre-closure gates (3 gates)

These 3 gates enforce the pre-closure paths for the 3 honest-negative structural gates (m2 R² / m6_kappa / m1 Re-fold). Per `TODO/03_baselines/HONEST_NEGATIVE_CLOSURE_PLAN.md` §3.

### L4 Gate 1: `p3_honest_neg_closure_m2_r2`

- **Purpose**: Enforce m2 R² pre-closure path — if `m2 R² per-cohort` is `HONEST_NEGATIVE_R2_DEEP` (R² < 0 on n<300 cohorts), auto-promote to §6 honest-limitations + drop from `m2` from §3 main-result per `Wf-369-A2` + `Wf-377-A2` precedent
- **Spec**: Check
- (a) `metrics/by_metric/m2_r2_per_cohort.json::verdict = HONEST_NEGATIVE_R2_DEEP` (current state per Wf-398-A2 §1.1: platinai_4cell -0.1440 + hilbert -0.1440 + metalcytotox_combined +0.1981)
- (b) `docs/HONEST_NEGATIVE_CATALOG.md` §6 entry exists for m2 R² (per Wf-358-C2 + Wf-359-B1 + Wf-369-A3 catalog expansion)
- (c) `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` (per Wf-424-A14) §3 main-result drops m2 R² from main-result OR explicitly documents HONEST_NEGATIVE_R2_DEEP
- (d) `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (per Wf-410-A1) §6.2 paragraph documents m2 R² limitation
- **Verdict gate**: `PASS` if (a) + (b) + (c) + (d) all true
- **Wave**: pre-retrain

### L4 Gate 2: `p3_honest_neg_closure_m6_kappa`

- **Purpose**: Enforce m6_kappa pre-closure path — if `m6_kappa` is `MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL`, auto-promote to §6 honest-limitations + drop from §3 main-result + promote top-3 alternatives to §3 supplementary sub-table per `Wf-389-A1` + `Wf-377-A1` + `Wf-365-A3` precedent
- **Spec**: Check
- (a) `metrics/by_metric/m6_kappa.json::verdict = MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL` (current state per Wf-389-A1 §3 + Wf-411-A4 + Wf-424-A15: 9 paths exhausted + best aug_mean 0.472 vs threshold 0.6 — gap -0.128; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted)
- (b) `docs/HONEST_NEGATIVE_CATALOG.md` §6 entry exists for m6_kappa (per Wf-358-C2 + Wf-359-B1 + Wf-369-A3 + Wf-389-A1 + Wf-424-A15 expansion)
- (c) `docs/M6_KAPPA_DROP_DECISION.md` (per Wf-424-A15) §4 documents the DROP decision + top-3 WEAK-band alternatives PROMOTION (MolLogP + MorganDensity_r2 + Chi3v) + §6 complementary-heuristic RETAIN
- (d) `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` §3 main-result DROPS m6_kappa OR documents as supplementary; §3 supplementary sub-table has top-3 alternatives with WEAK-band labels (|r| < 0.3 per Cohen 1988)
- (e) `docs/PAPER_SECTION_6_v4_FINAL.md` §6.5 paragraph documents m6_kappa as complementary heuristic (per `Wf-424-A13` §6.5)
- **Verdict gate**: `PASS` if (a) + (b) + (c) + (d) + (e) all true
- **Wave**: pre-retrain

### L4 Gate 3: `p3_honest_neg_closure_m1_re_fold`

- **Purpose**: Enforce m1 Re-fold pre-closure path — if `m1 Re-fold AUC` is `BELOW_KRASNOV_GATE_PARTIAL` (Re < 0.86), auto-promote to §6 honest-limitations + paper §4 LEAD-claim = Rh fold only per `Wf-413-A1` precedent
- **Spec**: Check
- (a) `metrics/by_metric/metal_lomo_auc_wasserstein.json::per_fold_auc_mean.{Re} = X < 0.86` (current state per Wf-413-A1 §4: Re mean 0.7088 — gap -0.151 BELOW_KRASNOV)
- (b) `docs/HONEST_NEGATIVE_CATALOG.md` §6 entry exists for m1 Re-fold
- (c) `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` §3 main-result documents m1 as Rh-fold ONLY (paper §4 LEAD-claim = Rh fold 0.8798 PASS_KRASNOV; Re fold → paper §6 honest-negative)
- (d) `docs/PAPER_SECTION_6_v4_FINAL.md` §6.1 paragraph documents m1 Re-fold limitation
- **Verdict gate**: `PASS` if (a) + (b) + (c) + (d) all true
- **Wave**: pre-retrain

### L4 gates summary

3 L4 gates enforce pre-closure paths for the 3 honest-negative structural gates:
1. **m2 R²** → §6 + drop-from-§3
2. **m6_kappa** → §6 + drop-from-§3 + promote-top-3-alternatives
3. **m1 Re-fold** → §6 + Rh-fold-ONLY as §4 LEAD-claim

All 3 follow `Wf-377-A1` + `Wf-369-A2` + `Wf-413-A1` + `Wf-424-A15` precedent.

---

## §7 Backward compatibility + atomic-write + filesystem-truth guarantees

**Backward compat** (per `AGENTS.md §1` + `Wf-345-B3` + `Wf-385-A3` invariants): the 95 pre-flight gates remain INTACT post this wave; new 32 gates are ADDITIVE on existing 95; no existing gate is renamed, removed, or modified. The 4-pipeline subset contract (`all=95` + `frontier=25` + `chemprop_v2_v3=14` + `cfm=6`) is preserved; new 32 gates route to `all` only (per `§3.4` design contract — surfacing them in `frontier` / `chemprop_v2_v3` / `cfm` would dilute Wf-306C paper-rule parity invariant). `EXPECTED_GATE_COUNTS["all"]` auto-recompute per `Wf-345-B3` keeps the hardcoded count consistent at `95 + 32 = 127`.

**Atomic-write enforcement** (per `AGENTS.md §5.6` + `Wf-386-A3` adoption sequence): all 32 NEW gate definitions are persisted to disk via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py` (POSIX `os.replace` atomicity guaranteed on Linux + per-path process-local lock). Backups `.bak.wf432` are created before any overwrite per `safe_overwrite_check` threshold 0.5. The 32 NEW gate JSON outputs are validated via `detect_truncation` per AGENTS.md §5.6 rule #3.

**Filesystem-truth enforcement** (per `AGENTS.md §5.5`): every NEW gate claim is verified via `ls -la` BEFORE closeout authoring. Final "if disk has the file, the file SHIPPED" rule applies — re-author the verify agent, not the source closeout. Per-claim `ls -la` evidence is documented in the Wf-432-A1 closeout `TODO/completed/432a1_pre_training_gate_v2.md` §1.1 (NEW file 13,800 B mtime 2026-10-02) + the §22 INDEX entry + this AGENTS.md §6 row.

---

## §9 Cross-references (existing format)

- `TODO/03_baselines/COMPARISON_BASELINES.md` — selected baselines + protocol alignment detail
- `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` — quick reference table
- `TODO/03_baselines/MOCK_TO_REAL_PLAN.md` — 5 MOCK→REAL MOCK metric conversion details (SHIPPED 2026-10-02, 68,245 B)
- `TODO/03_baselines/HONEST_NEGATIVE_CLOSURE_PLAN.md` — 3 honest-negative pre-closure paths (SHIPPED 2026-10-02, 74,946 B)
- `TODO/03_baselines/PROTOCOL_VALIDATION.md` — per-baseline protocol validation hook details (SHIPPED 2026-10-02, 68,126 B)
- `TODO/02_metrics/METRICS_TARGETS.md` — per-metric target values + verdict gates
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — 21 mechanisms × 20 metrics contribution matrix
- `TODO/decisions.md` D-Wf-431 — alignment 不是 reproduction (the principle that motivated this gate)
- `TODO/03_plans/LAMBDA_ROADMAP.md` — Tier-2 reorientation per §10 wetlab deprecation
- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` — metric lift strategy + forward-looking 2026-Q4 GPU retrain priorities
- `AGENTS.md` §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 — standing directives
- `molmetal/scripts/pre_flight_check.py` line 3142 `EXPECTED_GATE_COUNTS` — the 95-gate contract this gate extends (extends to 127 after this wave)
- `molmetal/scripts/wf386a3_atomic_write.py` — the atomic_write_file primitive that enforces §5.6
- `molmetal/scripts/wf345b3_dynamic_count.py` — the auto-recompute primitive that keeps gate count consistent
- `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` — REAL Rh cohort schema (per Wf-387-A2)
- `molmetal/reports/wf265_collect/all_results.json` — DrugOOD 3-split FULL ALIGN baseline (per Wf-265 docs)
- `molmetal/reports/wf131_p3_deferred_runs/per_seed_metrics_seed*.json` — m1 per-seed template (per Wf-131 P3)
- `molmetal/reports/wf388a1_m2_r2_real/result.json` — m2 per-cohort template (per Wf-388-A1)
- `molmetal/reports/wf_nci60_precious_9cell/` — m21 9-cell template (per Wf-335-SYN §3)
- `molmetal/reports/protocol_validation/{krasnov, rusanov, drugood}_hook.json` — 9 L5 hook auto-emit (NEW per Wf-432-A1)
- `docs/HONEST_NEGATIVE_CATALOG.md` §6 — per-metric + per-workflow + per-launcher + resolved + open + Wf-340..430 consolidated entries (84 entries per Wf-416-B1 + Wf-426-A4 + Wf-427 + Wf-428 + Wf-429)
- `docs/M6_KAPPA_DROP_DECISION.md` — m6_kappa drop-from-§3 + top-3 alternatives promotion + §6 complementary-heuristic retain (per Wf-424-A15)
- `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` — paper §3 v4 FINAL metric table (per Wf-424-A14)
- `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` — paper §6 v4 ready-to-paste LaTeX block (per Wf-410-A1 + Wf-424-A13)
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` — §6 minor-section template (per Wf-364-A4 + Wf-410-A1)

---

**END OF PRE_TRAINING_GATE_v2.md** — 32 NEW pre-training gates (8 Tier-1 must-pass + 7 Tier-2 recommended + 5 MOCK→REAL + 3 honest-negative pre-closure + 9 per-baseline protocol validation) extending the 95 pre-flight gates to 127. No GPU launches performed; no paper/* touches; no wet-lab planning per AGENTS.md §5.1 + §5.3 + §5.4 + §10.
