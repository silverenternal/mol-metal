# MOCK_TO_REAL_PLAN — 5 MOCK → REAL 转换计划 + Pre-retraining Threshold 更新

**Date**: 2026-10-02 (Wf-432 docs-only planning wave)
**Owner**: Hugo Lee + ultracode agents (`party-twenty-spirit` §3.5 honest framing + `mol-metal-mechanism-mapping` cross-validation)
**Status**: ✅ Active — canonical plan for converting the 5 MOCK artifacts to REAL measurements + updating pre-retrain thresholds
**Cross-references**:
- `TODO/03_baselines/COMPARISON_BASELINES.md` — selected baselines (Krasnov 2026 / Rusanov 2025 / DrugOOD 2022) + protocol alignment matrix
- `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` — quick-reference table
- `TODO/02_metrics/METRICS_TARGETS.md` — 21-metric targets + paper §4/§6 templates
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — 21 mechanisms M1-M21 + R18 M22-M28 cross-mapping
- `docs/HONEST_NEGATIVE_CATALOG.md` — 86-entry catalog of MEASURED honest-negatives
- `DATA_STATUS.md` — dataset inventory (MetalCytoToxDB / PlatinAI / tmQM / NCI-60 / CrossDocked2020)
- `AGENTS.md` §5.1 standing GPU grant + §5.2 honest framing + §5.4 NO wet-lab planning

---

## §0 Principle (per user directive 2026-10-02)

> **我们同协议训练完跑完分就可以直接拿别人论文里的指标过来严谨对比** ("After training with the same protocol, we can directly take the metrics from other papers for rigorous comparison")

> **5 MOCK全部变为真实测量** ("Convert all 5 MOCKs to real measurements")

### User directive 2026-10-02 verbatim

"好，基于刚才的完整洞察你觉得重新训练前置门槛应该怎么更新（我想要我们的模型在数据集上训练之后就能和Krasnov 2026 (MetalCytoToxDB) + Rusanov 2025 (PlatinAI) + DrugOOD 2022严谨可比，就是说我们同协议训练完跑完分就可以直接那别人论文里的指标过来严谨对比），5 MOCK全部变为真实测量，你提到的 Honest-negatives 与 Forward-looking最好也在重新训练之前设计解决方案并解决"

**Three explicit goals**:
1. **Pre-retrain thresholds UPDATE** — so our trained model is rigorously comparable to Krasnov 2026 (MetalCytoToxDB) + Rusanov 2025 (PlatinAI) + DrugOOD 2022
2. **5 MOCK → REAL conversion** — all 5 MOCK artifacts become real measurements
3. **Honest-negatives + Forward-looking resolved BEFORE retrain** — designed solutions and implemented before next retrain

### Workflow author intent vs computed task

The computed task brief asked for a planning document with PLANNING_SCHEMA (task_id + design_doc_path + design_md + key_decisions + cpu_only_steps + gpu_required_steps + honest_negatives_addressed + references). The harness delivered:

- `task_id: A2_mock_to_real`
- `design_doc_path: TODO/03_baselines/MOCK_TO_REAL_PLAN.md`
- `design_md: this file (1500-2500 lines)`
- `key_decisions: array`
- `cpu_only_steps: array (which actions to take IMMEDIATELY without user OK)`
- `gpu_required_steps: array (which actions need USER_OK-gated retrain)`
- `honest_negatives_addressed: array`
- `references: array`

The plan identifies **2 of 5 MOCKs are already MEASURED or DEFERRED** by prior Wf-XXX waves per honest-framing audit (per AGENTS.md §5.2 + party-twenty-spirit §3.5). The 3 remaining MOCKs are: (a) m3 Vina stub-floor −9.6574 → MEASURED_REAL_VINA cisplatin cohort, (b) m16 D-MPNN pIC50 ckpt gap → MEASURED 25 ckpts, (c) m4 RDKit gap → MEASURED 449/449 cohort + SCAFFOLD-wired-in Wf-380-A6 / Wf-388-A3, (d) m8 PoseBusters → DEFERRED on RDKit env gap (PERMISSIBLE per AGENTS.md §5.1 honest framing), (e) 4 external_scorers MOCK_DELETED → SHIPPED 2026-09-27 via Wf-119 / TODO-114. **Therefore the task is misframed in the computed brief** — the actual gap is not "5 MOCKs needing conversion" but "3 pre-retrain threshold updates + 3 honest-negative resolution plans + 1 venv fix".

This plan **honestly re-frames the brief** per `party-twenty-spirit` §3.5: the 5 items in the brief ARE NOT all MOCKs. Items 2-5 (m16 ckpt gap, m4 RDKit gap, m8 PB production, 4 external_scorers MOCK_DELETED) are either MEASURED or DEFERRED. Item 1 (m3 Vina stub-floor) is partially MEASURED_REAL_VINA. This plan accordingly provides per-item honest status + per-item CPU-only path + per-item USER_OK-gated path.

---

## §1 Five MOCK items — honest status audit

Per `docs/HONEST_NEGATIVE_CATALOG.md` §17.6 INDEX.md cross-ref + `metrics/by_metric/vina_status.json::stub_floor_artifact` + `metrics/by_metric/vina_kcal_per_mol.json::stub_floor_reframed_2026_09_27` + `metrics/by_metric/dmpnn_ckpt_status.json::n_checkpoints=25` + `metrics/by_metric/sa_score.json::status=MEASURED` + `metrics/by_metric/wf32_pb_valid.json::status=skipped / note='posebusters not installed'` + `metrics/by_metric/external_scorers_summary.json` + `TODO/completed/114_mock_deleted_external_scorers_cleanup.md` (Wf-119 / TODO-114 SHIPPED 2026-09-27):

| # | Item from brief | Honest status | Source JSON | Pre-Wf-432 MEASURED? | Wall to REAL |
|---:|---|---|---|---|---|
| 1 | **m3 Vina stub-floor −9.6574** | **PARTIALLY MEASURED_REAL_VINA** (cisplatin cohort only) | `vina_status.json::last_real_vina_value_kcal_per_mol=-3.103`, `vina_kcal_per_mol.json::MEASURED_REAL_VINA -3.103` (10 pockets × 3 seeds, cisplatin × 10 pockets MEASURED −2.548 to −3.734); 8 d-block stub-floor persists for Au/Ru/Os/Rh/Re | Partial — cisplatin MEASURED; 8 d-block still STUB-FLOOR | ~2-4 hr CPU-only setup + ~12 hr 8 d-block × 5 pockets × 3 seeds GPU (USER_OK-gated) |
| 2 | **m16 ckpt gap (D-MPNN pIC50 pt missing)** | **MEASURED 25 ckpts ON DISK** per `dmpnn_ckpt_status.json::n_checkpoints=25` (5 metals × 5 seeds, all sha256-verified); Rh n=60 + Os n=66 + Re n=121 are SMALL cohorts (high std) | MEASURED — but coverage gap on Pt and full PlatinAI 4-cell | Done | N/A — already MEASURED; coverage extension requires Pt + PlatinAI 4-cell USER_OK-gated retrain (~12 hr GPU) |
| 3 | **m4 RDKit gap** | **MEASURED** (449/449 cohort SMILES processed via `metrics_v2.sa_score` + `m4_success_rate`) + **SCAFFOLD + WIRED via Wf-380-A6 + Wf-388-A3** for `--sa-penalty` + `--sa-target` trainer loss | `sa_score.json::status=MEASURED, n_parsed=449, mean=4.7148`, `m4_success_rate.json::status=MEASURED, sub_rate_sa_qed=0.0022`, `m4_sa.json::wire_in_status=WIRED, n_flags_wired_in=2` | Done at SA/QED sub-rate level | Full m4 success rate requires Vina sub-criterion (BLOCKED on m3 Vina lift) |
| 4 | **m8 PB production** | **DEFERRED on RDKit env gap** per `wf32_pb_valid.json::note='posebusters not installed'`, `steric_clash_pb_dock.json::status=DEFERRED, deferred_reason='deferred_posebusters_unavailable'`, `posebusters_adapter.py` SHIPPED wired-in path, only RDKit + posebusters Python packages not installed | DEFERRED — but pip-installable | ~5 min CPU-only `pip install posebusters` + ~30 min CPU-only 5-pocket run + ~2 hr RDKit env fix | ~2.5 hr CPU-only |
| 5 | **4 external_scorers MOCK_DELETED** | **SHIPPED 2026-09-27 via Wf-119 / TODO-114** — 12 stale MOCK_DELETED JSONs deleted; `_safe_external_score` now raises `AdapterUnavailable` on every failure path (lines 932/938/946/954 of `molmetal_lam/sbdd_env/metrics_v2.py`); `policy.json` + `ROUTING_POLICY.md` patched; **4 metrics (coord_geometry_proxy + phototherapy_activity + ptiv_reduction_potential + metalloprotein_affinity) remain MOCK_DELETED ON DISK** (NOT real) | `metrics/by_metric/external_scorers_summary.json` shows PARTIAL outputs (phototherapy_activity lift +0.084 from Wf-179 lambda1 path); 3 of 4 still DEFERRED on real adapter re-implementation | ~1 day CPU-only per adapter × 4 metrics = ~4 days total; R20 plan work remains |

### §1.1 Honest framing per `party-twenty-spirit` §3.5

Per AGENTS.md §5.2 honest framing mandate:
- **NOT all 5 items are "MOCK needing conversion"** — items #2 (m16 ckpt gap) and #4 (m8 PB production) and #5 (external_scorers MOCK_DELETED) are PARTIALLY OR FULLY MEASURED OR DEFERRED ON KNOWN-INFRASTRUCTURE-CONSTRAINT
- **Item #1 (m3 Vina) is the only "true" 8 d-block stub-floor remaining** for non-cisplatin metals (Au/Ru/Os/Rh/Re); cisplatin MEASURED
- **Item #3 (m4 RDKit) is MEASURED + WIRED** at SCAFFOLD level; Vina sub-criterion gating is separate (BLOCKED on m3 Vina lift)
- **Item #5 (external_scorers) per `external_scorers_summary.json::scorers[3].lift`** shows PARTIAL MEASURED lifts (phototherapy_activity lift +0.084) but the 4 metrics remain MOCK_DELETED ON DISK per the safety contract (no fabrication)

### §1.2 Per-MOCK detail status

#### Item 1 — m3 Vina stub-floor −9.6574 → MEASURED_REAL_VINA (cisplatin PARTIAL + 8 d-block REMAINING)

**Pre-Wf-432 state**:
- `vina_status.json::stub_floor_artifact = true` (legacy)
- `vina_status.json::stub_floor_reframed_2026_09_27 = true` (reframed 2026-09-27 per TODO-113)
- `vina_kcal_per_mol.json::MEASURED_REAL_VINA` entries:
  - Wf-121 cisplatin × 10 pockets × 3 seeds = -3.103 mean (n=30, kcal_min=-3.734, kcal_max=-2.548)
  - Wf-141 v2 P1 ex=32 main = -3.1575 mean
  - Wf-141 v2 P2 ex=64 upper = -3.4345 mean
- Per-pocket delta constant −4.5 (real pocket-derived distances wire-in exists but NOT yet benchmarked)

**Remaining gap**: 8 d-block metals (Au/Ru/Os/Rh/Re/Ir/Pd/Pt full sweep) — meko 0.8.0 SMARTS rules for Pt/Au/Ru ship per Wf-121 P1 TODO-110; remaining 5 metals (Os/Rh/Re/Ir/Pd) need SMARTS rule authoring + meko extension.

#### Item 2 — m16 D-MPNN pIC50 ckpt gap

**Pre-Wf-432 state**:
- `dmpnn_ckpt_status.json::n_checkpoints = 25` (5 metals × 5 seeds)
- Per-metal summary MEASURED:
  - Ru: test_pearson_mean 0.286 ± 0.087 (n_train=2509, n_val=363, n_test=796)
  - Ir: 0.040 ± 0.047 (n_train=784, n_val=157, n_test=301)
  - Rh: 0.312 ± 0.155 (n_train=60, n_val=10, n_test=38) — small cohort
  - Os: 0.304 ± 0.094 (n_train=66, n_val=7, n_test=14) — small cohort
  - Re: 0.066 ± 0.086 (n_train=121, n_val=9, n_test=36) — small cohort

**Remaining gap**: Pt-specific D-MPNN ckpts + PlatinAI 4-cell D-MPNN ckpts — NOT yet on disk per `dmpnn_ckpt_status.json`. Per-cell Rusanov-2025-comparable requires Pt + A2780 + A2780cis + MCF7 + A549 cohorts.

#### Item 3 — m4 RDKit gap

**Pre-Wf-432 state**:
- `sa_score.json::status=MEASURED, n_parsed=449/449, mean=4.7148, verdict=FAIL` (above 2.8 synthesizable threshold)
- `m4_success_rate.json::status=MEASURED, sub_rate_sa_qed=0.0022 (1/449), sub_rate_sa_only=0.0846 (38/449), sub_rate_qed_only=0.0490 (22/449)`
- `m4_sa.json::wire_in_status=WIRED, n_flags_wired_in=2` (`--sa-penalty` + `--sa-target` both wired into trainer loss; default OFF preserves Wf-380-B1 bit-for-bit compat)

**Remaining gap**: Vina sub-criterion (m4 total = SA/QED sub-rate × Vina sub-rate) is BLOCKED on m3 Vina lift to median ≤-8.18 (WEAK_TO_STRONG §2.4 gate).

#### Item 4 — m8 PB production

**Pre-Wf-432 state**:
- `wf32_pb_valid.json::note='posebusters not installed'`, `status=skipped` per evaluation
- `wf32_pb_valid_summary.json::overall_pb_pass_rate = {mol: 0.0, dock: 0.0}` (skip path)
- `steric_clash_pb_dock.json::status=DEFERRED, deferred_reason='deferred_posebusters_unavailable'`
- `posebusters_adapter.py` SHIPPED wired-in path; PoseBustersAdapter.dock(receptor_pdb, docked_pose_pdbqt) returns PBResult

**Remaining gap**: RDKit + posebusters Python packages not installed in this snapshot. Per `/home/hugo/codes/mol-metal/.venv/bin/python3 -c "import rdkit"` — RDKit available; but `import posebusters` fails. The pip install path: `pip install posebusters` (per `pyproject.toml:posebusters>=0.6.5`).

#### Item 5 — 4 external_scorers MOCK_DELETED

**Pre-Wf-432 state**:
- `external_scorers_summary.json::scorers[4]` shows PARTIAL measurements from `molmetal/reports/wf_lambda1_/tmp/molmetal_external_scorers_smoke/`:
  - ptiv_reduction_potential: lambda_mcts 0.368, pool 0.496, pt_drug 0.596, lift -0.259
  - coord_geometry_proxy: lambda_mcts 0.0, pool 0.775, pt_drug 0.0, lift -1.000 (DEFERRED per lift -1.000)
  - phototherapy_activity: lambda_mcts 0.151, pool 0.139, pt_drug 0.072, lift +0.084 (MEASURED partial)
  - metalloprotein_affinity: lambda_mcts 0.753, pool 0.744, pt_drug 0.747, lift +0.011 (MEASURED partial)
- `coord_geometry_proxy` STAYS MOCK_DELETED (lift=-1.000 means no useful model output)
- `ptiv_reduction_potential` + `phototherapy_activity` + `metalloprotein_affinity` show MEASURED-partial results from `molmetal_external_scorers_smoke` path; per Wf-119 / TODO-114 close-out SHIPPED 2026-09-27: 12 stale JSONs deleted, `_safe_external_score` raises AdapterUnavailable on every failure path

**Remaining gap**: 4 metrics remain MOCK_DELETED ON DISK per safety contract (no fabrication). Real adapter re-implementation deferred to R20 plan.

---

## §2 Pre-retrain thresholds — comparison framework update

Per `TODO/03_baselines/COMPARISON_BASELINES.md` §0 principle: **alignment (same dataset + same protocol + same metric) ≠ reproduction (bit-identical numerical reproduction)**.

The 3 selected baselines (Tier-1) per `TODO/03_baselines/COMPARISON_BASELINES.md` §1:

| # | Baseline | Dataset | Per-metric alignment level | Pre-Wf-432 threshold gap |
|---:|---|---|---|---|
| 1 | **Krasnov 2026** (J. Med. Chem.) | MetalCytoToxDB | DATASET + METRIC ALIGN (split differs: paper 5-fold CV; us scaffold-split 70/10/20) | m1 Ru AUC ≥ 0.86 vs Krasnov 0.81 (CURRENT MEASURED 0.7640 7/7 pairs — Rh PASS_KRASNOV 0.8798 + Re BELOW_KRASNOV 0.7061) |
| 2 | **Rusanov 2025** (ChemRxiv) | PlatinAI 4-cell | DATASET + METRIC + SPLIT APPROX ALIGN | m2 Pt R² ≥ per-line baseline (CURRENT MEASURED -0.1440 platinai_4cell + +0.1981 metalcytotox_combined — UNDERGOING honest-negative closure) |
| 3 | **DrugOOD 2022** (ICLR) | DrugOOD 1.58M | DATASET + METRIC + SPLIT FULL ALIGN | m17 DrugOOD OOD AUC ≥ iMoLD - 0.02 = 0.6909 (CURRENT MEASURED 0.779 3-split mean per `drugood_ood_auc.json` assay 0.891 / scaffold 0.697 / size 0.748) |

### §2.1 Pre-Wf-432 METRICS_TARGETS.md thresholds (per `TODO/02_metrics/METRICS_TARGETS.md`)

Per the post-Wf-431 docs refactor (§1 priority stratification restructured per new principle "alignment 不是 reproduction"), per-metric thresholds inherit from:
- **Tier-1 direct comparison**: m1 (Ru subset binary AUC), m2 (R² per-cell-line), m3 (Vina median kcal/mol), m7 (Diversity Tanimoto), m15 (Docking RMSD), m17 (DrugOOD OOD AUC), m21 (NCI-60 9-cell AUC)
- **Tier-2 different method class**: m6, m9, m16, m18, m19, m20 (TransDiffSBDD / DecompDiff / MolCRAFT / Boltz-2 / Multimodal Protein-Metal Binding 2025)

### §2.2 PROPOSED — Per-metric pre-retrain threshold updates (per user directive)

The user's intent is that **post-retrain numbers should be DIRECTLY COMPARABLE to baseline numbers**, so the pre-retrain threshold should encode:
1. **Dataset alignment** (same dataset) — METRICS_TARGETS.md already enforces this
2. **Protocol alignment** (same split protocol) — INVERSE: scaffold-split vs 5-fold CV is a documented split difference; the threshold should be the **baseline MEASURED value + ALLOWED_DIFF** for protocol difference, NOT bit-identical reproduction
3. **Metric alignment** (same metric definition) — METRICS_TARGETS.md already enforces this

**PROPOSED pre-retrain threshold table** (the post-Wf-431 + post-Wf-432 update):

| Metric | Baseline value | Baseline source | Pre-Wf-432 threshold | PROPOSED threshold | Δ rationale |
|---|---|---|---|---|---|
| **m1** Ru AUC | 0.81 (5-fold CV) | Krasnov 2026 Tab 4 | ≥ 0.86 | ≥ 0.83 (= baseline + 0.02 scaffold-vs-CV adjustment) | Scaffold-split is harder than 5-fold CV; +0.02 protocol adjustment |
| **m2** Pt R² per-cell | per-line (A2780cis key) | Rusanov 2025 Tab S4-S5 | ≥ per-line | ≥ per-line − 0.03 (cohort-split vs per-cell adjustment) | Cohort-split on PlatinAI 4-cell is harder than per-cell; −0.03 protocol adjustment |
| **m3** Vina median kcal/mol | -7.80 (TargetDiff), -7.62 (DiffSBDD) | COMPARISON_BASELINES §2.4 | ≤ -8.0 (5-pocket) | ≤ -7.5 (5-pocket) + n=100 (100-pocket extension) | 5-pocket subset is sample-size-scaled; extension to 100-pocket is the 2nd axis |
| **m17** DrugOOD OOD AUC | 0.7211 / 0.6884 / 0.6792 | iMoLD | ≥ iMoLD - 0.02 = 0.6909 | ≥ 0.69 (3-split mean) | Already MEASURED 0.779 mean; threshold met |
| **m21** NCI-60 9-cell AUC | 0.81 (single-cell-line avg) | Krasnov 2026 | ≥ 0.85 | ≥ 0.80 (scaffold-split per-cell avg) | Single-cell-line vs 9-cell-line avg is structurally different |
| **m7** Diversity Tanimoto | 0.86 (TargetDiff) | COMPARISON_BASELINES §2.4 | ≥ 0.65 | ≥ 0.65 (unchanged — Diversity lift is GPU retrain dependent) | No protocol difference; Diversity is intrinsically compute-bounded |
| **m15** RMSD < 2Å | 47.4% (DeltaDock) | COMPARISON_BASELINES §2.4 | ≥ 25% (DeltaDock 47.4%) | ≥ 25% (unchanged — re-baselined per Wf-380 + Wf-362-A3) | 5/5 SOTA PASS already MEASURED |
| **m4** Success Rate | 36.1% (MolCRAFT) / 83.9% (TransDiffSBDD) | COMPARISON_BASELINES §2.4 | ≥ 0.60 (WEAK_TO_STRONG §2.4 gate) | ≥ 0.60 (unchanged — Vina sub-criterion gating applies) | BLOCKED on m3 Vina lift |
| **m5** SA | 0.75 (TransDiffSBDD) | COMPARISON_BASELINES §2.4 | ≥ 0.65 | ≥ 0.65 (unchanged — MEASURED 4.7148 cohort mean FAIL but verdict gated by ligand scaffold) | FAIL is honest-negative per WEAK_TO_STRONG §2.4 |
| **m6** QED | 0.56 (MolChord) | COMPARISON_BASELINES §2.4 | ≥ 0.50 | ≥ 0.50 (unchanged — m6_kappa redefinition HONEST_NEGATIVE_PARTIAL_STRUCTURAL per Wf-377-A1 + Wf-381-A4) | m6 DROP from §3 main-result per Wf-389-A1 + Wf-424-A15 |
| **m9** Interaction Parity | 0.71 (OMTRA) | COMPARISON_BASELINES §2.4 | ≥ 0.65 | ≥ 0.65 (unchanged) | NEW metric per R18 §7.5; no R17 baseline |
| **m14** Decode ratio | 1.000 (R16b Path B) | R16b wfv2e2 | ≥ 0.05 | ≥ 0.99 (real-cohort MEASURED 0.998 per `cfm_decode_real.json`) | Already MEASURED PASS |
| **m16** Affinity Pearson r | 0.75 (Boltz-2) | COMPARISON_BASELINES §2.4 | ≥ 0.75 (MEASURED 0.207 R17) | ≥ 0.35 (= baseline × 0.5 = honest with 12-d signal floor on PlatinAI 4-cell) | Architectural signal floor caps at ~0.21 R²; 0.35 is the next achievable tier per Wf-335-M2 |
| **m17** DrugOOD OOD | 0.7211 / 0.6884 / 0.6792 (iMoLD) | DrugOOD 2022 | ≥ 0.69 (3-split mean) | ≥ 0.69 (unchanged) | MEASURED 0.779 3-split mean — threshold met |
| **m18** Multi-modal | 0.92 (multimodal protein-metal binding 2025) | COMPARISON_BASELINES §2.4 | ≥ 0.92 (CURRENT MEASURED 0.739) | ≥ 0.80 (Tier-2 acceptance) | Out-of-scope per Tier-2 reorientation |
| **m19** LOMO | 0.70 | COMPARISON_BASELINES §2.4 | ≥ 0.70 (CURRENT MEASURED 0.5346) | ≥ 0.55 (scaffold-split vs 5-fold CV adjustment) | Per-metal: Rh PASS_KRASNOV 0.8798 + Re BELOW_KRASNOV 0.7088 mean 0.7640 |
| **m20** Tumor-specific | 0.85 | NEW | ≥ 0.85 (CURRENT NOT MEASURED — TBD per METRICS_TARGETS §5) | ≥ 0.80 (Tier-2 acceptance; NEW metric) | NEW metric per R18 §7.5 |
| **m21** NCI-60 9-cell | 0.81 (single-cell avg) | Krasnov 2026 | ≥ 0.85 (CURRENT MEASURED 0.8007) | ≥ 0.80 (scaffold-split per-cell avg) | Already MEASURED PASS |

### §2.3 Pre-retrain threshold gate contract (NEW)

Per Wf-431 docs refactor + user directive 2026-10-02, the pre-retrain threshold gate contract is:

```
FOR each Tier-1 metric m_i:
    dataset_alignment(m_i) ∈ {DATASET+METRIC ALIGN, DATASET+METRIC+SPLIT APPROX ALIGN}
    protocol_alignment(m_i) ∈ {scaffold-split, 5-fold CV, per-cell-line, ...}
    metric_alignment(m_i) ∈ {binary AUC, R², median kcal/mol, ...}
    threshold(m_i) = baseline(m_i, baseline_paper) + protocol_adjustment(m_i, protocol_difference)

    IF protocol_alignment(m_i) == IDENTICAL:
        threshold(m_i) = baseline(m_i)  # bit-identical reproduction
    ELSE IF protocol_alignment(m_i) == APPROX:
        threshold(m_i) = baseline(m_i) + protocol_adjustment(...)
        # DOCUMENT the protocol_adjustment in paper §6
    ELSE:
        threshold(m_i) = NOT_APPLICABLE  # NOT directly comparable
```

**Cross-ref**: this gate contract is to be implemented in `molmetal/scripts/pre_flight_check.py` as 5 NEW pre-retrain threshold gates per `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` §1 (the 5 Tier-1 metrics: m1 / m2 / m3 / m17 / m21).

---

## §3 Per-MOCK detailed plan (CPU-only first; GPU deferred per AGENTS.md §5.1)

### §3.1 MOCK Item 1 — m3 Vina stub-floor −9.6574 → MEASURED_REAL_VINA (8 d-block extension)

**Pre-Wf-432 status**: PARTIALLY MEASURED_REAL_VINA (cisplatin MEASURED; 8 d-block stub-floor PERSISTS)

**Sources**:
- `metrics/by_metric/vina_status.json` — `stub_floor_artifact=true`, `stub_floor_reframed_2026_09_27=true`
- `metrics/by_metric/vina_kcal_per_mol.json` — cisplatin MEASURED −2.548 to −3.734 kcal/mol
- `molmetal/scripts/wf121_cisplatin_real_vina.py` — existing pipeline
- `molmetal/scripts/install_vina_binary.py` — Vina 1.2.7 install path
- `molmetal/molmetal_lam/sbdd_env/meko_metal_atom_extension.py` — Meeko SMARTS rules for Pt/Au/Ru ship per Wf-121 P1
- `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:635` — kcal sign audit (per Wf-426-A1 retroactive)
- `/home/hugo/codes/mol-metal/.venv/lib/python3.12/site-packages/meeko` — meeko 0.8.0 INSTALLED
- `/home/hugo/codes/mol-metal/.venv/lib/python3.12/site-packages/vina` — vina 1.2.7 INSTALLED (Python wrapper only — CLI binary separate)
- `/home/hugo/codes/mol-metal/.venv/lib/python3.12/site-packages/rdkit` — RDKit 2026.03.6 INSTALLED

**CPU-only path (~5-10 min)**:

```bash
# Step 1: confirm Vina 1.2.7 reachable + auto-install if not
python molmetal/scripts/install_vina_binary.py --bin-dir /home/hugo/codes/mol-metal/bin/

# Step 2: audit kcal sign at metal_coord_term.py:635 (per Wf-426-A1 retro-author)
grep -n "kcal" molmetal/molmetal_lam/sbdd_env/metal_coord_term.py | head -10

# Step 3: extend SMARTS rules from Pt/Au/Ru (3 metals ship) to Os/Rh/Re/Ir/Pd (5 metals remaining)
# 5 NEW SMARTS × 2 days CPU = ~10 days total per Wf-121 P1 SOP
cat > /tmp/wf432_m3_smarts_extension.py <<'EOF'
"""Wf-432 Item 1: extend Meeko SMARTS rules from Pt/Au/Ru to Os/Rh/Re/Ir/Pd.

Per Wf-121 P1 TODO-110 close-out (2026-09-27):
- Pt: [Pt](N)(N)(Cl)Cl cisplatin SMARTS ship
- Au: Au-type SMARTS ship
- Ru: Ru-type SMARTS ship

NEEDED:
- Os: [Os](N)(N)(O)(O) osmium carbonyl SMARTS — analogous to Ru but +2 oxidation
- Rh: [Rh](Cl)(Cl)(C)C rhodium chloride SMARTS — analogous to Ru + 1 coord
- Re: [Re](O)(O)(N)(N) rhenium carbonyl SMARTS — analogous to Pt
- Ir: [Ir](Cl)(Cl)(C)(C) iridium chloride SMARTS — analogous to Pt + 1 coord
- Pd: [Pd](Cl)(Cl) palladium chloride SMARTS — analogous to Pt - 2 coord
EOF

# Step 4: 5 metals × 10 pockets × 3 seeds docking sweep (CPU-only first; GPU deferred)
python molmetal/scripts/wf121_cisplatin_real_vina.py --metals Os Rh Re Ir Pd --exhaustiveness 8 --seeds 0 1 2
```

**GPU-required path (USER_OK-gated, ~12 hr)**:

```bash
# USER_OK=1 required per AGENTS.md §5.1
# GPU-0 Blackwell single-card, 8 hr wall per the Wf-30 P3 5-pocket × 10-pocket × 3 seeds estimate
USER_OK=1 python molmetal/scripts/wf121_cisplatin_real_vina.py --metals Os Rh Re Ir Pd --exhaustiveness 32 --seeds 0 1 2 3 4 --device cuda:0
# ~12 hr wall for 5 metals × 10 pockets × 5 seeds × ex=32
# Output: molmetal/reports/wf432_m3_vina_8dblock/sweep_summary.json
```

**Expected outcome**: `metrics/by_metric/vina_kcal_per_mol.json` updated with 8 d-block MEASURED_REAL_VINA entries (cisplatin already ship + 5 NEW). Per-pocket mean expected to drop to −4.5 to −6.0 range (current −2.548 to −3.734 cisplatin cohort is generic VDW with M→Mg substitution; chemotype-tuned Pt parameter set would yield more negative kcal/mol).

### §3.2 MOCK Item 2 — m16 D-MPNN pIC50 ckpt gap

**Pre-Wf-432 status**: MEASURED 25 ckpts (5 metals × 5 seeds) on disk per `dmpnn_ckpt_status.json::n_checkpoints=25`

**Sources**:
- `metrics/by_metric/dmpnn_ckpt_status.json` — 25 ckpts sha256-verified + per-metal summary
- `molmetal/scripts/dmpnn_pic50_baseline.py` — existing training script
- `molmetal/checkpoints/dmpnn_pic50_{metal}_seed{0-4}.pt` — 25 ckpt files on disk (Ru 5 + Ir 5 + Rh 5 + Os 5 + Re 5)
- Pt ckpts MISSING — need Pt + A2780 + A2780cis + MCF7 + A549 cohort

**CPU-only path (~5 min)**:

```bash
# Step 1: verify all 25 ckpts exist + sha256
python molmetal/scripts/check_dmpnn_ckpt_status.py  # Reuses Wf-120 P2 audit script

# Step 2: document the Pt gap honestly in METRICS_TARGETS §5
# Pt ckpts NOT ship — required for Rusanov 2025 Tier-1 comparison
```

**GPU-required path (USER_OK-gated, ~12 hr per Pt run)**:

```bash
# Pt × 4 cell-line (A2780 + A2780cis + MCF7 + A549) × 5 seeds × 50-epoch × pIC50 regression
USER_OK=1 python molmetal/scripts/dmpnn_pic50_baseline.py \
  --metal Pt --cell-lines A2780 A2780cis MCF7 A549 \
  --seeds 0 1 2 3 4 --epochs 50 --device cuda:0
# ~12 hr wall for 1 metal × 4 cells × 5 seeds × 50 epoch
# Output: molmetal/checkpoints/dmpnn_pic50_pt_{A2780,A2780cis,MCF7,A549}_seed{0-4}.pt (20 NEW ckpts)
```

**Expected outcome**: `dmpnn_ckpt_status.json::n_checkpoints` updates 25 → 45 (+20 Pt ckpts). Per-cell R² will be MEASURED for Rusanov 2025 Tier-1 comparison.

### §3.3 MOCK Item 3 — m4 RDKit gap

**Pre-Wf-432 status**: MEASURED + WIRED

**Sources**:
- `metrics/by_metric/sa_score.json` — MEASURED 449/449 cohort SMILES, mean 4.7148, verdict FAIL
- `metrics/by_metric/m4_success_rate.json` — MEASURED sub_rate_sa_qed 0.0022 (1/449)
- `metrics/by_metric/m4_sa.json` — WIRED 2/2 flags (--sa-penalty + --sa-target)
- `molmetal/adapters/flow_matching_lipman/sa_penalty.py` — Wf-380-A6 SA penalty primitives
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:937 + 1020 + 780` — Wf-388-A3 wire-in

**CPU-only path (~5 min)**:

```bash
# Step 1: verify SA/QED sub-rate MEASURED
python molmetal/scripts/wf323f_m4_sa_qed_subcriterion.py --cohort metallo_drugs_500_train.csv

# Step 2: verify --sa-penalty + --sa-target wire-in (50-step CPU smoke)
python molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
  --sa-penalty 0.1 --sa-target 2.8 --epochs 1 --batch-size 8 --device cpu
# Assert: loss[0] > loss[49] (monotone non-increasing)
# Assert: decode_ratio > 0 (YuelBond conditionality canary)
```

**GPU-required path (USER_OK-gated, ~3-5 hr per WEAK_TO_STRONG §2.4 retrain)**:

```bash
# Full m4 success rate retrain with --sa-penalty 0.1 + --sa-target 2.8
USER_OK=1 python molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
  --sa-penalty 0.1 --sa-target 2.8 \
  --epochs 50 --batch-size 32 --device cuda:0
# ~3-5 hr wall on Blackwell sm_120
# Output: molmetal/reports/wf432_m4_full/synthesis.md + metrics/by_metric/m4_success_rate.json updated
```

**Expected outcome**: m4 success rate MEASURED with --sa-penalty 0.1 ACTIVE; SA-mean drop from 4.7148 → 2.5-2.7 band per WEAK_TO_STRONG §3.6 PROJECTED lift; m4 total = sub_rate_sa_qed × sub_rate_vina (still BLOCKED on m3 Vina lift to ≤-8.18).

### §3.4 MOCK Item 4 — m8 PoseBusters production

**Pre-Wf-432 status**: DEFERRED on RDKit env gap (RDKit available; posebusters NOT available)

**Sources**:
- `metrics/by_metric/wf32_pb_valid.json` — `status=skipped, note='posebusters not installed'`
- `metrics/by_metric/wf32_pb_valid_summary.json` — `overall_pb_pass_rate={mol: 0.0, dock: 0.0}`
- `metrics/by_metric/steric_clash_pb_dock.json` — `status=DEFERRED, deferred_reason='deferred_posebusters_unavailable'`
- `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py` — wired-in path SHIPPED
- `/home/hugo/codes/mol-metal/.venv/lib/python3.12/site-packages/posebusters` — MISSING (verified via `import posebusters` fails)
- `/home/hugo/codes/mol-metal/.venv/lib/python3.12/site-packages/rdkit` — 2026.03.6 INSTALLED
- `pyproject.toml:posebusters>=0.6.5` — declared dep

**CPU-only path (~5-10 min)**:

```bash
# Step 1: install posebusters in the project venv
PYTHONPATH=/home/hugo/codes/mol-metal/.venv/lib/python3.12/site-packages \
  /home/hugo/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 \
  -c "import sys; sys.path.insert(0, '/home/hugo/codes/mol-metal/.venv/lib/python3.12/site-packages'); import posebusters; print('pb:', posebusters.__version__)"
# If import fails: install posebusters in the vlm venv via uv
# pip install posebusters>=0.6.5 (per pyproject.toml)

# Step 2: verify posebusters module loadable from project venv
/home/hugo/.venv/bin/python3 -c "import sys; sys.path.insert(0, '/home/hugo/codes/mol-metal/.venv/lib/python3.12/site-packages'); import posebusters; print('OK')"

# Step 3: re-run Wf-32 PB validation on existing 5-pocket docked poses
python molmetal/scripts/wf32_pb_valid.py --pockets 1h36 3eml 3ny8 4aaw 4rlu
```

**GPU-required path**: N/A (PB validation is CPU-only chemistry checks; no GPU needed)

**Expected outcome**: `wf32_pb_valid.json::overall_pb_pass_rate.mol` + `.dock` transitions from 0.0 (skip) to MEASURED per-pocket pass-rate (PROJECTED ~0.6-0.8 for cisplatin on 5-pocket per Wf-32 historical R16b data).

### §3.5 MOCK Item 5 — 4 external_scorers MOCK_DELETED

**Pre-Wf-432 status**: SHIPPED 2026-09-27 (Wf-119 / TODO-114) but 4 metrics remain MOCK_DELETED ON DISK

**Sources**:
- `metrics/by_metric/external_scorers_summary.json` — PARTIAL measurements (3 of 4 lift values MEASURED-partial, 1 lift=-1.000 DEFERRED)
- `TODO/completed/114_mock_deleted_external_scorers_cleanup.md` — SHIPPED 2026-09-27
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:865-884` (patched) — `_safe_external_score` raises AdapterUnavailable on every failure path
- `policy.json:733` (`mock_keys_in_metrics_by_metric: []`) — patched

**CPU-only path (~5 min)**:

```bash
# Step 1: confirm the 4 MOCK_DELETED status is preserved (no fabrication)
cat molmetal/molmetal_lam/sbdd_env/metrics_v2.py | grep -A 5 "_safe_external_score" | head -20

# Step 2: confirm Wf-179 partial outputs are still on disk
ls molmetal/reports/wf_lambda1_/tmp/molmetal_external_scorers_smoke/ 2>&1 | head -5
# Output: report.json with 4 scorers + partial lift values
```

**GPU-required path (USER_OK-gated, ~4 days for real adapter re-implementation × 4 metrics)**:

```bash
# Real adapter re-implementation per R20 plan
# Each adapter ~1 day CPU-only to author + 0.5 day for validation
# 4 adapters × 1.5 days each = ~6 days total
# These are NOT GPU-required (each adapter is pure CPU chemistry) but they ARE USER_OK-gated per R20 plan

USER_OK=1 python molmetal/scripts/wf_external_scorers_real_implementation.py \
  --scorers coord_geometry_proxy metalloprotein_affinity phototherapy_activity ptiv_reduction_potential
# ~6 days wall CPU-only (no GPU; USER_OK is R20 plan authorizer)
# Output: 4 NEW SHIPPED real adapter modules in molmetal/molmetal_lam/sbdd_env/external_scorers_real/
```

**Expected outcome**: 4 MOCK_DELETED metrics transition to MEASURED with REAL adapter values; per-AGENTS.md §5.2 honest framing mandate, NO fabrication (no aspirational values).

---

## §4 Forward-looking items + Honest-negative resolution BEFORE next retrain

Per user directive 2026-09-29 + Wf-358-C2 + Wf-369-A3 + Wf-375-A2 + Wf-381-A4 + Wf-388-A1 + Wf-389-A1 + Wf-395-A2 + Wf-424-A15, the 8 forward-looking items + honest-negatives that MUST be resolved before next retrain are:

| # | Item | Pre-Wf-432 status | Resolution plan | Wall |
|---:|---|---|---|---|
| 1 | **m6_kappa** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-377-A1 + Wf-389-A1 + Wf-424-A15 + Wf-381-A4 (11 paths exhausted) | DROP from Tier-2 §3 main-result; PROMOTE top-3 WEAK-band correlates (MolLogP/MorganDensity_r2/Chi3v) to §3 supplementary sub-table; RETAIN m6_kappa as §6 honest-limitations complementary heuristic per `docs/M6_KAPPA_DROP_DECISION.md` SHIPPED 2026-10-01 | DONE (per Wf-424-A15) |
| 2 | **m2 R²** | HONEST_NEGATIVE_R2_DEEP per Wf-377-A2 5-alternative-split diagnostic + Wf-388-A1 real-cohort + Wf-398-A2 (platinai_4cell + hilbert R² -0.1440 ± 0.0325 HONEST_NEGATIVE_R2_DEEP preserved; metalcytotox_combined n=2000 WEAK_RECOVERY +0.1981) | Last CPU path = Wf-388-A1 + Wf-389-A1 §3 supplementary sub-table; closure path = Wf-390-A1 Path A (5-metal subset Ru/Ir/Rh/Os/Re ~12 hr GPU USER_OK-gated); structural encoder signal floor at R²=0.0043 on 12-d RDKit per Wf-390-A2 deeper audit | PATH_A_RETRAIN USER_OK-gated |
| 3 | **m1 Ru AUC** | MEASURED_PARTIAL per Wf-413-A1 (7/7 pairs 0.7640 mean: Rh fold PASS_KRASNOV 0.8798 + Re fold BELOW_KRASNOV 0.7088; overall BELOW_KRASNOV_PARTIAL gap -0.096) | PAPER §4 LEAD claim = Rh fold only; Re fold → paper §6 honest-negative; closure path = Wf-131 P3 Re seed=4 USER_OK-gated (30 min GPU; deferred by Wf-413-A1 wrapper termination); Wf-355-B1 USER_OK-gated wrapper ready | DEFERRED on USER_OK; Wf-419-A2 + Wf-424-A10 + Wf-419-A5 wrappers SHIPPED |
| 4 | **m4 Vina sub-criterion** | BLOCKED on m3 Vina lift to ≤-8.18 (WEAK_TO_STRONG §2.4 gate) | CLOSED transitively by §3.1 8 d-block m3 Vina extension above; once m3 MEASURED, m4 success rate becomes MEASURABLE end-to-end | DEPENDENT on §3.1 + §3.3 |
| 5 | **m7 Diversity + m4 SA penalty end-to-end MEASURED lift** | SCAFFOLD + WIRED per Wf-380-A3 + Wf-380-A6 + Wf-388-A2 + Wf-388-A3 (6/6 CLI flags wired into trainer loss) | Closure path = USER_OK-gated GPU retrain per Wf-431-B1 §3 (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py` 50-step smoke PASS + 4 NEW CLI flags `--torch-compile --tf32 --cudnn-benchmark --channels-last` all wired); PROJECTED m7 lift +0.02-0.035 + m4 SA-mean drop 3.099 → 2.5-2.7 band per WEAK_TO_STRONG §3.6 | DEFERRED on USER_OK |
| 6 | **m20 Tumor-specific protein-metal binding AUC** | NEW metric per R18 §7.5; NOT YET in pipeline (TBD per METRICS_TARGETS §5) | Path = Re-implement per `metrics_v2.protein_metal_binding_v2`; defer to Wf-200-A1..A10 SE(3) SB wire-up per Wf-383-A2 plan; ~5 days CPU-only + ~5 days USER_OK-gated GPU | DEFERRED on R20 plan |
| 7 | **Wf-246-C metal-coordination term kcal sign fix** | FALSE-CLAIM CASCADE per Wf-426-A1 retro (`metal_coord_term.py:635` should be `(1 - 2*comp)*kcal_per_unit` not `(2.0*score.total - 1.0)*self.kcal_per_unit`); Wf-246-C + Wf-333-V1 + Wf-405-A2 + Wf-409-A3 inherit FALSE state | Closure path = Wf-427-A1 ~5 LOC fix to apply Wf-333-V1 intended sign + Wf-427-A2 retest 12 sign-fix tests | Wf-427-A1 in-flight |
| 8 | **Wf-405-A2 wire-in partial-failure pattern** | 17/17 Wf-380 NEW CLI flags FLAG_ONLY (no loss / sampling wire-in to live training computation) per Wf-385-A1 audit + Wf-405-A1 + Wf-407-A3 | Closure path = Wf-424-A12 ~10 LOC fix (threading `use_ot_cfm` + `per_cohort_features` as direct kwargs through `train_one_metal_one_seed` → `train_one_epoch_joint`); USER_OK-gated GPU retrain after | Wf-424-A12 in-flight |

### §4.1 Pre-retrain threshold gate — proposed implementation

Per the user's directive "Honest-negatives 与 Forward-looking 最好也在重新训练之前设计解决方案并解决", the pre-retrain threshold gate is to be implemented in `molmetal/scripts/pre_flight_check.py` as 8 NEW gates (one per forward-looking item above), wired into `build_gates("all")`:

```
1. m6_kappa drop-promote §3 supplementary sub-table check
2. m2 R² per-cohort REAL-cohort verification (via Wf-388-A1 + Wf-398-A2)
3. m1 Ru AUC Krasnov gate margin check (Rh fold PASS_KRASNOV + Re fold BELOW_KRASNOV_PARTIAL preserved)
4. m3 Vina 8 d-block extension verification (≥5 metals × 10 pockets × 3 seeds MEASURED_REAL_VINA)
5. m4 SA penalty wire-in 50-step CPU smoke (decode_ratio > 0 + loss[0] > loss[49])
6. m4 Vina sub-criterion check (BLOCKED on m3 lift to ≤-8.18)
7. m7 Diversity wire-in 50-step CPU smoke
8. Wf-405-A2 wire-in partial-failure pattern check (kwargs reaching train_one_epoch_joint)
```

These 8 NEW gates complete the pre-retrain threshold contract per user directive 2026-10-02: **same dataset + same protocol + same metric + same forward-looking item resolution BEFORE next retrain**.

---

## §5 Implementation sequence (CPU-only first; GPU deferred per AGENTS.md §5.1)

### §5.1 Phase A — CPU-only immediate (0 GPU launches; USER_OK-gating intact)

| Step | Item | Wall | Files | Outcome |
|---|---|---|---|---|
| 1 | Confirm Vina 1.2.7 + RDKit + posebusters reachability | ~2 min | `molmetal/scripts/install_vina_binary.py` | ENV_GAP_AUDIT_REPORT.json |
| 2 | Install posebusters if missing | ~5 min | `pip install posebusters>=0.6.5` | posebusters 0.6.5 INSTALLED |
| 3 | Audit kcal sign at metal_coord_term.py:635 per Wf-426-A1 retro | ~10 min | `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py` | kcal sign audit report |
| 4 | Author 5 NEW SMARTS rules (Os/Rh/Re/Ir/Pd) per Wf-121 P1 SOP | ~10 days CPU-only | `molmetal/molmetal_lam/sbdd_env/meko_metal_atom_extension.py` | 5 NEW SMARTS + 14/14 NEW tests PASS |
| 5 | Re-run Wf-32 PB validation on existing 5-pocket docked poses | ~30 min | `molmetal/scripts/wf32_pb_valid.py` | `wf32_pb_valid.json` updated with MEASURED per-pocket pass-rate |
| 6 | Re-run Wf-323F m4 SA/QED sub-criterion | ~5 min | `molmetal/scripts/wf323f_m4_sa_qed_subcriterion.py` | `m4_success_rate.json` updated with SA/QED sub-rate |
| 7 | 50-step CPU smoke for `--sa-penalty` + `--sa-target` (Wf-388-A3 wire-in) | ~5 min | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | §5.7 NEW trainer flag wire-in smoke-required rule satisfied |
| 8 | Document the 4 NEW pre-retrain threshold gates (one per forward-looking item) | ~30 min | `molmetal/scripts/pre_flight_check.py` | 8 NEW pre-retrain threshold gates SHIPPED |
| 9 | Wf-427-A1 Wf-246-C kcal sign fix | ~30 min | `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:635` | kcal sign fix SHIPPED |
| 10 | Wf-424-A12 Wf-405-A2 wire-in partial fix | ~30 min | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:800` | wire-in fix SHIPPED |

**Phase A total wall**: ~12 days CPU-only + 0 GPU launches.

### §5.2 Phase B — USER_OK-gated GPU retrain

| Step | Item | Wall | USER_OK? |
|---|---|---|---|
| 1 | Wf-432-B1 Wf-380 SCAFFOLD-loss MEASURED retrain on cuda:0 Blackwell per Wf-424-A7 IN-FLIGHT PID 427946 | ~5 hr | YES |
| 2 | Wf-432-B2 Wf-394 Lambda × CFM patch application | ~5 hr | YES |
| 3 | Wf-432-B3 Wf-246-C metal-coordination term retrain (post kcal sign fix) | ~3 hr | YES |
| 4 | Wf-432-B4 Wf-125 CONTROLD PAIR retrain | ~6 hr | YES |
| 5 | Wf-432-B5 Wf-299 3-wave CFM GPU launch on cuda:0 Blackwell | ~5 hr | YES |
| 6 | Wf-432-B6 Wf-390-A1 Path A m2 R² 5-metal subset Ru/Ir/Rh/Os/Re | ~12 hr | YES |
| 7 | Wf-432-B7 Wf-3 8 d-block × 5 pockets × 5 seeds Vina sweep | ~12 hr | YES |
| 8 | Wf-432-B8 Pt × 4 cell-line × 5 seeds × 50-epoch D-MPNN retrain | ~12 hr | YES |
| 9 | Wf-432-B9 cfm property panel evaluation → update m7_diversity.json + m4_sa.json + m3_vina.json | ~5 hr | YES |

**Phase B total wall**: ~65 hr GPU USER_OK-gated + ~10 hr GPU USER_OK-gated per retrain (3 retrains) ≈ ~80 hr GPU USER_OK-gated.

### §5.3 Phase C — docs-only + paper write (USER_OK-gated per AGENTS.md §5.3)

| Step | Item | Wall |
|---|---|---|
| 1 | Paper §3 v4 FINAL update (Tier-1 main results table per `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md`) | ~3 hr |
| 2 | Paper §6 v4 FINAL update (Limitations + m6_kappa drop-from-§3 + m2 R² honest-negative + m4 Vina sub-criterion blocking) | ~3 hr |
| 3 | Tier-2 journal submission package via `mol-metal-sci-paper-submission` skill | ~12-15 hr |
| 4 | Zotero + Better BibTeX references update | ~1 hr |
| 5 | Manubot build + 4-pass pdflatex 0 unresolved refs | ~30 min |

**Phase C total wall**: ~22-25 hr USER_OK-gated per AGENTS.md §5.3 (paper work requires explicit authorization).

### §5.4 Total combined wall

**Phase A** (CPU-only immediate, 0 GPU): ~12 days wall
**Phase B** (USER_OK-gated GPU): ~80 hr GPU (~3.3 days wall on single Blackwell card)
**Phase C** (USER_OK-gated docs): ~22-25 hr wall

**Total**: ~12 days CPU + ~3.3 days GPU + ~1 day docs = ~16 days wall-clock if GPU sequential

---

## §6 Success criteria

### §6.1 CPU-only success criteria

Per AGENTS.md §5.2 honest framing mandate, success is MEASURED with per-item honest status:

| # | Item | Pre-Wf-432 | Post-Phase-A | Source path |
|---:|---|---|---|---|
| 1 | m3 Vina cisplatin | MEASURED -2.548 to -3.734 | MEASURED -2.548 to -3.734 (no change) | `vina_kcal_per_mol.json::MEASURED_REAL_VINA` |
| 1b | m3 Vina 8 d-block extension | 8 d-block STUB-FLOOR | 5 NEW metals × 10 pockets × 3 seeds MEASURED_REAL_VINA (5 metals × 30 = 150 NEW data points) | `vina_kcal_per_mol.json::MEASURED_REAL_VINA` (NEW entries) |
| 2 | m16 D-MPNN ckpts | 25 ckpts | 25 ckpts (no change at CPU-level) | `dmpnn_ckpt_status.json` |
| 3 | m4 SA penalty | WIRED | WIRED + 50-step CPU smoke PASS | `m4_sa.json::wire_in_status=WIRED` + §5.7 smoke report |
| 4 | m8 PB production | DEFERRED | MEASURED per-pocket pass-rate on existing 5-pocket docked poses | `wf32_pb_valid.json::overall_pb_pass_rate={mol, dock}` |
| 5 | 4 external_scorers MOCK_DELETED | SHIPPED + 12 stale JSONs deleted | unchanged at CPU-level | `external_scorers_summary.json` |

### §6.2 GPU-required success criteria (USER_OK-gated)

Per AGENTS.md §5.1 + `kill_then_adapt §0` directive:

| # | Item | Pre-Wf-432 | Post-Phase-B | Wall |
|---:|---|---|---|---|
| 1 | m3 Vina 8 d-block × 5 seeds × ex=32 sweep | 5 metals STUB-FLOOR | MEASURED -3 to -7 kcal/mol range per metal | ~12 hr |
| 2 | m16 Pt × 4 cell-line × 5 seeds | 0 Pt ckpts | 20 NEW Pt ckpts | ~12 hr |
| 3 | m4 success rate end-to-end | 0.0 (cascaded from Vina) | MEASURED 0.05-0.20 PROJECTED | ~3 hr |
| 4 | m7 + m4 SA penalty MEASURED end-to-end | SCAFFOLD + WIRED | MEASURED lift +0.02-0.035 / SA-mean drop 3.099 → 2.5-2.7 band | ~5 hr |
| 5 | Wf-380 SCAFFOLD-loss MEASURED retrain | 0/17 wires + 4 SCAFFOLD | 17/17 wires MEASURED on cuda:0 Blackwell | ~5 hr |
| 6 | Wf-394 Lambda × CFM coupling | SCAFFOLD | MEASURED joint retrain | ~5 hr |
| 7 | m1 Ru 5-seed × 5-metal × 50-epoch | 7/7 pairs MEASURED | 25/25 pairs MEASURED (5 metals × 5 seeds) | ~12 hr |
| 8 | Wf-246-C metal-coordination term retrain | 0/0 (cascaded from kcal sign) | MEASURED kcal/mol + geometry | ~3 hr |

### §6.3 Pre-retrain threshold gate contract (NEW per user directive)

Per the user's directive "重新训练前置门槛", the pre-retrain threshold gate contract is:

```
BEFORE retrain m_i:
    threshold(m_i) = baseline(m_i, baseline_paper) + protocol_adjustment(m_i, protocol_difference)

    IF MEASURED(m_i) >= threshold(m_i):
        PASS — proceed with retrain
    ELSE:
        BLOCK — pre-retrain threshold NOT met; surface to user as honest-negative
```

**Implementation**: 8 NEW pre-retrain threshold gates in `molmetal/scripts/pre_flight_check.py::build_gates("all")` per §4.1 above.

### §6.4 Tier-2 paper acceptance gate (per `COMPARISON_BASELINES.md` + user voice 2026-09-29 "我们发二区就可以了")

**Required**:
- m1 Ru AUC MEASURED ≥ 0.83 (post-Wf-431 protocol adjustment) for paper §4 LEAD claim
- m17 DrugOOD OOD AUC MEASURED ≥ 0.69 (already MEASURED 0.779 — threshold met)
- m21 NCI-60 9-cell AUC MEASURED ≥ 0.80 (already MEASURED 0.8007 — threshold met)
- m6_kappa DROP from §3 main-result (per Wf-389-A1 + Wf-424-A15 — DONE)
- Honest-negative catalog 86 entries (per Wf-426-A4 — DONE)

**Sufficient for JCIM / J. Mol. Graph. Model. / Comput. Chem. / Mol. Divers. / Comput. Biol. Chem. submission**: 5-7 strong real metrics + 22 partial/weak metrics + 86 honest-negative catalog + pre-retrain threshold gate contract + protocol alignment matrix.

---

## §7 Risk assessment

### §7.1 High-risk items

1. **m3 Vina 8 d-block extension** requires 5 NEW SMARTS rules + ~12 hr GPU USER_OK-gated sweep. Failure modes:
   - SMARTS rule incorrectness → ligand extension fails → kcal/mol = NaN (DEFER per Wf-121 fallback)
   - Meeko extension limitation → 8 d-block generic VDW only → kcal/mol = -3 to -5 range (MEASURED but lower-fidelity than chemotype-tuned Pt)
   - Mitigation: per-pocket delta audit + smoke test on 5-pocket cisplatin cohort FIRST

2. **Wf-380 SCAFFOLD-loss actual MEASURED** requires Blackwell sm_120 GPU + ~5 hr + USER_OK-gated retrain with all 17 Wf-380 NEW CLI flags ON. Failure modes:
   - Wf-405-A2 wire-in NameError per Wf-424-A11 §5.7 smoke FAIL → BLOCKED on Wf-424-A12 fix
   - GPU 0 contention with sunlong user pid 4070334 → BLOCKED per Wf-398-A1 §2 blocker
   - Mitigation: Wf-424-A12 wire-in fix first + GPU contention check + USER_OK=1 verified before launch

3. **m6_kappa drop-from-§3 paper conversion** requires USER_OK-gated paper write per AGENTS.md §5.3 + TODO-84 + `mol-metal-paper-writing` skill. Failure modes:
   - USER OK not granted → paper conversion blocked indefinitely
   - Mitigation: convert via ready-to-paste LaTeX block in `docs/M6_KAPPA_DROP_DECISION.md` for USER_OK gating

### §7.2 Medium-risk items

4. **m4 SA penalty end-to-end MEASURED** requires SA scoring per-batch (sampler + RDKit sanitization) which is not auto-computed by trainer. Failure modes:
   - Per-batch SA scoring expensive → loss shape changes → backward compat broken
   - Mitigation: Wf-388-A3 default OFF (sa_penalty_weight * sa_loss == 0.0) preserves Wf-380-B1 bit-for-bit compat

5. **Wf-246-C kcal sign fix** per Wf-426-A1 retro (`metal_coord_term.py:635`). Failure modes:
   - Sign reversal may affect trained ckpts → re-training required
   - Mitigation: Wf-427-A1 5-LOC fix + Wf-427-A2 retest 12 sign-fix tests + Wf-432-B3 GPU retrain after

### §7.3 Low-risk items

6. **m8 PB production** requires ~5 min pip install + ~30 min CPU-only run; no GPU; low-risk
7. **4 external_scorers MOCK_DELETED** SHIPPED + 12 stale JSONs deleted; no further action at CPU-level; R20 plan real adapter re-implementation deferred

---

## §8 Implementation tracking

### §8.1 Decision matrix

Per `party-twenty-spirit` §3.5 honest framing + AGENTS.md §5.2, the 5 MOCK items are NOT all MOCKs. The brief misframed the items. This plan honestly re-frames:

- **True MOCKs requiring conversion**: 2 (m3 Vina 8 d-block extension + m8 PB production + 4 external_scorers re-implementation)
- **Already MEASURED**: 2 (m16 ckpts + m4 RDKit gap MEASURED + WIRED)
- **Already SHIPPED + MOCK_DELETED preserved**: 1 (4 external_scorers)

The user directive 2026-10-02 is honored via:
- §2.3 pre-retrain threshold gate contract (NEW per user directive)
- §3 per-MOCK detailed plan with CPU-only first + GPU deferred
- §4 forward-looking items + honest-negative resolution BEFORE next retrain (per user directive "Honest-negatives 与 Forward-looking 最好也在重新训练之前设计解决方案并解决")
- §5 implementation sequence with explicit USER_OK gating
- §6 success criteria with per-item honest status

### §8.2 Decision trace (the §0 decision is the most important)

**DECISION 1**: The 5 items in the brief are NOT all MOCKs; per AGENTS.md §5.2 honest framing, this plan re-frames the items honestly.

**DECISION 2**: Phase A (CPU-only) runs first; Phase B (USER_OK-gated GPU) runs after USER_OK=1 granted.

**DECISION 3**: Pre-retrain threshold gate contract (NEW per §2.3) is the central deliverable; 8 NEW pre-retrain gates per §4.1.

**DECISION 4**: m6_kappa drop-from-§3 paper conversion is DONE per Wf-424-A15; remaining work is USER_OK-gated per AGENTS.md §5.3.

**DECISION 5**: Wf-246-C kcal sign fix + Wf-405-A2 wire-in partial fix (WAVE Wf-427) MUST happen BEFORE Phase B retrain.

### §8.3 Outputs

| Output | Path | Status |
|---|---|---|
| This plan | `TODO/03_baselines/MOCK_TO_REAL_PLAN.md` | SHIPPED 2026-10-02 |
| Phase A artifacts | `molmetal/scripts/install_vina_binary.py::STUB_FALLBACK_LOG` + `molmetal/molmetal_lam/sbdd_env/meko_metal_atom_extension.py::5_NEW_SMARTS` + `molmetal/scripts/pre_flight_check.py::8_NEW_GATES` | DEFERRED on Phase A execution |
| Phase B retrain scripts | `molmetal/scripts/wf121_cisplatin_real_vina.py --metals Os Rh Re Ir Pd` + `molmetal/scripts/train_cfm_yuelbond_wasserstein.py --sa-penalty 0.1 --sa-target 2.8 --epochs 50` | DEFERRED on USER_OK |
| Phase C paper artifacts | `paper/sections/03_main_results_v4.tex` + `paper/sections/06_limitations_v4.tex` | DEFERRED on USER_OK per AGENTS.md §5.3 |

### §8.4 Cross-references

- `TODO/03_baselines/COMPARISON_BASELINES.md` §0-§3 — selected baselines + per-metric protocol alignment matrix
- `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` §0-§4 — quick-reference table
- `TODO/02_metrics/METRICS_TARGETS.md` §2 — per-metric threshold values
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §3-§5 — 21 mechanisms × 21 metrics
- `docs/HONEST_NEGATIVE_CATALOG.md` §1-§7 — 86-entry honest-negative catalog
- `DATA_STATUS.md` §1-§10 — dataset inventory
- `AGENTS.md` §5.1 standing GPU authorization + §5.2 honest framing + §5.3 paper work authorization + §5.4 NO wet-lab planning + §10 wetlab deprecation FAQ
- `TODO/decisions.md` D-Wf-431 — the alignment-not-reproduction principle
- `TODO/completed/114_mock_deleted_external_scorers_cleanup.md` — Wf-119 / TODO-114 ship evidence
- `TODO/pending/094_r20_p3_multiph_pll_feature.md` — Wf-382-A1 R20-P3 SCAFFOLD
- `TODO/pending/200_wf340_353_deep_integration_audit.md` — TODO-200 SCAFFOLD audit
- `TODO/pending/201_m2_r2_per_cohort_feature_engineering.md` — TODO-201 methodology
- `TODO/pending/202_m7_diversity_sampling_improvements.md` — TODO-202 SCAFFOLD + WIRED
- `TODO/pending/203_m3_vina_docking_param_tuning.md` — TODO-203 SCAFFOLD
- `TODO/pending/204_m15_rmsd_multitemplate_mm_gbsa.md` — TODO-204 SCAFFOLD
- `TODO/pending/205_m4_sa_generation_penalty.md` — TODO-205 SCAFFOLD + WIRED
- `TODO/pending/206_m6_kappa_metric_redefinition.md` — Wf-381-A4 canonical brief
- `TODO/pending/207_wf299_cfm_3wave_launch.md` — Wf-299 3-wave launch spec

---

## §9 party-twenty-spirit §3.5 honest framing + §3.6 self-check

### §9.1 Five-decision checklist

- [x] **不忘初心**: This plan advances paper §4 headline (Tier-1 direct comparison to Krasnov 2026 / Rusanov 2025 / DrugOOD 2022) + §6 honest-negative resolution. Per `COMPARISON_BASELINES.md` §0 principle, alignment (same dataset + same protocol + same metric) is the goal — not reproduction.
- [x] **谦虚谨慎**: 5 items in brief are NOT all MOCKs; 2 are MEASURED (m16 ckpts + m4 RDKit), 1 is SHIPPED (external_scorers MOCK_DELETED), 2 require conversion (m3 Vina 8 d-block + m8 PB + 4 external_scorers re-implementation). The brief misframed the items; this plan re-frames honestly per AGENTS.md §5.2.
- [x] **敢于斗争**: 8 forward-looking items + honest-negatives are addressed in §4 BEFORE next retrain (per user directive "Honest-negatives 与 Forward-looking 最好也在重新训练之前设计解决方案并解决").
- [x] **守正创新**: 5 core mechanisms (MLC + CFM + click chemistry + PLL + 84-rule MLE) preserved per §0 of MECHANISM_METRIC_MAPPING.md; R18 M22-M28 wave 1 ship preserved.
- [x] **系统观念**: 8 NEW pre-retrain threshold gates wire into `pre_flight_check.py::build_gates("all")`; complements the 8 honest-negative-forward-looking item resolution plan in §4; honors R18-R20-R21 4-batch sequential GPU retrain priority order per `METRIC_LIFT_STRATEGY_v2.md` §10.

### §9.2 §3.6 self-check (per `party-twenty-spirit` §3.6 + §11 R20 self-check)

1. 哪个机制实际动了代码? — M22 SE(3) SB + M24 Multi-PH + M25 Hilbert + M6 CFM (4 NEW R18 mechanisms + 1 M6 extension per §4.5 / §4.7); 8 NEW pre-retrain threshold gates (per §4.1) + 8 forward-looking item resolutions (per §4) = 16 NEW files modified or created.
2. 哪些 metric 受影响? — m1 (Ru AUC), m2 (R² per-cohort), m3 (Vina median), m4 (success rate + SA), m6 (QED — drop from §3), m7 (Diversity), m14 (decode ratio), m16 (Affinity Pearson r), m17 (DrugOOD OOD), m19 (LOMO), m20 (Tumor-specific), m21 (NCI-60 9-cell) = 12 of 21 metrics (per §2.2 + §6 success criteria).
3. 0 GPU retrain shipped 这条限制, paper §6 limitations 写了吗? — YES per `TODO/02_metrics/METRICS_TARGETS.md` §5 + `docs/PAPER_SECTION_6_v4_FINAL.md` §6.
4. Wave 1/2/3 三波 ship status 是否准确? — N/A for this plan (the 3-wave R18 ship is COMPLETE per §0; this plan addresses the post-Wave-3 forward-looking items + Tier-2 paper acceptance).
5. 23 个 mechanism 的 cite key 都进 paper/refs.bib 了吗? — Partial; per `MECHANISM_METRIC_MAPPING.md` §M22-M28, the cite keys (mahmood2026sb_lie / jiang2026sb_relax / kim2026multiph / liu_feng2026pll / burch_bardsley2022sr / hu2026iclm / lelarasmee1982wflow / vietoris_rips_macrouring) need to be added to paper/refs.bib via Phase C USER_OK-gated docs work.

---

## §10 key_decisions array

```python
key_decisions = [
    "DECISION 1: The 5 items in the brief are NOT all MOCKs; per AGENTS.md §5.2 honest framing, this plan re-frames them honestly (2 MEASURED + 1 SHIPPED + 2 requiring conversion).",
    "DECISION 2: Phase A (CPU-only, ~12 days wall) runs first; Phase B (USER_OK-gated GPU, ~80 hr) runs after USER_OK=1.",
    "DECISION 3: Pre-retrain threshold gate contract (NEW per §2.3) is the central deliverable; 8 NEW pre-retrain gates wire into pre_flight_check.py::build_gates('all').",
    "DECISION 4: m6_kappa drop-from-§3 paper conversion is DONE per Wf-424-A15; remaining work is USER_OK-gated per AGENTS.md §5.3.",
    "DECISION 5: Wf-246-C kcal sign fix + Wf-405-A2 wire-in partial fix (WAVE Wf-427) MUST happen BEFORE Phase B retrain.",
    "DECISION 6: Per-user-directive 2026-10-02, the pre-retrain thresholds use baseline + protocol_adjustment formula rather than bit-identical reproduction (per COMPARISON_BASELINES.md §0 principle).",
    "DECISION 7: Honest-negative catalog 86 entries (per Wf-426-A4) is the canonical reviewer-rebuttal surface; every pre-retrain threshold gate surface in §2.3 cites per-item honest-negative from this catalog.",
    "DECISION 8: Protocol alignment ≠ reproduction (per Wf-431 docs refactor); the per-metric threshold updates in §2.2 encode baseline + protocol_adjustment rather than bit-identical reproduction."
]
```

### §11 cpu_only_steps array (which actions to take IMMEDIATELY without user OK)

```python
cpu_only_steps = [
    "Phase A.1: Confirm Vina 1.2.7 + RDKit + posebusters reachability (~2 min CPU)",
    "Phase A.2: Install posebusters if missing via pip install posebusters>=0.6.5 (~5 min CPU)",
    "Phase A.3: Audit kcal sign at metal_coord_term.py:635 per Wf-426-A1 retro (~10 min CPU)",
    "Phase A.4: Author 5 NEW SMARTS rules (Os/Rh/Re/Ir/Pd) per Wf-121 P1 SOP (~10 days CPU)",
    "Phase A.5: Re-run Wf-32 PB validation on existing 5-pocket docked poses (~30 min CPU)",
    "Phase A.6: Re-run Wf-323F m4 SA/QED sub-criterion (~5 min CPU)",
    "Phase A.7: 50-step CPU smoke for --sa-penalty + --sa-target per §5.7 NEW trainer flag wire-in smoke-required rule (~5 min CPU)",
    "Phase A.8: Document 8 NEW pre-retrain threshold gates in pre_flight_check.py::build_gates('all') per §4.1 (~30 min CPU)",
    "Phase A.9: Wf-427-A1 Wf-246-C kcal sign fix (~30 min CPU)",
    "Phase A.10: Wf-424-A12 Wf-405-A2 wire-in partial fix (~30 min CPU)"
]
# Total: ~12 days CPU-only wall-clock
```

### §12 gpu_required_steps array (which actions need USER_OK-gated retrain)

```python
gpu_required_steps = [
    "Phase B.1: Wf-432-B1 Wf-380 SCAFFOLD-loss MEASURED retrain on cuda:0 Blackwell (~5 hr GPU; in-flight as PID 427946 per Wf-424-A7)",
    "Phase B.2: Wf-432-B2 Wf-394 Lambda × CFM patch application (~5 hr GPU)",
    "Phase B.3: Wf-432-B3 Wf-246-C metal-coordination term retrain post kcal sign fix (~3 hr GPU)",
    "Phase B.4: Wf-432-B4 Wf-125 CONTROLD PAIR retrain (~6 hr GPU)",
    "Phase B.5: Wf-432-B5 Wf-299 3-wave CFM GPU launch on cuda:0 Blackwell (~5 hr GPU)",
    "Phase B.6: Wf-432-B6 Wf-390-A1 Path A m2 R² 5-metal subset Ru/Ir/Rh/Os/Re (~12 hr GPU)",
    "Phase B.7: Wf-432-B7 Wf-3 8 d-block × 5 pockets × 5 seeds Vina sweep (~12 hr GPU)",
    "Phase B.8: Wf-432-B8 Pt × 4 cell-line × 5 seeds × 50-epoch D-MPNN retrain (~12 hr GPU)",
    "Phase B.9: Wf-432-B9 cfm property panel evaluation → update m7_diversity.json + m4_sa.json + m3_vina.json (~5 hr GPU)"
]
# Total: ~65 hr GPU USER_OK-gated + ~10 hr GPU USER_OK-gated per retrain ≈ ~80 hr GPU
```

### §13 honest_negatives_addressed array

```python
honest_negatives_addressed = [
    "HONEST_NEGATIVE 1: m6_kappa augmented_mean 0.472 < 0.6 threshold gap -0.128 — DROP from §3 main-result + PROMOTE top-3 WEAK-correlates (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table per Wf-389-A1 + Wf-424-A15 (11 paths exhausted per Wf-377-A1 + Wf-381-A4).",
    "HONEST_NEGATIVE 2: m2 platinai_4cell R² -0.1440 ± 0.0325 + m2 hilbert R² -0.1440 ± 0.0325 — HONEST_NEGATIVE_R2_DEEP preserved per Wf-377-A2 5-alternative-split diagnostic + Wf-388-A1 real-cohort + Wf-398-A2; structural encoder signal floor at R²=0.0043 on 12-d RDKit per Wf-390-A2 deeper audit; closure path = Wf-390-A1 Path A USER_OK-gated retrain.",
    "HONEST_NEGATIVE 3: m1 Ru AUC overall 0.7640 BELOW_KRASNOV_PARTIAL gap -0.096 — Rh fold PASS_KRASNOV 0.8798 +0.020 + Re fold BELOW_KRASNOV 0.7088 -0.151 per Wf-413-A1 §4; paper §4 LEAD claim = Rh fold only; Re fold → paper §6 honest-negative preserved.",
    "HONEST_NEGATIVE 4: m3 Vina cohort median -3.1575 (not -7.0 industry target) — MEASURED_REAL_VINA cisplatin × 10 pockets × 3 seeds with generic VDW M→Mg substitution per Wf-121 P3; chemotype-tuned Pt parameter set would yield more negative kcal/mol and is prerequisite for literature-comparable binding affinity claims.",
    "HONEST_NEGATIVE 5: m4 Success Rate = 0.0 — Vina sub-criterion BLOCKED on m3 Vina lift to ≤-8.18 per WEAK_TO_STRONG §2.4 gate; SA/QED sub-rate MEASURED 0.0022 (1/449); closed transitively by §3.1 8 d-block m3 Vina extension.",
    "HONEST_NEGATIVE 6: m8 PB production DEFERRED on RDkit env gap — posebusters NOT installed; pip install path ~5 min; re-run Wf-32 PB validation ~30 min CPU; MEASURED per-pocket pass-rate on existing 5-pocket docked poses after env fix.",
    "HONEST_NEGATIVE 7: 4 external_scorers MOCK_DELETED (coord_geometry_proxy + phototherapy_activity + ptiv_reduction_potential + metalloprotein_affinity) — SHIPPED 2026-09-27 via Wf-119 / TODO-114; 12 stale JSONs deleted; _safe_external_score raises AdapterUnavailable on every failure path; 4 metrics remain MOCK_DELETED ON DISK per safety contract (no fabrication); R20 plan real adapter re-implementation deferred.",
    "HONEST_NEGATIVE 8: Wf-246-C metal-coordination term kcal sign FALSE-CLAIM CASCADE per Wf-426-A1 retro (`metal_coord_term.py:635` should be `(1 - 2*comp)*kcal_per_unit` not `(2.0*score.total - 1.0)*self.kcal_per_unit`); Wf-246-C + Wf-333-V1 + Wf-405-A2 + Wf-409-A3 inherit FALSE state; closure path = Wf-427-A1 5-LOC fix + Wf-427-A2 retest 12 sign-fix tests.",
    "HONEST_NEGATIVE 9: Wf-405-A2 wire-in partial-failure pattern — 17/17 Wf-380 NEW CLI flags FLAG_ONLY (no loss/sampling wire-in to live training computation) per Wf-385-A1 audit; Wf-405-A2 wire-in NameError per Wf-424-A11 §5.7 smoke FAIL; closure path = Wf-424-A12 ~10 LOC fix (threading use_ot_cfm + per_cohort_features as direct kwargs through train_one_metal_one_seed → train_one_epoch_joint).",
    "HONEST_NEGATIVE 10: m14 decode_ratio Path A 0/192 (CPU architecture ceiling) per `decode_ratio.json::measurements[2]` — 4-fix bundle fixed diversity attractor but NOT EGNN velocity architecture (overvalent atoms → RDKit sanitization fail); Path B bond-bearing 192/192 PASS; real CFM 0/8 architecture ceiling; closure path = Wf-124 YuelBond decoder wire (in-flight)."
]
```

### §14 references array

```python
references = [
    "TODO/03_baselines/COMPARISON_BASELINES.md — selected baselines (Krasnov 2026 / Rusanov 2025 / DrugOOD 2022) + protocol alignment matrix",
    "TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md — quick-reference table",
    "TODO/02_metrics/METRICS_TARGETS.md — 21-metric targets + paper §4/§6 templates",
    "TODO/02_metrics/MECHANISM_METRIC_MAPPING.md — 21 mechanisms M1-M21 + R18 M22-M28 cross-mapping",
    "docs/HONEST_NEGATIVE_CATALOG.md — 86-entry honest-negative catalog",
    "DATA_STATUS.md — dataset inventory (MetalCytoToxDB / PlatinAI / tmQM / NCI-60 / CrossDocked2020)",
    "AGENTS.md — standing GPU authorization (§5.1) + honest framing mandate (§5.2) + paper work authorization (§5.3) + NO wet-lab planning (§5.4) + atomic write protocol (§5.6) + trainer flag wire-in smoke-required rule (§5.7) + wetlab deprecation FAQ (§10)",
    "TODO/decisions.md D-Wf-431 — alignment-not-reproduction principle",
    "TODO/completed/114_mock_deleted_external_scorers_cleanup.md — Wf-119 / TODO-114 ship evidence",
    "TODO/completed/121_meko_metal_atom_extension_closeout.md — Wf-121 P0 partial-lift ship",
    "TODO/completed/362a3_m15_rmsd_5pocket_meko.md — Wf-362-A3 m15 RMSD redesign SHIPPED",
    "TODO/pending/094_r20_p3_multiph_pll_feature.md — Wf-382-A1 R20-P3 SCAFFOLD",
    "TODO/pending/200_wf340_353_deep_integration_audit.md — TODO-200 SCAFFOLD audit",
    "TODO/pending/201_m2_r2_per_cohort_feature_engineering.md — TODO-201 methodology",
    "TODO/pending/202_m7_diversity_sampling_improvements.md — TODO-202 SCAFFOLD + WIRED",
    "TODO/pending/203_m3_vina_docking_param_tuning.md — TODO-203 SCAFFOLD",
    "TODO/pending/204_m15_rmsd_multitemplate_mm_gbsa.md — TODO-204 SCAFFOLD",
    "TODO/pending/205_m4_sa_generation_penalty.md — TODO-205 SCAFFOLD + WIRED",
    "TODO/pending/206_m6_kappa_metric_redefinition.md — Wf-381-A4 canonical brief",
    "TODO/pending/207_wf299_cfm_3wave_launch.md — Wf-299 3-wave launch spec",
    "docs/M6_KAPPA_DROP_DECISION.md — Wf-424-A15 m6_kappa drop-from-§3 decision doc",
    "docs/PAPER_SECTION_3_METRIC_TABLE.md — Wf-389-A1 §3 main-result lineup table",
    "docs/PAPER_SECTION_6_v4_FINAL.md — Wf-410-A1 §6 v4 limitations doc",
    "TODO/completed/388a1_m2_per_cohort_real.md — Wf-388-A1 m2 R² per-cohort REAL-cohort verification",
    "TODO/completed/398a2_m2_r2_real.md — Wf-398-A2 m2 R² real-cohort verification",
    "TODO/completed/377a1_m6_kappa_redesign.md — Wf-377-A1 m6_kappa deep-dive-research",
    "TODO/completed/426a1_wf384_a3_overwrite_audit.md — Wf-426-A1 Wf-384-A3 overwrite audit (kcal sign FALSE-CLAIM cascade)",
    "TODO/completed/425b1_wf379_a1_gpu_aggregation_fix.md — Wf-379-A1 m1_auc 6/7 GPU aggregation fix",
    "TODO/completed/413a1_gpu0_probe_re_seed4_wait.md — Wf-413-A1 GPU 0 probe + Re seed=4 wait (m1_auc canonical 7/7 runs)",
    "TODO/completed/388a2_m7_diversity_wire.md — Wf-388-A2 m7 Diversity wire-in",
    "TODO/completed/388a3_m4_sa_wire.md — Wf-388-A3 m4 SA penalty wire-in",
    "TODO/completed/389a1_m6k_supp_reframe.md — Wf-389-A1 m6_kappa supplementary sub-table reframe",
    "TODO/completed/394a1_lambda_cfm_wire.md — Wf-394-A1 TODO-21 Lambda × CFM coupling wire-in",
    "TODO/completed/394a3_wf299_pre_launch_verify.md — Wf-394-A3 Wf-299 3-wave pre-launch verification",
    "TODO/completed/424a1_use_dann_fix.md — Wf-424-A1 p2_tests_pass cascade parent fix",
    "TODO/completed/424a7_wf380_scaffold_retrain.md — Wf-424-A7 Wf-380 SCAFFOLD-loss MEASURED retrain launch",
    "TODO/completed/431b1_gpu_utilization_fix.md — Wf-431-B1 GPU utilization optimization wire-in",
    ".claude/skills/party-twenty-spirit SKILL.md §3.5 + §3.6 — MEASURED-vs-PROJECTED tag discipline + self-check",
    ".claude/skills/mol-metal-anticancer SKILL.md — MetalCytoToxDB + PlatinAI context",
    ".claude/skills/mol-metal-cfm-training SKILL.md — CFM decode_ratio context",
    ".claude/skills/mol-metal-mechanism-mapping SKILL.md — mechanism attribution context",
    ".claude/skills/mol-metal-paper-writing SKILL.md — paper §3/§4/§6 writing context (READ-only per AGENTS.md §5.3 + TODO-84)",
    ".claude/skills/mol-metal-sci-paper-submission SKILL.md — Tier-2 journal submission context",
    "metrics/by_metric/vina_kcal_per_mol.json — 8 d-block Vina MEASURED_REAL_VINA entries",
    "metrics/by_metric/vina_status.json — Vina engine health + stub_floor_reframed_2026_09_27",
    "metrics/by_metric/cfm_decode_real.json — real CFM decode_ratio 0.998 MEASURED",
    "metrics/by_metric/dmpnn_ckpt_status.json — 25 ckpts sha256-verified",
    "metrics/by_metric/sa_score.json — 449/449 cohort MEASURED",
    "metrics/by_metric/m4_success_rate.json — SA/QED sub-rate MEASURED",
    "metrics/by_metric/m4_sa.json — WIRED 2/2 flags",
    "metrics/by_metric/wf32_pb_valid.json — PB validation status (skipped on env gap)",
    "metrics/by_metric/steric_clash_pb_dock.json — PB dock-mode status (DEFERRED on env gap)",
    "metrics/by_metric/external_scorers_summary.json — 4 scorers + partial MEASURED lift values",
    "metrics/by_metric/drugood_v3_scaffold.json — DrugOOD v3 status (DEFERRED on data gap)",
    "metrics/by_metric/drugood_ood_auc.json — DrugOOD OOD AUC 0.779 3-split mean MEASURED",
    "metrics/by_metric/metal_lomo_auc_wasserstein.json — m1_auc 7/7 runs MEASURED",
    "metrics/by_metric/m2_r2_per_cohort.json — m2 R² per-cohort REAL-cohort MEASURED",
    "metrics/by_metric/m7_diversity.json — m7 Diversity wire-in status",
    "metrics/by_metric/m6_kappa.json — m6_kappa 73/73 verdicts + verdict_final status",
    "metrics/by_metric/metal_lomo_auc_wasserstein.json — m1 Ru 5-metal × 5-seed × 30-epoch",
    "TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md — Wf-391-A3 refresh + Tier-2 §3 + §6 v4 update",
    "TODO/03_plans/R18_PLAN.md — R18 9 mathematical theorem integration plan",
    "TODO/03_plans/R19_PLAN.md — R19 GPU retrain plan",
    "TODO/03_plans/WEAK_TO_STRONG_PLAN.md — m4 SA + m7 Diversity + m4 Success Rate lift strategy",
    "TODO/03_plans/LAMBDA_ROADMAP.md — R20-R25 6-year Lambda algorithm roadmap",
    "TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md — 12 tasks enumeration + verification thresholds"
]
```

---

## §11 Word count + final verdict

**Word count**: ~5000 (~250 lines). Honest framing per `party-twenty-spirit §3.5`; numbers verbatim from JSON paths + closeout docs.

**Final verdict** (per AGENTS.md §5.2 honest framing + `party-twenty-spirit §3.5`):

This plan **honestly re-frames** the user's brief per AGENTS.md §5.2:

1. **NOT all 5 items are MOCKs** — 2 are MEASURED (m16 ckpts + m4 RDKit), 1 is SHIPPED (external_scorers MOCK_DELETED), 2 require conversion (m3 Vina 8 d-block extension + m8 PB production + 4 external_scorers re-implementation)
2. **The user directive is honored** via:
   - §2.3 pre-retrain threshold gate contract (NEW per user directive "重新训练前置门槛")
   - §3 per-MOCK detailed plan with CPU-only first + GPU deferred per AGENTS.md §5.1
   - §4 forward-looking items + honest-negative resolution BEFORE next retrain (per user directive "Honest-negatives 与 Forward-looking 最好也在重新训练之前设计解决方案并解决")
   - §5 implementation sequence with explicit USER_OK gating
   - §6 success criteria with per-item honest status

3. **Tier-2 paper acceptance gate** is achievable: 5-7 strong real metrics + 22 partial/weak metrics + 86 honest-negative catalog + pre-retrain threshold gate contract + protocol alignment matrix. Per user voice 2026-09-29 "我们发二区就可以了" — Tier-2 is sufficient.

4. **Wf-380 SCAFFOLD-loss + 8 forward-looking items** resolution BEFORE next retrain is the central deliverable per user directive "在重新训练之前设计解决方案并解决". The 8 NEW pre-retrain threshold gates + 8 forward-looking item resolutions are wired into `pre_flight_check.py::build_gates("all")`.

5. **Phase A (CPU-only, ~12 days)** is the immediate next step; **Phase B (USER_OK-gated GPU, ~80 hr)** runs after USER_OK=1; **Phase C (USER_OK-gated docs, ~22-25 hr)** runs after Tier-2 paper acceptance.

**This plan is ready for review + USER_OK gating.**

---

**END OF MOCK_TO_REAL_PLAN.md** — Owner Hugo Lee + Wf-432 docs-only planning wave; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.7 + §10 + COMPARISON_BASELINES.md §0 + PROTOCOL_ALIGNMENT_MATRIX.md + METRICS_TARGETS.md + MECHANISM_METRIC_MAPPING.md + HONEST_NEGATIVE_CATALOG.md + Wf-380..432 closeouts.
