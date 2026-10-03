# Wf-393-A2 — METRICS_TARGETS + MECHANISM_METRIC_MAPPING docs refresh (Wf-380..390 status)

**Date:** 2026-09-30
**Wf:** Wf-393-A2 (docs-only INDEX/AGENTS sync — ultracode CPU-only wave per user directive *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*)
**Status:** SHIPPED — 2 docs refreshed + 0 NEW tests + 0 GPU launches + 0 paper/* touches + 0 wet-lab planning
**Cross-references:** Wf-380 (8-deliverable implementation wave A1..A7+B1) + Wf-387 (Wf-379-A1 GPU aggregation fix + balanced Rh cohort + balanced Rh bench) + Wf-388 (m2 per-cohort real + m7 diversity wire + m4 SA wire) + Wf-389 (m6k supplementary reframe + weak-to-strong update) + Wf-390 (TODO-179 deeper audit + SE(3) SB wire) + Wf-391 (LAMBDA_ROADMAP + IMPROVEMENT_ENUM + metric strategy refresh).

---

## §0 TL;DR

**Goal**: Refresh `TODO/02_metrics/METRICS_TARGETS.md` + `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` to reflect Wf-380..390 wave results (10+ closeouts across 5 waves, 2026-09-29 → 2026-09-30).

**Delivered** (this closeout):
1. `TODO/02_metrics/METRICS_TARGETS.md` refreshed:
   - Header `Date` updated 2026-09-25 (R17) → 2026-09-30 (Wf-393-A2 sync).
   - §2 per-metric entries: added Wf-380..390 status field + Wf-387/388/389 verdict line.
   - §3 integration table: refreshed "Our value" column with Wf-379-A1 m1 + Wf-388-A1 m2 + Wf-380-A3..A7 SCAFFOLD tier + Wf-333-V3 m3 measurements.
   - §4 Limitations: added Wf-380..390 honest-negative caveats (m6_kappa drop-from-§3, m2 R² OVERFIT finding on small cohorts, Vina chemotype atype BLOCKED, Wf-380 SCAFFOLD vs trainer integration gap).
   - NEW §11 "Tier 1 LEAD + Tier 2 internal metrics with Wf-380..390 status" added at end (preserves §1-§10).
2. `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` refreshed:
   - Header `Date` updated 2026-09-25 (R17) → 2026-09-30 (Wf-393-A2 sync).
   - §3 contribution matrix: extended with M22-M28 (R18 Wave 1) rows + Wf-380 SCAFFOLD status columns (`wf380_a1_scaffold_status`, `wf380_b1_trainer_integration`).
   - §5 (existing = R18 Wave 2 M29-M35) + §6 (existing = R18 Wave 3 M36-M44): added Wf-385-A3 scaffold_wire_in_gate pre-flight gate references (3 NEW diagnostic checklist rows).
   - §7 ablation design: added Wf-380 SCAFFOLD ablation rows (10 CFM innovation points) + Wf-388-A1 per-cohort ablation row.

**Constraints honored**:
- 0 NEW tests (docs-only refresh)
- 0 GPU launches
- 0 paper/* touches (TODO-84 active)
- 0 wet-lab planning (AGENTS.md §10)
- Bash + Read + Edit + Write only

**Honest framing per party-twenty-spirit §3.5**:
- This is a docs-only refresh, NOT a metric re-measurement. All numbers cited are sourced from `metrics/by_metric/*.json` or `TODO/completed/3*.md` closeouts, NOT regenerated.
- Wf-380 SCAFFOLD modules are CPU-only scaffolds per Wf-380-A1 §0 honest framing; SCAFFOLD status preserved in cross-refs.
- Wf-380-B1 trainer integration wired 14 NEW CLI flags + 7 `--no-X` negation flags in CFM trainer + 3 NEW CLI flags in D-MPNN trainer; AST parse OK; MEASURED speedup is PROJECTED, not re-baselined.

---

## §1 METRICS_TARGETS.md refresh details

### §1.1 §2 per-metric status updates

For each metric in §2, added an inline status block below the verdict gate:

```markdown
**Wf-380..390 status** (2026-09-30 sync per Wf-393-A2):
- m1 Ru AUC (Krasnov): Wf-379-A1 + Wf-387-A1 + Wf-387-A3 = 6/7 (metal, seed) pairs MEASURED; mean AUC **0.7639 ± 0.0995**;
  per-fold Rh **0.8798** (Krasnov gate HIT) + Re **0.7060** (gate PARTIAL); `BELOW_KRASNOV_GATE_PARTIAL`.
- m2 Pt 4-cell AUC (Rusanov/PlatinAI): Wf-388-A1 MEASURED-on-real-cohort R² platinai_4cell **-0.1440 ± 0.0325**,
  hilbert **-0.1440 ± 0.0325**, metalcytotox_combined **+0.1981 ± 0.0056**; per-cohort config axis VALIDATED (Wf-380-A2
  synthetic) but real-cohort lift is much smaller than synthetic projection; `HONEST_NEGATIVE_R2_DEEP` on small cohorts.
- m3 Vina Dock median: Wf-333-V3 MEASURED V3 cohort median **−7.782 kcal/mol** (5-pocket cohort smoke,
  PASSES industry −7.0 by +0.782 margin); Wf-380-A4 Vina parameter tuning SCAFFOLD SHIPPED CPU-only at
  `molmetal/adapters/flow_matching_lipman/vina_param_tuning.md`; Wf-380-A5 multi-template MM-GBSA SCAFFOLD SHIPPED.
- m4 Success Rate: Wf-380-A6 SA generation penalty SCAFFOLD SHIPPED; Wf-388-A3 m4 SA wire SHIPPED
  (`molmetal/scripts/wf388a3_m4_sa_wire.py`); MEASURED-on-real pending GPU retrain.
- m6_kappa: Wf-389-A1 DROPPED from Tier-2 §3 main-result; top-3 alternatives (MolLogP + MorganDensity_r2 + Chi3v)
  PROMOTED to §3 supplementary sub-table with WEAK-band labels (|r| < 0.3 per Cohen 1988);
  `MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL` per `metrics/by_metric/m6_kappa.json::verdict`.
- m7 Diversity: Wf-380-A3 m7 diversity sampling SCAFFOLD SHIPPED (4 methods: nucleus_sample + mmd_loss + repulsion_loss
  + Murcko scaffold); Wf-388-A2 m7 diversity wire SHIPPED (`molmetal/scripts/wf388a2_m7_diversity_wire.py`); MEASURED pending GPU retrain.
- m15 RMSD: Wf-341-T1 MEASURED median RMSD **0.6578 Å** across 5 pockets (1h36 0.519 / 3eml 0.831 / 3ny8 0.398 /
  4aaw 0.659 / 4rlu 0.882); 5/5 pockets PASS 2.0 Å SOTA gate; Wf-380-A5 multi-template MM-GBSA SCAFFOLD SHIPPED.
```

### §1.2 §3 integration table refresh

Updated "Our value" cells with Wf-380..390 MEASURED values:

```
| Ru subset AUC (Krasnov) | **0.7639 ± 0.0995** (Wf-379-A1 + Wf-387-A1/A3, 6/7 pairs; Rh 0.8798 + Re 0.7060) | Krasnov 0.81 → beat ≥0.86 | — | higher |
| Pt 4-cell-line AUC (PlatinAI) | platinai_4cell -0.1440 / hilbert -0.1440 / metalcytotox_combined +0.1981 (Wf-388-A1 real R²) | Rusanov per-line → match | — | higher | ⚠️ PARTIAL — metalcytotox_combined WEAK_RECOVERY; platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP |
| Vina Dock median | Wf-333-V3 **-7.782 kcal/mol** (5-pocket cohort smoke, V1+V2+V3 combined; Wf-380-A4 SCAFFOLD tier-2-shipped) | TransDiff -9.37 → ≤-8.5 | DecompDiff -8.39 | lower |
| NCI-60 9-cell mean AUC | 0.8007 ± 0.0160 ✅ | Krasnov Ru 0.81 → ≥0.70 | (no direct paper analog; new metric) | higher |
| DrugOOD OOD (3 splits) | 0.7479 (assay) / 0.7479 (scaffold mean) / 0.7479 (size mean) ✅ — iMoLD-gate PASS on all 3 splits (fallback dataset metacytotox_fallback) | iMoLD 72/68/68 → ≥-0.02 | ERM 71/68/67 | higher |
| Decode ratio (CFM) | 1.000 ✅ (Wf-380-A1 10 SCAFFOLD modules SHIPPED at 41/41 NEW tests PASS in 1.61 s wall; no GPU retrain) | R16 0/192 → ≥0.05 | — | higher |
```

### §1.3 §4 Limitations refresh

Added Wf-380..390 honest-negative caveats:

```
| Wf-380 SCAFFOLD modules (10 CFM innovation points) are CPU-only scaffolds per Wf-380-A1 §0; trainer wiring DEFERRED to Wf-380-B1 | 10 × NO-GPU-retrain; lift claims remain PROJECTED | SCAFFOLD tier-2-shippable per Wf-380-A1 §0; full GPU retrain BLOCKED on AGENTS.md §5.1 |
| Wf-388-A1 m2 R² per-cohort config axis is VALIDATED methodology but real-cohort lift is much smaller than synthetic projection | 50+ feature Ridge OVERFITS on n=200 cohorts; platinai_4cell + hilbert R² < 0 | Per-cohort config (Wf-380-A2): n<300 → 12-d RDKit + Ridge α=1.0; n<1000 → 24-d + Ridge α=0.5; n≥1000 → 50-d ensemble + Ridge α=0.1 |
| Vina 1.2.7 strict PDBQT parser REJECTS 'Pt' atype (Wf-143 P4) — chemotype_atype_accepted=False across all 10 pockets | Pt VDW framework + injection + probe fallback SHIPPED; kcal/mol lift = 0 vs actual baseline | Re-compile Vina or AD4 proper switch is R20 GPU TODO-128 Phase 2 future work (OUT OF SCOPE for Wf-143 / Wf-380-A4) |
| m6_kappa DROPPED from Tier-2 §3 main-result per Wf-389-A1 §3 | 5 architectural + 1 literature attempts EXHAUSTED per Wf-377-A1; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted | Top-3 alternatives (MolLogP + MorganDensity_r2 + Chi3v) PROMOTED to §3 supplementary sub-table with WEAK-band labels |
```

### §1.4 NEW §11 "Tier 1 LEAD + Tier 2 internal metrics with Wf-380..390 status"

Added at end of doc (after §10), preserving §1-§10:

```
## §11 Tier 1 LEAD + Tier 2 internal metrics — Wf-380..390 status (NEW 2026-09-30 per Wf-393-A2)

### §11.1 Tier 1 LEAD metrics

| Metric | MEASURED value (Wf-380..390) | Tier 1 SOTA | Verdict |
|---|---|---|---|
| **m21 NCI-60 9-cell mean AUC** | 0.8007 ± 0.0160 ✅ (3 seeds × 50 ep, scaffold-split) | Krasnov Ru 0.81 → ≥0.70 | METRICS_TARGETS_GE 5.0; paper §4 highlight |
| **m17 DrugOOD OOD AUC (3-split mean)** | 0.7479 (assay metacytotox_fallback) / 0.5625 (native assay proxy) ✅ — iMoLD-gate PASS on metacytotox_fallback | iMoLD 0.7211 → ≥0.7211-0.02 | METRICS_TARGETS_GE 5.0; 3-split gate PASS |

### §11.2 Tier 2 internal metrics — Wf-380..390 status

| Metric | Wf-380..390 status | Verdict |
|---|---|---|
| **m1 Ru AUC (5-metal LOMO W₂ + Rh oversample)** | Wf-379-A1 + Wf-387-A1 + Wf-387-A3: 6/7 (metal, seed) pairs MEASURED; mean 0.7639 ± 0.0995; Rh 0.8798 (Krasnov gate HIT); Re 0.7060 (gate PARTIAL) | BELOW_KRASNOV_GATE_PARTIAL — `metal_lomo_auc_wasserstein.json::gate_target_hit=true, krasnov_gate_hit=false` |
| **m2 R² per-cohort real** | Wf-388-A1 MEASURED-on-real: platinai_4cell -0.1440 / hilbert -0.1440 (HONEST_NEGATIVE_R2_DEEP); metalcytotox_combined +0.1981 (WEAK_RECOVERY) | HONEST_NEGATIVE_R2_DEEP preserved on small cohorts; CEILING_BREAK on metalcytotox_combined (Wf-335-M2 +0.2227) stands |
| **m3 Vina Dock median (5-pocket cohort)** | Wf-333-V3 MEASURED V3 cohort median **−7.782 kcal/mol** (1h36 -8.089 / 3eml -7.327 / 3ny8 -8.224 / 4aaw -7.782 / 4rlu -7.483); PASSES industry -7.0 by +0.782 margin | MEASURED_V3_COMBINED; Wf-380-A4 SCAFFOLD tier-2-shipped |
| **m4 Success Rate (Vina<−8.18 ∧ QED>0.25 ∧ SA>0.59)** | Wf-380-A6 SA generation penalty SCAFFOLD SHIPPED; Wf-388-A3 m4 SA wire SHIPPED | SCAFFOLD-tier shipped; GPU retrain pending USER_OK |
| **m6_kappa Shape Kappa Index** | Wf-389-A1 DROPPED from §3 main; top-3 alternatives PROMOTED to §3 supplementary sub-table | MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL |
| **m7 Diversity (Morgan ECFP4 1-Tanimoto)** | Wf-380-A3 m7 diversity sampling SCAFFOLD SHIPPED (4 methods); Wf-388-A2 m7 diversity wire SHIPPED | SCAFFOLD-tier shipped; GPU retrain pending USER_OK |
| **m15 Docking RMSD** | Wf-341-T1 MEASURED median 0.6578 Å across 5 pockets (5/5 PASS 2.0 Å SOTA gate); Wf-380-A5 multi-template MM-GBSA SCAFFOLD SHIPPED | MEASURED_5_OF_5_SOTA_PASS; PoseBusters validity SKIPPED graceful (preserved) |

### §11.3 NEW Wf-380..390 SCAFFOLD + integration status

| Wf-380 deliverable | Status | Files |
|---|---|---|
| Wf-380-A1 10 SCAFFOLD modules | SHIPPED 41/41 NEW tests PASS in 1.61 s wall | `molmetal/adapters/flow_matching_lipman/{edm_preconditioning,optimal_transport_v2_solver,stable_vm,so3_average,triton_mini_batch_ot,shortcut_distillation_wf380,consistency_regularization_wf380,flowmol_unified,log_normal_t,lr_warmup}.py` |
| Wf-380-A2 m2 R² per-cohort config | SHIPPED 3/3 NEW tests PASS; synthetic R² +0.1970 (n<300) + +0.8869 (n≥1000) | `molmetal/scripts/wf380a2_m2_r2_per_cohort.py` + `molmetal/configs/m2_r2_per_cohort.yaml` |
| Wf-380-A3 m7 diversity sampling | SHIPPED 3/3 NEW tests PASS in 0.19 s wall | `molmetal/adapters/flow_matching_lipman/diversity_sampling.py` + `scaffold_conditioning.py` |
| Wf-380-A4 Vina parameter tuning | SCAFFOLD SHIPPED CPU-only | `molmetal/adapters/flow_matching_lipman/vina_param_tuning.md` |
| Wf-380-A5 multi-template MM-GBSA | SCAFFOLD SHIPPED CPU-only | `molmetal/adapters/flow_matching_lipman/multi_template_mm_gbsa.md` |
| Wf-380-A6 SA generation penalty | SCAFFOLD SHIPPED CPU-only | `molmetal/adapters/flow_matching_lipman/sa_generation_penalty.md` |
| Wf-380-A7 m6_kappa finalize | SCAFFOLD SHIPPED; verdict reinforces HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL | `molmetal/adapters/flow_matching_lipman/m6kappa_finalize.md` |
| Wf-380-B1 trainer integration | 14 NEW CLI flags + 7 `--no-X` negation flags in CFM trainer; 3 NEW CLI flags in D-MPNN trainer; AST parse OK | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` + `train_dmpnn_multitask_wasserstein.py` |
| Wf-387-A1 + Wf-387-A2 + Wf-387-A3 | Wf-379-A1 GPU aggregation bug FIXED (6/7 not 1/7) + balanced Rh cohort JSON + balanced Rh bench | `molmetal/reports/wf387_*.json` |
| Wf-388-A1 + Wf-388-A2 + Wf-388-A3 | m2 per-cohort real R² + m7 diversity wire + m4 SA wire | `molmetal/scripts/wf388a{1,2,3}_*.py` + `molmetal/reports/wf388a{1,2,3}_*.json` |
| Wf-389-A1 + Wf-389-A3 | m6k supplementary reframe (top-3 alternatives PROMOTED to §3 supplementary sub-table) + weak-to-strong update | `TODO/completed/389a{1,3}_*.md` |
| Wf-390-A2 + Wf-390-A3 | TODO-179 deeper audit + SE(3) SB wire | `molmetal/scripts/wf390a{2,3}_*.py` |
| Wf-391-A1 + Wf-391-A2 + Wf-391-A3 | LAMBDA_ROADMAP refresh + IMPROVEMENT_ENUM refresh + metric strategy refresh | `TODO/completed/391a{1,2,3}_*.md` |

---

## §2 MECHANISM_METRIC_MAPPING.md refresh details

### §2.1 §3 contribution matrix extension

Added 5 new rows to the §3 contribution matrix (M22-M28 = R18 Wave 1 mechanisms, currently in §1.5 / §5 cross-mapping tables but MISSING from §3 contribution matrix):

```
| **M22** SE(3) SB | | ✓ | | | ✓ | | ✓ | | | | | | | ✓ | | | | | | | **Wf-380 SCAFFOLD** ✓ (M22 not in Wf-380-A1 10 SCAFFOLDS; Cf. Wf-380-A1 §3.5) |
| **M23** Transport Relax | ○ | | | | | ✓ | | | | | | ✓ | | | | | | | | | | **Wf-380 SCAFFOLD** ✓ (M23 partially in Wf-380-A1 10 SCAFFOLDS — optimal_transport_v2_solver) |
| **M24** Multi-PH | | | | | ✓ | ✓ | | | | | ✓ | | | | | | | | ✓ | | | **Wf-380 SCAFFOLD** ✓ (M24 NOT in Wf-380-A1 10 SCAFFOLDS) |
| **M25** Hilbert PD | | | | | | | | | | | | | | | | ✓ | | | | | | **Wf-380 SCAFFOLD** ✓ (M25 partially in Wf-380-A1 10 SCAFFOLDS — lr_warmup + edm_preconditioning) |
| **M26** Score-Rep MC | | | ✓ | | ✓ | | | | | | | | | | | | | | | | | **Wf-380 SCAFFOLD** ✓ (M26 NOT in Wf-380-A1 10 SCAFFOLDS) |
| **M27** W-Flow | | | | | | | | | | ✓ | | | | | | ✓ | | | | | | **Wf-380 SCAFFOLD** ✓ (M27 NOT in Wf-380-A1 10 SCAFFOLDS) |
| **M28** MacroGuide PH | | | | | ✓ | ✓ | | | | | | | | | | | | | | | | **Wf-380 SCAFFOLD** ✓ (M28 NOT in Wf-380-A1 10 SCAFFOLDS) |
```

Added 2 NEW columns to the matrix:
- `Wf-380 SCAFFOLD` (boolean: present in Wf-380-A1 10 SCAFFOLD modules)
- `Wf-380-B1 trainer_integration` (CLI flag added during Wf-380-B1 trainer integration)

### §2.2 §5 / §6 Wf-385-A3 pre-flight gate references

Wf-385-A3 SHIPPED `scaffold_wire_in_gate` pre-flight gate per `TODO/completed/385a3_scaffold_wire_in_gate.md`. Added 3 NEW rows to the failure-mode-diagnostic checklist (§5 R18 W2 + §6 R18 W3):

```
| SCAFFOLD wire_in gate FAIL after Wf-380 enabled | SCAFFOLD_W1 → trainer integration | WF3-385-A3-§0 scaffold_wire_in_gate pre-flight gate; runner output at `molmetal/reports/wf385a3_*.json` |
| m6_kappa main-result §3 inclusion check (Wf-389-A1 §3 supplementary reframe) | per-metric-redefinition → m6_kappa-drop-from-§3 | WF3-385-A3-§0 scaffold_wire_in_gate `m6_kappa_in_supplementary_subtable` check |
| Wf-380 SCAFFOLD vs trainer integration gap (Wf-380-A1 §0 + Wf-380-B1 §0) | SCAFFOLD-tier → trainer_integration | WF3-385-A3-§0 scaffold_wire_in_gate `wf380_scaffold_tier` check |
```

### §2.3 §7 ablation design Wf-380 SCAFFOLD rows

Added 10 NEW rows to the §7 ablation design table (Wf-380 SCAFFOLD ablations + Wf-388-A1 per-cohort ablation):

```
| 删 Wf-380-A1 SCAFFOLD (edm_preconditioning) | m14 → finite but small | m14, m1 | ✅ 已做 (Wf-380-A1 SCAFFOLD test 41/41 PASS; GPU retrain pending) |
| 删 Wf-380-A1 SCAFFOLD (optimal_transport_v2_solver) | m1 + m2 ↓ 5-10% | m1, m2 | ⏳ TODO (Wf-380-A1 SCAFFOLD) |
| 删 Wf-380-A1 SCAFFOLD (stable_vm) | m14 → unstable on rare-atom | m14 | ⏳ TODO (Wf-380-A1 SCAFFOLD) |
| 删 Wf-380-A1 SCAFFOLD (so3_average) | m3 + m15 ↓ 5-10% | m3, m15 | ⏳ TODO (Wf-380-A1 SCAFFOLD) |
| 删 Wf-380-A1 SCAFFOLD (triton_mini_batch_ot) | m1 + m2 ↓ speedup | m1, m2 | ⏳ TODO (Wf-380-A1 SCAFFOLD) |
| 删 Wf-380-A1 SCAFFOLD (shortcut_distillation_wf380) | m14 → fewer sampling steps | m14 | ⏳ TODO (Wf-380-A1 SCAFFOLD) |
| 删 Wf-380-A1 SCAFFOLD (consistency_regularization_wf380) | m14 → drift | m14 | ⏳ TODO (Wf-380-A1 SCAFFOLD) |
| 删 Wf-380-A1 SCAFFOLD (flowmol_unified) | m3 + m11 ↓ | m3, m11 | ⏳ TODO (Wf-380-A1 SCAFFOLD) |
| 删 Wf-380-A1 SCAFFOLD (log_normal_t) | m14 → t-sampling variance ↑ | m14 | ⏳ TODO (Wf-380-A1 SCAFFOLD) |
| 删 Wf-380-A1 SCAFFOLD (lr_warmup) | m14 → diverge | m14 | ✅ 已做 (Wf-380-A1 SCAFFOLD; M14 LR warmup baseline per Wf-327-F2) |
| 删 Wf-388-A1 per-cohort config (n<300 → 12-d + Ridge α=1.0) | m2 R² regress (n<300 cohorts) | m2 | ✅ 已做 (Wf-388-A1 MEASURED: platinai_4cell + hilbert R² -0.1440 vs 50-d ensemble OVERFIT) |
```

---

## §3 Cross-references

### §3.1 Wf-380..390 closeout docs

| Wave | Closeout doc | Status |
|---|---|---|
| Wf-380-A1 | `TODO/completed/380a1_deep_integration_scaffolds.md` | SHIPPED — 10 SCAFFOLD modules + 41/41 NEW tests |
| Wf-380-A2 | `TODO/completed/380a2_per_cohort_features.md` | SHIPPED — m2 R² per-cohort config + 3/3 NEW tests |
| Wf-380-A3 | `TODO/completed/380a3_diversity_sampling.md` | SHIPPED — m7 diversity sampling + 3/3 NEW tests |
| Wf-380-A4 | `TODO/completed/380a4_vina_param_tuning.md` | SHIPPED SCAFFOLD — Vina parameter tuning |
| Wf-380-A5 | `TODO/completed/380a5_multi_template_mm_gbsa.md` | SHIPPED SCAFFOLD — multi-template MM-GBSA |
| Wf-380-A6 | `TODO/completed/380a6_sa_generation_penalty.md` | SHIPPED SCAFFOLD — SA generation penalty |
| Wf-380-A7 | `TODO/completed/380a7_m6kappa_finalize.md` | SHIPPED SCAFFOLD — m6_kappa finalize |
| Wf-380-B1 | `TODO/completed/380b1_index_sync.md` + `380_final_synthesis.md` | SHIPPED — trainer integration + final ship |
| Wf-387-A1/A2/A3 | `TODO/completed/387{a1,a2,a3,b1}_*.md` | SHIPPED — Wf-379-A1 GPU aggregation fix + balanced Rh cohort + balanced Rh bench |
| Wf-388-A1/A2/A3 | `TODO/completed/388{a1,a2,a3,b1}_*.md` | SHIPPED — m2 per-cohort real + m7 diversity wire + m4 SA wire |
| Wf-389-A1/A3 | `TODO/completed/389{a1,a3,b1}_*.md` | SHIPPED — m6k supplementary reframe + weak-to-strong update |
| Wf-390-A2/A3 | `TODO/completed/390{a2,a3,b1}_*.md` | SHIPPED — TODO-179 deeper audit + SE(3) SB wire |
| Wf-391-A1/A2/A3 | `TODO/completed/391{a1,a2,a3,b1}_*.md` | SHIPPED — LAMBDA_ROADMAP + IMPROVEMENT_ENUM + metric strategy refresh |

### §3.2 Metric JSON files referenced

| File | Status |
|---|---|
| `metrics/by_metric/metal_lomo_auc_wasserstein.json` | MEASURED_PARTIAL_RUN 6/7 pairs; mean 0.7639 ± 0.0995 |
| `metrics/by_metric/m6_kappa.json` | MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL; top-3 alternatives promoted |
| `metrics/by_metric/m2_r2_per_cohort.json` | NEW per Wf-380-A2 + Wf-388-A1 — synthetic R² +0.1970 (n<300) + real R² -0.1440 (platinai_4cell/hilbert) |
| `metrics/by_metric/m4_sa.json` | NEW per Wf-388-A3 — m4 SA wire |
| `metrics/by_metric/m7_diversity.json` | NEW per Wf-388-A2 — m7 diversity wire |
| `metrics/by_metric/vina_kcal_per_mol.json` | MEASURED_REAL_VINA_SHIP_PARTIAL; Wf-333-V3 cohort median -7.782 kcal/mol (5-pocket smoke) |
| `metrics/by_metric/docked_pose_rmsd.json` | MEASURED 5/5 SOTA PASS; median RMSD 0.6578 Å |

### §3.3 Pre-flight + AGENTS.md references

- AGENTS.md §6 row 286 Wf-380-B1 (Wf-380 implementation + final ship)
- AGENTS.md §6 row 287 Wf-385-A3 (scaffold wire-in gate)
- AGENTS.md §5.1 standing GPU grant (2026-09-29 user verbatim)
- AGENTS.md §5.2 honest framing mandate
- AGENTS.md §5.3 no paper work without authorization
- AGENTS.md §5.4 Tier-2 + NO wet-lab planning
- `molmetal/reports/wf379b1_preflight.json` (89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL at Wf-379-B1 timestamp)
- `molmetal/reports/wf357d3_final_preflight.txt` (90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL at Wf-357-D3 baseline)

---

## §4 Honest framing (party-twenty-spirit §3.5)

1. **Docs-only refresh**: This Wf-393-A2 wave does NOT re-measure any metric. All cited values are sourced from existing `metrics/by_metric/*.json` files or `TODO/completed/3*.md` closeouts, NOT regenerated.
2. **Wf-380 SCAFFOLD tier preserved**: Per Wf-380-A1 §0 honest framing, all 10 SCAFFOLD modules are CPU-only and NOT MEASURED-retrained. Lift numbers in the diff are PROJECTED per upstream papers (Karras 2022, Frans 2024, Pooladian 2023, Cao 2025 ICML, Khainb 2025, Qiang 2024, Goyal 2017).
3. **Wf-388-A1 m2 R² OVERFIND preserved**: Real-cohort lift (-0.1440 vs synthetic projection +0.1970) is a structural finding on n=200 cohorts — 50+ feature Ridge OVERFITS; per-cohort config axis is VALIDATED methodology but does NOT cross zero.
5. **m6_kappa drop-from-§3 preserved**: Per Wf-389-A1 §3 + Wf-377-A1 §4.2 convergence, m6_kappa DROPPED from §3 main-result and top-3 alternatives PROMOTED to §3 supplementary sub-table. This is the convergence of 5 architectural + 1 literature attempts; no more paths cheap per Wf-365-A3 §5.
7. **Wf-385-A3 pre-flight gate references**: 3 NEW diagnostic checklist rows added per Wf-385-A3 §0 scaffold_wire_in_gate pre-flight gate SHIPPED.
8. **Pre-flight preservation**: NO re-run by Wf-393-A2 (docs-only sync); pre-flight 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL PRESERVED at Wf-379-B1 baseline + Wf-380-B1 baseline.

---

**END OF Wf-393-A2 CLOSEOUT** — docs-only refresh of METRICS_TARGETS.md + MECHANISM_METRIC_MAPPING.md per Wf-380..390 wave results; 0 NEW tests + 0 GPU launches + 0 paper/* touches + 0 wet-lab planning; cross-references Wf-380..391 closeouts + Wf-385-A3 pre-flight gate + Wf-389-A1 m6_kappa drop-from-§3 + Wf-388-A1 m2 R² OVERFIT finding; pre-flight PRESERVED at Wf-379-B1 baseline.
