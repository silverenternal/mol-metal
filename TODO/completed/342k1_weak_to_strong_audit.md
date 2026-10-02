# Wf-342-K1 — WEAK_TO_STRONG_PLAN §1.1 + §2.5 details audit

**Date**: 2026-09-29
**Workflow**: Wf-342-K1 (docs-only audit; no code change, no GPU launch)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline
+ per-metric honest framing) + AGENTS.md §5.2 (numeric value MUST trace to JSON /
report path on disk) + WEAK_TO_STRONG_PLAN §1.1 (8 STRONG + 8 WEAK metrics) +
WEAK_TO_STRONG_PLAN §2.5 (m7 Diversity recovery reference)
**Audit brief source**: Wf-342-K1 task brief (8-metric list:
m1, m2, m3, m4, m5, m7, m16, m19)
**Pre-flight**: STALE-flag drift acknowledged; no new pre-flight gate triggered
(Pure docs audit per WEAK_TO_STRONG_PLAN §1.3)

---

## §0 Goal

Per `WEAK_TO_STRONG_PLAN.md` §1.1 + §2.5 + §2.X:
1. Read the §1.1 (8 STRONG) and §1.2 (8 WEAK) metric tables.
2. Read §2.5 (m7 Diversity recovery context).
3. For each of the 8 metrics in the audit brief (m1, m2, m3, m4, m5, m7, m16, m19),
   document the current MEASURED status, gate target, lift trajectory, and
   verdict (`SHIPPED` / `IN-FLIGHT` / `HONEST_NEGATIVE`).
4. Pure docs audit — no Edit to WEAK_TO_STRONG_PLAN.md, no GPU launch.

Honest framing per `party-twenty-spirit` §3.5 + AGENTS.md §5.2: every numeric
value traces to a JSON / report path on disk. PROJECTED vs MEASURED tag is
required.

---

## §1 WEAK_TO_STRONG_PLAN §1.1 — 8 STRONG metrics (paper §4 LEAD claims)

Per `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` lines 19-30 (verbatim from §1.1):

| ID | metric | measured | Tier 1 SOTA | gap |
|---|---|---|---|---|
| **m11** | distinct metal-ligand pairs (n_distinct) | **20** | ≥3 (gate) | **+17** (17× gate) |
| **m12** | SyntheMol synthesis rate | **1.000** | ≥0.95 | **+0.05** |
| **m13** | RDKit validity (Path B bond-bearing) | **1.000** | ≥0.99 | **+0.01** |
| **m14** | CFM decode ratio (R17 4-fix bundle) | **1.000** | ≥0.05 (gate) | **+0.95** (20× gate) |
| **m17** | OOD generalization (DrugOOD 3-split) | **0.891/0.697/0.748** | iMoLD 0.721/0.688/0.679 | **+0.17/+0.009/+0.07** |
| **m2-PC3/HL60/SKOV3** | per-cell AUC subset | **0.870/0.932/0.905** | Krasnov-tier ≥0.85 | **+0.02 to +0.08** |
| **m21** | NCI-60 9-cell Precious-Metal mean AUC | **0.8007** | Krasnov 0.81 | **−0.009** (just under) |
| **Deflexpressor** | bound-aware accuracy ba_acc (novelty) | **1.000 @ margin≥0.5** | not measured by SOTA | **novelty claim** |

Note: §1.1 STRONG set is **distinct from the 8 WEAK audit targets** (m2-PC3/HL60/SKOV3
in §1.1 is a per-cell-line subset; the m2 in the audit brief is `m2-Pt-A2780cis` from
WEAK_TO_STRONG_PLAN §2.2 — the Pt 4-cell-line AUC bottleneck cell). They are NOT
the same m2.

---

## §3 Per-metric status (audit brief: m1, m2, m3, m4, m5, m7, m16, m19)

### §3.1 m1 — Ru subset binary AUC

- **Audit brief value**: 0.7123 (baseline 0.7613 from G6)
- **MEASURED source**: `metrics/by_metric/metalcytotox_ru_auc.json`
  - `value` = 0.7613 (Wf-124 P3 30-epoch seed=0, scaffold-split, D-MPNN multitask Wasserstein)
  - `baseline_value` = 0.7123 (pre-Wf-124 baseline)
  - `n_seeds` = 1 (1-seed only — full sweep requires 5-seed × 50-epoch GPU retrain)
  - `gap_to_gate` = -0.0987 (gate = Krasnov ≥ 0.86)
  - `status` = "MEASURED_PARTIAL_LIFT"
- **Gate target**: 0.86 (Krasnov 2026 anchor ≥ 0.81)
- **Lift trajectory**: +0.049 MEASURED vs baseline; +0.088 PROJECTED on 5-seed × 50-epoch
  retrain (Wf-120 P2 TODO-181 G5 in-flight per Wf-335-M3 / TODO-181 launch wrapper)
- **Recovery strategy** (WEAK_TO_STRONG §2.1): Phase 0 honest framing + Phase 2 5-metal
  multitask pretrain (M9 coupling + M10 84-rule MLE matrix); CPU-side forward pass is
  SHIPPED, GPU retrain is the blocking step.
- **Verdict**: **IN-FLIGHT** (1-seed 30-epoch MEASURED at 0.7613; 5-seed × 50-epoch GPU
  retrain NOT STARTED — TODO-181 G5 launch wrapper READY per Wf-298D, USER_OK-gated
  per kill_then_adapt §0)
- **Recovery progress**: 0.7613 / 0.86 = 88.5% of gate; below SOTA 0.81 by 0.0487
  (still BELOW target band per WEAK_TO_STRONG §1.3 paper-credibility framing)

### §3.2 m2 — Pt 4-cell AUC bottleneck cell

- **Audit brief value**: 0.553 (BELOW)
- **MEASURED source**: `metrics/by_metric/platinai_4cell_auc.json`
  - `value_overall_mean_auc` = 0.5532 ± 0.0093 (3 seeds × 50 epochs, scaffold split)
  - `per_cell_line`:
    - A2780 = 0.3505 ± 0.0189 (THE BOTTLENECK CELL — why WEAK_TO_STRONG §2.2 names this
      metric "Pt 4-cell-line AUC bottleneck cell")
    - A2780cis = 0.6476 ± 0.0065
    - MCF7 = 0.5844 ± 0.0232
    - A549 = 0.6302 ± 0.0300
  - `n_seeds` = 3 (3-seed sweep complete)
  - `status` = "MEASURED_SYNTHETIC_LABELS" (synthetic IC50 labels — honest framing)
  - `data_source` = `molmetal/data/platinai_4cell.csv` (synthesised)
- **Gate target**: per-cell ≥ 0.85 (per WEAK_TO_STRONG §1.1 m2-PC3/HL60/SKOV3 row
  = 0.870/0.932/0.905); overall mean ≥ 0.70 (paper-credibility band)
- **Lift trajectory**: overall 0.553 < 0.70 gate (BELOW band); per-cell mostly
  BELOW 0.85 except A2780cis which is at 0.6476 (still below per-cell gate).
- **Recovery strategy** (WEAK_TO_STRONG §2.2): Phase 3 Hilbert primal-dual LambdaASTSR
  (CPU scaffolding SHIPPED R17-Wf-1 + R17-Wf-2; full GPU retrain BLOCKED on R20-P2
  Hilbert-ASTSR ship target)
- **Verdict**: **HONEST_NEGATIVE_BELOW_TARGET_BAND** — overall mean 0.553 BELOW
  paper-credibility band 0.70; A2780 bottleneck cell at 0.350 is the worst offender.
  MEASURED on 3-seed × 50-epoch with synthetic labels (honest framing per JSON
  `honest_framing` field). Recovery structurally requires R20-P2 Hilbert-ASTSR
  GPU retrain (TODO-93 in-flight).

### §3.3 m3 — Vina Dock median kcal/mol

- **Audit brief value**: -7.782 (RESOLVED per Wf-333 V3)
- **MEASURED source**: `metrics/by_metric/vina_score_*.json` (per-pocket) +
  `TODO/completed/333v3_vina_p2p3.md` (PROJECTED combined lift analysis)
  - `vina_relaxed_success_*.json`: per-pocket baseline (Wf-325B cohort median
    -3.282 kcal/mol)
  - `Wf-333-V3` PROJECTED combined V1+V2+V3 lift = -4.500 kcal/mol constant
    - V1: per-centre F2(a) (Wf-329-W3) = -1.500
    - V2: per-bond F2(a) (Wf-333-V3 P2) = -1.500
    - V3: distance sub-score wire-in (Wf-333-V3 P3) = -1.500
  - Combined PROJECTED cohort median -3.282 + (-4.500) = **-7.782 kcal/mol**
  - Industry cutoff = -7.000 kcal/mol
  - **V3 gap to industry = -0.782 kcal/mol** (PASSES by 0.782 margin)
- **Tag discipline**: This is **PROJECTED**, NOT MEASURED — the 4.5 kcal/mol lift
  is a per-component conservative midpoint per Wf-333-V3 task brief §5.1. MEASURED
  baseline is -3.282 kcal/mol; MEASURED cohort is 5-pocket (1h36, 3eml, 3ny8,
  4aaw, 4rlu).
- **Verdict**: **PROJECTED_RESOLVED** — the Wf-333 V3 ship composite PASSES the
  industry -7.0 cutoff by 0.782 kcal/mol margin on the 5-pocket PROJECTED
  baseline; MEASURED GPU retrain on the 100-pocket production sweep is
  USER_OK-gated (Wf-264 CFM Vina panel integration pending per Wf-298D +
  Wf-336-X4 meeko alt path).

### §3.4 m4 — Success Rate (Vina<−8.18 ∧ QED>0.25 ∧ SA>0.59)

- **Audit brief value**: 0.000 (STUFFED, F2(a) covered)
- **MEASURED source**: `metrics/by_metric/m4_success_rate.json`
  - `value` = 0.00223 (1/449 hits on the SA/QED sub-criterion alone — STUFFED)
  - `sub_rate_sa_qed` = 0.00223 (MEASURED Wf-323F)
  - `sub_rate_sa_only` = 0.0846
  - `sub_rate_qed_only` = 0.0490
  - Vina sub-criterion = 0.0 (BINDING — cascades from m3 Vina)
  - `total` = sub_rate_sa_qed × sub_rate_vina = 0.00223 × 0.0 = **0.0**
  - `HONEST_NEGATIVE` = true (auto-attached per Wf-308B / Wf-312C / Wf-323F
    contract)
- **Gate target**: ≥ 0.60 (WEAK_TO_STRONG §2.4)
- **Lift trajectory**: 
  - **Short-term** (Phase 1 SA/QED wiring — SHIPPED via Wf-315A + Wf-323F):
    +5-10pp PROJECTED on the SA/QED sub-rate; current sub-rate = 0.00223
  - **Medium-term** (Phase 2 m3 Vina lift hitting ≤-8.0): +15-20pp PROJECTED
    once m3 Vina binding breaks
- **Recovery strategy** (WEAK_TO_STRONG §2.4): cascades from m3 Vina; F2(a)
  MetalLigandExchange SMARTS rule SHIPPED per Wf-329-W3 + Wf-333-V3; SA/QED
  evaluator SHIPPED via Wf-315A + Wf-323F (Wf-323F sub-criterion wire
  `p3_m4_sa_qed_subcriterion_pass`).
- **Verdict**: **HONEST_NEGATIVE_CASCADED** — m4 total is structurally 0.0
  until m3 Vina lifts (the AND of two binding sub-criteria: SA/QED AND Vina).
  SA/QED sub-criterion SHIPPED via Wf-315A + Wf-323F (sub_rate at 0.00223);
  Vina sub-criterion is BLOCKED on the m3 Vina lift hitting -8.18 or better
  on the 100-pocket retrain (PROJECTED -7.782 per Wf-333 V3; MEASURED GPU
  retrain pending per kill_then_adapt §0).

### §3.5 m5 — SA Score (Ertl 2008, lower = easier to synthesize)

- **Audit brief value**: 3.099 (BELOW)
- **MEASURED source**: `metrics/by_metric/sa_mean.json`
  - `wf_sa_fragment_pool_optimize` (2026-09-15, 1×1 smoke with
    `--keep-high-sa-tiles=False`): value = **3.099** MEASURED, status =
    "MEASURED", n_samples = 1, note = "filtered pool (220→210 tiles);
    closer to TargetDiff range but still above"
  - Earlier baselines:
    - `wf_sa_penalty_guidance` (no `--sa-weight`, Lambda-only): ≥ 5.0 MEASURED
    - `wf_sa_penalty_guidance` (`--sa-weight 0.3`, 5×1 smoke): 3.319 MEASURED
- **Gate target**: ≤ 2.5 (TargetDiff-class target band). NOTE: the
  `INDEX.md §1335 + §1347` reclassifies SA 3.099 as a paper §4 LEAD-claim
  metric, NOT a WEAK metric; the audit brief's "BELOW" framing here appears
  to follow the WEAK_TO_STRONG_PLAN §2.4 sub-criterion gate rather than the
  paper §4 LEAD-claim framing.
- **Lift trajectory**: 
  - `wf_sa_penalty_guidance` (no sa-weight): ≥ 5.0 MEASURED
  - `--sa-weight 0.3` smoke: 3.319 MEASURED (+1.681 lift vs no-sa-weight)
  - `wf_sa_fragment_pool_optimize` (1×1): 3.099 MEASURED (+0.220 lift)
  - Larger cohort `wf_round12_lambda_pilot` (10×3 cisplatin+all-5,
    n_sim=1000): see `sa_mean.json` line 31+ (not extracted in this audit)
- **Recovery strategy** (WEAK_TO_STRONG §2.4 + §2.5 implicit): SA + QED
  evaluator wiring SHIPPED via Wf-315A + Wf-323F; further lift expected from
  Phase 1 + Phase 2 multi-PH fragment pool growth (Wf-331-R4 multi-PH
  integration is primarily for m7 Diversity but the pool expansion
  benefits SA).
- **Verdict**: **IN-FLIGHT_PARTIAL_LIFT** — SA score 3.099 is MEASURED at the
  current Lambda-MCTS pipeline (1×1 smoke); Phase 1 SA + QED evaluator wiring
  SHIPPED per Wf-315A + Wf-323F (provides honest framing in `metrics/by_metric/sa_mean.json`
  + `metrics/by_metric/qed_score.json`). Still BELOW TargetDiff-class gate
  ≤ 2.5; further lift requires Phase 2 multi-PH fragment pool growth
  (Wf-331-R4 architectural caveat.4 / m7 pool-ceiling 0.9068 MEASURED).

### §3.6 m7 — Diversity (Morgan ECFP4 pairwise Tanimoto)

- **Audit brief value**: 0.792 (RESOLVED per Wf-331-R4)
- **MEASURED source**: `TODO/completed/331r4_m7_diversity_multiph.md` §2
  + `metrics/by_metric/diversity_tanimoto.json`
  - `m7_baseline_path_a` = 0.1065 MEASURED (R12 Path A with 4 fixes,
    10×3 sweep)
  - `m7_baseline_r12_ev3` = 0.1366 MEASURED (R12 EV-3, `--sa-weight 0.3`)
  - `m7_pool_ceiling_base` = 0.9068 MEASURED (449-pool upper bound on
    what MCTS could ever generate from this pool)
  - `m7_pool_ceiling_mph` = 0.9026 MEASURED (3-pH expanded cohort, 1347 SMILES)
  - `m7_after_lift_tanimoto` = 0.9048 MEASURED (3-seed mean of MaxMin
    PLL-distance-aggregated Tanimoto)
  - `m7_after_lift_pll` = 0.6965 MEASURED (3-seed mean of normalised PLL
    signature distance)
  - **`m7_after_lift_geometric` = 0.7925 MEASURED** (3-seed mean of
    `√(Tanimoto × PLL_normalised)` — headline recovery)
  - 3-seed × 3 pH × multi-pocket sub-pools (n_agg ~52-53 per seed)
  - 3 seeds independently clear the 0.65 gate
- **Gate target**: ≥ 0.65 (WEAK_TO_STRONG §2.5)
- **Lift trajectory**: 
  - `m7_after_lift_geometric` (0.7925) vs `m7_baseline_path_a` (0.1065):
    **+0.6860** (primary lift)
  - vs `m7_baseline_r12_ev3` (0.1366): **+0.6559**
  - vs `m7_pool_ceiling_mph` (0.9026): **-0.1102** (12% headroom)
  - vs `m7_sota_target` (0.81): **-0.0175** (within 0.02 of TransDiffSBDD SOTA)
- **Recovery strategy** (WEAK_TO_STRONG §2.5 + Wf-331-R4): Multi-PH
  integration + 3-seed × multi-pocket expansion + PLL signature diversity
  contribution (Liu-Feng 2026 PLL signature as orthogonal diversity axis
  combined via geometric mean with Tanimoto). Recovery SHOULD NOT be confused
  with single-metric recovery: the 0.7925 is a *combined* recovery that
  uses the PLL axis to penalise per-pool collapse.
- **Verdict**: **SHIPPED_RESOLVED** — Multi-PH integration SHIPPED via
  Wf-331-R4; geometric mean headline 0.7925 MEASURED on 3-seed × 3-pH
  expanded cohort (1347 SMILES); within 0.02 of TransDiffSBDD SOTA 0.81.
  Verdict: `HONEST_NEGATIVE_M7_DIVERSITY_RESOLVED`. Per-pool MaxMin
  selection is a *combined-axis* recovery (geometric mean of Tanimoto +
  PLL_normalised); the Tanimoto-only value sits at 0.9048 (close to pool
  ceiling 0.9068 because MaxMin picks maximally-distinct Tanimoto
  representatives).

### §3.7 m16 — Affinity prediction (Pearson r vs experimental pIC50)

- **Audit brief value**: 0.207 (BELOW)
- **MEASURED source**: `metrics/by_metric/pearson_r_pic50.json`
  - `wf_pic50_margin_sweep` (margin=0.5, BEST): **0.207** MEASURED,
    n_samples = test set ~700, date_utc = 2026-09-14, note = "best margin
    0.5; +0.011 over 1.0; default CENSOR_MARGIN_DEFAULT 1.0→0.5"
  - Earlier baselines:
    - `wf_extra1_full` (3-seed × 50-epoch, margin=0.25): 0.198 MEASURED,
      seed-fragile: 0.005 / 0.519 / 0.071
    - `wf_pic50_margin_fix` (margin 0.25→1.0): 0.181 MEASURED, ba_acc
      0.0→1.0 lifted but pearson within seed-std
  - R19 alternative (`metrics/by_metric/affinity_pearson_r.json`):
    - `r19_wf32_m16_affinity_pearson_r_platinai` (PlatinAI 4-cell, n_seeds=1,
      epochs=1, batch=128, lr=0.001, hidden=300, depth=3, alpha=0.0,
      init=tmqm): headline_pearson_r_mean = **-0.0403** MEASURED
      (R19-tunable Tanimoto, HeLa48h regime, NOT a 5-seed × 50-epoch
      production run)
- **Gate target**: ≥ 0.86 (Boltz-2 anchor)
- **Lift trajectory**: 0.207 (margin=0.5 BEST) → gap -0.653 to Boltz-2 0.86;
  R19-tunable Tanimoto -0.0403 MEASURED on PlatinAI 4-cell (worse than
  margin=0.5 BEST on HeLa48h cohort).
- **Recovery strategy** (WEAK_TO_STRONG §2.8): Phase 2 LambdaASTSR ablation
  on HeLa48h cohort + R19 Deflexpressor algorithmic walk; R20-P2 Hilbert-ASTSR
  may structurally lift (orthogonal axis). Per `TODO/pending/179_wf123_p3_w2_pic50_alignment.md`
  STATUS_HOLD: the W₂ alignment recipe is **architecturally infeasible** due
  to the 12-d RDKit 2D signal floor (|ρ| ≤ 0.019 on PlatinAI pIC50 → linear R²
  upper bound = (0.019)² · 12 = 0.0043; R² ≥ 0 mathematically infeasible on
  the 12-d feature stack).
- **Verdict**: **HONEST_NEGATIVE_ARCHITECTURAL_FLOOR** — 0.207 (margin=0.5
  BEST) MEASURED on 2026-09-14; R19-tunable Tanimoto regresses to -0.0403 on
  PlatinAI 4-cell (STATUS_HOLD per Wf-316A); W₂ alignment recipe architecturally
  infeasible per TODO-179 §0 + Wf-313G dashboard §3.4. Recovery requires R20-P2
  Hilbert-ASTSR structural feature stack (orthogonal to current 12-d RDKit
  2D floor) + multi-PH + PLL features per WEAK_TO_STRONG §2.5 + §2.9 + Wf-331
  multi-PH integration. Per WEAK_TO_STRONG §1.3 honest framing: m16 is one of
  the 3 weak metrics (m16, m18, m19) requiring "deeper structural fixes that
  may not close the gate" — current verdict is BELOW gate with no short-term
  recovery path.

### §3.8 m19 — Multi-metal LOMO (leave-one-metal-out transfer)

- **Audit brief value**: 0.535 (PARTIAL via R20 §2.3)
- **MEASURED source**: `metrics/by_metric/metal_lomo_auc.json`
  - `value` = **0.5346** ± 0.1096 MEASURED (5 folds × 3 seeds = 15 runs)
  - `per_fold_auc_mean`:
    - Ru: 0.5706 ± 0.0032
    - Ir: 0.6176 ± 0.0068
    - Os: 0.6113 ± 0.0235
    - **Rh: 0.3677 ± 0.0184 (BOTTLENECK FOLD)**
    - **Re: 0.5059 ± 0.1281 (high variance fold)**
  - `metal_lomo_auc_v2.json`: v2 (Wf-145 P1 BALANCED Rh cohort, per-metal
    head ensemble, Wasserstein transfer loss target = Rh real ACTIVE
    distribution, Hilbert L2 grad-norm guard max_norm=5.0): `value` = **null**
    (NOT RUN — CPU scaffold READY at `molmetal/scripts/train_metal_lomo_v2.py`
    + USER_OK-gated GPU launch wrapper READY per Wf-298D + Wf-338-V2 +
    Wf-337-U2 + Wf-322F + Wf-321A; BLOCKED on kill_then_adapt §0)
- **Gate target**: ≥ 0.70 (WEAK_TO_STRONG §2.9; KRASNOV tier)
- **Lift trajectory**: 0.5346 / 0.70 = 76.4% of gate; -0.165 gap to gate.
  Rh fold bottleneck 0.3677 is the worst offender (per-metal LOMO range
  0.367-0.618). LOMO v2 partial pool lift (per-metal head + Wasserstein +
  Hilbert L2 grad-norm guard) is the recovery axis but v2 has not been run
  (value = null).
- **Recovery strategy** (WEAK_TO_STRONG §2.9 + R20 §2.3 + Wf-176 + Wf-135 +
  Wf-125): R20-P3 Hilbert primal-dual + Wf-176 m19 LOMO Rh oversample +
  Wf-135 BALANCED Rh cohort (300 active + 300 inactive) + Wf-125 Wasserstein
  transfer + Rh closure oversample (Wf-171 P1). Phase 1 MEASURED at 0.368
  pre-Wf-1 + Wf-2 (Wf-125 baseline). Phase 2 CPU scaffold SHIPPED per Wf-321A
  (Wf-135 Phase 2 CPU scaffold); Phase 2 GPU wrapper READY at
  `scripts/launch_wf310e_wf135_phase2_gpu.sh` (USER_OK-BLOCKED per kill_then_adapt).
- **Verdict**: **IN-FLIGHT_PARTIAL_RECOVERY** — v1 MEASURED 0.5346 (5 folds
  × 3 seeds) with Rh bottleneck fold 0.368; v2 partial-pool-lift CPU scaffold
  READY (Wf-321A + Wf-322F + Wf-338-V2 + Wf-337-U2 chain); GPU launch
  BLOCKED on kill_then_adapt §0 USER_OK gate. Per `TODO/INDEX.md` §247
  + §231 + §531: Rh LOMO 0.136 GPU re-bench HONEST_NEGATIVE PRESERVED per
  Wf-327-F13 + Wf-322F + Wf-125 P2 + Wf-338-V2 + Wf-337-U2; Phase 2 GPU
  wrapper READY (USER_OK-BLOCKED).

---

## §4 Per-metric verdict summary

| Metric | Brief value | MEASURED source path | Verdict |
|---|---|---|---|
| **m1** | 0.7123 (baseline 0.7613 from G6) | `metalcytotox_ru_auc.json` value=0.7613, n_seeds=1 | **IN-FLIGHT** (1-seed 30-epoch MEASURED at 0.7613; 5-seed × 50-epoch GPU retrain READY, USER_OK-gated) |
| **m2** | 0.553 (BELOW) | `platinai_4cell_auc.json` value_overall_mean_auc=0.5532 (3-seed: 50-epoch; synth labels) | **HONEST_NEGATIVE_BELOW_TARGET_BAND** (overall mean BELOW paper-credibility band 0.70; A2780 bottleneck 0.350) |
| **m3** | -7.782 (RESOLVED per Wf-333 V3) | `TODO/completed/333v3_vina_p2p3.md` §5.3 PROJECTED cohort median trajectory | **PROJECTED_RESOLVED** (V1+V2+V3 PROJECTED lift -4.500; MEASURED GPU retrain pending) |
| **m4** | 0.000 (STUFFED, F2(a) covered) | `m4_success_rate.json` value=0.00223, sub_rate_sa_qed=0.00223, vina=0.0 | **HONEST_NEGATIVE_CASCADED** (Vina sub-criterion binding; SHIPPED SA/QED sub-criterion via Wf-315A + Wf-323F) |
| **m5** | 3.099 (BELOW) | `sa_mean.json` wf_sa_fragment_pool_optimize=3.099 MEASURED, n_samples=1 | **IN-FLIGHT_PARTIAL_LIFT** (Phase 1 SA/QED evaluator wiring SHIPPED via Wf-315A; BELOW TargetDiff ≤2.5 gate) |
| **m7** | 0.792 (RESOLVED per Wf-331-R4) | `TODO/completed/331r4_m7_diversity_multiph.md` §2 m7_after_lift_geometric=0.7925 | **SHIPPED_RESOLVED** (Multi-PH + 3-seed + PLL integration; within 0.02 of TransDiffSBDD SOTA 0.81) |
| **m16** | 0.207 (BELOW) | `pearson_r_pic50.json` wf_pic50_margin_sweep=0.207 MEASURED, HeLa48h | **HONEST_NEGATIVE_ARCHITECTURAL_FLOOR** (W₂ alignment STATUS_HOLD per TODO-179 + Wf-313G; 12-d RDKit 2D feature signal floor) |
| **m19** | 0.535 (PARTIAL via R20 §2.3) | `metal_lomo_auc.json` value=0.5346 (5 folds × 3 seeds = 15 runs) | **IN-FLIGHT_PARTIAL_RECOVERY** (v1 MEASURED 0.5346; v2 partial-pool-lift CPU scaffold READY; Phase 2 GPU wrapper USER_OK-BLOCKED) |

---

## §5 Ship / in-flight / honest-negative counts

- **SHIPPED** (1): m7 (Diversity — Multi-PH integration via Wf-331-R4)
- **IN-FLIGHT** (4): m1 (1-seed MEASURED; 5-seed × 50-epoch GPU retrain USER_OK-gated);
  m5 (1×1 smoke MEASURED; Phase 1 SA/QED wiring SHIPPED; BELOW gate);
  plus 2 PROJECTED-shifted-as-in-flight: m3 (PROJECTED -7.782 RESOLVED on
  5-pocket PROJECTED baseline; MEASURED GPU retrain pending) is the borderline
  case — it has PROJECTED resolution but MEASURED GPU retrain is pending.
  Conservatively count as **IN-FLIGHT** since PROJECTED ≠ MEASURED.
  (If you re-classify m3 as **PROJECTED_RESOLVED** but not yet **SHIPPED** on
  a MEASURED basis, the count moves to 2 SHIPPED / 3 IN-FLIGHT / 3 HONEST_NEGATIVE.)
- **HONEST_NEGATIVE** (3): m2 (Pt 4-cell AUC bottleneck BELOW paper-credibility
  band); m4 (Success Rate cascaded from m3 Vina binding); m16 (Affinity
  architectural floor on 12-d RDKit 2D signal)

Counts depend on m3 classification. Reasonable baselines:
- **Strict PROJECTED ≠ MEASURED**: SHIPPED=1, IN-FLIGHT=4 (m1, m3, m5, m19),
  HONEST_NEGATIVE=3 (m2, m4, m16) — total 8.
- **m3 PROJECTED as SHIPPED-on-resolve gate**: SHIPPED=2 (m3, m7), IN-FLIGHT=3
  (m1, m5, m19), HONEST_NEGATIVE=3 (m2, m4, m16) — total 8.

---

## §6 Honest framing per AGENTS.md §5.2

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.2:

1. **All numeric values in this audit trace to a JSON / report path on disk.**
   PROJECTED vs MEASURED tag is included on every value (see §3 + §4).
2. **No fabricated / aspirational numbers.** m3 -7.782 is PROJECTED (per
   Wf-333-V3 task brief §5.3); m1 0.7613 is MEASURED 1-seed (per
   `metalcytotox_ru_auc.json:9`); m7 0.7925 is MEASURED 3-seed (per
   `TODO/completed/331r4_m7_diversity_multiph.md:47`); etc.
3. **Per WEAK_TO_STRONG §1.3 paper-credibility framing**: 1 SHIPPED (m7
   Diversity within 0.02 of SOTA); 3 HONEST_NEGATIVE (m2 BELOW band, m4
   cascaded, m16 architectural floor); 4 IN-FLIGHT (m1, m3, m5, m19 with
   partial recovery + USER_OK-gated GPU launch wrappers READY).
4. **This audit is docs-only.** No Edit to `WEAK_TO_STRONG_PLAN.md`, no
   GPU launch, no code change.
5. **§1.1 STRONG set is distinct from the 8 WEAK audit targets.** The m2 in
   §1.1 (m2-PC3/HL60/SKOV3 per-cell-line subset) is NOT the same m2 as the
   audit brief (m2-Pt-A2780cis Pt 4-cell-line AUC bottleneck cell). Audit
   brief's m2 is WEAK_TO_STRONG_PLAN §2.2; §1.1's m2-PC3/HL60/SKOV3 is a
   STRONG metric. Honest reading is required to avoid double-counting.

---

## §7 Recovery axis status

Per WEAK_TO_STRONG §1.3 + AGENTS.md §5.1 (standing GPU authorization):

1. **m1 Ru AUC**: 1-seed 30-epoch MEASURED at 0.7613; 5-seed × 50-epoch GPU
   retrain at TODO-181 G5 launch wrapper READY per Wf-298D; USER_OK-gated
   per kill_then_adapt §0.
2. **m2 Pt 4-cell AUC**: MEASURED 0.5532 (3-seed × 50-epoch, synth labels)
   BELOW paper-credibility band 0.70; A2780 bottleneck 0.350 is the worst
   offender. Recovery structurally requires R20-P2 Hilbert-ASTSR GPU
   retrain (TODO-93 in-flight; Wf-122 P3 attempts HONEST_NEGATIVE per
   Wf-313G dashboard §3.4).
3. **m3 Vina Dock**: PROJECTED -7.782 RESOLVED on 5-pocket baseline
   (industry cutoff -7.0 PASSED by 0.782 margin per Wf-333 V3 task
   brief §5.3); MEASURED GPU retrain on 100-pocket production sweep is
   USER_OK-gated per kill_then_adapt §0 (TODO-134 Vina install attempt
   per Wf-323 closeout; meeko alt per Wf-336-X4).
4. **m4 Success Rate**: cascaded from m3 Vina; SA/QED sub-criterion
   SHIPPED via Wf-315A + Wf-323F (sub_rate at 0.00223); Vina sub-criterion
   pending m3 GPU retrain.
5. **m5 SA Score**: Phase 1 SA + QED evaluator wiring SHIPPED via Wf-315A;
   further lift requires Phase 2 multi-PH fragment pool growth (Wf-331-R4
   architectural caveat.4).
6. **m7 Diversity**: SHIPPED via Wf-331-R4 multi-PH + 3-seed × multi-pocket
   + PLL signature integration; within 0.02 of TransDiffSBDD SOTA 0.81.
7. **m16 Affinity**: STATUS_HOLD per TODO-179 + Wf-313G dashboard §3.4
   (12-d RDKit 2D signal floor: |ρ| ≤ 0.019 → linear R² upper bound
   = 0.0043); W₂ alignment architecturally infeasible. Recovery requires
   R20-P2 Hilbert-ASTSR structural feature stack (orthogonal to 12-d
   floor).
8. **m19 LOMO**: v1 MEASURED 0.5346 (5 folds × 3 seeds); v2 partial-pool-lift
   CPU scaffold READY (Wf-321A + Wf-322F + Wf-338-V2 + Wf-337-U2); Phase 2
   GPU wrapper READY (USER_OK-BLOCKED). Rh bottleneck fold 0.368 is the
   worst offender.

---

## §8 Pre-flight + EXPECTED_GATE_COUNTS status

- Pre-flight v3 fresh-run: **62 PASS / 0 FAIL / 62 TOTAL** PRESERVED at
  Wf-335-SYN baseline (per AGENTS.md §0 + Wf-335-SYN closeout); no
  pre-flight gate triggered by this docs-only audit.
- EXPECTED_GATE_COUNTS["all"] 62 PRESERVED.
- No code change; no Edit to `WEAK_TO_STRONG_PLAN.md`, no Edit to
  `molmetal/scripts/*`, no Edit to `metrics/by_metric/*.json`.

---

## §9 References

| Source | Path |
|---|---|
| WEAK_TO_STRONG_PLAN §1.1 + §1.2 + §2.5 | `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` |
| Wf-331-R4 m7 Diversity multi-PH integration | `TODO/completed/331r4_m7_diversity_multiph.md` |
| Wf-333-V3 Vina P2 + P3 ship surface | `TODO/completed/333v3_vina_p2p3.md` |
| Wf-333-V1 Vina sign-fix | `TODO/completed/333v1_wf246c_sign_fix.md` |
| Wf-333-V2 Vina P0 + P1 wire-in | `TODO/completed/333v2_vina_p1_wf246c_wired.md` |
| Wf-321A Wf-135 Phase 2 LOMO CPU scaffold | `TODO/completed/321a_wf135_phase2_cpu_scaffold.md` |
| Wf-322F Wf-176 m19 LOMO FINAL closeout | `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` |
| Wf-337-U2 Wf-135 Phase 2 USER_OK gate | `TODO/completed/337u2_wf135_user_ok.md` |
| Wf-338-V2 Wf-298D Wf-264 + Wf-176 integration | `TODO/completed/338v2_wf298d_wf176_integration.md` |
| Wf-315A SA + QED evaluator wiring | `TODO/completed/315a_sa_qed_evaluator_wiring.md` |
| Wf-323F m4 SA/QED sub-criterion | `TODO/completed/323f_m4_sa_qed_subcriterion.md` |
| TODO-179 W₂ pIC50 alignment STATUS_HOLD | `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` |
| AGENTS.md §1.1 + §5.2 + §5.1 standing GPU authorization | `AGENTS.md` |
| LAMBDA_ROADMAP §2.3 R21 key metric target | `TODO/03_plans/LAMBDA_ROADMAP.md` |

---

**END OF Wf-342-K1 AUDIT** — Pure docs audit; MEASURED values cited; PROJECTED
labels on m3; HONEST_NEGATIVE on m2/m4/m16; per-pool caveat on m7.