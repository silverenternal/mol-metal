# TODO 177 (closed) — Wf-313G Downstream Lift Dashboard for 6 strengthen-TODO consumers

**Date**: 2026-09-29 (Wf-313G ship)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ CLOSED (aggregation dashboard SHIPPED)
**Parent**: `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md`
**Sibling**: `TODO/completed/126_innovation_strengthen_roadmap.md` §3 (master M1+M2 plan)

---

## §1 Scope (closed)

Wf-313G produced a **unified downstream lift dashboard** that aggregates the
per-consumer MEASURED state for the 6 Wf-123 P3 M1+M2 strengthen-TODO
consumers:

| TODO | Consumer metric | Status (2026-09-29) |
|---|---|---|
| TODO-119 | m14 decode_ratio | MEASURED_PARTIAL |
| TODO-120 | m1 Ru AUC | MEASURED_PARTIAL |
| TODO-121 | m19 LOMO Rh bottleneck | HONEST_NEGATIVE |
| TODO-122 | L5 Hilbert R² (4-cell) | HONEST_NEGATIVE |
| TODO-123 | m15 RMSD redocking | MEASURED_PASS |
| TODO-125 | L12 OOD per-pocket Vina | HONEST_NEGATIVE |

**Verdict**: 1/6 PASS, 2/6 PARTIAL, 3/6 HONEST_NEGATIVE. No MEASURED-only
METRIC LIFT yet (per-consumer isolated lift is per-TODO Q4 2026 / R20 GPU
exercise per `party-twenty-spirit` §3.5).

---

## §2 Dashboard artefacts SHIPPED

- **`molmetal/reports/wf313g_downstream_lift_dashboard/dashboard.json`**
  — per-consumer MEASURED/PROJECTED/HONEST_NEGATIVE table (6 rows,
  ~250 LOC schema).

- **`molmetal/tests/test_wf313g_downstream_lift_dashboard.py`**
  — 12 tests:
    - 6 per-consumer schema validation tests (one per TODO)
    - 3 lift calculation arithmetic tests (Ru AUC over baseline, Hilbert R²
      delta, m15 RMSD against 2.0 Å SOTA gate)
    - 2 status classification tests (MEASURED vs PROJECTED + status_detail
      length enforcement)
    - 1 dashboard schema validation test (top-level fields + engine-tag
      invariant + honest-negative precedent + no-best-seed selection +
      no-paper-work disclosure + user-OK-required on all pending_actions)

---

## §3 Per-consumer honest framing (party-twenty-spirit §3.5)

### §3.1 TODO-119 (m14 decode_ratio) — MEASURED_PARTIAL

- **MEASURED**: 0.9645 ± 0.0000 (1 seed × 818 samples PlatinAI pt_subset,
  CPU, pocket=1h36, jitter=0.05)
  `metrics/by_metric/cfm_decode_real.json`
- **Baseline**: 1.0 (synthetic — Wf-124 ship on Pt-click clouds)
- **Lift_abs**: -0.0354 (-3.55%)
- **Caveat**: regression_disclosure (Wf-271B) warns single-value citation
  requires sample+seed+compute disclosure. M1+M2 isolated lift NOT measured.

### §3.2 TODO-120 (m1 Ru AUC) — MEASURED_PARTIAL

- **MEASURED**: 0.7860 ± 0.0145 (Wf-278A TODO-181 G5 50-epoch
  re-aggregation; 13/15 pairs MEASURED, 2 missing: Ru/seed2 + Ir/seed0)
  `metrics/by_metric/wf181_g5_50epoch_full_sweep.json:per_metal_in_domain.Ru`
- **Baseline (single-task chemprop v2)**: 0.7123
- **Lift_abs**: +0.0737 (+10.35%)
- **TARGET (PROJECTED)**: 0.86 (Krasnov 2026 m1 target)
- **Delta vs target**: -0.0740 (still need 5-seed full coverage +
  per-innovation W₂ ablation per TODO-182)
- **Note**: no best-seed selection — mean+std of 2 seeds reported (0.7716,
  0.8005); seed 2 missing honestly disclosed.

### §3.3 TODO-121 (m19 LOMO Rh bottleneck) — HONEST_NEGATIVE

- **MEASURED**: 0.6220 (1 seed × 1 fold; only Re fold present, Rh fold
  absent from per_fold_auc_mean)
  `metrics/by_metric/metal_lomo_auc_wasserstein.json`
- **Rh fold measured**: false
- **Verdict**: Wasserstein + Rh oversample regularisers ran but Rh-as-LODO
  fold was not produced. Per-fold AUC contains only Re — Rh bottleneck
  lift claim cannot be quoted from this run.
- **Overall target (0.6)**: HIT (Re 0.6220); Rh-specific target (0.5) NOT
  measured.

### §3.4 TODO-122 (L5 Hilbert R² 4-cell) — HONEST_NEGATIVE

- **MEASURED**: -5.66 ± 1.36 (Wf-122 P3 post-fix combined anchor;
  n_seeds_total=20; PlatinAI 4-cell strict-scaffold)
  `metrics/by_metric/hilbert_4cell_r2.json:aggregate_combined_r2_mean`
- **Pre-fix baseline (Wf-85 P3)**: -1.50 (ASTSR-collapse trick yielding R²
  ≈ -0.014 constant predictor)
- **Lift_abs**: -4.16 (post-fix is WORSE on raw R² because Wf-122 P1.2
  scale-guard correctly rejects the collapse trick)
- **TARGET (PROJECTED)**: R² ≥ 0.85 (LAMBDA_ROADMAP §4.2 L5)
- **Verdict (per Wf-139 P4 close-out TODO/completed**:
  `137_l5_l7_l10_arch_ceiling_honest.md`):
  **MEASURED_HONEST_NEGATIVE_R2_ARCHITECTURAL_CEILING**. 12-d RDKit |ρ| ≤
  0.019 on PlatinAI pIC50; linear R² upper bound = (0.019)² · 12 =
  0.0043; hence R² ≥ 0 mathematically infeasible on this feature stack.
  Wf-94 R20-P3 Multi-PH + PLL real feature augmentation (32-bin Fourier +
  128-d PLL signature, NOT zero-padded) is the only architectural fix.
- **Wf-129 P3 Multi-PH + PLL guarded retrain**: R² = -0.0780 ± 0.0148
  (less negative than Wf-122 P3 post-fix, but still below LAMBDA_ROADMAP §4.2
  L7 + R20-P3 gate R² ≥ +0.30). Regularisers PREVENT divergence but do
  NOT lift the architectural signal floor.
- **Paper §6 relegation**: do NOT cite in §4 main.

### §3.5 TODO-123 (m15 RMSD redocking) — MEASURED_PASS

- **MEASURED**: 0.6578 Å (5/5 pockets pass 2.0 Å SOTA gate; aggregate
  median across 15 runs = 5 pockets × 3 seeds; exhaustiveness=16;
  n_poses_per_seed=5)
  `metrics/by_metric/docked_pose_rmsd.json:aggregate_median_across_pockets`
- **Gate target**: ≤ 2.0 Å (Trott & Olson 2010 Vina paper convention)
- **Per-pocket**: 1h36=0.5739, 3eml=1.024, 3ny8=0.9506, 4aaw=0.4223, 4rlu=0.6578
  (median across 3 seeds each)
- **Difficulty counts**: EASY=1, MEDIUM=3, HARD=1
- **Caveat**: M1+M2 isolated lift NOT measured; this is the redesigned m15
  RMSD measurement after the Wf-246-B algorithm switch (RDKit
  rdMolAlign.GetBestRMS, STRICTER than legacy Kabsch+Hungarian).

### §3.6 TODO-125 (L12 OOD per-pocket Vina) — HONEST_NEGATIVE

- **MEASURED**: none on disk (per_pocket_vina_*.json files do NOT exist)
- **Closest**: `metrics/by_metric/vina_min_score.json` — 830c MMP13 single
  pocket, -4.16 ± 0.76 kcal/mol (10 organic SMILES, exhaustiveness=2)
- **Verdict**: per-pocket OOD Vina measurement NOT on disk; only single-pocket
  830c measurement exists (vina_min_score.json n=10, vina_score.json n=100
  CuAAC products on 830c).
- **Receptors unavailable per-pocket OOD**: 1h36 PDB missing; test_001-019
  manifests point to /mnt/storage/data/... not available; only 830c on disk.
- **Architectural fix**: L12 OOD Tanimoto<0.4 protocol + ≥25 metals ×
  per-pocket Vina is the R20-P5 ship target (Wf-130 P3).

---

## §4 Engine-tag invariant (Wf-123 P3 ship contract preserved)

- **`wasserstein_reg ∩ closure_catherine_bounded = ∅`**
- Source: `molmetal/tests/test_wasserstein_closure_integration.py`:
  `TestCombinedEngineTags::test_combined_engine_set_is_disjoint PASS`
- Date: 2026-09-27

---

## §5 Pre-flight v2 + test PASS

- **Pre-flight v2** (TODO-207): 43/43 PASS (2026-09-29 02:12:18 UTC)
- **Tests**: 12/12 PASS (`molmetal/tests/test_wf313g_downstream_lift_dashboard.py`)

```
test_per_consumer_schema[TODO-119-m14_decode_ratio] PASSED
test_per_consumer_schema[TODO-120-m1_ru_auc] PASSED
test_per_consumer_schema[TODO-121-m19_lomo_rh_bottleneck] PASSED
test_per_consumer_schema[TODO-122-L5_hilbert_r2_4cell] PASSED
test_per_consumer_schema[TODO-123-m15_rmsd_redocking] PASSED
test_per_consumer_schema[TODO-125-L12_ood_per_pocket_vina] PASSED
test_lift_calc_ru_auc_over_single_task_baseline PASSED
test_lift_calc_hilbert_r2_negative_delta PASSED
test_lift_calc_m15_rmsd_relative_to_gate PASSED
test_status_classification_measured_vs_projected PASSED
test_status_classification_status_detail_required PASSED
test_dashboard_schema_validation PASSED
```

---

## §6 Standing rules honored

1. **MEASURED vs PROJECTED tag discipline** (§3.5): all 6 consumers carry
   explicit MEASURED/PARTIAL/PROJECTED/HONEST_NEGATIVE labels; every numeric
   value traces to a JSON / report path on disk.

2. **"Kill then adapt" GPU directive** (§0): zero GPU launches; this is
   aggregation work only. All 6 pending follow-up Wf-NNN dry-runs declare
   `user_ok_required: true`.

3. **No paper work without authorization** (TODO-84 active): dashboard is
   aggregation metadata only; §6 relegation notes per TODO-122 L5 are
   READ-only citations to existing TODO/completed/* docs (no `paper/*`
   touches).

4. **Pre-flight v2 25/25 PASS gate** (TODO-207): 43/43 PASS at 2026-09-29
   02:12:18 UTC. No STALE flag (verified < 24 hr from refresh).

5. **Honest-negative close** (§3.5): TODO-121 + TODO-122 + TODO-125 all
   declared HONEST_NEGATIVE with full status_detail citing the precedent
   (Wf-86 TODO-91d, Wf-139 P4 TODO/completed/137, Wf-122 P3 4-cell
   honest-negative).

6. **No best-seed selection**: TODO-120 Ru AUC reports mean+std of 2 seeds
   (0.7716, 0.8005); seed 2 missing honestly disclosed (n_pairs_expected=15
   vs n_pairs_total=13).

---

## §7 Pending follow-up Wf (all BLOCKED on fresh user OK per kill_then_adapt §0)

| Wf | Wall | Hardware | Status |
|---|---:|---|---|
| Wf-124 P4 (TODO-119 m14 isolated lift) | ~6 hr | RTX 5090 + Blackwell | dry-run READY |
| Wf-124 P3.5 (TODO-180/181) | ~5 hr | RTX 5090 | dry-run READY |
| Wf-125 P2 (TODO-121 Rh fold) | ~3 hr | RTX 5090 | dry-run TBD |
| Wf-127 P3 (TODO-122 L5 joint sweep) | ~4 hr | Blackwell | BLOCKED on architectural signal floor; Wf-94 R20-P3 Multi-PH+PLL required first |
| Wf-126 P3 (TODO-123 SE(3) bridge) | ~2 hr | RTX 5090 | dry-run TBD |
| Wf-130 P3 (TODO-125 L12 per-pocket Vina) | ~6 hr | Blackwell | BLOCKED on PDB staging; per-pocket OOD Vina pool not built |

---

## §8 Cross-references

- Parent: `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md`
- Wf-123 P3 ship: `TODO/completed/131_wf124_p2_cfm_full.md`
- M1+M2 master plan: `TODO/completed/126_innovation_strengthen_roadmap.md`
- M1+M2 integration tests: `molmetal/tests/test_wasserstein_closure_integration.py`
  (25/25 PASS)
- Wf-139 P4 architectural ceiling close-out:
  `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md`
- Pre-flight v2: `molmetal/scripts/pre_flight_check.py` (43/43 PASS)

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: Wf-313G is
aggregation-only work. Per-consumer isolated M1+M2 lift is per-TODO Q4
2026 / R20 GPU exercise. The dashboard SHIPS a measurement-state ledger
for the 6 strengthen-TODO consumers; lift magnitudes will be re-aggregated
after each consumer's Wf-NNN follow-up run completes.

— Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md
§1§5; close 2026-09-29 (Wf-313G ship).
