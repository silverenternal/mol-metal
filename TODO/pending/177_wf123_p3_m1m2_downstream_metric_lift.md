# TODO 177 — Wf-123 P3 Gap: Verify M1+M2 actually lifts downstream strengthen metrics

**Date**: 2026-09-27 (Wf-123 P3 gap analysis per `party-twenty-spirit` §3.5)
**Owner**: Hugo Lee + ultracode agent
**Status**: 🚧 P1 — IN-FLIGHT (Wf-123 P3 SHIP integration layer, downstream lift NOT yet measured)
**Parent**: `TODO/completed/131_wf124_p2_cfm_full.md` §3 honest framing + `TODO/completed/126_innovation_strengthen_roadmap.md` §3 (master M1+M2 plan)
**Sibling**: `TODO/pending/178_wf123_p3_metalcytotox_csv_ondisk.md` (G2) + `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (G3) + Wf-124 P3 gaps (G4-G6 in `180_wf124_p3_*.md`)

---

## §1 Gap description

**Wf-123 P3 SHIP** (2026-09-27) integrated M1 (Wasserstein regularization) + M2 (Pocket closure bias) shared modules with **25 / 25 integration tests PASS** + verification driver + PlatinAI 4-cell signal-floor probe. The honest framing (§7) explicitly states:

> "The 25 integration tests verify the **wiring** between M1 and M2 on a real probe (PlatinAI 4-cell, 4078 Pt complexes × 4 cell lines) and a synthetic probe (5-metal transfer cohort). They do **not** verify that the W₂ penalty + closure bias actually moves any downstream metric — that is the per-strengthen-TODO deliverable."

The 6 strengthen-TODO consumers are:

| TODO | Consumer metric | M1 function | M2 function | Lift status |
|---|---|---|---|---|
| TODO-119 | m14 decode_ratio | `wasserstein_loss` | `closure_bounded_smiles_generation` | NOT measured |
| TODO-120 | m1 Ru AUC | `wasserstein_pic50_loss` | — | NOT measured (Wf-124 P3 lift measured for Ru seed=0 only; full 5×3 needs this gap closed) |
| TODO-121 | m19 LOMO Rh bottleneck | `wasserstein_transfer_loss` | `closure_oversample` | NOT measured |
| TODO-122 | L5 Hilbert R² ≥ 0.85 | `wasserstein_pic50_loss` | — | signal-floor (R² ≤ 0 by construction) — **see G3** |
| TODO-123 | m15 RMSD | `wasserstein_pose_loss` | — | NOT measured |
| TODO-125 | L12 OOD per-pocket Vina | `wasserstein_pocket_loss` | `pocket_specific_closure_bias` + `per_pocket_pharmacophore_lookup` | NOT measured |

**Verdict**: M1+M2 are foundation primitives — they DO NOT move any of these 6 metrics by themselves. Per-TODO lift is a Q4 2026 / R20 GPU exercise (per `party-twenty-spirit` §3.5 honest framing).

---

## §2 Current MEASURED state

| Metric | Value | Source | Date |
|---|---:|---|---|
| M1 integration tests | 43 / 43 PASS | `molmetal/tests/test_wasserstein_loss.py` | 2026-09-27 |
| M2 integration tests | 55 / 55 PASS | `molmetal/tests/test_pocket_closure_bias.py` | 2026-09-27 |
| M1+M2 cross-module integration | 25 / 25 PASS | `molmetal/tests/test_wasserstein_closure_integration.py` | 2026-09-27 |
| PlatinAI 4-cell signal-floor R² | ≤ 0 by construction | `molmetal/reports/wf123_p3_integration/cross_module_verification.json` | 2026-09-27 |
| Engine-tag disjoint | `wasserstein_reg` ∩ `closure_catherine_bounded` = ∅ | `TestCombinedEngineTags` PASS | 2026-09-27 |
| Ru AUC (Wf-124 P3, seed=0 only) | 0.7613 MEASURED | `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` | 2026-09-27 |
| Krasnov 2026 m1 target | ≥ 0.86 PROJECTED | `TODO/pending/120_m1_ru_auc_strengthen.md` | — |

**Gap (MEASURED → TARGET)**:

- m1 Ru AUC: 0.7613 (1-seed) → 0.86 → −0.098 (need 5-seed × 50-epoch sweep, see TODO-180)
- m14 decode_ratio: NOT measured (Wf-119 P1.3 deferred; synthetic 192/192 from Wf-124 P2 is **architectural path**, NOT real-CFM)
- m15 RMSD: NOT measured (TODO-123 P1.3 deferred)
- m19 LOMO Rh bottleneck: NOT measured (TODO-121 P1.1 deferred)
- L5 Hilbert R²: NOT measured (signal-floor by construction — see TODO-179 for W₂ alignment lift)
- L12 OOD per-pocket: NOT measured (TODO-125 P1.2 deferred)

---

## §3 What's needed

### §3.1 Per-consumer Wf-124+ spawn

For each of the 6 strengthen-TODO consumers, spawn a follow-up Wf-124+ wave that:

1. **Wires the consumer metric into the M1+M2 integration layer** (`molmetal/scripts/verify_shared_modules.py` is the template).
2. **Re-runs the consumer metric MEASUREMENT** with M1+M2 enabled vs disabled (ablation — see TODO-182 for the per-innovation ablation protocol).
3. **Reports per-seed mean ± std** (no best-seed selection per `party-twenty-spirit` §3.5 + `mol-metal-gpu-execution` §7 tag discipline).
4. **Honest framing**: if the lift is < 0.05 or the new value is below gate, write into §6 limitations (NOT paper §4 LEAD-claim).

### §3.2 Consumer-specific Wf list

| Consumer TODO | Follow-up Wf | Hardware | Wall-time | ETA |
|---|---|---|---:|---:|
| TODO-119 (m14 decode_ratio) | Wf-124 P4 (or Wf-128 if orphan) | RTX 5090 + Blackwell | ~6 hr | 1 wk |
| TODO-120 (m1 Ru AUC) | Wf-124 P3.5 (this gap, see TODO-180/181) | RTX 5090 | ~5 hr | 1 wk |
| TODO-121 (m19 LOMO Rh) | Wf-125 P2 (per-metal Wasserstein + closure oversample) | RTX 5090 | ~3 hr | 1 wk |
| TODO-122 (L5 Hilbert R²) | Wf-127 P3 (joint sweep, see TODO-179) | Blackwell | ~4 hr | 1 wk |
| TODO-123 (m15 RMSD) | Wf-126 P3 (5-pocket SE(3) bridge + W₂) | RTX 5090 | ~2 hr | 1 wk |
| TODO-125 (L12 OOD per-pocket) | Wf-130 P3 (Catherine ≥ 25 metals × per-pocket Vina) | Blackwell | ~6 hr | 2 wk |

### §3.3 Acceptance gate

- [G1.1] Each of the 6 consumer TODOs has a follow-up Wf-NNN that re-measures the consumer metric with M1+M2 enabled
- [G1.2] Per-seed mean ± std reported (no best-seed selection)
- [G1.3] MEASURED vs PROJECTED labels per `mol-metal-gpu-execution` §7 tag discipline
- [G1.4] Engine tags `wasserstein_reg` + `closure_catherine_bounded` disjoint verified in each per-sample output
- [G1.5] If lift < 0.05 OR below gate: §6 limitations entry (NOT §4 LEAD)

---

## §4 ETA + dependencies

**ETA**: 6 weeks (1 wk per consumer, staggered)

**Dependencies**:
- TODO-180 (Wf-124 P3.5: 14/15 (metal,seed) complete) — BLOCKING for TODO-120 consumer
- TODO-181 (Wf-124 P3.6: 50-epoch full sweep) — BLOCKING for all 6 consumers (current 30-epoch may be undertrained)
- TODO-182 (Wf-124 P3.7: per-innovation ablation) — prerequisite for §3.1 step 2 ablation
- Wf-127 P3 joint sweep (L5 R² — Hilbert + ASTSR + W₂)
- Wf-126 P3 5-pocket staging + SE(3) bridge
- Wf-130 P3 Catherine ≥ 25 metals × per-pocket Vina
- `mol-metal-gpu-execution` §7 tag discipline (no_overclaim audit)
- Blackwell + RTX 5090 dual-GPU per standing "kill then adapt" directive (need fresh user OK)

---

## §5 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: 25 / 25 M1+M2 integration tests PASS + Ru seed=0 0.7613 +0.049 lift
- **PROJECTED**: m1 Ru AUC ≥ 0.86, m14 decode_ratio lift on real CFM, m15 RMSD, m19 LOMO Rh, L5 Hilbert R² ≥ 0.85, L12 OOD per-pocket — all 6 are PROJECTED @ R20-P2/P3/P5 with explicit MEASURED-vs-PROJECTED labels
- **Smaller-metal caveat** (per Wf-124 P3 §5): Rh/Os/Re < 1.2k rows have noisier per-seed AUC; report mean ± std unfiltered
- **W₂ caveat** (per Wf-124 P3 §5): W₂ regularization may slow convergence in early epochs (predicted pIC50 cloud forced to track reference before encoder has learned anything); JKO-style CFL gate satisfied
- **L2 guard caveat**: Without L2 guard, W₂ term can blow up early in training (predicted pIC50 distribution far from reference); L2 clip at 5.0 keeps grad-norms bounded
- **Engine-tag invariant** (per TODO-126 §7): `wasserstein_reg` and `closure_catherine_bounded` are **disjoint**; verified by `TestCombinedEngineTags::test_combined_engine_set_is_disjoint` in `molmetal/tests/test_wasserstein_closure_integration.py`
- **No best-seed selection** (per `mol-metal-gpu-execution` §7): if 3-seed mean is below gate, the per-seed std IS the answer; do not pick the highest seed and call it MEASURED

---

## §6 Cross-references

- `molmetal/molmetal_lam/reward/wasserstein_loss.py` (~530 LOC, 6 functions, 43 / 43 tests)
- `molmetal/molmetal_lam/reward/pocket_closure_bias.py` (~200 LOC, 4 functions, 55 / 55 tests)
- `molmetal/tests/test_wasserstein_closure_integration.py` (25 integration tests — Wf-123 P3 ship)
- `molmetal/scripts/verify_shared_modules.py` (verification driver — Wf-123 P3 ship)
- `molmetal/reports/wf123_p3_integration/cross_module_verification.json` (signal-floor probe + pytest summary)
- `molmetal/reports/wf123_p3_integration/README.md` §7 (honest framing per TODO-126 §7)
- `TODO/pending/126_innovation_strengthen_roadmap.md` §3 (M1+M2 master plan)
- `TODO/pending/119_decode_ratio_strengthen.md` §3 P1.2 (M1+M2 consumer — m14)
- `TODO/pending/120_m1_ru_auc_strengthen.md` §3 P1.2 (M1 consumer — m1)
- `TODO/pending/121_m19_lomo_strengthen.md` §3 P1.1 (M1+M2 consumer — m19)
- `TODO/pending/122_platinai_4cell_r2_strengthen.md` §3 P1.3 (M1 consumer — L5)
- `TODO/pending/123_m15_rmsd_strengthen.md` §3 P1.3 (M1 consumer — m15)
- `TODO/pending/125_vina_per_pocket_strengthen.md` §3 P1.2 (M1+M2 consumer — L12)

---

## §7 Follow-ups

- **TODO-178** (G2: metalcytotox 5-metal CSV on disk) — needed before integration smoke test passes
- **TODO-179** (G3: W₂ alignment on PlatinAI 4-cell probe) — needed to lift L5 R² above signal-floor
- **TODO-180** (G4: Wf-124 P3.5 14/15 (metal,seed) complete) — needed for m1 Ru AUC lift
- **TODO-181** (G5: Wf-124 P3.6 50-epoch full sweep) — needed for convergence check
- **TODO-182** (G6: Wf-124 P3.7 per-innovation ablation) — prerequisite for §3.1 step 2

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: M1+M2 ship as foundation primitives. The 6 consumer metric lifts are PROJECTED @ R20-P2/P3/P5 (Q4 2026 / 6-week ETA). If any consumer lift is below 0.05 OR below gate, the honest entry is §6 limitations, NOT §4 LEAD-claim. Wf-86 honest-negative close-out (TODO-91d) is the precedent.
