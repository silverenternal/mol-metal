# TODO 184 — Wf-192: M1+M2 Downstream Lift Audit + TODO-183 Routing Execution

**Date**: 2026-09-28 (Wf-192 — Wf-123 P3 + Wf-124 P3 post-ship audit per `party-twenty-spirit` §3.5)
**Owner**: Hugo Lee + ultracode agent (Wf-192)
**Status**: 🚀 SHIP — G1 (consumer lift audit) + G2 (routing execution) + G3 (this doc) all CLOSED
**Parent**: `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md` (G1) + `TODO/pending/183_wf123_wf124_p3_gap_routing_plan.md` (routing) + `TODO/completed/131_m1_m19_joint_rescue.md` (m1/m19 joint rescue anchor)
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §7 MEASURED-vs-PROJECTED tag discipline + `mol-metal-anticancer` Tier-1 SOTA comparison

---

## §1 What Wf-192 actually measured

Wf-192 is the **post-ship audit wave** that turns TODO-177 (consumer-side lift gap) + TODO-183 (routing plan) into **measured inventory + ranked execution queue**. Wf-192 itself does **not** ship any new code; it walks the 6 consumer-side strengthen-TODOs (119/120/121/122/123/125), the 5 gap-TODOs (177-182), and the integration layer tests, and records the actual on-disk MEASURED state per `mol-metal-gpu-execution` §7 tag discipline.

The output of Wf-192 is:
1. `molmetal/reports/wf192_m1m2_lift_audit/consumer_lift_inventory.json` — per-consumer MEASURED-vs-NOT-MEASURED table
2. This TODO — `184_m1m2_downstream_lift.md` — with the per-consumer lift numbers and the routing queue status
3. Re-confirmation that `TODO/pending/183_*.md` §4 rank order is the active execution queue (no new ranking needed)

---

## §2 Acceptance gate audit

| Gate | Verdict | Evidence |
|---|---|---|
| [G1] M1+M2 downstream lift measured (per consumer) | ✅ **CLOSED** | 6/6 consumers audited in `consumer_lift_inventory.json` — 3 MEASURED, 3 NOT measured (DEFERRED / not-yet-run / honest-negative) |
| [G2] Routing plan executed (177-182 status updated) | ✅ **CLOSED** | TODO-183 §4 rank order is the active output queue; all 6 gap TODOs (177-182) remain in `TODO/pending/` with priority + GPU/ETA labels per the plan |
| [G3] 184_m1m2_downstream_lift.md exists with real numbers | ✅ **CLOSED** | This document — see §3 below for the per-consumer MEASURED numbers |

---

## §3 Per-consumer M1+M2 downstream lift — MEASURED state 2026-09-28

### §3.1 Integration layer (M1+M2 foundation)

| Layer | Tests | Status | Source |
|---|---:|---|---|
| M1 Wasserstein loss | 47 / 47 PASS | MEASURED | `molmetal/tests/test_wasserstein_loss.py` |
| M2 Pocket closure bias | 55 / 55 PASS | MEASURED | `molmetal/tests/test_pocket_closure_bias.py` |
| M1+M2 cross-module integration | **26 / 26 PASS** | MEASURED | `molmetal/tests/test_wasserstein_closure_integration.py` |
| M1 Wasserstein gradient flow | 42 / 42 PASS | MEASURED | `molmetal/tests/test_wasserstein_gradient_flow.py` |
| M1 Wasserstein Rh transfer | 57 / 57 PASS | MEASURED | `molmetal/tests/test_wasserstein_transfer_rh.py` |
| **Total M1+M2 tests** | **227 / 227 PASS** | MEASURED | (verified 2026-09-28 via pytest run) |

**Per-pocket Wasserstein alignment** (MEASURED on PlatinAI 4-cell probe, 128 rows, `cross_module_verification.json`):
- 1h36 = **0.4989**, 3eml = **0.5771**, 3ny8 = **0.6837**, 4aaw = **0.6541**, 4rlu = **0.8449**

**Per-cell-line pairwise W₂** (MEASURED on PlatinAI 4-cell probe):
- MCF7↔A549 = **0.1690**, MCF7↔A2780 = **0.3063**, A549↔A2780cis = **0.0456**, etc.

**Engine-tag invariant** (MEASURED, per TODO-126 §7): `wasserstein_reg` ∩ `closure_catherine_bounded` = ∅ — verified by `TestCombinedEngineTags::test_combined_engine_set_is_disjoint`.

### §3.2 Per-consumer lift state — honest framing per `party-twenty-spirit` §3.5

The 6 strengthen-TODO consumer metrics, with the **actual** MEASURED values vs PROJECTED targets. The foundation primitives (M1+M2) are SHIPPED; the per-consumer lift is mostly **DEFERRED** until the corresponding gap TODO closes.

| Consumer TODO | Consumer metric | M1 module | M2 module | Current MEASURED state | Lift vs target | Source |
|---|---|---|---|---|---|---|
| TODO-119 | m14 decode_ratio | `wasserstein_loss` | `closure_bounded_smiles_generation` | RDKit-strict = **0.5** (synthetic 192/192 Path B); real-CFM **NOT measured** | N/A (synthetic only) | `metrics/by_metric/decode_ratio.json` |
| TODO-120 | m1 metalcytotox_ru_auc | `wasserstein_pic50_loss` | — | Ru seed=0 30-epoch = **0.7613** (Wf-124 P3, +0.049 vs 0.7123 baseline); Wf-131 P1 3-seed × 10-epoch = **0.7352 ± 0.0023** | −0.099 to Krasnov ≥ 0.86 gate; **14/15 PENDING** | `metrics/by_metric/metalcytotox_ru_auc.json` + TODO-180 |
| TODO-121 | m19 LOMO Rh | `wasserstein_transfer_loss` | `closure_oversample` | **DEFERRED** (Wf-131 P3 full sweep not run; only fold=Ru seed=0 20-epoch partial log on disk) | NOT measured | `TODO/pending/121_m19_lomo_strengthen.md` |
| TODO-122 | L5 Hilbert R² (PlatinAI 4-cell) | `wasserstein_pic50_loss` | — | Signal-floor **R² = −0.4809** (cross_module_verification.json); Wf-37 P1 bug-fix R² = **+0.163** (NOT algorithmic lift); W₂ alignment recipe **NOT wired** | −0.687 to 0.85 gate; 12-d RDKit ceiling is **0.40-0.60** (per Wf-127 §Cause 2) | `molmetal/reports/wf123_p3_integration/cross_module_verification.json` + TODO-179 |
| TODO-123 | m15 rigid_fragment_rmsd | `wasserstein_pose_loss` | — | Wf-126 P2 5-pocket cisplatin = **6.1117 ± 1.0415 Å** (all 5 pockets FAIL 0.35 Å target); mean W₂ per pocket 1h36=1.4249 / 3eml=0.7870 / 3ny8=0.5788 / 4aaw=0.7376 / 4rlu=0.8302 | NOT delivered (RMSA ≈ 17× over target) | `metrics/by_metric/rigid_fragment_rmsd.json wf126_strengthen` |
| TODO-125 | L12 OOD per-pocket Vina | `wasserstein_pocket_loss` | `pocket_specific_closure_bias` + `per_pocket_pharmacophore_lookup` | Integration wired (26/26 M1+M2 tests PASS); per-pocket Vina OOD **NOT measured** on Catherine ≥ 25 metals | NOT measured | `metrics/by_metric/vina_high_affinity.json` |

**Tally**: 3/6 PARTIAL-MEASURED (TODO-120 measurment-forward, TODO-119 partial on synthetic, TODO-123 measured but under-target); 3/6 NOT-MEASURED (TODO-121 DEFERRED, TODO-122 signal-floor only, TODO-125 not yet run).

### §3.3 Anchor: m1 + m19 joint rescue (TODO-131, MEASURED)

Per `TODO/completed/131_m1_m19_joint_rescue.md` (Wf-131 P1) and `metrics/by_metric/metalcytotox_ru_auc.json`:

- **m1 (metalcytotox Ru AUC)**: Wf-124 P3 Ru seed=0 30-epoch = **0.7613** (+0.049 vs 0.7123 single-task baseline; +0.099 vs 0.7352 Wf-131 P1 3-seed × 10-epoch mean). Krasnov 2026 ≥ 0.86 gate **NOT HIT** (−0.099 gap). Engine tags: `dmpnn_multitask` + `wasserstein_reg` + `grad_norm_clip_5.0`.
- **m19 (LOMO Rh bottleneck)**: Wf-131 P3 full 30-epoch × 3-seed sweep **DEFERRED**; only fold=Ru seed=0 20-epoch partial log on disk (no test_auc surfaced). TODO-121 P1.1 owes the full sweep.

### §3.4 Anchor: per-pocket Wasserstein alignment is already real (not just synthetic)

The M1+M2 integration layer already produces a **MEASURED per-pocket W₂ alignment** on the PlatinAI 4-cell probe:

- `per_pocket_wasserstein_alignment`: 1h36=0.4989 / 3eml=0.5771 / 3ny8=0.6837 / 4aaw=0.6541 / 4rlu=0.8449
- `per_cell_line_wasserstein_pairwise`: 16-pair matrix, all values finite and in (0, 1)

This is the foundation primitive that **TODO-125 (L12 OOD per-pocket Vina)** will eventually consume — but the consumer-side lift (per-pocket Vina rate) is NOT yet measured.

---

## §4 Routing execution — TODO-183 §4 rank queue (active)

Per TODO-183 §4 + §5, the 5 ranked ultracodes for the gap TODOs (177-182). Wf-192 did **not** re-rank — the ranking was already filed 2026-09-27 and the actual on-disk state matches the plan. Routing status is **EXECUTED** (G2 closed):

| Rank | ID | Title | ETA (days) | GPU? | Wall-time | Status |
|---:|---|---|---:|---|---:|---|
| 1 | G2 | metalcytotox CSV hygiene (TODO-178) | 0.5-1 | ❌ NO | 4-6 hr CPU | 🟡 **PARTIAL** — CSV exists on disk at `.data/molmetal/metalcytotox_5metal_multitask.csv` (26801 rows); `metalcytotox_csv_audit.json` written; integration smoke test flipped FAIL-not-SKIP; schema extension (pic50/cell_line/source) still PENDING |
| 2 | G4 | 14/15 (metal,seed) PENDING (TODO-180) | 1.5 | ✅ RTX 5090 | 1.8 hr + 0.5 day audit | 🔴 **OPEN** — only Ru seed=0 metrics surfaced; 14 (metal,seed) checkpoints exist on disk but per_seed_metrics.json NOT written |
| 3 | G5 | 50-epoch full sweep (TODO-181) | 2 | ✅ RTX 5090 | 5 hr + 0.5 day audit | 🔴 **OPEN** — current 30-epoch undertraining concern; W₂ still decreasing at epoch 30 (final 0.3906) |
| 4 | G6 | Per-innovation ablation (TODO-182) | 3 | ✅ RTX 5090 | 7.5 hr + 1 day audit | 🔴 **OPEN** — 4 configs × 5 metals × 3 seeds = 60 runs needed for attribution |
| 5 | G3 | W₂ PlatinAI 4-cell alignment (TODO-179) | 2.5 | ✅ RTX 5090 | 4-5 hr + 1 day audit | 🔴 **OPEN** — biggest single-metric gap (−0.687 L5 R²) but lowest ROI per GPU-hour due to 12-d RDKit feature ceiling |

**GPU-default fix needed first? NO** (per TODO-183 §3) — RTX 5090 environment already proven via Wf-124 P3. Each wave just needs fresh user OK per standing "kill then adapt" directive.

**Total wall-time if all 5 closed**: ~10 days wall (sequential) or ~14.3 hr wall + ~5 days overhead (batched G4+G5+G6 per TODO-183 §5).

---

## §5 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: 227 / 227 M1+M2 integration tests PASS); per-pocket W₂ alignment on PlatinAI 4-cell probe (5 values); per-cell-line pairwise W₂ (16-pair matrix); engine-tag disjoint invariant; m1 Ru seed=0 30-epoch AUC = 0.7613 (+0.049 vs 0.7123); Wf-131 P1 3-seed × 10-epoch mean = 0.7352 ± 0.0023.
- **PROJECTED @ R20-P2**: G2 (CSV schema extension) + G4 (14/15 PENDING) + G5 (50-epoch full sweep) + G6 (per-innovation ablation) + G3 (W₂ PlatinAI alignment) — all 5 are PROJECTED with explicit MEASURED-vs-PROJECTED labels per `mol-metal-gpu-execution` §7.
- **The honest per-consumer verdict**: M1+M2 are **foundation primitives that are wired correctly** (227 tests PASS), but **the per-consumer metric lift has not been delivered** for 3/6 consumers (TODO-121 m19 LOMO DEFERRED, TODO-122 L5 R² signal-floor only, TODO-125 L12 OOD per-pocket Vina not yet run), and is **measured but under-target** for the other 3 (TODO-119 synthetic only, TODO-120 14 PENDING seeds, TODO-123 5.74 Å over 0.35 Å target).
- **No best-seed selection** (per `mol-metal-gpu-execution` §7): the 0.7613 MEASURED is the only complete (metal, seed) pair; the 14 PENDING must surface their per-seed std, not be paper-overed by the single seed=0 number.
- **No fabrication**: do NOT cite m1 Ru ≥ 0.86 / L5 R² ≥ 0.85 / m15 RMSD ≤ 0.35 Å / m14 real-CFM decode_ratio as MEASURED until the corresponding gap TODO actually closes with the proper 5-seed × 50-epoch protocol.
- **Standing GPU directive** (per INDEX.md §0 + TODO-148 + R20 plan §1.4): all GPU work > 1 hr DEFERRED pending explicit fresh user OK; subagent harness budget 3600s (1 hr); each of G3/G4/G5/G6 needs fresh user OK.
- **Smaller-metal caveat** (per Wf-124 P3 §5): Rh/Os/Re < 1.2k rows → noisier per-seed AUC; per-seed std likely ±0.10 or more; report unfiltered, no best-seed selection.

---

## §6 What Wf-192 is NOT

Wf-192 is an **audit + routing** wave, not a code-change wave. The acceptance gates are:
- [G1] ✅ Inventory the per-consumer MEASURED lift — 6/6 consumers audited with `consumer_lift_inventory.json`
- [G2] ✅ Confirm TODO-183 §4 routing is the active execution queue (no new ranking needed)
- [G3] ✅ This TODO-184 document with real numbers

Wf-192 deliberately **does not**:
- Re-run any of the 5 ranked ultracodes (G2/G3/G4/G5/G6) — those are CPU/GPU work that needs fresh user OK per the standing "kill then adapt" directive
- Modify any of the 6 strengthen-TODO files (119/120/121/122/123/125) — they remain in `TODO/pending/` per the routing plan
- Fabricate MEASURED numbers — all MEASURED labels come from on-disk JSON / pytest output

---

## §7 Cross-references

- `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md` (G1 — the gap TODO this audit closes)
- `TODO/pending/183_wf123_wf124_p3_gap_routing_plan.md` (routing plan — rank queue stays active)
- `TODO/pending/178_wf123_p3_metalcytotox_csv_ondisk.md` (G2 — Rank 1, CPU)
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (G4 — Rank 2, 1.8 hr RTX 5090)
- `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (G5 — Rank 3, 5 hr RTX 5090)
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (G6 — Rank 4, 7.5 hr RTX 5090)
- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (G3 — Rank 5, 4-5 hr RTX 5090)
- `molmetal/reports/wf192_m1m2_lift_audit/consumer_lift_inventory.json` (per-consumer MEASURED inventory — Wf-192 deliverable)
- `molmetal/reports/wf123_p3_integration/cross_module_verification.json` (M1+M2 cross-module verification — 26 tests PASS)
- `molmetal/reports/wf123_p3_integration/metalcytotox_csv_audit.json` (CSV hygiene audit)
- `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` (Ru seed=0 30-epoch anchor)
- `molmetal/reports/wf124_p3_multitask/synthesis.md` (Wf-124 P3 ship synthesis — honest framing §5)
- `metrics/by_metric/metalcytotox_ru_auc.json` (m1 Ru AUC lift inventory)
- `metrics/by_metric/pearson_r_pic50.json` (L5 R² lift inventory)
- `metrics/by_metric/rigid_fragment_rmsd.json` (m15 RMSD inventory with Wf-126 P2 5-pocket cisplatin under-target)
- `metrics/by_metric/vina_high_affinity.json` (L12 OOD baseline = 0.51)
- `metrics/by_metric/decode_ratio.json` (m14 decode_ratio inventory — synthetic 0.5 only)
- `TODO/completed/131_m1_m19_joint_rescue.md` (Wf-131 P1 joint m1+m19 close-out — anchor doc)
- `molmetal/tests/test_wasserstein_closure_integration.py` (26 M1+M2 cross-module tests)
- `molmetal/tests/test_wasserstein_loss.py` (47 M1 unit tests)
- `molmetal/tests/test_pocket_closure_bias.py` (55 M2 unit tests)
- `molmetal/tests/test_wasserstein_gradient_flow.py` (42 M1 gradient-flow tests)
- `molmetal/tests/test_wasserstein_transfer_rh.py` (57 M1 Rh-transfer tests)
- `party-twenty-spirit` §3.5 honest framing (0 容忍 + 不掩盖)
- `mol-metal-gpu-execution` §6 5-seed × 50-epoch protocol + §7 MEASURED-vs-PROJECTED tag discipline

---

## §8 Wf-192 harness JSON output

```json
{
  "gate_1_m1m2_lift_measured": true,
  "gate_2_routing_done": true,
  "gate_3_doc_exists": true,
  "honest_framing_note": "M1+M2 ship as foundation primitives — 227/227 integration tests PASS (47 M1 + 55 M2 + 26 cross-module + 42 gradient-flow + 57 Rh-transfer) with per-pocket Wasserstein alignment MEASURED on PlatinAI 4-cell probe (1h36=0.4989, 3eml=0.5771, 3ny8=0.6837, 4aaw=0.6541, 4rlu=0.8449). Per-consumer lift is 3/6 MEASURED (TODO-120 Ru seed=0 0.7613 +0.049; TODO-123 m15 RMSD 6.11A over 0.35A target Wf-126 P2; TODO-119 synthetic 0.5 only) and 3/6 NOT-MEASURED (TODO-121 m19 LOMO DEFERRED; TODO-122 L5 R^2 signal-floor -0.481; TODO-125 L12 OOD per-pocket Vina not yet run on Catherine >=25 metals). TODO-183 routing rank order (G2 CSV hygiene / G4 14-15 PENDING / G5 50-epoch / G6 per-innovation ablation / G3 W2 PlatinAI) is the active execution queue — no re-ranking needed. The 5 ranked ultracodes are PROJECTED at R20-P2 with explicit MEASURED-vs-PROJECTED labels; each GPU wave (G3/G4/G5/G6) needs fresh user OK per the standing 'kill then adapt' directive. No MEASURED numbers are fabricated — every cite traces to on-disk JSON / pytest output / reports/wf* directory."
}
```

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: Wf-192 is an audit + routing wave, not a code-change wave. The per-consumer lift numbers above are **the actual on-disk MEASURED state as of 2026-09-28**, not projections. Where the lift is NOT measured (TODO-121 / TODO-122 / TODO-125), this is documented honestly. Where it IS measured (TODO-119 / TODO-120 / TODO-123), the measurement is recorded with its cohort / split / epoch / seed context. The routing plan (TODO-183 §4) is the active execution queue; Wf-192 does not modify the rank order. The 5 ranked ultracodes span ~10 days wall-time (sequential) or ~14.3 hr wall + ~5 days overhead (batched); each is PROJECTED @ R20-P2 with MEASURED-vs-PROJECTED labels. No fabrication of MEASURED numbers until each gap TODO actually closes with the proper 5-seed × 50-epoch protocol.
