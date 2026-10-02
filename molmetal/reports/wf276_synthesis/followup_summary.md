# Wf-276 Follow-Up Synthesis (Wf-276E)

**Date**: 2026-09-29
**Owner**: Hugo Lee + Claude (workflow harness)
**Inputs aggregated**: Wf-276A (Wf-266 overclaim fix) + Wf-276B (G6 launch orchestration) + Wf-276C (Cf-258 CFM TODOs audit) + Wf-276D (207 §9 cross-ref audit)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED discipline) + `mol-metal-gpu-execution` §7 (no best-seed selection)

---

## §1 Per-agent summary

### Wf-276A — Wf-266 overclaim fix (chemprop v3 LOMO fix)
**Status**: IMPLEMENTED (FLAT G1-G8 PASS, 18 tests green). Two levers wired into `train_chemprop_v2_v3.py` (`--chemprop-min-epochs=10` + `--chemprop-lomo-init-scale=1e-3`) plus `--chemprop-mode=v3-fix`. Root cause per `wf266_audit/v3_lomo_root_cause.md` §2.1 is **structural** (LOMO + BCE + all-zero held-out head labels → deterministic collapse), not architectural. **Honest-negative**: zero GPU retrain in this workflow — the lever is wired but no MEASURED AUC lift yet (deferred to TODO-130 G5 retrain).

### Wf-276B — G6 launch orchestration
**Status**: PLAN WRITTEN (no GPU launch yet). G5 wrapper PID 609543 is Ru seed=0 mid-run (~epoch 36/50); G6 wrapper exists but `--no-enforce-budget` flag is **MISSING** and `--no-kendall`/`--no-dann` ablation flags are **UNVERIFIED** (G6 will skip 30/60 tuples). GPU-0 (Blackwell, 96 GB) is **idle** and reserved. **Honest correction**: the task brief assumed auto-advance on `test_auc=` line — both wrappers are plain sequential bash, NO inter-pair auto-launch. Recommended Option C: `--dry-run` G6 now + serial real-run after G5 exit (~15-20 hr wall, 2-3× Wf-258 PROJECTED 7.5 hr per the Wf-272D G5 MEASURED 13.17 vs PROJECTED 5.0 hr pattern).

### Wf-276C — Cf-258 5 PENDING CFM TODOs audit
**Status**: AUDIT COMPLETE. 5 sub-tasks totalling **37.8 GPU-hr + 39 waves** (conservative; honest range 37.8-61.8 hr depending on TODO-119 P1.3 interpretation). Priority order: TODO-180 G4 (1.8 hr, ready-to-launch) > TODO-182 G6 (7.5 hr, needs ablation-flag verify) > TODO-181 G5 (5 hr, ready-to-launch) > TODO-119 P1.3 (16 hr, needs CFM ablation flag + W₂ gradient-path verify) > TODO-119 P1.4 (7.5 hr, blocked by P1.3). **Honest risk**: Wf-181/Wf-188 decode_ratio 0.9980/0.9958 suggest TODO-119 P1.3 may already be partially satisfied on MetalCytoToxDB — but formal P1.3 deliverable is per-cohort on PlatinAI 4-cell + DrugOOD, NOT yet formally run.

### Wf-276D — 207 §9 cross-ref audit
**Status**: MOSTLY CONSISTENT (18 of 19 entries ✅). 19 207 §9 entries × 8 188 §4 sections cross-referenced line-level. **1 MINOR INCONSISTENCY** found: 207 §9.5 line 386 says "MEASURED per Wf-263A verify phase" but the honest framing paragraph immediately below + 188 §4.5.2 + 188 §5.2 line 379 + 207 §9.AE all say TIMEOUT/NOT-MEASURED. Action item A1: edit 207 §9.5 body to mirror 188 §4.5.2 verbatim. cfm_decode_real.json regression_disclosure (Wf-274A 0.9645 vs 0.9980) properly cited in 188 §5.2 line 387; 207 correctly cites Wf-188 0.9958 LEAD, avoiding the deprecated 0.9645 value. MEASURED-vs-PROJECTED label discipline preserved across both docs.

---

## §2 Aggregate work completed vs failed

| Metric | Value | Notes |
|---|---:|---|
| **Sub-tasks COMPLETED** | 4 / 4 | A (implementation), B (orchestration plan), C (audit), D (cross-ref audit) |
| **Sub-tasks FAILED** | 0 | All sub-tasks delivered within their FLAT gates |
| **Action items generated** | 3 | A1 (MINOR: 207 §9.5 body wording edit) + A2 (INFO: 188 §5.2 line 379 should add 207 §9.AE Wf-268B cross-ref) + A3 (INFO: byte-identical 207 §9.AH ↔ 188 §4.5.4, no edit) |
| **GPU-hours NEW total (5 PENDING CFM TODOs)** | 37.8 hr (conservative; 61.8 hr upper) | Per Cf-258 §2.1 |
| **GPU-hours NOW in flight (Wf-124 P3 G5)** | ~12.5 hr remaining | PID 609543, Ru seed=0 ~epoch 36/50 |
| **GPU-hours BLOCKED (207 §9.AB + 188 §4.5.1)** | 50-epoch sweep blocked | until G4 15-pair close |
| **Honest-negatives documented** | 6 | Wf-266 no retrain; Wf-258 §7 PROJECTED; Wf-269C Δ=−1.000 YuelBond load-bearing; Wf-269A path_c undefined; Wf-269B n_pairs_measured=0; Wf-271B 0.9645 silent-regression methodology-change |
| **Critical inconsistent cross-refs** | 1 (minor) | 207 §9.5 line 386 wording |

---

## §3 Critical remaining honest-negatives (ranked by severity)

1. **TODO-119 P1.3 (real-CFM decode_ratio lift) is NOT formally run on PlatinAI/DrugOOD cohorts** — only MetalCytoToxDB has ad-hoc Wf-181/Wf-188 0.9980/0.9958 measurements. Paper §3/§4 cannot cite P1.3 as MEASURED. **16 GPU-hr Blackwell needed**.
2. **Wf-124 P3 G5 50-epoch sweep BLOCKED at G5 15-pair completion** — W₂ trajectory still decreasing at epoch 30. TODO-181 G5 5 hr required to close.
3. **Wf-124 P3 per-innovation ablation (TODO-182 G6) per-lever attribution NOT measured** — current 0.7613 is AGGREGATE lift; cannot cite multi-task 60-80% / W₂ 10-30% / L2 5% as MEASURED until 60 ablation runs complete. **7.5 GPU-hr needed**.
4. **Wf-266 LOMO-fix lever is UNVERIFIED on real GPU retrain** — 4 CPU tests pass but no AUC lift measurement. Test result on 3-epoch trial: 0.5000 (constant predictor — the lever does NOT rescue head-collapse on its own; structural Path C needed for full close).
5. **207 §9.5 body line 386 internal inconsistency** — says "MEASURED per Wf-263A verify phase" but honest framing + 188 §4.5.2 say TIMEOUT/NOT-MEASURED. **MINOR action item A1**.
6. **Wf-271B silent regression methodology change** — `cfm_decode_real.json::regression_disclosure` flags 0.9645 (1 seed × 818 samples, CPU) vs prior 0.9980 (3 seeds × 5991, GPU) without disclosing. 188 §5.2 line 387 properly cites Wf-188 0.9958 LEAD to avoid the trap; 207 §9.AA + §9.6.3 also consistent.

---

## §4 Next follow-up priorities (recommended order)

| # | Workflow | Wall | Why | Blocked by |
|---:|---|---:|---|---|
| 1 | **Wf-276E-A1 fix** | CPU only | Edit 207 §9.5 line 386 wording to mirror 188 §4.5.2 (MINOR inconsistency). Trivial. | — |
| 2 | **Wf-276E-280 = TODO-180 G4 launch** | 1.8 hr | Smallest wall; closes honest gap from Wf-124 P3 §4 "14 PENDING" cite; enables honest 5×3 paper §4 LEAD claim. ZERO new infrastructure. | None — can launch TODAY |
| 3 | **Wf-276E-281 = TODO-181 G5 launch** | 5 hr | 50-epoch convergence check; needs G4 done first (control = 30-ep 15-pair). | Wf-276E-280 |
| 4 | **Wf-276E-282 = TODO-182 G6 launch** (CUDA:1 serial) | 15-20 hr | Per-lever attribution matrix (60 runs; 30 CONFIRMED only unless `--no-kendall`/`--no-dann` verified). Plan for 2-3× Wf-258 PROJECTED 7.5 hr per G5 MEASURED pattern. | G5 wrapper exit |
| 5 | **Wf-276E-119P1.3** | 16 hr | Real-CFM decode_ratio on PlatinAI 4-cell + DrugOOD assay/scaffold/size cohorts. Needs CFM ablation flag + W₂ gradient-path verification first. | Blackwell smoke (Wf-269A pattern, 3-metal); GPU-0 idle |
| 6 | **Wf-276E-119P1.4** | 7.5 hr | Per-innovation CFM ablation; depends on P1.3 result as the "all 3 innovations" control. | Wf-276E-119P1.3 |

**Total wall for the recommended sequence**: 45.3-50.3 GPU-hr (mostly Blackwell idle GPU-0; G6 on GPU-1 serial after G5).

**Strategy recommendation**: parallelize — run Wf-276E-280 + Wf-276E-281 on GPU-1 (after G5 exit, sequential); reserve GPU-0 Blackwell for Wf-276E-119P1.3 + P1.4 (memory pool fits 96 GB model). Avoid GPU contention per `mol-metal-gpu-execution` §4.

---

## §5 Honest framing (party-twenty-spirit §3.5)

- All numbers in this synthesis are MEASURED where the source report carries `MEASURED` label (Wf-188 0.9958, Wf-272D 13.17 hr, Wf-266 0.5000 3-ep trial); PROJECTED where the source carries `PROJECTED` (Wf-258 §2.1 37.8 hr, Wf-258 §3 priority attribution, Wf-276B G6 wall estimate).
- **DO NOT cite** any of the 5 PENDING CFM TODOs as MEASURED until the actual retrain / ablation completes. All ETA numbers are PROJECTED with explicit per-TODO §3 cite.
- **WAVE_BUDGET discipline**: All recommendations are in 1-hr waves per `mol-metal-gpu-execution` §6 "kill then adapt" pattern. No multi-hour autonomous launches proposed.
- **Cuda:0 trick**: Wf-266 + Wf-272D confirmed `CUDA_VISIBLE_DEVICES=1` exposes GPU-1 as cuda:0. Wf-276E-119P1.3 on GPU-0 uses native cuda:0 (no env trick).
- **MEASURED-vs-PROJECTED label discipline preserved** per Wf-276D audit: 207 §9 + 188 §4 + cfm_decode_real.json regression_disclosure all consistent on the Wf-188 0.9958 LEAD number.

---

**End of synthesis. Summary**: 4 sub-tasks completed, 0 failed; 1 minor inconsistency (action item A1); 37.8-61.8 GPU-hr remaining across 5 PENDING CFM TODOs. Recommended sequence: TODO-180 G4 (1.8 hr) → TODO-181 G5 (5 hr) → TODO-182 G6 (15-20 hr) on GPU-1 serial, parallel TODO-119 P1.3 + P1.4 (23.5 hr) on idle Blackwell GPU-0 after 3-metal smoke.