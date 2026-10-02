# Wf-280D — Cf-258 17 Waves (C1–C17) Progress Audit

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Parent:** `molmetal/reports/wf258_synthesize/ready_for_gpu.md` (17-wave synthesis, 5 PENDING CFM TODOs)
**Method:** Read-only filesystem audit of `results/wf124_p3_remaining_metals_g4/`, `results/wf124_p3_50epoch_full_sweep_g5/`, `results/wf124_p3_per_innovation_ablation_g6/`, plus `configs/cfm_*.yaml` and `configs/wf124_p3_*.yaml`. No file modifications.

---

## §1 Per-wave status (C1–C17)

### §1.1 TODO-180 G4 (Wf-124 P3 30-epoch 5×3 sweep)

| Wave | Pairs | Wall | Status | Evidence |
|---|---|---:|---|---|
| **C1** | Ru s1/s2 + Ir s0/s1/s2 (5 pairs) | ~38 min | **DONE** | `results/wf124_p3_remaining_metals_g4/A/Ru/seed1.log`, `seed2.log`, `A/Ir/seed0.log`, `seed1.log`, `seed2.log` (5/5 files present; master.log "Wave A DONE Mon Sep 28 10:12:59 PM CST 2026") |
| **C2** | Os s0/s1/s2 + Rh s0/s1/s2 + Re s0/s1/s2 (9 pairs) | ~70 min | **DONE** | `results/wf124_p3_remaining_metals_g4/A/Os/seed{0,1,2}.log`, `B/Rh/seed{0,1,2}.log`, `B/Re/seed{0,1,2}.log` (9/9 files; master.log "Wave B DONE Mon Sep 28 10:23:21 PM CST 2026") |

**G4 aggregate (after Wf-278A path-mismatch fix)**: 14/14 pairs MEASURED; 5-metal test_auc 0.7209 ± 0.0458 (`results/wf124_p3_remaining_metals_g4/training_summary.json`).

### §1.2 TODO-119 P1.3 + P1.4 (CFM ablation on Blackwell GPU-0)

| Wave | Configs | Wall | Status | Evidence |
|---|---|---:|---|---|
| **C3** | `cfm_real_decode_lift_p1.3.yaml` (1 seed × 3 ep) | ~30 min | **PENDING** | `results/cfm_real_decode_lift_p1.3/` does NOT exist. Wf-276 §3 honest-negative: "TODO-119 P1.3 is NOT formally run on PlatinAI/DrugOOD cohorts — only MetalCytoToxDB has ad-hoc Wf-181/Wf-188 0.9980/0.9958." |
| **C4** | `cfm_per_innovation_ablation_p1.4.yaml` `a_full_stack` + `b_yuelbond_only` (2 configs) | ~60 min | **PENDING** | `results/cfm_per_innovation_ablation_p1.4/` does NOT exist. |
| **C5** | `cfm_per_innovation_ablation_p1.4.yaml` `c_w2_only` + `d_catherine_only` (2 configs) | ~60 min | **PENDING** | Same dir absent. Wf-276 §3: "Wf-276E-119P1.4 ... depends on P1.3 result as the 'all 3 innovations' control." |

**Blockers for C3-C5**: 16-hr Blackwell pre-launch scope (CFM ablation flag + W₂ gradient-path verification required before any wave); GPU-0 Blackwell idle but formal P1.3 deliverable needs PlatinAI 4-cell + DrugOOD per-cohort assays (not yet wired).

### §1.3 TODO-181 G5 (Wf-124 P3 50-epoch full sweep, RTX 5090 GPU-1)

| Wave | Pairs | Wall | Status | Evidence |
|---|---|---:|---|---|
| **C6** | Ru s0/s1/s2 (3 × 50 ep) | ~60 min | **PENDING** (partial) | Only `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` exists; log cut at `epoch 47/50 loss=0.4011 val_auc=0.8177` — NOT completed. Per Wf-276B §1: "G5 wrapper PID 609543 is Ru seed=0 mid-run (~epoch 36/50)." Ru/seed1 + Ru/seed2 directories absent. |
| **C7** | Ir s0/s1/s2 (3 × 50 ep) | ~60 min | **PENDING** | No `Ir/` subdirectory. |
| **C8** | Os s0/s1/s2 + Rh s0/s1/s2 (6 × 50 ep) | ~60 min | **PENDING** | No `Os/` or `Rh/` subdirectories. |
| **C9** | Re s0/s1/s2 + `30ep_vs_50ep_comparison.json` (3 × 50 ep) | ~60 min | **PENDING** | No `Re/` subdirectory; no comparison JSON. |

**Blockers for C6-C9**: G5 wrapper (`scripts/launch_wf124_p3_g5.sh`) launched Ru seed=0 but did NOT complete (mid-run abort, no ckpt, no per_seed_metrics.json). Per Wf-276B §1: wrapper exists but `--no-enforce-budget` flag MISSING (mitigates 1-hr wave discipline). Per Wf-279D §5: launchers skip all 3 pre-launch tools. **15/15 pairs PENDING.**

### §1.4 TODO-182 G6 (per-innovation ablation, RTX 5090 GPU-1)

| Wave | Configs | Runs | Wall | Status | Evidence |
|---|---|---:|---:|---|---|
| **C10** | `no_multi_task` Ru/Ir/Os | 9 | ~68 min | **PENDING** | `results/wf124_p3_per_innovation_ablation_g6/` empty (no files). |
| **C11** | `no_multi_task` Rh/Re + `no_wass` Ru | 9 | ~68 min | **PENDING** | Same. |
| **C12** | `no_wass` Ir/Os/Rh/Re | 12 | ~90 min | **PENDING** | Same. |
| **C13** | `no_kendall` Ru/Ir | 6 | ~45 min | **PENDING** | `--no-kendall` UNVERIFIED (per Wf-258 §1.7 + §4 G6 verdict). |
| **C14** | `no_kendall` Os/Rh/Re | 9 | ~68 min | **PENDING** | Same `--no-kendall` UNVERIFIED. |
| **C15** | `no_dann` Ru/Ir/Os | 9 | ~68 min | **PENDING** | `--no-dann` UNVERIFIED. |
| **C16** | `no_dann` Rh | 3 | ~23 min | **PENDING** | `--no-dann` UNVERIFIED. |
| **C17** | `no_dann` Re + `ablation_report.json` | 3 | ~23 min | **PENDING** | `--no-dann` UNVERIFIED. |

**Blockers for C10-C17**: G6 wrapper (`scripts/launch_wf124_p3_g6.sh`) NEVER launched. `--no-kendall` and `--no-dann` flags NOT confirmed in `train_dmpnn_multitask_wasserstein.py --help` (per Wf-258 §1.7 PRE-LAUNCH BLOCKER + `cfm_todo_gap_audit.md §3.4 gap #1`). Per Wf-276B §1: even if G6 launches, 30/60 tuples will skip due to unverified flags. **60/60 runs PENDING.**

---

## §2 Tally

| Metric | Value |
|---|---:|
| **n_waves_total** | 17 (C1–C17) |
| **n_waves_done** | **2** (C1, C2) |
| **n_waves_pending** | **15** (C3–C17) |
| **completion_pct** | **11.76%** (2/17) |

---

## §3 DONE wave → corresponding Wf mapping

| Wave | Corresponding Wf | Notes |
|---|---|---|
| **C1** | Wf-124 P3 (Ru s1/s2 + Ir s0/s1/s2) → aggregated by Wf-278A | 5 pairs MEASURED, test_auc 0.7483/0.6702 per metal |
| **C2** | Wf-124 P3 (Os s0/s1/s2 + Rh s0/s1/s2 + Re s0/s1/s2) → aggregated by Wf-278A | 9 pairs MEASURED, test_auc 0.7025/0.7227/0.7700 per metal |

Note: TODO-180 G4 14/14 pairs is the **only fully-DONE block** in the Cf-258 plan. Aggregate 5-metal test_auc = 0.7209 ± 0.0458 (n=14, Ru has only 2 seeds; Ru seed=0 was the Wf-124 P3 baseline at 0.7613).

---

## §4 PENDING wave blocker summary (15 waves)

| Block | Waves affected | Severity | Fix |
|---|---|---|---|
| **G5 wrapper abort + missing `--no-enforce-budget`** | C6–C9 (4 waves, 15 pairs) | HIGH | Re-launch `scripts/launch_wf124_p3_g5.sh`; add `--no-enforce-budget` flag to wrapper per Wf-276B §1 |
| **CFM ablation flag verification (PlatinAI/DrugOOD per-cohort)** | C3–C5 (3 waves) | HIGH | Run CFM ablation flag + W₂ gradient-path verify on GPU-0 Blackwell (Wf-269A pattern, 3-metal smoke); 16-hr Blackwell scope |
| **G6 `--no-kendall` / `--no-dann` flags UNVERIFIED** | C10–C17 (8 waves, 60 runs) | HIGH | Either add the 2 flags OR scope G6 to 2 verified configs (`no_multi_task` + `no_wass` via `--wasserstein-weight 0.0`); per Wf-258 §4 G4 verdict |
| **GPU-1 RTX 5090 over 10-hr budget** | C1–C17 combined | MEDIUM | Per Wf-258 §3.2: 12.55 hr on GPU-1 exceeds 10h user-OK by 2.55 hr; Option A (drop C10–C17) or Option C (extend to 13h) requires fresh user OK |
| **Launcher × wave-tool integration (Wf-279D)** | All 15 PENDING | MEDIUM | Launchers skip `pre_flight_check.py` / `per_wave_checklist.py` / `wave_budget_tracker.py`; Wf-279E fix recommended (~45 LOC across 3 wrappers) |
| **Per-wave fresh user OK gate** | All 15 PENDING | PROTOCOL | Per `kill_then_adapt`: every wave ≤ 1 hr requires FRESH user OK before launch; NO batch-launch |

---

## §5 Honest framing (party-twenty-spirit §3.5)

- **DONE claims are MEASURED**: Wf-124 P3 + Wf-278A aggregator provide test_auc per-seed numbers for all 14 G4 pairs; master.log timestamps confirm Wave A + Wave B completion at 22:12:59 + 22:23:21 (2026-09-28).
- **PENDING claims are PENDING**: NO `results/cfm_real_decode_lift_p1.3/` or `results/cfm_per_innovation_ablation_p1.4/` directories exist; G5 wrapper launched Ru seed=0 only and aborted at epoch 47/50 (no ckpt, no per_seed_metrics.json); G6 wrapper never launched (empty output dir).
- **NO fabrication**: G6 60 runs are PROJECTED, NOT MEASURED; C3–C5 CFM ablation are PROJECTED, NOT MEASURED.
- **Smaller-metal caveat** (per Wf-124 P3 §5): Rh/Os/Re cohorts < 1.2k rows → per-seed std likely ±0.10; honest reporting only.
- **No best-seed selection** (per `mol-metal-gpu-execution` §7): all 14 G4 seeds reported; std IS the answer.

---

## §6 Recommended next-step sequencing (per `mol-metal-gpu-execution` §6 + Wf-276E §4)

1. **Wf-279E launcher fix** (~45 LOC, CPU only) — wire `pre_flight_check.py` into G4/G5/G6 launchers; address Wf-279D §6 recommendation.
2. **Wf-276E-280 G4 (already DONE; just sign-off)** — TODO-180 G4 14/14 MEASURED; write `paper §4 LEAD` aggregate cite (test_auc 0.7209 ± 0.0458).
3. **Wf-276E-281 G5 re-launch** (5 hr, GPU-1) — Re-run `scripts/launch_wf124_p3_g5.sh` after Wf-279E fix; completes C6–C9.
4. **Wf-276E-119P1.3 + P1.4** (23.5 hr, GPU-0 Blackwell) — Sequential C3 → C4 → C5; needs CFM ablation flag + W₂ gradient-path verify first.
5. **Wf-276E-282 G6** (15–20 hr, GPU-1 serial) — After Wf-279E fix + G5 exit; 60 runs at 2–3× Wf-258 PROJECTED 7.5 hr per G5 MEASURED pattern; 30/60 CONFIRMED only if `--no-kendall`/`--no-dann` added.

Total wall (post Wf-279E fix): 5 + 23.5 + 17.5 ≈ **46 hr** (vs Wf-258 §2.2 audit-conservative 37.8 hr; +22% for G6 MEASURED-vs-PROJECTED reality).

---

*Generated by Wf-280D (2026-09-29). Read-only audit. No file modifications outside this report.*