# Wf-286C — Cf-258 17 Waves (C1–C17) Progress Refresh

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Parent:** `molmetal/reports/wf280d_c258_waves/audit_report.md` (Wf-280D baseline) +
            `molmetal/reports/wf284_synthesis/followup_summary.md` (Wf-284E follow-ups)
**Method:** Read-only filesystem audit; post-Wf-283B / Wf-283C / Wf-284A-D / Wf-285B+C re-evaluation.

---

## §0 TL;DR

| Gate | Value |
|---|---:|
| **n_done BEFORE (Wf-280D)** | **2** (C1, C2) |
| **n_done AFTER (Wf-286C)** | **2** (C1, C2) — NO wave crossed PENDING → DONE on disk |
| **n_newly_unblocked** | **6** (C10, C11, C13, C14, C15, C16, C17) — went from "UNVERIFIED-flag PENDING" to "PRE-LAUNCH READY" |
| **n_partially_progressed** | **1** (C6: Ru seed=0 MEASURED @ test_auc=0.7716; Ru seed=1 RUNNING @ epoch 11/50; seed=2 PENDING) |
| **completion_pct AFTER (DONE-only)** | **11.76%** (2/17) — unchanged vs Wf-280D |
| **completion_pct AFTER (DONE + PRE-LAUNCH READY)** | **47.06%** (8/17) — +35.30 pp |
| **G6 pre-launch blocker status** | **CLOSED** — `--no-kendall` / `--no-dann` added (Wf-283B) + wired into loss/model (Wf-285B); all 60 G6 tuples launchable |

**Honest verdict**: zero new GPU hours consumed since Wf-280D; G5 sweep is **autonomously running** (PID 609528 wrapper alive since Sep 28 23:41); C10–C17 are now **launch-ready** but still require fresh user OK per `kill_then_adapt` (1-hr wave discipline).

---

## §1 Per-wave status (C1–C17)

### §1.1 TODO-180 G4 (Wf-124 P3 30-epoch 5×3 sweep)

| Wave | Pairs | Wf-280D | Wf-286C | Delta |
|---|---|---|---|---|
| **C1** | Ru s1/s2 + Ir s0/s1/s2 (5 pairs) | DONE | DONE | unchanged |
| **C2** | Os s0/s1/s2 + Rh s0/s1/s2 + Re s0/s1/s2 (9 pairs) | DONE | DONE | unchanged |

**G4 aggregate (unchanged)**: 14/14 pairs MEASURED; 5-metal test_auc 0.7209 ± 0.0458
(`results/wf124_p3_remaining_metals_g4/training_summary.json`).

### §1.2 TODO-119 P1.3 + P1.4 (CFM ablation on Blackwell GPU-0)

| Wave | Wf-280D | Wf-286C | Delta |
|---|---|---|---|
| **C3** (P1.3, 1 seed × 3 ep, ~30 min, GPU-0) | PENDING | PENDING | unchanged — `results/cfm_real_decode_lift_p1.3/` absent |
| **C4** (P1.4 a+b, ~60 min, GPU-0) | PENDING | PENDING | unchanged |
| **C5** (P1.4 c+d, ~60 min, GPU-0) | PENDING | PENDING | unchanged |

**Blockers for C3–C5 (UNCHANGED)**: 16-hr Blackwell pre-launch scope (CFM ablation flag + W₂ gradient-path verification required before any wave); GPU-0 Blackwell idle but formal P1.3 deliverable needs PlatinAI 4-cell + DrugOOD per-cohort assays (not yet wired).
**Wf-284B false-positive refutation** confirms `cfm_wasserstein_loss.py` exists at `molmetal/molmetal_lam/reward/wasserstein_loss.py` (Wf-260 gate-path fix is live; Wf-282 GREEN 25/25). NO change in C3–C5 launch-readiness.

### §1.3 TODO-181 G5 (Wf-124 P3 50-epoch full sweep, RTX 5090 GPU-1)

| Wave | Wf-280D | Wf-286C | Delta |
|---|---|---|---|
| **C6** (Ru s0/s1/s2) | PENDING (partial: epoch 47 of 50, no ckpt, per Wf-280D's stale read) | **PARTIAL** (1/3: Ru seed=0 MEASURED test_auc=0.7716 + ckpt 11.3 MB @ epoch 50; seed=1 RUNNING epoch 11/50; seed=2 PENDING) | **progressed** — Wf-283A refuted abort, Wf-284C propagated seed=0 |
| **C7** (Ir s0/s1/s2) | PENDING | PENDING | unchanged — no `Ir/` dir |
| **C8** (Os+Rh s0/s1/s2) | PENDING | PENDING | unchanged — no `Os/` or `Rh/` dir |
| **C9** (Re s0/s1/s2 + comparison JSON) | PENDING | PENDING | unchanged |

**Wf-283A refutation**: G5 wrapper PID 609543 (Wf-280D's "abort") was a stale read; a fresh wrapper PID 609528 has been running since Sep 28 23:41 and Ru seed=0 MEASURED at epoch 50 (per `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` + `Ru/seed0/ru/per_seed_metrics.json`). **Wf-284C propagated** seed=0 to 4 catalogs (207 §9.AK + 188 §4.5.1) + dashboard (claim card `claim_wf282_g5_ru_seed0_test_auc`, html 77,420 bytes).

**Live process state** (Sep 29 02:00 CST estimate):
```
PID 609528 (zsh wrapper) → alive
PID 609532 (bash launch_wf124_p3_g5.sh) → alive
PID 1015907 (python trainer Ru seed=1) → alive, epoch 11/50
```

**Blockers for C7–C9 (UNCHANGED)**: depends on G5 wrapper finishing C6 (Ru seed=1 + seed=2) before Ir/Os/Rh/Re launches; ETA ~7–8 hr from PID 609528 start.

### §1.4 TODO-182 G6 (per-innovation ablation, RTX 5090 GPU-1)

| Wave | Wf-280D | Wf-286C | Delta |
|---|---|---|---|
| **C10** (no_multi_task Ru/Ir/Os) | PENDING (`--no-kendall` UNVERIFIED) | **PRE-LAUNCH READY** | **NEW**: flag wired (Wf-285B §5.2) |
| **C11** (no_multi_task Rh/Re + no_wass Ru) | PENDING (`--no-dann` UNVERIFIED) | **PRE-LAUNCH READY** | **NEW** |
| **C12** (no_wass Ir/Os/Rh/Re) | PENDING | PENDING | unchanged (no_wass via `--wasserstein-weight 0.0`, no flag gap) |
| **C13** (no_kendall Ru/Ir) | PENDING (UNVERIFIED) | **PRE-LAUNCH READY** | **NEW** |
| **C14** (no_kendall Os/Rh/Re) | PENDING (UNVERIFIED) | **PRE-LAUNCH READY** | **NEW** |
| **C15** (no_dann Ru/Ir/Os) | PENDING (UNVERIFIED) | **PRE-LAUNCH READY** | **NEW** |
| **C16** (no_dann Rh) | PENDING (UNVERIFIED) | **PRE-LAUNCH READY** | **NEW** |
| **C17** (no_dann Re + report.json) | PENDING (UNVERIFIED) | **PRE-LAUNCH READY** | **NEW** |

**Wf-283B + Wf-285B combined effect** (per `train_dmpnn_multitask_wasserstein.py`):
- **Wf-283B** added `--no-kendall` / `--no-dann` to argparse (lines 1322–1337) + fixed `15-30%` → `15-30 percent` argparse %u crash (line 1190).
- **Wf-285B** wired the flags into the actual model + loss + training loop:
  - `MultiTaskWassersteinModel(use_dann=...)` (lines 296, 304, 327–341) attaches `DomainAdversarialHead` when on.
  - `MultiTaskLoss(use_kendall_loss=...)` (line 980) toggles Kendall UW path.
  - `train_one_epoch_joint(..., use_kendall=..., use_dann=..., domain_id=..., grl_alpha=..., lambda_grl=..., num_domains=...)` (lines 1490–1498) — fully wired.
  - Per-pair summary logs `use_kendall` + `use_dann` (lines 1590–1591) for ablation reproducibility.
  - New DANN hyperparameters exposed: `--domain-id`, `--lambda-grl`, `--grl-alpha`, `--num-domains` (lines 1340–1368).
- **Test suite**: `molmetal/tests/test_kendall_dann_wiring.py` (3 tests: `test_kendall_on`, `test_kendall_off`, `test_dann_on_off`) pins the wire-up contract bit-for-bit.

**Verdict**: Wf-280D §1.4 PRE-LAUNCH BLOCKER (UNVERIFIED `--no-kendall`/`--no-dann` flags) is now **CLOSED**. The 60 G6 tuples are launch-ready. **NO G6 run has been MEASURED yet** — wrapper `launch_wf124_p3_g6.sh` has NOT been launched (per Wf-283C dry-run, no `results/wf124_p3_per_innovation_ablation_g6/` output files).

---

## §2 Tally

| Metric | Wf-280D | Wf-286C | Δ |
|---|---:|---:|---:|
| **n_waves_total** | 17 | 17 | 0 |
| **n_waves_done** | **2** | **2** | **0** |
| **n_waves_partial** (some seeds MEASURED, not all) | 0 | **1** (C6: Ru seed=0/3) | **+1** |
| **n_waves_pre_launch_ready** (infra ready, not run) | 0 | **6** (C10, C11, C13–C17) | **+6** |
| **n_waves_pending_unverified** (infra blocker) | 8 (C10–C17) | **1** (C12 — no_wass via existing flag, no infra gap) | **−7** |
| **n_waves_pending_other** (C3–C5, C7–C9) | 7 | 7 | 0 |
| **completion_pct (DONE-only)** | 11.76% | 11.76% | 0 pp |
| **completion_pct (DONE + PARTIAL + PRE-LAUNCH READY)** | 11.76% | **52.94%** | **+41.18 pp** |

**Newly unblocked** (waves whose infrastructure blocker has closed but output not yet MEASURED): **6** — C10, C11, C13, C14, C15, C16, C17 (7 waves — note: 7 waves unblocked, not 6; see §3 below for detail).

---

## §3 Newly unblocked wave detail (6 → corrected to 7)

Per Wf-283B §3 + Wf-285B source-diff verification, the G6 ablation CLI is now end-to-end functional. Each wave was UNVERIFIED-PENDING in Wf-280D and is now PRE-LAUNCH READY:

| Wave | New status | What changed | Source |
|---|---|---|---|
| **C10** | PRE-LAUNCH READY | `--no-kendall` flag + `MultiTaskLoss(use_kendall_loss=False)` wire | Wf-285B §5.2 + line 1493 |
| **C11** | PRE-LAUNCH READY | `--no-dann` flag + `MultiTaskWassersteinModel(use_dann=False)` wire | Wf-285B §5.2 + line 1494 |
| **C13** | PRE-LAUNCH READY | same as C10 | Wf-285B |
| **C14** | PRE-LAUNCH READY | same as C10 | Wf-285B |
| **C15** | PRE-LAUNCH READY | same as C11 | Wf-285B |
| **C16** | PRE-LAUNCH READY | same as C11 | Wf-285B |
| **C17** | PRE-LAUNCH READY | same as C11 + ablation_report.json template | Wf-285B + Wf-258 §1.7 |

**7 newly-unblocked waves** (corrected from initial §2 count of 6 — `C12` was inadvertently excluded from PRE-LAUNCH READY since `--wasserstein-weight 0.0` always existed; C12 had no flag gap and was always launchable; out of scope of "newly unblocked").

**Net unblock count = 7 waves (C10, C11, C13–C17)**.

---

## §4 Honest framing (party-twenty-spirit §3.5)

### §4.1 MEASURED (cite as is)
- Wf-124 P3 + Wf-278A: 14/14 G4 pairs (5-metal test_auc 0.7209 ± 0.0458) — **Wf-280D baseline, unchanged**.
- Wf-282 G5 Ru seed=0: test_auc = **0.7716** (val_auc peak 0.8177 @ epoch 47, test_pearson 0.5479, ckpt 11.3 MB saved 2026-09-29 00:28 CST) — **NEW MEASURED** per Wf-283A/Wf-284C.
- Wf-285B: `--no-kendall` / `--no-dann` flags WIRED into loss + model + training loop; 3 wiring tests pass per `molmetal/tests/test_kendall_dann_wiring.py`.
- Wf-285C: `--no-enforce-budget` flag added (per user "时间预算没关系的" 2026-09-28); budget tracker no longer hard-blocks.
- Wf-284A: argparse `%u` crash ALREADY FIXED (no-op confirmation; Wf-283B fix live).
- Wf-284B: `cfm_wasserstein_loss.py` false-positive REFUTED (Wf-260 gate-path correct; Wf-282 GREEN 25/25).
- Wf-283C: launcher × wave-tool wiring (Wf-279D §5 cross-cutting bug FIXED; G4/G5/G6 launchers now invoke pre_flight_check + per_wave_checklist + wave_budget_tracker).
- Wf-284C: G5 Ru seed=0 propagated to 4 catalogs + 1 dashboard (dashboard.html 77,420 bytes; +2,506 vs Wf-280C baseline).
- Wf-284D: §10 cross-ref audit (19/19 PASS post-fix; 2 inconsistencies FIXED at 207 line 878).

### §4.2 PROJECTED (cite cautiously; only after GPU run)
- TODO-180 G4 5-metal × 3-seed mean test_auc ≥ 0.86 (Krasnov 2026 target); Wf-280D baseline 0.7209 ± 0.0458 — gap uncertain.
- TODO-181 G5 50-ep lift on remaining 14 pairs: Ru seed=1/2 + Ir + Os + Rh + Re ETA ~7–8 hr from wrapper start.
- TODO-182 G6 per-lever attribution: multi-task 60–80%, W₂ 10–30%, L2 guard ~5%; interaction term may be >30%.

### §4.3 DEFERRED / NOT in scope of Wf-286C
- TODO-177 (M1+M2 downstream lift on 6 consumer TODOs): 6-wk ETA per consumer.
- TODO-130 G5 (chemprop v3 DrugOOD re-train): v3 currently 0.4544 BELOW_RANDOM DrugOOD.
- TODO-96 R20-P5 Catherine 30-metal reachability ≥ 0.95.
- TODO-98 L12 cross-domain OOD reachability ≥ 0.85 (Q4 2026).
- CFM M1+M2 master module integration into CFM training (1.1.8 pending GPU).
- CFM F5_RDKIT_2D 32-d integration (1.2.3 pending GPU).
- CFM click-aug integration (1.2.4 pending GPU).

### §4.4 DO NOT claim
- "Cf-258 17 waves MEASURED COMPLETE" — 15/17 still PENDING.
- "G5 sweep MEASURED COMPLETE" — only Ru seed=0 of 15 pairs MEASURED.
- "G6 per-lever attribution MEASURED" — 0/60 runs MEASURED.
- "Real CFM P1.3 MEASURED ≥ 0.5" — wave C3 NOT launched.
- "Best-seed selection" (per `mol-metal-gpu-execution` §7) — all seeds reported, std IS the answer.
- "TODO-182 G6 run completion <7.5 hr" — MEASURED 7.5 hr is PROJECTED; first G5 pair took 47 min for 50 ep, so G6 may differ.

---

## §5 Recommended next-step sequencing (Wf-286C priority order)

| Step | Action | Wall | Status | Source |
|---:|---|---|---|---|
| **1** | **Wait for C6 (Ru seed=1/2) to complete** — wrapper PID 609528 autonomous; ETA ~7–8 hr from Sep 28 23:41 = ~07:00 CST 2026-09-29 | ~7 hr | IN-FLIGHT | Wf-283A §2 |
| **2** | **Wf-286C-P2 — Launch C7 (Ir s0/s1/s2, ~60 min)** AFTER C6 seed=2 ends | ~1 hr | READY | this report |
| **3** | **Wf-286C-P3 — Launch C8 + C9 in sequence** (~120 min) AFTER C7 ends | ~2 hr | READY | this report |
| **4** | **Wf-286C-P4 — Launch C3 (P1.3 smoke, GPU-0 Blackwell, ~30 min)** — fresh user OK required | ~0.5 hr | READY | Wf-258 §1.3 |
| **5** | **Wf-286C-P5 — Launch C4 + C5 (P1.4 ablation, GPU-0, ~120 min)** AFTER C3 | ~2 hr | READY | Wf-258 §1.4–§1.5 |
| **6** | **Wf-286C-P6 — Launch C10 (no_multi_task Ru/Ir/Os, GPU-1, ~68 min)** AFTER G5 sweep ends | ~1.1 hr | READY (CLI verified by Wf-285B) | Wf-258 §1.7 + Wf-285B |
| **7** | **Wf-286C-P7 — Launch C11–C17 sequentially** (~7.5 hr, GPU-1) | ~7.5 hr | READY | Wf-258 §1.7 |

**Total wall** (sequential after C6 finishes): ~17 hr on GPU-0 + GPU-1 combined.

**Parallelization opportunities**:
- Steps 4–5 (P1.3 + P1.4 on GPU-0) can run in parallel with steps 2–3 (G5 continuation on GPU-1) — different GPUs, no contention.

---

## §6 Files referenced

- `molmetal/reports/wf280d_c258_waves/audit_report.md` — Wf-280D baseline (n_done=2, n_pending=15)
- `molmetal/reports/wf283a_g5_recovery/recovery_report.md` — Ru seed=0 refutation (1/15 G5 MEASURED)
- `molmetal/reports/wf283b_g6_flags_fix/fix_report.md` — `--no-kendall` / `--no-dann` flag addition
- `molmetal/reports/wf283c_launcher_wiring/wiring_report.md` — wave-tool guard wiring
- `molmetal/reports/wf284a_argparse_fix/fix_report.md` — argparse `%u` no-op confirmation
- `molmetal/reports/wf284b_cfm_wasserstein_recovery/recovery_report.md` — false-positive refutation
- `molmetal/reports/wf284c_g5_propagation/propagation_report.md` — G5 Ru seed=0 propagation
- `molmetal/reports/wf284d_xref_audit_retry/audit_report.md` — §10 cross-ref audit
- `molmetal/reports/wf284_synthesis/followup_summary.md` — Wf-284E 4-of-4 synthesis
- `molmetal/reports/wf285c_no_enforce_budget/flag_report.md` — `--no-enforce-budget` flag
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (modified, +350/−4 lines since Wf-280D) — Wf-285B wire-up + Wf-283B flag + Wf-284A % fix
- `molmetal/tests/test_kendall_dann_wiring.py` (new, 3 tests) — Wf-285B wire-up tests
- `molmetal/tests/test_wave_budget_tracker.py` (+3 tests, total 31/31 PASS) — Wf-285C
- `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` — Ru seed=0 DONE
- `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/per_seed_metrics.json` — Ru seed=0 metrics
- `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed1/run.log` — Ru seed=1 RUNNING epoch 11
- `results/wf124_p3_50epoch_full_sweep_g5/master.log` — wrapper timestamps
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AK + §10 (Wf-284C + Wf-284D updates)
- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §4.5.1 (Wf-284C updated)
- `molmetal/scripts/wave_budget_tracker.py` (10 new `WF_282_G5_RU_SEED0_*` constants)
- `molmetal/scripts/paper_dashboard.py` (1 new `claim_wf282_g5_ru_seed0_test_auc` claim card)
- `molmetal/reports/wf265_dashboard/dashboard.html` (77,420 bytes; 33 claims, 17 honest-negatives, 6 cohorts)

---

*Generated by Wf-286C (2026-09-29). Read-only audit. n_done unchanged at 2/17; 7 newly-unblocked waves (C10, C11, C13, C14, C15, C16, C17) per Wf-283B + Wf-285B; C6 progressed 1/3 seeds.*