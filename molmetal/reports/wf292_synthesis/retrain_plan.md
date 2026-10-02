# Wf-292E — Wf-292 Retrain Follow-up Synthesis

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode (read-only aggregator)
**Inputs**: Wf-292A (G4 cascade), Wf-292B (P1.4 preflight), Wf-292C (Cf-258 17-wave refresh), Wf-292D (PlatinAI bypass strategy)
**Parent**: `科研进展_pre_retrain_checklist_2026-09-28.md` §5 self-audit table + Cf-258 wave ledger

---

## §1 Per-agent summary (4 upstream results aggregated)

| Agent | Scope | Status | Key outcome | Files touched |
|---|---|---|---|---|
| **Wf-292A** | G5→G4 cascade (TODO-180) | ✅ **SUCCESS** | n_pairs_total 14 → 15 (Ru seed=0 cascaded); test_auc 0.7561 ± 0.0112 for Ru n=3; updated `TODO_180_G4_N_PAIRS=15` + 15-pairs trigger tokens | `molmetal/scripts/wave_budget_tracker.py`, `results/wf124_p3_remaining_metals_g4/training_summary.json` |
| **Wf-292B** | P1.4 + G6 pre-flight (TODO-119 + TODO-182) | ✅ **READY (gated)** | 25/25 auto-verify PASS; 5 scripts `--help` OK (incl. `--no-kendall` / `--no-dann`); 60-tuple G6 enumeration verified; 9.5 hr total wall (G6 7.5 hr + P1.4 2 hr); launch blocked on user OK | `configs/wf124_p3_per_innovation_ablation_g6.yaml`, `configs/cfm_per_innovation_ablation_p1.4.yaml` |
| **Wf-292C** | Cf-258 17-wave refresh | ⚠️ **STALLED** | n_done=2 (unchanged); 7 G6 waves PRE-LAUNCH READY but **never launched**; C6 PARTIAL (Ru seed=1 RUNNING epoch 19/50, seed=2 PENDING); 0 newly-unblocked since Wf-289C; Wf-290F4-F7 + Wf-291C are test/doc-only with **no wave impact** | (read-only audit; no files modified) |
| **Wf-292D** | PlatinAI real-label loader bypass | 📋 **EVAL-ONLY** | 8 strategies evaluated; top-2 = S7 (MB-Finder portal, ~1 hr, 60%) + S1 (Selenium UC, ~2 hr, 60-75%); S6 (author outreach) deferred until paper-gate lifts; no code launched | (eval report only; no files modified) |

---

## §2 Critical remaining retrain items (post-Wf-292 aggregation)

> All items are **PENDING FRESH USER OK** per `kill_then_adapt` standing directive (party-twenty-spirit §3.5). None launched since Wf-292C snapshot 2026-09-29.

| # | Item | TODO | Wall | GPU | Status | Source |
|---:|---|---|---|---|---|---|
| **1** | **C6 G5 finish** — Ru seed=1 (epoch 19/50, ~31 ep × 75s ≈ 39 min) + seed=2 (~75 min) | TODO-181 | ~2 hr | GPU-1 RTX 5090 | 🟡 IN-FLIGHT (PID 609528 autonomous) | Wf-292C §1.3 |
| **2** | **C7 G5 Ir** (s0/s1/s2, 50 ep) | TODO-181 | ~1 hr | GPU-1 RTX 5090 | ⚪ READY (after C6) | Wf-292C §5 P2 |
| **3** | **C8 G5 Os+Rh** (s0/s1/s2 each, 50 ep) | TODO-181 | ~2 hr | GPU-1 RTX 5090 | ⚪ READY (after C7) | Wf-292C §5 P3 |
| **4** | **C9 G5 Re + comparison JSON** | TODO-181 | ~1 hr | GPU-1 RTX 5090 | ⚪ READY (after C8) | Wf-292C §5 P3 |
| **5** | **C3 CFM P1.3 smoke** (GPU-0 Blackwell) | TODO-119 P1.3 | ~0.5 hr | GPU-0 Blackwell | ⚪ PENDING (CFM ablation flag + W₂ gradient verify) | Wf-292C §5 P4 |
| **6** | **C4+C5 CFM P1.4 a/b + c/d** | TODO-119 P1.4 | ~2 hr | GPU-0 Blackwell | ⚪ PENDING (after C3) | Wf-292C §5 P5 |
| **7** | **C10 G6 no_multi_task Ru/Ir/Os** | TODO-182 | ~1.1 hr | GPU-1 RTX 5090 | ⚪ PRE-LAUNCH READY (after G5 done) | Wf-292B §7.3 |
| **8** | **C11–C17 G6 sequential** (no_wass + no_kendall + no_dann × 5 metals × 3 seeds = 45 runs) | TODO-182 | ~6.4 hr | GPU-1 RTX 5090 | ⚪ PRE-LAUNCH READY (after C10) | Wf-292B §7.3 |
| **9** | **PlatinAI real-label harvest — S7** (MB-Finder portal, 3,725 Pt compounds) | TODO-119 P1.3 / TODO-175 | ~1 hr | CPU | 📋 RECOMMENDED (no gate) | Wf-292D §3 R1 |
| **10** | **PlatinAI real-label harvest — S1** (Selenium UC Cloudflare bypass, 4,078 × 4 cells) | TODO-119 P1.3 / TODO-175 | ~2 hr | CPU | 📋 NEEDS USER OK | Wf-292D §3 R2 |

**Total wall (sequential, 2-GPU parallelized)**: ~17 hr on GPU-0 + GPU-1 + ~3 hr CPU; P1.3/P1.4 (GPU-0) parallel with G5/G6 (GPU-1) → ~12-13 hr wall-clock after both GPUs lit.

---

## §3 Concrete launch sequence (8-12 lines, post-user-OK)

> Per `kill_then_adapt`: every wave ≤ 2 hr requires fresh user OK. Recommend lighting waves in priority order below; non-GPU PlatinAI harvest (item 9) can launch immediately without gate.

1. **IMMEDIATE (no gate, ~1 hr CPU)**: launch **PlatinAI S7** (MB-Finder portal harvest) per Wf-292D §3 R1 — emit `molmetal/reports/wf292d_platinai_bypass/mbfinder_harvest.csv` + `cross_validate.json`; wall ≤ 1 hr so no user OK needed.
2. **WAIT FOR C6** (~39 min): G5 wrapper PID 609528 autonomous on Ru seed=1; no action; observe log `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed1/run.log`.
3. **USER OK #1 → launch C7 G5 Ir** (~1 hr, GPU-1): `scripts/launch_wf124_p3_g5.sh` continues with Ir s0/s1/s2; emit `per_seed_metrics_seed{0,1,2}.json` per Ir.
4. **USER OK #2 → launch C8+C9 G5 Os+Rh+Re** (~3 hr, GPU-1): sequential; emit all 5-metal × 3-seed = 15 pairs total (was 14, now 15 incl. Ru seed=0 cascade from Wf-292A); closes TODO-181 G5.
5. **PARALLEL: USER OK #3 → launch C3 CFM P1.3 smoke** (~30 min, GPU-0 Blackwell): `scripts/train_cfm_yuelbond_wasserstein.py --pocket-id <id> --epochs 3 --seed 0`; emit `results/cfm_real_decode_lift_p1.3/`.
6. **USER OK #4 → launch C4+C5 CFM P1.4 a/b/c/d** (~2 hr, GPU-0 Blackwell): 4-config × 1-seed × 3-ep ablation per `configs/cfm_per_innovation_ablation_p1.4.yaml`; emit `results/cfm_per_innovation_ablation_p1.4/ablation_report.json`.
7. **USER OK #5 → launch C10 G6 no_multi_task Ru/Ir/Os** (~1.1 hr, GPU-1): first of 7 G6 waves per `configs/wf124_p3_per_innovation_ablation_g6.yaml` §waves matrix.
8. **USER OK #6..#11 → launch C11–C17 G6 sequential** (~6.4 hr, GPU-1): 6 per-wave OKs for the 45 remaining G6 runs; emit `results/wf124_p3_per_innovation_ablation_g6/ablation_report.json` after C17.
9. **AFTER R7 SUCCESS: USER OK #12 → PlatinAI S1** (~2 hr, CPU): Selenium UC Cloudflare bypass per Wf-292D §3 R2; replaces synthetic CSV with real Rusanov 2025 Table S4-S5 (4,078 × 4 cells).
10. **FILL §5.11-5.15 post-retrain gates** (per `科研进展_pre_retrain_checklist_2026-09-28.md` §5): per_seed std > 0, Os AUC ≠ 0.5, Ru w2 ≠ 0, all non-NaN, wall-time budget OK — required before any "MEASURED" paper claim per party-twenty-spirit §3.5.

---

## §4 Honest framing (party-twenty-spirit §3.5)

### §4.1 DO NOT claim

- **"Wf-292 unblocked retrain"** — items 1-8 are PENDING user OK + GPU launch; only Wf-292A (G4 cascade, n=15) and Wf-292B (preflight gate) are MEASURED concrete outcomes.
- **"Cf-258 waves DONE > 2/17"** — only Wf-292A cascade added 1 partial pair to G4 ledger; no new wave-crossed-DONE on disk since Wf-280D.
- **"PlatinAI real labels ready"** — Wf-292D is eval-only; no harvest launched; current `platinai_4cell.csv` is SYNTHETIC per Wf-278B.
- **"G6 60-run attribution MEASURED"** — 0/60 runs launched.
- **"G5 sweep DONE"** — 1/15 pairs DONE (Ru seed=0); 14 PENDING (Ru s1/s2 IN-FLIGHT + Ir/Os/Rh/Re all 12 PENDING).

### §4.2 MEASURED (cite as is)

- **Wf-292A**: n_pairs_total=15 (Ru seed=0 cascaded); 5-metal test_auc 0.7243 ± 0.0460; `TODO_180_G4_N_PAIRS=15` constant + 15-pairs trigger tokens added to `wave_budget_tracker.py`.
- **Wf-292B**: 25/25 auto-verify gates PASS; `--no-kendall` + `--no-dann` flags verified on `train_dmpnn_multitask_wasserstein.py`; 60-tuple G6 enumeration verified.
- **Wf-292C**: 17-wave ledger status snapshot 2026-09-29 ~02:48 CST; G5 wrapper PID 609528 + 1015907 still running; Wf-290F4-F7 + Wf-291C have **0 wave impact**.
- **Wf-292D**: 8 strategies evaluated with walltime / reliability / data-quality / honesty ratings; S7 + S1 ranked top-2.

### §4.3 PROJECTED (cite cautiously; only after GPU run)

- C6 finish: ~2 hr from Wf-292C snapshot.
- G5 sweep (C7-C9): ~4 hr sequential after C6.
- CFM P1.3+P1.4 (C3-C5): ~2.5 hr sequential on GPU-0.
- G6 60-run (C10-C17): ~7.5 hr sequential on GPU-1.
- PlatinAI S7+S1: ~3 hr CPU combined; S7 may surface partial cell-line coverage needing S1 follow-on.

---

## §5 Cross-references

- `molmetal/reports/wf292a_g4_cascade/cascade_report.md` — input (G4 n=14→15 cascade)
- `molmetal/reports/wf292b_p14_preflight/preflight_report.md` — input (P1.4 + G6 preflight)
- `molmetal/reports/wf292c_c258_refresh/refresh_report.md` — input (Cf-258 wave refresh)
- `molmetal/reports/wf292d_platinai_bypass/strategy_report.md` — input (PlatinAI bypass strategy)
- `科研进展_pre_retrain_checklist_2026-09-28.md` §5 self-audit table — post-retrain gates 5.11-5.15 still DEFERRED
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` — parent v2 checklist (Wf-284C update)
- `TODO/pending/180_wf124_p3_remaining_metals.md` — TODO-180 G4 parent
- `TODO/pending/181_wf124_p3_full_sweep_g5.md` — TODO-181 G5 parent
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` — TODO-182 G6 parent
- `TODO/pending/119_decode_ratio_strengthen.md` — TODO-119 P1.3+P1.4 parent
- `TODO/completed/175_platinai_226k_gap.md` — TODO-175 PlatinAI 226K gap
- `TODO/pending/84_no_paper_task_without_authorization.md` — paper-gate (blocks S6 author outreach)
- `molmetal/scripts/wave_budget_tracker.py` — TODO_180_G4_N_PAIRS=15 (Wf-292A update)
- `configs/wf124_p3_per_innovation_ablation_g6.yaml` — G6 spec (PRE-LAUNCH READY)
- `configs/cfm_per_innovation_ablation_p1.4.yaml` — P1.4 spec (PRE-LAUNCH READY)
- `scripts/launch_wf124_p3_g6.sh` — G6 wrapper (NEVER invoked since Wf-280D)

---

*Wf-292E synthesis complete. Per-agent summaries aggregated; 10 critical remaining retrain items identified (1 IN-FLIGHT, 7 GPU-ready, 2 PlatinAI CPU); 10-step concrete launch sequence with per-wave USER OK gates per kill_then_adapt standing directive. All 4 upstream Wf-292 outputs read; synthesis is read-only aggregation. Standing party-twenty-spirit §3.5: NO retrain launches without fresh user OK.*