# Wf-292C — Cf-258 17 Waves (C1–C17) Progress Refresh (post-Wf-290/291 fixes)

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Parent reports:**
- `molmetal/reports/wf280d_c258_waves/audit_report.md` (Wf-280D baseline, n_done=2)
- `molmetal/reports/wf286c_c258_refresh/refresh_report.md` (Wf-286C refresh, n_done=2, 7 newly-unblocked)
- `molmetal/reports/wf289c_c258_refresh/refresh_report.md` (Wf-289C refresh, n_done=2, 0 newly-unblocked since Wf-286C)
- `molmetal/reports/wf291c_pipeline_root_cause/root_cause.md` (Wf-291C pipeline-regression class root-cause)

**Method:** Read-only filesystem audit + cross-reference of Wf-290F4/F5/F6/F7 + Wf-291C outputs against the Wf-280D baseline wave ledger.

---

## §0 TL;DR

| Gate | Wf-280D | Wf-286C | Wf-289C | **Wf-292C (this)** | Δ vs Wf-280D |
|---|---:|---:|---:|---:|---:|
| **n_done (DONE-only)** | **2** | **2** | **2** | **2** | **0** |
| **n_partially_progressed** | 0 | 1 (C6: Ru seed=0/3) | 1 (C6: Ru seed=0/3; seed=1 RUNNING epoch 15/50) | **1** (C6: Ru seed=0/3; seed=1 RUNNING **epoch 19/50**; seed=2 PENDING) | +1 |
| **n_pre_launch_ready** | 0 | 7 (C10, C11, C13–C17) | 7 | **7** (unchanged) | +7 |
| **n_pending_unverified** | 8 (C10–C17) | 1 (C12 only) | 1 (C12) | **1** (C12 unchanged) | −7 |
| **n_pending_other** (C3–C5, C7–C9) | 7 | 7 | 7 | **7** (unchanged) | 0 |
| **completion_pct (DONE-only)** | 11.76% | 11.76% | 11.76% | **11.76%** | 0 pp |
| **completion_pct (DONE + PARTIAL + PRE-LAUNCH READY)** | 11.76% | 52.94% | 52.94% | **52.94%** | +41.18 pp |
| **n_newly_unblocked_since_Wf_289C** | — | — | — | **0** | — |

**Honest verdict**: **NO new wave crossed PENDING → DONE on disk since Wf-280D.** The 7 G6 waves (C10, C11, C13, C14, C15, C16, C17) unblocked at Wf-286C remain PRE-LAUNCH READY but have NOT been launched (wrapper `scripts/launch_wf124_p3_g6.sh` never invoked). C6 partial progress (Ru seed=0 DONE + seed=1 RUNNING epoch 15 → 19) advanced 4 epochs since Wf-289C; seed=2 still PENDING. Per `kill_then_adapt`, every wave ≤ 1 hr still requires FRESH user OK before launch.

**Wave-impact analysis of Wf-290F4/F5/F6/F7 + Wf-291C**: **0 waves newly-unblocked, 0 waves progressed, 0 waves completed**. Wf-290F4-F7 fixes are **test-infrastructure only** (`test_metal_ligand_exchange_matrix`, `test_reproduce_R17`, `test_score_any_gpu_routing`, `test_synthemol_reactions`) — they unblock test collections but do NOT change Cf-258 wave-launch readiness. Wf-291C is a **doc-only pipeline-regression root-cause investigation** that does NOT change any wave status.

---

## §1 Per-wave status (C1–C17)

### §1.1 TODO-180 G4 (Wf-124 P3 30-epoch 5×3 sweep)

| Wave | Pairs | Wf-280D | Wf-286C | Wf-289C | **Wf-292C** | Delta |
|---|---|---|---|---|---|---|
| **C1** | Ru s1/s2 + Ir s0/s1/s2 (5 pairs) | DONE | DONE | DONE | **DONE** | unchanged |
| **C2** | Os s0/s1/s2 + Rh s0/s1/s2 + Re s0/s1/s2 (9 pairs) | DONE | DONE | DONE | **DONE** | unchanged |

**G4 aggregate (unchanged since Wf-278A)**: 14/14 pairs MEASURED; 5-metal test_auc **0.7209 ± 0.0458** (`results/wf124_p3_remaining_metals_g4/training_summary.json`).

### §1.2 TODO-119 P1.3 + P1.4 (CFM ablation on Blackwell GPU-0)

| Wave | Wf-280D | Wf-286C | Wf-289C | **Wf-292C** | Delta |
|---|---|---|---|---|---|
| **C3** (P1.3 smoke, ~30 min, GPU-0) | PENDING | PENDING | PENDING | **PENDING** | unchanged — `results/cfm_real_decode_lift_p1.3/` absent |
| **C4** (P1.4 a+b, ~60 min, GPU-0) | PENDING | PENDING | PENDING | **PENDING** | unchanged |
| **C5** (P1.4 c+d, ~60 min, GPU-0) | PENDING | PENDING | PENDING | **PENDING** | unchanged |

**Blockers for C3–C5 (UNCHANGED)**: 16-hr Blackwell pre-launch scope (CFM ablation flag + W₂ gradient-path verification required before any wave); GPU-0 Blackwell idle but formal P1.3 deliverable needs PlatinAI 4-cell + DrugOOD per-cohort assays (not yet wired).

**Wf-291C relevance**: Wf-291C identified the 6-occurrence pipeline-regression class (Wf-270/271/272/274/279/285) but **is doc-only and does NOT fix C3–C5's launch readiness**. The CFM ablation flag + W₂ gradient-path verification gate remains unaddressed.

### §1.3 TODO-181 G5 (Wf-124 P3 50-epoch full sweep, RTX 5090 GPU-1)

| Wave | Wf-280D | Wf-286C | Wf-289C | **Wf-292C** | Delta |
|---|---|---|---|---|---|
| **C6** (Ru s0/s1/s2) | PENDING (partial: epoch 47/50, no ckpt) | PARTIAL (Ru seed=0 MEASURED test_auc=0.7716; seed=1 RUNNING epoch 11/50) | PARTIAL (Ru seed=0 MEASURED; seed=1 RUNNING epoch 15/50) | **PARTIAL** (Ru seed=0 MEASURED; seed=1 RUNNING **epoch 19/50**; seed=2 PENDING) | seed=1 advanced epoch 15 → 19 (+4 epochs since Wf-289C) |
| **C7** (Ir s0/s1/s2) | PENDING | PENDING | PENDING | **PENDING** | unchanged — no `Ir/` dir |
| **C8** (Os+Rh s0/s1/s2) | PENDING | PENDING | PENDING | **PENDING** | unchanged — no `Os/` or `Rh/` dir |
| **C9** (Re s0/s1/s2 + comparison JSON) | PENDING | PENDING | PENDING | **PENDING** | unchanged — no `Re/` dir; no comparison JSON |

**Live process state** (Wf-292C snapshot, 2026-09-29 ~02:48 CST):
```
PID 609528 (zsh wrapper)             → alive (since Sep 28 23:41:04)
PID 609532 (bash launch_wf124_p3_g5.sh) → alive
PID 1015907 (python trainer Ru seed=1) → alive, epoch 19/50 (last log entry: 80.2s/epoch)
```

**Live log tail** (`results/wf124_p3_50epoch_full_sweep_g5/Ru/seed1/run.log`):
```
[Ru seed=1] epoch 15/50 loss=0.9474 cls=0.5162 reg=1.3785 w=0.7118 val_auc=0.6816 (50.4s)
[Ru seed=1] epoch 16/50 loss=0.8535 cls=0.4839 reg=1.2231 w=0.6291 val_auc=0.6877 (44.2s)
[Ru seed=1] epoch 17/50 loss=0.8059 cls=0.5018 reg=1.1099 w=0.6028 val_auc=0.7094 (76.6s)
[Ru seed=1] epoch 18/50 loss=0.8074 cls=0.4952 reg=1.1197 w=0.6022 val_auc=0.6918 (76.4s)
[Ru seed=1] epoch 19/50 loss=1.0045 cls=0.4775 reg=1.5314 w=0.7843 val_auc=0.7127 (76.2s)
```

**ETA to C6 completion**: 31 epochs remaining × ~75s/epoch ≈ **~39 more min** for Ru seed=1, then seed=2 (~75 min). C6 PARTIAL → DONE ETA: **~2 hr from now**.

**Blockers for C7–C9 (UNCHANGED)**: depends on G5 wrapper finishing C6 (Ru seed=1 + seed=2) before Ir/Os/Rh/Re launches.

### §1.4 TODO-182 G6 (per-innovation ablation, RTX 5090 GPU-1)

| Wave | Wf-280D | Wf-286C | Wf-289C | **Wf-292C** | Delta |
|---|---|---|---|---|---|
| **C10** (no_multi_task Ru/Ir/Os) | PENDING (`--no-kendall` UNVERIFIED) | PRE-LAUNCH READY | PRE-LAUNCH READY | **PRE-LAUNCH READY** | unchanged |
| **C11** (no_multi_task Rh/Re + no_wass Ru) | PENDING (`--no-dann` UNVERIFIED) | PRE-LAUNCH READY | PRE-LAUNCH READY | **PRE-LAUNCH READY** | unchanged |
| **C12** (no_wass Ir/Os/Rh/Re) | PENDING | PENDING | PENDING | **PENDING** (no flag gap; never launched) | unchanged |
| **C13** (no_kendall Ru/Ir) | PENDING (UNVERIFIED) | PRE-LAUNCH READY | PRE-LAUNCH READY | **PRE-LAUNCH READY** | unchanged |
| **C14** (no_kendall Os/Rh/Re) | PENDING (UNVERIFIED) | PRE-LAUNCH READY | PRE-LAUNCH READY | **PRE-LAUNCH READY** | unchanged |
| **C15** (no_dann Ru/Ir/Os) | PENDING (UNVERIFIED) | PRE-LAUNCH READY | PRE-LAUNCH READY | **PRE-LAUNCH READY** | unchanged |
| **C16** (no_dann Rh) | PENDING (UNVERIFIED) | PRE-LAUNCH READY | PRE-LAUNCH READY | **PRE-LAUNCH READY** | unchanged |
| **C17** (no_dann Re + report.json) | PENDING (UNVERIFIED) | PRE-LAUNCH READY | PRE-LAUNCH READY | **PRE-LAUNCH READY** | unchanged |

**G6 PRE-LAUNCH BLOCKER (Wf-280D §1.4)**: `--no-kendall` / `--no-dann` flags UNVERIFIED → **CLOSED at Wf-283B + Wf-285B**.
- Wf-283B: added `--no-kendall` / `--no-dann` to argparse (lines 1322–1337) + fixed `15-30%` → `15-30 percent` argparse %u crash.
- Wf-285B: wired flags into model + loss + training loop (`MultiTaskWassersteinModel(use_dann=...)`, `MultiTaskLoss(use_kendall_loss=...)`, `train_one_epoch_joint(...)`).
- Wf-285B tests: `molmetal/tests/test_kendall_dann_wiring.py` 3/3 PASS (per Wf-286C §1.4 + Wf-288D §1 cascade).

**Wf-290/291 wave-impact audit**:
- **Wf-290F4** (test_metal_ligand_exchange_matrix.py fix): **NO wave impact** — fixes 12-metal × 7-ligand = 84-pattern METAL_LIGAND_EXCHANGE_MATRIX; 659 tests pass. Pure test-infrastructure fix for the L-R17-B2 TODO-31 §2 matrix; does NOT change G6 CLI readiness.
- **Wf-290F5** (test_reproduce_R17.py fix): **NO wave impact** — adds `reproduce_R17` marker to `pyproject.toml [tool.pytest.ini_options].markers`; 4 tests pass + 1 skip. Fixes strict-markers collection error; no G6 launch impact.
- **Wf-290F6** (test_score_any_gpu_routing.py fix): **NO wave impact** — wires GPU_WRAPPER_TABLE (6 entries) + `gpu=` kwarg into `molmetal/scoring/score_any.py`; 50 GPU-routing tests pass + 25 Wf-43 backward-compat tests pass. Fixes Wf-50 Phase 4 routing layer; **does NOT touch `train_dmpnn_multitask_wasserstein.py`** so G6 CLI flags are unaffected.
- **Wf-290F7** (test_synthemol_reactions.py fix): **NO wave impact** — extends REAL_TO_LAMBDA 13→18 entries (benzimidazole_formation, oxadiazole_formation, tetrazole_formation, thiazole_formation, hantzsch_pyridine_synthesis); 38 tests pass + 2 skip. Fixes SyntheMol reaction-rule registry; no G6 launch impact.
- **Wf-291C** (pipeline-regression root-cause): **NO wave impact** — identifies 6-occurrence pipeline-regression class (Wf-270/271/272/274/279/285) caused by `StructuredOutput` returning schema-as-result; doc-only investigation.

**Verdict**: Wf-280D PRE-LAUNCH BLOCKER **CLOSED** at Wf-286C; **no NEW unblockers since Wf-286C OR Wf-289C**. G6 wrapper `scripts/launch_wf124_p3_g6.sh` has NOT been invoked; all 7 PRE-LAUNCH READY waves still need fresh user OK per `kill_then_adapt`.

---

## §2 Tally

| Metric | Wf-280D | Wf-286C | Wf-289C | **Wf-292C** | Δ since Wf-280D |
|---|---:|---:|---:|---:|---:|
| **n_waves_total** | 17 | 17 | 17 | **17** | 0 |
| **n_waves_done** | **2** | **2** | **2** | **2** | 0 |
| **n_waves_partial** | 0 | **1** (C6: 1/3 seeds MEASURED) | **1** (C6: 1/3 seeds MEASURED; seed=1 RUNNING) | **1** (C6: 1/3 seeds MEASURED; seed=1 RUNNING epoch 19/50) | +1 |
| **n_waves_pre_launch_ready** | 0 | **7** (C10, C11, C13–C17) | **7** (C10, C11, C13–C17) | **7** (C10, C11, C13–C17) | +7 |
| **n_waves_pending_unverified** | 8 (C10–C17) | 1 (C12) | 1 (C12) | **1** (C12) | −7 |
| **n_waves_pending_other** (C3–C5, C7–C9) | 7 | 7 | 7 | **7** | 0 |
| **completion_pct (DONE-only)** | 11.76% | 11.76% | 11.76% | **11.76%** | 0 pp |
| **completion_pct (DONE + PARTIAL + PRE-LAUNCH READY)** | 11.76% | 52.94% | 52.94% | **52.94%** | +41.18 pp |

**Newly unblocked since Wf-289C**: **0** waves. Wf-290F4/F5/F6/F7 + Wf-291C are **test-infrastructure / doc-only** and do NOT unblock any wave-launch blocker.

---

## §3 Newly unblocked wave detail (since Wf-289C = 0)

| Wave | Wf-289C status | Wf-292C status | What (if anything) changed | Source |
|---|---|---|---|---|
| **C10** | PRE-LAUNCH READY | PRE-LAUNCH READY | none | (Wf-290F4-F7 + Wf-291C are test/doc only; no G6 CLI delta) |
| **C11** | PRE-LAUNCH READY | PRE-LAUNCH READY | none | (same) |
| **C12** | PENDING (no flag gap; never launched) | PENDING (unchanged) | none | (same) |
| **C13** | PRE-LAUNCH READY | PRE-LAUNCH READY | none | (same) |
| **C14** | PRE-LAUNCH READY | PRE-LAUNCH READY | none | (same) |
| **C15** | PRE-LAUNCH READY | PRE-LAUNCH READY | none | (same) |
| **C16** | PRE-LAUNCH READY | PRE-LAUNCH READY | none | (same) |
| **C17** | PRE-LAUNCH READY | PRE-LAUNCH READY | none | (same) |
| **C3–C5** | PENDING (CFM ablation flag + W₂ gradient-path verify, GPU-0 Blackwell) | PENDING (unchanged) | none | Wf-291C identifies pipeline-regression class but is doc-only; CFM ablation flag verification gate remains unaddressed |
| **C6** | PARTIAL (Ru seed=1 RUNNING epoch 15/50) | PARTIAL (Ru seed=1 RUNNING epoch 19/50) | seed=1 advanced 4 epochs (15→19) but no new seed/ckpt completed | G5 wrapper PID 609528 autonomous |
| **C7–C9** | PENDING (depends on C6 G5 wrapper finishing) | PENDING (unchanged) | none | G5 wrapper still autonomously running (PID 609528); Ru seed=1 at epoch 19/50 |

**Verdict**: **0 newly-unblocked waves since Wf-289C.** The 7 G6 waves unblocked at Wf-286C remain PRE-LAUNCH READY; C3–C5 CFM ablation flag verification is the only remaining infrastructure gap, and no Wf-290/291 wave has addressed it.

---

## §4 Honest framing (party-twenty-spirit §3.5)

### §4.1 MEASURED (cite as is)

- **Wf-280D baseline**: 14/14 G4 pairs (5-metal test_auc 0.7209 ± 0.0458) — **unchanged since Wf-278A**.
- **Wf-282 + Wf-283A**: G5 Ru seed=0 MEASURED test_auc = **0.7716** (val_auc peak 0.8177 @ epoch 47, test_pearson 0.5479, ckpt 11.3 MB) — **unchanged since Wf-284C propagation**.
- **Wf-285B**: `--no-kendall` / `--no-dann` flags WIRED into loss + model + training loop; 3 wiring tests pass — **unchanged since Wf-286C**.
- **Wf-285C**: `--no-enforce-budget` flag added — **unchanged since Wf-286C**.
- **Wf-287A**: m14 LEAD anchor 0.9958 on n=26,801 (Ru/Ir/Os/Rh/Re per-metal breakdown) — **NEW MEASURED** (orthogonal to Cf-258 wave ledger; no wave impact).
- **Wf-287D**: 4/6 §9 gaps fixed (G1/G2/G5/G6), 2/6 partial (G3/G4), 0/6 unfixed — doc-audit finding, no wave impact.
- **Wf-288B**: paper §4 107 claims clean, 0 §3.5 violations — doc-audit finding, no wave impact.
- **Wf-288D**: Cf-263C 3 original deviations (G5/G6 CSV v2→v4 + G6 epochs 30→50) FIXED; 0 new deviations — **prior Wf-280D blocker already closed; no NEW unblocker**.
- **Wf-290F4**: METAL_LIGAND_EXCHANGE_MATRIX (12 metals × 7 ligands = 84 patterns) SHIPPED; 659/659 tests pass — **NEW MEASURED** (test-infrastructure; no wave impact).
- **Wf-290F5**: `reproduce_R17` marker REGISTERED in `pyproject.toml`; 4/4 tests pass + 1 skip — **NEW MEASURED** (test-infrastructure; no wave impact).
- **Wf-290F6**: GPU_WRAPPER_TABLE (6 entries) + `gpu=` kwarg WIRED into `molmetal/scoring/score_any.py`; 50/50 GPU-routing tests + 25/25 Wf-43 backward-compat tests pass — **NEW MEASURED** (test-infrastructure; no wave impact — does NOT touch `train_dmpnn_multitask_wasserstein.py`).
- **Wf-290F7**: REAL_TO_LAMBDA 13→18 entries SHIPPED; 38/38 tests pass + 2 skip — **NEW MEASURED** (test-infrastructure; no wave impact).
- **Wf-291C**: 6-occurrence pipeline-regression class root-cause IDENTIFIED (StructuredOutput returns schema-as-result); 3 long-term mitigations proposed — **NEW MEASURED** (doc-only; no wave impact).

### §4.2 PROJECTED (cite cautiously; only after GPU run)

- G5 C6 → DONE: Ru seed=1 + seed=2 remaining; ETA ~2 hr from Wf-292C snapshot (PID 609528 still autonomously running).
- G5 C7–C9 (Ir/Os/Rh/Re 50-ep sweep): ETA ~7–8 hr from G5 wrapper start (Sep 28 23:41) = ~06:00–07:00 CST 2026-09-29 (some pairs may complete earlier as wrapper auto-launches).
- G6 60-run ablation (C10–C17): 7 waves PRE-LAUNCH READY but wrapper never invoked; no projected wall without fresh user OK.
- C3–C5 CFM ablation (GPU-0 Blackwell): W₂ gradient-path verify + CFM ablation flag verify STILL PENDING; no projected wall without fresh user OK + pre-launch infrastructure work.

### §4.3 DEFERRED / NOT in scope of Wf-292C

- Wf-286A DrugOOD 5-seed: `gate_1_all_ran=false`, all gates 0 (wrapper aborted; cf. Wf-263A timeout pattern). Out of scope for Cf-258 waves.
- Wf-286B ECFP4 + BinaryClassificationFFN smoke: test_auc 0.8820 (rules out ECFP4 axis as bottleneck for Wf-253-D 0.4544 collapse; not a Cf-258 wave).
- TODO-177 (M1+M2 downstream lift on 6 consumer TODOs): 6-wk ETA per consumer.
- TODO-130 G5 (chemprop v3 DrugOOD re-train): v3 currently 0.4544 BELOW_RANDOM DrugOOD.
- TODO-96 R20-P5 Catherine 30-metal reachability ≥ 0.95.
- TODO-98 L12 cross-domain OOD reachability ≥ 0.85 (Q4 2026).
- CFM M1+M2 / F5_RDKIT_2D / click-aug integration into CFM training (1.1.8 / 1.2.3 / 1.2.4 pending GPU).

### §4.4 DO NOT claim

- "Cf-258 17 waves MEASURED COMPLETE" — 15/17 still PENDING.
- "G5 sweep MEASURED COMPLETE" — only Ru seed=0 of 15 pairs MEASURED; seed=1 RUNNING epoch 19/50; seed=2 PENDING.
- "G6 per-lever attribution MEASURED" — 0/60 runs MEASURED.
- "Real CFM P1.3 MEASURED ≥ 0.5" — wave C3 NOT launched.
- "G6 wrapper launched since Wf-286C" — wrapper NEVER invoked.
- "Best-seed selection" (per `mol-metal-gpu-execution` §7) — all seeds reported, std IS the answer.
- "TODO-182 G6 run completion <7.5 hr" — MEASURED 7.5 hr is PROJECTED; first G5 pair took 47 min for 50 ep, so G6 may differ.
- "Wf-290F4-F7 unblock Cf-258 waves" — 0 wave impact; all 4 fixes are test-infrastructure only and do NOT change G6 CLI readiness or C3–C5 CFM ablation flag verification.
- "Wf-291C unblocks Cf-258 waves" — 0 wave impact; doc-only root-cause investigation.
- "Wf-287/288/290/291 changes unblocked new waves" — 0 newly-unblocked waves since Wf-286C.

---

## §5 Recommended next-step sequencing (Wf-292C priority order)

| Step | Action | Wall | Status | Source |
|---:|---|---|---|---|
| **1** | **Wait for C6 (Ru seed=1 + seed=2)** — wrapper PID 609528 autonomous; ETA ~2 hr from Wf-292C snapshot | ~2 hr | IN-FLIGHT | Wf-292C §1.3 |
| **2** | **Wf-292C-P2 — Launch C7 (Ir s0/s1/s2, ~60 min)** AFTER C6 seed=2 ends | ~1 hr | READY | this report |
| **3** | **Wf-292C-P3 — Launch C8 + C9 in sequence** (~120 min) AFTER C7 ends | ~2 hr | READY | this report |
| **4** | **Wf-292C-P4 — Launch C3 (P1.3 smoke, GPU-0 Blackwell, ~30 min)** — fresh user OK required | ~0.5 hr | PENDING | Wf-258 §1.3 |
| **5** | **Wf-292C-P5 — Launch C4 + C5 (P1.4 ablation, GPU-0, ~120 min)** AFTER C3 | ~2 hr | PENDING | Wf-258 §1.4–§1.5 |
| **6** | **Wf-292C-P6 — Launch C10 (no_multi_task Ru/Ir/Os, GPU-1, ~68 min)** AFTER G5 sweep ends | ~1.1 hr | PRE-LAUNCH READY | Wf-258 §1.7 + Wf-285B |
| **7** | **Wf-292C-P7 — Launch C11–C17 sequentially** (~7.5 hr, GPU-1) | ~7.5 hr | PRE-LAUNCH READY | Wf-258 §1.7 |

**Total wall** (sequential after C6 finishes): ~17 hr on GPU-0 + GPU-1 combined.

**Parallelization opportunities**:
- Steps 4–5 (P1.3 + P1.4 on GPU-0) can run in parallel with steps 2–3 (G5 continuation on GPU-1) — different GPUs, no contention.
- Steps 6–7 (G6 on GPU-1) sequential after G5; no parallelism with steps 2–3.

**Wf-291F pipeline-harness root-cause fix** (per Wf-291C §5.4; ~4 hr, NOT parallelizable per Wf-280E §P2.2): would close the 6-occurrence pipeline-regression class; would NOT directly change Cf-258 wave status but WOULD prevent future Wf-XXX batches from regressing.

**Wf-291G re-emit Wf-285A/D** (~30 min CPU, parallelizable per Wf-280E §P2.1): reconstruct the 2 missing Wf-285 deliverables from sibling data; would close Wf-285 2-of-4 split to 4-of-4.

---

## §6 Files referenced

- `molmetal/reports/wf280d_c258_waves/audit_report.md` — Wf-280D baseline (n_done=2, n_pending=15)
- `molmetal/reports/wf286c_c258_refresh/refresh_report.md` — Wf-286C refresh (n_done=2, 7 newly-unblocked C10/C11/C13–C17, C6 PARTIAL)
- `molmetal/reports/wf289c_c258_refresh/refresh_report.md` — Wf-289C refresh (n_done=2, 0 newly-unblocked since Wf-286C)
- `molmetal/reports/wf290f4_fix/fix_summary.md` — METAL_LIGAND_EXCHANGE_MATRIX (12 metals × 7 ligands = 84 patterns); 659/659 tests pass; NO wave impact
- `molmetal/reports/wf290f5_fix/fix_summary.md` — `reproduce_R17` marker registration; 4/4 tests pass + 1 skip; NO wave impact
- `molmetal/reports/wf290f6_fix/fix_summary.md` — GPU_WRAPPER_TABLE (6 entries) + `gpu=` kwarg wire-up; 50/50 GPU-routing + 25/25 backward-compat tests pass; NO wave impact (does NOT touch `train_dmpnn_multitask_wasserstein.py`)
- `molmetal/reports/wf290f7_fix/fix_summary.md` — REAL_TO_LAMBDA 13→18 entries; 38/38 tests pass + 2 skip; NO wave impact
- `molmetal/reports/wf291c_pipeline_root_cause/root_cause.md` — 6-occurrence pipeline-regression class root-cause (StructuredOutput returns schema-as-result); 3 long-term mitigations proposed; NO wave impact (doc-only)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (modified, +350/−4 lines since Wf-280D) — Wf-285B wire-up + Wf-283B flag + Wf-284A % fix + Wf-285C no-enforce-budget
- `molmetal/scoring/score_any.py` (modified, Wf-290F6) — GPU_WRAPPER_TABLE (6 entries) + `gpu=` kwarg + `_gw_*` adapters; does NOT touch trainer
- `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` (modified, Wf-290F4) — METAL_LIGAND_EXCHANGE_MATRIX (84 patterns) + accessors
- `molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py` (modified, Wf-290F7) — REAL_TO_LAMBDA 13→18 entries
- `pyproject.toml` (modified, Wf-290F5) — `reproduce_R17` marker registration
- `molmetal/tests/test_kendall_dann_wiring.py` (new, 3 tests) — Wf-285B wire-up tests
- `molmetal/tests/test_wave_budget_tracker.py` (+3 tests, total 31/31 PASS) — Wf-285C
- `molmetal/tests/test_metal_ligand_exchange_matrix.py` (659/659 PASS) — Wf-290F4
- `molmetal/tests/test_reproduce_R17.py` (4/4 PASS + 1 skip) — Wf-290F5
- `molmetal/tests/test_score_any_gpu_routing.py` (50/50 PASS) — Wf-290F6
- `molmetal/tests/test_synthemol_reactions.py` (38/38 PASS + 2 skip) — Wf-290F7
- `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log` — Ru seed=0 DONE (Wf-282 + Wf-283A)
- `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/per_seed_metrics.json` — Ru seed=0 metrics
- `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed1/run.log` — Ru seed=1 RUNNING epoch 19/50 (PID 1015907)
- `results/wf124_p3_50epoch_full_sweep_g5/master.log` — wrapper timestamps (00:28:52 START Ru seed=1)
- `configs/wf124_p3_per_innovation_ablation_g6.yaml` — G6 config (PRE-LAUNCH READY but never launched)
- `configs/wf124_p3_50epoch_full_sweep_g5.yaml` — G5 config (autonomous wrapper running)
- `scripts/launch_wf124_p3_g6.sh` — G6 wrapper (NEVER invoked since Wf-280D)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AK (Wf-284C update) + §9.AH (Wf-269A/B/C catalog)

---

*Generated by Wf-292C (2026-09-29). Read-only audit. n_done unchanged at 2/17; 0 newly-unblocked waves since Wf-289C; C6 PARTIAL advanced seed=1 epoch 15 → 19; G5 wrapper PID 609528 still autonomously running; Wf-290F4-F7 + Wf-291C are test-infrastructure / doc-only with NO wave impact.*
