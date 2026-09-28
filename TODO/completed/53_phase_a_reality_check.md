# TODO-53 — Phase A Reality Check (synthesis report)

**Status:** ⚠️ **HONEST_NEGATIVE** — Phase A (R17 Tier-1 AUC fix) failed on 4 layers; do not continue Phase A repair, pivot to R18 SE(3) Schrödinger Bridge.
**Date:** 2026-09-25
**Author:** Hugo Lee
**Replaces:** `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` (marked SUPERSEDED, kept for traceability).
**Cross-refs:** `molmetal/reports/wf_phase_a_stage1_ru/final.md` (Stage 1.5 BLOCK); `molmetal/reports/wf_phase_a_stage2_pt/final.md` (Stage 2.5 BLOCK); `molmetal/reports/wf_phase_a_stage2_diag/a2780_diagnosis.md` (A2780 signal-absence); `molmetal/reports/wf_tmqm_pretrain/final.md` (tmQM +0.046 MEASURED).

---

## §1 Phase A 失败根因 (4 layers, all MEASURED)

Phase A launched Stage 1.4 (`train_ru_multitask.py`) and Stage 2.4 (`train_cisplatin_resistance.py`) simultaneously at 22:53. Both runs were starved before producing per-seed JSON. Diagnosis layered top-down:

| Layer | Evidence | Verdict |
|---|---|---|
| **L1 — ultracode contention** | Both runs at 22:53; cisplatin PID 3450108 reported `104% CPU`; `train_ru_multitask` logged only init line then stalled (Stage 1.5 §3) | Blackwell host has no serial scheduler; two ultracodes assumed GPU exclusivity |
| **L2 — wall-clock budget** | Recipe needs 5 seeds × 100 epoch × 2 ultracodes ≈ 30–36 h GPU; budget was 30 min/run (Stage 1.5 §3 + Stage 2.5 §1) | 30+ h work crammed into 60 min concurrent budget — structurally unrunnable |
| **L3 — tmQM pretrain arch mismatch** | tmQM ckpt = 12 tensors, hidden=128, `_GRUUpdate`; V4 encoder = 36 tensors, hidden=300, `GRUMessageUpdate`; state-dict keys incompatible → random-init fallback (Stage 1.5 §8) | Expected +0.046 tmQM lift was structurally unavailable; multi-task lift was PROJECTED, never MEASURED |
| **L4 — synthetic label ceiling** | A2780 AUC 0.350 traced to cross-line residual r < 0.03 (A2780↔A2780cis +0.003, ↔MCF7 +0.000, ↔A549 +0.005) — `build_platinai_4cell.py` carries no molecule-specific signal (`a2780_diagnosis.md` §8) | A2780cis 0.648 vs 0.65 gate is noise floor; real wet-lab data needed to claim ≥0.85 |

**Verdict (§1.5):** Phase A was **structurally unrunnable** at 30 min budget on Blackwell single-host. Four independent BLOCKs — any one would suffice. R17_AUC_RECOVERY_PLAN §3 timeline of "2 weeks" needed 4–6 weeks on Blackwell alone (sequential scheduling).

---

## §2 Per-phase pass/fail (R17 Phase A matrix)

| Stage | What | Result | Evidence |
|---|---|---|---|
| 1.1–1.3 | data prep, splits, baseline | ✅ PASS | split counts in Stage 1.5 §3 line 3 |
| 1.4 | train 5-seed × 100-ep multi-task D-MPNN + tmQM | ❌ BLOCK | only seed 0 init line; no ckpt, no metrics JSON (Stage 1.5 §3) |
| 1.5 | ship-gate eval | ❌ BLOCK | `metrics/by_metric/ru_5metal_multitask.json` missing — fallthrough BLOCK |
| 1.6 | ablation plan (Path A/B/C) | ➖ N/A | superseded by Phase A close-out |
| 2.1 | A2780 0.350 diagnosis | ✅ PASS | r < 0.03, labels well-formed (a2780_diagnosis.md §8–10) |
| 2.2–2.3 | feature ablation prep | ✅ PASS | 10-dim resistance fingerprint + focal γ=2.0 |
| 2.4 | train 5-seed × 100-ep + tmQM + focal + side feat | ❌ BLOCK | seed 0 epoch 40/100; 30-min budget exhausted (Stage 2.5 §5) |
| 2.5 | ship-gate eval | ❌ BLOCK | A2780cis 0.648 < 0.65 gate (Stage 2.5 §1) |
| tmQM pretrain (parallel) | D-MPNN pretrain 20k subset, 15-ep | ✅ SHIP | test_rmse 0.2718, cls_acc 0.965, 12/12 tensors load (wf_tmqm_pretrain §run results) |
| tmQM fine-tune ablation | `--init tmqm` vs `--init random` on Ru subset | ✅ SHIP | +0.046 lift MEASURED (0.7655 vs 0.7195) — single-seed, plain D-MPNN |

---

## §3 Honest-negative 路线总结

**Decision (this doc):**
1. Phase A = **CLOSED as honest-negative**, not "in-flight / pending fix"
2. A2780 dropped from §4 panel; report 3-cell subset {MCF-7, A549, A2780cis} with synthetic-label caveat
3. tmQM pretrain +0.046 cited **only** as plain-D-MPNN baseline ablation (single seed); not extended to V4 multi-task
4. R18 SE(3) Schrödinger Bridge (P0-1) is the next active plan; primary metric shifts from AUC leadership to Vina Dock geometric accuracy (target -8.5 → -10.5+)
5. R18 transport relaxation (P0-2) + Wasserstein GF (P0-4) paper §3 theory contribution

**Why R18 wins over Phase A:**

| Axis | Phase A | R18 |
|---|---|---|
| Resource | 30–36 h GPU Blackwell blocked | Math-only Phase 0+1 (CPU, 1 ultracode serial) |
| Theory novelty | Incremental AUC lift | First SE(3) SB applied to precious-metal SBDD |
| Failure mode | Stuck at AUC < gate | Lie group Euler fallback (R18_PLAN §10); no regression on R17 baseline |
| Paper §3 contribution | None (recipe-only) | +200 lines theory + 4 new math refs |

---

## §4 Paper §4 / §6 / abstract 改动列表

### §4.1 §4 headline rewrite (per `TODO/01_strategy/INSIGHTS_FROM_WORKFLOWS.md` §3)

> **Current:** "Mol-Metal achieves Ru AUC 0.7123 on MetalCytoToxDB (Tier-1 baseline Krasnov 0.81)"; "PlatinAI 4-cell AUC: MCF-7 0.584, A549 0.630, A2780 0.350, A2780cis 0.648".
>
> **New (§4.1):** "Mol-Metal contributions: (i) first SE(3) Schrödinger-Bridge generator (R18, in flight per §3.4); (ii) tmQM-pretrained cytotoxicity encoder with **MEASURED** +0.046 Ru AUC lift over random init on plain D-MPNN baseline; (iii) cisplatin-resistance-augmented PlatinAI 3-cell pipeline, A2780cis AUC 0.648 on synthetic-label regime. **We do NOT claim Tier-1 AUC leadership over Krasnov 2026 (0.81) or Rusanov 2025 (0.78) on this hardware.** R18 migration is the path to Vina Dock -8.5 → -10.5+ gains."

### §4.2 §4.5 honest-negative — add Gap 4/5/6

- **Gap 4**: Phase A Tier-1 AUC fix did not produce per-seed metrics (Stage 1.5 BLOCK + Stage 2.5 BLOCK).
- **Gap 5**: A2780 0.350 reflects synthetic-label signal-absence, not model defect (ruled out per Stage 2.1); 3-cell subset reported.
- **Gap 6**: tmQM pretrain +0.046 lift measured on plain D-MPNN only; V4 multi-task arch mismatch, random-init fallback.

### §4.3 §6 limitations — items (10)–(13)

- **(10)** Phase A Tier-1 AUC fix interrupted by parallel-ultracode contention + tmQM→V4 arch mismatch; no per-seed metrics; paper §4 does not claim Tier-1 leadership.
- **(11)** A2780 0.350 is signal-absence (cross-line residual r < 0.03); dropped from §4 panel; 3-cell subset reported with caveat.
- **(12)** Blackwell single-host resource ceiling: 30-min/run × 5-seed × 100-ep would require 4–6 weeks sequential; §4 measurements are MEASURED on this host.
- **(13)** tmQM pretrain +0.046 ablation scope: plain D-MPNN baseline only; V4 multi-task arch mismatch documented; Stage 1.6 deferred to R18 post-Phase-2.

### §4.4 Abstract rewrite

Drop "Tier-1 AUC leadership" claim. Replace with: "We report (i) MEASURED +0.046 Ru AUC lift from tmQM pretraining on the MetalCytoToxDB baseline encoder (single-task, plain D-MPNN, scaffold split); (ii) A2780cis AUC 0.648 on PlatinAI 4-cell synthetic-label regime; (iii) R18 SE(3) Schrödinger Bridge migration (in progress) targeting Vina Dock -10.5+. We do not claim AUC leadership over Krasnov 2026 or Rusanov 2025 on this hardware; the theoretical contribution is in the SE(3) SB framework, with empirical gains TBD after Phase 2."

---

## §5 R18 重排期 — Phase 0 + 1 start this week

Original plan (Phase A → wait → R18 Phase 2+). Revised plan: Phase 0 + 1 of R18 start **this week**, parallel with Phase A close-out.

| Week | Phase A close-out | R18 work |
|---|---|---|
| W1 (now) | Stage 1.5 + 2.5 final.md (DONE); §6 items (10–13); §4.5 Gap 4–6 | **Phase 0**: write `paper/sections/03_4_se3_lie_group.md` (~150 L) + `03_4_wasserstein_gf.md` (~80 L) + 5 bib entries (mahmood2026sb_lie, jiang2026sb_relax, wang_xujia2026ma, xue_xiaoping2026ot, kim_jung2026multi_ph) |
| W2–3 | §4 headline + abstract rewrite; Table 2 regen | **Phase 1**: implement `transport_relaxation.py` + ε-annealing + 1000-step smoke (m14 ≥ 0.5) |
| W4–6 | R18 transition docs (MERGE_GATE.md, R18_PLAN.md bump) | **Phase 2**: P0-1 SE(3) SB (3 weeks, biggest risk) |
| W7–8 | — | **Phase 3**: multi-PH + Hilbert primal-dual |
| W9 | — | **Phase 4**: paper final + Table 2 regen + skill update + R18 close-out |

---

## §6 不再 claim Tier-1 AUC 领先的决策

| Decision | Rationale |
|---|---|
| **D1**: Drop Tier-1 AUC leadership claim | R17 1-seed baseline 0.7123 < Krasnov 0.81 by 0.098; Stage 1.4 multi-task run interrupted, no measurement; A2780cis 0.648 < 0.65 gate. Honest framing per `MERGE_GATE.md` §6. |
| **D2**: Drop A2780 from §4 panel | Synthetic-label signal-absence (r < 0.03); same property affects MCF-7/A549/A2780cis but A2780 most extreme. 3-cell subset reported with caveat. |
| **D3**: R18 P0-1 SE(3) SB as next primary metric | First SE(3) SB application to precious-metal SBDD (per Mahmood 2026); primary metric = Vina Dock -8.5 → -10.5+. Rollback to midpoint ODE if blocked (R18_PLAN §10). |
| **D4**: Wet-lab validation deferred to R19 | Synthetic IC50 cannot lift above 0.65 ceiling; honest §6 item (11) documents gap. |
| **D5 (NEW)**: `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` → SUPERSEDED | Keep file with banner; archive after R18 Phase 0–1 close. |
| **D6 (NEW)**: R18 Phase 0 + 1 start this week | Independent of Phase A close-out; pure doc + CPU work; no Blackwell contention. |

---

## §7 honest summary — MEASURED vs PROJECTED vs VERIFIED FAIL

### §7.1 MEASURED (real, on this hardware)
1. Stage 1.4 training interrupted after 7 log lines (wf_phase_a_stage1_ru §3)
2. Stage 2.4 training interrupted at seed 0 epoch 40/100 (wf_phase_a_stage2_pt §5)
3. tmQM pretrain **plain D-MPNN** lift **+0.046** (Ru AUC 0.7195 → 0.7655, single seed, wf_tmqm_pretrain §fine-tune)
4. A2780 AUC 0.350 = signal-absence (cross-line residual r < 0.03, a2780_diagnosis §8)
5. A2780cis 4-cell baseline 0.648 ± 0.006 (3 seeds, n=4078)
6. DecompDiff/Pocket2Mol/TargetDiff 100×3 pipeline wired (R17 close-out)
7. 14/14 evaluators verified, 7/7 pitfall guards PASS, ≥217 unit tests PASS
8. tmQM encoder arch mismatch: baseline 12 tensors hidden=128 vs V4 DirectedMPNN 36 tensors hidden=300

### §7.2 PROJECTED (not measured, TBD)
1. Ru AUC ≥ 0.86 from 5-metal multitask + 3-seed × 100-ep — never ran
2. Pt A2780cis AUC ≥ 0.85 from 5-seed tmQM + resistance features — never landed
3. R18 P0-1 SE(3) SB Vina Dock -8.5 → -10.5+
4. R18 P1-1 multi-PH Diversity 0.65 → 0.75
5. R18 P1-4 Hilbert primal-dual LambdaASTSR R² ≥ 0.62 wall-time -50%

### §7.3 VERIFIED FAIL (cannot lift in this configuration)
1. Stage 1.4 random-init encoder cannot reach 0.86 Ru AUC (tmQM +0.046 unavailable; multi-task lift never measured)
2. Stage 2.4 30-min budget cannot fit 5-seed × 100-ep (12 h GPU required)
3. Concurrent ultracode Blackwell contention structurally unfixable without serial scheduling

---

## §8 Cross-references

- `molmetal/reports/wf_phase_a_stage1_ru/final.md` — Stage 1.5 BLOCK verdict (m1 Ru AUC fail)
- `molmetal/reports/wf_phase_a_stage2_pt/final.md` — Stage 2.5 BLOCK verdict (m2 A2780cis fail)
- `molmetal/reports/wf_phase_a_stage2_diag/a2780_diagnosis.md` — A2780 0.350 signal-absence
- `molmetal/reports/wf_tmqm_pretrain/final.md` — tmQM pretrain +0.046 ablation MEASURED
- `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` — Phase A plan (SUPERSEDED by this doc)
- `TODO/03_plans/R18_PLAN.md` — Phase B detailed plan (now active)
- `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` — Phase B strategic plan
- `TODO/01_strategy/INSIGHTS_FROM_WORKFLOWS.md` §3 — honest framing
- `paper/sections/04_results.tex` — §4 (rewrite to honest framing)
- `paper/sections/04_5_honest_negative.tex` — §4.5 (add Gap 4–6)
- `paper/sections/06_limitations.tex` — §6 (add items 10–13)

---

## §9 Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-25 | Phase A close as honest-negative | 4-layer structural failure, unrunnable on Blackwell single-host |
| 2026-09-25 | Drop A2780 from §4 panel | Synthetic-label signal-absence (Stage 2.1 ruled out alternatives) |
| 2026-09-25 | R18 P0-1 SE(3) SB as next primary | First SE(3) SB applied to precious-metal SBDD |
| 2026-09-25 | R18 Phase 0 + 1 start this week | Independent of Phase A close-out |
| 2026-09-25 | tmQM pretrain +0.046 cited as plain-D-MPNN only | V4 multi-task arch mismatch, random-init fallback |
| 2026-09-25 | §4 headline + abstract rewrite to honest framing | No Tier-1 AUC leadership claim |
| 2026-09-25 | §6 add items (10)–(13) | Document Phase A failure modes for reviewers |
| 2026-09-25 | `R17_AUC_RECOVERY_PLAN.md` → SUPERSEDED | This doc replaces it as canonical Phase A plan |

---

End of report (199 lines).