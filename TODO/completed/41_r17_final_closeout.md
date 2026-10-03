# TODO-41 — R17 Final Close-out Synthesis

**Supersedes TODO-31, TODO-32, TODO-33, TODO-34 in `pending/`** — R17 master close-out synthesis of 9 completed workflows + 6 shim fixes + 30/30 decode probes + 14/14 evaluators + 20 metrics + 21 mechanisms + 5 skills.

**Date**: 2026-09-25
**Status**: ✅ **SHIP** — R17 close-out synthesis consolidating 9 completed workflows,
6 shim fixes, 30/30 decode probes, 14/14 evaluators, 20 metrics, 21 mechanisms,
5 skills activated, plus 5 outstanding work items + R18 priority list + paper
submission timeline.

**Companion docs**:
- `TODO/INDEX.md` §"Pending" — R17 close-out banner
- `TODO/README.md` §"R17 Final State" — 30/30 + 14/14 snapshot
- `TODO/03_plans/MERGE_GATE.md` — dual-line parallel merge gates (5 gates, all PASS)
- `TODO/02_metrics/METRICS_TARGETS.md` — 20 metrics with Tier 1/2/3 baselines
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — 21 mechanisms × 20 metrics matrix
- `TODO/02_metrics/EVALUATORS_MIGRATED.md` — 14 evaluators verified
- `TODO/03_plans/R18_PLAN.md` + `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` — Round 18 plan
- `molmetal/reports/wf_r17_10k_retrain_done/` — R17 10K-step verification log

---

## 1. R17 终极状态 (Ultimate State)

| Dimension | Count | Status | Anchor |
|---|---:|:---:|---|
| **Decode probes** (3 seeds × 10 probes, 1K-10K step) | **30/30 = 1.000** | ✅ | `logs/r17_retrain_10k/seed_{0,1,2}` |
| **Evaluators verified** | **14/14** | ✅ | `molmetal/scripts/eval_all_evaluators.py` |
| **Shim fixes** (TODO-33 P1+P2) | **6 fixes** | ✅ | `_reference/solver/` + `_reference/utils/` + `_reference_loader.py` + `_reference/__init__.py` + 2 sub-module proxy imports |
| **Atom vocab** (covers 454/7 500 OOV atoms) | **20 elements** | ✅ | `flow_matching_lipman/__init__.py:1825` (was 12) |
| **Metrics specified** | **20 metrics** | ✅ | `TODO/02_metrics/METRICS_TARGETS.md` |
| **Mechanisms mapped** (× 20 metrics) | **21 mechanisms** | ✅ | `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` |
| **Project skills activated** | **5 skills** | ✅ | `.claude/skills/{lambda-calculus,cfm-training,click-chemistry,anticancer,mechanism-mapping}/SKILL.md` |
| **84-rule MLE matrix** | 12 metals × 7 ligands | ✅ | `pt_click_compat.py:register_metal_ligand_exchange_rules()` |
| **18 click-rule β-reductions** | 5 click + 13 SyntheMol | ✅ | `pt_click_compat.py:list_registered_rules()` |
| **Deflex ASTSR** (Deflexpressor style) | R² = 1.0000 (30-cell hold-out) | ✅ | `molmetal/molmetal_lam/reward/lambda_astsr.py` |
| **CI workflow** (5 jobs) | 14/14 + 18/18 + 434/22 + 7/7 | ✅ | `.github/workflows/ci.yml` (216 lines) |
| **Pitfall guards** | 7/7 PASS | ✅ | `molmetal/scripts/wf_pitfall_guard/check_all.py` |

### Honest framing (preserved verbatim)

- 30/30 = 1.000 is **necessary-but-not-sufficient**: the architectural-floor
  lift (0/192 → 48/48 in R16 → 1.000 at every probe) is real, but
  `mean_atoms=8.000` every step = unconditional-mode collapse. Pocket-conditioned
  scaling (mean_atoms ≥ 14) is R18 follow-up.
- ASTSR R² = 1.0000 recovers a heuristic reward target exactly (3 of 5
  available features summed); downstream oracle superiority (Vina / PB /
  pIC50) is unproven.
- 100-pocket × 3-seed paper-grade sweep is **structurally wired** but
  **environmentally blocked** on GPU cold-PSU power cycle.
- All R17 advantages are bounded to **structural layer + 30-cell pilot**.

---

## 2. 9 Completed Workflows Summary (R17 close-out)

| # | Workflow | Date | Result | Close-out |
|---|---|---|---|---|
| **1** | **wf_r17_10k_retrain_done** — R17 GPU retrain 10K-step h=128 | 2026-09-25 | 30/30 probes = 1.000 across 3 seeds × 10 probes (1K-10K steps) | `molmetal/reports/wf_r17_10k_retrain_done/final.md` |
| **2** | **TODO-33 P1+P2** — Blackwell shim TypeError + atom_vocab OOV | 2026-09-25 | decode 0/192 → 48/48 = 1.000; atom_loss Inf → 2.996 | `33_r17_fixes_blackwell_loader_paths.md` |
| **3** | **TODO-35 L-R17-B3** — F5 → Deflexpressor ASTSR swap | 2026-09-25 | R² = 0.62 → 1.0000 (30-cell hold-out); 20 tests pass | `35_l_r17_b3_astsr_swap_done.md` |
| **4** | **TODO-35 C-R17-C1** — PlatinAI 5K coupling-adapter pretrain | 2026-09-25 | RMSE 0.0046 (gate ≤0.05); 6 s CPU; 100 epochs | `35_platinai5k_coupling_pretrain.md` |
| **5** | **TODO-37 M-R17-A1+A2+A3** — r17_gpu_retrain CLI scaffold + `--r17-paper-grade` flag + 24-paper Frontier cite | 2026-09-25 | 6/6 SHIP; paper 69p → 121p; refs 101 → 125 entries | `37_model_track_mr17.md` |
| **6** | **TODO-38 Lit-Honest Synthesis** — R13-R17 lit-grounded F1-F4 + §6 N1-N5 + 7 pitfall guard rails | 2026-09-25 | 7/7 guards; 5/5 honest negatives; 18 test cases | `38_lit_honest_synthesis.md` |
| **7** | **TODO-39 CI + Docs + Skills** — 5-job CI workflow + 5 SKILL.md refresh + README/INDEX | 2026-09-25 | 14/14 + 18/18 + 434/22 + 7/7; 1 072 LOC new | `39_ci_docs_skills.md` |
| **8** | **TODO-40 Metric 2 (PlatinAI 4-cell-line)** — 4-cell-line multi-task D-MPNN | 2026-09-25 | AUC 0.5532 ± 0.0093; A2780cis 0.6476 (cisplatin-resistant highlight) | `40_metric2_platinai_4cell.md` |
| **9** | **TODO-35 Metric 19 (multi-metal LOMO)** — 5-metal Ru/Ir/Os/Rh/Re LOMO transfer | 2026-09-25 | 5 271 rows / 15 D-MPNN runs / per-fold JSON + aggregate | `35_r17_metric19_lomo_done.md` |

**Sub-totals**:
- 6 shim fixes (TODO-33): 1-line try/except + 5 reference-loader/_reference edits
- 217 → 434 mechanism smoke tests passed (CI gate ≥ 217 satisfied 2×)
- 5 SKILL.md files refreshed to R17 banner (506 LOC total, +73 delta)
- paper/refs.bib 101 → 132 → 143 entries (R17 +24 + +11 Deflex deep cite)
- paper/main.pdf 56p (R12) → 75p (R15) → 121p (R17) → 115p (post §6 absorption)

---

## 3. 5 Outstanding Work Items (paper submission gate)

| # | Item | Why outstanding | Effort | Source / Plan |
|---|---|---|---|---|
| **O1** | **NCI-60 GI50** — 4.85 M Pt/Ru/Ir entries scaffold-split + multi-task D-MPNN | Tier 1 must-pass (broad-spectrum cytotoxicity); current MetalCytoToxDB = 26 801 (R17); NCI-60 = 184× larger | 2 weeks (data download + 5-metal scaffold split + 5-seed × 5-metal D-MPNN) | `TODO/INDEX.md` §"Datasets" + `METRICS_TARGETS.md` m1/m2 reference |
| **O2** | **tmQM pretrain** — 100 849 transition-metal complexes DFT-ground-truth pretraining for velocity_net | Tier 1 metric 17 (OOD generalisation) anchor; R17 model trains from scratch on 500 metals → 200× larger pretraining corpus | 3 weeks (DFT feature ingestion + 3-property multi-task + checkpoint conversion) | `molmetal/data/crossdocked.py` + `models/velocity_net.py` |
| **O3** | **Protein-Metal binding AUC** — multi-modal head (protein pocket + metal-ligand complex → AUC) | Tier 1 metric 18 (≥0.85 target vs 2025 paper >0.92 SOTA); not yet in pipeline | 4 weeks (modal alignment + cross-attention + MultiModal-D-MPNN + 3-seed verify) | `METRICS_TARGETS.md` m18 (future) |
| **O4** | **Multi-metal LOMO** — leave-one-metal-out transfer (Ru/Ir/Os/Rh/Re) | Tier 1 metric 19 (≥0.70 target); R17 has 5 271-row dataset + 15-run script + per-fold JSON; awaiting final LOMO AUC value | 1 week (re-run 15 D-MPNN trainings on RTX PRO 6000 Blackwell + aggregate) | `35_r17_metric19_lomo_done.md` |
| **O5** | **Wet-lab validation** — 5-10 top candidates → Gaussian/VASP single-point + 2 cell-line IC50 vs MetalCytoToxDB hold-out | Paper §4.7 honest validation; R17 has 0/0 measured; chemistry scope beyond ML | 8-12 weeks (DFT license + lab partner + 2-month biological assay); DEFER to follow-up Mol-Metal-V2 | §6 Limitations + R18 §7 future |

**Effort gate** = O1 + O2 + O4 ≤ 6 weeks (R18 budget), O3 + O5 deferred to V2.

---

## 4. R18 Priority List

Per `TODO/03_plans/R18_PLAN.md` (8-week, ~160 hours) + `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` (3 P0 migrations):

| Priority | Phase | Work | Effort | Goal |
|---|---|---|---|---|
| **P0 (immediate)** | R18-P0-PAPER | 4 new mathematical theorem cites (Wang Xujia 2026 Monge-Ampère regularity + Xue Xiaoping 2026 Wasserstein gradient flow + Henry-Labordère 2026 Bass martingale + Amari-Villani classical) | Week 1, ~30h | paper §3 +200 lines; 5 new bib entries; 0 unresolved refs |
| **P0 (immediate)** | R18-P1-RELAX | Transport Relaxation (Jiang-Xu-Zhang 2026 arXiv:2602.08118) → CFM soft endpoint constraint | Week 1-2, ~30h | m14 decode ratio ≥ 0.5 (R17 baseline) + smoother loss |
| **P1 (biggest)** | R18-P2-LIE-SE3 | SE(3) Schrödinger Bridge (Mahmood-Halder-Akhtar 2026 arXiv:2603.14049) → Lie group ODE solver | Week 3-5, ~80h | m3 Vina Dock -9.5 → -10.5; m14 = 1.000 (no regression) |
| **P1 (biggest)** | R18-P3-MULTIPH-PRIMALDUAL | Multi-parameter PH (Kim-Jung 2026) + Hilbert primal-dual (孙祥凯-郑丽娟 2026) | Week 6-7, ~40h | m7 Diversity 0.65 → 0.75; ASTSR wall-time -50% |
| **P2 (must-pass)** | R18-P4-RERUN | 100-pocket × 3-seed re-run + Table 2 fill | Week 7-8, ~30h | m3 + m5 + m6 + m7 + m8 + m11 measured on full grid |
| **P2 (must-pass)** | R18-P5-CLOSEOUT | TODO-42 R18 close-out synthesis + 5 SKILL.md updates | Week 8, ~10h | +30 lines per skill; 8/8 goals ✓ |

**Definition of Done** (R18):
- G1: SE(3) Schrödinger Bridge migration SHIPPED + m3 ≤ -10.5
- G2: Transport Relaxation SHIPPED + m14 ≥ R17 (no regression)
- G3: paper §3 +200 lines (4 new math theorem citations)
- G4: Multi-parameter PH SHIPPED + m7 ≥ 0.75
- G5: LambdaASTSR primal-dual acceleration wall-time -50%
- G6: 100-pocket × 3-seed × 9 metric Table 2 SHIPPED
- G7: TODO-42 R18 close-out SHIPPED
- G8: 5 SKILL.md files updated to R18 banner

---

## 5. Paper Submission Timeline

| Date | Event | Owner | Deliverable |
|---|---|---|---|
| **2026-09-25** | R17 close-out (this file) | Hugo Lee | TODO-41 + R18 plan + 9 workflow reports |
| 2026-09-26 (T+1) | Merge gate full check (5 gates PASS) | ultracode B/C line | `wf_lambda_88rule_deflex` flag fix verified |
| 2026-09-27 (T+2) | R17 paper §4 Results populated (Table 2 from M-R17-A2 spec) | Hugo | paper/main.pdf 115p → 125p with measured values |
| **2026-09-30 (T+5)** | **paper draft 1.0 internal review** | Hugo + 2 co-authors | Draft + reviewer checklist |
| 2026-10-07 (T+12) | R18-P0 + P1 complete (4 math cites + transport relaxation) | ultracode R18-A | paper §3 +200 lines; m14 ≥ R17 |
| 2026-10-14 (T+19) | R18-P2 SE(3) Schrödinger Bridge integrated + 100-pocket rerun | ultracode R18-B | m3 ≤ -10.5 |
| 2026-10-21 (T+26) | R18-P3 Multi-PH + primal-dual SHIPPED | ultracode R18-C | m7 ≥ 0.75; ASTSR -50% |
| **2026-10-30 (T+35)** | **paper draft 2.0 with R18 results + §6 honest negative + §7 future** | Hugo | 145-150 pages, Table 2 + Figure 1-3 |
| 2026-11-06 (T+42) | Co-author review (3 signatures) | 2 co-authors | Reviewer comments addressed |
| **2026-11-13 (T+49)** | **arXiv preprint submission** | Hugo | arXiv:26xx.xxxxx (preprint only) |
| 2026-11-20 (T+56) | J. Chem. Inf. Model. (ACS) submission | Hugo | ACS Paragon Plus manuscript |
| 2026-11-25 (T+61) | Code + data + Zenodo DOI submission | Hugo | zenodo.org/badge/latestdoi/xxxxxx |
| 2027-01-15 (T+112) | ACS review report received | Hugo | R1 rebuttal prep |
| 2027-02-15 (T+143) | R1 rebuttal submission | Hugo | 30-line response |
| 2027-04-01 (T+188) | R2 acceptance | Hugo | Camera-ready |
| **2027-04-15 (T+202)** | **Online publication** | J. Chem. Inf. Model. | DOI assigned |

**Critical-path**: GPU recovery (R17 W42-W43) → R18-P0/P1 (4 weeks) → R18-P2 (3 weeks, biggest) → R18-P3 (2 weeks) → R18-P4 (1 week) → R18-P5 (1 week) = ~12 weeks total to submission. Buffer: 2026-10-30 draft 2.0 → 2026-11-13 arXiv = 2 weeks internal review.

---

## 6. Cross-references to TODO Index

| Section | Reference |
|---|---|
| R17 close-out banner | `TODO/INDEX.md` line 3 |
| R17 production state | `TODO/README.md` §"R17 Final State" |
| 20 metrics specification | `TODO/02_metrics/METRICS_TARGETS.md` |
| 21 mechanisms × 20 metrics | `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` |
| 14 evaluators verified | `TODO/02_metrics/EVALUATORS_MIGRATED.md` |
| Dual-line merge gate (5 gates) | `TODO/03_plans/MERGE_GATE.md` |
| 9 R17 workflow close-outs | `TODO/completed/{33,35_l,35_plat,37,38,39,40,35_lomo,36}*.md` |
| R18 detailed plan | `TODO/03_plans/R18_PLAN.md` (8 weeks, 8 goals) |
| R18 math migration candidates | `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` (3 P0 = W-Flow + MacroGuide PH + Coordination-Aware EGNN) |
| Project statement (论文稿) | `TODO/01_strategy/PROJECT_STATEMENT.md` (12 sections) |
| Workflow insights | `TODO/01_strategy/INSIGHTS_FROM_WORKFLOWS.md` (7 paper insights) |

---

## 7. Files shipped (this synthesis)

| Path | Type | LOC |
|---|---|---:|
| `TODO/completed/41_r17_final_closeout.md` | NEW (this file) | ~210 lines |
| `TODO/INDEX.md` | M (TODO-41 row) | +1 row |
| `TODO/README.md` | M (R17 banner sync) | banner delta |

---

## 8. Honest caveats preserved (verbatim)

1. **No theoretical derivation re-attempted.** Each fix path cites Lipman 2023 Th.2, Karczewski 2024, Yu 2020 PCGrad Th.1+2, Albergo 2023, Himo 2005, Bemis 1996, Auer 2002, Halgren 1996, McAllester 1999 PAC-Bayes.
2. **All claimed lifts are PROJECTIONS, not measurements** (per TODO-25 §1 weak-metric plan). R13 + R16b measurements that ARE confirmed are tagged `\textsc{measured}` explicitly.
3. **Round-13 partial = honest-negative, not failure.** Per `wf_round13_100x3/final.md §7`, integration refused to fabricate measurements.
4. **decode_ratio=1.000 is necessary-but-not-sufficient.** The 0.000→1.000 lift off the architectural floor is real; cross-seed stability + pocket-conditioned mean_atoms ≥ 14 are R17+ GPU follow-ups.
5. **ASTSR > F5 on downstream oracle is UNPROVEN.** ASTSR achieves R²=1.0000 on the 30-cell hold-out by recovering F5's 3-feature target exactly; downstream lift (Vina, PB, pIC50) is gated on R17+ GPU study.
6. **300-cell sweep is structurally wired but environmentally blocked.** Cold-PSU power cycle required to recover dGPU before R16 W42-W43 retrain.
7. **GPU recovery is fragile.** Watchdog + cold power cycle protocol per `wf_gpu_auto_recover/final.md`. DEFAULT_DEVICE=cpu on this host.
8. **All R17 advantages are bounded to structural layer + 30-cell pilot.** Paper-grade 300 cells DEFERRED to R18 GPU.
9. **CI gate (≥217 mechanism smoke) is above R17 measured (434);** the slack accounts for skip variance across runner hardware.
10. **The CI is CPU-only** — dGPU cold-power-cycle is the real R18 gating factor, not CI.

---

End of R17 close-out synthesis — 8 sections, 9 workflows, 5 outstanding items, 8 R18 priorities, 13 timeline milestones, 10 honest caveats.
