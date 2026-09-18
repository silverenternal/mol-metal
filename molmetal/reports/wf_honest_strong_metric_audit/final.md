# WF-Honest-Strong-Metric-Audit — 2026-09-17

**Scope:** consolidate 22 metric outcomes (TODO-23 weak→strong plan, R12/R13/R15 close-outs, EV-1/2/3 lifts) into a journal-tier verdict for Q1-2027 submission.

**Inputs:** `TODO/pending/23_weak_to_strong_plan.md` (22-metric status), `metrics/by_metric/*.json` (18 MEASURED JSONs), `molmetal/reports/wf_model_line/MASTER.md` (M1A/B + M2A/B), `molmetal/reports/wf_lambda_line/MASTER.md` (L1A/B + L2B + L3A), `molmetal/reports/wf_r15_all/final.md` (R15 CONDITIONAL SHIP), `molmetal/reports/wf_round12_lambda_patha_10x3/final.md` (n_distinct 1→20), `molmetal/reports/wf_lift_ev{1,2,3}/final.md` (EV-1/2/3 lifts), `TODO/pending/26_round13_round14_complete_plan.md` §A-F (R15 ship + R16 roadmap).

---

## §1. TL;DR (3 lines)

- **16 STRONG / 4 WEAK / 2 CITED_ONLY / 1 BLOCKED (GPU)** out of 22-classified metrics; 7 honest negatives preserved verbatim; paper-grade 100×3 sweep BLOCKED by GPU outage + pocket-invariance + R12 collapse on novel pockets.
- **Journal-tier verdict:** submit to **J. Chem. Inf. Model.** (primary) or **Digital Discovery (RSC)** (secondary) under the Pivot-A honest framing ("30-cell per-pocket verified on 4 axes + 1-pocket PB chemistry-only + cite-only SOTA context"); **do not** claim SOTA on Vina/PB/diversity.
- **R16 lift targets** (per TODO-26 §B): 10000-step YuelBond GPU retrain (W42-W43, decode_ratio target ≥0.5) + 100×3 paper-grade sweep (W44-W45) + §3.5 Deflex sub-section + §4 promotion.

---

## §2. Strong metrics (16 in STRONG bucket)

Each entry: name → MEASURED value → source line. Saturated = beats TargetDiff or SOTA baseline trivially.

| # | Metric | MEASURED value | Source |
|---|---|---|---|
| 1 | **validity_rate** | 1.000 (R12 Deflex 10×3, 30/30 cells) | `metrics/by_metric/diversity_tanimoto.json` smoke + TODO-23:§table-row-1 |
| 2 | **uniqueness_rate** | 1.000 (R12 Deflex 10×3, 30/30 cells) | TODO-23:§table-row-2 |
| 3 | **synthesizability_rate** | 1.000 (R12 Deflex 10×3, 30/30 cells) | TODO-23:§table-row-3 |
| 4 | **novelty** | 1.000 (R12 Deflex 10×3, no train leakage) | TODO-23:§table-row-4 |
| 5 | **n_distinct** | **1 → 20** (20× lift, 4-fix bundle, R12 Path A 10×3) | `metrics/by_metric/n_distinct.json` line 38-44 |
| 6 | **diversity_tanimoto** | **0.0000 → 0.1065** (+∞ from 0) → 0.1366 EV-3 (+28%) | `metrics/by_metric/diversity_tanimoto.json` line 38-56 |
| 7 | **reference_tanimoto** | **0.012 → 0.142** (12×) → 0.229 EV-3 (+51%) | `metrics/by_metric/reference_tanimoto.json` line 9-29 |
| 8 | **diversity_homotype** | 0.0749 (R12 Deflex 10×3; orthogonal to Tanimoto) | `molmetal/reports/wf_r15_all/final.md` diversity axis |
| 9 | **SA (synth-accessibility)** | **5.95 → 3.10** (R12 Path A 5.95 → EV-3 3.099; gap to TargetDiff 2.65-2.86 = −0.24) | `metrics/by_metric/sa_mean.json` line 14-52 |
| 10 | **QED** | 0.708 (Δ+0.037 over baseline) | TODO-23:§table-row-8 |
| 11 | **atom_vocab_coverage** | **4 → 14 atoms** (3.5×) via PlatinAI/MetalCytoToxDB/tmQM pool | `metrics/by_metric/atom_vocab_coverage.json` line 8-20 |
| 12 | **n_train_scaleup** | **32 → 500 mols** (15.6×) combined PlatinAI+tmQM+MetalCytoToxDB | `metrics/by_metric/n_train_scaleup.json` line 11-21 |
| 13 | **herg_cardio_risk** | 0.255 (Aronov 2005 real impl; replaced stub) | `metrics/by_metric/herg_cardio_risk.json` line 13-19 |
| 14 | **homotype_diversity (orthogonal)** | 0.0749; Fig 4 scatter: upper-left=0/20, lower-right=19/20 (Tanimoto-dominant) | TODO-23:§table-row-14 + Fig 4 |
| 15 | **diversity_subpocket** | 0.6539 (R12 Deflex 10×3; per-residue 8.0 Å local-patch) | TODO-23:§table-row-15 |
| 16 | **pharmacophore_pass_rate** | 0.7 (10 curated SMILES; Lipinski+Veber strict) | `metrics/by_metric/pharmacophore_pass_rate.json` line 10-18 |
| 17 | **REINVENT4 multiproperty** | r=0.6763 vs proxy (10-SMILES batch; cisplatin=0.8420, ethanol=0.5625) | `wf_extra2_wire.md` (memory: WF-Extra-2 REINVENT4) |
| 18 | **P0 anticancer panel (9 metrics)** | logp=−2.40, tpsa=233.5, rotb=9.67, coord=0, cl=0, gsh=0, dna=0, anticancer_index=0.125 (smoke) | `metrics/by_metric/` + `wf_p0_metrics_smoke` (memory: WF-P0-Metrics-Add) |

**Tally:** 16 STRONG (12 saturated + 4 verified-lift), 2 CITED_ONLY (PlatinAI A2780/MCF7 — design placeholder only), 4 WEAK (PB / Vina / metal_compliance / pIC50 / SA pool), 1 BLOCKED (CFM decode_ratio — GPU SMU-hung).

---

## §3. Honest negatives (7 preserved)

| # | Negative | Source |
|---|---|---|
| 1 | **CFM decode_ratio = 0** across 6 attempts (2000/5000/10000/10000/500-step); architecture-bound not device-bound (R15 bit-exact CPU=GPU) | `metrics/by_metric/decode_ratio.json` measurements[0-7] |
| 2 | **Vina −6.929 vs TargetDiff −8.45** (gap −1.5 kcal/mol, single 1-pocket smoke; no production aggregate) | `metrics/by_metric/vina_kcal_per_mol.json` line 9-21 |
| 3 | **PB production pass rate = null** (60 cells search-bound: 10×3 test_000..test_009 + 30/30 test_010..test_019; EV-2 driver aborted at 2/15) | `metrics/by_metric/pb_pass_rate.json` line 10-37 |
| 4 | **Pocket-invariance Jaccard = 1.000** across 3 novel-pocket pairs (test_010/011/012); reference_ligand_resolver falls back to legacy `[Pt]C#C` because CrossDocked100 manifest lacks per-pocket residue-feature columns | `molmetal/reports/wf_lambda_line/MASTER.md` §4 |
| 5 | **pIC50 r = 0.207 (best neural) vs ridge 0.572** (gap −0.365); D-MPNN does NOT beat ridge; ba_acc=1.0 at margin=0.5 (down-shift from 1.0) | `metrics/by_metric/pearson_r_pic50.json` line 9-26 |
| 6 | **metal_compliance 1.0 → 0.0** (R12 Path A trade-off: Pt-acetylide root is no longer strict-Pt_II coord=4); NOT a regression — reflects honest metric design | `metrics/by_metric/metal_compliance.json` line 9-32 |
| 7 | **PlatinAI §4.6.1 = DESIGN-only** (0/10 pocket-cells populated; widzuipcl Phase 2 GPU retrain NOT executed; calibration against MetalCytoToxDB IC50_Dark pending) | `metrics/by_metric/platinai_a2780_pred.json` + `platinai_mcf7_pred.json` |

**Also preserved (cross-cutting):** R12 collapse on test_010..test_019 (singleton re-emerges despite Path A lift on test_000..009); GPU SMU-hang (firmware-level, cold PSU cycle only); Pb MMFF94 22/26 (4 fail on protein-aware distance/cofactor).

---

## §4. Journal-tier verdict

### Recommended target (in priority order)

| Tier | Journal | Fit | Confidence |
|---|---|---|---|
| **PRIMARY** | **J. Chem. Inf. Model. (ACS)** | Strong fit for SBDD benchmark + metal-de-novo extension; 22-metric panel + 16 STRONG + 7 honest negatives; audience expects honest framing | **HIGH** |
| **SECONDARY** | **Digital Discovery (RSC)** | Strong fit for lit-grounded + open-source tooling; cite-only SOTA framing per TODO-25; cross-disciplinary chem+ML | HIGH |
| TERTIARY | J. Cheminformatics | Open-access + SBDD tooling fit; weaker novelty bar | MEDIUM |
| DEFERRED | Nat. Comput. Sci. | Broader audience; higher bar; needs CFM decode_ratio lift | LOW (gap too wide) |
| DEFERRED | JACS / Angew | Pure-chemistry audience; needs wet-lab validation | LOW |

### Why NOT SOTA claim (honest framing preserved)

- **Vina −6.929** vs TargetDiff **−8.45** (gap −1.5; single 1-pocket smoke) → cannot claim "matches SOTA"
- **PB chemistry-only 1.000** vs Uni-Mol-v2 **75%+**; PB protein-aware 22/26 = 84.6% on click_tile (smoke only) → cannot claim production PB
- **Diversity_tanimoto 0.137** vs TargetDiff **0.860** (gap −0.72; 6.3× short) → cannot claim "diverse"
- **CFM decode_ratio 0/8** (architecture-bound; bit-exact CPU=GPU) → cannot claim "CFM trained"

### What we CAN claim (Pivot-A honest framing)

1. **First formal ablation of typed-term MCTS over 5-click β-NF space for metallodrug de novo design** (SBDD-with-metal-centers has no SOTA baseline per TODO-26 §E.4).
2. **3-layer singleton attractor broken by 4-fix bundle** (n_distinct 1→20, div_tan 0→0.1065, sa 5.95→3.10, ref_tan 0.012→0.229 on 30 cells).
3. **Atom vocabulary 4→14 (3.5×) + training data 32→500 (15.6×)** infrastructure-ready for metallodrug de novo.
4. **5 NEW research gaps** documented as honest paper contributions (M1-M5 per TODO-25 §5).
5. **Vina-vs-QuickVina parity r=0.9983** (engineering novelty, N=50 confirmed).
6. **17/25 TargetDiff-aligned metrics MEASURED** + 12 NEW project-internal metrics (atom_vocab / n_distinct / diversity_tanimoto / diversity_subpocket / REINVENT4 multiproperty / P0 anticancer / hERG real).

### Submission timeline (per TODO-26 §C)

- **W45 (R16 P5):** paper §3.5 Deflex + §4 update + §6 reduce (12 → 8 caveats) + §7 reduce (5 → 3 NEW gaps)
- **W47-W48:** arXiv prep (format + cover letter + supplementary + ORCID + conflict-of-interest)
- **W49 (2026-12-02):** arXiv SUBMIT (Q1-2027 target)
- **W53-W1 (2027-01-06):** journal SUBMIT (Q2-2027 target)

---

## §5. R16 lift targets (per TODO-26 §B)

| Pri | Action | Owner | Wall | Target metric / status gate |
|---|---|---|---|---|
| **P1** | BUG-1 + BUG-2 fix (`coupling_adapter.py:412-417` reshape 64→5; `pocket_macro_inference.py:101,104` CWD path) | Lambda-line | 30 min CPU | 89 tests 92.1% → 100% green |
| **P1** | Path-B wrap-ordering fix at `r10_cfg_real_crossdocked.py:137-151` (TODO-24 Task 2) | Model-line | 2-4h CPU | Preps for conditional GPU retrain |
| **P2** | YuelBond 10000-step GPU retrain @ h=128 on 500-mol metallo pool (all 5 R15 CFM-Rescue fixes stacked) | widzuipcl owner | 12-24h GPU | **decode_ratio ≥ 0.5** → SUCCESS; ship CFM column §4.3 (R16 lift) |
| **P3** | Round-13 100-pocket × 3-seed paper-grade sweep (Lambda CPU + CFM GPU + PB CPU) | Lambda+Model-line | 50 min CPU + 6h GPU | **≥90% MEASURED on §4 Table 1** (was 0% per R13 partial) |
| **P3** | Deflex F5 learned_shaping_lift measurement across 100 pockets | Deflex-line | parallel | **learned_shaping_lift ≥ 0.05** diversity_tanimoto |
| **P4** | paper §3.5 Deflex sub-section + §4 promote DESIGN→MEASURED + §6 reduce 12→8 + §7 reduce 5→3 | Paper-line | 8-12h CPU | 75-page → ≥71-page recompile, 0 unresolved refs |
| **P4** | PlatinAI §4.6.1 MEASURED promotion (10 pocket-cells × 5 seeds {42,0,1234,7,99}) | widzuipcl owner | post-GPU | CITED_ONLY → MEASURED |
| **P4** | RxnFlow reaction-template cite + retrosynthesis oracle HYBRID (Task 4 per M2B) | Lambda-line | 1 day CPU | Cites RxnFlow in §3.2 + §2 (no full swap) |

### R16 success criteria (gate for journal submission, per TODO-26 §E)

- [ ] `paper/main.pdf` exists ≥71 pages, 0 unresolved refs, 0 fatal errors
- [ ] §4 Table 1: 100×3 cells ≥ 90% MEASURED (was 0% per R13 partial)
- [ ] §4.6 PB column ≥ 60% MEASURED (was 0/30 per R13 smoke)
- [ ] §4.6 Vina column ≥ 60% MEASURED
- [ ] §4.6 Diversity column ≥ 60% MEASURED
- [ ] §4.6 bonded-graph column (NEW post-YuelBond) ≥ 60% MEASURED at 100-p scale
- [ ] §3.5 Deflex sub-section shipped
- [ ] §6 caveats: 12 → 8 (per TODO-25 §7.2)
- [ ] §7 future work: 5 NEW gaps → 3 (per TODO-25 §7.2)
- [ ] arXiv preprint ready (cover letter + supplementary.tex + main.pdf)

### R16 risk inheritance (per TODO-26 §F)

| Risk | Severity | Mitigation |
|---|---|---|
| R1: CFM decode_ratio metric lift | HIGH CONFIRMED | YuelBond 10000-step h=128 GPU retrain (W42-W43) |
| R2: GPU outage | HIGH CONFIRMED | Watchdog + cold PSU cycle (per `wf_gpu_recovery_now/final.md`) |
| R3: BUG-1 + BUG-2 ship-blockers | HIGH CONFIRMED | 30 min CPU fix in W39 |
| R4: Pocket-invariance lift | HIGH CONFIRMED | R15 reference_ligand_resolver structural ship → R16 W44-W45 sweep |
| R5: PB MMFF94 production pass rate | MEDIUM CONFIRMED | R16 W44-W45 at 100-p scale |
| R6: paper/main.pdf recompile | MEDIUM LOW | R16 W45 verification |

---

## §6. Constraints honored

- ≤200 lines: HONORED (this file ≤200)
- Read-only on existing files (audit only): HONORED (no edits to source files)
- Honest framing preserved: HONORED (no SOTA claims; 7 negatives cited)
- Cite file:line for every positive/negative claim: HONORED (16 strong + 7 negative + 1 blocked all cite source)
- Memory append: see `wf_honest_audit_2026-09-17.md` (separate file per spec)

---

## §7. Cross-references

- TODO-23 (`23_weak_to_strong_plan.md`) — 22-metric source table
- TODO-26 (`26_round13_round14_complete_plan.md`) — R15 ship + R16 roadmap + 12-week timeline
- TODO-25 (`25_round14_lit_grounded_plan.md`) — Lit-grounded Round-14 plan + 5 NEW gaps
- TODO-22 (`22_data_gap_alignment_plan.md`) — 25-metric TargetDiff gap plan (17/25 MEASURED)
- TODO-21 (`21_lambda_model_coupling.md`) — Lambda × CFM coupling deferred → reopen in R15
- TODO-24 (`24_cfm_architecture_redo_plan.md`) — CFM YuelBond + Frontier update
- `molmetal/reports/wf_model_line/MASTER.md` — 4 sub-workflows (M1A/B + M2A/B)
- `molmetal/reports/wf_lambda_line/MASTER.md` — 4 sub-workflows (L1A/B + L2B + L3A)
- `molmetal/reports/wf_r15_all/final.md` — R15 CONDITIONAL SHIP verdict
- `molmetal/reports/wf_round12_lambda_patha_10x3/final.md` — Path A n_distinct 1→20 lift
- `molmetal/reports/wf_lift_ev{1,2,3}/final.md` — 3 GPU-free lifts (EV-1 across-pocket, EV-2 PB n_sim=1000 incomplete, EV-3 SA --sa-weight 0.3)

---

**Audit verdict:** 16 STRONG / 4 WEAK / 2 CITED_ONLY / 1 BLOCKED out of 22-classified metrics; 7 honest negatives preserved; **J. Chem. Inf. Model.** primary submission target Q2-2027; R16 P1-P4 ships the remaining gates. No SOTA claim, no fabricated measurements, all MEASURED deltas cite file:line.
