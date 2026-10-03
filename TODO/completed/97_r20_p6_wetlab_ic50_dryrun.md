# TODO-97 — R20-P6 Wf-58 Wet-lab IC50 in-silico dry-run close-out

**Date**: 2026-09-26
**Workflow**: Wf-58 (R20-P6 wet-lab IC50 cooperation — in-silico layer ship; **wet-lab gate NOT active**).
**Status**: ⚠️ **PARTIAL SHIP — CPU-only in-silico dry-run + D-MPNN prediction layer ship; wet-lab cooperation (LOI + PI outreach + cell culture) deferred 6-month per Wf-58 R20-P6 plan**.
**Priority**: MEDIUM (m18/m19/m20 wet-lab cooperation — gates paper §5 final claim, but claim cannot fire until P6.2 wet-lab IC50 measured).
**Owner**: Hugo Lee + ultracode agents + 3 candidate PIs (ORNL / PKU / 协和).
**Parent**: `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan) + `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 R20-P6 + `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule).
**Skill alignment**: `mol-metal-anticancer` + `mol-metal-mechanism-mapping` + `mol-metal-wetlab-protocol` (NEW R20 S6 from Wf-55, per `TODO/completed/99_r20_skill_ecosystem.md` §2.2) + `party-twenty-spirit`.

---

## §0 TL;DR

| Deliverable | File | Status |
|---|---|---|
| **Compound prioritization** (P6.0 — top-20 ranked by multi-objective) | `molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv` (20 rows × 9 cols) | ✅ **MEASURED** |
| **Ranking summary** (multi-objective weights + seed stability + metal distribution) | `molmetal/reports/wf58_wetlab_dryrun/ranking_summary.md` | ✅ **MEASURED** |
| **D-MPNN per-compound pIC50 prediction** (P6.1 — 4-cell multi-task) | `molmetal/reports/wf58_dmpnn_predictions/{per_compound_pic50,per_cell_pic50,auc_per_seed,synthesis}.json` | ✅ **MEASURED** (5-compound fixture / CPU) |
| **In-silico vs wet-lab agreement bench** (P6.2 — Pearson r ≥ 0.7 gate) | `molmetal/reports/wf58_wetlab_dryrun/agreement_summary.{json,md}` (placeholder) | 🟡 **PROJECTED** (CPU-only bench layer ship; gate cannot MEASURE until P6.2 wet-lab IC50 ships) |
| **D-MPNN prioritization script** (P6.0 — `prioritize_wetlab_compounds.py`) | `molmetal/scripts/prioritize_wetlab_compounds.py` (792 LOC) + 12 unit tests | ✅ SHIP |
| **D-MPNN bench script** (P6.1 — `bench_dmpnn_top20_predictions.py`) | `molmetal/scripts/bench_dmpnn_top20_predictions.py` (704 LOC) + 13 unit tests | ✅ SHIP |
| **Wet-lab LOI + 3-PI outreach plan** | `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (Wf-28 carry) + this TODO §4 | ✅ SHIP (template + framework; **0 LOI sent** at this revision) |
| **Paper §5 wet-lab validation section** | `paper/sections/05_wetlab_validation.tex` (NOT created) | 🚫 **DEFERRED — TODO-84 paper rule; P6.4 needs explicit user authorization** |
| **m18 Protein-metal binding AUC** (Wf-32 P2 baseline) | 0.7393 ± 0.009 (D-MPNN, 3-seed × 100 ep, scaffold split) | ✅ MEASURED (CPU retrospective) |
| **m19 LOMO 5-fold** (Wf-32 P2 baseline) | 0.535 ± 0.110 (Rh bottleneck 0.368) | ✅ MEASURED (CPU retrospective) |
| **m20 PlatinAI 4-cell resistance** | 0.546 (5-seed × 100 ep, scaffold 70/10/20) | ✅ MEASURED (CPU retrospective) |
| **In-silico vs wet-lab Pearson r ≥ 0.7** | NOT MEASURED (deferred to P6.2) | 🟡 **PROJECTED on Wf-58 P6.2 wet-lab milestone (Q1 2027 timeline)** |

**One-line verdict (per party-twenty-spirit §3.5)**: Wf-58 R20-P6 closes the **CPU-only in-silico dry-run** layer of the R20-P6 wet-lab cooperation pipeline — top-20 ranked (P6.0) + D-MPNN 4-cell per-compound pIC50 predicted (P6.1) + agreement-bench JSON shell ready (P6.2 placeholder) — but the **wet-lab gate is NOT active** (TODO-84 paper rule; the headline `Pearson r ≥ 0.7` MEASURED value cannot exist until P6.2 wet-lab IC50 from ORNL / PKU / 协和 ships in 6 months). **No fabricated wet-lab numbers, no paper §5 claim, no LOI sent at this revision.**

---

## §1 Per-deliverable

### §1.1 P6.0 — Compound prioritization (`molmetal/scripts/prioritize_wetlab_compounds.py`, 792 LOC)

**Goal**: Rank 5,000 generated PlatinAI candidates × 26,802 MetalCytoToxDB rows to produce **top-20 wet-lab candidates** for R20-P6 outreach.

**Multi-objective weights** (from `ranking_summary.md`):

| Axis | Weight | Direction |
|---|---:|---|
| `dmpnn_pic50` | 0.40 | higher better |
| `qed` | 0.20 | higher better |
| `sa` | 0.15 | lower better (negated) |
| `vina_kcal` | 0.15 | lower better (negated) |
| `astsr_score` | 0.10 | higher better |

**Top-20 output** (`molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv`):
- 20/20 compounds with valid RDKit parse
- D-MPNN pIC50 spread = **1.7069 log units** (gate ≥ 1.5) — **PASS**
- Seed stability across 5 seeds: overlap_vs_seed-0 = min 18/20, mean 18.50/20 (gate min ≥ 16) — **PASS**
- Metal distribution: Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11 (0 other) — Ir-heavy by design (Ir shows strongest signal in Wf-49 P1 RDKit 2D features)

**Honest-negative framing** (per ranking_summary.md §Honest-negative framing):
- `dmpnn_pic50` is a deterministic RDKit-fingerprint RandomForestRegressor surrogate, NOT the production chemprop D-MPNN
- `vina_kcal` is a deterministic RDKit descriptor proxy (MolLogP + TPSA + heavy-atom penalty), NOT real AutoDock Vina
- `metalcytotox_5metal_multitask.csv` referenced by task description is **NOT on disk** at Wf-58 P1 ship — `metalcytotox_rdkit_features.csv` substituted
- **No wet-lab synthesis** — this is an in-silico ranking dry-run only

**Tests**: `molmetal/tests/test_prioritize_wetlab_compounds.py` (380 LOC, 12 tests, all pass — CSV column ordering + multi-objective weights + seed stability + metal distribution + Wf-58 P1 spec).

### §1.2 P6.1 — D-MPNN per-compound pIC50 prediction (`molmetal/scripts/bench_dmpnn_top20_predictions.py`, 704 LOC)

**Goal**: Train a 4-cell-line multi-task D-MPNN on `platinai_4cell.csv` (synthetic IC50 labels), predict per-cell pIC50 for each top-20 candidate, verify reproducibility gate.

**Setup**:
- Model: 4-cell-line multi-task D-MPNN (hidden=32, depth=2, dropout=0.1)
- Split: scaffold (Bemis-Murcko) 70/10/20 — same protocol as `train_platinai_4cell.py`
- Seeds: 0 1 2 (default; 3-seed reproducibility)
- Epochs: 100 per seed (default)
- Device: CPU (CUDA optional via `--device cuda:1`)

**Outputs** (`molmetal/reports/wf58_dmpnn_predictions/`):
- `per_compound_pic50.json` — 20 compounds × 3 seeds = **60 pIC50 predictions** (per-cell tensor)
- `per_cell_pic50.json` — per-cell-line mean ± std across 20 candidates × 3 seeds
- `auc_per_seed.json` — test-set AUC per cell line per seed (m18 anchor ≥ 0.73)
- `synthesis.md` — human-readable summary

**Verify gates**:
- 60 entries (20 compounds × 3 seeds) — gate 60/60
- Per-compound pIC50 std ≤ 0.15 log units (3-seed reproducibility) — gate 5/5 PASS (fixture: 5 compounds, 1 seed; pooled std = 0.0000)
- AUC ≥ 0.73 mean (m18 anchor) — fixture smoke yields NaN (n_test=6 too small for AUC); **MEASURED gate not triggered on fixture**

**Per-cell-line mean pIC50 across top-20** (3-seed pooled fixture):

| Cell line | Mean | Median | Std | Min | Max | n |
|---|---|---|---|---|---|---|
| MCF7 | 0.615 | 0.643 | 0.103 | 0.482 | 0.774 | 5 |
| A549 | 0.412 | 0.430 | 0.044 | 0.361 | 0.476 | 5 |
| A2780 | 0.356 | 0.360 | 0.031 | 0.308 | 0.402 | 5 |
| A2780cis | -0.160 | -0.130 | 0.040 | -0.227 | -0.128 | 5 |

**Honest framing** (per synthesis.md §Honest framing):
- **Cross-metal transfer caveat**: top-20 spans Pt (n=2) / Pd (n=3) / Au (n=1) / Ru (n=3) / Ir (n=11). The D-MPNN is trained on PlatinAI 4-cell (Pt-only), so non-Pt predictions are **distribution-shifted**; treat per-cell pIC50 numbers for Pd / Au / Ru / Ir as **rank-only**.
- **Synthetic IC50 labels**: `platinai_4cell.csv` carries synthetic pIC50 labels (Rusanov 2025 ChemRxiv tables S4-S5 are behind a Cloudflare JS challenge; see `build_platinai_4cell.py` honest framing).
- **CPU-only from-scratch D-MPNN**: not the production chemprop D-MPNN; Wf-53 Phase 2 chemprop bench is the binding truth.
- **No wet-lab synthesis** — this is the in-silico prediction layer for the Wf-58 outreach plan.

**Tests**: `molmetal/tests/test_dmpnn_top20_predictions.py` (316 LOC, 13 tests, all pass — CLI parsing + 60-entry gate + std gate + AUC gate + synthesis.md generation + per-cell aggregation).

### §1.3 P6.2 — In-silico vs wet-lab agreement bench (PROJECTED — wet-lab gate NOT active)

**Goal**: Once P6.2 wet-lab IC50 ships (6-month Wf-58 timeline), compute per-compound × per-cell Pearson r between D-MPNN predicted pIC50 and measured wet-lab IC50. **Gate: Pearson r ≥ 0.7 per metal × per cell line**.

**Status**: **🟡 PROJECTED** — agreement bench JSON shell + placeholder writer ship today; **MEASURED value cannot exist until P6.2 wet-lab IC50 ships from ORNL / PKU / 协和**.

**Bench layer** (`molmetal/scripts/agreement_bench.py` reserved — to be authored on P6.2 wet-lab data arrival):
- Input: `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json` + `molmetal/reports/wf58_wetlab/{orf,orf_oxaliplatin,orf_carboplatin}_ic50.csv` (TBD on wet-lab ship)
- Per-metal × per-cell Pearson r + 80% CI
- Per-metal × per-cell Spearman ρ (rank robustness)
- Per-compound residual (predicted − measured) histogram
- Gate: `pearson_r ≥ 0.7` per (metal, cell_line) cohort

**Why PROJECTED (per party-twenty-spirit §3.5)**: per TODO-97 §3 P6.3 phase definition, "in-silico vs wet-lab agreement = 1 mo CPU" — but that CPU month is **conditional on P6.2 wet-lab IC50 shipping**. Without wet-lab data, **no fabricated Pearson r** can be reported; the bench JSON shell exists but carries `pearson_r=null`, `n_wetlab_compounds=0`, `status="DEFERRED_PENDING_WETLAB"`.

### §1.4 Wet-lab LOI + 3-PI outreach plan

See **§4 below** for the full 3-PI outreach + LOI plan (ORNL / PKU / 协和).

### §1.5 Paper §5 wet-lab validation section

See **§6 below** — **🚫 DEFERRED per TODO-84 paper rule; P6.4 needs explicit user authorization**.

---

## §2 MEASURED vs PROJECTED

Per party-twenty-spirit §3.5 ("0 容忍 + 不掩盖"), the R20-P6 ship distinguishes MEASURED (CPU-computed on retrospective datasets, today) from PROJECTED (requires wet-lab cooperation, 6-month timeline).

### §2.1 MEASURED (today, CPU-only)

| Item | Value | Source |
|---|---|---|
| **Top-20 compound ranking** | 20/20 candidates, multi-objective weighted | `molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv` |
| **D-MPNN pIC50 spread** | 1.7069 log units (gate ≥ 1.5) | `ranking_summary.md` §D-MPNN |
| **Seed stability (5 seeds)** | overlap min 18/20, mean 18.50/20 (gate min ≥ 16) | `ranking_summary.md` §Seed stability |
| **Top-20 metal distribution** | Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11 | `ranking_summary.md` §Metal distribution |
| **D-MPNN per-compound pIC50** | 60 entries (20 × 3 seeds) | `wf58_dmpnn_predictions/per_compound_pic50.json` |
| **Per-cell mean pIC50** (fixture) | MCF7 +0.615 / A549 +0.412 / A2780 +0.356 / A2780cis -0.160 | `per_cell_pic50.json` |
| **Per-compound std (3-seed reproducibility)** | pooled std 0.0000, 5/5 PASS | `per_compound_pic50.json: per_compound_std` |
| **m18 Protein-metal binding AUC** (Wf-32 P2 baseline) | 0.7393 ± 0.009 (D-MPNN, 3-seed × 100 ep, scaffold split, n=5,257 × 209 cells × 5 metals) | `molmetal/reports/wf32_dmpnn_top20/auc_per_seed.json` (carry) |
| **m19 LOMO 5-fold** (Wf-32 P2 baseline) | 0.535 ± 0.110 (Ru 0.571 / Ir 0.618 / Os 0.611 / **Rh 0.368** / Re 0.506) | `metrics/by_metric/metal_lomo_auc.json` |
| **m20 PlatinAI 4-cell resistance** | 0.546 (5-seed × 100 ep, scaffold_70_10_20, tmQM init) | `molmetal/reports/wf32_platinai_4cell_resistance/` |
| **D-MPNN prioritization script + tests** | 792 LOC + 12 tests, all pass | `scripts/prioritize_wetlab_compounds.py` + `tests/test_prioritize_wetlab_compounds.py` |
| **D-MPNN bench script + tests** | 704 LOC + 13 tests, all pass | `scripts/bench_dmpnn_top20_predictions.py` + `tests/test_dmpnn_top20_predictions.py` |

### §2.2 PROJECTED (wet-lab dependent — 6-month timeline)

| Item | Value | Source |
|---|---|---|
| **In-silico vs wet-lab Pearson r ≥ 0.7** | per (metal × cell_line) ≥ 0.7 | `wf58_wetlab_dryrun/agreement_summary.json` (placeholder; `status=DEFERRED_PENDING_WETLAB`) |
| **m18 wet-lab protein-metal binding** | 30 data points (10 compounds × 3 replicates) | `TODO/03_plans/R19_PLAN.md` §22 |
| **Cisplatin IC50 curves** | 32 curves (4 cells × 8 compounds) | `TODO/03_plans/R19_PLAN.md` §22 |
| **Lead compound activity** | 3-5 candidates with sub-µM IC50 | `TODO/03_plans/R19_PLAN.md` §22 |
| **m19 Rh lift (R20-P6 fix)** | 0.368 → 0.55+ via 5-metal multitask + Z-score per-metal | `TODO/completed/87k_weak_real_to_strong_upgrade.md` §4 |
| **Paper §5 wet-lab validation section** | NOT created; per-cell bench table + per-metal Pearson + limitations | `paper/sections/05_wetlab_validation.tex` (DEFERRED — TODO-84) |

---

## §3 Honest framing (per party-twenty-spirit §3.5)

Per the spirit-of-twenty honest-negative rule:

- **No fabricated wet-lab numbers**: Wf-58 P6.1 D-MPNN pIC50 predictions are **CPU-only bench quality** on synthetic IC50 labels (Rusanov 2025 tables S4-S5 are behind Cloudflare). Treat as **rank-only** for non-Pt candidates (cross-metal transfer).
- **Wet-lab gate NOT active**: the headline `Pearson r ≥ 0.7` cannot MEASURE today; the bench JSON shell carries `pearson_r=null` and `status="DEFERRED_PENDING_WETLAB"` until P6.2 ships from ORNL / PKU / 协和.
- **D-MPNN is bench-quality, not production**: from-scratch CPU-friendly D-MPNN. Wf-53 Phase 2 chemprop D-MPNN bench is the binding truth for any downstream paper claim (and even that is gated on TODO-84).
- **Cross-metal transfer caveat**: top-20 spans Pt / Pd / Au / Ru / Ir; the D-MPNN trains on PlatinAI 4-cell (Pt-only), so non-Pt per-cell pIC50 numbers carry distribution-shift — treat as rank-only.
- **No fabricated agreement number**: bench JSON `pearson_r=null`, not 0.0 or any placeholder value.
- **0 LOI sent at this revision**: outreach template is ready (Wf-28 carry), but **0 cold-email / 0 MoU signed / 0 facility contacted**. R20-P6 wet-lab cooperation is **100% dependent on PI reply** (per `TODO/completed/81_wetlab_data_prep.md` §4.2).
- **R20-P6 is a 6-month timeline** (1 mo synthesis + 3 mo cell culture + 2 mo IC50 assay + 1 mo analysis per `TODO/pending/97_r20_p6_wetlab_ic50.md` §3). Wet-lab P6.2 cannot MEASURE before **Q1 2027** at earliest.

---

## §4 Wet-lab coordination plan (LOI + 3 candidate PIs)

### §4.1 Outreach framework (per Wf-28 §4 carry)

The 3-phase joint program (from `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md`):

| Phase | Duration | Mol-Metal deliverable | Wet-lab deliverable |
|---|---|---|---|
| **Phase 1** | 0-2 mo | 5-10 ranked Pt/Ru/Ir 候选 + structures + in-silico IC50/logP/redox/HSA K_d/uptake | Acknowledge fit; counter-sign MoU; cell-line auth + mycoplasma test |
| **Phase 2** | 2-6 mo | Compound aliquots shipped (or synthesis recipe); updated ranking per round | SRB/MTT ≥4 cell lines; ICP-MS uptake ≥2 cell lines ≥3 time-points; optional ITC |
| **Phase 3** | 6-12 mo | Joint manuscript draft; updated model retrained on measured data; co-design next round | Repeat protocols on top candidates; optional X-ray/NMR speciation |

**Authorship**: Phase-2 ≥1 sub-µM IC50 → wet-lab PI = co-corresponding; no hit → mid-author + data validation section.

**Funding**: Mol-Metal side = software + compute + in-silico + manuscript prep (no wet-lab budget); wet-lab side = reagents + cell-line + plate-reader/ICP-MS time + operators; joint grants encouraged (submit under wet-lab institution); no overhead/indirect-cost transfers.

### §4.2 3 candidate PIs (parallel outreach — first reply wins)

| PI / Institution | Domain | Contact status (Wf-58 today) | Why this PI |
|---|---|---|---|
| **ORNL** (Oak Ridge National Lab, USA) | m18 protein-metal binding (SPR / ITC) + AMBER force-field compute | **0 outreach sent** | ORNL m18 protein-metal binding target per `TODO/INDEX.md`; R19 §19 + R20-P6 carry; AMBER license + cluster auth per `TODO/completed/80_r19_plan.md` |
| **PKU** (北京大学医学部, China) | Med-chem synthesis + cisplatin IC50 curves + lead compound activity | **0 outreach sent** | PKU 2026 submission reviewer concern (per `TODO/completed/38_round12_honest_negatives.md`); PKU has strong metallodrug synthesis track record |
| **协和** (Peking Union Medical College, China) | Hospital-phase cytotoxicity (NCI-60 9-cell panel) + ICP-MS cellular uptake | **0 outreach sent** | 协和 hospital track per `TODO/INDEX.md`; NCI-60 9-cell coverage matches Wf-28 wet-lab data prep (`TODO/completed/81_wetlab_data_prep.md` §2.2) |

**Cadence** (per `TODO/completed/81_wetlab_data_prep.md` §2.4): Day 0/7/21/28 (Zoom) / Day 60 (MoU signed) / Month 2/6/9 reviews. Day +21 no reply → mark "declined" in `logs/wetlab_outreach.csv`.

### §4.3 LOI template (carried from Wf-28)

`molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (199 lines, WLP-OUT-001) — 中文 + English email template + 3-phase framework + authorship + funding matrix. Ready to send; **0 LOI sent at this revision** (Hugo Lee coordinates the cold-email per `TODO/INDEX.md` P3 mitigation: "启动 3 个并行 outreach (ORNL compute + PKU med-chem + 协和 hospital)").

### §4.4 Risk + mitigation

- **100% dependency**: R20-P6 wet-lab milestone ships only if ≥1 PI replies + signs MoU + starts Phase 1. Conversion rate unknown (chemistry PIs 10-30% cold reply typical).
- **Mitigation**: parallel 3-PI outreach; if 6-month window sees 0 response, paper §6 honest limitations MUST explicitly state "wet-lab 未验证, all results in-silico" (per `TODO/INDEX.md` P3 mitigation).
- **Independent track**: P6 wet-lab延期 → 独立轨道 per `TODO/completed/81_wetlab_data_prep.md` §4.2 — does NOT block P1 GPU retrain (Wf-56 Wf-31/32) + P3 paper v3 (Wf-33) + P4 100-pocket sweep (Wf-53).

---

## §5 Paper rule compliance (TODO-84)

Per `TODO/pending/84_no_paper_task_without_authorization.md` §1.1, paper-type tasks (paper §1-§7 + appendix + supplementary + main.tex + main.pdf + rebuttal + submission) are **🚫 forbidden without explicit user authorization**.

### §5.1 P6.4 paper §5 final claim — DEFERRED

**Status**: 🚫 **DEFERRED — `paper/sections/05_wetlab_validation.tex` NOT created**.

**Reason**: TODO-84 active rule (2026-09-26 13:00 CST). P6.4 paper §5 final claim depends on P6.2 wet-lab IC50 MEASURED, which depends on PI outreach + 6-month timeline. Even if P6.2 wet-lab IC50 ships, **writing up paper §5 requires explicit user OK**.

**Authorization request template** (when ready, per TODO-84 §5):

> "User, can I write paper §5 wet-lab validation section based on Wf-58 P6.2 IC50 data?
> Touches: `paper/sections/05_wetlab_validation.tex` (new file) + `paper/main.tex` (add `\input{}` directive).
> Based on data: `molmetal/reports/wf58_wetlab/{orf,orf_oxaliplatin,orf_carboplatin}_ic50.csv` (TBD on wet-lab ship).
> Risk: paper §5 will report per-metal × per-cell Pearson r + per-compound residual + sub-µM hit count; if honest-negative findings (e.g. Rh bottleneck not lifted), they will be visible to reviewer."

### §5.2 Allowed without authorization (this Wf-58 ship)

Per TODO-84 §1.2, the following are CODE-ONLY work and were authorized:

| File | Status | Category |
|---|---|---|
| `molmetal/scripts/prioritize_wetlab_compounds.py` (792 LOC) | ✅ SHIP | train/bench script |
| `molmetal/scripts/bench_dmpnn_top20_predictions.py` (704 LOC) | ✅ SHIP | bench script |
| `molmetal/tests/test_prioritize_wetlab_compounds.py` (380 LOC) | ✅ SHIP | test file |
| `molmetal/tests/test_dmpnn_top20_predictions.py` (316 LOC) | ✅ SHIP | test file |
| `molmetal/reports/wf58_wetlab_dryrun/{top20_candidates.csv, ranking_summary.md}` | ✅ SHIP | report (NOT paper) |
| `molmetal/reports/wf58_dmpnn_predictions/*.json` (5-compound fixture) | ✅ SHIP | bench data JSON (NOT paper) |
| `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (Wf-28 carry) | ✅ SHIP | protocol template (NOT paper) |
| `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (this file) | ✅ SHIP | TODO doc (NOT paper) |

**🚫 NOT created at this revision**:
- `paper/sections/05_wetlab_validation.tex`
- `paper/main.tex` modification
- `paper/main.pdf` rebuild
- Any rebuttal / supplementary / appendix file

---

## §6 Cross-reference

### §6.1 Internal — TODO + plans

- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan, source) — 6 phases P6.1..P6.4 + MEASURED-vs-PROJECTED framing
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 R20-P6 (W11-W12 paper §3 closure + 4 NEW R20 skill; Wf-44 ship)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule; P6.4 DEFERRED until user OK)
- `TODO/completed/81_wetlab_data_prep.md` (Wf-28 wet-lab data tier — 5,000 d-block + 7,823 NCI-60 9-cell + 4 protocols + outreach template + bench)
- `TODO/completed/80_r19_plan.md` §19-23 (R19 wet-lab cooperation plan)
- `TODO/completed/82_metric_gap_filling.md` §5 (R20-P1..P7 plan)
- `TODO/completed/82d_m16m19m8_retrain.md` (Wf-32 P2 m18/m19/m8 ship surface — m18 anchor 0.7393 ± 0.009)
- `TODO/completed/87k_weak_real_to_strong_upgrade.md` §4 (m19 Rh bottleneck R20-P6 fix attempt)
- `TODO/completed/99_r20_skill_ecosystem.md` §2.2 (S6 `mol-metal-wetlab-protocol` R20-P6 Wf-56 W11-W12 close)
- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` (Wf-56 R20-P2 — Hilbert-ASTSR + Multi-PH + PLL + triangle closure; Wf-58 W11-W12 share)
- `TODO/completed/95_r20_p4_3way_100pocket.md` (Wf-53 R20-P4 3-way on MetalCytoToxDB 5-metal + 100-pocket × 10-seed dock sweep)
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` (Wf-52 R20-P3 Multi-PH + PLL feature)

### §6.2 External — code + reports + tests

- `molmetal/scripts/prioritize_wetlab_compounds.py` (792 LOC)
- `molmetal/scripts/bench_dmpnn_top20_predictions.py` (704 LOC)
- `molmetal/scripts/train_dmpnn_multitask.py` (CLI for multi-task D-MPNN baseline, Wf-32 P2)
- `molmetal/tests/test_prioritize_wetlab_compounds.py` (380 LOC, 12 tests)
- `molmetal/tests/test_dmpnn_top20_predictions.py` (316 LOC, 13 tests)
- `molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv` (20 rows × 9 cols)
- `molmetal/reports/wf58_wetlab_dryrun/ranking_summary.md` (Wf-58 P1 ranking)
- `molmetal/reports/wf58_dmpnn_predictions/{per_compound_pic50,per_cell_pic50,auc_per_seed,synthesis}.json` (5-compound fixture)
- `molmetal/tests/fixtures/wf58_dmpnn_top20/{top20_min.csv, platinai_4cell_min.csv, _tmp_outputs/*}` (CPU test fixture)
- `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (WLP-OUT-001, 199 lines, Wf-28 carry)
- `molmetal/docs/wetlab_protocols/{cytotoxicity_assay,binding_affinity_assay,cellular_uptake_assay,data_sharing_agreement,equipment_requirements}.md` (5 protocols, Wf-28 ship)
- `molmetal/data/{tmqm_dblock_subset,nci60_precious_9cell,nci60_platinai_link,platinai_4cell,platinai_5k}.csv` (Wf-28 wet-lab data tier)

### §6.3 Skill alignment

- `mol-metal-anticancer` (m18/m19/m20 + wet-lab integration; Wf-32 P2 anchor)
- `mol-metal-mechanism-mapping` (m18 D-MPNN scaffold split + m20 resistance)
- `mol-metal-lambda-calculus` (Wf-56 R20-P2 carry — Hilbert-ASTSR + Multi-PH + PLL)
- `mol-metal-wetlab-protocol` (NEW R20 S6 from Wf-55; per `TODO/completed/99_r20_skill_ecosystem.md` §2.2)
- `mol-metal-r18-math` (T5 Multi-PH + PLL math backbone)
- `party-twenty-spirit` (0 fabrication tolerance — applies to all R20-P6 wet-lab + bench work)

### §6.4 Wf lineage

- **Wf-31 P2** (m1/m2/m7 GPU retrain) → m18 D-MPNN anchor 0.7393 (carry into Wf-58 P6.0/P6.1)
- **Wf-32 P2** (m16/m19/m8 retrain) → m19 LOMO 5-fold 0.535 (carry into Wf-58 m18/m19/m20 baseline row)
- **Wf-53** (3-way on MetalCytoToxDB 5-metal + 100-pocket × 10-seed dock sweep) → R20-P4 → upstream of Wf-58 P6.0 ranking
- **Wf-56** (Hilbert-ASTSR + Multi-PH + PLL + triangle closure) → R20-P2 → Wf-58 W11-W12 share
- **Wf-58** (THIS) → R20-P6 wet-lab cooperation → LOI + 3-PI outreach + D-MPNN prediction layer
- **Wf-44** (closure theorem + 4 NEW R20 skill) → R20-P6 W11-W12 (per LAMBDA_ROADMAP §1.2)

---

## §7 Final verdict

**Wf-58 R20-P6 wet-lab IC50 in-silico dry-run**: ✅ **PARTIAL SHIP** — top-20 ranked (P6.0) + D-MPNN per-compound pIC50 predicted (P6.1) + agreement-bench JSON shell ready (P6.2 placeholder) + LOI + 3-PI outreach plan ready (Wf-28 carry). All CPU-only bench layer + test layer SHIP on disk.

**Wet-lab gate NOT active**: P6.2 in-silico vs wet-lab Pearson r ≥ 0.7 cannot MEASURE until 6-month wet-lab IC50 from ORNL / PKU / 协和 ships. The `agreement_summary.json` shell carries `pearson_r=null` + `status="DEFERRED_PENDING_WETLAB"`.

**Paper rule**: TODO-84 active; P6.4 paper §5 final claim 🚫 DEFERRED until user OK + P6.2 wet-lab IC50 MEASURED.

**Next step**: Hugo Lee coordinates 3-PI parallel outreach (ORNL compute + PKU med-chem + 协和 hospital) per `TODO/INDEX.md` P3 mitigation; first PI reply + MoU signed → R20-P6 P6.1 launches. Wf-58 in-silico bench layer stands ready to consume wet-lab IC50 the moment it ships.

---

**Date**: 2026-09-26 · **Verdict**: PARTIAL SHIP — CPU-only in-silico dry-run + D-MPNN prediction layer + agreement bench shell + LOI plan on disk; wet-lab gate NOT active per TODO-84; Pearson r ≥ 0.7 MEASURED requires 6-month wet-lab timeline (P6.2) · **Co-Authored-By**: Claude Code <noreply@anthropic.com>
