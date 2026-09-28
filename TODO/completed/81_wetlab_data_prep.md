# TODO 81 — Wf-28 Wet-Lab Data Prep synthesis (Phase 1-5 ship)

**Date**: 2026-09-26 (Wf-28 ship)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ SHIP (Phase 1-5 wet-lab data tier ready; outreach pending)
**Parent**: `TODO/03_plans/R19_PLAN.md` §19-23 (wet-lab 合作)
**Sibling inputs**:
- `TODO/03_plans/R19_PLAN.md` §19.1 wet-lab 合作总览 (ORNL/PKU/其他)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §1.2 (8 weak metrics — R19 wet-lab to close)
- `molmetal/reports/wf_wetlab_plan/final.md` (Top-10 candidate selection + ¥70-130k budget)
- `TODO/completed/69_pitfall_p4_p5_extended.md` (P5.2-ext wet-lab plumbing helpers)
- `TODO/completed/80_r19_plan.md` (Wf-27 R19 PLAN)

---

## §1 TL;DR

Wf-28 wet-lab data prep 5 phases 全部 ship, R19 wet-lab tier ready to launch:

| Deliverable | 状态 | 文件 | 规模 |
|---|---|---|---|
| **Phase 1** tmQM d-block 5K 候选 | ✅ SHIP | `molmetal/data/tmqm_dblock_subset.csv` | 5,000 complexes (30 d-block metals stratified) |
| **Phase 2** NCI-60 9-cell precious-metal prep | ✅ SHIP | `molmetal/data/nci60_precious_9cell.csv` | 7,823 long-form rows × 9 cell lines × 5 precious metals |
| **Phase 3** PlatinAI 4-cell cross-link | ✅ SHIP | `molmetal/data/nci60_platinai_link.csv` | cross-link table |
| **Phase 4** Wet-lab 协议 (4 protocols + outreach) | ✅ SHIP | `molmetal/docs/wetlab_protocols/*.md` | 7 docs / 1,266 lines |
| **Phase 5** End-to-end bench (compat verdict) | ✅ SHIP | `molmetal/reports/wf28_wetlab_data/bench_summary.md` | all 3 corpora PASS ±2σ comparability |

**关键 verdict**: Wf-28 准备好 wet-lab tier — 5,000 d-block 候选 + 7,823 NCI-60 rows + 4 protocol + outreach 模板 + bench compatibility 验证完成。R19 wet-lab milestone 1 (ORNL m18 protein-metal binding) 启动时,所有 data + protocol + template 都 on disk。

---

## §2 Per-deliverable

### §2.1 Phase 1 — tmQM d-block 5K subset

**Goal**: 从 tmQM 108K (Balcells 2020, MIT licensed) 抽取 5,000 wet-lab-feasible d-block 候选。

**Script**: `molmetal/scripts/extract_tmqm_dblock.py`
**Output**:
- `molmetal/data/tmqm_dblock_subset.csv` (5,000 rows × 14 cols)
- `molmetal/reports/wf28_wetlab_data/tmqm_dblock_summary.md`
- `molmetal/reports/wf28_wetlab_data/tmqm_dblock_stats.json`

**Filters applied**:
1. SMILES parseable by RDKit
2. Coordination number in [2, 9]
3. Heavy-atom window [8, 60]
4. SA-score computable (Ertl-Schuffenhauer)
5. Stratified random sample across 30 d-block metals

**Per-metal allocation** (top 6 锚定 metallodrug metals):

| Metal | n | Mean MW | Mean QED | Wet-lab 用 |
|---|---:|---:|---:|---|
| Pt | 411 | 659.9 | 0.277 | cisplatin / carboplatin analogues |
| Ru | 444 | 624.3 | 0.230 | NAMI-A / KP1339 衍生物 |
| Ir | 242 | 713.4 | 0.228 | photoactivatable C-H insertion |
| Pd | 533 | 588.8 | 0.252 | cross-coupling 副产物基线 |
| Au | 217 | 631.6 | 0.250 | auranofin 衍生物 |
| Rh | 219 | 598.4 | 0.224 | Rh(III) catalysts |

### §2.2 Phase 2 — NCI-60 9-cell precious-metal prep

**Goal**: NCI-60 9 cell line (MCF-7/A549/A2780/A2780cis/HCT-116/PC-3/HL-60/SK-OV-3/DU-145) × 5 precious metals (Ru/Ir/Rh/Os/Re) 的 cytotoxicity 训练 substrate。

**Script**: `molmetal/scripts/prepare_nci60_data.py`
**Output**:
- `molmetal/data/nci60_precious_9cell.csv` (7,823 rows × 4 cols)
- `molmetal/data/nci60_platinai_link.csv` (cross-link)
- `molmetal/reports/wf28_wetlab_data/nci60_summary.md`
- `molmetal/reports/wf28_wetlab_data/nci60_platinai_link.json`

**Cell-line coverage**:

| Cell line | n_rows | mean pIC50 | n_active (IC50 < 10 µM) | active_rate |
|----------:|-------:|-----------:|-------------------------:|------------:|
| MCF7 | 1,964 | 4.787 | 728 | 0.371 |
| A549 | 2,647 | 4.697 | 865 | 0.327 |
| A2780 | 1,095 | 4.894 | 466 | 0.426 |
| A2780cis | 564 | 4.849 | 235 | 0.417 |
| HCT116 | 701 | 4.873 | 310 | 0.442 |
| PC3 | 275 | 4.945 | 114 | 0.415 |
| HL60 | 225 | 5.077 | 121 | 0.538 |
| SKOV3 | 191 | 4.693 | 65 | 0.340 |
| DU145 | 161 | 5.055 | 83 | 0.516 |

### §2.3 Phase 3 — PlatinAI 4-cell cross-link

**Goal**: 把 precious-metal 9-cell 与 PlatinAI 4-cell (Pt complexes) join,生成 cross-link table。

**Outcome** (honest): 0/7,823 overlap by construction (precious-metal filter excludes Pt; PlatinAI 4-cell is Pt-only). 两个 corpora 是 complementary 而非 overlapping。这是 expected outcome,意味着 cross-cell r4 lambda-only 没法直接做。

### §2.4 Phase 4 — Wet-lab 协议 + outreach 模板

**Files**: `molmetal/docs/wetlab_protocols/` (7 docs / 1,266 lines)

| File | Lines | Status |
|---|---:|---|
| `cytotoxicity_assay.md` (WLP-CYTO-001) | 211 | ✅ Template ready (no wet-lab executed) |
| `binding_affinity_assay.md` | 218 | ✅ SPR / ITC protocol template |
| `cellular_uptake_assay.md` | 204 | ✅ ICP-MS quantification template |
| `collaborator_outreach_template.md` (WLP-OUT-001) | 199 | ✅ 中文 + English email template + framework |
| `data_sharing_agreement.md` | 145 | ✅ MoU template |
| `equipment_requirements.md` | 171 | ✅ BSL-2 + ICP-MS spec |
| `README.md` | 118 | ✅ Index |

**Outreach template** (核心): 3-phase joint program:
- Phase 1 (0-2 个月): 5-10 ranked Pt/Ru/Ir 候选 + in-silico prediction (IC50/logP/redox/HSA K_d/uptake)
- Phase 2 (2-6 个月): compound aliquots shipped + wet-lab SRB/MTT ≥4 cell lines + ICP-MS uptake ≥2 cell lines ≥3 time-points
- Phase 3 (6-12 个月): joint manuscript + updated model retrained on measured data

**Authorship**: Phase-2 sub-µM IC50 hit → co-corresponding author。无 hit → mid-author + data validation section。

**Follow-up cadence**: Day 0/7/21/28 (Zoom) / Day 60 (MoU signed) / Month 2/6/9 reviews。Day +21 无回复 → mark "declined" in `logs/wetlab_outreach.csv`。

### §2.5 Phase 5 — End-to-end bench

**Script**: `molmetal/scripts/bench_wetlab_data.py` (727 lines, RDKit + pandas CPU-only)
**Output**:
- `molmetal/reports/wf28_wetlab_data/bench_summary.json`
- `molmetal/reports/wf28_wetlab_data/bench_summary.md`

**Comparability verdict**: 3 candidate corpora (tmqm_dblock + nci60_precious_9cell + platinai_4cell) vs PlatinAI 5K reference。**3/3 OK** — 都在 ±2σ_ref envelope 内,wet-lab tier 在 r18 SBDD reference 同一 descriptor neighbourhood。

---

## §3 MEASURED bench metrics

**Date**: 2026-09-25 21:15:29 UTC | **Seed**: 20260926 | **CPU-only / RDKit + pandas**

| Corpus | n_candidates | n_rows | mean_mw | mean_logp | mean_qed | mean_sa | parse_fail | secs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **tmqm_dblock** | 4,982 | 5,000 | 592.0 | 3.76 | 0.257 | 0.552 | 0.00% | 9.27 |
| **nci60_precious_9cell** | 5,207 | 5,207 | 578.7 | 5.24 | 0.282 | 0.742 | 0.00% | 10.26 |
| **platinai_4cell** | 4,078 | 4,078 | 674.8 | 3.13 | 0.263 | 0.583 | 0.00% | 7.27 |
| **platinai_5k** (reference) | 5,000 | 5,000 | 657.9 | 4.19 | 0.233 | 0.579 | 0.04% | 9.43 |

**Descriptor stds** (reference):

| Corpus | σ_mw | σ_logp | σ_qed | σ_sa |
|---|---:|---:|---:|---:|
| tmqm_dblock | 162.1 | 4.60 | 0.144 | 0.112 |
| nci60_precious_9cell | 204.2 | 4.82 | 0.188 | 0.096 |
| platinai_4cell | 182.3 | 5.38 | 0.141 | 0.105 |
| platinai_5k (reference) | 184.4 | 4.76 | 0.138 | 0.100 |

**Comparability vs PlatinAI 5K (the r18 reference)**:

| Candidate → Reference | Δ_mw | Δ_logp | Δ_qed | Δ_sa | Verdict |
|---|---:|---:|---:|---:|---|
| tmqm_dblock → platinai_5k | -65.96 | -0.44 | +0.024 | -0.027 | **OK** |
| nci60_precious_9cell → platinai_5k | -79.19 | +1.05 | +0.048 | +0.163 | **OK** |
| platinai_4cell → platinai_5k | +16.92 | -1.06 | +0.030 | +0.004 | **OK** |

**Per-axis reading guide** (4 corpora all in expected metallodrug neighborhood): mean_mw 570-680 Da (above Ro5 500, expected for coordination complexes); mean_logp [3.1, 5.2]; mean_qed [0.23, 0.28] (low — metallodrug-shaped); mean_sa NCI-60 0.74 vs rest 0.55-0.58 (precious-metal long-form already passed synthesis, so easier).

**Honest framing**: ±2σ window intentionally wide (stricter KS-test/MMD verdict in `benchmark_r18_platinai.py`); mean_sa is Ertl SA unit-transformed to [0,1] (higher = easier, matches r18 direction); bench exercises actual CSV files on disk; wall-clock ≈ 35 s single CPU thread。

---

## §4 R19 wet-lab milestone 1 准备

**Wf-28 已准备好 R19 wet-lab milestone 1 (ORNL/PKU) 启动所需的 data + protocol + template**。

### §4.1 4 tiers all ready

- **Data tier**: ~5,000 d-block candidates (tmQM) + 7,823 NCI-60 precious-metal 9-cell + 4,078 PlatinAI 4-cell + all comparability-checked vs PlatinAI 5K (±2σ)
- **Protocol tier**: cytotoxicity (WLP-CYTO-001 SRB 72h) + binding_affinity (SPR primary + ITC confirm) + cellular_uptake (ICP-MS pg/cell) + data_sharing (MoU) + equipment_requirements (BSL-2)
- **Outreach tier**: collaborator_outreach_template (WLP-OUT-001) 中英文 + 3-phase framework + cadence (Day 0/7/21/28/60 + Month 2/6/9 reviews); sub-µM IC50 hit → co-corresponding authorship; no funds transfer
- **Bench tier**: bench_summary.{json,md} full provenance + descriptor statistics + comparability verdict; all CSVs on disk; all summary reports at `molmetal/reports/wf28_wetlab_data/`

### §4.2 Risk

- 100% 依赖 ORNL/PKU 等 facility 实际 reply outreach — **0 outreach sent** at this revision
- P4 wet-lab milestone 延期 → 独立轨道 (R19_PLAN §19.2 R-8),不阻塞 P1 GPU retrain + P3 paper v2

---

## §5 合作框架 (collaboration framework)

### §5.1 3-phase joint program (源自 outreach template)

| Phase | Duration | Mol-Metal deliverable | Wet-lab deliverable |
|---|---|---|---|
| **Phase 1** | 0-2 mo | 5-10 ranked Pt/Ru/Ir 候选 + structures + in-silico IC50/logP/redox/HSA K_d/uptake | Acknowledge fit; counter-sign MoU; cell-line auth + mycoplasma test |
| **Phase 2** | 2-6 mo | Compound aliquots shipped (or synthesis recipe); updated ranking per round | SRB/MTT ≥4 cell lines; ICP-MS uptake ≥2 cell lines ≥3 time-points; optional ITC |
| **Phase 3** | 6-12 mo | Joint manuscript draft; updated model retrained on measured data; co-design next round | Repeat protocols on top candidates; optional X-ray/NMR speciation |

### §5.2 R18 modules 提供 + Author/Funding

**Mol-Metal → wet-lab**: click-chemistry metallodrug generator (`pt_click_compat.py`) + lambda-combinator search (`proof_search.py` + `learned_shaping.py`) + coupled flow-matching backbone (`flow_matching_lipman/`) + 100-pocket Pt/Ru/Ir ligand-set scored against PlatinAI (30-50 top hits/pocket) + predicted IC50 (80% CI) + logP/redox/HSA K_d/counter-screen + open datasets (PlatinAI CC0 + tmQM CC-BY + MetalCytoToxDB Ru subset CC-BY-SA) + open-source pipeline (wet-lab team can audit every prediction)。

**Authorship**: Phase-2 ≥1 sub-µM IC50 → wet-lab PI = co-corresponding; Phase-2 alone (no hit) → mid-author + data validation section。

**Funding**: Mol-Metal side = software + compute + in-silico + manuscript prep (no wet-lab budget); wet-lab side = reagents + cell-line + plate-reader/ICP-MS time + operators; joint grants encouraged (submit under wet-lab institution); no overhead/indirect-cost transfers。

### §5.3 Joint paper outcome

Wf-28 准备好的 candidate libraries + in-silico predictions + protocol templates → wet-lab 验证 → experimental data → 共同 paper (Phase 3):
- m18 protein-metal binding 30 data points (10 compound × 3 replicate)
- cisplatin IC50 32 curves (4 cell × 8 compound)
- lead compound activity 3-5 candidates
- **Total**: 65-67 experimental data points → paper §4 + §S7 supplementary
- **R20+**: clinical trial 注册 → Phase I → Phase II

---

## §6 Cross-reference

### §6.1 Internal
- `TODO/03_plans/R19_PLAN.md` §19-23 (wet-lab 合作目标 ORNL/PKU/其他 + 3 milestones + 总结); §19.2 (Pitfall P5.2 wet-lab outreach — R17 wetlab_extensions.py skeleton); §24.1 NEW skill `mol-metal-wet-lab`
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §1.2 (8 weak metrics — R19 wet-lab future m18/m19/m20)
- `molmetal/reports/wf_wetlab_plan/final.md` (Top-10 candidate + ¥70-130k budget + 6-week timeline)
- `TODO/completed/69_pitfall_p4_p5_extended.md` (P5.2-ext wet-lab plumbing 5 helpers, R17 ship)
- `TODO/completed/60_pitfall_tier2.md` (P5.2 Tier-1 DONE; Tier-3 outreach DEFER → Wf-28 fill)
- `TODO/completed/80_r19_plan.md` (Wf-27 R19 PLAN §19)

### §6.2 External
- **Scripts**: `molmetal/scripts/{extract_tmqm_dblock,prepare_nci60_data,bench_wetlab_data}.py`
- **Data CSVs**: `molmetal/data/{tmqm_dblock_subset,nci60_precious_9cell,nci60_platinai_link,platinai_4cell,platinai_5k}.csv`
- **Reports**: `molmetal/reports/wf28_wetlab_data/{tmqm_dblock,nci60,bench}_*.{md,json}`
- **Protocols**: `molmetal/docs/wetlab_protocols/{cytotoxicity,binding_affinity,cellular_uptake,data_sharing,equipment}_*.md` + `collaborator_outreach_template.md` + `README.md`

### §6.3 Skill alignment
NEW R19 skill `mol-metal-wet-lab` (per R19_PLAN §24.1) + existing `mol-metal-anticancer` (m18/m19/m20 + wet-lab integration) + `party-twenty-spirit` (0 fabrication tolerance — applies to outreach timeline)。

---

## §7 Honest framing

**Wf-28 ship status**: 5/5 phases complete; data + protocol + outreach + bench tiers all on disk. x1-x5 outputs verifiable in `molmetal/reports/wf28_wetlab_data/` + `molmetal/docs/wetlab_protocols/`。

**Risks**: R19 wet-lab milestone 100% dependent on ORNL/PKU reply outreach (Wf-28 准备 data + protocol + template,但 **0 outreach sent** at this revision); outreach conversion rate unknown (chemistry PIs 10-30% cold reply typical); milestone 延期 → 独立轨道,不阻塞 P1 GPU retrain + P3 paper v2 (R19_PLAN §19.2 R-8)。

**Boundaries (what Wf-28 does NOT do)**: NOT actually send outreach (template ready, coordinator TBD); NOT actually run wet-lab experiment (CPU-only data prep + protocol design); NOT actually synthesize compound (only in-silico ranking + protocol templates); NOT guarantee wet-lab facility accept (depends on PI response)。

**Success metric**: Wf-28 ship = data + protocol + template + bench all on disk。R19 wet-lab milestone ship = (separately) ≥1 wet-lab PI signed MoU + started Phase 1 acknowledgement。

---

**Date**: 2026-09-26 (Wf-28 ship)
