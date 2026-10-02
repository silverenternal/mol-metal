# Wf-299D — TODO-119 P1.4 Cross-Cohort Generalization Plan

**Date:** 2026-09-29
**Workflow:** Wf-299D (TODO-119 P1.4 cross-cohort prep)
**Owner:** Hugo Lee + ultracode
**Status:** PLAN — no GPU execution yet; sequencing P1.4 ablation behind per-cohort data availability
**Honest framing gate:** party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED discipline)

---

## §0 TL;DR

| item                          | value                                                              |
|-------------------------------|--------------------------------------------------------------------|
| n_cohorts                     | **3** (PlatinAI 4-cell + DrugOOD ECFP4 + MetalCytoToxDB v2)         |
| CFM-backbone shared across    | chemprop_v2_mt, h=128, n_layers=3, joint_train, YuelBond bond head  |
| Per-cohort MEASURED target    | decode_ratio ≥ 0.5 (current honest-negative 0.0; P1.4 target gate) |
| Cross-cohort comparison       | Per-cohort decode_ratio table + per-cohort wasserstein + catherine  |
| Honest framing                | No MEASURED cross-cohort lift until all 3 cohorts reach ≥ 0.5      |

---

## §1 Per-cohort MEASURED target

Each cohort ships its own decoder-evaluation script + CSV row schema; the
CFM-backbone is **shared** (chemprop_v2_mt + YuelBond + W2 + Catherine
closure-bounded), but the **labels + splits + scale differ**.

| cohort                  | path / rows / split                                                | MEASURED target                | gate                       |
|-------------------------|-------------------------------------------------------------------|--------------------------------|----------------------------|
| PlatinAI 4-cell         | `.data/molmetal/platinai_4cell_mbfinder.csv` (101 lines, 100 mols, A549-only currently) + Wf-299A extension to 4 cell lines | decode_ratio ≥ 0.5 on A549 subset | Wf-299A ships MCF7/A2780/A2780cis first |
| DrugOOD ECFP4 (Wf-275B) | `.data/molmetal/drugood_ecfp4_scaffold.csv` (95,236 rows, 70/10/20 Bemis-Murcko split, 0 leak) | decode_ratio ≥ 0.5 on test (scaffold OOD) | classifier head → regress head switch for `reg_label` (already in CSV col `pIC50`) |
| MetalCytoToxDB v2 (Wf-261) | `.data/molmetal/metalcytotox_5metal_multitask_v2.csv` (26,632 rows, 5 metals: Ru/Ir/Os/Rh/Re, 70/10/20 Bemis-Murcko strict, 3069 unique scaffolds) | decode_ratio ≥ 0.5 on test (scaffold OOD, per-metal + pooled) | already in P1.4 config §data{} |

The MEASURED-vs-PROJECTED split is preserved: `target_decode_ratio: 0.5` is
the **honest-negative gate from current state 0.0**, NOT a projected lift.

---

## §2 Per-cohort data inspection (MEASURED)

### §2.1 PlatinAI 4-cell (Wf-295B + Wf-299A pending)

```
mb_id,imageId,smiles,mw,ceilLineCount,bioDataCount,has_mcf7,has_a549,has_a2780,has_a2780cis,n_cell_lines_with_data,n_biodata_total,total_cell_lines_listed
3662,MB-Pt-002305,[OH2][Pt]([Cl])([Cl])N12CCN(CC1)CC2,396.18,6,6,0,1,0,0,5,22,5
```

- **SMILES column:** `smiles` (RDKit-readable; `[Pt]` coordination SMILES)
- **Metal/labels:** Metal = Pt (encoded by SMILES `[Pt]` token + `imageId` prefix `MB-Pt-*`). Per-cell-line bio-data is **boolean coverage** (`has_*`) + counts, NOT raw IC50 numerics.
- **Train/val/test split:** **NONE shipped yet**. Wf-299A must:
  1. Extend MB-Finder harvest from 100 to ≥ 1000 Pt compounds
  2. Scrape numeric IC50 from the paper-gated references (DOIs already in `.meta.json`)
  3. Apply 70/10/20 Bemis-Murcko split (zero leak invariant)
- **Per-cell counts:** A549=100, MCF7=0, A2780=0, A2780cis=0 → single-paper slice
- **Honest caveat:** Wf-295B §2 binds: "Per-cell-line IC50 NUMERIC VALUES are NOT exposed by any of the 5 [MB-Finder API] endpoints". Cross-cohort comparison blocked on Wf-299A.

### §2.2 DrugOOD ECFP4 (Wf-275B)

```
smiles,scaffold,ecfp4_bits,ecfp4_popcount,pIC50,active,subset,split,idx
N#Cc1cc(...),__empty__...,01000000...,71,8.68,1,lbap_core_ic50_assay,train,0
```

- **SMILES column:** `smiles` (canonical RDKit; 95,236 rows; 0 RDKit failures)
- **Metal/labels:** **No metal** — DrugOOD is **organic-only** (this is the CFM-OOD transfer challenge: organic backbone + reg_label only). `pIC50` is from `reg_label` (continuous regression target).
- **Train/val/test split:** Bemis-Murcko `scaffold_split_largest_first` → train=66,665 / val=9,524 / test=19,047 (70/10/20 target fractions). **0 scaffold leak** train↔val, train↔test, val↔test.
- **ECFP4 features:** radius=2, nBits=2048, hex-packed, popcount stored separately (71-bit average).
- **Cross-cohort framing:** DrugOOD acts as **the metal-free OOD control** — if CFM achieves decode_ratio ≥ 0.5 here without any metal-conditioning, the lift on MetalCytoToxDB / PlatinAI is attributable to metal-conditioning, not just to the CFM architecture.

### §2.3 MetalCytoToxDB v2 (Wf-261)

```
smiles,metal,scaffold_id,split,activity,pic50,cell_line,source
N#Cc1ccc(...),Ru,c1ccc(...)...,train,0,4.4437,MCF-7,MetalCytoToxDB
```

- **SMILES column:** `smiles` (RDKit canonical; 26,632 rows; 169 null-scaffold rows dropped)
- **Metal/labels:** `metal` ∈ {Ru, Ir, Os, Rh, Re} + `pic50` continuous + `activity` 0/1 + `cell_line` (MCF-7 + 7 others)
- **Train/val/test split:** `scaffold_split: true` flag in P1.4 config; 3,069 unique Bemis-Murcko scaffolds; strict 70/10/20 zero-leak partition
- **Already the P1.4 primary cohort** — config `data.cohort_filter.metals: [Ru, Ir, Os, Rh, Re]` matches v2 file exactly
- **Cross-cohort role:** the **in-distribution benchmark** for P1.4 (matches the Wf-124 P2 protocol exactly so per-lever attribution is comparable)

---

## §3 Cross-cohort comparison table

After P1.4 (4 ablations × 3 epochs) completes, write
`molmetal/reports/wf299d_cross_cohort/results_table.md` with:

| cohort              | n_rows | split strategy | CFM MEASURED decode_ratio (wave a_full_stack) | MEASURED wasserstein_loss | MEASURED catherine_pass_rate | honest status |
|---------------------|--------|----------------|------------------------------------------------|---------------------------|------------------------------|---------------|
| PlatinAI 4-cell     | 100 → 1000+ (Wf-299A) | Bemis-Murcko (zero leak) | TBD (BLOCKED on Wf-299A) | TBD | TBD | BLOCKED — no real numeric labels yet |
| DrugOOD ECFP4       | 95,236 | Bemis-Murcko (Wf-275B, 0 leak) | TBD (P1.4 wave a) | TBD | TBD | READY — run P1.4 wave a on this CSV |
| MetalCytoToxDB v2   | 26,632 | Bemis-Murcko (Wf-261, 0 leak) | TBD (P1.4 wave a) | TBD | TBD | READY — already in P1.4 config |

Then per-lever attribution table (4 ablations × 3 cohorts × 3 metrics):

| ablation          | PlatinAI decode_ratio | DrugOOD decode_ratio | MetalCytoToxDB decode_ratio | per-lever Δ |
|-------------------|------------------------|----------------------|-----------------------------|-------------|
| a_full_stack      | TBD                    | TBD                  | TBD                         | control     |
| b_yuelbond_only   | TBD                    | TBD                  | TBD                         | Δ_yuelbond  |
| c_w2_only         | TBD                    | TBD                  | TBD                         | Δ_w2        |
| d_catherine_only  | TBD                    | TBD                  | TBD                         | Δ_catherine |

Interaction terms (YuelBond×W2, YuelBond×Catherine, W2×Catherine) deferred per P1.4 §honest_framing.deferred.

---

## §4 Honest framing expectations (party-twenty-spirit §3.5)

### §4.1 DO claim (after P1.4 completes)

- "CFM (chemprop_v2_mt + YuelBond + W2 + Catherine) achieves decode_ratio ≥ 0.5 MEASURED on DrugOOD test (scaffold OOD)"
- "CFM achieves decode_ratio ≥ 0.5 MEASURED on MetalCytoToxDB v2 test (scaffold OOD, 5-metal pooled)"
- "Per-lever attribution: 4 ablation deltas (Δ_yuelbond, Δ_w2, Δ_catherine, Δ_interaction) reported as MEASURED numbers with 1-seed × 3-epoch CI (NOT 5-seed × 50-epoch; that's R20-P3 promised scope, not P1.4 scope)"

### §4.2 DO NOT claim

- "Cross-cohort generalization" as a single MEASURED number — each cohort has its own MEASURED gate; aggregation only after each cohort passes individually.
- "Per-lever percentage attribution" — interaction term may be >30% per TODO-182 §3.4; report per-lever Δs as raw MEASURED numbers, NOT as % of total lift.
- "PlatinAI 4-cell decode_ratio" — BLOCKED until Wf-299A ships real numeric IC50 labels.
- "OOD transfer from organic → metallo" — DrugOOD is metal-FREE; transfer claim requires the CFM to have learned metal-conditioning, which only MetalCytoToxDB tests. Report DrugOOD as the **organic-only negative control**, NOT as transfer evidence.

### §4.3 Deferred (out of P1.4 scope)

- 5-seed × 50-epoch reruns for paper-grade CI (R20-P3 scope, NOT P1.4)
- 3 pairwise interaction ablations (YuelBond×W2, YuelBond×Catherine, W2×Catherine) — would double wave count to 8
- PlatinAI 4-cell cross-tab (BLOCKED on Wf-299A)
- Cross-cohort OOD test: train on MetalCytoToxDB → test on DrugOOD (Tanimoto<0.4 protocol from Wf-44 P2 closure OOD)
- ablation_report.json write + audit (post-wave)

---

## §5 Sequencing (Wf-299D outputs → downstream blockers)

```
Wf-299D (this plan, DONE)
    ↓
Wf-299A: PlatinAI MB-Finder harvest 100 → 1000+ Pt compounds
    ↓ (BLOCKS PlatinAI cohort in §3 table)
Wf-299E: P1.4 wave a_full_stack on DrugOOD + MetalCytoToxDB
    ↓
Wf-299F: P1.4 waves b/c/d on DrugOOD + MetalCytoToxDB
    ↓
Wf-299G: write results_table.md + 4-ablation × 2-cohort MEASURED report
    ↓
Wf-124 P3 paper §3 per-lever attribution paragraph (honest framing gate)
```

---

## §6 Cross-references

- `configs/cfm_per_innovation_ablation_p1.4.yaml` (this plan's parent)
- `molmetal/reports/wf275b_drugood_adapter/*.md` (DrugOOD adapter ship)
- `molmetal/reports/wf295b_platinai_loader/*.md` (PlatinAI loader ship)
- `molmetal/reports/wf261_update/*.md` (MetalCytoToxDB v2 strict scaffold split)
- `TODO/completed/119_decode_ratio_strengthen.md` §3 (P1.4 origin)
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (P1.4 parent TODO)
- `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` §1.2 (P1.4 audit)
- `molmetal/reports/wf292d_platinai_bypass/strategy_report.md` §3 R1 (Wf-299A strategy)
- `TODO/pending/130_drugood_official_datasets.md` (Wf-275B parent)

---

**Status:** PLAN COMPLETE — no GPU work executed. Awaiting user OK to launch Wf-299A (PlatinAI extension) + Wf-299E (P1.4 wave a on DrugOOD + MetalCytoToxDB).