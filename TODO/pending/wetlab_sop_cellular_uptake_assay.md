# SOP-Cellular-Uptake — Institutional-Tailored ICP-MS Cellular Uptake Assay

**Document ID:** WLP-UP-INST-001
**Version:** 1.0 (institutional-tailored template; no wet-lab data collected yet)
**Status:** **TEMPLATE — not executed at this revision** (Hugo Lee coordinates PI outreach per `TODO/INDEX.md` §12.3 P3 mitigation)
**Authoring date:** 2026-09-27
**Owner:** Mol-Metal wet-lab coordinator (TBD) + 3 institutional PIs (ORNL / PKU / 协和)

---

## 1. Purpose and Scope

This SOP provides **institution-specific tailoring** of the Wf-28 generic
cellular uptake assay (`molmetal/docs/wetlab_protocols/cellular_uptake_assay.md`,
WLP-UP-001) for the 3 candidate PIs (ORNL / PKU / 协和). The technical
content (ICP-MS Pt/Pd/Au/Ru/Ir quantification + microwave acid digestion)
is shared across all 3 institutions; the institutional tailoring covers:

- ICP-MS instrument + microwave digester model (per institutional core facility).
- Cell-line scope (per PI focus: ORNL A549/A2780cis resistance; PKU A549/MCF7; 协和 A549/MCF7/A2780cis).
- Time-point coverage (per PI: ORNL 3 time-points; PKU 3 time-points; 协和 3 time-points with extra replicates).
- Acid waste handling (per institutional acid-waste SOP).

The PI adapts the **institution-specific §3 tailoring** to their own
SOP numbering; the §1+§2 technical content remains unchanged from
WLP-UP-001.

## 2. Shared technical content (carry from WLP-UP-001)

### 2.1 Reagents (shared)

| Reagent | Vendor | Catalogue # |
|---|---|---|
| Trace-metal-grade HNO₃ (67–70%) | Fisher | A509 |
| Trace-metal-grade HCl (32–35%) | Fisher | A508 |
| H₂O₂ (30%, TraceMetal) | Fisher | H341 |
| Pt standard (1000 µg/mL in 10% HCl) | Inorganic Ventures | CGPT1 |
| Pd standard | Inorganic Ventures | CGPD1 |
| Au standard | Inorganic Ventures | CGAU1 |
| Ru standard | Inorganic Ventures | CGRU1 |
| Ir standard | Inorganic Ventures | CGIR1 |
| Internal standard: Y or In (1000 µg/mL) | Inorganic Ventures | CGY1 / CGIN1 |
| PBS Mg²⁺/Ca²⁺-free | Gibco | 14190 |
| RIPA lysis buffer (with protease inhibitors) | Sigma | R0278 |
| BCA Protein Assay Kit | Thermo | 23225 |

### 2.2 ICP-MS isotope selection (shared)

| Metal | Primary isotope | Polyatomic interference mitigation |
|---|---|---|
| Pt | ¹⁹⁵Pt | KED mode (He) to remove ¹⁷⁹Hf¹⁶O / ¹⁷⁹Hf¹⁶O |
| Pd | ¹⁰⁵Pd | KED mode (He) |
| Au | ¹⁹⁷Au | Standard mode (no major interferences) |
| Ru | ¹⁰¹Ru | KED mode (He) to remove ⁶¹Ni⁴⁰Ar / ⁸⁷Sr¹⁴N |
| Ir | ¹⁹³Ir | KED mode (He) |

### 2.3 Sample preparation (shared)

- Cell seeding: 2 × 10⁵ cells/well in 6-well plate (2 mL complete medium); 24 h attachment.
- Drug exposure: 10 µM single concentration (matches biological relevance); 3 replicates per time-point.
- Time-points: **1 h / 6 h / 24 h** post-exposure.
- Vehicle control: 0.5% DMSO.
- Positive control: cisplatin 10 µM.
- Wash: 3× cold PBS (third wash retained as procedural blank).
- Lysis: 300 µL RIPA + protease inhibitors; 5 min ice; sonicate 3 × 10 s.
- Centrifuge: 16 000 × g, 15 min, 4 °C.

### 2.4 Microwave acid digestion (shared)

- Lysate + 2 mL trace-metal-grade HNO₃ + 0.5 mL H₂O₂ (30%) per vessel.
- Digestion program: ramp to 200 °C over 15 min; hold 200 °C for 20 min; cool to RT.
- Include reagent blank (HNO₃ + H₂O₂ only) and digestion blank (untreated cells) per run.
- Final dilution: 10 mL with 2% HNO₃ + internal standard (Y or In at 10 µg/L).

### 2.5 QC acceptance (shared)

| Check | Criterion |
|---|---|
| Reagent blank | < LOD (typically < 0.01 ng/mL) |
| Digestion blank | < LOD |
| Spike recovery | 90–110% |
| Replicate CV | ≤ 15% |
| LOD per metal | Pt 0.005 ng/mL / Pd 0.01 / Au 0.005 / Ru 0.01 / Ir 0.005 |

### 2.6 Output schema (shared)

Per (compound × cell × time-point) record:

| Field | Type | Example |
|---|---|---|
| `compound_id` | str | "platinai_0454" |
| `metal` | str | "Ir" |
| `cell_line` | str | "A549" |
| `time_h` | int | 6 |
| `concentration_uM` | float | 10.0 |
| `ng_metal_per_1e6_cells` | float | 12.4 |
| `nmol_metal_per_mg_protein` | float | 0.85 |
| `cv_pct` | float | 8.5 |
| `qc_pass` | bool | true |

JSON output conforming to `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` schema (with uptake extension).

---

## 3. Institutional tailoring

### §3.1 ORNL tailoring

**ICP-MS instrument**: Thermo iCAP RQ (ORNL ICP-MS Core Facility); KED mode with He cell gas.
**Microwave digester**: CEM MARS 6 with PFA vessels (15-position rotor).
**Cell-line scope**:
- **A549** (primary; lung cancer reference).
- **A2780cis** (primary; cisplatin resistance model).
- Time-points: 1h / 6h / 24h.
- Optional (Phase 3): MCF7 + LOX-IMVI.
**Replicate count**: 3 biological replicates per (compound × cell × time-point).
**Authorship tie-in**: see `TODO/pending/wetlab_loi_ornl_protein_metal.md`.
**Note**: ORNL 主要 focus 是 protein-metal binding (见 `wetlab_sop_binding_affinity_assay.md` §3.1); uptake 实验是 secondary, 与 binding 实验并行。

### §3.2 PKU tailoring

**ICP-MS instrument**: Agilent 7900 (PKU 医药卫生分析中心); KED mode with He cell gas; 或 PerkinElmer NexION 350D (PKU 药学院)。
**Microwave digester**: Anton Paar Multiwave 7000 with quartz vessels。
**Cell-line scope**:
- **A549** (primary)。
- **MCF7** (primary)。
- 可选: A2780 + A2780cis (顺铂耐药对比)。
- Time-points: 1h / 6h / 24h。
**Replicate count**: 3 biological replicates per (compound × cell × time-point)。
**Authorship tie-in**: see `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md`。
**Note**: PKU 主要 focus 是 IC50 曲线 (顺铂类); uptake 实验是 secondary, 若 PKU 课题组 capacity 可用时执行。

### §3.3 协和 tailoring

**ICP-MS instrument**: Thermo iCAP RQ (协和医院中心实验室); KED mode with He cell gas; 或 Agilent 7900 (协和医院引进)。
**Microwave digester**: CEM MARS 6 (协和医院中心实验室)。
**Cell-line scope** (NCI-60 9-cell 同步 + uptake):
- **A549** (primary; lung)。
- **MCF7** (primary; breast)。
- **A2780cis** (primary; 顺铂耐药, 与 A2780 对比)。
- Time-points: 1h / 6h / 24h。
- Replicate count: 3 biological replicates (可扩展至 5 若 capacity 允许)。
**Hazardous waste**: ICP-MS 酸性废液按 协和医院 酸性废液 SOP (中和至 pH 6-8) 单独收集; Pt / Pd / Au / Ru / Ir 含量高的废液按金属废液 SOP (碱性-氯化物沉淀) 二次处理。
**Authorship tie-in**: see `TODO/pending/wetlab_loi_xiehe_nci60.md`。
**Note**: 协和 主要 focus 是 NCI-60 9-cell IC50 + ICP-MS uptake 联合 panel; uptake 是 primary, 与 IC50 同步执行。

---

## 4. Cross-institution harmonization

To enable cross-PI uptake comparison (`wetlab_ic50_analysis_pipeline.md` §3.4 resistance signature analysis), the following MUST be harmonized:

- **Same drug concentration** (10 µM).
- **Same time-points** (1h / 6h / 24h).
- **Same cell-seeding density** (2 × 10⁵ cells/well).
- **Same output unit** (ng metal per 1e6 cells + nmol metal per mg protein).
- **Same internal standard** (Y or In at 10 µg/L).

The PI is encouraged to use the **Mol-Metal bench harness** for uptake quantification to avoid per-PI re-implementation drift.

## 5. Institutional SOP numbering

| Institution | SOP # | Title | Owner |
|---|---|---|---|
| ORNL | ORNL-UP-MOLMETAL-001 | Mol-Metal Cellular Uptake Assay (institutional tailoring of WLP-UP-INST-001) | ORNL PI |
| PKU | PKU-UP-MOLMETAL-001 | Mol-Metal 细胞摄取测定 (institutional tailoring of WLP-UP-INST-001) | PKU PI |
| 协和 | 协和-UP-MOLMETAL-001 | Mol-Metal NCI-60 细胞摄取 ICP-MS 测定 (institutional tailoring of WLP-UP-INST-001) | 协和 PI |

## 6. Honest framing (party-twenty-spirit §3.5)

- This SOP is a **template** ready for institutional adaptation; **NO wet-lab hands validation, NO institutional QC, NO ng/cell measured**.
- 协和 是 uptake primary PI (per `wetlab_loi_xiehe_nci60.md`); ORNL + PKU uptake is secondary, contingent on PI capacity.
- The shared technical content (§2) is harmonized across all 3 institutions to enable cross-PI uptake comparison (§4).

## 7. Cross-references

- `molmetal/docs/wetlab_protocols/cellular_uptake_assay.md` (Wf-28 generic ICP-MS protocol, WLP-UP-001)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (companion IC50 analysis pipeline; uptake extension)
- `TODO/pending/wetlab_loi_ornl_protein_metal.md` (LOI-ORNL, uptake secondary)
- `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` (LOI-PKU, uptake secondary)
- `TODO/pending/wetlab_loi_xiehe_nci60.md` (LOI-协和, uptake primary)
- `TODO/pending/wetlab_sop_cytotoxicity_assay.md` (NEW institutional SOP-cytotoxicity)
- `TODO/pending/wetlab_sop_binding_affinity_assay.md` (NEW institutional SOP-binding)
- `TODO/completed/97b_wetlab_prework_docs.md` (v1 overstated close-out — corrected by 97c)
- `TODO/completed/97c_wf74_reclassification.md` (Wf-74 v2 close-out)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule; outreach NOT gated)

## 8. Revision history

| Rev | Date       | Author | Notes |
|-----|------------|--------|-------|
| 1.0 | 2026-09-27 | Wf-74  | Initial institutional-tailored SOP template (3 institutions: 协和 primary / ORNL+PKU secondary); ready for PI adaptation. |
