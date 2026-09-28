# SOP-Cytotoxicity — Institutional-Tailored Cytotoxicity Assay

**Document ID:** WLP-CYTO-INST-001
**Version:** 1.0 (institutional-tailored template; no wet-lab data collected yet)
**Status:** **TEMPLATE — not executed at this revision** (Hugo Lee coordinates PI outreach per `TODO/INDEX.md` §12.3 P3 mitigation)
**Authoring date:** 2026-09-27
**Owner:** Mol-Metal wet-lab coordinator (TBD) + 3 institutional PIs (ORNL / PKU / 协和)

---

## 1. Purpose and Scope

This SOP provides **institution-specific tailoring** of the Wf-28 generic
cytotoxicity assay (`molmetal/docs/wetlab_protocols/cytotoxicity_assay.md`,
WLP-CYTO-001) for the 3 candidate PIs (ORNL / PKU / 协和). The technical
content (8-dose × 3 biological replicates, SRB / MTT / CellTiter-Glo
read-out, 4-parameter logistic fit) is shared across all 3 institutions;
the institutional tailoring covers:

- Cell-line authentication + mycoplasma cadence (per institutional SOP).
- Plate-reader + incubator availability (per institutional equipment).
- Hazardous-waste handling (per institutional metal-waste SOP).
- IRB + biosafety approval (institutional requirement).

The PI adapts the **institution-specific §3 tailoring** to their own
SOP numbering; the §1+§2 technical content remains unchanged from
WLP-CYTO-001.

## 2. Shared technical content (carry from WLP-CYTO-001)

### 2.1 Reagents (shared)

| Reagent | Vendor | Catalogue # |
|---|---|---|
| RPMI-1640 (with glutamine) | Sigma | R8758 |
| FBS, heat-inactivated | Sigma | F2442 |
| Penicillin-Streptomycin | Gibco | 15140-122 |
| Trypsin-EDTA 0.25% | Gibco | 25200-056 |
| Sulforhodamine B (SRB) | Sigma | 230162 |
| Trichloroacetic acid (TCA) 10% | Sigma | T9159 |
| Tris base (unbuffered, pH 10.5) | Sigma | T1503 |
| DMSO, molecular-biology grade | Sigma | D8418 |
| 96-well clear flat-bottom plates | Corning | 3595 |
| Reference drug: cisplatin | Sigma | P4394 |

### 2.2 Dose grid (shared, per L13/L16)

```
concentrations_uM = [0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0]
                  = 10 ** linspace(-1.5, 2.0, 8)
```

### 2.3 QC acceptance (shared)

| Check | Criterion |
|---|---|
| Blank OD | Mean blank OD ≤ 0.05 |
| Vehicle OD | Vehicle OD ≥ 80% of untreated reference OD |
| Cisplatin IC50 (A549) | within 1.0–5.0 µM (historical QC range) |
| Z'-factor | ≥ 0.5 |
| Replicate CV | ≤ 15% across duplicate wells |

### 2.4 Output schema (shared, per `wetlab_ic50_analysis_pipeline.md` §1.1)

CSV columns: `plate_id`, `well_id`, `compound_id`, `concentration_uM`,
`metal`, `cell_line`, `replicate`, `assay`, `time_h`, `raw_readout`,
`background_well`, `vehicle_well`, `positive_ctrl_well`, `qc_flag`.

IC50 fit: 4-parameter log-logistic (Hill equation), `molmetal/metrics/cytotox_eval.py::fit_hill_4pl`.

---

## 3. Institutional tailoring

### §3.1 ORNL tailoring

**Cell-line source**: ATCC + ORNL in-house stock; passage ≤ 15 from thaw.
**Mycoplasma cadence**: monthly (ORNL standard); reject any positive line.
**Plate-reader**: Molecular Devices SpectraMax M5 (OD 540 nm + reference 630 nm) + Beckman Coulter Paradigm (luminescence for CTG).
**CO₂ incubator**: Thermo HERAcell 240i (37 °C, 5% CO₂).
**Hazardous waste**: Pt / Ru / Ir waste per ORNL metal-waste SOP-WST-014 (alkaline-chloride precipitation; collected as hazardous waste).
**Biosafety**: BSL-2; cabinet Class II Type A2.
**Authorship tie-in**: see `TODO/pending/wetlab_loi_ornl_protein_metal.md`.
**Cell panel scope**: **m18 protein-metal binding** primary; cytotoxicity 4-cell (MCF7 / A549 / A2780 / A2780cis) is **secondary**, run in parallel with SPR/ITC if PI agrees.

### §3.2 PKU tailoring

**Cell-line source**: PKU 细胞库 (北京大学医学部); STR 鉴定 + 支原体 (mycoplasma) 检测每月一次。
**Plate-reader**: Molecular Devices SpectraMax 190 (OD 540 nm + reference 630 nm) + PerkinElmer Victor X3 (luminescence)。
**CO₂ incubator**: Thermo Forma 3110 (37 °C, 5% CO₂)。
**细胞培养**: BSL-2; cabinet 苏州安泰 BSC-1300IIA2。
**Hazardous waste**: 顺铂 + Pt 类似物废液按 PKU 危险废物 SOP (碱性-氯化物沉淀) 收集; X-ray 单晶衍射样品按放射性废物 (实际无放射性, 按化学毒性) 收集。
**细胞 panel scope**: **顺铂 IC50 曲线** primary; 4-cell 基准 (MCF7 / A549 / A2780 / A2780cis) 必须; 可扩展 5 个 NCI-60 cell。
**Authorship tie-in**: see `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md`。

### §3.3 协和 tailoring

**Cell-line source**: 协和医院 细胞库 + NCI-60 DTP 标准来源; STR 鉴定 + 支原体检测每月一次 (协和医院 SOP-CL-022)。
**Plate-reader**: Molecular Devices SpectraMax M5 + PerkinElmer EnSight (luminescence + absorbance)。
**CO₂ incubator**: Thermo HERAcell 150i (37 °C, 5% CO₂)。
**细胞培养**: BSL-2; cabinet 协和医院 Class II Type A2 (净化年检半年一次)。
**IRB**: 协和医院 IRB approval 必须 (医院期临床前流程, 4-8 周审批周期); 项目编号 SOP-IRB-PHASE2-001。
**Hazardous waste**: Pt / Pd / Au / Ru / Ir 废液按 协和医院 危险废物 SOP (碱性-氯化物沉淀) 收集; ICP-MS 废液按酸性废液 SOP 单独收集。
**细胞 panel scope**: **NCI-60 9-cell 完整面板** (CCRF-CEM / A549 / HCT-116 / SF-539 / LOX-IMVI / OVCAR-3 / A498 / PC-3 / MCF7) primary; ICP-MS 摄取 ≥2 cell × 3 time-point。
**Authorship tie-in**: see `TODO/pending/wetlab_loi_xiehe_nci60.md`。

---

## 4. Cross-institution harmonization

To ensure in-silico vs wet-lab agreement (`wetlab_ic50_analysis_pipeline.md` §5.2) is comparable across institutions, the following MUST be harmonized:

- **Same compound stock concentration** (5 mM in DMSO; 100× final dilution).
- **Same dose grid** (8-dose half-log; 0.0316–100 µM).
- **Same exposure duration** (72 ± 2 h).
- **Same QC reference** (cisplatin 10 µM as positive control; QC range 1.0–5.0 µM on A549).
- **Same output CSV schema** (per §2.4 above).

The PI is encouraged to use the **Mol-Metal bench harness** (`molmetal/scripts/template_ic50_pipeline.py`) for IC50 fitting to avoid per-PI re-implementation drift.

## 5. Institutional SOP numbering

| Institution | SOP # | Title | Owner |
|---|---|---|---|
| ORNL | ORNL-CYTO-MOLMETAL-001 | Mol-Metal Cytotoxicity Assay (institutional tailoring of WLP-CYTO-INST-001) | ORNL PI |
| PKU | PKU-CYTO-MOLMETAL-001 | Mol-Metal 顺铂类细胞毒测定 (institutional tailoring of WLP-CYTO-INST-001) | PKU PI |
| 协和 | 协和-CYTO-MOLMETAL-001 | Mol-Metal NCI-60 9-cell 面板细胞毒测定 (institutional tailoring of WLP-CYTO-INST-001) | 协和 PI |

The institutional SOP # is assigned by the wet-lab PI on MoU signature; this template uses placeholders.

## 6. Honest framing (party-twenty-spirit §3.5)

- This SOP is a **template** ready for institutional adaptation; **NO wet-lab hands validation, NO institutional QC, NO biological replicate measured**.
- The PI is responsible for adapting §3 to their institutional SOP numbering + biosafety / IRB / hazardous-waste institutional SOPs.
- The shared technical content (§2) is harmonized across all 3 institutions to enable cross-PI in-silico vs wet-lab agreement analysis (`wetlab_ic50_analysis_pipeline.md` §5.2).

## 7. Cross-references

- `molmetal/docs/wetlab_protocols/cytotoxicity_assay.md` (Wf-28 generic protocol, WLP-CYTO-001)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (companion IC50 analysis pipeline)
- `TODO/pending/wetlab_loi_ornl_protein_metal.md` (LOI-ORNL, m18 binding primary)
- `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` (LOI-PKU, cisplatin IC50 curves)
- `TODO/pending/wetlab_loi_xiehe_nci60.md` (LOI-协和, NCI-60 9-cell panel)
- `TODO/pending/wetlab_sop_binding_affinity_assay.md` (NEW institutional SOP-binding)
- `TODO/pending/wetlab_sop_cellular_uptake_assay.md` (NEW institutional SOP-uptake)
- `TODO/completed/97b_wetlab_prework_docs.md` (v1 overstated close-out — corrected by 97c)
- `TODO/completed/97c_wf74_reclassification.md` (Wf-74 v2 close-out)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule; outreach NOT gated)

## 8. Revision history

| Rev | Date       | Author | Notes |
|-----|------------|--------|-------|
| 1.0 | 2026-09-27 | Wf-74  | Initial institutional-tailored SOP template (3 institutions: ORNL / PKU / 协和); ready for PI adaptation. |
