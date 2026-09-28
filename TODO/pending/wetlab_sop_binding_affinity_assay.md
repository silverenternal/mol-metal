# SOP-Binding — Institutional-Tailored Binding Affinity Assay (SPR / ITC)

**Document ID:** WLP-BIND-INST-001
**Version:** 1.0 (institutional-tailored template; no wet-lab data collected yet)
**Status:** **TEMPLATE — not executed at this revision** (Hugo Lee coordinates PI outreach per `TODO/INDEX.md` §12.3 P3 mitigation)
**Authoring date:** 2026-09-27
**Owner:** Mol-Metal wet-lab coordinator (TBD) + 3 institutional PIs (ORNL / PKU / 协和)

---

## 1. Purpose and Scope

This SOP provides **institution-specific tailoring** of the Wf-28 generic
binding affinity assay (`molmetal/docs/wetlab_protocols/binding_affinity_assay.md`,
WLP-BIND-001) for the 3 candidate PIs (ORNL / PKU / 协和). The technical
content (SPR + ITC, kinetics fit + thermogram analysis) is shared across
all 3 institutions; the institutional tailoring covers:

- SPR + ITC instrument availability (per institutional core facility).
- Protein target panel (per PI focus: ORNL protein-metal binding; PKU cisplatin-DNA; 协和 HSA / transferrin serum protein).
- Buffer + immobilization chemistry (per instrument compatibility).
- Cross-validation protocol (SPR + ITC cross-check on top-3 binders).

The PI adapts the **institution-specific §3 tailoring** to their own
SOP numbering; the §1+§2 technical content remains unchanged from
WLP-BIND-001.

## 2. Shared technical content (carry from WLP-BIND-001)

### 2.1 Proteins (shared initial panel)

| Target | Vendor / source | Notes |
|---|---|---|
| **Glutathione-S-transferase (GST)** | Sigma / recombinant | Cisplatin detoxification |
| **Cisplatin-DNA lesion** (1,2-d(GpG) intrastrand crosslink) | Midland Certified Reagent | Canonical cisplatin binding |
| **Recombinant p53 DNA-binding domain** | Addgene / recombinant | Tumor-suppressor rescue |
| **Metallothionein (MT-2A)** | Sigma / recombinant | Heavy-metal detoxification |
| (Optional) **HSA** | Sigma | Pharmacokinetic distribution |
| (Optional) **Transferrin** | Sigma | Serum protein binding |

### 2.2 Buffers (shared)

- **SPR running buffer**: HBS-EP+ (10 mM HEPES pH 7.4, 150 mM NaCl, 3 mM EDTA, 0.005% P-20).
- **ITC buffer**: 20 mM sodium phosphate, 150 mM NaCl, pH 7.4 (must be the same buffer for protein + ligand dilution).

### 2.3 SPR setup (shared)

- Chip: CM5 (carboxymethylated dextran).
- Immobilization: amine coupling (EDC/NHS); low-density target immobilization (~1000-3000 RU for kinetics-grade).
- Regeneration scouting: 10 mM NaOH pulse; optimize for each protein-ligand pair.
- Multi-cycle kinetics: 5-7 concentrations per compound; flow rate 30 µL/min; association 120 s + dissociation 300 s.

### 2.4 ITC setup (shared)

- Instrument: MicroCal PEAQ-ITC (200 µL cell + 40 µL syringe).
- Protein in cell: 30-80 µM.
- Ligand in syringe: 500-1500 µM (≥10× cell concentration).
- 19 injections (0.4 µL first + 18 × 2.0 µL); 25 °C; 750 rpm stirring.
- Buffer-into-buffer blank titration mandatory.

### 2.5 QC acceptance (shared)

| Check | Criterion |
|---|---|
| **SPR Rmax** | ≤ 100 RU (kinetics fit valid) |
| **SPR kinetics fit χ²** | ≤ 5 |
| **SPR KD reproducibility** | ≥ 1e-10 M (sub-nM) reproducible |
| **ITC N (stoichiometry)** | 0.8 ≤ N ≤ 1.2 (single binding site) |
| **ITC buffer match** | heats of dilution ≤ 0.05 µW |

### 2.6 Output schema (shared)

Per (compound × protein) record:
- SPR: `kon` (1/Ms), `koff` (1/s), `KD` (M), `Rmax` (RU), `chi2`.
- ITC: `KD` (M), `ΔH` (kcal/mol), `ΔS` (cal/mol/K), `N` (stoichiometry), `T` (°C).

JSON output conforming to `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` schema (with `KD_M` field per Wf-74 P3 extension).

---

## 3. Institutional tailoring

### §3.1 ORNL tailoring (primary)

**SPR instrument**: Biacore X100 or T200 (ORNL Biacore Core Facility).
**ITC instrument**: MicroCal PEAQ-ITC (ORNL Biacore Core Facility).
**Protein panel scope**:
- **GST** (primary; cisplatin detoxification pathway).
- **Cisplatin-DNA lesion 1,2-d(GpG)** (primary; canonical cisplatin binding).
- **p53 DBD** (primary; tumor-suppressor rescue).
- **MT-2A** (primary; heavy-metal detoxification).
- Optional: HSA + transferrin (Phase 3 extension).
**SPR chip**: CM5 (low-density amine coupling; ~2000 RU target immobilization).
**Regeneration scouting**: ORNL uses 10 mM NaOH + 1 M NaCl double-pulse for sticky Pt compounds.
**ITC cell loading**: protein at 50 µM; syringe at 1000 µM (20× cell).
**AMBER compute**: ORNL cluster access enables MM-GBSA per-compound binding free energy pre-screen; Mol-Metal team will share the pre-screen script (`molmetal/scripts/ammonia_gbsa.py` reserved).
**Authorship tie-in**: see `TODO/pending/wetlab_loi_ornl_protein_metal.md`.

### §3.2 PKU tailoring (secondary)

**SPR instrument**: Biacore T200 (PKU 医药卫生分析中心) — 若可用; 否则只用 ITC。
**ITC instrument**: MicroCal PEAQ-ITC 或 Nano-ITC (PKU 药学院)。
**Protein panel scope**:
- **Cisplatin-DNA lesion 1,2-d(GpG)** (primary; cisplatin 机制研究)。
- **p53 DBD** (primary; 抑癌 rescue)。
- 可选: GST, MT-2A。
**SPR chip**: CM5; cisplatin-DNA lesion 用 streptavidin chip + biotinylated DNA 固定。
**Authorship tie-in**: see `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md`。
**Note**: PKU 主要 focus 是 IC50 曲线 (顺铂类); binding 实验是 secondary, 仅在 PKU 课题组 capacity 可用时执行。

### §3.3 协和 tailoring (secondary)

**SPR instrument**: 协和医院 无 SPR; 若需要, 借用外部 facility 或外包 (北京蛋白质组研究中心)。
**ITC instrument**: MicroCal PEAQ-ITC (协和医院中心实验室)。
**Protein panel scope** (可选, 若 capacity 允许):
- **HSA** (primary; pharmacokinetic distribution, 临床相关)。
- **Transferrin** (primary; serum protein binding, 临床相关)。
- 可选: cisplatin-DNA lesion (与 PKU 分工)。
**SPR chip**: 不适用。
**ITC cell loading**: HSA at 30 µM; drug at 750 µM (25× cell)。
**Authorship tie-in**: see `TODO/pending/wetlab_loi_xiehe_nci60.md`。
**Note**: 协和 主要 focus 是 NCI-60 9-cell IC50 + ICP-MS 摄取; binding 实验是 tertiary, 仅在 协和 PI 同意且 capacity 可用时执行。

---

## 4. Cross-validation protocol (per PI agreement)

For top-3 binders identified by the PI's primary assay:

1. **SPR + ITC cross-check** — same compound × same protein measured on both instruments; report `KD_SPR / KD_ITC` ratio (should be within 0.5-2×).
2. **Replicate reproducibility** — 3 independent titrations; report `KD_mean ± std` and CV.
3. **Cross-PI validation** (optional) — top-3 binders from ORNL shipped to 协和 for HSA / transferrin secondary validation; results shared under data-sharing MoU.

## 5. Institutional SOP numbering

| Institution | SOP # | Title | Owner |
|---|---|---|---|
| ORNL | ORNL-BIND-MOLMETAL-001 | Mol-Metal Protein-Metal Binding Assay (institutional tailoring of WLP-BIND-INST-001) | ORNL PI |
| PKU | PKU-BIND-MOLMETAL-001 | Mol-Metal 顺铂-DNA 结合实验 (institutional tailoring of WLP-BIND-INST-001) | PKU PI |
| 协和 | 协和-BIND-MOLMETAL-001 | Mol-Metal HSA / Transferrin 血清蛋白结合 (institutional tailoring of WLP-BIND-INST-001) | 协和 PI |

## 6. Honest framing (party-twenty-spirit §3.5)

- This SOP is a **template** ready for institutional adaptation; **NO wet-lab hands validation, NO institutional QC, NO K_d measured**.
- ORNL is the primary binding PI (per `wetlab_loi_ornl_protein_metal.md` §7); PKU + 协和 binding is secondary/tertiary, contingent on PI capacity.
- The shared technical content (§2) is harmonized across all 3 institutions to enable cross-PI K_d cross-check (§4).

## 7. Cross-references

- `molmetal/docs/wetlab_protocols/binding_affinity_assay.md` (Wf-28 generic ITC protocol, WLP-BIND-001)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (companion IC50 analysis pipeline; K_d extension)
- `TODO/pending/wetlab_loi_ornl_protein_metal.md` (LOI-ORNL, primary binding PI)
- `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` (LOI-PKU, secondary)
- `TODO/pending/wetlab_loi_xiehe_nci60.md` (LOI-协和, tertiary)
- `TODO/pending/wetlab_sop_cytotoxicity_assay.md` (NEW institutional SOP-cytotoxicity)
- `TODO/pending/wetlab_sop_cellular_uptake_assay.md` (NEW institutional SOP-uptake)
- `TODO/completed/97b_wetlab_prework_docs.md` (v1 overstated close-out — corrected by 97c)
- `TODO/completed/97c_wf74_reclassification.md` (Wf-74 v2 close-out)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule; outreach NOT gated)

## 8. Revision history

| Rev | Date       | Author | Notes |
|-----|------------|--------|-------|
| 1.0 | 2026-09-27 | Wf-74  | Initial institutional-tailored SOP template (3 institutions: ORNL primary / PKU secondary / 协和 tertiary); ready for PI adaptation. |
