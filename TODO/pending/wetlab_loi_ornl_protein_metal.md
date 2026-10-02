# LOI-ORNL — Letter of Intent: Protein-Metal Binding Validation

**Document ID:** LOI-ORNL-2026-09
**Version:** 1.0 (draft, ready for review)
**Status:** **TEMPLATE_NOT_NEEDED_FOR_TIER2** — see AGENTS.md §5.4 + §10 + 2026-09-30 user reinforcement "搞清楚我们不用做湿实验，这一点我已经明确了好多遍了". Tier-2 journal acceptance does not require wet-lab data. Retained as **historical artifact only**; any agent action attempting to send this LOI must REFUSE per AGENTS.md §10.3.
**(Historical note: DRAFT — not sent at this revision)** (Hugo Lee coordinates the cold-email per `TODO/INDEX.md` §12.3 P3 mitigation; no action will be taken — wet-lab DEPRECATED per AGENTS.md §5.4 + §10)
**Target PI:** Oak Ridge National Laboratory (ORNL) protein-metal binding group
**Authoring date:** 2026-09-27
**Owner:** Hugo Lee (Mol-Metal project lead) — `hugo.lee@mol-metal.org`

---

## 1. Purpose

This Letter of Intent (LOI) proposes a short-term validation collaboration
between the Mol-Metal open-source project and an ORNL protein-metal
binding group. The collaboration will validate the top-ranked
metallodrug candidates from the Mol-Metal R18 release against a panel
of metalloprotein targets using SPR (Biacore) and ITC (MicroCal PEAQ).

## 2. Background — what Mol-Metal brings

Mol-Metal is an open-source metallodrug design pipeline (`mol-metal.org`,
GitHub `mol-metal`). The R18 release (2026-09) ships:

- **Click-chemistry metallodrug generator** (`molmetal/molmetal_lam/lam_chem/pt_click_compat.py`).
- **Lambda-combinator search** with metal-aware reward shaping (`molmetal/molmetal_lam/search_alg/proof_search.py`, `molmetal/molmetal_lam/reward/learned_shaping.py`).
- **Coupled flow-matching backbone** (`molmetal/adapters/flow_matching_lipman/`).
- **Pre-computed top-20 wet-lab candidates** for R20-P6 (Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11), see `molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv`.
- **In-silico prediction panel**: predicted IC50 (with 80% CI), logP, redox potential, HSA K_d, cellular uptake (pg/cell). D-MPNN pIC50 spread 1.7 log units across top-20.
- **Companion IC50 analysis pipeline** (`TODO/pending/wetlab_ic50_analysis_pipeline.md` + `molmetal/scripts/template_ic50_pipeline.py`) ready to consume wet-lab CSV the moment it ships.

Open datasets: PlatinAI (CC0), tmQM (CC-BY), MetalCytoToxDB Ru subset (CC-BY-SA).

## 3. What ORNL brings

ORNL has a strong protein-metal binding group with:

- BSL-2 wet-lab facility and trained operators.
- SPR (Biacore X100 / T200) and ITC (MicroCal PEAQ-ITC) instruments.
- AMBER force-field license + high-performance computing cluster access.
- Metalloprotein expression and purification pipeline.
- Track record in cisplatin-DNA lesion binding + GST / metallothionein / p53 DBD binding studies.

## 4. Proposed collaboration (3-phase, 12-month)

| Phase | Duration | Mol-Metal deliverable | ORNL deliverable |
|-------|----------|------------------------|-------------------|
| **Phase 1** | 0–2 mo | 5–10 ranked Pt / Ru / Ir candidates (3D structures, in-silico IC50, logP, redox, predicted K_d, predicted uptake); CPU analysis pipeline (template_ic50_pipeline.py) | Acknowledge fit; counter-sign MoU; protein expression + purification; SPR + ITC setup with reference compound |
| **Phase 2** | 2–6 mo | Compound aliquots shipped (or synthesis recipe if MoU allows in-house resynthesis); updated ranking per round | SPR + ITC on top-10 binders (per-metal cohort): Pt 2 + Pd 3 + Au 1 + Ru 3 + Ir 11 = 20 compounds × 3 proteins (GST / cisplatin-DNA lesion / p53 DBD) = 60 K_d measurements; ±std across ≥3 technical replicates |
| **Phase 3** | 6–12 mo | Joint manuscript draft; updated Mol-Metal model retrained on measured K_d | Optional X-ray / NMR speciation on top-3 binders; repeat SPR + ITC on next round of candidates |

60-day review cadence: Day 0/7/21 (email follow-up) → Day 28 (intro Zoom) → Day 60 (MoU signed) → Month 2/6/9 (phase reviews).

## 5. Authorship

- Phase-2 validation that surfaces ≥1 sub-µM K_d candidate for a clinically-relevant target (e.g. p53 DBD rescue, metallothionein displacement) qualifies the ORNL PI for **co-corresponding authorship**.
- Phase-2 validation alone (without a sub-µM hit) qualifies for a **mid-author position** with the data-validation section of the paper.
- Both teams agree on author order in writing before submission.
- Pre-print / conference abstracts: dry-lab PI and ORNL PI present jointly; first authorship on the abstract follows experimental contribution to the underlying figure.

## 6. Funding and equipment

- **Mol-Metal side**: software, compute, in-silico predictions, manuscript preparation. No wet-lab budget.
- **ORNL side**: protein expression + purification reagents, SPR + ITC consumables + instrument time, operator hours.
- **Joint grant applications encouraged** (NIH R01 / DOE BER / NSF). Submit under ORNL as fiscal host. Mol-Metal team can supply a letter of support and a 1-page research plan on request.
- **No overhead or indirect-cost transfers** are implied.

## 7. Target proteins (initial SPR + ITC panel)

| Target | Rationale | Vendor / source | Expected binding partner |
|---|---|---|---|
| **Glutathione-S-transferase (GST)** | Cisplatin detoxification pathway | Sigma / recombinant | Cisplatin + Pt-analogs |
| **Cisplatin-DNA lesion (1,2-d(GpG) intrastrand crosslink)** | Canonical cisplatin binding mode | Midland Certified Reagent | Cisplatin-class + Pt-analogs |
| **Recombinant p53 DNA-binding domain (p53 DBD)** | Tumor-suppressor rescue | Addgene / recombinant | Pt / Ru metallodrugs with DNA-affinity |
| **Metallothionein (MT-2A)** | Heavy-metal detoxification | Sigma / recombinant | Pt / Pd / Au |

Optional Phase-3 extension: **transferrin** (serum protein binding), **HSA** (pharmacokinetic distribution).

## 8. Expected outputs

- Per-compound × per-protein **K_d ± std** (n=3 technical replicates; SPR + ITC cross-validation).
- Per-compound **ΔH + ΔS + n (stoichiometry)** from ITC thermograms.
- Optional **kon / koff** from SPR kinetics fit (χ² ≤ 5).
- All raw SPR / ITC files uploaded to `data/wetlab/binding/<run_id>/` with checksums.
- JSON output conforming to `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` schema.

## 9. Timeline + deliverables

| Month | Milestone | Deliverable |
|---|---|---|
| 0 | MoU signed; protein expression initiated | MoU + protein prep |
| 1 | SPR + ITC calibration; reference compound (cisplatin) titrated | QC report |
| 2 | Top-10 Pt / Ru / Ir aliquots shipped from Mol-Metal | Compound arrival + QC |
| 3 | First SPR + ITC round (top-10 × 4 proteins) | 40 K_d measurements |
| 4–5 | Second SPR + ITC round (next 10 × 4 proteins) | 40 K_d measurements |
| 6 | Phase-2 review meeting | K_d summary report |
| 6–9 | Phase-3 speciation + next-round candidates | Optional X-ray / NMR |
| 9–12 | Manuscript draft + retrain model on measured K_d | Joint paper |

## 10. Risk and mitigation

- **AMBER license**: depends on ORNL cluster access. Mitigation: Mol-Metal can pre-compute MM-GBSA binding free energies on publicly-available GPUs as backup.
- **Cisplatin-DNA lesion synthesis**: commercial supply (Midland Certified Reagent) is reliable; 4-week lead time.
- **SPR regeneration conditions**: Pt compounds can be sticky on CM5 chips. Mitigation: use carboxymethylated dextran with low-density immobilization + 10 mM NaOH regeneration pulses.

## 11. Confidentiality and data sharing

- All wet-lab data shared under the data-sharing MoU template (`molmetal/docs/wetlab_protocols/data_sharing_agreement.md`).
- Joint pre-print embargo until both PIs approve submission.
- All R18 / R20 Mol-Metal code is open-source (CC-BY-4.0); the wet-lab K_d + ΔH + ΔS data becomes part of the open Mol-Metal dataset on publication.

## 12. Next steps

1. Hugo Lee sends this LOI via cold-email to ORNL PI (template at `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` §3).
2. Day +7 light follow-up; Day +21 second follow-up.
3. Day +28 intro Zoom (30 min) — confirm fit, protein availability, instrument access.
4. Day +60 MoU signed; Phase 1 launches.

## 13. Honest framing (party-twenty-spirit §3.5)

- This LOI is a **draft** ready for Hugo Lee's review; **0 LOI sent** at this revision.
- **No wet-lab K_d data MEASURED today**; in-silico K_d is RDKit-fingerprint RandomForestRegressor surrogate, NOT the production chemprop D-MPNN.
- The headline `Pearson r ≥ 0.7` between D-MPNN-predicted K_d and ORNL-measured K_d cannot MEASURE until ≥1 PI replies + MoU + Phase 1 launches + 6-month timeline completes (Q1 2027 earliest).
- Paper §5 wet-lab validation section 🚫 DEFERRED per TODO-84 paper rule; needs explicit user authorization.

## 14. Cross-references

- `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (Wf-28 generic email template, WLP-OUT-001)
- `molmetal/docs/wetlab_protocols/binding_affinity_assay.md` (Wf-28 generic ITC protocol, WLP-BIND-001)
- `molmetal/docs/wetlab_protocols/data_sharing_agreement.md` (Wf-28 data-sharing MoU template)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (companion IC50 analysis pipeline)
- `TODO/pending/wetlab_sop_binding_affinity_assay.md` (NEW institutional-tailored SOP)
- `TODO/completed/97b_wetlab_prework_docs.md` (v1 overstated close-out — corrected by 97c)
- `TODO/completed/97c_wf74_reclassification.md` (this Wf-74 v2 close-out)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 in-silico dry-run layer ship)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule; outreach NOT gated)

## 15. Revision history

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 1.0 | 2026-09-27 | Wf-74  | Initial institutional-tailored LOI draft (ready for Hugo Lee review; **0 sent**). |
