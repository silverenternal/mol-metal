# Data Sharing Agreement — Wet-Lab Validation Datasets

**Document ID:** WLP-DSA-001
**Version:** 1.0
**Status:** Template (no wet-lab data has been generated under this
agreement yet)
**Owner:** Mol-Metal wet-lab coordinator (TBD)
**Last reviewed:** 2026-09-26

---

## 1. Purpose

This Data Sharing Agreement (DSA) governs the release, licensing,
attribution, and anonymization of all wet-lab validation data generated
under a Mol-Metal / wet-lab collaboration. It is referenced from
`collaborator_outreach_template.md` (Phase-1 deliverable: counter-signed
MoU) and applies in addition to the wet-lab institution’s own IRB / IBC
policies.

## 2. Pre-existing Datasets (provided by Mol-Metal)

These datasets are shipped with the R18 release and are subject to
their original licenses. Re-distribution downstream must preserve the
attribution string.

| Dataset        | Contents                                                          | License    | Citation / source                           |
|----------------|-------------------------------------------------------------------|------------|---------------------------------------------|
| PlatinAI       | 4 007 Pt complexes with curated SMILES, structures, IC50 targets | **CC0 1.0** (public domain) | https://github.com/mol-metal/platinai |
| tmQM           | 86 665 transition-metal complexes, geometries, properties        | **CC-BY 4.0**                | Balcells & Skyner, https://github.com/aimat-lab/tmQM |
| MetalCytoToxDB | Ru-subset cytotoxicity measurements, D-MPNN multi-task, scaffold-split | **CC-BY-SA 4.0**         | mol-metal.org/MetalCytoToxDB                 |

Attribution strings for each dataset are versioned in
`molmetal/data/licenses.csv` and must be copied into any downstream
release that incorporates the corresponding rows.

## 3. New Wet-Lab Data — Release Timeline

Wet-lab data generated under a Mol-Metal collaboration fall into three
release tiers:

### 3.1 Tier 1 — Pre-print / project-internal

- **Window:** Generation → manuscript submission to a peer-reviewed
  journal.
- **Access:** Both teams (dry + wet) only.
- **Embargo:** Hard embargo. Data are not shared outside the named
  authors and their direct collaborators.

### 3.2 Tier 2 — Coincident with paper acceptance

- **Window:** On the date the joint paper is accepted for publication.
- **Access:** Released under **CC-BY 4.0** to the community, deposited
  in `data/wetlab/<assay>/<run_id>/` of the Mol-Metal repository with
  checksums and a `README.md` describing the run.
- **Embargo:** None after acceptance; reviewer access under the journal
  embargo counts as Tier 1.

### 3.3 Tier 3 — Public long-term archive (Zenodo)

- **Window:** **6 months after paper acceptance.**
- **Access:** DOI-issued deposit on Zenodo with the canonical
  `data/wetlab/` snapshot. License: **CC-BY 4.0**.
- **Rationale:** A 6-month buffer allows the wet-lab team to publish
  follow-up biological interpretation papers without preemptive
  competition, while still meeting the FAIR-data commitments of most
  funders and the Mol-Metal charter.

If the parties agree in advance (e.g. for thesis-driven timelines) the
Tier 3 release may be brought forward, but **never earlier than the
paper-acceptance date**.

## 4. Anonymization

### 4.1 Personal anonymization

- Operator initials are stored in the run metadata as opaque strings
  (e.g. `op_a3f9`). A mapping table
  (`logs/wetlab_operators_anonymized.csv`) is held only by the wet-lab
  PI and is **not** released.
- Email addresses, ORCID iDs, and signatures on MoUs are stripped from
  any public release.

### 4.2 Institutional anonymization

- Wet-lab institution names are released alongside the joint paper
  (consistent with normal authorship practice).
- The Tier 3 release does **not** strip the institution name; the
  community expects to know which lab generated the data.
- Internal cost codes, grant numbers, and student names are stripped
  from the public release.

### 4.3 Compound / cell-line anonymization

- Compound IDs use the Mol-Metal canonical scheme
  (`MC-<run>-<index>`), never the wet-lab’s internal lab notebook ID.
- Cell-line names are released as-is (they are public knowledge), but
  passage numbers and mycoplasma-test dates are aggregated to "passage
  ≤ 15 from thaw" and "tested negative, year-month" to avoid
  re-identification of specific culture batches.

## 5. Data Formats and Schemas

All wet-lab data must conform to the canonical schemas defined in:

- `cytotoxicity_assay.md` §10 — `cytotox_<compound_id>_<cell_line>.csv`
- `binding_affinity_assay.md` §8.3 — `binding_<compound_id>_<target>.json`
- `cellular_uptake_assay.md` §9 — `uptake_<compound_id>_<cell_line>.csv`

Raw plate-reader / ITC / ICP-MS exports are retained under
`data/wetlab/<assay>/<run_id>/raw/` with SHA-256 checksums.

## 6. Pre-existing Third-Party Data — Re-licensing

If the wet-lab partner contributes pre-existing data that was generated
under a **more restrictive** license than CC-BY 4.0, that data:

1. Must be flagged in the MoU at signing time.
2. Are excluded from the Tier 2/3 release unless re-licensed in writing
   by the original rights-holder.
3. May still be used internally for the joint paper with proper
   attribution.

## 7. Disputes and Withdrawal

- Either party may request withdrawal of a specific compound’s data
  prior to Tier 2 release if there is a substantive scientific or
  legal concern (e.g. contaminated batch, IRB issue). Withdrawn data
  are excluded from the release; the reason is logged.
- After Tier 2 release, withdrawal is not possible — the data are
  CC-BY 4.0 and may have been mirrored downstream.
- Disputes are resolved by joint PI discussion; failing that, by an
  independent mediator mutually agreed upon.

## 8. References

- Mol-Metal open-data charter (current revision).
- Creative Commons CC0 1.0, CC-BY 4.0, CC-BY-SA 4.0 — legal codes at
  creativecommons.org.
- FAIR Data Principles (Wilkinson et al., *Sci. Data* 3, 160018, 2016).

## 9. Revision History

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 0.1 | 2026-09-26 | Wf-28  | Initial template (no DSA signed yet).              |
