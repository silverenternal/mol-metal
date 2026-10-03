# Cytotoxicity Assay — Standard NCI-60 72-Hour SRB Protocol

**Document ID:** WLP-CYTO-001
**Version:** 1.0
**Status:** Template (no wet-lab data has been collected)
**Owner:** Mol-Metal wet-lab coordinator (TBD)
**Last reviewed:** 2026-09-26

---

## 1. Purpose and Scope

This protocol describes the standardized Sulforhodamine B (SRB) colorimetric
cytotoxicity assay used to determine the in-vitro growth-inhibitory activity of
metallodrug candidates (Pt, Ru, Ir, Os, Rh complexes) against a reference
panel of human tumour and non-tumour cell lines.

The protocol is adapted from the NCI-60 Developmental Therapeutics Program
72-hour drug exposure protocol (Skehan et al., *J. Natl. Cancer Inst.*, 1990)
and is intended to be executed by trained personnel in a BSL-2 wet-lab
facility. The companion analysis pipeline (`molmetal/metrics/cytotox_eval.py`)
expects dose-response tables in the schema described in §10.

## 2. Safety and Compliance

- All work involving live cells must be performed in a certified Class II
  biosafety cabinet under BSL-2 containment. See `README.md` for facility-wide
  PPE and waste-handling requirements.
- Metallodrugs are handled as acutely toxic and potentially carcinogenic.
  Nitrile gloves, lab coat, and safety goggles are required at all times.
- DMSO stocks are flammable and hygroscopic; store in a flammable-cabinet
  desiccator.

## 3. Materials and Reagents

| Reagent / Consumable                | Vendor (suggested) | Catalogue #   | Notes                               |
|-------------------------------------|--------------------|---------------|-------------------------------------|
| RPMI-1640 medium (with glutamine)   | Sigma / Gibco      | R8758 / 11875| Supplemented as in §3.1             |
| Foetal Bovine Serum (FBS), heat-inactivated | Sigma       | F2442         | Lot-tested for clonal growth        |
| Penicillin–Streptomycin (10 000 U/mL) | Gibco            | 15140-122     | Final 1×                             |
| Trypsin-EDTA (0.25 %), phenol red   | Gibco              | 25200-056     |                                     |
| Sulforhodamine B (SRB)              | Sigma              | 230162        | 0.4 % (w/v) in 1 % acetic acid     |
| Trichloroacetic acid (TCA), 10 %    | Sigma              | T9159         | Pre-chilled to 4 °C                 |
| Tris base (unbuffered, pH 10.5)     | Sigma              | T1503         | 10 mM for SRB solubilization        |
| Acetic acid (glacial)               | Sigma              | A6283         |                                     |
| DMSO, molecular-biology grade       | Sigma              | D8418         | For compound stock                  |
| 96-well clear flat-bottom plates    | Corning            | 3595          | Tissue-culture treated              |
| Reference drug: cisplatin           | Sigma              | P4394         | Positive control                    |

### 3.1 Complete growth medium

RPMI-1640 supplemented with 10 % FBS and 1 % penicillin–streptomycin.
Filter-sterilize through a 0.22 µm vacuum filter. Store at 4 °C for up to
4 weeks; pre-warm to 37 °C before use.

## 4. Equipment

- Class II Type A2 biosafety cabinet
- Humidified CO₂ incubator (37 °C, 5 % CO₂)
- Inverted phase-contrast microscope (4× and 10× objectives)
- Hemocytometer or automated cell counter (e.g. Countess II)
- Multichannel pipettes (8- and 12-channel, 10–300 µL)
- Single-channel pipettes (2, 20, 200, 1 000 µL)
- Plate-reading spectrophotometer with 540 nm filter (reference 630 nm)
- Vortex mixer, benchtop centrifuge with plate rotor
- Analytical balance (readability 0.1 mg)
- Ultrasonic bath (for compound dissolution)

## 5. Cell Line Maintenance

1. Maintain cultures in T25 or T75 flasks in complete RPMI-1640 at 37 °C,
   5 % CO₂.
2. Sub-culture at 70–80 % confluence (typically every 2–3 days). Use
   trypsinization; neutralize 1:1 with complete medium.
3. Maintain a master stock in liquid nitrogen (10 % DMSO + 90 % FBS) and
   passage cultures no more than 15 passages from thaw.
4. Mycoplasma-test cultures monthly; reject any positive line.

## 6. Compound Stock Preparation

1. Weigh 1–5 mg of the metallodrug on an analytical balance; record mass to
   the nearest 0.1 mg.
2. Dissolve in molecular-biology-grade DMSO to give a **5 mM stock**.
   Sonicate 5 min at room temperature if dissolution is incomplete.
3. Aliquot 20 µL into amber PCR tubes; store at −20 °C for ≤ 1 month,
   −80 °C for ≤ 6 months. Avoid repeated freeze–thaw (>3 cycles).
4. On the day of dosing, thaw one aliquot; vortex 30 s; dilute in complete
   medium immediately before addition. **Final DMSO concentration per well
   must not exceed 0.5 % v/v** (validated against vehicle-only controls).

## 7. Plate Layout and Dosing

A standardized 96-well layout is used to maximize statistical power while
conserving plate real-estate:

```
        col1   col2   col3   col4   col5   col6   col7   col8   col9   col10  col11  col12
row A   BLANK   BLANK   BLANK   BLANK   BLANK   BLANK   BLANK   BLANK   BLANK   BLANK   BLANK   BLANK
row B   veh    veh    veh    veh    veh    veh    veh    veh    veh    veh    veh    veh
row C   d1     d1     d2     d2     d3     d3     d4     d4     d5     d5     d6     d6
row D   d1     d1     d2     d2     d3     d3     d4     d4     d5     d5     d6     d6
row E   d7     d7     [compound at 100 µM in cols 3-12] ...
row F   ...                                                                    cisPt  cisPt
row G   ...
```

- **Rows A**: medium-only blanks (background subtraction).
- **Rows B**: vehicle (0.5 % DMSO) controls.
- **Rows C–H**: 8 serial half-log doses of test compound
  (100, 31.6, 10, 3.16, 1, 0.316, 0.1, 0.0316 µM).
- **Columns 11–12**: positive control (cisplatin) at 8 matched doses.

### 7.1 Cell seeding (Day 0)

1. Harvest log-phase cells; count and verify ≥ 95 % by trypan blue.
2. Resuspend at **5 × 10⁴ cells/mL** in complete medium.
3. Seed **100 µL per well = 5 000 cells/well** using a multichannel pipette.
4. Allow cells to attach for 24 h at 37 °C, 5 % CO₂ before drug addition.

### 7.2 Drug treatment (Day 1)

1. Prepare a 2× intermediate dosing plate by serially diluting the 5 mM
   stock in complete medium (final DMSO 0.5 %).
2. Add **100 µL of 2× drug solution** to each treatment well (existing
   100 µL is now at 1× final).
3. Return plates to the incubator for **72 ± 2 h**.

## 8. SRB Staining (Day 4)

1. Add 50 µL of cold 10 % TCA per well. Incubate 60 min at 4 °C to fix cells.
2. Decant; wash 5× with slow-running tap water; air-dry plate upside-down.
3. Add 100 µL of 0.4 % SRB (w/v in 1 % acetic acid). Stain 30 min at RT.
4. Decant SRB; wash 4× with 1 % acetic acid to remove unbound dye.
5. Air-dry plate completely (overnight at RT or 1 h at 60 °C).
6. Solubilize bound dye with **200 µL of 10 mM Tris (unbuffered, pH 10.5)**;
   shake 10 min on orbital shaker.
7. Read absorbance at **540 nm** (reference 630 nm) on a plate reader.

## 9. Acceptance Criteria

A plate is accepted if **all** of the following hold:

| Check | Criterion |
|-------|-----------|
| Blank OD | Mean blank OD ≤ 0.05 |
| Vehicle OD | Vehicle OD ≥ 80 % of untreated reference OD |
| Cisplatin IC50 (A549) | within 1.0–5.0 µM (historical QC range) |
| Z'-factor (vehicle vs blank) | ≥ 0.5 |
| Replicate CV | ≤ 15 % across duplicate wells |

## 10. Data Output Schema

The raw plate-reader export must be reduced to the following long-format
table and saved as `cytotox_<compound_id>_<cell_line>.csv`:

| column         | type   | example                       |
|----------------|--------|-------------------------------|
| compound_id    | str    | "MC-1024"                     |
| cell_line      | str    | "A549"                        |
| concentration_uM | float | 0.0316                        |
| replicate      | int    | 1                             |
| od_540         | float  | 0.412                         |
| od_630         | float  | 0.038                         |
| vehicle_od     | float  | 1.215                         |
| blank_od       | float  | 0.041                         |
| pct_growth     | float  | 33.2                          |
| plate_id       | str    | "2026-09-26-PT-002"           |
| operator       | str    | "operator_initials"           |

`pct_growth` is computed as:

    pct_growth = 100 * (od_treated - od_blank) / (od_vehicle - od_blank)

This schema is the canonical input for the downstream 4-parameter logistic
fit described in §11.

## 11. IC50 Determination (4-Parameter Logistic Fit)

The dose-response data are fit to the Hill equation:

    response(d) = bottom + (top - bottom) / (1 + (d / IC50) ** n)

where `d` is concentration (µM), `n` is the Hill slope, and `top` /
`bottom` are the upper and lower asymptotes. The fit is performed by
`molmetal/metrics/cytotox_eval.py::fit_hill_4pl`, which uses
`scipy.optimize.curve_fit` with the following constraints:

- `top ∈ [80, 120]`
- `bottom ∈ [−20, 20]`
- `n ∈ [0.5, 8]`
- `IC50 > 0` (log-parameterized)

Report:

- **IC50** with 95 % confidence interval from the covariance matrix.
- **Hill slope** (`n`) as a quality flag (n < 1 may indicate partial
  saturation artefacts).
- **GI50** (concentration for 50 % growth inhibition relative to vehicle)
  if `bottom < 0`.

## 12. References

- Skehan P. et al., *J. Natl. Cancer Inst.* 82, 1107 (1990).
- NCI-60 Developmental Therapeutics Program SOP, current revision.
- Keepers Y. et al., *Eur. J. Cancer* 27, 897 (1991) — SRB assay validation.
- Vichai V., Kirtikara K., *Nat. Protoc.* 1, 1112 (2006).

## 13. Revision History

| Rev | Date       | Author | Notes                                                |
|-----|------------|--------|------------------------------------------------------|
| 0.1 | 2026-09-26 | Wf-28  | Initial template (no wet-lab data yet).              |
