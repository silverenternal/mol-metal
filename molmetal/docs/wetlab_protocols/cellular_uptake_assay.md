# Cellular Uptake Assay — ICP-MS Quantification of Pt / Ru / Ir

**Document ID:** WLP-UP-001
**Version:** 1.0
**Status:** Template (no wet-lab data has been collected)
**Owner:** Mol-Metal wet-lab coordinator (TBD)
**Last reviewed:** 2026-09-26

---

## 1. Purpose and Scope

This protocol describes the standardized workflow for quantifying the
total cellular accumulation of platinum, ruthenium, and iridium in
cultured cells exposed to metallodrug candidates. The method couples
short-term drug exposure, cell lysis, and element quantification by
Inductively Coupled Plasma Mass Spectrometry (ICP-MS).

ICP-MS gives a direct, absolute read-out of metal content per cell,
uniquely capable of distinguishing between the parent complex, free
metal ion, and protein-bound species when coupled with size-exclusion
or reversed-phase chromatography (see §9 for extensions).

## 2. Safety and Compliance

- ICP-MS operation requires a dedicated instrument room with fume hood
  for acid digestion, oxygen supply, and argon (or mixed-gas) cylinder
  storage. Refer to the institutional ICP-MS facility SOP.
- Aqua regia (HCl : HNO₃ = 3 : 1) is highly corrosive and produces
  chlorine and nitrosyl chloride vapors. Perform digestions in a
  certified fume hood with eye protection and acid-resistant gloves.
- Pt, Ru, and Ir salts are toxic. Handle as for the parent compounds
  (see `README.md`).

## 3. Materials and Reagents

| Reagent                                | Vendor        | Catalogue # | Notes                          |
|----------------------------------------|---------------|-------------|--------------------------------|
| Trace-metal-grade HNO₃ (67–70 %)       | Fisher        | A509       |                                |
| Trace-metal-grade HCl (32–35 %)        | Fisher        | A508       |                                |
| Hydrogen peroxide (30 %, TraceMetal)   | Fisher        | H341       |                                |
| Pt standard (1 000 µg/mL in 10 % HCl)  | Inorganic Ventures | CGPT1   | Primary standard               |
| Ru standard (1 000 µg/mL in 10 % HCl)  | Inorganic Ventures | CGRU1   |                                |
| Ir standard (1 000 µg/mL in 10 % HCl)  | Inorganic Ventures | CGIR1   |                                |
| Internal standard: Y or In (1 000 µg/mL) | Inorganic Ventures | CGY1 / CGIN1 | Compensates for drift   |
| Phosphate-buffered saline (PBS), Mg²⁺/Ca²⁺-free | Gibco | 14190       |                                |
| RIPA lysis buffer (with protease inhibitors) | Sigma | R0278       |                                |
| BCA Protein Assay Kit                  | Thermo        | 23225       |                                |
| Metal-free 15 mL conical tubes         | VWR           | 89049-170  | Pre-checked lot                |
| Metal-free 1.5 mL Eppendorf tubes      | Eppendorf     | 0030122-78 | Pre-checked lot                |

## 4. Equipment

- ICP-MS instrument (Agilent 7900 / PerkinElmer NexION / Thermo iCAP RQ)
  tuned for high-sensitivity KED or DRC mode to remove polyatomic
  interferences on ¹⁹⁵Pt, ¹⁰¹Ru, ²⁰⁷Pb-analogues, and ¹⁹³Ir.
- Microwave digestion system (CEM MARS 6 or Anton Paar Multiwave 7000)
  with PFA or quartz vessels.
- Ultrasonic probe bath for cell lysis.
- Centrifuge (benchtop, ≥ 16 000 × g).
- Hemocytometer or automated cell counter.
- Class II biosafety cabinet, humidified CO₂ incubator.
- Pneumatic pipettes with metal-free tips.

## 5. Cell Culture and Drug Exposure

1. Seed cells in 6-well plates at **2 × 10⁵ cells/well** in 2 mL complete
   medium. Allow 24 h for attachment.
2. Treat with the metallodrug at a single concentration (commonly 10 µM
   to match biological relevance) for **1, 2, 4, 8, 24 h** time points.
   Use three replicate wells per time point.
3. Include vehicle (0.5 % DMSO) controls and a positive control
   (cisplatin, 10 µM).

## 6. Sample Harvest

All steps performed on ice with metal-free plastics.

1. Aspirate medium; wash cells **three times** with cold PBS. The third
   wash is retained and measured as a procedural blank (it should be
   below the detection limit; if not, residual extracellular drug is
   being measured).
2. Add 300 µL RIPA buffer with protease inhibitors; incubate 5 min on
   ice.
3. Scrape cells; transfer lysate to a metal-free 1.5 mL tube; sonicate
   3 × 10 s on ice.
4. Centrifuge at 16 000 × g for 15 min at 4 °C. Collect the supernatant
   ("cell lysate fraction").
5. Reserve a 20 µL aliquot for the BCA protein assay.
6. Cell pellets (insoluble fraction) are reserved for separate
   analysis if partitioning between soluble and protein-bound species
   is required.

## 7. Microwave Acid Digestion

1. Transfer the lysate to a PFA digestion vessel.
2. Add **2 mL trace-metal-grade HNO₃** and **0.5 mL H₂O₂ (30 %)** per
   vessel.
3. Include reagent blanks (HNO₃ + H₂O₂ only) and a digestion blank
   (untreated cells) on every run.
4. Run the microwave program (suggested):
   - Ramp to 200 °C over 15 min (1 200 W).
   - Hold at 200 °C for 20 min.
5. Cool to < 40 °C in the rotor; vent vessels in the fume hood.
6. Dilute the digest to a final acid concentration of 2 % HNO₃ (typically
   1:10 with ultra-pure water) for ICP-MS analysis. Add the internal
   standard (Y or In) at **10 µg/L** final concentration.

## 8. ICP-MS Quantification

### 8.1 Tuning

- Tune the instrument daily using the manufacturer's tuning solution to
   meet sensitivity ≥ 50 000 cps for ¹⁵⁹Tb in standard mode.
- For ¹⁹⁵Pt and ¹⁹³Ir, use He KED mode (4.5 mL/min He) to suppress
   polyatomic interferences (⁵⁶Ar¹³⁹Ba, ⁴⁰Ar¹⁵³Eu, etc.).
- For ¹⁰¹Ru, DRC mode with NH₃ reaction gas (0.5 mL/min) is preferred
   to remove Ni and Cu interferences.

### 8.2 Calibration curve

Prepare external calibration standards in 2 % HNO₃ at 0.1, 1, 10, 100,
1 000 µg/L Pt/Ru/Ir with 10 µg/L internal standard (Y or In).
Verify linearity (r² > 0.999) and check a quality-control standard every
10 samples (target deviation < 10 %).

### 8.3 Isotopes and detection limits

| Isotope | Natural abundance | Expected LOD (³σ of blank) |
|---------|-------------------|----------------------------|
| ¹⁹⁵Pt   | 33.8 %            | 0.005 µg/L                 |
| ¹⁹⁴Pt   | 32.9 %            | 0.01 µg/L                  |
| ¹⁰¹Ru   | 17.0 %            | 0.01 µg/L                  |
| ¹⁰²Ru   | 31.6 %            | 0.01 µg/L                  |
| ¹⁹³Ir   | 62.7 %            | 0.005 µg/L                 |

## 9. Output Schema

Save results as `uptake_<compound_id>_<cell_line>.csv`:

| column              | type   | example                  |
|---------------------|--------|--------------------------|
| compound_id         | str    | "MC-1024"                |
| cell_line           | str    | "A549"                   |
| exposure_h          | float  | 4.0                      |
| replicate           | int    | 1                        |
| element             | str    | "Pt" / "Ru" / "Ir"       |
| isotope             | str    | "195Pt"                  |
| ppb_measured        | float  | 3.45                     |
| pg_per_cell         | float  | 12.7                     |
| fmol_per_cell       | float  | 65.2                     |
| nmol_per_mg_protein | float  | 4.2                      |
| protein_ug_per_well | float  | 22.5                     |
| cell_count          | int    | 220000                   |
| plate_id            | str    | "2026-09-26-UP-002"      |
| instrument          | str    | "Agilent 7900"           |
| operator            | str    | "operator_initials"      |

`pg_per_cell` is computed from `ppb_measured`, the digest dilution
factor, and the cell count per well:

    metal_pg = ppb_measured * (diluted_volume_L) * (dilution_factor)
    pg_per_cell = metal_pg / cell_count

`nmol_per_mg_protein` is the preferred normalization for cross-line
comparison:

    nmol_per_mg_protein = (pg_per_cell * 1e-9 / metal_atomic_mass) /
                          (protein_ug_per_well * 1e-6 * cell_count / 1e6)

## 10. Quality Control

- Reagent blank < LOD/3.
- Untreated cell blank < LOD/3.
- Positive control (cisplatin-treated) within the lab's historical
  range (e.g. 0.5–5 pg Pt / cell at 10 µM, 4 h).
- Internal-standard recovery: 80–120 %.
- Triplicate CV ≤ 15 %.

## 11. Speciation Extensions (Optional)

The lysate / digest split (§6) can be analysed by hyphenated
techniques if parent-vs-fragment speciation is required:

- **SEC-ICP-MS** — quantify metal bound to high-MW (e.g. protein) vs
  low-MW (free or fragment) fractions. Column: Superdex 200 Increase
  5/150 GL.
- **RP-HPLC-ICP-MS** — separate intact complex from metabolites using
  a C18 column with an ammonium-formate/MeOH gradient.

These methods are out of scope for the standard protocol and must be
developed per compound class.

## 12. References

- Timerbaev A.R., *J. Anal. At. Spectrom.* 36, 2298 (2021).
- Wedeking A. et al., *Metallomics* 12, 678 (2020).
- U.S. EPA Method 6020B — Inductively Coupled Plasma Mass Spectrometry.
- Guidance on ICP-MS sample preparation: IUPAC Technical Report 2018.

## 13. Revision History

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 0.1 | 2026-09-26 | Wf-28  | Initial template (no wet-lab data yet).            |