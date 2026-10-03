# Binding Affinity Assay — Isothermal Titration Calorimetry (ITC)

**Document ID:** WLP-BIND-001
**Version:** 1.0
**Status:** Template (no wet-lab data has been collected)
**Owner:** Mol-Metal wet-lab coordinator (TBD)
**Last reviewed:** 2026-09-26

---

## 1. Purpose and Scope

This protocol describes the standardized Isothermal Titration Calorimetry
(ITC) workflow used to determine the equilibrium dissociation constant
(K_d), stoichiometry (n), enthalpy (ΔH), and entropy (ΔS) of metallodrug
binding to a purified protein target (e.g. a metalloenzyme, transferrin,
albumin, or a recombinant metalloprotein domain).

ITC is the preferred primary method because it provides a complete
thermodynamic profile in a single titration, requires no labelling or
immobilization, and is directly compatible with the high-ionic-strength
buffers needed to keep metal complexes soluble. The protocol is written
for a Malvern MicroCal PEAQ-ITC or equivalent microcalorimeter with a
200 µL sample cell and a 40 µL syringe.

## 2. Safety and Compliance

- The metallodrug is handled as acutely toxic. Gloves, lab coat, and
  safety goggles are mandatory. See `README.md` for facility-wide PPE
  requirements.
- The ITC instrument contains a precision-machined Hastelloy cell. Do not
  introduce halogenated solvents (chloroform, dichloromethane); these
  corrode the cell fittings and may invalidate the calibration.
- Buffer filtration and thorough degassing are mandatory to avoid
  bubble artefacts that destroy the baseline.

## 3. Materials and Reagents

| Reagent                          | Vendor (suggested) | Catalogue # | Notes                                  |
|----------------------------------|--------------------|-------------|----------------------------------------|
| Sodium phosphate (monobasic)     | Sigma              | S0751       | For 20 mM sodium phosphate, pH 7.4     |
| Sodium phosphate (dibasic)       | Sigma              | S0876       |                                        |
| Sodium chloride                  | Sigma              | S7653       | 150 mM final                           |
| EDTA (disodium dihydrate)        | Sigma              | E4884       | Optional, 0.1 mM to scavenge trace metal |
| Tris base (ultra-pure)           | Sigma              | T1503       | Alternative buffer if phosphate interferes |
| Purified protein target          | in-house / vendor  | —           | ≥ 95 % purity by SDS-PAGE; SEC-MALS preferred |
| Reference compound               | Vendor             | —           | E.g. apoprotein, known inhibitor        |

### 3.1 ITC Buffer (standard)

20 mM sodium phosphate, 150 mM NaCl, pH 7.4 (the same buffer must be used
to dissolve both protein and ligand — buffer mismatch is the most common
source of artefactual heats of dilution). Filter through a 0.22 µm
membrane and degas for 15 min at 25 °C under gentle stirring before
every session.

## 4. Equipment

- MicroCal PEAQ-ITC (or equivalent) with operating software
  (PEAQ-ITC Analysis, NanoAnalyze, or AFFINImeter)
- ÄKTA or FPLC for protein polishing immediately before ITC
- Size-exclusion column (Superdex 200 Increase 10/300 GL or equivalent)
- Ultracentrifugal concentrators (10 kDa MWCO)
- UV spectrophotometer (NanoDrop or cuvette-based) with A₂₈₀ extinction
- pH meter, calibrated daily with NIST-traceable buffers
- Vacuum filtration apparatus (0.22 µm PES membrane)
- Analytical balance (0.1 mg readability)
- Magnetic stirrer and clean stir bars (acid-washed, ITC-grade)

## 5. Sample Preparation

### 5.1 Protein

1. Thaw an aliquot of purified protein on ice.
2. Polish by size-exclusion chromatography into ITC buffer immediately
   before the titration. Collect the central 60 % of the monomer peak.
3. Concentrate to **30–80 µM** (exact concentration determined by A₂₈₀
   using the calculated extinction coefficient).
4. Verify monodispersity by dynamic light scattering (DLS): polydispersity
   index ≤ 0.2; reject otherwise.
5. Filter (0.22 µm) and degas for 15 min. Keep at the experimental
   temperature until loading.

### 5.2 Ligand (metallodrug)

1. Dissolve in ITC buffer at **500–1 500 µM** (≥ 10× cell concentration).
2. Confirm solubility by DLS or visual inspection; metallodrugs often
   form colloids or aggregates at high [Cl⁻] mismatch — adjust buffer
   if necessary.
3. Filter (0.22 µm) and degas for 15 min.

### 5.3 Buffer blank reference

Retain ≥ 1.5 mL of the final dialysate / SEC flow-through to perform the
buffer-into-buffer titration (§7.4).

## 6. Instrument Preparation

1. Power on the calorimeter ≥ 30 min before the run for thermal
   equilibration.
2. Rinse the cell three times with ITC buffer; do **not** introduce
   detergents or organic solvents.
3. Fill the reference cell with ultra-pure water (degassed).
4. Load the sample cell with **200 µL** of protein solution using the
   gel-loading tip provided. Avoid bubbles; tap the syringe to release
   trapped air.
5. Load the syringe with **40 µL** of ligand solution; assemble and
   insert into the cell.

## 7. Titration Protocol

### 7.1 Standard parameters

| Parameter           | Value                                |
|---------------------|--------------------------------------|
| Temperature         | 25 °C (or 37 °C for physiological)   |
| Reference power     | 5 µcal/s                             |
| Stir speed          | 750 rpm                              |
| Number of injections| 19                                   |
| Injection volume    | 2 µL (first) + 18 × 2 µL (rest)      |
| Injection spacing   | 180 s (extend to 240 s for slow kinetics) |
| Filter period       | 2 s                                  |

The first injection of 0.5 µL is discarded from the fit (it is
predominantly diffusional equilibration).

### 7.2 Recommended stoichiometry

Use a protein:ligand molar ratio of **1 : 10 to 1 : 20** in the cell
(matches c = 10–50, the optimum for accurate K_d determination in the
micromolar range). For tight binders (K_d < 10 nM), use competitive
displacement ITC.

### 7.3 Cleaning after run

1. Rinse the cell three times with 2 % SDS, then three times with
   ultra-pure water, then three times with ITC buffer.
2. Soak in 0.1 M NaOH for 30 min, then rinse thoroughly with water.
3. Inspect the cell visually; if baseline drift > 0.02 µW/s persists,
   contact service.

### 7.4 Buffer blank

Perform an identical titration of ligand into buffer (no protein) using
the same injection parameters. Subtract these heats from the main
titration to correct for dilution and mixing artefacts. A flat, small
signal (< 0.1 µcal/s) is expected.

## 8. Data Analysis

### 8.1 Fitting model

Fit the integrated heats to a **single-site binding model**:

    Q_i = n * M_tot * ΔH * V * ( [L]_i + [P]_tot + 1/K_d -
            sqrt(([L]_i + [P]_tot + 1/K_d)^2 - 4*[L]_i*[P]_tot)) / 2

where `Q_i` is the heat per injection, `V` the cell volume, `n` the
stoichiometry, `M_tot` the total protein concentration, and `[L]_i`
the cumulative ligand concentration after injection i.

The fit yields `K_d`, `n`, `ΔH`, and (by difference) `ΔS = −ΔH / T −
R ln K_a`.

### 8.2 Quality criteria

| Quantity       | Acceptable range                                  |
|----------------|---------------------------------------------------|
| n              | 0.7 – 1.3 (for 1 : 1 binding)                     |
| K_d            | within the dynamic range for c (= n·[P]/K_d): 1 ≤ c ≤ 100 |
| RMSD of residuals | ≤ 0.05 µcal/s                                 |
| Buffer blank   | Each integrated peak < 5 % of main-titration peak |

If `n` deviates substantially from 1, consider a "two-sites" model; check
for protein aggregation or ligand stoichiometric impurity.

### 8.3 Output schema

Save the fitted parameters to `binding_<compound_id>_<target>.json`:

| field           | type   | example                                |
|-----------------|--------|----------------------------------------|
| compound_id     | str    | "MC-1024"                              |
| target          | str    | "HSA"                                  |
| temperature_C   | float  | 25.0                                   |
| buffer          | str    | "20 mM NaPi, 150 mM NaCl, pH 7.4"      |
| n_sites         | float  | 1.05                                   |
| Kd_M            | float  | 2.3e-6                                 |
| Kd_CI_low_M     | float  | 1.9e-6                                 |
| Kd_CI_high_M    | float  | 2.8e-6                                 |
| dH_kcal_per_mol | float  | -8.4                                   |
| dS_cal_per_mol_K | float | 12.1                                   |
| rmsd_ucal_per_s | float  | 0.03                                   |
| instrument      | str    | "MicroCal PEAQ-ITC"                    |
| operator        | str    | "operator_initials"                    |
| run_date        | str    | "2026-09-26"                           |

## 9. Common Pitfalls and Mitigation

| Pitfall                                | Mitigation                                          |
|----------------------------------------|-----------------------------------------------------|
| Buffer mismatch → huge dilution heats  | Always use the same SEC flow-through for protein & ligand |
| Metal complex precipitation at high [Cl⁻] | Reduce [NaCl] to 50 mM; switch to HEPES buffer  |
| Protein aggregation → sloping baseline | Polish by SEC; verify by DLS before each run        |
| Tight binding beyond c = 100           | Use displacement ITC; lower [protein] to 1–5 µM     |
| Slow kinetics → mis-shaped peaks       | Increase spacing to 240–300 s                        |

## 10. References

- Wiseman T. et al., *Anal. Biochem.* 179, 131 (1989).
- Velazquez-Campoy A., Freire E., *Nat. Protoc.* 1, 1868 (2006).
- MicroCal PEAQ-ITC user manual (current revision).
- Bastos M., Olofsson G., *J. Solution Chem.* 51, 1097 (2022).

## 11. Revision History

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 0.1 | 2026-09-26 | Wf-28  | Initial template (no wet-lab data yet).            |
