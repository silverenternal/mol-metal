# TODO-Wf74 — IC50 Analysis Pipeline: from raw readout to pIC50 publication

**Date**: 2026-09-26
**Workflow**: Wf-74 Phase 3 (R20-P6 wet-lab cooperation — analysis layer doc)
**Status**: 🚀 **Active — CPU-only analysis pipeline doc** (template + scaffold only; no wet-lab IC50 MEASURED today)
**Priority**: MEDIUM (R20-P6 wet-lab milestone 2 / 3 dependency — gates paper §5 final claim, but the wet-lab gate is itself deferred 6-month)
**Owner**: Hugo Lee + ultracode agent (ultracode analog = single Claude Code pass)
**Parent**: `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan) + `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 P6.0 + P6.1 + P6.2 ship surface) + `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule)
**Companion script**: `molmetal/scripts/template_ic50_pipeline.py` (~200 LOC, pure NumPy/SciPy)
**Companion template**: `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json`

---

## §0 Why this doc exists

Wf-58 (R20-P6 wet-lab cooperation) ships the **CPU-only in-silico layer** today — top-20 ranked (P6.0), D-MPNN per-compound pIC50 predicted (P6.1), agreement-bench JSON shell ready (P6.2 placeholder) — but **the wet-lab IC50 is NOT yet measured**. When P6.2 wet-lab IC50 ships (6-month ORNL/PKU/协和 timeline per `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` §4), the wet-lab PI will hand us:

1. Raw plate-reader readouts (OD 540 nm, fluorescence, or luminescence)
2. Compound × concentration × replicate metadata
3. Background / vehicle / positive-control wells

… and we need a **reproducible analysis pipeline** that converts (1)+(2)+(3) into per-compound × per-cell pIC50 ± 95% CI, then compares against the Wf-58 D-MPNN predictions already on disk at `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json`.

**This document specifies the pipeline**. The companion script (`template_ic50_pipeline.py`) implements the analysis numerically with **no wet-lab data dependency** — it runs on a small synthetic fixture and produces the schema that real wet-lab data will populate.

**Constraint reminder** (per `party-twenty-spirit` §3.5): **no fabricated wet-lab numbers, no paper §5 claim, no LOI sent at this revision**. The pipeline doc + template script ship today; the wet-lab gate is PROJECTED on Wf-58 P6.2 timeline.

---

## §1 Raw data normalization

### §1.1 Input schema (per wet-lab PI)

The wet-lab PI delivers a **plate-wise long-form CSV** with columns:

| Column | Type | Example | Notes |
|---|---|---|---|
| `plate_id` | str | `P6-001-A` | plate + replicate letter |
| `well_id` | str | `C05` | 96-well row+col |
| `compound_id` | str | `platinai_0454` | matches Wf-58 top-20 `id` |
| `concentration_uM` | float | `0.316` | log-spaced (typically 8 doses; see §1.4) |
| `metal` | str | `Ir` | Pt/Pd/Au/Ru/Ir per Wf-58 top-20 |
| `cell_line` | str | `MCF7` | per `molmetal/data/nci60_precious_9cell.csv` |
| `replicate` | int | `1..3` | biological replicate (not technical) |
| `assay` | str | `SRB_540nm` | `SRB_540nm` / `MTT_570nm` / `CTG_RLU` / `AlamarBlue_RFU` |
| `time_h` | int | `72` | exposure duration (NCI-60 protocol) |
| `raw_readout` | float | `0.412` | absorbance / fluorescence / luminescence |
| `background_well` | bool | `False` | media-only well flag |
| `vehicle_well` | bool | `False` | DMSO 0.5% v/v well flag |
| `positive_ctrl_well` | bool | `False` | cisplatin 10 µM reference well |
| `qc_flag` | str | `pass` | `pass` / `cv_gt_20` / `z_prime_lt_0.5` / `outlier` |

### §1.2 Background subtraction

For each plate, compute the **plate-mean background** across all `background_well=True` rows:

```
bg_plate = mean(raw_readout | plate_id == P | background_well == True)
```

Then subtract per-well: `corrected_readout = raw_readout - bg_plate`.

**Why plate-wise, not well-wise**: media-only wells are located at row A (per `molmetal/docs/wetlab_protocols/cytotoxicity_assay.md` §7) and serve as the plate-level optical background; subtracting per-well would over-correct and inject well-edge evaporation artefacts.

### §1.3 Plate normalization (B-score variant)

After background subtraction, apply a **two-way median polish** to remove row + column edge effects (the most common systematic artefact in 96-well plate-reader readouts):

1. Compute the row median per plate → row effect `r_i`.
2. Compute the column median per plate → column effect `c_j`.
3. Normalize: `normalized_readout = corrected_readout / (r_i * c_j)`.

**Alternative** (if row/column effects are not separable): compute the **plate Z-prime (Z')** as a quality gate:

```
Z' = 1 - (3 * (sigma_pos + sigma_neg)) / |mu_pos - mu_neg|
```

where `mu_pos, sigma_pos` are mean/std of `positive_ctrl_well=True` and `mu_neg, sigma_neg` are mean/std of `vehicle_well=True`. **Reject the plate if Z' < 0.5** (per NCI-60 protocol + `molmetal/docs/wetlab_protocols/cytotoxicity_assay.md` §11).

### §1.4 Dose grid (log-spaced)

Standard 8-point dose grid per compound (per `cytotoxicity_assay.md` §8):

```
concentrations_uM = [0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0]
                  = 10 ** linspace(-1.5, 2.0, 8)   # log10(M) range
```

**Each (compound, cell_line) pair** must have ≥6 dose points × ≥3 biological replicates to be curve-fitting-eligible (see §2.4); otherwise marked `qc_flag=insufficient_doses`.

### §1.5 Per-well viability

Convert normalized readout to fractional viability relative to vehicle control:

```
viability = normalized_readout / median(vehicle_well == True) per plate
```

Clamp to `[0.0, 1.5]` to catch obvious artefacts (viability > 1.5 = hormesis or pipetting error; < 0.0 = cytolysis beyond assay floor).

---

## §2 Dose-response curve fitting

### §2.1 The 4-parameter log-logistic (Hill) equation

For each (compound, cell_line, replicate-set), fit:

```
f(c) = Bottom + (Top - Bottom) / (1 + (c / EC50) ** n_Hill)
```

where:

| Parameter | Symbol | Units | Typical range (metallodrugs) |
|---|---|---|---|
| Lower asymptote | `Bottom` | fraction viability | 0.0 (full kill) |
| Upper asymptote | `Top` | fraction viability | 0.95-1.05 |
| Half-maximal effective concentration | `EC50` | µM | 0.01 – 100 |
| Hill slope | `n_Hill` | dimensionless | 0.5 – 3.0 |

**Note on terminology**: for cytotoxicity, EC50 is conventionally reported as **IC50** (concentration for 50% growth inhibition). The fitted EC50 IS the IC50 when `Top ≈ 1.0` and `Bottom ≈ 0.0`.

### §2.2 Fit algorithm

**Primary**: SciPy `curve_fit` with `loss='soft_l1'` (robust to outlier wells) and `p0=[1.0, 0.0, 1.0, 1.5]` initial guess. **Bounds**: `Bottom ∈ [−0.2, 0.5]`, `Top ∈ [0.5, 1.5]`, `EC50 ∈ [1e-4, 1e3]`, `n_Hill ∈ [0.1, 5.0]`.

**Fallback** (if `curve_fit` does not converge after 50 iterations): weighted least-squares on `log10(viability / (1 - viability))` vs `log10(c)` linearisation, then re-fit with the linearised estimate as initial guess.

### §2.3 Pooling replicates

For each (compound, cell_line), pool all biological replicates by **fitting a single curve to the pooled (c, viability) points with replicate-stratified variance**:

```
weights = 1 / sigma_replicate**2   # inverse-variance weights
```

This produces a **per-compound × per-cell-line IC50 estimate** with replicate-aware uncertainty (see §3 for CI).

### §2.4 Fit-quality gates

| Gate | Criterion | Action on FAIL |
|---|---|---|
| **Convergence** | `curve_fit` info == 1..4 OR fallback R² ≥ 0.85 | mark `qc_flag=fit_fail` |
| **Hill slope plausibility** | `0.5 ≤ n_Hill ≤ 3.0` | mark `qc_flag=hill_outlier` |
| **Asymptote plausibility** | `Bottom ≤ 0.3` AND `Top ≥ 0.7` | mark `qc_flag=asymptote_outlier` |
| **Dose coverage** | ≥6 dose points span ≥2 log units with monotonic viability | mark `qc_flag=insufficient_doses` |
| **Replicate agreement** | pooled residual std ≤ 0.20 (viability units) | mark `qc_flag=replicate_mismatch` |

If any of the above gates FAIL, the compound is **excluded from §5 in-silico vs wet-lab comparison** (no fabricated pIC50).

---

## §3 pIC50 computation + 95% CI

### §3.1 Definition

```
pIC50 = -log10(IC50_M) = -log10(IC50_uM / 1e6) = 6 - log10(IC50_uM)
```

For IC50 in µM (the wet-lab SI unit per `molmetal/docs/wetlab_protocols/cytotoxicity_assay.md` §8.3), pIC50 ranges ~4.0 (IC50 = 100 µM) to ~8.0 (IC50 = 10 nM).

### §3.2 95% confidence interval (bootstrap)

For each fitted IC50, run **1,000-iteration residual bootstrap**:

1. Compute per-well residuals: `r_i = viability_observed_i - viability_predicted_i`.
2. For each bootstrap iteration `b ∈ {1..1000}`:
   - Resample residuals with replacement.
   - Construct `viability_bootstrap_b = viability_predicted + r_resampled`.
   - Re-fit (compound, cell_line) → IC50_b.
3. Compute **percentile CI**: `IC50_low = quantile(IC50_b, 0.025)`, `IC50_high = quantile(IC50_b, 0.975)`.

Convert to pIC50 CI: `pIC50_low = 6 - log10(IC50_high)`, `pIC50_high = 6 - log10(IC50_low)` (**note inversion**: lower IC50 = higher pIC50).

### §3.3 Per-compound pIC50 record schema

The per-compound × per-cell JSON record produced by this pipeline:

```json
{
  "rank": 1,
  "compound_id": "platinai_0454",
  "metal": "Ir",
  "cell_line": "MCF7",
  "ic50_uM": 0.412,
  "ic50_uM_ci_low": 0.318,
  "ic50_uM_ci_high": 0.534,
  "pic50": 6.385,
  "pic50_ci_low": 6.272,
  "pic50_ci_high": 6.498,
  "n_doses": 8,
  "n_replicates": 3,
  "hill_slope": 1.42,
  "fit_quality": "pass",
  "qc_flags": [],
  "dmpnn_pic50_predicted": 8.6172,
  "dmpnn_pic50_delta": 2.23
}
```

Where:
- `dmpnn_pic50_predicted` is the **Wf-58 P6.1 D-MPNN per-compound pIC50** (mean across 3 seeds, mean across 4 cell lines; see `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json` for the per-cell breakdown).
- `dmpnn_pic50_delta = dmpnn_pic50_predicted - pic50` (signed residual; positive = D-MPNN over-predicts potency).

**Honest framing**: the D-MPNN numbers are CPU-only bench quality on synthetic IC50 labels (per `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` §1.2); the `dmpnn_pic50_delta` is a **rank-comparison indicator**, not a calibrated absolute comparison.

### §3.4 pIC50 standardisation across cell lines

For each compound: `pic50_mean = mean(pic50 | cell_line in 4-cell set, qc_flag == 'pass')`, `pic50_std = std(same)`. The 4-cell reference set is MCF7 / A549 / A2780 / A2780cis (per `molmetal/data/platinai_4cell.csv` + Wf-58 P6.1 D-MPNN training data).

The **A2780cis / A2780 ratio** is the **cisplatin-resistance fold**: `resistance_fold = 10 ** (pic50_A2780cis - pic50_A2780)`. Compounds with `resistance_fold ≤ 2.0` are flagged as **resistance-bypass candidates** (potential lead compounds); `> 10.0` are **cross-resistant** (typical of cisplatin-analog scaffolds).

---

## §4 Statistical analysis

### §4.1 ANOVA per metal cohort

For each metal (Pt / Pd / Au / Ru / Ir), test whether pIC50 varies across cell lines:

```
one_way_anova(pic50 ~ cell_line | metal == M)
```

**Test**: `scipy.stats.f_oneway` (one-way ANOVA, equal variance assumed) or `scipy.stats.kruskal` (non-parametric fallback if Levene's test rejects equal variance).

**Significance level**: α = 0.05; report Bonferroni-corrected p-values across the 5 metals (α_eff = 0.01).

**Effect size**: η² = SS_between / SS_total.

### §4.2 Post-hoc (Tukey HSD)

If ANOVA rejects H0 (`p < α_eff`), run **Tukey HSD** pairwise comparisons across the 4 cell lines within each metal:

```
tukey_hsd(pic50_A, pic50_B, ..., pic50_D | metal == M)
```

Use `scipy.stats.tukey_hsd` (SciPy ≥ 1.8). Report:
- `mean_diff`, `p_adj`, `CI_low`, `CI_high` for each pair.
- Significance flag: `True` if `p_adj < 0.05`.

### §4.3 Resistance signature per metal

For each metal, fit a **linear model**: `pic50 = β_0 + β_1 * is_cis_resistant_cell`. The coefficient `β_1` is the **resistance slope** (positive = more potent in A2780cis, negative = cross-resistant).

| Metal | β_1 expected (cisplatin-like) | β_1 expected (resistance-bypass) |
|---|---:|---:|
| Pt | strongly negative | n/a |
| Pd | negative (similar mechanism) | variable |
| Au | weakly negative | variable |
| Ru | weakly negative / neutral | **positive (NAMI-A-class)** |
| Ir | neutral / weakly positive | **strongly positive (kinase-target class)** |

### §4.4 Multiple-testing correction across (metal × cell-line × replicate-set)

Bonferroni is conservative; for exploratory wet-lab bench, **Benjamini-Hochberg FDR** is acceptable:

```
p_adj_bh = scipy.stats.false_discovery_control(p_values, method='bh')  # SciPy ≥ 1.11
```

Report both raw p and BH-adjusted p; flag significance at `p_adj_bh < 0.05`.

---

## §5 In-silico vs wet-lab comparison

### §5.1 Input alignment

For each (compound, cell_line) present in **both** the wet-lab pIC50 table (§3.3) and the Wf-58 D-MPNN prediction table (`molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json`):

- Wet-lab: `pic50_wetlab` (this pipeline §3) with 95% CI.
- In-silico: `pic50_dmpnn = per_compound_pic50.json[compound_id].per_cell_pic50[cell_line]` (mean across 3 seeds).

### §5.2 Pearson r per (metal × cell_line)

For each (metal M, cell_line L) cohort with **n ≥ 4 compounds**, compute:

```
pearson_r = scipy.stats.pearsonr(pic50_wetlab, pic50_dmpnn)[0]
spearman_rho = scipy.stats.spearmanr(pic50_wetlab, pic50_dmpnn)[0]
```

**Gate** (per `TODO/pending/97_r20_p6_wetlab_ic50.md` §3 P6.3): `pearson_r ≥ 0.7` per (metal, cell_line).

**Bootstrap 95% CI on r**: resample (compound, pic50_wetlab, pic50_dmpnn) triples 1,000× and report `quantile(r_b, [0.025, 0.975])`.

### §5.3 Per-compound residual analysis

```
residual = pic50_dmpnn - pic50_wetlab
```

**Outlier flag**: `abs_residual > 2 * sigma_residual_pooled`. **Cross-metal transfer caveat**: residuals for Pd / Au / Ru / Ir are expected to be larger than for Pt (the D-MPNN trains on PlatinAI 4-cell Pt-only); the agreement gate `pearson_r ≥ 0.7` is applied **per metal** to allow for cohort-specific transfer.

### §5.4 Honest framing per `party-twenty-spirit` §3.5

Per the spirit-of-twenty honest-negative rule, the in-silico vs wet-lab comparison MUST distinguish:

| Source | Status today (Wf-74 Phase 3) | Status after P6.2 wet-lab |
|---|---|---|
| D-MPNN pIC50 (in-silico, CPU) | ✅ MEASURED (5-compound fixture) | MEASURED (20-compound full top-20 on P6.1 re-run) |
| Wet-lab pIC50 (per-compound × per-cell) | 🟡 DEFERRED (template ships, real data on P6.2) | MEASURED (Q1 2027 earliest per Wf-58 P6.2) |
| Pearson r ≥ 0.7 per (metal × cell_line) | 🟡 PROJECTED (no wet-lab → no r) | MEASURED gate fires on P6.2 wet-lab data arrival |
| Paper §5 wet-lab validation section | 🚫 DEFERRED (TODO-84 paper rule; needs explicit user OK + MEASURED wet-lab) | DEFERRED until user authorization |

**No fabricated Pearson r, no fabricated pIC50, no paper claim** — the template script ships today, runs on a small synthetic fixture, and writes a JSON whose `pearson_r=null` until real wet-lab data lands.

### §5.5 Output file schema

The final pipeline output (`molmetal/reports/wf74_ic50_pipeline/per_compound_pic50_wetlab.json`):

```json
{
  "meta": {
    "n_compounds": 20, "n_cell_lines": 4, "n_pass_qc": 17, "n_excluded": 3,
    "exclusion_reasons": ["hill_outlier×2", "asymptote_outlier×1"],
    "wetlab_data_source": "ORNL/PKU/协和 P6.2 wet-lab IC50 CSV (TBD)",
    "dmpnn_data_source": "molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json",
    "pipeline_version": "wf74-v1.0",
    "framing": "honest-negative: synthetic pIC50 from synthetic fixture until P6.2 wet-lab ships"
  },
  "compounds": [ /* §3.3 schema × n_compounds */ ],
  "anova_per_metal": { "Pt": {"F": ..., "p_raw": ..., "p_bh": ..., "eta_sq": ..., "tukey": [...]}, ... },
  "agreement_per_metal_cell": { "Pt-MCF7": {"pearson_r": ..., "pearson_r_ci_low": ..., "pearson_r_ci_high": ..., "spearman_rho": ..., "n": ..., "gate_pass": ...}, ... }
}
```

---

## §6 Next actions (Wf-74 close-out + R20-P6 P6.2 hand-off)

1. **Wf-74 close-out doc** on this Wf ship — code-only, no paper touch.
2. **Wf-58 P6.1 re-run** (5-compound fixture → 20-compound full top-20) on R20 GPU host when Wf-60 meko fork + Vina binary land.
3. **Wet-lab PI outreach** (ORNL compute + PKU med-chem + 协和 hospital) per `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` §4.2.
4. **P6.2 wet-lab IC50** ships from first PI reply + MoU signed (6-month timeline).
5. **This pipeline** consumes wet-lab CSV per §1.1 → emits `per_compound_pic50_wetlab.json` per §5.5.
6. **Paper §5** (TODO-84 DEFERRED until user authorization + MEASURED wet-lab agreement).

---

**Date**: 2026-09-26 · **Verdict**: 🚀 SHIP — pipeline doc (5 sections) + companion script (~200 LOC) + per-compound template JSON on disk; CPU-only; no paper §5 touches; honest framing per `party-twenty-spirit` §3.5; wet-lab gate NOT active. · **Co-Authored-By**: Claude Code <noreply@anthropic.com>
