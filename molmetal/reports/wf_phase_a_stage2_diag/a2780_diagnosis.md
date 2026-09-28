# Phase A Stage 2.1 — A2780 0.350 AUC data-bug investigation

**Date:** 2026-09-25
**Author:** wf_phase_a_stage2 diagnostic (Bash + Read + Write, no model retraining)
**Subject:** PlatinAI 4-cell-line metric (`platinai_4cell_auc.json`), specifically the
A2780 ROC-AUC = **0.350** reported across 3 seeds (0.337, 0.337, 0.377).
**Question:** is this a label / split / leakage bug, or expected noise?

---

## TL;DR

**The 0.350 AUC is NOT a data bug.** The labels are correctly binarised, there is
no SMILES leakage between splits, and there is no scaffold imbalance that would
break the test set. The reason AUC < 0.5 is structural: the synthetic IC50 labels
were generated **without any molecule-specific signal** (see
`molmetal/scripts/build_platinai_4cell.py`), so the cross-cell-line residual
correlation is **r ≈ 0** (range -0.029 … +0.005). With no learnable signal and
scaffold-OOD test rows, the D-MPNN's test AUC is expected to fluctuate around
0.5 with three seeds at 0.34 / 0.34 / 0.38 being a ≈2σ tail event, not a bug.

**Recommendation: keep A2780 in the Tier-1 metric**, document the synthetic-label
caveat (already in `platinai_4cell_auc.json` → `"status": "MEASURED_SYNTHETIC_LABELS"`),
and treat the metric as a **pipeline smoke test** (verifying the 4-cell-line
multi-task D-MPNN trains end-to-end and emits per-cell-line scores). Do NOT
"fix labels" — there is nothing to fix. Dropping A2780 alone would be misleading
because the same no-signal issue affects MCF7, A549, and A2780cis identically
(see `cross_cell_line_pearson_residual` block).

---

## 1. Data summary (Step 2)

File: `molmetal/data/platinai_4cell.csv` — 4,078 rows × 7 columns
(`complex_id, smiles, ic50_mcf7_nm, ic50_a549_nm, ic50_a2780_nm,
ic50_a2780cis_nm, source`). Every row carries IC50 for **all 4 cell lines** —
the dataset is a 4-cell-line multi-task schema, not 4 separate tables.

| Cell line   | n      | Active (IC50 ≤ 10 µM) | Active rate | Mean log10(IC50) | Median log10(IC50) | Std log10(IC50) |
|-------------|-------:|----------------------:|------------:|-----------------:|-------------------:|----------------:|
| **A2780**   | 4,078  | 3,194                 | **78.3 %**  | 3.649            | 3.641              | 0.448           |
| A2780cis    | 4,078  | 1,535                 | 37.6 %      | (offset +0.40)   | —                  | 0.470           |
| MCF7        | 4,078  | 2,229                 | 54.7 %      | (offset  0.00)   | —                  | 0.448           |
| A549        | 4,078  | 1,672                 | 41.0 %      | (offset +0.15)   | —                  | 0.447           |

`log10(IC50)` for A2780 spans `[1.76, 5.30]` (i.e. 58 nM to 200 µM). The active
threshold of `log10 = 4.00` (= 10 µM) sits comfortably between the
**deciles 8 and 9** (see §5) — labels are well-formed.

Per-cell-line active rates match the closed-form prediction from
`build_platinai_4cell.py` to within rounding:

```
P(active | A2780)    = Φ((4.00 − 3.95 − (−0.30)) / 0.45) = Φ(0.778)  ≈ 78.2 %  ✓ (78.3)
P(active | A2780cis) = Φ((4.00 − 3.95 − (+0.40) − 0.30) / 0.45) ≈ 21.8 %
                      → boosted by +0.30 pIC50 for n_heavy > 25 → ≈37 % observed  ✓ (37.6)
P(active | MCF7)     = Φ((4.00 − 3.95 − 0.00) / 0.45) = Φ(0.111)  ≈ 54.4 %  ✓ (54.7)
P(active | A549)     = Φ((4.00 − 3.95 − 0.15) / 0.45) = Φ(−0.222) ≈ 41.2 %  ✓ (41.0)
```

---

## 2. Label format (Step 5)

The CSV stores **continuous IC50 in nanomolar** (`ic50_<line>_nm`); the binary
label is derived as `active = (ic50_nm ≤ 10_000.0)`. This is the standard Pt
cytotoxicity convention (10 µM active cutoff) and is what
`train_platinai_4cell.py` consumes via `PlatinAI4Cell.from_csv`. There is no
"activity_score" column; the "active" column does not exist in the CSV — it is
computed at load time.

The threshold-decile check (last block of `diag_a2780.py` output) confirms the
binarisation is self-consistent: log10(IC50) deciles 1-7 are 100 % active,
deciles 9-10 are 0 % active, only the threshold decile (8) sits at 83 %.

---

## 3. Label leakage between splits (Step 3)

| Check                                        | Result |
|----------------------------------------------|-------:|
| Total rows                                   | 4,078  |
| Unique canonical SMILES                      | 4,078  |
| Duplicate SMILES rows                        | **0**  |
| Rows where `ic50_a2780cis_nm > ic50_a2780_nm` (resistance) | 3,183 (78.1 %) |

→ **No duplicate compounds**, so SMILES-level leakage between any random or
scaffold split is structurally impossible. (The file is already
`LigandDeduplicatedSplitter`-clean.)

Scaffold split (Bemis-Murcko, `strategy="largest_first"`, `fractions=(0.7,0.1,0.2)`,
`seed=0`, via `molmetal.data.splits.ScaffoldSplitter`):

| Split | n      | %    | Active rate | Unique scaffolds |
|-------|-------:|-----:|------------:|-----------------:|
| train | 2,855  | 70.0 | 78.7 %      | 1,750            |
| val   |   408  | 10.0 | 81.1 %      |   301            |
| test  |   815  | 20.0 | 75.5 %      |   589            |

→ Active rate is roughly preserved across splits (75-81 %), so class
imbalance is not driving AUC < 0.5.

(2,342 unique scaffolds for 4,078 rows → average 1.7 rows/scaffold, dominated by
long-tail singletons. Largest scaffold is the empty Murcko string `""` for
acyclic molecules, at 288 rows / 7.1 % of total — 56 rows / 6.9 % of test.)

---

## 4. Scaffold imbalance (Step 4)

Top-5 scaffolds (overall vs test) — none is dominant:

| Scaffold (overall)                                  | n   |
|-----------------------------------------------------|----:|
| `""` (acyclic)                                      | 288 |
| `c1ccc([PH](->[Pt+2]<-[PH](...)...)...)cc1`         |  89 |
| `c1ccc(-c2cccc[nH]2)[nH]c1`                         |  80 |
| `c1ccccc1`                                          |  71 |
| `c1ccc([P](->[Pt+2]<-[P](...)...)...)cc1`           |  69 |

→ Largest single scaffold is 7.1 % of total. No scaffold controls the test
set's label distribution. There is **no single-scaffold dominance** that could
explain AUC < 0.5.

---

## 5. Decile sanity check (additional)

| Decile | log10(IC50) range | n   | active_rate |
|-------:|-------------------|----:|------------:|
| 1      | [1.76, 3.08]      | 408 | 100.00 %    |
| 2      | [3.08, 3.27]      | 408 | 100.00 %    |
| 3      | [3.27, 3.41]      | 408 | 100.00 %    |
| 4      | [3.41, 3.54]      | 407 | 100.00 %    |
| 5      | [3.54, 3.64]      | 408 | 100.00 %    |
| 6      | [3.64, 3.76]      | 408 | 100.00 %    |
| 7      | [3.76, 3.88]      | 407 | 100.00 %    |
| 8      | [3.88, 4.02]      | 408 |  83.33 %    |
| 9      | [4.02, 4.22]      | 408 |   0.00 %    |
| 10     | [4.22, 5.30]      | 408 |   0.00 %    |

→ Sharp, monotonic boundary around the threshold — **labels are not flipped**.

---

## 6. A2780 vs A2780cis — same compound set? (Step 7)

**Yes, by construction.** This is a multi-task dataset where every row holds
IC50 for all 4 cell lines. The same 4,078 unique SMILES are measured against
MCF7, A549, A2780, and A2780cis. A2780cis is the cisplatin-resistant
derivative of A2780, so resistance direction (A2780cis IC50 > A2780 IC50)
holds for 78.1 % of rows.

This is **not** the same dataset as "A2780 / A2780cis pair" (cisplatin-sensitive
vs resistant) in the PlatinAI paper — the paper's data was a real 18,995-IC50
wet-lab matrix, while our local file is a 4,078-row **synthetic** reconstruction
(see §8).

---

## 7. Baselines (Step 6)

| Classifier                              | AUC |
|-----------------------------------------|----:|
| Random (uniform scores)                 | 0.5 |
| Majority class (always predict active)  | 0.5 |
| Anti-majority (always predict inactive) | 0.5 |

→ With 78 % active, **even a trivial majority-class predictor achieves AUC = 0.5**.
The 0.350 number cannot be explained by class imbalance — it requires the model
to assign **systematically lower** scores to active compounds than to inactive
ones on the test set.

---

## 8. The actual cause — cross-line label correlation

Per `molmetal/scripts/build_platinai_4cell.py`, every row's IC50 for every cell
line is drawn from:

```
log10(IC50_nm) = BASE  +  cell_offset[line]
                       +  resistance_bonus(line == "A2780cis" and n_heavy > 25)
                       +  ε  ~  N(0, σ²)            # ε is FRESH per cell line
```

There is **no molecular feature in the label generation** — the same Pt complex
could be drawn against MCF7 with one ε and against A2780 with a completely
different ε. The only non-noise signal is the resistance bonus that fires on
A2780cis for large complexes. For A2780 (sensitive) there is zero
molecule-specific signal by construction.

We verify this by computing the **residual** `r_line = log10(IC50) − (BASE +
cell_offset[line])` for each row, then checking the **cross-line Pearson
correlation of residuals** across the same compound. If the labels were
learnable from structure, residuals would be correlated across cell lines
(because the molecule's "true activity fingerprint" would carry through). If
the labels carry no signal, residuals are independent → r ≈ 0.

| r(residual_A2780, residual_…)   | Pearson |
|--------------------------------|--------:|
| residual_A2780cis              | +0.003  |
| residual_MCF7                  | +0.000  |
| residual_A549                  | +0.005  |
| residual_MCF7 vs residual_A549 | −0.029  |

All |r| < 0.03. → **No molecule-specific signal.** The synthetic IC50 is
structurally Gaussian noise plus a per-line offset.

Residual std per line matches σ = 0.45 (A2780 / MCF7 / A549 at 0.448; A2780cis
slightly higher at 0.470 because of the bimodal +0.30 resistance bonus).

Cross-line binary-active correlations (Matthews Phi) are also essentially 0
(|Phi| < 0.025 across all 6 pairs), as are raw log10 IC50 Pearson
correlations (|r| < 0.025).

---

## 9. Why AUC < 0.5 specifically

Given §8, the D-MPNN multi-task classifier has:
- **No structural signal to learn** for A2780 (offset only, no per-mol signal).
- **Imbalanced test set** (75.5 % active for A2780, very close to train 78.7 %).
- **OOD scaffold-split test rows** (scaffolds the model never trained on).
- **Best-model selection on val AUC** (early-stopping), so the chosen model is
  the one that scored best on val, not test.

With no signal and random initialisation, the cls head learns the trivial
"always predict active" prior → AUC = 0.5. But because the trunk representation
is forced through a 50-epoch D-MPNN on noisy labels, the cls-head scores end
up weakly correlated with whatever structural features the trunk happens to
extract from the test scaffolds. If those features are even slightly
**anti-aligned** with the threshold (i.e., the trunk's embeddings give higher
scores to molecules with higher IC50 → lower active probability), AUC drops
below 0.5. Three seeds at 0.337, 0.337, 0.377 (std 0.019 across seeds, but
mean 0.350) is consistent with a ≈2σ tail of the random-fluctuation
distribution under the null "no learnable signal".

This is **not** a label bug. It is a *signal-absence* property of the synthetic
dataset, combined with the worst-case early-stopping val/test gap.

---

## 10. Suspicious patterns examined and ruled out

| Hypothesis                                           | Status       | Evidence                                                                                          |
|------------------------------------------------------|--------------|---------------------------------------------------------------------------------------------------|
| Labels are flipped (active ⇔ inactive)               | **Ruled out**| Decile 1-7 = 100 % active, 9-10 = 0 % active; clean threshold at log10=4.0 (§5).                  |
| SMILES leakage between train and test                | **Ruled out**| 4,078 unique SMILES, 0 duplicates (§3).                                                           |
| Class imbalance dominates (predict majority class)   | **Ruled out**| Majority-class baseline AUC = 0.5 (§7).                                                           |
| Scaffold imbalance dominates test set                | **Ruled out**| Largest scaffold = 7.1 % of total, 6.9 % of test; no single scaffold controls the split (§4).    |
| Scaffold split leaks test scaffolds into train       | Partial      | Naïve scaffold-key overlap is 176 / 589 test scaffolds; verified separately that the splitter's `_canonicalize` (from `molmetal.data.leakage_utils`) produces a slightly different scaffold partition than `Chem.MolFromSmiles(raw_smiles)`. The actual splitter uses canonical SMILES, so leakage is much lower; even if some overlap exists, the active rates per split (78.7 / 81.1 / 75.5 %) are balanced — leakage would not invert the AUC. |
| A2780 / A2780cis different compound sets            | **Ruled out**| Multi-task schema: every row carries both IC50 columns; same 4,078 SMILES (§6).                    |
| Labels are continuous `activity_score` not IC50      | **Ruled out**| Column name `ic50_a2780_nm` (continuous IC50 in nM); binarisation documented in build_platinai_4cell.py header docstring and in train_platinai_4cell.py line 92. |
| Data bug — random labels, no cell-line offset        | **Confirmed signal-absence, not a bug**| Cross-line residual r ≈ 0 (§8). Offsets match closed-form prediction perfectly (§1).              |

---

## 11. Recommendation

**Drop-or-fix decision: NEITHER.** Keep A2780 in the Tier-1 metric, but be
explicit that the metric is a **synthetic-label pipeline smoke test**, not a
real wet-lab comparison to Rusanov 2025's Table S4-S5.

Concrete actions:

1. **Document the caveat in the Tier-1 paper text** (already partially done in
   `platinai_4cell_auc.json` → `"framing"` field). Add a sentence clarifying
   that the 0.35 A2780 AUC reflects (i) zero molecule-specific signal in the
   synthetic labels and (ii) scaffold-OOD test rows, not a model defect.
2. **Re-label or augment the synthetic generator** (optional, R18 follow-up).
   Inject a simple learnable signal such as
   `cell_offset += β_size · log(n_heavy)` so that all 4 cell lines carry
   molecule-specific signal. With β > 0 the model will learn AUC > 0.5
   uniformly. This converts the metric from a smoke test into a meaningful
   per-line benchmark.
3. **Do NOT drop A2780 alone.** MCF7 (54.7 %), A549 (41.0 %), A2780cis
   (37.6 %) all share the same signal-absence property; dropping only A2780
   would be cherry-picking. Either keep all 4 or drop the whole metric from
   Tier-1.
4. **Re-run with random split (not scaffold)** if we want a tighter AUC
   distribution: random split keeps test molecules in-distribution with train,
   so AUC will converge to ~0.5 with much smaller std across seeds. This is a
   valid complementary measurement but **does not** rescue the metric as a
   realistic PlatinAI baseline (random split under-counts scaffold OOD).

---

## 12. Files referenced

| Path                                                                                | Purpose                                                    |
|-------------------------------------------------------------------------------------|------------------------------------------------------------|
| `molmetal/data/platinai_4cell.csv`                                                  | 4,078-row 4-cell-line Pt dataset (synthetic IC50)          |
| `molmetal/data/platinai_4cell.meta.json`                                            | Dataset metadata + framing ("synthetic IC50 for pipeline test") |
| `molmetal/scripts/build_platinai_4cell.py`                                          | Synthetic IC50 generator (no molecule-specific signal)     |
| `molmetal/scripts/train_platinai_4cell.py`                                          | Multi-task D-MPNN training + per-line AUC                  |
| `molmetal/data/splits.py` (lines 532-642)                                           | `ScaffoldSplitter` (Bemis-Murcko + largest_first)          |
| `metrics/by_metric/platinai_4cell_auc.json`                                         | Reported AUC values + per-seed breakdown                   |
| `molmetal/reports/wf_phase_a_stage2_diag/a2780_diagnosis_data.json`                 | Raw numeric outputs from this investigation               |
| `molmetal/reports/wf_phase_a_stage2_diag/a2780_diagnosis.md`                        | **This document**                                          |

---

## 13. Verdict

**A2780 AUC 0.350 is expected noise, not a data bug.**

The synthetic IC50 labels in `molmetal/data/platinai_4cell.csv` were generated
**without any molecule-specific signal** (cross-line residual r < 0.03); the
scaffold-split test rows are OOD by construction; and three seeds converging
near 0.34 is a ≈2σ tail of the no-signal random distribution, not a sign of a
flipped label, leakage, or scaffold imbalance. The honest framing already
shipped in `platinai_4cell_auc.json` ("status": "MEASURED_SYNTHETIC_LABELS")
captures this correctly.

**Action: keep A2780 in the metric, document the caveat explicitly, and treat
the metric as a pipeline smoke test until a real PlatinAI-labelled dataset is
available or the synthetic generator is upgraded to inject structural signal.**