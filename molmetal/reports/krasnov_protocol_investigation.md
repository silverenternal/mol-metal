# Krasnov 2026 Protocol Investigation

## 1. SMILES Column Name

The SMILES column in `MetalCytoToxDB.csv` is **`SMILES_Ligands`** (not `SMILES`).
There is also a separate **`Counterion`** column containing the counterion SMILES.

---

## 2. Salt-Form / Multi-Component SMILES in Ru Subset

| Metric | Value |
|---|---|
| Ru subset size | 19,135 rows |
| Rows containing `.` (multi-component) | **19,135 / 19,135 (100%)** |
| Average components per entry | 3.32 |
| Median components per entry | 3.0 |
| Min / Max components | 1 / 6 |

The database stores metal complexes as **coordinated ligands + counterions in a single dot-separated SMILES string**.
RDKit fragment analysis (n=2000 Ru sample) shows three fragment size categories:

| Fragment type | Count | Share |
|---|---|---|
| Large ligand (>10 heavy atoms) | 3,333 | 50.3% |
| Small counterion (≤2 heavy atoms) | 1,740 | 26.2% |
| Medium (3-10 heavy atoms) | 1,559 | 23.5% |

Example multi-component Ru SMILES:
```
# 3 components:
Nc1ccc(-c2ccccn2)cc1.N#Cc1ccc(-c2ccccn2)cc1.[Cl-]

# 4 components:
Cc1cc(C)n2ncnc2n1.[Cl-].[Cl-].c1ccc(-c2ccccc2)cc1

# 5 components:
Cn1ccnc1.Cn1ccnc1.Cn1ccnc1.Cn1ccnc1.c1cnc2c(c1)c1nc[nH]c1c1cccnc12
```

The **Counterion column** holds the counterion SMILES separately (e.g., `[Cl-]`, `[BF4]-`, `O=S(=O)([O-])C(F)(F)F`),
confirming that the database intentionally stores ligand + counterion as separate fragments.
Only 6,298 / 19,135 Ru rows have `Counterion` = NaN (neutral complexes).

**Implication for our pipeline**: Our Morgan FP computation currently uses `SMILES_Ligands` as-is.
If we feed dot-separated SMILES to RDKit `MolFromSmiles`, the presence of counterions inflates the fingerprint
with irrelevant atoms. We should either (a) strip counterions before fingerprinting, or (b) use only the
largest fragment as the "active" species. The Krasnov paper explicitly lists "the influence of counterions"
as a **limitation they do not account for**, so their model also does not strip counterions — but they do at
least separate the columns in the source data.

---

## 3. Krasnov 2026 Protocol — What We Know

### Source: LinkedIn post by Jackson Burns (quoting Lev Krasnov, April 2026)

> "We trained **LightGBM classification models** to predict cytotoxicity class — reaching
> ROC-AUC = **0.81 for Ru** and **0.73 for Ir** complexes. The multimetal model enables
> predictions even for metals with limited data (Rh, Os, Re). We also openly discuss the
> current limitations: **ligand geometry and counterion effects are not yet accounted for**,
> the model handles only mononuclear complexes, and selectivity prediction remains out of
> reach for now."

### Additional details from GxP news article (May 2026)

- "An algorithm trained on studies published **before 2024** was successfully tested on **2025 publications**:
  in nine out of ten cases, the model correctly identified active compounds — twice the rate of random selection."
- "The researchers noted that the current version of the models has limitations: they do not account
  for complex geometry (the arrangement of ligands around the metal atom), **the influence of counterions**,
  or selectivity toward healthy cells."

### Reconstructed Protocol

| Parameter | Krasnov 2026 setting | Our current setting |
|---|---|---|
| **Model** | LightGBM (gradient boosted trees) | XGBoost (L1/L2) |
| **Task** | Binary classification (cytotoxicity class) | Regression (pIC50) → binary |
| **Features** | Metal + ligand composition (likely Morgan FP) | Morgan FP (2048 bits, radius 2) |
| **Split strategy** | **TEMPORAL** — train on pre-2024, test on 2025 | **RandomSplitter** (default) |
| **Label** | Cytotoxicity class (threshold unclear) | pIC50 > threshold → active |
| **Reported AUC** | Ru=0.81, Ir=0.73 | Ru=0.90, Ir=0.90 |

### Critical Difference: Split Strategy

The Krasnov paper's key claim is a **temporal external validation**: they train on all available data up to 2024
and evaluate on 2025 publications. This is far more stringent than a random split because:

1. **Temporal drift**: Compounds published in 2025 may use different scaffolds/ligands than those in earlier years.
2. **No data leakage**: No ligand or complex from 2025 appears in training.
3. **Realistic deployment**: In drug discovery, you predict on newly synthesized/measured compounds.

Our **RandomSplitter** allows the same ligand to appear in both train and test (across different cell lines),
which artificially inflates AUC. This is the most likely explanation for the ~0.09 gap in Ru (0.90 vs 0.81).

---

## 4. Implication for Our Baselines

Our current `baseline_ru_xgb.json` reports ROC-AUC = 0.90 with RandomSplitter.
The Krasnov paper's honest number is ROC-AUC = 0.81 with a temporal split.

The gap of ~0.09 is consistent with **train/test SMILES leakage** in the random split:
if the model has seen the same (or a very similar) ligand during training, it can "read off" the answer
rather than generalizing.

**Recommended action**: Our `--split ligand_dedup` or `--split temporal` should give numbers
closer to the Krasnov paper's 0.81. The `scaffold` split is the most stringent and would likely
produce a lower AUC — but that is the correct thing to report if we want to match (or beat) the
paper's methodology.

---

## 5. Recommendation

| Goal | Recommended Split |
|---|---|
| **Paper-comparable** (temporal, matching Krasnov) | `temporal` — train on Year < 2024, test on Year >= 2024 |
| **Stricter / more generalizable** | `scaffold` — Bemis-Murcko scaffold disjoint |
| **Quick diagnostic** | `ligand_dedup` — remove exact SMILES duplicates across splits |

**Caveats**: The exact label threshold (how "active" is defined in the paper) is not yet confirmed.
The paper says "cytotoxicity class" but does not publicly specify the IC50 cutoff.
If our threshold differs from Krasnov's, direct AUC comparison may be misleading.

**Note on counterions**: Neither Krasnov nor our pipeline explicitly strips counterions.
Both models see the full dot-separated SMILES. Any difference in counterion handling would affect
both equally, so it is not the source of the AUC gap.

---

## 6. References

- Krasnov et al., "Machine Learning Approach to Anticancer Activity Prediction of Transition-Metal Complexes Based on a Large-Scale Experimental Database," *J. Med. Chem.* 2026, 69, 8, 8838–8851. DOI: 10.1021/acs.jmedchem.5c02755
- GxP News, "Russian chemists have trained AI to identify promising metals for developing cancer drugs," May 7, 2026. https://gxpnews.net/en/2026/05/russian-chemists-have-trained-ai-to-identify-promising-metals-for-developing-cancer-drugs/
- LinkedIn post by Jackson Burns, April 13, 2026. https://www.linkedin.com/posts/jacksonwarnerburns_machine-learning-approach-to-anticancer-activity-activity-7449443869911375872-rCpi
- Zenodo: MetalCytoToxDB dataset (v2, Sept 2025). https://doi.org/10.5281/zenodo.17106822
