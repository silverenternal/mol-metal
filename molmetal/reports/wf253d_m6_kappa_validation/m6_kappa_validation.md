# m6_kappa Shape Kappa Index — MetalCytoToxDB pIC50 Validation

**Wf-253-D / TODO-130 G2** — closes the Wf-252 honest-negative (`do NOT claim as learned metallodrug score without validation`).

## Global summary

- N_total rows: **5460**
- N_parsed (RDKit canonical SMILES): **5460**
- Pearson r (m6_kappa vs pIC50): **0.0813**
  - 95% CI: [0.0549, 0.1076]
- Spearman rho: **0.1114**
- m6_kappa mean: **0.5434** (std 0.1157)
- pIC50 mean: **4.6679** (std 0.7076)

## Per-metal breakdown

| metal | n | Pearson r | 95% CI | Spearman rho | m6_kappa mean | pIC50 mean |
|---|---:|---:|---:|---:|---:|---:|
| Ru | 3849 | 0.0970 | [0.0656, 0.1282] | 0.1048 | 0.5417 | 4.6024 |
| Ir | 1250 | 0.0425 | [-0.0129, 0.0978] | 0.1212 | 0.5543 | 4.8791 |
| Os | 87 | 0.1377 | [-0.0751, 0.3386] | 0.3165 | 0.4824 | 5.0146 |
| Rh | 108 | -0.0335 | [-0.2210, 0.1565] | 0.0330 | 0.5076 | 4.6035 |
| Re | 166 | -0.1843 | [-0.3275, -0.0329] | -0.1247 | 0.5571 | 4.4563 |

## Verdict

> **NEGATIVE_WEAK (Shape Kappa has negligible linear association with pIC50)**

Honest framing: the Shape Kappa Index is a *normalised topological index*, not a learned metallodrug score.  A near-zero Pearson r is the EXPECTED outcome — the metric is reported in paper §3 as a side-by-side complement to RDKit Bickerton 2012 QED, NOT as a cytotoxicity predictor.  This validation explicitly closes the Wf-252 `do NOT claim without validation` honest-negative so the metric can ship in paper §3 with full disclosure.
