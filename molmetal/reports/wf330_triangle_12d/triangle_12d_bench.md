# Wf-330-T4 triangle convergence lift (F5_RDKIT_2D 12-d)

Per-dataset comparison: 5-d structural features (baseline) vs 12-d RDKit 2D (F5_RDKIT_2D).

**Recovery verdict**: `HONEST_NEGATIVE_PERSISTENT`

| Dataset | 5d hilbert R² | 12d hilbert R² | 17d hilbert R² | 12d-5d lift | 5d var_red | 12d var_red | 12d-5d var_red lift |
|---------|---------------|----------------|----------------|-------------|------------|-------------|--------------------|
| platinai_4cell | 0.0104 | 0.0362 | 0.0463 | +0.0259 | 0.0063 | 0.0036 | -0.0027 |
| metalcytotox | 0.0316 | 0.1227 | 0.1358 | +0.0911 | 0.0054 | 0.0036 | -0.0018 |

## Honest framing

* **MEASURED** (CPU): the 5-d vs 12-d vs 17-d columns above. Baseline 5-d reproduces Wf-329-W2 / Wf-330-F (`mean_hilbert_r2 = 0.4885` on PlatinAI 4-cell). The 12-d path's CPU-side lift is the column `12d-5d lift`.
* **PROJECTED** (GPU per AGENTS.md §5.1): the full R² ≥ 0.6 honesty gate requires the GPU launch in `molmetal/scripts/train_hilbert_astsr_end_to_end.py` (Wf-330-T1/T2 + Wf-331-T1). The CPU bench verifies the **integration pipeline** and the **feature schema**; the GPU retrain provides the R²-lift body.
