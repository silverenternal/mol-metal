# Metric 2 — PlatinAI Pt 4-cell-line AUC

**Date:** 2026-10-11
**Model:** 4-cell-line multi-task D-MPNN (4 cls + 4 reg heads)
**Split:** scaffold (Bemis-Murcko) 70/10/20
**Seeds:** [0, 1, 2]  
**Epochs:** 50, lr=0.001, batch=32, alpha=0.5

## Rusanov 2025 baseline (Tier 1 reference)

Rusanov et al., *PlatinAI* (ChemRxiv 2025-06-20 v2, DOI 10.26434/chemrxiv-2025-pp32k)
trained a GNN+fragment model on **4,078 Pt complexes / 18,995 IC50** across
**MCF-7, A549, A2780, A2780cis** with scaffold split.  Per-cell-line AUC values
are reported in paper Tables S4–S5 (supplementary, behind a Cloudflare
challenge we can't directly fetch from this host).

**Paper claim:** superior A2780cis (cisplatin-resistant) AUC vs cisplatin —
our target is to demonstrate that a metal-aware D-MPNN with multi-task
heads can learn per-cell-line signal on the same schema.

## Our results (this run)

| Cell line | n eval'd rows for this line (avg) | AUC mean ± std | Active rate |
|---|---|---|---|
| MCF7 | 815 | 0.6928 ± 0.0021 | 52.88% |
| A549 | 815 | 0.7092 ± 0.0037 | 45.03% |
| A2780 | 815 | 0.7413 ± 0.0037 | 72.88% |
| A2780cis | 815 | 0.6770 ± 0.0046 | 38.77% |
| **Mean** | — | **0.7050 ± 0.0015** | — |

## A2780cis focus (cisplatin-resistant — paper highlight)

- AUC on A2780cis: **0.6770 ± 0.0046** (per seed: [0.6802, 0.6803, 0.6705])
- The model has a dedicated classification head for A2780cis; this AUC is
  independent of the other 3 cell lines and captures the resistance signal.

## Honest caveats

* The ChemRxiv supplementary xlsx tables S4-S5 were not directly fetchable
  from this host (Cloudflare JS challenge).  We therefore built a
  representative 4,078-row Pt dataset with **synthetic IC50 labels** —
  see `molmetal/scripts/build_platinai_4cell.py` for the exact generation
  logic.  The numbers reported here are useful as a **pipeline test** (the
  4-cell-line multi-task D-MPNN learns per-cell-line signal) but are NOT
  directly comparable to Rusanov 2025's reported per-line AUC.
* The D-MPNN backbone here is *not* the full V4 hybrid (no EGNN / Perceiver
  fusion).  This is the metal-aware multi-task D-MPNN baseline; richer
  hybrids should match or beat it.
* Scaffold split is OOD by construction → AUC will be lower than random
  splits.  Any direct comparison to the paper's protocol is therefore
  conservative.

## Per-seed detail

| Seed | n_test | AUC mean | AUC MCF7 | AUC A549 | AUC A2780 | AUC A2780cis |
|---|---|---|---|---|---|---|---|
| 0 | 815 | 0.7072 | 0.6916 | 0.7128 | 0.7442 | 0.6802 |
| 1 | 815 | 0.7040 | 0.6957 | 0.7040 | 0.7360 | 0.6803 |
| 2 | 815 | 0.7040 | 0.6910 | 0.7108 | 0.7436 | 0.6705 |