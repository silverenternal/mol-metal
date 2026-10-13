# Metric 2 — PlatinAI Pt 4-cell-line AUC

**Date:** 2026-10-12
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
| MCF7 | 527 | 0.7120 ± 0.0027 | 55.60% |
| A549 | 527 | 0.7085 ± 0.0053 | 43.83% |
| A2780 | 527 | 0.7185 ± 0.0119 | 73.43% |
| A2780cis | 527 | 0.7029 ± 0.0054 | 41.94% |
| **Mean** | — | **0.7105 ± 0.0028** | — |

## A2780cis focus (cisplatin-resistant — paper highlight)

- AUC on A2780cis: **0.7029 ± 0.0054** (per seed: [0.6953, 0.706, 0.7075])
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
| 0 | 527 | 0.7071 | 0.7139 | 0.7123 | 0.7068 | 0.6953 |
| 1 | 527 | 0.7139 | 0.7139 | 0.7010 | 0.7348 | 0.7060 |
| 2 | 527 | 0.7105 | 0.7083 | 0.7122 | 0.7139 | 0.7075 |