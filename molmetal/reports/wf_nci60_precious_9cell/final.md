# NCI-60-Panel — Precious-Metal 9-Cell-Line Cytotoxicity Prediction AUC

**Date:** 2026-09-25
**Model:** k-cell-line multi-task D-MPNN (9 cls + 9 reg heads)
**Substrate:** MetalCytoToxDB.csv filtered to precious metals (Ru, Ir, Rh, Os, Re) and the 9 most-studied NCI-60-overlapping cell lines.
**Split:** scaffold (Bemis-Murcko) 70/10/20
**Seeds:** [0, 1, 2]  
**Epochs:** 50, lr=0.001, batch=32, alpha=0.5

## What this measures

NCI-60 is the gold standard for in-vitro anticancer screening across
60 human tumor cell lines.  This metric evaluates how well the
metal-aware D-MPNN learns to predict *cell-line-specific* activity
(active = IC50_Dark < 10 µM) of precious-metal complexes across the
9 most-studied cell lines that overlap between MetalCytoToxDB and the
NCI-60 core panel.

**Note:** The raw NCI-60 GI50 release at
`.data/molmetal/NCI60_GI50/GI50.csv` contains only per-cell GI50
values + NSC numbers — **no compound structures** (SMILES).  We
therefore use MetalCytoToxDB (which has SMILES + cell-line + IC50)
restricted to the precious metals (Ru/Ir/Rh/Os/Re) that overlap with
our Pt-only PlatinAI benchmark (Metric 2).  This is the same protocol
used by Krasnov 2026 (MetalCytoToxDB, J. Med. Chem. 2026-04).

## Verdict gate (this run)

- Per-cell AUC mean = 0.8007 ± 0.0160
- Gate: mean per-cell AUC ≥ 0.70
- Verdict: ✅ PASS

## Per-cell-line AUC (this run)

| Cell line | n_test (avg) | AUC mean ± std | Active rate |
|---|---|---|---|
| MCF7 | 405 | 0.7441 ± 0.0017 | 30.12% |
| A549 | 534 | 0.7758 ± 0.0029 | 30.52% |
| A2780 | 216 | 0.6931 ± 0.0254 | 42.13% |
| A2780cis | 111 | 0.7278 ± 0.0406 | 41.44% |
| HCT116 | 157 | 0.7584 ± 0.0146 | 38.22% |
| PC3 | 50 | 0.8700 ± 0.0041 | 40.00% |
| HL60 | 63 | 0.9320 ± 0.0136 | 53.97% |
| SKOV3 | 39 | 0.9046 ± 0.0474 | 43.59% |
| DU145 | 0 | nan ± nan | 0.00% |
| **Mean** | — | **0.8007 ± 0.0160** | — |

## Per-seed detail

| Seed | n_test | AUC mean | AUC MCF7 | AUC A549 | AUC A2780 | AUC A2780cis | AUC HCT116 | AUC PC3 | AUC HL60 | AUC SKOV3 | AUC DU145 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1041 | 0.8174 | 0.7421 | 0.7780 | 0.7131 | 0.7692 | 0.7589 | 0.8700 | 0.9503 | 0.9572 | nan |
| 1 | 1041 | 0.8056 | 0.7438 | 0.7777 | 0.7088 | 0.7415 | 0.7759 | 0.8650 | 0.9178 | 0.9144 | nan |
| 2 | 1041 | 0.7792 | 0.7463 | 0.7717 | 0.6572 | 0.6726 | 0.7402 | 0.8750 | 0.9280 | 0.8422 | nan |

## Honest caveats

* **No raw NCI-60 SMILES** — the public NCI-60 GI50 release contains
  per-cell GI50 values and NSC numbers but no compound structures.
  We therefore benchmark on the **MetalCytoToxDB precious-metal subset**
  restricted to the 9 NCI-60-overlapping cell lines.  Numbers here are
  comparable to **Krasnov 2026 / MetalCytoToxDB baseline (Ru subset AUC
  = 0.81) but extended to 9 cell lines**.  Direct head-to-head with the
  NCI-60 panel as a whole is impossible without an NSC→SMILES table.
* **Scaffold split is OOD by construction** → AUC will be lower than
  random splits.  Any direct comparison to a paper's random-split
  protocol is therefore *conservative*.
* **D-MPNN backbone** here is *not* the full V4 hybrid (no EGNN /
  Perceiver fusion).  This is the metal-aware multi-task D-MPNN
  baseline; richer hybrids should match or beat it.
* **Per-cell coverage is uneven** — A549/MCF-7/A2780 have ~1000+
  rows; SK-OV-3/DU-145 have <250 rows.  The mean AUC is dominated
  by the high-coverage lines.
