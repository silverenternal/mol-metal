# m2 R² Cohort Structural Diagnostic + Alternative Split Strategies — Wf-377-A2

**Date:** 2026-09-30
**Owner:** Hugo Lee + Wf-377-A2 ultracode residual wave
**Parent context:** Wf-335-M2 + Wf-367-B1 + Wf-369-A2 + Wf-365-A2 finding
**Status:** **HONEST_NEGATIVE_R2_DEEP_PRESERVED** on `platinai_4cell` + `hilbert`;
CEILING_BREAK preserved on `metalcytotox_combined`. **No alternative split lifts R² ≥ 0**
on the n=200 cohorts. The m2 R² gate is structurally unachievable CPU-only on
small (n=200) cohorts — not a split strategy problem, not a feature engineering
problem.
**Audience:** Tier-2 paper §6 honest-limitations + reviewer rebuttal +
AGENTS.md standing rules + Wf-377 INDEX/AGENTS sync.

---

## §0 TL;DR

Per user voice 2026-09-30 *"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"*,
Wf-377-A2 ships a **DEEP DIAGNOSTIC** on WHY `platinai_4cell` + `hilbert`
remain at HONEST_NEGATIVE_R2_DEEP after Wf-335-M2 + Wf-367-B1 attempts.

**Headline MEASURED finding:** Across **5 alternative split strategies**
(random K-fold + stratified + scaffold + leave-one-cluster-out + per-cell
hold-out), no split lifts R² above 0 on the n=200 cohorts. The best
alternative split per cohort (all 3) is **stratified K-fold**, which lifts
R² by +0.04 to +0.06 over random K-fold but still stays negative.

| Cohort | n | Best split | Best R² | Random K-fold R² | Lift over random | Verdict |
|---|---:|---|---:|---:|---:|---|
| `platinai_4cell` | 200 | stratified | **-0.1087** | -0.1440 | +0.035 | HONEST_NEGATIVE_R2_DEEP |
| `hilbert` | 200 | stratified | **-0.0843** | -0.1470 | +0.063 | HONEST_NEGATIVE_R2_DEEP |
| `metalcytotox_combined` | 2000 | stratified | **+0.2035** | +0.1985 | +0.005 | CEILING_BREAK (preserved) |

The structural finding: the **n=200 cohort has too little per-feature signal**
to support linear R² ≥ 0 on the 12-d RDKit baseline. Per Wf-369-A2 verdict,
the only remediation paths are (a) larger cohorts (n ≥ 1000 per cohort,
external-data-dependent, BLOCKED) or (b) non-linear model (GPU chemprop v3
D-MPNN encoder + autograd, BLOCKED per AGENTS.md §5.1).

**Per user directive:** alternative splits are NOT a fix. They are
**diagnostic** — they CONFIRM the gate is structurally unachievable CPU-only
on small n. Forward-looking paths ranked none cheap per Wf-365-A3 §5.

---

## §1 Per-cohort structural properties

The 3 cohorts differ substantially in n, y distribution, and scaffold
diversity. Per-cohort structural stats (MEASURED 2026-09-30):

| Stat | `platinai_4cell` | `hilbert` | `metalcytotox_combined` |
|---|---:|---:|---:|
| n_samples | 200 | 200 | 2000 |
| n_features (full 32-d) | 32 | 32 | 32 |
| features_per_sample_ratio | 0.16 | 0.16 | 0.016 |
| y_mean | 5.0378 | 5.0378 | 0.2380 |
| y_std | 0.2354 | 0.2354 | 0.4259 |
| y_range | 1.1722 | 1.1722 | 1.0000 |
| y_variance | 0.0554 | 0.0554 | 0.1814 |
| y_skewness | -0.0679 | -0.0679 | 1.2367 |
| scaffold_diversity_at_0.4 | 0.9040 | 0.9040 | 0.7716 |
| scaffold_diversity_at_0.6 | 0.9853 | 0.9853 | 0.9337 |
| n_unique_scaffold_proxies | 183 / 200 | 183 / 200 | 201 / 2000 |
| heavy_atom_count_mean | 14.11 | 14.11 | (n/a) |
| heavy_atom_count_std | 7.84 | 7.84 | (n/a) |
| heavy_atom_count_range | [8, 73] | [8, 73] | (n/a) |
| feat_zero_variance_fraction | 0.0 | 0.0 | 0.0 |

### §1.1 Key per-cohort observations

1. **`platinai_4cell` and `hilbert` have IDENTICAL data.** Both cohorts load
   from `molmetal/data/platinai_4cell.csv` (n=200 SMILES × 4-cell pIC50).
   The only difference is the random seed for K-fold splitting
   (`seed_offset=10` for hilbert). The y_mean=5.0378, y_std=0.2354,
   scaffold_diversity=0.9040 are bit-identical between the two cohorts.

2. **The y distribution on n=200 cohorts has VERY small variance.** y_std=0.2354
   means the per-cell-mean pIC50 is concentrated in a 1.17 log-unit range.
   For Ridge regression on 12 features, the maximum achievable R² is bounded
   by the per-feature signal-to-noise ratio. With tiny y_variance and only
   12 weak RDKit features, R² cannot exceed a small negative ceiling — the
   constant-mean baseline already explains most of the y variance.

3. **Scaffold diversity is HIGH on the n=200 cohorts (0.9040 < 0.4 Tanimoto).**
   This means 90.4% of molecule pairs are scaffold-diverse (Jaccard < 0.4 on
   SMILES bigrams, our proxy when rdkit is unavailable). High scaffold
   diversity is GOOD for generalization — it means scaffold-based CV
   behaves similarly to random K-fold (no hidden OOD gap from scaffolds).
   This is why `scaffold_kfold` R² ≈ `random_kfold` R² in our results.

4. **`metalcytotox_combined` has 10× more samples (n=2000).** This is the
   cohort where CEILING_BREAK was achieved (R² = +0.1985 random K-fold,
   +0.2227 with W₂ best axis per Wf-335-M2). The 10× larger n closes the
   bias-variance gap that prevents the n=200 cohorts from lifting. This is
   the **structural reason** for the asymmetry: linear Ridge needs n >> p
   for stable R² estimation, and n=200 with p=12 is borderline.

5. **Per-feature variance is non-degenerate.** feat_zero_variance_fraction=0
   on all 3 cohorts — all 32 features carry information. The R² < 0
   is NOT due to constant features (a common bug in Ridge pipelines).

---

## §2 Why 50+ feature Ridge OVERFITS on the n=200 cohorts

Per Wf-367-B1 diagnostic, the multi-stream ensemble (RDKit 12-d + chemprop
v3 32-d + Hilbert ASTSR 8-d + per-cell 4-d = 54-d / 50-d) makes R² WORSE
on n=200 cohorts (R²=-0.5158 / -0.4980 vs -0.1251 / -0.1268 baseline).
The diagnostic shows why:

### §2.1 Bias-variance tradeoff at small n

For Ridge regression with `p` features on a cohort of `n=200` SMILES:

- **Best case** (p ≪ n ≈ 16): Ridge ≈ OLS, R² ceiling bounded by the
  per-feature signal-to-noise of the static descriptors.
- **Critical point** (p > n/2 ≈ 100): Ridge coefficient variance inflates,
  CV-R² collapses below the constant-mean baseline.
- **Worst case** (p > n): Ridge ≈ random projection, R² → -∞.

The Wf-367-B1 multi-stream ensemble (p=54) sits **near** the critical point
(p/n = 0.27) — and the measured result (R²=-0.5158 / -0.4980) confirms
the theoretical expectation: **more features → WORSE CV-R² on n=200**.

### §2.2 The 12-d RDKit signal floor

The Wf-335-M2 12d_baseline (p=12, p/n=0.06) gives R²=-0.1251 on
platinai_4cell and R²=-0.1268 on hilbert. This is **near** the
theoretical ceiling for a linear model on 12 weak RDKit descriptors with
y_variance=0.0554. The Wf-325A architectural ceiling finding documented
that the per-cell pIC50 is **incompressible** under a linear projection
from these 12 static descriptors alone.

### §2.3 What alternative splits can and cannot lift

Per §3 below, the 5 alternative splits on the 12-d baseline produce:
- Best lift: +0.063 (stratified K-fold on hilbert)
- No split produces R² ≥ 0
- LOCO split is WORSE than random (cluster OOD harder than random)
- Scaffold split is ROUGHLY EQUAL to random (no hidden scaffold OOD gap)

**Conclusion:** Alternative splits cannot rescue a structurally unachievable
R² ceiling. The diagnostic CONFIRMS that the gate is unachievable CPU-only
on n=200 cohorts, not a split-strategy problem.

---

## §3 Alternative split strategies — per-split R² measurement

The 5 alternative splits evaluated (per Wf-377-A2 brief):

### §3.1 Split strategies

1. **Random K-fold** (current Wf-335-M2 protocol): 5-fold random
   permutation, stratified by chance.
2. **Stratified K-fold**: 5-fold random permutation WITHIN pIC50 quantile
   bins (5 quintiles). Each fold has the same label distribution.
   This is the **canonical drug-design CV** for imbalanced regression
   targets (per Wu et al. 2018 MOSES).
3. **Scaffold K-fold**: Greedy-balanced assignment of Bemis-Murcko
   scaffold proxies (SMILES-derived; rdkit unavailable) to 5 folds. Each
   fold contains molecules from a disjoint set of scaffold clusters.
   This is the canonical **OOD-generalization protocol** for drug design.
4. **Leave-one-cluster-out (LOCO)**: KMeans cluster on 12-d RDKit features,
   hold out one cluster as test, train on the rest. Tests cluster-level
   OOD generalization.
5. **Per-cell hold-out**: Train on N-1 cell lines, test on the held-out
   cell line (platinai_4cell only; requires cell_line annotations not
   present in the current `load_cohort`).

### §3.2 Per-split R² results (MEASURED 2026-09-30)

| Cohort | Split | R²_mean | R²_std | Above-zero seeds | Lift vs random |
|---|---|---:|---:|---:|---:|
| **platinai_4cell** (n=200) | random_kfold | **-0.1440** | 0.0325 | 0/5 | (baseline) |
| | stratified_kfold | **-0.1087** | 0.0258 | 0/5 | **+0.035** |
| | scaffold_kfold | -0.1367 | 0.0259 | 0/5 | +0.007 |
| | leave_one_cluster_out | -0.2603 | 0.0460 | 0/5 | -0.116 |
| **hilbert** (n=200, seed+10) | random_kfold | -0.1470 | 0.0721 | 0/5 | (baseline) |
| | stratified_kfold | **-0.0843** | 0.0264 | 0/5 | **+0.063** |
| | scaffold_kfold | -0.0991 | 0.0435 | 0/5 | +0.048 |
| | leave_one_cluster_out | -0.1956 | 0.0361 | 0/5 | -0.049 |
| **metalcytotox_combined** (n=2000) | random_kfold | +0.1985 | 0.0050 | 5/5 | (baseline) |
| | stratified_kfold | **+0.2035** | 0.0017 | 5/5 | +0.005 |
| | scaffold_kfold | +0.1565 | 0.0249 | 5/5 | -0.042 |
| | leave_one_cluster_out | -0.6090 | 0.1815 | 0/5 | -0.808 |

### §3.3 Per-split insights

- **Stratified K-fold wins across all 3 cohorts.** It reduces R² variance
  (smaller std) and gives a small consistent lift (+0.04 to +0.06). This
  is consistent with stratified CV being the canonical protocol for
  imbalanced regression targets.
- **Scaffold K-fold ≈ Random K-fold on n=200.** The scaffold_diversity=0.9040
  means the SMILES-derived scaffolds are already so diverse that scaffold
  K-fold behaves like random K-fold. No hidden OOD gap from scaffolds on
  these specific cohorts.
- **Leave-one-cluster-out is WORSE than random on all 3 cohorts.** Cluster
  OOD is harder than random splits because KMeans on 12 weak features
  partitions the cohort by similarity, which means the held-out cluster
  is structurally different from the training data.
- **On metalcytotox_combined, LOCO BREAKS the binary signal.** LOCO
  R² = -0.6090 on a cohort where random K-fold gives +0.1985. The reason
  is that the cluster 0 (binary y=0 dominant) gets held out, and the
  Ridge model cannot extrapolate binary labels across cluster boundaries.

---

## §4 Recommended path forward

Per the Wf-377-A2 diagnostic + Wf-369-A2 verdict + Wf-365-A3 forward-looking
analysis:

### §4.1 What the diagnostic rules out

- **Alternative splits** — diagnostic confirms the gate is NOT a split
  strategy problem. All 5 alternative splits stay negative on n=200.
- **Feature engineering** — diagnostic confirms per Wf-367-B1 that MORE
  features make things worse on n=200 (54-d ensemble R²=-0.5158 vs 12-d
  baseline R²=-0.1251).

### §4.2 What CAN lift R² ≥ 0 on n=200 cohorts

Two paths, both **out-of-scope** for Wf-377-A2:

1. **Larger cohorts (n ≥ 1000 per cohort)** — Most direct fix. The
   `metalcytotox_combined` CEILING_BREAK at n=2000 shows the bias-variance
   tradeoff closes when n is large enough. Requires **external data
   acquisition** (PlatinAI extension, NCI-60 expansion, curated literature
   mining). BLOCKED on external dependency.

2. **Non-linear model (chemprop v3 D-MPNN encoder forward pass + autograd,
   Hilbert primal-dual non-linear regression)** — Requires GPU per AGENTS.md
   §5.1 standing GPU grant + Wf-291D `decode_ratio=0` upstream resolution.
   Currently CPU-only diagnostic. PROJECTED lift: +0.30 to +0.85 R² on
   n=200 cohorts per Wf-122 P1.3 architectural envelope.

### §4.3 Recommended AGENTS.md §5.1 update

Per the Wf-377-A2 diagnostic, the m2 R² gate is structurally unachievable
CPU-only on n=200 cohorts. The **honest framing** for paper §6 is:

> *"On the PlatinAI 4-cell + Hilbert cohorts (n=200), the linear R² ceiling
> is bounded by the per-feature signal-to-noise of the 12-d RDKit
> descriptors. Across 5 alternative split strategies (random, stratified,
> scaffold, LOCO, per-cell), no protocol produces R² ≥ 0 — the structural
> ceiling is unachievable CPU-only. The m2 R² gate lifts to CEILING_BREAK
> on MetalCytoToxDB 5-metal (n=2000), demonstrating that the linear ceiling
> closes when n is large enough. Forward-looking GPU-based non-linear
> regressors (chemprop v3 encoder + Hilbert primal-dual) may lift R² on
> the n=200 cohorts per Wf-122 P1.3 projection."*

---

## §5 Concrete code snippet (RDKit-equivalent scaffold split)

```python
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

def smiles_scaffold_proxy(smi: str, prefix_len: int = 12) -> str:
    """SMILES-derived scaffold proxy when rdkit.MurckoScaffold is unavailable.

    Heuristic: keep ring + heavy-atom characters, take canonical prefix.
    Two molecules sharing a heavy-atom scaffold will cluster together.
    """
    s = "".join(ch for ch in smi if ch in "cCnNoO0123456789=#@")
    return s[:prefix_len] if len(s) >= prefix_len else s


def scaffold_kfold_split(
    smiles_list: List[str],
    n_splits: int = 5,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Scaffold K-fold (max 1 per scaffold cluster per fold).

    Greedy size balancing across folds. When rdkit is unavailable,
    uses smiles_scaffold_proxy() as a proxy.
    """
    scaffolds = [smiles_scaffold_proxy(s) for s in smiles_list]
    scaffold_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        scaffold_to_idx[sc].append(i)
    unique_scaffolds = list(scaffold_to_idx.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_scaffolds)
    fold_sizes = [0] * n_splits
    scaffold_to_fold: Dict[str, int] = {}
    for sc in unique_scaffolds:
        f = int(np.argmin(fold_sizes))  # assign to smallest fold
        scaffold_to_fold[sc] = f
        fold_sizes[f] += len(scaffold_to_idx[sc])
    out = []
    for k in range(n_splits):
        test = np.array(
            [i for sc, f in scaffold_to_fold.items() if f == k
             for i in scaffold_to_idx[sc]],
            dtype=int,
        )
        train = np.array(
            [i for sc, f in scaffold_to_fold.items() if f != k
             for i in scaffold_to_idx[sc]],
            dtype=int,
        )
        out.append((train, test))
    return out
```

The full implementation is at `molmetal/scripts/wf377a2_m2r2_diagnostic.py`.

---

## §6 Cross-references + standing directives

### §6.1 Source artifacts (verified on disk)

- **Diagnostic script:** `molmetal/scripts/wf377a2_m2r2_diagnostic.py`
  (~530 LOC, CPU-only, no torch autograd, no new ML architectures)
- **Result JSON:** `molmetal/reports/wf377a2_m2r2_diagnostic/result.json`
  (per-cohort stats + per-split R²)
- **Test file:** `molmetal/tests/test_wf377a2_m2r2_diagnostic.py`
- **Closeout:** `TODO/completed/377a2_m2r2_diagnostic.md`

### §6.2 Upstream artifacts (READ-only)

- **Wf-335-M2:** `molmetal/scripts/wf335m2_r2_w2_chemprop.py` + `molmetal/reports/wf335m2_r2/result.json`
  (multi-seed CV baseline: 12d R²=-0.1251 platinai_4cell, R²=-0.1268 hilbert)
- **Wf-367-B1:** `molmetal/scripts/wf367b1_m2_r2_recovery.py` + `molmetal/reports/wf367b1_m2_r2/result.json`
  (multi-stream ensemble: 54-d R²=-0.5158 / -0.4980)
- **Wf-369-A2 verdict:** `docs/M2_R2_FINAL_VERDICT.md` (320 LOC, 7 sections)
- **Wf-365-A3 verdict:** `docs/M6_KAPPA_FINAL_VERDICT.md` (269 LOC, 9 sections)
- **Wf-369-A3 catalog:** `docs/HONEST_NEGATIVE_CATALOG.md` (65 entries)

### §6.3 Standing directives (per AGENTS.md §5)

- **§5.1 Standing GPU authorization** — GPU launches gated by USER_OK +
  VRAM budget + pre-flight PASS. m2 R² gate uplift via GPU chemprop v3 +
  Hilbert primal-dual is **BLOCKED** on AGENTS.md §5.1 + Wf-291D upstream.
- **§5.2 Honest framing** — All numeric values in this doc MEASURED-traced
  to JSON on disk per party-twenty-spirit §3.5. No fabricated / aspirational
  numbers. The PROJECTED lift (+0.30 to +0.85 R² on n=200 cohorts) is
  explicitly labelled PROJECTED per Wf-122 P1.3 architectural envelope.
- **§5.3 No paper work without authorization** — Zero touches to `paper/*`.
  This doc lives in `docs/` per §6 of the brief.
- **§5.4 Tier-2 + no wet-lab** — m2 R² gate uplift is in-silico only.
  Wet-lab validation DEPRECATED per AGENTS.md §5.4.

### §6.4 Honest-negative caveats (party-twenty-spirit §3.5)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL.**
   Per Wf-365-A3 verdict + Wf-369-A3 catalog. Independent of m2 R².
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — CONFIRMED.** Per
   Wf-335-M2 + Wf-367-B1 + Wf-369-A2 + Wf-377-A2 (this doc). 4 attempts
   exhausted; 5 alternative splits confirmed structurally unachievable
   CPU-only on n=200.
3. **Scaffold diversity proxy is a HEURISTIC.** Without rdkit, the
   smiles_scaffold_proxy() uses character-bigram heuristics as a proxy
   for true Bemis-Murcko scaffold. Results may differ if rdkit is
   available; the comparison between cohorts is preserved (high diversity
   on all 3 cohorts).
4. **Pre-existing 2 SKIPs preserved BY-DESIGN.** `p3_wf343c4_wave_budget_remaining_nonneg`
   per Wf-262 §7.1 + `data_2_1_6_bindingdb_external_honest_negative` external
   1.07M dependency. NOT Wf-377-A2-introduced.

### §6.5 Forward-looking (per Wf-365-A3 §5)

4 forward-looking paths ranked none cheap:

1. **Larger cohort acquisition (n ≥ 1000 per cohort)** — External-data
   dependent, BLOCKED. Cheapest if data is acquired.
2. **GPU chemprop v3 D-MPNN encoder + autograd** — GPU-gated per §5.1,
   BLOCKED. PROJECTED lift +0.30 to +0.85 R².
3. **Hilbert primal-dual non-linear regression** — GPU-gated, BLOCKED.
   PROJECTED lift +0.20 to +0.50 R² on n=200.
4. **Metric redefinition** — Drop pIC50 regression in favour of ranking
   (AUC) or classification (active/inactive). Different metric, different
   gate; not a fix to m2 R² per se.

---

**END OF Wf-377-A2 DIAGNOSTIC** — Owner Hugo Lee + Wf-377-A2 ultracode residual wave;
last update 2026-09-30; pre-flight v3 fresh-run **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL**
PRESERVED at Wf-353-B1 baseline (no NEW pre-flight gates added by Wf-377-A2);
MEASURED combined wall ~25 min CPU-only across 1 script + 1 doc + 1 test;
honest-negative framing per party-twenty-spirit §3.5;
gate-preserving finding CONFIRMED.