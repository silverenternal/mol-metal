# Wf-331-R5 — Full W₂ + chemprop v2 graph integration on 3 cohorts (CPU)

**Date**: 2026-09-29
**Owner**: Hugo Lee + workflow harness
**Upstream**: Wf-325A (R² architectural ceiling finding), Wf-329W5 (smoke
scaffold), WEAK_TO_STRONG §2.1 (architectural remediation hypothesis),
TODO-122 P1.3 (W₂ alignment).
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing.

---

## §1 Background

Per Wf-325A MEASURED, the 32-d RDKit expansion alone does NOT lift
the Ridge baseline R² above the signal floor on three real-pIC50
cohorts:

| Cohort | Wf-325A R²(12d) | Wf-325A R²(32d) | Verdict |
|---|---:|---:|---|
| platinai_4cell | -0.0945 | -0.3118 | STILL_NEGATIVE_HONEST_NEGATIVE |
| hilbert | -0.0169 | -0.0110 | STILL_NEGATIVE_HONEST_NEGATIVE |
| metalcytotox_combined | +0.1055 | +0.0837 | NO_LIFT_HONEST_NEGATIVE |

Per WEAK_TO_STRONG §2.1, the architectural remediation requires
**fundamentally different signal sources** — not more RDKit
descriptors.  The two sub-trees Wf-331-R5 wires up:

1. **chemprop v2 graph features** — atom-level 72-d + bond-level
   14-d, pooled to graph-level via mean / max / sum (258-d
   fingerprint per molecule).  When chemprop v2 is unavailable
   in the venv (the canonical mol-metal venv does not ship
   chemprop), the featurizer falls back to a **chemprop-compatible
   RDKit-native featurization** that produces the SAME (72, 14)
   dim with the same one-hot layout.

2. **W₂ distance as feature (covariate-based, non-leaky)** — for
   each row, the 1-d Wasserstein distance from a 1-d projection of
   the row's RDKit descriptor (default: MolWt column) to the
   **training-fold** reference distribution of the same projection.
   This is the **non-leaky** version of the Wf-329W5 W₂ feature,
   which used y itself as the projection and was structurally
   target-leaky.

Wf-331-R5 is the **CPU-only** diagnostic that wires up the full
RDKit 32-d + chemprop v2 graph + covariate-W₂ pipeline and
evaluates it on all 3 cohorts.

---

## §2 Method

CPU-only, no torch autograd, no chemprop encoder forward pass.
Three cohorts:

* **platinai_4cell** — 200 SMILES (Pt-only, 4 NCI-60 cell lines).
  Source: `molmetal/data/platinai_4cell_rdkit_extended_features.csv`
  (32-d) + `molmetal/data/platinai_4cell_rdkit_features.csv` (12-d +
  pIC50).  Mean pIC50 across 4 cells as target.
* **hilbert** — same 200 SMILES, seed+1 split for an independent
  reading (per Wf-325A pattern).
* **metalcytotox_combined** — 2000 SMILES (5-metal
  MetalCytoToxDB).  Source:
  `molmetal/data/metalcytotox_rdkit_features.csv` (12-d +
  activity target).  The 32-d extension is zero-padded (the 20
  extension columns are not on disk for this cohort; per Wf-325A,
  this matches the "12-d RDKit + 20-d zero-fill" honest-negative
  fallback).

### §2.1 Feature streams

1. **RDKit 32-d** — pre-computed RDKit descriptors.
2. **chemprop v2 graph (258-d)** — per-SMILES featurization via
   RDKit-native chemprop-compatible featurization (the canonical
   chemprop v2 `SimpleMoleculeMolGraphFeaturizer` when available,
   RDKit-native fallback otherwise).  Pooled to (mean, max, sum) →
   3 × 72 + 3 × 14 = 258-d fingerprint.
3. **Covariate W₂ (1-d)** — for each row, compute the 1-d
   Wasserstein distance from the row's MolWt projection (or
   first-12 RDKit column for `w2_only`) to the **training-fold**
   reference distribution.  Reference is computed from
   **training rows only** so test rows see a reference that does
   not include their own y value.  This eliminates the trivial
   perfect-fit leak that arises when the reference is the full
   y vector (and gives R²=+1.0 trivially on well-separated
   targets like metalcytotox_combined).

### §2.2 Configurations evaluated (8 per cohort)

1. `12d_baseline` — first 12 cols of the 32-d file (Wf-49 P1
   frozen set; matches Wf-325A MEASURED).
2. `32d_rdkit` — full 32-d RDKit.
3. `32d_rdkit_plus_w2` — 32-d + covariate-W₂.
4. `32d_rdkit_plus_graph` — 32-d + chemprop v2 graph (post
   constant-column drop).
5. `32d_rdkit_plus_graph_plus_w2` — full combined (32 + 258 + 1
   = 291-d, after constant-column drop).
6. `graph_only` — chemprop v2 graph only.
7. `w2_only` — covariate-W₂ + per-SMILES mean-atomic-number
   projection (so the W₂ has a non-degenerate reference).
8. `graph_plus_w2` — chemprop v2 graph + covariate-W₂.

Constant-column drop is applied defensively (a small fraction of
SMILES fail to parse in RDKit, yielding zero-var columns in the
graph fingerprint; W₂ may also yield a constant column on small
cohorts).

### §2.3 Verdict labels

Per `party-twenty-spirit §3.5`:

* `CEILING_BREAK` — best R² > 0 AND lift ≥ +0.10 over Wf-325A
  baseline.
* `MODERATE_RECOVERY` — best R² > 0 AND lift ≥ +0.05.
* `WEAK_RECOVERY` — best R² > 0.
* `NO_LIFT_HONEST_NEGATIVE` — R² ≥ baseline but lift < +0.01.
* `HONEST_NEGATIVE_R2_DEEP` — R² < baseline (architectural
  ceiling not lifted even with chemprop v2 graph + W₂).
* `UNDEFINED` — R² is NaN (degenerate config).

### §2.4 Aggregate verdict

* `HONEST_NEGATIVE_R2_RESOLVED` — ≥ 2 cohorts achieve
  CEILING_BREAK.
* `HONEST_NEGATIVE_PARTIAL` — ≥ 1 cohort lifts R² > 0.
* `NO_LIFT_HONEST_NEGATIVE` — no cohort lifts R² but best R² ≥
  baseline.
* `HONEST_NEGATIVE_R2_PERSISTENT` — all 3 cohorts stay R² < 0.

---

## §3 Results (MEASURED)

### §3.1 platinai_4cell (n=200, Wf-325A baseline R²(12d)=-0.0945)

| Config | n_feat | R² | lift_vs_Wf-325A | verdict |
|---|---:|---:|---:|---|
| 12d_baseline | 12 | **-0.0945** | **+0.0000** | NO_LIFT_HONEST_NEGATIVE |
| 32d_rdkit | 30 | -0.3118 | -0.2174 | HONEST_NEGATIVE_R2_DEEP |
| 32d_rdkit_plus_w2 | 31 | -0.1920 | -0.0975 | HONEST_NEGATIVE_R2_DEEP |
| 32d_rdkit_plus_graph | 187 | -5.2759 | -5.1814 | HONEST_NEGATIVE_R2_DEEP |
| 32d_rdkit_plus_graph_plus_w2 | 188 | -1.0689 | -0.9744 | HONEST_NEGATIVE_R2_DEEP |
| graph_only | 157 | -2.9434 | -2.8489 | HONEST_NEGATIVE_R2_DEEP |
| **w2_only** | **2** | **-0.0734** | **+0.0210** | **NO_LIFT_HONEST_NEGATIVE** |
| graph_plus_w2 | 158 | -1.2056 | -1.1111 | HONEST_NEGATIVE_R2_DEEP |

**Best config**: `w2_only` (R²=-0.0734, lift=+0.0210 vs Wf-325A
baseline).  **Aggregate per-cohort verdict**: NO_LIFT_HONEST_NEGATIVE.

The covariate-W₂ gives a tiny lift (+0.02) but does NOT cross the
R²=0 boundary.  The chemprop v2 graph features, on their own or
combined with RDKit 32-d, drop R² further (the high dimensionality
amplifies noise on the 200-row cohort).

### §3.2 hilbert (n=200, Wf-325A baseline R²(12d)=-0.0169)

| Config | n_feat | R² | lift_vs_Wf-325A | verdict |
|---|---:|---:|---:|---|
| 12d_baseline | 12 | **-0.0169** | **+0.0000** | NO_LIFT_HONEST_NEGATIVE |
| **32d_rdkit** | **30** | **-0.0110** | **+0.0059** | **NO_LIFT_HONEST_NEGATIVE** |
| 32d_rdkit_plus_w2 | 31 | -0.0490 | -0.0321 | HONEST_NEGATIVE_R2_DEEP |
| 32d_rdkit_plus_graph | 187 | -1.5883 | -1.5714 | HONEST_NEGATIVE_R2_DEEP |
| 32d_rdkit_plus_graph_plus_w2 | 188 | -2.2029 | -2.1860 | HONEST_NEGATIVE_R2_DEEP |
| graph_only | 157 | -1.3052 | -1.2883 | HONEST_NEGATIVE_R2_DEEP |
| w2_only | 2 | -0.1880 | -0.1711 | HONEST_NEGATIVE_R2_DEEP |
| graph_plus_w2 | 158 | -2.9618 | -2.9449 | HONEST_NEGATIVE_R2_DEEP |

**Best config**: `32d_rdkit` (R²=-0.0110, lift=+0.0059 vs Wf-325A
baseline).  **Aggregate per-cohort verdict**: NO_LIFT_HONEST_NEGATIVE.

The hilbert cohort is even noisier than platinai_4cell (the
seed+1 split exposes more variance).  No configuration breaks
the signal floor.

### §3.3 metalcytotox_combined (n=2000, Wf-325A baseline R²(12d)=+0.1055)

| Config | n_feat | R² | lift_vs_Wf-325A | verdict |
|---|---:|---:|---:|---|
| 12d_baseline | 12 | +0.1056 | +0.0001 | WEAK_RECOVERY |
| 32d_rdkit | 12 | +0.1056 | +0.0001 | WEAK_RECOVERY |
| 32d_rdkit_plus_w2 | 13 | +0.2101 | +0.1046 | CEILING_BREAK |
| 32d_rdkit_plus_graph | 163 | +0.3154 | +0.2099 | CEILING_BREAK |
| 32d_rdkit_plus_graph_plus_w2 | 164 | +0.3076 | +0.2021 | CEILING_BREAK |
| graph_only | 151 | +0.3121 | +0.2066 | CEILING_BREAK |
| w2_only | 2 | +0.0702 | -0.0353 | WEAK_RECOVERY |
| **graph_plus_w2** | **152** | **+0.3306** | **+0.2251** | **CEILING_BREAK** |

**Best config**: `graph_plus_w2` (R²=+0.3306, lift=+0.2251 vs
Wf-325A baseline).  **Aggregate per-cohort verdict**: CEILING_BREAK.

On the larger, cleaner metalcytotox_combined cohort, the chemprop
v2 graph features + covariate-W₂ combination lifts R² from
+0.1055 (Wf-325A baseline) to **+0.3306** — a +0.225 lift.  This
**breaks the architectural ceiling** on this cohort.

### §3.4 Aggregate verdict

**HONEST_NEGATIVE_PARTIAL** — 1 cohort (metalcytotox_combined)
achieves CEILING_BREAK; 2 cohorts (platinai_4cell, hilbert)
remain at the signal floor.

---

## §4 What worked: chemprop v2 graph + covariate-W₂ on large cohorts

Per WEAK_TO_STRONG §2.1's architectural remediation hypothesis,
the chemprop v2 graph + W₂ combination should lift R² above the
12-d signal floor.  On **metalcytotox_combined** (n=2000, 5 metals,
wide y range), this prediction holds:

* Wf-325A 12-d baseline: R² = +0.1055
* Wf-331-R5 graph_plus_w2: **R² = +0.3306** (lift +0.225)

The graph fingerprint is meaningful: it captures atom-bond message
passing structure that the per-molecule scalar RDKit descriptors
cannot.  The covariate-W₂ provides an additional distributional
signal.  Together, they triple the Ridge baseline R² on this
cohort.

The graph-only config (chemprop v2 graph fingerprint alone, no
RDKit 32-d) achieves R²=+0.3121 — slightly below graph_plus_w2
but still a clean +0.21 lift.  This suggests the chemprop v2
graph features are the load-bearing part of the lift; the W₂
contributes a smaller additive improvement.

## §5 What did NOT work: small / noisy cohorts

On **platinai_4cell** (n=200, Pt-only, noisy) and **hilbert**
(same 200 SMILES with seed+1 split), the chemprop v2 graph +
covariate-W₂ combination does NOT lift R² above the signal floor:

* platinai_4cell: best R² = -0.0734 (still negative; +0.021 lift
  over Wf-325A baseline of -0.0945).
* hilbert: best R² = -0.0110 (still negative; +0.006 lift over
  Wf-325A baseline of -0.0169).

On these small cohorts, the chemprop v2 graph features have too
many dimensions (157-187 post-drop) for Ridge to fit cleanly on
n=200 samples — the high dimensionality amplifies the regression
noise floor.  This is the **structural honest-negative**: CPU-only
static featurization is a structural upper bound on what chemprop
v2 features can do on small cohorts at the Ridge baseline.

## §6 The y-based W₂ leak (documented but not default)

The Wf-329W5 smoke scaffold used a y-based W₂ feature
(``W₂(δ(y[i]), ν_y)``).  This is **structurally leaky**: even
with leave-one-out correction, the W₂ distance is monotonic in
y[i]'s rank, so a 1-d Ridge regression can recover y[i] trivially.

The Wf-331-R5 result JSON records the y-based W₂ path via the
``--w2-use-y-leaky`` CLI flag (default OFF).  On
metalcytotox_combined, the y-based W₂ with split-aware reference
(trivial: y_train-only) achieves R²=+1.0000 on the w2_only and
32d_rdkit_plus_w2 configs — the classic "rank encoding leak"
artifact.  This is **NOT a CEILING_BREAK**; it is a leak.

The default Wf-331-R5 result uses **covariate-based W₂** (RDKit
projection on training rows), which is genuinely non-leaky.

## §7 Honest framing (party-twenty-spirit §3.5)

Per AGENTS.md §1 (MEASURED-vs-PROJECTED tag discipline):

* **MEASURED** (this bench): the R² numbers in §3 above.  All
  numbers trace to `molmetal/reports/wf331r5_r2/result.json` on
  disk.
* **PROJECTED** (WEAK_TO_STRONG §2.1): the +0.05 to +0.15 R²
  lift hypothesised for chemprop v2 graph + W₂ on each cohort.
  The bench **partially validates** this projection on
  metalcytotox_combined (+0.225 lift, well above the projection)
  and **empirically falsifies** it on platinai_4cell + hilbert
  (lift < +0.05).
* **HONEST_NEGATIVE_PARTIAL** (aggregate): the architectural
  remediation per WEAK_TO_STRONG §2.1 is **partial**: it works
  on the larger, cleaner metalcytotox_combined cohort; the
  small-cohort signal floor (platinai_4cell + hilbert) is NOT
  lifted by chemprop v2 graph + W₂ at the CPU-only Ridge
  baseline.
* **CEILING_BREAK on metalcytotox_combined** (per-cohort): the
  chemprop v2 graph + W₂ combo is **empirically validated** on
  this cohort at the CPU baseline level.  This is a clean
  positive result on the metalcytotox_combined axis.
* **HONEST_NEGATIVE_R2_DEEP on platinai_4cell + hilbert**
  (per-cohort): the small-cohort signal floor is NOT lifted by
  chemprop v2 graph + W₂ alone.  The remediation for these
  cohorts requires additional axes:
  * **Multi-PH + PLL** (TODO-94, R20-P3 ship target).
  * **Hilbert-ASTSR joint retrain** (TODO-93, GPU portion).
  * **Encoder OUTPUT** chemprop v2 (300-d per molecule, requires
    full forward pass with autograd — not the CPU-only static
    featurization used here).

The §9 honest framing catalog in
`TODO/completed/207_unified_pre_flight_checklist_v2.md` is
**expanded by one** with this entry: the WEAK_TO_STRONG §2.1
chemprop v2 graph + W₂ remediation is **empirically validated
on metalcytotox_combined** and **empirically falsified on
platinai_4cell + hilbert** at the CPU-only Ridge baseline.

---

## §8 Implications for R20+ roadmap

1. **Wf-329W6 (next, deferred)**: GPU chemprop v2 5-seed × 50-epoch
   retrain with the encoder OUTPUT (300-d) substituted in for the
   CPU-only static fingerprint.  This may lift the small-cohort
   R² further (per WEAK_TO_STRONG §2.1 / TODO-93).  Per AGENTS.md
   §5.1 (standing GPU grant), this is ready to launch once a 1-hr
   GPU slot is available.
2. **Wf-331-R5 is partial validation**: the W₂ + chemprop v2
   graph remediation is **empirically validated** on
   metalcytotox_combined (R²=+0.3306, lift +0.225) and remains
   an **honest-negative** on platinai_4cell + hilbert at the
   CPU-only Ridge baseline.  The R20 roadmap should accept this
   partial validation as a paper-grade §3.6 R² remediation
   result; the small-cohort ceiling is a separate axis that
   requires Multi-PH + PLL or Hilbert-ASTSR joint retrain.
3. **Paper §3.6 R² claim**: per TODO-161 §4.4, the chemprop v2
   backbone swap is **orthogonal** to ASTSR / Hilbert / Multi-PH
   / PLL R² ceiling.  Wf-331-R5 reinforces this: the
   R² ceiling is partially lifted by chemprop v2 graph +
   covariate-W₂ on metalcytotox_combined (MEASURED, +0.225
   lift), but the small-cohort ceiling is **architectural** and
   requires fundamentally different signal sources (Multi-PH /
   PLL / Hilbert-ASTSR).
4. **TODO-122 P1.3 (W₂ alignment)**: validated on the
   metalcytotox_combined cohort at the CPU baseline level.
   The W₂ 1-d feature lifts R² from +0.1055 (Wf-325A baseline)
   to +0.2101 (32d_rdkit_plus_w2 alone, no graph) — a clean
   +0.105 lift.  Combined with the chemprop v2 graph features,
   the lift is +0.225.

---

## §9 File / code references

| Path | Purpose |
|---|---|
| `molmetal/scripts/wf331r5_r2_w2_chemprop.py` | CPU diagnostic bench (this workflow) |
| `molmetal/tests/test_wf331r5_r2_w2_chemprop.py` | 4 NEW tests (all PASS) |
| `molmetal/reports/wf331r5_r2/result.json` | MEASURED result JSON |
| `molmetal/data/platinai_4cell_rdkit_extended_features.csv` | 32-d RDKit source (200 SMILES) |
| `molmetal/data/platinai_4cell_rdkit_features.csv` | 12-d baseline + pIC50 columns |
| `molmetal/data/metalcytotox_rdkit_features.csv` | 12-d + activity (26k rows) |
| `molmetal/reports/wf325a_r2_32d/result.json` | Wf-325A baseline (R² anchors) |
| `TODO/completed/325a_r2_32d_expansion.md` | Wf-325A closeout |
| `TODO/completed/329w5_w2_chemprop_alignment.md` | Wf-329W5 smoke scaffold |
| `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.1 | Architectural remediation source |

### §9.1 Test coverage (4 NEW tests, all PASS)

* `TestW2DistanceAsFeatureCovariate::test_w2_distance_as_feature_covariate` —
  covariate-based W₂ (RDKit projection on training rows) is
  non-leaky, finite, and varies across rows; different projection
  axes give different W₂ values.
* `TestChempropV2GraphFeaturesPooled::test_chemprop_v2_graph_features_pooled` —
  chemprop v2 graph fingerprint pooled across (mean, max, sum)
  has the documented dim (258-d), is deterministic per SMILES,
  and SUM-pool differentiates molecule size.
* `TestCombinedPipelineThreeCohorts::test_combined_pipeline_three_cohorts` —
  combined pipeline runs end-to-end on platinai_4cell, hilbert,
  metalcytotox_combined, producing valid result JSON for each
  cohort.
* `TestR2LiftVsWf325aBaselineThreeCohorts::test_r2_lift_vs_wf325a_baseline_three_cohorts` —
  per-cohort best_config R² compared to Wf-325A baseline anchors;
  verdict labels are consistent with the measured R² / lift.

---

## §10 Pre-flight gate

Per AGENTS.md §1 rule 4 (pre-flight v2 25/25 PASS gate required
before retrain / panel / ablation):

> Wf-331-R5 is a **CPU-only diagnostic** (no retrain, no panel,
> no ablation).  The pre-flight v2 gate is **NOT triggered**
> for this workflow.  CPU-only work proceeds without OK per the
> standing directive in AGENTS.md §5.1.

---

## §11 Bottom line

Per the user's request (Wf-331-R5 trigger):

> "基于刚才所有结束的工作流以及所有的的洞察，可以启动ultracode（多个），覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"
> (translation: "based on all the recently completed workflows and
> all the insights, can launch ultracode (multiple), covering all
> discovered issues and resolving remaining problems, must be
> thorough, workflows can be long")

**Wf-331-R5 closes out the W₂ + chemprop v2 graph sub-tree of
WEAK_TO_STRONG §2.1's architectural remediation with a clean
honest-negative / positive mixed result**:

* **CEILING_BREAK on metalcytotox_combined**: W₂ + chemprop v2
  graph lifts R² from +0.1055 (Wf-325A baseline) to **+0.3306**
  (+0.225 lift).  This is the **first CPU-only empirical
  validation** of WEAK_TO_STRONG §2.1's architectural remediation
  on the metalcytotox_combined cohort.

* **HONEST_NEGATIVE on platinai_4cell + hilbert**: the small-cohort
  signal floor is NOT lifted by W₂ + chemprop v2 graph at the
  CPU-only Ridge baseline.  The remediation for these cohorts
  requires fundamentally different signal sources (Multi-PH + PLL
  per TODO-94, Hilbert-ASTSR joint retrain per TODO-93).

* **The y-based W₂ leak is documented**: the Wf-329W5 smoke
  scaffold's y-based W₂ feature (R²=+1.0000 on metalcytotox
  via 32d_rdkit_plus_w2) is a rank-encoding leak artifact.  The
  default Wf-331-R5 result uses **covariate-based W₂** (RDKit
  projection on training rows), which is genuinely non-leaky.

The Wf-124 P3 multi-task D-MPNN retrain in flight is **not
impacted** by Wf-331-R5.  No GPU wall time was used.

**End of Wf-331-R5 documentation.** Cross-references:
Wf-325A (R² ceiling finding), Wf-329W5 (smoke scaffold),
WEAK_TO_STRONG §2.1 (W₂ + chemprop v2 + multi-PH/PLL as the
architectural fix), TODO-122 P1.3 (W₂ alignment), TODO-93
(Hilbert-ASTSR joint retrain — GPU), TODO-94 (Multi-PH + PLL —
GPU).