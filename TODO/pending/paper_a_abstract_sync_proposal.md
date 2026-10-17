# PAPER-SYNC proposal — `paper_a/main.tex` abstract (BLOCKED-ON-USER)

**Status:** PROPOSAL ONLY. Nothing has been written to `paper_a/*` (AGENTS.md §5.3,
TODO-84). Per `feedback_paper_a_edit_scope.md`, the hunks below are offered for
review first.

**Scope:** the abstract only (`paper_a/main.tex` lines 44–79). The body sections
(03_method / 04_results / 05_ablation) carry the same class of defect and are NOT
covered here.

**Root cause (single, not a scattering of typos).** The abstract was written
against `metrics/by_metric/fsma_paper_stats.json`, which is the **chemprop-v2
reference** curve. The in-house D-MPNN migration produced two newer artifacts —
`fsma_inhouse_kcurve_ep50_synth.json` and `fsma_ablations_inhouse_ep50.json` —
and the abstract was never re-synced. So the abstract attributes the *reference*
model's numbers to the *paper's* model. Every stale value below is the chemprop
value; every replacement is MEASURED on the in-house model.

Evidence that the abstract is quoting the reference: the abstract's
`+0.0084 / +0.0379 / +0.0685` are **exactly**
`fsma_inhouse_kcurve_ep50_synth.json → per_k.{20,40,80}.chemprop.mean`
(`0.008426 / 0.037935 / 0.068512`), and `fsma_paper_stats.json →
families.C1_kshot_curve_headonly.p_raw.head_k5 = 0.024881813973408932` is
bit-identical to that file's `per_k.5.chemprop.p_one_sided_gt0`.

---

## Defect 1 — the K-curve numbers (lines 57–60)

> **Current:** "...rising monotonically from $+0.0084$ at $K{=}20$ to $+0.0379$
> at $K{=}40$ and $+0.0685$ at $K{=}80$ (Holm--Bonferroni $p<10^{-6}$; $50/50$
> metal$\times$seed cells positive at $K{=}80$, with all five metals individually
> significant)"

**MEASURED, in-house D-MPNN** — `metrics/by_metric/fsma_inhouse_kcurve_ep50_synth.json`:

| K | delta (`per_k.K.inhouse_ep50.mean`) | Holm (`holm.inhouse_ep50.p_one_sided_holm.K`) | cells>0 (`n_positive`) |
|---|---|---|---|
| 5 | +0.0129 | 0.108 (n.s.) | 29/50 |
| 10 | +0.0141 | 0.108 (n.s.) | 27/50 |
| 20 | **+0.0389** | 0.00753 | 37/50 |
| 40 | **+0.0624** | 7.51e-5 | 40/50 |
| 80 | **+0.0996** | 1.76e-8 | **46/50** |

All five metals individually significant at K=80
(`per_k.80.inhouse_ep50_per_metal.*.p_one_sided_gt0`): Ru 2.9e-3, Ir 3.5e-4,
Rh 7.8e-7, Os 7.1e-4, Re 6.2e-6.

**Two substantive changes, not just digits:**
1. `$p<10^{-6}$` is true only at K=80 (Holm 1.76e-8). At K=20 the Holm p is
   **7.5e-3**. The current phrasing attaches a K=80-strength p-value to the whole
   curve — reword to "family-significant at $K\ge20$".
2. `50/50` cells positive at K=80 is the **chemprop** count. The in-house model is
   **46/50**. (chemprop: `per_k.80.chemprop.n_positive = 50`.)

## Defect 2 — the recovery range (lines 61–62)

> **Current:** "at $K{=}80$ it recovers $74.9$--$98.5\%$ of the within-metal ceiling"

**MEASURED** — `fsma_inhouse_kcurve_ep50.json → per_run[k=80].adapted_auc` ÷
`fsma_paper_stats.json → recovery_to_ceiling.ceiling_within_metal_mean_auc`:

Ru 76.3% / Ir 87.9% / Rh 101.6% / Os 96.2% / Re 91.5% → **76.3–101.6%**.
`>100%` is expected and documented in the ceiling artifact ("the LOMO source pool
adds information").

## Defect 3 — the matched-budget control numbers (lines 62–66)

> **Current:** "a from-scratch model ... scores $-0.0948$ and a model retrained on
> the source pool plus the $K$ ligands scores $-0.1619$, against \fsma's $+0.0171$
> (K$=20$)"

**MEASURED** — `fsma_ablations_inhouse_ep50.json →
families.C2_controls_matched.k20.members`:

| arm | current | MEASURED |
|---|---|---|
| kshot | +0.0171 | **+0.0389** |
| from_scratch | -0.0948 | **-0.0697** |
| concat_retrain | -0.1619 | **-0.1682** |

(The qualitative claim — adaptation beats both controls by a wide margin —
survives; the magnitudes do not.)

## Defect 4 — "the one real knob" / "samplers are indistinguishable" (lines 67–71) — **RESOLVED 2026-10-17**

> **Current:** "a head-only scope removes the only significant negative in the
> full-scope curve --- while the support-set construction rule is *not*: four
> samplers (random, stratified, scaffold-diverse, activity-scaffold) are
> statistically indistinguishable"

**MEASURED** — `fsma_ablations_inhouse_ep50.json`:

* `families.C4_adapt_scope.members.full_k5 = -0.0027`, and `full_k5` is **not** in
  `C4_adapt_scope.significant_after_holm_0.05`. There is **no significant negative
  in the full-scope curve**, so "removes the only significant negative" is stale.
  The defensible version: full-scope is *not* significantly worse at K=5, i.e. the
  scope choice is not what makes the effect appear.
* `families.C3_support_sampler` reports four **one-sample** tests against zero
  (random +0.0359 Holm 0.010; scaffold-diverse +0.0249 Holm 0.043; stratified
  +0.0116 n.s.; activity-scaffold +0.0031 n.s.). Those carry **no information**
  about whether the arms differ from each other, so the sentence originally had
  no backing artifact.

**Resolution (user directive 2026-10-17: "补跑 sampler 间对照检验").** The missing
test was run — `molmetal/scripts/fsma_c3_sampler_pairwise.py` →
`metrics/by_metric/fsma_c3_sampler_pairwise.json`, on the same 25 paired
`(metal, seed)` cells at K=20:

| test | result |
|---|---|
| Friedman across the 4 samplers | p = **0.128** (on delta-vs-zero-shot: 0.246) |
| max pairwise |Δ| (random vs activity-scaffold) | +0.0304, raw p₂ = 0.0233, **Holm 0.140** |
| all 6 pairwise Holm (absolute AUC) | all ≥ **0.140** |
| all 6 pairwise Holm (delta vs zero-shot) | all ≥ **0.073** |

**Verdict: `SAMPLERS_INDISTINGUISHABLE`.** The abstract's sentence is therefore
*correct* and should be **kept**, with two changes:
1. cite the new artifact, and
2. drop the "the only significant negative" half (stale, per the bullet above).

**Pairing caveat (now recorded in the artifact).** Unlike the C2 controls, the C3
arms do **not** share a query set: the sampler draws the K support rows and the
query is held-out-minus-support, so `n_query` is identical but the zero-shot AUC
is not. The test is therefore reported twice — on the absolute adapted AUC and on
the delta-vs-zero-shot — and the verdict requires both to agree. They do.

## Defect 5 — G2 wording, the cross-backbone sentence (lines 71–74)

> **Current:** "the same protocol run with fingerprint-plus-XGBoost warm-started
> gradient boosting **reproduces the curve**, reaching $+0.0786$ at $K{=}80$
> ($p=2\times10^{-7}$)"

`+0.0786` is **correct** (`fsma_xgb_warmstart_kshot.json`, pooled over
`per_metal.*.by_k.80.paired_delta_warm_vs_zero.per_seed`, n=25, 25/25 positive).
What is missing is G2 (`gap_analysis_2026_10_16.gaps[G2]`): on the XGB backbone the
**matched-budget control separation does not reproduce** —

| contrast | D-MPNN (in-house) | XGB warm-start |
|---|---|---|
| adaptation vs zero-shot, K=20 | +0.0389 | +0.0197 |
| **adaptation vs concat-retrain, K=20** | **+0.207** (0.0389 vs -0.1682) | **-0.0002** (p=0.51, n.s.) |

So the curve reproduces but the *"the lift is the adaptation, not the extra labels"*
control does not. Proposed honest qualifier: "...reproduces the K-curve
(+0.0786 at K=80, 25/25 cells positive); the matched-budget control separation,
however, does not survive the backbone swap — at K=20 warm-started boosting is
indistinguishable from retraining on source+K (−0.0002, p=0.51), so the
'adaptation ≫ retrain' claim is backbone-specific."

## Verified-correct (no change)

* "within-metal 5-fold ROC-AUC is 0.75–0.85" — matches
  `fsma_paper_stats.json → recovery_to_ceiling.ceiling_within_metal_mean_auc`
  (Ru 0.8545 / Ir 0.7570 / Rh 0.7681 / Os 0.7498 / Re 0.8187) ✓
* "$+0.0786$ at $K{=}80$" ✓ (see Defect 5)

## Flagged, not resolved

* "falls to 0.61–0.69 under cross-metal LOMO" — I could not pin the backing
  artifact. `ex1_i2_rich_metal_cap512.json` (arm `none`, K=10) gives per-metal
  zero-shot 0.617 / 0.653 / 0.621 / 0.663 / 0.631 → 0.62–0.66;
  `fsma_inhouse_kcurve_ep50.json` K=80 zero-shot gives 0.486–0.667. Neither is
  0.61–0.69. **Please name the intended LOMO source** or accept 0.62–0.66 from EX1.

---

## Requested action

Approve or amend Defects 1–3 and 5 (pure digit/wording corrections, each traceable
to a MEASURED artifact), and pick an option for Defect 4 (a / b / c). On approval I
will apply the hunks to `paper_a/main.tex` only, rebuild, and re-verify 0 undefined
references.
