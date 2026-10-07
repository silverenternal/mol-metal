# Wf-R19-S — Robustness Fix Plan (D-MPNN / Re LOMO seed spread)

> **SUPERSEDED 2026-10-07** by Plan A (see `TODO/pending/plan_a_overview.md`). This file moved to `TODO/completed/` for archival reference. The cosine + warmup + AdamW port is still a **PRECURSOR to Plan A Step 2** (`TODO/pending/plan_a_step2_cfm_prior.md` §3.2) — CFM training must be stable before fine-tuning the prior head.

**Status:** DRAFT plan. No GPU spent yet. No commit. No paper/* touched (TODO-84).
**Date:** 2026-10-07

## 0. PREMISE CORRECTION (MEASURED from disk — overrides R19-A/R19-B input)

The upstream brief says seed spread = **0.144** (0.5647 / 0.6479 / 0.7088). That number is
**not** what is on disk for the Re fold.

`molmetal/reports/dmpnn_re_w2off_50ep_3seed/aggregate.json` (MEASURED):

```
per_run_auc   = [0.57647, 0.61537, 0.58327]   # seeds 0,1,2
range         = 0.0389     stdev = 0.0208
n_test        = 166        n_pos = 27
auc_resolution    = 0.0607   auc_resolution_se = 0.0350
source        = molmetal/scripts/train_metal_lomo_wasserstein.py
```

**The metric's own sampling SE at n=166 (0.0350) is LARGER than the observed seed spread
(0.0208).** Honest-negative (§5.2): the model is *not* demonstrably "too init-sensitive" on
this fold — the measurement cannot resolve a 0.02 difference. Any "fix" that shrinks 0.021
to 0.012 is inside the noise. **Fix 0 below is therefore mandatory before Fix 1-4, and
buying 5 hr of GPU for it first would be the mistake.**

Second correction: R19-A audited `train_dmpnn_multitask_wasserstein.py`, but the Re numbers
come from `train_metal_lomo_wasserstein.py`. In the LOMO trainer, MEASURED by grep:
`random.seed` = 0 hits, `scheduler` = 0 hits, `init.kaiming/xavier/zeros` = 0 hits,
`--use-l2-guard` default = **False**. (`train_dmpnn_multitask_wasserstein.py` is different:
it already has `random.seed` at L1656, a `build_lr_scheduler` at L1904, and
`--use-l2-guard` default **True** at L2706.) Fixes below target the LOMO trainer.

---

## 1. FIX ORDERING (all Δ = PROJECTED unless marked MEASURED)

### Fix 0 — Fold-bootstrap CI on the EXISTING 3-seed predictions (MANDATORY GATE, 0 GPU)
- **What:** CPU bootstrap (B=2000) of ROC-AUC on the n=166 Re fold; report per-seed CI +
  paired CI for (max-min). No trainer change.
- **Why:** §5.2 honest framing; standard binomial-auc SE (Hanley & McNeil 1982). R19-B's own
  suggestion, which must be honored before any GPU spend.
- **Wall:** 0.3 hr, CPU only.
- **Expected Δ:** not a variance fix — it *sizes the noise floor*. Outcome decides whether
  Fix 1-4 are worth 5 hr.
- **Risk:** if CI half-width > ~0.07, the 3-seed panel is under-powered and even 5 seeds
  won't resolve a fix; the correct response is a bigger fold / repeated 5-fold, not more seeds.

### Fix 1 — Seeding completeness + grad clip ON (LOMO trainer, flag-only + ~10 LOC)
- **What:** in `train_metal_lomo_wasserstein.py` add `random.seed(seed)` beside L844-845
  (`torch.manual_seed` / `np.random.seed`), pass an explicit `torch.Generator` to the train
  sampler, and flip `--use-l2-guard` default to `True` (`--no-l2-guard` to disable).
- **Why:** any python-`random`-driven shuffle is currently an uncontrolled RNG, so part of the
  residual spread is not model sensitivity at all; Goh et al. 2017 (grad clip).
- **Wall:** 0.3 hr coding + 0.2 hr §5.7 50-step CPU smoke + 2.5 hr GPU (3 seeds x 50 ep).
- **Expected Δ:** PROJECTED 0.0208 -> ~0.015-0.018. **Only trustworthy if Fix 0 CI is tight.**
- **Risk:** changing the l2-guard default breaks bit-for-bit baseline comparability with
  Wf-125/Wf-131 panels. Prefer a *new* flag default than mutating the old one if any archived
  rerun must stay reproducible.

### Fix 2 — Explicit weight init (kaiming_normal + zero-init final logit)
- **What:** `init.kaiming_normal_` on `nn.Linear` (nonlinearity='relu'), `xavier_uniform_` on
  embeddings, `init.zeros_` on the final `nn.Linear(hidden//2, 1)` **bias** in
  `molmetal/baselines/dmpnn_multitask.py` (0 such calls today).
- **Why:** He et al. 2015; zero-init final logit removes a seed-symmetric output offset.
- **Wall:** 0.4 hr coding + 0.2 hr smoke + 2.5 hr GPU.
- **Expected Δ:** PROJECTED 0.015-0.018 -> ~0.012-0.015. Weakest evidence base of the three.
- **Risk:** interacts with Fix 3 (warmup) — with warmup, the LR multiplier already damps the
  init-scale effect, so Fix 2's marginal gain may vanish. Consider skipping if Fix 3 lands.

### Fix 3 — Cosine anneal + 5-epoch warmup (port existing helper, ~20 LOC)
- **What:** port `build_lr_scheduler()` from `train_dmpnn_multitask_wasserstein.py:329-385`
  (already implemented, already unit-wired) into the LOMO trainer; default
  `--lr-scheduler cosine --lr-warmup-epochs 5`. Zero new algorithm.
- **Why:** Smith & Topin 2018; Loshchilov & Hutter 2017 (decoupled wd).
- **Wall:** 0.5 hr coding + 0.2 hr smoke + 2.5 hr GPU.
- **Expected Δ:** PROJECTED 0.015-0.018 -> ~0.010-0.013. Highest expected yield per LOC.
- **Risk:** per-epoch `scheduler.step()` semantics under `--gradient-accumulation-steps N`
  (the D-MPNN trainer documents the total_steps divisor) — copy that, don't re-derive.

### Fix 4 — Seed-averaged prediction ensemble (0 GPU, report-only)
- **What:** score the **mean of the 3 (later 5) per-seed test predictions** per molecule
  instead of the mean of per-seed AUCs; add both columns to `aggregate.json`.
- **Why:** arXiv:2603.20724 reports 4x std reduction from 10-seed averaging — but on
  ogbg-molhiv (41k). Not transferable to n=166; direction is right, magnitude is NOT.
- **Wall:** 0.2 hr coding, 0 GPU. **This is the cheapest real deliverable.**
- **Expected Δ:** PROJECTED ~1/sqrt(3) on the ensemble member variance.
- **Risk:** reporting an ensemble metric where the paper tables a single-model metric is a
  comparability break. Must be a separate column, never a replacement (§5.2).

### Fix 5 — EMA / weight averaging (DEFER)
- **What:** in-loop EMA (decay 0.999) evaluated alongside raw weights.
- **Why:** Izmailov et al. 2018 (SWA) — wider optima; Huang et al. 2017 snapshot ensembles.
- **Wall:** 1 hr coding + 2.5 hr GPU.
- **Expected Δ:** PROJECTED, **unmeasured on this data; no paper reports seed-spread on
  n=166 folds.** Do NOT start until Fix 0-3 are MEASURED.
- **Risk:** lowest-evidence item in the plan. SAM (Foret 2020) is the same class, 2-3x wall —
  explicitly out of scope.

---

## 2. GPU WALL BUDGET (RTX 5090 / Blackwell, sequential, one fold = Re, n=166)

| Fix | Coding | Smoke (§5.7) | GPU (3 seeds x 50 ep) | Cumulative |
|---|---:|---:|---:|---:|
| Fix 0 (bootstrap) | 0.3 hr | 0 | **0** | 0.3 hr |
| Fix 1 (seed + clip) | 0.3 hr | 0.2 hr | 2.5 hr | 3.3 hr |
| Fix 2 (init) | 0.4 hr | 0.2 hr | 2.5 hr | 6.4 hr |
| Fix 3 (cosine) | 0.5 hr | 0.2 hr | 2.5 hr | 9.4 hr |
| Fix 4 (ensemble) | 0.2 hr | 0 | **0** | 9.6 hr |
| Fix 5 (EMA) | 1.0 hr | 0.2 hr | 2.5 hr | 13.3 hr |

**Recommended envelope: 3.3 hr (Fix 0 + Fix 1), decide after.** Full 1-3 = 9.4 hr.
Fix 5 is gated. §5.8: this plan is documentation, not a workflow — 0 tokens spent on a wf.

---

## 3. VERIFICATION PROTOCOL (every fix must clear this before being called SHIPPED)

Per AGENTS.md §5.7 (trainer flag wire-in requires a 50-step smoke) and §5.5 (filesystem is
the source of truth; `ls -la` evidence in the closeout).

1. **Smoke (CPU, <60 s):** `run_smoke_50step()` from
   `molmetal/scripts/wf399a1_smoke_protocol.py`; assert `parse_args` accepts the new flags,
   trainer imports, `decode_ratio > 0` for CFM, and `loss[0] > loss[49]`. A flag that only
   exists in `parse_args` is **FLAG_ONLY**, not SHIPPED.
2. **Panel:** re-run the Re fold **unchanged** vs the Fix-1/2/3 flag set, 3 seeds {0,1,2},
   50 epochs, `w_w=0.0 w2_real=false rh_oversample=500` — i.e. reproduce
   `dmpnn_re_w2off_50ep_3seed` exactly, so the diff is attributable to the fix alone.
3. **Panel replication:** for the winning config, run **seeds {0..4}** (5 seeds) so the
   spread estimate has 10 dof instead of 2. Report `mean ± stdev` AND `range`, plus
   `auc_resolution_se` from `aggregate.json` alongside — a spread below the resolution SE is
   reported as **INDISTINGUISHABLE FROM NOISE**, not as a win.
4. **Attribution:** the seed-only change (Fix 1's `random.seed` + generator) must be
   isolated *first*, before any hyperparameter moves, so "uncontrolled RNG" and
   "hyperparameter sensitivity" are not confounded.
5. **Honest framing on every number:** MEASURED only with a JSON path; everything in §1
   marked PROJECTED stays PROJECTED in the closeout.

---

## 4. HONEST NEGATIVE (mandatory, §5.2)

- The 0.144 spread in the R19-A/R19-B brief is **not reproducible from disk**; the measured
  Re 3-seed spread is **0.0389 range / 0.0208 stdev**, which is *smaller* than the fold's own
  AUC resolution SE of 0.0350. Any claim that the model is "too init-sensitive" is currently
  **UNSUPPORTED** at n=166.
- Every paper cited (Izmailov 2018, Huang 2017, arXiv:2603.20724, Smith & Topin 2018) reports
  seed spreads on datasets 2-3 orders of magnitude larger than n=166. **No published
  seed-spread number is transferable to this fold.** All Δ values above are PROJECTED.
- R19-A's audit of `train_dmpnn_multitask_wasserstein.py` does not describe the trainer that
  produced the Re numbers. Findings must be re-derived against
  `train_metal_lomo_wasserstein.py` before any are acted on.

## 5. SOURCES

- Izmailov et al. 2018, SWA — https://arxiv.org/abs/1803.05407
- Huang et al. 2017, Snapshot Ensembles — https://arxiv.org/abs/1704.00109
- Multi-RF Fusion / Multi-GNN Blending — https://arxiv.deeppaper.ai/papers/2603.20724v1
- Smith & Topin 2018 (cosine/exponential restart) — cited in-code at
  `train_dmpnn_multitask_wasserstein.py:1889`
- He et al. 2015 (init); Srivastava et al. 2014 (dropout); Goh et al. 2017 (grad clip)
