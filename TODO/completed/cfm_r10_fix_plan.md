# CFM Round 7 — deep review + fix plan (Wf-R10-A2)

> **SUPERSEDED 2026-10-07** by Plan A (see `TODO/pending/plan_a_overview.md`). This file moved to `TODO/completed/` for archival reference. The σ floor fix is still a **PRECURSOR to Plan A Step 2** (`TODO/pending/plan_a_step2_cfm_prior.md` §3.1) and must be completed before Step 2 starts.

Source: `molmetal/reports/cfm_round7_f5f6_retrain/run/per_epoch_metrics.json` +
`events.jsonl` (20 epochs, 16060 s = 4.46 hr). All numbers MEASURED from those.

## A. train_loss 9.17 is a Bayes FLOOR, not a loss-imbalance bug

- ep19 decomposition: `bond_ce 0.0306` + `0.0*w2` (weight already 0.0) +
  `assemble 0.0` (weight 0.0) => the 9.1717 is **entirely `fm_term`**.
- `cb/diversity/shortcut/consistency/so3/edm` are all default OFF/zero
  (trainer L1665-1700; config sets none) => they contribute exactly 0.0.
  **GPU-D's "turn those off" is a no-op.**
- With `fm_noise_sigma 2.505` (matched to data std) the target `dx = x1-x0` has
  variance 2*2.505^2 = 12.55 and an irreducible floor. Simulated optimum/coord:
  sigma 2.505 -> 9.86 | 1.0 -> 3.94 | 0.5 -> 1.97. **MEASURED fm_term = 9.14 =
  93% of the 9.86 floor.** Meanwhile bond_ce fell 0.44 -> 0.031 (14x). Not diverging.
- Fix is **sigma, not the cascade**: 2.505 -> 0.5 drops the floor ~5x and lifts max
  R^2 0.607 -> 0.849 (simulated). Caveat: sigma was RAISED to 2.505 in R7-N1 to hold
  radius-graph edge density flat (0.186->0.259 vs 0.843->0.259) — A/B, not swap.

## B. gen_validity 0.63 -> 0.29 -> 0.67 is under-powered eval + a still-rising trend

- Seed spread within an epoch is 0.02-0.19 (ep11 0.1875); binomial 1-sigma at
  n=192 is 0.034. Observed spread is 3-5x => **eval under-powered, not solver noise.**
- Epoch-0 `gen_validity 0.630` came with `single_fragment_ratio 0.033` /
  `largest_fragment_frac 0.706` — fragmented clouds parse trivially. **The 0.630
  is a fragmentation artifact, not a ceiling**, which is why the curve looks collapsed.
- From ep8: `corr(gen_validity, epoch) = 0.636`, `corr(., exact_type) = 0.379`.
  **Still improving at ep19 — cut by the epoch cap, not early stop.**
- Solver/threshold is not the lever: `bond_best_f1_threshold` sweeps 0.40 -> 0.85
  while `bond_f1` sits flat at 0.996 (recall 0.999 / precision 0.994). No headroom.
- Fix is **eval power + more epochs**: `n_steps 100 -> 50` halves the 13-min
  gen_eval; spend it on `gen_eval_n 64 -> 256` x 4 seeds (n=1024, 1-sigma 0.015).

## C. gate 0.266 vs 0.8 target: product metric, under-reported

- `gate = gen_validity * bond_exact_type_rate` (trainer L759-762; verified by
  recomputing all 20 products). Target 0.400.
- `exact_type_rate` is **all-or-nothing at 42 truth bonds/complex**: 0.4374 needs
  per-bond-type acc 0.9805; 0.50 needs 0.9836; 0.70 needs 0.9915. The brief's
  "0.437 -> 0.70" is a **1.1 pp per-bond push on a saturated head**, not a 37 pp gap.
- **gate_value 0.26596 is stale**: that is ep17's product; ep19's own is **0.29163
  (+9.7%)**. `_best_gate` is stored before ep19 folds in, so every wave has quoted
  a stale-low gate.
- Path to 0.400 = 0.80 x 0.50, both halves. gen_validity 0.667 -> 0.80 is the bigger
  move and is where headroom is. Order: **B first, then C's bond-type head, then A.**

## Fix ordering

- **GPU-D** (~3.5 hr, highest value): `epochs 20 -> 40`, `--lr-scheduler cosine` +
  `--warmup-steps`, `early-stop-metric combined`, `patience 8`. Cut mid-ascent.
- **GPU-E** (~2.5 hr, eval only, on D's checkpoint): `gen_eval_n 256`,
  `gen_eval_seeds "0 1 2 3"`, `method midpoint`, `n_steps 50`. Re-measure with
  1-sigma 0.015 instead of 0.034. No retraining.
- **GPU-F** (~3.5 hr, conditional): `fm_noise_sigma 0.5` A/B at 40 epochs, only if
  GPU-D's gate plateaus < 0.35.

## Honest framing

- train_loss 9.17 is at the floor for sigma 2.505, NOT non-convergence. Any plan
  that "removes loss terms" is a no-op — they are already 0.
- `decode_ratio 0.3725` is a PARSE count, explicitly anti-correlated with
  correctness (`decode_ratio_caveat`); chasing 0.8 there would be metric-gaming.
- GPU-D/E/F wall times PROJECTED from MEASURED 803 s/epoch mean (ep19 1317 s).
- Targets 0.80 gen_validity / 0.50 exact_type are PROJECTED, not promised.
