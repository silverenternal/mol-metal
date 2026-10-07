# Rh LOMO AUC blocker — fix plan (Wf-R9-A3)

> **SUPERSEDED 2026-10-07** by Plan A (see `TODO/pending/plan_a_overview.md`). This file moved to `TODO/completed/` for archival reference. The Rh LOMO data fix is still a **PRECURSOR to Plan A Step 3** (`TODO/pending/plan_a_step3_closed_loop.md` §3.4) and must be completed before Step 3 starts.

## §0 TL;DR
Rh LOMO AUC 0.4106 ± 0.0821 (MEASURED, `molmetal/reports/dmpnn_clean5v5_w2_real/aggregate.json`).
Root cause is **statistical, not architectural**: the Rh test fold is 48-60 rows
(MEASURED, `Rh/per_seed_metrics.json` seeds 0-4), giving a Hanley-McNeil AUC
standard error of **0.082-0.097** (COMPUTED, hm(n1,n0,0.5)). The ±0.082 spread is
the entire signal — 0.41 is statistically indistinguishable from 0.50.
The training CSV `molmetal/data/metal_lomo.csv` carries only **108 Rh rows / 51
unique ligands / 35 actives** (MEASURED) while the source `.data/molmetal/MetalCytoToxDB.csv`
carries **1134 Rh rows / 339 unique ligands / 534 with IC50<=10 uM** (MEASURED).
**The fix is to widen the Rh block ~10x, not to tune the model.**

## §1 Three candidate fixes
(a) **Data widening (RECOMMENDED)** — rebuild the Rh block of `metal_lomo.csv` from
    `MetalCytoToxDB.csv` (1134 rows, 339 ligands, 51/51 ligand overlap confirmed).
    n_test 48 -> ~567, AUC SE 0.097 -> 0.024 (COMPUTED). Smallest change, largest
    information gain; makes every subsequent (b)/(c) result interpretable.
(b) **W2-REAL w=0.5 + 80 ep on the Rh fold only** — 5 seeds, no new data. Tests
    whether the transfer regulariser is under-weighted. Cannot be scored: SE stays
    ~0.09, so only a >0.2 swing would be detectable.
(c) **Rh class-weight re-tune** (`--pos-weight-mode fixed --pos-weight 2-5`) —
    lowest value. `pos_rate_train=0.324` (MEASURED, seed 0) already implies
    `auto` pos_weight ~2.08; the deficit is n_test, not the loss weighting.
    NOTE: `--pos-weight-mode` choices are `auto|fixed|none`; **`sqrt` is NOT a
    valid choice** (MEASURED, trainer argparse L1372-1378).

## §2 Recommended order
1. (a) first — it is a prerequisite for measuring anything else.
2. (b) second, but only after (a) lands, run on all 5 metals so the aggregate
   gate (0.5833 MEASURED, target 0.70) is not degraded by a Rh-only path.
3. (c) only if (a)+(b) still leave Rh < 0.5 with SE <= 0.03.

## §3 Wall estimate
R8 (5 metals x 5 seeds x 50 ep) took ~3.5 hr MEASURED (dir mtime 11:50 -> 15:23).
- (a) rebuild is CPU-only, < 5 min; (a) re-run of 5 Rh seeds x 50 ep ~= **0.9 hr** PROJECTED.
- (b) 5 Rh seeds x 80 ep ~= **1.4 hr** PROJECTED.
- (c) ~= 0.9 hr PROJECTED.
Total if all three run: **~3.2 hr** on one GPU. All are USER_OK-gated per AGENTS.md §5.1.

## §4 Honest framing
No projection here claims any candidate will lift Rh above chance. (a) only buys
statistical power; a truthful outcome is "Rh AUC = 0.50 +/- 0.03", which is a
publishable honest-negative and a better paper artefact than an under-powered 0.41.
