# Wf-R10 D-MPNN fix plan (MEASURED-driven, 2026-10-06)

> **SUPERSEDED 2026-10-07** by Plan A (see `TODO/pending/plan_a_overview.md`). This file moved to `TODO/completed/` for archival reference. The Rh widen fix is still a **PRECURSOR to Plan A Step 1 + Step 3** (`TODO/pending/plan_a_step1_dmpnn_value.md` §3, `plan_a_step3_closed_loop.md` §3.4) and must be completed before those steps start.

Sources: `reports/dmpnn_clean5v6_w2_real_30ep/metrics.json` (R9),
`molmetal/data/metal_lomo.csv`, `reports/wf125_p1_rh_oversample/synthetic_rh_cohort*.json`,
`molmetal/scripts/train_metal_lomo_wasserstein.py` (L206-300 splitter, L425-513 append, L975-1100 W2 site).

## 1. Rh systematic failure — 0.3713, 5/5 seeds below 0.5
ROOT CAUSE (iii): the Rh training class is **82 % fabricated**. `append_synthetic_rh_to_dataset`
(L425) injects 500 synthetic Rh rows against only 108 real Rh rows in `metal_lomo.csv`
-> Rh train block = 608 rows, 500 synthetic. Their labels are NOT measured: the Active=1 half
is `similarity_transfer_to_real_active_rh`, `label_is_measured=false`, median max-sim **0.183**
(self-declared "weak evidence"), and the regression head gets a hardcoded `pIC50=7.0/5.0` (L503).
Model fits "synthetic-Rh chemistry => active"; the LOMO test fold (48 real Rh rows, synthetics
excluded at L296) measures the absence of that rule. Systematic, not noise: measured seed
sem 0.0239 << the fold's own `auc_resolution_se=0.042`, z vs chance = **-5.4**.
Corroboration: every wave with `n_synthetic_rh_appended=500` scores Rh 0.338-0.411 — the
constant is the cohort, not epochs.
FIX: (1) `--rh-oversample 0` — highest leverage, zero cost, removes a fabricated-label
leak per §5.2. (2) If still needed, keep the cohort **unlabelled** (aux/W2-embedding only),
never as CE/regression targets. (3) NO MetalCytoToxDB re-split: `data/MetalCytoToxDB.csv` has
600 rows and metals Cd/Pb/Cu/Zn/Ni/Mn — **no Rh/Ru/Ir/Os/Re**. The brief's "1134 rows" is FALSE
on disk; Rh is data-limited at 108 rows and no re-partition can fix it.

## 2. W2 no lift at 30 ep (R9 vs R7 d=-0.0021)
ROOT CAUSE (ii)+(iii): injection CADENCE, not weight. W2 is built once per epoch
(L984 `epoch % 5 == 0 or last`) and applied to the **first batch only** (one-shot
`w2_applied` flag, L1082). At 30 ep that is 7 single-batch backwards at w=0.05 against
~1300 optimiser steps = **~0.5 % of gradient mass**. At 50 ep the ratio is identical
(11 / ~2100), so the R8 +0.024 is inside seed noise (Re sem alone = 0.044) — there is no
real epoch-count effect. The stale-direction caveat (L1076) is secondary to the frequency.
FIX: (1) cadence first — compute every epoch, spread the backward over the first
`min(8, n_batches)` batches, or keep one-shot and raise w to 0.5. (2) Ablate one variable
per run: (w=0.05, every-epoch) vs (w=0.5, every-5). (3) Report a paired per-seed delta over
>=5 seeds; a single-seed +0.024 is not evidence.

## 3. Re per-seed variance — 0.539-0.778, sd 0.089
ROOT CAUSE (i): the fold is too small to separate model from evaluation variance. Re has
n=166, `pos_rate=0.163`, n_test=166 n_pos=27, `auc_resolution_se=0.043`. Measured sd 0.089 is
~2x the floor the fold imposes, so ~half the spread is the 27-positive test set, half is
init/sampler. Re also has the lowest positive rate of the 5 metals (0.163 vs Ru 0.263 /
Ir 0.448) while `pos_weight_mode=auto` + `balanced_sampler` are ALREADY on — an imbalance fix
is unlikely to help, and oversampling would worsen it.
FIX: (1) Reduce eval variance first — repeated LOMO `n_folds>=5` for Re and Os (Os n=87).
(2) Report Re as mean +- sd over folds x seeds; anything inside +-0.09 is unresolvable.
(3) Only then capacity down (`--depth`/`--hidden` + `--use-l2-guard`). NOT more oversampling.

## GPU priority (single-venv rule §5.9 -> `.venv/bin/python`)

| # | Run | Change | Wall (PROJECTED) |
|---|-----|--------|------------------|
| **A** | Rh real-only | `--rh-oversample 0`, 5 metals x 5 seeds, 50 ep | ~0.9 hr |
| **B** | W2 cadence | w=0.05 every-epoch, 5 metals x 3 seeds, 50 ep | ~3 hr |
| **C** | Re/Os repeated LOMO | `n_folds=5`, Re+Os, 3 seeds | ~1 hr |

Run **A** first: it removes a fabricated-label leak and is the only change that can move the
headline (`rh_target_hit=false` vs `target_rh_auc=0.5`). B and C are measurement-quality work —
expect NO headline movement, only better resolution. Pre-flight PASS (§4) + VRAM check per §5.1.

## Honest framing (§5.2)
All metrics above MEASURED from the JSON/CSV paths cited. Wall clocks are PROJECTED (no GPU run
executed). `target_rh_auc=0.5` remains an HONEST NEGATIVE until a run with real-only Rh clears it.
