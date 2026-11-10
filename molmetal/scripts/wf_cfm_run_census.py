# SPDX-License-Identifier: Apache-2.0
"""Census of every CFM run already on disk, and what each one already answers.

WHY THIS EXISTS
---------------
On 2026-10-10 a full-corpus run was found on disk — ``cfm_r9_fix_v2``,
``n_train=26715``, ``bond_type_macro_f1 = 0.9470`` — while the canonical
config was running ``n_train=256`` at ``bond_type_macro_f1 = 0.3770``.  The
26715 was being printed in the loader line ("26715/26715 requested complexes
have a Wiberg BO block") during runs that were in flight at the time.  Two
things followed from not reading it:

  * a probe that scaled 256 -> 4096 and concluded data volume was not the
    constraint, when the real comparison was 256 -> 26715; and
  * a proposed batch-size A/B for which the disk already held eight variants.

This script reads every ``per_epoch_metrics.csv`` under ``molmetal/reports``
and emits one row per run, plus the config each run actually used (from the
``config`` block its own metrics JSON recorded — never re-derived, never
retyped).  The point is that the NEXT change can be checked against what has
already been measured before another run is launched.

The `already_answers` column is the load-bearing one. It maps each run to the
question its existence closes, so a proposal can be dropped rather than run.

CONNECTIVITY (Wf-A114, 2026-10-30; the 0.203-0.348 claim CORRECTED
2026-10-31 by Wf-A152)
------------------------------------------------------------
This census recorded ``gen_validity`` and ``decode_ratio`` and nothing that
could see a fragmented molecule.  That is not a small gap: ``gen_validity``
counts a sample as valid when RDKit can sanitise the string, and a pile of
loose atoms sanitises.  MEASURED across the runs on disk, ``gen_validity``
reaches 1.000 on pools whose honest ``gen_single_fragment_ratio`` is
0.203-0.348 — i.e. most of the "valid" molecules are not one connected
molecule at all.  The census now carries each run's OWN recorded
``gen_single_fragment_ratio`` / ``gen_largest_fragment_frac`` plus the
``cfm_fragmentation_gate`` verdict computed from them, so the axis is
visible next to the blind one instead of being absent.

**CORRECTION (2026-10-31, Wf-A152) — the 0.203-0.348 sentence above is
WITHDRAWN as a claim about this corpus and is PRESERVED VERBATIM as
history.**  MEASURED over the 59 runs on disk: the 9 runs whose
``gen_validity_max`` reaches 1.000 carry LAST-epoch
``gen_single_fragment_ratio`` values of 0.0, 0.002840909090909091 (twice),
null (twice), 0.12215909090909091, 0.6666666666666666, 0.75 and
0.8571428571428571 — 0 of the 9 sit in 0.203-0.348.  The only 2 runs whose
LAST-epoch ratio lands in 0.203-0.348 are
``cfm_blk_c5_repair_2026_10_09`` (0.3333333333333333) and
``cfm_r17_eval512`` (0.23563218390804597), and NEITHER has a
``gen_validity_max`` of 1.000 (0.8541666666666666 and 0.0849609375).  The
fragmentation problem this section describes is real — it is what the
``gen_single_fragment_ratio`` / ``cfm_fragmentation_gate`` columns exist to
show — but the specific co-occurrence this sentence asserted does not hold
on this corpus.

RUN INTEGRITY + COHORT STRATIFICATION (Wf-A115, 2026-10-30)
-----------------------------------------------------------
Three defects in the census were measured on disk and are fixed here.

1. **THE LOADER COULD NOT SEE THE RUNS.**  The scan was
   ``REPORTS.iterdir()`` — one level deep — so 17 runs nested under a wave
   directory (``cfm_round3_retrain/run``, ``wf299_cfm_main/wave_a/*``,
   ``wfa113_t42_bond_ce/*``) were structurally invisible.  The scan is now
   ``REPORTS.rglob("per_epoch_metrics.csv")``; a run's ``run`` name is its
   path relative to ``REPORTS`` and ``run_path`` is the absolute directory.

2. **RUN INTEGRITY WAS RECORDED NOWHERE.**  ``n_epochs_run`` is a ROW COUNT.
   It is not "how many epochs the model trained": MEASURED, many runs START
   above epoch 0 because they resumed (``cfm_c7_fullgen`` starts at 4,
   ``cfm_bond_head_20ep`` at 1, ``cfm_r11_eval256`` at 19 — and that one has
   exactly ONE recorded epoch, with ``train_loss = nan``, out of a declared
   20).  Every run now carries an ``integrity`` block: the epoch index range,
   how many epochs were recorded vs declared, whether the history is
   contiguous from zero, whether it was a resume, and whether any recorded
   ``train_loss`` is nan.

3. **THE SERIES WAS READ AT CSV PRECISION.**  The per-epoch series now comes
   from ``per_epoch_metrics.json`` (full float precision) with the CSV as a
   fallback, and the source is recorded per run in
   ``integrity.series_source``.  This is not cosmetic: the published bar
   0.41145833333333337 is the JSON value (109/192 x 0.7247706422018348); the
   4-decimal CSV renders it 0.41147500000000004, a 1.7e-5 gap that means
   the artifact and the published number disagree.

The cohort axis is promoted to a first-class field (``run["n_train"]``) and
stratified in the new top-level ``by_n_train`` block, with a
``budget_matched`` block that compares cohorts at the SAME epoch index rather
than at whatever epoch each cohort happened to stop at, and a ``per_run_bar``
block that replaces the cohort median with a NAMED run.

THE BAR IS A NAMED RUN, NOT A COHORT MEDIAN
-------------------------------------------
MEASURED on disk: the best-epoch reading is a max over a noisy series (the
census's own words: "optimistically biased"), and the final-epoch reading is
dominated by WHERE EACH COHORT STOPPED.  The two cohorts are out of phase in
the same oscillation — ~~five of the eight clean n_train=256 runs read 0.0 at
epoch 8 and only take off around epoch 10-14, while the full-corpus runs are
already productive by epoch 8 — so the ordering between them flips at almost
every budget.~~ **CORRECTION (2026-10-31, Wf-A153) — the struck cohort counts
AND the struck "flips at almost every budget" clause are REFUTED by the
corpus on disk.**  A cohort median therefore measures the BUDGET, not the
data size.  ``per_run_bar`` names the run a comparison has to beat, with the
reading, the cohort and the budget stated alongside the number.

**CORRECTION (2026-10-31, Wf-A153) — MEASURED over the 59 runs on disk
(``metrics/by_metric/cfm_run_census.json`` -> ``runs[]`` and
``budget_matched``, plus the per-epoch rows under ``molmetal/reports/``):**

* the population is the **8** clean n_train=256 runs that carry an
  end-to-end rate (of **14** clean n_train=256 runs);
* at epoch 8 **4** of those 8 read exactly 0.0 (cfm_fm_ab_2026_10_09,
  cfm_fm_control_2026_10_09, cfm_frag_control, cfm_pcgrad_off_2026_10_15),
  cfm_pcgrad_on_2026_10_14 reads 0.0625, and **3** carry NO epoch-8 rate at
  all (cfm_blk_c5_repair_2026_10_09, cfm_bond_head_30ep_onecycle_2026_10_09,
  cfm_frag_assemble): each of those three HAS an epoch-8 row with
  ``gen_validity_n > 0`` and ``gen_validity_val == 0`` (48 drawn / 0 scored,
  48 / 0 and 128 / 0), so the rate is UNDEFINED and has been published as
  null since Wf-A127, NOT as the 0.0 the struck sentence counted;
* "only take off around epoch 10-14": of the 4 that read 0.0 at epoch 8, 3
  first read >= 0.10 at epoch 9 (cfm_fm_control) or epoch 10 (cfm_fm_ab,
  cfm_pcgrad_off) and 1 (cfm_frag_control) first reads >= 0.10 at epoch 24;
* "the full-corpus runs are already productive by epoch 8": of the **11**
  clean n_train=26715 runs, **8** read a non-zero end-to-end rate at epoch 8,
  1 reads exactly 0.0 (cfm_tmqm_all6_retrain) and 2 have NO epoch-8 row at
  all (cfm_fm_distort_2026_10_11 stops at epoch 7,
  cfm_fullcorpus_2026_10_10_attempt1_epsgate stops at epoch 3);
* "the ordering between them flips at almost every budget": the census's OWN
  published field refutes this — ``budget_matched.flipped_budgets`` is
  ``[9, 11, 25]``, i.e. **3** flips across the **10** budget transitions, and
  the n_train=256 cohort median is the HIGHER of the two at **8 of the 11**
  budgets.

THE "DEFAULT THAT LIES" (Wf-A114 finding D; the trainer side is repaired,
the CENSUS side is what this census still has to carry)
-----------------------------------------------------------------
The trainer used to write ``gen_validity`` / ``gen_single_fragment_ratio`` /
``gen_largest_fragment_frac`` / ``gen_metal_ratio`` as 0.0 when the
generation eval did NOT run that epoch, which is indistinguishable from a
measured 0.  The predicate that actually says "the eval ran" is
``gen_validity_n > 0``; ``gen_eval_n`` is per-seed bookkeeping and is NOT
reliable (MEASURED: rows carry ``gen_eval_n == 0`` while ``gen_validity_n >
0``, e.g. ``cfm_tmqm_all6_retrain`` epoch 10 reads 12 of 16).  The census now
reports the last row's rate columns as null — NOT 0.0 — when that predicate
says the eval did not run, and says so per run.

The trainer no longer writes that placeholder (Wf-A114/T85, 2026-10-30: it
writes ``null`` / the literal ``nan``), but the HISTORIC artifacts on disk
still carry it, and the census reads those artifacts.  Wf-A127 (2026-10-31)
closes the second hole the trainer's repair exposed in the census: a rate
over the SCORED rows (``gen_validity_val``) is 0/0 — undefined — when the
eval drew samples and NONE of them parsed, which the drawn-count predicate
(``gen_validity_n > 0``) does not catch.  MEASURED: 66 of the 576 per-epoch
rows across the 59 census runs, and 2 of those runs END on such a row.  The
census now nulls those rates too, so an empty denominator is published as
undefined and never as a measured 0.0.  See ``_normalise_row`` and the
SKIP-cause section of ``_fragmentation_verdict``.

THE POPULATION CHIMERA (Wf-A114E, 2026-10-30)
--------------------------------------------
``per_epoch_metrics.json`` is written ONCE, at the end of training, to a
FIXED path (``train_cfm_yuelbond_wasserstein.py:20381``).  Two runs launched
into the same ``--out-dir`` therefore CLOBBER each other's record, and the
survivor is whichever finished LAST.  MEASURED: 4 of the 59 census runs were
scored under a run name they do not own — the census read a TRUNCATED PREFIX
of a longer run while the directory's own name pointed at the other one.

``events.jsonl`` is opened in APPEND mode, so it keeps EVERY population that
ever wrote into the directory, each with its own ``done`` event.  It is the
ground truth this census used to ignore (``grep -n events.jsonl`` found
nothing).  The census now reads it and CROSS-CHECKS every run against it:

  * ``integrity.population_verdict`` says which population the census
    believes it scored, and ``integrity.population_reason`` says WHY;
  * a run whose per-epoch file is a truncated prefix of a longer population
    is RE-DERIVED from the events rows of the matching population, so the
    truncated prefix is never scored;
  * a run whose conflict cannot be resolved is REFUSED and excluded from
    every headline population rather than scored on a guess.

The verdict is per RUN and the classes are not interchangeable: a RESUMED
run's events stream carries the PRE-resume epochs, so preferring the longer
series would double-count epochs the run never trained; and a partial
stream carries fewer epochs than the file, so preferring events would lose
epochs.  Both classes stay exactly as they were.

SCOPE
-----
Only runs that wrote ``per_epoch_metrics.csv``.  Runs killed before the first
write leave no row here — that is stated in `not_counted` rather than papered
over, because two runs from 2026-10-10 died that way (the epsilon-gate
verification pair) and their logs remain as the only record of what they
proved.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REPORTS = REPO / "molmetal" / "reports"

#: Configuration keys that plausibly move bond-type quality. Pulled from each
#: run's own recorded `config` dict, so a value here is what that run used.
#:
#: Wf-A114 (2026-10-30) — the four velocity-head levers are here because the
#: trainer's own `config` block now records all six of them.  Before this,
#: a run pair that differed only in `velocity_head_init_knots` (MEASURED: the
#: value is 9 on some configs on disk and 5 on others) was INDISTINGUISHABLE
#: in its own artifact, because the census filtered the key out with
#: `{k: cfg.get(k) for k in CFG_KEYS if k in cfg}` — the 2x2 attribution
#: (head x init) could not be read off any run.
CFG_KEYS = (
    "n_train", "n_val", "batch_size", "hidden_dim", "n_layers", "lr",
    "epochs", "bond_predictor", "use_tmqm_init", "bond_head_warmup_epochs",
    "early_stop_metric", "use_egnn_velocity",
    # Wf-A115 — `bond_class_weights` (NOT `class_weights`).  The CFM
    # trainer's dests are `bond_class_weights` / `bond_class_weight_power`
    # (train_cfm_yuelbond_wasserstein.py:12199 / :12214).  `class_weights`
    # is the name the OTHER trainer uses
    # (wf_blk_g1_bond_head_scaling.py:322 -- `--class-weights`), so it
    # resolved to NOTHING in every CFM run: MEASURED, ~~0 of 43 on-disk CFM
    # runs record either name in their config block.~~
    # **CORRECTION (2026-10-31, Wf-A153): 0 of 59** — the population size in
    # the struck rendering ("43") is STALE: the census now scores 59 runs and
    # all 59 carry a `config` block, so the measured statement is "0 of 59"
    # for `bond_class_weights` and `bond_class_weight_power` (neither name
    # appears in ANY run's config block).  The tuple's THIRD key behaves
    # differently and the struck rendering said nothing about it: 3 of the 59
    # runs DO record `use_fm_objective` in their own `config` block —
    # wf385a3_scaffold_wire_in/train (false),
    # wfa113_t42_bond_ce/wfa113_t42 (true) and
    # wfa113_t42_bond_ce/wfa113_t42_lr0 (true).  MEASURED 2026-10-31 from
    # `metrics/by_metric/cfm_run_census.json` -> `runs[]` -> `runs[].config`.
    "bond_class_weights",
    "bond_class_weight_power", "use_fm_objective",
    # Wf-A114 — the velocity-head levers (the other two, `velocity_head` and
    # `vector_channels`, were already in the trainer's config block but were
    # NOT in this tuple, so the census filtered them out too).
    #
    # Wf-A115 — these six ALSO live in `HEAD_LEVER_KEYS`.  That is not a
    # duplicate definition: `config` is the run's OWN record (whatever the
    # run wrote under `config`), while `head_levers` is the MERGED view
    # (config first, then the `provenance` block).  A lever the run recorded
    # only in `provenance` is null in `config` and present in
    # `head_levers` — the two blocks differ BY DESIGN, and the test suite
    # pins that they agree on every key they share.
    "velocity_head", "vector_channels", "velocity_head_init",
    "velocity_head_init_knots", "vector_channel_norm_gain", "t_embedding_dim",
)

#: Wf-A115 — the six head levers, read in this order: the run's own
#: ``config`` block first (the trainer writes ``velocity_head`` and
#: ``vector_channels`` there), then its ``provenance`` block (which carries
#: ``velocity_head`` / ``velocity_head_init`` / ``vector_channel_norm_gain``
#: / ``t_embedding_dim``).  A key that is in NEITHER is recorded as null.
HEAD_LEVER_KEYS = (
    "velocity_head", "vector_channels", "velocity_head_init",
    "velocity_head_init_knots", "vector_channel_norm_gain", "t_embedding_dim",
)

#: Wf-A115 — the budgets the ``budget_matched`` block compares at.  Each
#: budget K means "the epoch whose INDEX is K-1", so a run that stopped at
#: epoch 4 contributes nothing to K=9.
BUDGET_MATCHED_BUDGETS = (5, 9, 11, 12, 13, 14, 15, 18, 20, 25, 28)

#: Wf-A115 — the two cohorts the budget-matched comparison is between.
BUDGET_MATCHED_COHORTS = (256, 26715)

#: Wf-A115 — the numbers that were published and are now WITHDRAWN, with the
#: reason each is withdrawn.  This dict is written both at the top level and
#: inside ``by_n_train`` (the spec placed it "同块"); it is ONE object, so
#: the two copies cannot drift.
RETRACTED_NUMBERS: Dict[str, str] = {
    "0.572276953125": (
        "cfm_r11_eval256 — 1 epoch recorded, resumed@19, train_loss nan "
        "— WITHDRAWN as a bar. MEASURED: its own per_epoch_metrics carries "
        "exactly one epoch (index 19) out of a declared 20, the history is "
        "not contiguous from zero, and its train_loss is nan. The "
        "full-precision JSON value of the same cell is 0.572265625; the "
        "0.572276953125 form is the 4-decimal CSV rendering."
    ),
    "3.2x inversion": (
        "WITHDRAWN. The n_train axis is UNRESOLVED: the inversion is an "
        "artifact of the READING (best-epoch max over a noisy series) and "
        "of the BUDGET (the cohorts are out of phase), not of the data "
        "size. See `budget_matched.ordering_flips` — ~~the ordering between "
        "the two cohorts flips at nearly every budget.~~ "
        "**CORRECTION (2026-10-31, Wf-A153): the struck clause is REFUTED by "
        "the census's OWN published field — `budget_matched.flipped_budgets` "
        "is [9, 11, 25], i.e. 3 flips across the 10 budget transitions, and "
        "the n_train=256 cohort median is the HIGHER of the two at 8 of the "
        "11 budgets.**  MEASURED 2026-10-31 from "
        "`metrics/by_metric/cfm_run_census.json` -> "
        "`budget_matched.flipped_budgets` and "
        "`budget_matched.ordering_flips`."
    ),
}

#: Question -> the runs whose existence already answers it. Each string says
#: WHAT the run measured, so a proposal that is already covered is visible as
#: such rather than having to be rediscovered.
#:
#: Wf-A115 — entry 9's hard-coded prose ("gen_validity reaches 1.000 on
#: pools that sit at 0.203-0.348") had no source.  Wf-A152 (2026-10-31) —
#: the claim does not hold on this corpus either, so entry 9's measured
#: text is now COMPUTED by ``_already_answers()`` from the runs on disk (the
#: span over the runs that RECORD ``gen_single_fragment_ratio_end``, and the
#: population whose ``gen_validity_max`` reaches 1.000) instead of being
#: retyped here; the 2026-10-30 rendering is preserved verbatim inside that
#: computed text.  The three new entries are added by ``_already_answers()``
#: so their numbers are computed from the runs on disk rather than retyped.
ALREADY_ANSWERS: Dict[str, List[str]] = {
    "does the bond head keep improving with more epochs at n_train=256?": [
        "cfm_bond_head_100ep_2026_10_09 (typeF1 0.6370 @100ep vs 0.3770 @30ep, same n_train=256)",
        "cfm_bond_head_20ep_2026_10_09 (typeF1 0.3236 @19ep)",
    ],
    "does training on the FULL tmQM corpus beat the 256-cloud cohort?": [
        "cfm_r9_fix_v2 (n_train=26715 -> typeF1 0.9470, exact_type 0.3849)",
        "cfm_r11_eval256 (n_train=26715 -> typeF1 0.9456, exact_type 0.4374)",
        "cfm_r17_eval512 (n_train=26715 -> typeF1 0.7795, exact_type 0.0652; NOTE worse than r9)",
    ],
    "does flow matching help or hurt, at fixed n_train=256?": [
        "cfm_fm_ab_2026_10_09 (fm=true: decode 0.5833, gv 0.0625)",
        "cfm_fm_control_2026_10_09 (fm=false: decode 0.2500, gv 0.0000)",
    ],
    "does the assemble/connectivity term help or hurt?": [
        "cfm_frag_assemble (typeF1 0.1756) vs cfm_frag_control (typeF1 0.2703)",
        "-> NULL on bond type; both collapse gv to ~0.04",
    ],
    "is bond_exact_type_rate ALWAYS zero (i.e. is `combined` always constant)?": [
        "NO. cfm_r9_fix_v2 = 0.3849, cfm_r11_eval256 = 0.4374, "
        "cfm_bond_head_100ep = 0.1302. ~~It is zero in the n_train=256 runs "
        "(cfm_fm_*, cfm_blk_c5_repair, cfm_frag_*) but NOT in the full-corpus runs.~~ "
        "**CORRECTION (2026-10-31, Wf-A153): the struck clause is REFUTED — "
        "TWO n_train=256 runs read a NON-ZERO `bond_exact_type_rate`, so the "
        "zero is not exclusive to the full-corpus cohort.**  MEASURED "
        "2026-10-31 over the 59 runs in "
        "`metrics/by_metric/cfm_run_census.json` -> `runs[]` -> "
        "`runs[].bond_exact_type_rate`: 33 runs read exactly 0.0, 16 read "
        "non-zero and 10 record no value; the non-zero n_train=256 runs are "
        "cfm_bond_head_100ep_2026_10_09 (0.13020833333333334) and "
        "cfm_bond_head_20ep_2026_10_09 (0.010416666666666666).",
        "So the constant-0 gate is a property of the SMALL cohort, not a "
        "universal defect. ~~cfm.yaml declares no early_stop_metric, so it "
        "inherits `combined` and inherits this.~~ "
        "**CORRECTION (2026-10-31, Wf-A153): the struck clause is REFUTED — "
        "`configs/cfm.yaml` DOES declare one, `early_stop_metric: "
        "bond_type_f1_eps` (configs/cfm.yaml:310).**  MEASURED 2026-10-31 "
        "over the 59 runs: 0 of them record `early_stop_metric` in their own "
        "`config` block (the requested criterion is read from the run's "
        "`early_stop_control` block instead), and the requested criterion is "
        "`bond_type_f1` in 25 runs, `combined` in 16, `bond_type_f1_eps` in 1 "
        "and absent in 17 (`runs[].early_stop_guard.requested_metric`).",
    ],
    "does use_tmqm_init actually load pretrained weights end-to-end?": [
        "2026-10-10 live run: '[load_tmQM_pretrained] Loaded tmQM-pretrained "
        "encoder: 47/51 params transferred' inside a full CFM trainer run.",
        "~~ckpt.yaml (shipped) does NOT declare it; 9 ablation configs do.~~ "
        "**CORRECTION (2026-10-31, Wf-A153): there is NO `configs/ckpt.yaml` "
        "on disk or in git (`git ls-files configs/` finds no file of that "
        "name), so the struck clause names a file that does not exist; and "
        "the count of configs that declare the flag is 70, not 9 (of the 263 "
        "`configs/*.yaml` / `*.yml` files on disk, 114 mention the key at all, "
        "70 declare it `true`, 44 declare it `false`, 149 do not mention it, "
        "and 21 of the 70 are named `cfm_*`).**  MEASURED 2026-10-31 by "
        "scanning `configs/*.yaml` on disk.",
    ],
    "does the epsilon-constraint gate wire correctly?": [
        "2026-10-10 live run logged "
        "'criterion=bond_type_f1_eps value=0.0613 (gen_validity 0.9753 >= "
        "epsilon 0.8500; bond_type_macro_f1 0.0613 scored +0.0613)'.",
        "NOT answered: which epoch it ultimately selects (both runs killed at "
        "ep5/ep2 of 30). cfm_r9_fix_v2 is the closest existing evidence "
        "(best @ ep7 under `combined`, gate_value 0.2205).",
    ],
    "yuelbond vs dmpnn bond head (isolated)?": [
        "2026-10-10 head-only A/B, same val split: yuelbond typeF1 0.7653 / "
        "exact 0.0573 vs dmpnn typeF1 0.6593 / exact 0.0169 at n_train=256; "
        "same ordering at 1024. yuelbond wins on the type axis.",
        "NOT answered: inside joint CFM training.",
    ],
    "does the generated pool hold together as MOLECULES?": [
        "NO — and the answer is now in this census, per run, as "
        "`gen_single_fragment_ratio_end` + `fragmentation_gate` + "
        "`failure_mode`. "
        "MEASURED — computed from the runs on disk by `_already_answers()`, "
        "not retyped here: the span of `gen_single_fragment_ratio_end` over "
        "the runs that RECORD it, and the population whose `gen_validity_max` "
        "reaches 1.000. The 2026-10-30 rendering of this note (a hard-coded "
        "0.0000-0.8772 span and a 0.203-0.348 co-occurrence that the data "
        "does not support) is preserved verbatim inside that computed text.",
        "The cause is measured in HONEST_NEGATIVE_CATALOG §6.170/§6.187/§6.195: "
        "the sampler returns the prior because the velocity field does not "
        "transport, and the field is worst exactly where the ODE starts "
        "(t -> 0).",
        "NOT answered: whether any configuration on disk closes it. No run has "
        "used the vector_channels + lsq-init head combination at full-corpus "
        "scale, and no run has emitted `gen_molecule_validity`.",
        "The per-cohort numbers are in `by_n_train` (and "
        "`by_n_train_depth1_only` for the loader scope the original cohort "
        "medians were measured on).",
    ],
}


from molmetal.eval.bond_metric_schema import (
    bond_ap_definition,
    bond_ap_legend,
    stamp_counts,
)
from molmetal.scripts.cfm_cohort_recovery import (
    COHORT_SOURCE_CONFIG,
    recover_cohort,
)


def _num(v) -> Optional[float]:
    """One numeric value, NaN-aware, from a JSON scalar or a CSV cell.

    Wf-A115 — this replaces the old ``_f`` body, which read
    ``float(row.get(key, "") or "nan")``.  The ``or`` there was a trap for
    JSON rows: a value of ``0`` or ``0.0`` is falsy, so it became the string
    ``"nan"`` and the number was LOST.  That is how a measured 0.0 train_loss
    would have been reported as a nan, and how a measured 0.0
    ``gen_validity_n`` would have hidden the fact that the eval never ran.

    NaN is recognised in both encodings: a JSON float ``nan`` (``v != v``)
    and the CSV string ``"nan"``.  A missing key and an unparseable cell are
    both None; callers that need to tell "absent" from "nan" use
    ``_is_nan``.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() == "nan":
            return None
        try:
            x = float(s)
        except ValueError:
            return None
    else:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
    return None if x != x else x


def _f(row: Dict[str, object], key: str) -> Optional[float]:
    """One numeric column of one row, or None when it is absent / nan."""
    return _num(row.get(key))


def _is_nan(row: Dict[str, object], key: str) -> bool:
    """True when the key is PRESENT and its value is nan.

    Wf-A115 — the difference between "this row has no ``train_loss`` column"
    and "this row's ``train_loss`` is nan" matters: the first is a file that
    never had the column, the second is a run whose loss did not come out.
    """
    if key not in row:
        return False
    return _num(row.get(key)) is None


def _col(rows: List[Dict[str, object]], key: str) -> List[float]:
    """Every finite value of one column, in row order."""
    out = []
    for r in rows:
        v = _f(r, key)
        if v is not None:
            out.append(v)
    return out


def _epoch_index(row: Dict[str, object]) -> Optional[int]:
    v = row.get("epoch")
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _empty_denominator_reason(
    gen_validity_n: Optional[int],
    gen_validity_val: Optional[int],
) -> str:
    """Why the ratio's denominator is empty — the two ways it can be.

    Wf-A127 (2026-10-31) — ``denominator_empty`` is ONE basis with TWO
    sub-causes, and the old reason string named only the first, so it was
    FALSE for the second:

    * the eval drew NOTHING (``gen_validity_n == 0``) — the original case;
    * the eval drew samples but SCORED no rows (``gen_validity_val == 0``
      of ``gen_validity_n``) — every chemistry rate is then 0/0 and the
      trainer's 0.0 is a placeholder.

    Both are "the denominator is empty", so both keep the SAME basis; only
    the sentence differs, and it must be TRUE for the row it describes.
    """
    if gen_validity_n is not None and int(gen_validity_n) == 0:
        return (
            "gen_single_fragment_ratio was recorded but the generation "
            "eval drew nothing (gen_validity_n == 0), so the ratio has "
            "an empty denominator"
        )
    if gen_validity_val is not None and int(gen_validity_val) == 0:
        return (
            "gen_single_fragment_ratio was recorded but the generation "
            "eval scored no rows (gen_validity_val == 0 of "
            f"gen_validity_n == {int(gen_validity_n or 0)}), so the ratio "
            "has an empty denominator"
        )
    return (
        "gen_single_fragment_ratio was recorded but its denominator is "
        "empty, so the ratio is 0/0 and undefined"
    )


def _fragmentation_verdict(
    single_fragment_ratio: Optional[float],
    largest_fragment_frac: Optional[float],
    gen_validity: Optional[float] = None,
    field_present: bool = False,
    *,
    gen_validity_n: Optional[int] = None,
    gen_validity_val: Optional[int] = None,
) -> Dict[str, object]:
    """PASS / FAIL / SKIP for one run's honest connectivity numbers.

    Wf-A114 (2026-10-30) — this column is the reason the fragmentation was
    ~~invisible for 41 runs.  ``gen_validity`` was 1.000 on pools whose honest
    ``gen_single_fragment_ratio`` was 0.203-0.348, and the census recorded
    only the blind one.~~ **CORRECTION (2026-10-31, Wf-A153) — BOTH numbers in
    the struck sentence are REFUTED by the corpus on disk; the superseded
    rendering is preserved VERBATIM inside the strike.**  MEASURED 2026-10-31
    from ``metrics/by_metric/cfm_run_census.json`` -> ``runs[]`` (59 runs),
    from ``budget_matched``/``fragmentation_summary`` in the same file, and
    from the per-epoch rows under ``molmetal/reports/``:

    * the 0.203-0.348 CO-OCCURRENCE does not hold: of the **329** (run,
      epoch) pairs that carry BOTH ``gen_validity`` and
      ``gen_single_fragment_ratio``, **40** read ``gen_validity == 1.0`` and
      **0 of those 40** carry a single-fragment ratio inside
      [0.203, 0.348];
    * the **9** runs whose ``gen_validity_max`` reaches 1.000 carry LAST-epoch
      ``gen_single_fragment_ratio`` values 0.0, 0.002840909090909091 (twice),
      0.12215909090909091, 0.6666666666666666, 0.75, 0.8571428571428571 and
      null (twice) — **0 of the 9** sit in 0.203-0.348;
    * the only **2** runs whose LAST-epoch ratio DOES sit in 0.203-0.348 are
      ``cfm_blk_c5_repair_2026_10_09`` (0.3333333333333333) and
      ``cfm_r17_eval512`` (0.23563218390804597), and NEITHER has a
      ``gen_validity_max`` of 1.000 (0.8541666666666666 and 0.0849609375);
    * **no population of 41 exists in the 59-run corpus.**  What the corpus
      carries instead is: **30** runs publish a RECORDED last-epoch
      single-fragment ratio and **29** publish null (11 ``field_absent`` +
      18 ``denominator_empty``); the ``per_epoch_metrics.csv`` of **10** runs
      lacks the ``gen_single_fragment_ratio`` column entirely (their JSON
      carries the key but no row carries a value, so 11 runs in total have
      no epoch row with a value for it); **40** runs publish a null
      ``decode_single_fragment_ratio_end`` and **27** a null
      ``gen_validity_end``.

    The fragmentation gap this column closes is real — it is what
    ``gen_single_fragment_ratio_end`` / ``cfm_fragmentation_gate`` exist to
    show — but NEITHER count the struck sentence asserted holds on this
    corpus.  The thresholds are the declared policy of
    ``molmetal/scripts/cfm_fragmentation_gate.py``; see that module's
    docstring for why they are policy and not a discovered optimum.

    TWO DENOMINATORS, AND THE VERDICT USES THE HONEST ONE
    ------------------------------------------------------
    ``gen_single_fragment_ratio`` is conditioned on the sample being
    RDKit-parseable (the trainer's own denominator).  A string RDKit rejects
    is not a molecule, so it must not leave the denominator — but the
    trainer's column does leave it out.  The end-to-end rate is therefore
    ``gen_validity x gen_single_fragment_ratio``: the share of ALL samples
    that are one connected, parseable molecule.

    MEASURED on ``cfm_fullcorpus_b64_2026_10_11`` at epoch 11: the
    conditional number is 0.7879 and the end-to-end number is
    0.0859 x 0.7879 = 0.0677.  Reading the conditional number alone
    overstates the generator by more than 10x.

    TWO DISTINCT SKIP CAUSES (Wf-A125 G8)
    -------------------------------------
    A ``None`` ratio has TWO causes, and the old single reason string
    ("no gen_single_fragment_ratio recorded for this run") was FALSE for
    one of them:

    * ``field_absent`` — the column was never written by this run at all.
      MEASURED 2026-10-31: **11** of the 29 SKIP runs.
    * ``denominator_empty`` — the column WAS written (the trainer's 0.0
      placeholder) but the generation eval drew NOTHING
      (``gen_validity_n == 0``), so the ratio's denominator is empty and
      the measurement is degenerate.  MEASURED 2026-10-31: **18** of the 29
      SKIP runs (e.g. ``c7_b32_4096_2026_10_20``, whose single epoch row
      is ``gen_single_fragment_ratio=0.0, gen_validity_n=0``).

    *(The two counts above were MEASURED 2026-10-31 against the committed
    artifact ``metrics/by_metric/cfm_run_census.json``: 29 SKIP rows, of
    which 11 carry the ``field_absent`` reason string and 18 carry the
    ``denominator_empty`` one.  This docstring previously said 9 / 18 —
    the split Wf-A125 G8 REFUTED — and was corrected by Wf-A127;
    catalog §6.238.  It was then corrected AGAIN by Wf-A150: the Wf-A127
    republication moved two runs from FAIL/end_to_end to
    SKIP/denominator_empty, so the split became 11 / 18 over 29 SKIP rows.
    The superseded 11 / 16 over 27 wording is preserved here and in
    catalog §6.262.)*

    ``field_present`` is the INPUT FLAG that carries the fact the caller
    already knows: the column was RECORDED (raw value present) even though
    the usable rate is None.  It is the only thing that separates the two
    causes.  It is NOT a third ``basis`` value — the ``basis`` key emitted
    in the artifact is one of ``field_absent`` / ``denominator_empty`` /
    ``end_to_end`` / ``conditional`` (the first two only ever appear
    together with ``status == "SKIP"``).

    Wf-A127 (2026-10-31) — ``denominator_empty`` HAS TWO SUB-CAUSES
    -------------------------------------------------------------
    The reason string above named only the first, and was therefore FALSE
    for the second:

    * the eval drew NOTHING (``gen_validity_n == 0``) — the original case;
    * the eval drew samples but SCORED no rows (``gen_validity_val == 0``)
      — every chemistry rate is then 0/0 and the trainer's 0.0 is a
      placeholder, not a measurement.

    MEASURED 2026-10-31 on disk: **66** of the **576** per-epoch rows
    across the **59** census runs are in the second sub-case, spread over
    **5** runs; on **2** of them it is the LAST row
    (``cfm_fm_control_2026_10_09``, ``cfm_bond_head_30ep_onecycle_2026_10_09``),
    and the committed artifact published their undefined rate as a measured
    ``0.0`` with a ``fragmentation_gate`` of FAIL on it.

    THE REPAIR (Wf-A127): the census now nulls a chemistry rate whose
    scored denominator is empty, so those two runs' gates become SKIP /
    ``denominator_empty`` instead of FAIL on a 0.0 that was never defined.
    The convention is the TRAINER's own (Wf-A114/T85,
    ``molmetal/tests/test_cfm_gen_metric_sentinel.py``): a rate over the
    SCORED rows is ``None`` when nothing was scored, and the end-to-end
    rate inherits that undefined factor; a rate over the DRAWN samples
    (``gen_validity``, ``gen_molecule_validity``) stays a number, and a
    measured 0 of N stays ``0.0``.

    **THE COMMITTED ARTIFACT IS NOT REGENERATED HERE** (AGENTS.md §5.5
    rule 6 — published metrics are append/supersede, never silently
    edited), so it ~~still carries the **11 / 16** split and the two 0.0
    rates~~ **now carries the 11 / 18 split and two `null` rates** — the
    committed artifact was re-published by Wf-A127 and again by Wf-A150,
    and `changelog.wf_a127` records the move.  A
    regeneration moves those **2** runs from FAIL to SKIP /
    ``denominator_empty`` — **11 / 18**, ``('FAIL','end_to_end')`` 27 -> 25
    — and every cohort aggregate that carried their placeholder 0.0
    changes with them.  MEASURED 2026-10-31 by regenerating the census
    into a scratch directory (the published file untouched) and diffing it
    against a regeneration from the pre-repair code, leaf by leaf: **0**
    differing leaves between the committed artifact and the pre-repair
    regeneration, and **254** differing leaves between the pre-repair and
    the repaired regeneration — **251** of them data (the two runs' 3
    end-rate fields + their 2 gates + the cohort / ``budget_matched``
    aggregates that carried their placeholder 0.0) and **3** of them this
    wave's own ``changelog.wf_a127`` entries.  Re-quoting those
    aggregates is a wave action.
    """
    from molmetal.scripts.cfm_fragmentation_gate import (
        DEFAULT_MAX_FRAGMENTED_RATIO,
        DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
    )

    if single_fragment_ratio is None:
        if field_present:
            return {
                "status": "SKIP",
                "reason": _empty_denominator_reason(
                    gen_validity_n, gen_validity_val
                ),
                "basis": "denominator_empty",
            }
        return {
            "status": "SKIP",
            "reason": (
                "gen_single_fragment_ratio was never recorded in any epoch row "
                "of this run (no epoch row carries a value for it)"
            ),
            "basis": "field_absent",
        }
    single = float(single_fragment_ratio)
    if gen_validity is not None:
        measured = float(gen_validity) * single
        basis = "end_to_end"
    else:
        measured = single
        basis = "conditional"
    fragmented = 1.0 - measured
    reasons = []
    status = "PASS"
    if fragmented > DEFAULT_MAX_FRAGMENTED_RATIO:
        status = "FAIL"
        reasons.append(
            f"fragmented_ratio {fragmented:.4f} > "
            f"{DEFAULT_MAX_FRAGMENTED_RATIO:.4f}"
        )
    if (
        largest_fragment_frac is not None
        and float(largest_fragment_frac) < DEFAULT_MIN_LARGEST_FRAGMENT_FRAC
    ):
        status = "FAIL"
        reasons.append(
            f"largest_fragment_frac {float(largest_fragment_frac):.4f} < "
            f"{DEFAULT_MIN_LARGEST_FRAGMENT_FRAC:.4f}"
        )
    if basis == "conditional":
        reasons.append(
            "NO gen_validity column for this run: the verdict is on the "
            "CONDITIONAL single_fragment_ratio (over the parseable rows "
            "only), which overstates the generator"
        )
    return {
        "status": status,
        "fragmented_ratio": fragmented,
        "measured_rate": measured,
        "single_fragment_ratio_conditional": single,
        "basis": basis,
        "reasons": reasons,
        "thresholds": {
            "max_fragmented_ratio": DEFAULT_MAX_FRAGMENTED_RATIO,
            "min_largest_fragment_frac": DEFAULT_MIN_LARGEST_FRAGMENT_FRAC,
        },
    }


def _gv(row: Dict[str, object]) -> Optional[float]:
    """The parse rate over the samples the eval actually drew.

    ``gen_validity_n == 0`` means the eval drew nothing, so there is no rate
    — that is None, never 0.0.
    """
    n, v = _f(row, "gen_validity_n"), _f(row, "gen_validity_val")
    return v / n if n and v is not None else None


#: Wf-A99 — one glyph per definition so the table stays narrow.
_DEF_CODE = {
    "existence_ap_v2": "v2",
    "no_bond_ap_v1": "v1",
    "undeclared": "?",
}

#: Wf-A115 — the failure-mode threshold.  0.5 is the project's OWN
#: ``DEFAULT_MAX_FRAGMENTED_RATIO`` (``cfm_fragmentation_gate.py``), so the
#: label boundary is the same boundary the gate already draws — no new
#: number is invented here.
_FRAGMENTATION_THRESHOLD = 0.50

_DATE_SUFFIX = re.compile(r"(\d{4})_(\d{2})_(\d{2})")


def _normalise_row(row: Dict[str, object], source: str) -> Dict[str, object]:
    """One per-epoch row, in one shape, whichever file it came from.

    Wf-A115 — the CSV and the JSON do NOT have the same column names and do
    NOT have the same column SET.  MEASURED: the JSON row keys are the
    dataclass field names (``decode_ratio_rdkit_strict``, not
    ``decode_ratio``), and older runs' CSVs carry fewer columns than newer
    ones (``cfm_tmqm_all6_retrain`` has 18 columns, ~~``cfm_r9_fix_v2`` has
    39~~ **CORRECTION (2026-10-31, Wf-A153): ``cfm_r9_fix_v2``'s
    ``per_epoch_metrics.csv`` has 35 columns, not 39** — MEASURED 2026-10-31
    by counting the header fields of
    ``molmetal/reports/cfm_r9_fix_v2/per_epoch_metrics.csv`` (35 fields; the
    11 rows of its ``per_epoch_metrics.json`` carry 44 distinct keys) — so a
    column that is absent from one run's CSV is not a zero, it is a column
    that run never wrote.

    The generation RATE columns are nulled HERE when the eval did not run
    (``gen_validity_n == 0``), because in the CSV those cells carry the
    trainer's ``0.0000`` placeholder and in the JSON they carry ``null``.
    A 0.0 that was never measured is not a measurement.

    Wf-A127 (2026-10-31) — THE CHEMISTRY RATES HAVE A SECOND, DIFFERENT
    DENOMINATOR, AND THE 0.0 PLACEHOLDER SLIPS THROUGH THE FIRST ONE.
    ------------------------------------------------------------------
    There are TWO denominators in a generation-eval row, and the single
    predicate ``gen_validity_n > 0`` only guards the first:

    * ``gen_validity`` — the RDKit PARSE rate.  Its denominator is the
      number of samples the eval DREW (``gen_validity_n``).
    * ``gen_single_fragment_ratio`` / ``gen_largest_fragment_frac`` /
      ``gen_metal_ratio`` — the CHEMISTRY rates.  Their denominator is the
      number of rows the eval SCORED, i.e. the rows RDKit accepted
      (``gen_validity_val``; see ``summarise_decode_chemistry``'s
      ``n_chemistry_scored``, which is exactly the accepted-row count).

    When the eval drew samples but NONE of them parsed, ``gen_validity_n``
    is > 0 (the first predicate passes) while the SCORED count is 0, so
    every chemistry rate is 0/0 — UNDEFINED — and the trainer publishes
    its 0.0 placeholder.  MEASURED 2026-10-31 on disk: **66** of the
    **576** per-epoch rows across the **59** census runs have
    ``gen_validity_val == 0`` with ``gen_validity_n > 0``, spread over
    **5** runs; on **2** of them it is the LAST row, which is where the
    census publishes it as a measurement.

    So the chemistry rates are nulled when the SCORED count is empty, and
    the drawn-count predicate keeps governing the drawn-denominator rates
    (``gen_validity``, ``gen_molecule_validity``).  This is the SAME
    convention the trainer itself uses since Wf-A114/T85
    (``_chem_val(..., scored_denominator=True)`` returns None when
    ``n_chemistry_scored == 0``): the census was simply not yet reading
    the pre-Wf-A114 artifacts through that rule.
    """
    n = _f(row, "gen_validity_n")
    v = _f(row, "gen_validity_val")
    measured = n is not None and n > 0
    # Wf-A127 — the SCORED-row denominator is empty for this row.  Only
    # PROVED emptiness nulls a value: a row that does not carry
    # `gen_validity_val` at all says nothing about its denominator, so its
    # value is kept (it may be a real measurement).
    scored_empty = bool(measured and v is not None and v == 0)
    out: Dict[str, object] = {
        "epoch": _epoch_index(row),
        "train_loss": _f(row, "train_loss"),
        "train_loss_nan": _is_nan(row, "train_loss"),
        "gen_validity_n": int(n) if n is not None else None,
        "gen_validity_val": int(v) if v is not None else None,
        "gen_validity": (v / n) if (n and v is not None) else None,
        "gen_eval_ran": measured,
        # Bond columns — same names in both files.
        "bond_f1": _f(row, "bond_f1"),
        "bond_type_macro_f1_pooled": _f(row, "bond_type_macro_f1_pooled"),
        "bond_exact_type_rate": _f(row, "bond_exact_type_rate"),
        "bond_pr_auc": _f(row, "bond_pr_auc"),
        "bond_pr_auc_definition": row.get("bond_pr_auc_definition"),
        # Wf-A115 -- the criterion that ACTUALLY gated this epoch.  The
        # trainer DEGRADES an unmeasurable criterion, so the resolved name
        # is not always the requested one; without this column the census
        # cannot see a run that gated on something its config never asked
        # for.  See `_early_stop_guard`.
        "early_stop_metric": row.get("early_stop_metric"),
        "early_stop_value": _f(row, "early_stop_value"),
        "early_stop_criterion_note": row.get("early_stop_criterion_note"),
    }
    # Wf-A127 — the two denominator families are separated.  The
    # SCORED-denominator rates are undefined when the eval scored no rows;
    # the DRAWN-denominator rates are undefined when the eval drew none.
    for key in (
        "gen_single_fragment_ratio",
        "gen_largest_fragment_frac",
        "gen_metal_ratio",
    ):
        raw = _f(row, key)
        out[key + "_raw"] = raw
        out[key] = (
            None if (scored_empty or not measured) else raw
        )
    for key in ("gen_molecule_validity",):
        raw = _f(row, key)
        out[key + "_raw"] = raw
        out[key] = raw if (measured and raw is not None) else None
    # `decode_ratio` is named `decode_ratio` in the CSV and
    # `decode_ratio_rdkit_strict` in the JSON; they are the same number
    # (the CSV renders it to 4 decimals).
    dec = _f(
        row,
        "decode_ratio_rdkit_strict" if source == "json" else "decode_ratio",
    )
    # Wf-A115 -- the DECODE path is a DIFFERENT measurement from the
    # GENERATION eval, and it has its OWN denominator: `decode_ratio_n` is
    # the number of val-split geometries the bond decoder was run on.  The
    # generation-eval predicate (`gen_validity_n == 0`) says NOTHING about
    # it, and using that predicate here nulled 27 of 59 runs' real
    # `decode_ratio` (MEASURED: e.g. `c7_b32_4096_2026_10_20` publishes
    # decode_ratio = 1.0 with decode_ratio_n = 192 while its generation
    # eval drew nothing).
    #
    # Older CSVs carry no `decode_ratio_n` column at all (the column was
    # added later), so the predicate falls back to "a ratio was recorded at
    # all" -- a recorded ratio IS a measurement.
    dec_n = _f(row, "decode_ratio_n")
    decode_measured = bool(
        (dec_n is not None and dec_n > 0)
        or (dec_n is None and dec is not None)
    )
    out["decode_ratio_n"] = int(dec_n) if dec_n is not None else None
    out["decode_measured"] = decode_measured
    out["decode_ratio"] = dec
    out["decode_ratio_measured"] = dec
    dsfr = _f(row, "decode_single_fragment_ratio")
    out["decode_single_fragment_ratio_raw"] = dsfr
    out["decode_single_fragment_ratio"] = dsfr if decode_measured else None
    return out


def _load_series(d: Path) -> Tuple[List[Dict[str, object]], str, Dict]:
    """(rows, series_source, metrics_json) for one run directory.

    Wf-A115 — the JSON is preferred because the CSV rounds every rate to 4
    decimals, and a census whose published bar disagrees with its own
    artifact by 1.7e-5 is a census that cannot be checked.
    """
    jd: Dict = {}
    jp = d / "per_epoch_metrics.json"
    if jp.is_file():
        try:
            jd = json.load(jp.open(encoding="utf-8"))
        except Exception:
            jd = {}
    pem = jd.get("per_epoch_metrics")
    if isinstance(pem, list) and pem:
        return [_normalise_row(r, "json") for r in pem], "json", jd
    csvp = d / "per_epoch_metrics.csv"
    rs = list(csv.DictReader(csvp.open(encoding="utf-8")))
    return [_normalise_row(r, "csv") for r in rs], "csv", jd


#: Wf-A114E — the append-only event stream every CFM run writes beside its
#: per-epoch metrics.  ``per_epoch_metrics.json`` is written ONCE, at the end
#: of training, to a FIXED path, so two runs launched into the same
#: ``--out-dir`` clobber each other's record and the survivor is whichever
#: finished LAST.  ``events.jsonl`` is opened in APPEND mode, so it keeps
#: EVERY population that ever wrote into the directory — including each
#: one's ``done`` event.  MEASURED: 51 of the 60 run directories carry one.
EVENTS_FILENAME = "events.jsonl"

#: Wf-A114E — the event kinds the census reads.  ``train_epoch`` is the ROW
#: source (one row per epoch, carrying the same measurements the per-epoch
#: file carries); ``done`` closes a POPULATION (one process that trained in
#: this directory) and declares the epoch budget it was given and the epochs
#: it ran; ``gen_eval`` is one row per (epoch, seed) and is used ONLY to
#: cross-check the epoch set — MEASURED, one epoch can carry rows from more
#: than one population, so it cannot delimit a population on its own.
EVENT_KIND_TRAIN_EPOCH = "train_epoch"
EVENT_KIND_DONE = "done"
EVENT_KIND_GEN_EVAL = "gen_eval"


def _read_events(d: Path) -> Optional[Dict[str, object]]:
    """The event stream of one run directory, or None when it has none.

    Wf-A114E — the stream is append-only and ordered by ``ts``, so it holds
    every population that ever wrote into the directory.  A malformed line is
    SKIPPED, not fatal: a census that dies on one bad line reads nothing.
    """
    p = d / EVENTS_FILENAME
    if not p.is_file():
        return None
    train_epoch: List[Dict[str, object]] = []
    done: List[Dict[str, object]] = []
    gen_eval: List[Dict[str, object]] = []
    gen_eval_epochs: set = set()
    try:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                kind = obj.get("event")
                if kind == EVENT_KIND_TRAIN_EPOCH:
                    train_epoch.append(obj)
                elif kind == EVENT_KIND_DONE:
                    done.append(obj)
                elif kind == EVENT_KIND_GEN_EVAL:
                    gen_eval.append(obj)
                    ep = _epoch_index(obj)
                    if ep is not None:
                        gen_eval_epochs.add(ep)
    except OSError:
        return None
    train_epoch.sort(key=lambda r: (r.get("ts") or 0.0))
    done.sort(key=lambda r: (r.get("ts") or 0.0))
    gen_eval.sort(key=lambda r: (r.get("ts") or 0.0))
    return {
        "present": True,
        "train_epoch": train_epoch,
        "done": done,
        "gen_eval": gen_eval,
        "gen_eval_epochs": sorted(gen_eval_epochs),
        "segments": _event_segments(train_epoch, done, gen_eval),
    }


def _event_segments(
    train_epoch: List[Dict[str, object]],
    done: List[Dict[str, object]],
    gen_eval: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    """One SEGMENT per ``done`` event: the population that event closed.

    Wf-A114E — the boundary is the ``done`` event's ``ts``: segment ``i``
    holds the ``train_epoch`` rows whose ``ts`` falls in
    ``(ts(done[i-1]), ts(done[i])]``.  A segment can hold MORE epochs than
    its ``done`` event ran — the LEADING extra epochs are a killed
    pre-resume attempt that wrote into the same directory.  MEASURED:
    ``cfm_c7_fullgen_2026_10_20``'s segment holds 18 epochs while its
    ``done`` event ran 14, and the 14 are exactly epochs 4..17.

    The ``gen_eval`` rows are segmented on the same window: they are one row
    per (epoch, seed) and carry the DENOMINATOR the generation eval used
    (``n_total``), which is the validation size the run of record measured
    against — the one cohort fact the per-epoch file lost when it was
    overwritten.
    """
    segs: List[Dict[str, object]] = []
    prev: Optional[float] = None
    for i, dn in enumerate(done):
        hi = dn.get("ts")
        hi_val = float(hi) if isinstance(hi, (int, float)) else float("inf")
        rows = [
            r for r in train_epoch
            if (prev is None or (r.get("ts") or 0.0) > prev)
            and (r.get("ts") or 0.0) <= hi_val
        ]
        grows = [
            r for r in (gen_eval or [])
            if (prev is None or (r.get("ts") or 0.0) > prev)
            and (r.get("ts") or 0.0) <= hi_val
        ]
        segs.append({
            "index": i,
            "done_ts": hi,
            "n_epochs_run": dn.get("n_epochs_run"),
            "declared_epochs": dn.get("n_epochs"),
            "epoch_indices": sorted({
                e for e in (_epoch_index(r) for r in rows) if e is not None
            }),
            "gen_eval_n_total": sorted({
                r.get("n_total") for r in grows if r.get("n_total") is not None
            }),
            "gen_eval_n_per_seed": sorted({
                r.get("n_per_seed")
                for r in grows
                if r.get("n_per_seed") is not None
            }),
            "rows": rows,
        })
        prev = hi_val
    return segs


def _run_of_record_index(segs: List[Dict[str, object]]) -> Optional[int]:
    """The index of the RUN OF RECORD: the longest population, earliest tie.

    Wf-A114E — a directory can hold several populations (a real run, then a
    smoke that overwrote the metrics file).  The run of record is the one
    that trained the MOST epochs; ties are broken by the earliest ``done``
    timestamp.  MEASURED: ``cfm_fullcorpus_2026_10_10`` holds a 14-epoch
    population followed by two 1-epoch ones.
    """
    if not segs:
        return None

    def _key(i: int) -> Tuple[int, float]:
        n = segs[i].get("n_epochs_run")
        n = int(n) if isinstance(n, (int, float)) else 0
        ts = segs[i].get("done_ts")
        ts = float(ts) if isinstance(ts, (int, float)) else 0.0
        return (n, -ts)

    return max(range(len(segs)), key=_key)


def _events_integrity(
    decision: Dict[str, object], ev: Optional[Dict[str, object]]
) -> Dict[str, object]:
    """The flat ``integrity.*`` block that records the population cross-check.

    Wf-A114E — every field is a COUNT or an INDEX SET, so a reader can
    re-derive the verdict from the artifact alone: how many ``done`` events
    the stream holds, how many epochs each declared, which epoch indices the
    stream carries, and which of those the census scored.
    """
    segs = list(ev.get("segments") or []) if ev else []
    done = list(ev.get("done") or []) if ev else []
    te = list(ev.get("train_epoch") or []) if ev else []
    eps = sorted({e for e in (_epoch_index(r) for r in te) if e is not None})
    n_run = [
        int(dn["n_epochs_run"])
        for dn in done
        if isinstance(dn.get("n_epochs_run"), (int, float))
    ]
    ro_idx = decision.get("run_of_record_segment_index")
    ro_seg = None
    for s in segs:
        if s.get("index") == ro_idx:
            ro_seg = s
            break
    return {
        "events_present": bool(ev),
        "events_done_count": len(done),
        "events_done_n_epochs_run": n_run,
        "events_max_n_epochs_run": max(n_run) if n_run else None,
        "events_train_epoch_row_count": len(te),
        "events_epoch_indices": eps,
        "events_epoch_index_start": eps[0] if eps else None,
        "events_epoch_index_end": eps[-1] if eps else None,
        "events_gen_eval_epoch_indices": (
            list(ev.get("gen_eval_epochs") or []) if ev else []
        ),
        # Wf-A114E — the GENERATION SAMPLING budget the run of record ran
        # with (`n_total` / `n_per_seed` in the `gen_eval` event rows: the
        # molecules sampled per seed, NOT the cohort size).  It is one of the
        # three population-identifying signals the diagnosis names, and it is
        # the only one the events stream still carries for a run whose
        # per-epoch file was overwritten.  `n_train`/`n_val` are NOT in the
        # stream, so they stay null for a re-derived run.
        "events_gen_eval_n_total": list(
            (ro_seg or {}).get("gen_eval_n_total") or []
        ),
        "events_gen_eval_n_per_seed": list(
            (ro_seg or {}).get("gen_eval_n_per_seed") or []
        ),
        "events_segment_count": len(segs),
        "events_segments": [
            {
                "index": s.get("index"),
                "done_ts": s.get("done_ts"),
                "n_epochs_run": s.get("n_epochs_run"),
                "declared_epochs": s.get("declared_epochs"),
                "epoch_indices": list(s.get("epoch_indices") or []),
                "gen_eval_n_total": list(s.get("gen_eval_n_total") or []),
            }
            for s in segs
        ],
        "run_of_record_segment_index": decision.get(
            "run_of_record_segment_index"
        ),
        "run_of_record_epoch_indices": list(
            decision.get("run_of_record_epoch_indices") or []
        ),
        "population_conflict": bool(decision.get("conflict")),
        "population_resolved": bool(decision.get("resolved")),
        # Wf-A116 — the run whose record could NOT be corroborated by the
        # events stream (the events stream is a FOREIGN launch).  It keeps
        # its own per-epoch file and its row; it is flagged, not refused.
        "population_uncorroborated": bool(decision.get("uncorroborated")),
        # Wf-A116 — WHICH population the per-epoch file turned out to be
        # (None = the run of record), and the epochs on which the two
        # records disagree.
        "population_matched_segment_index": decision.get(
            "matched_segment_index"
        ),
        "population_disagreed_epochs": list(
            decision.get("disagreed_epochs") or []
        ),
        "population_verdict": str(decision.get("verdict") or ""),
        "population_reason": str(decision.get("reason") or ""),
        "population_source": str(decision.get("source") or ""),
    }


#: Wf-A116 — the tolerance of the population cross-check.  The per-epoch CSV
#: stores ``train_loss`` rounded to 5 decimals and the JSON stores it at full
#: precision, so two records of the SAME measurement differ by the rounding
#: and by nothing else.  MEASURED: the file's epoch 1 of
#: ``cfm_bond_head_20ep_2026_10_09`` reads 0.83485 (5 dp) while the matching
#: events row reads 0.834846742451191.
POP_MATCH_RTOL = 1e-4


def _loss_value(
    row: Dict[str, object], nan_key: str = ""
) -> Tuple[Optional[float], bool]:
    """``(value, is_nan)`` of one row's ``train_loss``.

    Wf-A116 — a MISSING loss and a NAN loss are different facts: the first
    says the row never carried the column, the second says the loss did not
    come out.  The normalised per-epoch rows keep that distinction in
    ``train_loss_nan``; the raw events rows keep it in the value itself.
    """
    if nan_key:
        return _num(row.get("train_loss")), bool(row.get(nan_key))
    raw = row.get("train_loss")
    if raw is None:
        return None, False
    if isinstance(raw, str):
        s = raw.strip()
        if s.lower() == "nan":
            return None, True
        try:
            x = float(s)
        except ValueError:
            return None, False
        return (None, True) if x != x else (x, False)
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return None, False
    return (None, True) if x != x else (x, False)


def _loss_agreement(
    a: Optional[float],
    a_nan: bool,
    b: Optional[float],
    b_nan: bool,
) -> Optional[bool]:
    """Do two ``train_loss`` values describe the same measurement?

    ``True`` = the same, ``False`` = a disagreement, ``None`` = SKIP (one of
    the two sides carries no loss at all, so the pair says nothing).
    """
    if a is None and not a_nan:
        return None
    if b is None and not b_nan:
        return None
    if a_nan and b_nan:
        return True
    if a_nan or b_nan:
        return False
    return abs(a - b) <= POP_MATCH_RTOL * max(1.0, abs(a), abs(b))


def _population_match(
    file_rows: List[Dict[str, object]],
    pop_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """Do the per-epoch FILE rows and one events POPULATION describe the SAME population?

    Wf-A116 — the cross-check used to compare EPOCH INDEX SETS only, so it
    could not tell a truncated record of the run of record from ANOTHER
    launch's record that happened to occupy the same directory.  It is now
    VALUE-AWARE and SET-AWARE:

      * for every epoch index the file carries, there must EXIST a row in the
        population with that epoch index whose ``train_loss`` agrees within
        ``POP_MATCH_RTOL``; ``nan == nan`` agrees, and a MISSING loss on
        either side is SKIPPED and never counted as a disagreement;
      * "SOME row at that epoch" — one epoch can carry rows from more than
        one population (MEASURED: ``cfm_bond_head_20ep_2026_10_09``'s
        population holds epoch 1 twice, 0.82975046 from a killed pre-resume
        attempt and 0.83484674 from the run of record), so the FIRST row at
        an epoch index is not the row to compare against.

    The returned dict carries the disagreements and the epochs the population
    does not carry, so the verdict can NAME them.
    """
    by_epoch: Dict[int, List[Dict[str, object]]] = {}
    for r in pop_rows:
        e = _epoch_index(r)
        if e is not None:
            by_epoch.setdefault(e, []).append(r)
    conflicts: List[Dict[str, object]] = []
    absent: List[Dict[str, object]] = []
    shared: List[int] = []
    for r in file_rows:
        e = _epoch_index(r)
        if e is None:
            continue
        cands = by_epoch.get(e)
        if not cands:
            absent.append({
                "epoch": e,
                "file_train_loss": _num(r.get("train_loss")),
            })
            continue
        fv, fnan = _loss_value(r, "train_loss_nan")
        pop_vals: List[Optional[float]] = []
        decided: Optional[bool] = None
        for c in cands:
            pv, pnan = _loss_value(c)
            pop_vals.append(pv)
            res = _loss_agreement(fv, fnan, pv, pnan)
            if res is None:
                # one side carries no loss: the pair says nothing
                continue
            if res:
                decided = True
                break
            if decided is None:
                decided = False
        shared.append(e)
        if decided is False:
            conflicts.append({
                "epoch": e,
                "file_train_loss": fv,
                "population_train_loss": pop_vals,
            })
    return {
        "same": not conflicts and not absent,
        "conflicts": conflicts,
        "absent_epochs": absent,
        "shared_epochs": shared,
    }


def _fmt_conflicts(conflicts: List[Dict[str, object]]) -> str:
    """One line per disagreeing epoch: ``epoch 0: file 2.07847 vs events 1.40822``."""
    out: List[str] = []
    for c in conflicts:
        fv = c.get("file_train_loss")
        pv = c.get("population_train_loss") or []
        pvs = "/".join(
            "nan" if v is None else f"{v:.5f}" for v in pv
        ) or "none"
        out.append(
            f"epoch {c.get('epoch')}: file "
            f"{'nan' if fv is None else f'{fv:.5f}'} vs events {pvs}"
        )
    return "; ".join(out)


def _rederive_from_events(
    seg: Dict[str, object], R: List[int]
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """The run of record's OWN events rows, normalised, and the config they carry.

    Wf-A116 — one code path for both "the file is a truncated prefix of the
    run of record" and "the file is ANOTHER population's record": in both
    cases the census scores the RUN OF RECORD and re-derives its series from
    the events rows, so the numbers are identical either way.
    """
    want = set(R)
    rws = [r for r in seg.get("rows") or [] if _epoch_index(r) in want]
    rws.sort(key=lambda r: (_epoch_index(r) or 0))
    cfg: Dict[str, object] = {}
    declared = seg.get("declared_epochs")
    if isinstance(declared, (int, float)):
        cfg["epochs"] = int(declared)
    for r in rws:
        lr = _num(r.get("lr"))
        if lr is not None:
            cfg["lr"] = lr
            break
    return [_normalise_row(r, "events") for r in rws], cfg


def _population_decision(
    census_rows: List[Dict[str, object]], ev: Optional[Dict[str, object]]
) -> Dict[str, object]:
    """Which POPULATION the census is scoring for this run — and on what basis.

    Wf-A114E — the verdict is decided PER RUN, because the classes are not
    interchangeable: a RESUMED run's stream carries the pre-resume epochs
    (re-importing them would double-count), and a PARTIAL stream carries
    fewer epochs than the metrics file (trusting it would lose epochs).

    Wf-A116 — the cross-check is now VALUE-AWARE.  It used to compare EPOCH
    INDEX SETS only, so a per-epoch file written by a DIFFERENT launch that
    reused the directory was scored as if the two records were one
    population.  MEASURED: ``wf299_cfm_main/wave_a/full_stack``'s file
    (train_loss 2.07847 / 1.84889 / 1.65890) and its events rows (1.40822 /
    1.22302) DISAGREE at every shared epoch, and the census called that
    "the events stream is a PARTIAL record" — MEASURED FALSE.

    The decision table, first match wins:

      * ``events_absent``            — no ``events.jsonl``: nothing to check.
      * ``class_c_events_shorter``   — the stream holds FEWER epochs than the
        metrics file AND agrees with it on every epoch they share.  The file
        wins; the stream is a PARTIAL record of the same run.
      * ``events_no_done``           — the stream holds no ``done`` event, so
        it cannot be segmented into populations at all, and it agrees with
        the file.
      * ``no_conflict``              — the file IS the run of record.
      * ``class_b_resume_prehistory``— the file IS the run of record, but the
        segment also holds a killed pre-resume attempt.  NOT a defect; the
        run stays out of the CLEAN population because it resumed.
      * ``class_a_truncated_prefix`` — the file is a contiguous PREFIX of the
        run of record AND agrees with it on every epoch they share: a
        genuine truncation of the SAME population.  The census re-derives
        the series from the events rows of the matching population.
      * ``class_b_resume_tail``      — the file's epochs are a subset of the
        run of record but NOT a prefix (the file holds the earlier ones).
      * ``class_a_other_population`` — the file is ANOTHER population's
        record (its values match a different segment of the same stream and
        DISAGREE with the run of record).  The census scores the run of
        record and re-derives the series from its events rows.
      * ``class_d_foreign_launch``   — the file matches NO population: the
        events stream is a FOREIGN launch that reused the directory.  The
        file is the run's own record and is KEPT, and the run is flagged
        UNCORROBORATED.
      * ``unmatched_conflict``       — the file's epochs and the run of
        record's epochs are disjoint: the two records disagree about which
        epochs this run trained.  The run is REFUSED.

    NOTE on the order of the first two rules: the draft table put "no done
    event" first, but MEASURED that collapses ``class_c`` into "no done
    event" for the one run that is BOTH (the stream is short AND has no done
    event).  The class-C test is therefore evaluated FIRST, so the verdict
    names the more specific fact; the reason string still records that the
    stream carries no ``done`` event.
    """
    C = sorted({
        e for e in (_epoch_index(r) for r in census_rows) if e is not None
    })
    out: Dict[str, object] = {
        "verdict": "events_absent",
        "conflict": False,
        "resolved": False,
        "uncorroborated": False,
        "reason": "",
        "run_of_record_segment_index": None,
        "run_of_record_epoch_indices": [],
        "run_of_record_ts": [],
        "matched_segment_index": None,
        "disagreed_epochs": [],
        "rows": None,
        "config": None,
        "done": None,
        "source": "per_epoch_metrics.json",
    }
    if ev is None:
        out["reason"] = (
            "this run directory has no events.jsonl, so the population "
            "cross-check has no second source and the per-epoch file is the "
            "only record of this run"
        )
        return out

    te = list(ev.get("train_epoch") or [])
    E = sorted({e for e in (_epoch_index(r) for r in te) if e is not None})
    segs = list(ev.get("segments") or [])
    done = list(ev.get("done") or [])

    # -- class C: the events stream is the SHORTER record ------------------
    if len(E) < len(C):
        m_all = _population_match(census_rows, te)
        if not m_all["conflicts"] and set(E).issubset(set(C)):
            out["verdict"] = "class_c_events_shorter"
            out["reason"] = (
                f"events.jsonl carries {len(E)} epoch(s) {E} and the per-epoch "
                f"file carries {len(C)}; the events stream is a PARTIAL "
                f"record — it agrees with the per-epoch file at every epoch "
                f"they share ({m_all['shared_epochs']}) and is missing "
                f"epoch(s) {sorted(set(C) - set(E))} — so the per-epoch file "
                f"is kept and no epoch is lost"
                + (
                    " (the stream also carries no done event, so it cannot be "
                    "segmented into populations)"
                    if not segs else ""
                )
            )
            return out
        # else: the two records DISAGREE on the values, so the class-C
        # reading ("a partial record of this run") is MEASURED FALSE and the
        # population cross-check below decides the verdict.

    # -- the populations of the stream -------------------------------------
    # A segment is one population: the rows between two ``done`` events.  Its
    # OWN rows are its LAST ``n_epochs_run`` epochs — the leading extra ones
    # are a killed pre-resume attempt.
    pops: List[Dict[str, object]] = []
    for s in segs:
        eps = list(s.get("epoch_indices") or [])
        nr = s.get("n_epochs_run")
        nr = int(nr) if isinstance(nr, (int, float)) else 0
        own = set(eps[-nr:]) if nr else set()
        rows = [
            r for r in s.get("rows") or []
            if (not nr) or _epoch_index(r) in own
        ]
        pops.append({"index": s.get("index"), "rows": rows, "segment": s})
    if not pops:
        # no ``done`` event: the whole stream is one un-segmented population
        pops.append({"index": None, "rows": list(te), "segment": None})

    i_star = _run_of_record_index(segs) if segs else None
    seg = segs[i_star] if i_star is not None else None
    n_run = seg.get("n_epochs_run") if seg else None
    n_run = int(n_run) if isinstance(n_run, (int, float)) else 0
    seg_epochs = list(seg.get("epoch_indices") or []) if seg else []
    # The population's OWN epochs are the LAST ``n_epochs_run`` epochs of the
    # segment: any leading extra epochs are a killed pre-resume attempt.
    R = seg_epochs[-n_run:] if n_run else []
    want_epochs = set(R)
    ro_rows = (
        [r for r in seg.get("rows") or [] if _epoch_index(r) in want_epochs]
        if seg else list(te)
    )
    out["run_of_record_segment_index"] = i_star
    out["run_of_record_epoch_indices"] = list(R)
    # Wf-A115 — the ``ts`` of the run-of-record population's OWN epoch rows.
    # These are the timestamps that TIE a log to this population: the trainer
    # stamps every ``train_epoch`` row and prints the whole event dict, so a
    # log echoes the same ``ts`` values.  A log that echoes every one of them
    # is this population's log; a log that echoes only some of them belongs to
    # a sibling population.  See ``molmetal/scripts/cfm_cohort_recovery.py``.
    out["run_of_record_ts"] = [
        float(r["ts"])
        for r in ((seg.get("rows") or []) if seg else [])
        if _epoch_index(r) in want_epochs
        and isinstance(r.get("ts"), (int, float))
    ]
    if i_star is not None:
        out["done"] = done[i_star]

    m_ro = _population_match(census_rows, ro_rows)
    out["disagreed_epochs"] = [c.get("epoch") for c in m_ro["conflicts"]]

    if m_ro["same"]:
        # -- the file IS the run of record: today's length-based classes ---
        if i_star is None:
            out["verdict"] = "events_no_done"
            out["reason"] = (
                f"events.jsonl carries {len(E)} epoch(s) {E} but no done "
                f"event, so it cannot be segmented into populations; it "
                f"agrees with the per-epoch file at every epoch they share "
                f"({m_ro['shared_epochs']}), and the per-epoch file carries "
                f"{len(C)} and is kept"
            )
            return out
        if C == R and len(seg_epochs) == n_run:
            out["verdict"] = "no_conflict"
            out["reason"] = (
                f"the per-epoch file's {len(C)} epoch(s) ARE the run of record "
                f"(segment {i_star}, {n_run} epoch(s) run, no extra epochs in "
                f"the segment)"
            )
        elif C == R and len(seg_epochs) > n_run:
            out["verdict"] = "class_b_resume_prehistory"
            out["reason"] = (
                f"the per-epoch file's {len(C)} epoch(s) ARE the run of record, "
                f"but segment {i_star} also holds {len(seg_epochs) - n_run} "
                f"EARLIER epoch(s) {seg_epochs[:len(seg_epochs) - n_run]} from a "
                f"killed pre-resume attempt in the same directory; the run "
                f"RESUMED, so its history is not contiguous from zero and it "
                f"stays out of the CLEAN population"
            )
        elif len(C) < len(R) and R[:len(C)] == C:
            # -- CLASS A: the census file is a TRUNCATED PREFIX of the SAME
            #    population (the values agree at every shared epoch) -------
            rows, cfg = _rederive_from_events(seg, R)
            out["verdict"] = "class_a_truncated_prefix"
            out["conflict"] = True
            out["resolved"] = True
            out["rows"] = rows
            out["config"] = cfg
            out["source"] = "events.jsonl"
            out["reason"] = (
                f"the per-epoch file holds {len(C)} epoch(s) {C}, a contiguous "
                f"PREFIX of the run of record (segment {i_star}: {len(R)} "
                f"epoch(s) {R[0]}..{R[-1]}) — the file was OVERWRITTEN by a "
                f"shorter run that wrote into the same directory, and it "
                f"agrees with the run of record at every epoch they share "
                f"({m_ro['shared_epochs']}). The series is RE-DERIVED from the "
                f"events rows of the matching population; the truncated prefix "
                f"is not scored"
            )
        elif set(C).issubset(set(R)):
            out["verdict"] = "class_b_resume_tail"
            out["reason"] = (
                f"the per-epoch file's {len(C)} epoch(s) {C} are a SUBSET of the "
                f"run of record ({len(R)} epoch(s) {R[0]}..{R[-1]}) but NOT a "
                f"prefix of it; the file holds the EARLIER epochs, so the run "
                f"RESUMED and the file is kept as the run's own record"
            )
        else:
            out["verdict"] = "unmatched_conflict"
            out["conflict"] = True
            out["resolved"] = False
            out["reason"] = (
                f"the per-epoch file's {len(C)} epoch(s) {C} and the run of "
                f"record's {len(R)} epoch(s) {R[0] if R else None}.."
                f"{R[-1] if R else None} are neither equal, nor a prefix, nor a "
                f"subset: the two records disagree about which epochs this run "
                f"trained. The run is REFUSED — it keeps its row and this "
                f"reason, and it is excluded from every headline population"
            )
        return out

    # -- the file is NOT the run of record ---------------------------------
    # Is it ANOTHER population's record of the same stream?
    other: Optional[Dict[str, object]] = None
    for p in pops:
        if p.get("index") == i_star:
            continue
        if _population_match(census_rows, p["rows"])["same"]:
            other = p
            break

    if other is not None:
        # -- the file is ANOTHER population's record ----------------------
        rows, cfg = _rederive_from_events(seg, R) if seg else ([], {})
        o_seg = other.get("segment") or {}
        out["verdict"] = "class_a_other_population"
        out["conflict"] = True
        out["resolved"] = True
        out["matched_segment_index"] = other.get("index")
        out["rows"] = rows
        out["config"] = cfg
        out["source"] = "events.jsonl"
        out["reason"] = (
            f"the per-epoch file's {len(C)} epoch(s) {C} are the record of "
            f"population {other.get('index')} (segment "
            f"{other.get('index')}, done ts {o_seg.get('done_ts')}), NOT of the "
            f"run of record (segment {i_star}, {len(R)} epoch(s) "
            f"{R[0] if R else None}..{R[-1] if R else None}): the two records "
            f"DISAGREE at epoch(s) {[c.get('epoch') for c in m_ro['conflicts']]}"
            f" ({_fmt_conflicts(m_ro['conflicts'])}). The census scores the RUN "
            f"OF RECORD and re-derives the series from its events rows; the "
            f"other population's file is not scored"
        )
        return out

    # -- the file matches NO population ------------------------------------
    if m_ro["conflicts"]:
        # -- the events stream is a FOREIGN launch ------------------------
        out["verdict"] = "class_d_foreign_launch"
        out["uncorroborated"] = True
        out["reason"] = (
            f"the per-epoch file's {len(C)} epoch(s) {C} and the events "
            f"stream's {len(E)} epoch(s) {E} DISAGREE on the VALUES at "
            f"epoch(s) {[c.get('epoch') for c in m_ro['conflicts']]} "
            f"({_fmt_conflicts(m_ro['conflicts'])}); the events stream matches "
            f"NO population of this directory — it is a FOREIGN launch that "
            f"reused the directory, not a partial record of this run. The "
            f"census does NOT assert which of the two records is the run: the "
            f"per-epoch file is KEPT as the run's own record (no epoch is "
            f"lost) and the run is flagged UNCORROBORATED — its record could "
            f"not be corroborated by a second source"
        )
        return out

    out["verdict"] = "unmatched_conflict"
    out["conflict"] = True
    out["resolved"] = False
    out["reason"] = (
        f"the per-epoch file's {len(C)} epoch(s) {C} and the run of "
        f"record's {len(R)} epoch(s) {R[0] if R else None}.."
        f"{R[-1] if R else None} are neither equal, nor a prefix, nor a "
        f"subset, and they share no epoch at all: the two records disagree "
        f"about which epochs this run trained. The run is REFUSED — it keeps "
        f"its row and this reason, and it is excluded from every headline "
        f"population"
    )
    return out


def _integrity(
    rows: List[Dict[str, object]],
    cfg: Dict,
    source: str,
    decision: Optional[Dict[str, object]] = None,
    ev: Optional[Dict[str, object]] = None,
) -> Dict:
    """What the run's own epoch record says about how much of it happened.

    Wf-A115 — ``n_epochs_run`` (a row count) is NOT "epochs the model
    trained".  MEASURED: ``cfm_r11_eval256`` has exactly ONE recorded epoch
    (index 19) with ``train_loss = nan`` out of a declared 20; ``cfm_c7_fullgen``
    starts at epoch 4; ``cfm_bond_head_20ep`` starts at epoch 1.  A run that
    RESUMED has no row for the epochs it did before the resume, so its
    history is not contiguous from zero and the census's "best epoch" scan
    can only see the epochs present in that run's own file.

    Wf-A114E — the block now also carries the POPULATION cross-check: the
    ``events.jsonl`` facts and the verdict the census reached about which
    population it is scoring.  Without it the block was internally consistent
    and externally wrong: a clobbered run looked like a complete 1-epoch run.
    """
    eps = [r["epoch"] for r in rows if r.get("epoch") is not None]
    uniq = sorted(set(eps))
    nan_eps = [
        r["epoch"]
        for r in rows
        if r.get("train_loss_nan") and r.get("epoch") is not None
    ]
    declared = cfg.get("epochs")
    try:
        declared_n = int(declared) if declared is not None else None
    except (TypeError, ValueError):
        declared_n = None
    start = uniq[0] if uniq else None
    end = uniq[-1] if uniq else None
    out = {
        "epoch_index_start": start,
        "epoch_index_end": end,
        "epochs_recorded": len(uniq),
        "epochs_declared": declared_n,
        "epoch_history_contiguous_from_zero": (
            bool(uniq) and uniq == list(range(len(uniq)))
        ),
        "resumed": bool(start is not None and start != 0),
        "history_completion_frac": (
            (len(uniq) / declared_n) if declared_n else None
        ),
        "nan_train_loss": bool(nan_eps),
        "nan_train_loss_epochs": nan_eps,
        "series_source": source,
        "notes": {
            "n_epochs_run": (
                "the number of DATA ROWS in the run's per-epoch file. It is "
                "NOT the number of epochs the model trained: a run that "
                "resumed at epoch 19 has one row, and a run that early-stopped "
                "has fewer rows than its declared budget."
            ),
            "epoch_history_contiguous_from_zero": (
                "True only when the recorded epoch indices are 0..n-1 with no "
                "gap and no offset. A resumed run starts above 0 and is "
                "therefore NOT contiguous, even though its rows are dense."
            ),
            "history_completion_frac": (
                "epochs_recorded / epochs_declared. A value below 1.0 means "
                "the run stopped early or was killed; it is NOT a measure of "
                "training quality."
            ),
        },
    }
    # Wf-A114E — the population cross-check, flat, on the same block.
    out.update(_events_integrity(decision or {}, ev))
    return out


def _head_levers(cfg: Dict, prov: Dict) -> Tuple[Dict, Optional[str], Dict]:
    """(values, source, per-key source) for the six head levers.

    Wf-A115 — ``config`` first, then ``provenance``.  A key in neither is
    null, and null means NOT RECORDED (it does NOT mean the argparse
    default).  MEASURED on the depth-1 corpus: 0 of 42 runs carry a
    ``provenance`` block (every one predates 2026-10-25), so ``provenance``
    is currently a FALLBACK, not the primary source.  ``velocity_head_init_knots``
    is in neither on that corpus, so it is null for every run there; the
    trainer's newer ``config`` block does carry it, and where it does the
    value is reported rather than nulled.
    """
    values: Dict[str, object] = {}
    per_key: Dict[str, Optional[str]] = {}
    for key in HEAD_LEVER_KEYS:
        if key in cfg and cfg.get(key) is not None:
            values[key] = cfg.get(key)
            per_key[key] = "config"
        elif key in prov and prov.get(key) is not None:
            values[key] = prov.get(key)
            per_key[key] = "provenance"
        else:
            values[key] = None
            per_key[key] = None
    if any(s == "config" for s in per_key.values()):
        source: Optional[str] = "config"
    elif any(s == "provenance" for s in per_key.values()):
        source = "provenance"
    else:
        source = None
    return values, source, per_key


def _failure_mode(
    gen_validity: Optional[float], e2e: Optional[float]
) -> str:
    """One label per run for how it failed, on the connectivity axis.

    Wf-A115 — the 0.5 boundary is the project's own
    ``DEFAULT_MAX_FRAGMENTED_RATIO``; no new number is invented.

      * ``unknown`` — the parse rate is not measured, so nothing can be said.
      * ``unparseable_dominated`` — fewer than half the samples even parsed.
      * ``parseable_fragmented`` — the samples parse, but fewer than half of
        ALL samples are one connected molecule.
      * ``ok`` — the parse rate is at least half AND at least half of all
        samples are one connected molecule.
    """
    if gen_validity is None:
        return "unknown"
    if gen_validity < _FRAGMENTATION_THRESHOLD:
        return "unparseable_dominated"
    if e2e is None:
        # The parse rate is known but the molecule-level rate is not; that is
        # not evidence of fragmentation, so it is not labelled as such.
        return "unknown"
    if e2e < _FRAGMENTATION_THRESHOLD:
        return "parseable_fragmented"
    return "ok"


def _rate_stats(values: List[float]) -> Dict[str, object]:
    """The five numbers every rate block carries."""
    if not values:
        return {
            "n": 0, "min": None, "max": None, "mean": None, "median": None,
            "n_at_or_above_0.50": 0,
        }
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "median": statistics.median(ordered),
        "n_at_or_above_0.50": sum(1 for v in ordered if v >= 0.50),
    }


def _is_clean(run: Dict) -> bool:
    """The clean-run predicate: contiguous from epoch 0 AND no nan loss.

    Wf-A114E — and the population conflict must be RESOLVED.  A run whose
    per-epoch file and whose events stream disagree about which epochs it
    trained, and which the census could not resolve, is REFUSED: it keeps its
    row and its reason, but it does not enter a headline population.  A
    RESOLVED conflict (the truncated-prefix class, re-derived from the
    events) is NOT excluded — that is the whole point of the re-derivation.
    """
    integ = run.get("integrity") or {}
    return bool(
        integ.get("epoch_history_contiguous_from_zero")
        and not integ.get("nan_train_loss")
        and not (
            integ.get("population_conflict")
            and not integ.get("population_resolved")
        )
    )


def _cohort_sort_key(label: str) -> Tuple[int, float]:
    try:
        return (0, float(label))
    except (TypeError, ValueError):
        return (1, 0.0)


def _by_n_train(runs: List[Dict]) -> Dict[str, object]:
    """Cohort-stratified end-to-end rates, stratified by ``n_train``.

    Wf-A115 — the cohort label is the run's OWN ``n_train`` value, verbatim.
    It is NOT binarised into "small" / "large": MEASURED the corpus holds
    n_train values of 16, 32, 64, 100, 128, 256, 400, 512, 2048, 4096 and
    26715, and a binary split would throw away exactly the axis in question.

    Wf-A114 verify:integration D1 — this label is a COHORT AXIS, and it is
    LOOSER than the harness's.  MEASURED: the ``n_train=26715`` label mixes
    two different validation protocols (``n_val=192`` in ~~8 runs~~ **9
    runs**, ``n_val=6714`` in 5 runs — 14 runs in all), so its median is a
    median over two cohorts.  The mixing is
    now VISIBLE in every block (``n_val_breakdown`` / ``n_val_mixed``) and
    the STRICT cohort — the ``(n_train, n_val)`` PAIR — is in the sibling
    ``by_cohort`` block, which is the definition
    ``molmetal/scripts/cfm_cohort_matched_compare.py`` enforces.
    """
    cohorts: Dict[str, List[Dict]] = {}
    for r in runs:
        cohorts.setdefault(str(r.get("n_train")), []).append(r)

    out: Dict[str, object] = {}
    for label in sorted(cohorts, key=lambda k: (k == "None", _cohort_sort_key(k))):
        members = cohorts[label]
        with_e2e = [
            r for r in members
            if r.get("gen_single_molecule_rate_end") is not None
        ]
        last_vals = [float(r["gen_single_molecule_rate_end"]) for r in with_e2e]
        best_vals = [
            float(r["gen_single_molecule_rate_best"])
            for r in members
            if r.get("gen_single_molecule_rate_best") is not None
        ]
        clean = [r for r in with_e2e if _is_clean(r)]
        excluded = [r for r in with_e2e if not _is_clean(r)]
        clean_last = [float(r["gen_single_molecule_rate_end"]) for r in clean]
        # Wf-A150 — the CLEAN population of the BEST-epoch reading is
        # "CLEAN AND carries a best-epoch rate".  It used to be
        # "CLEAN AND carries a LAST-epoch rate AND carries a best-epoch
        # rate", i.e. it intersected the two readings.  That intersection is
        # UNDECLARED: the block's own ``predicate`` string says nothing about
        # a last-epoch rate, and the sibling ``best_epoch`` block (and the
        # top-level ``fragmentation_summary``) are both computed over the
        # NON-intersected population.  The intersection became visible only
        # after Wf-A127 published the empty-denominator chemistry rate as
        # ``null``: two CLEAN runs lost their last-epoch rate while KEEPING
        # their best-epoch rate, and they silently fell out of this block.
        # The population is now the same one the block's predicate names.
        clean_best = [
            float(r["gen_single_molecule_rate_best"])
            for r in members
            if _is_clean(r)
            and r.get("gen_single_molecule_rate_best") is not None
        ]
        n_val_breakdown: Dict[str, int] = {}
        for r in members:
            n_val_breakdown[str(r.get("n_val"))] = (
                n_val_breakdown.get(str(r.get("n_val")), 0) + 1
            )
        out[label] = {
            "n_train": members[0].get("n_train"),
            "n_runs": len(members),
            "n_runs_with_e2e": len(with_e2e),
            "n_val_values": sorted(
                {r.get("n_val") for r in members},
                key=lambda v: (v is None, v),
            ),
            "n_val_breakdown": n_val_breakdown,
            "n_val_mixed": len(n_val_breakdown) > 1,
            "cohort_axis_note": (
                "this block is stratified by n_train ONLY.  When "
                "`n_val_mixed` is true the median below is a median over "
                "MORE THAN ONE cohort -- the strict (n_train, n_val) cohort "
                "is in the sibling `by_cohort` block, and that is the "
                "definition molmetal/scripts/cfm_cohort_matched_compare.py "
                "enforces."
            ),
            "last_epoch": _rate_stats(last_vals),
            "best_epoch": _rate_stats(best_vals),
            "clean_only": {
                "n_excluded": len(excluded),
                "excluded_runs": sorted(r["run"] for r in excluded),
                "last_epoch": _rate_stats(clean_last),
                "best_epoch": _rate_stats(clean_best),
                "predicate": (
                    "integrity.epoch_history_contiguous_from_zero AND NOT "
                    "integrity.nan_train_loss"
                ),
            },
        }
    return out


def _by_cohort(runs: List[Dict]) -> Dict[str, object]:
    """Cohort-stratified end-to-end rates, stratified by the STRICT cohort.

    Wf-A114 verify:integration D1 — a cohort is the PAIR
    ``(n_train, n_val)``, not ``n_train`` alone.  This block is the census's
    answer to the same question
    ``molmetal/scripts/cfm_cohort_matched_compare.py::axis_cohort`` asks, so
    the two artifacts agree on what a cohort is.

    The label is ``"<n_train>/<n_val>"``.  The stats are the census's own
    (``_rate_stats``), and the ``clean_only`` predicate is the census's own
    ``_is_clean``.
    """
    cohorts: Dict[str, List[Dict]] = {}
    for r in runs:
        label = f"{r.get('n_train')}/{r.get('n_val')}"
        cohorts.setdefault(label, []).append(r)

    out: Dict[str, object] = {}
    for label in sorted(cohorts, key=lambda k: _cohort_sort_key(k.split("/")[0])):
        members = cohorts[label]
        with_e2e = [
            r for r in members
            if r.get("gen_single_molecule_rate_end") is not None
        ]
        last_vals = [float(r["gen_single_molecule_rate_end"]) for r in with_e2e]
        best_vals = [
            float(r["gen_single_molecule_rate_best"])
            for r in members
            if r.get("gen_single_molecule_rate_best") is not None
        ]
        clean = [r for r in with_e2e if _is_clean(r)]
        excluded = [r for r in with_e2e if not _is_clean(r)]
        clean_last = [float(r["gen_single_molecule_rate_end"]) for r in clean]
        # Wf-A150 — same repair as ``_by_n_train``: the CLEAN population of
        # the BEST-epoch reading is "CLEAN AND carries a best-epoch rate",
        # NOT the intersection with the last-epoch population.  The block's
        # ``predicate`` names the former.
        clean_best = [
            float(r["gen_single_molecule_rate_best"])
            for r in members
            if _is_clean(r)
            and r.get("gen_single_molecule_rate_best") is not None
        ]
        n_train, n_val = label.split("/", 1)
        out[label] = {
            "n_train": members[0].get("n_train"),
            "n_val": members[0].get("n_val"),
            "n_runs": len(members),
            "n_runs_with_e2e": len(with_e2e),
            "runs": sorted(r["run"] for r in members),
            "last_epoch": _rate_stats(last_vals),
            "best_epoch": _rate_stats(best_vals),
            "clean_only": {
                "n_excluded": len(excluded),
                "excluded_runs": sorted(r["run"] for r in excluded),
                "last_epoch": _rate_stats(clean_last),
                "best_epoch": _rate_stats(clean_best),
                "predicate": (
                    "integrity.epoch_history_contiguous_from_zero AND NOT "
                    "integrity.nan_train_loss"
                ),
            },
        }
    return out


def _budget_matched(runs: List[Dict]) -> Dict[str, object]:
    """Compare the two cohorts at the SAME epoch index.

    Wf-A115 — a run contributes to budget K only if K-1 is IN ITS OWN epoch
    sequence.  A run that stopped at epoch 4 contributes nothing at K=9; it
    does not contribute its epoch-3 value either.  That is the whole point:
    comparing cohorts at the epoch each cohort HAPPENED to reach compares the
    budget, not the data.

    MEASURED (clean runs, i.e. contiguous from epoch 0 with no nan loss):
    ~~the ordering between n_train=256 and n_train=26715 flips at nearly every
    budget, because the two cohorts are OUT OF PHASE in the same oscillation
    (five of the eight clean 256-cohort runs read 0.0 at epoch 8 and only
    take off around epoch 10-14, while the full-corpus runs are already
    productive by epoch 8).  The full-corpus cohort's surviving sample
    collapses from 8 runs to 1 past epoch 18, because its runs stopped early.~~
    **CORRECTION (2026-10-31, Wf-A153) — the struck paragraph is REFUTED in
    its counts; MEASURED over the 59 runs on disk
    (``metrics/by_metric/cfm_run_census.json`` -> ``runs[]`` and
    ``budget_matched``, plus the per-epoch rows under
    ``molmetal/reports/``):** the ordering of the two COHORT MEDIANS flips at
    **3** of the **10** budget transitions, not at nearly every budget — the
    census's own published ``budget_matched.flipped_budgets`` is
    ``[9, 11, 25]`` — and the n_train=256 median is the HIGHER of the two at
    **8 of the 11** budgets.  The population is the **8** clean n_train=256
    runs that carry an end-to-end rate (of **14** clean n_train=256 runs), of
    which **4** read exactly 0.0 at epoch 8 (cfm_fm_ab_2026_10_09,
    cfm_fm_control_2026_10_09, cfm_frag_control, cfm_pcgrad_off_2026_10_15),
    cfm_pcgrad_on_2026_10_14 reads 0.0625 and **3** carry NO epoch-8 rate at
    all (cfm_blk_c5_repair_2026_10_09, cfm_bond_head_30ep_onecycle_2026_10_09,
    cfm_frag_assemble — each with ``gen_validity_n > 0`` and
    ``gen_validity_val == 0``, so the rate is UNDEFINED and published as null
    since Wf-A127).  Of those 4, 3 first read >= 0.10 at epoch 9
    (cfm_fm_control) or epoch 10 (cfm_fm_ab, cfm_pcgrad_off) and 1
    (cfm_frag_control) at epoch 24.  Of the **11** clean n_train=26715 runs,
    8 read a non-zero end-to-end rate at epoch 8, 1 reads exactly 0.0
    (cfm_tmqm_all6_retrain) and 2 have no epoch-8 row at all.  The
    full-corpus cohort's surviving sample collapses from **10** runs at K=5
    (9 at K=9, 7 at K=11, 4 at K=12, 3 at K=13, 2 at K=15) to **1** run from
    K=18 onward, because its runs stopped early.
    """
    out_cohorts: Dict[str, Dict[str, object]] = {}
    for n in BUDGET_MATCHED_COHORTS:
        budgets: Dict[str, object] = {}
        for K in BUDGET_MATCHED_BUDGETS:
            ep = K - 1
            reached = [
                r for r in runs
                if r.get("n_train") == n
                and _is_clean(r)
                and ep in (r.get("_epoch_indices") or [])
            ]
            vals = [
                float(v)
                for v in (
                    (r.get("_e2e_by_epoch") or {}).get(ep) for r in reached
                )
                if v is not None
            ]
            budgets[str(K)] = {
                "epoch_index": ep,
                "n_values": len(vals),
                "n_reached": len(reached),
                "median": statistics.median(vals) if vals else None,
                "mean": (sum(vals) / len(vals)) if vals else None,
                "max": max(vals) if vals else None,
                "min": min(vals) if vals else None,
            }
        out_cohorts[str(n)] = budgets

    # Which budget flips the ordering?  Recorded explicitly, because "the
    # ordering flips" is the claim and the budgets are the evidence.
    flips: List[Dict[str, object]] = []
    prev_leader: Optional[str] = None
    for K in BUDGET_MATCHED_BUDGETS:
        a = out_cohorts[str(BUDGET_MATCHED_COHORTS[0])][str(K)]
        b = out_cohorts[str(BUDGET_MATCHED_COHORTS[1])][str(K)]
        ma, mb = a["median"], b["median"]
        if ma is None or mb is None:
            leader = None
        elif ma > mb:
            leader = str(BUDGET_MATCHED_COHORTS[0])
        elif mb > ma:
            leader = str(BUDGET_MATCHED_COHORTS[1])
        else:
            leader = "tie"
        flips.append({
            "K": K,
            "epoch_index": K - 1,
            "median_n_train_256": ma,
            "median_n_train_26715": mb,
            "leader": leader,
            "previous_leader": prev_leader,
            "flipped": bool(prev_leader is not None and leader != prev_leader),
            "n_values_256": a["n_values"],
            "n_reached_256": a["n_reached"],
            "n_values_26715": b["n_values"],
            "n_reached_26715": b["n_reached"],
        })
        prev_leader = leader

    survived = {
        str(n): {
            str(K): out_cohorts[str(n)][str(K)]["n_values"]
            for K in BUDGET_MATCHED_BUDGETS
        }
        for n in BUDGET_MATCHED_COHORTS
    }

    _comp: Dict[str, Dict[str, int]] = {}
    for n in BUDGET_MATCHED_COHORTS:
        _c: Dict[str, int] = {}
        for r in runs:
            if r.get("n_train") != n:
                continue
            _c[str(r.get("n_val"))] = _c.get(str(r.get("n_val")), 0) + 1
        _comp[str(n)] = _c

    return {
        "budgets": list(BUDGET_MATCHED_BUDGETS),
        "reading": "end-to-end rate at epoch K-1 (final-epoch reading)",
        "rule": (
            "a run contributes to budget K only if the epoch index K-1 is IN "
            "that run's own epoch sequence. A run that stopped before it does "
            "not contribute at all."
        ),
        "cohort_scope": (
            "clean runs only (contiguous from epoch 0 AND no nan train_loss), "
            "across every run the loader found"
        ),
        "cohort_axis": (
            "this block is stratified on n_train ALONE.  MEASURED: the "
            "n_train=26715 cohort holds two validation protocols (see "
            "`cohort_composition`), so its median is a median over two "
            "cohorts.  The strict (n_train, n_val) cohorts are in the "
            "`by_cohort` block."
        ),
        "cohort_composition": _comp,
        "cohorts": out_cohorts,
        "ordering_flips": flips,
        "flipped_budgets": [f["K"] for f in flips if f["flipped"]],
        "n_train_axis": "UNRESOLVED",
        "n_train_axis_note": (
            "No cohort median is a valid bar for the n_train axis. MEASURED: "
            "the best-epoch reading is a max over a noisy series (the census's "
            "own note: 'optimistically biased'), and the final-epoch reading "
            "is dominated by WHERE EACH COHORT STOPPED. Under the "
            "budget-matched comparison above the ordering between the two "
            "cohorts flips at the budgets listed in `flipped_budgets`, "
            "because the cohorts are out of phase in the same oscillation. "
            "The full-corpus cohort's surviving sample collapses from "
            f"{survived.get('26715', {}).get('9')} runs at K=9 to "
            f"{survived.get('26715', {}).get('28')} run at K=28, because its "
            "runs stopped early — a cohort median over 1 surviving run is a "
            "median over one run. Use `per_run_bar` instead."
        ),
        "n_values_by_budget": survived,
    }


def _per_run_bar(runs: List[Dict]) -> Dict[str, object]:
    """The bar as a NAMED run, not a cohort median.

    Wf-A115 — the rule is: same reading, same cohort, same budget, against a
    NAMED run.  The values below are read from those runs' own records; the
    run NAMES are the declaration, the numbers are measured.
    """
    by_name = {r["run"]: r for r in runs}

    def _pick(run: str, reading: str) -> Optional[Dict]:
        r = by_name.get(run)
        if r is None:
            return None
        if reading == "final_epoch":
            value = r.get("gen_single_molecule_rate_end")
            epoch = (r.get("_last_row") or {}).get("epoch")
        else:
            value = r.get("gen_single_molecule_rate_best")
            epoch = r.get("gen_single_molecule_rate_best_epoch")
        return {
            "run": run,
            "n_train": r.get("n_train"),
            "n_val": r.get("n_val"),
            "cohort": f"n_train={r.get('n_train')}",
            "cohort_strict": f"{r.get('n_train')}/{r.get('n_val')}",
            "cohort_strict_note": (
                "the STRICT cohort is the (n_train, n_val) PAIR -- the "
                "definition molmetal/scripts/cfm_cohort_matched_compare.py::"
                "axis_cohort enforces.  The `cohort` field above is the older "
                "n_train-only label and is kept for continuity."
            ),
            "reading": reading,
            "value": value,
            "epoch": epoch,
            "integrity": r.get("integrity"),
        }

    primary = _pick("cfm_r9_fix_v2", "final_epoch")
    secondary = _pick("cfm_fullcorpus_seed1", "best_epoch")
    stretch_runs = [
        s for s in (
            _pick("cfm_fm_ab_2026_10_09", "best_epoch"),
            _pick("cfm_pcgrad_off_2026_10_15", "best_epoch"),
        ) if s is not None
    ]

    stretch: Dict[str, object] = {
        "runs": stretch_runs,
        "pass_fail": False,
        "report_only": True,
        "caveat": (
            "a DIFFERENT cohort (n_train=256) AND a different budget from the "
            "primary bar. It is reported, never used as a pass/fail threshold."
        ),
    }
    if stretch_runs:
        first = stretch_runs[0]
        stretch.update({
            "run": first["run"],
            "value": first["value"],
            "cohort": first["cohort"],
            "cohort_strict": first["cohort_strict"],
            "epoch": first["epoch"],
            "n_train": first["n_train"],
            "n_val": first["n_val"],
        })

    out: Dict[str, object] = {
        "rule": (
            "bar = PER-RUN + same reading + same cohort + same budget vs a "
            "NAMED run. A cohort median is NOT a bar: MEASURED, it measures "
            "the budget and the reading, not the thing being compared."
        ),
    }
    if primary is not None:
        out["primary"] = primary
    if secondary is not None:
        out["secondary"] = secondary
    out["stretch"] = stretch
    out["caveats"] = [
        "the primary bar is a FINAL-EPOCH reading; the secondary is a "
        "BEST-EPOCH reading (a max over a noisy series) and is therefore not "
        "interchangeable with it",
        "the stretch number is from a different cohort and a different "
        "budget, and is report-only",
    ]
    return out


def _run_date_key(run: Dict) -> Tuple[str, str]:
    """Sort key inside a cohort: the run's own date suffix, else its mtime.

    Wf-A115 — a run name carries its date (``..._2026_10_09``); the runs that
    do not are ordered by their directory's mtime, formatted into the same
    shape so the two are comparable.
    """
    name = run["run"]
    leaf = name.rsplit("/", 1)[-1]
    m = _DATE_SUFFIX.search(leaf)
    if m:
        return (m.group(0), name)
    try:
        from datetime import datetime

        return (datetime.fromtimestamp(run.get("_mtime") or 0).strftime("%Y_%m_%d"), name)
    except Exception:
        return ("0000_00_00", name)


def _paired_headline(runs: List[Dict]) -> Tuple[Dict[str, Dict], List[str]]:
    """Detect a run whose HEADLINE improved while its GUARDED number did not.

    Wf-A115 — the headline is ``gen_validity_max`` (the number the line
    actually gates on) and the guard is the honest end-to-end rate at the
    run's BEST epoch.  A run is flagged when it beat every EARLIER run in
    its cohort on the headline while failing to beat them on the guard —
    i.e. the number that is quoted went up and the number that means
    something did not.
    """
    cohorts: Dict[object, List[Dict]] = {}
    for r in runs:
        cohorts.setdefault(r.get("n_train"), []).append(r)

    per_run: Dict[str, Dict] = {}
    flagged_names: List[str] = []
    for n_train, members in cohorts.items():
        ordered = sorted(members, key=_run_date_key)
        priors: List[Dict] = []
        for r in ordered:
            headline = r.get("gen_validity_max")
            guarded = r.get("gen_single_molecule_rate_best")
            prior_headlines = [
                p.get("gen_validity_max") for p in priors
                if p.get("gen_validity_max") is not None
            ]
            prior_guardeds = [
                p.get("gen_single_molecule_rate_best") for p in priors
                if p.get("gen_single_molecule_rate_best") is not None
            ]
            prior_headline_max = max(prior_headlines) if prior_headlines else None
            prior_guarded_max = max(prior_guardeds) if prior_guardeds else None
            flagged = False
            if priors and headline is not None and guarded is not None:
                if prior_headline_max is not None and prior_guarded_max is not None:
                    flagged = (
                        headline > prior_headline_max
                        and guarded <= prior_guarded_max
                    )
            per_run[r["run"]] = {
                "flagged": flagged,
                "headline_gen_validity_max": headline,
                "prior_headline_max": prior_headline_max,
                "delta_headline": (
                    headline - prior_headline_max
                    if (headline is not None and prior_headline_max is not None)
                    else None
                ),
                "guarded_e2e_best": guarded,
                "prior_guarded_max": prior_guarded_max,
                "delta_guarded": (
                    guarded - prior_guarded_max
                    if (guarded is not None and prior_guarded_max is not None)
                    else None
                ),
                "cohort_n_train": n_train,
                "prior_runs": sorted(p["run"] for p in priors),
            }
            if flagged:
                flagged_names.append(r["run"])
            priors.append(r)
    return per_run, sorted(flagged_names)


#: Wf-A115 -- the early-stop criterion -> the per-epoch column(s) that carry
#: it.  A run that SELECTED a criterion and never emitted the column could
#: not have measured it, so the selection was a promise the artifact cannot
#: keep.  ``combined`` is a composite and needs both halves.
EARLY_STOP_COLUMN: Dict[str, List[str]] = {
    "decode_ratio": ["decode_ratio"],
    "gen_validity": ["gen_validity"],
    "molecule_validity": ["gen_molecule_validity"],
    "bond_exact_type": ["bond_exact_type_rate"],
    "bond_type_f1": ["bond_type_macro_f1_pooled"],
    "bond_type_f1_eps": ["bond_type_macro_f1_pooled"],
    "combined": ["gen_validity", "bond_type_macro_f1_pooled"],
}


def _early_stop_guard(
    rows: List[Dict[str, object]], cfg: Dict, jd: Dict
) -> Dict[str, object]:
    """Did this run gate on the criterion it said it would gate on?

    Wf-A115 -- there was NO detector for this, and that is a hole, not a
    nicety.  ``configs/c9_cfm_transport_fixed.yaml`` selects
    ``early_stop_metric: molecule_validity``; the trainer DEGRADES a
    criterion it cannot measure (it falls back down a ladder and says so in
    ``early_stop_criterion_note``), so a c9 run can silently gate on
    something else -- or on nothing that moves -- while its config still
    claims ``molecule_validity``.

    Three states, all derived from the run's OWN artifact:

      * ``ok`` -- the run requested a criterion, emitted the column(s) that
        criterion is measured from, and gated on the requested criterion.
      * ``selected_but_not_emitted`` -- the run REQUESTED a criterion whose
        column it never emitted.  The criterion was never measurable from
        this artifact.
      * ``degraded`` -- the run requested one criterion and every epoch
        gated on a DIFFERENT one.  This is the silent case: the config says
        X, the run stopped on Y.
      * ``not_recorded`` -- the artifact does not say what it selected at
        all.  Blind, which is not the same as fine.
    """
    ctrl = jd.get("early_stop_control") or {}
    if not isinstance(ctrl, dict):
        ctrl = {}
    requested = ctrl.get("requested_metric") or cfg.get("early_stop_metric")
    requested = str(requested).strip().lower() if requested else None
    resolved = sorted(
        {
            str(r.get("early_stop_metric")).strip().lower()
            for r in rows
            if r.get("early_stop_metric")
        }
    )
    row_keys: set = set()
    for r in rows:
        row_keys |= set(r.keys())
    required = EARLY_STOP_COLUMN.get(requested) if requested else None
    emitted = [c for c in (required or []) if c in row_keys]
    missing = [c for c in (required or []) if c not in row_keys]
    degraded_to = (
        sorted(set(resolved) - {requested}) if (requested and resolved) else []
    )

    if requested is None:
        state = "not_recorded"
    elif missing:
        state = "selected_but_not_emitted"
    elif degraded_to:
        state = "degraded"
    else:
        state = "ok"

    out: Dict[str, object] = {
        "state": state,
        "requested_metric": requested,
        "required_columns": required,
        "emitted_columns": emitted,
        "missing_columns": missing,
        "resolved_metrics": resolved,
        "degraded_to": degraded_to,
    }
    if state == "selected_but_not_emitted":
        out["note"] = (
            "the run selected a criterion whose column(s) it never emitted, "
            "so the criterion was never measurable from this artifact"
        )
    elif state == "degraded":
        out["note"] = (
            "the run selected one criterion and gated on another; the "
            "selection in the config is not the criterion that stopped the "
            "run"
        )
    elif state == "not_recorded":
        out["note"] = (
            "the artifact records neither the requested criterion nor the "
            "resolved one; a reader cannot tell what stopped this run"
        )
    return out


def load_run(d: Path) -> Optional[Dict]:
    """Read one run directory into one census row."""
    csv_path = d / "per_epoch_metrics.csv"
    if not csv_path.is_file():
        return None
    rows, source, jd = _load_series(d)
    # Wf-A115 — the guard is on DATA ROWS, not on file lines.  A run with
    # exactly ONE recorded epoch (cfm_r11_eval256, cfm_r17_eval512) is
    # precisely the run whose integrity the census has to expose, so it must
    # not be dropped for being short.  Only a file with NO data row at all is
    # not a run.
    if not rows:
        return None
    cfg = jd.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    prov = jd.get("provenance") or {}
    if not isinstance(prov, dict):
        prov = {}

    # ------------------------------------------------------------------
    # Wf-A114E — L1 (cross-check) + L2 (resolution).  The per-epoch file is
    # written ONCE to a FIXED path, so a directory can hold the record of a
    # DIFFERENT, SHORTER run than the one it is named for.  events.jsonl is
    # append-only and holds every population that ever wrote here, so it is
    # the cross-check the census needs.
    # ------------------------------------------------------------------
    ev = _read_events(d)
    decision = _population_decision(rows, ev)
    resolved = bool(decision.get("resolved"))
    if resolved:
        # CLASS A — the file is a truncated prefix.  Score the events rows of
        # the matching population instead, and drop every block that came
        # from the CLOBBERED file (its config, its provenance and its
        # early-stop control belong to the other population).
        rows = list(decision.get("rows") or [])
        source = "events"
        cfg = dict(decision.get("config") or {})
        prov = {}
        jd = {}
    else:
        decision["source"] = (
            "per_epoch_metrics.json" if source == "json"
            else "per_epoch_metrics.csv"
        )

    last = rows[-1]
    gvs = [r["gen_validity"] for r in rows if r.get("gen_validity") is not None]
    # The honest END-TO-END rate, paired ROW BY ROW.  ``gen_validity`` is the
    # RDKit parse rate over the same samples, so their product is the share of
    # ALL samples that are one connected molecule — the number that answers
    # "did the model produce a molecule".  Pairing per row (not per column)
    # matters: a run whose last epoch lacks a validity would otherwise be
    # scored with two different epochs' numbers.
    #
    # Wf-A115 — the LAST row's product, and null when that row's generation
    # eval did not run.  It used to be "the last row that HAD both", which
    # silently reported an earlier epoch's number under an "end" name.
    paired_epochs: List[Tuple[int, float]] = []
    for r in rows:
        c = r.get("gen_single_fragment_ratio")
        v = r.get("gen_validity")
        if c is not None and v is not None and r.get("epoch") is not None:
            paired_epochs.append((int(r["epoch"]), float(c) * float(v)))
    e2e_rate = (
        float(last["gen_single_fragment_ratio"]) * float(last["gen_validity"])
        if (
            last.get("gen_single_fragment_ratio") is not None
            and last.get("gen_validity") is not None
        )
        else None
    )
    best_pair = max(paired_epochs, key=lambda x: x[1]) if paired_epochs else None

    # Wf-A115 — the last row's gen_* rate columns are null when that row's
    # generation eval did not run (``gen_validity_n == 0``); the trainer
    # wrote 0.0 there, which is indistinguishable from a measured 0.
    last_measured = bool(last.get("gen_eval_ran"))
    # Wf-A115 -- the DECODE path's own predicate.  It is NOT `last_measured`:
    # the decode runs on the val-split geometry and has its own denominator
    # (`decode_ratio_n`), so a run whose GENERATION eval drew nothing can
    # still carry a fully measured `decode_ratio`.
    decode_measured = bool(last.get("decode_measured"))

    # The last row that DID carry a measurement (for the "last measured"
    # variant of the fragmentation verdict).
    #
    # Wf-A127 — "a measurement" for the FRAGMENTATION verdict means the
    # connectivity rate itself is defined, i.e. the eval SCORED rows.  A
    # row whose scored denominator is empty carries a 0.0 placeholder, not
    # a measurement, so it cannot be the row whose verdict is preserved.
    last_measured_row = None
    for r in rows:
        if (
            r.get("gen_eval_ran")
            and r.get("gen_validity") is not None
            and r.get("gen_single_fragment_ratio") is not None
        ):
            last_measured_row = r

    integrity = _integrity(rows, cfg, source, decision, ev)
    head_levers, head_source, head_per_key = _head_levers(cfg, prov)
    es_guard = _early_stop_guard(rows, cfg, jd)
    # Wf-A114E — which file the run's CONFIG block came from, and whether
    # that block is complete.  A class-A run's config was re-derived from the
    # events stream, which carries no n_train / n_val / batch_size, so the
    # block is PARTIAL and it is flagged as a conflict rather than silently
    # completed from the clobbered file.
    if resolved:
        config_source = "events.jsonl"
        config_complete = False
        config_conflict = True
    else:
        config_source = decision.get("source") or "per_epoch_metrics.json"
        config_complete = bool(cfg)
        config_conflict = False

    # Wf-A115 — the COHORT of the RUN OF RECORD.
    #
    # The cohort is the PAIR ``(n_train, n_val)``.  It used to be read out of
    # the per-epoch file's own config block — but that file is written ONCE,
    # to a FIXED path, so a later run launched into the same directory
    # OVERWRITES it and the survivor is whichever finished LAST.  For a
    # class-A run the config block was re-derived from the events stream,
    # which carries no ``n_train`` / ``n_val`` at all, so the run came out
    # with ``n_train = None`` and dropped out of EVERY cohort axis — including
    # the one it belongs to.
    #
    # The cohort is recoverable, but only from an artifact that can be tied to
    # the RUN OF RECORD.  ``molmetal/scripts/cfm_cohort_recovery.py`` owns that
    # rule: a run log attests the run of record iff it echoes EVERY ``ts`` of
    # that population.  A sibling population's log is NOT evidence about the
    # run of record — MEASURED: ``cfm_fullcorpus_2026_10_10`` holds a
    # 26715/192 run beside a 64/32 CPU smoke in ONE directory.
    #
    # So the two cases are kept apart:
    #   * the per-epoch file IS the run of record  -> its own config block
    #     states the cohort (``per_epoch_metrics.json:config``);
    #   * the per-epoch file is NOT the run of record -> the cohort comes from
    #     a ts-matched log, or it is reported as UNATTESTED and the run stays
    #     out of the cohort axes rather than being given a neighbour's cohort.
    file_is_run_of_record = not resolved and (
        decision.get("verdict") != "unmatched_conflict"
    )
    cohort = recover_cohort(
        d,
        list(decision.get("run_of_record_ts") or []),
        config_cohort=(
            (cfg.get("n_train"), cfg.get("n_val"))
            if file_is_run_of_record
            else None
        ),
    )
    cohort_source = str(cohort.get("source") or "")
    cohort_reason = str(cohort.get("reason") or "")
    cohort_evidence = cohort.get("evidence")
    n_train = cohort.get("n_train")
    n_val = cohort.get("n_val")

    out: Dict[str, object] = {
        # Wf-A115 — the run name is the path RELATIVE to molmetal/reports, so
        # two runs with the same leaf name in different wave directories stay
        # distinguishable.  A directory OUTSIDE the reports tree (a synthetic
        # run in a test's tmp_path) keeps its absolute path — it has no
        # relative name to be had.
        "run": _run_name(d),
        "run_path": str(d),
        # Wf-A115 — `n_epochs_run` is a ROW COUNT, kept under its old name
        # so no consumer breaks, and documented as such in
        # `integrity.notes`.  `integrity.epochs_recorded` is the count of
        # DISTINCT epoch indices.
        "n_epochs_run": len(rows),
        "n_train": n_train,
        # Wf-A114 verify:integration D1 -- `n_val` is promoted to a
        # first-class field for the same reason `n_train` is: a cohort is the
        # PAIR (n_train, n_val), and the `n_train=26715` label alone mixes
        # `n_val=192` with `n_val=6714` (two different validation protocols
        # under one label).
        "n_val": n_val,
        "config": {k: cfg.get(k) for k in CFG_KEYS if k in cfg},
        # Wf-A114E — the provenance of that config block.
        "config_source": config_source,
        "config_complete": config_complete,
        "config_conflict": config_conflict,
        # Wf-A115 — the provenance of that COHORT, and what it is based on.
        # `cohort_source == "unattested"` means NO artifact on disk ties a
        # cohort to this run's run of record; the run keeps its row and its
        # reason and stays out of the cohort axes.
        "cohort_source": cohort_source,
        "cohort_reason": cohort_reason,
        "cohort_evidence": cohort_evidence,
        "integrity": integrity,
        # Wf-A115 -- "did this run gate on what it said it would gate on?".
        # See `_early_stop_guard` for why this is load-bearing rather than
        # decorative: the trainer DEGRADES an unmeasurable criterion, so a
        # config that selects `molecule_validity` is not evidence that the
        # run did.
        "early_stop_guard": es_guard,
        "head_levers": head_levers,
        "head_levers_source": head_source,
        "head_levers_source_per_key": head_per_key,
        "head_levers_recorded": any(v is not None for v in head_levers.values()),
        "head_levers_note": (
            "null = NOT RECORDED, not 'radial' / 'zero' / 'learnable'. "
            "A lever the run never wrote down is not back-filled with the "
            "argparse default, because a default is not a measurement."
        ),
        "bond_f1": last.get("bond_f1"),
        "bond_type_macro_f1": last.get("bond_type_macro_f1_pooled"),
        "bond_exact_type_rate": last.get("bond_exact_type_rate"),
        "bond_pr_auc": last.get("bond_pr_auc"),
        # Wf-A99 (2026-10-20) — runs written before Wf-A97 carry the
        # NO_BOND AP here, runs after it carry the existence AP, and
        # nothing in the old rows said which.  Stamp per row so the
        # census cannot present the two as one column.
        "bond_pr_auc_definition": bond_ap_definition(last),
        # Wf-A115 — null when the DECODE path did not run on the last row.
        # `decode_ratio_measured` keeps the raw source value: decode_ratio
        # is a BOND-decode number, measured on the val split, and the
        # generation-eval predicate does not apply to it.  Both are kept so
        # the null is self-explaining and nothing measured is lost.
        # MEASURED: gating this on the generation predicate nulled the real
        # measurement for 27 of 59 runs.
        "decode_ratio": last.get("decode_ratio") if decode_measured else None,
        "decode_ratio_measured": last.get("decode_ratio_measured"),
        "decode_ratio_n": last.get("decode_ratio_n"),
        "decode_eval_ran_on_last_epoch": decode_measured,
        "gen_eval_ran_on_last_epoch": last_measured,
        "gen_validity_max": max(gvs) if gvs else None,
        "gen_validity_end": gvs[-1] if gvs else None,
        "gen_validity_last_row": last.get("gen_validity"),
        # --- honest connectivity (higher single_fragment_ratio is BETTER) ---
        "gen_single_fragment_ratio_end": (
            last.get("gen_single_fragment_ratio") if last_measured else None
        ),
        "gen_single_fragment_ratio_max": max(
            [
                r["gen_single_fragment_ratio"] for r in rows
                if r.get("gen_single_fragment_ratio") is not None
            ]
        ) if any(r.get("gen_single_fragment_ratio") is not None for r in rows) else None,
        # The end-to-end rate: ONE connected molecule per SAMPLE.  This is
        # the number to quote; the conditional column above overstates the
        # generator whenever some samples fail to parse.
        "gen_single_molecule_rate_end": e2e_rate,
        # The run's BEST epoch by the same end-to-end rate.  A max over ~30
        # noisy epochs is optimistically biased, so it is NOT interchangeable
        # with the last-epoch figure — it is the number a run that early-stops
        # at its best epoch would deliver, and the honest bar for a re-run.
        "gen_single_molecule_rate_best": best_pair[1] if best_pair else None,
        "gen_single_molecule_rate_best_epoch": best_pair[0] if best_pair else None,
        "gen_largest_fragment_frac_end": (
            last.get("gen_largest_fragment_frac") if last_measured else None
        ),
        "gen_molecule_validity_end": (
            last.get("gen_molecule_validity") if last_measured else None
        ),
        "decode_single_fragment_ratio_end": (
            last.get("decode_single_fragment_ratio") if decode_measured else None
        ),
        "decode_single_fragment_ratio_end_measured": last.get(
            "decode_single_fragment_ratio_raw"
        ),
    }
    # Wf-A114 — the verdict at the run's LAST epoch.  Wf-A115 — the last
    # epoch's generation eval may not have run, in which case there is no
    # number to judge and the verdict is a SKIP with a reason.
    #
    # Wf-A125 G8 — a None ratio has TWO causes and the verdict must say
    # which.  ``field_present`` is "the column WAS recorded for this run",
    # which is what separates "the column was never written" from "the
    # column was written but the generation eval drew nothing".
    #   * the eval RAN at the last epoch -> the column is present iff that
    #     epoch's row carries it;
    #   * the eval DID NOT run -> the trainer may have written nothing at
    #     all for that epoch, so the RUN-level presence decides.
    _last_raw_present = last.get("gen_single_fragment_ratio_raw") is not None
    _run_raw_present = any(
        r.get("gen_single_fragment_ratio_raw") is not None for r in rows
    )
    _field_present = (
        _last_raw_present if last_measured else (_last_raw_present or _run_raw_present)
    )
    out["fragmentation_gate"] = _fragmentation_verdict(
        out["gen_single_fragment_ratio_end"],
        out["gen_largest_fragment_frac_end"],
        out["gen_validity_end"] if last_measured else None,
        field_present=_field_present,
        gen_validity_n=last.get("gen_validity_n"),
        gen_validity_val=last.get("gen_validity_val"),
    )
    out["fragmentation_gate"]["epoch"] = last.get("epoch")
    if not last_measured:
        out["fragmentation_gate"]["reasons"] = list(
            out["fragmentation_gate"].get("reasons") or []
        ) + [
            "the LAST epoch's generation eval did not run "
            "(gen_validity_n == 0), so there is no rate to judge"
        ]
    elif out["fragmentation_gate"]["status"] == "SKIP":
        # Wf-A127 — the OTHER empty-denominator sub-case: the eval DID draw
        # samples but scored no rows (`gen_validity_val == 0`), so the
        # chemistry rates are 0/0 and there is still no rate to judge.
        out["fragmentation_gate"]["reasons"] = list(
            out["fragmentation_gate"].get("reasons") or []
        ) + [
            "the LAST epoch's generation eval drew samples but scored no "
            "rows (gen_validity_val == 0), so the chemistry rates have an "
            "empty denominator and there is no rate to judge"
        ]
    # Wf-A115 — the same verdict at the last epoch that DID carry a
    # measurement, so nothing measured is lost when the run's last epoch is
    # unmeasured.
    #
    # Wf-A127 — the trigger is now "the last epoch's verdict is a SKIP",
    # not "the last epoch's eval did not run": the second empty-denominator
    # sub-case (scored no rows) also has no rate to judge, and its last
    # MEASURED verdict must survive alongside it.
    if (
        last_measured_row is not None
        and out["fragmentation_gate"]["status"] == "SKIP"
    ):
        lm = last_measured_row
        out["fragmentation_gate_last_measured"] = _fragmentation_verdict(
            lm.get("gen_single_fragment_ratio"),
            lm.get("gen_largest_fragment_frac"),
            lm.get("gen_validity"),
            field_present=(
                lm.get("gen_single_fragment_ratio_raw") is not None
            ),
            gen_validity_n=lm.get("gen_validity_n"),
            gen_validity_val=lm.get("gen_validity_val"),
        )
        out["fragmentation_gate_last_measured"]["epoch"] = lm.get("epoch")

    # Wf-A115 — the failure-mode label, on the last epoch and on the best
    # epoch (the last epoch alone would label a good run "unknown").
    out["failure_mode"] = _failure_mode(
        out["gen_validity_end"], out["gen_single_molecule_rate_end"]
    )
    best_gv = None
    if best_pair is not None:
        for r in rows:
            if r.get("epoch") == best_pair[0]:
                best_gv = r.get("gen_validity")
                break
    out["failure_mode_best"] = _failure_mode(
        best_gv, best_pair[1] if best_pair else None
    )

    if "n_train" in cfg and cfg.get("batch_size"):
        out["optimizer_steps_per_epoch"] = -(-cfg["n_train"] // cfg["batch_size"])
    if resolved:
        # Wf-A114E — the clobbered file's `gate_metric` / `gate_value` /
        # `best` blocks belong to the OTHER population.  The run of record's
        # own `done` event carries the same three facts, measured by the
        # run's OWN gate metric (which is NOT the census's end-to-end rate).
        _done = decision.get("done") or {}
        out["gate_metric"] = _done.get("gate_metric")
        out["gate_value"] = _done.get("gate_value")
        out["best_epoch"] = _done.get("best_epoch")
    else:
        out["gate_metric"] = jd.get("gate_metric")
        out["gate_value"] = jd.get("gate_value")
        if isinstance(jd.get("best"), dict):
            out["best_epoch"] = jd["best"].get("epoch")
    # Wf-A115 — internal helpers, dropped before serialisation.
    out["_epoch_indices"] = [r["epoch"] for r in rows if r.get("epoch") is not None]
    out["_e2e_by_epoch"] = {int(k): v for k, v in paired_epochs}
    out["_last_row"] = {"epoch": last.get("epoch")}
    try:
        out["_mtime"] = d.stat().st_mtime
    except OSError:
        out["_mtime"] = 0.0
    return out


def _public_run(run: Dict) -> Dict:
    """Strip the internal helpers (``_``-prefixed) before serialisation."""
    return {k: v for k, v in run.items() if not k.startswith("_")}


def _run_name(d: Path) -> str:
    """The run's name: its path RELATIVE to the reports tree, POSIX-style.

    Wf-A114E — a directory outside the reports tree (a synthetic run under a
    test's ``tmp_path``) has no relative name, so it keeps its absolute path.
    Every on-disk run lives under the reports tree, so nothing on disk is
    affected by the fallback.
    """
    try:
        return d.relative_to(REPORTS).as_posix()
    except ValueError:
        return d.as_posix()


def _population_summary(runs: List[Dict]) -> Dict[str, object]:
    """The census-wide view of the population cross-check (Wf-A114E).

    One block that says how many runs carry an ``events.jsonl``, how the
    verdicts distribute, and — for every conflicted run — what the census
    did about it: re-derived the series from the matching population, or
    refused the run.
    """
    verdicts: Dict[str, int] = {}
    conflicted: List[Dict[str, object]] = []
    uncorroborated: List[Dict[str, object]] = []
    for r in runs:
        integ = r.get("integrity") or {}
        v = str(integ.get("population_verdict") or "")
        verdicts[v] = verdicts.get(v, 0) + 1
        if integ.get("population_uncorroborated"):
            uncorroborated.append({
                "run": r["run"],
                "verdict": v,
                "series_source": integ.get("series_source"),
                "epochs_recorded": integ.get("epochs_recorded"),
                "disagreed_epochs": integ.get("population_disagreed_epochs"),
                "reason": integ.get("population_reason"),
            })
        if integ.get("population_conflict"):
            conflicted.append({
                "run": r["run"],
                "verdict": v,
                "resolved": bool(integ.get("population_resolved")),
                "series_source": integ.get("series_source"),
                "population_source": integ.get("population_source"),
                "epochs_recorded": integ.get("epochs_recorded"),
                "run_of_record_epoch_indices": integ.get(
                    "run_of_record_epoch_indices"
                ),
                "reason": integ.get("population_reason"),
            })
    return {
        "n_runs": len(runs),
        "n_runs_with_events": sum(
            1 for r in runs
            if (r.get("integrity") or {}).get("events_present")
        ),
        "verdict_counts": dict(sorted(verdicts.items())),
        "n_conflicted": len(conflicted),
        "n_conflicted_resolved": sum(
            1 for c in conflicted if c["resolved"]
        ),
        "n_conflicted_refused": sum(
            1 for c in conflicted if not c["resolved"]
        ),
        "conflicted_runs": conflicted,
        # Wf-A116 — the runs whose record could not be corroborated by the
        # events stream.  They are NOT refused: their own per-epoch file is
        # kept and scored; they are FLAGGED, because the second source
        # contradicts them and the census does not assert which record is the
        # run.
        "n_uncorroborated": len(uncorroborated),
        "uncorroborated_runs": uncorroborated,
        "note": (
            "Wf-A114E: the trainer writes per_epoch_metrics.json ONCE, at the "
            "end of training, to a FIXED path, so two runs launched into the "
            "same --out-dir clobber each other's record and the survivor is "
            "whichever finished LAST. events.jsonl is append-only and keeps "
            "every population, so the census now cross-checks every run "
            "against it and says, per run, WHICH population it scored and "
            "WHY. A run whose file is a truncated prefix of a longer "
            "population is re-derived from the events rows of the matching "
            "population; a run whose two records disagree irreconcilably is "
            "REFUSED and excluded from every headline population. "
            "Wf-A116: the cross-check is VALUE-AWARE — the per-epoch file is "
            "matched against the events rows of each population, so a file "
            "written by ANOTHER launch that reused the directory is not "
            "scored as if it were the run of record. `class_a_other_"
            "population` = the file is another population's record (the run "
            "of record is scored instead); `class_d_foreign_launch` = the "
            "file matches no population (the events stream is a foreign "
            "launch; the file is kept and the run is flagged UNC"
            "ORROBORATED)."
        ),
    }


def _already_answers(runs: List[Dict]) -> Dict[str, List[str]]:
    """The static catalogue plus the three Wf-A115 entries.

    Wf-A115 — the three new entries reference the artifact's own blocks
    rather than retyping numbers, so they cannot drift from the runs on disk.
    """
    out = {k: list(v) for k, v in ALREADY_ANSWERS.items()}

    e2e = [
        r for r in runs
        if r.get("gen_single_molecule_rate_end") is not None
    ]
    clean = [r for r in e2e if _is_clean(r)]
    depth1_clean = [r for r in clean if "/" not in r["run"]]
    d1_last = [float(r["gen_single_molecule_rate_end"]) for r in depth1_clean]
    # Wf-A152 — the BEST-epoch reading's population is NOT the intersection
    # with the LAST-epoch reading.  It used to be
    #     depth1_clean AND carries a best-epoch rate
    # i.e. CLEAN AND carries a LAST-epoch rate AND carries a best-epoch
    # rate -- a RETIRED population that no predicate ever named, and that
    # dropped the two runs Wf-A127 moved to a null last-epoch rate
    # (cfm_fm_control_2026_10_09, cfm_bond_head_30ep_onecycle_2026_10_09):
    # they are CLEAN and KEEP their best-epoch rate.  MEASURED: the depth-1
    # CLEAN LAST-epoch population is n=22 and the depth-1 CLEAN BEST-epoch
    # population is n=24 -- the same two numbers the artifact's own
    # `by_n_train_depth1_only` block carries, as the sum of
    # `*.clean_only.best_epoch.n` (24) against `*.clean_only.last_epoch.n`
    # (22).
    depth1_clean_best = [
        r for r in runs
        if _is_clean(r)
        and "/" not in r["run"]
        and r.get("gen_single_molecule_rate_best") is not None
    ]
    d1_best = [
        float(r["gen_single_molecule_rate_best"]) for r in depth1_clean_best
    ]

    def _med(vals: List[float]) -> str:
        return repr(statistics.median(vals)) if vals else "n/a"

    def _max(vals: List[float]) -> str:
        return repr(max(vals)) if vals else "n/a"

    out["does fragmentation quality depend on n_train?"] = [
        "UNRESOLVED. The previously published '3.2x inversion' is WITHDRAWN: "
        "it is an artifact of the READING (best-epoch = a max over a noisy "
        "series, which the census itself labels 'optimistically biased') and "
        "of the BUDGET (the two cohorts are out of phase in the same "
        "oscillation).",
        "Under the budget-matched comparison (only runs that actually "
        "REACHED epoch K-1 contribute) the ordering between n_train=256 and "
        "n_train=26715 flips at the budgets listed in "
        "`budget_matched.flipped_budgets` — so no cohort median is a valid "
        "bar. See `budget_matched` and `budget_matched.n_train_axis`.",
        "The 0.572276953125 end-to-end rate that was made the primary bar is "
        "ALSO WITHDRAWN: it comes from cfm_r11_eval256, a run with exactly ONE "
        "recorded epoch (index 19, resumed) whose train_loss is nan, out of a "
        "declared 20. See `retracted_numbers` and "
        "`runs[cfm_r11_eval256].integrity`.",
        f"MEASURED on the clean runs (contiguous from epoch 0, no nan "
        f"train_loss) on the depth-1 loader scope. The two readings have TWO "
        f"populations and they are NOT the same: LAST-epoch reading "
        f"n={len(d1_last)} (CLEAN AND carries a last-epoch rate): median "
        f"{_med(d1_last)}, max {_max(d1_last)}, "
        f"{sum(1 for v in d1_last if v >= 0.50)} of {len(d1_last)} at or "
        f"above 0.50; BEST-epoch reading n={len(d1_best)} (CLEAN AND carries "
        f"a best-epoch rate — a last-epoch rate is NOT required, so this is "
        f"NOT the intersection of the two readings): median {_med(d1_best)}, "
        f"max {_max(d1_best)}, "
        f"{sum(1 for v in d1_best if v >= 0.50)} of {len(d1_best)} at or "
        f"above 0.50. Wf-A152 (2026-10-31) — the best-epoch statistic was "
        f"computed over a RETIRED population: the INTERSECTION of 'has a "
        f"last-epoch rate' and 'has a best-epoch rate', which no predicate "
        f"ever named and which dropped the two CLEAN runs that KEEP a "
        f"best-epoch rate while their last-epoch rate went null on an empty "
        f"denominator (cfm_fm_control_2026_10_09, "
        f"cfm_bond_head_30ep_onecycle_2026_10_09). SUPERSEDED RENDERING, "
        f"preserved verbatim: "
        "MEASURED on the clean runs (contiguous from epoch 0, no nan "
        "train_loss; n=22 on the depth-1 loader scope): final-epoch "
        "median 0.05338541666666667, max 0.5208333333333334, 1 of 22 at "
        "or above 0.50; best-epoch median 0.259765625, max 0.9375, 6 of "
        "22 at or above 0.50.",
        "The per-cohort numbers are in `by_n_train` and "
        "`by_n_train_depth1_only`. A bar must be a PER-RUN, same-reading, "
        "same-cohort, same-budget comparison against a NAMED run — see "
        "`per_run_bar`.",
    ]

    frag = {
        r["run"]: r
        for r in runs
        if r["run"] in ("cfm_frag_assemble", "cfm_frag_control")
    }
    if "cfm_frag_assemble" in frag and "cfm_frag_control" in frag:
        a, b = frag["cfm_frag_assemble"], frag["cfm_frag_control"]
        out["does the assemble/connectivity loss help?"] = [
            "NO — and it made fragmentation WORSE on both readings. "
            "MEASURED (end-to-end single-molecule rate, full-precision JSON):",
            f"  cfm_frag_assemble: best-epoch "
            f"{a.get('gen_single_molecule_rate_best')}, last-epoch "
            f"{a.get('gen_single_molecule_rate_end')}",
            f"  cfm_frag_control:  best-epoch "
            f"{b.get('gen_single_molecule_rate_best')}, last-epoch "
            f"{b.get('gen_single_molecule_rate_end')}",
            "So the connectivity-loss lever was already tried on disk and the "
            "honest end-to-end number is LOWER with it, on both readings. "
            "This is a settled negative, not an open question.",
        ]

    # Wf-A152 — the pool-integrity entry's numbers are COMPUTED here, from
    # the runs on disk, because the 2026-10-30 rendering carried a hard-coded
    # span ("0.0000 to 0.8772") and a hard-coded co-occurrence ("gen_validity
    # reaches 1.000 on pools that sit at 0.203-0.348") that the data does
    # not support.
    _band_lo, _band_hi = 0.203, 0.348
    _sfr_pairs = [
        (r["run"], r.get("gen_single_fragment_ratio_end")) for r in runs
    ]
    _sfr_recorded = [(n, float(v)) for n, v in _sfr_pairs if v is not None]
    _sfr_depth1 = [v for n, v in _sfr_recorded if "/" not in n]
    _gvmax = {r["run"]: r.get("gen_validity_max") for r in runs}
    _gv1 = [
        (r["run"], r.get("gen_single_fragment_ratio_end"))
        for r in runs
        if r.get("gen_validity_max") == 1.0
    ]
    _gv1_in_band = [
        (n, v) for n, v in _gv1
        if v is not None and _band_lo <= float(v) <= _band_hi
    ]
    _sfr_in_band = [
        (n, v) for n, v in _sfr_recorded if _band_lo <= v <= _band_hi
    ]
    _band_runs_no_gv1 = [n for n, _v in _sfr_in_band if _gvmax.get(n) != 1.0]
    _sfr_null = [
        r for r in runs if r.get("gen_single_fragment_ratio_end") is None
    ]
    _sfr_null_not_run = [
        r["run"] for r in _sfr_null if not r.get("gen_eval_ran_on_last_epoch")
    ]
    _sfr_null_empty = [
        r["run"] for r in _sfr_null if r.get("gen_eval_ran_on_last_epoch")
    ]

    def _rate_text(value) -> str:
        return "null" if value is None else repr(float(value))

    _gv1_rendered = ", ".join(
        f"{n} = {_rate_text(v)}" for n, v in sorted(_gv1)
    )
    _band_rendered = ", ".join(
        f"{n} = {_rate_text(v)}" for n, v in sorted(_sfr_in_band)
    )
    _pool_entry = (
        "NO — and the answer is now in this census, per run, as "
        "`gen_single_fragment_ratio_end` + `fragmentation_gate` + "
        "`failure_mode`. "
        f"MEASURED 2026-10-31 (Wf-A152) over the {len(runs)} runs on disk: "
        f"the honest single-fragment ratio is `gen_single_fragment_ratio_end` "
        f"(HIGHER IS BETTER — it is the fraction of the pool that is one "
        f"connected molecule), and it is RECORDED for "
        f"{len(_sfr_recorded)} of the {len(runs)} runs; the other "
        f"{len(_sfr_null)} publish null — for {len(_sfr_null_not_run)} of them "
        f"the last row's generation eval did not run (gen_validity_n == 0), "
        f"and for {len(_sfr_null_empty)} the eval ran but SCORED no rows, so "
        f"the denominator was empty ({', '.join(sorted(_sfr_null_empty))}). "
        f"Over the {len(_sfr_recorded)} recorded runs it spans "
        f"{repr(min(v for _n, v in _sfr_recorded))} to "
        f"{repr(max(v for _n, v in _sfr_recorded))}; over the depth-1 loader "
        f"scope ({len(_sfr_depth1)} runs with a recorded value) it spans "
        f"{repr(min(_sfr_depth1))} to {repr(max(_sfr_depth1))}. "
        f"WITHDRAWN — the earlier claim that gen_validity reaches 1.000 on "
        f"pools that sit at 0.203-0.348 does NOT hold on this corpus: the "
        f"{len(_gv1)} runs whose `gen_validity_max` reaches 1.000 carry "
        f"last-epoch single-fragment ratios of {_gv1_rendered} — "
        f"{len(_gv1_in_band)} of {len(_gv1)} sit in 0.203-0.348. The runs "
        f"whose last-epoch ratio DOES sit in 0.203-0.348 are "
        f"{_band_rendered}, and {len(_band_runs_no_gv1)} of them have a "
        f"`gen_validity_max` that is NOT 1.000. So the metric the line has "
        f"been gating on cannot see this failure. SUPERSEDED RENDERING, "
        f"preserved verbatim: "
        "NO — and the answer is now in this census, per run, as "
        "`gen_single_fragment_ratio_end` + `fragmentation_gate` + "
        "`failure_mode`. MEASURED 2026-10-30 (Wf-A114) over every run on "
        "disk: the honest single-fragment ratio spans 0.0000 to 0.8772 "
        "(HIGHER IS BETTER — it is the fraction of the pool that is one "
        "connected molecule), and gen_validity reaches 1.000 on pools "
        "that sit at 0.203-0.348. So the metric the line has been gating "
        "on cannot see this failure."
    )
    out["does the generated pool hold together as MOLECULES?"][0] = _pool_entry

    out["what is the correct bar?"] = [
        "A PER-RUN, same-reading, same-cohort, same-budget comparison against "
        "a NAMED run. The named bars are in `per_run_bar`:",
        "  primary  — cfm_r9_fix_v2 (n_train=26715), FINAL-epoch reading, "
        "value 0.41145833333333337 (MEASURED, full-precision JSON).",
        "  secondary — cfm_fullcorpus_seed1 (n_train=26715), BEST-epoch "
        "reading, value 0.5260416666666666. A best-epoch reading is a max "
        "over a noisy series and is NOT interchangeable with the final-epoch "
        "one.",
        "  stretch  — 0.9375 at n_train=256 (cfm_fm_ab_2026_10_09 and "
        "cfm_pcgrad_off_2026_10_15, both at epoch 12, both with full "
        "30-epoch histories). Different cohort AND different budget from the "
        "primary bar: REPORT-ONLY, never a pass/fail.",
        "Caveats: `runs[].integrity` records that `n_epochs_run` is a ROW "
        "COUNT and that several runs resumed above epoch 0; `budget_matched` "
        "records that the cohort ordering flips with the budget; "
        "`retracted_numbers` lists the numbers that were published and are "
        "withdrawn.",
    ]
    return out


def _not_counted_run(d: Path) -> Dict[str, object]:
    """Why one run directory was NOT counted — with the evidence, not a shrug.

    Wf-A116 — the loader used to drop a directory silently
    (``if r: runs.append(r)``), so a chimaera directory disappeared from the
    census with no trace: the artifact said ``n_runs = 59`` while 60
    directories carried a ``per_epoch_metrics.csv``, and the ``not_counted``
    prose did not mention the 60th.  MEASURED: ``molmetal/reports/
    cfm_default`` holds a ``per_epoch_metrics.csv`` with ZERO data rows (and a
    zero-row ``per_epoch_metrics.json``) while its ``events.jsonl`` holds 3
    ``done`` events (n_epochs_run 1, 4, 0) and 5 ``train_epoch`` rows.
    """
    csv_path = d / "per_epoch_metrics.csv"
    jpath = d / "per_epoch_metrics.json"
    n_csv: Optional[int] = None
    try:
        with csv_path.open(encoding="utf-8") as fh:
            n_csv = sum(1 for _ in csv.DictReader(fh))
    except OSError:
        n_csv = None
    n_json: Optional[int] = None
    if jpath.is_file():
        try:
            jd = json.load(jpath.open(encoding="utf-8"))
        except Exception:
            jd = {}
        pem = jd.get("per_epoch_metrics")
        n_json = len(pem) if isinstance(pem, list) else 0
    ev = _read_events(d)
    evidence: Dict[str, object] = {
        "per_epoch_metrics_csv_rows": n_csv,
        "per_epoch_metrics_json_rows": n_json,
        "events_present": bool(ev),
    }
    if ev:
        done = list(ev.get("done") or [])
        te = list(ev.get("train_epoch") or [])
        evidence["events_done_count"] = len(done)
        evidence["events_done_n_epochs_run"] = [
            int(dn["n_epochs_run"])
            for dn in done
            if isinstance(dn.get("n_epochs_run"), (int, float))
        ]
        evidence["events_train_epoch_row_count"] = len(te)
    return {
        "run": _run_name(d),
        "run_path": str(d),
        "reason": (
            f"the per-epoch file carries NO data row ({n_csv} row(s) in "
            f"per_epoch_metrics.csv, {n_json} row(s) in "
            f"per_epoch_metrics.json), so there is no series to score; the "
            f"directory is listed here instead of being dropped silently"
        ),
        "evidence": evidence,
    }


def main() -> int:
    runs = []
    not_counted_runs: List[Dict[str, object]] = []
    for csv_path in sorted(REPORTS.rglob("per_epoch_metrics.csv")):
        r = load_run(csv_path.parent)
        if r:
            runs.append(r)
        else:
            # Wf-A116 — a directory that carries a per-epoch file but holds
            # no row in it is a CHIMAERA (e.g. cfm_default: a zero-row file
            # beside an events stream that holds 3 done events and 5
            # train_epoch rows).  It is published, not dropped.
            not_counted_runs.append(_not_counted_run(csv_path.parent))

    _defs = stamp_counts(runs)
    print(
        "bond_pr_auc definitions present: "
        + ", ".join(f"{d}={n}" for d, n in sorted(_defs.items()))
    )
    for d in sorted(_defs):
        print(f"  {_DEF_CODE[d]:>4} = {d}: {bond_ap_legend(d)}")
    print()

    hdr = (
        f"{'run':<46}{'ep':>4}{'rec':>5}{'n_train':>9}{'bs':>4}{'step/ep':>9}"
        f"{'hid':>5}{'lr':>8}{'typeF1':>8}{'exactT':>8}{'gv_max':>8}{'gv_end':>8}"
        f"{'sFrag':>8}{'molRt':>8}{'decSF':>8}{'prAUC':>8}{'def':>4}{'fragGate':>9}"
        f"{'failMode':>22}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(runs, key=lambda x: -(x["bond_type_macro_f1"] or -1)):
        c = r["config"]
        integ = r["integrity"]

        def g(k, w, f="{}"):
            v = c.get(k)
            return (f.format(v) if v is not None else " " * (w - 3) + "n/a").rjust(w)

        def num(v, w):
            return f"{v:>{w}.4f}" if isinstance(v, float) else "n/a".rjust(w)

        print(
            f"{r['run']:<46}{r['n_epochs_run']:>4}{integ['epochs_recorded']:>5}"
            f"{g('n_train', 9)}{g('batch_size', 4)}"
            f"{str(r.get('optimizer_steps_per_epoch', 'n/a')).rjust(9)}"
            f"{g('hidden_dim', 5)}{g('lr', 8)}"
            f"{num(r['bond_type_macro_f1'], 8)}{num(r['bond_exact_type_rate'], 8)}"
            f"{num(r['gen_validity_max'], 8)}{num(r['gen_validity_end'], 8)}"
            f"{num(r['gen_single_fragment_ratio_end'], 8)}"
            f"{num(r['gen_single_molecule_rate_end'], 8)}"
            f"{num(r['decode_single_fragment_ratio_end'], 8)}"
            f"{num(r['bond_pr_auc'], 8)}{_DEF_CODE[r['bond_pr_auc_definition']]:>4}"
            f"{r['fragmentation_gate']['status']:>9}"
            f"{r['failure_mode']:>22}"
        )

    # Wf-A114 (2026-10-30) — the summary the census existed for, now applied
    # to the axis that was missing: connectivity.  Before this, the census
    # could not answer "which runs on disk produce fragments", because it
    # recorded the blind validity and nothing else.
    _frag_statuses = [r["fragmentation_gate"]["status"] for r in runs]
    # Wf-A115 -- the HEADLINE population is the CLEAN runs: contiguous-from-
    # zero epoch history AND no nan train_loss.  A run whose history is one
    # row, resumed above epoch 0, with a nan loss, cannot support a verdict
    # about what a training run delivers, and the block used to lead with
    # exactly such a run: ``cfm_r11_eval256`` (ONE epoch, resumed@19,
    # train_loss nan) was the max of the end-to-end distribution AND was
    # counted as a PASS.  The degenerate runs are NAMED, not silently
    # dropped, and the all-runs distribution is kept beside the clean one so
    # nothing measured is lost.
    _clean_runs = [r for r in runs if _is_clean(r)]
    _degenerate_runs = [r["run"] for r in runs if not _is_clean(r)]
    _clean_statuses = [r["fragmentation_gate"]["status"] for r in _clean_runs]
    _e2e = [
        float(r["gen_single_molecule_rate_end"])
        for r in _clean_runs
        if r.get("gen_single_molecule_rate_end") is not None
    ]
    _e2e_sorted = sorted(_e2e)
    _mid = len(_e2e_sorted) // 2
    _best = [
        float(r["gen_single_molecule_rate_best"])
        for r in _clean_runs
        if r.get("gen_single_molecule_rate_best") is not None
    ]
    _best_sorted = sorted(_best)
    _bmid = len(_best_sorted) // 2
    # The same two distributions over ALL runs, for the reader who wants the
    # corpus as it stands rather than the clean subset.
    _e2e_all = [
        float(r["gen_single_molecule_rate_end"])
        for r in runs
        if r.get("gen_single_molecule_rate_end") is not None
    ]
    _e2e_all_sorted = sorted(_e2e_all)
    _mid_all = len(_e2e_all_sorted) // 2
    _best_all = [
        float(r["gen_single_molecule_rate_best"])
        for r in runs
        if r.get("gen_single_molecule_rate_best") is not None
    ]
    _best_all_sorted = sorted(_best_all)
    _bmid_all = len(_best_all_sorted) // 2
    # Runs whose LAST epoch is total collapse (validity 0.0) although some
    # earlier epoch reached a real validity.  For these the last-epoch
    # end-to-end rate reads 0.0 and understates the run by its full range.
    _collapsed = [
        r["run"]
        for r in runs
        if r.get("gen_validity_end") == 0.0
        and (r.get("gen_validity_max") or 0.0) > 0.5
    ]
    # Wf-A152 — the collapse runs' BEST-epoch rates, so the note can name the
    # surviving reading next to the null one.
    _collapsed_best = ", ".join(
        f"{r['run']} = {r.get('gen_single_molecule_rate_best')}"
        for r in runs
        if r["run"] in _collapsed
    )
    frag_summary = {
        "n_runs_with_verdict": len(_frag_statuses),
        "n_pass": _frag_statuses.count("PASS"),
        "n_fail": _frag_statuses.count("FAIL"),
        "n_skip": _frag_statuses.count("SKIP"),
        "failing_runs": [
            r["run"] for r in runs if r["fragmentation_gate"]["status"] == "FAIL"
        ],
        "runs_with_a_recorded_number": [
            r["run"]
            for r in runs
            if r["gen_single_fragment_ratio_end"] is not None
        ],
        "n_runs_with_an_end_to_end_rate": len(_e2e),
        "end_to_end_single_molecule_rate": {
            "n": len(_e2e),
            "min": _e2e_sorted[0] if _e2e_sorted else None,
            "max": _e2e_sorted[-1] if _e2e_sorted else None,
            "mean": (sum(_e2e) / len(_e2e)) if _e2e else None,
            "median": (
                _e2e_sorted[_mid]
                if _e2e_sorted and len(_e2e_sorted) % 2
                else (
                    (_e2e_sorted[_mid - 1] + _e2e_sorted[_mid]) / 2.0
                    if _e2e_sorted
                    else None
                )
            ),
            "n_below_0.50": sum(1 for x in _e2e if x < 0.50),
            "n_below_0.10": sum(1 for x in _e2e if x < 0.10),
            "basis": (
                "gen_validity_end x gen_single_fragment_ratio_end, paired "
                "row by row: the share of ALL sampled molecules that are ONE "
                "connected, RDKit-parseable molecule"
            ),
            "population": (
                "CLEAN runs only: contiguous-from-zero epoch history AND no "
                "nan train_loss (see `integrity`). The degenerate runs are "
                "named in `degenerate_runs_excluded_from_the_headline`; the "
                "all-runs distribution is in "
                "`end_to_end_single_molecule_rate_all_runs`"
            ),
        },
        "best_epoch_single_molecule_rate": {
            "n": len(_best),
            "min": _best_sorted[0] if _best_sorted else None,
            "max": _best_sorted[-1] if _best_sorted else None,
            "mean": (sum(_best) / len(_best)) if _best else None,
            "median": (
                _best_sorted[_bmid]
                if _best_sorted and len(_best_sorted) % 2
                else (
                    (_best_sorted[_bmid - 1] + _best_sorted[_bmid]) / 2.0
                    if _best_sorted
                    else None
                )
            ),
            "n_at_or_above_0.50": sum(1 for x in _best if x >= 0.50),
            "n_at_or_above_0.10": sum(1 for x in _best if x >= 0.10),
            "basis": (
                "the MAXIMUM over the run's epochs of "
                "gen_validity x gen_single_fragment_ratio, paired row by row. "
                "A max over a noisy epoch series is optimistically biased — it "
                "is the number a run that early-stops at its best epoch would "
                "deliver, NOT the number the run delivers on average"
            ),
            "population": (
                "CLEAN runs only — see "
                "`end_to_end_single_molecule_rate.population`"
            ),
        },
        # Wf-A115 -- the SAME two distributions over ALL runs, so the clean
        # subset above is a choice the reader can see and undo.
        "end_to_end_single_molecule_rate_all_runs": {
            "n": len(_e2e_all),
            "min": _e2e_all_sorted[0] if _e2e_all_sorted else None,
            "max": _e2e_all_sorted[-1] if _e2e_all_sorted else None,
            "mean": (sum(_e2e_all) / len(_e2e_all)) if _e2e_all else None,
            "median": (
                _e2e_all_sorted[_mid_all]
                if _e2e_all_sorted and len(_e2e_all_sorted) % 2
                else (
                    (_e2e_all_sorted[_mid_all - 1] + _e2e_all_sorted[_mid_all])
                    / 2.0
                    if _e2e_all_sorted
                    else None
                )
            ),
            "n_below_0.50": sum(1 for x in _e2e_all if x < 0.50),
            "n_below_0.10": sum(1 for x in _e2e_all if x < 0.10),
            "population": (
                "ALL runs, degenerate ones included. The maximum over this "
                "population is the WITHDRAWN cfm_r11_eval256 value — see "
                "`retracted_max`"
            ),
        },
        "best_epoch_single_molecule_rate_all_runs": {
            "n": len(_best_all),
            "min": _best_all_sorted[0] if _best_all_sorted else None,
            "max": _best_all_sorted[-1] if _best_all_sorted else None,
            "mean": (sum(_best_all) / len(_best_all)) if _best_all else None,
            "median": (
                _best_all_sorted[_bmid_all]
                if _best_all_sorted and len(_best_all_sorted) % 2
                else (
                    (
                        _best_all_sorted[_bmid_all - 1]
                        + _best_all_sorted[_bmid_all]
                    )
                    / 2.0
                    if _best_all_sorted
                    else None
                )
            ),
            "n_at_or_above_0.50": sum(1 for x in _best_all if x >= 0.50),
            "n_at_or_above_0.10": sum(1 for x in _best_all if x >= 0.10),
            "population": "ALL runs, degenerate ones included",
        },
        # Wf-A115 -- the degenerate runs are NAMED, not dropped in silence.
        "degenerate_runs_excluded_from_the_headline": _degenerate_runs,
        "degenerate_note": (
            "a run is degenerate when its epoch history is not contiguous "
            "from zero (it resumed above epoch 0, so its early epochs are "
            "absent) OR its train_loss is nan somewhere. Such a run cannot "
            "support a statement about what a training run delivers: "
            "cfm_r11_eval256 recorded ONE epoch (index 19) of a declared 20 "
            "with train_loss nan, and it was the maximum of the "
            "all-runs end-to-end distribution before this wave"
        ),
        "verdict_counts_clean_only": {
            "n_runs": len(_clean_statuses),
            "n_pass": _clean_statuses.count("PASS"),
            "n_fail": _clean_statuses.count("FAIL"),
            "n_skip": _clean_statuses.count("SKIP"),
            "note": (
                "the fragmentation verdict over the clean runs only; the "
                "all-runs counts above are in `n_pass` / `n_fail` / "
                "`n_skip`"
            ),
        },
        # Wf-A115 -- the pointer the block was missing: the number that was
        # published as the headline is WITHDRAWN, and the block said so
        # nowhere.
        "retracted_max": {
            "value": "0.572265625",
            "run": "cfm_r11_eval256",
            "status": "WITHDRAWN",
            "why": (
                "one recorded epoch (index 19) of a declared 20, resumed "
                "above epoch 0, train_loss nan — see `retracted_numbers` at "
                "the top level of this artifact. The 0.572276953125 form "
                "that was published is the 4-decimal CSV rendering of this "
                "JSON value"
            ),
            "in_this_block": (
                "the value is the max of "
                "`end_to_end_single_molecule_rate_all_runs`, NOT of the "
                "clean-run headline above"
            ),
        },
        "runs_collapsed_at_their_last_epoch": _collapsed,
        "retracted_numbers": RETRACTED_NUMBERS,
        "collapsed_note": (
            "gen_validity_end == 0.0 while gen_validity_max > 0.5: the last "
            "epoch is a total collapse. The last-epoch end-to-end rate is "
            "PUBLISHED AS null for these runs, NOT as 0.0: the generation eval "
            "DREW samples (gen_validity_n > 0) and SCORED none of them "
            "(gen_validity_val == 0), so the last-epoch denominator is EMPTY "
            "and the rate is undefined (the Wf-A127 null-vs-0.0 convention). "
            "gen_validity itself stays a number — it is a measured 0 of N — "
            "and it is the chemistry rate over the SCORED rows that has no "
            "denominator. MEASURED 2026-10-31: the "
            f"{len(_collapsed)} runs named in "
            f"`runs_collapsed_at_their_last_epoch` ({_collapsed_best}) carry "
            "gen_single_molecule_rate_end = null and "
            "gen_single_fragment_ratio_end = null, and their BEST-epoch rates "
            "are the surviving reading. Wf-A152 (2026-10-31) — the previous "
            "rendering said the rate 'reads 0.0', which is what the artifact "
            "publishes as null; SUPERSEDED RENDERING, preserved verbatim: "
            "gen_validity_end == 0.0 while gen_validity_max > 0.5: the last "
            "epoch is a total collapse, so the last-epoch end-to-end rate "
            "reads 0.0 and understates the run by its full range"
        ),
    }
    print()
    print(
        f"fragmentation gate: PASS={frag_summary['n_pass']} "
        f"FAIL={frag_summary['n_fail']} SKIP={frag_summary['n_skip']} "
        f"of {frag_summary['n_runs_with_verdict']} runs"
    )
    _es = frag_summary["end_to_end_single_molecule_rate"]
    if _e2e:
        print(
            f"end-to-end single-molecule rate over {len(_e2e)} run(s): "
            f"median={_es['median']:.4f}  min={_es['min']:.4f}  "
            f"max={_es['max']:.4f}  below 0.50: {_es['n_below_0.50']}"
        )
    _bs = frag_summary["best_epoch_single_molecule_rate"]
    if _best:
        print(
            f"best-epoch end-to-end rate over {len(_best)} run(s): "
            f"median={_bs['median']:.4f}  max={_bs['max']:.4f}  "
            f">=0.50: {_bs['n_at_or_above_0.50']}"
        )
    if _collapsed:
        print(
            "collapsed at their last epoch (validity 0.0, max > 0.5): "
            + ", ".join(_collapsed)
        )

    # Wf-A115 — the cohort-stratified view, the budget-matched comparison,
    # the named-run bar, and the paired-headline detector.
    by_n_train = _by_n_train(runs)
    depth1_runs = [r for r in runs if "/" not in r["run"]]
    by_n_train_depth1 = _by_n_train(depth1_runs)
    # Wf-A114 verify:integration D1 -- the STRICT cohort, keyed on the
    # (n_train, n_val) PAIR.  This is the definition
    # `cfm_cohort_matched_compare.py::axis_cohort` enforces, so the census and
    # the harness now agree on what a cohort is.
    by_cohort = _by_cohort(runs)
    by_cohort_depth1 = _by_cohort(depth1_runs)
    budget_matched = _budget_matched(runs)
    per_run_bar = _per_run_bar(runs)
    paired_per_run, paired_flagged = _paired_headline(runs)
    for r in runs:
        r["paired_headline_divergence"] = paired_per_run.get(r["run"])
    by_n_train["retracted_numbers"] = RETRACTED_NUMBERS

    _resume = [r["run"] for r in runs if r["integrity"]["resumed"]]
    _nanloss = [r["run"] for r in runs if r["integrity"]["nan_train_loss"]]
    _notclean = [r["run"] for r in runs if not _is_clean(r)]

    print()
    print(
        f"run integrity: {len(_resume)} run(s) resumed above epoch 0: "
        + ", ".join(_resume)
    )
    print(
        f"run integrity: {len(_nanloss)} run(s) carry a nan train_loss: "
        + ", ".join(_nanloss)
    )
    print(
        f"run integrity: {len(_notclean)} run(s) are not clean "
        "(contiguous-from-zero AND no nan loss): " + ", ".join(_notclean)
    )
    # Wf-A114E — the population cross-check, in the console too.
    _pop = _population_summary(runs)
    print(
        f"population check: {_pop['n_runs_with_events']}/{_pop['n_runs']} "
        f"run(s) carry an events.jsonl; verdicts: "
        + ", ".join(f"{k}={v}" for k, v in _pop["verdict_counts"].items())
    )
    for _c in _pop["conflicted_runs"]:
        print(
            f"  CONFLICT {_c['run']}: {_c['verdict']} "
            f"({'re-derived from events' if _c['resolved'] else 'REFUSED'})"
        )
    print()
    print()
    print("cohort (by_n_train):")
    for label, blk in by_n_train.items():
        if not isinstance(blk, dict) or "n_runs" not in blk:
            continue
        cl = blk["clean_only"]
        print(
            f"  n_train={label:<8} runs={blk['n_runs']:<3} "
            f"e2e={blk['n_runs_with_e2e']:<3} "
            f"last med={blk['last_epoch']['median']} "
            f"best med={blk['best_epoch']['median']} | "
            f"clean={cl['last_epoch']['n']}/{cl['best_epoch']['n']} "
            f"excluded={cl['n_excluded']}"
        )
    print()
    print(
        "budget-matched ordering flips at K = "
        + ", ".join(str(k) for k in budget_matched["flipped_budgets"])
    )
    print(f"n_train axis: {budget_matched['n_train_axis']}")
    print()
    if per_run_bar.get("primary"):
        p = per_run_bar["primary"]
        print(
            f"primary bar: {p['run']} ({p['cohort']}, {p['reading']}) = "
            f"{p['value']}"
        )
    if per_run_bar.get("secondary"):
        s = per_run_bar["secondary"]
        print(
            f"secondary bar: {s['run']} ({s['cohort']}, {s['reading']}) = "
            f"{s['value']}"
        )
    if per_run_bar.get("stretch", {}).get("value") is not None:
        print(
            "stretch (report-only, different cohort+budget): "
            f"{per_run_bar['stretch']['value']}"
        )
    print(
        f"paired headline divergences: {len(paired_flagged)} -> "
        + ", ".join(paired_flagged)
    )

    # Wf-A152 — the counts the wf_a115 correction quotes, measured here.
    _n_decode_ratio_null = sum(
        1 for r in runs if r.get("decode_ratio") is None
    )
    _n_decode_eval_not_run = sum(
        1 for r in runs if not r.get("decode_eval_ran_on_last_epoch")
    )
    _n_decode_ratio_measured_null = sum(
        1 for r in runs if r.get("decode_ratio_measured") is None
    )
    _n_dsfr_end_null = sum(
        1 for r in runs if r.get("decode_single_fragment_ratio_end") is None
    )
    _n_dsfr_raw_null = sum(
        1 for r in runs
        if r.get("decode_single_fragment_ratio_end_measured") is None
    )

    out = {
        "metric": "cfm_run_census",
        "purpose": (
            "Every CFM run on disk with its own recorded config and results, so "
            "a proposal can be checked against what is already measured before "
            "another run is launched."
        ),
        "source": "each run's own per_epoch_metrics.{csv,json}; nothing re-derived",
        "n_runs": len(runs),
        "corpus": {
            "loader": "REPORTS.rglob('per_epoch_metrics.csv')",
            "n_runs_loaded": len(runs),
            # Wf-A116 — the directories the loader SAW, and the ones it could
            # not score.  MEASURED: 60 directories carry a per_epoch_metrics.csv
            # and 59 of them are scored; cfm_default is the 60th (a zero-row
            # file beside an events stream) and is listed in `not_counted_runs`.
            "n_run_directories_with_per_epoch_csv": len(runs) + len(
                not_counted_runs
            ),
            "n_runs_not_counted": len(not_counted_runs),
            "n_runs_depth1": len(depth1_runs),
            "n_runs_nested": len(runs) - len(depth1_runs),
            "run_name": "path relative to molmetal/reports (POSIX)",
            "run_path": "absolute path of the run directory",
            # Wf-A115 — the leaf directory name, so a consumer that keys the
            # census by directory name (cfm_fragmentation_audit.py does:
            # `census.get(d.name)`) can find a nested run.  The list makes the
            # ambiguity explicit instead of silently dropping a run.
            "run_index_by_dir_name": {
                name: sorted(
                    r["run"] for r in runs if r["run"].rsplit("/", 1)[-1] == name
                )
                for name in sorted(
                    {r["run"].rsplit("/", 1)[-1] for r in runs}
                )
            },
            "series_source": {
                "json": sum(
                    1 for r in runs if r["integrity"]["series_source"] == "json"
                ),
                "csv": sum(
                    1 for r in runs if r["integrity"]["series_source"] == "csv"
                ),
                # Wf-A114E — the "events" bucket: runs whose per-epoch file
                # was a truncated prefix and whose series was RE-DERIVED
                # from the events stream of the matching population.
                "events": sum(
                    1 for r in runs if r["integrity"]["series_source"] == "events"
                ),
            },
            "note": (
                "Wf-A115: the loader used to be REPORTS.iterdir() — one level "
                "deep — so the nested runs were structurally invisible. The "
                "corpus is therefore LARGER than the one the pre-Wf-A115 "
                "cohort medians were measured on; `by_n_train_depth1_only` "
                "keeps that earlier scope reproducible."
            ),
        },
        "bond_pr_auc_definitions": _defs,
        "bond_pr_auc_definition_legend": {
            d: bond_ap_legend(d) for d in sorted(_defs)
        },
        "runs": [_public_run(r) for r in runs],
        # Wf-A114E — the population cross-check, census-wide.  A conflict
        # must be VISIBLE in the artifact, never silently resolved.
        "population_conflicts": _population_summary(runs),
        "fragmentation_summary": frag_summary,
        # Wf-A115 -- the early-stop guard, aggregated.  The per-run block is
        # `runs[].early_stop_guard`; this is the census-wide answer to "does
        # every run on disk gate on the criterion it says it gates on?".
        "early_stop_guard": {
            "n_runs": len(runs),
            "n_ok": sum(
                1 for r in runs
                if (r.get("early_stop_guard") or {}).get("state") == "ok"
            ),
            "n_degraded": sum(
                1 for r in runs
                if (r.get("early_stop_guard") or {}).get("state") == "degraded"
            ),
            "n_selected_but_not_emitted": sum(
                1 for r in runs
                if (r.get("early_stop_guard") or {}).get("state")
                == "selected_but_not_emitted"
            ),
            "n_not_recorded": sum(
                1 for r in runs
                if (r.get("early_stop_guard") or {}).get("state")
                == "not_recorded"
            ),
            "degraded_runs": [
                {
                    "run": r["run"],
                    "requested_metric": (r["early_stop_guard"] or {}).get(
                        "requested_metric"
                    ),
                    "gated_on": (r["early_stop_guard"] or {}).get(
                        "resolved_metrics"
                    ),
                }
                for r in runs
                if (r.get("early_stop_guard") or {}).get("state") == "degraded"
            ],
            "selected_but_not_emitted_runs": [
                {
                    "run": r["run"],
                    "requested_metric": (r["early_stop_guard"] or {}).get(
                        "requested_metric"
                    ),
                    "missing_columns": (r["early_stop_guard"] or {}).get(
                        "missing_columns"
                    ),
                }
                for r in runs
                if (r.get("early_stop_guard") or {}).get("state")
                == "selected_but_not_emitted"
            ],
            "not_recorded_runs": [
                r["run"]
                for r in runs
                if (r.get("early_stop_guard") or {}).get("state")
                == "not_recorded"
            ],
            "why": (
                "the trainer DEGRADES an early-stop criterion it cannot "
                "measure, so a config that selects a criterion is not "
                "evidence that the run used it. c9 selects "
                "`molecule_validity`; a run that selected it and never "
                "emitted `gen_molecule_validity` was gated on something else"
            ),
        },
        "cohort_axis": {
            "definition": (
                "a cohort is the PAIR (n_train, n_val).  `by_cohort` is "
                "stratified on that pair and is the definition "
                "molmetal/scripts/cfm_cohort_matched_compare.py::axis_cohort "
                "enforces.  `by_n_train` is stratified on n_train ALONE and is "
                "the older, LOOSER axis."
            ),
            "strict_block": "by_cohort",
            "loose_block": "by_n_train",
            "why_two_blocks": (
                "MEASURED: the n_train=26715 label holds ~~13 runs~~ **14 "
                "runs** that split into two validation protocols (n_val=192 in "
                "9 runs and n_val=6714 in 5), so "
                "its median is a median over two cohorts.  The loose block is "
                "kept because it is the axis the published cohort medians were "
                "measured on; the strict block is the one a comparison may "
                "use."
            ),
            "n_val_is_promoted": (
                "each run's own `n_val` is a first-class field (read from its "
                "own config block), so the cohort axis is not re-derived"
            ),
        },
        "by_cohort": by_cohort,
        "by_cohort_depth1_only": by_cohort_depth1,
        "by_n_train": by_n_train,
        "by_n_train_depth1_only": by_n_train_depth1,
        "budget_matched": budget_matched,
        "per_run_bar": per_run_bar,
        "paired_headline_divergences": paired_flagged,
        "retracted_numbers": RETRACTED_NUMBERS,
        "already_answers": _already_answers(runs),
        "not_counted": (
            "Runs killed before their first per_epoch_metrics.csv write leave no "
            "row. Two exist from 2026-10-10 (wf_cfm_eps_gate_2026_10_10 and "
            "wf_cfm_eps_gate_tmqminit_2026_10_10, stopped at ep5/ep2 of 30); "
            "their master.log holds the only record of the epsilon-gate note and "
            "the tmQM weight load. "
            "A third, and the one the fragmentation question is about: "
            "cfm_c8_fullgen_2026_10_24 died in epoch 3 with a CUDA OOM in "
            "triton_kernels/fused_mlp.py:889 before it ever wrote a "
            "per_epoch_metrics.csv, so its 3 epochs are NOT in this census. "
            "Its master.log and generated_smiles.csv survive and are scored by "
            "molmetal/scripts/cfm_fragmentation_gate.py; the c8 config header "
            "says 18 epochs and the run did 3. "
            "And one directory DOES carry a per_epoch_metrics.csv but is still "
            "not counted: cfm_default — the file holds ZERO data rows (and a "
            "zero-row per_epoch_metrics.json) while its events.jsonl holds 3 "
            "done events (n_epochs_run 1, 4, 0) and 5 train_epoch rows, so "
            "the directory is a CHIMAERA with no per-epoch series to score. It "
            "is listed, with its evidence, in `not_counted_runs`."
        ),
        # Wf-A116 — every directory that carries a per-epoch file but is NOT
        # a scored run, with the reason and the evidence.  The loader used to
        # drop these silently, so the corpus count and the directories on
        # disk disagreed with no trace.
        "not_counted_runs": not_counted_runs,
        "n_not_counted_runs": len(not_counted_runs),
        "headline_gap": {
            "best_on_disk_typeF1": "cfm_r9_fix_v2 0.9470 (n_train=26715)",
            "canonical_config_typeF1": "cfm_fm_control_2026_10_09 0.3770 (n_train=256)",
            "ratio": 2.51,
            "cause_candidate": (
                "n_train 26715 vs 256 = 104x more complexes; ~~with batch 16 vs 64 "
                "that is ~417x vs 4 optimizer steps per epoch. NOT yet "
                "attributed to a single variable — r9 also differs in batch "
                "size, lr and epoch budget.~~ "
                "**CORRECTION (2026-10-31, Wf-A153): the batch sizes and the "
                "step counts are REFUTED — BOTH runs use `batch_size` 16, so "
                "the step counts are 1670 vs 16 (104x), not ~417x vs 4, and "
                "r9 does NOT differ in batch size.**  MEASURED 2026-10-31 from "
                "the two runs' own recorded blocks in "
                "`metrics/by_metric/cfm_run_census.json` -> `runs[]`: "
                "cfm_r9_fix_v2 `config.batch_size` 16 / "
                "`optimizer_steps_per_epoch` 1670, and "
                "cfm_fm_control_2026_10_09 `config.batch_size` 16 / "
                "`optimizer_steps_per_epoch` 16 (26715/16 = 1669.7, 256/16 = "
                "16).  NOT yet attributed to a single variable — r9 also "
                "differs in lr and epoch budget."
            ),
        },
        "changelog": {
            "wf_a150": [
                "MEANING CHANGED — the `clean_only.best_epoch` reading in "
                "`by_n_train`, `by_n_train_depth1_only`, `by_cohort` and "
                "`by_cohort_depth1_only` is now computed over the "
                "POPULATION ITS OWN `predicate` STRING NAMES "
                "(`integrity.epoch_history_contiguous_from_zero AND NOT "
                "integrity.nan_train_loss` AND the run carries a best-epoch "
                "rate).  It used to be computed over the INTERSECTION of "
                "that predicate with the LAST-epoch population (the run also "
                "carries a last-epoch rate) — an intersection the predicate "
                "never mentioned, and which disagreed with the sibling "
                "`best_epoch` block and with the top-level "
                "`fragmentation_summary.best_epoch_single_molecule_rate`, "
                "both of which are computed over the NON-intersected "
                "population.",
                "WHY IT BECAME VISIBLE ONLY NOW — the intersection "
                "separated from the non-intersected population for the "
                "first time under Wf-A127, which published the "
                "empty-denominator chemistry rate as `null`.  The two runs "
                "that lost their last-epoch rate (cfm_fm_control_2026_10_09, "
                "cfm_bond_head_30ep_onecycle_2026_10_09) are CLEAN and KEEP "
                "their best-epoch rate, so they fell out of the "
                "`clean_only.best_epoch` block while remaining inside the "
                "sibling `best_epoch` block.",
                "MEASURED 2026-10-31 — the n_train=256 cohort's "
                "`clean_only.best_epoch` moves from n=6 / median 0.59375 / "
                "mean 0.58984375 / 3 at or above 0.50 to n=8 / median "
                "0.7708333333333333 / mean 0.6376953125 / 5 at or above "
                "0.50.  The two ADDED runs are cfm_fm_control_2026_10_09 "
                "(best-epoch rate 0.6875) and "
                "cfm_bond_head_30ep_onecycle_2026_10_09 (best-epoch rate "
                "0.875) — the same two runs the artifact already names in "
                "`fragmentation_summary.runs_collapsed_at_their_last_epoch`.",
                "NO OTHER BLOCK MOVED — `fragmentation_summary` (the "
                "headline), `budget_matched`, `by_n_train.last_epoch`, "
                "`by_n_train.best_epoch` and every `clean_only.last_epoch` "
                "are UNCHANGED by this repair; a rebuild diff against the "
                "previous artifact shows exactly four changed blocks, each "
                "differing only in `clean_only.best_epoch`.  The "
                "budget-matched flip set is unchanged at [9, 11, 25].",
            ],
            "wf_a127": [
                "MEANING CHANGED — a chemistry rate whose denominator is "
                "EMPTY is now published as null, never as the trainer's 0.0 "
                "placeholder.  There are TWO denominators in a generation-"
                "eval row: gen_validity_n (the samples the eval DREW) and "
                "gen_validity_val (the rows the eval SCORED).  The "
                "chemistry rates (gen_single_fragment_ratio, "
                "gen_largest_fragment_frac, gen_metal_ratio) are over the "
                "SCORED rows, so when the eval drew samples and NONE of "
                "them parsed they are 0/0 — undefined.  The census used to "
                "gate them on the DRAWN count, so the 0.0 placeholder was "
                "published as a measurement and the fragmentation gate "
                "FAILED on it.  MEASURED 2026-10-31 on disk: 66 of the 576 "
                "per-epoch rows across the 59 runs have gen_validity_val == "
                "0 with gen_validity_n > 0, over 5 runs; on 2 of them it is "
                "the LAST row (cfm_fm_control_2026_10_09, "
                "cfm_bond_head_30ep_onecycle_2026_10_09).  Those two runs' "
                "fragmentation_gate moves from FAIL to SKIP / "
                "denominator_empty, and gen_single_fragment_ratio_end, "
                "gen_largest_fragment_frac_end and gen_single_molecule_rate_end "
                "move from 0.0 to null.  The convention is the trainer's own "
                "(Wf-A114/T85, molmetal/tests/test_cfm_gen_metric_sentinel.py): "
                "a rate over the SCORED rows is None when nothing was "
                "scored, the end-to-end rate inherits that undefined factor, "
                "and a rate over the DRAWN samples (gen_validity, "
                "gen_molecule_validity) stays a number — a measured 0 of N "
                "stays 0.0.",
                "MEANING CHANGED — 'denominator_empty' now has TWO "
                "sub-causes and its reason string names the one that is "
                "true: the eval drew NOTHING (gen_validity_n == 0), or the "
                "eval drew samples but SCORED no rows (gen_validity_val == "
                "0).  The old string claimed the first for both.  The BASIS "
                "value is unchanged — one basis, two causes.",
                "CONSUMER IMPACT — the two runs leave the end-to-end rate "
                "distributions, so every cohort aggregate that carried "
                "their placeholder 0.0 shifts: "
                "('FAIL','end_to_end') 27 -> 25, "
                "('SKIP','denominator_empty') 16 -> 18, and the n_train=256 "
                "cohort's last-epoch median moves with them.  A "
                "fragmentation_gate_last_measured block is now emitted for "
                "a run whose last epoch is a SKIP, so the last MEASURED "
                "verdict survives beside the SKIP.",
            ],
            "wf_a125": [
                "MEANING CHANGED — fragmentation_gate SKIP now carries one of "
                "TWO bases instead of a single conflated reason.  "
                "'field_absent' means gen_single_fragment_ratio was never "
                "recorded in any epoch row of the run; 'denominator_empty' "
                "means it WAS recorded (the trainer's 0.0 placeholder) but "
                "the generation eval drew nothing (gen_validity_n == 0), so "
                "the ratio has an empty denominator.  The old reason string "
                "('no gen_single_fragment_ratio recorded for this run') was "
                "FALSE for the denominator_empty runs.  MEASURED on this "
                "working tree, over the 27 SKIP runs the census finds: 11 "
                "field_absent and 16 denominator_empty.  The split is decided "
                "from the census's OWN normalised per-epoch row (the "
                "<key>_raw flag), never from an offline events.jsonl read: "
                "wf299_cfm_main/wave_a/full_stack carries the field in "
                "events.jsonl but its per-epoch series does not, so the "
                "census reports field_absent for it.",
            ],
            "wf_a115": [
                "LOADER: REPORTS.iterdir() -> REPORTS.rglob('per_epoch_metrics.csv'). "
                "The nested runs are now visible; `run` is the path relative to "
                "REPORTS and `run_path` the absolute directory.",
                "SERIES SOURCE: the per-epoch series now comes from "
                "per_epoch_metrics.json (full float precision) with the CSV as "
                "a fallback; `integrity.series_source` records which. This is "
                "why the primary bar is 0.41145833333333337 and not the "
                "4-decimal CSV rendering 0.41147500000000004.",
                "ADDED per run: integrity, n_train, head_levers, "
                "head_levers_source, head_levers_source_per_key, "
                "head_levers_recorded, failure_mode, failure_mode_best, "
                "paired_headline_divergence, run_path, "
                "gen_eval_ran_on_last_epoch, gen_validity_last_row, "
                "decode_ratio_measured, "
                "decode_single_fragment_ratio_end_measured, "
                "fragmentation_gate.epoch, fragmentation_gate_last_measured.",
                "ADDED at the top level: by_n_train, by_n_train_depth1_only, "
                "budget_matched, per_run_bar, paired_headline_divergences, "
                "retracted_numbers, corpus, changelog.",
                "MEANING CHANGED — gen_single_fragment_ratio_end, "
                "gen_largest_fragment_frac_end, gen_molecule_validity_end: "
                "these now read null when the LAST row's generation eval did "
                "not run (gen_validity_n == 0) instead of the trainer's 0.0 "
                "placeholder. MEASURED on the depth-1 corpus: 119 of 496 "
                "generation rows across 25 runs carry that placeholder, and "
                "TWO of them are LAST-epoch rows. The DECODE-side numbers "
                "(decode_ratio, decode_single_fragment_ratio_end) are DECODE "
                "metrics, measured independently of the generation eval: they "
                "read null on the DECODE predicate (the decode eval did not "
                "run on the last epoch), NOT on the generation predicate, and "
                "their raw values are kept in decode_ratio_measured / "
                "decode_single_fragment_ratio_end_measured. Wf-A152 "
                "CORRECTION (2026-10-31) — the previous rendering of this "
                "note listed decode_ratio and "
                "decode_single_fragment_ratio_end among the columns that "
                "'now read null when the LAST row's generation eval did not "
                "run'. THAT CLAIM DOES NOT HOLD. MEASURED over the "
                f"{len(runs)} census runs: {_n_decode_ratio_null} of the "
                f"{len(runs)} carry a null decode_ratio, and "
                f"{_n_decode_eval_not_run} of the {len(runs)} have the decode "
                f"eval failing on the last epoch (and "
                f"{_n_decode_ratio_measured_null} carry a null "
                f"decode_ratio_measured), so the decode-side null case does "
                f"not arise at all on this corpus. {_n_dsfr_end_null} of the "
                f"{len(runs)} DO carry a null "
                f"decode_single_fragment_ratio_end, and for every one of them "
                f"the field was never recorded on the last row (the raw "
                f"decode_single_fragment_ratio_end_measured is null for "
                f"{_n_dsfr_raw_null} of them too), which is a field-absent "
                f"case and not a 'the generation eval did not run' case. "
                f"SUPERSEDED RENDERING, preserved verbatim: "
                "MEANING CHANGED — gen_single_fragment_ratio_end, "
                "gen_largest_fragment_frac_end, gen_molecule_validity_end, "
                "decode_ratio, decode_single_fragment_ratio_end: these now read "
                "null when the LAST row's generation eval did not run "
                "(gen_validity_n == 0) instead of the trainer's 0.0 placeholder. "
                "MEASURED on the depth-1 corpus: 119 of 496 generation rows "
                "across 25 runs carry that placeholder, and TWO of them are "
                "LAST-epoch rows. The decode-side numbers are DECODE metrics, "
                "measured independently of the generation eval, so their raw "
                "values are kept in decode_ratio_measured / "
                "decode_single_fragment_ratio_end_measured.",
                "MEANING CHANGED — fragmentation_gate is now the verdict at "
                "the run's LAST epoch (SKIP when that epoch's eval did not "
                "run) instead of the verdict at the last epoch that carried "
                "a number; the latter is preserved as "
                "`fragmentation_gate_last_measured`.",
                "MEANING CLARIFIED — n_epochs_run is a ROW COUNT and was "
                "always one; it is not 'epochs trained'. "
                "integrity.epochs_recorded (distinct epoch indices) and the "
                "rest of the integrity block are the new, explicit fields.",
                "CONSUMER IMPACT: "
                "molmetal/scripts/cfm_fragmentation_audit.py reads "
                "gen_single_fragment_ratio_end into its conditional "
                "statistics, so its median over the corpus changes now that "
                "the placeholder 0.0 rows are null. It also joins the census "
                "by DIRECTORY NAME (`census.get(d.name)`), so with the run "
                "name now a relative path the join still works for every "
                "depth-1 run and comes back EMPTY for a nested one; "
                "`corpus.run_index_by_dir_name` lists the nested runs by leaf "
                "name so that consumer's owner can resolve them. "
                "molmetal/scripts/c7_scale_matched_pools.py reads only "
                "`config` from the census and is unaffected for the depth-1 "
                "runs it iterates.",
            ],
        },
    }
    dest = REPO / "metrics" / "by_metric" / "cfm_run_census.json"
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(str(dest), json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
