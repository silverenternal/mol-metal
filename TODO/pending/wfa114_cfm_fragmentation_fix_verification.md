# Wf-A114 follow-up — run c9 and measure the fragmentation fix (GPU-BLOCKED)

**Wave:** Wf-A114 (follow-up)
**Created:** 2026-10-30
**Status:** ⏸️ **BLOCKED — GPU authorisation withheld by explicit user directive**
(*"现在不要运行任何gpu相关的实验"*, 2026-10-30). This item exists so the fix is
not forgotten; it is not a work-in-progress.
**Predecessor:** `TODO/completed/wfa114_cfm_fragmentation_audit.md`

---

## §1 What is blocked, and what is not

**SHIPPED (CPU-only, no GPU needed):**

  * the read-only fragmentation audit and its report —
    `docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md`;
  * three measurement instruments — `cfm_fragmentation_gate.py`,
    `cfm_fragmentation_audit.py`, and the best-epoch view in
    `wf_cfm_run_census.py`;
  * the trainer's `n_frags_raw` sentinel fix (an unparseable SMILES now writes
    `-1` = UNKNOWN instead of claiming one connected molecule);
  * `configs/c9_cfm_transport_fixed.yaml` — the fix config, with
    `early_stop_metric: molecule_validity`;
  * **the head-lever recording (§4.2)** — the trainer's `config` and
    `provenance` blocks now carry **all six** head levers, and the census
    publishes the merged view as `head_levers`; only the **56 historical nulls**
    remain unrecoverable.

**BLOCKED (needs a GPU slot):** running c9 and measuring whether it produces
connected molecules.

## §2 The protocol to run when a GPU slot is authorised

1. **Pre-flight (AGENTS.md §1.4, mandatory):**
   `.venv/bin/python molmetal/scripts/pre_flight_check.py` — 25/25 PASS, and
   the report must not be STALE.
2. **Launch (AGENTS.md §5.11 — YAML-only, no CLI hyperparameters):**
   `bash scripts/start.sh configs/c9_cfm_transport_fixed.yaml`
   The wrapper may pass only `--config` / `--out-dir` / `--ckpt-dir` / `--help`.
3. **Score:**
   * `.venv/bin/python molmetal/scripts/cfm_fragmentation_gate.py --smiles-csv <the run's generated_smiles.csv>`
   * `.venv/bin/python molmetal/scripts/cfm_fragmentation_audit.py`
   * `.venv/bin/python molmetal/scripts/wf_cfm_run_census.py`

## §3 The bar (MEASURED baselines, from the census artifact)

**⚠️ The previous bar in this section — 0.572276953125 — was WRONG and is
RETRACTED (`docs/HONEST_NEGATIVE_CATALOG.md` §6.200).** It was the end-to-end
rate of `cfm_r11_eval256`, a run that recorded **ONE** epoch, resumed at epoch
19 of a declared 20, with `train_loss = nan`. It was also internally
inconsistent: defined as a *best*-epoch number and applied as a *final*-epoch
threshold. The **"3.2x n_train inversion"** that justified the old table is
also retracted — it is a best-epoch-only artefact, and the ordering reverses
on the last-epoch reading and then **alternates** under budget matching. The
full-precision JSON value of the withdrawn cell is `0.572265625`; the
`0.572276953125` form is the 4-decimal CSV rendering.

**The bar is a NAMED RUN, compared on a matching reading, cohort and budget.**
c9 runs at **n_train = 26715**. Do not use 0.9375 as c9's pass/fail line: it
belongs to an **n_train = 256** run, and the cohort ordering is
**UNRESOLVED** — it flips with the reading and with the epoch budget, and it
flips with which population is quoted. MEASURED
(`metrics/by_metric/cfm_run_census.json` → `by_n_train`):

| reading | population | 26715 median (n) | 256 median (n) | leader | ratio |
|---|---|---|---|---|---|
| last epoch | CLEAN only | **0.125** (11) | ~~**0.026041666666666664**~~ **0.03645833333333333** (6) | **26715** | ~~**4.80×**~~ **3.43×** |
| last epoch | ALL runs (loose) | **0.12357954545454546** (14) | ~~**0.026041666666666664**~~ **0.03645833333333333** (6) | **26715** | ~~**4.75×**~~ **3.39×** |
| best epoch | CLEAN only | **0.35546875** (11) | **0.7708333333333333** (8) | **256** | **2.17×** |
| best epoch | ALL runs (loose) | **0.296875** (14) | **0.7708333333333333** (8) | **256** | **2.60×** |

**The strict (n_train, n_val) cohorts** (`by_cohort`, the axis
`cfm_cohort_matched_compare.py` enforces) read: `26715/192` last-epoch median
**0.19856770833333331** (CLEAN, n=8) / **0.12215909090909091** (ALL, n=9),
best-epoch median **0.365234375** (CLEAN, n=8) / **0.35546875** (ALL, n=9);
`26715/6714` last-epoch median **0.125** (CLEAN, n=3) / **0.125** (ALL, n=5),
best-epoch median **0.125** (CLEAN, n=3) / **0.125** (ALL, n=5); `256/192`
last-epoch median ~~**0.026041666666666664**~~ **0.03645833333333333** (n=6), best-epoch median
**0.7708333333333333** (n=8). **Still no cohort median is a bar.**

And under **budget matching** (the rate at epoch K-1, counting only runs that
actually REACHED epoch K-1) the winner **alternates**: the ordering flips at
**~~K = 9, 12, 14, 18 and 20~~ K = 9, 11 and 25** (`budget_matched.flipped_budgets` — **CHANGED** by Wf-A150; see
catalog §6.262).
No cohort median is a bar.

| Bar | Value | Baseline run | Strict cohort | Reading |
|---|---|---|---|---|
| **PRIMARY** | **> 0.41145833333333337** | **`cfm_r9_fix_v2`** (ep 10) | **26715/6714** | **final epoch** (budget 11 epochs) |
| **SECONDARY** | **> 0.5260416666666666** | **`cfm_fullcorpus_seed1`** (ep 13) | **26715/192** | **best epoch** (budget 14 epochs) |
| **CONTEXT** — project-wide last-epoch median (**27 CLEAN** of 30 runs) | ~~**0.057291666666666664**~~ **0.0625** | — | — | last epoch |
| **CONTEXT** — project-wide last-epoch median (ALL 30 runs) | ~~**0.05989583333333333**~~ **0.06380208333333333** | — | — | last epoch |
| **STRETCH (report, not pass/fail)** | 0.9375 | `cfm_fm_ab_2026_10_09`, `cfm_pcgrad_off_2026_10_15` | **256/192** | best epoch |
| generation-side gate | **10 pools, PASS=0, FAIL=10** | — | — | — |

> **⚠️ CORRECTION (2026-10-30, Wf-A114C).** The previous version of this
> table said *"Both baselines are full-corpus runs (n_train = 26715) … the best
> of their kind among the 10 such runs in the cohort"*, quoted the primary bar
> as `0.411475000000` and the secondary as `0.526017187500`, and quoted the
> project-wide median as `0.04166875` over *"27 runs"*. **All four are wrong
> against the artifact on disk.** (i) The two baselines are **NOT the same
> cohort**: a cohort is the PAIR `(n_train, n_val)`, and the primary bar is
> `26715/6714` while the secondary is `26715/192` — they share `n_train` and
> nothing else, so they are **not** comparable to each other and neither is
> transferable to c9 without naming the cohort. (ii) The bar values are
> `0.41145833333333337` and `0.5260416666666666` — the 4-decimal
> `per_epoch_metrics.csv` rendering, not the JSON's full precision; the
> harness (`cfm_cohort_matched_compare.py`) reads the JSON. (iii) The
> project-wide median is now quoted over the **CLEAN** population
> (contiguous-from-zero epoch history AND no `nan` train_loss): **median
> ~~0.057291666666666664 over 29 clean runs of 32~~ **0.0625 over 27 clean runs of 30**, of which **~~25 are below
> 0.50~~ 23 are below 0.50** and **~~18 are below 0.10~~ 16 are below 0.10**. The all-runs median is
> ~~`0.05989583333333333` over 32 runs~~ **`0.06380208333333333` over 30 runs**, and **its maximum, `0.572265625`, is the
> WITHDRAWN `cfm_r11_eval256` value**. (**SUPERSEDED 2026-10-31** — this
> sub-clause originally read *median 0.03125 … 28 are below 0.50 and 22 are
> below 0.10 … all-runs median 0.03580729166666667*; the figures moved when the
> census artifact was regenerated — see the dated note at the end of §3.)
> (iv) The cohort ratios "4.69× / 3.24×" were loose-axis numbers that mixed a
> clean run set with a degenerate one; the measured values are in the table
> above. Source: `metrics/by_metric/cfm_run_census.json` (59 runs) and
> `metrics/by_metric/cfm_cohort_matched_bar.json`.

**To count as a win, c9 must clear ALL THREE of the primary set:**

  * end-to-end single-molecule rate **> 0.41145833333333337** on a **final-epoch**
    reading against `cfm_r9_fix_v2` **in the `26715/6714` strict cohort** (and,
    if a best-epoch claim is made, **> 0.5260416666666666** against
    `cfm_fullcorpus_seed1` **in the `26715/192` strict cohort** — on a
    best-epoch reading, naming the run it beat);
  * `largest_fragment_frac_mean > 0.90`;
  * the census gate verdict **PASS** for `cfm_c9_transport_fixed_2026_10_30`.

**Report the stretch number (0.9375) separately.** Clearing it would be a real
result and would also mean the cohort effect was broken; **not** clearing it is
**not** a failure by itself.

**Beating the median (~~0.057291666666666664~~ 0.0625) is NOT a win** — that is the existing
baseline, and **11 of the 29 CLEAN runs** already reach ≥ 0.50 at their best
epoch (`fragmentation_summary.best_epoch_single_molecule_rate.
n_at_or_above_0.50`) while ending at 0.0000-0.5000 at their last epoch.

### §3.1 Do not re-try the connectivity-loss lever

`cfm_frag_assemble` (assemble/connectivity term) vs `cfm_frag_control`: best-epoch
**0.1953125 vs 0.28125**, last-epoch **0.015625 vs 0.03125** (the census JSON's
full-precision values, read from `cfm_run_census.json` → `runs[*].
gen_single_molecule_rate_best` / `gen_single_molecule_rate_end`; the 4-decimal
CSV rendering is 0.195314 / 0.281259 / 0.015625 / 0.031252 — **these numbers
did NOT move** in the 2026-10-31 re-quote) — the term made
fragmentation **worse** on both views (census `already_answers` already records
it as "NULL on bond type"). Both runs sit at `gen_validity` 0.039-0.047, so the
comparison is confounded by low parse rates — but it is **not** an A/B, and it
is the only connectivity-specific loss lever on disk. It was not the fix.

### §3.2 STILL OPEN — the c7 fullgen pool spans TWO populations, and its verdict was deliberately NOT re-baselined

**MEASURED 2026-10-31** (`metrics/by_metric/cfm_fragmentation_audit.json` →
`runs[cfm_c7_fullgen_2026_10_20].pool_population_composition`):

  * the pool holds **12,669 rows** over **epochs 0..17**;
  * the run of record ran **epochs 4..17** — the **2,816 rows of epochs 0..3**
    belong to an earlier, **KILLED PRE-RESUME ATTEMPT** that wrote into the same
    directory (`n_rows_outside_run_of_record = 2816`); the sibling pool
    `cfm_c7_fullgen_ep4_partial_2026_10_20` (2,816 rows, epochs 0..3) is that
    attempt on its own;
  * **the composition is now PUBLISHED** (that is what Wf-A121 added —
    `docs/HONEST_NEGATIVE_CATALOG.md` §6.206).

**What is still OPEN:** the **fragmentation verdict was DELIBERATELY NOT
re-baselined** on the run of record's epochs 4..17. The published verdict for
this pool still mixes the two populations. Re-baselining it is a **change to a
published metric artifact**, and it is **not done**. A c9 comparison must
therefore either (a) re-baseline this pool first, or (b) quote this pool with
its population caveat attached — **the mixed-population verdict is not a c9
baseline**.

### §3.3 Dated note — what moved on 2026-10-31 (Wf-A119/A124 re-quote)

Every figure in §3 was re-derived on 2026-10-31 against
`metrics/by_metric/cfm_run_census.json` and
`metrics/by_metric/cfm_fragmentation_audit.json`. **0 GPU** (read-only JSON +
`git` reads).

| figure | before (2026-10-30 quote) | now (2026-10-31, MEASURED) |
|---|---|---|
| CLEAN last-epoch median | 0.03125 | ~~**0.057291666666666664**~~ **0.0625** |
| ALL-runs last-epoch median | 0.03580729166666667 | ~~**0.05989583333333333**~~ **0.06380208333333333** |
| CLEAN best-epoch median | 0.1953125 | **0.35546875** |
| "≥ 0.50 at best epoch" | 8 of 29 | **11 of 29** |
| below 0.50 / below 0.10 (CLEAN, last epoch) | 28 / 22 | ~~**25 / 18**~~ **23 / 16** |
| generation-side gate | 9 pools, PASS=0, FAIL=9 | **10 pools, PASS=0, FAIL=10** |
| generation-side row totals | 21,379 rows / 21,378 parseable / 1 unparseable | **21,445 rows / 21,444 parseable / 1 unparseable** |
| 26715 last-epoch median (CLEAN / ALL) | 0.09635416666666666 (10) / 0.12215909090909091 (13) | **0.125 (11) / 0.12357954545454546 (14)** |
| 256 last-epoch median (CLEAN / ALL) | 0.020833333333333332 (11) / 0.020833333333333332 (11) | ~~**0.026041666666666664 (8) / 0.026041666666666664 (8)**~~ **0.03645833333333333 (6) / 0.03645833333333333 (6)** |
| 26715 best-epoch median (CLEAN / ALL) | 0.296875 (10) / 0.23828124999999997 (13) | **0.35546875 (11) / 0.296875 (14)** |
| 256 best-epoch median (CLEAN / ALL) | 0.3333333333333333 (11) / 0.3333333333333333 (11) | **0.7708333333333333 (8) / 0.7708333333333333 (8)** |

**Did NOT move (verified 2026-10-31):** the PRIMARY / SECONDARY / STRETCH bars
(0.41145833333333337 / 0.5260416666666666 / 0.9375, unchanged
`per_run_bar`); the `cfm_frag_assemble` vs `cfm_frag_control` lever A/B; the
`cfm_r11_eval256` WITHDRAWN maximum **0.572265625**; and the **UNRESOLVED**
verdict on the `n_train` axis (the ordering still flips with the reading and
with the epoch budget).

**Did move (2026-10-31, Wf-A150):** `budget_matched.flipped_budgets` — it read
~~**[9, 12, 14, 18, 20]**~~ (the "still" above) and is now **[9, 11, 25]** (the
census's empty-denominator republication moved the two collapsing runs out of
the end-to-end distribution, and the `clean_only.best_epoch` population was
corrected to the one its own `predicate` names — catalog §6.262).

**Why the medians moved:** `cfm_run_census.json` was **regenerated in the
working tree** after §6.201 was written. **MEASURED:** the census's own
`changelog` block records **only `wf_a115`** (the recursive loader
`REPORTS.rglob` + the full-precision JSON series), so the artifact does **not**
name the later regeneration; and **both artifacts are UNCOMMITTED** (`git
status --porcelain` → ` M metrics/by_metric/cfm_run_census.json`, `??
metrics/by_metric/cfm_fragmentation_audit.json` — the audit artifact has **no
commit at all**). The waves evidenced as having touched these two artifacts
since this TODO was written: **Wf-A115** (census loader/precision), **Wf-A117/
Wf-A118** (§6.202, the pool-kind inversion that took the generation side from
9 pools to 10), **Wf-A120** (§6.204/§6.205, the UNIFORM EPOCH RULE), **Wf-A121**
(§6.206, the population/cohort owners), **Wf-A122** (§6.207, the audit's
honesty fields) and **Wf-A123** (§6.210/§6.211/§6.213, the registry re-point
and the false ledger provenance).

## §4 The 2×2 that would attribute it (do this, or the result is unattributable)

c9 differs from c8 in the coordinate-head bundle **and** the early-stop
criterion. A c9-vs-c8 comparison confounds them. Run the other arm:

| | `bond_type_f1` (c8's gate) | `molecule_validity` (c9's gate) |
|---|---|---|
| **c8's head bundle** (radial) | c8 — already on disk | *(missing)* |
| **c9's head bundle** (`vector_channels`) | *(missing)* | c9 |

The two missing cells are two additional runs of the same protocol. Without
them, a c9 win cannot be attributed to the head bundle or to the early-stop.

### §4.1 The head axis is a FIVE-lever bundle, not one variable

**MEASURED** by resolving both configs through
`parse_args(["--config", cfg])` (249 resolved args each): c8 and c9 differ in
**six** resolved args, not two.

| arg | c8 | c9 | axis |
|---|---|---|---|
| `velocity_head` | `radial` | `vector_channels` | head |
| `velocity_head_init` | `zero` | `lsq` | head |
| `vector_channel_norm_gain` | `learnable` | `fixed` | head |
| `t_embedding_dim` | `0` | `32` | head |
| `fm_batch_chunk_ratio` | `0.0` | `2.0` | head (batching) |
| `early_stop_metric` | `bond_type_f1` | `molecule_validity` | early-stop |

(`velocity_head_init_knots` is `9` in both — not a confound.) So the 2×2
attributes to the **bundle**, not to "the head"; a reader must not describe the
result as "the vector_channels head won".

### §4.2 The head levers ARE observable from a finished run — the recording is DONE; only the historical nulls remain

**✅ THE ACTION IS DONE (MEASURED 2026-10-31).** The requirement — *"Record the
levers in the run's `config` block and in the census's `CFG_KEYS` **before the
GPU run**"* — is **SATISFIED on disk**. It is **no longer an open to-do**:

  * the trainer's **`config` block** (the one written into
    `per_epoch_metrics.json`) now carries **ALL SIX** levers —
    `velocity_head` (`:20622`), `vector_channels` (`:20625`),
    `velocity_head_init` (`:20626`), `velocity_head_init_knots` (`:20629`),
    `vector_channel_norm_gain` (`:20632`), `t_embedding_dim` (`:20635`) in
    `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wf-A114, 2026-10-30);
  * the trainer's **`provenance` block** carries the same six at
    `train_cfm_yuelbond_wasserstein.py:20469-20486`;
  * the census reads a **MERGED** view — **`config` first, then `provenance`** —
    through `HEAD_LEVER_KEYS` (`molmetal/scripts/wf_cfm_run_census.py:194`,
    applied in `_head_levers` at `:1398-1428`), and publishes it per run as
    `head_levers` / `head_levers_source` / `head_levers_source_per_key`. **A
    lever that is in NEITHER block stays `null` — "NOT RECORDED", never
    back-filled with the argparse default, because a default is not a
    measurement.**

**What remains is ONLY the HISTORICAL nulls.** MEASURED over the census's
**59 runs**: **3 carry a non-null lever** —

  * `wf385a3_scaffold_wire_in/train` — **all six**, every one of them from the
    `config` block;
  * `wfa113_t42_bond_ce/wfa113_t42` and `wfa113_t42_bond_ce/wfa113_t42_lr0` —
    **five of six**; `velocity_head_init_knots` is `null` for both (it is in
    neither their `config` nor their `provenance` block), and
    `velocity_head_init` / `vector_channel_norm_gain` / `t_embedding_dim` come
    from `provenance` rather than `config`.

The other **56 of 59** census runs record **none** of the six: they **PREDATE**
the change, and their nulls **cannot be recovered**. **So the 2×2 is auditable
for any run launched from now on, and is NOT auditable for the 56 historical
runs.**

`velocity_head_init_knots` is **live** (dest at
`train_cfm_yuelbond_wasserstein.py:14870`, consumed at `:16060` →
`model._lsq_knots`, read at `:7285`) and `configs/` on disk already use both
`9` and `5` for it — so two runs differing **only** in that number are still
**indistinguishable** if both are among the 56 historical nulls, and
**distinguishable** for any new run.

## §5 What must be written when it runs

  * the run's `per_epoch_metrics.csv` must carry the **`molecule_validity`**
    column — this is blind spot 3. **MEASURED 2026-10-30: it is no longer
    true that no CSV on disk carries it.** The column exists in
    `molmetal/reports/wf385a3_scaffold_wire_in/train/per_epoch_metrics.csv`
    (and its `results/gate_scratch/` copy), because that run was regenerated
    **after** the trainer's column contract shipped. It is **1 run of 59** —
    every other run on disk predates the column. The requirement therefore
    stands, but the reason changes from *"the column does not exist yet"* to
    *"the column exists and must not be dropped"*: a c9 CSV that silently
    lacks it would be indistinguishable from a pre-fix run.
  * the run's `per_epoch_metrics.json` `config` block must record **all six
    head levers** (§4.2), so the census can name the arm — **✅ this is now
    automatic and is no longer a code change**: the trainer's `config` block
    writes all six (`train_cfm_yuelbond_wasserstein.py:20622-20635`) and the
    census's merged view (`config` first, then `provenance`) publishes them as
    `head_levers`. It remains a **check on the new run's artifact**, not
    something to implement;
  * a **§6.201** honest-negative entry (or an update to §6.200) recording the
    measured end-to-end rate at the final epoch, the best-epoch rate, the
    **named baseline run it was compared against**, and whether the bar was
    met;
  * `docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md` §6 ("what is still
    unmeasured") updated to close the items the run settles;
  * the fix's status changes from **UNRUN** to **MEASURED** in the closeout file
    and in `TODO/INDEX.md`.

## §6 Constraint reminders

  * §1.4 pre-flight PASS before any launch; the STALE flag (24 hr) blocks PASS.
  * §5.11 YAML-only launch — `--extra-flags` is **refused** by `scripts/start.sh`.
  * §5.2 every number in the write-up traces to a JSON/report path on disk,
    with MEASURED vs PROJECTED tagged.
  * §5.3 zero touches to `paper/*` without explicit user authorisation.
  * §5.4 **no wet-lab** — nothing in this item touches wet-lab.
