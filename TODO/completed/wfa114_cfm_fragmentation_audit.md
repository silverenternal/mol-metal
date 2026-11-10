# Wf-A114 — CFM 碎片化只读审计 + 三盲点修复 (CPU-only, 0 GPU)

**Wave:** Wf-A114
**Date:** 2026-10-30
**Status:** ✅ SHIPPED (audit + instruments + config + trainer fix). ❌ **The fix is
UNRUN** — no c9 run exists; the verification is a separate pending item.
**User directive (verbatim):** *"现在不要运行任何gpu相关的实验，搞清楚这个项目里的cfm模型
生成的分子为什么在多数情况下都是碎片，做只读校验审计再做对应的修复，顺着数据链路一路查下来并修复"*

**GPU used:** 0. Everything in this wave is CPU-only: RDKit re-scoring of SMILES
already on disk, JSON re-reads, one 500-step CPU capacity A/B, YAML round-trips,
unit tests. No training, no GPU sampling, no sweep.

---

## §1 The question and the answer

**Why does the CFM model emit fragments?** The chain, link by link:

| Link | Verdict | Evidence |
|---|---|---|
| Data | ✅ sound | reconstruction pools on *real* tmQM geometry: single-fragment **0.7500** (`cfm_fullcorpus_b64_2026_10_11`) / **0.6744** (`cfm_objective_terms_2026_10_11`) — **2** reconstruction pools. (The third figure, **0.9697**, is `cfm_fullcorpus_2026_10_10`, a **GENERATION** pool — it left the reconstruction list; see §6.202.) |
| Decoder | ✅ exonerated | c8 on *real* validation clouds: `decode_single_fragment_ratio` 0.953125 (ep0) / 0.9739583333333334 (ep1) / 0.9375 (ep2) — `molmetal/reports/cfm_c8_fullgen_2026_10_24/events.jsonl` (re-verified 2026-10-31, **UNCHANGED**) |
| Field | ❌ **the defect** | the trained field's direction cosine at `t = 0.05` is **-0.04** against a target endpoint cosine of **+0.736** (§6.187); the sampler integrates from `t = 0` upward, so the first ~10% of every trajectory is where the field is worse than a constant |
| Metric | ❌ **blind** | `gen_validity` is an RDKit **parse rate**; `C.C.C` parses. **6 pools carry the blind-validity signature** (`gen_validity` = 1.0000 at the pool's own epoch while the pool is fragmented) — of those, **4** also carry `gen_validity_end` = 1.0000 and **2** read `null` at their last epoch (`cfm_fragmentation_audit.json` → `blind_validity_signature`, n = 6) |

**Generation side, project-wide: 10 generation pools, PASS=0, FAIL=10.**

**The second answer — the metric was reading the wrong epoch.** Over the SSOT's
**59 runs** (~~32~~ **30** carry both columns; ~~**29 of those are CLEAN**~~ **27 of those are CLEAN** — contiguous-from-
zero epoch history AND no `nan` train_loss), the **last-epoch** end-to-end rate
has median ~~**0.057291666666666664**~~ **0.0625** over the CLEAN population
(~~**25 of 29 below 0.50**~~ **23 of 27 below 0.50**), and ~~**0.05989583333333333**~~ **0.06380208333333333** over all 30 runs (~~27 of 32 below 0.50~~ **25 of 30 below 0.50**; its
maximum **0.572265625** is the WITHDRAWN `cfm_r11_eval256` value — the
`0.572276953125` form is the 4-decimal CSV rendering). The **best-epoch** rate
has median **0.35546875** (CLEAN) and max **0.9375**, with **11 of 29 at or above
0.50**. Two runs **collapse at their last epoch** to `gen_validity_end == 0.0`
while their best epoch reads 0.8750 / 0.6875.

> **⚠️ RE-QUOTED (2026-10-30, Wf-A114C).** This paragraph originally read
> *"Over the 27 SSOT runs … last-epoch median 0.04166875 (26 of 27 below 0.50)
> … best-epoch median 0.23827330729166665 and max 0.9374541666666667, with 8 of
> 27 at or above 0.50"*. Those figures were measured over the run set as it
> stood during the Wf-A114 audit and do **not** reproduce against the disk as
> of 2026-10-30: the census now loads **59 runs** (42 depth-1 + 17 nested, after
> Wf-A115 widened the loader to `REPORTS.rglob`), it separates the **CLEAN**
> population from the degenerate one, and it reads the per-epoch series from
> `per_epoch_metrics.json` (full float precision) rather than the 4-decimal
> CSV. Source: `metrics/by_metric/cfm_run_census.json` →
> `fragmentation_summary`; full re-quote in `docs/HONEST_NEGATIVE_CATALOG.md`
> §6.201.

> **⚠️ RE-QUOTED (2026-10-31, Wf-A119/A124).** The figures above were
> re-derived on 2026-10-31 against the two artifacts on disk
> (`metrics/by_metric/cfm_run_census.json`, `metrics/by_metric/cfm_fragmentation_audit.json`).
> **What moved, and why:**
>
>   * **The pool census moved 9 → 10 generation pools and 21,379 → 21,445
>     rows** because **one pool was reclassified**, not because a pool was
>     created: `cfm_fullcorpus_2026_10_10` (66 rows) is an **era-B
>     free-generation** pool (its header carries no `metal_z`, `sanitized == 0`)
>     and was filed as a reconstruction until **Wf-A117/Wf-A118** reclassified
>     it — recorded in `docs/HONEST_NEGATIVE_CATALOG.md` **§6.202**, whose banner
>     already carries the corrected 10 / FAIL=10 and 2 / PASS=1 / FAIL=1
>     counts. **This file now carries the same correction.**
>   * **The census medians moved** (CLEAN last epoch 0.03125 →
>     **0.057291666666666664**; ALL-runs 0.03580729166666667 →
>     **0.05989583333333333**; CLEAN best epoch 0.1953125 →
>     **0.35546875**; "8 of 29 at or above 0.50" → **11 of 29**), because
>     `cfm_run_census.json` was **regenerated in the working tree** after
>     §6.201 was written. **MEASURED:** the census's own `changelog` block
>     records **only `wf_a115`** (the recursive loader `REPORTS.rglob` + the
>     full-precision JSON series), and the artifact **does not name** the
>     later regeneration; **both artifacts are UNCOMMITTED** (`git status
>     --porcelain` → ` M metrics/by_metric/cfm_run_census.json`, `??
>     metrics/by_metric/cfm_fragmentation_audit.json` — the audit artifact has
>     **no commit at all**). The waves that are evidenced as having touched
>     these two artifacts since this file was written are **Wf-A115** (census
>     loader/precision), **Wf-A117/Wf-A118** (§6.202 pool-kind inversion),
>     **Wf-A120** (§6.204/§6.205, the UNIFORM EPOCH RULE), **Wf-A121**
>     (§6.206, the population/cohort owners), **Wf-A122** (§6.207, the audit's
>     honesty fields) and **Wf-A123** (§6.210/§6.211/§6.213, the registry
>     re-point and the false ledger provenance).
>   * **What did NOT move:** the decoder exoneration row (c8's
>     `decode_single_fragment_ratio` 0.953125 / 0.9739583333333334 / 0.9375,
>     re-read from `events.jsonl`); the per-run **PRIMARY / SECONDARY /
>     STRETCH** bars (0.41145833333333337 / 0.5260416666666666 / 0.9375);
>     `budget_matched.flipped_budgets` (~~still [9, 12, 14, 18, 20]~~ **now [9, 11, 25]**); the
>     `cfm_frag_assemble` vs `cfm_frag_control` A/B; and the **UNRESOLVED**
>     verdict on the `n_train` axis.
>
> **0 GPU** was used by this re-quote (read-only JSON + `git` reads).
> **⚠️ SUPERSEDED (2026-10-31, Wf-A150).** The census re-published the
> **empty-denominator** chemistry rate as `null` instead of the trainer's `0.0`
> placeholder — 66 of 576 per-epoch rows over 5 runs carry
> `gen_validity_val == 0` with `gen_validity_n > 0`, and on **2** of them it is
> the **LAST** row (`cfm_fm_control_2026_10_09`,
> `cfm_bond_head_30ep_onecycle_2026_10_09`), so those two runs leave the
> end-to-end distribution. MEASURED 2026-10-31 from
> `metrics/by_metric/cfm_run_census.json` -> `fragmentation_summary`, the
> CURRENT values are: CLEAN last-epoch median **0.0625** (n=**27**, mean
> 0.17755582555110858, max 0.5572916666666666, **23 of 27** below 0.50,
> **16** below 0.10); CLEAN best-epoch median **0.35546875** (n=29,
> **11 of 29** at or above 0.50); ALL-runs last-epoch median
> **0.06380208333333333** (n=**30**, mean 0.18361505123463406, max
> 0.572265625 — WITHDRAWN, **25 of 30** below 0.50, **17** below 0.10);
> ALL-runs best-epoch median **0.34440104166666663** (n=32, **12 of 32** at
> or above 0.50); the budget-matched flip set is **K = 9, 11 and 25**; the
> gate verdicts are PASS/FAIL/SKIP **5 / 25 / 29** over 59 runs and
> **4 / 23 / 28** over the 55 CLEAN runs. The superseded figures above are
> **PRESERVED VERBATIM** as history (AGENTS.md §5.5 rule #6); the finding of
> record is catalog **§6.262**.

So the model *can* hold a molecule together; the training oscillates violently
and the early-stopping criterion cannot see the axis that went unstable.

## §2 The three blind spots (all fixed)

1. **The reported metric.** `gen_validity` is a parse rate; `gen_single_fragment_ratio`
   is over *parsed rows only*. The honest end-to-end denominator is
   `gen_validity × gen_single_fragment_ratio`. The gate, the audit and the census
   now report the end-to-end rate **and name their basis**
   (`pool_end_to_end` / `trainer_columns_derived` / `trainer_conditional_column`).
2. **The early-stop criterion.** **50** configs on disk select an early-stop
   metric: **46** set `early_stop_metric: bond_type_f1`, **3** set
   `bond_type_f1_eps`, **1** sets `molecule_validity`. The first 49 are blind to
   connectivity. The strict `molecule_validity` was added to
   `EARLY_STOP_METRICS` on 2026-10-27 (commit `27777532`);
   `configs/c9_cfm_transport_fixed.yaml` is the only config that selects it.
3. **The measurement itself.** At audit time the newest
   `per_epoch_metrics.csv` on disk was dated 2026-10-21, six days **before**
   `molecule_validity` existed, so **no CSV on disk carried the column** — the
   strict criterion could not have been read even if a config had selected it.
   **MEASURED 2026-10-30 (Wf-A114C): the column now exists in exactly 1 run of
   59** (`molmetal/reports/wf385a3_scaffold_wire_in/train/per_epoch_metrics.csv`,
   regenerated after the trainer's column contract shipped) — the blind spot is
   **narrowed, not closed**.

## §3 The trainer bug (fixed)

`train_cfm_yuelbond_wasserstein.py` wrote `n_frags_raw = 1` and
`largest_frag_frac = 1.0` for **any SMILES RDKit cannot parse** — the row
claimed to be one connected molecule holding all its atoms.

* **MEASURED impact:** 1 row of **21,445** in the generation pools on disk
  (`molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`); the **10**
  generation pools hold **21,445 rows, 21,444 parseable, 1 unparseable**
  (MEASURED over `metrics/by_metric/cfm_fragmentation_audit.json` → `runs[*]`;
  the pool set grew from 9 to 10 by the §6.202 reclassification of
  `cfm_fullcorpus_2026_10_10`, +66 rows).
* **FIX:** the counts now come from `decode_chemistry` — the same function the
  audit scores the pool with — and an unparseable SMILES writes the sentinel
  **`-1` = UNKNOWN**. MEASURED bit-exact for every parseable row: the audit's
  own column cross-check covers **9 of the 10** generation pools — **21,378
  rows, 0 mismatches** (`csv_column_cross_check`, `n_checked` summed over
  `runs[*]`) — and over those rows the old inline logic and `decode_chemistry`
  agree on `n_fragments` (**0 differences**) and `largest_fragment_frac` (**0
  differences**). **The 66-row era-B pool `cfm_fullcorpus_2026_10_10` is NOT
  covered by that receipt** (`csv_column_cross_check.n_checked = 0` — it has no
  trainer-written `n_frags_raw` column to compare against).

## §4 The config (UNRUN)

`configs/c9_cfm_transport_fixed.yaml` — c8's protocol held fixed (n_train 26715,
batch 64, 18 epochs, lr 3e-4, `fm_noise_sigma` 2.485, scaffold split) plus:

| key | value | why |
|---|---|---|
| `velocity_head` | `vector_channels` | a learned per-atom 3-vector head, not a radial scalar field |
| `vector_channels` | 8 | C = 8 learned 3-vectors per atom |
| `velocity_head_init` | `lsq` | least-squares gate init against the training targets |
| `velocity_head_init_knots` | 9 | knot count for the LSQ fit |
| `vector_channel_norm_gain` | `fixed` | the field's magnitude is not trainable-away |
| `t_embedding_dim` | 32 | the head is conditioned on `t` |
| `fm_batch_chunk_ratio` | 2.0 | chunked FM batch (only effective because `_fm_padding_invariant()` is True under `vector_channels`) |
| `early_stop_metric` | `molecule_validity` | blind spot 2 — gates on connectivity |

**No run on disk ever used these levers.** `--velocity-head` exists only from
2026-10-22 (`540cb8cd`), `--velocity-head-init` only from 2026-10-25
(`3ceb008e`), **no run's `per_epoch_metrics.json` records any head setting**,
c8's config declines `vector_channels` over the 32 GiB cap, and every other
`vector_channels` config on disk is smoke-scale (n_train 32 / 64 / 200 / 1200 /
1600) except c9.

## §5 The CPU A/B — direction, not effect size

`metrics/by_metric/cfm_c9_head_ab_capacity.json` — **`device: cpu`,
`gpu_used: false`**, 500 steps, 8 held-out clouds, 464 atoms, same cohort and
seed, both fields fresh:

| | arm A (radial, c8's head) | arm B (c9) |
|---|---|---|
| gap closed | 0.026603877033031947 | **0.03661264969945691** |
| direction cosine | 0.1685085892677307 | **0.19569902122020721** |
| mean \|v\| | 0.6298840641975403 | **0.7988280653953552** |
| verdict | CANNOT_FIT | **CANNOT_FIT** |

against a `mean_target_norm` of **6.028980731964111** — neither arm comes close
to the target's scale, so both are `CANNOT_FIT` and neither number is an effect
size. The LSQ init's real win is at **step 0**, where the fit is analytic: at
`t = 0.02` the ratio to the zero baseline is **0.570944** and the cosine
**0.573981**, against the trained c8 field's measured **-0.04** direction at
`t = 0.05` (§6.187). Whether it survives 18 epochs is **not measured**.

## §6 Honest negatives (ACTIVE)

Full text: `docs/HONEST_NEGATIVE_CATALOG.md` §6.199 (net +1, cumulative 253 →
254, ACTIVE 209 → 210) **and §6.200** (the retraction of two claims this file
originally published — net +1, cumulative 254 → 255, ACTIVE 210 → 211) **and
§6.201** (the re-quote of every figure in this file against the 59-run census —
net +1, cumulative 255 → 256, ACTIVE 211 → 212). The load-bearing ones:

  * The fix is **UNRUN**; `molmetal/reports/cfm_c9_transport_fixed_2026_10_30/`
    does not exist on disk.
  * **c9 differs from c8 in TWO ways** (head + early-stop), so a c9-vs-c8
    comparison is not a single-variable A/B. The separating design is 2×2.
  * The best-epoch number is **optimistically biased** (a max over a noisy
    epoch series); quote it together with the last-epoch view or not at all.
  * **RETRACTED (§6.200): the "n_train trend runs backwards / 3.2× inversion"
    claim and the 0.572276953125 bar.** The inversion was a **best-epoch-only**
    artefact: on the **last-epoch** reading the ordering REVERSES (CLEAN
    population: n_train 26715 median ~~**0.09635416666666666**~~ **0.125**, n=11 vs n_train
    256 median ~~**0.020833333333333332**~~ **0.03645833333333333**, n=6 — full corpus ahead ~~4.63×~~ **3.43×**), and
    under **budget matching** (rate AT epoch K-1, counting only runs that
    REACHED epoch K-1) the winner **alternates** — the ordering flips at
    **~~K = 9, 12, 14, 18 and 20~~ K = 9, 11 and 25**. So the n_train axis is **UNRESOLVED**, not a
    trend. The 0.572276953125 bar was the rate of `cfm_r11_eval256`, a run with
    **ONE** recorded epoch (resumed at epoch 19 of a declared 20,
    `train_loss = nan`), and was defined as a best-epoch number but applied as a
    final-epoch threshold. **Replacement bar: a NAMED run** — final epoch
    **> 0.41145833333333337** vs `cfm_r9_fix_v2` (strict cohort **26715/6714**);
    best epoch **> 0.5260416666666666** vs `cfm_fullcorpus_seed1` (strict cohort
    **26715/192**). **The two bars are NOT the same cohort** — a cohort is the
    PAIR `(n_train, n_val)`, so they are not comparable to each other. The
    census's own `already_answers` records a
    *different* ordering on the bond-head axis (full corpus typeF1 0.9470 vs
    the 256-cohort canonical 0.3770), and `cfm_r9_fix_v2` — the run the census
    names best-on-disk for type-F1 — has a best-epoch fragmentation rate of
    ~~**0.473976562500**~~ **0.4739583333333333**. **Consequence: 0.9375 remains a 256-cohort number, not
    a transferable bar for a full-corpus run** — and neither is any cohort
    median.
  * **The obvious fix was already tried and did not work.**
    `cfm_frag_assemble` vs `cfm_frag_control`: best-epoch **0.1953125** vs
    **0.28125**, last-epoch **0.015625** vs **0.03125** (the census JSON's
    full-precision values; the 4-decimal CSV rendering is 0.195314 / 0.281259 /
    0.015625 / 0.031252 — **these numbers did NOT move** in the 2026-10-31
    re-quote) — the connectivity term made fragmentation **worse** on both
    views (not an A/B; both sit at `gen_validity` 0.039-0.047). The fix could
    not be a new loss term.
  * **c8's config claims 18 epochs; the run reached 3** (CUDA OOM in epoch 3's
    backward pass, `master.log:257`), leaving no `per_epoch_metrics.csv` —
    which is why c8 is **absent from the 59-run census**. (The census's
    `not_counted` field already states this explicitly, so it is a
    documented scope limit, not an unnoticed omission.)
  * c8's config carries a §5.11 dead key: `protocol.scheduler.warmup_epochs: 5`
    (`configs/c8_fullgen_b64_2026_10_24.yaml:740`) aliases `train.lr_warmup_epochs`
    (`:323`). c9 does not carry it.
  * **Two mistakes this audit made and corrected** (recorded so they are not
    repeated): the first draft averaged the two pool pipelines into one mean
    (0.3310) and one verdict; the gate initially aligned the trainer's
    `n_frags_raw` column positionally and reported **2299 false mismatches**.
    Post-fix cross-check: **9 of the 10** generation pools (**21,378 rows**),
    **0 mismatches** (the 10th, era-B `cfm_fullcorpus_2026_10_10`, has no
    trainer-written column to check — `n_checked = 0`).
  * **No pool is refused any more** (`cfm_fragmentation_audit.json` →
    `summary.n_runs_refused = 0`, `refused_runs = []`).
    `cfm_ode_link_probe` (12 rows against 1536 samples drawn) is **scored**,
    but its **end-to-end rate reads `null`**, so the generation-side end-to-end
    stats have **n=9 of the 10** pools (`summary.generation_side.
    single_molecule_rate_end_to_end.n = 9`).
  * **One pre-existing test failure, not caused by this wave.**
    `molmetal/tests/test_cfm_next_action_resolver.py::test_settled_mcnemar_moves_perf_out_of_the_dead_branch`
    fails identically with this wave's changes stashed.

## §7 Verification

| Check | Result |
|---|---|
| `pytest molmetal/tests/test_wfa114_cfm_fragmentation_gate.py molmetal/tests/test_wfa114_c9_config_binding.py -q` | **53 passed** |
| six related regression files (`test_c7_fullgen_pool_quality.py`, `test_c7_scale_matched_pools.py`, `test_decode_largest_fragment.py`, `test_cfm_yuelbond.py`, `test_wfa113_m11_sampled_output_gate.py`, `test_wfa113_connectivity_prior_zero_bonds.py`) | **79 passed** |
| broad sweep `pytest molmetal/tests/ -q -x -k "cfm or yuelbond or fragment or c7_ or decode or pool"` | 335 passed, 6 skipped, **1 failed (pre-existing, proven by stash)** |
| §5.11 YAML round-trip on `configs/c9_cfm_transport_fixed.yaml` | strict mode raises nothing; every head lever lands with the intended value **and type** |

## §8 Files (§5.5 filesystem-truth, `ls -la` 2026-10-30)

| Path | Bytes | Status |
|---|---|---|
| `docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md` | 23967 | NEW |
| `docs/HONEST_NEGATIVE_CATALOG.md` | 1431089 | MODIFIED (§6.199 appended) |
| `molmetal/scripts/cfm_fragmentation_gate.py` | 24232 | NEW |
| `molmetal/scripts/cfm_fragmentation_audit.py` | 25575 | NEW |
| `molmetal/scripts/wf_cfm_run_census.py` | 25963 | MODIFIED (best-epoch view) |
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | 1010394 | MODIFIED (`n_frags_raw` sentinel) |
| `configs/c9_cfm_transport_fixed.yaml` | 22224 | NEW (UNRUN) |
| `molmetal/tests/test_wfa114_cfm_fragmentation_gate.py` | 21983 | NEW |
| `molmetal/tests/test_wfa114_c9_config_binding.py` | 6908 | NEW |
| `metrics/by_metric/cfm_run_census.json` | 74253 | REGENERATED (42 runs) |
| `metrics/by_metric/cfm_fragmentation_audit.json` | 48175 | REGENERATED (13 pools) |
| `metrics/by_metric/cfm_c9_head_ab_capacity.json` | 7625 | NEW (CPU A/B) |

**Addendum — `ls -la` re-read 2026-10-31 (Wf-A119/A124 re-quote).** The table
above is the **2026-10-30 receipt** and is left exactly as it was taken. The two
regenerated artifacts have since moved on disk, and **both are UNCOMMITTED**:

| Path | Bytes (2026-10-31) | Status |
|---|---|---|
| `metrics/by_metric/cfm_run_census.json` | **548654** | REGENERATED (**59 runs**) — working tree only (`git status` → ` M`); last commit touching it is `ee7a987a` |
| `metrics/by_metric/cfm_fragmentation_audit.json` | **163537** | REGENERATED (**13 pools**, 10 generation / 2 reconstruction / 1 unknown) — **never committed** (`git status` → `??`) |

## §9 Cross-refs

  * `docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md` — the full report.
  * `docs/HONEST_NEGATIVE_CATALOG.md` §6.187 (the field's low-`t` defect),
    §6.199 (this wave's entry), §6.200/§6.201 (the retraction + the 59-run
    re-quote), **§6.202 (the pool-kind inversion that makes the generation
    side 10 pools)**, §6.204/§6.205 (Wf-A120, the UNIFORM EPOCH RULE),
    §6.206/§6.207 (Wf-A121/Wf-A122, the population/cohort/cross-check
    owners) and §6.211/§6.213 (Wf-A123, the registry re-point and the false
    ledger provenance).
  * `TODO/pending/wfa114_cfm_fragmentation_fix_verification.md` — the GPU
    follow-up this wave hands over.
  * `molmetal/scripts/c7_scale_matched_pools.py:68-95` — the canonical
    pool-header rule that separates the two pipelines.
