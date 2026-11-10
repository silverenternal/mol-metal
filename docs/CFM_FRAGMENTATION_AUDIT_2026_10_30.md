# CFM Fragmentation Audit — why the CFM model emits fragments, and what was fixed

**Date:** 2026-10-30
**Wave:** Wf-A114
**Status:** READ-only audit completed. The fixes are a config change, a trainer
bug fix and three measurement instruments. **The fix itself is NOT MEASURED** —
no GPU experiment was run (§7).
**Discipline:** every number is **MEASURED** and traces to a path on disk.
Nothing here is PROJECTED except where the word appears.

**Standing directive honoured:** *"现在不要运行任何gpu相关的实验"*. Everything in
this wave is CPU-only: RDKit re-scoring of SMILES already on disk, JSON re-reads,
one 500-step CPU capacity A/B, YAML round-trips, and a unit test. No training, no
GPU sampling, no sweep.

> **UPDATE (2026-10-31, Wf-A121C/Wf-A122).** The **"CPU-only" claim above is
> now true rather than merely intended** — but only because the smoke test now
> **forces** it. Wf-A121C/Wf-A122 changed
> `molmetal/tests/test_smoke_flag.py::test_smoke_run_completes_within_budget_on_cpu`
> so that the test was renamed for what it does: it now launches the smoke
> script with `CUDA_VISIBLE_DEVICES` emptied (so torch reports no CUDA) **and**
> the script's explicit `MOLMETAL_SMOKE_DEVICE=cpu` override, and it asserts
> `config.device == "cpu"`. MEASURED wall-clock, from that test's own
> docstring: **63 s** for the subprocess on CPU on this host (2026-10-31), with
> the pytest call phase at **75 s** against a 240 s ceiling. Before Wf-A121C the
> same test — named `…_on_cpu` — launched the script with **no** env override
> and therefore ran **on the GPU** on a GPU host.
>
> ⚠️ **The GPU-free claim does not extend to a FULL-SUITE pytest run.** On
> 2026-10-31 the pytest process held **1444 MiB** of GPU 0 and the smoke child
> **2218 MiB** before it was killed (MEASURED, recorded in
> `molmetal/tests/test_smoke_flag.py`). A full-suite run must be forced onto
> the CPU (`CUDA_VISIBLE_DEVICES=""`); the Wf-A114 instruments' own test files
> in §8 are CPU-only, but running the *whole* suite is not.

---

## §0 TL;DR — the answer

1. **Fragmentation is not a decoder bug and not a data bug.** The same decoder on
   the same *real* validation geometry produces one connected molecule 93.75% of
   the time. On the model's *own* generated geometry it produces one molecule
   0%-25% of the time. The input cloud is the difference, so the defect is in the
   **velocity field**.
2. **The project was reading a metric that cannot see the failure.**
   `gen_validity` is an RDKit *parse* rate, and `C.C.C` parses. Six generation
   pools carry the blind-validity signature — `gen_validity = 1.0000` at their
   own pool epoch while being fragmented.
3. **The honest denominator is one connected molecule per sample drawn**:
   `gen_validity × gen_single_fragment_ratio`. Over the **27 CLEAN** SSOT runs
   carrying both columns (**30** runs carry them; **3 are degenerate** — a
   non-contiguous epoch history or a `nan` train_loss), the **last-epoch**
   median is ~~**0.057291666666666664**~~ **0.0625** (**23 of 27 below 0.50**, 16 of 27 below
   0.10). Over ALL 30 runs the median is ~~**0.05989583333333333**~~ **0.06380208333333333** (25 of 30
   below 0.50) — and its **maximum, 0.572265625, is the WITHDRAWN
   `cfm_r11_eval256` value** (a 1-epoch, resumed, `train_loss = nan` run).
4. **But the model's BEST epoch is a different story, and nobody was looking at
   it.** The **best-epoch** median is **0.35546875** (clean) and the maximum is
   **0.9375**; **11 of the 29 CLEAN runs reach ≥ 0.50 at their best epoch** (12
   of 32 over all runs) — and 2 of them then **collapse to 0.0** at their last
   epoch. The model can hold a molecule together. The training is unstable, and
   the early-stopping criterion cannot see the axis that went unstable.

5. **Generation side, project-wide: 10 generation pools, PASS=0, FAIL=10.**
6. **Three blind spots, three fixes:** the reported metric, the early-stop
   criterion, and the measurement itself. Plus one trainer bug that wrote
   "one molecule" for a SMILES that does not parse.

> **CORRECTION (2026-10-31, Wf-A116/A121/A122) — bullets 2, 3 and 4.**
> * **Bullet 2** read *"Four generation pools carry `gen_validity = 1.0000` while
>   being fragmented"*. The audit artifact now publishes **six**:
>   `metrics/by_metric/cfm_fragmentation_audit.json` →
>   `summary.n_blind_validity_signature` = **6**, and
>   `blind_validity_signature[]` carries six rows. The two that joined are
>   `cfm_c7_fullgen_ep4_partial_2026_10_20` (pool epoch 3) and
>   `cfm_c8_fullgen_2026_10_24` (pool epoch 2) — both publish
>   `gen_validity_at_pool_epoch` = **1.0** with `gen_validity_n` = **704**,
>   because their event streams carry **no `done` event** and are therefore
>   **one** population, not two. Their `fragmented_ratio`s (0.9552556818181818
>   and 0.7495265151515151) both exceed the 0.50 gate, so **their PASS/FAIL
>   verdicts did not change** — the count that moved is the *number of pools
>   carrying the signature*, not the number of failures.
> * **Bullets 3 and 4** previously read
> *"last-epoch median **0.03125** (28 of 29 below 0.50, 22 of 29 below 0.10)"*,
> *"ALL 32 runs the median is **0.03580729166666667** (30 of 32 below 0.50)"*,
> *"best-epoch median **0.1953125** (clean)"* and *"**8 of the 29 CLEAN runs**
> reach ≥ 0.50 … (9 of 32 over all runs)"*. The census made the population
> identity **VALUE-AWARE** and re-published the summary; every one of those
> figures moved. New values, all from
> `metrics/by_metric/cfm_run_census.json` →
> `fragmentation_summary.end_to_end_single_molecule_rate` (**CLEAN**, n = ~~29~~ **27**,
> min 0.0, **median ~~0.057291666666666664~~ 0.0625**, mean ~~0.16531059620275626~~ 0.17755582555110858, **max
> 0.5572916666666666**, `n_below_0.50` = ~~25~~ **23**, `n_below_0.10` = ~~18~~ **16**),
> `…_all_runs` (n = ~~32~~ **30**, **median ~~0.05989583333333333~~ 0.06380208333333333**, max **0.572265625**
> — still the WITHDRAWN `cfm_r11_eval256` value — `n_below_0.50` = ~~27~~ **25**),
> `best_epoch_single_molecule_rate` (**median 0.35546875**, max **0.9375**,
> `n_at_or_above_0.50` = 11) and `…_all_runs` (**median
> 0.34440104166666663**, `n_at_or_above_0.50` = 12). The population SIZE did
> not move (~~29 CLEAN of 32~~ **27 CLEAN of 30**), but its MEMBERSHIP did: `cfm_c7_fullgen_2026_10_20`
> is now **degenerate** (the census's own verdict `class_b_resume_prehistory` —
> its per-epoch file holds epochs **4..17** (14 rows, `epoch_index_start` = 4),
> the run **RESUMED**, and its events segment also carries **4 earlier epochs
> (0..3) from a killed pre-resume attempt** in the same directory, so its
> history is not contiguous from zero), and it leaves the CLEAN headline. The CLEAN
> last-epoch **maximum** therefore moved from 0.5 to **0.5572916666666666**
> (`cfm_round7_f5f6_retrain/run` and its `run_argmax_baseline` twin). The
> **conclusion is unchanged**: the field produces a connected molecule in a
> small minority of samples at the last epoch, and the best epoch is a much
> more generous (and optimistically biased) view of the same runs.

> **⚠️ SUPERSEDED (2026-10-31, Wf-A152).** The "New values" above are the
> Wf-A116/A121 readings; the census was re-published twice since (Wf-A127
> and Wf-A150) and the CURRENT values are: CLEAN last-epoch median
> **0.0625** (n=**27**, mean 0.17755582555110858, max 0.5572916666666666,
> **23 of 27** below 0.50, **16** below 0.10); ALL-runs last-epoch median
> **0.06380208333333333** (n=**30**, **25 of 30** below 0.50); the population
> is **27 CLEAN of 30**. MEASURED 2026-10-31 from
> `metrics/by_metric/cfm_run_census.json` -> `fragmentation_summary`. The
> superseded figures above are **PRESERVED VERBATIM** as history (AGENTS.md
> §5.5 rule #6); the finding of record is catalog **§6.264**.

---

## §1 The question and the instruments

**Question:** why does the CFM model generate fragments in most cases; audit
read-only first, then fix, tracing the data chain end to end.

| Instrument | Path | What it does |
|---|---|---|
| The gate | `molmetal/scripts/cfm_fragmentation_gate.py` | Scores a SMILES pool on an **honest end-to-end denominator** and names its basis |
| The audit | `molmetal/scripts/cfm_fragmentation_audit.py` | Walks every CFM run dir, classifies each pool by **which pipeline produced it**, re-scores it with the trainer's own `decode_chemistry`, joins with the SSOT |
| The census | `molmetal/scripts/wf_cfm_run_census.py` | The SSOT over all 59 CFM runs; now carries last-epoch **and** best-epoch end-to-end rates |

**Artifacts:** `metrics/by_metric/cfm_run_census.json` (59 runs),
`metrics/by_metric/cfm_fragmentation_audit.json` (13 pools),
`metrics/by_metric/cfm_c9_head_ab_capacity.json` (the CPU A/B).

---

## §2 The chain, link by link

### §2.1 Link 1 — the DATA is not the problem

Two of the thirteen pools on disk are **reconstruction** pools (the bond head
runs on *real* tmQM geometry), one is **unknown**, and the other **ten are
generation** pools (the sampler):

| Pool | pipeline | conditional single-fragment ratio | end-to-end single-molecule rate | verdict |
|---|---|---|---|---|
| `cfm_fullcorpus_b64_2026_10_11` | reconstruction | 0.7500 | 0.7500 (n=40) | PASS |
| `cfm_objective_terms_2026_10_11` | reconstruction | 0.6744 | 0.6744 (n=43) | FAIL |
| `wf124_p2_cfm_full` | **unknown** | 0.4468 | — | excluded from both sides |
| `cfm_fullcorpus_2026_10_10` | **generation** | 0.9697 | **0.3333 (n=192)** | FAIL |

Source: `metrics/by_metric/cfm_fragmentation_audit.json` → `runs[*].pipeline`.
**MEASURED.** On real geometry the same decoder and bond head produce a single
connected molecule 67%-75% of the time.

> **⚠️ CORRECTED (2026-10-30, Wf-A114D).** This section originally listed
> **three** reconstruction pools and named `cfm_fullcorpus_2026_10_10` as one,
> with the reason *"the bond head applied to REAL tmQM geometries (never the
> sampler)"*. **That reason was wrong.** The pool's CSV header carries
> `metal_z`, but its CONTENT carries `sanitized = 0` and `metal_z = ''` — the
> signature of the **2026-10-12+ ODE path** (`wf_cfm_decode_dump_smiles.py` →
> `evaluate_generation_validity`), not the pre-2026-10-12 bond-head-on-real-
> geometry path. The header alone does not decide the pipeline.
> `molmetal/scripts/cfm_pool_kind.py` now applies a **content-aware** rule
> (`sanitized == 1` AND `metal_z` non-empty ⇒ reconstruction; `sanitized == 0`
> AND `metal_z` empty ⇒ generation; both or neither ⇒ unknown), and the
> generation-side end-to-end rate for this pool is **0.3333 (64/192)**, not the
> conditional 0.9697. The **conditional** ratio 0.9697 is unchanged — it was
> always the RDKit parse-and-fragment count over the 66 parsed rows.
> Source: `metrics/by_metric/cfm_fragmentation_audit.json` →
> `summary.n_generation_pools` / `n_reconstruction_pools` / `n_unknown_pools`
> = 10 / 2 / 1; per-pool `pool_kind_reason`.

### §2.2 Link 2 — the DECODER is not the problem either

c8 records `decode_single_fragment_ratio` per epoch in
`molmetal/reports/cfm_c8_fullgen_2026_10_24/events.jsonl`:

| epoch | `decode_single_fragment_ratio` | `gen_validity` |
|---|---|---|
| 0 | 0.953125 | 1.0 |
| 1 | 0.9739583333333334 | 1.0 |
| 2 | 0.9375 | 1.0 |

This is measured on the **real validation clouds**. c7's equivalent end-of-run
figure is 0.7135
(`metrics/by_metric/cfm_run_census.json` → `runs[19] ~~`runs["cfm_c7_fullgen_2026_10_20"]`~~`).

**Same decoder, same bond head, same code path.** On real geometry: one molecule
93.75%-97.40% of the time. On the model's own generated geometry: 0%-25%. The
difference is the input cloud → the defect is in the **field**.

### §2.3 Link 3 — the FIELD emits fragmented clouds

Every generation pool on disk, re-scored with the trainer's own `decode_chemistry`:

| run | pool rows | samples drawn | conditional single-fragment | end-to-end rate | verdict |
|---|---|---|---|---|---|
| `cfm_pcgrad_off_2026_10_15` | 1175 | 1440 | 0.5421 | 0.4424 | FAIL |
| `cfm_pcgrad_on_2026_10_14` | 1155 | 1440 | 0.5333 | 0.4278 | FAIL |
| `cfm_fullcorpus_2026_10_10` | 66 | 768 | 0.9697 | 0.3333 (denominator 192) | FAIL |
| `cfm_c8_fullgen_2026_10_24` | 2112 | 2112 | 0.2505 | 0.2505 | FAIL |
| `cfm_c7_fullgen_2026_10_20` | 12669 | 12672 | 0.0851 | 0.0851 | FAIL |
| `cfm_c7_fullgen_ep4_partial_2026_10_20` | 2816 | 2816 | 0.0447 | 0.0447 | FAIL |
| `c7_sixterms_smoke_2026_10_20` | 704 | 704 | 0.0028 | 0.0028 | FAIL |
| `c7_vramprobe_2026_10_20` | 704 | 704 | 0.0028 | 0.0028 | FAIL |
| `c7_fullgen_smoke_2026_10_20` | 32 | 32 | 0.0000 | 0.0000 | FAIL |
| `cfm_ode_link_probe` | 12 | 1536 | 0.0000 | not computable (§4.3) | FAIL |

Source: `metrics/by_metric/cfm_fragmentation_audit.json` → `runs[*].pipeline ==
"generation"`. **MEASURED.** PASS = 0, FAIL = 10. Median conditional
single-fragment ratio on the generation side: **0.0649203119169274**.

> **CORRECTION (2026-10-31, Wf-A116).** This table listed **nine** generation
> pools and read *"PASS = 0, FAIL = 9. Median conditional single-fragment ratio
> on the generation side: **0.0447**"*. The tenth pool —
> **`cfm_fullcorpus_2026_10_10`**, the 66-row full-corpus pool — is classified
> **generation** by CONTENT (the header carries `metal_z`, but the content
> carries `sanitized = 0` and an empty `metal_z`, the signature of the
> 2026-10-12+ ODE path), and it was missing from this table even though §2.1
> already carries the reclassification. Its row is added above. The artifact's
> own summary: `summary.generation_side` → `n_pools` = **10**,
> `gate_pass` = **0**, `gate_fail` = **10**, `gate_skip` = 0, and
> `single_fragment_ratio.median` = **0.0649203119169274** (n = 10, min 0.0, max
> 0.9696969696969697, mean 0.24311538894689289). The **headline finding is
> unchanged**: no generation pool passes the fragmentation gate.

Note the ordering: the *best* generation pool, `cfm_pcgrad_off`, was trained on
**n_train = 256**. The full-scale runs — c7 (`cfm_c7_fullgen_2026_10_20`, 0.0851),
c8 (`cfm_c8_fullgen_2026_10_24`, 0.2505) and
`cfm_fullcorpus_2026_10_10` (0.3333), all n_train 26715 — sit below it.
**More data did not produce connected molecules.**

### §2.4 Link 4 — the FIELD's defect is at low `t`, and it was already documented

`docs/HONEST_NEGATIVE_CATALOG.md` §6.187 measured the trained c8 field:

* *"At `t = 0.05` the field is 64% worse than predicting zero"* — the field's
  direction at low `t` is actively anti-correlated with the target;
* the **target's** endpoint cosine `v(0.05)·v(0.95)` is **+0.736**, so the
  training signal at low `t` is well-posed;
* the trained field's measured direction cosine at `t = 0.05` is **-0.04**.

A CFM trajectory is `x_t = (1-t)x_0 + t·x_1` with `x_0 ~ N(0, σ²I)` and
`fm_noise_sigma = 2.485`. The ODE integrates the field from `t = 0` upward, so
the first ~10% of the trajectory is exactly where this field is worse than a
constant. A trajectory that starts in the wrong direction at low `t` never
recovers into a single connected geometry: the sampler spends its budget
separating atoms instead of arranging them.

**This is the mechanism.** The decoder is not mis-assembling a good cloud — it is
handed a cloud whose atoms are already in the wrong places.

### §2.5 Link 5 — the FIELD is also UNSTABLE, and the early-stop gate cannot see it

This is the finding that changed the conclusion of the whole audit. The census's
end-to-end rate was computed on each run's **last** epoch. Reading the **best**
epoch of the same series, over the same runs:

| view | population | n | min | median | max | ≥ 0.50 |
|---|---|---|---|---|---|---|
| last epoch | **CLEAN** | **27** | 0.0000 | ~~**0.057291666666666664**~~ **0.0625** | **0.5572916666666666** | ~~**25 of 29 below 0.50**~~ **23 of 27 below 0.50** |
| last epoch | ALL | 30 | 0.0000 | ~~**0.05989583333333333**~~ **0.06380208333333333** | *0.572265625 (WITHDRAWN)* | ~~27 of 32 below 0.50~~ **25 of 30 below 0.50** |
| **best epoch** | **CLEAN** | **29** | 0.0000 | **0.35546875** | **0.9375** | **11 of 29** |
| **best epoch** | ALL | 32 | 0.0000 | **0.34440104166666663** | **0.9375** | 12 of 32 |

Source: `metrics/by_metric/cfm_run_census.json` → `fragmentation_summary` →
`end_to_end_single_molecule_rate` (and `…_all_runs`) and
`best_epoch_single_molecule_rate` (and `…_all_runs`).

> **CORRECTION (2026-10-30, Wf-A114C).** This table previously read
> *"over the same 27 runs"* with `last epoch | 27 | 0.0000 | 0.0417 | 0.5723 |
> 1 of 27` and `best epoch | 27 | 0.0000 | 0.2383 | 0.9375 | 8 of 27`.  The
> population grew (the census now loads **59 runs**, of which **32** carry both
> columns, of which **29 are CLEAN**) and the **headline population changed**:
> it is now the **CLEAN** runs, with the all-runs distribution carried beside
> it rather than mixed in.  The `0.5723` maximum is the **WITHDRAWN**
> `cfm_r11_eval256` value and is labelled as such in the artifact
> (`fragmentation_summary.retracted_max`).
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

> **CORRECTION (2026-10-31, Wf-A116/A121).** This table was then corrected a
> second time, because the census made the population identity **VALUE-AWARE**
> and re-published the summary. The cells changed to their current values
> (CLEAN last-epoch median ~~**0.057291666666666664**~~, CLEAN last-epoch max
> **0.5572916666666666**, CLEAN best-epoch median **0.35546875**, all-runs
> medians ~~**0.05989583333333333**~~ / **0.34440104166666663**); the *n* column
> ~~did not move (29 CLEAN of 32)~~. The CLEAN last-epoch **max** is no longer 0.5
> but **0.5572916666666666**, the rate of `cfm_round7_f5f6_retrain/run` and its
> `run_argmax_baseline_20261007T1529` twin — two runs the earlier population
> did not contain. Source: `metrics/by_metric/cfm_run_census.json` →
> `fragmentation_summary.end_to_end_single_molecule_rate` /
> `…_all_runs` / `best_epoch_single_molecule_rate` / `…_all_runs`.
>
> **CORRECTION (2026-10-31, Wf-A152).** The values the Wf-A116/A121
> correction above published as "their current values" were themselves
> superseded when the census re-published the summary a third time: the
> **CLEAN last-epoch median** moved ~~**0.057291666666666664**~~ ->
> **0.0625** (n=27) and the **ALL-runs last-epoch median** moved
> ~~**0.05989583333333333**~~ -> **0.06380208333333333** (n=30), so the *n*
> column DID move after all (29 CLEAN of 32 -> 27 CLEAN of 30 for the
> last-epoch reading). The values that did NOT move: CLEAN last-epoch max
> **0.5572916666666666**, CLEAN best-epoch median **0.35546875** (n=29),
> ALL-runs best-epoch median **0.34440104166666663** (n=32). The superseded
> rendering is **PRESERVED VERBATIM** above (AGENTS.md §5.5 rule 6). Source:
> `metrics/by_metric/cfm_run_census.json` -> `fragmentation_summary`.
> Catalog **§6.264**.
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

The runs that reach ≥ 0.50 at their best epoch (**CLEAN** runs, 11 of 29):

| run | n_train / n_val | best epoch | best rate | last epoch |
|---|---|---|---|---|
| `cfm_pcgrad_off_2026_10_15` | 256 / 192 | 12 | **0.9375** | 0.0625 |
| `cfm_fm_ab_2026_10_09` | 256 / 192 | 12 | **0.9375** | 0.041666666666666664 |
| `cfm_bond_head_30ep_onecycle_2026_10_09` | 256 / 192 | 1 | 0.875 | ~~**0.0000**~~ **null** |
| `cfm_pcgrad_on_2026_10_14` | 256 / 192 | 11 | 0.8541666666666666 | 0.25 |
| `cfm_fm_control_2026_10_09` | 256 / 192 | 10 | 0.6875 | ~~**0.0000**~~ **null** |
| `cfm_round7_f5f6_retrain/run` | — / — | 19 | 0.5572916666666666 | 0.5572916666666666 |
| `cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529` | — / — | 19 | 0.5572916666666666 | 0.5572916666666666 |
| `cfm_fullcorpus_seed1` | 26715 / 192 | 13 | 0.5260416666666666 | 0.38020833333333337 |
| `cfm_fullcorpus_2026_10_10` | 26715 / 192 | 13 | 0.5208333333333334 | 0.5208333333333334 |
| `cfm_fullcorpus_seed2` | 26715 / 192 | 24 | 0.5026041666666666 | 0.40624999999999994 |
| `cfm_round3_retrain/d1_smoke2` | 64 / 32 | 0 | 0.5 | 0.5 |

Over ALL runs the same ≥ 0.50 list has **12** members — the extra one is
`cfm_r11_eval256` (26715 / 6714, epoch 19, 0.572265625), the **degenerate**
run whose value is WITHDRAWN as a bar.

> **CORRECTION (2026-10-31, Wf-A116/A121).** The list above previously held
> **eight** rows and the text read *"8 of 29"* / *"9 of 32"*. Three runs joined
> the CLEAN list: **`cfm_round7_f5f6_retrain/run`** and
> **`cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529`** (both
> `n_train`/`n_val` **unattested** — no config block and no cohort on disk —
> best epoch 19, rate **0.5572916666666666**, which is also their last-epoch
> rate) and **`cfm_fullcorpus_2026_10_10`** (26715 / 192, best epoch 13,
> **0.5208333333333334**). No row was removed. Source:
> `metrics/by_metric/cfm_run_census.json` → `runs[*].gen_single_molecule_rate_best`
> / `gen_single_molecule_rate_best_epoch` / `gen_single_molecule_rate_end`,
> and `fragmentation_summary.best_epoch_single_molecule_rate.n_at_or_above_0.50`
> = **11** (CLEAN) / `best_epoch_single_molecule_rate_all_runs.n_at_or_above_0.50`
> = **12** (all).

Two runs **collapse at their last epoch** — `gen_validity_end == 0.0` while
`gen_validity_max > 0.5` (census → `runs_collapsed_at_their_last_epoch`).
`cfm_fm_control` reads 0.6875 at epoch 10 and 0.0000 at epoch 30; its epoch 6
reads `gen_validity = 1.0`, `gen_single_fragment_ratio = 0.0` and
`gen_largest_fragment_frac = 0.0286` — the model was emitting atoms that all
parse, none of which form a molecule, with the largest fragment holding 2.9% of
the atoms.

**So the honest answer to "why are the molecules fragments" is two-part:**

* the field's low-`t` direction is wrong (§2.4), and
* the training oscillates violently, and the early-stopping criterion gates on
  `bond_type_f1`, which is blind to connectivity — so the checkpoint that
  survives is not the checkpoint that holds the molecules together (§5.3).

### §2.6 Link 6 — the METRIC was blind, and that is why this survived

`molmetal/scripts/train_cfm_yuelbond_wasserstein.py:9011-9025` defines the
generation validity metric:

> `molecule_validity` / `gen_validity_definition`: *"rdkit_parse_rate…
> Disconnected (C.C.C) and metal-free molecules PASS"*

Six pools carry the blind-validity signature — `gen_validity` saturated at
**1.0000** at the pool's own epoch while the pool is fragmented
(`metrics/by_metric/cfm_fragmentation_audit.json` → `blind_validity_signature`,
`summary.n_blind_validity_signature` = **6**):

| run | `gen_validity` (census) | pool-epoch validity | n | pool epoch | fragmented ratio | conditional single-fragment | single-molecule rate |
|---|---|---|---|---|---|---|---|
| `c7_fullgen_smoke_2026_10_20` | 1.0000 | 1.0000 | 32 | 0 | 1.0000 | 0.0000 | 0.0000 |
| `c7_sixterms_smoke_2026_10_20` | 1.0000 | 1.0000 | 704 | 0 | 0.9972 | 0.0028 | 0.0028 |
| `c7_vramprobe_2026_10_20` | 1.0000 | 1.0000 | 704 | 0 | 0.9972 | 0.0028 | 0.0028 |
| `cfm_c7_fullgen_2026_10_20` | 1.0000 | 1.0000 | 704 | 17 | 0.9149 | 0.0851 | 0.0851 |
| `cfm_c7_fullgen_ep4_partial_2026_10_20` | **null** (census-less) | 1.0000 | 704 | 3 | 0.9553 | 0.0447 | 0.0447 |
| `cfm_c8_fullgen_2026_10_24` | **null** (census-less) | 1.0000 | 704 | 2 | 0.7495 | 0.2505 | 0.2505 |

The two census-less rows are **absent from the census** (`cfm_c8_fullgen_2026_10_24`
never wrote a `per_epoch_metrics.csv` — it OOM'd in epoch 3, §6.7;
`cfm_c7_fullgen_ep4_partial_2026_10_20` has no per-epoch file either), so their
`gen_validity_end` is **null by construction** and the census-sourced value
cannot be quoted for them. Their pool-epoch validity is **1.0000** with
**n = 704**, read from their own event stream: because that stream carries **no
`done` event** it cannot be segmented into populations, so the whole stream is
**ONE** population. The `single-molecule rate` column is the run's own
published end-to-end rate — it is **no longer null** for these two rows.

> **CORRECTION (2026-10-31, Wf-A121/A122).** This table listed **four** rows
> (the four above with a census `gen_validity`) and the text read *"Four pools
> carry the signature `gen_validity = 1.0000` while being fragmented"*. It now
> lists **six**: `cfm_c7_fullgen_ep4_partial_2026_10_20` (pool epoch 3) and
> `cfm_c8_fullgen_2026_10_24` (pool epoch 2) joined, both publishing
> `gen_validity_at_pool_epoch` = **1.0** with `gen_validity_n` = **704**.
> Their `fragmented_ratio`s — **0.9552556818181818** and
> **0.7495265151515151** — both exceed the 0.50 gate, so **their PASS/FAIL
> verdicts did not change** (both still FAIL). Wf-A122 added the additive
> columns: the pool-epoch validity, its `n`, the pool epoch, and a
> non-null `single_molecule_rate` (previously null for the two census-less
> runs). The census-sourced `gen_validity_end` **still reads null for those
> two**, and that is correct — it is shown as null above rather than hidden.

> **OPEN ITEM (2026-10-31, Wf-A121 — deliberately NOT re-baselined).** The pool
> **`cfm_c7_fullgen_2026_10_20` spans TWO populations**. Its 12,669 rows cover
> epochs 0..17, of which **2,816 rows (epochs 0..3) belong to a KILLED
> PRE-RESUME ATTEMPT** that wrote into the same directory, while the run of
> record ran epochs **4..17** (`pool_population_composition` in the audit
> artifact: `n_rows_total` = 12669, `n_rows_at_pool_epoch` = 704,
> `run_of_record_epochs` = [4..17], `n_rows_outside_run_of_record` = **2816**).
> The composition is now published **additively**, but the **fragmentation
> verdict was deliberately NOT re-baselined on the run of record's epochs** —
> the table above and the §2.3 row still score the whole pool file. The
> census's own verdict for this run is `class_b_resume_prehistory`
> (`metrics/by_metric/cfm_run_census.json` →
> `runs[19].integrity.population_verdict ~~`runs["cfm_c7_fullgen_2026_10_20"].integrity.population_verdict`~~`), which is
> why the run is **degenerate** and outside the CLEAN headline population
> (§2.5). Its **cohort is attested**: **n_train 26,715 / n_val 192**, source
> `per_epoch_metrics.json:config` — it was never "unattested".

A stricter definition already exists in the trainer — `molecule_is_valid` at
`:8650` (RDKit-strict **AND** `n_fragments == 1` **AND** `has_metal` **AND**
`n_heavy > 1`) — and it was added as an early-stop criterion on 2026-10-27
(Wf-A113, commit `27777532`). **No run on disk has ever emitted it**: when
this paragraph was written (2026-10-30) the newest `per_epoch_metrics.csv` on
disk was dated 2026-10-21, six days before the metric existed, so no CSV
carried the column.

> **CORRECTION (2026-10-30, Wf-A114C).** The claim *"no CSV on disk carries the
> column"* is **no longer true**. MEASURED: `gen_molecule_validity` is present
> in `molmetal/reports/wf385a3_scaffold_wire_in/train/per_epoch_metrics.csv`
> (and its `results/gate_scratch/` copy) — **1 run of the 59** the census
> loads.  That run was regenerated **after** the trainer's column contract
> shipped, so it is the first run on disk that could carry the column; every
> other run predates it.  The §2.3 finding ("the measurement itself never
> existed") therefore stands for the 58 runs that predate it, and the honest
> statement today is: **the column exists, exactly one run carries it, and the
> c9 run must not be one that silently lacks it**.

---

## §3 The two denominators (the single most important correction)

The trainer writes two numbers that both look like "how good is the generation",
and they have different denominators:

| Name | Denominator | Meaning |
|---|---|---|
| `gen_single_fragment_ratio` | **PARSED rows only** | of the molecules RDKit could parse, the share that are one connected molecule |
| `gen_validity` | all sampled | the share RDKit could parse at all |

The trainer's generation CSV holds the sampler's raw output, and a SMILES that
fails to parse still gets a row. So **the conditional ratio cannot be read as an
end-to-end quality number**.

The honest end-to-end rate is:

```
single_molecule_rate = gen_validity x gen_single_fragment_ratio
```

Worked example, `cfm_fullcorpus_b64_2026_10_11`: `gen_validity = 0.0859`,
`gen_single_fragment_ratio = 0.7879` → end-to-end **0.0677**, against a
conditional 0.7879. The conditional number says "79% of what parses is one
molecule"; the end-to-end number says "6.8% of what we sample is one molecule."
Both are true. Only the second is a generation-quality number.

**This correction changed the census verdict.** Before it, the census gate read
PASS=14 / FAIL=27 on the conditional number. After it: ~~**PASS=5 / FAIL=27 / SKIP=27**~~
over **59** runs (and ~~**PASS=4 / FAIL=25 / SKIP=26**~~ **PASS=4 / FAIL=23 / SKIP=28** over the **55
CLEAN** runs). The five PASSes are `cfm_fullcorpus_2026_10_10` (0.5208333333333334),
`cfm_round3_retrain/d1_smoke2` (0.5), `cfm_round7_f5f6_retrain/run` and
`cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529` (both
0.5572916666666666) — and `cfm_r11_eval256`, the **degenerate** 1-epoch run
whose last epoch reads the WITHDRAWN 0.572265625 (it is the only one of the
five outside the CLEAN population, so the clean count is four). MEASURED over
the census's 59 runs; the per-run verdicts live in
`metrics/by_metric/cfm_run_census.json` → `runs[*].fragmentation_gate`
(counters: `fragmentation_summary.n_pass` / `n_fail` / `n_skip` and
`verdict_counts_clean_only`).

> **CORRECTION (2026-10-30, Wf-A114C).** This paragraph previously read
> *"PASS=1 / FAIL=40 / SKIP=1 over 42 runs. The single PASS is
> `cfm_r11_eval256`"*.  The run set grew from 42 to **59** and the SKIP count
> from 1 to **27** — the new runs are mostly short smokes whose generation
> eval never ran, so the gate SKIPs them.  The **single PASS** became
> **two**: the degenerate `cfm_r11_eval256` plus one clean run.

> **CORRECTION (2026-10-31, Wf-A116/A121).** The counts have moved again:
> **PASS=5 / FAIL=27 / SKIP=27** over the 59 runs, and **PASS=4 / FAIL=25 /
> SKIP=26** over the 55 CLEAN runs (the previously-stated `PASS=2 / FAIL=30 /
> SKIP=27` and `PASS=1 / FAIL=28 / SKIP=26` are stale). Two clean runs joined
> the PASS set — `cfm_round7_f5f6_retrain/run` and its
> `run_argmax_baseline_20261007T1529` twin, both at 0.5572916666666666 — and
> one clean run joined it as well: `cfm_fullcorpus_2026_10_10` at
> 0.5208333333333334. The **FAIL** count fell from 30 to 27 because those same
> three runs moved out of FAIL. The gate's *policy* did not change: the
> thresholds are still `max_fragmented_ratio` 0.5 and
> `min_largest_fragment_frac` 0.9.
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

> **EPOCH RULE (2026-10-31, Wf-A121).** The audit now publishes the pool's
> generation validity at the **POOL'S OWN EPOCH** — the resolved population's
> aggregate row that carries `gen_validity_n` — and keeps the run's
> **last-epoch** value **beside it, labelled with its own epoch** (they are
> different measurements). Three pools were re-read, because their recorded
> series' last row is not at their pool's epoch
> (`metrics/by_metric/cfm_fragmentation_audit.json` →
> `runs[*].verdict.gen_validity_provenance.superseded`):
>
> | pool | superseded value (epoch) | value at the pool's own epoch |
> |---|---|---|
> | `cfm_fullcorpus_2026_10_10` | 0.9140625 (epoch 0, n=128) | **0.3255208333333333** (epoch 10, n=768) |
> | `cfm_fullcorpus_b64_2026_10_11` | 0.0859375 (epoch 11, n=768) | **0.3763020833333333** (epoch 8, n=768) |
> | `cfm_objective_terms_2026_10_11` | 0.07421875 (epoch 8, n=768) | **0.234375** (epoch 5, n=768) |
>
> The worked example above is unaffected: it quotes the **trainer's own
> columns** (`gen_validity_end` × `gen_single_fragment_ratio_end`), which the
> census still publishes unchanged (0.0859375 × 0.7878787878787878 =
> 0.06770833333333333). What moved is the *pool-level* validity, which is a
> different number read at a different epoch. The two census-less runs now
> also carry a `recorded_vs_rederived` block (they did not before): the
> recorded side at their pool's own epoch, the last-epoch value kept
> separately with its own epoch and source, and the reason named.

All three instruments now carry the basis explicitly (`pool_end_to_end`,
`trainer_columns_derived`, `trainer_conditional_column`), so a reader can never
be handed a conditional number without being told.

---

## §4 Three corrections made during the audit

### §4.1 The two pool pipelines must not be averaged

`molmetal/scripts/c7_scale_matched_pools.py:68-95` defines the canonical rule: a
pool's CSV **header** says which pipeline produced it — `metal_z` =
reconstruction, `n_frags_raw` = generation. The audit's first draft averaged
both sides into one mean (0.3310) and reported one gate verdict. That is wrong:
a reconstruction pool measures *decoder fidelity on real geometry*, a generation
pool measures *the generator*. They are now reported separately
(`summary.generation_side` / `summary.reconstruction_side` /
`summary.unknown_side`).

### §4.2 The gate's own CSV alignment bug

The gate initially aligned the trainer's `n_frags_raw` column to the re-scored
rows positionally and reported **2299 false mismatches**. Fixed by using the
pool's `row_indexes` and zipping on the index. Post-fix cross-check: **9 pools
checked, 0 mismatches**.

### §4.3 The drawn-count join and its coverage guard

`cfm_ode_link_probe` holds 12 pool rows against 1536 samples drawn (0.8%). The
pool is not a sample of that run's generation, so an end-to-end rate over it
would be meaningless. The audit restricts the drawn count to the epochs the pool
covers and **refuses** to compute the rate when the pool covers < 50% of the
drawn samples; the entry carries an explicit note instead of a number.

---

## §5 The three blind spots, and the fixes

### §5.1 Blind spot 1 — the reported metric. FIXED in the instruments.

`gen_validity` is a parse rate. Fixed by making the gate, the audit and the
census report the end-to-end rate and name their basis (§3).

### §5.2 Blind spot 2 — the trainer's own CSV column. FIXED in the trainer.

`molmetal/scripts/train_cfm_yuelbond_wasserstein.py` wrote `n_frags_raw = 1` and
`largest_frag_frac = 1.0` for **any SMILES RDKit cannot parse** — i.e. the row
claimed to be one connected molecule holding all its atoms. A string that does
not parse is not a molecule, so the column that carries connectivity truth
asserted connectivity for exactly the rows where it is unknown.

* **MEASURED impact:** **1 row** in the generation pools on disk
  (`molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`) — the
  generation side now holds **21,445 rows** in total across **10** pools, of
  which **21,444 are parseable** and exactly **1** is not. The **9
  trainer-written** pools (the ones that carry the trainer's own
  `n_frags_raw` column, and therefore the ones the trainer's inline literal
  ever touched) account for **21,379** of those rows, **21,378** of them
  parseable; the tenth pool, `cfm_fullcorpus_2026_10_10` (66 rows), was
  written by the 2026-10-12+ ODE path and carries no such column.
* **FIX:** the counts now come from `decode_chemistry` — the same function the
  audit scores the pool with — and an unparseable SMILES writes the sentinel
  **`-1` = UNKNOWN** instead of a fabricated `1`. MEASURED bit-exact for every
  parseable row: over all **21,378** parseable rows in the trainer-written
  generation pools on disk, the old inline logic (the HEAD literal,
  re-evaluated row by row) and `decode_chemistry` agree on `n_fragments`
  (**0 differences**) and on `largest_fragment_frac` (**0 differences**).
  Re-verified 2026-10-30 (Wf-A114C) against the pools as they stand now.

> **CORRECTION (2026-10-30, Wf-A114C).** This bullet previously read *"1 row of
> **21,679** in the generation pools"* and *"over all **21,379** parseable
> rows"*.  Both were off: the generation pools hold **21,379 rows in total**,
> and the parseable subset is **21,378** (the 1 unparseable row is inside the
> 21,379).  The `21,679` figure was a digit transposition of `21,379`.  The
> **bit-exactness claim itself REPRODUCES**: re-evaluating the HEAD inline
> literal row by row against `decode_chemistry` over every parseable row of
> every generation pool on disk gives **0 differences** on `n_fragments` and
> **0 differences** on `largest_fragment_frac`.

> **CORRECTION (2026-10-31, Wf-A116).** The generation side now holds **10**
> pools, not 9: `cfm_fullcorpus_2026_10_10` (66 rows) is classified
> **generation** by content (§2.1, §2.3). So the totals above are
> **21,445 rows / 21,444 parseable / 1 unparseable** over the ten pools, and
> **21,379 / 21,378 / 1** over the nine **trainer-written** pools — the scope
> the trainer's inline literal can apply to, because the tenth pool carries no
> `n_frags_raw` column. Source:
> `metrics/by_metric/cfm_fragmentation_audit.json` → `runs[*]` →
> `n_rows_in_csv` / `n_scored` / `n_unparseable` for the ten rows whose
> `pipeline` is `generation`.

### §5.3 Blind spot 3 — the early-stopping criterion. FIXED in the c9 config.

**50 configs on disk set `early_stop_metric`**: **46** set `bond_type_f1`, **3**
set `bond_type_f1_eps`, and **1** — `configs/c9_cfm_transport_fixed.yaml`, this
wave's own fix — sets `molecule_validity`. The trainer's *default* is `combined`
= `gen_validity × bond_exact_type` — half-blind, because `gen_validity` is the
parse rate. The strict criterion `molecule_validity` has been accepted by
`--early-stop-metric` since 2026-10-27 and, **before this wave, no config
selected it**.

That matters because of §2.5: the runs reach 0.9375 at their best epoch and then
collapse, and the criterion that decides which checkpoint survives cannot see
that happening.

* **FIX:** `configs/c9_cfm_transport_fixed.yaml` now sets
  `early_stop_metric: molecule_validity`. Degradation is safe: if the generation
  eval did not run, the criterion degrades to `bond_exact_type` →
  `gen_validity` → `decode_ratio` and the degradation is recorded per epoch.
  c9 runs the generation eval every epoch (`gen_eval_n: 128` × 6 seeds), so the
  strict criterion is available.

### §5.4 The coordinate-head arm: `configs/c9_cfm_transport_fixed.yaml`

| key | value | why |
|---|---|---|
| `velocity_head` | `vector_channels` | a learned per-atom 3-vector head, not a radial scalar field |
| `vector_channels` | 8 | C = 8 learned 3-vectors per atom |
| `velocity_head_init` | `lsq` | least-squares gate init against the training targets |
| `velocity_head_init_knots` | 9 | knot count for the LSQ fit |
| `vector_channel_norm_gain` | `fixed` | the field's magnitude is not trainable-away |
| `t_embedding_dim` | 32 | the head is conditioned on `t` |
| `fm_batch_chunk_ratio` | 2.0 | chunked FM batch; only effective because `_fm_padding_invariant()` is True under `vector_channels` |

Everything else is c8's protocol held fixed: n_train 26715, batch 64, 18 epochs,
lr 3e-4, `fm_noise_sigma` 2.485, scaffold split.

**No run on disk ever used these levers.** `--velocity-head` exists only from
2026-10-22 (`540cb8cd`); `--velocity-head-init` only from 2026-10-25
(`3ceb008e`); ~~**no run's `per_epoch_metrics.json` records any head
setting**~~; c8's config declines `vector_channels` over the 32 GiB cap; every
other `vector_channels` config on disk is smoke-scale (n_train 32 / 64 / 200 /
1200 / 1600) except c9.

> **CORRECTION (2026-10-31).** Two sentences in this paragraph are now **false**
> against the census and are struck through above. **"No run on disk ever used
> these levers"** and **"no run's `per_epoch_metrics.json` records any head
> setting"** no longer hold:
> `metrics/by_metric/cfm_run_census.json` → `runs[*].head_levers_recorded` is
> **true for 3 of the 59 runs**.
> * `wfa113_t42_bond_ce/wfa113_t42` and `wfa113_t42_bond_ce/wfa113_t42_lr0`
>   record `velocity_head` = `vector_channels`, `vector_channels` = **8**,
>   `velocity_head_init` = `lsq`, `vector_channel_norm_gain` = `fixed`,
>   `t_embedding_dim` = **32** — **five of the six** levers in the table
>   above, at `n_train` = **64** (cohort 64/32, smoke scale); only
>   `velocity_head_init_knots` is null. `velocity_head` and `vector_channels`
>   come from the run's own `config` block (`config_source` =
>   `per_epoch_metrics.json`); the other three come from `provenance`
>   (`runs[*].head_levers_source_per_key`).
> * `wf385a3_scaffold_wire_in/train` records the **radial** head
>   (`velocity_head` = `radial`, `velocity_head_init` = `zero`,
>   `vector_channel_norm_gain` = `learnable`, `t_embedding_dim` = 0, and also
>   `vector_channels` = 8 and `velocity_head_init_knots` = 9) at
>   `n_train` = **16** (cohort 16/8), all six lever values sourced from its
>   `config` block (`head_levers_source_per_key` = `config` for all six).
>   It is **not** the c9 bundle — `velocity_head` is `radial`, not
>   `vector_channels`; the gain is `learnable`, not `fixed`;
>   `t_embedding_dim` is 0, not 32 — so it is evidence that the levers are
>   now *written down*, not that the c9 bundle was used.
>
> **What survives:** **no run at the c9 scale (n_train 26715) records these
> levers**, and every run that does is smoke-scale. The c8 config still
> declines `vector_channels` over the 32 GiB cap. The §7 attribution problem is
> therefore still open for the runs that matter — but the census can now
> **read** the levers where a run wrote them down.

### §5.5 The CPU A/B: direction, not effect size

`metrics/by_metric/cfm_c9_head_ab_capacity.json` — **device `cpu`, `gpu_used:
false`**, 500 steps, 8 held-out clouds, 464 atoms, same cohort and seed for both
arms, both fields fresh:

| | arm A (radial, c8's head) | arm B (c9 combination) |
|---|---|---|
| final MSE | 13.9954 | 13.8515 |
| direction cosine | 0.1685 | 0.1957 |
| mean &#124;v&#124; | 0.6299 | 0.7988 |
| gap closed | 0.0266 | **0.0366** |
| verdict | CANNOT_FIT | **CANNOT_FIT** |

against a `mean_target_norm` of **6.0290** — neither arm comes close to the
target's scale, so both are labelled `CANNOT_FIT` and neither number is an
effect size. The LSQ init's real win is at **step 0**, where the fit is
analytic: at `t = 0.02` the ratio to the zero baseline is **0.5709** and the
cosine **0.5740**, against the trained c8 field's measured **-0.04** direction at
`t = 0.05` (§6.187). That is a statement about the *initialisation*, which is
what the init is for. Whether it survives 18 epochs is **not measured**.

---

## §6 What is still unmeasured (honest negatives)

ACTIVE. Full text: `docs/HONEST_NEGATIVE_CATALOG.md` §6.199.

1. **The fix is unrun.** No c9 run exists. Every statement about c9 is a
   statement about the *config* and the *CPU capacity probe*, not about
   generated molecules.
2. **c9 now differs from c8 in TWO ways** — the coordinate head and the
   early-stop criterion. A c9-vs-c8 comparison is **not** a single-variable A/B.
   The design that would separate them is 2×2 (head × early-stop); c9 is the
   (fixed, fixed) corner.
3. **The best-epoch number is optimistically biased.** It is a max over a noisy
   epoch series — the number a run that early-stops at its best epoch would
   deliver, not the number the run delivers on average. Both views are recorded
   (last epoch median ~~**0.057291666666666664**~~ **0.0625**, best epoch median
   **0.35546875**, max **0.9375**; over the CLEAN populations of 27 (last epoch)
   / 29 (best epoch)) and neither should be quoted alone.
4. **The best on-record rate is a snapshot of a noisy series.**
   `cfm_fullcorpus_seed2` (radial head) reaches end-to-end **0.4961 at epoch 13**
   and **0.5026 at epoch 24**, but the neighbouring epochs read 0.4518 (e14),
   0.2044 (e15), 0.4531 (e16). Source:
   `molmetal/reports/cfm_fullcorpus_seed2/per_epoch_metrics.json` →
   `per_epoch_metrics[*]`. Beating 0.9375 means beating a **best-of-epochs
   maximum**, which is a much weaker bar than it looks.
5. **RETRACTED (§6.200 of `docs/HONEST_NEGATIVE_CATALOG.md`): the "n_train
   trend runs backwards / 3.2× inversion" claim.** The 3.2× is a
   **best-epoch-only** artefact, and the axis is **UNRESOLVED** rather than
   reversed: the ordering flips with the **reading**, with the **population**,
   and with the **epoch budget**. MEASURED over the census's 59 runs
   (`by_n_train`):

   | reading | population | 26715 median (n) | 256 median (n) | leader | ratio |
   |---|---|---|---|---|---|
   | last epoch | CLEAN | 0.125 (11) | ~~0.026041666666666664 (8)~~ **0.03645833333333333 (6)** | **26715** | ~~**4.80×**~~ **3.43×** |
   | last epoch | ALL (loose) | 0.12357954545454546 (14) | ~~0.026041666666666664 (8)~~ **0.03645833333333333 (6)** | **26715** | ~~**4.75×**~~ **3.39×** |
   | best epoch | CLEAN | 0.35546875 (11) | 0.7708333333333333 (8) | **256** | **2.17×** |
   | best epoch | ALL (loose) | 0.296875 (14) | 0.7708333333333333 (8) | **256** | **2.60×** |

   And under **budget matching** (rate AT epoch K-1, counting only runs that
   REACHED epoch K-1) the winner **alternates**: the ordering flips at
   **~~K = 9, 12, 14, 18 and 20~~ K = 9, 11 and 25**. **The n_train axis is UNRESOLVED, not a
   trend**, and no cohort median is a bar. The **0.572276953125** figure that
   was used as the primary bar is the rate of `cfm_r11_eval256`, a run with
   **ONE** recorded epoch (resumed at epoch 19 of a declared 20, `train_loss =
   nan`) — and it
   was defined as a best-epoch number yet applied as a final-epoch threshold.
   **Consequence: the 0.9375 bar is a 256-cohort number** — it is not a
   transferable target for a full-corpus run (§7), and neither is any cohort
   median.

   > **CORRECTION (2026-10-31, Wf-A116/A121).** This table previously read
   > `last epoch | CLEAN | 0.09635416666666666 (10) | 0.020833333333333332
   > (11) | 26715 | 4.63×`, `last epoch | ALL (loose) | 0.12215909090909091
   > (13) | 0.020833333333333332 (11) | 26715 | 5.86×`, `best epoch | CLEAN |
   > 0.296875 (10) | 0.3333333333333333 (11) | 256 | 1.12×` and `best epoch |
   > ALL (loose) | 0.23828124999999997 (13) | 0.3333333333333333 (11) | 256 |
   > 1.40×`. Every cell moved when the census re-published `by_n_train`:
   > the 26715 cohort now holds **14** runs with an end-to-end rate (11 CLEAN)
   > and the 256 cohort ~~8~~ **6** with a last-epoch rate and **8** with a
   > best-epoch rate (all CLEAN), and the medians are now
   > **0.125 / ~~0.026041666666666664~~ 0.03645833333333333** (last epoch, CLEAN),
   > **0.12357954545454546 / ~~0.026041666666666664~~ 0.03645833333333333** (last epoch, all),
   > **0.35546875 / 0.7708333333333333** (best epoch, CLEAN) and
   > **0.296875 / 0.7708333333333333** (best epoch, all). Source:
   > `metrics/by_metric/cfm_run_census.json` → `by_n_train["26715"]` and
   > `by_n_train["256"]`, `last_epoch.median` / `best_epoch.median` and the
   > `clean_only` siblings. **The axis is still UNRESOLVED**: the leader flips
   > with the reading and with the budget, and the budget-matched ordering
   > still flips at ~~**K = 9, 12, 14, 18 and 20**~~ **K = 9, 11 and 25**
   > (`budget_matched.flipped_budgets`, ~~unchanged~~ re-measured 2026-10-31 — the set MOVED when Wf-A127/Wf-A150 moved two runs from FAIL to SKIP).
   > The ratios are larger than
   > before because the 256 cohort's median moved down, not because the
   > 26715 cohort improved.
> **⚠️ POPULATION CORRECTION (2026-10-31, Wf-A150).** The cohort block's
> `clean_only.best_epoch` reading was computed over an **UNDECLARED
> population** — "CLEAN **AND** carrying a LAST-epoch rate" — instead of the
> population its own `predicate` string names ("CLEAN **AND** carrying a
> BEST-epoch rate"). The Wf-A127 republication made the difference visible:
> two CLEAN runs lost their last-epoch rate while KEEPING their best-epoch
> rate, so they silently fell out of that block. The block is now computed
> over the population its `predicate` names. MEASURED 2026-10-31 from
> `metrics/by_metric/cfm_run_census.json` -> `by_n_train` / `by_cohort`: the
> n_train=256 cohort's `clean_only.best_epoch` moves from
> ~~n=6 / median 0.59375 / mean 0.58984375 / 3 at or above 0.50~~ to
> **n=8 / median 0.7708333333333333 / mean 0.6376953125 / 5 at or above
> 0.50** (the two added runs are `cfm_bond_head_30ep_onecycle_2026_10_09` at
> 0.875 and `cfm_fm_control_2026_10_09` at 0.6875); the n_train=256
> cohort's `last_epoch` moves from
> ~~median 0.026041666666666664 (n=8)~~ to **0.03645833333333333 (n=6)** for the
> same reason; **no other block moved**. The superseded figures are
> **PRESERVED VERBATIM** as history (AGENTS.md §5.5 rule #6); the finding of
> record is catalog **§6.262**.
6. **The failure mode changed between eras and was never explained.** On the
   **trainer's own columns**, the Oct-10/11 full-corpus runs
   (n_train = 26715) read `gen_validity` **0.048177-0.596354** with `sFrag`
   **0.6129-0.8772** — mostly *unparseable*. The Oct-20 c7 generation runs
   read `gen_validity` **1.000** with `sFrag` **0.0000-0.1222** — parseable but
   *fragmented*. The headline `gen_validity` therefore went **up** (0.05 → 1.00)
   while the thing it was standing in for did not. Whether the field improved is
   **not established**.

   > **CORRECTION (2026-10-31, Wf-A116).** The `gen_validity` range read
   > *"0.048177-0.386719"*. Its **upper** bound moved to **0.596354**: the
   > census's value-aware re-derivation now reads
   > `cfm_fullcorpus_2026_10_10`'s last-epoch `gen_validity` from its
   > **resolved run of record** (14 rows, epochs 0..13,
   > `integrity.population_verdict` = `class_a_other_population`) instead of the
   > one-row per-epoch file of a *different* population. MEASURED:
   > `metrics/by_metric/cfm_run_census.json` →
   > `runs[26].gen_validity_end ~~`runs["cfm_fullcorpus_2026_10_10"].gen_validity_end`~~` =
   > **0.5963541666666666**. The `sFrag` range (**0.6129-0.8772**) and the
   > Oct-20 c7 figures (**1.000**, **0.0000-0.1222**) are unchanged. The
   > **conclusion is unchanged**: the Oct-10/11 full-corpus runs are mostly
   > *unparseable* and the Oct-20 c7 runs are parseable but *fragmented*, so the
   > headline `gen_validity` went up while the thing it stood in for did not.
7. **c8's config header claims 18 epochs; the run reached 3.** c8 OOM'd during
   epoch 3's backward pass (`molmetal/reports/cfm_c8_fullgen_2026_10_24/master.log`,
   `torch.OutOfMemoryError: CUDA out of memory`) and left **no**
   `per_epoch_metrics.csv` — which is why c8 is **absent from the 59-run census**
   and why its numbers here come from `events.jsonl` rather than the census.
8. **c8's config carries a §5.11 dead key**: `protocol.scheduler.warmup_epochs: 5`
   (`configs/c8_fullgen_b64_2026_10_24.yaml:740`) is an alias for a dest that
   `train.lr_warmup_epochs` already sets (`:323`). It is suppressed but still
   escalated under `WFA2_CONFIG_STRICT=1`, which is why the shipped c8 reference
   config cannot pass its own binding gate. c9 does not carry it.
9. **The census's own gate was blind until this wave.** Its `gate_metric` was
   never a fragmentation metric and its verdict used the conditional number.
   Both are fixed, but the 59 runs' *historical* verdicts were computed under the
   old rule, so "PASS=1" is a re-derivation over the same recorded columns, not
   a re-run.
10. **One pre-existing test failure, not caused by this wave.**
    `molmetal/tests/test_cfm_next_action_resolver.py::test_settled_mcnemar_moves_perf_out_of_the_dead_branch`
    fails on the current tree. MEASURED pre-existing: it fails identically with
    this wave's changes to the trainer and the census stashed.

---

## §7 What to run when a GPU slot is authorised

Not part of this wave (no GPU was used). When it is:

1. `python molmetal/scripts/pre_flight_check.py` — required by AGENTS.md §1.4.
2. `bash scripts/start.sh configs/c9_cfm_transport_fixed.yaml` (YAML-only launch
   per §5.11 — no CLI hyperparameters).
3. Score with `python molmetal/scripts/cfm_fragmentation_gate.py --smiles-csv <csv>`
   and with `python molmetal/scripts/cfm_fragmentation_audit.py`.
4. **The bar is a NAMED RUN, not a cohort median and not the global maximum.**
   c9 runs at n_train = 26715, and the 0.9375 figure belongs to an
   **n_train = 256** run (§6.5) — it is not a transferable target, and neither
   is any cohort median, because the cohort ordering is UNRESOLVED (§6.5,
   §6.200). Both baselines below are full-corpus runs with an **intact** epoch
   history (contiguous from epoch 0, `train_loss` not `nan`) — but they are
   **not the same cohort**: a cohort is the PAIR `(n_train, n_val)`, and the
   primary bar is **`26715/6714`** while the secondary is **`26715/192`**.
   They share `n_train` and nothing else.
   * **primary bar:** end-to-end single-molecule rate **> 0.41145833333333337**
     on a **final-epoch** reading, against **`cfm_r9_fix_v2`** (11 rows, epochs
     0..10, declared 40; strict cohort **26715/6714**) — **and**
     `largest_fragment_frac_mean > 0.90`;
   * **secondary bar (if a best-epoch claim is made):** **> 0.5260416666666666**
     against **`cfm_fullcorpus_seed1`** (best epoch 13; 15 rows, epochs 0..14,
     declared 50; strict cohort **26715/192**) — on a best-epoch reading, naming
     the run it beat;
   * **context:** the project-wide last-epoch median ~~**0.057291666666666664**~~ **0.0625**
     over the **27 CLEAN** runs of 30 (all-runs median ~~**0.05989583333333333**~~ **0.06380208333333333**
     over 30);
   * **stretch bar (reported separately, not a pass/fail):** **> 0.9375**
     — the 256/192-cohort maximum, reached by `cfm_fm_ab_2026_10_09` and
     `cfm_pcgrad_off_2026_10_15`. Clearing it would be a real result and would
     also mean the cohort effect was broken; not clearing it is **not** a
     failure by itself.
5. **The 2×2 that would attribute it:** run c9's head with c8's
   `bond_type_f1` gate as the other arm, so the head effect and the early-stop
   effect separate. Without it, a c9-vs-c8 comparison confounds them (§6.2).
   Note the head axis is a **five-lever bundle** (`velocity_head`,
   `velocity_head_init`, `vector_channel_norm_gain`, `t_embedding_dim`,
   `fm_batch_chunk_ratio` — MEASURED by resolving both configs through
   `parse_args(["--config", cfg])`: 6 of 249 resolved args differ), and that
   ~~**no artifact records which head a run used** (the census's `CFG_KEYS`
   carries none of the six head levers; 0 of 42 runs record any of them)~~
   **the census now carries all six levers for 3 of its 59 runs**
   (`runs[*].head_levers_recorded` — `wf385a3_scaffold_wire_in/train`,
   `wfa113_t42_bond_ce/wfa113_t42`, `wfa113_t42_bond_ce/wfa113_t42_lr0`; §5.4)
   — but **56 of 59 runs still record no head lever at all**, and none of the
   three that do is at the c9 scale — so the levers must still be recorded in
   the run's `config` block and in the census **before** the GPU run, or the
   attribution will not be readable afterwards
   (`docs/HONEST_NEGATIVE_CATALOG.md` §6.200).

---

## §8 Reproduction

```bash
cd /home/hugo/codes/mol-metal

# CPU-only: empty CUDA_VISIBLE_DEVICES so torch reports no GPU (see the
# 2026-10-31 incident in the header note above)
export CUDA_VISIBLE_DEVICES=""

# the two instruments, CPU-only
.venv/bin/python molmetal/scripts/cfm_fragmentation_audit.py
.venv/bin/python molmetal/scripts/cfm_fragmentation_gate.py --smiles-csv <a pool csv>

# the census
.venv/bin/python molmetal/scripts/wf_cfm_run_census.py

# the tests
.venv/bin/python -m pytest molmetal/tests/test_wfa114_cfm_fragmentation_gate.py \
    molmetal/tests/test_wfa114_c9_config_binding.py -q

# the c9 config round-trip (§5.11), CPU-only
.venv/bin/python -m pytest molmetal/tests/test_wfa114_c9_config_binding.py -q
```

## §9 File index

| Path | Role |
|---|---|
| `docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md` | this document |
| `docs/HONEST_NEGATIVE_CATALOG.md` §6.199 | the honest-negative entry |
| `molmetal/scripts/cfm_fragmentation_gate.py` | the gate (end-to-end denominator) |
| `molmetal/scripts/cfm_fragmentation_audit.py` | the audit (pipeline-split, disk-wide) |
| `molmetal/scripts/wf_cfm_run_census.py` | the SSOT over 59 CFM runs (last-epoch + best-epoch) |
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | the `n_frags_raw` sentinel fix (§5.2) |
| `configs/c9_cfm_transport_fixed.yaml` | the fix (unrun) |
| `molmetal/tests/test_wfa114_cfm_fragmentation_gate.py` | the instruments' tests |
| `molmetal/tests/test_wfa114_c9_config_binding.py` | the §5.11 round-trip + the early-stop criterion |
| `metrics/by_metric/cfm_run_census.json` | artifact: 59 runs |
| `metrics/by_metric/cfm_fragmentation_audit.json` | artifact: 13 pools |
| `metrics/by_metric/cfm_c9_head_ab_capacity.json` | artifact: the CPU A/B |

---

## §10 Correction log — 2026-10-31 (Wf-A116 / A120 / A121 / A122)

This audit is a dated report. The waves below changed the **artifacts it
quotes**, not the audit's conclusions. Every number that moved is listed here
with the artifact path it now matches. **Nothing in this section is
PROJECTED** — each row is a value read out of a JSON on disk.

| # | where | was | now | artifact path |
|---|---|---|---|---|
| 1 | §0.2, §2.6 | four pools carry the blind-validity signature | **six** | `cfm_fragmentation_audit.json` → `summary.n_blind_validity_signature` = 6, `blind_validity_signature[]` (6 rows) |
| 2 | §0.3, §2.5, §6.3, §7 | last-epoch median **0.03125** | ~~**0.057291666666666664**~~ **0.0625** | `cfm_run_census.json` → `fragmentation_summary.end_to_end_single_molecule_rate.median` |
| 3 | §0.3, §2.5 | 28 of 29 below 0.50 / 22 of 29 below 0.10 | ~~**25 of 29**~~ **23 of 27** / ~~**18 of 29**~~ **16 of 27** | same block → `n_below_0.50` = ~~25~~ **23**, `n_below_0.10` = ~~18~~ **16** |
| 4 | §0.3, §2.5, §7 | all-runs median **0.03580729166666667** | ~~**0.05989583333333333**~~ **0.06380208333333333** | `fragmentation_summary.end_to_end_single_molecule_rate_all_runs.median` |
| 5 | §0.3, §2.5 | 30 of 32 below 0.50 | ~~**27 of 32**~~ **25 of 30** | same block → `n_below_0.50` = ~~27~~ **25** |
| 6 | §2.5 | CLEAN last-epoch max **0.5** | **0.5572916666666666** | `fragmentation_summary.end_to_end_single_molecule_rate.max` |
| 7 | §0.4, §2.5, §6.3 | best-epoch median **0.1953125** | **0.35546875** | `fragmentation_summary.best_epoch_single_molecule_rate.median` |
| 8 | §0.4, §2.5 | 8 of 29 CLEAN ≥ 0.50 (9 of 32 all) | **11 of 29** (**12 of 32**) | `best_epoch_single_molecule_rate.n_at_or_above_0.50` = 11 / `best_epoch_single_molecule_rate_all_runs.n_at_or_above_0.50` = 12 |
| 9 | §2.5 | all-runs best-epoch median **0.18229166666666669** | **0.34440104166666663** | `best_epoch_single_molecule_rate_all_runs.median` |
| 10 | §2.5 | 8-row ≥ 0.50 list | **11 rows** (+ `cfm_round7_f5f6_retrain/run`, + `…/run_argmax_baseline_20261007T1529`, + `cfm_fullcorpus_2026_10_10`) | `cfm_run_census.json` → `runs[*].gen_single_molecule_rate_best` / `gen_single_molecule_rate_best_epoch` / `gen_single_molecule_rate_end` |
| 11 | §3 | PASS=2 / FAIL=30 / SKIP=27 | ~~**PASS=5 / FAIL=27 / SKIP=27**~~ **PASS=5 / FAIL=25 / SKIP=29** | `fragmentation_summary.n_pass` / `n_fail` / `n_skip` |
| 12 | §3 | PASS=1 / FAIL=28 / SKIP=26 (55 CLEAN) | ~~**PASS=4 / FAIL=25 / SKIP=26**~~ **PASS=4 / FAIL=23 / SKIP=28** | `fragmentation_summary.verdict_counts_clean_only` |
| 13 | §2.3 | 9 generation pools, PASS=0 / FAIL=9, median **0.0447** | **10 pools**, PASS=0 / **FAIL=10**, median **0.0649203119169274** | `cfm_fragmentation_audit.json` → `summary.generation_side` (`n_pools` 10, `gate_fail` 10, `single_fragment_ratio.median`) |
| 14 | §5.2 | generation pools hold **21,379** rows, **21,378** parseable | **21,445** rows / **21,444** parseable over 10 pools; **21,379 / 21,378** over the 9 trainer-written pools | `cfm_fragmentation_audit.json` → `runs[*].n_rows_in_csv` / `n_scored` / `n_unparseable` where `pipeline` = `generation` |
| 15 | §6.5 | 26715 CLEAN last median **0.09635416666666666 (10)** | **0.125 (11)** | `cfm_run_census.json` → `by_n_train["26715"].clean_only.last_epoch` |
| 16 | §6.5 | 26715 all last median **0.12215909090909091 (13)** | **0.12357954545454546 (14)** | `by_n_train["26715"].last_epoch` |
| 17 | §6.5 | 256 last-epoch median **0.020833333333333332 (11)** | ~~**0.026041666666666664 (8)**~~ **0.03645833333333333 (6)** | `by_n_train["256"].last_epoch` (all 6 of its 6 rated runs are CLEAN) |
| 18 | §6.5 | 26715 CLEAN best median **0.296875 (10)** | **0.35546875 (11)** | `by_n_train["26715"].clean_only.best_epoch` |
| 19 | §6.5 | 26715 all best median **0.23828124999999997 (13)** | **0.296875 (14)** | `by_n_train["26715"].best_epoch` |
| 20 | §6.5 | 256 best median **0.3333333333333333 (11)** | **0.7708333333333333 (8)** | `by_n_train["256"].best_epoch` |
| 21 | §1, §7, §9 | the census covers **42** runs | **59** runs | `cfm_run_census.json` → `n_runs` = 59, `corpus.n_runs_loaded` = 59 |
| 22 | §3 | pool validity of `cfm_fullcorpus_2026_10_10` read as **0.9140625** (epoch 0) | **0.3255208333333333** at the pool's own epoch 10 (n=768) | `cfm_fragmentation_audit.json` → `runs[6].verdict.gen_validity_provenance ~~`runs["cfm_fullcorpus_2026_10_10"].verdict.gen_validity_provenance`~~` (`.value`, `.superseded`) |
| 23 | §3 | pool validity of `cfm_fullcorpus_b64_2026_10_11` read as **0.0859375** (epoch 11) | **0.3763020833333333** at the pool's own epoch 8 (n=768) | same block, run `cfm_fullcorpus_b64_2026_10_11` |
| 24 | §3 | pool validity of `cfm_objective_terms_2026_10_11` read as **0.07421875** (epoch 8) | **0.234375** at the pool's own epoch 5 (n=768) | same block, run `cfm_objective_terms_2026_10_11` |
| 25 | §5.4, §7 | "**no run's `per_epoch_metrics.json` records any head setting**"; "0 of **42** runs record any head lever" | **3 of 59 runs record the levers** (`wf385a3_scaffold_wire_in/train`, `wfa113_t42_bond_ce/wfa113_t42`, `wfa113_t42_bond_ce/wfa113_t42_lr0`); **56 of 59 record none** | `cfm_run_census.json` → `runs[*].head_levers_recorded` / `head_levers` / `head_levers_source_per_key` |
| 26 | §6.6 | Oct-10/11 full-corpus `gen_validity` range **0.048177-0.386719** | **0.048177-0.596354** | `cfm_run_census.json` → `runs[26].gen_validity_end` ~~`runs["cfm_fullcorpus_2026_10_10"].gen_validity_end`~~ = 0.5963541666666666 (the resolved run of record, epochs 0..13) |
| 27 | §2.5, §2.6 | census cohort identity read from a single population verdict | **value-aware**: `class_a_other_population` **4**, `class_b_resume_prehistory` **2**, `class_d_foreign_launch` **1**, `events_absent` **9**, `no_conflict` **43** over 59 runs (4 conflicted, **4 resolved**, 0 refused) | `cfm_run_census.json` → `population_conflicts.verdict_counts` / `n_conflicted` / `n_conflicted_resolved` / `n_conflicted_refused` |
| 28 | §2.6 | `cfm_fullcorpus_2026_10_10` cohort unknown | strict cohort **26715/192** (9 runs), `cohort_source` = `run_log:ts_matched` | `cfm_run_census.json` → `by_cohort["26715/192"].runs` / `runs[26].cohort_source ~~`runs["cfm_fullcorpus_2026_10_10"].cohort_source`~~` |
| 29 | §2.5 | `None/None` cohort (no attestable `n_train`/`n_val`) | **3 runs**: `cfm_round3_retrain/run`, `cfm_round7_f5f6_retrain/run`, `cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529` — all three `cohort_source` = `unattested` | `cfm_run_census.json` → `by_cohort["None/None"].runs` / `runs[*].cohort_source` |
| 30 | §0.4, §2.5, §3, §6.6, §7 | the run was cited by NAME — `runs["<name>"]` — which **does not resolve** (`runs` is a LIST) | **8 sites** re-pointed to the list index (`runs[19]`, `runs[26]`, `runs[28]`, `runs[6]`); the superseded path is **PRESERVED VERBATIM** in a strike | `cfm_run_census.json` -> `runs` (59-element LIST, keyed by `run`); `cfm_fragmentation_audit.json` -> `runs` (13-element LIST, keyed by `run`) |

> **⚠️ CORRECTION (2026-11-01, Wf-A153).** The artifact paths in this
> document cited the run as `runs["<name>"]`, but `runs` is a **LIST** in
> both artifacts — not a mapping — so **none of those citations
> resolved**. MEASURED 2026-11-01: `cfm_run_census.json` -> `runs` is a
> **59-element LIST** keyed by `run`; `cfm_fragmentation_audit.json` ->
> `runs` is a **13-element LIST** keyed by `run`. **8 sites** in this
> document were re-pointed to the list index (`runs[19]`, `runs[26]`,
> `runs[28]`, `runs[6]`); the superseded name-keyed path is **PRESERVED
> VERBATIM** in a strike at each site (AGENTS.md §5.5 rule 6). **No value
> moved** — every leaf the paths were quoted for was re-read at the
> resolved index and is unchanged (e.g. `runs[26].gen_validity_end` =
> **0.5963541666666666**, `runs[19].decode_single_fragment_ratio_end` =
> **0.7135416666666666**, `runs[28].gen_validity_end` = **0.0859375**).
> Catalog **§6.264**.

**The census's population identity is now value-aware, and the "unattested"
wording in §2.5 is now census-backed rather than an inference.** Under the
value-aware re-derivation the four conflicted runs are resolved by *value*
comparison, not by file order, and their verdicts are named:
`cfm_fullcorpus_2026_10_10`, `cfm_round3_retrain/run`,
`cfm_round7_f5f6_retrain/run` and
`cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529` are
`class_a_other_population` (the per-epoch file records a **different**
population's epochs than the run of record — the census scores the run of
record and re-derives the series from its `events.jsonl` rows);
`cfm_bond_head_20ep_2026_10_09` and `cfm_c7_fullgen_2026_10_20` are
`class_b_resume_prehistory`; `wf299_cfm_main/wave_a/full_stack` is
`class_d_foreign_launch`. `cfm_fullcorpus_2026_10_10` therefore moved **out of**
the unknown cohort and **into** the strict **26715/192** cohort, and its
`gen_validity_end` is read from the resolved run of record (14 epochs, 0..13) —
that is the whole of the §6.6 range move. `cfm_c7_fullgen_2026_10_20` is
`class_b_resume_prehistory` (the census is the authority for that), and its
cohort **is** attested — **26,715 / 192**, source `per_epoch_metrics.json:config`
(§2.6). The three runs in the `None/None` cohort have **no** attestable
`n_train`/`n_val` (`cohort_source` = `unattested`), which is the only sense in
which the word "unattested" survives in this report.

**Re-read against the artifacts in scope and NOT changed** (the number still
matches): the headline **PASS=0 / FAIL=10** on the generation side (§0.5, §2.3)
— `cfm_fragmentation_audit.json` → `summary.generation_side.gate_pass` = 0,
`gate_fail` = 10; the `0.9375` best-epoch maximum — `cfm_run_census.json` →
`fragmentation_summary.best_epoch_single_molecule_rate.max`; the WITHDRAWN
`0.572265625` all-runs maximum and its `cfm_r11_eval256` attribution —
`fragmentation_summary.retracted_max` and `…_all_runs.max`; c7's decoder
figure **0.7135** — `runs[19].decode_single_fragment_ratio_end ~~`runs["cfm_c7_fullgen_2026_10_20"].decode_single_fragment_ratio_end`~~`
= 0.7135416666666666; the §3 worked example (0.0859 × 0.7879 → 0.0677 — the
**trainer's** last-epoch columns, which the census still publishes unchanged:
`runs[28].gen_validity_end ~~`runs["cfm_fullcorpus_b64_2026_10_11"].gen_validity_end`~~` = 0.0859375,
`gen_single_fragment_ratio_end` = 0.7878787878787878,
`gen_single_molecule_rate_end` = 0.06770833333333333); the §5.2 **1**
unparseable row — `runs[*].n_unparseable` (only `cfm_c7_fullgen_2026_10_20`
carries one); the §7 bars (primary **0.41145833333333337** vs
`cfm_r9_fix_v2`, secondary **0.5260416666666666** vs `cfm_fullcorpus_seed1`,
stretch **0.9375**) — `cfm_run_census.json` → `per_run_bar`; the §6.5
budget-matched flip list (**~~K = 9, 12, 14, 18 and 20~~ K = 9, 11 and 25**) —
`budget_matched.flipped_budgets`; and the 13-pool census with 9 pools
cross-checked and 0 mismatches — `cfm_fragmentation_audit.json` →
`summary.n_runs_considered` = 13, `n_pools_with_csv_cross_check` = 9,
`n_pools_with_csv_mismatch` = 0.

**Not re-verified in this pass** (their artifacts were out of scope for this
correction wave): the §2.2 c8 per-epoch decoder table (read from
`molmetal/reports/cfm_c8_fullgen_2026_10_24/events.jsonl`), the §5.2
bit-exactness re-check, and the §5.5 CPU A/B table (read from
`metrics/by_metric/cfm_c9_head_ab_capacity.json`). They are carried over
unchanged, and no wave since 2026-10-30 has touched those artifacts.

The module docstring figure **0.0085** (Wf-A122: it belongs to
`cfm_fullcorpus_2026_10_10`, not to the b64 run) does **not** appear in this
report — there is nothing to strike.
