# Wf-A114E — the census scores a DIFFERENT population than the run it names

**Date:** 2026-10-30
**Status:** diagnosis COMPLETE, fix NOT yet applied
**GPU:** 0. Everything here is a read of JSON already on disk.
**User directive (verbatim):** *"好，那你启动ultracode把这些问题都修一下，但是禁止使用gpu"*

---

## §1 The defect in one sentence

`wf_cfm_run_census.py` reads **one** file per run — `per_epoch_metrics.{json,csv}` — and
that file is **overwritten by whichever run wrote into the directory last**. A run
directory can therefore hold the record of a *different, shorter, later* run than the
one the directory is named for, and the census scores that later run's numbers under
the earlier run's name, with no warning.

`events.jsonl` is append-only and therefore keeps **every** population that ever wrote
into the directory — including each one's `done` event. It is the ground truth the
census never reads.

---

## §2 Evidence — the `done` events vs the file on disk

MEASURED by reading each run's `events.jsonl` `done` events and its
`per_epoch_metrics.json`. **4 of 59** census runs are mis-scored this way.

### 2.1 `cfm_fullcorpus_2026_10_10` — the generation pool of record's own run

| source | content |
|---|---|
| `events.jsonl` done #1 | `n_epochs_run: 14, best_epoch: 10` ← the real Wf-124 P2 run |
| `events.jsonl` done #2 | `n_epochs_run: 1, best_epoch: 0` |
| `events.jsonl` done #3 | `n_epochs_run: 1, best_epoch: 0` |
| `per_epoch_metrics.json` **on disk** | `config = {epochs: 1, batch_size: 8, n_train: 64, n_val: 32, pocket_id: "1h36", device: "cpu"}` — the **1-epoch CPU smoke's** config, not the 14-epoch run's |

The census therefore loads this run as **1 epoch**, `epochs_declared: 1`,
`history_completion_frac: 1.0`, `resumed: false`, `contiguous_from_zero: true` — i.e.
it looks like a *complete* run, and the `done` event of the real 14-epoch run is
invisible to it.

| view | census (reads `per_epoch_metrics`) | `events.jsonl` (epoch 13) |
|---|---|---|
| epochs | 1 (0) | 14 (0…13) |
| `gen_validity` last | 0.9140625 | 0.5963541666666666 |
| `gen_single_fragment_ratio` last | 0.008547008547008548 | 0.8733624454148472 |
| **end-to-end last** | **0.0078125** | **0.5208333333333334** |
| **end-to-end best** | **0.0078125** | **0.5208333333333334** (epoch 13) |

The census's value is **66.7× smaller** and — decisively — **below** the 0.50 gate when
the real one is **above** it.

### 2.2 The three round-3 / round-7 runs

| run | census `n_epochs` | `events.jsonl` done #1 | census last e2e | events last e2e (ep 19) | census best e2e | events best e2e |
|---|---|---|---|---|---|---|
| `cfm_round3_retrain/run` | 4 | `n_epochs_run: 20, best_epoch: 18` | 0.0 | **0.29874213836477986** | 0.022135416666666668 | **0.47395833333333326** (ep 17) |
| `cfm_round7_f5f6_retrain/run` | 4 | `n_epochs_run: 20, best_epoch: 19` | 0.02734375 | **0.5572916666666666** | 0.02734375 | **0.5572916666666666** |
| `cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529` | 4 | `n_epochs_run: 20, best_epoch: 19` | 0.0078125 | **0.5572916666666666** | 0.0078125 | **0.5572916666666666** |

Each of these directories holds a `done` event for the 20-epoch run **and** `done`
events for the 4-epoch runs that wrote after it (`n_epochs_run: 4`, `best_epoch: 0`).
Their `per_epoch_metrics.json` still carries `config.epochs = 20` (the YAML budget)
but only **4 rows** in `per_epoch_metrics` — so the census reads
`epochs_declared: 20, epochs_recorded: 4, history_completion_frac: 0.2` and treats the
run as *legitimately early-stopped*.

**Two of the four flip across the 0.50 gate**: `cfm_fullcorpus_2026_10_10`
(0.0078 → 0.5208) and `cfm_round7_f5f6_retrain/run` (0.0273 → 0.5573). All three
round-3/round-7 runs sit in the **CLEAN** population today (contiguous-from-zero,
no `nan` train_loss) and therefore sit inside the medians that
`docs/HONEST_NEGATIVE_CATALOG.md` §6.201 publishes.

---

## §2.3 The append-log trace, line by line (MEASURED 2026-10-30)

`events.jsonl` is a **three-population append log** for
`cfm_fullcorpus_2026_10_10` (99 lines). Each population is a run of `train_epoch`
rows terminated by its own `done` event:

| lines | `total_epochs` | `gen_validity_n` | `train_loss` range | `done` |
|---|---|---|---|---|
| 4–84 | **50** | 768 | 9.93 → 9.15 | `{n_epochs: 50, n_epochs_run: 14, best_epoch: 10, gate_value: 0.9194141126592557, wall_time_s: 17839.6}` |
| 89–91 | **1** | **128** | 13.82 | `{n_epochs: 1, n_epochs_run: 1, best_epoch: 0, wall_time_s: 35.2}` |
| 96–98 | **1** | **128** | 13.68 | `{n_epochs: 1, n_epochs_run: 1, best_epoch: 0, wall_time_s: 117.9}` |

The census's `per_epoch_metrics.json` holds **exactly the last 1-epoch row**
(`epoch 0`, `gen_validity 0.9140625`, `gen_single_fragment_ratio
0.008547008547008548`) with the smoke's config
(`{epochs: 1, batch_size: 8, n_train: 64, n_val: 32, pocket_id: "1h36",
device: "cpu"}`).

**So the census reports, under the name of the full-corpus run, the numbers of a
1-epoch CPU smoke.** The real run's own per-epoch series:

| epoch | `gen_validity` | n | `gen_single_fragment_ratio` | end-to-end |
|---|---|---|---|---|
| 0 | 0.08203125 | 768 | 0.36507936507936506 | 0.003 |
| 10 (**the pool of record's epoch**) | 0.3255208333333333 | 768 | 0.872 | 0.28385416666666663 |
| 13 (**last**) | 0.5963541666666666 | 768 | 0.8733624454148472 | **0.5208333333333334** |

Note the epoch-0 row of the real run (`gen_validity_n = 768`) is NOT the row the
census reads (`gen_validity_n = 128`) — the census's row is the smoke's, and it
coincides with the smoke's identical values because the same smoke was run twice.

### `cfm_round3_retrain/run` (151 lines)

| lines | `total_epochs` | `gen_validity_n` | `train_loss` range | `done` |
|---|---|---|---|---|
| 3–100 | 20 | **192** | 0.4506 → 0.0609 | `{n_epochs: 20, n_epochs_run: 20, best_epoch: 18, wall_time_s: 23742.9}` |
| 105–125 | 20 | **768** | 14.36 → 13.01 | `{n_epochs: 20, n_epochs_run: 4, best_epoch: 0}` |
| 130–150 | 20 | **768** | 14.46 → 13.09 | `{n_epochs: 20, n_epochs_run: 4, best_epoch: 0}` |

`per_epoch_metrics.json` = the **last** 4-epoch population's rows (epochs 0–3),
`config.epochs = 20`, `n_train = 256`. Real run's epoch 19:
`gv 0.8333333333333334 × sf 0.3584905660377358` = **0.29874213836477986**.

### `cfm_round7_f5f6_retrain/run` (176 lines)

| lines | `gen_validity_n` | `train_loss` range | `done` |
|---|---|---|---|
| 3–100 | **192** | 10.02 → 9.17 | `{n_epochs_run: 20, best_epoch: 19, wall_time_s: 16060.1}` |
| 105–125 | 768 | 14.22 → 13.01 | `{n_epochs_run: 4, best_epoch: 0}` |
| 130–150 | 768 | 14.29 → 13.03 | `{n_epochs_run: 4, best_epoch: 0}` |
| 155–175 | 768 | 14.28 → 13.68 | `{n_epochs_run: 4, best_epoch: 0}` |

`per_epoch_metrics.json` = the **last** 4-epoch population's rows; its epoch 3 is
`gv 0.85546875 × sf 0.0319634703196347` = 0.02734375 — exactly the census's
`gen_single_molecule_rate_end`. The real run's epoch 19:
`gv 0.6666666666666666 × sf 0.8359375` = **0.5572916666666666**.

---

## §2.4 How a population can be told apart — three independent signals

1. **`done` events.** Every population ends with one, carrying `n_epochs_run`,
   `n_epochs` (the declared budget) and `best_epoch`. The census can segment the
   event log into populations on these boundaries.
2. **`train_loss` regime.** The real runs and the later ones sit in disjoint loss
   regimes (0.45–0.06 / 10.0–9.2 vs 13.0–14.5).
3. **`gen_validity_n`.** The real runs evaluate over 192 (or 768); the later ones
   over a different denominator (128 for the CPU smoke, 768 for the 4-epoch ones).

None of these is a heuristic: each is a value the trainer itself wrote. A resolver
that segments on `done` and cross-checks the segment's `n_epochs_run` against the
`per_epoch_metrics` row count can name the population unambiguously.

**The row-count coincidence to watch for:** the later 4-epoch populations also
declare `n_epochs: 20` in their config — the same declared budget as the real run —
so "declared budget" alone cannot separate them; only the `done` event's
`n_epochs_run` and the row count together can.

---

## §3 Scope over all 59 census runs

MEASURED: for every run with an `events.jsonl`, compare the census's
`integrity.epochs_recorded` / `epoch_index_end` against the epoch indices in that
file's `gen_eval` / `train_epoch` events.

**7 of 59 diverge.** They split into three classes, and only the first is a defect:

| class | runs | verdict |
|---|---|---|
| **A — census reads a TRUNCATED PREFIX of a longer run in the same dir** | `cfm_fullcorpus_2026_10_10` (1 of 14), `cfm_round3_retrain/run` (4 of 20), `cfm_round7_f5f6_retrain/run` (4 of 20), `cfm_round7_f5f6_retrain/run_argmax_baseline_20261007T1529` (4 of 20) | **DEFECT — mis-scored** |
| **B — explained by a RESUME** (`epoch_index_start > 0`, `resumed: true`): the census's own file starts at epoch 4 / epoch 1, and `events.jsonl` holds the pre-resume epochs from an earlier run | `cfm_c7_fullgen_2026_10_20` (14 rows, epochs 4–17), `cfm_bond_head_20ep_2026_10_09` (19 rows, epochs 1–19) | **NOT a defect — already flagged** (`resumed`, `contiguous_from_zero: false` → excluded from the CLEAN population) |
| **C — `events.jsonl` is the SHORTER record** (a partial stream: no `done` event at all) | `wf299_cfm_main/wave_a/full_stack` (census 3 epochs, events 2, `events` types = `gen_eval, lr_step, train_epoch` only) | **NOT a defect in the census** — but it shows `events.jsonl` is not always complete, so it cannot be blindly trusted as the sole authority |

**Why class B and C must not be "fixed" the same way:** a resolver that always
preferred the longer series would silently re-import the pre-resume epochs of a
resumed run into that run's history — a different defect (double-counting epochs that
a resumed run never trained). A resolver that always preferred `events.jsonl` would
lose `wf299_cfm_main`'s third epoch. **The resolution must be per-run and recorded.**

---

## §4 Why the existing integrity block cannot catch it

`wf_cfm_run_census.py::_integrity` (line 572) records `epoch_index_start/end`,
`epochs_recorded`, `epochs_declared`, `epoch_history_contiguous_from_zero`,
`resumed`, `history_completion_frac`, `nan_train_loss`, `series_source`. All of these
read **the same overwritten file** that holds the wrong population. They are
internally consistent and externally wrong — the run looks like a *complete 1-epoch
run*, not a clobbered 14-epoch one.

There is no `events.jsonl` read anywhere in the script:
`grep -n "events.jsonl" molmetal/scripts/wf_cfm_run_census.py` → **no match**.

---

## §5 The upstream cause (the writer, not the reader)

`molmetal/scripts/train_cfm_yuelbond_wasserstein.py:20381` writes
`(out_dir / "per_epoch_metrics.json").write_text(...)` — **once, at end of training**,
to a **fixed path**. Two runs launched into the same `--out-dir` therefore clobber
each other's record, and the survivor is whichever finished last. `events.jsonl` is
opened in append mode (`--events-jsonl-path`, default `<out_dir>/events.jsonl`), so it
keeps both.

This is why the directories above are chimeras: **the run name is the directory name,
but the directory is not a run.** The name is not a key.

---

## §6 What must NOT change

* **No GPU.** No training, no sampling, no sweep.
* **No new argparse flag** unless the task explicitly says so (§5.7 then applies).
* **No change to any number that is currently correct.** The 55 non-divergent runs'
  census entries must be **bit-identical** after the fix. This is the acceptance test.
* **`docs/HONEST_NEGATIVE_CATALOG.md` §6.201 and `TODO/completed/wfa114_cfm_fragmentation_audit.md`
  quote the CLEAN-population medians.** Those medians will move. They must be
  re-quoted in the same wave, with the new census as the cited source, and the
  retraction log must record that they moved.
* **No touches to `paper/`.** `metrics/paper_tables/` is regenerated by
  `molmetal/scripts/paper_metric_tables.py` and must stay byte-identical in T7.
* **The 3 resume-class runs must stay excluded from the CLEAN population.** A fix that
  readmits them would be a false improvement.

---

## §7 The fix, in three layers

**L1 — the census must cross-check (read-only, safe).**
For each run directory: read `events.jsonl`, collect its `done` events, and compare
`max(n_epochs_run)` and the epoch index set against the `per_epoch_metrics` series.
Record the result in `integrity` as a new field (e.g. `population_conflict`,
`events_done_epochs`, `events_last_epoch`). A conflict must be **visible in the
artifact**, not silently resolved.

**L2 — the census must not score a truncated prefix.**
When a conflict is detected, the census must either (a) re-derive the per-epoch
series from the `events.jsonl` rows of the *matching* population, or (b) mark the run
`CONFLICTED` and exclude it from the headline population while still listing it with
its reason. **Which of the two is correct per run must be decided by evidence, not by
a single blanket rule** — see §3's class B and C.

**L3 — the writer must stop the clobber.**
`per_epoch_metrics.json` must carry a **run fingerprint** (the config identity that
already distinguishes the populations: `epochs`, `n_train`, `n_val`, `pocket_id`,
`seed`, `device`) and the trainer must **refuse to overwrite a file whose fingerprint
differs** — or write to a run-scoped filename. This is the fix that stops the defect
at the source. It must not require a new CLI flag.

---

## §8 Acceptance criteria (all MEASURED, all CPU)

1. `grep -n "events.jsonl" molmetal/scripts/wf_cfm_run_census.py` → **matches**.
2. The census artifact carries a population-conflict field for **every** run, and the
   **4 class-A runs** carry their conflict.
3. The **55 non-divergent runs**' census entries are **bit-identical** before/after.
4. The class-B (resume) runs are **still excluded** from the CLEAN population.
5. `cfm_fullcorpus_2026_10_10`'s reported end-to-end rate is **0.5208333333333334**
   (events-derived) — or the run is refused with a reason — and **never 0.0078125**
   presented as that run's rate.
6. The trainer refuses to clobber a `per_epoch_metrics.json` from a different run.
7. The census's CLEAN-population medians are re-quoted in
   `docs/HONEST_NEGATIVE_CATALOG.md` and `TODO/completed/wfa114_cfm_fragmentation_audit.md`,
   with the OLD values recorded as retracted.
8. `metrics/paper_tables/T7_generation.md` and `.csv` are **byte-identical** after
   `paper_metric_tables.py --out-dir metrics/paper_tables/` re-runs.
9. `pytest molmetal/tests/ -q` shows **no new failure** against the pre-existing
   baseline.
