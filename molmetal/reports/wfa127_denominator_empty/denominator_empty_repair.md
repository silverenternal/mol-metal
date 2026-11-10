# Wf-A127 — the empty denominator was published as a measured 0.0

**Date:** 2026-10-31 · **Mode:** CPU-only, read-only audit + one code path repaired.
**No GPU was used.  No training, no sampling, no sweep.**

---

## 1 — The defect

A generation-eval row carries **TWO denominators**:

| column | meaning | denominator of |
|---|---|---|
| `gen_validity_n` | the samples the eval **DREW** | `gen_validity` (the RDKit parse rate) |
| `gen_validity_val` | the rows the eval **SCORED** (RDKit accepted) | `gen_single_fragment_ratio`, `gen_largest_fragment_frac`, `gen_metal_ratio` |

When the eval drew samples and **NONE** of them parsed, the drawn count is
`> 0` and the scored count is `0`, so every chemistry rate is **0/0 —
undefined**.  The trainer's 0.0 there is a *placeholder*, not a measurement.

The census gated the chemistry rates on the **DRAWN** count only, so the
placeholder 0.0 was published into `metrics/by_metric/cfm_run_census.json`
as if it were a measurement, and the fragmentation gate **FAILED** on it.

### Where (file:line)

| what | where |
|---|---|
| the DRAWN-count-only gate that let the empty SCORED denominator through | `molmetal/scripts/wf_cfm_run_census.py` — `_normalise_row`, the `for key in ("gen_single_fragment_ratio", "gen_largest_fragment_frac", "gen_metal_ratio")` loop (**the defect**) |
| where emptiness is now tested | `_normalise_row` — `scored_empty = bool(measured and v is not None and v == 0)` |
| the reason string for the two sub-causes | `_empty_denominator_reason(gen_validity_n, gen_validity_val)` (new helper, added above `_fragmentation_verdict`) |
| where the 0.0 was written into the artifact | `load_run` — `gen_single_fragment_ratio_end` / `gen_largest_fragment_frac_end` / `gen_single_molecule_rate_end`, and inside `fragmentation_gate` (`measured_rate`, `single_fragment_ratio_conditional`, `fragmented_ratio`) |
| where the gate verdict is formed | `_fragmentation_verdict(...)` — now takes `gen_validity_n=` and `gen_validity_val=` and returns `SKIP` / `denominator_empty` with **no** `measured_rate` |

## 2 — The convention followed (NOT invented)

The convention is the **trainer's own**, defined by Wf-A114/T85 in
`molmetal/tests/test_cfm_gen_metric_sentinel.py`:

* over the **SCORED** rows → `None` when nothing was scored
  (`gen_single_fragment_ratio`, `gen_largest_fragment_frac`,
  `gen_metal_ratio`, and the end-to-end rate, which "inherits the
  undefined factor");
* over the **DRAWN** samples → a real number, and a measured 0 of N stays
  `0.0` (`gen_validity`, `gen_molecule_validity`).

The census now holds the same convention, and the two are cross-checked by
`molmetal/tests/test_wfa127_denominator_empty.py::TestTheCensusAgreesWithTheTrainer`.

## 3 — MEASURED, 2026-10-31, on disk

Scope: `molmetal/reports/**/per_epoch_metrics.csv` (the census's own
loader, `REPORTS.rglob("per_epoch_metrics.csv")`).

| quantity | value |
|---|---|
| `per_epoch_metrics.csv` files | **60** |
| per-epoch rows | **576** |
| rows with `gen_validity_val == 0` and `gen_validity_n > 0` (empty **SCORED** denominator) | **66** |
| rows with `gen_validity_n == 0` (empty **DRAWN** denominator) | **143** |
| runs containing ≥ 1 empty-scored row | **5** |
| runs whose **LAST** row has an empty scored denominator | **2** — `cfm_fm_control_2026_10_09`, `cfm_bond_head_30ep_onecycle_2026_10_09` (both `gen_validity_n = 48`, `gen_validity_val = 0`) |

Per-run empty-scored row counts: `cfm_blk_c5_repair_2026_10_09` 19,
`cfm_bond_head_30ep_onecycle_2026_10_09` 25, `cfm_fm_control_2026_10_09` 9,
`cfm_frag_assemble` 7, `cfm_frag_control` 6.

### The published artifact — BEFORE vs AFTER

The committed artifact was **NOT** rewritten (AGENTS.md §5.5: published
metrics are append/supersede, never silently edited).  The "AFTER" column
is a regeneration into a **scratch directory** (`/tmp/wfa127_probe/after2/`),
produced with `census.REPO` monkeypatched so `metrics/` was never a write
target.

Gate census over the 59 structured run rows
(`runs[].fragmentation_gate.{status,basis}`):

| status / basis | committed (BEFORE) | repaired regeneration (AFTER) |
|---|---|---|
| `FAIL` / `end_to_end` | **27** | **25** |
| `PASS` / `end_to_end` | **5** | **5** |
| `SKIP` / `denominator_empty` | **16** | **18** |
| `SKIP` / `field_absent` | **11** | **11** |

`fragmentation_summary`: `n_fail` 27 → 25, `n_skip` 27 → 29,
`n_runs_with_an_end_to_end_rate` 29 → 27.

### Delta isolation (leaf-by-leaf diff of the two JSON documents)

* committed artifact vs a regeneration from the **pre-repair** code:
  **0** differing leaves — the regeneration reproduces the published file
  exactly, so the committed artifact is a faithful product of the old code.
* pre-repair vs repaired regeneration: **254** differing leaves —
  **251** data + **3** this wave's own `changelog.wf_a127` entries.
  The 251 data leaves are the two runs' 3 end-rate fields each, their two
  `fragmentation_gate` / `fragmentation_gate_last_measured` blocks, and
  every cohort / `budget_matched` aggregate that carried their placeholder
  0.0.

### Knock-on aggregate changes the WAVE MUST RE-QUOTE

| aggregate | before | after |
|---|---|---|
| `by_n_train.256.last_epoch.median` | 0.026041666666666664 | 0.03645833333333333 |
| `by_n_train.256.last_epoch.mean` | 0.052734375 | 0.0703125 |
| `by_n_train.256.clean_only.best_epoch.median` | 0.7708333333333333 | 0.59375 |
| `budget_matched.cohorts.256.12.median` | 0.3125 | 0.8125 |
| `budget_matched.cohorts.256.12.n_values` | 8 | 4 |
| `budget_matched.flipped_budgets` | [9, 12, 14, 18, 20] | [9, 11, 25] |
| `already_answers.…n_train?[3]` (depth-1 loader scope) | n=24 | n=22 |

Any document that quotes a cohort median or the ordering flips is now
STALE and must be re-quoted from the regenerated artifact.

## 4 — Tests

New file: `molmetal/tests/test_wfa127_denominator_empty.py` — **18 tests**
(row normalisation, the gate, the two real runs on disk, the census-vs-trainer
agreement, and "the repair does not rewrite the published artifact").

MEASURED, CPU-only (`CUDA_VISIBLE_DEVICES=""`), 2026-10-31:

| scope | result |
|---|---|
| the 14 named regression files (census integrity + population, Wf-A125 registry sha, Wf-A127 residue closure, the sentinel, the Wf-A114 fragmentation gate, the epoch rules, the cohort compare, the pool kind, the published-tree guards, the Wf-A123 audit truth) | **397 passed, 0 failed** |
| `test_wfa127_denominator_empty.py` + `test_wfa127_residue_closure.py` | **38 passed** |
| `test_wfa127_residue_closure.py` alone | **20 passed** |

`metrics/` was verified byte-identical before and after every run
(`git status --porcelain metrics/` — 34 entries, unchanged;
`metrics/by_metric/cfm_run_census.json` mtime still `Oct 31 09:53`).

## 5 — What could NOT be verified

* **The published artifact still carries the 0.0** — by design (§5.5).  The
  repair is in the code path; regenerating the artifact is a wave action,
  and every aggregate that quotes the old 0.0 is stale until it is done.
* **No run on disk exercises the post-Wf-A114/T85 shape.**  The trainer now
  writes `null` / the literal `nan` for these rates, and the census's
  `_num()` maps both to `None`; a run written that way would be classified
  `field_absent`, not `denominator_empty`, because the census's
  `field_present` detection cannot distinguish "the column was never
  written" from "the column was written as null".  **The census therefore
  cannot distinguish a post-repair artifact's null from a missing column.**
  Not fixed here — it is a different defect class (the census's input-flag
  detection) and no run on disk would exercise it.
* **No GPU, no training, no sampling.**  Nothing here needs a GPU; the
  census is a CPU-only file reader.

## 6 — Premise corrections

1. The task's catalog cross-reference "§6.234" is **wrong**.  The entry
   that records this defect as an OPEN honest-negative is
   `docs/HONEST_NEGATIVE_CATALOG.md` **§6.231 (G8)** — its "SCOPE — what
   this entry does NOT cover" says: *"THE DEEPER DEFECT IS IN A TRAINER AND
   IS OUT OF SCOPE HERE.  The trainer records `0.0` where the truth is
   'undefined'; the census can only report the degeneracy, not fix it."*
   §6.234 is the G11 WARN gate.
2. The trainer at HEAD does **not** record 0.0 for an empty denominator —
   Wf-A114/T85 already repaired it (it writes `null` / `nan`).  The
   residual defect was in the **census**, which passes the historic
   placeholder through because it gates on the DRAWN count only.  The fix
   therefore lands in `molmetal/scripts/wf_cfm_run_census.py`, not in the
   trainer.
3. The quoted **11 / 16** split is **CORRECT**, verified independently
   against the committed artifact's structured run rows (not against prose):
   `('SKIP','field_absent') = 11`, `('SKIP','denominator_empty') = 16`.
