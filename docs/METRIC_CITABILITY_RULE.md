# Which numbers count as paper-citable

Status: ACTIVE 2026-10-20. Implemented in `molmetal/scripts/registry.py`
(`assess_citability`), surfaced by `registry.py cite`, counted by
`registry.py check`.

The question this answers is not "is this number real?" but "may I put it in
the paper, and if so, in a table or only in prose?" Those are different
questions, and conflating them is how a single-seed result acquires a `± 0.000`
and starts to look replicated.

---

## 1. The three marks

`registry.py cite` prints one line per paper-line metric.

| Mark | Meaning | Where it may appear |
|---|---|---|
| `[CITE]` | Table-grade. A real measurement, source in version control, protocol disclosed well enough to reproduce the number. | Results table, abstract, claim. |
| `[NUM ]` | Quotable with a stated caveat. The number is real, but at least one protocol fact is missing or the file itself declines the claim. | Prose only, with the caveat attached. Never a table mean. |
| `[NO  ]` | Do not quote as a result of this metric. | Nowhere, except as an explicit honest-negative. |

`[NUM ]` is not a softer `[CITE]`. It means: if this number goes in a table
without its caveat, the table is making a claim the evidence does not support.

---

## 2. The predicate

Two booleans, computed from the metric JSON alone. No prose is parsed and no
number is recomputed — the rule judges *documentation*, never the value.

### `quotable` — is this a measurement of this metric?

Fails if any of the following holds:

| Condition | Why |
|---|---|
| No status token | An unlabelled number cannot be distinguished from an aspiration. |
| `status` starts with `PROJECTED`, `DEFERRED`, `BLOCKED`, `REINDEX_PENDING`, `SUPERSEDED`, `UNMEASURED`, `DESIGN`, `CHECKPOINT`, `WIRED`, `PARTIAL`, `MEASURED_SYNTHETIC_LABELS`, `MEASURED_PILOT_MOCK` | These describe something other than a measurement. |
| `value` is `None` | No scalar headline. Unless the file declares `no_headline_reason`, in which case that reason is printed instead of a generic gap. |
| `cisplatin_reference_mean_is_control` / `value_is_control` | A positive control is not a model score. |
| `value_is_half_synthetic` / `value_is_synthetic` | A value averaged over real and synthetic folds is not a measurement of the real task. |
| `source_json` is not tracked by git | An untracked source cannot be cited. |

#### A status is a token, not a sentence

`status` is read from `status`, else from `verdict` / `honest_framing` / `tag`
— but only when the value is a **token**: either the whole string matches
`^[A-Z][A-Z0-9_+\-]{2,}$`, or it leads with one followed by a separator
(`MEASURED — …`, `VERIFIED_LIFT — …`, `MEASURED (3-metal x 3-seed …)`).

This is deliberately strict. `verdict` in this repo is also used for narrative
findings — *"THE HEAD IS NOT THE RING DEFECT. On clean geometry …"*, *"NO CLEAR
DIFFERENCE — the discordant flips do not clear the exact test"*. An earlier,
looser version copied those into `status` and 36 rows ended up asserting a
provenance their value did not support: `status` held a paragraph. Those rows
now report as "narrative verdict, no status token" (`status_narrative_source`
names the field), which is the honest description.

The `status_source` column records where the token came from (`status`,
`verdict`, `tag`, `verdict:leading_token`) so the aliasing is auditable rather
than invisible.

#### The `value` column has a provenance too

`value_basis` records the JSON path the headline number was read from. It
exists because the generic extractor preferred `aggregate.best_measured` — the
*best* of five pocket means — and `sa_mean` reported 2.2884 from the smallest
pocket (4aaw, n=14) while the file's own five-pocket mean is 2.4486. A best-of
is not a central tendency. `sa_mean` now declares
`headline_key: aggregate.closure_bounded_5pocket_simple_mean`; the extractor
also tries `aggregate.mean` before `aggregate.best_measured`, and whatever path
wins is written to `value_basis`.

### `headline_grade` — is the protocol disclosed well enough for a table?

Requires `quotable`, plus all of:

1. **Run count declared.** `n_seeds`, `n_runs`, `n_seeds_total`, a non-empty
   `seeds` list, or `aggregate.n_runs_measured` / `aggregate.n_seeds` /
   `aggregate.n_runs`.
2. **Not a single run.** `n < 2` yields "quote as a single-run measurement, not
   a table mean". A table mean of one run is not a mean.
3. **No std attached to a single run.** If `n == 1` and a *run-level* std field
   is present (`value_std`, or any name containing `std` together with `seed`
   or `run`), the row is flagged. A bare `std` is treated as per-sample spread
   (`m6_kappa`: 449 molecules) and is not an error bar.
4. **Dataset declared.** One of `dataset`, `data_source`, `cohort`,
   `cohorts_tested`.
5. **Split declared.** A `split` field, or a `protocol` string mentioning
   `fold` or `split`.
6. **The file does not decline the claim.** `status` starting with
   `MEASURED_NOT_A_CLAIM` or `MEASURED_BUT_NOT_A_QUALITY_METRIC` stays
   quotable but can never be table-grade.

### Optional disclosure field

`headline_caveat` (string) is printed under the row whatever the mark. Use it
when the number is table-grade but a reader needs one sentence of context — a
cohort alias, a superseded baseline, a single-seed warning.

---

## 3. Why these five facts

Anchored on the **ML Reproducibility Checklist v2.0** (Pineau et al., 2020),
which asks, for every reported result, for:

- *"the exact number of training and evaluation runs"*;
- *"a clear definition of the specific measure or statistics used to report
  results"*;
- *"a description of results with central tendency (e.g. mean) & variation
  (e.g. error bars)"*.

The run count and the variation are the two that get skipped most often, and
they are the two that a std of `0.0` fakes most convincingly. R's `sd()`
returns `NA` for a length-one vector, not `0`; this rule takes the same
position. `metalcytotox_ru_auc.json` carried `value_std: 0.0` at
`n_seeds: 1` until 2026-10-20 — the field is now `null` with a stated reason.

The dataset and split requirements are the project's own addition: a number
whose cohort is unnamed cannot be compared to anything, and this repo has
already published a gap-table row that compared ROC-AUC against an R² target
because the file, not the value, was misidentified.

---

## 4. Declaring the facts

Add the missing keys to `metrics/by_metric/<metric>.json` and re-run:

```bash
.venv/bin/python molmetal/scripts/registry.py generate
.venv/bin/python molmetal/scripts/registry.py cite
```

`generate` is now safe to run: `_preserve_underived_datasets` carries over rows
that no generator source declares and prints them to stderr. Before 2026-10-20
a plain regenerate shrank `datasets.yaml` from 21 rows to 11 — dropping the ten
hand-added `rusanov_2025_*` rows — and `check_datasets` then reported `11/11
PASS`. A generator must not delete what it cannot re-derive, and a smaller
registry that still looks green is the worst failure mode available.

Do **not** invent a protocol fact to reach `[CITE]`. If a run genuinely has no
seed to vary (a deterministic pass over a fixed cohort), declare `n_runs: 1`
and say so in `n_runs_note`; the row stays `[NUM ]`, which is the correct
answer for a single pass. `l6_closure` and `m6_kappa` are declared that way.

---

## 5. Provenance when the measurement date is unknown

Two different dates matter, and conflating them is how a 2026 probe inherits a
2024 citation year:

| Field | Means | Source |
|---|---|---|
| `metric_timestamp` + `metric_timestamp_source` | When the *measurement* was made | `timestamp` / `date` / `date_utc`, else the latest ISO date under a **date-named key** anywhere in the file (path recorded) |
| `date_added_to_repo` | When the *file* entered mainline history | `git log --first-parent` |

`date_added_to_repo` is a W3C PROV `generatedAtTime` for the file entity —
Git2PROV maps commit time to the activity that produced it. It is not a
substitute for the measurement date; it is what remains when none was recorded.

The nested-date rule searches *keys that claim to be dates*, not values that
look like dates. 119 of 142 undated files contain a date somewhere; only 23
contain one under a key that claims to be a date. The other 96 would have
picked up citation years and target dates out of prose.

`--first-parent` rather than `--diff-filter=A`: 20 metric files reached mainline
through a merge commit, and `git log --diff-filter=A` shows no file changes for
a merge, so those files looked like they had never been committed at all.

---

## 6. What the rule does not do

- **It does not judge the value.** A `[CITE]` honest-negative
  (`l5_hilbert_r2` at −5.66 against a 0.85 gate) is table-grade evidence that
  the method fails. The mark is about whether the number may be reported, not
  whether it is good.
- **It does not require a status on a row with nothing to qualify.** A status
  qualifies a *claim*. `carries_claim` is true only when the row has a scalar,
  so a probe log or a design note with no number is a record, not an unlabelled
  measurement. Counting the two together made the evidence string read like 174
  broken claims when the real number was 15.
- **It does not gate a launch.** Pre-flight checks provenance and freshness,
  not citability. `registry.py check` counts the marks in its evidence string;
  `registry.py cite` is the report.
- **It does not parse prose.** A file that describes its protocol only in an
  `honest_framing` paragraph does not pass. The five facts must be fields.

---

## 7. State (2026-10-20, MEASURED)

`registry.py cite` — 11 paper-line metrics:

| Mark | n | Metrics |
|---|---|---|
| `[CITE]` | 4 | `m19_lomo` 0.5575, `m2_r2_per_cell` −0.0300, `m21_nci60_9cell` 0.7294, `l5_hilbert_r2` −5.6625 |
| `[NUM ]` | 5 | `m14_decode_ratio`, `m7_diversity`, `m1_ru_auc`, `l6_closure`, `m6_kappa` |
| `[NO  ]` | 2 | `cfm_bond_accuracy` (no scalar by design, probe scale), `m3_vina_dock` (positive control) |

`registry.py check` on the same registry:

```
394 current, 0 stale, 0 missing | 11 paper-line (9 quotable, 4 headline-grade)
| 0 paper-line + 0 artifact claims without a status (15 hold a narrative verdict instead)
| 0 paper-line + 198 artifact rows record no measurement date (198 carry a commit date instead, 0 carry neither)
| 0 paper-line + 118 artifact rows labelled MEASURED with no standardised scalar (117 hold numbers under a file-specific key, 1 hold none)
```

The zeros are the point: every paper-line row has a status, a date and — if it
claims MEASURED — either a scalar or a declared reason it has none. Every
artifact row with a scalar now carries a status token too; the 15 that do not
are rows where a conclusion was written into `verdict` as prose, which is
recorded rather than promoted.

The 118 unstandardised scalars are not debt. They are files whose headline
lives under a file-specific key (`per_ring_size`, `decode_rate_by_arm`,
`quantity`, `per_metal`), where no generic extractor can know which number is
the claim. 117 of them do hold numbers; the registry's `value` column is empty
because the headline has not been *declared*, and `headline_key` is the
declaration.

Open items the marks expose, none of them resolved by editing this document:

- **`m1_ru_auc` is a single seed with an unrecoverable cohort filter.** The
  same file holds a traceable DOI-grouped 10-fold result at 0.7717 ± 0.0380
  which is *higher*, but also single-seed and on a cohort that does not
  replicate the paper (591 vs 23 cell lines). Promoting it is a paper-narrative
  decision, not a registry one.
- **`cfm_bond_accuracy` becomes declarable when the 20-epoch production run
  lands.** Until then its own caveat says exact-match and F1 at production
  scale are unmeasured.
- **`m2_r2_per_cell`'s headline double-weights one cohort** (hilbert is
  aliased to platinai_4cell; independent-only mean is 0.0270). Both are far
  below the gate, so the verdict holds either way.
- **`m7_diversity`'s 0.8690 is `aggregate.best_measured`** (best of 4 runs),
  while the file also reports `cohort_mean_3_runs` = 0.8674. The difference is
  0.0016 and the file declines the claim anyway, so it is disclosed via
  `value_basis` rather than changed.
