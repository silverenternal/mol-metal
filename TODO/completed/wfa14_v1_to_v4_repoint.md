# Wf-A14 — repointed the leaky v1 MetalCytoTox defaults at the clean v4

**Date:** 2026-10-03
**Status:** SHIPPED (working tree)
**Authorised by:** user directive 2026-10-03 — *"这真的得修复"*

---

## §0 What was wrong

`.data/molmetal/metalcytotox_5metal_multitask.csv` (v1) is the default data
input of a large number of trainers and benchmarks, and **its shipped split
leaks**. MEASURED by `check_data_leakage.py`, not inherited as folklore:

```
metalcytotox_5metal: n_total=26801 unique_smiles=6782 unique_scaffolds=3069
  random_split:   strict=FAIL  (1935 shared canon SMILES; 1213 shared scaffolds)
  scaffold_split: strict=PASS  (0 shared scaffolds)     <- a RE-computed split is clean
  shipped_split:  strict=FAIL  (48 shared scaffolds)    <- the SHIPPED one leaks
```

Independent confirmation from the gate that now covers it —
`p0_metalcytotox_v1_known_leaky`, live output:

```
PASS | KNOWN-LEAKY v1 artifact still leaky as expected: rows=26801,
      scaffold overlaps train/val=29, train/test=48, val/test=32 (total=109),
      sha256=f167c11931ec… (bytes frozen by p0_csv_hash_verify)
```

So the file's own published split shares scaffolds across train/val/test.
Training on it inflates test performance. It was also, until this wave, the
default in **134 places across 70 live files**.

## §1 Why this is not a science change

It would have been, if v4 were some other dataset. It is not — the project
already decided this and wrote it down, verbatim from
`TODO/completed/m14_lead_deployment_precondition.md:154`:

> *"The v4 cohort **supersedes** the v2 cohort (26,633 lines per TODO-207 §1.4
> post-Wf-261) and **the v1 cohort (26,802 lines, leaky)**."*

and v4 was already in use elsewhere (`wf219_temporal_split_driver.py`, the LOMO
CSV per `355b1_wf131_p3_deferred.md` and `379a1_wf131_p3_7runs.md`).

**This change completed a migration the project had already committed to; it
did not make a new decision.** Everything below follows from that.

## §2 What v1 and v4 actually are

| | v1 | v4 |
|---|---:|---:|
| rows | 26,801 | 26,632 |
| columns | 8 | **11** (+`SMILES_Ligands`, `Metal`, `IC50_Dark_value`) |
| relation | — | **strict subset of v1** — v4-only keys = 0, v1-only = 39 |

Not a re-split of identical bytes: v4 drops 39 `(smiles, metal)` tuples (169
rows) and adds provenance columns. A metric produced on v1 is therefore **not
numerically comparable** to one produced on v4. That is the honest cost of this
change and it is why §6 records it rather than burying it.

## §3 What changed

**134 occurrences across 70 files.** Representative set:

```
molmetal/scripts/train_{transport,se3sb,multiph,multimetal_lomo,ru_multitask,ru_5metal_se3sb}.py
molmetal/scripts/benchmark_{r18_metalcytotox,wf25_5seed}.py
molmetal/scripts/bench_{hilbert_astr_real_metalcytotox,astr_only_baseline,hilbert_only_baseline,lambda_cfm_e2e}.py
molmetal/scripts/wf25_{hilbert,multiph,combined}_real_eval.py
molmetal/tests/test_{wasserstein_closure_integration,data_leakage}.py
```

Method: a **dry-run-first** script (`/tmp/wfa14_v1_to_v4.py`), not a `sed`. The
dry run earned its place twice:

- **Hazard 1 — the hash-freeze list.** `pre_flight_check.py` holds v1 at L691
  and v4 at L695 of `gated_csv_specs()`. A blind replace would have produced
  **two v4 entries and silently dropped v1's freeze protection**, so the leaky
  artifact would have become free to drift precisely while being "cleaned up".
  `pre_flight_check.py` and `check_data_leakage.py` are therefore excluded —
  their v1 references are deliberate (freeze it, measure it).
- **Hazard 2 — candidate lists.** 6 of the 67 files carry fallback chains
  (e.g. `test_wasserstein_closure_integration.py:110-111`). Checked before
  applying: the `molmetal/data/` and `data/` fallbacks are **already dead for
  both v1 and v4** (only `.data/molmetal/` holds the files), so first-existing
  still wins and no new dead path was introduced.

The substitution is also structurally safe: the versioned names are
`..._multitask_v2.csv` / `_v3` / `_v4`, none containing the bare
`..._multitask.csv` substring, so it cannot produce `..._v4_v4.csv`. Verified
post-apply: 0 such strings.

## §4 What did NOT change

- **v1 is untouched on disk**, still hash-frozen by `p0_csv_hash_verify`
  (blocking ON). A run that deliberately passes the old path reproduces the old
  behaviour exactly. Only the defaults moved.
- **The gate's exclusion of v1 from the zero-leak gate stays**, now with a
  truthful rationale. The earlier "not a trainer input" justification was
  FALSE (a prior lane corrected it); this wave then made the roster it named
  historical.
- `METALCYTOTOX_V1_DEFAULT_CONSUMERS` was **rewritten as a historical record**.
  It listed the repointing as "the open remedy, out of this gate's file
  ownership, so it is REPORTED, not fixed" — this wave did the fix, so the
  roster would otherwise have become a stale list of live consumers.

## §5 Verification

```
configs parse                      9/9
test_wasserstein_closure_integration + test_data_leakage   42 passed (111.97s)
_v4_v4 double-substitution         0
frozen spec list still holds v1    yes (pre_flight_check.py:691 intact)
p0_metalcytotox_v1_known_leaky     PASS | 109 overlaps, sha256 f167c11931ec…
gate count                         105 runtime == 105 constant
```

## §6 Honest framing

- **MEASURED**: 134 occurrences / 70 files repointed; 42 tests pass; 9/9 configs
  parse; the v1 leak is 109 pairwise scaffold overlaps (29/48/32); v1 remains
  byte-frozen.
- **NOT CLAIMED**: no model was retrained and no shipped metric was recomputed.
  Every existing number produced by the affected scripts was produced on the
  **leaky v1 split** and is now known to be an optimistic estimate. Repointing
  the default does not retroactively fix those numbers — it stops new ones from
  inheriting the defect.
- **OPEN**: whether any of those v1-based numbers are cited in the paper is not
  established here and would need its own audit. That is a genuinely
  uncomfortable question and it is deliberately not glossed.
- **NOT ACTIONED**: 21 test modules still fail at collection (pre-existing,
  unrelated to this change) and the tree-wide whitespace/format cleanup is
  still pending. Both are recorded separately.
