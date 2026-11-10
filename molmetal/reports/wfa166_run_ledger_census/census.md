# Wf-A166 -- run-ledger provenance census

MEASURED 2026-11-01T03:36:11.118482+00:00 -- READ-ONLY, CPU-only, 0 GPU.

- Ledger: `/home/hugo/codes/mol-metal/molmetal/reports/run_ledger.jsonl`
- **Denominator: 418 rows** / 396 distinct metrics /
  398 distinct `source_json` paths / 6 distinct `git_sha` values.

## Per-axis counts (denominator = 418 rows)

| axis | count | share |
|---|---|---|
| A. row sha == file on disk now | 387 | 92.6% |
| B. row sha == some committed version of that path | 409 | 97.8% |
| C. row `git_sha` resolves to a commit object | 418 | 100.0% |
| D. that commit's changed files contain the row's `source_json` | 1 | 0.2% |
| E. that commit is in some branch or tag | 417 | 99.8% |
| F. row `git_sha` == the ledger's modal sha | 384 | 91.9% |
| G1. row carries a non-null `data_sha` | 30 | 7.2% |
| G2. the `data_source` path resolves to a file on disk | 30 | 7.2% |
| G3. that file's sha256 == the row's `data_sha` | 30 | 7.2% |
| G4. row carries a non-null `config_sha` | 6 | 1.4% |
| G5. (of the rows carrying a config_sha) the protocol's `trainer_config` path resolves on disk | 5 | 1.2% |
| G6. (of the rows carrying a config_sha) that file's sha256 == the row's `config_sha` | 0 | 0.0% |

Distinct `source_json` paths whose file is MISSING on disk: 0.

## `git_sha` histogram

| git_sha | rows | share | commit exists | reachable |
|---|---|---|---|---|
| `d86c879a540e79ebefcda820e83ddff24f5477ad` | 384 | 91.9% | True | True |
| `8b9ceb52830f3203cf028ed12943fdcaff0e0ebb` | 11 | 2.6% | True | True |
| `e0d473cc7baec3d454c9515694ca17fdadd631c3` | 11 | 2.6% | True | True |
| `1b9470a0e2129e3c4d982da4453e034a6cc12e24` | 10 | 2.4% | True | True |
| `7e30d6ea509a8a42d953a77a166d69457ef7f037` | 1 | 0.2% | True | True |
| `975414bc6fb8ef20ca53429f287a4d3297184cad` | 1 | 0.2% | True | False |

## Rows whose named commit DOES touch their `source_json`

| line | metric_id | source_json | git_sha |
|---|---|---|---|
| 23 | l5_hilbert_r2 | `metrics/by_metric/hilbert_4cell_r2.json` | `7e30d6ea509a8a42d953a77a166d69457ef7f037` |

## Rows whose named commit is UNREACHABLE from any branch or tag

| line | metric_id | source_json | git_sha |
|---|---|---|---|
| 408 | chembl29_metal_scan | `metrics/by_metric/chembl29_metal_scan.json` | `975414bc6fb8ef20ca53429f287a4d3297184cad` |

Resolver agreement: **AGREE** (398 paths, 754 blobs).

## The column contract

| column | writer | kind | validated by |
|---|---|---|---|
| `metric_id` | `molmetal/scripts/run_ledger.py:214` | live pointer | run_ledger.py:305-326 (staleness() joins the ledger to the registry rows on this key) |
| `value` | `molmetal/scripts/run_ledger.py:215` | historical | run_ledger.py:322 (_same_value) -- decides the 'reformatted' vs 'stale' split |
| `source_json` | `molmetal/scripts/run_ledger.py:216` | live pointer | run_ledger.py:107-118 (_resolve_source_json) + :517-528 (uncommitted_provenance resolves it and compares shas) |
| `source_json_sha256` | `molmetal/scripts/run_ledger.py:217` | historical | run_ledger.py:320 (vs the registry row) and run_ledger.py:523-528 (vs disk AND vs git history) |
| `git_sha` | `molmetal/scripts/run_ledger.py:218` | historical | NOTHING. It is read only to be echoed into a report (run_ledger.py:547, :874; wfa156_ledger_provenance_guard.py:576). registry.py never opens the ledger (0 hits). |
| `config_sha` | `molmetal/scripts/run_ledger.py:220` | historical | NOTHING in the repo. The PATH is NOT stored in the row, so the hash is not self-describing. |
| `data_sha` | `molmetal/scripts/run_ledger.py:219` | historical | NOTHING in the repo. The PATH is not stored in the row either (only echoed into extra.data_source, run_ledger.py:651). |
| `timestamp` | `molmetal/scripts/run_ledger.py:223` | historical | NOTHING. Echoed at run_ledger.py:317. Never parsed, never checked monotonic. |
| `protocol_id` | `molmetal/scripts/run_ledger.py:221` | UNDEFINED | NOTHING. No producer, no consumer, no validator, no doc. |
| `host` | `molmetal/scripts/run_ledger.py:222` | historical | NOTHING. |
| `seed` | `molmetal/scripts/run_ledger.py:224` | historical | NOTHING. seed_from_disk skips on metric_id, not on this flag (run_ledger.py:626-629). |
| `extra` | `molmetal/scripts/run_ledger.py:226-227` | historical | NOTHING. Read only to be echoed (run_ledger.py:547). |

## The three candidate commits, side by side

- **A** -- the commit recorded in the ledger row (HEAD at ledger-row-write time) (recorded in every row).
- **B** -- the commit the metric JSON itself records in provenance.git_commit (only some metric JSONs carry it).
- **C** -- the unique commit whose version of source_json has exactly the row's source_json_sha256.

Rows carrying a candidate B: **54**. A == B in **0** of them; A == C in **0** of all 418 rows; B == C in **0**.

Recoverability: **409/418** rows have a recoverable producing commit (**409** of them have EXACTLY ONE holding commit); **9/418** have none.

> recovering a holding commit identifies the commit whose TREE held the metric file at that content; it does NOT identify the commit whose tree the measurement RAN on.  The ledger never recorded the latter for any row.

The full per-row table (every axis, every row) is `census.json`.
