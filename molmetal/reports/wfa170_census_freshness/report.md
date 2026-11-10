# Wf-A170 — census-freshness guard

- status: **FAIL**  (generated 2026-11-04T12:06:42+0000, git head `1b9470a0e2129e3c4d982da4453e034a6cc12e24`)
- guard: `molmetal/scripts/wfa170_census_freshness_guard.py`
- JSON artefacts scanned: **4251** (27 unparseable)
- artefacts carrying >= 1 SHA-256 pin: **92** (denominator)
- SHA pins checked: **2550** — FRESH 2096, STALE **255**, MISSING_TARGET 199, SKIPPED_TOO_LARGE 0, UNPAIRED 2476

## Primary scope (`target_sha256` + `target_file`)

- artefacts: **1** (denominator)
- stale: **1** — `molmetal/reports/wfa159_timeout_verdict_census/timeout_verdict_census.json`
  - `molmetal/reports/wfa159_timeout_verdict_census/timeout_verdict_census.json` -> `molmetal/scripts/pre_flight_check.py`: STALE (recorded `ece6fd18bc09…`, current `2c9bb8c2908c…`)

## Generalized scope (any `<stem>_sha256` + named file)

- artefacts: **91** (denominator)
- stale: **254** — `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T040421Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T040835Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T042355Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T042855Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T042925Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T043155Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T043425Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T043819Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T043925Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T044426Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T165220Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T165430Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T165930Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T170406Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T174511Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T175231Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T191553Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T191730Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T192303Z.json`, `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T192446Z.json` …

## Stale pins (255 total; the first 40 are listed here, the full list is in `report.json`)

| artefact | pointer | sha key | target | recorded | current |
|---|---|---|---|---|---|
| `molmetal/reports/wfa159_timeout_verdict_census/timeout_verdict_census.json` | `$` | `target_sha256` | `molmetal/scripts/pre_flight_check.py` | `ece6fd18bc09…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T040421Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `ca60e4c704f3…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T040835Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `ca60e4c704f3…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T042355Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `315ab6035196…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T042855Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T042925Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T043155Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T043425Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T043819Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T043925Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T044426Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T165220Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T165430Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T165930Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T170406Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `b9aff1642670…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T174511Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1b7aa5b8eb8d…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T175231Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1b7aa5b8eb8d…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T191553Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1fabd0fba08a…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T191730Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1fabd0fba08a…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T192303Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1fabd0fba08a…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T192446Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1fabd0fba08a…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T193042Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1fabd0fba08a…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T193216Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1fabd0fba08a…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T193459Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1fabd0fba08a…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T193827Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1fabd0fba08a…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T194508Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `8791b125d9b9…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T194646Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `8791b125d9b9…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T194852Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `8791b125d9b9…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T195028Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `8791b125d9b9…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261103T195205Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `8791b125d9b9…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261104T093720Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `97efdb7ad45b…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261104T093859Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `97efdb7ad45b…` | `2c9bb8c2908c…` |
| `molmetal/reports/pre_flight_gpu_ready/preflight_20261104T094039Z.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `97efdb7ad45b…` | `2c9bb8c2908c…` |
| `molmetal/reports/wf146_chemprop_lomo/pre_flight.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1b7aa5b8eb8d…` | `2c9bb8c2908c…` |
| `molmetal/reports/wf30_pocket_prep/830c/prep_meta.json` | `$` | `source_pdb_sha256` | `/home/hugo/codes/mol-metal/molmetal/data/mmp13_real/830c.pdb` | `276bc4b8d73d…` | `20257683bc3d…` |
| `molmetal/reports/wf30_pocket_prep/pocket_status.json` | `$.pockets[0]` | `source_pdb_sha256` | `/home/hugo/codes/mol-metal/molmetal/data/mmp13_real/830c.pdb` | `276bc4b8d73d…` | `20257683bc3d…` |
| `molmetal/reports/wf310e_r20_p3_drugood_full_launch/pre_flight.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `1b7aa5b8eb8d…` | `2c9bb8c2908c…` |
| `molmetal/reports/wf_preflight/preflight_wf409a3_metal_coord_retrain.json` | `$.provenance` | `gate_source_sha256` | `molmetal/scripts/pre_flight_check.py` | `ca60e4c704f3…` | `2c9bb8c2908c…` |
| `molmetal/reports/wfa164_coordination_sphere_narrative/default_pool_diff.json` | `$.the_four_literals_IN_A_DIFFERENT_POOL_ARTIFACT` | `pool_sha256` | `molmetal/reports/wfa164_coordination_sphere_narrative/synthetic_pool_artifact.json` | `f02f56af027f…` | `8bb9fece4584…` |
| `molmetal/reports/wfa164_posebusters_narrative/default_pool_diff.json` | `$` | `published_artifact_sha256` | `metrics/by_metric/wfa155_posebusters_validity.json` | `3772fa658077…` | `6464edc16252…` |

## Advisory posture

- A FAIL-ing freshness gate over the **GENERALIZED** scope **WOULD** block launches today — 255 stale pin(s) of 2549 checked across 91 artefact(s).
- A FAIL-ing freshness gate over the **PRIMARY** scope only **WOULD** block launches today — 1 stale pin(s) of 1 checked across 1 artefact(s).
- The proposed pre-flight hook is ADVISORY (WARN-equivalent): it never changes the exit code and never changes `EXPECTED_GATE_COUNTS`.

## Deployment

- Patch SPEC: `patch_preflight_gate.diff` in this directory.  It adds the freshness check to `molmetal/scripts/pre_flight_check.py` as an ADVISORY hook in `main()` — NOT registered in `build_gates()`, so `EXPECTED_GATE_COUNTS` stays 129 (`all`) / 131 (`launch`) and no PINNED gate count moves.
- Proven with `git apply --check` against a SCRATCH copy of the file (rc=0); the working tree was NOT modified.
- Cost: the scan re-hashes every sha-pinned artefact.  MEASURED on this host: ~15 s against a ~411 s full pre-flight.
- To apply: `git apply molmetal/reports/wfa170_census_freshness/patch_preflight_gate.diff` (the main thread owns `pre_flight_check.py`).
- GITIGNORE: `.gitignore:158` ignores `molmetal/reports/**/*.json`, so THIS `report.json` is an on-disk-only artefact unless it is force-added (`git add -f`).  The same rule already applies to the Wf-A159 census artefact it replaces.  `report.md` and `patch_preflight_gate.diff` are NOT ignored and carry the headline numbers durably.

## Residual blind spot — 2476 SHA pin(s) with no named file

These record a SHA-256 but name no sibling file, so the pin cannot be checked.  They are NOT counted as stale — they are the honest limit of this guard.

- `a_side_sha256`: 1 pin(s)
- `after_run1_sha256`: 1 pin(s)
- `after_run2_sha256`: 1 pin(s)
- `after_sha256`: 1 pin(s)
- `b_side_sha256`: 1 pin(s)
- `before_sha256`: 2 pin(s)
- `branch_a_proof_sha256`: 1 pin(s)
- `caveat_sha256`: 2 pin(s)
- `cli_run_artefact_sha256`: 1 pin(s)
- `committed_sha256`: 4 pin(s)
- `config_run_artefact_sha256`: 1 pin(s)
- `config_sha256`: 7 pin(s)
- `correction_of_ledger_sha256`: 4 pin(s)
- `disk_sha256`: 499 pin(s)
- `driver_sha256`: 7 pin(s)
- `eprint_sha256`: 302 pin(s)
- `file_sha256`: 93 pin(s)
- `git_diff_binary_sha256`: 1 pin(s)
- `git_status_sha256`: 1 pin(s)
- `headline_caveat_sha256`: 4 pin(s)
- `live_worktree_sha256_at_probe_time`: 1 pin(s)
- `payload_caveat_sha256`: 3 pin(s)
- `payload_sha256`: 35 pin(s)
- `payload_sha256_after`: 6 pin(s)
- `payload_sha256_before`: 6 pin(s)
- `payload_sha256_now`: 6 pin(s)
- `physical_evaluator_sha256`: 7 pin(s)
- `pool_sha256_at_run_time`: 2 pin(s)
- `prior_ledger_sha256`: 2 pin(s)
- `r3_sha256_after`: 1 pin(s)
- `r3_sha256_before`: 1 pin(s)
- `registry_caveat_sha256`: 3 pin(s)
- `registry_metrics_yaml_sha256_after`: 1 pin(s)
- `registry_metrics_yaml_sha256_before`: 1 pin(s)
- `registry_py_sha256`: 1 pin(s)
- `registry_sha256`: 2 pin(s)
- `registry_source_json_sha256_now`: 2 pin(s)
- `retrieval_sha256`: 3 pin(s)
- `run_ledger_jsonl_sha256_after`: 1 pin(s)
- `run_ledger_jsonl_sha256_before`: 1 pin(s)
- `runner_sha256`: 7 pin(s)
- `scripts_start_sh_sha256`: 1 pin(s)
- `seed0_sha256`: 1 pin(s)
- `seed1_sha256_current`: 1 pin(s)
- `sha256`: 208 pin(s)
- `sha256_at_commit`: 454 pin(s)
- `sha256_current`: 1 pin(s)
- `sha256_of_the_on_disk_extracted_text`: 2 pin(s)
- `sha256_pre_retrain_backup`: 1 pin(s)
- `sibling_sha256`: 1 pin(s)
- `source_file_sha256`: 1 pin(s)
- `source_json_sha256`: 8 pin(s)
- `superseded_census_sha256`: 4 pin(s)
- `target_sha256_after_save`: 1 pin(s)
- `test_row_id_sha256`: 25 pin(s)
- `value_sha256`: 741 pin(s)

