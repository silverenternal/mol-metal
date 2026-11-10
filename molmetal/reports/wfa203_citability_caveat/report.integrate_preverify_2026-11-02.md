# Wf-A203 — the citability caveat for the two headline-grade rows that had none

**Date:** 2026-11-02 · **GPU:** none (every python/pytest call prefixed `CUDA_VISIBLE_DEVICES=""`) ·
**venv:** `/home/hugo/codes/mol-metal/.venv/bin/python` only

## 0. The gap (MEASURED census)

`registry/metrics.yaml` holds **582** metric rows, **11** of them `registry_role: paper_line`.
Of those 11, **exactly two** were `quotable: true` AND `headline_grade: true` AND carried no
`headline_caveat`:

| metric_id | value | status | caveat before |
|---|---:|---|---|
| `m19_lomo` | 0.5575471269535148 | `MEASURED_BELOW_TARGET_BASELINE` | absent |
| `m21_nci60_9cell` | 0.7293802091886383 | `MEASURED_HONEST_NEGATIVE` | absent |

There is **no third row** in that class. `registry.py cite` printed a bare number for both, so a
reader got `0.5575` / `0.7294` with none of the disclosures the payloads themselves already carry.

A wider class (`quotable: true`, `headline_grade: false`, no caveat) has **6** rows; four of them
(`l6_closure`, `m14_decode_ratio`, `m6_kappa`, `m7_diversity`) are `headline_grade: false`, so
`cite` prints a `not headline:` reason instead of a caveat, and `m1_ru_auc` already has one.
`docs/METRIC_CITABILITY_RULE.md:288-292` calls `headline_caveat` an **optional** disclosure field
and says it is printed "whatever the mark" — so the absence of one is a **disclosure gap, not a
rule breach**. There is no mechanical enforcement for it in the rule.

## 1. What changed

| file | before | after |
|---|---|---|
| `registry/metrics.yaml` | `b3a8522f9ebfed2f220311e1e1da2414e3709e3a8a0ec455538b5a42732d4f72` (682 753 B) | `9ddd971eae8e333ed7bf169660da28f95ab500528e53d6a4eb1c17df1480e684` (684 704 B) |
| `molmetal/reports/run_ledger.jsonl` | `3a5a8cd0690a0b854c5bfde6e6f73dd62f07e340be7699106b7e8098be135479` (237 992 B, 432 lines) | `c464cd5ea07875a24467cfac9ce710399e821a01050c9f8d73f278ae0fae087e` (243 800 B, 434 lines) |
| `molmetal/tests/test_wfa203_citability_caveat.py` | — | NEW |
| `molmetal/reports/wfa203_citability_caveat/report.{json,md}` | — | NEW |

The two metric payloads were **not** touched by this agent — the m21/m19 repair agents had already
landed them (sha `8ff6b226…` and `2522c808…`, byte-identical before and after this wave).

### 1.1 The two caveats (verbatim, byte-identical in payload and registry row)

**`m21_nci60_9cell`** (837 chars):

> 0.7294 is the 9-cell panel mean and it clears this metric's own 0.7 gate. But DU145 (n_test=7) and SKOV3 (n_test=3, n_val=0) both score AUC = 1.0 on single-digit test sets, where such an AUC is not informative, while the other seven cells average 0.6521; the identity (7 x 0.6520602689568208 + 1.0 + 1.0) / 9 == 0.7293802091886383 therefore holds exactly and the headline is carried by the two smallest test sets. Under the recorded minimum-test-set filters the panel mean is 0.6521 over 7 cells (n_test>=10) and 0.5923 over 5 cells (n_test>=20), both below the 0.81 Krasnov 2026 comparator, so on this panel the honest reading is an explicit miss, not a tie, and not a win. The comparison is also protocol-mismatched: this panel is a Bemis-Murcko scaffold split, whereas Krasnov's 0.81 is a within-metal stratified CV on a random split.

**`m19_lomo`** (957 chars):

> RETRACTED HEADLINE: the 0.5575 in this file is the 5-metal EQUAL-WEIGHT mean, and docs/PREDICTIVE_LINE_INDEX.md §3 (retraction log, dated 2026-10-19) withdrew the claim 「LOMO 0.5575，差 0.04」 and replaced it with the resolvable-subset 0.5974, because 0.5575 averages in the Rh 51 / Os 36 / Re 74 folds whose unique counts are below MIN_UNIQUE_FOR_GATE=200. The quotable figure per that index is 0.5974 (resolvable subset: Ru 1,311 / Ir 538), with 0.5947 as the seed-ensemble robustness, and the index's own instruction is 「以本索引和台账为准」, i.e. the index and the claim ledger (docs/CLAIM_LEDGER.md) take precedence over registry.py cite; 0.5575 must not be quoted as the LOMO headline. This 0.5575 also sits below the metric's own gate constant of 0.70 (gate_target: 0.7), which gate_target_provenance records as an INTERNAL GATE_AUC constant and NOT a literature value, and real_only_fold_value is 0.3812 (the Rh fold), the weakest fold in that equal-weight mean.

`registry.py cite` now prints both under their rows.

## 2. Method — registry rows

**Two steps, in this order.**

1. **The caveat line was inserted surgically** into each row's `headline_caveat: null` slot, rendered
   with the generator's own serialiser — `yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)` —
   so the bytes are exactly what `registry.py generate` would emit for the same value. The
   whole-file round-trip check confirms it: the written `registry/metrics.yaml` still satisfies
   `yaml.safe_dump(yaml.safe_load(text), sort_keys=False, allow_unicode=True) == text` (**True**).
   Written via `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file` (POSIX `os.replace`);
   `safe_overwrite_check` -> `None`; `detect_truncation` -> `None`.
2. **The sha was refreshed with the repo's own tool**:
   `registry.py repoint m21_nci60_9cell --apply` and `registry.py repoint m19_lomo --apply`.

**Why that order, and why repoint is safe here.** `repoint` is a text-level surgical edit
(`molmetal/scripts/registry.py:2181-2197` rewrites only the row's `source_json_sha256:` line;
`:2199-2205` the row's `last_refreshed:` line; `:2210-2216` the FILE-level `generated_at:` header).
It never touches `headline_caveat`, and it re-parses the result before writing (`:2218-2229`), so a
hand-added caveat line survives it — confirmed here by the fact that both caveats are present in the
file after the repoint.

**Diff proof — only the two rows changed.** Parsed-YAML before/after:

- rows changed: `['m19_lomo', 'm21_nci60_9cell']` — every other one of the 582 rows is identical;
- keys changed per row: `['headline_caveat', 'last_refreshed', 'source_json_sha256']`;
- non-metric sections changed: **`generated_at`** only. This is tool-mandated: `repoint` moves the
  file-level header together with the row it refreshes (Wf-A127) so the header does not date itself
  before one of its own rows. It is the one change outside the two rows and is reported here
  explicitly.

## 3. Durability — a regenerate would NOT erase the caveats (T2)

`registry.py generate` rebuilds each row from its payload. The chain:

```
molmetal/scripts/registry.py:1239    caveat = payload.get("headline_caveat")
molmetal/scripts/registry.py:1245    "headline_caveat": (caveat.strip() if isinstance(caveat, str) and caveat.strip() else None)
molmetal/scripts/registry.py:1426    rec.update(assess_citability(payload, rec["value"], _rel(src_path), tracked))
molmetal/scripts/registry.py:1515    (the same call in the disk-walk pass)
molmetal/scripts/registry.py:2036    metrics, covered = build_metrics(repo_root, datasets, protocols)
molmetal/scripts/registry.py:2072    text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
molmetal/scripts/registry.py:2079    atomic_write_file(path, text)
```

`m19_lomo` and `m21_nci60_9cell` are in `LINE_METRICS` (`registry.py:112`, `registry.py:124`), so
they take the `:1426` path.

**Live proof (read-only).** `build_metrics(repo_root, datasets, protocols)` was called directly —
`registry.py generate` was **NOT** run. The two rows it produces are identical to the on-disk rows
**except for the clock stamp** `last_refreshed`; `headline_caveat` matches byte-for-byte in both.
`assess_citability(payload, value, ...)` was also called on both payloads and returned exactly the
payload's caveat string, with `quotable: true` and `headline_grade: true`.

## 4. Method — ledger rows (APPEND ONLY)

Two rows appended as the **last two lines** (433 = `m21_nci60_9cell`, 434 = `m19_lomo`) using the
repo's own `molmetal.scripts.run_ledger.record_run` (O_APPEND + fsync, `run_ledger.py:162-172`).

**Append-only proof:** the first 237 992 bytes of the new file hash to
`3a5a8cd0690a0b854c5bfde6e6f73dd62f07e340be7699106b7e8098be135479` — byte-identical to the
pre-append file. Nothing was rewritten, reordered or deleted.

Fields, and whether they are required or optional (per `run_ledger.py:175-235`):

| field | value | status |
|---|---|---|
| `metric_id` | `m21_nci60_9cell` / `m19_lomo` | REQUIRED (`run_ledger.py:206`) |
| `value` | 0.7293802091886383 / 0.5575471269535148 | REQUIRED (positional, no default) |
| `source_json` | the relative path | REQUIRED (`run_ledger.py:180`) |
| `source_json_sha256` | the CURRENT on-disk sha | REQUIRED by the provenance gate (`run_ledger.py:517-521`) |
| `git_sha` | `1b9470a0e2129e3c4d982da4453e034a6cc12e24` | optional (defaults to HEAD) |
| `data_sha` / `config_sha` | `null` | optional — no data or config file changed |
| `protocol_id` | `null` | optional |
| `host` | `ZhaoLab` | optional (matches rows 17/19) |
| `timestamp` | `2026-11-01T18:37:48.173157+00:00` | optional — **real clock** (`date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00`) |
| `seed` | `true` | optional — "not a fresh training run", matching rows 17/19 |
| `extra` | see below | optional |

**`extra`** carries: `wf: "Wf-A203"`, `edit_kind: "metadata_only_citability_caveat"`,
`value_unchanged: true`, `committed: false`, `commit_blob_equals_disk: false`,
`committed_git_sha` = HEAD, `committed_sha256` = the **OLD** sha, `disk_sha256` = the new sha,
`correction_of_ledger_sha256` = the prior latest row's sha, `previous_git_sha`, `measurement_timestamp`,
`edit_timestamp`, `value_key`, `data_source: null`, and a `reason` that states in words: Wf-A203
(2026-11-02), **METADATA ONLY**, the **VALUE is UNCHANGED**, and the bytes on disk are **NOT
COMMITTED** — no commit was authorised in this wave, so `commit_blob_equals_disk` is FALSE and
`committed_sha256` names the blob that *is* committed (the HEAD blob, i.e. the **prior** bytes).

**No commit equality is fabricated.** MEASURED: `git cat-file -p HEAD:<path>` hashes to the OLD sha
for both files (`a2bdc1ad…` and `90a9c0c0…`), so the row's `git_sha` names the commit that
reproduces the **prior** bytes — never the current ones. The m19 row additionally records the
pre-existing looseness it replaces: row 17's own `git_sha` (`e0d473cc…`) names a commit whose blob
for that path hashes to `0a3d3cae…`, NOT `90a9c0c0…`.

**Post-write gate state (MEASURED):** `uncommitted_provenance` — `open_uncommitted_metrics` is still
**2** (unchanged), and the two OPEN rows are `wfa155_ligand_connected_rate` and
`wfa155_metal_coordination_sphere` — **not** ours. `run_ledger.staleness_summary()` is back to
`{'current': 399, 'reformatted': 0, 'stale': 0, 'no_run': 183}`.

## 5. The test

`molmetal/tests/test_wfa203_citability_caveat.py` — **27 assertions, all pass**.

```
CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
    molmetal/tests/test_wfa203_citability_caveat.py -q
27 passed, 1 warning in 13.28s
```

T1 `cite` prints a `caveat:` line for both, carrying the required numbers (m21: `0.6521`, `0.5923`,
`0.81`, `DU145`, `SKOV3`, `AUC = 1.0`, `n_test=7`, `n_test=3`; m19: `0.5974`, `0.5947`, `0.3812`) ·
T2 the measured values are unchanged (`0.7293802091886383`, `0.5575471269535148`) · T3 each registry
row's sha equals the payload's sha on disk · T4 no forbidden claim word (the explicit list — beat,
beats, beating, outperform(s/ing), state-of-the-art, state of the art, sota, best-in-class, best
performing, best-performing, significant improvement, significantly improves, superior performance,
best — matches either caveat; the check is case-insensitive with alphanumeric-boundary guards) ·
T5 `registry.py check` reports no STALE/MISSING marker for either metric · T6 the ledger's latest row
names the current on-disk sha and the unchanged value · T7 the registry row's caveat is byte-identical
to the payload's and `assess_citability` reproduces it from the payload alone (the durability proof) ·
T8 the Wf-A203 ledger row does not fabricate a commit equality.

**Bite test (negative control, run and reverted).** Blanking the m21 registry caveat and corrupting
the m19 ledger's latest sha produced **5 failures** (3 cite/registry + 1 ledger + 1 registry-row);
both files were then restored byte-identically (`RESTORED sha == EXPECTED sha` for each). The
assertions are not vacuous.

**One assertion was adapted, and here is exactly how.** The task asked for "registry.py check exits
0". It **cannot** exit 0 on this tree: two PRE-EXISTING rows (`wfa155_ligand_connected_rate`,
`wfa155_metal_coordination_sphere`) are STALE because the Wf-A155 wave edited their metric JSONs,
and that wave's files are explicitly out of this agent's scope. The test therefore asserts the
**exact STALE set** instead (no row outside the known pre-existing set is stale, no row is missing,
and the exit code agrees with that set) — which is strictly stronger against regression than a bare
"exits 0", and it is stated in the test's own docstring.

**Environment note (not caused by this wave).** The repo-wide Wf-A123 session guard
(`molmetal/tests/conftest.py::_published_tree_guard`) compares every published metric artifact's
sha256 at session start against session teardown and raises a teardown ERROR if any moved — no
matter which test moved it. The Wf-A155 wave's `wf_a155_posebusters_validity.py` is **running in
this working tree right now** (`ps` confirms) and rewrites its own
`metrics/by_metric/wfa155_metal_coordination_sphere.json`, alternating between two serializations, so
a session that straddles one of its writes trips that guard. Evidence that this is not the test's
doing: (a) this module writes nothing under `metrics/` — `registry.py check` and `registry.py cite`
leave that file's sha256 **and** mtime unchanged (MEASURED); (b) the unrelated pre-existing
`molmetal/tests/test_wfa188_registry_ledger_coverage.py` fails on the **same two** `wfa155_*` rows for
the same reason; (c) the 27 assertions pass in every session — only the guard's teardown error is
environment-dependent. The clean run above is a session that straddled no Wf-A155 write.

## 6. Invariants (T5)

| invariant | required | observed | verdict |
|---|---|---|---|
| `git status --porcelain metrics/` md5 | `a022e54fd2938b9f333fd8438f6d3d27` | `753b04b2a241039eb538601f35bb0504` | **DEVIATION — structural, explained below** |
| metrics/ files touched by me | 0 | **0** | PASS |
| sha256 `metrics/by_metric/nci60_precious_9cell_auc.json` | unchanged | `8ff6b226ee282348709f2acb865c14e0bf937fe92b81598cb0bc0138cbcd637a` before **and** after | PASS |
| sha256 `metrics/by_metric/metal_lomo_auc_wasserstein.json` | unchanged | `2522c80890e18b4a293163a625f0d23831bae34d00c4ccd1e7b4ea3033b9449e` before **and** after | PASS |
| only the two registry rows changed | yes | `['m19_lomo', 'm21_nci60_9cell']` (+ the tool-mandated `generated_at` header) | PASS |
| ledger append-only | yes | first 237 992 bytes hash to the pre-append sha | PASS |

**The md5 deviation, stated plainly.** The required baseline `a022e54f…` corresponds to a 38-line
porcelain list. The two **authorised** payload repairs in this wave (the m21 and m19 repair agents)
each turned a file that was clean-vs-HEAD into a modified one, so the list grew to 40 lines and the
md5 moved to `753b04b2…`. The porcelain line is path-only, so the md5 moves whenever a path is *added*
to the list — the task's own note ("a content edit does not move it") holds only for paths already
listed. **This agent wrote nothing under `metrics/`**: the md5 read `753b04b2…` before my first write
and reads `753b04b2…` after my last, and both payloads' sha256 are byte-identical before and after.

## 7. OPEN items — NOT closed

1. **Producer/artefact parity is still UNMET.** `molmetal/scripts/train_nci60_10cell.py:1214-1270`
   and `molmetal/scripts/train_metal_lomo_wasserstein.py:3159-3282` build a **fresh** payload dict and
   would **ERASE** the new `headline_caveat` (and ~21-24 other hand-added keys, including the nci60
   `test_set_size_funnel` the caveat cites) on a re-run. Neither producer was edited: both are
   outside this agent's write scope, and the m19 producer's payload does not reproduce the on-disk
   artefact at all, so a trainer-emitted copy of this caveat would be a **dangling** reference. The
   durable fix is a value-agnostic trainer-side caveat template — a separate change.
2. **The bytes on disk are UNCOMMITTED.** No commit was authorised in this wave, so
   `registry/metrics.yaml`, the two payload caveats and the two ledger rows are working-tree state
   only. The ledger rows say so explicitly.
3. **`registry.py check` still exits 1** (`REGISTRY: STALE`) because of the two pre-existing
   `wfa155_*` rows owned by the Wf-A155 wave. Not this wave's to fix.
4. **`docs/PREDICTIVE_LINE_INDEX.md` has no m21 entry at all**, so the m21 caveat is currently the
   only machine-readable ruling on which m21 number is quotable. That index is out of scope here.
5. **No mechanical rule prevents a third caveat-less headline row.** `docs/METRIC_CITABILITY_RULE.md`
   still calls `headline_caveat` optional; the guard here covers exactly the two rows this wave
   repaired.

## 8. Files written by this agent

- `registry/metrics.yaml` (the two rows + the tool-mandated `generated_at`)
- `molmetal/reports/run_ledger.jsonl` (two appended lines)
- `molmetal/tests/test_wfa203_citability_caveat.py` (new)
- `molmetal/reports/wfa203_citability_caveat/report.json` (new)
- `molmetal/reports/wfa203_citability_caveat/report.md` (this file)
