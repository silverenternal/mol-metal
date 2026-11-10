# Wf-A203 — citability caveats for the two headline-grade registry rows that carried none

**Date:** 2026-11-02 · **GPU:** none (every python/pytest call prefixed
`CUDA_VISIBLE_DEVICES=""`) · **venv:** `/home/hugo/codes/mol-metal/.venv/bin/python` only ·
**HEAD:** `1b9470a0e2129e3c4d982da4453e034a6cc12e24`

**Measurement policy.** Every number below is MEASURED in the 2026-11-02 REPORT run and carries
a `file:line` or a named-command citation. **No number in this report is PROJECTED.** Where a
figure was measured by an upstream Wf-A203 agent and not re-measured here, it is attributed to
that agent inside `verifiers` / `open_items`. Note also that m19's `0.70` gate is itself declared
**INTERNAL** (not a literature value) by the payload at
`metrics/by_metric/metal_lomo_auc_wasserstein.json:8`.

---

## 1. THE DEFECT

Two paper-line registry rows published their headline numbers with **no caveat attached**, so
`registry.py cite` printed a bare number for each while the metrics' own payloads already
carried a contradicting disclosure. MEASURED at HEAD: `registry/metrics.yaml` carries 498 metric
rows, 11 of them `registry_role: paper_line`; **exactly two** were `quotable: true` AND
`headline_grade: true` AND carried no caveat — `m19_lomo` (`headline_caveat: null` at
`git show HEAD:registry/metrics.yaml:245`, row block starts `:216`) and `m21_nci60_9cell`
(`headline_caveat: null` at `git show HEAD:registry/metrics.yaml:326`, row block starts `:295`);
the wider `quotable && no caveat` class held 6 rows, the other 4 being `headline_grade: false`.
The m21 payload's own nested reading at
`metrics/by_metric/nci60_precious_9cell_auc.json:538` says the panel mean of 0.7294 is *"an
explicit miss — not a tie, and not a win"* under the recorded minimum-test-set filters (0.6521
over 7 cells at `:459`, 0.5923 over 5 cells at `:508`, both below the 0.81 comparator at `:8`
and `:536`), and `honest_negative.statement` at `:432` records that the headline drops from
0.8007 to 0.7294 on the fixed scaffold split; the m19 headline 0.5575 had already been
**RETRACTED** in `docs/PREDICTIVE_LINE_INDEX.md:1405` (section 3 header at `:1396`) — *"2026-10-19
| 「LOMO 0.5575，差 0.04」 | 可解析子集 **0.5974**"* — and replaced by the resolvable-subset 0.5974,
yet the registry still printed 0.5575 as the quotable LOMO number with no pointer to either
ruling; the index's known-conflict table at `:185-196` records that `registry.py cite` says quote
0.5575 while the claim ledger says quote 0.5974, and rules 「以本索引和台账为准」 at `:193`. The
defect is a **disclosure gap in the machine-readable layer**, not a wrong measurement — both
measured values are unchanged to the last bit. And it is a gap rather than a rule breach:
`docs/METRIC_CITABILITY_RULE.md:288-292` still calls `headline_caveat` an *"Optional disclosure
field"* that is *"printed under the row whatever the mark"* (mark table `:18-25`), so nothing
mechanically required one.

---

## 2. THE REPAIR

One new top-level key `headline_caveat` was appended to each metric payload; the registry row was
given the identical string; the row sha was refreshed with the repo's own tool; one ledger row per
metric was appended. **The measured VALUE of both metrics is unchanged.**

### 2.1 `m21_nci60_9cell` — payload `metrics/by_metric/nci60_precious_9cell_auc.json:549`

> 0.7294 is the 9-cell panel mean and it clears this metric's own 0.7 gate. But DU145 (n_test=7) and SKOV3 (n_test=3, n_val=0) both score AUC = 1.0 on single-digit test sets, where such an AUC is not informative, while the other seven cells average 0.6521; the identity (7 x 0.6520602689568208 + 1.0 + 1.0) / 9 == 0.7293802091886383 therefore holds exactly and the headline is carried by the two smallest test sets. Under the recorded minimum-test-set filters the panel mean is 0.6521 over 7 cells (n_test>=10) and 0.5923 over 5 cells (n_test>=20), both below the 0.81 Krasnov 2026 comparator, so on this panel the honest reading is an explicit miss, not a tie, and not a win. The comparison is also protocol-mismatched: this panel is a Bemis-Murcko scaffold split, whereas Krasnov's 0.81 is a within-metal stratified CV on a random split.

### 2.2 `m19_lomo` — payload `metrics/by_metric/metal_lomo_auc_wasserstein.json:298`

> RETRACTED HEADLINE: the 0.5575 in this file is the 5-metal EQUAL-WEIGHT mean, and docs/PREDICTIVE_LINE_INDEX.md §3 (retraction log, dated 2026-10-19) withdrew the claim 「LOMO 0.5575，差 0.04」 and replaced it with the resolvable-subset 0.5974, because 0.5575 averages in the Rh 51 / Os 36 / Re 74 folds whose unique counts are below MIN_UNIQUE_FOR_GATE=200. The quotable figure per that index is 0.5974 (resolvable subset: Ru 1,311 / Ir 538), with 0.5947 as the seed-ensemble robustness, and the index's own instruction is 「以本索引和台账为准」, i.e. the index and the claim ledger (docs/CLAIM_LEDGER.md) take precedence over registry.py cite; 0.5575 must not be quoted as the LOMO headline. This 0.5575 also sits below the metric's own gate constant of 0.70 (gate_target: 0.7), which gate_target_provenance records as an INTERNAL GATE_AUC constant and NOT a literature value, and real_only_fold_value is 0.3812 (the Rh fold), the weakest fold in that equal-weight mean.

### 2.3 What changed on disk (MEASURED)

| artefact | before | after |
|---|---|---|
| `metrics/by_metric/nci60_precious_9cell_auc.json` | 23805 B · `a2bdc1ad6c514ac7d0b5e36cc24106ca4a1d0348581b59cf1a943bec3abf8ce1` | 24667 B · `8ff6b226ee282348709f2acb865c14e0bf937fe92b81598cb0bc0138cbcd637a` |
| `metrics/by_metric/metal_lomo_auc_wasserstein.json` | 19900 B · `90a9c0c095605fbe20e49a2dfb830a6a890b961b9204330d6cc234bdb6d4aa74` | 20913 B · `2522c80890e18b4a293163a625f0d23831bae34d00c4ccd1e7b4ea3033b9449e` |
| `registry/metrics.yaml` | `b3a8522f…` (682753 B) | `9ddd971e…` (684704 B) |
| `molmetal/reports/run_ledger.jsonl` | `3a5a8cd0…` (237992 B, 432 lines) | `c464cd5e…` (243800 B, 434 lines) |

Key-set diff, MEASURED: m21 added `['headline_caveat']`, removed `[]`, changed `[]` (38 → 39
top-level keys); m19 added `['headline_caveat']`, removed `[]`, changed `[]` (55 → 56 top-level
keys). Every pre-existing key/value pair is byte-identical.

### 2.4 Registry state (MEASURED)

| row | `source_json_sha256` | value | status | verdict | caveat |
|---|---|---|---|---|---|
| `m19_lomo` (`registry/metrics.yaml:216`) | `2522c808…` (`:218`) | 0.5575471269535148 (`:220`) | `MEASURED_BELOW_TARGET_BASELINE` (`:222`) | `HONEST_NEGATIVE` (`:225`) | `:245` (957 ch) |
| `m21_nci60_9cell` (`registry/metrics.yaml:306`) | `8ff6b226…` (`:308`) | 0.7293802091886383 (`:310`) | `MEASURED_HONEST_NEGATIVE` (`:312`) | `PASS` (`:315`) | `:337` (837 ch) |

Both row shas equal the on-disk payload shas; `registry.py check` reports both rows `current`.
Method: the caveat was inserted into each row's `headline_caveat: null` slot rendered with the
generator's own serialiser (`yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)`), then
`registry.py repoint <id> --apply` refreshed the sha. `repoint` is a text-level surgical edit
(`molmetal/scripts/registry.py:2181-2197` the sha line, `:2199-2205` the row `last_refreshed`,
`:2210-2216` the file-level `generated_at` header) that never touches `headline_caveat` and
re-parses before writing (`:2218-2229`). **Durability:** `registry.py generate` rebuilds each row
from the payload — `registry.py:1239` (`payload.get("headline_caveat")`) → `:1245-1247` → `:1426`
→ `:1515` → `:2072` → `:2079` — so a regenerate REPRODUCES the caveat. `generate` was **not** run;
`build_metrics()` was called read-only and reproduces both rows identically except the clock
stamp.

### 2.5 Ledger state (MEASURED) — append-only

`molmetal/reports/run_ledger.jsonl` now has 434 lines; the two new rows are the last two, and the
prior rows are left in place (AGENTS.md §5.5 / §5.6). Append-only proof: the first **237992**
bytes of the file hash to `3a5a8cd0690a0b854c5bfde6e6f73dd62f07e340be7699106b7e8098be135479`
— the pre-append sha (MEASURED this run).

| row | line | `source_json_sha256` | value | `committed` | `commit_blob_equals_disk` | `correction_of_ledger_sha256` |
|---|---|---|---|---|---|---|
| `m21_nci60_9cell` | 433 | `8ff6b226…` | 0.7293802091886383 | `false` | `false` | `a2bdc1ad…` (prior row 429) |
| `m19_lomo` | 434 | `2522c808…` | 0.5575471269535148 | `false` | `false` | `90a9c0c0…` (prior row 17) |

`committed_sha256` on each row names the sha of the blob that IS committed — the HEAD blob, i.e.
the **PRIOR** bytes: `git cat-file -p HEAD:metrics/by_metric/nci60_precious_9cell_auc.json |
sha256sum` → `a2bdc1ad…` and `git cat-file -p
HEAD:metrics/by_metric/metal_lomo_auc_wasserstein.json | sha256sum` → `90a9c0c0…` (both
MEASURED this run). **No commit equality is fabricated.**

---

## 3. THE VERIFIER VERDICTS — verbatim

Both verifiers were read-only, ran 0 GPU, and used only `.venv/bin/python`.

### 3.1 `verify:m21` — overall verdict (verbatim)

> # OVERALL VERDICT: **PROVEN** (the m21 repair is sound and honestly framed) — 8 of 9 checks PROVEN; check 9's *literal* md5 clause is REFUTED but its cause is measured and the delta is exactly the two authorised payload lines.

Per-check verdicts (verbatim):

> 1. Payload carries a non-empty string `headline_caveat` — VERDICT: PROVEN
> 2. Diff vs `git show HEAD:metrics/by_metric/nci60_precious_9cell_auc.json` — VERDICT: PROVEN (exactly one difference)
> 3. Every number in the caveat sourced — VERDICT: PROVEN
> 4. No over-claim — VERDICT: PROVEN (clause by clause vs `test_set_size_funnel.verdict` :538)
> 5. `registry.py cite` prints the caveat — VERDICT: PROVEN (verbatim, my run)
> 6. Registry row sha == disk sha; row is `current` — VERDICT: PROVEN
> 7. Ledger LATEST row — VERDICT: PROVEN
> 8. BITE — VERDICT: PROVEN (3 independent bites, all in /tmp shadow repos)
> 9. Invariants and write scope — VERDICT: PROVEN for the substance; **REFUTED** for the literal md5 clause

Residual defects found (verbatim, not softened):

1. **Producer↔artefact parity UNMET** (verified myself): `molmetal/scripts/train_nci60_10cell.py:1211-1214,1270` and `molmetal/scripts/train_metal_lomo_wasserstein.py:3159,3282` build a FRESH payload dict and write the canonical path; `grep -c headline_caveat` = **0** in both. A re-run of either trainer erases the new caveat (plus ~21-24 other hand-added keys, incl. the whole `test_set_size_funnel` block the caveat cites). Needs a separate, value-agnostic producer change.
2. **All bytes UNCOMMITTED** — registry, ledger and both payloads are working-tree only (no commit authorised); the ledger row says so explicitly, but the caveat is not yet in version control.
3. **`registry.py check` still exits 1 (`REGISTRY: STALE`)** — caused solely by the two pre-existing `wfa155_*` rows owned by a concurrent wave.
4. **`docs/PREDICTIVE_LINE_INDEX.md` has no m21 entry** — the m21 caveat is the only machine-readable ruling on which m21 number is quotable; `docs/**` is out of scope, so this gap stays open.
5. **No mechanical rule blocks a third caveat-less headline row** — `docs/METRIC_CITABILITY_RULE.md:288-292` still calls `headline_caveat` an "Optional disclosure field".
6. **The caveat is duplicated in two places** (payload + registry row). The registry copy is what `cite` prints; the payload copy is what makes it durable across `registry.py generate`. Drift between them is guarded only by `test_registry_row_caveat_is_byte_identical_to_the_payload` (T7) — verified passing.
7. **Minor framing nit:** the caveat's opening clause ("it clears this metric's own 0.7 gate") is the one affirmative sentence; a reader who reads only the first sentence gets the flattering half, though it is immediately qualified and the 0.7 is explicitly the metric's own internal gate (not the 0.81 paper comparator).
8. **Concurrency:** other waves are writing this working tree during verification (a Wf-A200 pytest was live at 02:50; the `wfa155` metric's sha moved mid-run). The m21 artefacts were sha-stable across my whole run, but any repo-wide `check`/ledger reading is a moving target.
9. **The task's literal metrics-porcelain md5 invariant is structurally unreachable** given the two authorised payload edits (both files were clean-vs-HEAD); the exact required hash is recovered by excluding those two lines.

### 3.2 `verify:m19` — overall verdict (verbatim)

> ## OVERALL VERDICT: **PROVEN** (8 of 9 checks clean; check 9's literal md5 constant is a defective invariant, not a defect in the wave)

Per-check verdicts (verbatim):

> 1. `headline_caveat` present at top level, non-empty, a string — VERDICT: PROVEN
> 2. Value + every pre-existing key unchanged vs `git show HEAD:…` — VERDICT: PROVEN
> 3. Every number in the caveat is verbatim sourceable — VERDICT: PROVEN
> 4. The caveat's retraction characterisation matches §3's 2026-10-19 row — VERDICT: PROVEN
> 5. `registry.py cite` prints the caveat for `m19_lomo` — VERDICT: PROVEN
> 6. Registry row sha == payload sha; `registry.py check` says current — VERDICT: PROVEN
> 7. Ledger LATEST row names the current sha + unchanged value, claims no false commit equality — VERDICT: PROVEN
> 8. Both bites fire — VERDICT: PROVEN
> 9. Metrics porcelain md5 / no unexpected path / DO-NOT-TOUCH trees — VERDICT: REFUTED on the literal md5 constant; PROVEN on substance

Residual defects found (verbatim, not softened):

1. **The metrics-porcelain-md5 invariant is structurally unreachable.** `a022e54fd2938b9f333fd8438f6d3d27` is the md5 of the porcelain *excluding* the two payloads this wave was ordered to edit; after the authorized edits the true value is `753b04b2a241039eb538601f35bb0504`. Any wave that edits a clean-vs-HEAD metrics file breaks it. A path-set-based invariant would be sound; the content of the constant should be recomputed per wave.
2. **`registry.py check` exits 1 on this tree** because of two pre-existing STALE `wfa155_*` rows owned by the concurrent Wf-A155 wave. `m19_lomo` is current. Not fixable without touching another wave's files.
3. **`p3_run_ledger_provenance` pre-flight FAILs** on two pre-existing `wfa155_*` OPEN findings; `m19_lomo`/`m21_nci60_9cell` are not among them.
4. **Producer↔artefact parity is UNMET for m19.** `molmetal/scripts/train_metal_lomo_wasserstein.py:3159-3244` builds a **fresh** payload dict and writes it at `:3282` (guarded by `--allow-canonical-out-metric`) — a re-run of that trainer would **erase** the caveat. The caveat lives only in the payload file, not in the producer. (Same class for m21 / `train_nci60_10cell.py:1214-1270`.) Not edited by this wave.
5. **All current bytes are UNCOMMITTED** — sha `2522c808…` is in 0 of 843 commits. The ledger row says so truthfully (`committed: false`, `commit_blob_equals_disk: false`). No commit was authorized, so this is a stated state, not a lie.
6. **`check_metrics` cannot detect a blanked registry-row caveat** (it compares only the sha) — bite 2b returned `PASS`. The real guard is the wave's own byte-identity test; a registry-level caveat check would close this.
7. **No mechanical rule blocks a third caveat-less headline row.** `docs/METRIC_CITABILITY_RULE.md:288-292` still calls `headline_caveat` "Optional disclosure field", and `docs/PREDICTIVE_LINE_INDEX.md` has no `m21` entry. Both DO-NOT-TOUCH for this wave.
8. **The caveat does not disclose the unresolved protocol conflict.** Index `:185-196` records the "已知冲突（未修，需要裁定）": `registry.py cite` says quote 0.5575, the ledger says quote 0.5974, both conforming to their own definitions, and truly fixing it requires changing `configs/locks/protocol_lock.yaml::m19_lomo`'s `target_metric` — explicitly **not** done, left as a pending adjudication item. The caveat reproduces the index's ruling (`以本索引和台账为准`) but not that open adjudication item. Disclosure-completeness observation, not a mismatch with the §3 row.
9. **Pre-existing docs lag:** `docs/PREDICTIVE_LINE_INDEX.md:1438` still calls the wasserstein headline "half-synthetic", while the payload's own `value_is_half_synthetic` is `False` (scope string: the flag was left at `true` after the Wf-R43 real-only re-measurement; the half-synthetic 0.7577 survives under `synthetic_augmented_baseline_preserved_for_diff`). The caveat correctly does **not** repeat the stale claim.
10. **Concurrency artifact:** `metrics/by_metric/wfa155_metal_coordination_sphere.json` changes bytes on every Wf-A155 write, which trips the repo-wide `conftest.py::_published_tree_guard` at session teardown for any test session that straddles one of its writes — including `molmetal/tests/test_wfa203_citability_caveat.py` (the wave's own test file, which writes nothing under `metrics/`).

**Both verifiers REFUTED the task's literal metrics-porcelain md5 clause and PROVED its cause:**
the required constant `a022e54fd2938b9f333fd8438f6d3d27` is the md5 of the porcelain *excluding*
the two payloads the wave was ordered to edit. The REPORT run re-measured this independently —
excluding those two paths the md5 is `a022e54fd2938b9f333fd8438f6d3d27`, the exact required
baseline, and the full porcelain is `753b04b2a241039eb538601f35bb0504` over 40 lines.

---

## 4. OPEN ITEMS — NOT closed

**OPEN-1 (owner: USER) — the m21 registry row's `verdict: PASS` contradicts the payload's own
"explicit miss" reading. This is a CLAIM change and belongs to the USER.**
`registry/metrics.yaml:315` reads `verdict: PASS` for `m21_nci60_9cell`, while the payload's own
nested reading at `metrics/by_metric/nci60_precious_9cell_auc.json:538` says *"an explicit miss —
not a tie, and not a win"* and `:432` says the headline drops from 0.8007 to 0.7294; the same row
carries `status: MEASURED_HONEST_NEGATIVE` (`:312`), so the row contradicts itself. **Mechanism,
MEASURED:** (1) the `verdict` **string** is a hand-written static literal in `METRIC_TARGETS` —
`molmetal/scripts/registry.py:192-197` — applied by the blanket `rec.update(METRIC_TARGETS.get(
metric_id, {}))` at `:1423`, which **overwrites** the payload-derived verdict read at `:1392`
(`payload.get("verdict")`); the m21 payload has **no top-level `verdict` key at all** (MEASURED:
`"verdict" in payload` → False — its `test_set_size_funnel.verdict` is a *nested* key the registry
never reads), so the literal is the only verdict the row can carry. (2) The gap table is genuinely
**mechanical**: `registry.py:1917-1926` computes `hit = cur >= target` and
`gap = round(cur - target, 4)`, and `registry.py gaps` prints m21 as `gap= +0.0294 PASS` and
counts it at target (line2_dmpnn 1/4 at target). So the value-vs-target comparison and the literal
agree, and both disagree with the payload's filter-based reading. **Why it is not fixed here:**
changing the verdict changes what the paper line **claims** about m21 — a scientific/claim
decision, not a mechanical repair, and AGENTS.md §5.3 / TODO-84 put paper-facing claims under the
user's authorisation. The same override applies to m19: `METRIC_TARGETS` at `registry.py:179-184`
sets `m19_lomo` verdict `HONEST_NEGATIVE`, overriding the payload's own top-level `verdict:
BELOW_TARGET_BASELINE_REAL_ONLY` (`metrics/by_metric/metal_lomo_auc_wasserstein.json:87`).
**Options for the user:** (i) retag m21's verdict from `PASS` to `HONEST_NEGATIVE` (mirroring
m19's pattern) — a CLAIM change needing user OK; (ii) leave `verdict: PASS` and rely on the new
`headline_caveat` line (`registry/metrics.yaml:337`) to carry the disclosure — the wave's shipped
state; (iii) change the target or the estimand so `PASS` is defensible — needs
`configs/locks/protocol_lock.yaml` surgery, which the index already flags as a separate
adjudication item (`:195-196`).

**OPEN-2 (census beyond the two repaired rows) — four more `quotable` rows carry no caveat.**
Pre-repair census (HEAD registry, 498 rows, 11 paper-line): 6 quotable-without-caveat rows —
`m14_decode_ratio`, `m7_diversity`, `m19_lomo`, `m21_nci60_9cell`, `l6_closure`, `m6_kappa` — of
which exactly 2 were also `headline_grade: true` (the two repaired). After the repair the class
holds 4 rows (`m14_decode_ratio`, `m7_diversity`, `l6_closure`, `m6_kappa`), **all
`headline_grade: false`** — a different and weaker defect, because `registry.py cite` prints a
`not headline:` reason for each instead of a bare number (`registry.py:2287-2293` prints the
caveat line only when the row carries one). All 3 headline-grade rows now carry a caveat (m19 957
ch, m21 837 ch, `m2_r2_per_cell` 636 ch). The payload declarations these 4 rows do not carry
(MEASURED): `metrics/by_metric/decode_ratio.json` — `status: MEASURED_BUT_NOT_A_QUALITY_METRIC`
plus a `MEASURED_TARGET_HIT_ON_BOTH_SYNTHETIC_AND_REAL` verdict; `metrics/by_metric/
m7_diversity.json` — `verdict: MEASURED_NOT_A_CLAIM — … It is a collapse alarm, and at the cohort
sizes that currently exist it cannot be compared to any published value.`;
`metrics/by_metric/l6_closure.json` — `status: MEASURED_BELOW_GATE` plus an `honest_framing`
that says 2981/5000 complexes reach closure (0.5962) against a 0.95 gate and 1 of 30 metals
passes; `metrics/by_metric/m6_kappa.json` — `HONEST_NEGATIVE: true` plus an
`honest_negative_reason` (aug_mean MEASURED 0.472 vs threshold 0.6, gap -0.128) and an
`honest_framing_note` that says *"do NOT claim it as a learned metallodrug score without
validation against MetalCytoToxDB labelled pIC50."* **NOT fixed** — outside this wave's two-row
scope.

**OPEN-3 (producer/artefact parity — UNMET) — a trainer re-run ERASES both caveats.**
`grep -c headline_caveat` = 0 in `molmetal/scripts/train_nci60_10cell.py`,
`train_metal_lomo_wasserstein.py`, `pa16_nci60_funnel.py` and
`wf_a55_nci60_aggregate_refresh.py` (MEASURED this run). The two **full rewriters** are
`train_nci60_10cell.py` (canonical path is the *unguarded default* at `:1213`; fresh payload dict
at `:1215` with MEASURED 21 top-level keys; write at `:1270`) — the on-disk payload now has 39
keys, so a re-run would erase the caveat plus ~18 hand-added keys **including the whole
`test_set_size_funnel` block the caveat cites** — and `train_metal_lomo_wasserstein.py` (fresh
payload at `:3159`; canonical write at `:3282`, guarded by `--allow-canonical-out-metric` at
`:3253`; the on-disk payload now has 56 keys, so ~21 hand-added keys would be erased). The two
**round-trip editors PRESERVE** the caveat: `pa16_nci60_funnel.py:132` (default path `:41`) and
`wf_a55_nci60_aggregate_refresh.py:240` (default path `:37`) both load the whole doc, set one key
and re-dump — **but `pa16_nci60_funnel.py` is listed as a ZERO-REFERENCE script at
`docs/PREDICTIVE_LINE_INDEX.md:1465`, so no new work should depend on it.**
`molmetal/scripts/wf226_aggregate.py` is **not** a writer for our file: `:42` uses a different
metric id (`metal_lomo_auc_wasserstein_wf226`), `:9`/`:73-74` write to
`molmetal/reports/wf226_lomo_sweep/`, the id appears 0 times in `registry/metrics.yaml` and no
wf226 file is tracked under `metrics/by_metric` (MEASURED). The durable fix is a **value-agnostic
trainer-side caveat template** — a separate change; hard-coding this caveat in the trainer would
emit a **FALSE** caveat on a future re-measurement (the m19 producer does not reproduce the
current on-disk artefact at all). **NOT fixed** — both producers are outside this wave's write
scope and were deliberately left alone.

**OPEN-4 (owner: USER) — all of this wave's bytes are UNCOMMITTED; no commit was authorised.**
`git log --all --oneline -S 'headline_caveat' -- metrics/by_metric/nci60_precious_9cell_auc.json
metrics/by_metric/metal_lomo_auc_wasserstein.json` returns EMPTY (MEASURED this run): the new
payload bytes are in **no** commit. HEAD is `1b9470a0…` and its blobs for the two payloads are the
**PRIOR** bytes (`a2bdc1ad…`, `90a9c0c0…`, both MEASURED via `git cat-file -p | sha256sum`).
`registry/metrics.yaml` and the two appended ledger rows are working-tree state only as well.
Honesty measures taken: both ledger rows carry `committed: false` and
`commit_blob_equals_disk: false`, and `committed_sha256` names the sha of the blob that *is*
committed. **Action required:** a commit (and a review) is needed to bring the caveat into version
control. None was authorised in Wf-A203, so none was made.

**OPEN-5 (process) — the task's literal metrics-porcelain md5 invariant is structurally
unreachable.** Required `a022e54fd2938b9f333fd8438f6d3d27`; observed
`753b04b2a241039eb538601f35bb0504` over 40 lines; **excluding the two authorised payload paths the
md5 is exactly the required baseline** (MEASURED this run). Porcelain is path-only, so a
clean-vs-HEAD → modified transition necessarily adds a line, and both target payloads were
clean-vs-HEAD at the Measure baseline. The constant should be restated per wave as a path-set
invariant.

**OPEN-6 (other wave) — `registry.py check` exits 1 and `p3_run_ledger_provenance` FAILs, both on
two pre-existing `wfa155_*` rows, not on ours.** `registry.py check` → `[STALE] metrics: 580
current, 2 stale, 0 missing | 11 paper-line (8 quotable, 3 headline-grade)`; the 2 STALE rows are
`wfa155_ligand_connected_rate` and `wfa155_metal_coordination_sphere`. The gate → `STATUS FAIL —
ledger_rows=434 metrics=399 OPEN_never_committed=2 superseded_never_committed=9`, the 2 OPEN
findings being the same two metrics. **Neither `m19_lomo` nor `m21_nci60_9cell` is an OPEN
finding** (MEASURED this run). Owned by the concurrent Wf-A155 wave; its files are on this wave's
DO-NOT-TOUCH list.

**OPEN-7 (docs) — `docs/PREDICTIVE_LINE_INDEX.md` has no m21 entry at all.**
`grep -c -e m21 -e nci60 -e NCI-60 docs/PREDICTIVE_LINE_INDEX.md` = 2, and neither hit is a metric
entry: `:1465` names `pa16_nci60_funnel.py` in the zero-reference-script list and `:1474` is a
data-blockade table row. So the m21 caveat (`registry/metrics.yaml:337`) is currently the **only**
machine-readable ruling on which m21 number is quotable — the index has no LOMO-style artifact
table, no known-conflict table and no retraction row for m21. `docs/**` is out of scope here.

**OPEN-8 (process) — no mechanical rule blocks a third caveat-less headline row.**
`docs/METRIC_CITABILITY_RULE.md:288-292` still calls `headline_caveat` *"Optional disclosure
field"*; no gate enforces caveat presence on a `headline_grade: true` row, and this wave's test
covers exactly the two repaired rows.

**OPEN-9 (process) — the caveat is duplicated (payload + registry row) and the registry-level
copy is not machine-checked.** `registry.py check` compares only the sha
(`molmetal/scripts/registry.py:1665-1667`) — it cannot detect a blanked registry-row caveat (both
verifiers measured this: the m19 bite 2b returned `PASS`). The real guard is the wave's own
byte-identity test (T7), verified passing.

**OPEN-10 (docs) — the m19 caveat does not disclose the unresolved protocol conflict.**
`docs/PREDICTIVE_LINE_INDEX.md:185-196` records 「已知冲突（未修，需要裁定）」: `registry.py cite`
says quote 0.5575, the claim ledger says quote 0.5974, and truly removing the conflict requires
changing `configs/locks/protocol_lock.yaml::m19_lomo`'s `target_metric` — explicitly not done
(`:195-196`). The caveat reproduces the index's ruling (「以本索引和台账为准」) but not that pending
adjudication item. (Surfaced by `verify:m19`.)

**OPEN-11 (docs, pre-existing) — the index still calls the m19 headline "half-synthetic".**
`docs/PREDICTIVE_LINE_INDEX.md:1438` says so, while the payload's own `value_is_half_synthetic`
is `false` (`metrics/by_metric/metal_lomo_auc_wasserstein.json:10`; the scope string at `:11`
explains the flag was left at `true` after the Wf-R43 real-only re-measurement and that the
half-synthetic 0.7577 survives under `synthetic_augmented_baseline_preserved_for_diff`). The new
caveat correctly does **not** repeat the stale claim. (Surfaced by `verify:m19`.)

**OPEN-12 (other wave / tooling) — a concurrent writer trips the repo-wide published-tree guard
at session teardown.** `molmetal/tests/conftest.py::_published_tree_guard` compares every published
metric artifact's sha256 at session start against teardown and raises a teardown ERROR if any
moved; `metrics/by_metric/wfa155_metal_coordination_sphere.json` is rewritten by the concurrent
Wf-A155 script, so any session straddling one of its writes trips the guard — including this
wave's own test file, which writes nothing under `metrics/` (MEASURED: `registry.py check` and
`registry.py cite` leave that file's sha256 **and** mtime unchanged).

**OPEN-13 (framing nit, disclosed not changed) — the m21 caveat's opening clause is its one
affirmative sentence.** *"it clears this metric's own 0.7 gate"* is the flattering half; a reader
who reads only the first sentence gets it. It is immediately qualified, and the 0.7 is explicitly
the metric's own internal gate (`registry/metrics.yaml:330`; the analogous m19 0.7 is declared
INTERNAL at `metrics/by_metric/metal_lomo_auc_wasserstein.json:8`) rather than the 0.81 paper
comparator.

---

## 5. THE TEST

```
CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
    molmetal/tests/test_wfa203_citability_caveat.py -q
27 passed, 1 warning in 9.39s
```

MEASURED 2026-11-02 (REPORT run), **exit code 0**. Test file
`molmetal/tests/test_wfa203_citability_caveat.py`, 17374 B, sha256
`60fb7ea7d0fdfffdd7a0877714aaf35b1345cc8cada5451703d449b568ee2877`.

T1 `cite` prints a `caveat:` line for both, carrying the required numbers · T2 the measured values
are unchanged (0.7293802091886383 / 0.5575471269535148) · T3 each registry row's sha equals the
payload's sha on disk · T4 no forbidden claim word in either caveat · T5 `registry.py check`
reports no STALE/MISSING marker for either metric · T6 the ledger's latest row names the current
on-disk sha and the unchanged value · T7 the registry row's caveat is byte-identical to the
payload's and `assess_citability` reproduces it from the payload alone (the durability proof) · T8
the Wf-A203 ledger row does not fabricate a commit equality.

**One assertion was adapted, and how.** "registry.py check exits 0" is unreachable on this tree
(two PRE-EXISTING `wfa155_*` rows are STALE, owned by another wave), so the test pins the **exact
STALE set** plus exit-code consistency instead — strictly stronger against regression — and the
adaptation is stated in the test's own docstring.

**Other gates, MEASURED 2026-11-02 (REPORT run):** `registry.py check` → exit 1, `580 current, 2
stale, 0 missing` (the 2 are the pre-existing `wfa155_*` pair; neither repaired metric is stale) ·
`registry.py cite` → exit 0, prints a `caveat:` line under both `[CITE] m19_lomo` and `[CITE]
m21_nci60_9cell` · `registry.py gaps` → `m21_nci60_9cell gap= +0.0294 PASS`, `m19_lomo gap=
-0.1425 HONEST_NEGATIVE` · `p3_run_ledger_provenance` → STATUS FAIL on the 2 pre-existing
`wfa155_*` OPEN findings only.

---

## 6. INVARIANTS

| invariant | required | observed (MEASURED) | verdict |
|---|---|---|---|
| `git status --porcelain metrics/` md5 | `a022e54fd2938b9f333fd8438f6d3d27` | `753b04b2a241039eb538601f35bb0504` (40 lines) | **DEVIATION — structural, see OPEN-5** |
| the same md5 **excluding the two authorised payloads** | — | `a022e54fd2938b9f333fd8438f6d3d27` | **exact required baseline** |
| `metrics/` files touched by the REPORT agent | 0 | **0** (porcelain md5 identical before and after the report writes) | PASS |
| payload sha `nci60_precious_9cell_auc.json` | unchanged by REPORT | `8ff6b226…` before and after | PASS |
| payload sha `metal_lomo_auc_wasserstein.json` | unchanged by REPORT | `2522c808…` before and after | PASS |
| registry sha / ledger sha unchanged by REPORT | — | `9ddd971e…` / `c464cd5e…` | PASS |
| only the two registry rows changed by the wave | yes | `['m19_lomo','m21_nci60_9cell']` + the tool-mandated `generated_at` header | PASS |
| ledger append-only | yes | first 237992 bytes hash to the pre-append sha `3a5a8cd0…` | PASS |

---

## 7. PRIOR RENDERING PRESERVED (AGENTS.md §5.5 rule #6)

The superseded rendering of this wave's report is the INTEGRATE agent's **pre-verification**
report. It is preserved **VERBATIM** (byte-identical copies) because this rendering *corrects* it
— it contains no verifier verdicts and its OPEN list is 5 items against this rendering's 13.

| superseded file | preserved as | sha256 | bytes | written at |
|---|---|---|---:|---|
| `report.md` | `molmetal/reports/wfa203_citability_caveat/report.integrate_preverify_2026-11-02.md` | `7b6dd678b2ab3b66b15d298df64b6433aec8c7c0cdbf9c431eb70292ebfaa749` | 17581 | 2026-11-02 02:46:08 +0800 |
| `report.json` | `molmetal/reports/wfa203_citability_caveat/report.integrate_preverify_2026-11-02.json` | `9dd861aac056251f05930bcf6ce72f29080f3c04546fd6f1cde1819c890fcf5e` | 12465 | 2026-11-02 02:45:26 +0800 |

---

## 8. FILES WRITTEN BY THE REPORT AGENT

- `molmetal/reports/wfa203_citability_caveat/report.json` (this wave's report, machine-readable)
- `molmetal/reports/wfa203_citability_caveat/report.md` (this file)
- `molmetal/reports/wfa203_citability_caveat/report.integrate_preverify_2026-11-02.md`
  (byte-identical preservation of the superseded rendering)
- `molmetal/reports/wfa203_citability_caveat/report.integrate_preverify_2026-11-02.json`
  (byte-identical preservation of the superseded rendering)

Nothing else was touched: no `metrics/` file, no `paper/**`, no `docs/**`, no `TODO/**`, no
`AGENTS.md`/`README.md`, no `registry/*.yaml` other than the wave's own `registry/metrics.yaml`
edit, no `pre_flight_check.py`, and no other wave's report/test tree.

---

## 9. ADDENDUM — 2026-11-02 (REPORT run, dated, append-only)

This addendum is **ADDITIVE and DATED** per AGENTS.md §5.5 rule #6. Nothing in §1-§8 above is
edited, reworded or removed; the renderings this section supersedes are preserved VERBATIM at
`report.integrate_preverify_2026-11-02.md` / `.json` (the INTEGRATE agent's pre-verification
rendering) and `report.reportagent_v1_2026-11-02.md` / `.json` (this REPORT agent's own first
rendering, taken before this addendum).

### 9.1 Overwrite-risk disclosure (MEASURED — both targets)

AGENTS.md §5.6 requires `safe_overwrite_check` on overwrite writes. **It FIRED on both
targets** and the ratios are recorded here rather than suppressed:

| target | `safe_overwrite_check` ratio | threshold | MEASURED at |
|---|---|---|---|
| `report.json` | `0.9449901768172888` | `0.5` | `molmetal/scripts/wf386a3_atomic_write.py:269` (threshold), `:298` (ratio), `:308` (returned) |
| `report.md` | `0.9841269841269842` | `0.5` | same |

The risk was **neutralised BEFORE the write**: the superseded rendering was first copied
byte-identically to the dated preservation paths and the copy's sha256 was asserted equal to the
original's (MEASURED — `report.integrate_preverify_2026-11-02.md` =
`7b6dd678b2ab3b66b15d298df64b6433aec8c7c0cdbf9c431eb70292ebfaa749`,
`report.integrate_preverify_2026-11-02.json` =
`9dd861aac056251f05930bcf6ce72f29080f3c04546fd6f1cde1819c890fcf5e`). The ratio is high because
this rendering is a **restructure-and-extend** of the prior one (verifier verdicts + 13 open
items + per-number citations added), not because content was destroyed — every prior statement is
recoverable byte-for-byte from the preservation paths. `detect_truncation` returned `None`
(= no truncation detected) for both writes (MEASURED, `wf386a3_atomic_write.py:312`).

### 9.2 Gate outputs re-measured in the REPORT run (verbatim)

* `registry.py check` → **exit 1**, stdout:
  `[STALE] metrics: 580 current, 2 stale, 0 missing | 11 paper-line (8 quotable, 3 headline-grade)`
  with the only 2 STALE rows being `wfa155_ligand_connected_rate` and
  `wfa155_metal_coordination_sphere`; final line `REGISTRY: STALE`. Neither repaired metric is
  STALE.
* `registry.py cite` → **exit 0**, 33 non-empty output lines. `[CITE] m19_lomo` and
  `[CITE] m21_nci60_9cell` each carry their `caveat:` line. The four remaining
  quotable-without-caveat rows are `[NUM ]` and each prints a `not headline:` reason —
  MEASURED verbatim: `m14_decode_ratio` → *"status 'MEASURED_BUT_NOT_A_QUALITY_METRIC': the file
  itself declines the claim"*; `m7_diversity` → *"status 'MEASURED_NOT_A_CLAIM': the file itself
  declines the claim"* + *"definition was changed (REDEFINED); values measured before the change
  are not comparable"*; `l6_closure` → *"single run: quote as a single-run measurement, not a
  table mean"*; `m6_kappa` → *"single run: quote as a single-run measurement, not a table mean"*.
  This confirms the OPEN-2 claim that those 4 are a weaker class than the 2 repaired
  headline-grade rows.
* `registry.py gaps` → **exit 0**: `m21_nci60_9cell value=0.72938020 target=0.7 gap= +0.0294
  PASS`; `m19_lomo value=0.55754712 target=0.7 gap= -0.1425 HONEST_NEGATIVE`;
  `line2_dmpnn: 1/4 at target`. This is the **mechanical** value-vs-target derivation that
  OPEN-1 rests on.
* `p3_run_ledger_provenance` → **STATUS FAIL**, `wall_sec` 0.788, evidence verbatim:
  `p3_run_ledger_provenance: ledger_rows=434 metrics=399 OPEN_never_committed=2
  superseded_never_committed=9 OPEN wfa155_ligand_connected_rate sha a60933e8281f src
  metrics/by_metric/wfa155_ligand_connected_rate.json OPEN wfa155_metal_coordination_sphere sha
  aa67ed49b645 src metrics/by_metric/wfa155_metal_coordination_sphere.json`. **Neither
  `m19_lomo` nor `m21_nci60_9cell` appears in the OPEN list.**
* `pytest molmetal/tests/test_wfa203_citability_caveat.py -q` → **exit 0**, `27 passed, 1
  warning in 9.28s`. §5 above records `27 passed, 1 warning in 9.39s` from the earlier run of the
  same command in the same session: the assertion count (27) and the exit code (0) are identical
  across both runs and both are MEASURED — the difference is pytest wall-clock reporting only.
* `git status --porcelain metrics/ | md5sum` was `753b04b2a241039eb538601f35bb0504` **before and
  after** the pytest run, and `a022e54fd2938b9f333fd8438f6d3d27` after excluding the two
  authorised payload paths — i.e. the wave's test file writes nothing under `metrics/`.

### 9.3 Files written by the REPORT agent (final)

| path | role |
|---|---|
| `molmetal/reports/wfa203_citability_caveat/report.json` | the wave report, machine-readable |
| `molmetal/reports/wfa203_citability_caveat/report.md` | the wave report, prose |
| `molmetal/reports/wfa203_citability_caveat/report.integrate_preverify_2026-11-02.md` | VERBATIM preservation of the INTEGRATE agent's pre-verification rendering |
| `molmetal/reports/wfa203_citability_caveat/report.integrate_preverify_2026-11-02.json` | VERBATIM preservation of the INTEGRATE agent's pre-verification rendering |
| `molmetal/reports/wfa203_citability_caveat/report.reportagent_v1_2026-11-02.md` | VERBATIM preservation of this REPORT agent's first rendering |
| `molmetal/reports/wfa203_citability_caveat/report.reportagent_v1_2026-11-02.json` | VERBATIM preservation of this REPORT agent's first rendering |

Nothing else was touched: no `metrics/` file, no `paper/**`, no `docs/**`, no `TODO/**`, no
`AGENTS.md`/`README.md`, no `registry/*.yaml` other than the wave's own `registry/metrics.yaml`
edit, no `pre_flight_check.py`, and no other wave's report/test tree.
