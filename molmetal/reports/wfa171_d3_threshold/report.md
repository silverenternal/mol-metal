# Wf-A171 D3 — the L5 protocol-validation hooks PASS on file existence, not on the spec threshold

**Wave:** Wf-A171 · **Branch:** D3 · **Date:** 2026-11-01
**Tag:** MEASURED (every number below was re-derived from disk by the commands quoted in §6)
**Owner of this report:** `molmetal/reports/wfa171_d3_threshold/`
**Status:** the repair is STAGED, NOT APPLIED. `molmetal/scripts/pre_flight_check.py` is UNTOUCHED.

---

## 1. The defect, stated exactly

All nine §4 L5 per-baseline protocol-validation hooks funnel through ONE helper in
`molmetal/scripts/pre_flight_check.py`:

- `_emit_protocol_validation_hook_json` — `molmetal/scripts/pre_flight_check.py:12322`
- `_hook_passthrough_payload` — `molmetal/scripts/pre_flight_check.py:12384`

The PASS branch of `_hook_passthrough_payload` (`molmetal/scripts/pre_flight_check.py:12445-12453`)
keys on two things only:

```python
    return GateResult(
        name, priority, pipeline, "PASS",
        f"{baseline_name} {hook_id} hook emitted with source_present=True "
        f"at {hook_dir}",
        time.perf_counter() - t0,
    )
```

`source_present` is "the source path exists" (`pre_flight_check.py:12404-12406`).
Nothing in the helper compares the source against the threshold that
`TODO/03_plans/PRE_TRAINING_GATE_v2.md` states for the hook. The helper's own
docstring (`pre_flight_check.py:12395-12401`) says so: *"PASS criteria (all required):
(a) hook_dir on disk + emit hook JSON; (b) source metric JSON OR source dir on disk"*.

The spec thresholds exist (`TODO/03_plans/PRE_TRAINING_GATE_v2.md:272, 280, 288, 298,
306, 314, 324, 332, 340`) and the sources do not meet them (§3).

**Blast radius today: zero.** `build_gates()` (`pre_flight_check.py:13957`) contains
no `.exists()` call and no `runners +=` site for the nine L5 gates — the 24-gate
wiring is still staged and unapplied in
`molmetal/reports/wfa169_dead_gate_repair/regen_block.py`. MEASURED: a live import
of `build_gates` gives `all=129, frontier=25, chemprop_v2_v3=18, cfm=8`, identical to
`EXPECTED_GATE_COUNTS` (`pre_flight_check.py:5689`). The defect is therefore a
**latent false-green**, not a live one — and the repair must stay that way.

---

## 2. The nine spec thresholds, quoted

| # | hook_id | plan-doc line | Verdict-gate line (verbatim) |
|---|---|---|---|
| 1 | `krasnov_m1_per_seed_per_split_5fold_5metal` | 272 | "`PASS` if 5+ metals × 5 seeds × 5 folds (125+ records) present; else `WARN`" |
| 2 | `krasnov_m21_per_cell_line_per_seed` | 280 | "`PASS` if 9 cell-lines × 5+ seeds (45+ records) present; else `WARN`" |
| 3 | `krasnov_protocol_alignment_diffs_documented` | 288 | "`PASS` if §6 paragraph exists for each Tier-1 LEAD metric × Krasnov" |
| 4 | `rusanov_m2_per_cell_line_per_cohort` | 298 | "`PASS` if 4 cell-lines × 3 cohorts (12 records) present; else `WARN`" |
| 5 | `rusanov_m2_4cell_baseline_comparison` | 306 | "`PASS` if at least 1 cell-line `delta > +0.03` (Tier 1 match)" |
| 6 | `rusanov_protocol_alignment_diffs_documented` | 314 | "`PASS` if §6 paragraph exists for each Tier-1 LEAD metric × Rusanov" |
| 7 | `drugood_m17_assay_split_auc` | 324 | "`PASS` if `delta_imold > -0.02` (Tier 1 match)" |
| 8 | `drugood_m17_scaffold_split_auc` | 332 | "`PASS` if `delta_imold > -0.02`" |
| 9 | `drugood_m17_size_split_auc` | 340 | "`PASS` if `delta_imold > -0.02`" |

### 2a. HONEST NEGATIVE — the brief's premise is only true for 3 of the 9

The D3 brief says "the spec gives each hook a record-count threshold". MEASURED:
**only three hooks carry a record count** (125 / 45 / 12). Four carry a **value**
threshold (`delta_imold > -0.02` ×3, `delta > +0.03` ×1) and two carry a
**doc-presence** threshold ("a §6 paragraph exists for each Tier-1 LEAD metric ×
baseline"). A module that assumed a uniform record count would have reported four
of these hooks against a threshold that does not exist. The module therefore models
four threshold kinds: `record_count`, `value_gt`, `count_above_value`, `doc_presence`.

---

## 3. The measured nine-row table

Counting key = the JSON key named in the module's cited constant block; counts were
obtained by JSON parsing, never by grep.

| # | hook_id | spec threshold | source path (primary) | counting key | MEASURED | verdict |
|---|---|---|---|---|---|---|
| 1 | `krasnov_m1_per_seed_per_split_5fold_5metal` | 125 records (L272) | `metrics/by_metric/metal_lomo_auc_wasserstein.json` | `per_run_auc` | **25** (shortfall_n **100**) | **SHORTFALL** |
| 2 | `krasnov_m21_per_cell_line_per_seed` | 45 records (L280) | `metrics/by_metric/nci60_precious_9cell_auc.json` | `per_cell_line` | **9** (shortfall_n **36**) | **SHORTFALL** |
| 3 | `krasnov_protocol_alignment_diffs_documented` | §6 paragraph for m1 + m21 (L288, L287) | `docs/HONEST_NEGATIVE_CATALOG.md` | §6 entries per metric | **2 / 2** metrics covered (m1: §6.58, §6.64; m21: §6.51, §6.64, §6.65, §6.150c) | **MEETS** |
| 4 | `rusanov_m2_per_cell_line_per_cohort` | 12 records (L298) | `metrics/by_metric/m2_r2_per_cohort.json` | `per_cohort` | **3** (shortfall_n **9**) | **SHORTFALL** |
| 5 | `rusanov_m2_4cell_baseline_comparison` | ≥1 cell-line `delta > +0.03` (L306) | `molmetal/reports/wf388a1_m2_r2_real/result.json` (L305) | per-cell-line delta | **MISSING** — source not on disk | **MISSING_SOURCE** |
| 6 | `rusanov_protocol_alignment_diffs_documented` | §6 paragraph for m2 (L314, L313) | `docs/HONEST_NEGATIVE_CATALOG.md` | §6 entries per metric | **1 / 1** metric covered (m2: §6.37, §6.39, §6.51, §6.58, §6.64, §6.65) | **MEETS** |
| 7 | `drugood_m17_assay_split_auc` | `delta_imold > -0.02` (L324) | `molmetal/reports/wf265_collect/all_results.json` (L323) | `per_axis_auc_delta_vs_imold.assay` | key **ABSENT** in the spec source; alternate `metrics/by_metric/drugood_ood_auc.json` measures **+0.009347711256380342** | **MEETS** (primary = MISSING_KEY) |
| 8 | `drugood_m17_scaffold_split_auc` | `delta_imold > -0.02` (L332) | `molmetal/reports/wf265_collect/all_results.json` (L331) | `per_axis_auc_delta_vs_imold.scaffold` | key **ABSENT**; alternate measures **+0.001844904852242868** | **MEETS** (primary = MISSING_KEY) |
| 9 | `drugood_m17_size_split_auc` | `delta_imold > -0.02` (L340) | `molmetal/reports/wf265_collect/all_results.json` (L339) | `per_axis_auc_delta_vs_imold.size` | key **ABSENT**; alternate measures **−0.0028530871876835118** | **MEETS** (primary = MISSING_KEY) |

Summary: **3 SHORTFALL, 4 MEETS, 1 MISSING_SOURCE, 1 MISSING_KEY** (rows 7-9 share
the MISSING_KEY primary) — **3 of 9 hooks are below their own spec threshold and
1 of 9 has no source at all.**

### 3a. Notes that must travel with the table

* **Row 1.** The source stores **one scalar per (metal, seed) run** (`per_run_auc`,
  25 entries = 5 metals × 5 seeds) plus `per_fold_auc_mean` over the 5 held-out
  metals. The spec asks for 125 **(metal × seed × fold)** records; the per-fold
  values are not stored per run, so the 125 tuples are not reconstructible from
  this file even though `n_folds=5` is recorded. The 25-vs-125 gap is therefore
  not a rounding artefact of the counting key — it is a real granularity gap.
* **Row 2.** Every cell record carries `per_seed_available: false`; the per-seed
  records the spec asks for do not exist at all.
* **Row 4.** `metrics/by_metric/m2_r2_per_cohort.json` holds 3 cohorts
  (`platinai_4cell`, `hilbert`, `metalcytotox_combined`) and has **no key
  containing "cell" anywhere in the file** — the (cell × cohort) axis is absent,
  not just short.
* **Row 5.** BOTH sources named at plan-doc line 305 are absent from disk:
  `molmetal/reports/wf388a1_m2_r2_real/` and
  `molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline/`. Nothing can be
  measured, so the module reports MISSING_SOURCE, never 0.
* **Rows 7-9.** The spec source `molmetal/reports/wf265_collect/all_results.json`
  exists (15 top-level keys) but contains **none** of `0.7479 / 0.7211 / 0.7163 /
  0.6884 / 0.6879 / 0.6792 / 0.675` and **no** key matching
  `assay|scaffold|size|split|imold|erm`. The hook PASSes on the existence of a file
  that contains no drugood-split data whatsoever. The deltas live in
  `metrics/by_metric/drugood_ood_auc.json` (the file the L3 gate
  `p3_mock_to_real_m17_drugood_assay_REAL_3_seeds` already reads) and there the
  three thresholds genuinely MEET.
* **Rows 7-9, second honest negative.** The plan doc's "current state" annotation
  (`metacytotox_fallback assay 0.7479`) does **not** match the measured metric:
  `drugood_ood_auc.json` gives `per_axis_auc.assay = 0.7304477112563803`
  (`assay_split_REAL = 0.7304`). The value `0.7479` occurs only inside the prose
  string `assay_split_REAL_honest_framing`. **The spec's quoted current-state
  number is stale**; the thresholds still MEET either way, but the number quoted
  in the plan doc is not the measured one.
* **Rows 3 and 6.** The plan doc cites "`PROTOCOL_ALIGNMENT_MATRIX.md` §1.4"
  (lines 287 and 313). The canonical file is
  `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` (it exists;
  `TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md` does not) and it has the headings
  §0, §1, §2, §3, §4 — **there is no §1.4**. The two doc-presence hooks are
  therefore measured against the §6 catalog only.

---

## 4. The repair (staged, not applied)

### 4a. The module — `molmetal/scripts/wfa171_l5_threshold.py`

Standalone, importable, torch-free, read-only (it never writes a metric file).
Public surface:

* `parse_plan_specs(path=None) -> dict[hook_id, Threshold]` — parses the nine
  thresholds out of the **Verdict-gate line** of each §4 hook block in
  `TODO/03_plans/PRE_TRAINING_GATE_v2.md`, **at runtime, from disk**. Nothing
  numeric is hard-coded for the thresholds.
* `default_specs() -> list[HookSpec]` — the nine specs, thresholds from disk.
* `evaluate_hook(spec, repo_root=None) -> HookRecord` — accepts a `HookSpec` **or**
  the brief's 4-tuple `(hook_id, spec_threshold, source_path, counting_key)`.
* `evaluate_hooks(specs=None) -> list[HookRecord]` — all nine.
* `evaluate_hook_by_id(hook_id) -> HookRecord | None`.
* `format_evidence(record) -> str` — the one-line evidence fragment.
* `strict_enabled(env=None) -> bool` — the opt-in strict mode.

`HookRecord` carries `threshold`, `measured_count`, `measured_value`,
`meets_threshold`, `source_present`, `shortfall_n`, `shortfall_pp`, `status`, plus
`primary_*` and `alternates` provenance so no measurement is hidden.

**Documented constant block (as the brief requires, and said so in the docstring):**
the plan doc names the source *files* in prose but never names the JSON *key* that
holds the records, and for two hooks it names a file that is not on disk. The
counting keys + source paths therefore live in the clearly-marked
`COUNTING_SOURCES` constant block, each entry citing the plan-doc line it came
from. The module docstring states this split in as many words.

### 4b. The staged patch — `molmetal/reports/wfa171_d3_threshold/patch_d3_threshold.diff`

A unified diff against `molmetal/scripts/pre_flight_check.py`, **NOT APPLIED**.
5 hunks, +50 / −3 lines. It:

1. adds `from molmetal.scripts import wfa171_l5_threshold as _wfa171_l5`
   (next to the existing `atomic_write_file` import at line 325);
2. measures the source against the spec threshold inside `_hook_passthrough_payload`;
3. appends `threshold + measured count + SHORTFALL/MISSING` to **all three**
   evidence strings (source-missing WARN, not-emitted WARN, PASS);
4. adds the **opt-in strict mode**: `WFA171_L5_STRICT=1` turns a shortfall into
   **WARN — never FAIL**.

**What the patch does NOT do:** it adds no gate, removes no gate, touches no
`runners +=` site, does not touch `EXPECTED_GATE_COUNTS`, and does not change any
verdict by default (PASS stays PASS). The 24-gate wiring remains an unmade user
decision (it needs a `paper/refs.bib` edit, AGENTS.md §5.3).

---

## 5. Verification (all MEASURED)

| check | command | result |
|---|---|---|
| patch applies | `git apply --check -p1 <patch>` on a /tmp scratch copy | **rc = 0** |
| gate counts unchanged | live import of the patched copy (symlink farm in /tmp) vs. the working tree | `all=129, frontier=25, chemprop_v2_v3=18, cfm=8` **both**, == `EXPECTED_GATE_COUNTS` |
| test suite | `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa171_l5_threshold.py -q` | **22 passed**, 0 failed |
| module is torch-free | fresh interpreter, module loaded by path | `NO_TORCH`; no `import torch` statement in the AST |
| module is read-only | mtime census of `metrics/by_metric/` before/after `evaluate_hooks()` | unchanged |

The gate-count check uses a symlink farm whose `molmetal/scripts/pre_flight_check.py`
is the patched copy; the farm resolves to the real repo contents, so the patched
module sees the same data files as the working tree.

---

## 6. Reproduction

```bash
cd /home/hugo/codes/mol-metal

# the measurement (CPU only, no GPU)
CUDA_VISIBLE_DEVICES="" .venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from molmetal.scripts import wfa171_l5_threshold as M
for r in M.evaluate_hooks():
    print(r.hook_id, r.status, r.measured_count, r.measured_value,
          r.meets_threshold, r.shortfall_n, r.shortfall_pp)"

# the staged patch, checked but NOT applied
mkdir -p /tmp/d3chk/molmetal/scripts
cp molmetal/scripts/pre_flight_check.py /tmp/d3chk/molmetal/scripts/
(cd /tmp/d3chk && git apply --check -p1 \
  /home/hugo/codes/mol-metal/molmetal/reports/wfa171_d3_threshold/patch_d3_threshold.diff)

# the tests
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  molmetal/tests/test_wfa171_l5_threshold.py -q
```

---

## 7. Files written by this branch

| path | bytes | what |
|---|---|---|
| `molmetal/scripts/wfa171_l5_threshold.py` | 35820 | the standalone, torch-free threshold module |
| `molmetal/tests/test_wfa171_l5_threshold.py` | 16771 | 22 tests |
| `molmetal/reports/wfa171_d3_threshold/patch_d3_threshold.diff` | 4237 | the STAGED patch (unapplied) |
| `molmetal/reports/wfa171_d3_threshold/threshold_table.json` | 15001 | the machine-readable nine-row table |
| `molmetal/reports/wfa171_d3_threshold/report.md` | this file | this report |

All five were written through `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file`
and verified with `detect_truncation` (all returned `None`).

---

## 8. Open items and uncertainties (not smoothed over)

1. **The strict mode is off by default and therefore changes nothing today.** It is
   a repair that cannot be observed until the 24-gate wiring lands; that wiring is
   blocked on a user decision (it touches `paper/refs.bib`, AGENTS.md §5.3).
2. **The doc-presence rule is a judgement, made reproducible.** A §6 entry
   qualifies for metric `mX` when it mentions `mX` (token-boundary regex) **and**
   a split/protocol-difference token. The token list is in the module and can be
   argued with. MEASURED: the covered metric set is identical when every
   `##`/`###`/`####` heading is used as an entry boundary instead of `###` only,
   so rows 3 and 6 do not hinge on the choice — but the *entry counts* do
   (m21: 4 entries under the `###` rule, 3 under the all-levels rule).
3. **The DrugOOD alternate source is an editorial judgement.** The spec names
   `all_results.json`; that file has no drugood-split content. Measuring the
   threshold against `metrics/by_metric/drugood_ood_auc.json` reports the real
   delta, but it is a *different file* than the one the spec and the gate name.
   Both facts are in the record (`primary_status = MISSING_KEY`,
   `measured_source_path` = the alternate). A reader who wants only the spec's
   own source should read the primary columns.
4. **The plan doc's "current state" annotation for m17 is stale** (§3a, rows 7-9):
   the quoted `0.7479` is not the measured `0.7304477112563803`. The doc is in
   `TODO/**`, which this branch may not write; this report is the record.
5. **`TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md` does not exist** and the
   canonical matrix has no §1.4 (§3a, rows 3 and 6). Wf-A169's staged
   `OLD_MATRIX`/`NEW_MATRIX` constants already correct the path; the §1.4
   sub-reference remains dangling in the spec.
6. **The 125-vs-25 gap is a granularity gap, not a key-selection artefact.** If a
   future repair wants per-(metal × seed × fold) records, they must be produced —
   they do not exist in `metrics/by_metric/metal_lomo_auc_wasserstein.json`.
7. **This branch wrote nothing outside its ownership list.** No edit to
   `molmetal/scripts/pre_flight_check.py`, no edit to `docs/**`, `AGENTS.md`,
   `TODO/**`, `configs/**`, `metrics/**`, `registry/**`, `paper/**`, or
   `molmetal/reports/run_ledger.jsonl`. No commit, no push, no stash.
