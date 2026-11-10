# Wf-A114 (Cluster A) — the trainer must EMIT `molecule_validity` + the honest end-to-end rate

**Date**: 2026-10-30
**Cluster**: A (trainer + column contract)
**GPU**: none. 0 training runs. 0 CUDA calls. CPU-only tests + one 1-epoch CPU
smoke driven by a PRE-EXISTING test (not by this cluster).
**Verdict**: SHIPPED (writer-level).  NO training was run, so NO new generation
number is MEASURED here — this change makes the NEXT run's numbers honest; it
does not produce one.

---

## 1. The defect (already MEASURED, not re-derived here)

The trainer wrote `gen_validity = _gen_ok / max(_gen_tot, 1)`.  When the
generation eval drew NO samples, that expression is a MEASURED `0.0000` —
indistinguishable from "every sample the model drew failed to parse".
MEASURED on disk over the census's own 59 runs
(`metrics/by_metric/cfm_run_census.json` → `runs[*].run_path`, **538 data
rows**): **143 rows carry `gen_validity_n == 0`; 140 of them publish the
fabricated `0.0000`** — spread across **16 runs**.  (The other 3 of the 143
already carry the post-fix `nan` sentinel, all in
`wf385a3_scaffold_wire_in/train`, which was re-generated after this cluster
shipped.)  **16 of the 140 are LAST-epoch rows, in 16 distinct runs** —
including `cfm_bond_head_100ep_2026_10_09` epoch 99 and
`cfm_bond_head_20ep_2026_10_09` epoch 19 — so those runs' headline number was
invented by the writer.

> **CORRECTION (2026-10-30, Wf-A114C).** The original text here read *"119 of
> 496 generation rows across 25 runs"* and *"two of them are LAST-epoch
> rows"*.  Both figures were measured over the run set as it stood during the
> Wf-A114 audit and do **not** reproduce against the disk as of 2026-10-30:
> the run set grew (the census now loads 59 runs, and the c7 calibration /
> `wfa113_t42` / `wf385a3` runs landed after the audit), which moved the
> denominators to 538 rows and 143 zero-sample rows, and the LAST-epoch count
> from 2 to **16 across 16 runs**.  The corrected figures above are re-derived
> from the CSV files themselves over the census's run set, not from the audit.

The predicate is **`gen_validity_n == 0`** and nothing else.  `gen_eval_n == 0`
is NOT the predicate: **no row on disk carries `gen_eval_n` literally `0`
alongside `gen_validity_n > 0` (0 rows)**.  The near-miss case is different —
**11 epoch rows** (`cfm_tmqm_all6_retrain`, every epoch) carry
`gen_validity_n > 0` and **no `gen_eval_n` column at all**, because that run's
CSV header predates the column; the per-seed record is *absent*, not zero
(e.g. `cfm_tmqm_all6_retrain` epoch 10 reads `gen_validity` 0.75 = 12 of 16).

## 2. What changed, with file:line

### `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`

| line | change |
|---|---|
| 1103 | `GEN_UNMEASURED_PREDICATE = "gen_validity_n == 0"` — the ONE named predicate |
| 1106 | `_gen_measured(gen_n) -> bool` — `gen_n > 0` |
| 1119 | `_chem_val(chem, key) -> Optional[float]` — key PRESENCE, never truthiness; NaN/inf are NOT MEASURED |
| 1146 | `_end_to_end_rate(gen_validity, gen_single_fragment_ratio)` — the product; None if either is None |
| 1166 | `_gen_metric_fields(gen_ok, gen_tot, chem) -> Dict` — ONE source of truth for the epoch record, the CSV row and the event stream |
| 1940-2035 | the `EpochMetrics` rate fields are `Optional[float] = None`; `gen_end_to_end_rate` (1977), `gen_molecule_validity_total` (1976), `gen_measured` (1981) added; the COUNT fields stay `int` |
| 1568-1578 | the early-stop degradation note names the predicate: `f"{req} degraded to {x} (its own input was unmeasured: gen_validity_n == 0)"` |
| 18912 | the early-stop call now receives `gen_validity=None` when the epoch drew no samples |
| 19148 / 19304 | the epoch record and the `train_epoch` event both take `**_gen_fields` — they cannot drift apart |
| 20225 | new top-level JSON key `gen_metric_sentinel_note` |
| 2118 / 2196 / 2212 | module-level `CSV_HEADER`, `_fmt_rate` (None -> the literal `nan`), `csv_row` |
| 20338 / 20343 | the per_epoch_metrics.csv WRITER now calls `list(CSV_HEADER)` and `csv_row(m)` |
| 19876-19893 | `provenance` block now carries all SIX velocity-head levers |
| 20029-20042 | `config` block now carries all SIX velocity-head levers |
| 18149 / 18157 / 18790-18800 | per-seed `gen_validity` / `molecule_validity` are `None` when that seed drew no samples; `molecule_validity_n` added |

### `molmetal/scripts/wf_cfm_run_census.py` (line 133-143) — SPEC-MANDATED EXCEPTION TO THE FILE-OWNERSHIP LIST

The spec's item 8 states the census's `CFG_KEYS` MUST be widened, because
`{k: cfg.get(k) for k in CFG_KEYS if k in cfg}` filters every key that is not
in the tuple — so the new head levers would have been recorded by the trainer
and then dropped by the census, leaving the 2x2 attribution undecidable.
MEASURED: `velocity_head_init_knots` takes the value 9 on some configs on disk
and 5 on others, so two runs that differ only in that knob were
indistinguishable in their own artifact.  Six keys added; nothing else in the
file was touched by this cluster.  (A sibling cluster, Wf-A115, is rewriting
this file concurrently and has since added its own `HEAD_LEVER_KEYS` at :150
with the same six names; this cluster's edit survived that rewrite and is
consistent with it.)

### `molmetal/tests/test_cfm_gen_metric_columns.py` (NEW, 9 tests)

## 3. Two bugs this cluster found while implementing

**(a) The CSV writer was still the old one.**  The module-level `CSV_HEADER` /
`csv_row` existed, but the ACTUAL `per_epoch_metrics.csv` writer still carried
its own hand-maintained header literal and its own hand-maintained row literal,
and formatted the rate cells as `f"{m.gen_validity:.4f}"`.  With
`gen_validity` now `Optional[float]`, that expression raises `TypeError:
unsupported format string passed to NoneType.__format__` — the trainer would
have CRASHED on the first epoch whose generation eval drew no samples.  The
writer is now wired to the shared header and row builder (lines 20338 / 20343),
and test 2 pins that wiring.

Bit-exactness of the refactor was MEASURED by evaluating the OLD row literal
(from `git show HEAD:`) against the NEW `csv_row` on one fully-populated
epoch: **0 differences across all 52 pre-existing columns**, and exactly the
5 new columns added.

> **SCOPE OF THE "BIT-EXACT" CLAIM (2026-10-30, Wf-A114C).** The comparison
> above is **NAME-keyed**: each cell is looked up by column NAME.  It is **not**
> positional, and the two are not equivalent here.  MEASURED against the HEAD
> header literal: the old header has **52** columns, the new one has **57**,
> and the **5 new columns are INSERTED at index 34-38**
> (`gen_molecule_validity`, `gen_molecule_validity_n`,
> `gen_molecule_validity_total`, `gen_end_to_end_rate`, `gen_measured`),
> which shifts **18 pre-existing columns** (`gen_metal_ratio` at old index 34
> → new index 39, through `early_stop_criterion_note` at old index 51 → new
> index 56).  So a positional diff over the raw header lists would report
> **18 positional differences**, not 0.  What the claim establishes is the
> stronger property that actually matters: **no pre-existing column's NAME or
> VALUE changed** — the refactor added columns and moved nothing in place.
> 0 columns were removed.

**(b) `train_cfm_yuelbond_wasserstein.py --help` crashed on HEAD.**  argparse
%-formats every help string, so a bare `%` inside one is a `ValueError` AT
PRINT TIME, not at parse time.  Eight help strings carried one (e.g.
`~95% prior`, `81.8% of Pt`, `+51% epoch time`).  MEASURED on HEAD
(1b9470a0:11876) AND on the pre-fix working tree, identically:
`ValueError: unsupported format character 'p' (0x70) at index 123`, rc=1.
Consequence: every `--help` probe of this trainer was dead, including the
§5.7 smoke protocol's first check.  FIXED in this cluster's owned file by
escaping the eight bare `%` as `%%` (a help-text-only change; it cannot
alter training).  `--help` now exits rc=0 with 2131 lines.  No `%(name)s`
expansion existed in any help string, so no legitimate argparse substitution
was broken (verified: `grep -n "%%(" ` returns nothing).

## 4. Verification (CPU-only, no GPU, no training)

```
.venv/bin/python -m pytest molmetal/tests/test_cfm_gen_metric_columns.py \
    molmetal/tests/test_wfa113_molecule_validity.py \
    molmetal/tests/test_cfm_d3_gen_eval_multiseed.py \
    molmetal/tests/test_wf_a70_w3_metric_plumb.py \
    molmetal/tests/test_wfa114_cfm_fragmentation_gate.py \
    molmetal/tests/test_cfm_combined_gate_non_degenerate.py -q
  -> 106 passed (9 new + 97 pre-existing), 0 failed, 6.55 s

.venv/bin/python -m pytest molmetal/tests/test_cfm_yuelbond_wasserstein.py \
    molmetal/tests/test_cfm_bond_f1_gate.py \
    molmetal/tests/test_wf_blkc_m3_gen_validity_floor.py \
    molmetal/tests/test_cfm_d4_metal_presence_gate.py \
    molmetal/tests/test_wfa81_decode_failure_taxonomy.py \
    molmetal/tests/test_cfm_f6_t_per_cloud.py \
    molmetal/tests/test_wf_c5_gen_head_lr_isolation.py \
    molmetal/tests/test_wfa113_best_block_one_epoch.py \
    molmetal/tests/test_wf_a70_w3_metric_plumb.py \
    molmetal/tests/test_cfm_todo119_decode_ratio.py -q
  -> 123 passed, 1 skipped, 0 failed
```

No pre-existing test asserts on the `per_epoch_metrics.csv` COLUMN SET
(grepped `CSV_HEADER|fieldnames|csv_row|per_epoch_metrics.csv` over
`molmetal/tests/`): every hit is an unrelated CSV writer.  Every reader of the
trainer's CSV that touches a RATE column goes through
`float(...) -> NaN -> None` (`wf_cfm_run_census.py::_num`,
`cfm_fragmentation_gate.py::_f`), so the literal `nan` does not pollute a
median; the two readers of COUNT columns (`wf_blk_c13_epsilon_constraint.py`,
`wf_blk_c13_adaptive_head_replay.py`) read `gen_validity_val` /
`gen_validity_n`, which stay bare ints.

## 5. The broad sweep: 32 failures, ALL pre-existing on HEAD (proved, not asserted)

A wide sweep (`pytest molmetal/tests -q -k "cfm or census or fragment or
molecule_validity or per_epoch"`) reported
**32 failed, 988 passed, 18 skipped, 1 xfailed** in 926.78 s.  NONE of the 32
is caused by this cluster.  PROOF: a detached `git worktree` at HEAD
(1b9470a0) was checked out at `/tmp/wfa114b_head` and the SAME tests were run
there.  Every failure reproduces there with the SAME test id and the SAME
cause.

| failure group | n | root cause | HEAD proof |
|---|---|---|---|
| `test_wf350c2_detect_anomaly.py` | 6 | `--detect-anomaly` is in NEITHER trainer — `git show HEAD:...` has 0 hits for both the CFM and the D-MPNN trainer, and `train_dmpnn_multitask_wasserstein.py` is CLEAN in git status (this cluster never touched it) | identical 6 failures in the HEAD worktree |
| `test_wf356a1_data_quality_all_launchers.py` | 4 | `scripts/launch_wf124_p3_g4.sh` and `scripts/launch_wf300_cfm_main.sh` DO NOT EXIST on disk ("No such file or directory", rc=127) | identical 4 failures in the HEAD worktree |
| `test_wf_cfm_sweep_launcher_gate.py` | 5 | a required checkpoint is absent: `MISSING: .../cfm_patience10_seed0/checkpoints/cfm_yuelbond_wasserstein_seed0_last.pt` | identical 5 failures in the HEAD worktree |
| `test_wf415a2_preflight_path.py::test_preflight_pipeline_cfm_strict` | 1 | the live cfm decode canary's 50-step smoke loses only 0.151% against a 0.3% floor | the HEAD worktree's gate prints the SAME NUMBERS: `loss[0]=1.4227431118488312, loss[-1]=1.4205920100212097, decrease_pct=0.1511939723838226` — bit-identical, so this change does not touch the loss trace at all |
| `test_wf_blk_c5_cfm_collapse_repair_smoke.py` | 14 (13 now) | 3 were the `--help` crash (FIXED by §3b — the file is 14 failed / 8 passed before the escape and 13 failed / 8 passed after); the other 11 are two pre-existing disagreements — (i) `--bond-predictor` choices extended past the test's `{yuelbond,dmpnn}` expectation — the choices tuple is IDENTICAL on HEAD — and (ii) `--entropy-reg-weight` default `0.1` vs the test's expected `0.0` — `default=0.1` is IDENTICAL on HEAD; the 11 smoke failures are the trainer EXITING 1 because the test config's `yuelbond_vocab: pt` does not cover the 5-metal cohort ("Z [44,45,75,76,77] ... 64/64 metal atoms OOV"), a config/cohort mismatch | the WHOLE FILE was run in the HEAD worktree with the SAME `--help` escape: **13 failed, 8 passed — the identical 13 test ids**.  Before the escape this tree was 14 failed / 8 passed.  Net effect of this cluster: +1 green, 0 red |
| `test_wfa114_cfm_fragmentation_gate.py::test_gate_uses_the_trainers_own_decode_chemistry` | 1 | a TEST-ORDER artifact, not a code defect: `cfm_fragmentation_gate._trainer_decode_chemistry()` imports the trainer LAZILY, so after `test_wf350c2_detect_anomaly.py::_reload_module` the identity check compares two different module objects. Both files are UNTRACKED (another cluster's new files) and neither was touched here | PASSES in isolation (it is in the 106-passed run above) and FAILS in the HEAD worktree too (the files are not in HEAD) |

Net effect of the `--help` repair on the sweep: the blk_c5 argparse class goes
from 3 failures to 2, and the whole blk_c5 file goes from 14 failed / 8 passed
to 13 failed / 8 passed — identical to the HEAD worktree with the same escape.
No test that was green before this cluster went red, and one that was red
became green.

`--help` was also verified to be byte-stable in its OPTION SET: the 256 flag
names parsed out of `--help` on HEAD+escape and on this tree are identical
(`diff` empty), so the escape and the Cluster-A change added and removed no
flag.

## 6. What this cluster did NOT do, and why

1. **No 50-step CPU smoke of the changed path.**  AGENTS.md §5.7 requires it
   only for a NEW trainer FLAG; this change adds no argparse flag.  The spec's
   own verification (§10) is pytest-only and CPU-only.  A trainer smoke
   would have to construct a sampler and touch the CUDA path, which the
   NO-GPU constraint forbids.  The cheaper WRITER-LEVEL test is shipped
   instead and is labelled as such: 9 tests that exercise the row builder,
   the header, the dataclass, the early-stop degradation and the census
   reader — i.e. every code path this change touches, minus the trainer loop
   itself.
2. **No census re-run.**  `metrics/by_metric/cfm_run_census.json` is NOT in
   this cluster's file ownership, and a sibling cluster (Wf-A115) is rewriting
   `wf_cfm_run_census.py` concurrently and re-ran the census at 20:02.  The
   CFG_KEYS widening takes effect on the next census run.
3. **No new MEASURED generation number.**  The numbers quoted in §1 are the
   audit's, cited to the audit; nothing here is a new measurement.
4. **The two remaining blk_c5 argparse failures are NOT fixed here.**  Fixing
   them means changing a TRAINING DEFAULT (`--entropy-reg-weight` 0.1 -> 0.0)
   or narrowing the `--bond-predictor` choices tuple — both alter training
   semantics or a shipped checkpoint's head, which is another wave's decision,
   not a reporting-layer fix.  They are reported instead, with the HEAD proof
   that they pre-date this cluster.

## 7. Known follow-up (census-side, NOT this cluster)

The census's `_col` now drops the `nan` cells (required by the spec, and
correct for a MEDIAN), but the census's `*_end` columns are read as
"the last MEASURED epoch", not "the last epoch".  For a run whose LAST epoch
drew no samples, `gen_validity_end` will name an EARLIER epoch's value
instead of the fabricated `0.0000` — more honest than before, but still not
"the end".  The census should read the last ROW's value (None when
unmeasured) and keep the last-MEASURED value in a separate column.  That is
a census-side change and is flagged here rather than made.
