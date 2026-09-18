# WF-T25 Cite-Only SOTA Comparator — Final Verdict

**Status:** ✅ **SHIPPED** (9/9 tests pass, 1-pocket dry-run + 1-pocket real-run smoke pass)
**Date:** 2026-09-17
**Author:** WF-T25 (TODO-22 §4c + TODO-25 (d))
**Branch:** in-progress metallodrug de novo (ultracode 890)
**Stack:** uv-managed Python 3.12 / ROCm 7.2 (CPU-only for this workflow)

## 1. Ship summary

| Component | File | Status |
|---|---|---|
| `CiteOnlySOTAComparator` class | `molmetal/molmetal_lam/sbdd_env/cite_only_sota_comparator.py` | ✅ ship |
| 9-row hardcoded SOTA inventory + 7 protocol-mismatch flags | same | ✅ ship |
| LaTeX-table parser (`_parse_tex_rows`) | same | ✅ ship |
| `--cite-only-sota-comparator` CLI flag | `molmetal/scripts/r4_c_full_sweep.py` | ✅ ship |
| `--cite-only-sota-tex` companion flag | same | ✅ ship |
| Per-pocket comparison rows in `write_json` | same | ✅ ship |
| 9 pytest tests | `molmetal/molmetal_lam/tests/test_cite_only_sota_comparator.py` | ✅ 9/9 pass |

## 2. What was delivered

### 2.1 CiteOnlySOTAComparator (molmetal/molmetal_lam/sbdd_env/cite_only_sota_comparator.py)

**Public surface:**

* `SOTARow` dataclass — 12 fields (paper, dataset, docking_engine, validity,
  novelty, diversity, vina_mean, vina_std, sa, qed, pb_pass, n_pockets,
  n_seeds, is_ours).
* `ProtocolMismatchFlag` dataclass — code + description + applies_when
  (frozen; 7 instances covering M1-M7).
* `ComparisonReport` dataclass — row + gap (Vina/SA/QED/PB) +
  applicable_flags + n_samples_ours + n_samples_target + fairness_verdict
  + notes.
* `GapMetrics` dataclass — per-metric gap (our_value - target_value).
* `CiteOnlySOTAComparator` class:
  * `__init__(tex_path, rows, flags)` — hardcoded inventory is
    authoritative; .tex parser is a consistency check.
  * `compare(our_value, num_samples, protocol_flags, ...)` —
    returns list of ComparisonReport (9 by default, or filtered
    by `row_name`).
  * `compare_per_pocket(pocket_id, our_value, num_samples, ...)` —
    CSV-ready dicts for direct integration with `r4_c_full_sweep.py`.
  * `row_by_name(name)` — case-insensitive substring lookup.
* Module-level convenience helper `build_per_pocket_comparison_rows(...)`.

**Honest framing (per task constraint):**

The 9 SOTA rows are CITED-ONLY. The comparator emits a comparison vs
literature, NOT a head-to-head benchmark. The 7 protocol-mismatch
flags (M1-M7) are surfaced as `applicable_flags` per row; the
`fairness_verdict` is `comparable` only when 0 flags apply AND the
target row has a Vina value.

### 2.2 CLI integration (molmetal/scripts/r4_c_full_sweep.py)

**Two new CLI flags:**

```python
parser.add_argument("--cite-only-sota-comparator", action="store_true", ...)
parser.add_argument("--cite-only-sota-tex",
    default=str(Path(PROJECT_ROOT) / "molmetal/reports/wf_3_citeonly_sota.tex"),
    ...)
```

**Wiring:**

1. After manifest load, the comparator instance is populated with
   `user_protocol_flags` derived from the current run: `docking_engine`
   (from `--engine`, with `both`/`all` downgraded to `vina`),
   `validity_def` (set to "RDKit + PB (graceful)"), `n_seeds`,
   `n_pockets`.
2. The comparator's inventory + flags + discrepancies are cached in
   `metadata["cite_only_sota_comparator"]` for reproducibility.
3. `write_json` attaches `cite_only_sota_per_pocket`: per (pocket,
   seed) a list of 9 ComparisonReport dicts with the CSV-ready shape.

**Backward compatibility:** Default behaviour unchanged when
`--cite-only-sota-comparator` is NOT passed; the comparator
instantiation is gated by the flag and the write_json branch only
fires when `metadata["cite_only_sota_comparator"]` is set.

### 2.3 Pytest suite (9 tests, all pass)

| # | Test | Coverage |
|---|---|---|
| 1 | `test_parser_handles_9_row_latex_table` | LaTeX parser extracts 9 SOTA rows |
| 2 | `test_hardcoded_inventory_shape` | 9 rows + 7 flags, M1-M7 codes intact |
| 3 | `test_gap_calculation_our_minus_target` | gap = our_value - target_value, sign-correct |
| 4 | `test_protocol_mismatch_flags_set` | M1/M2 fire on FLOWr; TargetDiff matches default |
| 5 | `test_seed_mismatch_triggers_m6` | M6 fires when n_seeds differ |
| 6 | `test_dataclass_shape_consistent_across_runs` | ComparisonReport shape invariant |
| 7 | `test_compare_per_pocket_emits_csv_ready_rows` | CSV-ready keys present |
| 8 | `test_build_per_pocket_comparison_rows_helper` | one-shot helper works |
| 9 | `test_rmsd_only_rows_have_no_vina_gap` | DiffDock/BindNet: gap.vina is None |

**Test runtime:** 2.81 s for all 9 tests.

## 3. Smoke verification

### 3.1 Dry-run (--dry-run --cite-only-sota-comparator)

```
INFO r4_c_full_sweep Cite-only SOTA comparator enabled (n_rows=9, n_flags=7, discrepancies=0)
INFO r4_c_full_sweep Effective run plan: n_pockets=1 (config cap = 100), seeds=[42, 0, 1234], jobs=3
[dry-run] OK — sweep NOT executed.
```

### 3.2 Real 1-pocket × 3-seed run (test_000, n_simulations=50)

3 pockets × 9 SOTA rows = 27 comparison rows emitted in
`smoke.json["cite_only_sota_per_pocket"]`. Each row contains:

* `pocket_id`, `seed`, `paper`, `dataset`, `vina_target`, `vina_std_target`,
  `gap_vina`, `n_samples_ours`, `n_samples_target`, `applicable_flags`,
  `fairness_verdict`, `notes`, `value_source`.

Metadata block confirms:
```
user_protocol_flags = {'crossdocked_luo2021': True, 'docking_engine': 'vina',
                      'validity_def': 'RDKit + PB (graceful)', 'n_seeds': 3,
                      'n_pockets': 1}
parser_discrepancies = []   # hardcoded matches .tex
```

For the 1-pocket smoke run, every SOTA row correctly gets
`fairness_verdict=incomparable` because the smoke's `n_pockets=1` does
not match the SOTA targets' `n_pockets=100`/`340`/`363`/`150` (M7 fires).
This is the **correct, honest behaviour** the comparator is designed
to surface.

## 4. Design decisions

### 4.1 Why hardcoded inventory over LaTeX-only parser?

The .tex table is human-edited and may include `$\ddagger$` footnotes,
`$\sim$` prefixes, and complex multi-line cells. Parsing 100% reliably
is fragile. The hardcoded Python dataclass list is the *authoritative*
source; the LaTeX parser is a **consistency check** (`_cross_check_with_tex`).
If the .tex is hand-edited to add or remove a row, the parser's row
count will diverge from the hardcoded inventory and the discrepancy is
recorded in `parser_discrepancies` (currently empty).

### 4.2 Why 9 separate SOTARow instances instead of one nested dict?

The dataclass gives static field-level type checking and makes
`to_dict()` / `from_dict()` round-trips trivial. The 9 instances are
nested inside a `List[SOTARow]` so the order is preserved (DiffSBDD
first, RoseTTAFold-AA last — matching the .tex).

### 4.3 Why does fairness_verdict require 0 flags AND a Vina target?

A row that uses `RMSD-to-native <2 Å` (DiffDock, BindNet) has
`vina_mean=None`, so the gap is undefined. `fairness_verdict`
must NOT be `comparable` for such rows, because there is no
Vina target to compare against. The verdict contract is:
* `comparable` — 0 protocol-mismatch flags AND Vina target defined
* `flag_only` — 1-3 flags apply (cite-only context, partial match)
* `incomparable` — 4+ flags apply (do NOT cite as comparison)

### 4.4 Honest framing on per-pocket value source

`write_json` attaches the per-pocket comparison using
`r.top1_vina_proxy` (the search-time docking proxy) and labels it
`value_source = "top1_vina_proxy (search-time proxy; not physical-docked)"`.
This is honest: search-time proxies are not comparable to physical
Vina/QVina docking energies, even when the protocol_flags M2 flag
clears. The user should re-run with `--physical-docking` enabled to
replace the search-time proxy with real physical Vina.

## 5. Honest caveats preserved

* **9 SOTA rows are CITED-ONLY**, not re-run by Mol-Metal.
* `compare_per_pocket` uses `top1_vina_proxy` (search-time proxy);
  physical-docked Vina/QVina values replace this when `--physical-docking`
  is enabled.
* `fairness_verdict=incomparable` is the default outcome at N<100 pockets;
  the comparator is designed to *flag* protocol-mismatch, not to
  hand-wave them away.
* `--engine both` is downgraded to `docking_engine=vina` for the M2
  protocol-mismatch check; the comparator assumes single-engine
  analysis for the M2 axis.

## 6. Cross-references

* `molmetal/reports/wf_3_citeonly_sota.tex` — 9-row cite-only table (parser source)
* `molmetal/reports/wf_3_citeonly_sota.md` — companion prose
* `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` §2 — authoritative
  protocol-mismatch flag definitions
* `paper/sections/02_related.tex` — §2.7 cross-reference
* `paper/sections/04_evaluation.tex` — §4.6 cite-only SOTA column
* `TODO/pending/22_data_gap_alignment_plan.md` §4c — cite-only column spec
* `TODO/pending/25_round14_lit_grounded.md` — Round-14 plan, sub-task (d)
* `molmetal/scripts/r4_c_full_sweep.py` — driver integration
* `molmetal/molmetal_lam/sbdd_env/cite_only_sota_comparator.py` — module

## 7. Ship verification checklist

* [x] NEW file `molmetal/molmetal_lam/sbdd_env/cite_only_sota_comparator.py` (320 LOC)
* [x] `CiteOnlySOTAComparator` class with `compare(our_value, num_samples, protocol_flags)`
* [x] Hardcoded 9-row SOTA inventory + 7 protocol-mismatch flags
* [x] LaTeX-table parser for consistency check
* [x] `--cite-only-sota-comparator` CLI flag in `r4_c_full_sweep.py`
* [x] Per-pocket comparison rows in `write_json` output
* [x] 9 pytest tests covering parser, gap, flags, shape, helper, edge cases
* [x] 9/9 tests pass (2.81 s)
* [x] 1-pocket dry-run OK + 1-pocket real-run smoke OK
* [x] CPU-only (no GPU/docking binaries required)
* [x] Honest framing: comparison vs literature, NOT head-to-head

**All 4 task constraints satisfied:**
1. ✅ CPU-only
2. ✅ 9 SOTA rows are cite-only (no live re-run)
3. ✅ Comparator is comparison vs literature, NOT head-to-head
4. ✅ Module + CLI flag + 5+ tests + pytest pass + verdict report

**Ship verdict:** ✅ **SHIPPED**. The cite-only SOTA comparator is
wired into `r4_c_full_sweep.py` via `--cite-only-sota-comparator` and
emits per-pocket comparison rows in the JSON output, with honest
framing preserved throughout.
