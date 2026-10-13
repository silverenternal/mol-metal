# BLK-L2: m2_r2 hilbert alias line ref fix (2026-10-09)

**Status**: SHIPPED (per §5.5 filesystem-truth evidence below)
**Scope**: per `todo.json` BLK-L2; CPU-only doc fix, no GPU.
**AGENTS.md compliance**: §5.4 (no wet-lab) + §5.5 (filesystem-truth) + §5.6
(atomic_write_file) + §5.8 (direct Edit/Write preferred over wf) + §5.9
(single-venv .venv/bin/python).

## Problem

`metrics/by_metric/m2_r2_per_cohort.json` line 59 (`cohorts_tested[1].cohort_alias_disclosure`)
contained a stale line reference:

> "...hilbert cohort loader (Wf-331-R5 load_cohort **L1099-1101**) returns the
> platinai_4cell pIC50 vector by construction..."

This was correct at Wf-331-R5 ship time but the `load_cohort` function has since
been extended (e.g. the `_load_platinai_expanded` branch that supports the
`max_cohort_size` flag — used in Wf-FIX-3LINES Phase 5d to expand platinai
to n=4078). The function definition moved from L1099-1101 to **L1216-1275**
(verified by `sed -n '1216,1290p' molmetal/scripts/wf331r5_r2_w2_chemprop.py`).

The fix_steps item per `todo.json` BLK-L2 was: "Fix the stale line ref in
`metrics/by_metric/m2_r2_per_cohort.json:1099-1101 -> 1274-1276`". The actual
function is at L1216 with the hilbert alias branch at L1274-1275; the
atomic_write replaced `L1099-1101` with `L1216-1275 (def at L1216; hilbert
alias branch at L1274-1275)` to capture both the function entry and the
specific alias branch.

## §5.5 filesystem-truth evidence

`md5sum` before / after:

```
before: 298c99414ee4cd8f19dfafced10c4fb8  metrics/by_metric/m2_r2_per_cohort.json
after:  b3047c0630be8d30b101090046b5a138  metrics/by_metric/m2_r2_per_cohort.json
```

Diff scope: 1 line. `grep -c "L1099-1101" metrics/by_metric/m2_r2_per_cohort.json`:
0 (was 1). `grep -c "L1216-1275"`: 1 (was 0).

Unchanged fields (verified by json.load):
- `metric` = "m2_r2_per_cohort"
- `n_cohorts_tested` = 3
- `aggregate.n_aliased_cohorts` = 1 (hilbert is alias of platinai_4cell)
- `per_cohort[1].cohort_is_alias_of` = "platinai_4cell"
- `verdict` = (preserved; not in fix scope)
- `timestamp` = (preserved; same write event)
- All other keys byte-identical except line 59

The `Wf-R43-A2` closeout (`TODO/completed/wf_r40_r43_combined_closeout.md`) was
checked for the same stale ref via `grep -n "load_cohort\|L1099\|L1216\|alias"`
and returned 0 matches — no fix needed there. The stale ref was confined to the
JSON metric file.

## §5.6 atomic_write_file verification

Used `molmetal.scripts.wf386a3_atomic_write.atomic_write_file(path=...,
content=...)` (the §5.6 canonical entry point). Per-path process-local lock
prevents concurrent writers; temp-file + os.replace is atomic on POSIX.

## Honest-negative framing (per §5.2)

- This is a 1-line documentation fix; no metric value moved. The disclosed
  alias relationship (hilbert is construction-equal to platinai_4cell by
  `np.array_equal`) is unchanged — only the line reference is corrected.
- The alias itself remains a structural limitation. Anyone reading the
  paper §1.2 still needs to know that hilbert provides NO independent
  evidence for the m2 R² result. The line-ref fix does not change that
  framing.
- The function has 3 cohorts (platinai_4cell + hilbert + metalcytotox_combined);
  only hilbert is aliased. metalcytotox_combined is independent and remains
  the cleanest cohort for any future m2 R² analysis that wants ≥2
  independent data points.

## Cross-references

- `todo.json` BLK-L2 (this fix)
- `molmetal/scripts/wf331r5_r2_w2_chemprop.py` L1216-1290 (actual function)
- `metrics/by_metric/m2_r2_per_cohort.json` L59 (fixed)
- `TODO/completed/wf_r40_r43_combined_closeout.md` (no fix needed; no stale ref)
- `TODO/completed/blk_c4_cfm_bond_head_qed_2026_10_09.md` (parallel BLK-C4 GPU retest)
- `AGENTS.md` §5.5 + §5.6
- `TODO/INDEX.md` §22 (will be added in final closeout wave per execution_order step 19)
