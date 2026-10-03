# Wf-330-B: Catherine 30-metal bench — 30/30 6/6 PASS verification

**Status:** SHIPPED (CPU-only)
**Date:** 2026-09-29
**Depends on:** Wf-330-A (P1+P6 BFS-bug fix)

---

## MEASURED outcome

```
$ PYTHONPATH=. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/bench_catherine_30metals.py --metals 30 --max-depth 1 \
    --output-dir molmetal/reports/wf330_catherine_30metals

n_metals                 : 30
6/6 PASS                 : 30 / 30
total properties PASS    : 180 / 180
wall_time_s              : 1.1959
[PASS] Catherine 30-metal bench ran on 30 metals (>= 30 expected, full 30-metal default)
```

Per-metal cohort: `Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn` (3d) + `Y, Zr, Nb, Mo,
Tc, Ru, Rh, Pd, Ag, Cd` (4d) + `La, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg` (5d+La).
Each metal: P1=PASS, P2=PASS, P3=PASS, P4=PASS, P5=PASS, P6=PASS.

## Honest framing

- **MEASURED**: 30/30 metals 6/6 PASS, 180/180 properties PASS on CPU
  (CadButEaas/.venv-vlm Python 3.12.13 + torch 2.14.0+cu130) — wall 1.20 s.
- **PROJECTED** (per the original skill §6 ledger): the bench was expected to be
  0/30 in CPU-only env without torch; the fix delivers 30/30.

## Skill acceptance gates

Per `mol-metal-catherine-dblock-bench` §4:

| Gate | Status |
|---|---|
| G2_catherine_30metal (CLI on ≥9 metals, exit 0) | **PASS** (CLI on 30 metals, exit 0) |
| G4_all_tests_pass (12 tests in `test_catherine_30metals.py`) | **PASS** (12/12) |
| G5_smoke_runs (--smoke 3-metal exit 0) | **PASS** (3/3 metals 6/6 PASS) |
| G6_no_breaking_changes (`bench_closure_dblock.TASK_SPEC_METALS` 9-metal subset unchanged) | **PASS** (subset preserved) |

## Output artefacts

- `molmetal/reports/wf330_catherine_30metals/per_metal_pass.json` (30 metals)
- `molmetal/reports/wf330_catherine_30metals/sweep_summary.md` (markdown table)
- `molmetal/reports/wf255_catherine_30metals/` (smoke 3-metal run, also updated)

## R20-P5 ship target

This delivery completes the R20-P5 (TODO-96) ship surface for the Catherine
30-metal bench. Paper §3.4 closure-theorem main result can now cite
**30/30 metals 6/6 PASS** with concrete MEASURED numbers per metal.

## Cross-references

- `molmetal/math_skills/catherine_properties.py` (verifier definitions)
- `TODO/completed/330a_catherine_bench_p1_p6_bug_fix.md` (the fix)
- `TODO/completed/96_r20_p5_closure_catherine.md` (R20-P5 closeout)
