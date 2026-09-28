# Wf-68 Phase 1 (R20-P5.1) — Catherine 6/6 Properties × 25+ d-block metals

**Date**: 2026-09-26T08:55:38Z
**#metals**: 25 / **#seeds**: 5 / **max_depth**: 1 / **#properties**: 6
**Mode**: synthetic fallback (CPU)

**Parent plan**: `TODO/pending/96_r20_p5_closure_catherine.md` §3 P5.1
**Back-link**: Wf-57 Phase 3 (`molmetal/reports/wf57_catherine_25metals/per_metal_pass.json`)

## Headline verdict

- **25 / 25 metals × 6 / 6 properties PASS** (target >= 25 / 25)
- **0 metals** with at least one Catherine property FAIL
- **25 / 25 metals** ran on synthetic fallback (no RDKit / molmetal_lam)

## Per-property pass-rate

| Property | #metals PASS / #metals total | Pass-rate |
|---|---:|---:|
| P1_reflexivity | 25 / 25 | 100.00% |
| P2_transitivity | 25 / 25 | 100.00% |
| P3_symmetry | 25 / 25 | 100.00% |
| P4_dblock | 25 / 25 | 100.00% |
| P5_reachability | 25 / 25 | 100.00% |
| P6_composability | 25 / 25 | 100.00% |

## Per-metal 6/6 verdict

| Metal | 6/6 | P1 | P2 | P3 | P4 | P5 | P6 | synth |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Sc | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Ti | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| V | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Cr | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Mn | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Fe | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Co | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Ni | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Cu | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Zn | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Y | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Zr | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Nb | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Mo | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Tc | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Ru | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Rh | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Pd | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Ag | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Cd | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Hf | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Ta | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| W | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Re | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |
| Os | YES | PASS | PASS | PASS | PASS | PASS | PASS | YES |

## File outputs

- `catherine_6_props.json` — combined sweep verdict + back-link to Wf-57
- `synthesis.md` — this file (Wf-68 P5.1 ship surface)
- `{metal}/catherine_props.json` — per-metal 6/6 verdict (one file per metal)
- `{metal}/bench_summary.json` — per-metal closure summary that fed the verifiers

