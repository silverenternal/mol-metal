# Wf-77 Phase 2 — ASTSR-only R² baseline synthesis

**Date**: 2026-09-26
**Constraint**: ASTSR-only (no Hilbert, no Multi-PH); per party-twenty-spirit §3.5.
**Feature set**: F5_RDKIT_2D 12-d RDKit 2D descriptors (Wf-49 Phase 2 ship).

## Cross-dataset summary

| Dataset | n_samples | overall R² mean ± std | per-target (mean ± std) | collapse? |
|---|---:|---|---|---|
| platinai_4cell | 4078 | -0.4028±0.0000 | ic50_mcf7_nm: -0.5340±0.0258<br>ic50_a549_nm: -0.5905±0.0045<br>ic50_a2780_nm: -0.5258±0.0129<br>ic50_a2780cis_nm: -0.4883±0.0054 | yes |
| metalcytotox_5metal | 20000 | -0.2841±0.1230 | activity: -0.2843±0.1230 | yes |

## Notes

- **ANCHOR role**: this ASTSR-only R² is the Wf-56 R20-P2 complement of Wf-77 P1 Hilbert-only.  Any composite R² lift on the same data is attributed to the Hilbert dual / Multi-PH augmentation rather than to the ASTSR search alone.
- **Feature set**: F5_RDKIT_2D 12-d RDKit 2D descriptors (Wf-49 Phase 2 ship).  The legacy F5 5-feature set is NOT used here because Wf-49 §Cause 2 (CRITICAL) confirmed it is non-informative on pIC50.
- **CPU-only**: the ASTSR evolutionary search runs in fp32 tournament selection; no GPU acceleration is required or used.
- **Honest framing**: per party-twenty-spirit §3.5, negative R² values are NOT hidden.  If `overall R² < 0` consistently across 5 seeds, the composite stack must lift *above* this number to claim any attribution.

## Per-seed R² detail

### platinai_4cell

| seed | overall R² | wall_s | collapse |
|---:|---:|---:|:---:|
| 0 | -0.4028 | 7.523 | yes |
| 1 | -0.4028 | 5.703 | yes |
| 2 | -0.4028 | 6.135 | yes |
| 3 | -0.4028 | 5.143 | yes |
| 4 | -0.4028 | 5.805 | yes |

### metalcytotox_5metal

| seed | overall R² | wall_s | collapse |
|---:|---:|---:|:---:|
| 0 | -0.4348 | 8.431 | yes |
| 1 | -0.1837 | 8.422 | yes |
| 2 | -0.1837 | 9.131 | yes |
| 3 | -0.4348 | 8.837 | yes |
| 4 | -0.1837 | 8.685 | yes |

## Best AST (lisp form, full-data refit per seed)

### platinai_4cell

- seed 0: `v_MolMR`
- seed 1: `v_MolWt`
- seed 2: `v_TPSA`
- seed 3: `v_MolWt`
- seed 4: `(mul v_MolMR (mul v_NRotB c_2.0))`

### metalcytotox_5metal

- seed 0: `c_0.0`
- seed 1: `c_0.5`
- seed 2: `c_0.5`
- seed 3: `c_0.5`
- seed 4: `c_0.5`

