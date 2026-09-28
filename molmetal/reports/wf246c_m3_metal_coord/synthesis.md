# Wf-246-C Synthesis: cisplatin × 1h36 smoke

**Date**: 2026-09-28  
**Workflow**: Wf-246-C TODO-130 m3/m10 REDESIGN — Vina metal-coordination term
**Engine**: AutoDock Vina 1.2.7 (Python binding)  
**Exhaustiveness**: 8  
**Seeds**: 0, 1, 2


## §1 Metal-coordination composite

- **Composite**: 0.9970443349753695
- **kcal/mol adjustment**: +0.994
- **Pocket donor counts**: {'N': 91, 'O': 110, 'S': 2, 'P': 0}


## §2 Per-seed results

| Seed | Plain Vina (kcal/mol) | Metal-coord (kcal/mol) | Δ (kcal/mol) | Status |
|---:|---:|---:|---:|---|
| 0 | -3.594 | -2.600 | +0.994 | ok |
| 1 | -3.576 | -2.582 | +0.994 | ok |
| 2 | -3.576 | -2.582 | +0.994 | ok |

## §3 Aggregate (n=3/3)

- **Plain Vina mean**: -3.582 ± 0.010 kcal/mol
- **Metal-coord mean**: -2.588 ± 0.010 kcal/mol
- **Mean Δ**: +0.994 kcal/mol


## §4 Honest framing

- This is a **smoke test** (1 pocket × 3 seeds × 1 ligand), NOT a
  full 5-metal × 3-seed × 10-pocket sweep. The purpose is to
  verify the metal-coordination term's plumbing end-to-end.
- The kcal/mol adjustment is a **soft bonus/penalty** added on
  top of Vina's kcal/mol — the docked pose is unchanged (Vina's
  MC search is not affected).
- For organic-only ligands the adjustment is exactly **0.0**
  (backward-compatible with all earlier Wf-30 / Wf-121 sweeps).
- The metal-coord composite (1.0 for cisplatin) reflects:
  geometry=1.0 (Pt(II) CN=4 square-planar) + HSAB=1.0
  (Pt prefers N donors from cisplatin's 2 NH3) + distance=None
  (no 3-D pocket context supplied in this smoke).
- kcal/mol label: `MEASURED_GENERIC_ATOM` + `MEASURED_METAL_COORD_TERM`.
  Full chemotype-tuned VDW awaits Wf-143 P3 ship (Pt/Mg atype rewrite).

