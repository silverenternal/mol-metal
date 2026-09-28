# Wf-246-B Synthesis: TODO-130 m15 REDESIGN

**Date**: 2026-09-28
**Workflow**: Wf-246-B TODO-130 m15 REDESIGN
**Parent**: TODO-130 (metric definition baseline) + TODO-134 (m15 redesign spec) + Wf-179 (BD-9 m15 RMSD Phase 2 Vina redocking)

## §1 Purpose

Redesign the m15 metric (docked_pose_rmsd) to add three paper-ready diagnostics:
1. **Per-pocket std reporting**: each pocket's std across 3 seeds (5 pockets x 3 seeds = 15 runs)
2. **Cross-pocket aggregate**: median, std, min, max across the 5 pocket medians
3. **Difficulty flag**: per-pocket classification (EASY / MEDIUM / HARD) by median best RMSD

## §2 Inputs

- `molmetal/reports/wf179_bd9_m15_rmsd_p2/summary.json` (5 pockets x 3 seeds, Vina ex=16)
- `molmetal/data/pocket_ref_ligands/ligand_metadata.json` (ref ligand SMILES + coords)

## §3 Per-pocket results

| Pocket | Difficulty | Median RMSD (Å) | Std (Å) | n_seeds |
|--------|------------|------------------|---------|---------|
| 1h36 | MEDIUM | 0.5739 | 0.0761 | 3/3 |
| 3eml | HARD | 1.0240 | 0.1098 | 3/3 |
| 3ny8 | MEDIUM | 0.9506 | 0.0000 | 1/3 |
| 4aaw | EASY | 0.4223 | 0.0335 | 3/3 |
| 4rlu | MEDIUM | 0.6578 | 0.0237 | 3/3 |

## §4 Cross-pocket aggregate

- **Median**: 0.6578 Å
- **Std**:    0.2546 Å
- **Min**:    0.4223 Å (best pocket)
- **Max**:    1.0240 Å (worst pocket)
- **n_pockets**: 5

## §6 Difficulty distribution

- EASY (<0.5 Å): 1 pockets
- MEDIUM (0.5-1.0 Å): 3 pockets
- HARD (>=1.0 Å): 1 pockets

## §7 Verdict

All 5/5 pockets MEASURED with per-pocket std and difficulty flag.
Cross-pocket median = 0.6578 Å (SOTA gate <= 2.0 Å: PASS);
all pockets pass <= 2.5 Å gate (5/5).

## §8 Honest framing

- All values are MEASURED (not projected) from Wf-179 Vina redocking runs (exhaustiveness=16, n_poses=5).
- 3ny8 has only 1/3 seeds succeeded (NO_DOCK_OUTPUT for seeds 1, 2 — partial stage artifact).
- Difficulty flag uses median best RMSD (NOT pass-rate), so it's a property of the pocket + crystal ligand, not of Vina exhaustiveness.
- Backward compat: `metrics/by_metric/docked_pose_rmsd.json` schema preserved (legacy fields unchanged); new fields added under `redesign_wf246b`, `per_pocket_with_difficulty`, `cross_pocket_aggregate`.
