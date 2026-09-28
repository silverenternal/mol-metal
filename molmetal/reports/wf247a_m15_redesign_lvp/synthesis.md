# Wf-247-A TODO-130 m15 REDESIGN — Ligand-vs-Pocket RMSD

Engine: RDKit `rdMolAlign.GetBestRMS` (symmetrizeConjugatedTerminalGroups=True)

Source: `molmetal/reports/wf179_bd9_m15_rmsd_p2/` (5-pocket × 3-seed × 5-pose docked)
Crystal: `molmetal/data/pocket_ref_ligands/ligand_metadata.json`

| Pocket | n_poses | median (Å) | best (Å) | succ@0.5A | succ@1.0A | succ@2.0A | Status |
|---|---|---|---|---|---|---|---|
| 1h36 | 15 | 4.6907 | 4.3078 | 0.0000 | 0.0000 | 0.0000 | MEASURED |
| 3eml | 15 | 3.8333 | 3.4528 | 0.0000 | 0.0000 | 0.0000 | MEASURED |
| 3ny8 | 5 | 3.0744 | 2.7741 | 0.0000 | 0.0000 | 0.0000 | MEASURED |
| 4aaw | 15 | 3.2488 | 2.3293 | 0.0000 | 0.0000 | 0.0000 | MEASURED |
| 4rlu | 15 | 4.6305 | 4.5157 | 0.0000 | 0.0000 | 0.0000 | MEASURED |

Cross-pocket aggregate: median = 3.8333 Å, min = 3.0744 Å, max = 4.6907 Å, std = 0.6736 Å.

Mean success at 2.0 Å (Tier-1 SOTA threshold): 0.0000 (0/5 pockets ≤ 2.0 Å).

## Honest framing note

RMSD computed by RDKit rdMolAlign.GetBestRMS (canonical SOTA path), NOT the Wf-179 hand-rolled Kabsch+Hungarian matcher.  Crystal ligand coords come from molmetal/data/pocket_ref_ligands/ligand_metadata.json (Wf-139 P1 ship).  Best-of-5-pose-per-seed headline per Vina evaluation protocol.  For 3ny8 only seed 0 succeeded under Wf-179 (10 poses expected, 5 measured); all other pockets have 15 poses. GetBestRMS uses substructure-match alignment, which is STRICTER than Kabsch+Hungarian; reported RMSDs are typically HIGHER than the legacy Wf-179 numbers (which we preserve as 'legacy_kabsch_hungarian_median_A' per pocket).  This is honest framing of the GetBestRMS limitation, not a bug.
