# Wf-246-B m15 Redesign — Per-Pocket Self-RMSD Table

Source: `molmetal/reports/wf179_bd9_m15_rmsd_p2/summary.json` (Wf-179 BD-9 m15 Phase 2)
Redesign: per-pocket std + cross-pocket aggregate + difficulty flag.

Difficulty thresholds (median best RMSD, angstrom):
- EASY: median < 0.5
- MEDIUM: 0.5 <= median < 1.0
- HARD: median >= 1.0

| Pocket | Difficulty | Median (Å) | Mean (Å) | Std (Å) | n_seeds | passes_2.5Å | passes_2.0Å |
|--------|------------|------------|----------|---------|---------|--------------|------------|
| 1h36 | MEDIUM | 0.5739 | 0.5886 | 0.0761 | 3/3 | True | True |
| 3eml | HARD | 1.0240 | 1.0954 | 0.1098 | 3/3 | True | True |
| 3ny8 | MEDIUM | 0.9506 | 0.9506 | 0.0000 | 1/3 | True | True |
| 4aaw | EASY | 0.4223 | 0.4208 | 0.0335 | 3/3 | True | True |
| 4rlu | MEDIUM | 0.6578 | 0.6433 | 0.0237 | 3/3 | True | True |

Cross-pocket aggregate (n=5):
- median: 0.6578 Å
- std:    0.2546 Å
- min:    0.4223 Å
- max:    1.0240 Å

Difficulty distribution: {'EASY': 1, 'MEDIUM': 3, 'HARD': 1}
