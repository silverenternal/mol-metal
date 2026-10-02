# Wf-310D Vina pt-strip full aggregation -- claim cards

Date: 2026-09-29
n_total: 100  n_parse_ok: 97  n_pdbqt_ok: 97  n_docked: 97
grand mean kcal/mol: -5.564  std: 1.767
wall_min: 0.00  n_pockets: 1

## Card 1 -- per-pocket summary (MEASURED)

| pocket | n_docked | mean_kcal_mol | std | best_kcal_mol | in_band |
|--------|----------|---------------|-----|---------------|---------|
| 1h36 | 97 | -5.564 | 1.767 | -10.299 | no |

## Card 2 -- pass-gate verdict (MEASURED)

- Field-avg band: [-9.5, -7.5] kcal/mol
- Wf-301B smoke (CPU, 5 mols on 1h36): -4.60 kcal/mol
- Full grand mean: -5.564 kcal/mol
- n_docked: 97/100

## Card 3 -- honest-negative flags (per Wf-301B honest-negative)

- Wf-301B smoke mean (-4.60) is ABOVE field-avg band (-7.5 to -9.5);
  recovery axis is *being able to dock at all* (pt-strip enables
  docking; absolute binding affinity is a follow-up).
- Top-1 kcal/mol may exceed -9.5 even when grand mean is in band
  (per Wf-301B honest-negative: -10.299 vs -9.5).
- Full 4-pocket panel provides per-pocket spread; use min/max to
  bracket the pocket-cohort variance.

## Card 4 -- wall + GPU cost

- Wall: 0.0 min (0.00 hr) -- MEASURED.
- GPU: CPU-only (Vina binary is single-threaded; cuda:1 unused).
- ~1-2 hr PROJECTED per Wf-306A Phase-B budget.

## Card 5 -- verdict

HONEST_NEGATIVE_DOCK -- n_docked=97/100; 1-pocket grand mean -5.564 kcal/mol; pt-strip recovery axis not fully realised.
