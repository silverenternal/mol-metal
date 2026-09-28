# Wf-30 Phase 4 — m3 Vina Dock median MEASURED + lift analysis

Generated: 2026-09-26T04:51:43+0000

## TL;DR

- **m3 Vina Dock median (se3sb)**: **-9.6574 kcal/mol**
- **m3 Vina Dock median (baseline)**: **-9.6574 kcal/mol**
- **Lift (se3sb − baseline)**: **+1.2963e-08 kcal/mol** (≈ 0; floating-point noise)
- **Stub scorer?** Yes (`scorer=stub`, `use_real_vina=false`). The Phase-3 sweep on the demo metal-drug cohort could not exercise real Vina because meeko rejects Pt/Au/Ru atoms during PDBQT conversion. See `notes.md` for the honest-negative rationale.
- **Ship gate (≤ -8.5 kcal/mol)**: stub-pass (se3sb median -9.6574 < -8.5), but this is a property of the stub floor, not a real-method result. **Reported as honest negative**: no quantitative lift can be claimed.

## Definition

**m3 Vina Dock median** = median of the 10 per-pocket best (most negative) Vina kcal/mol values across the WF-30 10-pocket cohort, where the per-pocket best is taken across the 5-candidate × 5-seed sweep.

- Cohort (10 pockets): 1fmc, 1g9v, 1h36, 2hnx, 3eml, 3ny8, 4aaw, 4rlu, 4rn0, 830c
- Candidates: 5 demo metal-drugs (cisplatin, carboplatin, oxaliplatin, auranofin, ruthenium-arene)
- Seeds: 5 (0..4)
- Total evaluations per method: 250

## Per-pocket best (most negative kcal/mol)

| Pocket | se3sb best | baseline best | Δ (se3sb − baseline) |
|---|---:|---:|---:|
| 1fmc | -9.657403 | -9.657404 | +8.9035e-08 |
| 1g9v | -9.657403 | -9.657403 | +1.6318e-08 |
| 1h36 | -9.657404 | -9.657404 | +8.0356e-08 |
| 2hnx | -9.657403 | -9.657403 | +4.9593e-08 |
| 3eml | -9.657403 | -9.657403 | +7.9658e-08 |
| 3ny8 | -9.657404 | -9.657404 | +7.6646e-11 |
| 4aaw | -9.657404 | -9.657404 | +1.8333e-07 |
| 4rlu | -9.657404 | -9.657404 | -3.8840e-08 |
| 4rn0 | -9.657403 | -9.657403 | -5.8236e-08 |
| 830c | -9.657404 | -9.657404 | +5.4221e-10 |

## Aggregate statistics (across the 10 per-pocket best values)

| Stat | se3sb | baseline |
|---|---:|---:|
| median | -9.657404 | -9.657404 |
| mean | -9.657404 | -9.657404 |
| min | -9.657404 | -9.657404 |
| max | -9.657403 | -9.657403 |
| std | 0.000000 | 0.000000 |

## Lift analysis

### vs R17 baseline (-1.2885 kcal/mol on 830c, real Vina, 100 CuAAC click products)

- R17 reference: **-1.2885 kcal/mol** (`metrics/by_metric/vina_score.json`).
- Phase 4 se3sb m3 median: **-9.6574 kcal/mol**.
- Naive delta (Phase 4 − R17): **-8.3689 kcal/mol**.
- **This naive delta is NOT directly comparable**: the Phase 4 sweep used the stub scorer (real Vina would NaN-out for Pt/Au/Ru demo candidates because meeko rejects them), the cohort is 10 pockets (not just 830c), and the candidate pool is metal-drugs (not CuAAC click products).

### vs SOTA target (-9.37 kcal/mol TransDiffSBDD)

- SOTA target: **-9.37 kcal/mol** (TransDiffSBDD CrossDocked, Guan 2024).
- Phase 4 se3sb m3 median: **-9.6574 kcal/mol** — formally meets the SOTA value (within stub determinism).
- **Caveat**: this is a stub-floor artifact. Real-Vina Vina scores on a real CrossDocked-style run are expected to be considerably less negative than -9.37 for this candidate pool.

### vs Ship gate (≤ -8.5 kcal/mol)

- Ship gate: **-8.5 kcal/mol**
- Phase 4 se3sb m3 median: **-9.6574 kcal/mol** — formally passes.
- **Caveat**: the pass is a stub-scorer property (the stub's uniform ±2.0 kcal/mol jitter bottom-out is -9.66 kcal/mol across all pockets). It does not validate any method-level capability.

## Honest-negative verdict

**No quantitative lift can be claimed.**

- Per-pocket best kcal/mol for `se3sb` and `baseline` agree to ~1e-7 to 1e-11 kcal/mol (floating-point noise from the stub's deterministic jitter).
- The aggregate median lift is **+1.3e-08 kcal/mol** (effectively zero).
- This is consistent with the Phase-3 finding that the SE(3) SB trajectory offers no quantitative signal at this exhaustiveness with the current parameter-free stub velocity field.

### Why we did not run real Vina on the full metal-cohort sweep

- The demo metal-drug candidates (cisplatin, carboplatin, oxaliplatin, auranofin, ruthenium-arene) are rejected by meeko 0.8.0 at PDBQT conversion (`UFFTYPER: Unrecognized atom type: Pt+2 (0)`, `atom number N has None type, ... non finite charge, charge: nan`).
- A real-Vina sweep on the metal cohort would therefore record `vina_kcal_mol=None` for every one of the 500 evaluations and yield `n_docked=0` in `summary.json`.
- We did run a 50-eval real-Vina smoke on the 5-organic-molecule cohort (CCO/benzene/toluene/TEA/benzamide) at 830c (`real_vina_organic/`). That adapter path works end-to-end (~1.3 s/docking) and is the recommended next-step path if the metal-cohort meeko blocker is resolved (see `notes.md` § Real Vina path on the metal cohort).

### Recommendations for a future lift measurement

- Train the W2.1 EGNN velocity field on the demo cohort so the SE(3) SB initial pose is biased toward an actual sub-pocket (not the pocket centroid).
- Re-run the sweep at Vina exhaustiveness=2 (or 1) so the initial pose actually matters for the search.
- Extend meeko (or pick a non-meeko PDBQT writer like PlatinAI / MolSimplify) to handle Pt/Au/Ru so real Vina can run on the demo metal cohort.
- If none of the above ships in time, switch the headline candidate set to the organic reference set (CCO / benzene / toluene / TEA / benzamide) and re-run with `--use-real-vina`. 250 evals would take ~25 min on the current RTX 5090 host.

## Sources

- `molmetal/reports/wf30_dock/per_pocket_metrics.json` — per-(pocket, method, seed) kcal/mol table
- `molmetal/reports/wf30_dock/se3sb/metrics.json` — 250 raw se3sb evaluations
- `molmetal/reports/wf30_dock/baseline/metrics.json` — 250 raw baseline evaluations
- `molmetal/reports/wf30_dock/best_m3_summary.json` — Phase-3 best-of-N pre-aggregation
- `molmetal/reports/wf30_dock/notes.md` — Phase-3 honest-negative rationale
- `molmetal/reports/wf30_dock/real_vina_organic/` — real-Vina smoke on the organic cohort (works)
- `molmetal/reports/wf30_dock/m3_lift_analysis.json` — this analysis in JSON form
- `metrics/by_metric/vina_score.json` — R17 reference (real Vina, 830c, 100 CuAAC click products)
- `metrics/by_metric/vina_kcal_per_mol.json` — updated with this measurement
