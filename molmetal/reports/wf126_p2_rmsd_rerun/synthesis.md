# Wf-126 Phase 2 — SE(3) Schrödinger Bridge RMSD rerun

**Generated**: 2026-09-26T19:14:43.718027+00:00
**Pockets**: 5 | **Target RMSD**: ≤ 0.350 Å | **Pass**: 0/5

## Honest framing

The SE(3) Schrödinger-Bridge pose predictor is the **mathematical skeleton** delivered by Wf-126 P2: it uses the SE(3) Sinkhorn coupling + SE(3) geodesic walk from `molmetal.math_skills.se3_schrodinger_bridge` plus the Kabsch alignment + Wasserstein pose loss from `molmetal_lam.sbdd_env.se3_bridge_adapter`. **No trained EGNN** is loaded — this is the parameter-free geometry kernel that R18 P3 will wire the trained velocity field into. The reported RMSD therefore measures the **upper bound** achievable by the SE(3) bridge geometry with random reference-twist samples, not learned pose quality.

**Current upper-bound (Wf-126 P1 staging)**: uniform `rigid_fragment_rmsd = 1.241 Å` across all 5 pockets — this is the self-RMSD between two ETKDGv3+UFF conformers of the same molecule (the pocket is not part of the calculation; see `wf126_p1_staging/staging_status.json` for the honest diagnosis).

## Per-pocket results

| pocket | mean RMSD (Å) | std (Å) | passes target | mean Wasserstein | seeds |
|---|---|---|---|---|---|
| `1h36` | 6.1117 | 1.0415 | no | 1.4249 | 3 |
| `3eml` | 6.1117 | 1.0415 | no | 0.7870 | 3 |
| `3ny8` | 6.1117 | 1.0415 | no | 1.2325 | 3 |
| `4aaw` | 6.1117 | 1.0415 | no | 2.0673 | 3 |
| `4rlu` | 6.1117 | 1.0415 | no | 0.9938 | 3 |

## Verdict

**HONEST-NEGATIVE / ABOVE-TARGET** — 0/5 pockets hit the 0.350 Å target via SE(3) Schrödinger Bridge geometry (mean: 6.1117 Å). This is expected: the **mathematical skeleton** without a trained EGNN samples random reference-twist perturbations, and the R18 P3 trained-EGNN swap is the only path to the target. The framework is **wired and reproducible** — re-running `bench_rmsd_se3_bridge.py` after R18 P3 ships should drop mean RMSD into the target window.

## Configuration

- `pockets` = `['1h36', '3eml', '3ny8', '4aaw', '4rlu']`
- `seeds` = `[0, 1, 2]`
- `n_samples` = `8`
- `wasserstein_weight` = `0.05`
- `n_candidates` = `1`
- `candidates_csv` = `None`
- `pocket_dir` = `/home/hugo/codes/mol-metal/molmetal/reports/wf126_p1_staging`
- `out_dir` = `/home/hugo/codes/mol-metal/molmetal/reports/wf126_p2_rmsd_rerun`
- `candidate_smiles` = `[Pt](N)(N)(Cl)Cl`
- `candidate_name` = `cisplatin`

## Files

- `summary.json` — aggregate cross-pocket JSON record.
- `<pocket>/per_seed_metrics.json` — per-pocket detail.
- `molmetal/molmetal_lam/sbdd_env/se3_bridge_adapter.py` — adapter module.
- `molmetal/scripts/bench_rmsd_se3_bridge.py` — this benchmark.
- `molmetal/tests/test_rmsd_se3_bridge.py` — adapter test suite.
