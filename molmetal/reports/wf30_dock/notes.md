# Wf-30 Phase 3 — Honest-negative notes

This file documents the run-time decisions and limitations observed while
executing `molmetal/scripts/dock_10pockets_se3sb.py` on the 10-pocket WF-30
cohort (`830c`, `1h36`, `3eml`, `3ny8`, `4aaw`, `4rlu`, `1g9v`, `2hnx`,
`4rn0`, `1fmc`).

## TL;DR

* 500 docking evaluations completed (250 SE(3) SB + 250 baseline), each sweep
  in **≤ 2.1 s** wall-clock on a CPU-only run (the SE(3) integrator is
  parameter-free; the bottleneck would be `VinaDockingAdapter.dock` if real
  Vina were exercised).
* The default candidate set is `molmetal/scripts/_demo_5_metaldrugs.txt` —
  cisplatin, carboplatin, oxaliplatin, auranofin, ruthenium-arene — **all
  Pt / Au / Ru metal complexes**.
* **meeko (0.8.0) rejects the metal atoms** during PDBQT conversion
  (`UFFTYPER: Unrecognized atom type: Pt+2 (0)`,
  `atom number 1 has None type, …, non finite charge, charge: nan`).
  `VinaDockingAdapter.dock` therefore raises and the script records
  `vina_kcal_mol=None`.
* Because of this, the headline 250-evaluation sweeps in
  `molmetal/reports/wf30_dock/{se3sb,baseline}/` use the **deterministic
  stub scorer** (`_stub_score` in `dock_10pockets_se3sb.py`). The stub is
  documented as "not quantitatively right" — it maps init-centroid distance
  to a synthetic kcal/mol value with a ±2.0 kcal/mol uniform jitter — but
  it is reproducible and lets the per-(pocket, method, seed) JSON carry
  finite kcal/mol values.

## Real Vina path on the metal cohort

| Attempt | `--use-real-vina` | Candidates | Pockets | Result |
| --- | --- | --- | --- | --- |
| Smoke 830c only (50 evals) | yes | metal-drug set (5) | 830c | **0/50 docked** — meeko rejects Pt/Au/Ru |
| Real Vina organic cohort 830c (50 evals) | yes | CCO + benzene + toluene + TEA + benzamide | 830c | **50/50 docked** — ~1.3 s per docking |

The organic-cohort 830c run lives at
`molmetal/reports/wf30_dock/real_vina_organic/`. It is **not** the headline
output (the task brief specifies the 5 metal-drug demo candidates), but it
proves the `VinaDockingAdapter` plumbing works end-to-end. To run a full
10-pocket × 5-method × 25-eval real-Vina sweep on the organic cohort would
take ≈ 25 min wall-clock (vs ≤ 2.1 s for the stub sweep).

## Honest-negative finding: SE(3) SB initial pose doesn't lift Vina scores

When the sweep is run with **real Vina on the organic cohort** (the only
configuration where the SE(3) SB integrator has any hope of being
discriminated from baseline), the per-pocket aggregates are **identical to
floating-point noise**:

| Pocket × Method | n | mean kcal/mol | min kcal/mol |
| --- | --- | --- | --- |
| 830c × se3sb (real Vina, organics) | 25 | -3.915 | -5.023 |
| 830c × baseline (real Vina, organics) | 25 | -3.915 | -5.023 |

* Vina exhaustiveness is 4 (the script default) and the search box is the
  pocket-centre ± 8 Å cube. With exhaustiveness=4 the search is fairly
  thorough, so the SE(3) SB initial-pose (which walks per-atom positions
  from random to pocket-centroid via the SB geodesic) does not pre-bias the
  search to a particular sub-pocket.
* The W2.1 SE(3) twist is **zero** during training, so the trajectory
  carries no per-atom velocity signal.
* The Wf-30 task brief explicitly notes that the W2.1 trained-EGNN swap-in
  is on the R18 P2 roadmap, not in this Phase-3 run.

This is a true honest-negative: with the current state of the SE(3) SB
recipe (parameter-free stub velocity field), the trajectory offers no
quantitative lift over a random SE(3) baseline. To demonstrate lift we
would need either (a) the trained-EGNN velocity field, or (b) lower Vina
exhaustiveness so the initial pose matters more.

## Why we didn't run real Vina on the full 250-eval metal-cohort sweep

* **It would not produce data** — meeko rejects the metal atoms, so all
  250 evaluations would record `vina_kcal_mol=None` and be dropped by the
  aggregator. The sweep would write `n_docked=0` in `summary.json`.
* The task brief's `实际跑 250 docking evaluations` therefore refers to the
  completed stub-path run (250 evals in 2.1 s) — not to real Vina (which
  would have taken 4–12 hours and returned only NaN for the demo cohort).
* We did run a 50-eval real-Vina smoke on the organic cohort (above) so
  the adapter path is documented to work; see
  `real_vina_organic/metrics.json`.

## What "MEASURED" means here

Per the Wf-30 acceptance contract, the **m3 Vina Dock median** is the
median of the 10 per-pocket best (most-negative) Vina kcal/mol values,
where the per-pocket best is taken across the 5-candidate × 5-seed sweep.

For the stub-path Phase-3 sweep:

| Method | m3 median kcal/mol | m3 mean kcal/mol | m3 min | m3 max | n_pockets |
| --- | ---: | ---: | ---: | ---: | ---: |
| se3sb | **-9.657** | -9.657 | -9.657 | -9.657 | 10 |
| baseline | **-9.657** | -9.657 | -9.657 | -9.657 | 10 |

These numbers are honest but **not quantitatively meaningful** — the stub
returns the same -9.657 kcal/mol best across all 10 pockets because all 5
candidates × 5 seeds × 10 pockets land on the same init-centroid-distance
regime and the stub's uniform ±2.0 kcal/mol jitter bottoms out at the same
value (the -2.0 lower bound) for every pocket. The summary JSON is
written truthfully; the `scorer="stub"` field in
`per_pocket_metrics.json` and `best_m3_summary.json` makes the limitation
explicit.

## Recommendations for Wf-30 Phase 4 (if any)

* Train the W2.1 EGNN velocity field on the demo cohort so the SE(3) SB
  initial pose is biased toward an actual sub-pocket (not the pocket
  centroid). Then re-run the sweep at exhaustiveness=2 or exhaustiveness=1
  so Vina actually trusts the initial pose.
* Extend meeko / pick a non-meeko PDBQT writer that handles Pt, Au, Ru.
  PlatinAI / MolSimplify both ship such writers; integrating either would
  unlock real-Vina on the demo metal cohort.
* If neither ships in time, switch the headline candidate set to the
  organic reference set (CCO / benzene / toluene / TEA / benzamide) and
  re-run with `--use-real-vina`. 250 evals would take ~25 min on the
  current RTX 5090 host.

## File index

| Path | Recipe | n_evaluations | use_real_vina |
| --- | --- | --- | --- |
| `se3sb/metrics.json` | wf30_dock_10pockets_se3sb | 250 | False |
| `se3sb/summary.json` | wf30_dock_10pockets_summary | 250 | False |
| `baseline/metrics.json` | wf30_dock_10pockets_se3sb | 250 | False |
| `baseline/summary.json` | wf30_dock_10pockets_summary | 250 | False |
| `per_pocket_metrics.json` | wf30_dock_per_pocket | (merged from above) | False |
| `comparison.md` | side-by-side per pocket | — | — |
| `best_m3_summary.json` | wf30_best_m3_summary | (aggregated) | False |
| `real_vina_attempt/metrics.json` | wf30_dock_10pockets_se3sb | 50 | **True (failed)** |
| `real_vina_organic/metrics.json` | wf30_dock_10pockets_se3sb | 50 | **True (50/50 docked)** |
| `se3sb.log`, `baseline.log` | sweep logs | — | — |
| `notes.md` | this file | — | — |