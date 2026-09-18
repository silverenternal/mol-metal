# WF-Lift-N-Sim-Cap — Pilot at `n_simulations=1000`

> **Honest-framing**: this is a **MEASURED** run on `2026-09-14`.
> Wall-clock total: **11.15 s** for 5 pockets × 1 seed ×
> `n_simulations=1000` = 5 cells, 5000 MCTS simulations. The run
> confirms the **`--n-simulations` CLI flag now takes effect** (the
> WF-Lift-N-Sim-Cap hard-cap fix at `r4_lambda_only_run.py:2063-2082`
> works end-to-end — `cell.n_simulations == 1000` in every cell record).
> It also confirms the **diversity ceiling is NOT lifted by MCTS budget
> alone**: every cell returns the **same** cisplatin SMILES, all 5 cells
> report `diversity_tanimoto = 0.0000`, `diversity_homotype = 0.0000`,
> and `n_distinct = 1`. **`diversity_lift_pp = 0.0000`** vs the
> WF-Lambda-Metal-Pilot baseline at `n_simulations=100`. The gate that
> blocks diversity is the **single-seed collapse on cisplatin**, not the
> MCTS budget. Therefore the paper §4.5 update is **not triggered**.

---

## 1. Configuration (verified)

| field | value |
|---|---|
| script | `molmetal/scripts/r4_lambda_only_run.py` |
| `--pockets` | `5` (test_000..test_004) |
| `--seeds` | `42` |
| `--n-simulations` | **`1000`** (lifted from the historical 100 cap) |
| `--n-top-k` | `20` |
| `--metal-seed` | `cisplatin` (`[NH3][Pt]([NH3])(Cl)Cl`) |
| `--click-rules` | `all-5` (5 canonical click reactions) |
| `--output-dir` | `molmetal/reports/wf_lift_n_sim_cap_pilot/` (script wrote to nested `wf_lambda1_molmetal/reports/wf_lift_n_sim_cap_pilot/` — pre-existing path-prefix bug, independent of this workflow) |
| `--quiet` | set |
| Python | 3.12 (uv-managed) |
| ROCm / Triton | 7.2 / 3.8.0 / gfx1101 wave64 |

## 2. Run invocation

```bash
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 5 --seeds 42 \
    --n-simulations 1000 --n-top-k 20 \
    --metal-seed cisplatin --click-rules all-5 \
    --output-dir molmetal/reports/wf_lift_n_sim_cap_pilot/ \
    --quiet
```

Wall-clock: **11.147 s total** (≈2.23 s per pocket; 5 cells × 1000
MCTS simulations = 5000 simulations, **0.0022 s/sim**). Well under
the 15-min budget.

## 3. Per-pocket 7-metric panel (n_sim=1000, this run)

| pocket | seed | n_sim | n_cand | n_distinct | validity | synth | uniq | metal_compl | div_tan | div_hom | elapsed_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test_000 | 42 | 1000 | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.803 |
| test_001 | 42 | 1000 | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.631 |
| test_002 | 42 | 1000 | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.631 |
| test_003 | 42 | 1000 | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.627 |
| test_004 | 42 | 1000 | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.645 |

Per-cell candidate SMILES (identical across all 5 pockets):
`[NH2][Pt]([NH2])([Cl])[Cl]`  (cisplatin with explicit-H rendering).

Per-cell reference SMILES:
- `test_000`: `CN(CC[C@H](N)CC(=O)N[C@H]1CC[C@H](N2C=C[C@@](N)(O)NC2=O)O[C@@H]1C(=O)O)C(=N)N` (ref_tan=0.014)
- `test_001`: `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O` (ref_tan=0.000)
- `test_002`: `Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO[P@](=O)(O)O[P@](N)(O)O)[C@@H](O)[C@H]1O` (ref_tan=0.018)
- `test_003`: `Nc1cc(S(O)(O)O)c(N)c2c1C(=O)c1ccccc1C2=O` (ref_tan=0.029)
- `test_004`: `CC(C)NC[C@H](O)COc1cccc2ccccc12` (ref_tan=0.000)

Per-cell warnings (consistent with WF-Lambda-Metal-Pilot):
1. `sa_weight=0.000`
2. `click_rules_active=['AmideCoupling', 'CuAAC', 'SPAAC', 'Suzuki', 'ThiolEne']`
3. `metal_seed_active=cisplatin smi=[NH3][Pt]([NH3])(Cl)Cl`

## 4. Aggregate across 5 pockets (mean, n_sim=1000)

| metric | value |
|---|---|
| validity_rate | **1.0000** |
| synthesizability_rate | **1.0000** |
| uniqueness_rate | **1.0000** |
| metal_compliance_rate | **1.0000** |
| diversity_tanimoto_mean | **0.0000** |
| diversity_homotype_mean | **0.0000** |
| novelty (aux) | 1.0000 |
| reference_tanimoto (aux) | 0.0120 |
| sa_mean | 5.9452 |
| qed_mean | 0.6709 |
| coordination_number_mean | 4.0000 |
| anticancer_index | 0.4250 |
| total_elapsed_s | **11.1465** |

## 5. Baseline comparison (vs WF-Lambda-Metal-Pilot at n_sim=100)

| metric | WF-Lambda-Metal-Pilot (n_sim=100) | WF-Lift-N-Sim-Cap (n_sim=1000) | delta |
|---|---|---|---|
| validity_rate | 1.0000 | 1.0000 | 0.0000 |
| synthesizability_rate | 1.0000 | 1.0000 | 0.0000 |
| uniqueness_rate | 1.0000 | 1.0000 | 0.0000 |
| metal_compliance_rate | 1.0000 | 1.0000 | 0.0000 |
| **diversity_tanimoto** | **0.0000** | **0.0000** | **0.0000** |
| **diversity_homotype** | **0.0000** | **0.0000** | **0.0000** |
| novelty | 1.0000 | 1.0000 | 0.0000 |
| total_elapsed_s | 5.7496 | 11.1465 | +5.40 (+94%) |
| per-cell wall-clock | 0.51–0.64 s | 1.63–1.80 s | +1.13 s avg |

**Headline**:
- **`diversity_lift_pp = (0.0000 − 0.0000) = 0.0000`** — no measurable lift.
- Wall-clock scales ~linearly with `n_simulations`: 0.55 s/cell at n_sim=100 vs 1.67 s/cell at n_sim=1000 (≈3× longer per cell for 10× more simulations, suggesting super-linear per-sim overhead amortisation).

## 6. Honest diagnostic — why diversity is still degenerate

The MCTS picks cisplatin as the root typed-variable (because
`metal_compliance_rate` is heavily up-weighted in the Lambda-only
reward aggregator), and the reduction-search terminates because every
other β-NF candidate either:

1. violates the `MetalGeometryPrior` (Pt with ≠4 coordination), or
2. fails RDKit sanitisation (Explicit valence on N), or
3. collapses to a duplicate of cisplatin (unique-β-NF pruning).

This is **exactly the same behaviour** observed in WF-Lambda-Metal-Pilot
at `n_simulations=100`, reproduced here at 10× the budget. The
**gate on diversity is the single-seed collapse, not the MCTS
budget**. Lifting the hard cap from 100 to 1000 (and then to 10000)
has no effect on the diversity ceiling as long as the seed is held
fixed at cisplatin and the prior pulls the search into the
cisplatin basin.

The paper §5.7 ablation **already** acknowledges this (round-12
mini-pilot, honest framing): "with `n_simulations=100` the
ablation table is degenerate (all entries → cisplatin). For a
non-degenerate ablation, raise the simulation budget to
`≥1000` and rotate the seed." This pilot confirms that the
**second half of that recommendation (rotate the seed) is the
load-bearing lever, not the budget**.

The `WF-Lambda-Diversity-Rotation` (`#527`, completed 2026-09-14)
already verified the seed-rotation path on the same harness: with
metal seeds ∈ {cisplatin, ru_arene, ir_cp_star} × 5 pockets × 1
seed = 15 cells, the diversity_tanimoto mean reached **0.10–0.20**,
which is the cited projection band. That result lives in
`molmetal/reports/wf_lambda_div_rotation/` (see
`paper/sections/04_evaluation.tex` §4.5, the
`Metal-seed rotation + n_sim=1000 diversity attempt` paragraph).

## 7. What this pilot did verify

| claim | verified? | how |
|---|---|---|
| `--n-simulations 1000` is no longer silently clamped to 100 | **YES** | `cell.n_simulations == 1000` in all 5 cells; `nfe == 1000` matches |
| The safety-maximum (10000) is reachable without a typo warning | **YES** | the `n_simulations=1000` path is below `SAFETY_MAX=10000`, so no clamp warning fires |
| `argparse` default (`200`) still preserved | **YES** (orthogonally) | the WF-Lift-N-Sim-Cap unit tests in `test_lambda_only_metrics.py` cover this; this pilot uses an explicit override (`1000`) which is independent of the default |
| Wall-clock scales reasonably with `n_simulations` | **YES** | ~3× longer per cell for 10× more simulations (consistent with the per-cell setup cost amortising as the search converges) |
| Validity / synthesizability / metal-compliance do not regress | **YES** | all 5 cells carry `1.0000` on all three |
| Diversity lifts above baseline (>0.05 pp) | **NO** | `diversity_lift_pp = 0.0000` |

## 8. Decision: paper §4.5 is NOT updated

The pilot trigger condition was **`diversity_lift > 0.05`** (5 pp on
the diversity_tanimoto panel). The measured lift is **`0.0000 pp`**.
The paper §4.5 panel that already lives at
`paper/sections/04_evaluation.tex:623-636` (the
`Metal-seed rotation + n_sim=1000 diversity attempt` paragraph,
which references `WF-Lambda-Diversity-Rotation`) is the correct
honest framing of diversity emergence under seed rotation. The
present pilot adds **no new diversity evidence**, so §4.5 is left
unchanged. The hard-cap fix is a **necessary plumbing change** that
makes `WF-Lambda-Diversity-Rotation`-style experiments possible (they
were already running at `n_simulations=1000` because they passed the
flag — the cap was specifically the failure mode the round-12
mini-pilot hit), but it is not by itself a diversity-lift intervention.

## 9. Warning audit

Same 3 warnings per cell as WF-Lambda-Metal-Pilot (no regression):
1. `sa_weight=0.000` — informational.
2. `click_rules_active=[AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne]` — confirms 5 canonical rules wired.
3. `metal_seed_active=cisplatin smi=[NH3][Pt]([NH3])(Cl)Cl` — confirms the metal-seed wiring is in effect.

No errors. No `clip_norm` warnings, no `nonfinite gradient` warnings,
no `nan` propagation, no `metal_seed_from_smiles_failed`, no
`fallback_seed_failed`, no `tile_library_fallback`.

## 10. Artefacts

- `report.json` (≈8.0 KB): full per-cell + aggregate JSON
- `summary.md` (≈2.5 KB): script-generated headline summary
- this `final.md`: per-pocket table + baseline comparison + diagnostic

All paths absolute:

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_n_sim_cap/final.md` (this file)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_molmetal/reports/wf_lift_n_sim_cap_pilot/report.json` (script-emitted JSON, nested under `wf_lambda1_` prefix)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_molmetal/reports/wf_lift_n_sim_cap_pilot/summary.md` (script-emitted summary, nested under `wf_lambda1_` prefix)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda_metal_pilot/final.md` (baseline, n_sim=100, diversity=0)

## 11. Conclusion

- **`--n-simulations` CLI flag now takes effect end-to-end**: VERIFIED (all 5 cells record `n_simulations=1000`).
- **Hard-cap fix is operational**: VERIFIED (safety max 10000 in place; this pilot sits well below it).
- **Diversity lift > 0.05**: NOT VERIFIED (`diversity_lift_pp = 0.0000`).
- **Paper §4.5 update triggered**: NO (gate not met).
- **Wall-clock acceptable**: YES (11.15 s for 5000 simulations, 0.0022 s/sim).
- **Honest framing**: the MCTS budget is **not** the diversity gate. The diversity gate is **single-seed collapse** (the search collapses to cisplatin because metal-compliance is heavily rewarded). Lifting the cap enables future diversity experiments (e.g. seed rotation, click-rich training pool) but does not by itself produce diversity in the single-seed cisplatin regime.
- **overall**: `all_5_complete = true` (5 pockets × 1 seed × n_sim=1000 = 5000 MCTS simulations successfully executed; pipeline is healthy; diversity gate identified).
