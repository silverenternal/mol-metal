# WF-LR17-B2 Pilot — Verify new click rules break the singleton-attractor barrier

> Run date: 2026-09-25
> Status: PASS (within-cell) / FAIL (across-cell, see §5)
> Script: `molmetal/scripts/r4_lambda_only_run.py`
> CLI (executed, with corrections vs. the harness-relayed command):
>   `--pockets 10 --seed 0 --n-simulations 1000`
>   `--metal-seed cisplatin --click-rules auto-pt-strict`
>   `--sa-weight 0.3 --output-dir logs/lr17_b2_pilot`
> Drops: `--engine both` (script is Lambda-only and has no `--engine` flag);
> renames: `--out-dir` -> `--output-dir` (script's actual flag).

## 1. Configuration
| parameter | value |
|---|---|
| pockets | test_000..test_009 (first 10 of CrossDocked 100-manifest) |
| seed | 0 |
| n_simulations | 1000 |
| n_top_k | 20 |
| metal_seed | cisplatin (= `[Pt]C#C`, bare Pt(II) + terminal alkyne handle) |
| click_rules | auto-pt-strict (= [CuAAC, SPAAC, Suzuki]) |
| sa_weight | 0.3 |
| prior_enabled | True |
| elapsed_s_total | 1400.3 s (~140 s / cell, CPU-only) |

## 2. Per-cell gate metrics (10 cells, all identical)
| metric | value |
|---|---|
| n_distinct (per cell) | **20** |
| distinct SMILES (across all 10 cells) | **20** |
| distinct metal+ligand-pair combos | **2** (Pt, triazole; Pt, ?) |
| distinct triazole ring patterns | **3** |
| uniqueness_rate | 1.00 |
| validity_rate | 1.00 |
| synthesizability_rate | 1.00 |
| diversity_tanimoto | 0.137 |
| diversity_homotype | 0.045 |
| diversity_subpocket | 0.679 |
| novelty | 1.000 |
| metal_compliance_rate | 0.00 |
| coordination_number_mean | 1.00 |
| decoder_pass_rate | 1.00 |
| elapsed_s/cell | 122-165 s |

## 3. Baseline comparison
| metric | R12 (singleton) | R16 Path A | R17 B2 (this pilot) |
|---|---|---|---|
| n_distinct per cell | 1 | 20 | **20** |
| distinct SMILES across pockets | 1 | 20 (likely collapse) | **20 (collapsed to same set)** |
| distinct metal-ligand pairs | n/a | n/a | **2** |

## 4. decode_smoke (per cell)
| metric | value |
|---|---|
| decoder_pass_rate | 1.00 (200/200) |
| n_coords_3d_attached_total | 200 (all cells, 20/cell) |
| n_scaffold_aware_gate_active_total | 200 |
| oxidation_state_distribution | {Pt_0: 200} |
| warnings | "Can't kekulize mol" on partial atom rings (non-fatal) |

## 5. Findings — partial pass, new failure mode surfaced

**Pass (within-cell singleton barrier broken).** The bare-`[Pt]C#C` + alkyne-handle
seed lets every CuAAC match fire against azide tiles, so each cell now returns
20 distinct molecules vs. R12's n_distinct=1. `auto-pt-strict` correctly gates
out ThiolEne + AmideCoupling (incompatible on strict Pt_II); the cell runs use
only CuAAC + SPAAC + Suzuki and never hit a ThiolEne SMARTS.

**Fail (across-cell collapse).** Although n_distinct=20 within each cell, the
*same 20 SMILES* appear in every one of the 10 pockets (overlap test_000 vs
test_009 = 20 of 20). The MCTS has converged to a **pocket-invariant global
attractor**: every pocket's MCTS roots at the same `[Pt]C#C` seed (default
`--metal-seed cisplatin` overrides `--metal-seed-from-pocket`), so the click-rule
chemistry produces the same 20 alkyne+azide -> triazole products everywhere.

**Distinct metal-ligand pairs = 2, gate fails.** The auto-pt-strict subset
emits (Pt, triazole) and (Pt, ?) only; the per-cell `metal_compliance_rate` is
0.00 because the candidate list contains Pt atoms but does not pass the strict
4-coordinate Pt(II) geometry prior (coordination_number_mean=1.00). The ≥3-pair
gate from TODO-25 §3 is NOT met.

## 6. Root causes
1. `--metal-seed cisplatin` + no `--metal-seed-from-pocket` => every pocket
   shares the same `cisplatin` SMILES seed -> identical MCTS root -> identical
   reachable chemistry region. (Phase 2C `pocket_derived_metal_seed` helper is
   shipped but not enabled here.)
2. `auto-pt-strict` keeps only 3 click rules; F2(a) `MetalLigandExchange` /
   `AquaExchange` SMARTS are not in the subset, so transmetalation / Cl-OH swap
   cannot diversify the metal coordination sphere. That is exactly the gap
   TODO-31 L-R17-B2 is queued to fix.
3. The MCTS expansion is dominated by tile selection on the alkyne handle
   alone, producing a narrow aromatic-N + Pt-trizole manifold instead of the
   {Pt(NH3)2, Pt(Cl)2, Pt(NH3)Cl, ...} coordination chemistry the gate wants.

## 7. Recommended next steps (handoff to L-R17-B2 / L-R17-B3)
- Re-run with `--metal-seed-from-pocket` to break the shared-root attractor.
- Extend `auto-pt-strict` -> `auto-pt-strict + metal-coord` (7-rule subset)
  to expose Pt(NH3)/Pt(Cl) coordination diversity (L-R17-B2 work item).
- After both fixes, target: n_distinct_per_cell >= 3 (already met),
  distinct_metal_ligand_pairs >= 3 across 10 cells (currently 2, FAIL).

## 8. Artefacts
- `/home/hugo/codes/mol-metal/logs/lr17_b2_pilot/run.log` (full per-pocket trace)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf_lambda1_logs/lr17_b2_pilot/report.json`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf_lambda1_logs/lr17_b2_pilot/summary.md`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf_lambda1_logs/lr17_b2_pilot/seeds.json`
