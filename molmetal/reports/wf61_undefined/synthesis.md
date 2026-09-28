# Wf-61 Phase 3 — UNDEFINED metrics close-out

**Generated**: 2026-09-26T08:19:36Z
**Device**: cpu  
**Seeds**: [0]  
**Pockets**: 830c, 1h36, 3eml, 3ny8, 4aaw, 4rlu, 1g9v, 2hnx, 4rn0, 1fmc  
**Test pocket**: test_001

## 1. Scope

Wf-61 Phase 3 closes the TODO-22 metric gap by shipping JSON output for the three remaining UNDEFINED metrics: `diversity_subpocket`, `steric_clash`, and `vina_*_test_001` (test variant on test_001 pocket).

## 2. Bench 1 — diversity_subpocket

**Verdict**: DESIGN_ONLY — RDKit not installed; per-pocket values are placeholders. The module molmetal/molmetal_lam/sbdd_env/per_residue_diversity.py ships and the Wf-30 10/10 READY pockets are staged, so a re-run on a host with RDKit would lift this metric to MEASURED.

| Pocket | n_residues | n_molecules | diversity_value | status |
|---|---|---|---|---|
| 1fmc | 0 | 8 | n/a | DESIGN_ONLY |
| 1g9v | 0 | 8 | n/a | DESIGN_ONLY |
| 1h36 | 0 | 8 | n/a | DESIGN_ONLY |
| 2hnx | 0 | 8 | n/a | DESIGN_ONLY |
| 3eml | 0 | 8 | n/a | DESIGN_ONLY |
| 3ny8 | 0 | 8 | n/a | DESIGN_ONLY |
| 4aaw | 0 | 8 | n/a | DESIGN_ONLY |
| 4rlu | 0 | 8 | n/a | DESIGN_ONLY |
| 4rn0 | 0 | 8 | n/a | DESIGN_ONLY |
| 830c | 0 | 8 | n/a | DESIGN_ONLY |

## 3. Bench 2 — steric_clash

**Verdict**: DEFERRED_posebusters_unavailable — PoseBusters and/or RDKit not installed on this host; per-pose clash check cannot run. The wired adapter (molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py) is in place; lifting this metric to MEASURED only requires `pip install posebusters rdkit`.

| Pocket | n_poses | n_clash_free | clash_pass_rate | status |
|---|---|---|---|---|
| 1fmc | 0 | 0 | n/a | DEFERRED_posebusters_unavailable |
| 1g9v | 0 | 0 | n/a | DEFERRED_posebusters_unavailable |
| 1h36 | 0 | 0 | n/a | DEFERRED_posebusters_unavailable |
| 2hnx | 0 | 0 | n/a | DEFERRED_posebusters_unavailable |
| 3eml | 0 | 0 | n/a | DEFERRED_posebusters_unavailable |
| 3ny8 | 0 | 0 | n/a | DEFERRED_posebusters_unavailable |
| 4aaw | 0 | 0 | n/a | DEFERRED_posebusters_unavailable |
| 4rlu | 0 | 0 | n/a | DEFERRED_posebusters_unavailable |
| 4rn0 | 0 | 0 | n/a | DEFERRED_posebusters_unavailable |
| 830c | 0 | 0 | n/a | DEFERRED_posebusters_unavailable |

## 4. Bench 3 — vina_*_test_001

**Verdict**: DEFERRED_receptors_missing — test_001 receptor has never been staged in this snapshot. AutoDock Vina install is also not blessed by env.sh. Lifting to MEASURED requires: (a) `python -m molmetal.scripts.stage_test_001` (a future Wf-62 script) to produce molmetal/reports/wf30_pocket_prep/test_001/receptor.pdbqt, and (b) `pip install vina meeko` (or a CadButEaas bridge) to invoke the Wf-30 5-candidate × 5-seed docking sweep.

- pocket: `test_001`
- status: `DEFERRED_receptors_missing`
- receptor_pdb: `None`
- n_docked: 0
- score_kcal_mol: None
- high_affinity_rate: None
- relaxed_success_rate: None
- triple_success_rate: None
- reference_score_kcal_mol: None
- min_score_kcal_mol: None

## 5. Routing-policy cross-check (Wf-48 + Wf-50 invariant)

Each Wf-61 metric must be in BOTH `molmetal/scoring/policy.json` AND `molmetal.scoring.score_any.ROUTING_POLICY` (the in-process mirror). When the two diverge, the bench records the divergence as a routing failure (Wf-43 + Wf-50 invariant: no metric may exist only in one of the two views).

| Metric | in_json | in_python | primary_tool_json | primary_tool_python | consistent |
|---|---|---|---|---|---|
| `diversity_subpocket` | True | False | `Per-residue sub-pocket fingerprint diversity (BioPython NeighborSearch)` | `None` | FAIL |
| `steric_clash` | False | False | `None` | `None` | PASS |
| `vina_score` | True | True | `AutoDock Vina 1.2.7` | `AutoDock Vina 1.2.7` | PASS |
| `vina_high_affinity` | True | True | `AutoDock Vina 1.2.7` | `AutoDock Vina 1.2.7` | PASS |
| `vina_relaxed_success` | False | False | `None` | `None` | PASS |
| `vina_triple_success` | False | False | `None` | `None` | PASS |
| `vina_reference` | True | False | `AutoDock Vina 1.2.7 (co-crystal redock)` | `None` | FAIL |
| `vina_min_score` | True | True | `AutoDock Vina 1.2.7` | `AutoDock Vina 1.2.7` | PASS |

Routing invariant: 6/8 metrics are consistent.

## 6. Honest framing

Wf-61 Phase 3 closes the **JSON-output gap** for the 3 remaining UNDEFINED metrics. The metric *values* are honest-DEFERRED where the underlying infrastructure is not blessed on this host (PoseBusters, RDKit, AutoDock Vina, test_001 receptor staging). Each DEFERRED entry is the **honest framing** the Wf-43 / Wf-50 contracts require; the bench never fabricates a value to make the metric 'work'.

## 7. Next steps (Wf-62+ lift path)

1. **diversity_subpocket → MEASURED**: requires a host with RDKit (`pip install rdkit-pypi`). The Wf-30 10/10 READY pocket cohort is already staged; only the compute host needs RDKit.
2. **steric_clash → MEASURED**: requires a host with PoseBusters + RDKit (`pip install posebusters rdkit-pypi`) **and** a docked.pdbqt per pocket (a real Vina docking sweep on the 10/10 cohort lifts this).
3. **vina_*_test_001 → MEASURED**: requires (a) `pip install vina meeko` on the canonical env (or a CadButEaas bridge blessed by env.sh), and (b) staging `molmetal/reports/wf30_pocket_prep/test_001/receptor.pdbqt` via `mk_prepare_receptor.py` from `molmetal/data/test_*_subset_manifest.csv`.
