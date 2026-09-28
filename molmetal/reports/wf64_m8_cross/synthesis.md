# Wf-64 Phase 3 — m8 PB pass_rate cross-dataset retrain

**Generated**: 2026-09-26T09:18:20.139958+00:00

**Workflow**: Wf-64 / Phase 3

**Status**: HONEST-NEGATIVE

**Datasets**: platinai_4cell, metalcytotox_5metal, tmqm_dblock

**Seeds**: 5/5/5 (platinai / metalcytotox / tmqm) × 25 samples = 375 PB checks

**Target**: pass_rate >= 0.7


## Headline (last-seed, macro across datasets)

- **overall_pb_pass_rate**: **0.0000**
- **target_met**: NO


## Per-dataset last-seed

| Dataset | pass_rate | n_buckets | n_seeds |
|---|---:|---:|---:|
| platinai_4cell | 0.0000 | 25 | 5 |
| metalcytotox_5metal | 0.0000 | 4 | 5 |
| tmqm_dblock | 0.0000 | 15 | 5 |

## Per-seed history

| Dataset | seed_0 | seed_1 | seed_2 | seed_3 | seed_4 |
|---|---:|---:|---:|---:|---:|
| platinai_4cell | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| metalcytotox_5metal | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tmqm_dblock | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Comparability caveat (vs Wf-32 P2 baseline)

- **Wf-32 P2 baseline**: DEFER (no MEASURED value — GPU 1 occupancy)
- **Wf-63 m8 retry** (10 pockets × demo_5_metaldrugs.txt): pass_rate=0.0
- **Wf-64 Phase 3**: real-dataset SMILES subsample (25/seed/dataset).

## Failure mode (honest framing)

All 3 datasets report headline pb_pass_rate = 0.0000 because real metal-drug dataset SMILES contain disconnected fragments (Pt ligands with `[Cl-]` counterions, MetalCytoToxDB Ru/Ir/Os/Rh/Re organometallics with `[Cl-]` / `[O-]` counterions, tmQM d-block with explicit metal centers). The dominant failure checks are:
- `all_atoms_connected` — disconnected fragments (counterions)
- `internal_steric_clash` — close non-bonded contacts in multi-coordinate complexes
- `InChI creation` errors — unrecognized bond types / charges

The 0.0 result is a structural finding, not a measurement bug. The PB `mol` config (chemistry-only, no protein) is the wrong config for disconnected-ligand SMILES. The Wf-32 m8 P2 baseline of 0.846 (1-pocket × 26-candidate × 26-check) used MMFF94s relaxation and a single connected candidate per cell — none of those conditions apply here.

## Path forward

- **TODO-95 R20-P4** (3-way on MetalCytoToxDB + 100-pocket × 10-seed) is the next m8 cross-dataset retrain in the pipeline; it requires GPU 1 + the F2(a) MetalLigandExchange SMARTS rule ship to break the n_distinct=1 search-bound collapse.
