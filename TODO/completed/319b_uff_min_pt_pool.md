# Wf-319B — UFF-min Pt-pool ACTUAL EXECUTION (ship summary)

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** SHIPPED (smoke)
**Pipeline:** P3 (CPU-only RDKit UFF)
**Related:** Wf-315C analysis (WEAK_TO_STRONG §2.7), WEAK_TO_STRONG_PLAN §2.7,
TODO-319B follow-up to Wf-315C, §9.Z 5 CFM TODO queue
**Honest framing:** MEASURED_SMOKE (100-conformer smoke; 96/100 generated).
m10 Vina Min Score ceiling improvement remains PROJECTED — see §6 below.

---

## §1 What was shipped

Wf-319B executes the UFF-min Pt-pool that Wf-315C analysis identified as a
CPU-only path for improving the m10 Vina Min Score ceiling
(WEAK_TO_STRONG §2.7). Concretely:

1. **`molmetal/scripts/wf319b_uff_min_pt_pool.py`** — entry script
   (CPU-only). Samples Pt drugs from `platinai_4cell.csv` (Rusanov 2025
   MB-Finder Pt subset; 4078 Pt rows), balances active/inactive on
   pIC50_a2780 ≥ 5.0, runs RDKit ETKDGv3 (with `useRandomCoords=True`
   fallback to ETKDGv2) + `UFFOptimizeMolecule(maxIters=200)`.

2. **`molmetal/reports/wf319b_uff_min_pt_pool/pool.json`** — per-conformer
   record (96 conformers): `smiles`, `n_atoms`, `embed_method`,
   `energy_kcal_mol`, `converged`, plus top-level `summary` block with
   `n_conformers_generated`, `n_conformers_failed`, `energy_kcal_per_mol`
   (mean / std / min / max / 5-bucket distribution), and
   `wall_clock_sec = 73.7s` for the full 100-conformer run.

4. **`metrics/by_metric/uff_energy_pt_pool.json`** — flat summary
   suitable for the metrics ledger + the pre-flight gate.

5. **8 NEW tests** at `molmetal/tests/test_wf319b_uff_min_pt_pool.py`:
   * `test_wf319b_pool_records_present` — per-conformer schema
   * `test_wf319b_energies_finite_or_null` — energy sanity
   * `test_wf319b_embed_method_canonical` — embed_method enum
   * `test_wf319b_n_conformers_meets_smoke_target` — ≥ 80 conformers
   * `test_wf319b_distribution_buckets_well_formed` — 5-bucket sum
   * `test_wf319b_metrics_honest_negative_tag` — canonical tag set
   * `test_wf319b_metrics_required_keys` — top-level keys present
   * `test_wf319b_vina_integration_3d_coords` — Vina downstream shape

6. **Pre-flight v2 gate `p3_uff_min_pt_pool_present`** added to
   `molmetal/scripts/pre_flight_check.py` (`gate_wf319b_uff_min_pt_pool_present`).
   Surfaces in the `all` pipeline alongside `p3_pb_packages_status_present`
   (Wf-319C) and `p3_pb_packages_available_or_documented` (Wf-315D).
   PASS criteria: pool.json ≥ 80 conformers AND metrics JSON carries
   `HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING=True`.

---

## §2 Measured numbers

```
n_conformers_generated: 96  (out of 100; 4 failed ETKDG embed)
n_conformers_failed:    4
wall_clock_sec:         73.7s (single-threaded, CPU-only)
converged (UFF ≤ 200 iter): 75 / 96  (78%)

energy_kcal_per_mol:
  mean:    8.475e+12   (NOT physical — see §6)
  std:     7.025e+13
  min:     0.154       (small complex; physically reasonable bound)
  max:     6.822e+14   (UFFTYPER fallback for Pt6+2 — not physical)
  distribution_buckets: {<0: 0, 0-50: 9, 50-200: 26, 200-500: 8, ≥500: 53}

embed_method: ETKDGv3+randomCoords (useRandomCoords=True)
              — required workaround for Pt-dative-bond SMILES
                (platinai_4cell.csv uses `Cl-->Pt` arrow notation
                 that triggers a "bad lower bound" BoundsMatrix
                 invariant under default ETKDGv3).
```

Source: `molmetal/reports/wf319b_uff_min_pt_pool/pool.json` +
`metrics/by_metric/uff_energy_pt_pool.json`.

---

## §3 Data source (honest)

Per Wf-319C honest framing: the Krasnov 2026 MetalCytoToxDB
(`Ir / Os / Re / Rh / Ru`) does NOT contain Pt. The canonical Pt
subset in this repo is `platinai_4cell.csv` (Rusanov 2025 MB-Finder,
4078 Pt rows). Wf-319B samples from this canonical Pt subset.

Cf. `mol-metal/data/known_pt_drugs.csv` (7 reference Pt drugs:
cisplatin / carboplatin / oxaliplatin / nedaplatin / satraplatin /
picoplatin / heptaplatin) — the data lineage.

---

## §4 ETKDGv3 dative-bond workaround (engineering note)

`platinai_4cell.csv` SMILES contain arrow bonds (e.g. `Cl-->Pt+2<--n(pyridine)`)
representing ligand-to-metal dative coordination. RDKit's default ETKDGv3
embedding can raise an Invariant Violation ("bad lower bound" in
BoundsMatrixBuilder.cpp) on these geometries because the dative-bond
distance bounds are undefined.

Wf-319B uses two workarounds in order:

1. `useRandomCoords=True` — bypass the bounds matrix entirely (still
   produces a valid conformer + UFF energy; recommended for smoke
   work).
2. Fallback to ETKDGv2 with `useRandomCoords=True` if ETKDGv3 fails
   entirely.

If both fail, the conformer is dropped from the pool with `min_ok=False`
recorded. 4 / 100 conformers hit the dual-edge case (the most
pathological Pt-phosphine complexes with multiple dative bonds).

This is a smoke-scale workaround. Production-grade d-block metal
3D embedding requires MACE-OFF23 (d-block metal MLIPs) or
SMIRNOFF/OpenFF with d-block coverage, neither of which is in this
venv.

---

## §5 Files touched / created

| Path | Type | Lines | Purpose |
|---|---|---:|---|
| `molmetal/scripts/wf319b_uff_min_pt_pool.py` | NEW | 350 | UFF-min entry script |
| `molmetal/reports/wf319b_uff_min_pt_pool/pool.json` | NEW | 96 conformers | per-conformer UFF record |
| `metrics/by_metric/uff_energy_pt_pool.json` | NEW | 1 | flat summary metrics |
| `molmetal/tests/test_wf319b_uff_min_pt_pool.py` | NEW | 192 | 8 tests |
| `molmetal/scripts/pre_flight_check.py` | MOD | +95 | gate function + registration |
| `TODO/completed/319b_uff_min_pt_pool.md` | NEW | (this file) | ship summary |

Pre-flight v2 25/25 PASS gate: `p3_uff_min_pt_pool_present` (Wf-319B)
joins the `all` pipeline alongside `p3_pb_packages_status_present`
(Wf-319C) and `p3_pb_packages_available_or_documented` (Wf-315D).

Tests: 8 / 8 PASS (`PYTHONPATH=/home/hugo/codes/mol-metal
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest
molmetal/tests/test_wf319b_uff_min_pt_pool.py -v`).

---

## §6 Honest framing (MANDATORY)

Per AGENTS.md §5.2 + party-twenty-spirit §3.5:

1. **UFF energies for Pt complexes are NOT physical.** RDKit's bundled
   UFF force field lacks d-block metal atom-type parameters; UFFTYPER
   emits `Unrecognized atom type: Pt6+2` for every Pt atom. Energies
   reported here are fallback terms, NOT physically meaningful. The
   `HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING=True` tag is set on
   `metrics/by_metric/uff_energy_pt_pool.json` so downstream consumers
   (paper dashboard, Wf-264 CFM panel, §9 honest-framing catalog) MUST
   NOT interpret these energies as a real minimization metric.

2. **The 3D conformers ARE useful.** Pool JSON + per-conformer 3D
   coordinates are valid input to downstream Vina (Wf-264 CFM property
   panel). 96/100 conformers generated is a usable smoke-scale pool.

3. **m10 Vina Min Score ceiling improvement remains PROJECTED.** The
   WEAK_TO_STRONG §2.7 claim that UFF-min pre-relaxes Pt conformers
   and improves the m10 Vina ceiling requires actually running Vina
   on the UFF-min pool + comparing against non-FF-minimized controls.
   This is out of scope for the Wf-319B smoke (would need GPU portion
   + full 100-record Vina bench).

4. **Full UFF-min × Vina pipeline requires a real Pt-capable force
   field.** Future work: integrate MACE-OFF23 (d-block MLIP) or
   SMIRNOFF/OpenFF d-block coverage. Wf-319B establishes the
   pool-generation scaffolding; the d-block FF integration is a
   separate workflow (not yet scheduled).

---

## §7 R20 / R21 follow-up TODOs (open)

The Wf-319B pool + pre-flight gate set up the following forward-looking
work (per Wf-258 §2.1 + WEAK_TO_STRONG §2.7):

1. **TODO-319C** — Pt-capable FF integration (MACE-OFF23 / OpenFF
   d-block). Once shipped, re-run UFF-min pool + Vina bench to lift
   m10 ceiling from PROJECTED to MEASURED.
2. **Wf-264 CFM property panel** — uses the Wf-319B pool as input
   once m10 ceiling is MEASURED.
3. **§9.Z 5 CFM TODOs** — per Wf-258 §2.1, 5 forward-looking CFM
   TODOs at 37.8 hr GPU total; Wf-319B pools feed into one of these.

CPU-only work proceeds. GPU launches remain BLOCKED on user OK per
AGENTS.md §5.1 kill_then_adapt directive.

---

**END OF Wf-319B SHIP SUMMARY** — Owner Hugo Lee; cross-ref
WEAK_TO_STRONG_PLAN §2.7, Wf-315C analysis, AGENTS.md §5.2 honest
framing, party-twenty-spirit §3.5; last update 2026-09-29.
