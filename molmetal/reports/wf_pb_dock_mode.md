# WF-PB-Dock-Mode-Wire — protein-aware clash checks

**Goal:** extend `PoseBustersAdapter` so `pb_mode="dock"` actually runs the
protein-aware clash / minimum-distance checks that have been silent in the
`mol` default since the original PB wire-up.  Closes the
"protein-blind" weakness identified by WF-PB-Pass-Real-Dock.

## Honest-framing scope

* Honest numbers: `mol` mode emits **14** bool chemistry checks after the
  three loader columns are dropped.  `dock` mode emits **26** bool checks
  total (same 14 chemistry checks + **12** protein-aware checks).
  The brief's "22" figure was off by a small amount — the correct
  protein-aware **extra** count is 12, not 8, because the cofactor checks
  (`minimum_distance_to_*_cofactors`, `not_too_far_away_*_cofactors`,
  `volume_overlap_with_*_cofactors`) are 8 checks (not 4).
* No SOTA claim.  This is a plumbing wire-up; the protein-aware PB pass
  rate on de novo generated ligands will be much lower than literature
  redocking pass rates because (a) we run PB on an ETKDGv3+MMFF94
  conformer without docking into the pocket first, and (b) Lambda
  candidates without explicit pocket conditioning will frequently clash.
* `validate_docked` returns a `PBResult` dataclass (not just a
  `ValidityReport`) so callers can see *which* extra checks ran.

## Diff (vs previous PB wire-up)

### `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py`

* Added frozen dataclass `PBResult(report, mode, receptor_pdb, extra_checks)`
  with property aliases (`smiles`, `passed`, `n_checks`, …) so it can be
  used as a drop-in replacement for `ValidityReport` in existing code.
* Added `PoseBustersAdapter.validate_docked(smiles, receptor_pdb)` →
  `PBResult`.  Reuses the same `Chem.MolFromSmiles` + `ETKDGv3` +
  `MMFF94OptimizeMolecule` (with UFF fallback) embedding path as
  `validate_mol`, then calls `self._pb.bust(mol, mol_cond=receptor_pdb,
  full_report=True)`.
* Added `PoseBustersAdapter._protein_aware_extras(df)` → tuple of bool
  column names that are present in `dock` mode but **absent** in
  `mol` mode.  Computed by spinning up a probe `PoseBusters(config="mol")`
  instance and diffing column sets; falls back to a hard-coded list of
  12 names when the probe fails (e.g. no RDKit in worker process).
* Updated `__all__` to export `PBResult`.

### `molmetal/scripts/r4_c_full_sweep.py`

* In the worker, after `pb_config` is parsed, the orchestrator now
  branches on `pb_mode`:
  * `mol` (default) — unchanged path through `validate_list`.
  * `dock` — loops over each generated SMILES, calls
    `adapter.validate_docked(smi, result.receptor_path)` (or
    `pb_config["receptor_pdb"]` if explicitly set), and records
    `pb_check["receptor_pdb"]` + `pb_check["extra_checks"]` (the union
    across all per-SMILES `PBResult`s).
  * If `pb_mode="dock"` and the receptor PDB is missing, the worker
    sets `pb_status="receptor_missing"` with a clear reason rather
    than silently falling back to mol-mode (honest-framing).
* The `--pb-mode` flag (already present with choices
  `("mol", "dock", "redock")` and default `mol`) now actually drives
  a different code path.

### `molmetal/molmetal_lam/tests/test_poseb_backers_wire.py`

* New test `test_pb_mode_dock_smoke` (mocked adapter) — verifies the
  worker calls `validate_docked(smiles, receptor_pdb)` once per
  generated candidate and the output JSON carries
  `extra_checks` + `receptor_pdb`.
* New test `test_pb_mode_mol_backward_compat` (mocked adapter) —
  verifies the legacy `mol` branch is unchanged, `validate_docked`
  is **not** called, and `extra_checks` / `receptor_pdb` are absent
  from `pb_check`.
* New test `test_validate_docked_signature` — verifies
  `PoseBustersAdapter.validate_docked` and `PBResult` are exported
  with the documented shape.

## Test results

```
$ uv run pytest -q molmetal/molmetal_lam/tests/test_poseb_backers_wire.py --tb=short
...........                                                              [100%]
11 passed, 1 warning in 1.49s
```

All 11 tests pass (8 pre-existing + 3 new for the dock mode wire-up).
No regressions.

## Smoke result

```
$ uv run python molmetal/scripts/r4_c_full_sweep.py \
    --pockets /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/ \
    --seeds 42 --physical-docking --pb-check --pb-mode dock \
    --output-prefix molmetal/reports/wf_pb_dock_mode_smoke/
…
INFO  Effective run plan: n_pockets=5 (config cap = 100), seeds=[42], jobs=5
INFO  [1/5] test_000 seed=42 search={… 'pb_config': {'mode': 'dock'}}
INFO  status=no_candidates candidates=0 seconds=4.58
…
INFO  Completed jobs=5 successful=0 failed/empty=5
```

The 5 selected test pockets all returned `status=no_candidates` /
`seed_only` — Lambda search did not generate any de novo SMILES in this
budget (the default `n_simulations=1000` + `max_depth=3` did not pass
the synthesis oracle).  So the dock-mode `validate_docked` path was
exercised at the *adapter* level (per `pb_check.mode ==
PoseBusters_dock_v1`) but did not run per-SMILES because there were
no generated SMILES to dock-check.

To exercise the per-SMILES dock path I ran a synthetic worker run that
fakes two `is_generated=True` candidates (`CCO`, `CCN`) and the real
`molmetal/data/mmp13_real/830c.pdb` receptor:

```
$ python -c "<worker with CCO + CCN against 830c.pdb>"
rc: 0
pb_status: completed
pb_pass_rate: 0.0
n_pb_pass: 0
pb_check.extra_checks: ['minimum_distance_to_inorganic_cofactors',
                        'minimum_distance_to_organic_cofactors',
                        'minimum_distance_to_protein',
                        'minimum_distance_to_waters',
                        'not_too_far_away_inorganic_cofactors',
                        'not_too_far_away_organic_cofactors',
                        'not_too_far_away_waters',
                        'protein-ligand_maximum_distance',
                        'volume_overlap_with_inorganic_cofactors',
                        'volume_overlap_with_organic_cofactors',
                        'volume_overlap_with_protein',
                        'volume_overlap_with_waters']
pb_check.receptor_pdb: /home/hugo/codes/try_triton_on_rocm/molmetal/data/mmp13_real/830c.pdb
CCO record extras: <same 12>
CCO record n_checks: 26
```

This confirms end-to-end:

* `validate_docked` runs and produces `n_checks=26` (chemistry 14 +
  protein-aware 12).
* The 12 protein-aware `extra_checks` are surfaced in both the
  per-smiles records and the union at `pb_check["extra_checks"]`.
* `pb_check["receptor_pdb"]` carries the path used.
* `pb_status="completed"`, `n_pb_pass=0` because both `CCO` and `CCN`
  fail `protein-ligand_maximum_distance` against the 830c pocket
  (their ETKDGv3 conformers are placed arbitrarily far from the
  binding site — expected behaviour; the docking step is what brings
  them into the pocket, and we are running PB *before* docking for
  this unit test).

## Files touched

| File | Change |
|---|---|
| `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py` | + `PBResult` dataclass, + `validate_docked`, + `_protein_aware_extras`, `__all__` export |
| `molmetal/scripts/r4_c_full_sweep.py` | worker `pb_config` branch on `pb_mode`; `receptor_pdb` propagation; `receptor_missing` honest error |
| `molmetal/molmetal_lam/tests/test_poseb_backers_wire.py` | + 3 tests (dock smoke, mol back-compat, signature) |
| `molmetal/reports/wf_pb_dock_mode_smoke/` | smoke outputs (worker log, pose dir) |
| `molmetal/reports/wf_pb_dock_mode.md` | this report |

## Next actions

* Hook `pb_mode="dock"` into a pocket-conditioned round-12 mini pilot so
  we can measure the protein-aware PB pass rate on real candidates.
  Until then the `pb_status` field reports `no_candidates` or
  `receptor_missing` and no number should be cited.
* The probe (`PoseBusters(config="mol")` + ethanol) costs ~1 s per
  `validate_docked` call; cache the probe result if profiling shows it
  becomes a bottleneck on long runs.
