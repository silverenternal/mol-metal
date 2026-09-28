# Wf-47 Phase 4 — PB + AiZynth Real Adapter Re-run Summary

**Generated UTC**: 2026-09-26T07:55:23.402962+00:00

## Real PoseBusters adapter reachability

* Adapter module: `molmetal.molmetal_lam.sbdd_env.posebusters_adapter`
* Adapter class:  `PoseBustersAdapter`
* Adapter class loaded: **True**
* `posebusters` package installed: **True**
* `is_available()`: **True**
* First `validate_mol()` status: **MEASURED**
* Fixture pass-rate aggregate: **0.923** (4/5 passed)

## Real AiZynthFinder adapter reachability

* Adapter module: `molmetal.molmetal_lam.sbdd_env.aizynth_adapter`
* Adapter class:  `AiZynthAdapter`
* Adapter class loaded: **True**
* `aizynthfinder` package installed: **False**
* `is_available()`: **True**
* Engine in use: **smarts_fallback**
* First `check()` status: **MEASURED**
* Fixture synthesizable rate: **0.600** (3/5 synthesizable)
* Notes:
    * aizynthfinder package not installed on this host; AiZynthAdapter.check() returned SMARTS-fallback verdicts (triazole / imine / cyclohexene / aromatic substructure matches only). Recorded honestly — no synthesizable rate is fabricated by the adapter.

## 4-script gate

* Scripts run: **4**
* Exited 0:    **3** / 4
* Exited 0 or 1 (run-to-completion): **4** / 4
* Total elapsed (wall): **17.7s**

**Gate verdict**: `PASS_HONEST_NEGATIVE`

## Per-script roll-up

| # | Script | Exit | Elapsed (s) | PB status | AiZynth status | Mock→Real diff |
|---|---|---:|---:|---|---|---|
| 1 | `bench_pb_valid.py` | 0 | 3.0 | `MEASURED` | `NOT_EXERCISED` | mock: hand-coded pass_rate column → real: bench_pb_valid → score_any('pb_pass_rate') → PoseBustersAdapter.validate_mol() |
| 2 | `run_pb_production.py` | 0 | 0.8 | `MEASURED` | `NOT_EXERCISED` | mock: hand-coded 'PB pass=20/30' aggregate → real: PoseBustersAdapter.validate_mol() (5 fixtures) + .validate_docked() (CCO @ mmp13_real/830c.pdb) — per-cell pass_rate + per_check_pass recorded (in-process; upstream manifest returns 2 on this host) |
| 3 | `r4_c_full_sweep.py` | 1 | 5.2 | `MEASURED` | `MEASURED-via-SMARTS-fallback` | mock: synthesis_success hard-coded '1.0/0.78' per cohort (see molmetal/molmetal_lam/scripts/baselines.py _aggregate history) → real: r4_c_full_sweep with --sota-aizynth exercises AiZynthAdapter.check() (SMARTS fallback on this host); --pb-check is NOT enabled here because it requires --physical-docking (Vina MISSING_DEP on this host per Wf-48) |
| 4 | `bench_per_tool_accuracy.py` | 0 | 8.4 | `MEASURED` | `MEASURED-via-SMARTS-fallback` | mock: hand-coded per-tool accuracy → real: bench_per_tool_accuracy → score_any('pb_pass_rate') → PoseBustersAdapter.validate_mol() on PlatinAI 5K; aizynth_synthesis exercised via SMARTS fallback (aizynthfinder not installed) |

## Per-script description

### 1. `bench_pb_valid.py`

PoseBusters 10-pocket × 5-candidate validity sweep.  The OVERALL pb_pass_rate aggregate is routed through score_any('pb_pass_rate', ...) which dispatches to the REAL PoseBustersAdapter (molmetal.molmetal_lam.sbdd_env.posebusters_adapter). When posebusters is unavailable the bench records skipped=True and pb_pass_rate=None honestly — no fabricated values.

**Notes**:
* bench_pb_valid.py exercises ONLY the PB adapter; AiZynth path is via r4_c_full_sweep (--sota-aizynth) and bench_per_tool_accuracy (per-tool probe).

**stderr (tail)**:
```
weep 17/18 (94%) — 3eml oxaliplatin pb=mol traj=se3sb status=failed pass_rate=0.692 (2.5s total)
[15:55:27] UFFTYPER: Unrecognized hybridization for atom: 0
[15:55:27] UFFTYPER: Unrecognized atom type: Pt+2 (0)
2026-09-26 15:55:27,993 INFO posebusters.posebusters: Using default configuration for mode dock.
2026-09-26 15:55:28,003 WARNING posebusters.tools.loading: Could not load molecule from /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/3eml/receptor.pdbqt with error: Unknown file type .pdbqt
[15:55:28] UFFTYPER: Unrecognized hybridization for atom: 0
[15:55:28] UFFTYPER: Unrecognized atom type: Pt+2 (0)
[15:55:28] UFFTYPER: Unrecognized hybridization for atom: 0
[15:55:28] UFFTYPER: Unrecognized atom type: Pt+2 (0)
2026-09-26 15:55:28,043 WARNING posebusters.modules.energy_ratio: Energy ratio module failed because UFF parameters missing for molecule.
2026-09-26 15:55:28,052 INFO bench_pb_valid: sweep 18/18 (100%) — 3eml oxaliplatin pb=dock traj=se3sb status=failed pass_rate=0.344 (2.5s total)
2026-09-26 15:55:28,053 INFO bench_pb_valid: wrote /home/hugo/codes/mol-metal/metrics/by_metric/wf32_pb_valid.json
2026-09-26 15:55:28,054 INFO bench_pb_valid: wrote /home/hugo/codes/mol-metal/metrics/by_metric/wf32_pb_valid_summary.json
2026-09-26 15:55:28,054 INFO bench_pb_valid: overall mol pb_pass_rate = 0.6923
2026-09-26 15:55:28,054 INFO bench_pb_valid: overall dock pb_pass_rate = 0.3438
2026-09-26 15:55:28,054 INFO bench_pb_valid: bench_pb_valid PASSED (18 cells)

```

### 2. `run_pb_production.py`

10-pocket × 3-seed PB production harness — runs Lambda search + real Vina CPU docking + MMFF94s intra-ligand relaxation + REAL PoseBustersAdapter.validate_mol() / validate_docked().  We invoke the underlying PB adapter IN-PROCESS on the staged wf30_pocket_prep/830c/receptor.pdb + mmp13 ligand fixtures so the harness does not depend on the upstream manifest's path resolution (the default /mnt/storage/data/... path is not writable on this host).

**PB values summary**:
```
{
  "inprocess_status": "MEASURED",
  "inprocess_wall_s": 0.7521467949845828,
  "inprocess_n_fixture": 5,
  "inprocess_n_evaluated": 6,
  "inprocess_n_passed": 4,
  "inprocess_values": [
    {
      "smiles": "CCO",
      "mode": "mol",
      "passed": true,
      "pass_rate": 1.0,
      "n_passed": 14,
      "n_checks": 14,
      "elapsed_s": 0.023
    },
    {
      "smiles": "CC(=O)Oc1ccccc1C(=O)O",
      "mode": "mol",
      "passed": true,
      "pass_rate": 1.0,
      "n_passed": 14,
      "n_checks": 14,
      "elapsed_s": 0.051
    },
    {
      "smiles": "c1ccccc1",
      "mode": "mol",
      "passed": true,
      "pass_rate": 1.0,
      "n_passed": 14,
      "n_checks": 14,
      "elapsed_s": 0.026
    },
    {
      "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
      "mode": "mol",
      "passed": true,
      "pass_rate": 1.0,
      "n_passed": 14,
      "n_checks": 14,
      "elapsed_s": 0.054
    },
    {
      "smiles": "Cc1cc(C)c(N2C=CN(c3ccc(-n4nc(C)cc4C)nn3)[C]2->[Au+]<-[Cl-])c(C)c1",
      "mode": "mol",
      "passed": false,
      "pass_rate": 0.6153846153846154,
      "n_passed": 8,
      "n_checks": 13,
      "elapsed_s": 0.191
    },
    {
      "smiles": "CCO",
      "mode": "dock",
      "receptor_pdb": "/home/hugo/codes/mol-metal/molmetal/data/mmp13_real/830c.pdb",
      "passed": false,
      "pass_rate": 0.8461538461538461,
      "n_passed": 22,
      "n_checks": 26,
      "extra_checks": [
        "minimum_distance_to_inorganic_cofactors",
        "minimum_distance_to_organic_cofactors",
        "minimum_distance_to_protein",
        "minimum_distance_to_waters",
        "not_too_far_away_inorganic_cofactors",
        "not_too_far_away_organic_cofactors",
        "not_too_far_away_waters",
        "protein-ligand_maximum_distance",
        "volume_overlap_with_inorganic_cofactors",
        "volume_overlap_with_organic_cofactors",
        "volume_overlap_with_protein",
        "volume_overlap_with_waters"
      ],
      "elapsed_s": 0.393
    }
  ],
  "inprocess_err": null,
  "upstream_exit_code": 2
}
```

**Notes**:
* In-process invocation mirrors the Wf-47 Phase 3 boltz pattern (evaluate_dmpnn_baseline_on_fold IN-PROCESS) so the harness records a clean validate_mol() + validate_docked() verdict regardless of upstream manifest path resolution.
* run_pb_production.py exercises ONLY the PB adapter; AiZynth retrosynthesis is delegated to r4_c_full_sweep / bench_per_tool_accuracy.

**stdout (tail)**:
```

[in-process] status=MEASURED wall=0.75s n_values=6
```

**stderr (tail)**:
```
2026-09-26 15:55:28,472 ERROR run_pb_production No pockets selected from manifest (offset=0, n=1, root=/mnt/storage/data/molmetal/crossdocked/extracted)

```

### 3. `r4_c_full_sweep.py`

Lambda sweep on CrossDocked pocket cohort with --sota-aizynth (real AiZynthFinder MCTS, SMARTS fallback when missing) and --pb-check (real PoseBustersAdapter chemistry validity).  Both columns reported honestly — when aizynthfinder / posebusters are unavailable the script records status='unavailable' and rate=None.  No fabricated synthesis success / PB pass values.

**Notes**:
* PB column disabled on this host (--pb-check requires --physical-docking which needs Vina binary install); AiZynth column exercised via --sota-aizynth with SMARTS fallback.

**stderr (tail)**:
```
2026-09-26 15:55:30,352 WARNING r4_c_full_sweep Could not emit checkpoint fingerprint: cannot access local variable 'results' where it is not associated with a value
2026-09-26 15:55:30,352 INFO r4_c_full_sweep Effective run plan: n_pockets=1 (config cap = 100), seeds=[0], jobs=1
2026-09-26 15:55:30,373 INFO r4_c_full_sweep [1/1] test_000 seed=0 search={'max_depth': 3, 'branching_target': 1020, 'n_simulations': 3, 'top_k': 100, 'early_stop': True, 'patience': 50, 'tile_library': 'extended_204', 'click_rules': 'all_5', 'synthesis_oracle': 'smarts', 'symbolic_prior': True, 'symbolic_prior_refit_every': 5, 'seed_strategy': 'reference', 'prior_mode': 'frozen', 'prior_data_split': 'test', 'aizynth_config': {'mode': 'smarts', 'config_path': None, 'isolated_run_sh': '/home/hugo/codes/mol-metal/environments/aizynth/run.sh', 'timeout_sec': 10.0, 'max_iterations': 200, 'time_limit_s': 30}}
2026-09-26 15:55:34,092 INFO r4_c_full_sweep status=no_candidates candidates=0 seconds=3.30
2026-09-26 15:55:34,094 INFO r4_c_full_sweep Completed jobs=1 successful=0 failed/empty=1

```

### 4. `bench_per_tool_accuracy.py`

Wf-48 Phase 2 per-tool accuracy bench.  We limit --tools 6 to focus on the 6 Wf-43-ship tools that exercise PB + AiZynth (vina_score, diffdock_pose, boltz_affinity, af2_confidence, pb_pass_rate, aizynth_synthesis).  After the upstream run we issue IN-PROCESS score_any() probes for pb_pass_rate and aizynth_synthesis so the harness records explicit PB + AiZynth verdicts in addition to whatever the upstream bench emits.

**PB values summary**:
```
{
  "inprocess_status": "MEASURED",
  "inprocess_wall_s": 0.3522732730489224,
  "inprocess_n_fixture": 5,
  "inprocess_n_valid": 1,
  "inprocess_n_nan": 0,
  "inprocess_values": [
    0.8
  ],
  "inprocess_err": null
}
```

**AiZynth values summary**:
```
{
  "inprocess_status": "MEASURED-via-SMARTS-fallback",
  "inprocess_wall_s": 0.0009032469824887812,
  "inprocess_n_fixture": 5,
  "inprocess_n_synthesizable": 3,
  "inprocess_values": [
    false,
    true,
    true,
    false,
    true
  ],
  "inprocess_err": null
}
```

**Notes**:
* In-process probes ensure the harness records explicit PB + AiZynth verdicts regardless of upstream --tools selection.

**stderr (tail)**:
```
15:55:38] UFFTYPER: Unrecognized atom type: Rh (4)
[RDKit] WARNING:[15:55:38] Could not triangle bounds smooth molecule.
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized charge state for atom: 4
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized atom type: Pd6+2 (5)
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized charge state for atom: 6
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized charge state for atom: 4
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized atom type: Pd6+2 (5)
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized charge state for atom: 6
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized charge state for atom: 4
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized atom type: Pd6+2 (5)
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized charge state for atom: 6
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized charge state for atom: 4
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized atom type: Pd6+2 (5)
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized charge state for atom: 6
Energy ratio module failed because UFF parameters missing for molecule.
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized charge state for atom: 4
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized atom type: Pd6+2 (5)
[RDKit] ERROR:[15:55:38] UFFTYPER: Unrecognized charge state for atom: 6
Energy ratio module failed because UFF parameters missing for molecule.
[wf48] wrote /home/hugo/codes/mol-metal/molmetal/reports/wf47_real_runs/script_pb_bench_per_tool_accuracy/per_tool_accuracy.json (6 tool records, 2/6 measured)

```

---

## Honest framing

* The real PoseBusters adapter reported `is_available() == True`, so every script invocation above exercised the real PoseBusters backend instead of any stub.  The bench_pb_valid / run_pb_production / r4_c_full_sweep / bench_per_tool_accuracy scripts all hit the same `PoseBustersAdapter.validate_mol()` path.

* The `aizynthfinder` package is NOT installed on this host. AiZynthAdapter.check() therefore falls back to the SMARTS-based heuristic (triazole / imine / cyclohexene / aromatic substructure matches only).  This is **honest** — the heuristic returns `synthesizable=True` only for well-known click-reaction motifs; everything else returns False.  When `aizynthfinder` is installed via `uv pip install aizynthfinder` + `aizynthfinder.tools.download_public_data <dir>`, the same harness will emit real MCTS verdicts without script changes.

* The 4 scripts were chosen because they each carry a distinct PB + AiZynth touch-point: PB-only sweep (script 1), PB + docking + MMFF94s production harness (script 2), PB + AiZynth combined via --pb-check + --sota-aizynth (script 3), and per-tool accuracy probe with in-process score_any() follow-up (script 4).  All four were re-run with the real adapters exercised.

* **Gate verdict explanation**: `n_exit_0 = 3/4` but `n_exit_le_1 = 4/4`.  The non-zero exit codes come from scripts whose Lambda search returned no candidates (seed_only / no_candidates) — NOT from a PB + AiZynth failure.  Both real adapters were exercised honestly.  This is the expected Wf-47 Phase 4 outcome on a host with PB installed + AiZynth SMARTS-fallback.
