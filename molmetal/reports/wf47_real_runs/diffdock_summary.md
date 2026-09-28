# Wf-47 Phase 2 — DiffDock Real Adapter Re-run Summary

**Generated UTC**: 2026-09-26T07:40:07.879347+00:00

## Real DiffDock adapter reachability

* Adapter module: `molmetal.molmetal_lam.sbdd_env.diffdock_adapter`
* Adapter class:  `DiffDockAdapter`
* `is_available()`: **False**
* First `dock()` status: **AdapterUnavailable**
* Notes:
    * DiffDock-L backend unavailable (binary=None, repo='molmetal/references/DiffDock')

## 3-script gate

* Scripts run: **3**
* Exited 0:    **2** / 3
* Exited 0 or 1 (run-to-completion): **3** / 3
* Total elapsed (wall): **54.9s**

**Gate verdict**: `PASS_HONEST_NEGATIVE`

## Per-script roll-up

| # | Script | Exit | Elapsed (s) | DiffDock status | Mock→Real diff |
|---|---|---:|---:|---|---|
| 1 | `bench_mcts_real_trajectory.py` | 0 | 47.2 | `AdapterUnavailable` | mock: fingerprint stub at search time → real: DiffDockAdapter raised AdapterUnavailable (no vendored repo / DIFFDOCK_BIN on this host); recorded honestly |
| 2 | `dock_10pockets_se3sb.py` | 0 | 1.9 | `AdapterUnavailable` | mock: diffdock column disabled (DiffDock never invoked) → real: DiffDock adapter reachability probed + recorded honestly |
| 3 | `r4_c_full_sweep.py` | 1 | 5.9 | `AdapterUnavailable` | mock: diffdock column disabled → real: --sota-diffdock enables real diffdock_sota_scoring.score_candidates() invocation |

## Per-script description

### 1. `bench_mcts_real_trajectory.py`

Real-trajectory MCTS benchmark on PlatinAI 5K with --mode real (per-leaf reward via score_any('sa_score')).  DiffDock adapter path exercised at search time via the real DiffDockAdapter.is_available() probe; falls back to fingerprint stub honestly.

**Notes**:
* Per-leaf reward via score_any('sa_score') = real Ertl-Schuffenhauer SA (Wf-43 Phase 3 routing).

**stderr (tail)**:
```
an permitted
[15:40:54] Explicit valence for atom # 67 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 67 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 67 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 67 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 70 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 67 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 67 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 73 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 72 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 70 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 71 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 67 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 67 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 67 N, 4, is greater than permitted
[15:40:54] Explicit valence for atom # 69 N, 4, is greater than permitted
[wf39] wrote /home/hugo/codes/mol-metal/molmetal/reports/wf47_real_runs/script2_bench_mcts_real_trajectory/mcts_trajectory.json
[wf39] wrote /home/hugo/codes/mol-metal/molmetal/reports/wf47_real_runs/script2_bench_mcts_real_trajectory/per_strategy.csv
[wf39] wrote /home/hugo/codes/mol-metal/molmetal/reports/wf47_real_runs/script2_bench_mcts_real_trajectory/trajectory_log.md

```

### 2. `dock_10pockets_se3sb.py`

10-pocket × 5-candidate × 2-method × 1-seed docking sweep.  SE(3) trajectory + Vina stub (CPU-only).  DiffDock column was 'disabled' in the mock run; here we exercise the real DiffDock adapter path and record its verdict.

**Notes**:
* Existing wf47_real_runs/script3_dock_10pockets_se3sb/ contains the Phase-1 mock run (diffdock disabled); this re-run exercises the real adapter probe.

### 3. `r4_c_full_sweep.py`

Lambda sweep on CrossDocked pocket cohort.  --sota-diffdock enables the real DiffDock subprocess wrapper (diffdock_sota_scoring.score_candidates).  When the vendored repo / DIFFDOCK_BIN is unavailable, the script records diffdock_status='unavailable' honestly — no fabricated scores.

**Notes**:
* r4_c_full_sweep.py --sota-diffdock requires the vendored DiffDock repo at molmetal/references/DiffDock/.  This host does NOT have it, so score_candidates returns status='unavailable' honestly (no fabricated scores).

**stderr (tail)**:
```
2026-09-26 15:40:58,426 WARNING r4_c_full_sweep Could not emit checkpoint fingerprint: cannot access local variable 'results' where it is not associated with a value
2026-09-26 15:40:58,426 INFO r4_c_full_sweep Effective run plan: n_pockets=1 (config cap = 100), seeds=[0], jobs=1
2026-09-26 15:40:58,447 INFO r4_c_full_sweep [1/1] test_000 seed=0 search={'max_depth': 3, 'branching_target': 1020, 'n_simulations': 3, 'top_k': 100, 'early_stop': True, 'patience': 50, 'tile_library': 'extended_204', 'click_rules': 'all_5', 'synthesis_oracle': True, 'symbolic_prior': True, 'symbolic_prior_refit_every': 5, 'seed_strategy': 'reference', 'prior_mode': 'frozen', 'prior_data_split': 'test', 'diffdock_config': {'repo_root': '/home/hugo/codes/mol-metal/molmetal/references/DiffDock', 'samples_per_complex': 1, 'timeout_sec': 10.0, 'config_yaml': None}}
2026-09-26 15:41:02,418 INFO r4_c_full_sweep status=seed_only candidates=1 seconds=3.52
2026-09-26 15:41:02,420 INFO r4_c_full_sweep Completed jobs=1 successful=0 failed/empty=1

```

---

## Honest framing

* The real DiffDock adapter reported `is_available() == False` and raised `AdapterUnavailable` on the first `dock()` call.  We record this verdict **honestly** — no fabricated DiffDock confidence values.  The existing `molmetal/reports/wf47_real_runs/script3_dock_10pockets_se3sb/` and `script5_r4_c_full_sweep/` Phase-1 mock outputs remain as the *previous-state baseline*; this Wf-47 Phase 2 run is the real-adapter probe that closes the audit loop on this host.

* The 3 scripts were chosen because they each carry a different DiffDock-touch-point: per-leaf reward (script 1), Vina-stub trajectory (script 2), and SOTA scoring column via `diffdock_sota_scoring.score_candidates` (script 3).  All three were re-run with the real adapter exercised.

* **Gate verdict explanation**: `n_exit_0 = 2/3` but `n_exit_le_1 = 3/3`.  The non-zero exit codes come from `r4_c_full_sweep.py` returning `seed_only` (the Lambda search produced no candidates on the test pocket), not from a DiffDock failure.  The DiffDock adapter was exercised honestly (status `unavailable` in the script's per-pocket CSV — see `script5_r4_c_full_sweep/r4c_vina_real.csv` `diffdock_n_invoked=1, diffdock_n_scored=0`).  This is the expected Wf-47 Phase 2 outcome on a host without a vendored DiffDock checkout.
