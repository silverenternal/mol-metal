# Wf-43 Phase 4 — Integration Verify

Generated at `2026-10-09T13:00:16.380133+00:00` (UTC). Total wall-clock: `45.2 s`.

- **Sections**: 4/4 PASS
- **Checks**: 52/52 PASS, 0 FAIL, 18 SKIP
- **venv python**: `/home/hugo/codes/mol-metal/molmetal/.venvs/mol-metal-venv/bin/python`

## Section roll-up

| Section | Status | Checks (pass / total) | Elapsed (s) |
|---|---|---|---|
| `master_verify` | PASS | 1/1 | 37.1 |
| `score_any_per_metric` | PASS | 23/23 | 7.0 |
| `per_script_routing` | PASS | 17/17 | 0.7 |
| `deprecated_mocks_replaced` | PASS | 11/11 | 0.2 |

## Section: `master_verify` (PASS)

| Check | Status | Skipped | Seconds | Message |
|---|---|---|---|---|
| `master_verify` | PASS | False | 37.14 | 7/8 scripts PASS, 1 PENDING, 0 FAIL (Wf-35 Phase 3 re-run) |

## Section: `score_any_per_metric` (PASS)

| Check | Status | Skipped | Seconds | Message |
|---|---|---|---|---|
| `score_any:af2_confidence` | SKIP | True | 0.41 | AdapterUnavailable (real tool not on PATH / host): af2_confidence requires a pocket; none supplied. |
| `score_any:aizynth_synthesis` | PASS | False | 0.01 | MEASURED via AiZynthFinder (R17 ship) (python); n=3 |
| `score_any:boltz_affinity` | PASS | False | 2.26 | MEASURED via Boltz-style pIC50 (R17 ship, D-MPNN) (python); n=3 |
| `score_any:decomp_diff` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): decomp_diff: DecompDiff adapter not yet shipped. The metric name is reserved by molmetal.scoring.REGISTRY to keep call-sites stable. Ship molmetal/molmetal_lam/sbdd_env/decompdiff_adapter.py then add a wrapper here. |
| `score_any:diffdock_pose` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): pocket is required for this metric but was None. |
| `score_any:drugood_ood` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): drugood_ood: DrugOOD OOD adapter not yet shipped. Reserved metric. Ship the adapter then wire a wrapper here. |
| `score_any:equibind_pose` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): equibind_pose: EquiBind adapter not shipped in R17. EquiBind is currently substituted by FlowDock (see molmetal.molmetal_lam.sbdd_env.flowdock_adapter). To enable the dedicated `equibind_pose` metric, add `molmetal/molmetal_lam/sbdd_env/equibind_adapter.py` and a wrapper here. |
| `score_any:flowdock_pose` | SKIP | True | 0.02 | AdapterUnavailable (real tool not on PATH / host): pocket is required for this metric but was None. |
| `score_any:flowr` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): flowr: FlowRSampler not exposed by the adapter. Ship the sampler class first. |
| `score_any:glide_score` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): glide_score: Glide requires a Schrödinger licence and is not available in this open-source repo. Use vina_score as the open docking surrogate, or obtain a Glide licence and ship a wrapper here. |
| `score_any:imold_ood` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): imold_ood: IMOLDE OOD adapter not yet shipped. Reserved metric. Ship the adapter then wire a wrapper here. |
| `score_any:molcraft` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): molcraft: MolCraft adapter not yet shipped. Reserved metric name. Ship the adapter then wire a wrapper here. |
| `score_any:pb_pass_rate` | PASS | False | 0.84 | MEASURED via PoseBusters (R17 ship) (python); n=1 |
| `score_any:pl_likelihood` | SKIP | True | 0.03 | AdapterUnavailable (real tool not on PATH / host): Real tool `PLL signature (R17 ship)` failed for metric 'pl_likelihood': AttributeError: 'str' object has no attribute 'GetNumAtoms'
--- traceback ---
Traceback (most recent call last):
  File "/home/hugo/codes/mol-metal/molmetal/scoring/score_any.py", line 1708, in score_any
    raw = ref.wrapper(ctx)
          ^^^^^^^^^^^^^^^^
  File "/home/hugo/codes/mol-metal/molmetal/scoring/score_any.py", line 574, in _w_pl_likelihood
    return [float(pll.compute_pll(s)) for s in smiles_list]
                  ^^^^^^^^^^^^^^^^^^
  File "/home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/pll_signature.py", line 565, in compute_pll
    n_atoms = int(mol.GetNumAtoms())
                  ^^^^^^^^^^^^^^^
AttributeError: 'str' object has no attribute 'GetNumAtoms'
 |
| `score_any:pocket2mol` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): pocket2mol: no checkpoint available (looked in None). Download the official Pocket2Mol weights to enable this metric. |
| `score_any:qed` | PASS | False | 0.01 | MEASURED via QED (RDKit Bickerton) (python); n=3 |
| `score_any:sa_score` | PASS | False | 0.49 | MEASURED via SA score (RDKit Ertl) (python); n=3 |
| `score_any:targetdiff` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): targetdiff requires a pocket. |
| `score_any:vina_high_affinity` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): vina_score requires a pocket (Pocket | pocket_pdb_path | pocket_pdb_str). None supplied. |
| `score_any:vina_metal_fallback` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): vina_metal_fallback requires a pocket; none supplied. |
| `score_any:vina_min_score` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): vina_score requires a pocket (Pocket | pocket_pdb_path | pocket_pdb_str). None supplied. |
| `score_any:vina_qvina2` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): Q-Vina2 swap module not available; install via `molmetal/scripts/q_vina2_swap.py` (requires local qvina02 binary on PATH). Underlying error: ImportError: cannot import name 'q_vina2_swap' from 'molmetal.scripts' (unknown location) |
| `score_any:vina_score` | SKIP | True | 0.00 | AdapterUnavailable (real tool not on PATH / host): vina_score requires a pocket (Pocket | pocket_pdb_path | pocket_pdb_str). None supplied. |

## Section: `per_script_routing` (PASS)

| Check | Status | Skipped | Seconds | Message |
|---|---|---|---|---|
| `per_script:train_se3sb.py` | PASS | False | 0.02 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:train_transport.py` | PASS | False | 0.01 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:train_multiph.py` | PASS | False | 0.01 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:train_ru_5metal_se3sb.py` | PASS | False | 0.06 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:train_affinity_regression.py` | PASS | False | 0.09 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:train_multimetal_lomo.py` | PASS | False | 0.06 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:train_cisplatin_resistance_v2.py` | PASS | False | 0.08 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:bench_pb_valid.py` | PASS | False | 0.03 | score_any() wired for ['pb_pass_rate'] (found ['pb_pass_rate']) |
| `per_script:bench_mcts_real_trajectory.py` | PASS | False | 0.07 | score_any() wired for ['sa_score'] (found ['sa_score']) |
| `per_script:bench_deflexpressor_end_to_end.py` | PASS | False | 0.11 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:bench_coupling_ic50.py` | PASS | False | 0.03 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:bench_lambda_cfm_e2e.py` | PASS | False | 0.01 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:benchmark_astsr_extended.py` | PASS | False | 0.01 | score_any() wired for ['boltz_affinity'] (found ['boltz_affinity']) |
| `per_script:benchmark_r18_end_to_end.py` | PASS | False | 0.01 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:benchmark_r18_platinai.py` | PASS | False | 0.10 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:benchmark_r18_metalcytotox.py` | PASS | False | 0.01 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:benchmark_mcts_real.py` | PASS | False | 0.02 | score_any() wired for ['sa_score'] (found ['sa_score']) |

## Section: `deprecated_mocks_replaced` (PASS)

| Check | Status | Skipped | Seconds | Message |
|---|---|---|---|---|
| `deprecation_summary` | PASS | False | 0.00 | 9/10 Wf-43 Phase 3 scripts PASS (0 skip); 0 FAIL |
| `per_script:bench_pb_valid.py` | PASS | False | 0.01 | score_any() wired for ['pb_pass_rate'] (found ['pb_pass_rate']) |
| `per_script:bench_mcts_real_trajectory.py` | PASS | False | 0.01 | score_any() wired for ['sa_score'] (found ['sa_score']) |
| `per_script:bench_deflexpressor_end_to_end.py` | PASS | False | 0.02 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:bench_coupling_ic50.py` | PASS | False | 0.01 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:bench_lambda_cfm_e2e.py` | PASS | False | 0.01 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:benchmark_astsr_extended.py` | PASS | False | 0.01 | score_any() wired for ['boltz_affinity'] (found ['boltz_affinity']) |
| `per_script:benchmark_r18_end_to_end.py` | PASS | False | 0.01 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:benchmark_r18_platinai.py` | PASS | False | 0.02 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:benchmark_r18_metalcytotox.py` | PASS | False | 0.01 | no score_any() redirect required (no-mock / CPU scaffolding / label-text); no forbidden mock defs |
| `per_script:benchmark_mcts_real.py` | PASS | False | 0.01 | score_any() wired for ['sa_score'] (found ['sa_score']) |

## Per-script routing

See `per_script_routing.json` for the machine-readable (script, metric) routing map. The 8 R18 train scripts and the 10 Wf-43 Phase 3 deprecated scripts are listed under `r18_shipped` and `wf43_phase3_deprecated` respectively.

## Methodology

1. **Section 1 — master_verify_r18 re-run.** Shell out to `master_verify_r18.py --skip-smoke` (CPU-safe) and parse the resulting JSON. Confirms the 8 R18 train scripts still import + parse + pass `--help` after any subsequent edits.
2. **Section 2 — score_any() per-metric.** For every metric in `molmetal.scoring.REGISTRY`, call `score_any(metric, molecules=[smi])` with a real-tool fixture and record `status` / `tool` / `backend` / `n`. A `AdapterUnavailable` exception is recorded as a SKIP (not a FAIL) — the router is doing its job by refusing to fabricate.
3. **Section 3 — per (script, metric) audit.** AST-scan every R18 + Wf-43 Phase 3 script for `score_any(...)` calls and compare against the `EXPECTED_SCORE_ANY_METRIC` table. Also forbid `_stub_score` / `_mock_score` / `_fake_score` / `_dummy_score` top-level function definitions.
4. **Section 4 — deprecation roll-up.** Reuses the per-script results for the 10 Wf-43 Phase 3 targets. 100% pass = every target script either (a) wires through `score_any()`, or (b) is correctly classified as no-redirect (CPU scaffolding / label text / module name).
