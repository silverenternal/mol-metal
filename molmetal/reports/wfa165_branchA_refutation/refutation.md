# Wf-A165 — Branch A refutation (adversarial review)

**VERDICT: NOT REFUTED** — every claim branch A made was reproduced from the real
files on disk. 0 GPU used (`CUDA_VISIBLE_DEVICES=""` on every call, bash variable
expansion only, torch never imported). Single venv
`/home/hugo/codes/mol-metal/.venv/bin/python`.

## Checks

| # | Check | Verdict | Evidence |
|---|---|---|---|
| C1 | Denominator: 113 tracked `*.sh`, 27 with an export line, 23 defeaters at HEAD | PASS | `git ls-files -- '*.sh'` = 113; grep = 27 files / 28 lines (28 because branch B gave `scripts/start.sh` a second export). HEAD enumerated by hand from `git show HEAD:<f>`: 24 defeaters = 23 (branch A) + 1 (`scripts/start.sh`, branch B) + 3 blank-mask exporters left alone. Branch's breakdown reproduced exactly: 18x `=0`, 1x `=1` (`scripts/launch_cfm_fm_control_2026_10_09.sh:37`), 1x `${CUDA_VISIBLE_DEVICES:-0}` (`molmetal/scripts/setup_molmetal_env.sh:52`), 2x `${GPU}`, 1x `${PA17_GPU:-0}` |
| C2 | Repair real + default preserved | PASS | Independent re-extraction of the real line from each of the 23 files, evaluated in `bash -c` against the same prelude, compared with the HEAD line: unset-ambient old == new on all 23 (`env.sh` [0]==[0]; `launch_cfm_fm_control_2026_10_09.sh` [1]==[1]; `launch_dmpnn_clean5v4_loso_d1d3.sh` [1]==[1]; `launch_cfm_round7_f5f6_retrain.sh` [0]==[0]). 0 default drifts, 0 lock breaks (`''`->`[]`), 0 choice swaps (`'1'`->`[1]`), 0 no-ops |
| C3 | Regression test really bites | PASS | `pytest molmetal/tests/test_wfa165_shell_launchers.py -q` -> 7 passed. NEGATIVE CONTROL re-run in an isolated sandbox (`/tmp/a165_negctl`, repo untouched): mutation A `export CUDA_VISIBLE_DEVICES=0` back into `env.sh` -> 3 failed / 4 passed (the branch's "3 of the 7" reproduced); mutation B `:-` form into `setup_molmetal_env.sh` -> same 3 failed; both reverted, baseline back to 7 passed |
| C4 | Proof is real, not transcribed | PASS | My own extractor reproduced 23/23 proof cases. `proof.json` census vs disk: 113 scanned, 90 not-applicable (all with a non-empty `reason`), 23 proved, 23 repaired, `defeater_sites_unrepaired: []`, 4 left alone, `files_any_fail: 0`, every recorded `to` string matches the current disk line |
| C5 | Post-repair census over the WHOLE tree | PASS (reported, **not clean**) | `wfa165_gpu_lock_guard.py --advisory` over 9550 tracked files (2987 enumerated = 2874 `*.py` + 113 `*.sh`): **89 findings, 54 defeat the ambient empty mask**. M1=35 (32 unmask + 3 forces_blank), M2=0, M3=27 (22 unmask), M4=27 (all forces_blank). My re-run matches the guard's own `census.json` exactly. HEAD baseline measured in an isolated sandbox of HEAD's 113 `*.sh`: M1=58 (55 unmask) + M2=1 = **56 unmask -> now 32**; M1 site-list diff = 23 removals (22 branch A + 1 branch B), 0 additions |
| C6 | Nothing broke — default path | PASS | `env.sh:61` unset->[0]; `launch_cfm_fm_control_2026_10_09.sh:38` (default 1) unset->[1]; `setup_molmetal_env.sh:53` unset->[0]; `launch_cfm_round7_f5f6_retrain.sh:118` unset->[0]; `launch_dmpnn_clean5v4_loso_d1d3.sh:149` unset->[1] — each identical to HEAD. `set -uo pipefail` harnesses rc=0. `git diff --numstat` = 2/1 on each of the 23; no mode change; `bash -n` clean on all 23. Hardcoded `cuda:N` in the repaired launchers are VISIBLE indices, so they stay coherent. Guard + `start.sh` tests: 24 passed |
| C7 | Scope discipline | PASS | mtime census of all 9550 tracked files: branch A's edits are exactly the 23 shell launchers (22 at 11:27:15 + `env.sh` at 11:29:34) plus its own `proof.json`, `wfa165_shell_launcher_repair.py`, `test_wfa165_shell_launchers.py`. Zero writes to `paper/**`, `docs/**`, `AGENTS.md`, `README.md`, `TODO/**`, `configs/**`, `metrics/**`, `registry/**`, `run_ledger.jsonl`, `molmetal_lam/**`, `pre_flight_check.py`, `promote_metrics.py`, `wf_a155_ligand_connected_rate.py` |

## Post-repair census (the wave headline)

* denominator: 9550 tracked files; 2987 enumerated (2874 `*.py` + 113 `*.sh`); 0 unreadable
* **89 findings total, 54 of which defeat an ambient `CUDA_VISIBLE_DEVICES=""` lock**
* M1 hard override 35 (32 unmask + 3 forces_blank) · M2 empty-default substitution **0** · M3 env-dropping spawn 27 (22 unmask) · M4 mask-dropping default 27 (all forces_blank)
* HEAD -> now: shell-mechanism unmask sites **56 -> 32**; M1 **58 -> 35**
* all 32 surviving unmask sites are the **command-prefix** form

## Surviving concerns

1. **HIGH (for the wave, not a false claim by branch A)** — the lock still does not hold
   repo-wide. 32 of 35 M1 findings still defeat it and **all 32 are the command-prefix
   form** (`CUDA_VISIBLE_DEVICES=0 cmd`) that branch A declined to repair, calling it
   "a 4th mechanism outside this branch's mechanism list". The wave's own guard places
   the command-prefix form *inside* M1 ("export / plain assignment / per-command prefix
   form"), so that carve-out is a scoping decision, not a property of the tool. 20 of
   the 32 are LIVE `scripts/` launchers: `scripts/launch_wf124_p3_g5.sh:380`,
   `scripts/launch_wf124_p3_g6.sh:388`, `scripts/launch_wf125_p1_controlled_pair.sh:376`,
   `scripts/launch_wf131_p3_deferred_runs.sh:304`, `scripts/launch_wf131_p3_pt_4cells.sh:523`,
   `scripts/launch_wf146_deferred_runs.sh:437`, `scripts/launch_wf299_cfm_main.sh:531`,
   `scripts/launch_wf308c_todo220_gpu.sh:178` and `:204`,
   `scripts/launch_wf310b_g6_final.sh:248`,
   `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh:429`,
   `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh:448`,
   `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:494`,
   `scripts/launch_wf310e_wf135_phase2_gpu.sh:235`,
   `scripts/launch_wf310e_wf219_production_gpu.sh:236`,
   `scripts/launch_wf310f_frontier_math_gpu.sh:248`,
   `scripts/launch_wf390a1_m2_r2_path_a.sh:394`,
   `scripts/launch_wf394a1_lambda_cfm_patch.sh:383`,
   `scripts/recover_g6_failed_tuples.sh:386`, `scripts/retry_g5_missing_pairs.sh:68`;
   11 in `TODO/archive/` (read-only), 1 in `results/cfm_full_retrain/launch_loop.sh:10`.
   Branch A disclosed this honestly with a durable artefact rather than hiding it —
   but the wave must not claim the GPU lock holds.
2. **MEDIUM** — the rule is line-level and cannot see a DEFAULT flip: `${CUDA_VISIBLE_DEVICES-1}`
   and `${CUDA_VISIBLE_DEVICES-0}` are both "safe", so a silent default change would pass all
   7 tests. No flip exists today (proved independently for all 23), but the test does not
   protect the property the branch cares most about (`molmetal/tests/test_wfa165_shell_launchers.py:147,276`
   never evaluate the unset-ambient case).
3. **MEDIUM** — `env.sh` and `molmetal/scripts/setup_molmetal_env.sh` are sourced by CI
   (`.github/workflows/ci.yml:118`) and by the Makefile flow; an ambient mask now propagates
   where HEAD forced 0. The unset path is byte-identical, so no CI behaviour changes today;
   `setup_molmetal_env.sh:22` still advertises the old idiom in its header comment.
4. **LOW** — the single exemption (`scripts/start.sh`, branch B) is line-level and valid only
   while branch B's ambient-presence capture survives; the branch said so and the test asserts
   it, so the exemption is guarded, not blind.
5. **LOW (out of branch A's remit)** — M3 contributes 22 of the 54 remaining lock-defeating
   sites; all are Python (`molmetal/molmetal_lam/sbdd_env/training_utils.py:136,217`,
   `molmetal/scripts/_common_train_args.py:585`, ...).

## Artefacts

* `molmetal/reports/wfa165_branchA_refutation/refutation.json`
* `molmetal/reports/wfa165_branchA_refutation/refutation.md` (this file)
* `molmetal/reports/wfa165_branchA_refutation/post_repair_census.json` (my guard re-run)
* `molmetal/reports/wfa165_branchA_refutation/independent_behaviour.json` (my own extraction)

Sandboxes (repo working tree never modified): `/tmp/a165_negctl` (mutation test),
`/tmp/a165_head_tree` (HEAD baseline census).
