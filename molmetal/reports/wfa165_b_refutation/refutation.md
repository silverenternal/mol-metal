# Wf-A165 branch B — refutation (scripts/start.sh YAML-derived GPU mask)

VERDICT: **NOT REFUTED** (core repair) — 1 BLOCKING surviving concern (S1), 3 minor (S2-S5).
0 GPU throughout; every call ran with `CUDA_VISIBLE_DEVICES=""`; single venv `.venv/bin/python`.

## Checks

| # | Check | Verdict | Evidence |
|---|---|---|---|
| C1 | Defect gone from the named lines | PASS | pre-repair `export CUDA_VISIBLE_DEVICES="$_cvd"` (HEAD:scripts/start.sh) -> now scripts/start.sh:721-742; capture at :66-73; exports only at :728/:738, both conditional |
| C1b | Claim 5 (only the yaml env file is sourced; no CUDA key in it) | PASS | scripts/start.sh:641 `. "$YAML_ENV_FILE"`; `lines=[...]` at :571-629 has no CUDA key; :182 pre-flight is a child; setup_molmetal_env.sh appears only in a comment at :57 |
| C2 | Regression test passes | PASS | `pytest molmetal/tests/test_wfa165_start_shell.py -q` -> **9 passed** (CUDA lock ON, 0 GPU tests run) |
| C2b | Negative control bites | PASS | rule flags pre-repair (`ambient='' -> '0'`, `ambient='1' -> '0'`) and the real file with the block reverted; current block: 0 violations |
| C3 | Proof is real, not transcribed | PASS | re-extracted the real blocks and re-ran all 6 cases myself: stdout byte-identical to proof.json in all 6 |
| C4 | Post-repair census, whole tracked tree | PASS (caveats S2/S3) | 89 findings / 2987 tracked *.py+*.sh read; M1 35, M2 0, M3 27, M4 27; 54 defeat an ambient empty mask; start.sh 0 (pre-repair: 1, M1 @HEAD:672) |
| C4b | Census coverage | FAIL (gap) | the guard enumerates TRACKED files only; 3 of this wave's own sites are untracked -> 89 tracked / 92 with them |
| C5 | Default path preserved | PASS per file / FAIL composed | start.sh cuda:0/cuda:3/auto: stdout AND mask identical; env.sh '0'->'0'; setup_molmetal_env.sh '0'->'0'; launch_cfm_fm_control (default 1) '1'->'1'; 5 more launchers unchanged. COMPOSED: launch_cfm_fm_control '0' -> '1' (S1) |
| C5b | start.sh called directly still lets the yaml decide | PASS | direct: '0'->'0' for all 4 yamls; nci60_9cell_probe (gpu: cuda:1) -> '1' via launch_all_training.sh |
| C6 | Scope discipline | PASS | branch B wrote only scripts/start.sh (11:30:09), its own test (11:27:32) and its own report dir; no forbidden path |

## Post-repair census (headline)

`CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa165_gpu_lock_guard.py --json --advisory --out ...`
- denominator: 2987 tracked *.py/*.sh enumerated = 2987 read (0 unreadable); 9550 tracked files total; 72 files with findings
- **89 findings**: M1_hard_override 35 (32 unmask), M2_empty_default_substitution 0, M3_env_dropping_spawn 27 (22 unmask, 5 unknown), M4_mask_dropping_default 27 (forces_blank, 0 unmask)
- 54 of 89 defeat an ambient empty mask; 5 unresolved env sites (not counted)
- per-site file:line list: `refutation.json` -> `census.sites` (89 entries)

## Surviving concerns

**S1 (BLOCKING, composed path)** — `scripts/launch_cfm_fm_control_2026_10_09.sh` + `configs/cfm_fm_control_2026_10_09.yaml`
(`compute.gpu: cuda:0`): composed mask PRE `'0'` -> POST `'1'`. The launcher always exports CVD, so
start.sh can no longer distinguish "operator asked for 1" from "launcher defaulted to 1", and the yaml's
device becomes dead. GPU 0 = RTX PRO 6000 Blackwell 98 GB, GPU 1 = RTX 5090 32 GB
(`molmetal/molmetal_lam/sbdd_env/training_utils.py:86-90`). Contradicts scripts/start.sh:705-708 and
AGENTS.md 5.11. The other three CVD-exporting launchers are unchanged.

**S2 (provenance)** — the branch's "2 -> 0" and "184" no longer reproduce with the shipped guard
(rewritten 11:38:51, after the branch's proof at 11:31:04): now 1 -> 0 and 89.

**S3 (coverage)** — 3 of the wave's own sites are in untracked files and outside the census:
`molmetal/tests/test_wfa165_gpu_lock_guard.py:161,312` (M4) and `molmetal/tests/test_wfa165_start_shell.py:147` (M3).
Quote the census with its denominator: 89 tracked-tree / 92 including them.

**S4 (latent)** — scripts/start.sh:728 re-exports the *current* `$CUDA_VISIBLE_DEVICES` while :729 prints the
*captured* `$_AMBIENT_CVD`; identical today, silently divergent after any future write between :73 and :721.

**S5 (advisory)** — the TUI's `--gpu $GPU` (scripts/start.sh:875) is display-only
(`scripts/_tui_dashboard.py:1042`) and is not remapped when the ambient mask wins, so in S1's composed path
the dashboard can name a different physical card than the run uses.
