# Wf-A165 — SYNTHESIS (what survives)

**Role:** synthesis agent. Did not perform the work; received 3 branch reports + 3 refutations, and
re-verified every load-bearing claim against disk.
**Date:** 2026-11-01 · **git_head:** `1b9470a0e2129e3c4d982da4453e034a6cc12e24` · **repo_root:** `/home/hugo/codes/mol-metal`
**Environment:** 0 GPU (every call carried `CUDA_VISIBLE_DEVICES=""`; `torch` never imported);
single venv `/home/hugo/codes/mol-metal/.venv/bin/python`; read-only — the only file written by this
agent is this document.
**Tags:** every number below is MEASURED and carries its on-disk path. PROJECTED: none.

---

## §1 The post-repair GPU lock-defeater census

### 1.1 Denominator (MEASURED)

| quantity | value | source |
|---|---|---|
| tracked files total | 9550 | `molmetal/scripts/wfa165_gpu_lock_guard.py` census → `denominator.tracked_files_total` |
| files **enumerated** (tracked `*.py` + `*.sh`, `git ls-files -z`, no `os.walk`) | 2987 (2874 `.py` + 113 `.sh`) | same, `denominator.files_enumerated` / `py_files` / `sh_files` |
| files **read** | 2987 | same, `denominator.files_read` |
| unreadable / lossy-decoded / parse failures | 0 / 0 / 0 | same, `files_unreadable`, `files_decoded_lossy`, `parse_failures` |
| files carrying ≥1 finding | **52** | same, `denominator.files_with_findings` |
| findings total | **68** | `findings_total` |
| findings that **defeat** an ambient `CUDA_VISIBLE_DEVICES=""` lock | **33** | `findings_defeat_ambient_empty_mask` |
| unresolved `env=` sites (reported, NOT counted) | 5 | `unresolved_env_sites` |
| `counts_match_lists` | true | `self_check` |

**Authoritative run = mine, 2026-11-01T03:54:31Z** (`wfa165_gpu_lock_guard.py --json`, stdout captured
to `/tmp/a165_guard_now.json`; the tool's own `--out` artefact is
`molmetal/reports/wfa165_gpu_lock_guard/census.json` @ 03:40:55Z, **older — the tree moved**).

### 1.2 The tree moved under the wave — three census snapshots, one git_head

| snapshot | when (UTC) | findings | defeats | M1 | M2 | M3 | M4 | files |
|---|---|---|---|---|---|---|---|---|
| branch C `census.json` | 03:40:55Z | 89 | 54 | 35 | 0 | 27 | 27 | 72 |
| refuter C re-run | 03:48:36Z | 69 | 34 | 15 | 0 | 27 | 27 | 53 |
| **this synthesis** | **03:54:31Z** | **68** | **33** | **14** | **0** | **27** | **27** | **52** |

The M1 35 → 15 → 14 collapse is **not** a measurement drift: concurrent wave **Wf-A167** landed the
command-prefix repair (`CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-N}"`) into the live launchers
during the wave (verified on disk, e.g. `scripts/launch_wf124_p3_g5.sh:380`,
`scripts/launch_wf308c_todo220_gpu.sh:179`). `git_head` was **identical** across all three runs
(refuter C's C4), so `git_head + generated` does **not** pin a census over a dirty tree — a
working-tree anchor is still missing.

### 1.3 Per-mechanism finding counts + every file:line

**M1 `hard_override` — 14 findings, 11 defeat the lock** (OPEN, but see §6 O2):
- defeats, all in the read-only archive `TODO/archive/wf437_simplify/`:
  `launch_wf124_p3_g4.sh:75` · `launch_wf300_cfm_main.sh:144` · `launch_wf300_cfm_main.sh:196` ·
  `launch_wf306a_drugood_vina_gpu.sh:236` · `launch_wf306b_wf219_ru_gpu.sh:162` ·
  `launch_wf310c_t179_w2_gpu.sh:134` · `launch_wf310c_t179_w2_gpu.sh:177` ·
  `launch_wf310c_todo220_gpu.sh:139` · `launch_wf310c_todo220_gpu.sh:193` ·
  `launch_wf310c_todo220_gpu.sh:221` · `launch_wf74_v2_drugood_dann.sh:332`
- does **not** defeat (forces the blank mask ⇒ lock-safe):
  `molmetal/reports/wf229_ablation/quick_pilot/run_7_remaining.sh:15` ·
  `molmetal/reports/wf256_apply/wf229_quick_pilot_5epoch_BACKUP/run_7_remaining.sh:15` ·
  `scripts/_retired/launch_wf_blk_d2_chemprop_v3_frozen_5seed.sh:50`
- **zero M1 sites remain in live `scripts/`** (measured 03:54:31Z).

**M2 `empty_default_substitution` — 0 findings, 0 defeats.** The `${CUDA_VISIBLE_DEVICES:-…}` idiom
appears in **5 tracked `.sh` files** (measured: `git ls-files -z -- '*.sh'` + grep) —
`TODO/archive/wf437_simplify/launch_wf306a_drugood_vina_gpu.sh:264` ·
`scripts/launch_dmpnn_clean5v3_loso.sh:130` · `:148` · `scripts/launch_wf299_cfm_main.sh:560` ·
`scripts/launch_wf310f_frontier_math_gpu.sh:276` · `scripts/start.sh:741` — all of them
**non-assignment reads**, so M2 = 0 is correct on the mechanism, not on the scan.

**M3 `env_dropping_spawn` — 27 findings, 22 defeat the lock, 5 unknown-effect** (OPEN, §6 O3/O4):
- 22 defeats (all `python_env_no_inherit`, all in `molmetal/tests/`):
  `test_four_domain_sampler.py:419` · `test_wf125_controlled_pair_wrapper.py:86,145,180,211` ·
  `test_wf330c1_c25_25metals.py:171` · `test_wf335r2_wireup_reverify.py:206` ·
  `test_wf337g1_g6_recovery.py:348` · `test_wf341c2_edm_precond.py:201,215,241` ·
  `test_wf349a1_edm_preconditioning.py:255,280` · `test_wf353b2_wf343c2_c3_verify.py:52` ·
  `test_wf356a2_anchor_refresh.py:152` · `test_wf357d1_fused_adam.py:116` ·
  `test_wf361c1_top_m6_kappa_improvement.py:116` · `test_wf414a1_m2r2_launcher.py:117,153,183` ·
  `test_wf_blkc_geom_classifier.py:270,293`
- 5 unknown-effect (symbolic / unresolvable — the guard does not guess):
  `molmetal/molmetal_lam/sbdd_env/training_utils.py:136` · `:217` ·
  `molmetal/scripts/_common_train_args.py:585` · `molmetal/scripts/bench_weak_real_lifting.py:268` ·
  `molmetal/scripts/bench_weak_real_multi_seed.py:159`

**M4 `mask_dropping_default` — 27 findings, 0 defeats** (all `forces_blank` ⇒ harmless):
`molmetal/scripts/integration_verify_scoring.py:258` · `wf226_lomo_cpu_sweep.py:118` ·
`wf229_ablation_launcher.py:161` · `wf385a3_scaffold_wire_in_gate.py:542` ·
`wf404a5_wf29_attr_fix_smoke.py:101` · `wf409a4_wf29_extended_smoke.py:93` ·
`wf_a51_leakage_audit.py:718` · `molmetal/tests/conftest.py:55` · `test_amp_context.py:29` ·
`test_cfm_infrastructure.py:193` · `test_dmpnn_scatter_wiring.py:38` ·
`test_egnn_coord_scatter_wired.py:49` · `test_end_to_end_cached_path.py:36` ·
`test_lipman_import_isolation.py:21` · `test_lipman_sampling_contract.py:12` ·
`test_metal_hybrid_v3.py:22` · `test_metal_lomo_v2.py:429,755` ·
`test_platinai_4cell_checkpoint_roundtrip.py:104` · `test_pocket_conditioned_lipman.py:33` ·
`test_smoke_flag.py:200` · `test_wf328w7_dmpnn_pyg.py:46` · `test_wf328w9_triton_edge.py:57` ·
`test_wf_blk_c5_cfm_collapse_repair_smoke.py:81,140` · `test_wf_blk_d1_attentivefp_smoke.py:66` ·
`test_wf_blk_d2_chemprop_v3_frozen_head_smoke.py:61`

**Headline (defensible): 33 lock-defeating sites remain, of which 11 are in a read-only archive and
22 are in `molmetal/tests/`.** No live `scripts/` launcher defeats the lock any more.

---

## §2 Per-file repairs + the three-case proof

### 2.1 Branch A — 23 shell launchers (`export` form)

Rule applied everywhere: `export CUDA_VISIBLE_DEVICES="<literal>"` →
`export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-<same literal>}"` (no-colon default ⇒ an
explicitly-set ambient, including the empty lock, survives; the original default is kept verbatim).
The AFTER line was **extracted from the file, never hand-typed**.

| file | HEAD line | now line | BEFORE | AFTER | (a) ambient unset | (b) ambient `""` | (c) ambient `1` | original default | all-3 PASS |
|---|---|---|---|---|---|---|---|---|---|
| `env.sh` | 60 | 61 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `molmetal/scripts/setup_molmetal_env.sh` | 52 | 53 | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/_retired/launch_wf_blk_d1_attentivefp_5seed.sh` | 52 | 53 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/_retired/launch_wf_fix3lines_bace.sh` | 42 | 43 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/_retired/launch_wf_fix3lines_cfm.sh` | 61 | 62 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/_retired/launch_wf_fix3lines_cfm_100ep.sh` | 67 | 68 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/_retired/launch_wf_fix3lines_cfm_100ep_qed.sh` | 34 | 35 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/_retired/launch_wf_fix3lines_cfm_30ep_onecycle.sh` | 58 | 59 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/_wf_r28_gpu_master.sh` | 44 | 45 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_cfm_fm_ab_2026_10_09.sh` | 33 | 34 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_cfm_fm_control_2026_10_09.sh` | 37 | 38 | `export CUDA_VISIBLE_DEVICES=1` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}"` | unset→`[1]` | `""`→`[]` | `1`→`[1]` | [1] | PASS |
| `scripts/launch_cfm_frag_ab.sh` | 68 | 69 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_cfm_gen_collapse_fm.sh` | 79 | 80 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_cfm_round7_f5f6_retrain.sh` | 117 | 118 | `export CUDA_VISIBLE_DEVICES="${GPU}"` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${GPU}}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_dmpnn_clean5v3_loso.sh` | 131 | 132 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_dmpnn_clean5v4_loso_d1d3.sh` | 148 | 149 | `export CUDA_VISIBLE_DEVICES="${GPU}"` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${GPU}}"` | unset→`[1]` | `""`→`[]` | `1`→`[1]` | [1] | PASS |
| `scripts/launch_m19_lomo_4metal_sensitivity.sh` | 55 | 56 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_pa14_moleculenet_cap512_d4.sh` | 69 | 70 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_pa15_metalcond_lomo.sh` | 71 | 72 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_pa17_chemprop_v2_moleculenet.sh` | 61 | 62 | `export CUDA_VISIBLE_DEVICES="${PA17_GPU:-0}"` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${PA17_GPU:-0}}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_pa17_morgan_mlp_moleculenet.sh` | 63 | 64 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_pa3_chemprop_lomo.sh` | 65 | 66 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |
| `scripts/launch_wf_fix3lines_cfm_c5_repair.sh` | 154 | 155 | `export CUDA_VISIBLE_DEVICES=0` | `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"` | unset→`[0]` | `""`→`[]` | `1`→`[1]` | [0] | PASS |

**Re-verified by this agent against `molmetal/reports/wfa165_shell_launchers/proof.json`:** 23/23
entries have case (a) = the file's original default, case (b) = `[]` (lock respected), case (c) = `[1]`
(explicit choice wins) ⇒ **23 all-3-PASS, 0 BAD**; and **0 on-disk drift** (each file's line N still
equals the recorded `extracted_line`). Denominator recorded in the same artefact: 113 tracked `*.sh`
scanned, 27 with an export line, 90 "not applicable" each with a reason, 23 defeater sites at HEAD,
23 repaired, `defeater_sites_unrepaired: []`.

Left alone on purpose: `molmetal/reports/**` archived run scripts + `scripts/_retired/**` (their export
FORCES the blank mask ⇒ lock-safe, cannot re-enable a GPU) and `scripts/start.sh` (owned by branch B).

### 2.2 Branch B — `scripts/start.sh` (only file changed)

- **BEFORE** (`git show HEAD:scripts/start.sh`): `if [[ -n "$_cvd" ]]; then export CUDA_VISIBLE_DEVICES="$_cvd"; echo "[start.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from yaml compute.gpu=$GPU)"; else <WARN> fi`
- **AFTER** (now `scripts/start.sh:712-743`): `if [[ "$_AMBIENT_CVD_SET" == "1" ]]; then export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"; echo "[start.sh] CUDA_VISIBLE_DEVICES='$_AMBIENT_CVD' (ambient mask honoured — yaml compute.gpu=$GPU NOT applied)"; [[ -z "$_AMBIENT_CVD" ]] && echo "<WARN> empty = no-GPU lock"; elif [[ -n "$_cvd" ]]; then export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-$_cvd}"; echo "(from yaml compute.gpu=$GPU)"; else <WARN unchanged>; fi`
- **NEW capture block** `scripts/start.sh:66-73`: `_AMBIENT_CVD_SET` / `_AMBIENT_CVD` via
  `${CUDA_VISIBLE_DEVICES+set}` (POSIX-safe, `set -u`-safe; unset ≠ empty).
- **Three-case proof** (`molmetal/reports/wfa165_start_shell/proof.json`, 6 cases, re-read by me):
  (a) ambient unset + yaml `cuda:0` ⇒ mask `"0"`, stdout byte-identical to pre-repair;
  (a′) ambient unset + yaml `cuda:3` ⇒ `"3"`;
  (b) ambient `""` ⇒ mask stays SET-EMPTY + "ambient mask honoured … NOT applied" + no-GPU WARN
  (pre-repair gave `"0"` = the defect);
  (c) ambient `"1"` ⇒ mask `"1"` (pre-repair gave `"0"`).
- **Whole-file property (measured):** exactly 2 exports, at `:728` and `:738`, both at conditional
  depth 1; the presence test at `:69` precedes them; no unconditional export.
- The real blocks were extracted by sentinel markers and executed in `bash -c` with a stubbed `$GPU`;
  `scripts/start.sh` itself was never executed.

---

## §3 Do the regression tests bite? + the refuters' verdicts

| branch | test file | my run (CPU, `CUDA_VISIBLE_DEVICES=""`) | negative control | refuter verdict |
|---|---|---|---|---|
| A | `molmetal/tests/test_wfa165_shell_launchers.py` | **7 passed** | YES — re-adding `export CUDA_VISIBLE_DEVICES=0` to `env.sh` (and `:-` to `setup_molmetal_env.sh`) fails 3 of the 7 in an isolated sandbox; reverted ⇒ 7 pass. Independently reproduced by refuter A (`/tmp/a165_negctl`, repo untouched). | **NOT REFUTED** — C1–C4, C6, C7 PASS; C5 PASS-but-NOT-CLEAN |
| B | `molmetal/tests/test_wfa165_start_shell.py` | **9 passed** | YES — the pre-repair block **and** the real file with the block reverted are both FLAGGED (`ambient "" → "0"`, `ambient "1 → "0"`); current block: 0 violations. Independently reproduced by refuter B. | **NOT REFUTED (core repair survives)** — C1–C6 PASS, but **1 BLOCKING surviving concern (S1)** + S2–S5 |
| C | `molmetal/tests/test_wfa165_gpu_lock_guard.py` | **15 passed** | YES — negative control on `git show HEAD:` copies of the 4 edited launchers ⇒ 4 findings, all defeats=yes; post-repair 0. Two guard mutants (shell classifier, python scanner) each break 3 negative-control tests. Independently reproduced by refuter C. | **REFUTED** — the census numbers reproduce exactly, but the closing claim "All deliverables are in place and verified" does not: **C1c, C1d, C3c (blocking), C6 FAIL** |

**Combined: 31 passed / 0 failed** (`pytest molmetal/tests/test_wfa165_shell_launchers.py
molmetal/tests/test_wfa165_start_shell.py molmetal/tests/test_wfa165_gpu_lock_guard.py -q`, 56 s, CPU).

---

## §4 Files created / modified

**Branch A — modified (23 tracked `.sh`):** the 23 rows of §2.1. **Created:**
`molmetal/tests/test_wfa165_shell_launchers.py`;
`molmetal/reports/wfa165_shell_launchers/proof.json`;
`molmetal/reports/wfa165_shell_launchers/wfa165_shell_launcher_repair.py`.
**Branch B — modified (1):** `scripts/start.sh`. **Created:**
`molmetal/tests/test_wfa165_start_shell.py`; `molmetal/reports/wfa165_start_shell/proof.json`;
`molmetal/reports/wfa165_start_shell/make_proof.py`.
**Branch C — created (4):** `molmetal/scripts/wfa165_gpu_lock_guard.py`;
`molmetal/tests/test_wfa165_gpu_lock_guard.py`;
`molmetal/reports/wfa165_gpu_lock_guard/census.json`;
`molmetal/reports/wfa165_gpu_lock_guard/mechanism_index.json`.
**Refuters — created (read-only, own dirs):** `molmetal/reports/wfa165_branchA_refutation/`
(`refutation.json`, `refutation.md`, `post_repair_census.json`, `independent_behaviour.json`),
`molmetal/reports/wfa165_b_refutation/` (`refutation.json`, `refutation.md`),
`molmetal/reports/wfa165_c_refutation/` (`refutation.json`, `refutation.md`).
**This agent — created (1):** `molmetal/reports/wfa165_synthesis/synthesis.md`.
Note: `molmetal/reports/**/*.json` is git-ignored (artefacts are on disk, not in git); the guard and the
three test files are **untracked** (`git status` ⇒ `??`), so the guard's own census does not cover the
wave's own sites.

---

## §5 OPEN items — what the main thread must surface to the user

**O1 — HIGH / BLOCKING (refuter B's S1), CONFIRMED by me: the composed launch path now lands on a
different GPU than the yaml asks for.**
`scripts/launch_cfm_fm_control_2026_10_09.sh:38` exports `"${CUDA_VISIBLE_DEVICES-1}"` and then
delegates to `scripts/start.sh "$YAML"` (`:54`); `configs/cfm_fm_control_2026_10_09.yaml:203` says
`compute.gpu: cuda:0`. Because the launcher *always* exports, `start.sh` can no longer tell "the
operator asked for GPU 1" from "the launcher defaulted to 1", so the yaml's device is dead in that
path. Measured by composing the **real** blocks in `bash -c`: ambient unset ⇒
`[start.sh] CUDA_VISIBLE_DEVICES='1' (ambient mask honoured — yaml compute.gpu= NOT applied)`,
whereas at HEAD the yaml-derived value (`0`) won ⇒ composed mask **"0" → "1"**. GPU 0 = RTX PRO 6000
Blackwell 98 GB, GPU 1 = RTX 5090 32 GB (`molmetal/molmetal_lam/sbdd_env/training_utils.py:86-90`).
This contradicts `scripts/start.sh:705-708` + AGENTS.md §5.11 (yaml = device SSOT). The **lock still
holds** (ambient `""` ⇒ the mask stays empty ⇒ no GPU); what broke is the yaml's device authority.
Scope: of the 4 repaired launchers that delegate to `start.sh`
(`launch_cfm_fm_ab`, `launch_cfm_fm_control`, `launch_cfm_gen_collapse_fm`,
`launch_wf_fix3lines_cfm_c5_repair`) only `launch_cfm_fm_control` flips, because only it defaults to
`1` (the other three default to `0` and all four yamls say `cuda:0`).
`scripts/launch_dmpnn_clean5v4_loso_d1d3.sh` also defaults to `1` (`GPU="${GPU:-1}"`, `:99`) but does
not delegate to `start.sh` (grep: 0 hits) ⇒ no yaml conflict there.

**O2 — HIGH: M1 is not zero; 11 defeating sites remain, all in the read-only archive.**
`TODO/archive/wf437_simplify/` (9 files, 11 sites — full list in §1.3) + 3 lock-safe `forces_blank`
sites. Branch A deliberately did not touch the archive; it needs an owner decision (rewrite history
vs. accept + declare). Separately: branch A's "4th mechanism" (32 command-prefix sites) was left
open by Wf-A165 but **repaired by concurrent Wf-A167** (verified: the 20 live `scripts/` sites now
read `"${CUDA_VISIBLE_DEVICES-N}"`; `scripts/launch_wf124_p3_g5.sh:380` etc.). The wave may claim the
live tree is clean only after a census re-taken on the settled tree — the numbers moved 89 → 69 → 68
findings in 14 minutes under one unchanged `git_head`.

**O3 — MEDIUM: 22 M3 defeats in `molmetal/tests/`** (full list §1.3): Python that spawns a child with
an `env=` that does not inherit `os.environ`, so the ambient `""` lock never reaches the child. Outside
branch A's remit (branch A = shell), inside the guard's remit. OPEN.

**O4 — MEDIUM: 5 M3 sites with unknown effect** — `training_utils.py:136,217`,
`_common_train_args.py:585`, `bench_weak_real_lifting.py:268`, `bench_weak_real_multi_seed.py:159`.
The guard refuses to guess (`effect=unknown`, `defeats=unknown`); a human must read them. OPEN.

**O5 — BLOCKING, CONFIRMED by me: the guard itself crashes on a form it claims to support.**
`molmetal/scripts/wfa165_gpu_lock_guard.py:335` does `m.group("mods")` on `_SHELL_ENV_CMD_RE`
(`^env\s+CUDA_VISIBLE_DEVICES\s*=\s*(?P<rest>.*)$`), which has **no `mods` group** ⇒ `IndexError: no
such group`. Reproduced directly:
`scan_shell_text("env CUDA_VISIBLE_DEVICES=0 cmd")` → `IndexError`, while
`"CUDA_VISIBLE_DEVICES=0 cmd"` and `"export CUDA_VISIBLE_DEVICES=0"` classify fine. Via the CLI the
traceback exits 1 — indistinguishable from "findings ⇒ gate" — and **no `--out` artefact is written**
(`scan_repository` has no per-file try/except, `:1117`). 0 occurrences in the tracked tree today, so
the census numbers are unaffected; the defect is latent and must be fixed before this guard gates a
real launch. One-line fix: `m.groupdict().get("mods") or ""` won't help either (`group()` raises
before the `or`) — the regex needs a `mods` group, or the code must branch on `env_cmd`.

**O6 — REPORT INTEGRITY (refuter C's C1c): branch C's Wf-A163 cross-check does not trace to disk.**
Branch C's report quotes `lock_defeaters=54, lock_forcers=41, files_enumerated_in_scope=6311,
files_read=2993, hits_by_kind={272,293,1306,70,11}`. The artefact
`molmetal/reports/wfa163_gpu_tooling_census/gpu_tooling_census.json` (regenerated 11:50, **after**
branch C's own census) currently reads `files_enumerated_in_scope=6339, files_read=3011,
hits_by_kind={274,293,1389,58,11}`; the copy branch C embedded in its own
`mechanism_index.json → wf_a163_census_verification.artifact_denominator` says `6325 / 3005` and its
sibling `note` says `54 / 41 / 6311` — **the artefact contradicts itself**. Three different numbers,
none of them the quoted one ⇒ AGENTS.md §5.2 violation; the Wf-A163 cross-check must be re-quoted from
disk (and the Wf-A163 artefact is itself a moving target — it was regenerated under the same tree).

**O7 — REPORT INTEGRITY (refuter C's C1d): the repo-wide `:-` scan figure is wrong.** Branch C's
report says the `${CUDA_VISIBLE_DEVICES:-…}` idiom appears in exactly **1** tracked `.sh`; my own scan
finds **5 files** (§1.3, M2). M2 = 0 remains correct (all 5 are non-assignment reads) and the guard
does fire M2 on the real pre-repair `setup_molmetal_env.sh` — but the stated scan result is wrong and
must be corrected before the report is quoted.

**O8 — LOW (refuter C's C6): undeclared blind spots.** `unset CUDA_VISIBLE_DEVICES` and
`env -u CUDA_VISIBLE_DEVICES` are neither detected nor listed in the guard's declared-limits section
(0 occurrences today).

**O9 — LOW: the guard enumerates TRACKED files only.** The wave's own untracked files are invisible to
it (the guard + 3 test files), and Makefile/container/untracked CVD setters are outside its scope.

**O10 — LOW, pre-existing (not introduced by Wf-A165):** `scripts/launch_dmpnn_clean5v3_loso.sh:148`
prints `CUDA_VISIBLE_DEVICES: 1` even under the lock and when unset, while the script exports `0` —
the echo was left untouched by the export repair. Reporting defect, not a lock defect.

---

## §6 Items the main thread may safely close

1. **Branch A's 23 `export`-form defeaters** — repaired 23/23, three-case proof 23/23 (re-verified by
   me from the artefact, 0 drift), `bash -n` clean, modes preserved, 7/7 tests, refuter A
   **NOT REFUTED**. CLOSE.
2. **Branch B's `scripts/start.sh` yaml-mask override** — the ambient mask (including the empty lock)
   now wins; yaml still decides when the ambient is unset; 9/9 tests; refuter B **NOT REFUTED on the
   core repair**. CLOSE the core; **keep O1 open** (it is a composition defect that only appears when
   a launcher pre-exports a non-zero default).
3. **M2 `empty_default_substitution`** — 0 findings repo-wide; the `:-` idiom now appears only in
   non-assignment reads. CLOSE.
4. **The command-prefix ("4th mechanism") sites in live `scripts/`** — repaired by concurrent
   Wf-A167, verified on disk (20/20 sites now `"${CUDA_VISIBLE_DEVICES-N}"`); the guard's census
   confirms 0 M1 sites under live `scripts/` at 03:54:31Z. CLOSE for the live tree, conditional on a
   census re-taken once Wf-A167 settles (the tree is still moving).
5. **M4's 27 sites** — all `forces_blank`, 0 defeats ⇒ lock-safe by construction, not defects. CLOSE
   (keep the list as declared limits).

---

## §7 Provenance of this document

Every number above was produced by this agent from files on disk (guard `--json` re-run at
03:54:31Z; `wfa165_shell_launchers/proof.json`; `wfa165_start_shell/proof.json`;
`wfa165_gpu_lock_guard/census.json` + `mechanism_index.json`;
`wfa163_gpu_tooling_census/gpu_tooling_census.json`; the three refutation artefacts; `git show HEAD:`;
`pytest` runs with `CUDA_VISIBLE_DEVICES=""`). 0 GPU used. Only this file was written.
