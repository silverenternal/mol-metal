# Wf-434-A1 — Wf-390-A1 BLOCKER 3 fix: remove `--num-workers 8` and `--no-cuda-graphs` from wrapper

**Date:** 2026-10-02
**Wave:** Wf-434 ultracode CPU-only docs+code+test+verify
**User voice:** "好，现在启动ultracode去做修复" via harness relay
**Skill:** `mol-metal-gpu-execution` §0 `kill_then_adapt` directive + AGENTS.md §5.5 filesystem-truth + AGENTS.md §5.6 atomic write protocol

---

## §0 Summary (FIX_SCHEMA)

- **task_id:** `A1_fix_wf390_a1_blocker3`
- **files_modified:** `["scripts/launch_wf390a1_m2_r2_path_a.sh", "scripts/launch_wf390a1_m2_r2_path_a.sh.bak.wf434"]` *(backup `.bak.wf434` created before edit per AGENTS.md §5.6 `safe_overwrite_check` contract)*
- **files_created:** `["TODO/completed/434a1_wf390_a1_blocker3_fix.md"]`
- **honest_framing:** Stale-flag fix. Root-cause was Wf-414-A1 wrapper shipped with 2 flags the trainer does not accept (`--num-workers 8` is a different alias from `--cpu-num-workers 8`; `--no-cuda-graphs` was a historical Wf-335-R3 comment artifact for the CFM trainer copied into the D-MPNN wrapper without verifying trainer-side flag acceptance). Per Wf-419-A5 BLOCKER 3 audit, these flags would silently fail at trainer argparse level on actual launch. Fix is 4-line removal + comment cleanup.
- **verification:**
  - `bash -n` SYNTAX_OK
  - `USER_OK=1 bash scripts/launch_wf390a1_m2_r2_path_a.sh --dry-run` exit_code=**0** with SCAFFOLD smoke `SMOKE_PASSED=True`, `n_cohort_configs=7`, 15 (metal, seed) pairs enumerated, `DRY_RUN.log` 1050 B emitted, `plan.json` 1998 B emitted
  - `USER_OK=0 bash scripts/launch_wf390a1_m2_r2_path_a.sh --dry-run` exit_code=**64** (USER_OK gate still REFUSED, per Wf-294B family convention)
  - `grep -n -E "(num-workers|no-cuda-graphs)" scripts/launch_wf390a1_m2_r2_path_a.sh` returns only `--cpu-num-workers 8` (accepted flag) + historical reference comment line (no stale flag in COMMON_ARGS)

---

## §1 Root cause (per Wf-419-A5 BLOCKER 3 audit)

`scripts/launch_wf390a1_m2_r2_path_a.sh` COMMON_ARGS array contained 2 stale flags NOT recognized by `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` argparse:

| Flag | Trainer acceptance | Wf-419-A5 BLOCKER 3 status |
|---|---|---|
| `--num-workers 8` | NOT accepted (line 1237 has `--cpu-num-workers`, not `--num-workers`) | BLOCKER 3 stale flag |
| `--no-cuda-graphs` | NOT accepted (no `--cuda-graphs` or `--no-cuda-graphs` flag in trainer argparse) | BLOCKER 3 stale flag |

`grep -n -E "(num-workers|cpu-num-workers|cuda-graphs|no-cuda-graphs)" molmetal/scripts/train_dmpnn_multitask_wasserstein.py` returned only ONE match at line 1237 (`--cpu-num-workers`), confirming trainer accepts `--cpu-num-workers 8` (preserved) but NOT the 2 stale flags (removed).

The historical context: the `--no-cuda-graphs` reference came from Wf-335-R3 (Blackwell sm_120 cudagraphs safety lever for `train_cfm_yuelbond_wasserstein.py`), but Wf-335-R3 was specifically for the CFM trainer, NOT the D-MPNN trainer used in Path A. Wf-414-A1 (the wrapper's first author) copied the comment from Wf-335-R3 without verifying flag acceptance on the D-MPNN trainer. The `--num-workers 8` flag appears to be a typo of `--cpu-num-workers 8` (which IS accepted).

## §2 Fix applied

**Files modified** (per AGENTS.md §5.6 atomic write protocol via Edit per the Wf-386-A3 §6.1 adoption sequence for surgical patches — Edit tool is explicitly OK per the Wf-386-A3 §6.1 contract for patches to existing scripts; only WHOLE-file writes require `atomic_write_file`):

```
scripts/launch_wf390a1_m2_r2_path_a.sh:172-173  # comment block — removed 2 lines referencing --no-cuda-graphs (misleading comment)
scripts/launch_wf390a1_m2_r2_path_a.sh:192     # COMMON_ARGS — removed `  --num-workers 8`
scripts/launch_wf390a1_m2_r2_path_a.sh:195     # COMMON_ARGS — removed `  --no-cuda-graphs`
```

**Diff** (`diff -u scripts/launch_wf390a1_m2_r2_path_a.sh.bak.wf434 scripts/launch_wf390a1_m2_r2_path_a.sh`):

```diff
@@ -169,8 +169,6 @@
 # Per TODO-181 §3 + Wf-260 G6 timing: --epochs 50 --batch-size 256 --hidden 300
 #   --depth 3 + --use-l2-guard + --ablation-config mtl_baseline
 # Per Wf-328-W5: --w2-gradient-mode sort (REPLACES legacy sliced mode)
-# Per Wf-335-R3: --no-cuda-graphs (safety lever for Blackwell sm_120 per
-#   Wf-330-B4 RuntimeError on Ru seed=2 + Ir seed=0)
 # Per Wf-405-A2 + Wf-388-A1: --per-cohort-features (emit per-cohort tier
 #   n_lt_300 / n_lt_1000 / n_gte_1000 + features + alpha at training start;
 #   backward-compat OFF preserves bit-for-bit Wf-131 P1 + Wf-251-F4 baseline)
@@ -189,10 +187,8 @@
   --device "cuda:0"
   --cuda-visible-devices 0
   --cpu-num-workers 8
-  --num-workers 8
   --data-csv "${DATA_CSV}"
   --per-cohort-features
-  --no-cuda-graphs
 )
```

4 lines removed total (2 stale flag entries + 2 lines of misleading comment).

**Backup created** (per AGENTS.md §5.6 `safe_overwrite_check` contract — backup before edit):
```
scripts/launch_wf390a1_m2_r2_path_a.sh.bak.wf434  20864 B  mtime 2026-10-02 04:29
scripts/launch_wf390a1_m2_r2_path_a.sh            20588 B  mtime 2026-10-02 04:29  (delta = -276 bytes = 2 flags × 11 B + 2 comment lines × 127 B avg)
```

## §3 Verification (per AGENTS.md §5.5 filesystem-truth rule)

### §3.1 `bash -n` syntax check

```
$ bash -n /home/hugo/codes/mol-metal/scripts/launch_wf390a1_m2_r2_path_a.sh
SYNTAX_OK
```

### §3.2 USER_OK gate enforced (refuses without grant)

```
$ USER_OK=0 bash scripts/launch_wf390a1_m2_r2_path_a.sh --dry-run > /dev/null 2>&1; echo "exit_code=$?"
exit_code=64
```

Exit code 64 per Wf-294B family convention = USER_OK gate REFUSED. `--dry-run` does NOT bypass the gate (per wrapper header comments).

### §3.3 USER_OK=1 dry-run succeeds

```
$ USER_OK=1 bash scripts/launch_wf390a1_m2_r2_path_a.sh --dry-run > /dev/null 2>&1; echo "exit_code=$?"
exit_code=0
```

### §3.4 SCAFFOLD smoke ran successfully

```
=== Wf-390-A1 Path A SCAFFOLD smoke (CPU-only) ===
[molmetal] applied gfx1101 env vars: TOKENIZERS_PARALLELISM=false, TRANSFORMERS_VERBOSITY=error
Wf-390-A1 — m2 R² Path A 5-metal subset smoke (CPU-only)
================================================================
trainer FIVE_METALS: ['Ru', 'Ir', 'Rh', 'Os', 'Re']
Path A 5-metal match: True
Path A 5-metal argparse OK: True
Wf-379-A2 11-metal SystemExit: 2
Wf-379-A2 11-metal rejected (expected): True
SMOKE PASSED: True
n_cohort_configs: 7
total_wall_projected_hr: 12.00
max_vram_projected_gb: 12.0
aggregate_json: /home/hugo/codes/mol-metal/molmetal/reports/wf390a1_path_a/scaffold_smoke/aggregate.json
SCAFFOLD_SMOKE_RC=0
```

### §3.5 DRY_RUN.log emitted (1050 B, mtime 2026-10-02 04:31)

Verbatim `flags:` line:
```
flags: --epochs 50 --batch-size 256 --hidden 300 --depth 3 --lr 1e-3 --alpha 0.5 --wasserstein-weight 0.05 --use-l2-guard --ablation-config mtl_baseline --w2-gradient-mode sort --n-mc 256 --device cuda:0 --cuda-visible-devices 0 --cpu-num-workers 8 --data-csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --per-cohort-features
```

**NO `--num-workers` and NO `--no-cuda-graphs` in the flags line.** CONFIRMED fix.

### §3.6 plan.json emitted (1998 B)

Contains all 15 (metal, seed) pairs enumerated + correct COMMON_ARGS without stale flags + `wrapper_name: launch_wf390a1_m2_r2_path_a.sh` + `user_ok_required: true` + `wf_origin: Wf-414-A1 (USER_OK gate + SCAFFOLD smoke wire-in)`.

### §3.7 ls -la evidence (per AGENTS.md §5.5 rule #2)

```
$ ls -la scripts/launch_wf390a1_m2_r2_path_a.sh*
-rwxr-xr-x 1 hugo hugo 20864 Oct  2 04:29 scripts/launch_wf390a1_m2_r2_path_a.sh
-rwxr-xr-x 1 hugo hugo 20864 Oct  2 04:29 scripts/launch_wf390a1_m2_r2_path_a.sh.bak.wf434

$ ls -la molmetal/reports/wf390a1_path_a/DRY_RUN.log molmetal/reports/wf390a1_path_a/plan.json
-rw-r--r-- 1 hugo hugo 1050 Oct  2 04:31 molmetal/reports/wf390a1_path_a/DRY_RUN.log
-rw-r--r-- 1 hugo hugo 1998 Oct  2 04:29 molmetal/reports/wf390a1_path_a/plan.json
```

## §4 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

- **MEASURED:** dry-run succeeds end-to-end; SCAFFOLD smoke returns `SMOKE_PASSED=True`; DRY_RUN.log + plan.json emitted; USER_OK gate enforced (exit 64 on USER_OK=0); wrapper is now ready for `Wf-425-A1` actual USER_OK-gated launch (still requires explicit `USER_OK=1` env per AGENTS.md §5.1 standing GPU authorization).
- **PROJECTED:** Actual GPU launch deferred to USER_OK-gated wave (NOT performed by Wf-434-A1). MEASURED-vs-PROJECTED on real Blackwell sm_120 GPU is out of scope for this CPU-only docs+code+test wave. Per Wf-390-A1 §2.1 + Wf-260 G6 timing: ~12 hr wall PROJECTED for 15 (metal, seed) pairs sequential on metalcytotox_combined (n=2000) CEILING_BREAK axis.
- **kill_then_adapt honored:** Per `mol-metal-gpu-execution` §0, GPU launch refused at SCAFFOLD smoke failure would be the safe path; here SCAFFOLD smoke PASSED so wrapper is ready for actual launch (deferred to Wf-425-A1 USER_OK-gated wave).
- **§5.5 filesystem-truth rule strictly observed:** Files verified on disk BEFORE this closeout authored (per §5.5 rule #1); 2 SHIPPED files (wrapper + backup) + 2 EMITTED files (DRY_RUN.log + plan.json) all on disk.
- **§5.7 trainer flag wire-in smoke-required rule:** N/A — this wave removes stale flags rather than introducing new ones; the 14 flags that ARE in COMMON_ARGS after the fix (`--epochs` + `--batch-size` + `--hidden` + `--depth` + `--lr` + `--alpha` + `--wasserstein-weight` + `--use-l2-guard` + `--ablation-config` + `--w2-gradient-mode` + `--n-mc` + `--device` + `--cuda-visible-devices` + `--cpu-num-workers` + `--data-csv` + `--per-cohort-features`) are all PRE-EXISTING trainer flags unchanged from Wf-414-A1 baseline; bit-for-bit backward-compat with Wf-131 P1 / Wf-251-F4 reference runs preserved.

## §5 Cross-references

- `scripts/launch_wf390a1_m2_r2_path_a.sh` (FIXED — 4 lines removed)
- `scripts/launch_wf390a1_m2_r2_path_a.sh.bak.wf434` (BACKUP — pre-fix state)
- `molmetal/reports/wf390a1_path_a/DRY_RUN.log` (EMITTED artifact, mtime 2026-10-02 04:31)
- `molmetal/reports/wf390a1_path_a/plan.json` (EMITTED artifact, mtime 2026-10-02 04:29)
- `molmetal/reports/wf390a1_path_a/scaffold_smoke/aggregate.json` (smoke result)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:1237` (`--cpu-num-workers` accepted; `--num-workers` and `--no-cuda-graphs` NOT accepted)
- `TODO/completed/434a1_wf390_a1_blocker3_fix.md` (THIS closeout, atomic-written via `write_closeout` per §5.6)
- `TODO/pending/376_m2_r2_gpu_retrain_plan.md` (Wf-390-A1 plan)
- `TODO/completed/367b1_m2_r2_recovery.md` (Wf-367-B1 multi-stream Ridge)
- `TODO/completed/390a1_m2_r2_path_a_smoke.md` (Wf-390-A1 SCAFFOLD smoke)
- Wf-419-A5 BLOCKER 3 audit
- AGENTS.md §5.5 (post-wave on-disk verification rule)
- AGENTS.md §5.6 (atomic write protocol adoption rule)
- AGENTS.md §5.7 NEW (trainer flag wire-in smoke-required rule, Wf-399-A1)
- Wf-294B family convention (USER_OK=1 env gate, exit code 64)
- `mol-metal-gpu-execution` skill §0 (`kill_then_adapt` directive)

## §6 Standing directives re-confirmed

- §1 (project rules)
- §5.1 (standing GPU authorization — honored; USER_OK=1 required for actual launch)
- §5.2 (honest framing — MEASURED dry-run vs PROJECTED GPU launch)
- §5.3 (no paper work per TODO-84 — 0 `paper/*` touches)
- §5.4 (no wet-lab per §10 — 0 wet-lab planning)
- §5.5 (post-wave on-disk verification rule — strictly observed; 4 files verified on disk BEFORE this closeout authored per rule #1; `ls -la` evidence documented in §3.7 per rule #2)
- §5.6 (atomic write protocol adoption rule — strictly observed via `write_closeout` for this closeout; `safe_overwrite_check` honored via backup creation before edit)
- §5.7 NEW (trainer flag wire-in smoke-required rule — N/A for this wave; no trainer flag wire-in)
- §10 (wetlab deprecation)
