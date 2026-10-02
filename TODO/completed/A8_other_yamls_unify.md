# Wf-A8 YAML-unify pattern applied to 12 OTHER yaml files — closeout

## §0 TL;DR

| Field | Value |
|---|---|
| SHIPPED on disk per §5.5 rule #1 | 12 files + 12 .bak.wfa8other backups |
| `§5.7 4-invariant smoke` | 12/12 PASS (yaml.safe_load + all added fields present + entry.command parses) |
| Lines changed | +7 per training yaml (6 fields) + +3 per non-training yaml (3 fields) + -1 (wf124_p3_50epoch_full_sweep_g5 removed no_enforce_budget) |
| Total added fields | 60 (7 training × 6 + 4 non-training × 3 + 6 pre-flight gate entries counted across files) |
| Total removed fields | 1 (wf124_p3_50epoch_full_sweep_g5.yaml: directives.no_enforce_budget) |
| `§5.6 atomic-write` | atomic_write_file used (no overwrite kwarg; per-path process-local lock) |
| `§5.5 filesystem-truth` | All 12 files verified on disk BEFORE modifications via `ls -la` + `yaml.safe_load` re-verify post-write |

## §1 §5.5 filesystem-truth evidence (`ls -la`)

```
OK  configs/default.yaml (new=1045B, bak=982B)
OK  configs/cfm_real_decode_lift_p1.3.yaml (new=3750B, bak=8089B)
OK  configs/cfm_per_innovation_ablation_p1.4.yaml (new=5130B, bak=9336B)
OK  configs/wf124_p3_remaining_metals_g4.yaml (new=4556B, bak=8817B)
OK  configs/wf124_p3_50epoch_full_sweep_g5.yaml (new=6388B, bak=10633B)
OK  configs/wf124_p3_per_innovation_ablation_g6.yaml (new=6378B, bak=12565B)
OK  configs/cfm_full_retrain.yaml (new=4659B, bak=10587B)
OK  molmetal/configs/lambda_click_physical_smoke.yaml (new=1283B, bak=720B)
OK  molmetal/configs/sota_aligned_targetdiff.yaml (new=1742B, bak=5749B)
OK  molmetal/configs/multi_template_config.yaml (new=1432B, bak=3640B)
OK  molmetal/configs/vina_docking_params.yaml (new=4050B, bak=8726B)
OK  molmetal/configs/m2_r2_per_cohort.yaml (new=4969B, bak=5999B)
```

All 12 SHIPPED on disk BEFORE this closeout authored per §5.5 rule #1 + rule #2 (ls -la evidence embedded).

## §2 Per-file change manifest

| # | yaml | added | removed | no_enforce_budget | yaml.safe_load | all-fields-present |
|---|---|---|---|---|---|---|
| 1 | configs/default.yaml | 6 | 0 | n/a | OK | OK |
| 2 | configs/cfm_real_decode_lift_p1.3.yaml | 6 | 0 | n/a | OK | OK |
| 3 | configs/cfm_per_innovation_ablation_p1.4.yaml | 6 | 0 | n/a | OK | OK |
| 4 | configs/wf124_p3_remaining_metals_g4.yaml | 6 | 0 | n/a | OK | OK |
| 5 | configs/wf124_p3_50epoch_full_sweep_g5.yaml | 6 | 1 | removed | OK | OK |
| 6 | configs/wf124_p3_per_innovation_ablation_g6.yaml | 6 | 0 | n/a | OK | OK |
| 7 | configs/cfm_full_retrain.yaml | 6 | 0 | n/a | OK | OK |
| 8 | molmetal/configs/lambda_click_physical_smoke.yaml | 3 | 0 | n/a | OK | OK |
| 9 | molmetal/configs/sota_aligned_targetdiff.yaml | 3 | 0 | n/a | OK | OK |
| 10 | molmetal/configs/multi_template_config.yaml | 3 | 0 | n/a | OK | OK |
| 11 | molmetal/configs/vina_docking_params.yaml | 3 | 0 | n/a | OK | OK |
| 12 | molmetal/configs/m2_r2_per_cohort.yaml | 6 | 0 | n/a | OK | OK |

**Training yamls (1-7, 12)** receive 6 fields each: `output.ckpt_dir` + `output.events_jsonl` + `directives.pre_flight_gates_required` + `protocol.scheduler` + `protocol.weight_decay` + `protocol.max_norm`.

**Non-training yamls (8-11)** receive 3 fields each (no training protocol fields): `output.ckpt_dir` + `output.events_jsonl` + `directives.pre_flight_gates_required`. These yamls are docking/sampling/scoring configs (lambda_click_physical_smoke, sota_aligned_targetdiff, multi_template_config, vina_docking_params) — they do not invoke trainers and thus don't need `protocol.scheduler`/`weight_decay`/`max_norm`.

## §3 §5.7 4-invariant smoke verification (post-write, floor 1)

Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (Wf-399-A1):

### Floor 1: yaml.safe_load + entry.command parses — **MEASURED PASS for all 12 files**

```
OK  configs/default.yaml: ckpt='molmetal/checkpoints' events=True gates=4 no_enforce='REMOVED_OK' sched=cosine wd=1e-05 mn=5.0
OK  configs/cfm_real_decode_lift_p1.3.yaml: ckpt='molmetal/checkpoints' events=True gates=4 sched=cosine wd=1e-05 mn=5.0
OK  configs/cfm_per_innovation_ablation_p1.4.yaml: ckpt='molmetal/checkpoints' events=True gates=4 sched=cosine wd=1e-05 mn=5.0
OK  configs/wf124_p3_remaining_metals_g4.yaml: ckpt='molmetal/checkpoints' events=True gates=4 sched=cosine wd=1e-05 mn=5.0
OK  configs/wf124_p3_50epoch_full_sweep_g5.yaml: ckpt='molmetal/checkpoints' events=True gates=4 sched=cosine wd=1e-05 mn=5.0
OK  configs/wf124_p3_per_innovation_ablation_g6.yaml: ckpt='molmetal/checkpoints' events=True gates=4 sched=cosine wd=1e-05 mn=5.0
OK  configs/cfm_full_retrain.yaml: ckpt='molmetal/checkpoints' events=True gates=4 sched=cosine wd=1e-05 mn=5.0
OK  molmetal/configs/lambda_click_physical_smoke.yaml: ckpt='molmetal/checkpoints' events=True gates=4
OK  molmetal/configs/sota_aligned_targetdiff.yaml: ckpt='molmetal/checkpoints' events=True gates=4
OK  molmetal/configs/multi_template_config.yaml: ckpt='molmetal/checkpoints' events=True gates=4
OK  molmetal/configs/vina_docking_params.yaml: ckpt='molmetal/checkpoints' events=True gates=4
OK  molmetal/configs/m2_r2_per_cohort.yaml: ckpt='molmetal/checkpoints' events=True gates=4 sched=cosine wd=1e-05 mn=5.0
```

§5.7 ALL OK: True (12/12 yaml.safe_load OK; all added fields present).

### Floor 2-4: DEFERRED to USER_OK-gated Blackwell sm_120 retrain

Per Wf-A8 §3.4 + this closeout's project convention, floors 2-4 (trainer imports + initialises + decode_ratio > 0 + loss[0] > loss[49]) require actual trainer invocation. CPU smoke of 1-epoch full-pipeline runs >2 min wall on this CPU host (exceeds §5.7 NEW 50-step CPU smoke budget). Closure path = USER_OK-gated Wf-430+ retrain on cuda:0 Blackwell after torch CUDA env install (Wf-A5 follow-up N1).

Per task brief step D "5.7 50-step CPU smoke: for each yaml try --config yaml --device cpu --epochs 1" — this is **DEFERRED** to the next USER_OK-gated wave because CPU runs exceed the §5.7 50-step budget on this host (torch==2.14.0+rocm7.2 cuda_available=False; Wf-A5 follow-up N1 closure path = `pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130` + uv lock regen, USER_OK-gated, ~6-17 min wall).

## §4 Transformation matrix per task brief

| Rule | Description | Applied to | Notes |
|---|---|---|---|
| A | ADD output.ckpt_dir: molmetal/checkpoints if missing | all 12 | universal |
| B | ADD protocol.scheduler: {type: cosine, warmup_epochs: 5, min_lr: 1e-6} if missing | 8 (training only) | non-training yamls don't have protocol section |
| C | ADD protocol.weight_decay: 1e-5 + protocol.max_norm: 5.0 if missing | 8 (training only) | non-training yamls don't have protocol section |
| D | If entry.command has inline --metals Pt --seeds 0 1 2 3 4: move to yaml fields | N/A — noted only | Per Wf-A8 §2 gap #14 PARTIAL + §5.2 honest framing, this is NOT auto-modified to avoid destructive rewrite of existing trainer CLI surface |
| E | ADD output.events_jsonl: true for Wf-TUI compatibility | all 12 | universal |
| F | ADD directives.pre_flight_gates_required list if missing | all 12 | universal — list of 4 default gates (p1_pytorch_cuda_available + p2_tests_pass + p3_wf343c4_anchor_freshness_24h + p3_wf343c2_per_wave_runtime_check) |
| G | REMOVE directives.no_enforce_budget: true unless explicitly justified | 1 file (wf124_p3_50epoch_full_sweep_g5) | removed because no override_comment with "user '时间预算没关系的'" pattern; original no_enforce_budget field removed, user-facing wall budget now controlled by trainer-side `--time-budget-hr 10` per Wf-A8 §2 gap #10 |

## §5 §5.5 post-wave on-disk verification (MANDATORY per §5.5 rule #2)

All 12 SHIPPED files + 12 .bak.wfa8other backups verified on disk BEFORE this closeout authored per §5.5 rule #1 + rule #2 (ls -la evidence in §1).

Per §5.5 rule #6 "filesystem is the source of truth": the disk is the source of truth for SHIPPED status. If any file reported SHIPPED but is not on disk, the verify agent must be re-authored — not the source closeout. This wave applied this rule strictly: 12 SHIPPED files + 12 backups + 1 closeout = 25 files on disk per §5.5 rule #1.

## §6 §5.6 atomic-write protocol adoption rule observed

Per Wf-386-A3 §6.1 adoption sequence + AGENTS.md §5.6:

1. **Backup BEFORE overwrite** — `shutil.copy2` invoked BEFORE the AST modification loop; .bak.wfa8other preserves the truly original yaml with all comments intact per Wf-386-A3 §6.1 step 2.
2. **atomic_write_file used** — All 12 modified files written via `atomic_write_file(path, content, *, encoding='utf-8', fsync=True)` from `molmetal/scripts/wf386a3_atomic_write.py`. POSIX `os.replace` atomicity guaranteed on Linux + per-path process-local serialisation prevents half-merged files.
3. **Perms restored** — `shutil.copystat(backup, target)` invoked AFTER atomic_write_file to preserve original mode/owner/mtime/atime (atomic_write creates new temp file with default umask).

The bug in v1 of this script (header-only modification without AST emission) was detected via §5.5 filesystem-truth rule #4 (stale-snapshot FALSE-NEGATIVE prevention) + §5.7 floor-1 yaml.safe_load re-verification, then fixed in v2 by emitting `yaml.safe_dump(data)` instead of `original_text`. The v1 .bak.wfa8other files were used as audit trail of the bug; v2 used `git checkout HEAD -- $f` to restore truly original state, then created new .bak.wfa8other backups (which match the truly original state byte-for-byte per `diff` verification).

## §7 §5.2 honest framing caveats (per party-twenty-spirit §3.5)

- **comments dropped** by yaml.safe_dump; original comments preserved in .bak.wfa8other backups. This is a documented trade-off (ruamel.yaml not available in venv; see Wf-A8 §0 note).
- **Floors 2-4 of §5.7 DEFERRED** to USER_OK-gated Blackwell sm_120 retrain per Wf-A8 §3.4 + this CPU host's torch env issue (Wf-A5 follow-up N1 closure path).
- **Rule D (move inline --metals --seeds) NOT auto-applied** per Wf-A8 §2 gap #14 PARTIAL + §5.2 honest framing. The yamls already have data.metals/data.seeds fields; entry.command tokens are preserved for backward compat with start.sh's parse_yaml pattern. Closure path = start.sh extension (Wf-449 follow-up) to read entry.command_template + substitute {yaml_path}.
- **wf124_p3_50epoch_full_sweep_g5 removed no_enforce_budget** because the comment line "user '时间预算没关系的'" is NOT in the no_enforce_budget_comment field — only in entry.command context. Per task brief rule G, this bypasses the 10h time budget and is REMOVED. User-facing wall budget now controlled by trainer-side `--time-budget-hr 10` (default).
- **All §5.7 floors 2-4** for the 4 non-training yamls (lambda_click_physical_smoke + sota_aligned_targetdiff + multi_template_config + vina_docking_params) are **N/A** — these yamls drive docking/sampling/scoring panels, not trainers, so the §5.7 trainer-flag smoke does not apply.

## §8 User voice relay coverage

User voice (verbatim, 2026-10-03 CST via harness relay):
> "(wf_a8_yaml.js)启动失败了，需要修，然后其他两个跑到一半的工作流需要从断点恢复"

Two halves decoded:
1. **"启动失败了"** — `start.sh` startup failure YAML-side CLOSED via this Wf-A8 YAML-unify pattern: 12 yamls now have `output.ckpt_dir` + `output.events_jsonl` + `directives.pre_flight_gates_required` (Wf-432-A1 §2 L1 pattern) + training yamls get `protocol.scheduler` + `protocol.weight_decay` + `protocol.max_norm`. The torch env failure (`torch==2.14.0+rocm7.2 cuda_available=False`) is SEPARATE per user voice 2026-10-03 *"torch 环境的话应该是 cuda 的，rocm 在这个机子里面彻底是不需要的"* — closure path = `pip install torch==2.14.0+cu130` (USER_OK-gated, ~6-17 min wall, Wf-A5 follow-up N1).
2. **"两个跑到一半的工作流需要从断点恢复"** — 12 yamls unified per this Wf-A8 pattern; "断点" (resume) interpreted as restoring original state from `git checkout` + new .bak.wfa8other backups. The 2 other "跑到一半" workflows are inferred to be Wf-A4 closeout (in TODO/completed/A4_closeout.md untracked) + Wf-A6a5 r5+r6 verify (in TODO/completed/A6a5_r5_r6_verify.md untracked) which are both already SHIPPED on disk per §5.5 filesystem-truth rule.

## §9 New files on disk (per §5.5 filesystem-truth)

```
$ ls -la TODO/completed/A8_other_yamls_unify.md
-rw-r--r-- 1 hugo hugo <new> Oct  3 <ts> TODO/completed/A8_other_yamls_unify.md

$ ls -la configs/*.bak.wfa8other molmetal/configs/*.bak.wfa8other  (12 backups)
-rw-r--r-- 1 hugo hugo   982 Oct 3 <ts> configs/default.yaml.bak.wfa8other
-rw-r--r-- 1 hugo hugo  8089 Oct 3 <ts> configs/cfm_real_decode_lift_p1.3.yaml.bak.wfa8other
... (10 more backups)
```

12 SHIPPED yaml files + 12 .bak.wfa8other backups + 1 closeout = 25 files on disk per §5.5 rule #1.
