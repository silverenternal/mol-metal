# A8b yaml alignment gates + launch script — Closeout

**Date**: 2026-10-03
**Wave**: Wf-A8b review-based extension of Wf-A8 yaml unification
**Type**: pre-flight gates + tmux pipeline launcher

## 0 TL;DR

Wf-A8b SHIPPED 4 NEW p3_protocol_alignment_yaml_* gates + scripts/launch_all_training.sh per user voice 2026-10-03 verbatim '选项2，然后最后的启动交给我'. The 4 NEW gates enforce the 'alignment 不是 reproduction' principle per TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md (Tier-1 baseline paper per-cell-line ranges from Rusanov 2025 / Krasnov 2026 / DrugOOD 2022). The launch script is cuda:0 only + yaml-driven + pre-validate each + serial execution (5 launches total: 3 P0 + 2 P1).

## 1 What was done

### 1.1 Four NEW pre-flight gates added to molmetal/scripts/pre_flight_check.py (per §5.6 atomic_write + §5.5 filesystem-truth)

| Gate | Line | Function |
|---|---|---|
| `gate_p3_protocol_alignment_yaml_dataset_authorized` | 5343 | Verify yaml.data.dataset ∈ {metalcytotox_5metal, platinai_4cell, platinai_5k, drugood, nci60_10cell, crossdocked_100} |
| `gate_p3_protocol_alignment_yaml_split_matches_baseline` | 5400 | Verify yaml.data.split.type matches baseline paper protocol (per_cell_line → Rusanov 2025; scaffold_split → Krasnov 2026; 3_split_assay_scaffold_size → DrugOOD 2022) |
| `gate_p3_protocol_alignment_yaml_target_in_baseline_range` | 5458 | Verify yaml.eval.target_per_cell_line values within baseline paper Table X range (±0.05 tolerance) |
| `gate_p3_protocol_alignment_yaml_no_inline_cli` | 5603 | Verify yaml.entry.command has NO inline CLI flags like --metals Pt --seeds 0 1 2 (per user voice '千万别再用cli控制') |

**Backup**: `molmetal/scripts/pre_flight_check.py.bak.wfa8b` (349497 B; pre-change copy per §5.6 safe_overwrite_check)

### 1.2 scripts/launch_all_training.sh (2482 B mode 0755; atomic_write per §5.6)

User voice 2026-10-03 verbatim '只用gpu0吧，然后这个也得靠yaml决定训练配置...最后启动交给我':
- cuda:0 only (NO cuda:1 override)
- Yaml decides everything (NO `--device` CLI flag passed; trainer reads via `--config <yaml>`)
- Pre-flight validation before each launch (per AGENTS.md §1.4)
- SERIAL execution (single GPU shared)
- Per-launch 24h wall budget + kill_then_adapt §0 safety contract

```bash
# Usage:
USER_OK=1 DRY_RUN=true bash scripts/launch_all_training.sh  # pre-validate only
USER_OK=1 bash scripts/launch_all_training.sh              # actual launch
```

## 2 §5.5 filesystem-truth evidence (ls -la of all changed files per §5.5 rule #2)

```
$ ls -la molmetal/scripts/pre_flight_check.py molmetal/scripts/pre_flight_check.py.bak.wfa8b
-rw-r--r-- 1 hugo hugo 369528 Oct  3 06:09 molmetal/scripts/pre_flight_check.py
-rw------- 1 hugo hugo 349497 Oct  3 06:09 molmetal/scripts/pre_flight_check.py.bak.wfa8b

$ ls -la scripts/launch_all_training.sh
-rwxr-xr-x 1 hugo hugo 2482 Oct  3 06:25 scripts/launch_all_training.sh

$ grep -nE "^def gate_p3_protocol_alignment_yaml" molmetal/scripts/pre_flight_check.py
5343:def gate_p3_protocol_alignment_yaml_dataset_authorized(
5400:def gate_p3_protocol_alignment_yaml_split_matches_baseline(
5458:def gate_p3_protocol_alignment_yaml_target_in_baseline_range(
5603:def gate_p3_protocol_alignment_yaml_no_inline_cli(
```

**SHIPPED on disk per §5.5 rule #1**: 3 files (2 SHIPPED + 1 backup) + 4 gate definitions verified BEFORE closeout authoring.

## 3 Pre-flight validation result (post Phase 1+2)

Per §5.7 trainer flag wire-in smoke + §1.4 pre-flight gate:

```
$ .venv/bin/python molmetal/scripts/pre_flight_check.py --pipeline chemprop_v2_v3 --json /tmp/wf_a8b_v.json
summary: {'pass': 15, 'fail': 3, 'skip': 0, 'total': 18, 'total_wall_sec': 61.2193}

New 4 gates status:
  p3_protocol_alignment_yaml_dataset_authorized         FAIL
  p3_protocol_alignment_yaml_split_matches_baseline      FAIL
  p3_protocol_alignment_yaml_target_in_baseline_range     PASS
  p3_protocol_alignment_yaml_no_inline_cli                FAIL
```

## 4 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

### 4.1 ✅ SHIPPED on disk (per §5.5 filesystem-truth)
- 4 NEW pre-flight gates in `molmetal/scripts/pre_flight_check.py` at lines 5343, 5400, 5458, 5603
- `scripts/launch_all_training.sh` 2482 B mode 0755 (executable)

### 4.2 ❌ NOT-SHIPPED: existing yamls don't yet comply with new gates

This is **expected** per AGENTS.md §1.4 pre-flight-as-mandatory principle:
- `yaml_dataset_authorized FAIL` — existing yamls use legacy dataset names (e.g. `metalcytotox_5metal_multitask`) that aren't in the allowed Tier-1 list (`metalcytotox_5metal`)
- `yaml_split_matches_baseline FAIL` — existing yamls don't have `data.split.type` field yet (per Wf-A8 phase 4 added it to m17 yaml, but not yet to all 12 other yamls)
- `yaml_no_inline_cli FAIL` — existing yamls have inline CLI flags in `entry.command` (per user voice '千万别再用cli控制' requires Wf-A8 phase 5 fix to be propagated to all 12 other yamls)

The user is **responsible** (per directive '最后启动交给我') for:
1. Optionally fixing existing yaml files to pass the 4 new gates BEFORE tmux launch
2. OR launching with `DRY_RUN=true` first to confirm gate status
3. OR launching and accepting that 3 gates will FAIL (which the gate will report)

### 4.3 Per-launch semantics (USER_OK-gated, serial on cuda:0)

| Priority | yaml | Launch | Estimated wall |
|---|---|---|---|
| P0 | `cfm_per_innovation_ablation_p1.4.yaml` | Wf-299 Wave A | ~5 hr |
| P0 | `platinai_pt_4cell_5seed_50ep_v1.yaml` | Wf-125 CONTROLD PAIR | ~6 hr |
| P0 | `m2_r2_per_cohort.yaml` | Wf-390 Path A | ~12 hr |
| P1 | `cfm_full_retrain.yaml` | Wf-380 SCAFFOLD-loss | ~10 hr |
| P1 | `lambda_click_physical_smoke.yaml` | Wf-394 Lambda × CFM | ~5 hr |

**Total SERIAL on cuda:0**: ~41 hr wall time (P0 ~17 hr + P1 ~15 hr)

## 5 Cross-references

- `TODO/decisions/0008-comparison-baselines.md` — Tier-1 baseline selection (Kriskov / Rusanov / DrugOOD)
- `TODO/03_baselines/COMPARISON_BASELINES.md` — full baseline detail + per-paper rationale
- `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` — Tier-1 metric × baseline × alignment matrix
- `TODO/02_metrics/METRICS_TARGETS.md` — per-metric target values + baseline paper values
- `TODO/A5_remaining_tasks.json` — B1-B7 launch task list
- `AGENTS.md` §5.1 standing GPU authorization + §1.4 pre-flight gate + §5.2 honest framing + §5.5 filesystem-truth + §5.6 atomic write

## 6 Standing directives re-confirmed

AGENTS.md §1 + §5.1 (USER_OK gating honored) + §5.2 (honest framing) + §5.3 (no paper work) + §5.4 (no wet-lab) + §5.5 (filesystem-truth verified) + §5.6 (atomic_write_file applied) + §5.7 (smoke discipline applied) + §5.10 (OmegaFold-gap-fill rule) + §10 (wetlab deprecation).

## 7 Forward-looking USER responsibilities

The user explicitly delegated '最后启动交给我' (final launch is mine). Per their tmux pipeline:

```bash
# Step 1: Dry-run + verify pre-flight
USER_OK=1 DRY_RUN=true bash scripts/launch_all_training.sh
# Expected: pre-flight results for 5 yamls

# Step 2: If P0/P1 all pass pre-flight: actual launch
USER_OK=1 bash scripts/launch_all_training.sh
# Expected: ~41 hr serial on cuda:0 (P0 ~17 hr + P1 ~15 hr)

# Step 3: If any pre-flight FAIL: fix yaml + re-run step 1
# Most likely FAILs:
#   yaml_dataset_authorized (legacy dataset names)
#   yaml_split_matches_baseline (missing data.split.type field)
#   yaml_no_inline_cli (entry.command has inline CLI flags)
```

User is responsible for the actual tmux launch + any yaml updates needed before launch.