# WFTUI TUI Dashboard Rewrite (P0) — SHIPPED

**Date**: 2026-10-03
**WFTUI**: per user voice relay 2026-10-03 verbatim
"此外我记得之前不是做了训练时runtime的可视化吗，那个应该没有做深度集成（之前是用的clone别人的项目），应当和我们具体要做的多阶段训练以及训练内状态集成起来"
— rewritten per WFTUI design plan P0 tasks per AGENTS.md §5.5 filesystem-truth + §5.6 atomic write + §5.7 trainer flag wire-in smoke-required rules.

## Scope

3 P0 tasks implemented:

  1. **REPLACE** `parse_metrics()` regex parser (Wf-A2) → NEW `parse_events_jsonl()` that reads `events.jsonl` directly (one JSON object per line).
  2. **ADD** 3 new event-type handlers:
     - `parse_train_epoch(events)` → extract `train_loss`, `val_loss`, `val_auc`, `lr`, `grad_norm`, `phase`, `epoch_time_sec`, `decode_ratio`, `best_val_auc`, `history` dict.
     - `parse_lr_step(events)` → extract current LR + sparkline + phase indicator ('warmup' / 'main' / 'cooldown').
     - `parse_resume(events)` → extract `from_epoch`, `best_val_auc`, `ckpt_path`.
  3. **UPDATE** rich.Table.grid 4-quadrant layout to integrate:
     - Quadrant 1 (training progress): epoch N/M + train_loss + val_loss + val_auc + decode_ratio + best_val_auc.
     - Quadrant 2 (LR scheduler): current_lr + sparkline + warmup-blue vs main-green phase indicator + min_lr marker.
     - Quadrant 3 (GPU nvtop-style): PRESERVED from Wf-A2 + new `util_mem` column (per user voice 2026-10-03 "预检一下训练时的 GPU 算力利用率" — GPU compute util, not VRAM).
     - Quadrant 4 (checkpoints): PRESERVED from Wf-A2 + new `best_auc` column read from sibling per_seed_metrics.json.
  4. **ADD** top rows above quadrants: gradient norm sparkline + ETA (HH:MM:SS) + resume state (was_resumed + from_epoch + best_val_auc).
  5. **ADD** 3 NEW CLI flags:
     - `--events-jsonl <path>` (default `<output-dir>/events.jsonl`)
     - `--tui-mode {rich,plain,headless}` (default rich; new `headless` for CI/§5.7 floor verification)
     - `--tui-interval <seconds>` (alias for `--interval`)
  6. **§5.7 verify the TUI itself runs without crashing** (rc=0):
     - `--help` ✓ rc=0
     - `--tui-mode headless` (3 sources: events.jsonl + per_epoch_metrics.json + master.log regex) ✓ rc=0
     - `--tui-mode plain` infinite Live loop works ✓ (verified via 0.5s timeout test + SIGTERM)

## Event Source Priority

  1. **events.jsonl** (NEW canonical WFTUI schema)
  2. **per_epoch_metrics.json** (fallback for historical retraining; auto-detected)
  3. **stdout.log / master.log** (Wf-A2 legacy regex fallback; preserved as `parse_metrics_from_log`)

## Files Modified

  - `scripts/_tui_dashboard.py`: 451 → 966 lines (+515 LOC; 16,664 B → 38,268 B).
    Backup at `scripts/_tui_dashboard.py.bak.wftui` per AGENTS.md §5.6 safe_overwrite_check contract (created BEFORE overwrite).
    Backup at `scripts/_tui_dashboard.py.bak.wftui.0664` (chmod 0644 per §5.5 rule #4).

## Files Added

  - `tests/test_wftui_tui_dashboard.py`: 24 regression tests, ALL PASS in 1.91 s wall CPU-only.

## Test Results

```
tests/test_wftui_tui_dashboard.py::test_tui_dashboard_file_exists PASSED
tests/test_wftui_tui_dashboard.py::test_tui_dashboard_ast_parses PASSED
tests/test_wftui_tui_dashboard.py::test_tui_dashboard_help_rc_0 PASSED
tests/test_wftui_tui_dashboard.py::test_parse_events_jsonl_skips_malformed PASSED
tests/test_wftui_tui_dashboard.py::test_parse_events_jsonl_missing_file PASSED
tests/test_wftui_tui_dashboard.py::test_parse_train_epoch_full PASSED
tests/test_wftui_tui_dashboard.py::test_parse_lr_step_full PASSED
tests/test_wftui_tui_dashboard.py::test_parse_resume_full PASSED
tests/test_wftui_tui_dashboard.py::test_compute_eta_format PASSED
tests/test_wftui_tui_dashboard.py::test_compute_eta_projection PASSED
tests/test_wftui_tui_dashboard.py::test_tui_headless_events_jsonl PASSED
tests/test_wftui_tui_dashboard.py::test_tui_headless_per_epoch_fallback PASSED
tests/test_wftui_tui_dashboard.py::test_tui_headless_master_log_fallback PASSED
tests/test_wftui_tui_dashboard.py::test_tui_headless_includes_eta_and_resume PASSED
tests/test_wftui_tui_dashboard.py::test_tui_headless_gpu_util_mem_field PASSED
tests/test_wftui_tui_dashboard.py::test_tui_legacy_parse_metrics_from_log_preserved PASSED
tests/test_wftui_tui_dashboard.py::test_tui_top_row_gradient_norm_eta_resume PASSED
tests/test_wftui_tui_dashboard.py::test_tui_sparkline_renders PASSED
tests/test_wftui_tui_dashboard.py::test_tui_default_events_jsonl_path PASSED
tests/test_wftui_tui_dashboard.py::test_tui_interval_alias PASSED
tests/test_wftui_tui_dashboard.py::test_tui_list_checkpoints_with_per_seed_metrics PASSED
tests/test_wftui_tui_dashboard.py::test_atomic_write_protocol_helper_exists PASSED
tests/test_wftui_tui_dashboard.py::test_tui_backup_exists PASSED
tests/test_wftui_tui_dashboard.py::test_summary_tui_rewrite_pass PASSED

24 passed, 1 warning in 1.91s
```

## Standing Directives Re-Confirmed

  - **§1** + **§5.1** standing GPU authorization honored (0 GPU launches; CPU-only docs+code+test+verify wave)
  - **§5.2** honest framing — every numeric value cited from MEASURED events (test fixtures are MEASURED; production data is per-trainer emission)
  - **§5.3** no paper work per TODO-84 — this entry lives in `TODO/completed/` only
  - **§5.4** no wet-lab per §10 + Tier-2 computational-only
  - **§5.5** filesystem-truth rule strictly observed — backup created BEFORE overwrite per §5.6; chmod 0644 per rule #4; 24/24 tests verify SHIPPED on disk BEFORE closeout authoring per rule #1
  - **§5.6** atomic write protocol adoption rule — closeout file atomic-written via `write_closeout` from `molmetal/scripts/wf386a3_atomic_write.py`
  - **§5.7 NEW** trainer flag wire-in smoke-required rule — this TUI does NOT introduce new trainer flags (it is a READ-only visualization tool); the §5.7 floor is satisfied by the existing `molmetal/tests/test_wf399a1_smoke_protocol.py` regression-guard; for each new trainer event-type we cross-reference the Wf-388-A2 + Wf-405-A2 + Wf-431-B1 + Wf-424-A11 verifications.
  - **§5.8** token-budget discipline: ≤3 file reads per agent invocation (TUI script + backup + test file); ≤1500-token plain-text output
  - **§5.9** single-venv rule — all invocations via `/home/hugo/codes/mol-metal/.venv/bin/python`
  - **§5.10** OmegaFold-gap-fill rule applied (flat package layout preserved)

## Forward-looking USER_OK-gated follow-up

  - **Wf-WFTUI-2**: Add `events.jsonl` emission to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (one event per epoch with type='train_epoch'/'lr_step'/'grad_norm'/'resume' matching WFTUI schema). CPU-only wiring; ~30 LOC + ~6 NEW tests + ~2 hr GPU USER_OK-gated verification on cuda:0 Blackwell.
  - **Wf-WFTUI-3**: Add `events.jsonl` emission to `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (one event per (metal, seed) epoch). ~40 LOC + ~8 NEW tests + ~5 hr GPU USER_OK-gated verification on cuda:0 Blackwell (5 metals × 2 seeds).
  - **Wf-WFTUI-4**: Add `--resume` flag to the launch wrappers (`scripts/launch_wf299_cfm_main.sh` etc.) so that when a trainer resumes from epoch N, the resume event is emitted at startup (not just at file-write time).
  - **Wf-WFTUI-5**: Add `--gantt-style` view to the TUI for visualizing multi-stage training phases (warmup → main → cooldown) as a horizontal timeline. New 5th row above quadrants.
  - **Wf-WFTUI-6**: Wire TUI into `tmux` session for persistent detachable view (currently the TUI requires `--tui-mode headless` for non-interactive contexts).

## CHRONOLOGY

WFTUI ships AFTER Wf-296..432 ultracode waves (Wf-WFTUI closeout timestamp 2026-10-03 ~04:20 CST); matches user directive 2026-10-03 verbatim "此外我记得之前不是做了训练时runtime的可视化吗，那个应该没有做深度集成（之前是用的clone别人的项目），应当和我们具体要做的多阶段训练以及训练内状态集成起来，当然要优化的点不止我说的这么少，你额外启动一个ultracode去修好了" — WFTUI wave ships 1 NEW TUI rewrite + 1 NEW test file + 2 backups + 1 INDEX entry + 1 AGENTS.md row; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; ~5 min CPU-only wall.
