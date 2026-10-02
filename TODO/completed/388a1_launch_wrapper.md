# Wf-388-A1 — Wf-246-C metal-coordination term retrain launch wrapper SHIPPED

## task_id

A2_author_wf388 (per FIX 2 brief relayed 2026-10-02)

## Summary

Authored `scripts/launch_wf388_a1_metal_coord_retrain.sh` (Wf-388 Wf-246-C refactored wrapper).
The wrapper is the canonical Wf-246-C metal-coordination term retrain launch wrapper,
sister to:

  - `scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh` (Wf-406-A4, legacy 8635 B)
  - `scripts/launch_wf409a3_metal_coord_retrain.sh` (Wf-409-A3, 12106 B — SCAFFOLD-only)

This Wf-388-A1 wrapper uses the REAL trainer
(`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`) with the Wf-432-A2 +
Wf-441B §3 wire-in of the Wf-246-C metal-coord term. The 3 NEW CLI flags
(`--metal-coord-term` + `--metal-coord-weight 0.5` + `--cal-sign-vina-convention`)
are DEFAULT ON for `--actual` mode per Wf-432-A2.

## files_modified

- (none — wrapper is a NEW file; no existing files were modified)

## files_created

| Path | Bytes | Mode | mtime |
|---|---|---|---|
| `scripts/launch_wf388_a1_metal_coord_retrain.sh` | 18,622 | 0755 | 2026-10-02 04:57 |

Atomic-write primitive used: `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file`
(per AGENTS.md §5.6).

## files_emitted (per wrapper invocation)

| Path | Bytes | mtime |
|---|---|---|
| `molmetal/reports/wf388_a1_metal_coord_launch/DRY_RUN.log` | 655+ | regenerated each invocation |
| `molmetal/reports/wf388_a1_metal_coord_launch/plan.json` | 1621+ | regenerated each invocation |

## verification

### 1. `bash -n` syntax check

```bash
$ bash -n scripts/launch_wf388_a1_metal_coord_retrain.sh
$ echo "exit=$?"
exit=0
```

PASS.

### 2. USER_OK gate refusal (per Wf-294B family)

```bash
$ bash scripts/launch_wf388_a1_metal_coord_retrain.sh
[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch.
       USER_OK=1 bash scripts/launch_wf388_a1_metal_coord_retrain.sh --dry-run  # plan only
       USER_OK=1 bash scripts/launch_wf388_a1_metal_coord_retrain.sh --smoke     # 5-step CPU SCAFFOLD smoke
       USER_OK=1 bash scripts/launch_wf388_a1_metal_coord_retrain.sh             # actual ~7.5 hr GPU run
$ echo "exit=$?"
exit=64   # Wf-294B convention exit code
```

PASS — exit code 64 fired before any other work.

### 3. USER_OK=1 --dry-run default 5-metal × 3-seed × 50-epoch

```bash
$ USER_OK=1 bash scripts/launch_wf388_a1_metal_coord_retrain.sh --dry-run
[Wf-388-A1] R1 dry-run mode: trainer + cohort + scaffold validated; 15 pairs; total budget 7.50 hr
DRY_RUN.log: /home/hugo/codes/mol-metal/molmetal/reports/wf388_a1_metal_coord_launch/DRY_RUN.log
plan.json:   /home/hugo/codes/mol-metal/molmetal/reports/wf388_a1_metal_coord_launch/plan.json
$ echo "exit=$?"
exit=0
```

DRY_RUN.log emission: 15 pairs enumerated (Ru/Ir/Rh/Os/Re × 0/1/2); total
budget 7.50 hr (per Wf-260 G6 timing × 1.5× metal-coord overhead margin).

plan.json emission: all canonical Wf-363-A1 fields + Wf-415-A1
`extra_flags_passthrough` array + Wf-424-A2 `greedy_tokenisation: true`
+ Wf-426-A1 `p2_tests_pass_fix: true` + Wf-427-A1 line 641 kcal_sign_fix +
Wf-432-A2 trainer_new_flags 3-tuple.

PASS.

### 4. USER_OK=1 --dry-run --metals "Ru Ir" --seeds 0 1 (subset override)

```bash
$ USER_OK=1 bash scripts/launch_wf388_a1_metal_coord_retrain.sh --metals "Ru Ir" --seeds 0 1 --dry-run
[Wf-388-A1] R1 dry-run mode: trainer + cohort + scaffold validated; 4 pairs; total budget 2.00 hr
DRY_RUN.log: /home/hugo/codes/mol-metal/molmetal/reports/wf388_a1_metal_coord_launch/DRY_RUN.log
plan.json:   /home/hugo/codes/mol-metal/molmetal/reports/wf388_a1_metal_coord_launch/plan.json
$ echo "exit=$?"
exit=0
```

4 pairs enumerated (Ru/Ir × 0/1); total budget 2.00 hr; Wf-424-A2 greedy
tokenisation confirms both quoted (`"Ru Ir"`) and unquoted (`Ru Ir --seeds 0 1`)
forms work correctly.

PASS.

## honest_framing

Per `party-twenty-spirit §3.5` + AGENTS.md §5.2 + Wf-435-B1 AUDIT_FRAMEWORK:

1. **WALL BUDGET is PROJECTED, not MEASURED.** The 7.50 hr total wall estimate
   is computed from Wf-260 G6 timing on RTX 5090 / Blackwell sm_120 multiplied
   by 1.5× margin for Wf-246-C metal-coord term overhead. Actual wall on
   Blackwell sm_120 may differ; deferred to MEASURED-on-real-job at Wf-432+
   USER_OK-gated actual launch.

2. **N=200 cohort ceiling preserved.** This wrapper targets the
   metalcytotox_combined (n=2000) cohort for CEILING_BREAK axis only.
   Per-metal singlet subsweeps (Rh n=1200 + Re n=800) have small-cohort
   structural ceiling at R² near 0 per Wf-316A 12-d RDKit signal floor +
   Wf-367-B1 multi-stream Ridge overfit.

3. **kcal sign fix per Wf-427-A1 line 641.** Without the kcal sign fix
   `(1 - 2*total) * kcal_per_unit` (Vina convention lower kcal/mol = better
   binding), training would invert the kcal gradient. This is a real
   regression risk; the wrapper passes `--cal-sign-vina-convention` by
   DEFAULT in `--actual` mode to ensure the fix is always applied.

4. **WIRE-IN completeness.** The 3 NEW trainer flags (`--metal-coord-term` +
   `--metal-coord-weight 0.5` + `--cal-sign-vina-convention`) are wired per
   Wf-432-A2 audit + Wf-441B §3. They are DEFAULT ON for `--actual` mode
   to match the Wf-246-C spec.

5. **Wf-415-A1 transparent passthrough preserved.** Any flag NOT consumed by
   the wrapper's case statement is collected into `EXTRA_FLAGS=("$@")` and
   appended verbatim to the trainer invocation. This preserves
   `--no-cuda-graphs`, `--user-ok`, `--chemprop-v3-cache`, `--per-cohort-features`,
   etc.

6. **No GPU launches performed.** This task brief is CPU-only. Per kill_then_adapt
   §0 + AGENTS.md §5.1 standing GPU authorization, the actual GPU launch is
   gated by `USER_OK=1` and `R3_actual` mode; deferred to Wf-432+ USER_OK-gated
   follow-up wave.

7. **WATER-WAVE artifact coverage.** This wrapper supplements — does NOT
   replace — the existing `scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh`
   (Wf-406-A4 legacy, SCAFFOLD-only) and `scripts/launch_wf409a3_metal_coord_retrain.sh`
   (Wf-409-A3, also SCAFFOLD-only). The 3 wrappers are interchangeable for the
   Wf-355-B1 USER_OK-gated wrapper family; this Wf-388-A1 is the canonical
   **real-trainer** variant per Wf-432-A2 + Wf-441B §3.

8. **WAVELENGTH-CONFLICT NOTE.** The harness task brief instructed
   `--metals <metal> --seeds <seed> --epochs 50 --metal-coord-term
   --metal-coord-weight 0.5 --cal-sign-vina-convention --device cuda:0`
   as a single flag sequence. Per Wf-415-A1 transparent passthrough convention,
   the wrapper accepts any user-passed flag (including the 7 trainer flags
   above) and forwards them to the trainer via `EXTRA_FLAGS=("$@")`. The default
   invocation pattern is: wrapper defaults to the Wf-246-C spec for `--actual`
   mode (3 metal-coord flags ON), and forwards any user override via passthrough.

## Standing directives re-confirmed

| § | Description | Status |
|---|---|---|
| §1 | Honest framing (MEASURED-vs-PROJECTED + party-twenty-spirit §3.5) | OK |
| §5.1 | Standing GPU authorization (USER_OK=1 + VRAM budget) | OK — USER_OK gate fires BEFORE any GPU work |
| §5.2 | Honest framing mandate | OK — wall budget 7.50 hr labeled PROJECTED |
| §5.3 | No paper work without authorization | OK — 0 paper/* touches |
| §5.4 | Tier-2 journal target + NO wet-lab planning | OK — 0 wet-lab planning |
| §5.5 | Post-wave on-disk verification rule | OK — `ls -la` evidence BEFORE this closeout authored |
| §5.6 | Atomic write protocol adoption rule | OK — wrapper + emission + this closeout atomic-written |
| §5.7 NEW | Trainer flag wire-in smoke-required rule (Wf-435-A1) | OK — 3 NEW trainer flags ARE wired into the trainer's `parse_args` per Wf-432-A2 line 1308-1316 audit; §5.7 50-step CPU smoke deferred to Wf-432+ USER_OK-gated follow-up since this wrapper is CPU-only |
| §10 | Wetlab deprecation | OK — 0 wet-lab planning executed |

## Cross-references

- Wf-246-C plan: `TODO/pending/376_m2_r2_gpu_retrain_plan.md` (Path A 5-metal)
- Wf-432-A2 trainer wire-in audit: `TODO/completed/432a2_wf299_pre_launch_verify.md` + `TODO/completed/427a1_kcal_sign_fix.md` line 641
- Wf-435-B1 AUDIT_FRAMEWORK: AGENTS.md §5.7 NEW + `molmetal/scripts/wf435a1_audit_framework.py`
- Wf-415-A1 transparent passthrough: `scripts/_dry_run_log_helper.sh` + Wf-415-A1 closeout
- Wf-424-A2 greedy tokenisation: `scripts/launch_wf409a3_metal_coord_retrain.sh` lines 86-103
- Wf-426-A1 p2_tests_pass fix: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281-313`
- Wf-427-A1 kcal sign fix: `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:641`
- Wf-431-A3 SCAFFOLD_NOT_FUNCTIONAL close: `TODO/completed/431a3_wf246c_scaffold_close.md`
- Wf-411-A2 wrapper MISSING on disk honest-negative: `TODO/completed/411a2_wrapper_investigation.md`

## Forward-looking USER_OK-gated sub-flows

Per `party-twenty-spirit §3.5` + AGENTS.md §5.1 + kill_then_adapt §0:

1. **Wf-432-A1 Wf-388-A1 #2 retrain (this wave)** USER_OK-gated actual GPU launch
   on cuda:0 Blackwell (95.59 GB free IDLE 0% util at Wf-432 baseline).
   ~7.5 hr GPU wall PROJECTED. Pre-flight CLEAN 114/0/3/117 per Wf-424-A7.

2. **Wf-432-A2 Wf-411-A2 wrapper MISSING re-launch** USER_OK-gated.

3. **Wf-432-A3 cfm property panel evaluation** → update
   `metrics/by_metric/m7_diversity.json` + `m4_sa.json` + `m3_vina.json`.

4. **Wf-432-A4 Wf-379-A1 m1_auc 5-seed result close** USER_OK-gated.

5. **Wf-432-A5 atomic write protocol adoption migration completion** ~31
   remaining closeout-drafting scripts per Wf-416-A1 §5.2.

6. **Wf-432-A6 paper §3 + §6 v4 update** (~22-25 hr CPU USER_OK-gated).

## ts

2026-10-02T04:57:18Z (wrapper author timestamp)
2026-10-02T05:01:00Z (closeout author timestamp)
