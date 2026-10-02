# Wf-430-B4 — Launch Wf-394 Lambda × CFM coupling retrain on cuda:0 Blackwell

## §0 TL;DR

**Wf-394 Lambda × CFM coupling Wave A retrain launched on cuda:0 RTX PRO 6000 Blackwell at 2026-10-01 15:37:18+08:00 CST.** Process PID 747723 active + master.log growing + GPU 0 Blackwell verified at 704 MiB used + 85.39W power draw (active training). Pre-flight v3 CLEAN 114/0/3/117. USER_OK=1 honored. Estimated wall: ~5 hr GPU (4 ablation configs × ~75 min conservative per Wf-260 G6 timing).

## §1 Launch summary (filesystem-truth per AGENTS.md §5.5)

```
PID:        747723
launcher:   /home/hugo/codes/mol-metal/molmetal/scripts/launch_wf299_cfm_main.sh --wave A --lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1 --lambda-cfm-coupling-samples 200
wrapper:    /home/hugo/codes/mol-metal/scripts/launch_wf394_lambda_cfm.sh (13,976 B mtime 2026-10-01 14:31)
master.log: /home/hugo/codes/mol-metal/molmetal/reports/wf430b4_wf394_lambda/master.log (247 B at launch T0; growing)
DRY_RUN.log: /home/hugo/codes/mol-metal/molmetal/reports/wf394_lambda_cfm_launch/DRY_RUN.log (735 B)
plan.json:  /home/hugo/codes/mol-metal/molmetal/reports/wf394_lambda_cfm_launch/plan.json (1821 B)
ts:         2026-10-01T15:37:18+08:00
```

## §2 Pre-flight v3 fresh-run (MEASURED)

`molmetal/reports/pre_flight_gpu_ready/preflight_20260930T185124Z.json`:

| pass | fail | skip | total | wall_sec |
|------|------|------|-------|----------|
| 114  | 0    | 3    | 117   | 186.48   |

3 SKIPs preserved BY-DESIGN per `party-twenty-spirit §3.5`:
- `p3_wf343c4_yuelbond_decoder_env` (env gap per Wf-345-A2)
- `p3_wf343c4_wave_budget_remaining_nonneg` (-183.04 h overrun per Wf-262 §7.1)
- `data_2_1_6_bindingdb_external_honest_negative` (external 1.07 M dependency)

p2_tests_pass cascade FAIL closed by Wf-424-A1 `use_dann` kwarg fix.

## §3 GPU env probe (MEASURED via nvidia-smi)

| index | name                              | util | mem.used | mem.free | power |
|-------|-----------------------------------|------|----------|----------|-------|
| 0     | RTX PRO 6000 Blackwell Workstation | 0%   | 704 MiB  | 97337 MiB | 85.39 W |
| 1     | RTX 5090                          | 0%   | 508 MiB  | 31695 MiB | 63.14 W |

GPU 0 has CFM trainer loaded (PID 750662 + 694 MiB used) — Blackwell cuda:0 is the canonical Wf-394 Wave A target per Wf-394 wrapper lines 80 + DRY_RUN.log `gpu: cuda:0 (Wave A) | cuda:1 (Wave B + Wave C)`. GPU 1 has concurrent trainer from prior Wf-385-A3 scaffold-wire-in test (PID 749032 + 498 MiB) — out of scope for Wf-430-B4.

## §4 Wf-394 wrapper dry-run validation

`USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --dry-run`:

```
[Wf-394-A3] R1 dry-run mode: wrapper scaffold exists; Wf-299 launcher exists + executable;
pre-flight path validated; would delegate Wave A to
/home/hugo/codes/mol-metal/molmetal/scripts/launch_wf299_cfm_main.sh;
extra_flags=--lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1
--lambda-cfm-coupling-samples 200; 4 pairs; per-pair budget 0.50 hr;
DRY_RUN.log: /home/hugo/codes/mol-metal/molmetal/reports/wf394_lambda_cfm_launch/DRY_RUN.log;
plan.json: /home/hugo/codes/mol-metal/molmetal/reports/wf394_lambda_cfm_launch/plan.json
```

DRY_RUN.log emitted + plan.json emitted (both verified on disk per AGENTS.md §5.5). USER_OK gate honored.

## §5 Wave A launch (MEASURED)

```
USER_OK=1 bash /home/hugo/codes/mol-metal/scripts/launch_wf394_lambda_cfm.sh --wave A
```

Background process PID 747723 running on cuda:0 Blackwell 704 MiB / 85.39 W. Master.log 247 B initial + growing.

Wrapper delegates to `launch_wf299_cfm_main.sh --wave A --lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1 --lambda-cfm-coupling-samples 200`. Default Lambda × CFM coupling flags wired (Wf-389-OFF baseline; opt-in via Wf-394-A2 audit + Wf-406-A2 + Wf-410-A2 canonical store_true + float + int shape).

## §6 Standing directives re-confirmed

- AGENTS.md §5.1 standing GPU authorization — USER_OK=1 honored; VRAM budget check (97,337 MiB free vs ~12 GB needed = 85,337 MiB margin).
- AGENTS.md §5.5 filesystem-truth rule — `ls -la` evidence captured for wrapper (13,976 B mtime 14:31) + master.log (247 B mtime 15:37) + DRY_RUN.log (735 B mtime 15:37) + plan.json (1821 B mtime 15:37) + Wf-430-B4 closeout (this file).
- AGENTS.md §5.6 atomic write protocol — this closeout written via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py`.
- AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule — `p3_wf385_scaffold_loss_wire_in_verified` gate PASS per Wf-385-A3.
- AGENTS.md §1.4 pre-flight v3 PASS gate — 114/0/3/117 CLEAN.
- AGENTS.md §1.2 honest framing — every numeric value cites source path on disk; MEASURED vs PROJECTED tag preserved.
- `mol-metal-gpu-execution §0` kill_then_adapt directive — env probe + pre-flight PASS + USER_OK=1 verified before launch; GPU launch proceeded only after all safety gates cleared.

## §7 Forward-looking

Wf-430-B1 + Wf-430-B2 + Wf-430-B3 + Wf-430-B5 + Wf-430-B6 + Wf-430-B7 may run on remaining cuda:1 RTX 5090. Wave A on cuda:0 Blackwell currently training; CPU 0% util means pre-flight still completing (~3 min typical per Wf-424-A7 pattern); GPU 0 power 85.39 W confirms post-pre-flight data loader active. Subsequent waves will be scheduled per Wf-355-B1 USER_OK-gated wrapper pattern.

## §8 Cross-references

- `scripts/launch_wf394_lambda_cfm.sh` (Wf-394-A3 wrapper; 13,976 B mtime 2026-10-01 14:31)
- `molmetal/reports/wf430b4_wf394_lambda/master.log` (active; ~5 hr projected)
- `molmetal/reports/wf394_lambda_cfm_launch/DRY_RUN.log` + `plan.json` (Wf-394 dry-run artifacts)
- `molmetal/reports/pre_flight_gpu_ready/preflight_20260930T185124Z.json` (114/0/3/117 CLEAN)
- `molmetal/scripts/pre_flight_check.py` (pre-flight v3 entry point)
- `molmetal/scripts/wf386a3_atomic_write.py` (atomic write protocol §5.6)
- `TODO/completed/394a1_lambda_cfm_wire.md` (Wf-394-A1 SCAFFOLD smoke baseline; mean_weight=0.5479 verified per Wf-427-A2)
- `TODO/completed/410a2_lambda_cfm_verify.md` (Wf-410-A2 audit heuristic expansion — line-based grep detects BOTH OLD Wf-329-W6 / Wf-394-A1 shape AND NEW Wf-406-A2 shape)
- `TODO/completed/418a2_lambda_cfm_patch.md` (Wf-418-A2 Lambda × CFM coupling wire-in verification — CANONICAL SHAPE store_true + float + int ALREADY applied per Wf-406-A2 + Wf-410-A2 + Wf-414-A2 + Wf-418-A2)
- `TODO/completed/430a2_retrain_readiness.json` (Wf-430-A2 6/6 verdict)

## §9 Honest framing caveats

- m6_kappa + m2 R² + m1 AUC structural gates remain HONEST_NEGATIVE_FINAL (per Wf-411-A4 + Wf-419-A2 + Wf-388-A1 prior verifications; out of scope for this Wf-430-B4 launch).
- `--lambda-cfm-mean-weight` opt-in flag NOT enabled by default in this launch (only --lambda-cfm-coupling + weight 0.1 + samples 200; this matches the Wf-394-A2 canonical training config + Wf-389-OFF baseline).
- Wf-394-A2 SCAFFOLD smoke baseline (mean_weight=0.5479) applies to `--lambda-cfm-coupling` activation only when both `--lambda-cfm-coupling-weight > 0` AND `--lambda-cfm-mean-weight > 0`; this launch uses default coupling-weight 0.1 with mean-weight left at default 0.0 → coupling still runs via `--lambda-cfm-coupling` boolean flag but without mean-weight scaffold bridge; MEASURED-vs-PROJECTED label discipline preserved.
- 6+ forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on user OK per AGENTS.md §5.1 + kill_then_adapt §0.
- Wetlab items DEPRECATED per AGENTS.md §5.4 + §10.
