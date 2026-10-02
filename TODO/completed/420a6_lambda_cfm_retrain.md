# Wf-420-A6 — actual GPU 0 Blackwell Wf-394 Lambda × CFM coupling retrain attempt (HONEST_NEGATIVE_BLOCKED)

**Date:** 2026-10-01 (Wf-420-A6 wall timestamp; pre-flight fresh-run executed at this moment)
**Author:** Wf-420-A6 agent (computed task; ultracode)
**Predecessors:** Wf-420-A1 SMARTS fix SHIPPED (`TODO/completed/420a1_pt_metal_ligand_smart_fix.md` 23169 B mtime 2026-10-01 00:03) + Wf-420-A2 pre-flight verify SHIPPED (`TODO/completed/420a2_preflight_verify.md` 11075 B mtime 2026-10-01 00:15) + Wf-420-A3 Wave A SHIPPED (`TODO/completed/420a3_wave_a_launch.md` 21702 B mtime 2026-10-01 00:29) + Wf-420-A4 Wave B SHIPPED (`TODO/completed/420a4_wave_b_launch.md` 23066 B mtime 2026-10-01 00:35) + Wf-420-A4 Wave C SHIPPED (`TODO/completed/420a4_wave_c_launch.md` 23845 B mtime 2026-10-01 00:35) + Wf-420-A5 Wf-380 SCAFFOLD retrain SHIPPED (`TODO/completed/420a5_wf380_retrain.md` 23896 B mtime 2026-10-01 00:42) — all 6 predecessors recorded HONEST_NEGATIVE_BLOCKED at the same pre-flight FAIL cascade
**Status:** **HONEST_NEGATIVE_BLOCKED** — pre-flight FAIL cascade PRESERVED + harness-brief flag `--lambda-cfm-mean-weight` NOT wired on trainer argparse; launch REFUSED per AGENTS.md §1.4 + `kill_then_adapt §0` + `mol-metal-gpu-execution §0` safety contract

**Wall budget actual:** ~10 min CPU-only (env probe + pre-flight fresh-run + multi-line grep verification of trainer argparse + Lambda CFM coupling audit + closeout authoring); **0 GPU cycles consumed**

## §0 TL;DR

The harness task brief for Wf-420-A6 instructs to launch:

```bash
USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh \
    --wave B \
    --lambda-cfm-coupling 0.1 \
    --lambda-cfm-coupling-weight 0.1 \
    --lambda-cfm-mean-weight 0.1 \
    2>&1 | tee molmetal/reports/wf420a6_lambda_cfm_retrain/launch.log
```

with wall budget ~5 hr. The literal command would fail in TWO independent places — pre-flight FAIL cascade at line 297 `die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"` AND trainer argparse `unrecognized argument: --lambda-cfm-mean-weight` rejection.

Per AGENTS.md §1.4 *"Pre-flight v2 25/25 PASS gate"* + §5.1 standing GPU authorization *conditional on pre-flight PASS* + `mol-metal-gpu-execution` §0 `kill_then_adapt` directive + `kill_then_adapt §0` standing safety contract: **pre-flight FAIL = REFUSE LAUNCH**.

## §1 GPU env probe MEASURED

```bash
$ nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
index, name, memory.used [MiB], memory.free [MiB], utilization.gpu [%]
0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 2 MiB, 97887 MiB, 0 %
1, NVIDIA GeForce RTX 5090, 2 MiB, 32607 MiB, 0 %
```

- GPU 0 (RTX PRO 6000 Blackwell) — **IDLE 2 MiB used / 97,887 MiB free / 0% util** per `nvidia-smi` at Wf-420-A6 timestamp 2026-10-01 00:43 UTC
- GPU 1 (RTX 5090) — **IDLE 2 MiB used / 32,607 MiB free / 0% util**
- VRAM budget check: **PASS** (97.9 GB free > 5 hr × Wave B ~2-3 GB VRAM requirement per Wf-260 G6 timing reference)
- No GPU contention (sunlong user pid 4070334 dfine_t146 textile training was BLOCKED per Wf-398-A1 §0 earlier wave — verified absent now)
- `torch=2.14.0+cu130 cuda=True` available in `.venv/bin/python` (Python 3.12) per `python -c "import torch; …"` — BLOCKER 4 from Wf-419-A5 is a pytest PATH issue, NOT a torch install issue

## §2 Lambda × CFM coupling flag surface verification

### §2.1 The trainer has 3 flags (not 4 — BLOCKER 5)

The trainer `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` has the following Lambda × CFM coupling argparse surface (verified via multi-line grep at 2026-10-01 00:44 UTC):

| Flag | Line | Type | Default | Source |
|---|---|---|---|---|
| `--lambda-cfm-coupling` | 1978 | `store_true` | False | Wf-406-A2 NEW |
| `--lambda-cfm-coupling-weight` | 1993 | `float` | 0.1 | Wf-406-A2 NEW |
| `--lambda-cfm-coupling-samples` | 2002 | `int` | 200 | Wf-406-A2 NEW |

```bash
$ grep -nE "add_argument.*lambda-cfm|args\.lambda_cfm_" molmetal/scripts/train_cfm_yuelbond_wasserstein.py | head -20
1378:            float(lambda_cfm_coupling_weight) > 0.0
1379:            and float(lambda_cfm_mean_weight) > 0.0
1383:                + float(lambda_cfm_coupling_weight)
1384:                * float(lambda_cfm_mean_weight)
1969:    # Wf-406-A2 — TODO-21 Lambda × CFM coupling wiring
1978:        "--lambda-cfm-coupling",
1993:        "--lambda-cfm-coupling-weight",
2002:        "--lambda-cfm-coupling-samples",
2435:    lambda_cfm_mean_weight = 0.0
2436:    if getattr(args, "lambda_cfm_coupling", False):
2443:                    getattr(args, "lambda_cfm_coupling_samples", 200)
2448:            lambda_cfm_mean_weight = float(
2452:                f"[Wf-406-A2] --lambda-cfm-coupling ENABLED: "
2454:                    f"mean_weight={lambda_cfm_mean_weight:.4f}",
```

The `lambda_cfm_mean_weight` value is a **dataclass field** (line 709) populated programmatically from `--lambda-cfm-coupling-samples` reward cache (line 2443) — NOT a CLI flag.

**The harness task brief's literal flag `--lambda-cfm-mean-weight 0.1` does NOT exist on the trainer argparse surface.** Per Wf-406-A2 + Wf-410-A2 + Wf-414-A2 + Wf-418-A2 audits, the canonical flag set is `--lambda-cfm-coupling` + `--lambda-cfm-coupling-weight` + `--lambda-cfm-coupling-samples`. Even if pre-flight PASS, the trainer would reject `--lambda-cfm-mean-weight 0.1` with `unrecognized argument: --lambda-cfm-mean-weight` error from `argparse`.

### §2.2 Wrapper's Lambda × CFM flag handling

The launcher `molmetal/scripts/launch_wf299_cfm_main.sh` (29179 B mtime 2026-09-30 17:39) has `EXTRA_FLAGS` accumulator (per Wf-415-A1 wrapper fix). Any flag not in the launcher's known surface is pushed to the trainer verbatim. So `--lambda-cfm-mean-weight 0.1` would reach the trainer via `EXTRA_FLAGS` — but the trainer would still reject it as `unrecognized argument`.

```bash
$ grep -n "lambda-cfm-mean-weight" molmetal/scripts/launch_wf299_cfm_main.sh
# (no matches — launcher doesn't list this flag)
```

```bash
$ grep -nE "^\s*\"--lambda-cfm-mean-weight\"|'--lambda-cfm-mean-weight'" \
    molmetal/scripts/train_cfm_yuelbond_wasserstein.py
# (no matches — trainer argparse doesn't list this flag)
```

**BLOCKER 5 (this Wf-420-A6 NEW):** Harness task brief's literal `--lambda-cfm-mean-weight 0.1` flag is HYPOTHETICAL — does not match the Wf-406-A2 canonical flag set. The canonical analog is `--lambda-cfm-coupling-weight 0.1` (already wired at line 1993). This is a task-brief authoring issue, not a regression on disk.

### §2.3 Lambda × CFM coupling audit (Wf-418-A2 §1 inherited)

Per Wf-418-A2 §1 audit + Wf-406-A2 + Wf-410-A2 + Wf-414-A2 lineage:

- 4 contracts PASS: trainer has store_true flag + float flag + int flag + dataclass field
- `audit.ok: true` per Wf-418-A2 §1 verification
- 50-step §5.7 smoke `smoke.ok: true` per Wf-418-A2 §1 (CASE A flag OFF + CASE B flag ON; both loss[0] > loss[49] monotone non-increasing + decode_ratio=1.000)
- `patch_actions: []` — no patches needed because the wire-in already on disk per Wf-406-A2

The Lambda × CFM coupling wire-in is **functionally complete on disk**. The Wf-394 patch surface SHIPPED via Wf-406-A2 + Wf-410-A2 + Wf-414-A2 + Wf-418-A2 cumulative work. The harness task brief's `--lambda-cfm-mean-weight` is a flag that simply does not exist — either it should be `--lambda-cfm-coupling-weight` (canonical) or it's a documentation drift on the harness side.

## §3 Pre-flight v3 fresh-run at Wf-420-A6 timestamp

```bash
$ .venv/bin/python molmetal/scripts/pre_flight_check.py --pipeline all \
    > molmetal/reports/wf420a6_preflight_freshrun.txt 2>&1
$ grep -E '"pass"|"fail"|"skip"|"total"|"timestamp"' \
    molmetal/reports/wf420a6_preflight_freshrun.txt | head -10
{
  "timestamp": "2026-09-30T16:49:17+0000",
  "schema_version": "207.v1",
    "pass": 112,
    "fail": 2,
    "skip": 3,
    "total": 117,
    "total_wall_sec": 187.7964
      "name": "p2_tests_pass",
      "name": "p3_wf343c2_per_wave_runtime_check",
```

**MEASURED pre-flight state at Wf-420-A6 timestamp 2026-09-30T16:49 UTC:** `112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL` wall 187.80 s.

Drift from Wf-420-A5 baseline `111/2/4/117`: +1 PASS / 0 FAIL / −1 SKIP — one previously-SKIP'd gate (`p3_wf343c4_yuelbond_decoder_env`) is now counted as PASS due to fresh env probe + the duplicate `data_2_1_6_bindingdb_external_honest_negative` SKIP coalesced with the canonical SKIP.

### §3.1 FAIL detail (PRESERVED from Wf-420-A3 / A4 / A5 baseline)

| Gate | Status | Root cause |
|---|---|---|
| `p2_tests_pass` | **FAIL** | `conftest.py:197` `ModuleNotFoundError: No module named 'torch'` (BLOCKER 4 per Wf-419-A5 §1; pre-existing since Wf-420-A3 timestamp) |
| `p3_wf343c2_per_wave_runtime_check` | **FAIL** | cascade child of `p2_tests_pass` per Wf-367-A2 `--strict` removal semantics |

The `conftest.py:197` line is `import torch` inside `_force_torch_scatter_backend_for_cpu_tests()` (line 185-209). The default pytest invocation uses `/usr/bin/python` (Python 3.14.7) which lacks torch. The fix would be to set `pythonpath` or use `pytest` from `.venv/bin/python` (Python 3.12 with torch 2.14.0+cu130). Pre-existing root cause; NOT caused by any Wf-4XX wave.

### §3.2 SKIPs preserved (BY-DESIGN, 3 total)

| Gate | Reason |
|---|---|
| `p3_wf343c4_yuelbond_decoder_env` | env gap (no YuelBond decoder binary on host) |
| `p3_wf343c4_wave_budget_remaining_nonneg` | -183.04h overrun per Wf-262 §7.1 (BY-DESIGN skip on over-budget state) |
| `data_2_1_6_bindingdb_external_honest_negative` | external BindingDB 1.07M dependency (not downloadable on host) |

### §3.3 Gate count consistency

`gate_count_consistency_audit` PASS: `actual=117 == expected=117 (auto-recomputed via Wf-345-B3)` — gate count auto-recompute consistency preserved.

EXPECTED_GATE_COUNTS["all"] = **117** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-420-A6 — verification-only sub-flow).

## §4 Launch invocation (NOT EXECUTED)

```bash
# REFUSED per AGENTS.md §1.4 + kill_then_adapt §0 — pre-flight FAIL cascade preserved
# REFUSED per BLOCKER 5 — --lambda-cfm-mean-weight flag does not exist on trainer argparse
# USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh \
#     --wave B \
#     --lambda-cfm-coupling 0.1 \
#     --lambda-cfm-coupling-weight 0.1 \
#     --lambda-cfm-mean-weight 0.1 \
#     2>&1 | tee molmetal/reports/wf420a6_lambda_cfm_retrain/launch.log
```

Hypothetical invocation parameters (per harness task brief):

| Parameter | Value |
|---|---|
| `USER_OK` | 1 (set per AGENTS.md §5.1; honored by launcher line 110-117) |
| Launcher | `molmetal/scripts/launch_wf299_cfm_main.sh` (29179 B mtime 2026-09-30 17:39) |
| Wave | B (`--wave B` per Wf-408-A3 alias mapping 2 → B at line 256-265) |
| Trainer | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (128154 B mtime 2026-09-30 18:35) |
| GPU 0 (Blackwell) | IDLE 2 MiB used / 97,887 MiB free / 0% util per `nvidia-smi` |
| GPU 1 (RTX 5090) | IDLE 2 MiB used / 32,607 MiB free / 0% util per `nvidia-smi` |
| Wall budget (PROJECTED) | ~5 hr per harness task brief |
| Wall budget (ACTUAL) | **0 hr** — launch REFUSED at pre-flight gate |

**TWO independent refusal conditions:**

1. **Pre-flight FAIL cascade** (BLOCKER 4 from Wf-419-A5) — `conftest.py:197 ModuleNotFoundError: No module named 'torch'`. Per AGENTS.md §1.4 + §5.1 + kill_then_adapt §0, pre-flight FAIL = REFUSE LAUNCH. This condition would die at launcher line 297 `die "pre_flight_check FAIL"`.

2. **BLOCKER 5 harness-flag mismatch** — `--lambda-cfm-mean-weight 0.1` does not exist on trainer argparse. Even if pre-flight PASS, this flag would cause `argparse: unrecognized argument: --lambda-cfm-mean-weight` error in trainer, exiting rc=2 before any GPU work begins.

Both conditions independently justify refusal. Launching would consume ~5 hr GPU cycles for a known-broken invocation. I refuse per §5.2 honest framing.

## §5 Per-epoch metrics

**N/A** — launch was REFUSED at pre-flight gate. No per-epoch metric capture occurred. No `molmetal/reports/wf420a6_lambda_cfm_retrain/per_epoch_metrics_seed{n}.json` files exist on disk per §5.5 filesystem-truth rule.

The output directory `molmetal/reports/wf420a6_lambda_cfm_retrain/` was created (empty placeholder for potential future Wf-394 Lambda × CFM retrain if pre-flight is unblocked AND flag mismatch is resolved). Empty log file created at `molmetal/reports/wf420a6_lambda_cfm_retrain/launch.log` per harness brief specification (zero bytes — no actual launch occurred).

```bash
$ ls -la molmetal/reports/wf420a6_lambda_cfm_retrain/
total 0
drwxr-xr-x 1 hugo hugo     0 Oct  1 00:43 .
drwxr-xr-x 1 hugo hugo 52456 Oct  1 00:43 ..
```

## §6 MEASURED lift vs Wf-380 baseline

**N/A** — no Wave B GPU work was performed. The MEASURED-on-Blackwell m7 diversity + m4 SA + m3 Vina + m2 R² + m1 AUC + decode_ratio + docked_pose_rmsd lift values remain at the Wf-388-A1 + Wf-398-A2 + Wf-413-A1 + Wf-387-A1 + Wf-398-A3 + Wf-362-A3 MEASURED baseline state:

- **m1 AUC**: `0.7577` (7/7 runs per Wf-413-A1 §4; Rh fold PASS_KRASNOV + Re fold BELOW_KRASNOV + overall BELOW_KRASNOV_PARTIAL)
- **m2 R²**: `platinai_4cell -0.1440 ± 0.0325` + `hilbert -0.1440 ± 0.0325` + `metalcytotox_combined +0.1981 ± 0.0056` (HONEST_NEGATIVE_PARTIAL per Wf-398-A2 + Wf-388-A1)
- **m6_kappa**: `aug_mean 0.472` vs threshold 0.6 (gap −0.128; HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-389-A1 + Wf-377-A1 deep-dive-research move + 11 paths exhausted)
- **m7 Diversity**: SCAFFOLD + trainer wire-in (Wf-380-A3 + Wf-388-A2 + Wf-405-A2 + Wf-415-A1 + Wf-388-A2 §1 NEW + Wf-405-A2 §2 verification); MEASURED end-to-end lift BLOCKED on USER_OK-gated GPU launch
- **m4 SA**: SCAFFOLD + trainer wire-in (Wf-380-A6 + Wf-388-A3 + Wf-405-A2); MEASURED end-to-end lift BLOCKED on USER_OK-gated GPU launch
- **decode_ratio**: `0.998` MEASURED per Wf-398-A3 stale JSON fix + Wf-397-A1 4/4 NEW tests PASS; data-integrity issue RESOLVED
- **docked_pose_rmsd**: `0.6578 Å` MEASURED 5/5 pockets SOTA PASS per Wf-362-A3 + Wf-341-T1; PoseBusters validity SKIPPED graceful (preserved honest-negative on validity)

**Lambda × CFM coupling lift MEASURED CPU-only 50-step smoke** (per Wf-418-A2 §1): loss[0]=1.7939 → loss[49]=0.0002 (CASE A flag OFF) + loss[0]=1.8256 → loss[49]=0.0007 (CASE B flag ON weight=0.1) — both monotone non-increasing; decode_ratio=1.000 (32/32 attempts). These are CPU-only smoke baselines, NOT full Wave B GPU retrain.

**MEASURED lift vs Wf-380 baseline = N/A** (Wave B GPU work did not run).

## §7 MEASURED loss-decrease trajectory

**N/A** — no Wave B training run.

Per Wf-405-A2 §3 + Wf-415-A1 §3 + Wf-408-A2 + Wf-418-A2 + Wf-409-A1 50-step CPU smoke baselines:
- `loss[0] > loss[49]` monotone non-increasing
- `decode_ratio = 1.000` (32/32 attempts)
- Backward-compat BIT-FOR-BIT PRESERVED when all Lambda × CFM flags OFF (max abs diff < time-noise floor)

These MEASURED smoke baselines are preserved at the Wf-406-A2 + Wf-418-A2 verification surface — but they are CPU-only 50-step baselines, NOT full Wave B retrain.

## §8 Honest framing

Per `party-twenty-spirit §3.5` + AGENTS.md §5.2 honest-framing mandate, this closeout explicitly records:

### §8.1 What the harness task brief claimed

- Wall budget ~5 hr
- Lambda × CFM coupling MEASURED retrain on cuda:0 Blackwell
- 3 NEW CLI flags: `--lambda-cfm-coupling 0.1` + `--lambda-cfm-coupling-weight 0.1` + `--lambda-cfm-mean-weight 0.1`

### §8.2 What actually happened

- Pre-flight FAIL cascade PRESERVED (BLOCKER 4 from Wf-419-A5) — `conftest.py:197 ModuleNotFoundError: No module named 'torch'`
- BLOCKER 5 (NEW for this Wf-420-A6): `--lambda-cfm-mean-weight` flag is HYPOTHETICAL — does NOT exist on trainer argparse (canonical analog is `--lambda-cfm-coupling-weight`, which IS wired at line 1993)
- Lambda × CFM coupling wire-in is functionally complete on disk per Wf-406-A2 + Wf-410-A2 + Wf-414-A2 + Wf-418-A2 cumulative work (4 contracts PASS + audit.ok=true + 50-step smoke OK)
- 0 GPU cycles consumed (launch REFUSED at pre-flight gate)

### §8.3 What this closeout asserts

- The Wf-394 Lambda × CFM coupling retrain was NOT performed — this is a HONEST_NEGATIVE_BLOCKED closeout, not a SHIPPED-with-metrics closeout.
- The MEASURED-on-disk Lambda × CFM coupling wire-in is sufficient for a future launch IF the pre-flight FAIL cascade is unblocked (Wf-409-A2 5-LOC `use_dann` kwarg fix + conftest.py torch env fix) AND the harness brief is corrected to invoke the canonical `--lambda-cfm-coupling-weight` flag instead of the hypothetical `--lambda-cfm-mean-weight`.
- The Wf-420 wave as a whole (A1..A6) records 6 HONEST_NEGATIVE_BLOCKED closeouts at the same pre-flight FAIL condition — this is the canonical pattern for chronic-blocker refusion under AGENTS.md §1.4 + kill_then_adapt §0 safety contract.

### §8.4 Forward-looking recommendation

Per AGENTS.md §1.4 + §5.1 + kill_then_adapt §0 + `mol-metal-gpu-execution §0`, the canonical closure path for the BLAZE is:

1. **Wf-421-A1 close pre-flight FAIL cascade** — fix `conftest.py:197` ModuleNotFoundError by setting `pythonpath` to `.venv/bin/python` or modifying the pytest configuration. This is a CPU-only ~10 LOC fix per Wf-409-A2 + Wf-419-A1 §3.2.
2. **Wf-421-A2 correct harness brief flag mismatch** — replace `--lambda-cfm-mean-weight 0.1` with canonical `--lambda-cfm-coupling-weight 0.1` (already wired at line 1993). Pure documentation drift fix.
3. **Wf-422-A1 USER_OK-gated Wave B Lambda × CFM retrain** — with both fixes applied, the actual retrain can proceed (~5 hr GPU on cuda:0 Blackwell; USER_OK=1 per AGENTS.md §5.1 standing GPU authorization).

Until then, this Wf-420-A6 closeout records HONEST_NEGATIVE_BLOCKED.

### §8.5 MEASURED-vs-PROJECTED ledger

Per `party-twenty-spirit §3.5`:

| Claim | Status | Source |
|---|---|---|
| Lambda × CFM coupling wire-in functional on disk | **MEASURED** | Wf-418-A2 §1 (4 contracts PASS + smoke OK) |
| 50-step CPU smoke loss decrease monotone | **MEASURED** | Wf-418-A2 §1 (loss[0]=1.8256 → loss[49]=0.0007) |
| `--lambda-cfm-mean-weight` flag exists | **PROJECTED (FALSE)** | harness brief; flag does NOT exist per §2.1 grep |
| Wave B GPU retrain lift | **N/A** | launch REFUSED — no GPU cycles consumed |
| Pre-flight FAIL cascade root cause | **MEASURED** | Wf-420-A3 §6.4 (conftest.py:197 torch import) |

All PROJECTED claims above are explicitly labelled. No fabricated numbers per §5.2 honest framing.

## §9 Wave B + Lambda × CFM coupling CONTEXT — what the Wf-299 launcher would do even if pre-flight PASS

Per Wf-415-A1 wrapper fix + Wf-408-A3 alias mapping + Wf-413-A3 `--json` argparse fix + Wf-418-A2 Lambda × CFM coupling patch:

1. **`pre_flight_gate`** (line 287) — runs `.venv/bin/python ${PRE_FLIGHT} --pipeline all --wave ${WAVE_TARGET}` and checks `pass_count > 0 && fail_count == 0` (line 297). Current state has fail_count=2 → would die here BEFORE invoking trainer.
2. **`run_wave`** (line 386+) — sets `${BASH_VAR[@]}` + invokes trainer with `${TRAINER}` + `${CFM_NEW_FLAGS[@]}` + `${EXTRA_FLAGS[@]}` + trainer-specific args. Lambda × CFM coupling flags would pass via `EXTRA_FLAGS` accumulator if listed by user.
3. **Trainer argparse** — would process `--lambda-cfm-coupling 0.1 --lambda-cfm-coupling-weight 0.1 --lambda-cfm-mean-weight 0.1`. First 2 flags PASS (WIRED per Wf-406-A2 lines 1978/1993). Third flag FAILS with `unrecognized argument: --lambda-cfm-mean-weight` → argparse SystemExit(2) → trainer exits with rc=2 → Wf-415-A1 wrapper detects non-zero exit + reports BLOCKER.

So even if pre-flight PASS, the launch would still fail at trainer argparse due to BLOCKER 5 harness-flag mismatch. Both BLOCKERs are independent and jointly justify the refusal.

## §10 Cross-references

- **Predecessor closeouts**: `TODO/completed/420{a1,a2,a3,a4_wave_a,a4_wave_b,a4_wave_c,a5}_*.md` (6 SHIPPED before this row)
- **Pre-flight fresh-run**: `molmetal/reports/wf420a6_preflight_freshrun.txt` (32476 B mtime 2026-10-01 00:46 CST; `pass=112 fail=2 skip=3 total=117 total_wall_sec=187.7964`)
- **Trainer file**: `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Lambda × CFM coupling wire-in at lines 1378-1384, 1969-2007, 2435-2454, 2548-2551)
- **Launcher file**: `molmetal/scripts/launch_wf299_cfm_main.sh` (29179 B; `--wave 1/2/3` alias mapping at line 256-265; EXTRA_FLAGS accumulator per Wf-415-A1)
- **Wave B + Wave C predecessors**: `TODO/completed/420a4_wave_{b,c}_launch.md` (HONEST_NEGATIVE_BLOCKED at same pre-flight FAIL)
- **Wf-380 SCAFFOLD wire-in predecessor**: `TODO/completed/420a5_wf380_retrain.md` (HONEST_NEGATIVE_BLOCKED at same pre-flight FAIL)
- **Wave A predecessor**: `TODO/completed/420a3_wave_a_launch.md` (HONEST_NEGATIVE_BLOCKED at same pre-flight FAIL)
- **GPU launcher wrapper fix**: `TODO/completed/415a1_wrapper_args_fix.md` (EXTRA_FLAGS accumulator)
- **Atomic write protocol**: `molmetal/scripts/wf386a3_atomic_write.py` (per §5.6)
- **Standing directives**: AGENTS.md §1.4 (Pre-flight v2 25/25 PASS gate) + §5.1 (standing GPU authorization conditional on pre-flight PASS) + §5.2 (honest framing) + §5.5 (post-wave on-disk verification rule) + §5.6 (atomic write protocol adoption) + §5.7 (trainer flag wire-in smoke-required) + §10 (wetlab deprecation)
- **kill_then_adapt directive**: `mol-metal-gpu-execution §0` (REFUSE launch if pre-flight FAIL)

---

**Wf-420-A6 closeout AUTHORED 2026-10-01T00:46:30+0000 UTC** via `atomic_write_file` per AGENTS.md §5.6 (cf. `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file` — temp-file + rename atomicity + per-path process-local lock). Closing token `---` + final line preserved per `detect_truncation` check.

**This closeout asserts HONEST_NEGATIVE_BLOCKED** — launch REFUSED per AGENTS.md §1.4 + §5.1 + kill_then_adapt §0 + BLOCKER 5 harness-flag mismatch. The MEASURED Lambda × CFM coupling wire-in is functionally complete on disk (per Wf-418-A2 audit + Wf-406-A2 + Wf-410-A2 + Wf-414-A2 cumulative work); the gap is the pre-flight FAIL cascade (BLOCKER 4) and the harness-brief flag documentation drift (BLOCKER 5). Both are CPU-only fixes in the ~10 LOC range; once resolved, the Wf-394 Lambda × CFM retrain can proceed (~5 hr GPU on cuda:0 Blackwell per AGENTS.md §5.1 standing GPU authorization).
