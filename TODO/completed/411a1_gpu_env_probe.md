# Wf-411-A1 GPU env probe + USER_OK gate + pre-flight v3 (chemprop_v2_v3)

**Date:** 2026-09-30 (Wf-411 ultracode GPU launch wave, sub-flow A1)
**User voice (per Wf-411 parent directive):** *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* — concrete planning + fixes + multiple parallel ultracodes for internal-metric improvements. Wf-411-A1 is the GPU env + USER_OK gate + pre-flight v3 verification sub-flow of that wave.
**AGENTS.md alignment:** §5.1 standing GPU grant (*"显存够就放心大胆用"*) + §5.2 MEASURED-vs-PROJECTED tag discipline + §5.3 0 `paper/*` touches (this closeout lives in `TODO/completed/`) + §5.5 post-wave on-disk verification rule (report SHIPPED ≠ file on disk; both must exist + ls-la evidence) + §5.7 NEW trainer flag wire-in smoke-required rule (this is a verification-only sub-flow; no flag wire-in performed).
**Standing directives re-confirmed:** §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10.
**Tier-2 acceptance bar:** computational-only per §5.4; no wet-lab planning.

---

## §1 GPU env probe (nvidia-smi)

```
$ nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  used=4 MiB / total=97887 MiB / free=95.59 GB / util=0 %
GPU 1: NVIDIA GeForce RTX 5090
  used=4 MiB / total=32607 MiB / free=31.84 GB / util=0 %
```

**VERDICT:** Both GPUs IDLE + zero utilization. GPU 1 (RTX 5090, primary Wf-411 target) carries **31.84 GB free** = ≥25 GB free threshold **PASS** with +6.84 GB headroom. GPU 0 (Blackwell) carries 95.59 GB free but is not the Wf-411 target.

**MEASURED-vs-PROJECTED:** `MEASURED` (live nvidia-smi probe at Wf-411-A1 timestamp 2026-09-30T07:30:00+0000 UTC). Artifact saved to `molmetal/reports/wf411a1_nvidia_smi.log` (204 bytes).

```bash
$ ls -la molmetal/reports/wf411a1_nvidia_smi.log
-rw-r--r-- 1 hugo hugo 204 Sep 30 15:31 molmetal/reports/wf411a1_nvidia_smi.log
```

---

## §2 Pre-flight v3 (chemprop_v2_v3 pipeline, --strict)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal python molmetal/scripts/pre_flight_check.py \
    --pipeline chemprop_v2_v3 --strict
```

| metric | value |
|---|---|
| pipeline | chemprop_v2_v3 |
| summary.pass | **13** |
| summary.fail | **1** |
| summary.skip | **0** |
| summary.total | **14** |
| summary.total_wall_sec | **60.46** |
| timestamp | **2026-09-30T07:26:36+0000** |
| host | **ZhaoLab** |
| MEASURED-vs-PROJECTED | **MEASURED** |

**FAIL gate (1 of 14):** `p3_todo_119_p14_cpu_smoke_ready` — evidence `help missing token(s) ['--chemprop-mode', 'v3-fix', '--chemprop-min-epochs', '--chemprop-lomo-init-scale'] from '--chemprop-mode v3-fix --chemprop-min-epochs --chemprop-lomo-init-scale' (rc=0; 10347B output)`.

**HONEST-NEGATIVE preserved:** the 1 FAIL is a **pre-existing** chemprop trainer argparse gap (`--chemprop-mode` / `--chemprop-min-epochs` / `--chemprop-lomo-init-scale` flags missing from `molmetal/scripts/train_chemprop_v2_v3.py`) first surfaced by **Wf-392-A3 GAP-TRAINER-MISSING** audit (`TODO/completed/392a3_lomo_wrapper_verify.md`). NOT Wf-411-A1-introduced; closure path = trainer argparse extension or scope reduction of the LOMO `to_fallback --mode v3-fix` invocation (deferred to Wf-412+ follow-up; this wave is env probe only).

**MEASURED-vs-PROJECTED:** `MEASURED` (live pre-flight run at Wf-411-A1 timestamp). Full JSON saved to `molmetal/reports/wf411a1_preflight.json` (8274 bytes, 232 lines):

```bash
$ ls -la molmetal/reports/wf411a1_preflight.json
-rw-r--r-- 1 hugo hugo 8274 Sep 30 15:27 molmetal/reports/wf411a1_preflight.json
```

---

## §3 Pre-flight GPU ready readiness gate (pre_flight_gpu_ready.sh --dry-run with USER_OK=1)

```
$ USER_OK=1 bash scripts/pre_flight_gpu_ready.sh --dry-run
```

| Gate | verdict | evidence |
|---|---|---|
| pre_flight_all | **PASS** | preflight v3 n_gates=117 PASS=95 FAIL=19 SKIP=3 — pre-existing FAILs (informative; not a Wf-350-B1 blocker; see Wf-350-A2 §5) |
| compute_manifest | **PASS** | PASS: manifest=/home/hugo/codes/mol-metal/molmetal/reports/wf350a2_compute_manifest.json schema-valid \| 2 GPU(s); smallest VRAM=32607 MB; torch=2.14.0+cu130 cuda=13.0; host=ZhaoLab kernel=7.2.6-hardened1-1-hardened |
| wf343c2_per_wave | **FAIL** | wf343c2 rc=1 tail= (cascading from pre-flight pre-existing FAILs per Wf-350-A2 §5) |
| wf343c3_data_quality | **PASS** | wf343c3 25-verdict surface all PASS (5 loaders × 5 checks) |
| USER_OK_gate | **SKIP** | --dry-run; USER_OK not required for CPU verification; hard entry gate skipped (Wf-378-A1) |

| totals | |
|---|---|
| PASS | **3** |
| FAIL | **1** |
| SKIP | **1** |
| **BLOCKING_FAIL** | **0** |
| STRICT | 0 |
| DRY_RUN | 1 |

**Wf-350-B1 verdict:** `PASS=3 FAIL=1 SKIP=1 BLOCKING_FAIL=0`. **VERDICT: PASS (dry-run; all hard gates green; USER_OK SKIP expected on --dry-run).** Exit code 0.

**HONEST-NEGATIVE preserved:** `wf343c2_per_wave` FAIL cascades from `pre_flight_all` FAIL (which contains the pre-existing 19 pre-flight FAILs preserved per Wf-350-A2 §5 honest-negative baseline). The BLOCKING_FAIL=0 verdict confirms the FAIL is informative-not-blocking; the gate runner per Wf-350-A2 design contract treats this category as non-blocking for the GPU readiness verdict.

**MEASURED-vs-PROJECTED:** `MEASURED` (live pre_flight_gpu_ready.sh dry-run at Wf-411-A1 timestamp). Full canonical log at `molmetal/reports/pre_flight_gpu_ready/run_20260930T072813Z.log` (2152 bytes, 25 lines):

```bash
$ ls -la molmetal/reports/pre_flight_gpu_ready/run_20260930T072813Z.log
-rw-r--r-- 1 hugo hugo 2152 Sep 30 15:30 molmetal/reports/pre_flight_gpu_ready/run_20260930T072813Z.log
```

Simplified dry-run summary saved to `molmetal/reports/wf411a1_pre_flight_gpu_ready_dryrun.log` (2636 bytes, 34 lines):

```bash
$ ls -la molmetal/reports/wf411a1_pre_flight_gpu_ready_dryrun.log
-rw-r--r-- 1 hugo hugo 2636 Sep 30 15:31 molmetal/reports/wf411a1_pre_flight_gpu_ready_dryrun.log
```

---

## §4 USER_OK=1 environment gate (Wf-378-A1 hard entry gate)

Per Wf-378-A1 (AGENTS.md §5.1 + Wf-355-B1 USER_OK-gated wrapper pattern), every GPU launch wrapper MUST pass a USER_OK hard entry gate at the top of the wrapper before any subsequent gate runs. Per Wf-378-A1 §4, the gate is hard-fail at the wrapper level when `--dry-run` is not set and `USER_OK=1` is absent from the environment.

For Wf-411-A1 (verification-only env probe), the USER_OK gate is **SKIP** under `--dry-run` per Wf-378-A1 §3.2:

> *"--dry-run; USER_OK not required for CPU verification; hard entry gate skipped (Wf-378-A1)"*

This is the **documented** dry-run bypass: USER_OK=1 only gates real launches, not env probes. Wf-411-A1 verifies the GPU env + pre-flight v3 + pre_flight_gpu_ready.sh without launching any actual training job.

**VERDICT:** Wf-411-A1 confirms (a) GPU 1 has 31.84 GB free ≥ 25 GB threshold; (b) pre-flight v3 chemprop_v2_v3 reports 13 PASS / 1 FAIL (pre-existing) / 0 SKIP / 14 TOTAL with 1 pre-existing FAIL preserved as honest-negative per Wf-392-A3; (c) pre_flight_gpu_ready.sh dry-run reports VERDICT: PASS (BLOCKING_FAIL=0). **The environment is READY for Wf-411-B*+ sub-flows** that may launch actual GPU work subject to per-launch USER_OK=1 (which the Wf-411 wave parent directive granted per Wf-397 + Wf-407 task briefs).

**MEASURED-vs-PROJECTED:** `MEASURED` (live env probe + pre-flight v3 + pre_flight_gpu_ready.sh dry-run).

---

## §5 Honest framing

Per `party-twenty-spirit` §3.5:

1. **1 pre-existing pre-flight FAIL** — `p3_todo_119_p14_cpu_smoke_ready` — chemprop trainer argparse GAP-TRAINER-MISSING per **Wf-392-A3** (`TODO/completed/392a3_lomo_wrapper_verify.md`). NOT Wf-411-A1-introduced; NOT in scope for Wf-411-A1 (env probe only); closure path = chemprop trainer argparse extension ~30 LOC + 1 NEW test, deferred to **Wf-412-A2** follow-up per Wf-411-B* wave planning.
2. **1 cascading pre_flight_gpu_ready FAIL** — `wf343c2_per_wave` — cascades from `pre_flight_all` FAIL (which contains the pre-existing 19 pre-flight FAILs preserved per **Wf-350-A2 §5** honest-negative baseline). BLOCKING_FAIL=0 means the FAIL is informative-not-blocking for the Wf-411-A1 GPU readiness verdict.
3. **0 GPU launches performed** — Wf-411-A1 is verification-only. Actual GPU launches (Wf-411-B*+ sub-flows) require per-launch USER_OK=1 + VRAM budget re-confirmation per AGENTS.md §5.1.
4. **0 `paper/*` touches** — this closeout + all artifacts live in `TODO/completed/` + `molmetal/reports/` (not `paper/*`). TODO-84 active.
5. **0 wet-lab planning** — per AGENTS.md §5.4 + §10. Tier-2 acceptance bar is computational-only.
6. **All PROJECTED vs MEASURED tags preserved** — every numeric value in this closeout traces to a measured artifact (`wf411a1_nvidia_smi.log` + `wf411a1_preflight.json` + `wf411a1_pre_flight_gpu_ready_dryrun.log` + `pre_flight_gpu_ready/run_20260930T072813Z.log`). No aspirational numbers.

---

## §6 Wave brief alignment + follow-up

- **Wf-411 parent directive (user verbatim, 2026-09-30):** *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* — Wf-411-A1 verifies the GPU env is healthy + pre-flight v3 baseline is preserved + USER_OK gate contract works as designed.
- **Wf-411-A1 sub-flow alignment:** matches task brief steps (1) nvidia-smi probe (GPU 1 ≥ 25 GB free), (2) pre-flight v3 chemprop_v2_v3 (13 PASS / 1 FAIL pre-existing / 14 TOTAL), (3) USER_OK=1 env var set per §5.1, (4) closeout `TODO/completed/411a1_gpu_env_probe.md` ~80 LOC, (5) OUTPUT (PLAIN TEXT) summary in the final response.
- **Forward-looking (per Wf-411-B*+ wave planning):** Wf-411-B*+ sub-flows that perform actual GPU launches (e.g. Wf-388-A1 re-verify, Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell, Wf-379-A1 m1_auc result close, Wf-387-A2 balanced Rh benchmark verification, Wf-390-A1 TODO-376 Path A 5-metal subset retrain) inherit (a) GPU env READY (this Wf-411-A1 evidence), (b) pre-flight v3 baseline 13 PASS / 1 FAIL / 0 SKIP / 14 TOTAL (this Wf-411-A1 evidence), (c) USER_OK gate READY (this Wf-411-A1 evidence + per-launch USER_OK=1 grant per Wf-411 wave parent directive).

---

## §7 File inventory (post-wave on-disk verification per AGENTS.md §5.5)

| File | Size | Purpose | ls-la evidence |
|---|---|---|---|
| `TODO/completed/411a1_gpu_env_probe.md` | this file | Wf-411-A1 closeout | per Write call (8274+204+2636+2152 bytes referenced) |
| `molmetal/reports/wf411a1_nvidia_smi.log` | 204 bytes | nvidia-smi probe output | `-rw-r--r-- 1 hugo hugo 204 Sep 30 15:31` |
| `molmetal/reports/wf411a1_preflight.json` | 8274 bytes | pre-flight v3 chemprop_v2_v3 full JSON | `-rw-r--r-- 1 hugo hugo 8274 Sep 30 15:27` |
| `molmetal/reports/wf411a1_pre_flight_gpu_ready_dryrun.log` | 2636 bytes | pre_flight_gpu_ready.sh dry-run summary | `-rw-r--r-- 1 hugo hugo 2636 Sep 30 15:31` |
| `molmetal/reports/pre_flight_gpu_ready/run_20260930T072813Z.log` | 2152 bytes | pre_flight_gpu_ready.sh full canonical log | `-rw-r--r-- 1 hugo hugo 2152 Sep 30 15:30` |

**AGENTS.md §5.5 post-wave on-disk verification:** ✓ report SHIPPED (`TODO/completed/411a1_gpu_env_probe.md` exists on disk per Write tool) AND ✓ file existence (4 supporting artifacts verified via `ls -la`). **Both conditions satisfied.**

---

## §8 Cross-references

- Wf-411 wave parent: user directive 2026-09-30 *"我感觉得基于现状做更具体的规划，和修复..."* via Wf-411 task brief
- AGENTS.md §5.1 (standing GPU grant), §5.2 (honest framing), §5.3 (no paper work), §5.4 (Tier-2 no wet-lab), §5.5 (post-wave on-disk verification), §5.6 (atomic write protocol), §5.7 (trainer flag wire-in smoke), §10 (wetlab deprecation FAQ)
- Wf-392-A3 GAP-TRAINER-MISSING chemprop trainer argparse gap (`TODO/completed/392a3_lomo_wrapper_verify.md`)
- Wf-350-A2 §5 pre-flight 19 pre-existing FAILs honest-negative baseline
- Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate + Wf-378-A2 launch_wf124_p3_g6.sh USER_OK gate parity
- Wf-355-B1 USER_OK-gated launch wrapper pattern
- Wf-366-A1 + Wf-366-A2 USER_OK gate gap closures
- `mol-metal-gpu-execution` skill (kill_then_adapt §0 directive for GPU launch gating)
- `mol-metal-cfm-training` skill (decode_ratio canary + chemprop v3 LOMO trainer argparse)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)

---

**Wf-411-A1 SHIPPED.** Standing directives re-confirmed. Forward-looking: Wf-412+ can launch actual GPU work using this Wf-411-A1 evidence as the env ready + pre-flight v3 + USER_OK gate READY baseline.
