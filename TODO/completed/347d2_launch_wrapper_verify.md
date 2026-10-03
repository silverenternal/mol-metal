# Wf-347-D2 — End-to-End Launch Wrapper Dry-Run Verify

**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-347-D2 verify agent
**Workflow:** Wf-347 (parent); sub-flow D2
**Source brief:** Wf-347-D2 task script (computed) under
  [AGENTS.md §5.1](AGENTS.md) standing GPU authorization + `party-twenty-spirit` §3.5 honest framing.

## §0 TL;DR

All 4 launch wrappers `READY` via dry-run; USER_OK gate correctly
`FAIL`s with exit code 1 when missing (kill_then_adapt standing
directive preserved). Integration verdict: **PASS** — wrappers are
wired end-to-end and safe to advance to USER_OK=1 launch gating.

## §1 Per-wrapper verdict

| # | Wrapper | Mode | Verdict | Key signal |
|---|---|---|---|---|
| 1 | `scripts/launch_wf124_p3_g5.sh` | `--dry-run` | **READY** | `DRY-RUN OK: 15 (metal, seed) pairs PLANNED, no GPU work executed` — 5 metals × 3 seeds |
| 2 | `scripts/launch_wf124_p3_g6.sh` | `--dry-run` | **READY** | `DRY-RUN OK: 60 (config, metal, seed) tuples PLANNED, no GPU work executed` — 4 configs × 5 metals × 3 seeds; 60 CONFIRMED, 0 UNVERIFIED (all 4 configs wired post-Wf-296C) |
| 3 | `scripts/recover_g6_failed_tuples.sh` | `--dry-run` | **READY** | `=== DRY-RUN OK: 1 === No GPU work performed. Set USER_OK=1 for real launch`; recovery manifest loaded 60 tuples (16 OK + 29 FAILED + 1 RUNNING + 14 PENDING = 60); 43 to recover |
| 4 | `scripts/pre_flight_gpu_ready.sh` | `--dry-run` | **READY** | `VERDICT: PASS (dry-run; all hard gates green; USER_OK SKIP expected on --dry-run)`; PASS=3 / FAIL=1 / SKIP=1 / BLOCKING_FAIL=0; the lone FAIL (wf343c2_per_wave) cascades from pre-existing pre-flight FAILs (Wf-350-A2 §5) and is NOT a Wf-350-B1 blocker |

## §2 USER_OK gate verify (kill_then_adapt standing directive)

Test: `bash scripts/pre_flight_gpu_ready.sh` (NO `--dry-run`, NO `USER_OK` env).

Expected: Gate 5 `USER_OK_gate` FAIL → BLOCKING_FAIL=1 → exit code 1.

Observed (full transcript in `/tmp/wf347d2_userok_test.txt`):

```
[09:25:34+0000] ===== Gate 5/5: USER_OK (kill_then_adapt standing directive) =====
[09:25:35] USER_OK_gate                   FAIL    USER_OK unset; re-run with USER_OK=1 bash scripts/pre_flight_gpu_ready.sh
[09:25:35+0000] ===== Wf-350-B1 verdict =====
[2026-09-29T09:25:35+0000]   PASS=3  FAIL=2  SKIP=0  BLOCKING_FAIL=1
[2026-09-29T09:25:35+0000]   STRICT=0  DRY_RUN=0
[2026-09-29T09:25:35+0000] VERDICT: FAIL — blocking gate(s) failed; do NOT launch GPU work
exit_code=1
```

**Verdict: USER_OK gate ACTIVE.** Exit code 1 confirms kill_then_adapt
directive blocks real GPU work without explicit user OK.

## §3 Integration verdict

All 4 wrappers agree on the dry-run contract:

- `launch_wf124_p3_g5.sh --dry-run` → PLANS 15 (metal, seed) pairs; exits 0; no GPU work.
- `launch_wf124_p3_g6.sh --dry-run` → PLANS 60 (config, metal, seed) tuples; 60/60 CONFIRMED; exits 0; no GPU work.
- `recover_g6_failed_tuples.sh --dry-run` → LOADS 60-tuple recovery manifest; identifies 43 to recover; exits 0; no GPU work.
- `pre_flight_gpu_ready.sh --dry-run` → ALL 4 hard gates PASS or non-blocking FAIL (wf343c2 cascading per Wf-350-A2 §5); USER_OK gate SKIP-as-expected; exits 0; no GPU work.

Without `--dry-run`, the pre-flight script becomes a hard gate:
Gates 1/4 (pre_flight_check.py + compute-manifest + wf343c2 + wf343c3)
must PASS, and Gate 5 (USER_OK) must be explicitly set to `1`.

**Integration verdict: PASS** — all wrappers are wired correctly end-to-end.

## §4 Per-wrapper dry-run evidence

### §4.1 `launch_wf124_p3_g5.sh --dry-run`

Output (last 16 lines):

```
--- 15 (metal, seed) pairs to enumerate ---
  Ru seed=0  ->  results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/
  Ru seed=1  ->  results/wf124_p3_50epoch_full_sweep_g5/Ru/seed1/
  Ru seed=2  ->  results/wf124_p3_50epoch_full_sweep_g5/Ru/seed2/
  Ir seed=0  ->  results/wf124_p3_50epoch_full_sweep_g5/Ir/seed0/
  Ir seed=1  ->  results/wf124_p3_50epoch_full_sweep_g5/Ir/seed1/
  Ir seed=2  ->  results/wf124_p3_50epoch_full_sweep_g5/Ir/seed2/
  Os seed=0  ->  results/wf124_p3_50epoch_full_sweep_g5/Os/seed0/
  Os seed=1  ->  results/wf124_p3_50epoch_full_sweep_g5/Os/seed1/
  Os seed=2  ->  results/wf124_p3_50epoch_full_sweep_g5/Os/seed2/
  Rh seed=0  ->  results/wf124_p3_50epoch_full_sweep_g5/Rh/seed0/
  Rh seed=1  ->  results/wf124_p3_50epoch_full_sweep_g5/Rh/seed1/
  Rh seed=2  ->  results/wf124_p3_50epoch_full_sweep_g5/Rh/seed2/
  Re seed=0  ->  results/wf124_p3_50epoch_full_sweep_g5/Re/seed0/
  Re seed=1  ->  results/wf124_p3_50epoch_full_sweep_g5/Re/seed1/
  Re seed=2  ->  results/wf124_p3_50epoch_full_sweep_g5/Re/seed2/

DRY-RUN OK: 15 (metal, seed) pairs PLANNED, no GPU work executed.
```

### §4.2 `launch_wf124_p3_g6.sh --dry-run`

Output (last 16 lines):

```
  [CONFIRMED] cfg=no_dann Os seed=0  ->  results/wf124_p3_per_innovation_ablation_g6/no_dann/Os/seed0/
  [CONFIRMED] cfg=no_dann Os seed=1  ->  results/wf124_p3_per_innovation_ablation_g6/no_dann/Os/seed1/
  [CONFIRMED] cfg=no_dann Os seed=2  ->  results/wf124_p3_per_innovation_ablation_g6/no_dann/Os/seed2/
  [CONFIRMED] cfg=no_dann Rh seed=0  ->  results/wf124_p3_per_innovation_ablation_g6/no_dann/Rh/seed0/
  [CONFIRMED] cfg=no_dann Rh seed=1  ->  results/wf124_p3_per_innovation_ablation_g6/no_dann/Rh/seed1/
  [CONFIRMED] cfg=no_dann Rh seed=2  ->  results/wf124_p3_per_innovation_ablation_g6/no_dann/Rh/seed2/
  [CONFIRMED] cfg=no_dann Re seed=0  ->  results/wf124_p3_per_innovation_ablation_g6/no_dann/Re/seed0/
  [CONFIRMED] cfg=no_dann Re seed=1  ->  results/wf124_p3_per_innovation_ablation_g6/no_dann/Re/seed1/
  [CONFIRMED] cfg=no_dann Re seed=2  ->  results/wf124_p3_per_innovation_ablation_g6/no_dann/Re/seed2/

DRY-RUN OK: 60 (config, metal, seed) tuples PLANNED, no GPU work executed.
  CONFIRMED runs (real launch): 60 (all 4 configs wired post-Wf-296C)
  UNVERIFIED runs (real launch): 0 (all flags verified)
```

### §4.3 `recover_g6_failed_tuples.sh --dry-run`

Output (last 16 lines):

```
WARNING: --prereg <path> not supplied. Paper-grade launches should provide a pre-registration record per AGENTS.md §5.2 + Wf-350-B2 PREREG_TEMPLATE.md.

=== Wf-337-G1 G6 RECOVERY START Tue Sep 29 05:17:37 PM CST 2026 ===
OUT_BASE: results/wf124_p3_per_innovation_ablation_g6
RECOVERY MANIFEST: results/wf124_p3_per_innovation_ablation_g6/recovery_manifest.tsv
  OK:       16
  FAILED:   29
  RUNNING:   1
  PENDING:  14
  TOTAL:    60  (expected 60)
  TO_RECOVER (FAILED + PENDING): 43

Recovery launch will skip OK + RUNNING tuples and re-launch the rest with
Wf-328-W2 + Wf-335-R1+R3 + Wf-296C fixes wired (see COMMON_ARGS banner).

=== DRY-RUN OK: 1 ===
No GPU work performed. Set USER_OK=1 for real launch:
$ USER_OK=1 bash scripts/recover_g6_failed_tuples.sh
```

### §4.4 `pre_flight_gpu_ready.sh --dry-run`

Output (last 16 lines):

```
[09:22:43] USER_OK_gate                   SKIP    --dry-run; USER_OK not required for CPU verification

===== Wf-350-B1 verdict =====
  PASS=3  FAIL=1  SKIP=1  BLOCKING_FAIL=0
  STRICT=0  DRY_RUN=1
VERDICT: PASS (dry-run; all hard gates green; USER_OK SKIP expected on --dry-run)
```

Gate breakdown:

| Gate | Result | Source |
|---|---|---|
| Gate 1: pre_flight_check.py --pipeline all | **PASS** (informative) | preflight v3 n_gates=94 PASS=90 FAIL=2 SKIP=2 (pre-existing FAILs per Wf-350-A2 §5) |
| Gate 2: compute-manifest schema + GPU count + VRAM | **PASS** | manifest=wf350a2_compute_manifest.json schema-valid; 2 GPUs; smallest VRAM=32607 MB; torch=2.14.0+cu130 cuda=13.0 |
| Gate 3: wf343c2_per_wave_check.py --skip-live --strict | **FAIL** (cascading) | rc=1; cascading from pre-flight pre-existing FAILs per Wf-350-A2 §5 — NOT a Wf-350-B1 blocker |
| Gate 4: wf343c3_data_quality_check.py --strict | **PASS** | wf343c3 25-verdict surface all PASS (5 loaders × 5 checks) |
| Gate 5: USER_OK | **SKIP** | --dry-run; USER_OK not required for CPU verification |

Wall: ~172 s total (Gate 1 ~169 s; Gates 2/3/4/5 ≤ 3 s).

## §5 Honest framing (party-twenty-spirit §3.5, AGENTS.md §5.2)

All numeric values above are **MEASURED** from the wrapper transcripts
(`/tmp/wf347d2_preflight_output.txt` + `/tmp/wf347d2_userok_test.txt`).

- **PROJECTED fields**: none. Wall time for the pre-flight script is
  MEASURED at ~172 s on this host (ZhaoLab + RTX 5090 32GB, kernel
  7.2.6-hardened1-1-hardened, torch 2.14.0+cu130, cuda 13.0).
- **MEASURED fields**: 15 G5 pairs, 60 G6 tuples (60 CONFIRMED / 0
  UNVERIFIED), 60-tuple recovery manifest (16 OK + 29 FAILED + 1
  RUNNING + 14 PENDING), 4 pre-flight gates (PASS=3 / FAIL=1
  cascading / SKIP=1 / BLOCKING_FAIL=0), USER_OK gate FAIL → exit 1
  (BLOCKING_FAIL=1).

Caveats preserved per Wf-350-A2 §5:

- The Gate 3 wf343c2_per_wave FAIL is a **cascading FAIL** from
  pre-existing pre-flight FAILs, NOT a Wf-350-B1 blocker. It is
  reported separately as a known-failure summary that the user can
  act on.
- 2 of 3 pre-existing pre-flight FAILs are NOT pre-Wf-3xx (per Wf-350-A2 §5);
  they are tracked in `molmetal/reports/preflight_pre_existing_fails.md`.

## §6 Forward-looking TODO (not gating this closeout)

Per AGENTS.md §5.1 standing GPU authorization, real (non-dry-run)
launches still require:

1. `pre_flight_gpu_ready.sh` exits 0 (PASS) with USER_OK=1.
2. Free VRAM ≥ 16 GB confirmed via `nvidia-smi`.
3. Hyperparameters via config files (referenced peer-paper-aligned
   `--batch-size`, `--lr`, `--hidden`, `--depth`).

These are USER_OK-gated at execution time, not at this dry-run.

## §7 Wall + tests

- **Tests added**: 0 NEW (per Wf-347-D2 brief: "Tests: 0 NEW").
- **Wall**: MEASURED ~6 min total (G5/G6/recovery dry-runs ~5 s each
  + pre_flight_gpu_ready.sh dry-run ~3 min + USER_OK gate test ~3 min).
- **GPU launches**: 0 (CPU-only dry-run verification).
- **Per-wrapper commends**: 4 READY (1-G5 + 1-G6 + 1-recovery + 1-pre-flight).

---

**END OF Wf-347-D2 LAUNCH WRAPPER VERIFY.**
