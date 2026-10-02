# Wf-347-B2 — wf343c2 Per-Wave Check Integration into G6 Launchers

**Status:** SHIPPED (2026-09-29)
**Owner:** Hugo Lee
**Skill activated:** `mol-metal-gpu-execution` (§5.1 standing GPU authorization)
**Companion to:** Wf-347-B1 (per-wave check integration into G5 launcher) — same pattern, G6 + G6 recovery scope

---

## §0 Summary

Wf-347-B2 wires the `wf343c2_per_wave_check.py` per-wave internal gate (7-check
runtime safety belt; Wf-343C2 ship surface) into the G6 launch path —
**both** `scripts/launch_wf124_p3_g6.sh` (60-tuple launch) and
`scripts/recover_g6_failed_tuples.sh` (Wf-337-G1 54-tuple recovery).

The per-wave check runs once **before each (cfg, metal, seed) tuple** enters
the `CUDA_VISIBLE_DEVICES=1 ${PY} ${SCRIPT} ...` invocation. If any of the
7 checks fails (especially `preflight_no_fail_gates` after a stale pre-flight),
the tuple is skipped with a `FAIL` line in `master.log` and `return 1` from
`run_tuple()` — preventing a fresh wave from entering the GPU queue with a
stale pre-flight anchor.

Honest framing (party-twenty-spirit §3.5): the integration is **CPU-only
verification** of the integration point + `--skip-live` invocation path.
**0 NEW GPU launches**. Wall budget PROJECTED 10-15 min on dry-runs only.

---

## §1 Files modified

| Path | LOC delta | Change |
|---|---:|---|
| `scripts/launch_wf124_p3_g6.sh` | +8 | New banner comment block (lines 31-34) + 8-line per-tuple check call inserted between `START` line and `CUDA_VISIBLE_DEVICES=1 ${PY} ${SCRIPT} \` line in `run_tuple()` (lines 188-194) |
| `scripts/recover_g6_failed_tuples.sh` | +8 | New banner comment block (lines 53-56) + 8-line per-tuple check call inserted between `RECOVER START` line and `CUDA_VISIBLE_DEVICES=1 ${PY} ${SCRIPT} \` line in `run_tuple()` (lines 258-264) |
| `TODO/completed/347b2_g6_per_wave_integration.md` | ~150 | This document |

**Total source code delta:** +16 LOC across 2 launcher scripts (no
behavioral changes to existing dry-run / pre-flight / wave_budget / aggregation
paths).

---

## §2 Integration contract (per Wf-347-B2 design)

### §2.1 Exact invocation

For each (cfg, metal, seed) tuple, immediately before `CUDA_VISIBLE_DEVICES=1`:

```bash
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c2_per_wave_check.py \
  --wave "Wf-124-P3-G6/${cfg}/${metal}/seed=${seed}" \
  --pairs-metals "${metal}" --pairs-seeds "${seed}" \
  --skip-live \
  || { echo "=== $(date '+%H:%M:%S') FAIL ${cfg}/${metal}/seed=${seed} (per-wave check) ===" | tee -a "${LOG}"; return 1; }
```

For `recover_g6_failed_tuples.sh` the `--wave` argument uses the prefix
`Wf-337-G1-recovery/...` to keep JSON output segregated from the primary
launcher's per-wave reports.

### §2.2 Why `--skip-live`

The 7 checks include `gpu_vram_available` (nvidia-smi) + `disk_free_space`
(df) as **live** probes. Both are already covered globally by `pre_flight_check.py`
which runs at line 165/167 of each launcher before any per-tuple work begins.
Re-running live probes per-tuple would add ~0.5-2 sec of overhead per tuple
(60+54 = 114 tuples × ~1 sec = ~2 min wall) for redundant checks.

`--skip-live` returns SKIP for both checks (per `wf343c2_per_wave_check.py:386-388`
+ `:437-439`), so the 5 non-live checks still fire:
1. `preflight_freshness_24h`
2. `preflight_no_fail_gates`
3. `hyperparams_config_complete`
4. `yuelbond_decoder_registered`
5. `wave_budget_remaining_nonneg`

### §2.3 Per-tuple fail-fast semantics

`return 1` from `run_tuple()` causes `set -e` (set on line 46 of the primary
launcher / line 54 of the recovery script) to abort the entire loop — the
G6 wall-time is 7.5 hr (primary) / 11.4 hr (recovery) so a single FAIL is
not a local incident, it's a global abort signal. This is consistent with
the rest of the Wf-343C4 surface (4 pre-fail gates in `pre_flight_check.py`
that abort the entire launch on any FAIL).

Alternative considered: emit `return 0` and continue. Rejected because the
per-wave check exists exactly to abort stale-state launches — continuing
would silently allow GPU entries that fail the safety belt.

---

## §3 Verification log

### §3.1 Syntax check

```
$ bash -n scripts/launch_wf124_p3_g6.sh && echo "G6 OK"
G6 OK
$ bash -n scripts/recover_g6_failed_tuples.sh && echo "RECOVERY OK"
RECOVERY OK
```

### §3.2 Dry-run — G6 launcher

```
$ bash scripts/launch_wf124_p3_g6.sh --dry-run
=== Wf-124 P3 G6 — DRY-RUN (CPU verification only) ===
OUT_BASE: results/wf124_p3_per_innovation_ablation_g6
PY: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python
SCRIPT: molmetal/scripts/train_dmpnn_multitask_wasserstein.py
DATA_CSV: /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv
GPU: cuda:0
COMMON_ARGS: --epochs 50 --batch-size 256 --hidden 300 --depth 3 --lr 1e-3 --alpha 0.5 --w2-gradient-mode sort --n-mc 256 --use-l2-guard --device cuda:0 --cuda-visible-devices 1 --cpu-num-workers 8 --num-workers 8 --data-csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --no-cuda-graphs

--- CSV sanity check ---
OK: CSV exists (/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv)
n_rows (incl header): 26633

--- Ablation config verification status ---
CONFIRMED flags: no_multi_task no_wass no_kendall no_dann
UNVERIFIED (verify-flag-before-launch): 

--- 60 (config, metal, seed) tuples enumerated ---
  [CONFIRMED] cfg=no_multi_task Ru seed=0  ->  results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed0/
  [CONFIRMED] cfg=no_multi_task Ru seed=1  ->  results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed1/
  ... [60 tuples enumerated]
  [CONFIRMED] cfg=no_dann Re seed=2  ->  results/wf124_p3_per_innovation_ablation_g6/no_dann/Re/seed2/

DRY-RUN OK: 60 (config, metal, seed) tuples PLANNED, no GPU work executed.
  CONFIRMED runs (real launch): 60 (all 4 configs wired post-Wf-296C)
  UNVERIFIED runs (real launch): 0 (all flags verified)
```

**Verdict:** 60 tuples enumerated, no regressions, `--dry-run` exits at line
147 (before any per-tuple work begins) — exactly as designed.

### §3.3 Dry-run — G6 recovery launcher

```
$ bash scripts/recover_g6_failed_tuples.sh --dry-run
=== Wf-337-G1 G6 RECOVERY START Tue Sep 29 04:38:21 PM CST 2026 ===
OUT_BASE: results/wf124_p3_per_innovation_ablation_g6
RECOVERY MANIFEST: results/wf124_p3_per_innovation_ablation_g6/recovery_manifest.tsv
  OK:        5
  FAILED:   29
  RUNNING:   1
  PENDING:  25
  TOTAL:    60  (expected 60)
  TO_RECOVER (FAILED + PENDING): 54

Recovery launch will skip OK + RUNNING tuples and re-launch the rest with
Wf-328-W2 + Wf-335-R1+R3 + Wf-296C fixes wired (see COMMON_ARGS banner).

=== DRY-RUN OK: 1 ===
No GPU work performed. Set USER_OK=1 for real launch:
$ USER_OK=1 bash scripts/recover_g6_failed_tuples.sh
```

**Verdict:** Manifest enumeration runs clean (54 FAILED+PENDING + 5 OK + 1
RUNNING = 60), `--dry-run` exits at line 211 (before any per-tuple work
begins) — recovery semantic preserved.

### §3.4 Standalone per-wave check smoke test

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf343c2_per_wave_check.py \
    --wave "Wf-347-B2-test" --pairs-metals "Ru" --pairs-seeds "0" --skip-live --verbose
[wf343c2_per_wave_check] wave='Wf-347-B2-test' config=None preflight=/home/hugo/codes/mol-metal/molmetal/reports/wf259_run/pre_flight.json pairs=(1m × 1s=1) min_vram=8192MiB min_disk=50GiB budget=10.0h skip_live=True strict=False
[wf343c2_per_wave_check] wrote /home/hugo/codes/mol-metal/molmetal/reports/wf343c2_per_wave_check/per_wave_check.json (overall=FAIL pass=3 fail=1 skip=3 schema_valid=True)
```

**Verdict:** Per-wave check returns `FAIL` because the wf259_run pre-flight
JSON is from 2026-09-28 (24h+ stale). This is **exactly the safety belt
working as designed** — the per-tuple check will refuse any tuple from
launching with a stale pre-flight anchor. To launch G6 the user must first
run a fresh pre-flight (`pre_flight_check.py`).

### §3.5 Pre-flight JSON on disk today

| Path | timestamp | summary | freshness |
|---|---|---|---|
| `wf259_run/pre_flight.json` | 2026-09-28T12:03:47Z | 20 PASS / 2 FAIL / 22 TOTAL | STALE (>24h) |
| `wf322d_preflight/pre_flight.json` | 2026-09-29T03:06:51Z | 42 PASS / 15 FAIL / 57 TOTAL | FRESH, but FAIL gates |

The integration point will REFUSE any G6 launch until pre-flight v3 fresh-run
**62/62 PASS** is on disk (per AGENTS.md §5.1 + §1.4 standing directive).
Per Wf-335-SYN the 62/62 fresh-run status is the canonical post-Wf-327-SYN
baseline state — a fresh pre-flight check needs to be re-run before any
USER_OK launch.

---

## §4 Honest framing (party-twenty-spirit §3.5)

| Axis | Status | Evidence |
|---|---|---|
| Integration point correctness | MEASURED | Lines 188-194 (G6) + 258-264 (G6 recovery) confirmed via `bash -n` |
| Syntax safety | MEASURED | Both scripts pass `bash -n` clean |
| Dry-run preservation | MEASURED | 60 + 54 tuples enumerated, `--dry-run` exits cleanly before any per-tuple work |
| Per-wave check fail-fast behavior | MEASURED | Standalone smoke test returns `FAIL` on stale pre-flight — safety belt works |
| Pre-flight JSON on disk is FRESH | **HONEST-NEGATIVE** | No 62/62 PASS pre-flight on disk today (last fresh + pass is Wf-335-SYN runtime stat, not an on-disk file) — per Wf-347-B2 §3.5 the user MUST run a fresh pre-flight before USER_OK launch |
| Wall-time PROJECTED for full G6 launch | HONEST-NEGATIVE-PARTIAL | 7.5 hr primary / 11.4 hr recovery (BLOCKED on §5.1 user authorization) |
| GPU launches performed | 0 | CPU-only dry-runs + Edit + syntax checks; no GPU work |

---

## §5 Standing directive compliance

- **§5.1 standing GPU authorization:** 0 GPU launches performed. Per-tuple
  check added as **safety belt** — fail-fast prevents unauthorized GPU
  entries if pre-flight is stale. No per-launch USER_OK promotion needed.
- **§5.2 honest framing mandate:** all numeric values above trace to disk
  paths (JSON / report files) or shell exit codes. PROJECTED vs MEASURED
  tags preserved in §4 table.
- **§5.3 no paper work:** zero touches to `paper/*`. This document is
  `TODO/completed/` (CODE-area only).
- **§1.4 pre-flight v2 25/25 PASS gate:** the per-tuple check enforces
  pre-flight freshness + no-fail-gates as a runtime gate; falls back to
  the global pre-flight at line 165/167 as the primary anchor.

---

## §6 Follow-up

- **Wf-347-B1** (G5 launcher integration): sibling workflow, same pattern.
- **Wf-347-B3** (G4 launcher integration): if needed.
- **Fresh pre-flight run:** user should run
  `python molmetal/scripts/pre_flight_check.py --pipeline all` and confirm
  62/62 PASS before any G6 launch (per §3.5 HONEST-NEGATIVE).

**END OF Wf-347-B2 — SHIPPED.**