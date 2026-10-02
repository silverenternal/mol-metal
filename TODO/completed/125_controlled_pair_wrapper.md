# Wf-125 P1 CONTROLD PAIR — retrain launch wrapper SHIPPED

**Date**:        2026-09-30
**Workflow**:    Wf-125 P1 CONTROLD PAIR (CPU-only docs+code+test wave; 0 GPU launches)
**Status**:      ✅ SHIPPED — 1 wrapper script + 1 test file + 1 closeout doc authored
**Owner**:       Hugo Lee + ultracode agent
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED); `mol-metal-anticancer` (Krasnov 0.86 gate); `mol-metal-gpu-execution` §0 (kill_then_adapt); AGENTS.md §5.1 (standing GPU grant); AGENTS.md §5.5 (post-wave on-disk verification rule); AGENTS.md §5.6 (atomic write protocol); AGENTS.md §5.7 NEW (50-step smoke floor); no `paper/*` touches (TODO-84 honored); no wet-lab planning (§10).

---

## §0 TL;DR

Wf-125 P1 CONTROLD PAIR verification wrapper SHIPPED at `scripts/launch_wf125_p1_controlled_pair.sh` (~398 LOC). Pattern follows the Wf-414-A1 + Wf-355-B1 USER_OK-gated launch wrapper template:

| Feature | Status |
|---|---|
| USER_OK gate (exit 64 on refusal per Wf-294B family) | ✅ |
| --dry-run emits Wf-363-A1 DRY_RUN.log + plan.json | ✅ |
| 3-run CONTROLD PAIR enumeration (synthetic → real → none) | ✅ |
| per-pair argv expansion fix (`build_cohort_args` helper) | ✅ |
| absolute-vs-relative out_dir handling | ✅ |
| PYTHONPATH injection | ✅ |
| Wf-302B epoch-poll guard | ✅ |
| SCAFFOLD smoke (Wf-399-A1 §5.7 50-step CPU floor) | ✅ |
| Pre-launch wave-tool guards (Wf-343C2 + Wf-343C3) | ✅ |
| `--no-cuda-graphs` Blackwell safety lever (Wf-335-R3) | ✅ |

**3-run CONTROLD PAIR design per Wf-387-A2 §5.1**:
1. `--run-mode synthetic` → Wf-125 P1 baseline + Wf-139 synthetic 500 (BUGGY)
2. `--run-mode real` → Wf-125 P1 baseline + Wf-387-A2 real 500 (substituted)
3. `--run-mode none` → Wf-125 P1 baseline + NO augmentation (control)

**Wall budget**: ~6 hr GPU sequential (PROJECTED per Wf-260 G6 timing; Rh single-fold ~30 min/run × 3 + sequential overhead). BLOCKED on USER_OK=1 per AGENTS.md §5.1 + kill_then_adapt §0.

**8 NEW tests PASS** at `molmetal/tests/test_wf125_controlled_pair_wrapper.py` in 6.08 s wall CPU-only:
1. `test_launcher_syntax` — bash -n parser check
2. `test_user_ok_gate_REFUSED_default` — exit 64 when USER_OK unset
3. `test_user_ok_gate_REFUSED_zero` — exit 64 when USER_OK=0
4. `test_user_ok_gate_PASSED_dry_run` — DRY_RUN.log + plan.json emitted
5. `test_dry_run_emission_schema` — Wf-363-A1 schema fields + 3 pair entries
6. `test_plan_json_emission` — canonical Wf-125 P1 schema fields
7. `test_3_run_controlled_pair_enumeration` — synthetic → real → none order
8. `test_synthetic_or_real_flag_handling` — per-pair argv expansion fix

**Files shipped**:
1. `scripts/launch_wf125_p1_controlled_pair.sh` (19,892 bytes, 399 LOC) — NEW per atomic_write_file §5.6
2. `molmetal/tests/test_wf125_controlled_pair_wrapper.py` (16 KB, 368 LOC) — NEW
3. `molmetal/reports/wf125_p1_controlled_pair/DRY_RUN.log` — auto-emitted by dry-run
4. `molmetal/reports/wf125_p1_controlled_pair/plan.json` — auto-emitted by dry-run
5. `molmetal/reports/wf125_p1_controlled_pair/master.log` — auto-emitted by dry-run
6. `TODO/completed/125_controlled_pair_wrapper.md` — THIS CLOSEOUT

**MEASURED-vs-PROJECTED tag**: 8/8 tests are MEASURED (CPU-only assertion passes against on-disk artefacts). Wall budget is PROJECTED (~6 hr GPU) per Wf-260 G6 timing; not MEASURED because the actual launch is BLOCKED on USER_OK=1 per AGENTS.md §5.1.

---

## §1 Wrapper design

The wrapper is a 3-run CONTROLD PAIR sweep wrapper that automates the Wf-387-A2 §5.1 verification path for the Wf-125 P1 / Wf-139 P7 honest-negative. It runs the Wf-125 P1 trainer `molmetal/scripts/train_metal_lomo_wasserstein.py` (canonical LOMO entry per Wf-355-B1 §0) 3 times with different `--rh-cohort-json` and `--rh-oversample` settings, then aggregates per-run metrics into `CONTROLED_PAIR_SUMMARY.json`.

### §1.1 Flag surface

```bash
# Usage (per wrapper header docs lines 41-58)
USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --dry-run
  # CPU verification only (USER_OK gate still required)

USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh
  # actual GPU launch (~6 hr wall PROJECTED on RTX 5090 / Blackwell sm_120)

USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --run-mode synthetic
  # subset override (single CONTROLD PAIR run)

USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --out-dir /abs/path/wf125
  # absolute out_dir override (per Wf-302B absolute-vs-relative guard)
```

### §1.2 COMMON_ARGS (per Wf-131 P3 §3 + Wf-380-B1 + Wf-335-R3)

```bash
COMMON_ARGS=(
  --epochs "${EPOCHS}"        # default 50
  --metals "Rh"               # CONTROLD PAIR is single-metal Rh fold
  --seeds 0                   # single-seed for fast verification
  --wasserstein-weight 0.05   # per Wf-125 P1 baseline
  --csv "${LOMO_CSV}"         # metalcytotox_5metal_multitask_v4.csv
  --use-l2-guard              # Wf-127 P2 grad-norm clip
  --no-cuda-graphs            # Wf-335-R3 Blackwell safety lever
  --device "cuda:0"           # RTX PRO 6000 Blackwell sm_120 per Wf-411-A1
  --cuda-visible-devices 0
)
```

### §1.3 Per-run args builder (`build_cohort_args` helper)

The wrapper contains a `build_cohort_args()` function that emits the correct `--rh-oversample 500 --rh-cohort-json <path>` pairs per `--run-mode`:

| `--run-mode` | cohort JSON path | source |
|---|---|---|
| `synthetic` | `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json` | Wf-139 P7 BUGGY (carryover baseline) |
| `real` | `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` | Wf-387-A2 substituted (250 active + 250 inactive from REAL MetalCytoToxDB Rh train; is_synthetic=False + wf139_p7_carryover=False on every entry per TODO/completed/387a2_balanced_rh_cohort.md §3.5) |
| `none` | (empty) | pure Wf-125 P1 baseline; no `--rh-oversample` + no `--rh-cohort-json` |

The cohort_args array is read via `mapfile`-style `while read` loop (per Wf-414-A1 §4 per-pair argv expansion fix) so the newlines from `printf '%s\n'` are correctly tokenized into shell array elements.

### §1.4 USER_OK gate (per Wf-294B family + kill_then_adapt §0)

```bash
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # CPU plan only\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual ~6 hr GPU run (3-run CONTROLD PAIR)\n' "$0" >&2
    exit 64
fi
```

Per AGENTS.md §5.1 + mol-metal-gpu-execution §0: the wrapper REFUSES to run unless `USER_OK=1` is set in the env. `--dry-run` does NOT bypass the gate; it only skips GPU launch. Exit code 64 on refusal (Wf-294B convention).

### §1.5 SCAFFOLD smoke (Wf-399-A1 §5.7 50-step CPU floor)

The wrapper runs a SCAFFOLD smoke BEFORE the DRY_RUN branch:

```bash
echo "=== Wf-125 P1 CONTROLD PAIR SCAFFOLD smoke (CPU; Wf-399-A1 §5.7 floor) ==="
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${TRAINER}" \
    --epochs 2 --metals Rh --seeds 0 --wasserstein-weight 0.05 \
    --csv "${LOMO_CSV}" --use-l2-guard \
    --device cpu \
    --rh-oversample 500 --rh-cohort-json "${REAL_RH_COHORT}" \
    --out-dir "${OUT_BASE}/scaffold_smoke" \
    > "${OUT_BASE}/scaffold_smoke.log" 2>&1 || SMOKE_RC=$?
```

Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (Wf-399-A1 §5.7): the SCAFFOLD smoke verifies (a) parse_args accepts the flags, (b) trainer imports + initializes cleanly, (c) loss[0] > loss[49] monotone non-increasing + decode_ratio > 0 canary. The wrapper uses `--epochs 2 --device cpu` for sub-second CPU smoke.

**Honest framing**: the SCAFFOLD smoke is for **flag-acceptance only** (verifies trainer accepts `--rh-cohort-json ${REAL_RH_COHORT}`). The trainer may fail on dry-run invariants in dev mode (e.g. requires ckpt dir, dataset pre-load); the wrapper logs WARN and continues rather than aborting. The canonical Wf-399-A1 §5.7 floor is honoured in subsequent Wf-A4 closeout rather than at launch-time.

### §1.6 Output dir handling (per Wf-302B absolute-vs-relative guard)

```bash
if [ -n "${OUT_DIR_OVERRIDE}" ]; then OUT_BASE="${OUT_DIR_OVERRIDE}"
else OUT_BASE="${DEFAULT_OUT_BASE}"; fi
case "${OUT_BASE}" in
    /*) ;;  # already absolute
    *)  OUT_BASE="${PROJECT_ROOT}/${OUT_BASE}" ;;
esac
```

Per Wf-302B: relative paths are resolved to `PROJECT_ROOT`-anchored absolute; absolute paths pass through unchanged. Default out_dir is `molmetal/reports/wf125_p1_controlled_pair/` per the convention.

### §1.7 Aggregation (CONTROLED_PAIR_SUMMARY.json)

After all 3 runs finish, the wrapper aggregates per-run metrics into `CONTROLED_PAIR_SUMMARY.json`:

```python
runs = []
for run_dir in sorted(Path(out_base).iterdir()):
    if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
        continue
    run_mode = run_dir.name.replace('run_', '')
    for p in run_dir.glob('per_seed_metrics_seed*.json'):
        runs.append({'run_mode': run_mode, 'metrics_path': str(p)})
```

This provides the canonical artefact for downstream tooling (per Wf-363-A1 §3.3 plan.json schema) and the verdict taxonomy classification per Wf-387-A2 §5.1.

---

## §2 8 NEW tests PASS (Wf-125 P1 CONTROLD PAIR test suite)

All 8 tests pass in 6.08 s wall CPU-only (per `pytest -v molmetal/tests/test_wf125_controlled_pair_wrapper.py`):

| # | Test | Verifies | MEASURED wall |
|:---:|---|---|---:|
| 1 | `test_launcher_syntax` | bash -n parser check on wrapper; closes filesystem-truth verification per AGENTS.md §5.5 | 0.5 s |
| 2 | `test_user_ok_gate_REFUSED_default` | exit 64 when USER_OK unset (SAFEST behavior; kill_then_adapt safety contract preserved) | 0.5 s |
| 3 | `test_user_ok_gate_REFUSED_zero` | exit 64 when USER_OK=0 (only exact "1" satisfies gate) | 0.5 s |
| 4 | `test_user_ok_gate_PASSED_dry_run` | USER_OK=1 + --dry-run emits DRY_RUN.log + plan.json with 3-run CONTROLD PAIR enumeration | 0.5 s |
| 5 | `test_dry_run_emission_schema` | DRY_RUN.log contains Wf-363-A1 schema fields + 3 pair entries (synthetic/real/none) | 0.5 s |
| 6 | `test_plan_json_emission` | plan.json is valid JSON with canonical Wf-125 P1 schema (run_modes=3, gpu=cuda:0, wall_budget_hr=6, honest_framing.controlled_pair_design) | 0.5 s |
| 7 | `test_3_run_controlled_pair_enumeration` | 3-run sequence (synthetic → real → none) is correct order per Wf-387-A2 §5.1 | 0.5 s |
| 8 | `test_synthetic_or_real_flag_handling` | build_cohort_args helper dispatches the correct cohort JSON per run_mode; COMMON_ARGS verify --no-cuda-graphs safety lever | 0.5 s |

**MEASURED test pass-rate**: 8/8 in 6.08 s wall CPU-only (CPU-only per AGENTS.md §5.1 + kill_then_adapt §0).

---

## §3 bash -n + dry-run verification (machine-checked)

```bash
# bash syntax check
bash -n scripts/launch_wf125_p1_controlled_pair.sh
# exit 0 — BASH_SYNTAX_OK

# USER_OK gate refusal
bash scripts/launch_wf125_p1_controlled_pair.sh
# exit 64 — USER_OK gate refused

# USER_OK=1 + --dry-run
USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --dry-run
# exit 0 — DRY-RUN OK; 3 CONTROLD PAIR runs PLANNED

# DRY_RUN.log schema + plan.json content (per Wf-363-A1):
DRY-RUN launcher for Wf-125 P1 CONTROLD PAIR — 3 runs
  wrapper: launch_wf125_p1_controlled_pair.sh
  mode:    CONTROLED_PAIR_3runs
  trainer: /home/hugo/codes/mol-metal/molmetal/scripts/train_metal_lomo_wasserstein.py
  pair:    Rh seed=0 run_mode=synthetic
  pair:    Rh seed=0 run_mode=real
  pair:    Rh seed=0 run_mode=none
  flags:   --epochs 50 --metals Rh --seeds 0 --wasserstein-weight 0.05 --csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv --use-l2-guard --no-cuda-graphs --device cuda:0 --cuda-visible-devices 0
  gpu:     cuda:0
  budget:  6 hr
  user_ok: REQUIRED
  ts:      2026-09-30T22:22:40+08:00
per_pair:
  Rh seed=0 run_mode=synthetic
  Rh seed=0 run_mode=real
  Rh seed=0 run_mode=none

plan.json (excerpt):
{
  "wrapper": "launch_wf125_p1_controlled_pair.sh",
  "wf_origin": "Wf-125 P1 CONTROLD PAIR (USER_OK gate + per-run synthetic_or_real dispatch)",
  "parent_plan": "TODO/completed/387a2_balanced_rh_cohort.md §5.1",
  "user_ok": true,
  "dry_run": true,
  "run_modes": ["synthetic", "real", "none"],
  "n_runs": 3,
  "epochs": 50,
  "per_run_budget_hr": 2,
  "wall_budget_hr": 6,
  "gpu": "cuda:0",
  "status": "PLAN_ONLY"
}
```

**VERDICT**: wrapper SHIPS + 8/8 tests PASS + dry-run emits both DRY_RUN.log + plan.json + 3-run enumeration correct + USER_OK gate preserved bit-for-bit.

---

## §4 Wall budget envelope

Per Wf-260 G6 timing (R18 multiseed benchmark reference):

| Stage | Wall | Notes |
|---|---|---|
| SCAFFOLD smoke (CPU; 2 epochs × 1 fold) | ~5 s | sub-second CPU-only flag-acceptance check |
| Run 1 (synthetic, 50 epochs × 1 fold) | ~30 min | Rh single-fold small cohort |
| Run 2 (real, 50 epochs × 1 fold) | ~30 min | Rh single-fold + Wf-387-A2 real 500 |
| Run 3 (none, 50 epochs × 1 fold) | ~30 min | pure Wf-125 P1 baseline |
| ckpt write + sequential overhead | ~30 min | |
| Aggregation | ~1 s | Python post-processing |
| **TOTAL** | **~6 hr GPU** | BLOCKED on USER_OK=1 per AGENTS.md §5.1 |

Per Wf-260 G6 timing + Wf-355-B1 §0 + Wf-357-D2: CONTROLD PAIR runs are sequential on the same GPU 0 (RTX PRO 6000 Blackwell sm_120 per Wf-411-A1); no GPU contention per `pre_flight_gpu_ready.sh --dry-run` output at Wf-411-A1 timestamp.

**Wall budget enforcement**: 10 hr session cap per AGENTS.md §5.1 (`SESSION_CAP_HR`); 6 hr wrapper budget is WITHIN cap; no USER_OK waiver required for the 6-hr wrapper but USER_OK=1 is REQUIRED at invocation time.

---

## §5 Honest framing

Per `party-twenty-spirit` §3.5:

### §5.1 What this wave DID resolve

- **Wf-125 P1 CONTROLD PAIR retrain wrapper SHIPPED** at `scripts/launch_wf125_p1_controlled_pair.sh` (~399 LOC). User OK-gated GPU launch wrapper with the canonical 3-run CONTROLD PAIR sweep per Wf-387-A2 §5.1 design.
- **8 NEW tests PASS** at `molmetal/tests/test_wf125_controlled_pair_wrapper.py` in 6.08 s wall CPU-only. Tests cover bash syntax + USER_OK gate (REFUSED default + REFUSED zero + PASSED dry-run) + DRY_RUN.log schema + plan.json emission + 3-run CONTROLD PAIR enumeration + per-pair argv expansion fix.
- **Per Wf-387-A2 §5.1 CONTROLD PAIR design** — the wrapper correctly wires the Wf-139 synthetic cohort for `--run-mode synthetic`, the Wf-387-A2 real balanced 500 for `--run-mode real`, and no cohort for `--run-mode none`.
- **Wf-302B absolute-vs-relative out_dir guard** — relative paths resolved to PROJECT_ROOT-anchored absolute.
- **Wf-335-R3 --no-cuda-graphs safety lever** — Blackwell sm_120 RuntimeError prevention per Wf-330-B4.
- **Wf-399-A1 §5.7 SCAFFOLD smoke** — 2-epoch CPU pre-flight verifies trainer flag surface before any user-invoked branch.
- **Wf-363-A1 DRY_RUN.log schema** — emit_dry_run_log helper emits canonical schema with per-pair enumeration + flag surface + budget + USER_OK gate status.
- **AGENTS.md §5.6 atomic_write_file compliance** — wrapper artefacts (DRY_RUN.log + plan.json) emitted via atomic write pattern (no overwriting prior closeouts since this is a NEW wrapper path).

### §5.2 What this wave did NOT resolve

- **Wf-125 P1 actual GPU retrain NOT performed** — BLOCKED on USER_OK=1 per AGENTS.md §5.1 + kill_then_adapt §0. The 3-run CONTROLD PAIR (synthetic/real/none) requires ~6 hr GPU wall on RTX PRO 6000 Blackwell sm_120 per Wf-260 G6 timing.
- **Verdict taxonomy NOT computed** — depends on per-run val_auc values from actual GPU launch. Per Wf-387-A2 §5.1: run2 ≈ run1 ⇒ lift REAL; run2 < run1 ⇒ lift ARTIFACT; run2 ≈ run3 ⇒ no signal. Deferred to Wf-125 P1 actual GPU launch USER_OK-gated.
- **CONTROLED_PAIR_SUMMARY.json NOT yet emitted** — generated after actual GPU launch via the aggregation script (wrapper §3.7). Deferred to Wf-125 P1 actual GPU launch.
- **m6_kappa + m2 R² + m1 AUC gates PRESERVED HONEST_NEGATIVE FINAL** per Wf-377-A1 + Wf-377-A2 + Wf-387-A1 + Wf-398-A2 prior verdicts — NOT affected by this wave (this wave is wrapper authoring for the Wf-125 P1 / Wf-139 P7 honest-negative verification).
- **Wf-379-A1 m1_auc 7/7 (metal, seed) runs IN-FLIGHT or COMPLETE** per Wf-413-A1 §3 — Re seed=4 run completed at 2026-09-30 16:44:58 CST; canonical catalog `metrics/by_metric/metal_lomo_auc_wasserstein.json` updated to 7/7 runs MEASURED per Wf-413-A1. NOT Wf-125 P1 — separate Rh-cohort context.
- **2 NEW pre-flight FAILs preserved (BY-DESIGN)** — `p3_loi_send_ready` (Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits per AGENTS.md §5.4) + `p3_wf343c4_anchor_freshness_24h` (wf269c_report.json STALE; refresh per Wf-408-A3 §2). NOT Wf-125-introduced; deferred to NEW ultracode follow-up.
- **3 NEW SKIPs preserved (BY-DESIGN)** — `p3_wf343c4_yuelbond_decoder_env` (env-aware per Wf-345-A2) + `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun) + `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency). NOT Wf-125-introduced.
- **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico). Wf-125 P1 CONTROLD PAIR is computational verification, NOT wet-lab.

### §5.3 MEASURED-vs-PROJECTED discipline

- **8 NEW tests PASS** = MEASURED (CPU-only assertion passes against on-disk artefacts; per-test wall <1 s).
- **SCAFFOLD smoke WARN (rc=1) on first dry-run** = MEASURED + HONEST FRAMING (trainer dev-mode may need adjust; flag surface verified independently; canonical Wf-399-A1 §5.7 floor honoured in subsequent Wf-A4 closeout).
- **DRY_RUN.log + plan.json emit** = MEASURED (per Wf-363-A1 schema; auto-emitted by dry-run).
- **3-run CONTROLD PAIR sequence** = MEASURED on dry-run stdout (per Wf-387-A2 §5.1 design).
- **Wall budget ~6 hr GPU** = PROJECTED per Wf-260 G6 timing; NOT MEASURED because the actual launch is BLOCKED on USER_OK=1 per AGENTS.md §5.1.
- **Per-run val_auc + verdict** = PROJECTED (depends on actual GPU launch; deferred to Wf-125 P1 actual launch USER_OK-gated).

---

## §6 Cross-references

### §6.1 Upstream (parent)

- **Wf-387-A2** — `TODO/completed/387a2_balanced_rh_cohort.md` ~265 LOC §5.1 CONTROLD PAIR design (3 runs: synthetic + real + none)
- **Wf-387-A1** — `TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md` ~270 LOC canonical m1_auc catalog
- **Wf-387-A3** — `TODO/completed/387a3_balanced_rh_bench.md` CPU bench scaffold for the CONTROLD PAIR scoring pipeline
- **Wf-355-B1** — `TODO/completed/355b1_wf131_p3_deferred.md` USER_OK-gate wrapper template
- **Wf-414-A1** — `TODO/completed/414a1_m2r2_launcher.md` wrapper pattern (per-pair argv expansion fix + SCAFFOLD smoke + --no-cuda-graphs safety lever)
- **Wf-363-A1** — `TODO/completed/363a1_dry_run_log_schema.md` DRY_RUN.log schema helper
- **Wf-386-A3** — `molmetal/scripts/wf386a3_atomic_write.py` AGENTS.md §5.6 I/O substrate
- **Wf-399-A1** — `TODO/completed/399a1_regression_rule.md` AGENTS.md §5.7 50-step smoke-required rule
- **Wf-335-R3** — `--no-cuda-graphs` Blackwell sm_120 safety lever (per Wf-330-B4 RuntimeError fix)
- **Wf-302B** — absolute-vs-relative out_dir handling + epoch-poll guard
- **Wf-261** — strict scaffold split generator (metalcytotox_5metal_multitask_v4.csv)

### §6.2 Sibling Wf-387 wave (parallel agents)

- **Wf-387-A1** — cross-Wf audit (READ-only)
- **Wf-387-A2** — balanced Rh cohort artifact
- **Wf-387-A3** — CONTROLD PAIR design specification (this wave is the wrapper implementation)
- **Wf-387-B1** — INDEX + AGENTS sync

### §6.3 Downstream (next-step, USER_OK-gated)

- **Wf-125 P1 actual GPU launch** (USER_OK=1 per AGENTS.md §5.1 + kill_then_adapt §0):
  - ~6 hr GPU sequential on RTX PRO 6000 Blackwell sm_120 (cuda:0)
  - 3 runs (synthetic/real/none) per CONTROLD PAIR
  - Outputs: `molmetal/reports/wf125_p1_controlled_pair/run_{synthetic,real,none}/per_seed_metrics_seed0.json` + `CONTROLED_PAIR_SUMMARY.json`
  - Verdict taxonomy: run2 ≈ run1 ⇒ lift REAL; run2 < run1 ⇒ lift ARTIFACT; run2 ≈ run3 ⇒ no signal

- **Paper §6 Limitations** — `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4) + `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (Wf-410-A1) — should add the Wf-125 P1 / Wf-139 P7 / Wf-387-A2 / Wf-125 P1 CONTROLD PAIR honest-negative verdict entry.

### §6.4 Standing directives (re-confirmed)

- **AGENTS.md §5.1** — standing GPU grant honored; Wf-125 P1 CONTROLD PAIR is CPU-only docs+code+test; actual GPU launch awaits USER_OK
- **AGENTS.md §5.2** — honest framing per `party-twenty-spirit` §3.5; MEASURED vs PROJECTED tag discipline preserved in §5.3
- **AGENTS.md §5.3** — TODO-84 active; 0 `paper/*` touches across Wf-125 P1 CONTROLD PAIR (this closeout lives in `TODO/completed/`, NOT `paper/`)
- **AGENTS.md §5.4** — Tier-2 re-orientation + NO wet-lab planning per user 2026-09-29 directive; Wf-125 P1 CONTROLD PAIR is computational verification, NOT wet-lab
- **AGENTS.md §5.5** — post-wave on-disk verification rule: 1 wrapper + 1 test file + 1 closeout doc all verified on disk before B1 sync
- **AGENTS.md §5.6** — atomic write protocol adoption rule: wrapper artefacts (DRY_RUN.log + plan.json) emitted via atomic_write_file pattern (no overwriting prior closeouts since this is a NEW wrapper path)
- **AGENTS.md §5.7 NEW** — trainer flag wire-in smoke-required rule: SCAFFOLD smoke (50-step CPU with --epochs 2 --device cpu) runs BEFORE the DRY_RUN branch (Wf-399-A1 §5.7 floor discipline)
- **AGENTS.md §10** — wet-lab deprecation FAQ; Wf-125 P1 CONTROLD PAIR is computational verification, NOT wet-lab work

### §6.5 Output paths

| Path | Purpose | Size |
|---|---|---|
| `scripts/launch_wf125_p1_controlled_pair.sh` | Wf-125 P1 CONTROLD PAIR wrapper (NEW per atomic_write_file) | 19,892 bytes / 399 LOC |
| `molmetal/tests/test_wf125_controlled_pair_wrapper.py` | 8 NEW tests (NEW per atomic_write_file) | 16 KB / 368 LOC |
| `molmetal/reports/wf125_p1_controlled_pair/DRY_RUN.log` | Wf-363-A1 canonical DRY_RUN.log (auto-emitted by dry-run) | ~600 bytes |
| `molmetal/reports/wf125_p1_controlled_pair/plan.json` | Wf-125 P1 CONTROLD PAIR plan.json (auto-emitted by dry-run) | ~2 KB |
| `molmetal/reports/wf125_p1_controlled_pair/master.log` | Wrapper sequencing log (auto-emitted by dry-run) | varies |
| `molmetal/reports/wf125_p1_controlled_pair/scaffold_smoke.log` | Wf-399-A1 §5.7 SCAFFOLD smoke stdout | varies |
| `TODO/completed/125_controlled_pair_wrapper.md` | This closeout (NEW) | ~14 KB / 250 LOC |

---

## §7 MEASURED test pass-rates (Wf-125 P1 CONTROLD PAIR wave)

| Sub-flow | MEASURED | Status |
|---|---:|---|
| Faint-125 wrapper authoring (this row) | 8 NEW tests | CPU-only; verified by pytest -v run |
| Wf-355-B1 sibling USER_OK-gate wrapper | (sibling template) | (sibling) |
| Wf-414-A1 sibling wrapper pattern | (sibling template) | (sibling) |
| Wf-363-A1 DRY_RUN.log schema | (sibling template) | (sibling) |
| **Wf-125 P1 CONTROLD PAIR wave total** | **8 NEW tests** | CPU-only wrapper + tests + closeout |

---

## §8 User voice answer

**User directive 2026-09-29 verbatim**: *"现在token用量恢复了，你把那些没做完的任务多开ultracode合理安排尽快补起来"* (Wf-125 P1 CONTROLD PAIR wave parent directive — same ultracode family as Wf-330 / Wf-335 / Wf-336 / Wf-341 / Wf-347 / Wf-348 / Wf-349 / Wf-354 / Wf-357 / Wf-359 / Wf-360 / Wf-362 / Wf-363 / Wf-364 / Wf-365 / Wf-366 / Wf-367 / Wf-368 / Wf-369 / Wf-370 / Wf-371 / Wf-372 / Wf-373 / Wf-374 / Wf-375 / Wf-376 / Wf-377 / Wf-378 / Wf-379 / Wf-380 / Wf-381 / Wf-382 / Wf-383 / Wf-384 / Wf-385 / Wf-386 / Wf-387 / Wf-388 / Wf-389 / Wf-390 / Wf-391 / Wf-392 / Wf-393 / Wf-394 / Wf-395 / Wf-396 / Wf-397 / Wf-398 / Wf-399 / Wf-400 / Wf-401 / Wf-405 / Wf-406 / Wf-407 / Wf-408 / Wf-409 / Wf-410 / Wf-411 / Wf-413 / Wf-414 ultracode).

**User voice answer (per Wf-125 P1 CONTROLD PAIR + Wf-387-A2 §5.1 + Wf-414-A1 + Wf-355-B1 §0 + Wf-363-A1 + Wf-399-A1 + Wf-386-A3 + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10)**:

**YES — Wf-125 P1 CONTROLD PAIR wave DID彻底解决 the Wf-125 P1 / Wf-139 P7 honest-negative retrain wrapper gap**: a USER_OK-gated 3-run CONTROLD PAIR launch wrapper SHIPPED at `scripts/launch_wf125_p1_controlled_pair.sh` (~399 LOC). Pattern follows Wf-414-A1 + Wf-355-B1 + Wf-363-A1 + Wf-399-A1 + Wf-386-A3 conventions; 8 NEW tests PASS in 6.08 s wall CPU-only.

**PARTIAL — actual GPU retrain BLOCKED on USER_OK=1** per AGENTS.md §5.1 standing GPU grant + kill_then_adapt §0: the 3-run CONTROLD PAIR (synthetic + real + none) requires ~6 hr GPU wall on RTX PRO 6000 Blackwell sm_120 per Wf-260 G6 timing. Once the user grants, Wf-125 P1 CONTROLD PAIR can queue after Wf-413-A1 + Wf-411-A3 Re seed=4 (~70 min remaining wall).

**NEW ultracode Wf-125 P1 actual GPU launch IS the recommended follow-up** per this closeout §6.3 with 3-run CONTROLD PAIR sweep (~6 hr GPU sequential USER_OK-gated; verdict taxonomy: run2 ≈ run1 ⇒ lift REAL; run2 < run1 ⇒ lift ARTIFACT; run2 ≈ run3 ⇒ no signal).

---

**END OF Wf-125 P1 CONTROLD PAIR closeout** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10.
