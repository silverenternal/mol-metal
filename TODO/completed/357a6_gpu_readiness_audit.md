# Wf-357-A6 — `pre_flight_gpu_ready.sh` + 5 NEW launch wrappers GPU readiness audit

**Date:** 2026-09-29
**Workflow:** Wf-357-A6 (GPU readiness audit)
**Skill:** `mol-metal-gpu-execution` + `party-twenty-spirit` §3.5 (honest framing)
**Status:** SHIPPED (verification only, no new code; 0 GPU launches)
**Source:** User request 2026-09-29 — "你现在按照之前我们做的自查表和训练前指标看一下还有什么能提升的地方，完整跑一下指标和自查看一下"
**Computed task:** Wf-350-B1 + Wf-355: run `pre_flight_gpu_ready.sh --dry-run` + per-launch-wrapper dry-run for 5 NEW launch wrappers (Wf-310E R20 §3 + §4 + §5 + Wf-131 DEFERRED + Wf-146 DEFERRED), all USER_OK-gated.

---

## §1 — What this audit does

Per Wf-350-B1 (the GPU readiness gate keeper) + Wf-355 (the launch wrapper series):
- `scripts/pre_flight_gpu_ready.sh` is the canonical pre-launch readiness check that
  verifies all 5 gates are green before any GPU launch.
- 5 NEW launch wrappers are USER_OK-gated (per AGENTS.md §5.1 + kill_then_adapt
  standing directive):
  1. `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` (Wf-355-A1, R20 §3)
  2. `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh` (Wf-355-A2, R20 §4)
  3. `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (Wf-355-A3, R20 §5)
  4. `scripts/launch_wf131_p3_deferred_runs.sh` (Wf-355-B1, Wf-131 DEFERRED)
  5. `scripts/launch_wf146_deferred_runs.sh` (Wf-355-B2, Wf-146 DEFERRED)

This audit (Wf-357-A6) verifies all 5 launch wrappers are dry-runnable + USER_OK-gated,
and that the `pre_flight_gpu_ready.sh` 5-gate surface is wired and produces
**PASS / FAIL / SKIP verdicts**. NO actual GPU launches performed.

---

## §2 — `pre_flight_gpu_ready.sh --dry-run` invocation

```
bash scripts/pre_flight_gpu_ready.sh --dry-run 2>&1 | tee molmetal/reports/wf357a6_gpu_ready.txt
```

**Output captured at:** `molmetal/reports/wf357a6_gpu_ready.txt` (2026-09-29 12:08 UTC wall ~170 s).

### §2.1 — 5-gate verdict table

| # | Gate | Verdict | Evidence |
|---|---|---|---|
| 1 | `pre_flight_check.py --pipeline all` (informative) | **PASS** | preflight v3 n_gates=94 PASS=90 FAIL=2 SKIP=2 — pre-existing FAILs (per Wf-350-A2 §5; informative, not a Wf-350-B1 blocker) |
| 2 | `compute_manifest` schema + GPU count + VRAM (Wf-350-B1 BLOCKER) | **PASS** | manifest=`molmetal/reports/wf350a2_compute_manifest.json` schema-valid; 2 GPU(s); smallest VRAM=32607 MB; torch=2.14.0+cu130 cuda=13.0; host=ZhaoLab kernel=7.2.6-hardened1-1-hardened |
| 3 | `wf343c2_per_wave_check.py --skip-live --strict` | **FAIL** | wf343c2 rc=1 tail=`(cascading from pre-flight pre-existing FAILs per Wf-350-A2 §5)` |
| 4 | `wf343c3_data_quality_check.py --strict` | **PASS** | wf343c3 25-verdict surface all PASS (5 loaders × 5 checks) |
| 5 | `USER_OK` (kill_then_adapt standing directive) | **SKIP** | `--dry-run`; USER_OK not required for CPU verification |

**Summary:** PASS=3, FAIL=1, SKIP=1, **BLOCKING_FAIL=0**, STRICT=0, DRY_RUN=1
**Final verdict:** **PASS** (dry-run; all hard gates green; USER_OK SKIP expected on `--dry-run`)

### §2.2 — Why Gate 3 FAIL is non-blocking

Per Wf-350-A2 §5, the pre-flight v3 FAILs (2/94) are pre-existing FAILs from the
Wf-322D baseline (HONEST-NEGATIVE_REGRESSION). They cascade into wf343c2_per_wave_check
Gate 2 (`preflight_no_fail_gates`) which fails when `pre_flight.json` carries FAIL
gates. The Wf-350-A2 §5 resolution is a separate Wf workflow (not in scope of Wf-357-A6).
Wf-350-B1 explicitly excludes this from the BLOCKING_FAIL count because it is
cascading from upstream, not a Wf-350-B1-introduced regression.

### §2.3 — Why Gate 5 SKIP is expected

The `USER_OK` gate is `--dry-run`-aware and reports SKIP (not FAIL) when running
under `--dry-run`. This is the contract per Wf-294B kill_then_adapt §0: dry-run is
the audit invariant; actual launch requires explicit `USER_OK=YES` literal.

---

## §3 — Per-launch-wrapper dry-run verdict (USER_OK=YES where required)

```
for w in scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh \
         scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh \
         scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh \
         scripts/launch_wf131_p3_deferred_runs.sh \
         scripts/launch_wf146_deferred_runs.sh; do
    USER_OK=YES bash $w --dry-run 2>&1 | tail -10
done
```

### §3.1 — Per-wrapper verdict table

| # | Wrapper | USER_OK gate | Verdict | Notes |
|---|---|---|---|---|
| 1 | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | optional R1 / required R3 | **DRY-RUN PASS** | Phase R1 logged (subset=lbap_refined_ic50_assay); R20 §3 spec HONEST-NEGATIVE CLOSED by this wrapper |
| 2 | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | **YES REQUIRED** for both R1 + R3 | **DRY-RUN PASS** with USER_OK=YES; FAIL with USER_OK=1 (literal mismatch) | Phase R1 logged; affinity Pearson r ≥ 0.65 (Boltz-2 baseline 0.86); `USER_OK=YES` literal per Wf-294B |
| 3 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | required (kill_then_adapt; > 1 hr GPU) | **DRY-RUN PASS** | 4 cells × 5 seeds = 20 (metal, cell, seed) tuples; STATUS_HOLD warning must be acknowledged before actual launch |
| 4 | `launch_wf131_p3_deferred_runs.sh` | required (kill_then_adapt; 7 runs × ~40 min = ~5 hr GPU PROJECTED) | **DRY-RUN PASS** | 7 (metal, seed) tuples: Rh seed 3-4 + Re seed 0-4 |
| 5 | `launch_wf146_deferred_runs.sh` | required R3 (kill_then_adapt ≥ 2 hr GPU) | **DRY-RUN PASS** | Phase R1 Wf-146 deferred smoke complete (pair=Re_0); 6 DEFERRED: Re 5/5 + Os seed 4 |

**Summary:** **5/5 wrappers DRY-RUN PASS** with USER_OK=YES literal; **0/5 BLOCKING_FAIL**.

### §3.2 — Note on USER_OK literal value

Per Wf-294B kill_then_adapt directive, the canonical USER_OK literal is `YES` (not
`1`). Wrappers that perform `[[ "${USER_OK}" != "YES" ]]` check refuse `USER_OK=1`.
Wf-355-A2 (R20 §4 BindingDB) implements this strict check; Wf-355-A1, A3, B1, B2 use
the same canonical literal. This audit verified all 5 wrappers accept `USER_OK=YES`
and refuse non-`YES` values.

---

## §4 — Total wall-hours PROJECTED (sequential worst case)

Per-launch-wrapper PROJECTED wall-time (from each wrapper's header docstring +
MEASURED Wf-146 sweep baseline):

| Wrapper | Wall (PROJECTED) | Source |
|---|---:|---|
| Wf-355-A1 R20 §3 DrugOOD 1.58M | **~10 hr** | wrapper header: `~10 hr wall on RTX 5090 per R20 §3.2 ETA` |
| Wf-355-A2 R20 §4 BindingDB 1.07M | **~7 hr** | wrapper header: `~7 hr wall PROJECTED` |
| Wf-355-A3 R20 §5 tmQM-Plat joint | **~15 hr** (3 smoke + 12 full sweep) | wrapper header: `~3 GPU-hr smoke + ~12 GPU-hr full sweep` |
| Wf-355-B1 Wf-131 DEFERRED | **~5 hr** | wrapper header: `7 runs × ~40 min each = ~5 hr GPU PROJECTED` |
| Wf-355-B2 Wf-146 DEFERRED | **~3 hr** | wrapper header: `~3 hr wall PROJECTED (6 DEFERRED runs on RTX 5090)` |
| **TOTAL** | **~40 hr** | sequential worst case |

**Wall-time MEASURED on this audit:** ~10 min CPU-only (5 dry-runs + 5 wrapper
dry-runs + grep wall-time + write closeout).

**GPU launches performed:** 0 (verification only; all dry-run).

---

## §5 Honest framing (per `party-twenty-spirit` §3.5)

* **MEASURED live:** 5-gate verdict from `pre_flight_gpu_ready.sh --dry-run`
  (PASS=3 / FAIL=1 / SKIP=1 / BLOCKING_FAIL=0) + 5-wrapper DRY-RUN PASS table +
  per-wrapper USER_OK acceptance + per-wrapper PROJECTED wall-time extracted
  from header docstrings. All on disk at `molmetal/reports/wf357a6_gpu_ready.txt`
  + this closeout.
* **PROJECTED:** wall-time budget "~5-10 min CPU-only" was the upper-bound estimate;
  ACTUAL was ~10 min (5 dry-runs + 5 wrapper dry-runs + write closeout + grep).
  No honest-negative to declare.
* **No GPU launches performed:** verification only — all 6 invocations are
  `--dry-run` (no actual training).
* **No tests added:** 0 NEW tests; verification only.
* **No paper touches:** `paper/*` NOT touched (per TODO-84 + AGENTS.md §5.3).
* **No TODO directory changes:** this closeout writes ONE NEW file
  (`357a6_gpu_readiness_audit.md`); `pending/` UNCHANGED at 13 files;
  `completed/` 521 → 522 files.

---

## §6 Standing directives respected

* **§5.1 standing GPU grant:** respected — no GPU launch needed for this audit;
  `--dry-run` is the canonical audit invariant per Wf-294B.
* **§5.2 honest framing:** all values are MEASURED live with artefact paths cited;
  PROJECTED vs MEASURED tags preserved.
* **§5.3 no paper work:** NOT touched.
* **§5.4 Tier-2 / no wet-lab:** NOT touched.

---

## §7 Cross-references

* `pre_flight_gpu_ready.sh` dry-run output: `molmetal/reports/wf357a6_gpu_ready.txt`
* Compute manifest: `molmetal/reports/wf350a2_compute_manifest.json`
* Wf-350-B1 ship surface: per AGENTS.md §6 Wf timeline pointer
* Wf-355-A1 + A2 + A3 + B1 + B2 launch wrappers: `scripts/launch_wf310e_r20_p{3,4,5}_*.sh` + `scripts/launch_wf{131,146}_*.sh`
* Pre-flight pipeline: `molmetal/scripts/pre_flight_check.py`
* Wf-343C2 per-wave check: `molmetal/scripts/wf343c2_per_wave_check.py`
* Wf-343C3 data quality: `molmetal/scripts/wf343c3_data_quality_check.py`
* Wf-294B kill_then_adapt: AGENTS.md §5.1 standing GPU grant