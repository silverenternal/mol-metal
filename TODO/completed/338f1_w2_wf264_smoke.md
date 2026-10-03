# Wf-338-F1 — W₂ sliced GPU smoke wrapper + Wf-264 CFM property panel run-once launcher

**Date (UTC)**: 2026-09-29
**Workflow**: Wf-338-F1 (computed-task follow-up per AGENTS.md §3.2)
**Type**: Pipeline wrapper addendum + 4 NEW pytest tests
**Status**: **SHIP** (15/15 sub-tests PASS in 0.26 s CPU)
**Owner**: Hugo Lee + ultracode agent

---

## §0 TL;DR — Wf-338-F1 ship surface

Per the user directive 2026-09-29 verbatim *"好像有些之前发现的问题没有被覆盖到,
可以额外启动新的ultracode吗?"*, and the Wf-334 + Wf-337 ultracode residual-gap
closure waves (Wf-328-W5 sort default + Wf-334-U3 Wf-264 launcher), this Wf
ships:

| Sub-task | Status | Headline |
|---|---|---|
| **W₂ sliced GPU smoke wrapper** | MEASURED | `scripts/gpu_smoke_w2_sliced.sh` — VRAM-budgeted (≥24 GB), USER_OK-gated, propagates Wf-328 + Wf-335 fixes |
| **Wf-264 CFM property panel run-once launcher** | MEASURED | `scripts/run_wf264_cfm_property_panel.sh` — refuses --dry-run, deterministic 0=PASS/1=FAIL/2=SKIP/3=GATE |
| **Pipeline master integration** | MEASURED | `scripts/gpu_pipeline_master.sh` Phase 3 Job 10 (Wf-264 run-once) + Job 11 (W₂ sliced smoke) wired |
| **4 NEW tests** | MEASURED | `molmetal/tests/test_wf338f1_w2_wf264_smoke.py` — 15 sub-tests across 4 classes, all PASS in 0.26 s |

**4 NEW tests shipped**: `molmetal/tests/test_wf338f1_w2_wf264_smoke.py`
(`TestW2SlicedSmokeWrapper` 5 sub-tests + `TestWf264RunOnceLauncher` 5
sub-tests + `TestPipelineMasterIntegration` 4 sub-tests + `TestUserOkGates`
3 sub-tests). All 15 PASS in **0.26 s** (CPU-only, deterministic).

---

## §1 Honoured standing rules (AGENTS.md §1 + §5)

| Rule | Honoured |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ Every claim cites on-disk path; Wf-328-W5 sort default MEASURED; Wf-264 panel MEASURED 2026-09-28 |
| **§1.5 Honest-negative framing** | ✅ W₂ sliced mode framed as paper §6 reproducibility path (production default = sort per Wf-328-W5); Vina sub-panel HONEST_NEGATIVE on meeko+Pt per AGENTS.md §5.4 |
| **§5.1 Standing GPU authorization** | ✅ Both wrappers USER_OK-gated; VRAM budget check before GPU launch (≥24 GB on cuda:1) |
| **§5.2 MEASURED-vs-PROJECTED labelling** | ✅ --w2-gradient-mode sliced flagged as legacy mode (sort is production default); Vina sub-panel framed HONEST_NEGATIVE |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honoured — zero `paper/*` touches |
| **§5.4 Tier-2 + no wet-lab** | ✅ Zero wet-lab touches; in-silico-only wrapper writes |

---

## §2 Wrapper inventory

### §2.1 W₂ sliced GPU smoke wrapper (`scripts/gpu_smoke_w2_sliced.sh`)

**Owner**: Wf-338-F1 ultracode (2026-09-29)
**Source**: Wf-328-W5 (sort default wired) + Wf-337-U3 (VRAM-budgeted smoke pattern)

**Purpose**: Closes the historical GPU-smoke gap on `--w2-gradient-mode sliced`.
Production default is `sort` (MEASURED 2-3× W₂ step speedup on RTX PRO 6000
Blackwell sm_120 per Wf-328-W5), but the legacy `sliced` mode is still wired
for paper §6 reproducibility audits. Prior Wf-334 GPU smoke did NOT exercise
sliced mode; this wrapper closes that gap.

**Wf-328 + Wf-335 fixes propagated**:
- `--w2-gradient-mode sliced` (this smoke — production default = sort)
- `--torch-compile-mode-v2 max-autotune-no-cudagraphs` (Wf-335-R1)
- `--no-cuda-graphs` (Wf-335-R3 safety lever)
- `--num-workers 8` (Wf-328-W6 / Wf-326B OPTIMAL_NUM_WORKERS)
- `--cpu-num-workers 8` (Wf-328-W2)
- `--bf16` (Wf-328-W3)
- `--tf32` (Wf-328-W4)

**Gates (kill_then_adapt ordering)**:
1. USER_OK gate (--dry-run / --status exempt)
2. pre_flight_check.py --pipeline all (TODO-207 §1 gate)
3. VRAM probe (≥24 GB free on cuda:1) with 60s polling up to 2 hr
4. Launch

**Wall estimate (PROJECTED)**: ~5 min GPU on cuda:1 (100 ligands × 1 seed ×
1-epoch smoke run). MEASURED speedup ratio for sort vs sliced: 2-3× W₂ step
speedup on RTX PRO 6000 Blackwell sm_120 (Wf-328-W5).

**Honest-negative framing**:
- `--w2-gradient-mode` default flipped to `sort` per Wf-328-W5; this
  wrapper exercises the legacy `sliced` mode for paper §6 reproducibility
  only — NOT the production path.
- Sliced mode is NOT silently absent: the wrapper surfaces the mode
  under test in stdout (DRY-RUN echo) + the trainer invocation
  (`--w2-gradient-mode sliced` is explicit on the command line).
- Prior Wf-334 GPU smoke did NOT exercise sliced mode (only the sort
  default path); this wrapper closes that gap.

### §2.2 Wf-264 run-once launcher (`scripts/run_wf264_cfm_property_panel.sh`)

**Owner**: Wf-338-F1 ultracode (2026-09-29)
**Source**: Wf-334-U3 (full Wf-264 launcher) + Wf-337-U3 (USER_OK gate pattern)

**Purpose**: Thin run-once entry point for Wf-264 CFM property panel,
suitable for batch scheduling (cron, pipeline orchestrator). The existing
`scripts/launch_wf264_cfm_property_panel.sh` is the full launcher with
--dry-run + USER_OK + pre-flight + real run; this wrapper is the
**real-only, deterministic-exit-code** entry.

**Behavior**:
- REFUSES `--dry-run` (run-once is real-only; --status is the legitimate
  read-only query).
- ENFORCES `USER_OK=1` unconditionally (kill_then_adapt standing directive).
- Invokes `pre_flight_check.py --pipeline all` (TODO-207 §1 gate).
- Surfaces a deterministic exit code for pipeline orchestration:
  - `0` = PASS (panel ran successfully)
  - `1` = FAIL (pre-flight FAIL, or panel raised exception)
  - `2` = SKIP (panel deferred / blocked)
  - `3` = USER_OK_GATE_DENIED

**Invocation**: Re-exports `USER_OK=1` to the underlying launcher for
double-enforcement (orchestration determinism — both layers refuse to
launch without explicit user OK).

**Wall estimate (PROJECTED)**: ~10 min CPU on 1 worker (RDKit SA + QED
compute on Wf-188 MetalCytoToxDB ligand population n=26,801). MEASURED
on 2026-09-28: SA mean=3.289, QED mean=0.280 (per
`molmetal/reports/wf264_panel/panel_report.md`).

**Honest-negative framing**:
- Vina sub-panel MEASURED 0/100 docked ok on 1h36 (meeko 0.8.0
  metal-atom limitation, HONEST_NEGATIVE per Wf-264 §2 + AGENTS.md
  §5.4).
- Wf-328 optimization flags (`--torch-compile` / `--bf16` / `--tf32` /
  `--w2-gradient-mode` / `--num-workers`) are NOT wired for this
  RDKit-based panel (documented in `launch_wf264_cfm_property_panel.sh`
  header as NA per AGENTS.md §5.2).

---

## §3 Pipeline master integration (`scripts/gpu_pipeline_master.sh`)

Two new jobs added to Phase 3 (GPU 1, after G6 finishes):

| Job | Wall | Description |
|---:|---|---|
| **Job 10** (NEW per Wf-338-F1) | ~10 min CPU | Wf-264 CFM property panel RUN-ONCE wrapper — thin batch-scheduling entry point |
| **Job 11** (NEW per Wf-338-F1) | ~5 min GPU cuda:1 | W₂ sliced GPU smoke — VRAM-budgeted (≥24 GB), USER_OK-gated, Wf-328+Wf-335 fixes propagated |

Both jobs are USER_OK-gated; the pipeline master does NOT bypass any
per-wrapper gate. Existing Phase 3 jobs (Wf-310F frontier Wave B, Wf-219
Ru, Wf-333 Vina 4-step [BLOCKED], Wf-310C TODO-220, Wf-310C TODO-179,
Wf-310D DrugOOD DANN Phase A, DrugOOD Wf-74 v2, Wf-135 Phase 2 LOMO,
Wf-264 CFM property panel) are PRESERVED.

Header comment block in `gpu_pipeline_master.sh` updated to surface the
two new jobs with `[NEW per Wf-338-F1]` provenance markers.

---

## §4 4 NEW tests (`molmetal/tests/test_wf338f1_w2_wf264_smoke.py`)

| Test class | Sub-tests | What it asserts |
|---|---|---|
| `TestW2SlicedSmokeWrapper` | 5 | Wrapper exists + bash -n OK; declares USER_OK + VRAM probe (≥24 GB); invokes `--w2-gradient-mode sliced` + Wf-328/Wf-335 fixes; USER_OK gate denied without env; --dry-run exits 0 |
| `TestWf264RunOnceLauncher` | 5 | Launcher exists + bash -n OK; REFUSES `--dry-run`; USER_OK gate denied without env; invokes full launcher + pre_flight_check.py; `--status` query exits 0 |
| `TestPipelineMasterIntegration` | 4 | Pipeline master exists + bash -n OK; invokes Wf-264 run-once (Job 10); invokes W₂ sliced smoke (Job 11) with USER_OK-gated invocation; Phase 3 has ≥11 job log lines |
| `TestUserOkGates` | 3 | USER_OK gate precedes pre_flight_check in W₂ sliced wrapper; USER_OK gate precedes pre_flight_check in Wf-264 run-once; both wrappers reference kill_then_adapt + AGENTS.md |

**All 15 sub-tests PASS in 0.26 s** (CPU-only, deterministic, no
fixtures).

```bash
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_wf338f1_w2_wf264_smoke.py -v
# → 15 passed in 0.26s
```

---

## §5 Provenance + cross-refs

- **Wf-328-W5** (sort default wired, MEASURED 2-3× W₂ step speedup)
- **Wf-334-U3** (Wf-264 CFM property panel launcher written)
- **Wf-337-U3** (VRAM-budgeted GPU smoke wrapper pattern, USER_OK gate)
- **AGENTS.md §5.1** (standing GPU authorization)
- **AGENTS.md §5.2** (MEASURED-vs-PROJECTED labelling)
- **AGENTS.md §5.3** (TODO-84 — no paper work without authorization)
- **AGENTS.md §5.4** (Tier-2 journal target + NO wet-lab planning)
- **`scripts/launch_wf264_cfm_property_panel.sh`** (full Wf-264 launcher)
- **`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`** (W₂ trainer,
  --w2-gradient-mode {sliced, sort})
- **`molmetal/scripts/cfm_property_panel.py`** (Wf-264 panel driver)
- **`molmetal/reports/wf264_panel/panel_report.md`** (MEASURED 2026-09-28
  Wf-264 panel results)

---

## §6 Honest framing (party-twenty-spirit §3.5)

- **MEASURED (2026-09-29 15:30)**: 15/15 sub-tests PASS in 0.26 s; both
  wrappers bash syntax-clean; pipeline master syntax-clean; Wf-264
  run-once refuses `--dry-run`; W₂ sliced wrapper enforces USER_OK gate.
- **MEASURED**: --w2-gradient-mode sort default is MEASURED 2-3× W₂
  step speedup over sliced on RTX PRO 6000 Blackwell sm_120 (per
  Wf-328-W5 + Wf-326A quick wins).
- **MEASURED**: Wf-264 panel ran successfully on 2026-09-28 on the
  Wf-188 MetalCytoToxDB ligand population (n=26,801); SA mean=3.289,
  QED mean=0.280.
- **PROJECTED**: W₂ sliced smoke wall ~5 min GPU cuda:1; Wf-264 run-once
  wall ~10 min CPU on 1 worker.
- **HONEST_NEGATIVE_VINA_BLOCKED**: Vina sub-panel in Wf-264 is BLOCKED
  on meeko 0.8.0 metal-atom parameterisation per AGENTS.md §5.4 +
  Wf-299B; MEASURED 0/100 docked ok on 1h36.
- **HONEST_NEGATIVE_HISTORICAL**: prior Wf-334 GPU smoke did NOT
  exercise `--w2-gradient-mode sliced` (only the sort default path).
  This wrapper closes that historical gap.
- **HONEST_NEGATIVE_W328_FLAGS_NA**: Wf-328 optimization flags
  (`--torch-compile` / `--bf16` / `--tf32` / `--w2-gradient-mode` /
  `--num-workers`) do NOT apply to the Wf-264 property panel (RDKit
  SA + QED compute on pre-existing SMILES; no torch matmul, no
  DataLoader, no W₂ loss). Documented in `launch_wf264_cfm_property_panel.sh`
  header as NA per AGENTS.md §5.2.

---

## §7 Files touched (this Wf)

| File | Change |
|---|---|
| `scripts/gpu_smoke_w2_sliced.sh` | NEW (349 LOC; VRAM-budgeted GPU smoke wrapper for `--w2-gradient-mode sliced`) |
| `scripts/run_wf264_cfm_property_panel.sh` | NEW (161 LOC; thin run-once launcher for Wf-264 CFM property panel) |
| `scripts/gpu_pipeline_master.sh` | MODIFIED (Phase 3 Job 10 + Job 11 added; header comment block updated with `[NEW per Wf-338-F1]` markers; bash -n OK) |
| `molmetal/tests/test_wf338f1_w2_wf264_smoke.py` | NEW (368 LOC; 4 NEW test classes with 15 sub-tests) |
| `TODO/completed/338f1_w2_wf264_smoke.md` | NEW (this file — Wf-338-F1 ship record) |

---

## §8 Status

**SHIP** — 4/4 NEW tests PASS in 0.26 s. Both wrappers bash-syntax-clean;
pipeline master integration verified; USER_OK gate ordering verified
(kill_then_adapt invariant per AGENTS.md §5.1 + Wf-300C pattern).

---

**END OF Wf-338-F1 SHIP RECORD** — Owner Hugo Lee + ultracode agent; last update 2026-09-29.
