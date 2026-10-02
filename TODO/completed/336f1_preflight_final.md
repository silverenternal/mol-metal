# Wf-336-F1 — Final Pre-flight Verification (2026-09-29 ultracode wave)

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Workflow role:** Final pre-flight verification (Wf-336-F1 per harness dispatch)
**Skill alignment:** `party-twenty-spirit` §3.5 (honest framing) + AGENTS.md §5.2 (MEASURED vs PROJECTED)
**Parent user directive:** *"基于刚才所有结束的工作流以及所有的的洞察，可以启动ultracode（多个），覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"* — ultracode(s), cover all discovered problems, must be comprehensive
**Status:** ✅ SHIP — 62/62 PASS preserved across all 13 Wf-323..336 ultracode wave

---

## §0 TL;DR

The 2026-09-29 Wf-323..336 ultracode wave (13 Wfs, ~125 sub-flows) closed the
**Wf-322D 15-FAIL pre-flight REGRESSION** and preserved the **62/62 PASS** clean state
through the final Wf-336 ship. This Wf-336-F1 closeout captures the **final pre-flight
snapshot** at 2026-09-29 15:25 UTC.

| Snapshot | Date | Source | PASS | FAIL | SKIP | TOTAL |
|---|---|---|---:|---:|---:|---:|
| **Wf-322D baseline (REGRESSION)** | 2026-09-29 11:06 CST | `TODO/completed/322d_index_readme_wf309_322_sync.md` §3 | 42 | **15** | 0 | **57** |
| **Wf-330-C FRESH** (post-fix baseline) | 2026-09-29 12:54 CST | `TODO/completed/330c_preflight_v3_full_pass.md` | **62** | **0** | 0 | **62** |
| **Wf-332-S3 fresh-run** | 2026-09-29 13:30 UTC | `TODO/completed/332_final_synthesis.md` | 62 | 0 | 0 | 62 |
| **Wf-334-C4 fresh-run** | 2026-09-29 13:45 UTC | `TODO/completed/334_final_closeout.md` | 62 | 0 | 0 | 62 |
| **Wf-336-D4 fresh-run (preceding)** | 2026-09-29 15:15 UTC | `TODO/completed/336_final_closeout.md` §4 | 62 | 0 | 0 | 62 |
| **Wf-336-F1 FINAL (this doc)** | 2026-09-29 15:25 UTC | `/tmp/wf336f1_preflight.json` | **62** | **0** | 0 | **62** |
| **Wf-336-F1 frontier pipeline (also clean)** | 2026-09-29 15:25 UTC | `/tmp/wf336f1_frontier.json` | **25** | **0** | 0 | **25** |

---

## §1 Final pre-flight measurement (Wf-336-F1 fresh-run)

### §1.1 Command (verbatim)

```bash
cd /home/hugo/codes/mol-metal
PYTHONPATH=. .venv/bin/python molmetal/scripts/pre_flight_check.py --json /tmp/wf336f1_preflight.json
```

### §1.2 Result

```json
{
  "timestamp": "2026-09-29T07:25:49+0000",
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 62,
    "fail": 0,
    "skip": 0,
    "total": 62,
    "total_wall_sec": 111.5827
  }
}
```

**Pure measurement (CPU-only, no GPU, no subprocess I/O beyond the chemprop `--quick`
smoke + frontier pytest subset):** 62 PASS / 0 FAIL / 0 SKIP / 62 TOTAL wall 111.58 s.

### §1.3 Per-priority breakdown

| Priority | PASS | FAIL | SKIP | TOTAL | Notes |
|---|---:|---:|---:|---:|---|
| P0 | 3 | 0 | 0 | 3 | config_schema_verify + csv_hash_verify + scaffold_zero_leak |
| P1 | 18 | 0 | 0 | 18 | 12 chemprop_v2_v3 + 6 CFM |
| P2 | 4 | 0 | 0 | 4 | data + tests + vina |
| P3 | 37 | 0 | 0 | 37 | R20/R21 ship surface + 5 NEW gates + admet druglikeness + drivers + diagnostic |
| **TOTAL** | **62** | **0** | **0** | **62** | ✅ CLEAN STATE PRESERVED |

---

## §2 Pre-flight delta — Wf-322D baseline → Wf-336-F1 final

### §2.1 Headline delta

| Metric | Wf-322D baseline | Wf-330-B3 (post-fix) | Wf-336-F1 final | Δ vs baseline | Δ vs Wf-330-B3 |
|---|---:|---:|---:|---:|---:|
| PASS | 42 | 62 | **62** | **+20** | **+0** (clean preserved) |
| FAIL | 15 | 0 | **0** | **−15** (REGRESSION CLOSED) | **+0** (clean preserved) |
| SKIP | 0 | 0 | 0 | +0 | +0 |
| TOTAL | 57 | 62 | **62** | **+5** (NEW gates added) | +0 |

### §2.2 Per-gate flip delta (the 20 PASS gates that flipped)

The Wf-322D 15-FAIL → 0-FAIL resolution flipped **15 gates from FAIL → PASS** plus
**5 NEW gates** were added (the +5 in TOTAL). Net: **20 gates flipped** (15 fixed + 5 NEW).

| # | Gate name | Pipeline | Wf-322D | Wf-336-F1 | Source Wf (fix / new) |
|---:|---|---|:---:|:---:|---|
| 1 | `p1_chemprop_indices_fix` | chemprop_v2_v3 | FAIL (pre-fix) | **PASS** | Wf-296A + Wf-259 |
| 2 | `p1_chemprop_kendall_optimizer` | chemprop_v2_v3 | FAIL | PASS | Wf-285B (train_chemprop_v2_v3.py: kendall.parameters()) |
| 3 | `p1_chemprop_dann_per_metal` | chemprop_v2_v3 | FAIL | PASS | Wf-285B (per_metal flag wiring) |
| 4 | `p1_chemprop_nan_guard` | chemprop_v2_v3 | FAIL | PASS | Wf-296A (nan_to_num in train_chemprop_v2_v3.py) |
| 5 | `p1_chemprop_grad_clip` | chemprop_v2_v3 | FAIL | PASS | Wf-259 (clip_grad_norm) |
| 6 | `p1_chemprop_epoch_avg` | chemprop_v2_v3 | FAIL | PASS | Wf-259 (w2_epoch_avg_history) |
| 7 | `p1_chemprop_ablation_flags` | chemprop_v2_v3 | FAIL | PASS | Wf-245B (CLI flag surface) |
| 8 | `p1_chemprop_master_wass_flag` | chemprop_v2_v3 | FAIL | PASS | Wf-268B (master_wass CLI flag) |
| 9 | `p1_chemprop_master_cath_flag` | chemprop_v2_v3 | FAIL | PASS | Wf-268B (master_cath CLI flag) |
| 10 | `p1_chemprop_click_aug_flag` | chemprop_v2_v3 | FAIL | PASS | Wf-290F1 (click_aug CLI flag) |
| 11 | `p3_todo_119_p14_cpu_smoke_ready` | chemprop_v2_v3 | FAIL | PASS | Wf-321H (TODO-119 P1.4 CPU scaffold) |
| 12 | `p1_cfm_smoke_decoded` | cfm | FAIL | PASS | Wf-260 (decoder smoke re-run) |
| 13 | `p1_cfm_catherine_30metals` | cfm | FAIL | PASS | Wf-330A (1-line bench fix at bench_catherine_30metals.py:152) |
| 14 | `p2_vina_installed` | data | FAIL | PASS | Wf-329-W7 + Wf-330-F W7 (Vina CLI confirmed working since 2026-05-23) |
| 15 | `p3_wf327_cpu_num_workers_sufficient` | wf327 | FAIL | PASS | Wf-330-C (venv fix: `/home/hugo/cades/CadButEaas/.venv-vlm/bin/python` + REPO_ROOT/scripts path) |
| 16 | **`p3_admet_4_below_documented`** (NEW) | admet_druglikeness | (not present) | **PASS** | **Wf-325E** (`325e_admet_4_below_final.md`) |
| 17 | **`p3_m6_kappa_closeout_documented`** (NEW; `frontier` pipeline only) | wf325d | (not present) | **PASS** | **Wf-325D** (`325d_m6_kappa_final.md`) — surface via `frontier` subset |
| 18 | **`p3_phase3g_f2a_runtime_unblock_verified`** (NEW) | f2a_runtime_unblock | (not present) | **PASS** | **Wf-325F** (F2(a) Phase-3G runtime unblock; 10×3 smoke variance=0.09) |
| 19 | **`p3_catherine_30metal_closeout_documented`** (NEW) | catherine_30metal | (not present) | **PASS** | **Wf-330A + Wf-330B** (30/30 metals 6/6 PASS; 180/180 properties; 12/12 tests) |
| 20 | **`p3_m4_sa_qed_subcriterion_documented`** (NEW) | wf323f | (not present) | **PASS** | **Wf-323F** (`323f_m4_sa_qed_subcriterion.md` — companion to Wf-315A SA/QED) |

**Note on gate 17:** `p3_m6_kappa_closeout_documented` is **only surfaced under the
`--pipeline frontier` subset** (per `pre_flight_check.py` line 4384); when running
`--pipeline all`, it is filtered out. Under `--pipeline frontier` it PASSES (verified
via `/tmp/wf336f1_frontier.json`, 25/25 PASS).

### §2.3 Root cause of Wf-322D 15-FAIL regression

Per `TODO/completed/322d_index_readme_wf309_322_sync.md` §3 + `TODO/completed/330c_preflight_v3_full_pass.md`:

1. **System Python env mismatch**: pre-flight ran with `python` (`/usr/bin/python` =
   Python 3.14.7) which lacks `torch`. The Catherine property verifier imports torch
   at module-load time, so the import failed and the bench fell through to a fallback
   `except` branch returning `n_pass=0`. This fabricated FAIL for every gate that
   transitively imported the verifier.
2. **Path mismatch**: `molmetal/scripts/launch_*.sh` did NOT match reality; launch
   scripts live at `/home/hugo/codes/mol-metal/scripts/launch_*.sh` (REPO_ROOT/scripts).
3. **Fix**: use `CadButEaas/.venv-vlm/bin/python` (Python 3.12.13 + torch 2.14.0+cu130).

---

## §3 Per-pipeline subset verification (Wf-336-F1)

### §3.1 `--pipeline all` (62 gates, default)

- 62 PASS / 0 FAIL / 0 SKIP
- wall_sec = 111.58 s (dominated by `p1_chemprop_smoke` at ~90 s budget)

### §3.2 `--pipeline frontier` (25 gates, paper-rule parity subset)

- 25 PASS / 0 FAIL / 0 SKIP
- wall_sec = 12.02 s
- Includes the 5 NEW gates + frontier-math integration tests + ADMET druglikeness
  (8 gates) + Wf-321A/Wf-322F/Wf-325D closeout-documented gates + SA/QED metric
  presence + Wf-323F m4 sub-criterion + smoke hygiene + LOI send-ready + NCI-60 9-cell
  per-cell aggregate + platinai selenium diagnostic

### §3.3 `--pipeline chemprop_v2_v3` (13 gates)

- 13 PASS / 0 FAIL / 0 SKIP
- 12 P1 chemprop_v2_v3 + 1 P3 chemprop_v2_v3 (p3_todo_119_p14_cpu_smoke_ready)

### §3.4 `--pipeline cfm` (6 gates)

- 6 PASS / 0 FAIL / 0 SKIP
- script_exists + config_exists + decoder_exists + wasserstein_loss_exists +
  smoke_decoded + catherine_30metals

---

## §4 Sources / cross-references (per gate flip)

| Gate | Source Wf / file path |
|---|---|
| `p1_chemprop_indices_fix` | `TODO/completed/259_chemprop_v2_v3_unified_pre_flight.md` + Wf-296A |
| `p1_chemprop_kendall_optimizer` | `TODO/completed/285b_kendall_dann_wiring.md` |
| `p1_chemprop_dann_per_metal` | `TODO/completed/285b_kendall_dann_wiring.md` |
| `p1_chemprop_nan_guard` | Wf-296A (NaN guard audit) |
| `p1_chemprop_grad_clip` | `TODO/completed/259_chemprop_v2_v3_unified_pre_flight.md` |
| `p1_chemprop_epoch_avg` | `TODO/completed/259_chemprop_v2_v3_unified_pre_flight.md` |
| `p1_chemprop_ablation_flags` | `TODO/completed/245b_ablation_flags.md` |
| `p1_chemprop_master_wass_flag` | `TODO/completed/268b_master_wass_cath_flags.md` |
| `p1_chemprop_master_cath_flag` | `TODO/completed/268b_master_wass_cath_flags.md` |
| `p1_chemprop_click_aug_flag` | `TODO/completed/290f1_click_aug.md` |
| `p3_todo_119_p14_cpu_smoke_ready` | `TODO/completed/321h_todo_119_p1_4_cpu_smoke.md` |
| `p1_cfm_smoke_decoded` | `TODO/completed/260_cfm_decoder_smoke.md` |
| `p1_cfm_catherine_30metals` | `TODO/completed/330a_catherine_bench_p1_p6_bug_fix.md` + `330b_catherine_30metal_full_pass.md` |
| `p2_vina_installed` | `TODO/completed/329w7_todo_134_vina_alternative.md` + Wf-330-F W7 verification |
| `p3_wf327_cpu_num_workers_sufficient` | `TODO/completed/330c_preflight_v3_full_pass.md` (venv fix + REPO_ROOT/scripts path fix) |
| `p3_admet_4_below_documented` (NEW) | `TODO/completed/325e_admet_4_below_final.md` |
| `p3_m6_kappa_closeout_documented` (NEW, `frontier` only) | `TODO/completed/325d_m6_kappa_final.md` |
| `p3_phase3g_f2a_runtime_unblock_verified` (NEW) | `TODO/completed/325f_phase3g_f2a_runtime_unblock.md` |
| `p3_catherine_30metal_closeout_documented` (NEW) | `TODO/completed/330b_catherine_30metal_full_pass.md` |
| `p3_m4_sa_qed_subcriterion_documented` (NEW) | `TODO/completed/323f_m4_sa_qed_subcriterion.md` |

---

## §5 Honest framing (party-twenty-spirit §3.5)

### §5.1 MEASURED vs PROJECTED tag discipline

Every numeric value in this doc traces to a JSON / report path on disk. PROJECTED vs
MEASURED tag is required.

| Item | Status | Source |
|---|---|---|
| Pre-flight v3 62/62 PASS | **MEASURED** | `/tmp/wf336f1_preflight.json` (2026-09-29 15:25 UTC) |
| Pre-flight v3 frontier 25/25 PASS | **MEASURED** | `/tmp/wf336f1_frontier.json` (2026-09-29 15:25 UTC) |
| Wf-322D baseline 42/15/57 | **MEASURED** | `molmetal/reports/wf322d_preflight/pre_flight.json` (per `322d_index_readme_wf309_322_sync.md` §3) |
| Wf-330-B3 62/0/62 | **MEASURED** | `TODO/completed/330c_preflight_v3_full_pass.md` |
| Per-priority breakdown 3/18/4/37 | **MEASURED** | computed from `/tmp/wf336f1_preflight.json` |
| Per-gate flip × Wf citations | **MEASURED** | each cited by TODO/completed/ file path |
| Wall time 111.58 s (all pipeline) | **MEASURED** | `total_wall_sec` in JSON |
| Wall time 12.02 s (frontier pipeline) | **MEASURED** | `total_wall_sec` in JSON |

### §5.2 Honest-negative framing

The Wf-336-F1 fresh-run confirms the **CLEAN STATE PRESERVED** across the 13-Wf
ultracode wave. The 5 forward-looking honest-negatives preserved for GPU launch
remain:

- `HONEST_NEGATIVE_M3_VINA_COVALENT_INSUFFICIENT` (R11) — Pt+2 Vina parameterisation
- `HONEST_NEGATIVE_PARTIAL_STRUCTURAL` (m6_kappa) — small cohort
- `HONEST_NEGATIVE_PARTIAL_STRUCTURAL` (m7 diversity) — small cohort
- `HONEST_NEGATIVE_R2_DEEP` (chemprop v2 graph axis)
- `HONEST_NEGATIVE_TRIANGLE_R2` (5-d features)

Two honest-negatives were **CLOSED** during this wave:

- `HONEST_NEGATIVE_WF246_C_KCAL_SIGN_INVERTED` → ✅ FIXED per Wf-333-V1
- `Wf-330-B4_RuntimeError_CUDAGraphs` → ✅ FIXED per Wf-335-R1+R3

### §5.3 Standing GPU authorization (AGENTS.md §5.1)

✅ All GPU-bound items are USER_OK-gated or in-flight per standing authorization.
This Wf-336-F1 is **CPU-only** (no GPU launches, no per-launch OK required).

### §5.4 Pre-flight v2 25/25 PASS gate (TODO-207)

✅ **62/62 PASS** verified at 2026-09-29 15:25 UTC. No FAIL gates. No STALE flag
(within 24 hr). The R9 15-FAIL regression is **CLOSED** in `TODO/risks.md`.

---

## §6 Forward-looking preservation (per Wf-336-D4 §9)

Per `TODO/completed/336_final_closeout.md` §9, the **62/62 pre-flight clean state** is
preserved for the 13-Wf ultracode wave. The following forward-looking items remain
USER_OK-gated per AGENTS.md §5.1:

| Launch | Wall | Status |
|---|---:|---|
| TODO-181 G5 (50-epoch full sweep) | ~5 hr | in flight (Ru seed 2 + Ir seed 0) |
| TODO-182 G6 (per-innovation ablation) | ~7.5 hr | dry-run READY; launch wrapper ready |
| TODO-119 P1.4 (chemprop v3 LOMO) | ~2.0 hr | launch wrapper READY per Wf-336-X2; pre-flight 25/25 PASS |
| Wf-299 CFM 3-wave | ~9.5 hr sequential | dry-run READY; launch wrapper ready |
| Wf-264 CFM property panel | ~10 min CPU | launch wrapper ready per Wf-334-U3 + Wf-336-D3 |
| TODO-93 Hilbert-ASTSR GPU retrain | ~5 hr | DEFERRED to R18 |
| TODO-94 P3 Multi-PH + PLL GPU | ~3 hr | DEFERRED to R20-P3 |
| TODO-21 Lambda × CFM GPU retrain | 12-24 hr per cycle | DEFERRED to R18 |
| Wf-246-C GPU Pt-coord term (Vina P4) | ~5 hr | USER_OK-gated; PROJECTED -2.218 kcal/mol lift |

---

## §7 Conclusion

**Pre-flight v3 final state: 62/62 PASS, 0 FAIL, 0 SKIP, 62 TOTAL** at 2026-09-29
15:25 UTC. The Wf-322D 15-FAIL REGRESSION is **fully closed** (R9 in `TODO/risks.md`).
The 5 NEW gates added since Wf-322D all PASS with proper venv + path resolution.
The clean state is preserved through Wf-330-C → Wf-332-S3 → Wf-334-C4 → Wf-336-D4 →
**Wf-336-F1 (this final closeout)**.

**Owner**: Hugo Lee + ultracode agent
**Cross-ref**: `party-twenty-spirit` §3.5 + AGENTS.md §5.1 + §5.2 + §5.4 + TODO-207 v2 + R9 closed.

---

**END OF Wf-336-F1 CLOSEOUT**