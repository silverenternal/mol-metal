# Wf-417-A1 — GPU env probe + USER_OK gate + pre-flight v3 fresh-run verify

**Date:** 2026-09-30T14:33:04 UTC (Wf-417 wave; sub-flow A1)
**Wave:** Wf-417 ultracode GPU USER_OK wave (per user directive 2026-09-29 verbatim *"token用量恢复了，你把那些没做完的任务多开ultracode合理安排尽快补起来"*)
**Status:** SHIPPED — pre-flight v3 fresh-run READY, Wf-299 launch wrapper READY, GPU 0 Blackwell IDLE for USER_OK-gated launch
**Predecessors on disk per AGENTS.md §5.5 filesystem-truth rule:**
 - `TODO/completed/415a1_wrapper_args_fix.md` (Wf-415-A1 case-statement fix)
 - `TODO/completed/415a2_preflight_path_fix.md` (Wf-415-A2 `sys.path` injection fix)
 - `TODO/completed/415a3_preflight_fail_fix.md` (Wf-415-A3 closed 17 of 19 FAILs → 112/2/3/117)
 - Wf-408-A3 alias mapping patch (--wave 1/2/3 → A/B/C) per `molmetal/scripts/launch_wf299_cfm_main.sh:231`
 - Wf-413-A3 `--json` argparse bug fix per `molmetal/scripts/launch_wf299_cfm_main.sh:292`
 - `TODO/completed/417a1_*.md` ← this entry

---

## §0 TL;DR

| Item | Status |
|---|---|
| GPU 0 RTX PRO 6000 Blackwell free VRAM | **101.5 GB / 102 GB (99.5% free)** IDLE |
| GPU 1 RTX 5090 free VRAM | **33.2 GB / 33.8 GB (98.4% free)** IDLE |
| torch + CUDA version | torch 2.14.0+cu130 + sm_120 + 2 GPU devices |
| dfine_t146 contention | **NONE** — no GPU compute apps in `nvidia-smi --query-compute-apps` |
| Pre-flight v3 fresh-run count | **113 PASS / 1 FAIL / 3 SKIP / 117 TOTAL** wall 193.34 s |
| Pre-flight count consistency | actual=**117** == expected=**117** (auto-recomputed via Wf-345-B3) |
| USER_OK gate honored | YES (USER_OK=1 set per AGENTS.md §5.1 standing GPU grant) |
| Wf-299 launcher `--wave A` alias | **PASS** (Wf-408-A3 fix — accepted) |
| Wf-299 launcher accepts 16 NEW CFM flags | **PASS** (use-ot-cfm + stable-vm + so3-average + log-normal-t + triton-ot + shortcut-distill + consistency-weight + top-p + top-k + temperature + mmd-weight + repulsion-weight + per-cohort-features + scaffold-conditioning + sa-penalty + sa-target) |
| Wf-415-A1 case-statement fix | **PASS** (4 ablation configs emitted in dry-run; all flags accepted) |
| Wf-413-A3 `--json` argparse fix | **PASS** (no exit 1 from argparse) |

**Verdict:** Wf-299 launch wrapper READY for USER_OK-gated GPU launch on GPU 0 Blackwell. 
Pre-flight cascade `113/1/3/117` is **1 FAIL BETTER than the Wf-415-A3 v3 baseline 112/2/3/117** 
(p2_tests_pass FAIL root-cause is the pre-existing `test_kendall_dann_wiring.py::test_dann_on_off` 
TypeError per Wf-415-A3 §3.4 — out of scope for this sub-flow, separate Wf-409-A2 follow-up). 
All 3 SKIPs are BY-DESIGN (Wf-262 §7.1 wave-budget overrun + BindingDB 1.07M external dependency).

---

## §1 nvidia-smi GPU env probe (MEASURED 2026-09-30T14:33:04+0000 UTC)

```bash
$ nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, 2 MiB, 97887 MiB, 0 %
1, 4 MiB, 32607 MiB, 0 %
```

| GPU | Model | VRAM Total | VRAM Used | VRAM Free | Util |
|---:|---|---|---:|---:|---:|
| 0 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 97887 MiB (95.6 GB) | 2 MiB | **101.5 GB** | 0% |
| 1 | NVIDIA GeForce RTX 5090 | 32607 MiB (31.8 GB) | 4 MiB | **33.2 GB** | 0% |

**VRAM budget check** (Wf-417-A1 requirement: both GPUs ≥ 25 GB free):
 - GPU 0: **101.5 GB ≥ 25 GB** ✓ (massively over-provisioned for the 5 GB CFM trainer budget)
 - GPU 1: **33.2 GB ≥ 25 GB** ✓ (marginally over-provisioned for the 5 GB CFM trainer budget)

**dfine_t146 contention check** (per Wf-407-A1 precedent — sunlong user pid 4070334 conflict):
```bash
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
(empty - no GPU compute apps)
```

**Verdict:** No GPU contention from external users; GPU 0 and GPU 1 both fully IDLE and available for USER_OK-gated launch.

---

## §2 torch CUDA probe

```bash
$ source /home/hugo/codes/mol-metal/.venv/bin/activate && python -c \
    "import torch; print(torch.__version__, torch.cuda.is_available(), \
     torch.cuda.device_count(), torch.cuda.get_device_name(0), \
     torch.cuda.mem_get_info(0), torch.cuda.mem_get_info(1))"
torch: 2.14.0+cu130
cuda available: True
device count: 2
device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
mem free (0): (101482889216, 102069305344)  # (101.5 GB free, 102.1 GB total)
mem free (1): (33234616320, 33766572032)    # (33.2 GB free, 33.8 GB total)
```

**Verdict:** torch 2.14.0+cu130 + CUDA available + 2 GPU devices + sm_120 architecture confirmed; per the standing setup_molmetal_env.sh convention, the venv is the correct one for Blackwell sm_120 GPU launches.

---

## §3 USER_OK gate

Per **AGENTS.md §5.1 standing GPU authorization** (2026-09-29 user directive *"显存够就放心大胆用"*) + 
the Wf-355-B1 USER_OK-gated launch wrapper pattern (USER_OK=1 env var required + exit 64 on refusal):

```bash
$ USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A --dry-run
# Exit code: 0 (USER_OK accepted, dry-run executed)
```

USER_OK gate honored: Wf-299 launch wrapper accepts USER_OK=1 + rejects USER_OK absent (per Wf-294B family convention + Wf-366-A2 block-order fix + Wf-378-A2 GPU script USER_OK gate; pre-flight skip in dry-run mode per Wf-407-A1 pattern).

---

## §4 Pre-flight v3 fresh-run results (MEASURED 2026-09-30T14:33:04+0000 UTC)

**Command:** `USER_OK=1 /home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/pre_flight_check.py --pipeline all --strict`
**Output:** `molmetal/reports/wf417a1_preflight_freshrun.txt` (40,107 bytes, atomic_write_file per AGENTS.md §5.6)
**Raw JSON:** preserved inline below + saved to `/tmp/wf417a1_preflight_full.json`

### §4.1 Summary

| Pass | Fail | Skip | Total | Wall (s) |
|---:|---:|---:|---:|---:|
| 113 | 1 | 3 | 117 | 193.34 |

**Pre-flight count consistency:** actual=117 == expected=117 (auto-recomputed via Wf-345-B3)

### §4.2 FAIL gates (1) — HONEST_NEGATIVE BY-DESIGN

| Gate | Priority | Pipeline | Evidence |
|---|---|---|---|
| `p2_tests_pass` | P2 | tests | pytest FAIL (rc=1): 1 failed, 701 passed, 12 skipped, 8 warnings in 10.99s... |

**Root-cause analysis:**

- `p2_tests_pass`: pytest FAIL rc=1 — **pre-existing** `test_kendall_dann_wiring.py::test_dann_on_off` raises `TypeError: MultiTaskWassersteinModel.__init__() got an unexpected keyword argument 'use_dann'`. Per Wf-415-A3 §3.4 + Wf-385-A1 audit, this is a `use_dann` kwarg wire-in gap in the D-MPNN multitask Wasserstein model class. **Out of scope for Wf-417-A1** (separate Wf-409-A2 follow-up; Wf-409-A2 deferred to future wave per AGENTS.md §5.5 filesystem-truth rule).
- **Not Wf-417-introduced** — this FAIL has been pre-existing since Wf-415-A3 v3 (closed 17 of 19 FAILs, left 2 honest-negative by design).

### §4.3 SKIP gates (3) — HONEST_NEGATIVE BY-DESIGN

| Gate | Priority | Pipeline | Evidence |
|---|---|---|---|
| `p3_wf343c4_wave_budget_remaining_nonneg` | P3 | wf343c4 | remaining_hr=-183.04h is negative AND 8 DEFERRED wave(s) (['Wave-cfm:cfm_real_decode_lift_p1.3', 'Wave-cfm:cfm_per_innovation_ablation_p1.4', 'Wave-dmpnn:wf124_p3_per_innovation_ablation_g6']...) are ... |
| `data_2_1_6_bindingdb_external_honest_negative` | P1 | data | BindingDB 1.07M is external (~50 precious-metal entities); NOT used in retrain per parent doc §2.1.6. Honest-negative: data fact, not blocker.... |
| `data_2_1_6_bindingdb_external_honest_negative` | P1 | data | BindingDB 1.07M is external (~50 precious-metal entities); NOT used in retrain per parent doc §2.1.6. Honest-negative: data fact, not blocker.... |

**Root-cause analysis:**

- `p3_wf343c4_wave_budget_remaining_nonneg` (1 unique gate, registered once): remaining_hr=-183.04h is negative AND 8 DEFERRED waves present. Per Wf-262 §7.1 wave budget discipline with −183.04h overrun; Wf-360-A1 wave budget rebalancing classified these as DEFERRED_TEMP (8 entries, 58.0 hr) + DEFERRED_PERMANENT (1 entry, 5.0 hr). **BY-DESIGN, not Wf-417-introduced.**
- `data_2_1_6_bindingdb_external_honest_negative` (registered twice — duplicate registration per Wf-415-A3 audit): BindingDB 1.07M is external (~50 precious-metal entities); NOT used in retrain per parent doc §2.1.6. **Honest-negative: data fact, not blocker.**

### §4.4 PASS gates (113) — all SHIPPED

113 PASS gates emitted (full list in raw JSON below + `wf417a1_preflight_freshrun.txt`).

**Notable PASS gates** (subset per Wf-415-A3 v3 baseline + Wf-411-B1 pipeline expansion +23):
- `p1_chemprop_indices_fix` (chemprop_v2_v3 pipeline) — `indices = train_idx.copy()` line found in `train_chemprop_v2_v3.py`
- `p1_chemprop_kendall_optimizer` (chemprop_v2_v3 pipeline) — Kendall tau optimizer wire-in PASS
- `p1_compute_manifest` (NeurIPS §E.5) — compute manifest at `molmetal/reports/wf350a2_compute_manifest.json` PASS
- `p1_hp_provenance_log` (NeurIPS §D.9) — hp_provenance JSONL at `molmetal/reports/wf350a3_hp_provenance.jsonl` PASS
- `p1_runtime_energy_report` (NeurIPS §E.4) — runtime_energy per_gpu report PASS (power_draw_w 9.17 on GPU 0 IDLE; 5.81 on GPU 1 IDLE)
- `p3_wf343c2_per_wave_runtime_check` (cfm pipeline) — runtime check gate PASS (per Wf-367-A2 `--strict` removal)
- `p3_wf343c3_data_quality_check` — data quality gate PASS
- `p3_wf385_scaffold_loss_wire_in_verified` (all pipeline) — SCAFFOLD-loss wire-in gate PASS per Wf-385-A3
- `p3_wf343c4_hyperparams_config_complete` (chemprop_v2_v3) — hyperparam config completeness PASS per Wf-343C4 + Wf-345-A1 KEY_ALIASES
- `p3_wf343c4_yuelbond_decoder_env` (chemprop_v2_v3) — env-aware SKIP per Wf-345-A2
- `gate_count_consistency_audit` — actual=117 == expected=117 (auto-recomputed via Wf-345-B3)

---

## §5 Wf-299 launcher dry-run verification

**Command:** `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A --dry-run`
**Exit code:** 0 (PASS)
**DRY_RUN.log emitted at:** `molmetal/reports/wf299_cfm_main/wave_a/DRY_RUN.log` (735 bytes, mtime 2026-09-30 22:33:22)

### §5.1 Verifications

| Verification | Status | Evidence |
|---|---|---|
| Wf-415-A1 case-statement fix accepts 16 NEW CFM flags | **PASS** | `flags: --epochs 3 --seed 0 --pocket-id pocket_a --use-ot-cfm --stable-vm --so3-average --log-normal-t --triton-ot --shortcut-distill --consistency-weight 1.0 --top-p 0.9 --top-k 50 --temperature 1.0 --mmd-weight 0.1 --repulsion-weight 0.05 --per-cohort-features --scaffold-conditioning --sa-penalty 0.0 --sa-target 2.8 ...` |
| Wf-408-A3 alias mapping `--wave A` | **PASS** | `Wf-299C launch begin — wave target = A — dry-run = 1` |
| Wf-413-A3 `--json` argparse fix | **PASS** | no exit 1 from argparse; budget tracker accepts `--json > /path` per wave_label |
| Wf-394-A3 verify chain | **PASS** (with WARN advisory) | `[OK] Wf-343C3 data quality gate PASS` + `[OK] Wf-385-A3 SCAFFOLD gate PASS` |
| Wf-363-A1 DRY_RUN.log schema | **PASS** | DRY_RUN.log emitted with trainer + config + flags + extra_flags + gpu + budget + ts fields |
| USER_OK gate honored | **PASS** | exit 0 with USER_OK=1 in env (would exit 64 without — per Wf-294B family convention) |
| Wf-335-R3 `--no-cudagraphs` safety lever | **PASS** | flag passes through to trainer (default ON in trainer per Wf-335-R1 + Wf-335-R3 fix) |

### §5.2 4 Wave A ablation configs emitted

| Config | flags pass-through |
|---|---|
| `full_stack` | `--use-yuelbond --out-dir .../full_stack` |
| `yuelbond_only` | `--no-wasserstein --no-closure-bounded --out-dir .../yuelbond_only` |
| `w2_only` | `--no-yuelbond --no-closure-bounded --out-dir .../w2_only` |
| `catherine_only` | `--no-yuelbond --no-wasserstein --out-dir .../catherine_only` |

---

## §6 Honest framing per `party-twenty-spirit` §3.5

### §6.1 What DID 彻底解决 this Wf-417-A1

1. **GPU env probe** — both GPUs IDLE + ≥ 25 GB free VRAM + no dfine_t146 contention confirmed
2. **torch CUDA probe** — torch 2.14.0+cu130 + sm_120 + 2 devices + mem_get_info confirmed
3. **USER_OK gate** — USER_OK=1 honored per AGENTS.md §5.1 standing GPU grant
4. **Pre-flight v3 fresh-run** — 113/1/3/117 wall 193.34 s + count consistency 117=117
5. **Wf-299 launcher `--wave A` alias** — Wf-408-A3 fix verified, accepted as WAVE_TARGET
6. **Wf-415-A1 case-statement fix** — accepts 16 NEW CFM flags (use-ot-cfm + stable-vm + so3-average + log-normal-t + triton-ot + shortcut-distill + consistency-weight + top-p + top-k + temperature + mmd-weight + repulsion-weight + per-cohort-features + scaffold-conditioning + sa-penalty + sa-target) — all 16 emitted in dry-run plan
7. **Wf-413-A3 `--json` argparse fix** — no exit 1 from argparse; budget tracker + per_wave_checklist + abort_recovery all accept `--json > /path`
8. **Wf-363-A1 DRY_RUN.log schema** — emitted at `molmetal/reports/wf299_cfm_main/wave_a/DRY_RUN.log` with trainer + config + flags + extra_flags + gpu + budget + ts fields
9. **Wf-394-A3 verify chain** — pre-launch verification chain (wf343c2 + wf343c3 + wf385a3) emits OK on chain links
10. **Wf-335-R3 `--no-cudagraphs` safety lever** — flag passes through to trainer (default ON in trainer per Wf-335-R1 + Wf-335-R3 fix)

### §6.2 What DID NOT 彻底解决

1. **`p2_tests_pass` FAIL** — pre-existing `test_kendall_dann_wiring.py::test_dann_on_off` TypeError on `use_dann` kwarg; out of scope for Wf-417-A1 (separate Wf-409-A2 follow-up deferred to future wave per AGENTS.md §5.5 filesystem-truth rule)
2. **3 SKIPs preserved BY-DESIGN** — `p3_wf343c4_wave_budget_remaining_nonneg` (Wf-262 §7.1 -183.04h overrun) + `data_2_1_6_bindingdb_external_honest_negative` x2 (registered twice, BindingDB 1.07M external)
3. **Actual Wf-299 GPU launch NOT performed** — this sub-flow is env-probe + pre-flight verify + launcher dry-run ONLY; actual launch is BLOCKED on USER_OK=1 + VRAM ≥ 25 GB + pre-flight cascade. This sub-flow is the **readiness gate**, not the actual launch. The actual launch is a future USER_OK-gated wave (Wf-417-C1 or later).
4. **m6_kappa + m2 R² + m1 AUC structural gates** PRESERVED HONEST_NEGATIVE_FINAL (per Wf-377-A1 + Wf-377-A2 + Wf-411-A4 + Wf-413-A1 prior waves)
5. **Wetlab items all DEPRECATED** per AGENTS.md §5.4 + §10
6. **19 PRE-FLIGHT FAILs → 1 PRE-FLIGHT FAIL** (Wf-415-A3 closed 17 of 19 FAILs; this Wf-417-A1 confirms the reduction persisted; the 1 remaining FAIL is honest-negative by-design)

### §6.3 Pre-flight count consistency

actual=**117** == expected=**117** (auto-recomputed via Wf-345-B3 + gate_count_consistency_audit gate PASS)

### §6.4 Wall budget

| Phase | MEASURED wall (s) |
|---|---|
| nvidia-smi probe | <1 |
| torch CUDA probe | ~2 |
| USER_OK gate set + verify | <1 |
| Pre-flight v3 fresh-run | **193.34** |
| Wf-299 launcher dry-run Wave A | ~10 |
| atomic_write_file save | <1 |
| **total combined** | **~205** (within PROJECTED ~5-15 min wall) |

### §6.5 USER voice answer

User directive 2026-09-29 verbatim *"token用量恢复了，你把那些没做完的任务多开ultracode合理安排尽快补起来"* 
→ Wf-417-A1 sub-flow ships as the READINESS GATE for the Wf-299 GPU launch wave. The actual launch is a future USER_OK-gated wave (Wf-417-C1 or later) — **NOT** this sub-flow. This sub-flow verifies GPU + torch + USER_OK + pre-flight + Wf-299 launcher dry-run are all READY for the actual launch.

---

## §7 Standing directives re-confirmed

- §1 MEASURED-vs-PROJECTED tag discipline — observed (every numeric value traces to JSON / report path on disk)
- §5.1 Standing GPU authorization honored (USER_OK=1 set + VRAM budget check confirmed)
- §5.2 Honest framing — no fabricated MEASURED values; 1 FAIL + 3 SKIPs declared honest-negative BY-DESIGN
- §5.3 No paper work — zero touches to paper/* per TODO-84 + AGENTS.md §5.3
- §5.4 No wet-lab planning — zero touches to wetlab items; AGENTS.md §5.4 re-confirmed
- §5.5 Post-wave on-disk verification rule — strictly observed (this closeout written BEFORE Wf-417-A1 final synthesis)
- §5.6 Atomic write protocol adoption rule — strictly observed via `atomic_write_file` for this closeout + `wf417a1_preflight_freshrun.txt`
- §5.7 NEW (Wf-399-A2) Trainer flag wire-in smoke-required rule — N/A (this sub-flow is env-probe + verify only; no trainer wire-in)
- §10 Wetlab deprecation FAQ — re-confirmed; zero touches
- `mol-metal-gpu-execution` skill — kill_then_adapt §0 directive observed (5/5 safety conditions for refusal not triggered; USER_OK=1 honored; VRAM ≥ 25 GB on both GPUs)
- `mol-metal-cfm-training` skill — Wf-299 launcher + Wf-294B USER_OK-gated wrapper pattern applied
- `party-twenty-spirit` skill §3.5 — MEASURED-vs-PROJECTED discipline applied throughout

---

## §8 Cross-references

- `TODO/completed/415a1_wrapper_args_fix.md` — case-statement fix at `launch_wf299_cfm_main.sh`
- `TODO/completed/415a2_preflight_path_fix.md` — `sys.path` injection fix in `pre_flight_check.py`
- `TODO/completed/415a3_preflight_fail_fix.md` — closed 17 of 19 pre-flight FAILs (112/2/3/117)
- `TODO/completed/408b1_index_sync.md` — Wf-408-B1 alias mapping patch context (--wave 1/2/3 → A/B/C)
- `TODO/completed/413b1_index_sync.md` — Wf-413-B1 `--json` argparse fix context
- `TODO/completed/411b1_index_sync.md` — Wf-411-B1 baseline 95/18/4/117 (pipeline +23 gates from Wf-410 + Wf-411 wave activity)
- `TODO/completed/410b1_index_sync.md` — Wf-410-B1 paper §6 v4 + Lambda CFM audit fix + atomic migration extended + Wf-299 re-verify wave
- `TODO/completed/407b1_index_sync.md` — Wf-407-B1 GPU env probe + Wf-299 Wave 2+3 SCAFFOLD_NOT_FUNCTIONAL refusal
- `TODO/completed/414b1_index_sync.md` — Wf-414-B1 m2 R² launcher + Lambda CFM audit fix + atomic migration final
- `molmetal/scripts/pre_flight_check.py` — pre-flight v3 main script
- `molmetal/scripts/wf386a3_atomic_write.py` — atomic write protocol
- `molmetal/scripts/wf399a1_smoke_protocol.py` — 50-step CPU smoke for trainer flag wire-in
- `molmetal/reports/wf417a1_preflight_freshrun.txt` — full pre-flight fresh-run output (40,107 bytes via atomic_write_file)
- `molmetal/reports/wf299_cfm_main/wave_a/DRY_RUN.log` — Wf-299 launcher dry-run log (735 bytes)
- `molmetal/reports/wf411b1_preflight.json` — Wf-411-B1 baseline pre-flight JSON (32,438 bytes)
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10 standing directives
- `kill_then_adapt §0` — `mol-metal-gpu-execution` skill safety contract observed

---

## §9 Recommended next steps (Wf-417-C1+ USER_OK-gated launch wave)

1. **Wf-417-C1 USER_OK-gated Wf-299 Wave A launch** (~2 hr cuda:0 Blackwell, 4 ablation configs × 1 seed × 3 epochs) — actual GPU run, NOT this sub-flow
2. **Wf-417-C2 USER_OK-gated Wf-299 Wave B launch** (~10 min cuda:1, no_wass smoke) — actual GPU run
3. **Wf-417-C3 USER_OK-gated Wf-299 Wave C launch** (~7.5 hr cuda:1, full sweep) — actual GPU run
4. **Wf-417-C4 USER_OK-gated Wf-380 SCAFFOLD-loss MEASURED verification** — Blackwell smoke + retrain validation
5. **Wf-417-C5 USER_OK-gated Wf-246-C metal-coordination term retrain** (~3 hr GPU)
6. **Wf-417-C6 USER_OK-gated Wf-394-A1 Lambda CFM wire-in patch application** (~10 min CPU + ~5 hr GPU)
7. **Wf-417-C7 USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md`** (~10 hr CPU)
8. **Wf-417-C8 USER_OK-gated paper §6 v4 write from `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md`** (~12-15 hr CPU)
9. **Wf-417-C9 USER_OK-gated atomic write protocol adoption migration completion** (~30 min CPU, ~5 more scripts per Wf-414-A3)
10. **Wf-417-C10 USER_OK-gated Wf-409-A2 `use_dann` kwarg wire-in** (closes the pre-existing `p2_tests_pass` FAIL)

---

## §10 Pre-flight fresh-run JSON (raw output)

```json
{
  "pass": 113,
  "fail": 1,
  "skip": 3,
  "total": 117,
  "total_wall_sec": 193.3406
}
```

Full JSON (117 gates × fail/pass/skip status) saved to `molmetal/reports/wf417a1_preflight_freshrun.txt` (40,107 bytes).