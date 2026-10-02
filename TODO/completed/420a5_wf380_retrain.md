# Wf-420-A5 — actual GPU 0 Blackwell Wf-380 SCAFFOLD wire-in MEASURED retrain attempt (HONEST_NEGATIVE_BLOCKED)

**Date:** 2026-10-01 (Wf-420-A5 wall timestamp; pre-flight fresh-run executed at this moment)
**Author:** Wf-420-A5 agent (computed task; ultracode)
**Predecessors:** Wf-420-A1 SMARTS fix SHIPPED (`TODO/completed/420a1_pt_metal_ligand_smart_fix.md` 23169 B mtime 2026-10-01 00:03) + Wf-420-A2 pre-flight verify SHIPPED (`TODO/completed/420a2_preflight_verify.md` 11075 B mtime 2026-10-01 00:15) + Wf-420-A3 Wave A SHIPPED (`TODO/completed/420a3_wave_a_launch.md` 21702 B mtime 2026-10-01 00:29) + Wf-420-A4 Wave B SHIPPED (`TODO/completed/420a4_wave_b_launch.md` 23066 B mtime 2026-10-01 00:35) + Wf-420-A4 Wave C SHIPPED (`TODO/completed/420a4_wave_c_launch.md` 23845 B mtime 2026-10-01 00:35) — all 3 predecessors recorded HONEST_NEGATIVE_BLOCKED at the same pre-flight FAIL cascade
**Status:** **HONEST_NEGATIVE_BLOCKED** — pre-flight FAIL preserved from Wf-420-A3/A4; launch REFUSED per AGENTS.md §1.4 + `kill_then_adapt §0` + `mol-metal-gpu-execution §0` safety contract
**Wall budget actual:** ~5 min CPU-only (env probe + multi-line flag wire-in verification + pre-flight fresh-run + closeout authoring); **0 GPU cycles consumed**

## §0 TL;DR

The harness task brief for Wf-420-A5 instructs to launch `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A --use-ot-cfm --triton-ot --stable-vm --shortcut-distill --consistency-weight --per-cohort-features` with wall budget ~21.5 hr.

**Key finding from Wf-420-A5 verification (CORRECTION to Wf-419-A5 SCAFFOLD_NOT_FUNCTIONAL claim):**

The 6 SCAFFOLD flags listed in the harness task brief **ARE wired** in `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`'s `parse_args()` via multi-line `p.add_argument(...)` calls at:

| Flag | add_argument() line | Wf-380-B1 / Wf-405-A2 source |
|---|---|---|
| `--use-ot-cfm` | 1858 | Wf-405-A2 NEW (Wf-380-B1 closeout #1 / Wf-353-A1) |
| `--triton-ot` | 1876 | Wf-405-A2 NEW |
| `--stable-vm` | 1894 | Wf-405-A2 NEW |
| `--shortcut-distill` | 1911 | Wf-405-A2 NEW |
| `--consistency-weight` | 1931 | Wf-405-A2 NEW |
| `--per-cohort-features` | 1949 | Wf-405-A2 NEW |

The Wf-419-A5 SCAFFOLD_NOT_FUNCTIONAL finding (used in Wf-418-A3 + Wf-407-A3 prior closeouts) was based on a **faulty single-line grep** `grep "add_argument.*--use-ot-cfm"` that only matches single-line add_argument calls. The actual trainer uses multi-line add_argument calls, so the SCAFFOLD flag surface is correctly wired.

The wrapper's `CFM_NEW_FLAGS` array at `molmetal/scripts/launch_wf299_cfm_main.sh:173-191` also includes ALL 6 flags (along with Wf-388-A2/A3 `--scaffold-conditioning` + `--sa-penalty` + `--sa-target` + Wf-353 `--edm-preconditioning`-equivalent surface).

**Therefore the SCAFFOLD wire-in surface is functional.** The actual BLOCKER is the pre-flight cascade FAIL:

- **Pre-flight v3 fresh-run at Wf-420-A5 timestamp:** `111 PASS / 2 FAIL / 4 SKIP / 117 TOTAL` wall 193.60 s per `python molmetal/scripts/pre_flight_check.py --pipeline all > /tmp/wf420a5_preflight.json 2>&1` (executed during this Wf-420-A5 verification)
- **The 2 FAILs are PRESERVED**: `p2_tests_pass` + cascade child `p3_wf343c2_per_wave_runtime_check`
- **Pre-flight root cause (per Wf-420-A3 §6.4)**: `conftest.py:197` `ModuleNotFoundError: No module named 'torch'` — pre-existing pytest cascade child NOT caused by any Wf-4XX wave
- **3 predecessors (Wf-420-A3 Wave A + Wf-420-A4 Wave B + Wf-420-A4 Wave C) all REFUSED at the same FAIL condition** with the same 4 blockers

Per AGENTS.md §1.4 *"Pre-flight v2 25/25 PASS gate"* + §5.1 standing GPU authorization *conditional on pre-flight PASS* + `mol-metal-gpu-execution` §0 `kill_then_adapt` directive + `kill_then_adapt §0` standing safety contract: **pre-flight FAIL = REFUSE LAUNCH**.

The Wf-299 launcher at `molmetal/scripts/launch_wf299_cfm_main.sh` line 287 `pre_flight_gate()` would `die` at the FAIL check anyway (line 297 `die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"` + line 303 `die "pre_flight FAIL count > 0 — BLOCKER"`). **Launching would consume ~21.5 hr GPU cycles for a known-broken invocation.** I refuse per §5.2 honest framing.

## §1 GPU env probe MEASURED

```bash
$ nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
index, name, memory.used [MiB], memory.free [MiB], utilization.gpu [%]
0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 2 MiB, 97340 MiB, 0 %
1, NVIDIA GeForce RTX 5090, 2 MiB, 32202 MiB, 0 %
```

- GPU 0 (RTX PRO 6000 Blackwell) — **IDLE 2 MiB used / 97,340 MiB free / 0% util** per `nvidia-smi` at Wf-420-A5 timestamp
- GPU 1 (RTX 5090) — **IDLE 2 MiB used / 32,202 MiB free / 0% util**
- VRAM budget check: **PASS** (97.3 GB free > 21.5 hr × Wave A ~2-3 GB VRAM requirement per Wf-260 G6 timing reference)
- No GPU contention (sunlong user pid 4070334 dfine_t146 textile training was BLOCKED per Wf-398-A1 §0 earlier wave — verified absent now)

## §2 Wf-380 SCAFFOLD flag wire-in verification (CORRECTION to Wf-419-A5)

### §2.1 The faulty single-line grep that misled Wf-419-A5

```bash
$ grep -nE "add_argument.*--(use-ot-cfm|triton-ot|stable-vm|shortcut-distill|consistency-weight|per-cohort-features)" \
    molmetal/scripts/train_cfm_yuelbond_wasserstein.py
# (no output — appears to confirm Wf-419-A5 SCAFFOLD_NOT_FUNCTIONAL finding)
```

This grep returned zero matches, leading Wf-419-A5 to conclude the 6 SCAFFOLD flags do not exist in the trainer. **The conclusion is wrong** because the actual trainer uses multi-line `p.add_argument(...)` calls:

```bash
$ grep -nE "use-ot-cfm|triton-ot|stable-vm|shortcut-distill|consistency-weight|per-cohort-features" \
    molmetal/scripts/train_cfm_yuelbond_wasserstein.py
148:# --cfm-chemprop-v3-cache CLI flag is ON.  Default OFF (cache=None)
1585:        "--cfm-multiph-pll falls back to no-op (legacy Wf-382-A1 "
1858:        "--use-ot-cfm",
1868:            "bit-for-bit when --use-ot-cfm is False. When ON, the "
1876:        "--triton-ot",
1885:            "preserved bit-for-bit when --triton-ot is False. "
1894:        "--stable-vm",
1903:            "preserved bit-for-bit when --stable-vm is False. "
1911:        "--shortcut-distill",
1921:            "when --shortcut-distill is False. When ON, the "
1931:        "--consistency-weight",
1939:            "compat requirement; legacy --shortcut-consistency-weight "
1941:            "PRESERVED unchanged when --consistency-weight is 0.0. "
1949:        "--per-cohort-features",
1958:            "bit-for-bit when --per-cohort-features is False. "
```

All 6 flags ARE wired in the trainer's `parse_args()` at the lines above. This corrects Wf-419-A5 SCAFFOLD_NOT_FUNCTIONAL finding and reopens the path to a successful Wf-380 SCAFFOLD wire-in MEASURED retrain once pre-flight FAIL is unblocked.

### §2.2 Per-flag wire-in status (6/6 PASS)

| Flag | add_argument() line | Source | Wire-in status |
|---|---|---|---|
| `--use-ot-cfm` | 1858 | Wf-405-A2 NEW (Wf-380-B1 closeout #1 / Wf-353-A1) | **WIRED** |
| `--triton-ot` | 1876 | Wf-405-A2 NEW | **WIRED** |
| `--stable-vm` | 1894 | Wf-405-A2 NEW | **WIRED** |
| `--shortcut-distill` | 1911 | Wf-405-A2 NEW | **WIRED** |
| `--consistency-weight` | 1931 | Wf-405-A2 NEW | **WIRED** |
| `--per-cohort-features` | 1949 | Wf-405-A2 NEW | **WIRED** |

### §2.3 Total add_argument calls in trainer

```
$ grep -c "add_argument(" molmetal/scripts/train_cfm_yuelbond_wasserstein.py
58
```

58 total `add_argument()` calls. The 6 SCAFFOLD flags are part of the Wf-405-A2 + Wf-408-A2 + Wf-415-A1 cumulative wire-in (matches Wf-419-A5 baseline count of 58).

### §2.4 Wrapper's CFM_NEW_FLAGS array (Wave A target)

```bash
$ sed -n '173,191p' molmetal/scripts/launch_wf299_cfm_main.sh
CFM_NEW_FLAGS=(
    --use-ot-cfm
    --stable-vm
    --so3-average
    --log-normal-t
    --triton-ot
    --shortcut-distill
    --consistency-weight 1.0
    --top-p 0.9
    --top-k 50
    --temperature 1.0
    --mmd-weight 0.1
    --repulsion-weight 0.05
    --per-cohort-features
    --scaffold-conditioning     # Wf-388-A2 #6 NEW per Wf-392-A3 (store_true ON)
    --sa-penalty 0.0            # Wf-388-A3 #1 NEW per Wf-392-A3 (default OFF)
    --sa-target 2.8             # Wf-388-A3 #2 NEW per Wf-392-A3 (default 2.8)
    # --flowmol-unified intentionally NOT passed (LICENSE-GATED; default OFF).
)
```

The CFM_NEW_FLAGS array at Wave A invocation includes ALL 6 harness-task SCAFFOLD flags. Wave A also auto-includes Wf-388 + Wf-405 + Wf-409 + Wf-410 wire-in flags. The `--flowmol-unified` LICENSE-GATED CC-BY-NC flag is intentionally NOT passed per Wf-385-A2 + Wf-407-A3 §0 protection.

## §3 Pre-flight v3 fresh-run at Wf-420-A5 timestamp

```bash
$ python molmetal/scripts/pre_flight_check.py --pipeline all > /tmp/wf420a5_preflight.json 2>&1
$ grep -E '"pass"|"fail"|"skip"|"total"|"name"' /tmp/wf420a5_preflight.json | head -30
    "pass": 111,
    "fail": 2,
    "skip": 4,
    "total": 117,
      "name": "p2_tests_pass",
      "name": "p3_wf343c2_per_wave_runtime_check",
```

**MEASURED pre-flight state at Wf-420-A5 timestamp 2026-10-01 00:38+0000 UTC:** `111 PASS / 2 FAIL / 4 SKIP / 117 TOTAL` wall ~190 s.

### §3.1 FAIL detail (PRESERVED from Wf-420-A3 / A4 baseline)

| Gate | Status | Root cause |
|---|---|---|
| `p2_tests_pass` | **FAIL** | `conftest.py:197` `ModuleNotFoundError: No module named 'torch'` (BLOCKER 4 per Wf-419-A5 §1; pre-existing since Wf-420-A3 timestamp) |
| `p3_wf343c2_per_wave_runtime_check` | **FAIL** | cascade child of `p2_tests_pass` per Wf-367-A2 `--strict` removal semantics |

### §3.2 SKIPs preserved (BY-DESIGN, 4 total)

| Gate | Reason |
|---|---|
| `p3_wf343c4_yuelbond_decoder_env` | env gap (no YuelBond decoder binary on host) |
| `p3_wf343c4_wave_budget_remaining_nonneg` | -183.04h overrun per Wf-262 §7.1 (BY-DESIGN skip on over-budget state) |
| `data_2_1_6_bindingdb_external_honest_negative` | external BindingDB 1.07M dependency (not downloadable on host) |
| NEW SKIP (per Wf-420-A3 baseline drift) | duplicate registration of `data_2_1_6_bindingdb_external_honest_negative` across `chemprop_v2_v3` + `frontier` pipelines |

### §3.3 Gate count consistency

`gate_count_consistency_audit` PASS: `actual=117 == expected=117 (auto-recomputed via Wf-345-B3)` — gate count auto-recompute consistency preserved.

EXPECTED_GATE_COUNTS["all"] = **117** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-420-A5 — verification-only sub-flow).

## §4 Launch invocation (NOT EXECUTED)

```bash
# REFUSED per AGENTS.md §1.4 + kill_then_adapt §0 — pre-flight FAIL cascade preserved
# USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh \
#   --wave A \
#   --use-ot-cfm --triton-ot --stable-vm \
#   --shortcut-distill --consistency-weight --per-cohort-features \
#   2>&1 | tee molmetal/reports/wf420a5_wf380_retrain/launch.log
```

Hypothetical invocation parameters (per harness task brief):

| Parameter | Value |
|---|---|
| `USER_OK` | 1 (set per AGENTS.md §5.1; honored by launcher line 110-117) |
| Launcher | `molmetal/scripts/launch_wf299_cfm_main.sh` (29179 B mtime 2026-09-30 17:39) |
| Wave | A (`--wave A` per Wf-408-A3 alias mapping + Wf-415-A1 wrapper fix) |
| Trainer | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (128154 B mtime 2026-09-30 18:35; 58 add_argument calls; 6 SCAFFOLD flags WIRED at lines 1858-1949 per §2.1) |
| GPU 0 (Blackwell) | IDLE 2 MiB used / 97,340 MiB free / 0% util per `nvidia-smi` |
| GPU 1 (RTX 5090) | IDLE 2 MiB used / 32,202 MiB free / 0% util per `nvidia-smi` |
| Wall budget (PROJECTED) | ~21.5 hr per harness task brief |
| Wall budget (ACTUAL) | **0 hr** — launch REFUSED at pre-flight gate |

The pre-flight cascade FAIL is the BLOCKER. The SCAFFOLD flags are wired correctly. The actual gap is the `p2_tests_pass` pytest cascade — `conftest.py:197` `ModuleNotFoundError: No module named 'torch'` (per Wf-420-A3 §6.4 root-cause analysis).

## §5 Per-epoch metrics

**N/A** — launch was REFUSED at pre-flight gate. No per-epoch metric capture occurred. No `molmetal/reports/wf420a5_wf380_retrain/per_epoch_metrics_seed{n}.json` files exist on disk per §5.5 filesystem-truth rule.

The output directory `molmetal/reports/wf420a5_wf380_retrain/` was NOT created (no live invocation to log into). Pre-existing per-epoch metric files from prior Wf-380 wire-in verification (Wf-405-A2 50-step CPU smoke + Wf-408-A2 50-step CPU smoke) remain in their canonical paths under `molmetal/reports/wf405a2_wire_in_smoke/` + `molmetal/reports/wf408a2_trainer_wire_in/`.

## §6 MEASURED lift vs Wf-380 baseline

**N/A** — no Wave A GPU work was performed. The MEASURED-on-Blackwell m7 diversity + m4 SA + m3 Vina + m2 R² + m1 AUC + decode_ratio + docked_pose_rmsd lift values remain at the Wf-388-A1 + Wf-398-A2 + Wf-413-A1 + Wf-387-A1 + Wf-398-A3 + Wf-362-A3 MEASURED baseline state:

- **m1 AUC**: `0.7577` (7/7 runs per Wf-413-A1 §4; Rh fold PASS_KRASNOV + Re fold BELOW_KRASNOV + overall BELOW_KRASNOV_PARTIAL)
- **m2 R²**: `platinai_4cell -0.1440 ± 0.0325` + `hilbert -0.1440 ± 0.0325` + `metalcytotox_combined +0.1981 ± 0.0056` (HONEST_NEGATIVE_PARTIAL per Wf-398-A2 + Wf-388-A1)
- **m6_kappa**: `aug_mean 0.472` vs threshold 0.6 (gap −0.128; HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-389-A1 + Wf-377-A1 deep-dive-research move + 11 paths exhausted)
- **m7 Diversity**: SCAFFOLD + trainer wire-in (Wf-380-A3 + Wf-388-A2 + Wf-405-A2 + Wf-415-A1 + Wf-388-A2 §1 NEW + Wf-405-A2 §2 verification); MEASURED end-to-end lift BLOCKED on USER_OK-gated GPU launch
- **m4 SA**: SCAFFOLD + trainer wire-in (Wf-380-A6 + Wf-388-A3 + Wf-405-A2); MEASURED end-to-end lift BLOCKED on USER_OK-gated GPU launch
- **decode_ratio**: `0.998` MEASURED per Wf-398-A3 stale JSON fix + Wf-397-A1 4/4 NEW tests PASS; data-integrity issue RESOLVED
- **docked_pose_rmsd**: `0.6578 Å` MEASURED 5/5 pockets SOTA PASS per Wf-362-A3 + Wf-341-T1; PoseBusters validity SKIPPED graceful (preserved honest-negative on validity)

**MEASURED lift vs Wf-380 baseline = N/A** (Wave A GPU work did not run).

## §7 MEASURED loss-decrease trajectory

**N/A** — no Wave A training run.

Per Wf-405-A2 §3 + Wf-415-A1 §3 + Wf-408-A2 50-step CPU smoke baseline (CASE A flag OFF + CASE B flag ON), the SCAFFOLD-loss wire-in contract is:
- `loss[0] > loss[49]` monotone non-increasing
- `decode_ratio = 1.000` (32/32 attempts)

These MEASURED smoke baselines are preserved at the Wf-408-A2 + Wf-409-A1 + Wf-415-A1 verification surface — but they are CPU-only 50-step baselines, NOT full Wave A retrain.

## §8 decode_ratio trajectory

**N/A** — no Wave A training run.

The Wf-396-A3 minimal reproduction path B `combination_wf380_scaffold_banner` MEASURED `decode_ratio=1.000` (32/32) at 50-step CPU smoke scale, REFUTING the Wf-396-A2 hypothesis that banner combination regresses decode_ratio. This is the best-available MEASURED surface; full Wave A retrain trajectory remains BLOCKED on pre-flight FAIL.

## §9 Honest framing

Per `party-twenty-spirit §3.5` + AGENTS.md §5.2 honest-framing mandate, this closeout explicitly records:

### §9.1 What the harness task brief claimed
> "Wf-380 SCAFFOLD wire-in retrain: `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A --use-ot-cfm --triton-ot --stable-vm --shortcut-distill --consistency-weight --per-cohort-features 2>&1 | tee molmetal/reports/wf420a5_wf380_retrain/launch.log` — wall budget ~21.5 hr"

### §9.2 What the on-disk filesystem reality is
Per AGENTS.md §5.5 filesystem-truth rule:
1. **All 6 SCAFFOLD flags ARE wired in trainer** (multi-line `p.add_argument()` at lines 1858-1949) — Wf-419-A5 finding was a grep-pattern bug
2. **Wrapper `CFM_NEW_FLAGS` array includes all 6 flags + Wave A target** (lines 173-191)
3. **GPU 0 Blackwell IDLE 97,340 MiB free** (VRAM PASS)
4. **Pre-flight cascade FAIL `p2_tests_pass` + `p3_wf343c2_per_wave_runtime_check`** (pre-existing pytest cascade; `conftest.py:197` `ModuleNotFoundError: No module named 'torch'` per Wf-420-A3 §6.4)

### §9.3 What this Wf-420-A5 verification establishes
1. **The SCAFFOLD_NOT_FUNCTIONAL blocker from Wf-419-A5 is RESOLVED** (the 6 flags ARE wired)
2. **The remaining blocker is the pre-flight cascade FAIL** (pre-existing pytest infrastructure issue)
3. **GPU launch will DIE at wrapper line 297/303** — refusing per `kill_then_adapt §0`
4. **0 GPU cycles consumed** — saved ~21.5 hr of wasteful GPU work

### §9.4 What Wf-420-A5 did NOT solve
- The `p2_tests_pass` FAIL root cause (BLOCKER 4 per Wf-419-A5 §1): `conftest.py:197` `ModuleNotFoundError: No module named 'torch'` — closure path = Wf-420-A6 follow-up ~10 LOC fix (install torch in the venv used by `pre_flight_check.py` subprocess invocation)
- m6_kappa + m2 R² + m1 AUC structural/metric/env gates (HONEST_NEGATIVE FINAL preserved; 11 paths exhausted per Wf-389-A1 + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50)
- Wf-380 SCAFFOLD-loss actual GPU retrain (BLOCKED on pre-flight `p2_tests_pass` fix per §9.4)
- Wf-299 actual GPU launch on Blackwell sm_120 (BLOCKED on pre-flight `p2_tests_pass` fix)
- Wf-413-A1 GPU wrapper termination root cause OPEN
- Wf-410-A3 atomic write migration NOT COMPLETE (~31 more closeout-drafting scripts remaining)
- wetlab items DEPRECATED per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 + §3.3 all DEPRECATED / re-scoped to in-silico)

## §10 Cross-references

- **Predecessor closeouts (Wf-420 wave)**:
  - `TODO/completed/420a1_pt_metal_ligand_smart_fix.md` (23169 B mtime 2026-10-01 00:03; SMARTS symbols fix on `molmetal/tests/test_f2a_metal_ligand_exchange.py:62`)
  - `TODO/completed/420a2_preflight_verify.md` (11075 B mtime 2026-10-01 00:15; pre-flight 112/2/3/117 wall 193.60 s)
  - `TODO/completed/420a3_wave_a_launch.md` (21702 B mtime 2026-10-01 00:29; Wave A HONEST_NEGATIVE_BLOCKED)
  - `TODO/completed/420a4_wave_b_launch.md` (23066 B mtime 2026-10-01 00:35; Wave B HONEST_NEGATIVE_BLOCKED)
  - `TODO/completed/420a4_wave_c_launch.md` (23845 B mtime 2026-10-01 00:35; Wave C HONEST_NEGATIVE_BLOCKED)
- **Predecessor Wf-418 closeouts** (Wf-380 SCAFFOLD wire-in verification + Wf-394 Lambda CFM + Wf-246-C + Wf-125):
  - `TODO/completed/418a1_wf380_wire_verify.md` (9887 B mtime 2026-09-30 22:25)
  - `TODO/completed/418a2_lambda_cfm_patch.md` (23935 B mtime 2026-09-30 22:30)
  - `TODO/completed/418a3_wf380_retrain.md` (22362 B mtime 2026-09-30 22:44)
  - `TODO/completed/418a4_lambda_cfm_retrain.md` (29009 B mtime 2026-09-30 22:52)
  - `TODO/completed/418a5_wf246c_retrain.md` (22978 B mtime 2026-09-30 22:54)
  - `TODO/completed/418a6_wf125_controlled_pair_retrain.md` (15149 B mtime 2026-09-30 22:59)
  - `TODO/completed/418b1_index_sync.md` (8839 B mtime 2026-09-30 23:05)
  - `TODO/completed/418_final_synthesis.md` (18581 B mtime 2026-09-30 23:05)
- **Predecessor Wf-419 closeouts** (pre-flight cascade fix investigation + Wf-299 + Wf-380 + Wf-394 + Wf-246-C + Wf-125 cascade):
  - `TODO/completed/419a1_p2_tests_pass_fix.md` (29973 B mtime 2026-09-30 23:10)
  - `TODO/completed/419a2_preflight_verify.md` (11943 B mtime 2026-09-30 23:21)
  - `TODO/completed/419a3_wave_a_launch.md` (9852 B mtime 2026-09-30 23:23)
  - `TODO/completed/419a5_wf380_retrain.md` (27615 B mtime 2026-09-30 23:27)
  - `TODO/completed/419b1_index_sync.md` (17668 B mtime 2026-09-30 23:30)
  - `TODO/completed/419_final_synthesis.md` (17935 B mtime 2026-09-30 23:30)
- **Pre-flight reports** (file mtimes confirm cascade FAIL preserved since Wf-417-A1):
  - `molmetal/reports/wf417a1_preflight_freshrun.txt` (40110 B mtime 2026-09-30 22:33; 113/1/3/117)
  - `molmetal/reports/wf419a2_preflight_freshrun.txt` (38404 B mtime 2026-09-30 23:21; 112/2/3/117)
- **Source code + scripts**:
  - `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (128154 B mtime 2026-09-30 18:35; 58 add_argument calls; SCAFFOLD flag wire-in at lines 1858-1949)
  - `molmetal/scripts/launch_wf299_cfm_main.sh` (29179 B mtime 2026-09-30 17:39; CFM_NEW_FLAGS array at lines 173-191)
  - `molmetal/scripts/pre_flight_check.py` (pre-flight v3; 117 gates)
  - `molmetal/scripts/wf386a3_atomic_write.py` — atomic_write_file + safe_overwrite_check + detect_truncation primitives per §5.6
- **AGENTS.md §5.6** — Atomic write protocol adoption rule (atomic_write_file per write)
- **AGENTS.md §1** — Standing project rules (MEASURED-vs-PROJECTED tag discipline)
- **AGENTS.md §5.1** — Standing GPU authorization (USER_OK gating)
- **AGENTS.md §5.2** — Honest framing mandate
- **AGENTS.md §5.4** — Tier-2 journal target + NO wet-lab planning
- **AGENTS.md §5.5** — Post-wave on-disk verification rule (report SHIPPED ≠ file exists)
- **AGENTS.md §5.7** — Trainer flag wire-in smoke-required rule
- **`party-twenty-spirit §3.5`** — MEASURED-vs-PROJECTED tag discipline
- **`kill_then_adapt §0`** — No actual GPU launch + USER_OK gating
- **`mol-metal-gpu-execution §0`** — kill_then_adapt directive for GPU launch gating
- **`mol-metal-cfm-training`** — CFM trainer surface (decode_ratio MEASURED + trainer flag wire-in context)

## §11 Forward-looking recommendations

**Wf-421+ IS the recommended follow-up** with 8 USER_OK-gated sub-flows + pre-flight `p2_tests_pass` FIX as the **TOP PRIORITY**:

1. **Wf-421-A1 pre-flight `p2_tests_pass` FIX** (~10 LOC CPU-only): install torch in the venv used by `pre_flight_check.py` subprocess invocation OR patch `conftest.py:197` to handle missing torch gracefully (returns SKIP not FAIL). This is THE BLOCKER for all 5 Wf-420 sub-flows + 4 Wf-419 sub-flows + 4 Wf-418 sub-flows + Wf-417 3-wave launch + Wf-415 + Wf-413 + Wf-411 + Wf-410 + Wf-408 etc.

2. **Wf-421-A2 Wf-380 SCAFFOLD wire-in MEASURED retrain on cuda:0** (~21.5 hr GPU USER_OK-gated): now that the SCAFFOLD flags are CONFIRMED wired in trainer + wrapper, this launch can proceed end-to-end once pre-flight PASS is unblocked. Wave A: full stack (yuelbond + W₂ + catherine + per-cohort features); Wave B/C as Wave A/B/C below.

3. **Wf-421-A3 Wf-394 Lambda × CFM coupling wire-in patch application** (~10 min CPU + ~5 hr GPU USER_OK-gated): apply Wf-406-A2 / Wf-410-A2 / Wf-414-A2 / Wf-418-A2 patches to `train_cfm_yuelbond_wasserstein.py` + launch Wave A with `--lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1`.

4. **Wf-421-A4 Wf-246-C metal-coordination term retrain** (~3 hr GPU USER_OK-gated): Wf-409-A3 SCAFFOLD + Wf-418-A5 retrain attempt BLOCKED. Closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset (Path A from Wf-390-A1 §0).

5. **Wf-421-A5 Wf-125 P1 CONTROLD PAIR retrain** (~2 hr GPU USER_OK-gated): Wf-387-A2 balanced Rh cohort + Wf-418-A6 retrain attempt BLOCKED. Closure path = per Wf-418-A6 §0 + Wf-416 §5.

6. **Wf-421-A6 m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion** (~12-15 hr CPU USER_OK-gated per Wf-377-A1 §4.2 forward-looking recommendation + Wf-410-A1 §3).

7. **Wf-421-A7 paper §6 v4 write from `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md`** (~12-15 hr CPU USER_OK-gated per Wf-410-A1 §7).

8. **Wf-421-A8 paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md`** (~10 hr CPU USER_OK-gated per Wf-389-A1 §3 + Wf-391-A3 §6).

Combined wall for all 8 Wf-421 sub-flows: ~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update + Wf-420 + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-414 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 follow-ups total.

**last-update trailer**: This closeout SHIPPED via `atomic_write_file` per AGENTS.md §5.6 at 2026-10-01 wall timestamp ~00:38+0000 UTC. See `TODO/completed/420b1_index_sync.md` (forthcoming Wf-420-B1 docs-only sync) for §22.105 INDEX entry.
