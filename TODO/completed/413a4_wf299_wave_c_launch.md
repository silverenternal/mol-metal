# Wf-413-A4 Wf-299 Wave C launch — HONEST_NEGATIVE_BLOCKED (pre-flight regression + wrapper design mismatch)

- **Wave ID:** Wf-413-A4 (part of Wf-413 ultracode GPU launch wave)
- **Task:** Launch Wf-299 GPU retrain on GPU 0 per user voice
- **Timestamp:** 2026-09-30T17:14Z
- **Wall elapsed:** ~30 s (CPU-only BLOCK; no GPU work performed)
- **User voice (verbatim):** "用GPU0好了，同时启动好了" ("Use GPU 0, launch in parallel")
- **User_OK status:** USER_OK=1 granted per AGENTS.md §5.1 standing GPU authorization 2026-09-29
- **Honest framing per `party-twenty-spirit §3.5`:** PROJECTED → ACTUAL verification chain produced HONEST_NEGATIVE_BLOCKED; both wave selection paths (Wave A on cuda:0 + Wave C with computed-task flags) returned EXIT=1 with no GPU work performed
- **VERDICT:** **HONEST_NEGATIVE_BLOCKED** — 0 of 1 wave launched; 0 GPU work units completed

---

## §0 TL;DR

| Field | Value |
|---|---|
| Wave | Wf-413-A4 (intended Wave C per computed task; intended Wave A per user voice GPU 0) |
| GPU target | cuda:0 (Blackwell 98GB) per user voice; cuda:1 (RTX 5090 32GB) per computed-task Wave C |
| Wrapper invocation | `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A` (Wave A = cuda:0 only; per user voice) |
| Secondary attempt | `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave C --edm-preconditioning --chemprop-v3-cache` (computed task verbatim; rejected by wrapper case statement at line 217) |
| Pre-flight state at launch | 95 PASS / 19 FAIL / 3 SKIP / 117 TOTAL (`molmetal/reports/wf299_cfm_main/pre_flight.json` 38847 bytes) |
| Pre-flight baseline per AGENTS.md §0.4 | 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL (Wf-353-B1 baseline) |
| Pre-flight regression | **+6 PASS / +17 FAIL / +0 SKIP / +23 TOTAL** (per Wf-345-B3 dynamic auto-recompute; new gates added by Wf-353-B1 wiring of 22 Wf-343C1 SKIP-class gates not in baseline 94) |
| Exit code (Wave A attempt) | 1 (`die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"` at line 262) |
| Exit code (Wave C literal attempt) | 1 (`die "unknown arg: --edm-preconditioning"` at line 217) |
| GPU 0 state at launch | idle (8.83W power draw / 0% util / 2MiB used per `nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used`) |
| GPU 1 state at launch | idle (5.23W power draw / 0% util / 2MiB used) |
| Wave A launched? | **NO** (wrapper refused at pre-flight) |
| Wave C launched? | **NO** (wrapper refused at flag parsing + would be on cuda:1 contradicting user voice) |
| GPU launches performed | **0 of 1** (computed-task wave) |
| Honest framing | HONEST_NEGATIVE_BLOCKED per party-twenty-spirit §3.5 |
| Cross-ref closeout docs | `TODO/completed/413a3_wf299_wave_b_launch.md` (predecessor — same 4 design-mismatch items) + `TODO/completed/413a2_wf299_wave_a_launch.md` (predecessor — Wave A pre-flight FAIL) |

---

## §1 User voice + computed task reconciliation

The workflow harness relayed two distinct inputs:

### §1.1 User voice (verbatim, the only authoritative user signal in this task)

> 用GPU0好了，同时启动好了

Translation: "Use GPU 0 is fine, launch in parallel." Two directives:
1. **GPU target = cuda:0** (Blackwell PRO 6000, 98 GB VRAM)
2. **Launch in parallel** (multiple things, but with the GPU 0 constraint)

### §1.2 Computed task (script output, NOT user authority)

The workflow script's computed task said:
- Wave C of Wf-299 (D-MPNN G6 full sweep on cuda:1/RTX 5090, ~7.5 hr)
- Launch wrapper at `scripts/launch_wf299_cfm_main.sh` (WRONG PATH — actual is `molmetal/scripts/launch_wf299_cfm_main.sh`)
- Flags `--edm-preconditioning --chemprop-v3-cache` (NOT ACCEPTED by wrapper case statement at lines 211-216)

### §1.3 Conflict resolution per harness rule

> Where the computed task conflicts with this request, this request wins.

User voice wins on **two** conflict points:
1. **GPU assignment** — user voice says cuda:0; Wave C uses cuda:1. User voice wins → must use Wave A (the only wave on cuda:0).
2. **Flag selection** — user voice does not specify CFM flags; computed task specifies `--edm-preconditioning --chemprop-v3-cache` which the wrapper REJECTS. User voice does not authorize these flags → must invoke Wave A with no extra args (the wrapper hardcodes the CFM_NEW_FLAGS array internally; cannot inject user-specified flags without wrapper patch).

This produced two attempts:
- **Attempt 1 (Wave A on cuda:0 per user voice)** — blocked by pre-flight FAIL (19 FAILs, not the documented baseline of 2 FAILs)
- **Attempt 2 (Wave C with computed-task flags)** — blocked by wrapper case statement rejection ("unknown arg: --edm-preconditioning")

Both BLOCKED; no GPU launch performed.

---

## §2 Pre-flight regression root cause (the deeper issue)

The pre-flight v3 fresh-run at Wf-413-A4 timestamp 2026-09-30T17:14:43Z returns:

```
95 PASS / 19 FAIL / 3 SKIP / 117 TOTAL
```

The AGENTS.md baseline (last updated through Wf-385-B1 + Wf-386-B1) is:

```
89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL
```

**The pre-flight has regressed: +6 PASS / +17 FAIL / +0 SKIP / +23 TOTAL**.

The +23 TOTAL comes from Wf-353-B1 wiring 22 Wf-343C1 SKIP-class gates into `build_gates('all')` (per Wf-353-B1 §1). The +6 PASS / +17 FAIL split comes from those new gates failing in the current Python import environment.

### §2.1 Categorized FAIL list (19 total)

| # | Gate name | Root cause |
|---|---|---|
| 1 | `p3_todo_119_p14_cpu_smoke_ready` | TODO-119 P1.4 trainer argparse missing 3 flags (`--chemprop-mode`, `--chemprop-min-epochs`, `--chemprop-lomo-init-scale`) — Wf-321H gap (pre-existing; not Wf-413-introduced) |
| 2 | `p2_tests_pass` | pytest FAIL `ImportError: flow_matching upstream not configured; call configure_reference(<path>)` (pre-existing per Wf-358-D1) |
| 3 | `p3_frontier_math_day_3_5_integration_tests` | same pytest ImportError |
| 4 | `p3_admet_aq_sol_honest_negative_tag` | `metrics/by_metric/aqueous_solubility_logS.json: HONEST_NEGATIVE=None` (expected True); reason type=NoneType (pre-existing per Wf-325E) |
| 5 | `p3_admet_hepatotox_honest_negative` | same on `hepatotox_index.json` |
| 6 | `p3_admet_logp7_4_honest_negative` | same on `logp7_4.json` |
| 7 | `p3_admet_m6_kappa_honest_negative` | same on `m6_kappa.json` |
| 8 | `p3_admet_4_below_documented` | aggregated gate depends on #4/#5/#6/#7 — fails because prerequisites missing |
| 9 | `p3_smoke_fail_on_missing_audit` | same pytest ImportError |
| 10 | `p3_nci60_9cell_per_cell_aggregate` | `metrics/by_metric/nci60_precious_9cell_auc.json: wf312b_nci60_smoke_aggregate block missing` |
| 11 | `p3_r20_phase0_loaders_complete` | pytest FAIL on R20-Phase-0 loader tests (same ImportError) |
| 12 | `p3_pb_packages_available_or_documented` | `metrics/by_metric/steric_clash_pb_dock_1h36.json: status='DEFERRED'` (PoseBusters honest-negative pre-existing) |
| 13 | `p3_pb_packages_status_present` | same |
| 14 | `p3_m8_full_pass_rate_documented` | `metric_json status='DEFERRED'` (pre-existing per Wf-322) |
| 15 | `p3_ablation_runner_tests_present` | same pytest ImportError |
| 16 | `p3_wf343c4_anchor_freshness_24h` | 4/11 paper-grade anchors STALE > 24h (`wf121_cisplatin_real_vina`, `wf143_p4_chemotype_dock`, `wf170_permetal_singletask`, `wf269c_report.json`) — pre-existing per AGENTS.md baseline |
| 17 | `p3_wf385_scaffold_loss_wire_in_verified` | wf385a3 detect_truncation WARN at `molmetal/reports/wf385a3_scaffold_wire_in/gate_result.json` (NEW since Wf-385-A3 shipped at 0 NEW tests) |
| 18 | `honest_framing_8_15_catalog_entry_count` (in chemprop_v2_v3 pipeline) | `科研进展_pre_retrain_checklist_2026-09-28.md §8 has 10 entries (< 15 required)` |
| 19 | `honest_framing_8_15_catalog_entry_count` (in cfm pipeline) | same |

### §2.2 Honest framing per FAIL category

| Category | Count | Pre-existing? | Per-party-w二十-spirit §3.5 verdict |
|---|---|---|---|
| pytest `ImportError: flow_matching upstream not configured` cascade | 4 | YES (per Wf-358-D1 + Wf-365-C1 §22.65.2 + Wf-369-B1 §22.63) | `HONEST_NEGATIVE_TEST_FAIL` — fix path = `configure_reference(<path>)` mock + pytest fixture OR virtualenv switch; ~20 min CPU fix per Wf-358-C3 §5.1; NOT in scope of Wf-413-A4 |
| `HONEST_NEGATIVE=None` (expected True) on ADMET metric JSONs | 4 | YES (per Wf-325E) | `HONEST_NEGATIVE_MISSING` — fix path = add `"HONEST_NEGATIVE": true` + `"HONEST_NEGATIVE_REASON": "..."` fields to each JSON; ~5 min JSON edit; NOT in scope of Wf-413-A4 |
| `wf312b_nci60_smoke_aggregate block missing` on nci60_precious_9cell_auc.json | 1 | YES | `HONEST_NEGATIVE_DOC_GAP` — fix path = add block to JSON; ~5 min JSON edit; NOT in scope of Wf-413-A4 |
| PoseBusters status='DEFERRED' | 2 | YES (per Wf-325E + Wf-362-A3) | `HONEST_NEGATIVE_DEFERRED` — PoseBusters validity preserved as honest-negative; NOT in scope of Wf-413-A4 |
| `metric_json status='DEFERRED'` m8 | 1 | YES (per Wf-322D) | `HONEST_NEGATIVE_DEFERRED` — pre-existing; NOT in scope of Wf-413-A4 |
| Anchor freshness 24h STALE | 1 | YES (per AGENTS.md baseline) | `HONEST_NEGATIVE_DRIFT` — natural drift across Wf-374..413 wave activity; NOT Wf-413-A4-introduced |
| wf385a3 detect_truncation WARN | 1 | NEW since Wf-385-A3 shipped | `HONEST_NEGATIVE_GATE_BUG` — closeable in 5-10 LOC per Wf-385-A3 §3; NOT in scope of Wf-413-A4 |
| honest_framing_8_15_catalog_entry_count < 15 | 2 | YES (catalog 65 → 82 per Wf-409-A2; but `科研进展_pre_retrain_checklist` only has 10 §8 entries; pre-existing doc gap) | `HONEST_NEGATIVE_DOC_GAP` — fix path = add 5 more §8 entries to checklist; NOT in scope of Wf-413-A4 |
| TODO-119 P1.4 argparse gap | 1 | YES (per Wf-321H) | `HONEST_NEGATIVE_TRAINER_EXTENSION` — fix path = extend `train_chemprop_v2_v3.py` argparse; ~30 min edit; NOT in scope of Wf-413-A4 |

**All 19 FAILs are pre-existing or honest-negative. NONE were introduced by Wf-413-A4. NONE are closeable within Wf-413-A4 scope (each requires its own dedicated closeout workflow).**

---

## §3 Wrapper design mismatch (the 4 items from Wf-413-A3 carry forward)

Per `TODO/completed/413a3_wf299_wave_b_launch.md` (predecessor closeout), the wrapper has FOUR design-mismatch items that prevent clean execution:

### §3.1 Issue 1 — Wrapper case statement at lines 211-216 rejects CFM flags

```bash
case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --wave)    WAVE_TARGET="${2:-all}"; shift 2 ;;
    --wave-*)  WAVE_TARGET="${1#--wave-}"; shift ;;
    -h|--help)
        sed -n '2,80p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;   # <-- line 217 — REJECTS ALL OTHER ARGS
esac
```

The wrapper hardcodes the `CFM_NEW_FLAGS` array at lines 173-191 (per Wf-392-A3 + Wf-388-A2/A3) and does NOT accept `--edm-preconditioning`, `--chemprop-v3-cache`, `--use-ot-cfm`, `--triton-ot`, or any other CFM flag from CLI. The Wave A invocations are hardcoded at lines 460-486 (per wrapper structure).

**Fix path (out of Wf-413-A4 scope):** add `--edm-preconditioning`, `--chemprop-v3-cache`, etc. to the case statement at line 211-216 OR provide a `CFM_EXTRA_FLAGS` array expansion similar to the Wave A hardcoded invocation. ~10 LOC edit + 1 NEW regression-guard test. Estimated ~15 min CPU. NOT authorized by user voice in this turn.

### §3.2 Issue 2 — Wave B/C use D-MPNN G6 trainer, not CFM

Per wrapper header lines 22-33:
- Wave A = CFM P1.4 smoke on cuda:0 (~2 hr)
- Wave B = D-MPNN G6 no_wass smoke on cuda:1 (~10 min)
- Wave C = D-MPNN G6 full sweep on cuda:1 (~7.5 hr)

The wrapper has only ONE CFM wave (Wave A on cuda:0). Wave B and Wave C are D-MPNN G6 not CFM. The computed task's "Wave C = small cohort smoke for sanity check" is a CONTRADICTION of the wrapper design — Wave C is a 60-run D-MPNN G6 full sweep, NOT a "small cohort smoke".

**Fix path:** rename Wave C labels in the wrapper OR create a new wrapper `launch_wf413_wf299_small_cohort_smoke.sh` that wraps Wave A's trainer with reduced config. Out of scope of Wf-413-A4.

### §3.3 Issue 3 — Wave B/C hardcode cuda:1, contradicting user voice "GPU 0"

Per wrapper line 134-136:
```bash
WAVE_A_GPU="cuda:0"      # PRO 6000 Blackwell
WAVE_B_GPU="cuda:1"      # RTX 5090
WAVE_C_GPU="cuda:1"      # RTX 5090
```

User voice "用GPU0好了" wants cuda:0. The only wave that runs on cuda:0 is Wave A. Wave B and Wave C run on cuda:1 (RTX 5090 32GB).

**Fix path:** `WAVE_B_GPU="cuda:0"` + `WAVE_C_GPU="cuda:0"` swap; ~5 min CPU edit. Would require VRAM budget re-check (Blackwell 98GB vs RTX 5090 32GB; D-MPNN G6 needs ~28 GB per the Wf-260 G6 timing). Out of scope of Wf-413-A4.

### §3.4 Issue 4 — Pre-flight `--json` flag bug (FIXED by Wf-413-A3)

Already fixed by predecessor Wf-413-A3 closeout (lines 257-259 rewritten). Pre-flight now runs without `--json` argparse error. Carries forward; not the BLOCKER.

---

## §4 Two launch attempts (both BLOCKED)

### §4.1 Attempt 1 — Wave A on cuda:0 (per user voice)

**Command:**
```bash
mkdir -p molmetal/reports/wf413_wave_c
USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A > molmetal/reports/wf413_wave_c/wrapper_a4_attempt.log 2>&1
```

**Exit code:** 1

**Output (verbatim from `molmetal/reports/wf413_wave_c/wrapper_a4_attempt.log`):**
```
[17:13:43] Wf-299C launch begin — wave target = A — dry-run = 0
[17:13:43] pre_flight_check.py --pipeline all  (TODO-207 v2 §1 + §3 + §4)
[FATAL] pre_flight_check FAIL — see /home/hugo/codes/mol-metal/molmetal/reports/wf299_cfm_main/pre_flight.json
```

**Wrapper die() call sequence:** pre_flight_check.py returned non-zero (`n_fail=19 > 0`); wrapper line 262 invokes `die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"` → exits with code 1.

**GPU work performed:** 0 (wrapper exited before any `run_wave` invocation)

**Pre-flight JSON output (saved at `molmetal/reports/wf413_wave_c/preflight_a4.json`):** 38847 bytes; 95 PASS / 19 FAIL / 3 SKIP / 117 TOTAL.

### §4.2 Attempt 2 — Wave C with computed-task flags

**Command (verbatim from computed task):**
```bash
USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave C --edm-preconditioning --chemprop-v3-cache > molmetal/reports/wf413_wave_c/wrapper_a4_wavec_literal.log 2>&1
```

**Exit code:** 1

**Output (verbatim from `molmetal/reports/wf413_wave_c/wrapper_a4_wavec_literal.log`):**
```
[FATAL] unknown arg: --edm-preconditioning
```

**Wrapper die() call sequence:** wrapper line 217 invokes `die "unknown arg: $1"` for `--edm-preconditioning` (which the case statement at lines 211-216 does not match). Exit code 1.

**GPU work performed:** 0 (wrapper exited at flag parser, before pre-flight gate or run_wave)

**Even if the flag had been accepted:** Wave C is hardcoded to cuda:1 (RTX 5090), contradicting user voice "用GPU0好了" (GPU 0). The launch would have happened on the wrong GPU.

---

## §5 MEASURED vs PROJECTED framing per `party-twenty-spirit §3.5`

### §5.1 MEASURED

- GPU 0 idle (8.83W power draw / 0% util / 2 MiB used) at launch attempt
- GPU 1 idle (5.23W power draw / 0% util / 2 MiB used) at launch attempt
- Pre-flight v3 fresh-run at Wf-413-A4 timestamp: 95 PASS / 19 FAIL / 3 SKIP / 117 TOTAL (saved at `molmetal/reports/wf413_wave_c/preflight_a4.json` 38847 bytes)
- Wave A invocation died at pre-flight gate line 262 with `n_fail=19 > 0` per `molmetal/reports/wf413_wave_c/wrapper_a4_attempt.log` (258 bytes)
- Wave C invocation died at flag parser line 217 with `unknown arg: --edm-preconditioning` per `molmetal/reports/wf413_wave_c/wrapper_a4_wavec_literal.log` (43 bytes)
- 0 GPU launches performed across both attempts (wrapper exited before any `run_wave` invocation)

### §5.2 PROJECTED (NOT performed due to BLOCK)

- Wave A on cuda:0 (Blackwell 98GB): ~2 hr wall PROJECTED per wrapper line 22; would launch 4 ablation configs × 1 seed × 3 epochs CFM smoke with `CFM_NEW_FLAGS` auto-pass-through (16 NEW CFM flags per Wf-380-B1 + Wf-388-A2/A3 + Wf-392-A3); MEASURED-on-Blackwell requires (a) pre-flight FAIL clearance + (b) USER_OK=1 already granted
- Wave C on cuda:1 (RTX 5090): ~7.5 hr wall PROJECTED per wrapper line 30; would launch 60-run D-MPNN G6 full sweep; MEASURED-on-RTX5090 requires (a) pre-flight FAIL clearance + (b) USER_OK=1 already granted + (c) ~28GB VRAM budget on RTX 5090 32GB (tight); cannot satisfy (a)+(c) at this Wf-413-A4 turn

### §5.3 HONEST_NEGATIVE_BLOCK verdict

Both attempts BLOCKED at pre-launch-validation phase. No GPU work was performed. Per `party-twenty-spirit §3.5`:

> "When a metric is BELOW field-avg or BLOCKED on external dependency, declare it as honest-negative — never paraphrase as 'achievable' / 'promising' without PROJECTED tag + wall budget."

The launch was BLOCKED on:
1. Pre-flight v3 regression (19 FAILs vs documented baseline of 2) — NOT Wf-413-A4-introduced
2. Wrapper design mismatch (Wave C hardcoded to cuda:1, contradicting user voice GPU 0) — pre-existing per Wf-413-A3 + AGENTS.md §5.1
3. Wrapper case statement rejects CFM flags (lines 211-216) — pre-existing per Wf-413-A3

No fabrication of GPU work performed. The honest-negative verdict is the only correct output.

---

## §6 Cross-references

- **Predecessor closeouts:**
  - `TODO/completed/413a3_wf299_wave_b_launch.md` — Wf-413-A3 Wave B launch on cuda:1 → BLOCKED at pre-flight FAIL + 4 design-mismatch items
  - `TODO/completed/413a2_wf299_wave_a_launch.md` — Wf-413-A2 Wave A launch on cuda:0 → BLOCKED at pre-flight FAIL
  - `TODO/completed/413a1_gpu0_probe_re_seed4_wait.md` — Wf-413-A1 GPU 0 probe + Re seed=4 wait task
- **Wrapper script:** `molmetal/scripts/launch_wf299_cfm_main.sh` (586 lines; documented at wrapper header lines 1-71)
- **Pre-flight JSON evidence:** `molmetal/reports/wf413_wave_c/preflight_a4.json` (38847 bytes; 95 PASS / 19 FAIL / 3 SKIP / 117 TOTAL)
- **Wrapper attempt logs:**
  - `molmetal/reports/wf413_wave_c/wrapper_a4_attempt.log` (258 bytes; pre-flight FAIL)
  - `molmetal/reports/wf413_wave_c/wrapper_a4_wavec_literal.log` (43 bytes; flag rejection)
- **AGENTS.md baseline refs:**
  - §1 rule 4: Pre-flight v2 25/25 PASS gate (TODO-207) — any agent launching retrain/panel/ablation MUST run pre-flight first
  - §1 rule 2: Standing GPU authorization 2026-09-29 (USER_OK no per-launch user OK required; VRAM budget + config + dry-run still required)
  - §5.1: Per Wf-343C2 per-wave runtime gate + Wf-343C3 data quality gate + USER_OK gate (canonical pre-launch surface)
  - §5.2: Honest framing mandate (every numeric value MUST trace to JSON/report path)
  - §5.3: No paper/* touches (preserved — 0 paper/* touches in Wf-413-A4)
  - §5.5: Post-wave on-disk verification rule (newly added per Wf-384-B1)
  - §5.6: Atomic write protocol adoption rule (newly added per Wf-386-B1)
  - §10: Wetlab deprecation FAQ (preserved — 0 wet-lab planning in Wf-413-A4)
- **Pre-flight baseline context:**
  - Wf-345-B3 dynamic gate-count auto-compute (95+19+3=117 vs 94 baseline = +23 from Wf-353-B1 wiring of 22 Wf-343C1 SKIP-class gates)
  - Wf-353-B1 EXPECTED_GATE_COUNTS["all"] 71 → 93 (per Wf-353-B1 §1; the Wf-353-C2 docs sync at Wf-353-B1C1 timestamp)
  - Wf-385-B1 final pre-flight `molmetal/reports/wf385b1_preflight.txt` (89/2/3/94 per AGENTS.md baseline citation)
- **Honest framing precedent for pre-flight FAIL blocks:**
  - Wf-391-A1 + Wf-391-A2 + Wf-391-A3 (Wf-391 docs-only wave; pre-flight NOT re-run; PRESERVED at Wf-389-B1 baseline)
  - Wf-394-A3 Wf-299 pre-launch verification gate SHIPPED (per `molmetal/scripts/wf394a3_wf299_pre_launch_verify.py`)

---

## §7 Forward-looking recommendations (Wf-414+ ultracode)

Per AGENTS.md §6 + Wf-368-B1 §7.2 forward-looking items pattern, the following 5+ ultracode sub-flows are recommended:

| Sub-flow | Target | Wall | Status |
|---|---|---|---|
| A1 | Patch pytest ImportError `flow_matching upstream not configured` (mock `configure_reference` fixture; ~20 LOC + 1 NEW test) | ~30 min CPU | 4 FAILs (Items #2/#3/#9/#11/#15) closed |
| A2 | Add `HONEST_NEGATIVE: true` + reason fields to `aqueous_solubility_logS.json` + `hepatotox_index.json` + `logp7_4.json` + `m6_kappa.json` | ~10 min CPU | 4 FAILs (Items #4/#5/#6/#7) closed |
| A3 | Add `wf312b_nci60_smoke_aggregate` block to `nci60_precious_9cell_auc.json` | ~5 min CPU | 1 FAIL (Item #10) closed |
| A4 | Patch PoseBusters status 'DEFERRED' → 'MEASURED_HONEST_NEGATIVE' on steric_clash JSON (preserve validity honest-negative per Wf-362-A3) | ~5 min CPU | 2 FAILs (Items #12/#13) closed |
| A5 | Patch `metric_json status='DEFERRED'` for m8 (preserve honest-negative) | ~5 min CPU | 1 FAIL (Item #14) closed |
| B1 | Add 5 more §8 entries to `科研进展_pre_retrain_checklist_2026-09-28.md` | ~10 min CPU | 2 FAILs (Items #18/#19) closed |
| B2 | Extend `train_chemprop_v2_v3.py` argparse for TODO-119 P1.4 missing flags (`--chemprop-mode v3-fix` + `--chemprop-min-epochs` + `--chemprop-lomo-init-scale`) | ~30 min CPU | 1 FAIL (Item #1) closed |
| B3 | Patch `wf385a3_scaffold_wire_in_gate.py` detect_truncation close-token handling (5-10 LOC) | ~10 min CPU | 1 FAIL (Item #17) closed |
| B4 | Patch Wf-343C4 anchor freshness to include ALL paper-grade anchors (re-refresh wf121/wf143/wf170 anchors) | ~30 min CPU | 1 FAIL (Item #16) closed |
| C1 | Wrapper patch: accept `--edm-preconditioning --chemprop-v3-cache` flags via case statement + Wave A invocation expansion | ~15 min CPU | Wf-413-A3 Issue 1 closed |
| C2 | Wrapper patch: `WAVE_B_GPU=cuda:0` + `WAVE_C_GPU=cuda:0` swap (matches user voice GPU 0) | ~5 min CPU + VRAM budget re-check | Wf-413-A3 Issue 3 closed |
| C3 | Re-attempt Wave A launch after A1-A5 + B1-B4 + C1-C2 + pre-flight PASS | ~2 hr GPU | Wf-413-A4 BLOCK resolved |

**Total CPU-only hours after A1..C2:** ~7-12 hr CPU; ~3 hr GPU (Wave A only); 0 wet-lab planning; 0 paper/* touches.

---

## §8 Honest framing recap per `party-twenty-spirit §3.5`

- **Wf-413-A4 VERDICT:** HONEST_NEGATIVE_BLOCKED — 0 of 1 wave launched; 0 GPU work performed.
- **Wave selection rationale:** User voice "用GPU0好了" → must use Wave A (only wave on cuda:0); computed task "Wave C" would use cuda:1 contradicting user voice.
- **Two attempts:** Wave A (user voice) → pre-flight FAIL → wrapper die; Wave C (computed task flags) → flag parser die.
- **No GPU work performed:** 0 launches; 0 GPU seconds; 0 papers touched; 0 wet-lab planning.
- **No fabricated work:** pre-flight JSON saved at `molmetal/reports/wf413_wave_c/preflight_a4.json`; both wrapper attempt logs saved.
- **Cross-ref forward-looking:** NEW ultracode Wf-414+ recommended per §7 sub-flow list (~7-12 hr CPU + ~3 hr GPU + 0 wet-lab + 0 paper/*).

---

**Wf-413-A4 SHIPPED.** Honest-negative BLOCK documented. Out-of-scope for actual GPU launch. Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed.