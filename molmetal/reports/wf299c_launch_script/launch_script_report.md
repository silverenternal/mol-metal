# Wf-299C — TODO-119 P1.4 Launch Script Report

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Status**: READY on user OK (per Wf-295D)
**Constraint**: party-twenty-spirit §3.5 — every number MEASURED or honest-PROJECTED; NO `paper/` directory writes.

---

## §1 Deliverable

| Path | Purpose |
|---|---|
| `molmetal/scripts/launch_wf299_cfm_main.sh` | Three-wave orchestration shell (Wave-A P1.4 smoke → Wave-B G6 no_wass smoke → Wave-C G6 full sweep); per-wave OK gate (< 2 hr); per-wave abort criteria (TODO-207 v2 §7.2 8 patterns); pre-launch checks (`pre_flight_check.py` + `per_wave_checklist.py` + `wave_budget_tracker.py`); `--dry-run` mode. |

---

## §2 Wave Decomposition

| Wave | Scope | GPU | Wall-time (PROJECTED) | Source yaml | Source waves section |
|---:|---|---|---:|---|---|
| **A** | TODO-119 P1.4 smoke — 4 ablation configs × 1 seed × 3 epochs (a_full_stack / b_yuelbond_only / c_w2_only / d_catherine_only) | `cuda:0` (PRO 6000 Blackwell) | 2.0 hr | `configs/cfm_per_innovation_ablation_p1.4.yaml` | §waves (wave_a..wave_d, total_wall_time_estimate=2.0) |
| **B** | TODO-182 G6 C12 no_wass smoke — 2 metals × seed 0 (Ru + Ir) | `cuda:1` (RTX 5090) | 0.5 hr (smoke; user spec 10 min × 3 safety) | `configs/wf124_p3_per_innovation_ablation_g6.yaml` | §waves.wave_c (12 runs, 1.5 hr) compressed to 2-metal smoke |
| **C** | TODO-182 G6 full sweep — 60 runs × 5 metals × 3 seeds (4 ablation configs) split into 8 sub-waves × < 2 hr each | `cuda:1` (RTX 5090) | 7.5 hr (paper-grade subset; full 60-run PROJECTED = 12.5 hr per g6 yaml §waves.total_wall_time_estimate) | same as Wave-B | §runs.matrix (60 entries) + §waves (8 sub-waves a..h) |

**n_waves = 3** (top-level) + 8 sub-waves inside Wave-C.

---

## §3 Per-wave Wall-Time Budget (kill_then_adapt < 2 hr/wave hard cap)

| Wave | PROJECTED hr | OK gate (kill_then_adapt) | Abort on over-budget? |
|---:|---:|:---:|:---:|
| A | 2.0 | < 2 hr | yes → `abort_recovery.py --pattern TIMEOUT` (TODO-207 v2 §7.2.1) |
| B | 0.5 | < 2 hr | yes (smoke; < 30 min expected) |
| C | 7.5 | **< 2 hr/sub-wave** (split into 8 sub-waves per g6 yaml §waves; aggregate 7.5 hr is a paper-grade cap, NOT a per-wave OK gate) | yes per sub-wave; `per_wave_user_ok_required: true` between sub-waves (TODO-119 P1.4 §directives) |

Honest framing (party-twenty-spirit §3.5): all wall-time figures are PROJECTED (from yaml §waves + Wf-258 audit §1.3); MEASURED figures populate after each wave runs.

---

## §4 Per-wave Abort Criteria (TODO-207 v2 §7.2 — **abort_criteria_count = 8**)

The script implements all 8 abort patterns from `TODO/completed/207_unified_pre_flight_checklist_v2.md` §7.2:

| # | Pattern | Detection (in-script) | Recovery hook |
|---:|---|---|---|
| **7.2.1** | per-wave timeout (ckpt older than 2× projected wall-time) | `kill_then_adapt` deadline check (every 60s poll) | `abort_recovery.py --pattern TIMEOUT` |
| **7.2.2** | 3+ consecutive NaN/OOM/IndexError on same metal | post-wave `grep "NaN\|OOM\|IndexError" train.log` | `abort_recovery.py --pattern NaN` |
| **7.2.3** | GPU util < 70% for 5+ min | `nvidia-smi` polling inside run_wave loop | advisory + falls back to other GPU |
| **7.2.4** | Os AUC < 0.51 (Os collapse) | per-seed JSON `test_auc < 0.51` (post-wave) | `--single-metal Os --no-multi-task` (TODO-207 §1.2.1) |
| **7.2.5** | CFM decode collapse (min per_epoch decode_ratio < 0.05) | post-wave `grep min per_epoch decode_ratio` | `abort_recovery.py --cfm-collapse --restore-cf124p2-ckpt` |
| **7.2.6** | m7 Tanimoto < 0.04 | per-wave JSON `tanimoto_diversity < 0.04` | DEFER m7 claim (paper §6 honest-negative) |
| **7.2.7** | CHECKPOINT_LOADED_FORWARD_DEFERRED (forward-pass only, no `epoch_*` keys) | post-wave `jq 'keys \| contains(["epoch_*"])'` | mark honest-negative, NOT FAIL (TODO-207 §7.2.7) |
| **7.2.8** | partial pilot (n_completed < n_total) | post-wave `n_completed < n_total` | never cite as full; report `n/total partial` |

**abort_criteria_count = 8** (matches TODO-207 v2 §7.2 enumeration).

---

## §5 Pre-launch Checks (BLOCKER before any wave)

Per TODO-207 v2 §1 (P0 BLOCKER) + §6 (fail-fast chain) + §7 (wave budget tracker):

| # | Script | Gate | Severity |
|---:|---|---|---|
| 1 | `molmetal/scripts/pre_flight_check.py --pipeline all` | §1.1 user OK + §1.2 GPU live + §1.3 YAML schema + §1.4 CSV hash (24/24 target per Wf-260 calibration) | BLOCKER |
| 2 | `molmetal/scripts/per_wave_checklist.py --wave wave_<X>` | per-wave sub-checklist emit (NEW per Wf-262 §10.1) | VERIFY |
| 3 | `molmetal/scripts/wave_budget_tracker.py --budget-hr <hr>` | 10h × N-wave cap (`--no-enforce-budget` opt-out supported per TODO-207 §7.1.1) | DOCUMENTED |

All three are invoked BEFORE the first wave; pre-flight re-runs after each wave per kill_then_adapt standing directive.

---

## §6 kill_then_adapt Semantics (Per TODO-119 P1.4 §directives + Wf-295D)

1. **Per-wave user OK gate** between waves (per `per_wave_user_ok_required: true`); `--dry-run` skips gate.
2. **< 2 hr/wave hard cap** — Wave A and Wave B trigger `abort_recovery.py` on breach (TODO-207 v2 §7.2.1); Wave C's 7.5 hr aggregate is split into 8 sub-waves × < 2 hr each, with `per_wave_user_ok_required` between sub-waves.
3. **Per-wave OK gate** = pre_flight + per_wave_checklist + budget_tracker all re-emit; any FAIL → abort (TODO-207 §6 fail-fast chain).
4. **Honest framing preserved** (party-twenty-spirit §3.5): every PROJECTED number stays PROJECTED until MEASURED post-wave; per-wave JSON written to `molmetal/reports/wf299_cfm_main/wave_<X>/<sub>.json` for downstream audit.

---

## §7 Honest Framing (party-twenty-spirit §3.5)

**DO claim (post-Wf-299C)**:
- "Wf-299C orchestration script ships at `molmetal/scripts/launch_wf299_cfm_main.sh` with 3 waves (A/B/C), 8 sub-waves inside Wave-C, 8 abort criteria (TODO-207 v2 §7.2), 3 pre-launch checks (pre_flight_check + per_wave_checklist + wave_budget_tracker), per-wave < 2 hr kill_then_adapt OK gate, --dry-run mode."
- "Per-wave PROJECTED wall-times: A=2.0 hr, B=0.5 hr, C=7.5 hr (paper-grade subset; full 60-run PROJECTED = 12.5 hr per g6 yaml)."

**DO NOT claim (until waves run)**:
- "Wave A/B/C MEASURED wall-time" (PROJECTED until run).
- "Per-lever attribution on the G6 60-run sweep" (60 runs PENDING per TODO-182 §6).
- "Wave C sub-wave per_wave_user_ok auto-granted" (always requires user OK between sub-waves).

---

## §8 Cross-references

- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §1 (P0 BLOCKER), §6 (fail-fast chain), §7 (wave budget + abort criteria), §8 (pre_flight_check.py auto-verify).
- `configs/cfm_per_innovation_ablation_p1.4.yaml` (Wave A source spec).
- `configs/wf124_p3_per_innovation_ablation_g6.yaml` (Wave B + C source spec).
- `molmetal/scripts/pre_flight_check.py` (Wf-262 §0.5 SHIPPED).
- `molmetal/scripts/per_wave_checklist.py` (Wf-262 §0.5 SHIPPED).
- `molmetal/scripts/wave_budget_tracker.py` (Wf-262 §0.5 SHIPPED).
- `molmetal/scripts/abort_recovery.py` (Wf-262 §0.5 SHIPPED).
- `molmetal/reports/wf295d_launch_sequence/` (parent launch-sequence readiness, per task text).
- `molmetal/reports/wf262_audit/` (4-script ship acceptance gates).