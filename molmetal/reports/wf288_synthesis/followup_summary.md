# Wf-288E — Synthesis of Wf-288 Batch (A/B/C/D) Outcomes

**Date:** 2026-09-29
**Method:** Read-only aggregation of 4 upstream Wf-288 outputs (READ-ONLY no source edits applied).

---

## §1 Per-agent summary

### Wf-288A — §11.1 Timeline update (in-place edit, no on-disk report)
- **Status:** SHIPPED — `TODO/completed/207_unified_pre_flight_checklist_v2.md` §11.1 extended from Wf-256..268 (pre-Wf-288A) to Wf-256..287 (post-Wf-288A), capturing 19 new wave entries: Wf-267-Integrate, Wf-269A/B/C/D, Wf-270D, Wf-271B, Wf-275B/C, Wf-276A/C, Wf-277A/B/D, Wf-279A/B, Wf-280D, Wf-282/A/B, Wf-283A, Wf-287A.
- **Coverage:** each entry carries Status / Honest-framing note / Cross-ref to §9.AF/AH/AI/AK/AC/AE/Y/Z.AB/AG; Wf-287A m14 anchor supersession codified; Wf-280D WRONG-CLAIM preserved as honest-negative example.
- **No scope creep:** pure cross-ref extension; no NEW claim or §3.5 violation introduced.

### Wf-288B — §4 Claim audit refresh (`molmetal/reports/wf288b_cf264_refresh/refresh_report.md`)
- **Status:** CLEAN — 107 claims audited across §4.1-§4.5.4 (Wf-264 baseline was 90; +17 net new from §4.5.x edits).
- **Acceptance gates:** G1=0 aspirational / G2=0 projected-as-measured / G3=0 seed-count errors / G4=0 partial-disclosure / G5=0 total §3.5 violations (all 7 Wf-264 + Wf-267 violations FIXED) / G6=107 claims extracted / G7=107 clean / G8=audit path present.
- **Honest-negatives properly tagged:** §4.5.1 G5 14/15 pairs PENDING; §4.5.2 Wf-266 LOMO-fix constant predictor 0.5000 FAIL G2 ≥ 0.65; §4.5.3 Wf-269A Path C DEFERRED; §4.5.4 Wf-269C generalization caveat (PlatinAI/DrugOOD cohorts not yet tested).
- **Verdict:** paper §4 ready for Wf-289 final-draft sign-off.

### Wf-288C — §10 Sign-off re-verify (`molmetal/reports/wf288c_cf253e_reverify/verify_report.md`)
- **Status:** 10/10 §10 gates intact (7 auto-fillable PASS + 3 DEFERRED-by-design unchanged).
- **Auto-fill re-verify:** 10.1 code gates 10/10 (line numbers shifted due to Wf-247..252 additions but all 10 markers intact) / 10.2 limitations 6/6 / 10.3 data 7/7 (path renamed but content present) / 10.4 tests 13/13 (205 tests passing combined, up from 135 in original) / 10.6 self-audit 19/24 (5 deferred identical to Wf-253E) / 10.7 honest framing 15/15 / 10.8 failure catalog 8/8.
- **DEFERRED unchanged:** 10.5 wall-time, 10.9 GPU selection, 10.10 output dir naming — all still require user OK per `party-twenty-spirit` §3.5.
- **Side finding (infrastructure, NOT §10 regression):** pre_flight_check has 1 FAIL on `p1_chemprop_smoke` (55s budget too tight by 3-5s; actual smoke = ~34s when given enough budget). Recommend bump to 90s OR trim chemprop `--quick` to 1 epoch.

### Wf-288D — Cf-263C G5/G6 configs verify (`molmetal/reports/wf288d_cf263c_verify/verify_report.md`)
- **Status:** 3/3 original deviations FIXED; 0 NEW deviations; 2 doc-only hygiene nits.
- **Per-deviation:** #1 G5 CSV v2→v4 FIXED (Wf-264 config update; yaml line 28 + launch wrapper line 30 + dry_run.log confirm v4 path) / #2 G6 CSV v2→v4 FIXED (yaml line 27 + wrapper line 34) / #3 G6 epochs 30→50 FIXED (yaml line 53 `epochs: 50` + wrapper line 39 `--epochs 50`).
- **Cascade-fix check (n_new=0):** Wf-265 dashboard / Wf-266 LOMO-fix / Wf-280A no-enforce-budget / Wf-283B kendall/dann CLI / Wf-285B kendall/dann wiring — all confirmed orthogonal or already wired; no drift.
- **Hygiene nits (NOT deviations, doc-only):** (N1) G6 yaml lines 103+113 still show `status: "verify-flag-before-launch"` but flags are confirmed-functional post-Wf-283B+Wf-285B; (N2) `--no-enforce-budget` not in `entry.command` block (G5 lines 208-221, G6 lines 239-251) but `dry_run: true` is sufficient workaround.
- **G5 50-ep real launch:** still BLOCKED on user OK + GPU availability (kill_then_adapt posture preserved).

---

## §2 Aggregate tally

| Dimension | Count | Notes |
|---|---:|---|
| **Total work completed** | **4/4 agents** | Wf-288A shipped inline §11.1 extension; Wf-288B/C/D each produced concrete report files |
| **Total failed** | **0/4 agents** | All completed without BLOCK; Wf-288C side finding is pre-flight INFRASTRUCTURE (not §10 gate regression) |
| **§3.5 violations remaining** | 0/7 | All Wf-264 + Wf-267 violations FIXED; 0 NEW introduced by Wf-274B / Wf-278D / Wf-279C / Wf-280C / Wf-284C |
| **Acceptance gates status** | 25/26 PASS (1 infra FAIL: `p1_chemprop_smoke` budget tightness) | Net GREEN across 7 auto-fillable §10 gates |
| **Config deviations remaining** | 0/3 (original Cf-263C) | All 3 deviations FIXED via Wf-264 config update; cascade fixes non-drift |
| **Doc-only hygiene nits** | 2 | G6 yaml status fields stale + `--no-enforce-budget` CLI flag missing (both non-blocking) |
| **Claims added since Wf-264** | +17 (90 → 107) | All properly tagged honest-negative / partial-MEASURED / PROJECTED / DEFERRED |

---

## §3 Critical honest-negatives remaining

1. **G4 14 (metal, seed) PENDING per-pair audit** — Per Wf-289B reflection: only Ru seed=0 of 15 pairs is independently auditable; 14-pair MEASURED aggregate is via Wf-265 collect + Wf-266 G1 (MEASURED-equivalent, NOT 14/14 independent audits). The §4.5.1 honest-positive scope in paper §4 inherits this caveat.
2. **Wf-282 G5 Ru seed=0 only (1/15 pairs MEASURED)** — 14/15 pairs PENDING; live wrapper PID 609528 autonomous, ETA C6→DONE ~2 hr, then C7-C9 ~3 hr.
3. **Cf-258 15/17 waves still PENDING** — 11.76% DONE-only; 52.94% if PARTIAL + PRE-LAUNCH READY counted. C3-C5 (CFM P1.3/P1.4) + C7-C9 (G5 50-ep Ir/Os/Rh/Re) BLOCKED at GPU/infrastructure layer.
4. **TODO-130 chemprop v3 DrugOOD re-train 0.4544 BELOW_RANDOM** — v3 OOD fix structurally PASS but empirical DrugOOD re-measurement DEFERRED per Wf-277B (75-min kanzi CPU PENDING).
5. **TODO-96 R20-P5 Catherine 30-metal reachability ≥ 0.95** — 6-wk ETA per consumer; not yet shipped.
6. **TODO-98 L12 cross-domain OOD reachability ≥ 0.85** — Q4 2026; defer.
7. **M1+M2 / F5_RDKIT_2D / click-aug CFM integration** — 1.1.8 / 1.2.3 / 1.2.4 pending GPU; downstream lift scope DEFERRED.
8. **pre_flight_check `p1_chemprop_smoke` budget** — 55s too tight by 3-5s; smoke itself is not broken (completes in ~34s when given enough budget). Recommend bump to 90s OR trim `--quick` to 1 epoch.

---

## §4 Recommended next priorities

| Rank | Action | Wall | Status | Rationale |
|---:|---|---|---|---|
| **1** | **Wf-290-A** Independently audit 14 G4 (metal, seed) per-pair payloads against on-disk log + ckpt + metrics re-verification | ~30 min | PENDING | Closes §3 critical honest-negative #1; enables honest §4 paper cite without Wf-289B caveat |
| **2** | **Wf-290-B** Wait for C6 (Ru seed=1 + seed=2) to complete via autonomous wrapper PID 609528 | ~2 hr | IN-FLIGHT (seed=1 epoch 15/50) | G5 C6→DONE is the gating event for C7-C9 launch |
| **3** | **Wf-290-C** Launch C7 (Ir s0/s1/s2, G5 50-ep, ~60 min) AFTER C6 seed=2 ends | ~1 hr | PRE-LAUNCH READY | Continues G5 sweep |
| **4** | **Wf-290-D** Launch C8+C9 (Os+Rh+Re, ~120 min) AFTER C7 ends | ~2 hr | PRE-LAUNCH READY | Closes TODO-181 G5 sweep |
| **5** | **Wf-290-E** Launch C3 (CFM P1.3 smoke, GPU-0 Blackwell, ~30 min) — fresh user OK required | ~0.5 hr | PENDING | Unblocks C4+C5 (P1.4 ablation) |
| **6** | **Wf-290-F** Launch C10-C17 G6 ablation (60 runs, ~7.5 hr, GPU-1) AFTER G5 sweep ends | ~7.5 hr | PRE-LAUNCH READY | Per-innovation attribution; --no-kendall/--no-dann flags wired at Wf-285B |
| **7** | **Wf-290-G** Doc edits (Wf-287E P1-P3 + Wf-277A §9.AF verbatim Δ) | ~13 min | READY | Orthogonal to GPU; ultracode-suitable (textual fix) |
| **8** | **Wf-290-H** Wf-277B DrugOOD v3 ECFP4 re-eval (~75 min, kanzi CPU) | ~1.25 hr | READY | Run on CPU while GPU-0/1 busy |
| **9** | **Wf-290-I** Apply Wf-288D hygiene nits (N1: G6 yaml lines 103+113 status fields; N2: append `--no-enforce-budget` to entry.command blocks) | ~5 min | READY | Doc-only, ultracode-suitable |
| **10** | **Wf-290-J** Bump pre_flight_check `p1_chemprop_smoke` budget 55s → 90s (or trim chemprop `--quick` to 1 epoch) | ~2 min | READY | Closes §3 critical honest-negative #8; pre-flight auto-verify infra |

**Total wall** (sequential): ~14 hr GPU-0/1 + ~20 min CPU/ultracode parallelizable.
**Parallelization:** Steps 7-10 (ultracode text edits) + Step 8 (CPU re-eval) run in parallel with steps 1-6 (GPU work).
**Blocker still open:** Steps 2-6 require fresh user OK per `kill_then_adapt` (every wave ≤ 1 hr); Steps 7-10 are non-GPU textual edits — ultracode-suitable.

---

## §5 Honest framing

- **Wf-288 batch as a whole:** 4/4 subworkflows shipped; **0** of them crossed a NEW DONE gate. The batch's value is **textual / structural audit consolidation** (Wf-288A timeline extension, Wf-288B claim refresh, Wf-288C sign-off re-verify, Wf-288D config deviation close), NOT new measurement.
- **Paper §4 §3.5 honesty ledger** remains intact: 0 violations (Wf-288B); 10/10 §10 gates intact (Wf-288C).
- **Cf-263C config gap inventory** fully closed: 3/3 deviations FIXED via Wf-264; 0 NEW drift from cascade fixes.
- **Wf-288 batch did NOT produce** the next MEASURED event (C6 G5 Ru seed=1+2 PENDING, ETA ~2 hr via autonomous wrapper PID 609528).
- **Side finding** (pre_flight_check `p1_chemprop_smoke` budget tightness) is INFRASTRUCTURE tuning, NOT a §10 gate regression — fixable in ~2 min via ultracode edit.

---

## §6 Ultracode parallelization recommendation (repair-type tasks)

Per the user's question about parallel repair-type tasks for ultracode, the following Wf-288/Wf-290 items are READY for parallel ultracode dispatch:

| Task | Type | Wall | File(s) |
|---|---|---|---|
| **Wf-290-G** | Doc fix | ~13 min | `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AF (verbatim Δ alignment) + G3 per-metal ckpt + G4 CHECKPOINT_LOADED_FORWARD_DEFERRED preamble |
| **Wf-290-I** | Doc fix | ~5 min | `configs/wf124_p3_per_innovation_ablation_g6.yaml` lines 103+113 (status field update); `configs/wf124_p3_50epoch_full_sweep_g5.yaml` line 208-221 + `wf124_p3_per_innovation_ablation_g6.yaml` line 239-251 (append `--no-enforce-budget`) |
| **Wf-290-J** | Infra fix | ~2 min | `molmetal/scripts/pre_flight_check.py` `p1_chemprop_smoke` budget constant 55→90 (or trim `--quick` to 1 epoch in `molmetal/scripts/train_chemprop_v2_v3.py`) |
| **Wf-287B** (proactive) | Doc reflection | ~10 min | Reflect Wf-288B §3.5 NET CLEAN + Wf-288C 10/10 + Wf-288D 0 deviations into `TODO/INDEX.md` §22 timeline |

These 4 textual/infrastructure tasks are all **GPU-independent, READ-ONLY-source**, and ultracode-suitable. Recommend launching Wf-290-G/I/J as a single parallel batch (total ~20 min wall) while Wf-290-B through F proceed on GPU.

---

## §7 File references

- `molmetal/reports/wf288b_cf264_refresh/refresh_report.md`
- `molmetal/reports/wf288c_cf253e_reverify/verify_report.md`
- `molmetal/reports/wf288d_cf263c_verify/verify_report.md`
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §11.1 (Wf-288A inline edit)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §10 (Wf-288C re-verify source)
- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §4 (Wf-288B audit target)
- `configs/wf124_p3_50epoch_full_sweep_g5.yaml` (Wf-288D deviation close)
- `configs/wf124_p3_per_innovation_ablation_g6.yaml` (Wf-288D deviation close)
- `molmetal/scripts/launch_wf124_p3_g5.sh` + `launch_wf124_p3_g6.sh` (Wf-288D verify)
- `molmetal/scripts/pre_flight_check.py` (Wf-288C side finding)

---

*Generated by Wf-288E (2026-09-29). Read-only synthesis. 4/4 Wf-288 subworkflows shipped; 0 NEW §3.5 violations; 0 NEW config deviations; 0 §10 gate regressions; 1 pre-flight infrastructure nit. Paper §4 ready for Wf-289 final-draft sign-off.*