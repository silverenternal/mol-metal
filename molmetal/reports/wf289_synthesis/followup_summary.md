# Wf-289E — Synthesis of Wf-289 Batch (A/B/C/D) Outcomes

**Date:** 2026-09-29
**Method:** Read-only aggregation of 4 upstream Wf-289 outputs.

---

## §1 Per-agent summary

### Wf-289A — Broken imports fix (`molmetal/reports/wf289a_broken_imports/fix_report.md`)
- **Status**: COMPLETE — all 9 Wf-268C-flagged broken imports resolved at module-import time.
- **Outcomes**: collection errors 9→0; tests collected 7853→8984 (+1131 newly discoverable); wall-time 12.70s→8.42s (−34%).
- **Source-level fixes**: 3 defined-missing-symbols (`METAL_LIGAND_EXCHANGE_MATRIX` 84-rule + 17 NEW SMARTS in `pt_metal_ligand_exchange.py`; `METAL_SYMBOLS` 30-d-block in `atoms/combinators.py`), 1 repointed-broken-path (`_SYNTHEMOL_PATH` to repo-relative + re-entrant bootstrap in `syntemol_reactions.py`), 4 no-op cascade failures, 1 no-op marker.
- **Honest framing**: no test deleted/skipped, no mock fallback added; the `pt_metal_ligand_exchange.py` matrix + 17 NEW SMARTS are L-R17-B2 ship surface (TODO-31/TODO-34), not Wf-289A scope creep.

### Wf-289B — G4 14 PENDING caveat reflection (text-only, no on-disk report)
- **Status**: REFLECTED into TODO-188 §4.5.1 + TODO-207 §9.AI.4 + §9.Z Z.1.
- **Caveat**: Wf-268C honest framing clarifies that the §4.5.1 MEASURED fallback actuals (Ru 0.7527 / Ir 0.6702 / Os 0.7025 / Rh 0.7227 / Re 0.7700; overall 0.7236 ± 0.0439) are **MEASURED-equivalents** from Wf-124 P3 5×3×30 protocol aggregated via Wf-265 collect; only **Ru seed=0** (test_auc=0.7613, `wf124_p3_multitask/ru/per_seed_metrics_seed0.json`) is independently auditable; other 14 (metal, seed) pairs are **PENDING** per TODO-180 §1.
- **Why this matters**: §9.AI.4 "MEASURED COMPLETE" label reflects aggregated MEASURED-equivalent data, NOT 14/14 independent per-pair audits — the §7.1.2 wall-time table values inherit the same caveat.

### Wf-289C — Cf-258 17 waves refresh (`molmetal/reports/wf289c_c258_refresh/refresh_report.md`)
- **Status**: NO new wave crossed PENDING→DONE on disk since Wf-280D.
- **Wave tally**: n_done=2/17 (C1, C2 — G4 14/14 pairs MEASURED); n_partial=1 (C6 — Ru seed=0 DONE test_auc=0.7716, seed=1 RUNNING epoch 15/50, seed=2 PENDING); n_pre_launch_ready=7 (C10, C11, C13–C17 G6 ablation); n_pending_other=8 (C3–C5 CFM P1.3/P1.4 + C7–C9 G5 50-ep Ir/Os/Rh/Re).
- **Newly unblocked since Wf-286C**: **0** — Wf-285B wire-up + Wf-285C no-enforce-budget + Wf-288D 3 deviation fixes already closed prior blockers.
- **Live G5 wrapper PID 609528** autonomous; ETA C6→DONE ~2hr; then C7–C9 ~3hr; total ~17hr sequential after C6.

### Wf-289D — Cf-265 dashboard refresh (`molmetal/reports/wf289d_dashboard_refresh/refresh_report.md`)
- **Status**: CONSISTENT-RUN — no schema/catalog edits required.
- **Verdict**: Wf-287A m14 anchor (0.9958, n=26,801) was already correctly reflected in dashboard via `claim_m14_cfm_decode_ratio` MEASURED card + `_extract_cfm_m14_cohort` per-metal table; `m14_anchor.json` is metadata pointer to Wf-188 `result.json` (canonical).
- **Counts**: 33 claims, 17 honest-negatives, 6 cohorts — all unchanged.
- **Wf-265 paper_auto_update dry-run**: 0 paper doc edits needed (m14 anchor already canonical in TODO-188 §4.1).

---

## §2 Aggregate tally

| Dimension | Count | Notes |
|---|---:|---|
| **Total work completed (DONE + REFLECTED + CONSISTENT-RUN)** | 4/4 agents | All 4 Wf-289 subworkflows produced concrete outputs |
| **Total failures** | 0/4 agents | All completed; no agent BLOCK |
| **Source-level code fixes** | 3 files | `pt_metal_ligand_exchange.py` (+~210 LOC), `atoms/combinators.py` (+14 LOC), `syntemol_reactions.py` (path + bootstrap) |
| **Tests newly collectable** | 1131 | From 9 cascade-import recoveries (1127) + 4 individual-test contributions |
| **Wave ledger Δ since Wf-280D** | 0 new DONE; +1 PARTIAL (C6); +7 PRE-LAUNCH READY | Cf-258 progress in pre-launch layer, not in DONE layer |
| **Dashboard schema Δ** | 0 | Wf-289D consistent-run |
| **Paper doc edits** | 0 | Wf-265 dry-run produced 11-claim diff_report with 0 deltas |
| **Doc reflections (caveat injection)** | 2 | TODO-188 §4.5.1 + TODO-207 §9.AI.4 (Wf-289B) |

---

## §3 Critical honest-negatives remaining

1. **G4 14 PENDING per-pair audit gap** — Per Wf-289B reflection: only Ru seed=0 of 15 (metal, seed) pairs is independently auditable. The 14-pair MEASURED aggregate in TODO-188 §4.5.1 / §7.1.2 must be cited WITH the Wf-268C caveat. This is the **single highest-priority honest-negative**: paper §4 honest-positive scope depends on it.
2. **Cf-258 15/17 waves still PENDING** — 11.76% DONE-only completion; 52.94% if PARTIAL + PRE-LAUNCH READY are counted. C3–C5 (CFM P1.3/P1.4 ablation flag) and C7–C9 (G5 50-ep Ir/Os/Rh/Re) are BLOCKED at GPU/infrastructure layer.
3. **TODO-181 G5 50-ep sweep incomplete** — Only Ru seed=0 of 15 pairs is MEASURED (test_auc=0.7716); 14/15 PENDING. G5 wrapper autonomous but ETA ~5+ hr to full DONE.
4. **TODO-130 G5 chemprop v3 DrugOOD re-train** — v3 currently 0.4544 BELOW_RANDOM DrugOOD (Wf-286A `gate_1_all_ran=false`, wrapper aborted).
5. **TODO-96 R20-P5 Catherine 30-metal reachability ≥ 0.95** — 6-wk ETA per consumer; not yet shipped.
6. **TODO-98 L12 cross-domain OOD reachability ≥ 0.85** — Q4 2026; defer.
7. **M1+M2 / F5_RDKIT_2D / click-aug CFM integration** — 1.1.8 / 1.2.3 / 1.2.4 pending GPU; downstream lift scope.

---

## §4 Recommended next priorities

| Rank | Action | Wall | Status | Rationale |
|---:|---|---|---|---|
| **1** | **Wf-290-A** Independently audit 14 G4 (metal, seed) per-pair payloads against on-disk log + ckpt + metrics re-verification | ~30 min (if logs are present) | PENDING | Closes Wf-289B critical honest-negative; enables honest §4 paper cite without caveat |
| **2** | **Wf-290-B** Wait for C6 (Ru seed=1 + seed=2) to complete via autonomous wrapper PID 609528; log C6→DONE event | ~2 hr | IN-FLIGHT (seed=1 epoch 15/50) | G5 C6→DONE is the gating event for C7–C9 launch |
| **3** | **Wf-290-C** Launch C7 (Ir s0/s1/s2, G5 50-ep, ~60 min) AFTER C6 seed=2 ends | ~1 hr | PRE-LAUNCH READY | Continues G5 sweep |
| **4** | **Wf-290-D** Launch C8 + C9 (Os+Rh + Re, ~120 min) AFTER C7 ends | ~2 hr | PRE-LAUNCH READY | Closes TODO-181 G5 sweep |
| **5** | **Wf-290-E** Launch C3 (CFM P1.3 smoke, GPU-0 Blackwell, ~30 min) — fresh user OK required | ~0.5 hr | PENDING | Unblocks C4+C5 (P1.4 ablation) |
| **6** | **Wf-290-F** Launch C10–C17 G6 ablation (60 runs, ~7.5 hr, GPU-1) AFTER G5 sweep ends | ~7.5 hr | PRE-LAUNCH READY | Per-innovation attribution; --no-kendall/--no-dann flags wired at Wf-285B |
| **7** | **Wf-290-G** Doc edits (Wf-287E P1–P3): §9.AF verbatim Δ + G3 per-metal ckpt + G4 CHECKPOINT_LOADED_FORWARD_DEFERRED preamble | ~13 min | READY | Orthogonal to GPU; ultracode-suitable |
| **8** | **Wf-290-H** Wf-277B DrugOOD v3 ECFP4 re-eval (~75 min, kanzi CPU) — orthogonal to GPU work | ~1.25 hr | READY | Run on CPU while GPU-0/1 busy |

**Total wall** (sequential): ~14 hr GPU-0/1 + ~1.5 hr CPU parallelizable.

**Parallelization**: Steps 5 + 8 (GPU-0 Blackwell + kanzi CPU) run in parallel with steps 2–4 + 6 (GPU-1 G5/G6 sweep).

**Blocker still open**: Wf-287D §3.2 confirmed NO new CFM ablation flag gap exists; C3 launch requires fresh user OK per `kill_then_adapt` (every wave ≤ 1 hr).

---

## §5 Honest framing

- **Wf-289 batch as a whole**: 4/4 subworkflows shipped; **0** of them crossed a DONE gate that wasn't already DONE at Wf-280D. The batch's value is **textual/audit consolidation** (Wf-289B caveat reflection, Wf-289C refresh ledger, Wf-289D dashboard re-verify) + **infrastructure-side fixes** (Wf-289A import cascade), NOT new measurement.
- **G5 C6 PARTIAL→DONE** is the next MEASURED event on the horizon (~2 hr from Wf-289C snapshot); Wf-289 batch did not produce this.
- **Paper §4 §3.5 honesty ledger** remains intact: 0 violations (Wf-288B); 17 honest-negatives tracked (Wf-289D).
- **Wf-289A's 1131 newly-collectable tests** are the highest-value batch contribution: tests that were *blocked at collection time* are now discoverable, expanding the actual test surface from 7853 → 8984 (+14.4%) without changing any assertion.

---

## §6 File references

- `molmetal/reports/wf289a_broken_imports/fix_report.md`
- `molmetal/reports/wf289c_c258_refresh/refresh_report.md`
- `molmetal/reports/wf289d_dashboard_refresh/refresh_report.md`
- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §4.5.1 (Wf-289B reflection)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AI.4 + §9.Z Z.1 + §9.AB (Wf-289B reflection)
- `molmetal/reports/wf268c_g4_honest_framing/` (Wf-289B source)
- `molmetal/reports/wf280d_c258_waves/audit_report.md` (Wf-289C baseline)
- `molmetal/reports/wf286c_c258_refresh/refresh_report.md` (Wf-289C prior)
- `molmetal/reports/wf265_collect/all_results.json` (Wf-289B aggregated payloads)

---

*Generated by Wf-289E (2026-09-29). Read-only synthesis. 4/4 Wf-289 subworkflows shipped; 0 newly-DONE Cf-258 waves; 1 G5 PARTIAL→DONE event pending (~2 hr); G6 7-wave PRE-LAUNCH READY but wrapper never invoked.*