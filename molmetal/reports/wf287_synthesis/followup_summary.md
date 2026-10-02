# Wf-287E — Synthesis: Wf-287 batch outcomes + follow-up priorities

**Date**: 2026-09-29
**Scope**: aggregate Wf-287A (Wf-188 m14 anchor JSON), Wf-287B (§3.1.7/§3.1.14 cross-ref), Wf-287C (Cf-266 LOMO gate audit), Wf-287D (Cf-263B §9 6-gaps verify).
**Method**: Read each upstream report verbatim + cross-check gate verdicts.

---

## §1 Per-agent summary

| Agent | Output | Verdict | Key result |
|---|---|---|---|
| **Wf-287A** | `molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json` | **MEASURED_PASS** | m14 LEAD anchor = **0.9958** on n=26,801 (5-metal Ru/Ir/Os/Rh/Re, real CFM cuda); SUPERSEDES prior 0.9645 (PlatinAI pt_subset CPU). Per-metal {Ru:0.9948, Ir:0.9974, Os:1.000, Rh:1.000, Re:1.000}; 0.9645 carry-forward labelled `silent_regression_methodology_change_not_disclosed`. |
| **Wf-287B** | `wf287b_section31_verify/verify_report.md` | **CONSISTENT** | §3.1.14 gate (184 tests) verified post-Wf-282B; §3.1.7 ablation flags orthogonal (no cross-ref); 3 stale "145 tests" mentions fixed (lines 737, 750, 935). 0 structural changes. Re-verify pending via `pre_flight_check.py --gate 3.1.14`. |
| **Wf-287C** | `wf287c_cf266_gate_audit/audit_report.md` | **PASS 8/8** | All Wf-266 G1-G8 gates still PASS post Wf-282B / Wf-283B / Wf-284C; 4/4 Wf-266 tests PASS in 2.36s; 0 stale lever/dispatch/log references. G5 backward-compat 14 tests env-qualified (`pocket/.eval_envs/admet311/bin/python`). |
| **Wf-287D** | `wf287d_cf263b_gaps_verify/verify_report.md` | **4/6 fixed, 2/6 partial, 0/6 unfixed** | G1 CFM SA/QED/Vina ✅, G2 m7 Tanimoto gap ✅ (with caveat), G3 per-metal coverage **partial**, G4 CHECKPOINT_LOADED_FORWARD_DEFERRED **partial**, G5 m6_kappa NEGATIVE_WEAK ✅ (single-line), G6 5 CFM TODO forward-looking ✅ (§9.Z). |

**Totals**: 4/4 agents shipped successfully (0 agents failed). 4 Wf-287D gaps fixed (G1/G2/G5/G6), 2 partial (G3/G4), 0 unfixed.

---

## §2 Critical honest-negatives remaining

1. **Wf-287D G3 partial** — per-metal ckpt coverage (Ru 4/4, Ir 3/4, Os 1/4 MEASURED; 4/12 DEFERRED_NO_CHECKPOINT per Wf-253d) distributed across §9.4 + §9.AE + §9.AI.1 instead of single dedicated §9.N row. Aggregate `BELOW_RANDOM 0.4544` LEAD-disqualification IS preserved.
2. **Wf-287D G4 partial** — `CHECKPOINT_LOADED_FORWARD_DEFERRED` invariant has no dedicated §9.N entry; closest analog is §9.AE Wf-268B DrugOOD re-verify + backbone mismatch.
3. **Wf-287D G5 optional** — Re r=-0.1843 NEGATIVE per-metal cell not cited explicitly in §9.6.1.1 row (compact catalog form acceptable).
4. **Wf-287B re-verify pending** — `pre_flight_check.py --gate 3.1.14` actual pytest "184 passed" wall run not yet executed.
5. **Wf-287C G5 env-qualified** — 14 backward-compat chemprop tests not re-run in the original `pocket/.eval_envs/admet311/bin/python` evidence env this session.
6. **Wf-269C precondition gate** — m14 LEAD 0.9958 anchor conditional on YuelBond decoder active (without it decode_ratio → 0.000 per Wf-269C MEASURED); party-twenty-spirit §3.5 honesty preserved via `honest_framing` field in `m14_anchor.json`.

---

## §3 Next follow-up priorities (ranked)

| Pri | Task | Source | Effort | Parallel-safe |
|---|---|---|---|---|
| **P1** | Wf-277A §9.AF verbatim Δ alignment (`per_lever.json` no_w Δ=-0.013, no_cb Δ=-0.008, NOT Δ=0.000) | Wf-287D §4 + §9.AI.6 PENDING | ~5 min doc edit | **YES — ultracode-suitable (explicit marker in Wf-287D §4)** |
| **P2** | G3 per-metal ckpt coverage §9.AI.7-bis row | Wf-287D G3 partial | ~5 min doc edit | YES — pure text insert |
| **P3** | G4 CHECKPOINT_LOADED_FORWARD_DEFERRED §9.AE.0-bis or §7.2.7 cross-ref preamble | Wf-287D G4 partial | ~3 min doc edit | YES — pure text insert |
| **P4** | pre_flight_check.py --gate 3.1.14 wall-run + 184-test pytest verify | Wf-287B re-verify pending | ~60s wall | NO — depends on actual chemprop env |
| **P5** | G5 per-metal Re r=-0.1843 explicit cell in §9.6.1.1 | Wf-287D G5 optional | ~2 min doc edit | YES — pure text insert |
| **P6** | Wf-277B DrugOOD v3 ECFP4 re-eval (target ≥ 0.7137 on `lbap_refined_ic50_assay`) | Wf-287D + §9.AI.7 PENDING | ~75 min GPU wall | NO — GPU-only, kanzi contention |
| **P7** | Backward-compat 14-test re-run in `pocket/.eval_envs/admet311/bin/python` env | Wf-287C G5 env-qualified | ~3s wall | NO — needs admet311 venv |

**Repair-task parallel-launch candidates for ultracode** (from §2 critical-negatives + §3 P1/P2/P3/P5): the 4 doc edits are isolated single-file text edits against `TODO/completed/207_unified_pre_flight_checklist_v2.md` (P1 §9.AF Δ, P2 §9.AI.7-bis, P3 §9.AE.0-bis, P5 §9.6.1.1 Re cell). They are orthogonal to GPU retrain (P6) and pytest wall-runs (P4/P7), so they can ship as a 4-task ultracode batch while the GPU Wf-277B is queued behind the kanzi contention gate.

---

## §4 Honest framing (party-twenty-spirit §3.5)

- All 4 Wf-287 agents shipped without engine-level failure; the 2 Wf-287D "partial" gaps (G3/G4) are honest-positive-status — the underlying negative IS catalogued in §9, just not as a single dedicated row.
- m14 LEAD anchor (Wf-287A 0.9958) is the paper-ship value; prior 0.9645 carry is properly labelled as silent-regression (not retracted, not used in §9.6.3 cite).
- Wf-277A doc Δ mismatch is the highest-leverage remaining fix — small (5 min) and explicitly ultracode-suitable per Wf-287D §5.
- No new §9.N entries are required for the cfm_decode_real 0.9645 silent regression (on-disk `regression_disclosure` block mitigates per Wf-271B).
- No new §9.N entries are required for the Wf-284B false-positive (Wf-260 prior fix supersedes per Wf-287D §3.2).

---

**End of Wf-287E followup_summary. Aggregated n_agents=4, n_shipped=4, n_failed=0, n_partial_gaps=2, n_remaining_doc_edits=4 (ultracode-suitable), n_remaining_gpu_walls=1 (Wf-277B), n_remaining_pytest_walls=2 (3.1.14 + admet311 env).**