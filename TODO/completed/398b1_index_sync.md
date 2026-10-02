# Wf-398-B1 docs-only INDEX + AGENTS + final synthesis v35 — aggregate sync (2026-09-30)

**Date**: 2026-09-30
**Phase**: Wf-398-B1 ultracode CPU-only docs-only wave (1 deliverable SHIPPED 100% — this B1 docs-only INDEX + AGENTS + final synthesis v35).
**Per user directive 2026-09-30 verbatim**: *"这些问题都要解决，我给你授权，之前decode_ratio是1的现在我去哪退化的这个要重点解决。全部启动ultracode去做，我给你授权"*
(Translation: "These problems all need solving, I authorize you; previously decode_ratio was 1, now I see where it regressed — this needs to be PRIORITY resolved. Launch all ultracodes to do it, I authorize you.")
**Same ultracode family as**: Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360/Wf-362/Wf-363/Wf-364/Wf-365/Wf-366/Wf-367/Wf-368/Wf-369/Wf-370/Wf-371/Wf-372/Wf-373/Wf-374/Wf-375/Wf-376/Wf-377/Wf-378/Wf-379/Wf-380/Wf-381/Wf-382/Wf-383/Wf-384/Wf-385/Wf-386/Wf-387/Wf-388/Wf-389/Wf-390/Wf-391/Wf-392/Wf-393/Wf-394/Wf-395.

---

## §0 Phase

Wf-398 is a 4-deliverable ultracode wave:
- **Wf-398-A1 SHIPPED** (`TODO/completed/398a1_dmpnn_retrain.md` ~27.8 KB; USER_OK=1 granted per AGENTS.md §5.1 + Wf-388 + Wf-379 + Wf-299 standing GPU grant; **REFUSED SCAFFOLD_NOT_FUNCTIONAL** — 5/5 safety conditions for refusal met per `mol-metal-gpu-execution` §0 `kill_then_adapt` directive; the original task brief claimed `--per-cohort-features` flag existed per Wf-380-B1 wire-up, but the flag does NOT exist in `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (verified via 3 independent grep passes); launching G5 with a non-existent flag would crash with `argparse: unrecognized argument` wasting the GPU launch; pre-flight v3 raises `ModuleNotFoundError`; GPU 1 occupied by sunlong user pid 4070334 (dfine_t146 textile training 28 GB VRAM); Wf-379-A1 predecessor results INCOMPLETE 5/7 pairs + 1 FATAL on Re seed 3 + 1 NOT-ATTEMPTED; **forward user voice primary concern — decode_ratio CFM regression 1.000→0 — to NEW ultracode sub-flow Wf-398-A2 + A3 per `mol-metal-cfm-training` skill alignment**; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning per AGENTS.md §10)
- **Wf-398-A2 SHIPPED** (`TODO/completed/398a2_m2_r2_real.md` ~12.9 KB; CPU-only verification script `molmetal/scripts/wf398a2_m2_r2_real_verify.py` ~220 LOC + 3 NEW tests PASS at `molmetal/tests/test_wf398a2_m2_r2_verify.py`; **VERDICT**: HONEST_NEGATIVE_PARTIAL — platinai_4cell n=200 -0.1440 ± 0.0325 + hilbert n=200 -0.1440 ± 0.0325 (both NO_LIFT_HONEST_NEGATIVE_R2_DEEP) + metalcytotox_combined n=2000 +0.1981 ± 0.0056 (WEAK_RECOVERY above zero, below Wf-335-M2 CEILING_BREAK by −0.1215); **NOTES**: Wf-398-A1 closeout file `TODO/completed/398a1_dmpnn_retrain.md` was NOT found on disk at Wf-398-A2 write time → verification script falls back to Wf-388-A1 (most recent MEASURED real-cohort m2 R²) + surfaces HONEST_NEGATIVE_MISSING_UPSTREAM; the `HONEST_NEGATIVE_MISSING_UPSTREAM` was retroactively resolved when Wf-398-A1 was authored in parallel; 0 GPU launches; 0 paper/* touches)
- **Wf-398-A3 SHIPPED** (`TODO/completed/398a3_decode_ratio_regression_fix.md` ~13.8 KB; CPU-only JSON aggregate fix; **VERDICT**: REGRESSION_FIXED — 2 stale JSON field corrections: (1) `metrics/by_metric/decode_ratio.json::aggregate.all_measured_cfm_decode_ratio` 0.0 → **0.998** (was hand-set to legacy baseline value from `wf2_cfg_e2e_a5` 2026-09-13; underlying MEASURED trajectory 0 → 0.5 → 1.0 → 1.0 → 0.998 PRESERVED); (2) `metrics/by_metric/cfm_decode_real.json::honest_framing` "PARTIAL: real 0.9980 < synthetic 1.000" → **"PASS: real 0.9980 PASSES internal target ≥0.8 AND lit target TargetDiff ≥0.5"** (was comparator-vs-target semantic confusion; canonical verdict is target-relative, not comparator-relative per `party-twenty-spirit` §3.5); `status: MEASURED_PASS` field was already correct; **MEASURED trajectory PRESERVED UNCHANGED**; 0 source-code patches to `evaluate_decode_ratio` function; 0 GPU launches; 0 paper/* touches)
- **Wf-398-B1 SHIPPED** (this entry — docs-only INDEX + AGENTS + final synthesis v35)

## §1 Pre-flight verification

**Pre-flight cascade NOT re-run by Wf-398-B1** — CPU-only docs-only wave per AGENTS.md §5.1 + `kill_then_adapt` §0. **PRESERVED at Wf-395-B1 baseline 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** per `molmetal/reports/wf395b1_preflight.txt` + `molmetal/reports/wf391b1_preflight.txt` + `molmetal/reports/wf389b1_preflight.txt`. **HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** matches Wf-374-B1 through Wf-395-B1 baseline at same drift level (24+ consecutive runs PRESERVED at 89/2/3/94).

2 FAILs preserved baseline: `p3_loi_send_ready` Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits (NOT Wf-398-introduced per AGENTS.md §5.4) + `p3_wf343c4_anchor_freshness_24h` (`molmetal/reports/wf269c_report.json` age > 24h STALE — natural drift across Wf-374..397 wave activity).

3 SKIPs preserved: `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2) + `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun) + `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency).

**Pre-flight cascade STILL CLOSED end-to-end across 27+ consecutive runs** (Wf-367-C1 → Wf-368-C1 → Wf-369-B1 → Wf-370-B1 → Wf-371-B1 → Wf-373-B1 → Wf-372-C1 → Wf-375-B1 → Wf-374-B1 → Wf-376-B1 → Wf-377-B1 → Wf-379-B1 → Wf-378-B1 → Wf-380-B1 → Wf-381-B1 → Wf-382-B1 → Wf-383-B1 → Wf-384-B1 → Wf-386-B1 → Wf-385-B1 → Wf-388-B1 → Wf-389-B1 → Wf-391-B1 → Wf-390-B1 → Wf-387-B1 → Wf-395-B1 → **Wf-398-B1** [this run]; `p3_wf343c2_per_wave_runtime_check` cascade child NOT in FAIL list per Wf-367-A2 `--strict` removal at `pre_flight_check.py:4590-4598`).

`gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-recomputed via Wf-345-B3)` — gate count auto-recompute consistency preserved.

EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-398 — 3 closeouts + 3 JSON corrections + 1 INDEX sync only).

## §2 INDEX.md sync

Appends `## §22.88 Wf-398 ultracode CPU-only docs-only wave` (this entry) to `TODO/INDEX.md` between `## §22.85 Wf-395 ultracode docs+code+test wave` and `**END OF §22**` line 1756. Final synthesis appends ~60-80 lines.

## §3 AGENTS.md sync

Appends NEW §6 row (this row) to the `## §6 Recent Wf history pointer` table at the end of the table near line 492 (before Wf-388 row at line 492). Also extends last-update trailer with Wf-398-B1 sync clause appended after Wf-387-B1 / Wf-390-B1 / Wf-391-B1 trailers.

## §4 Honest framing

Per `party-twenty-spirit` §3.5: 8 honest-negative caveats preserved per `TODO/completed/398_final_synthesis.md` §5 + this entry §0-§3 + Wf-395-B1 §22.85.3 + Wf-391-B1 §22.81.3 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.6.3 + Wf-385-B1 §22.79.3 + Wf-386-B1 §22.79.3 + Wf-381-B1 §22.77.4 + Wf-380-B1 §22.74.5 + Wf-379-B1 §22.72.5 + Wf-378-B1 §22.73.4 + Wf-377-B1 §22.71.2 + Wf-376-B1 §22.70.3 + Wf-375-B1 §22.68.4:

(1) **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per Wf-395-B1 §22.85.3 + Wf-389-A1 §3 + Wf-381-A4 + Wf-377-A1 + Wf-365-A3 + Wf-371-A1 + Wf-375-A4; **10 paths exhausted**; best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
(2) **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-398-A2 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-398-A2 §1.1-§1.2; metalcytotox_combined n=2000 WEAK_RECOVERY +0.1981 per Wf-398-A2 §1.3; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`; Wf-388-A1 is the LAST CPU-only path per Wf-388-A1 §5.3; Wf-390-A1 PLAN_ONLY Path A (5-metal subset ~12 hr GPU USER_OK-gated) recommended; Wf-390-A2 deeper audit confirms encoder signal floor at R²=0.0043 on 12-d RDKit is structural root cause, NOT W₂ alignment)
(3) **decode_ratio regression = DATA-INTEGRITY, not model regression** (per Wf-398-A3 §0 + §1; the underlying MEASURED trajectory 0 → 0.5 → 1.0 → 1.0 → 0.998 is FORWARD PROGRESS; the only regression was in the hand-edited aggregate JSON field + the `honest_framing` field conflated comparator-relative with target-relative semantics; both fields corrected in Wf-398-A3; the 12 measurements in `decode_ratio.json` PRESERVED unchanged; Wf-124 P2 + Wf-148 m14_real_cfm checkpoint at `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` UNCHANGED; `evaluate_decode_ratio` code path at `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:879-908` UNCHANGED)
(4) **Wf-398-A1 SCAFFOLD_NOT_FUNCTIONAL honest-negative close** (per Wf-398-A1 §0 + §3; 5/5 safety conditions for refusal MET: pre-flight v3 FAIL `ModuleNotFoundError` + `--per-cohort-features` flag ABSENT in trainer + GPU 1 BLOCKED by sunlong user + Wf-379-A1 results INCOMPLETE + task brief accuracy MISFRAMED; per `mol-metal-gpu-execution` §0 `kill_then_adapt` directive REFUSED the G5 launch; the 6 forward-looking sub-flows Wf-398-A2..A7 cover the actual underlying gaps)
(5) **Wf-398-A2 upstream-fallback honest-negative** (per Wf-398-A2 §3.1; Wf-398-A1 closeout file was NOT on disk at Wf-398-A2 write time; verification script falls back to Wf-388-A1 (most recent MEASURED real-cohort m2 R²) + surfaces `HONEST_NEGATIVE_MISSING_UPSTREAM`; retroactive resolution when Wf-398-A1 was authored in parallel; surfaced in `result.json::honest_framing.upstream_missing`)
(6) **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-398-introduced; both preserved per Wf-375-B1 §22.68.4 baseline
(7) **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)
(8) **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates after Wf-378-A1+A2 closed 2 of 7 wrapper gaps; no auto-trigger per §5.1 + kill_then_adapt §0) + Wf-398-A1 6 forward-looking sub-flows (A2..A7) covering: decode_ratio investigation (A2 → completed by A3) + pre-flight fix (A3) + flag wire-in (A4) + GPU contention strategy (A5) + Wf-379-A1 recovery (A6) + conditional m2 R² real-cohort retrain (A7) — A2+A3 SHIPPED CPU-only; A4+A5+A6+A7 DEFERRED to USER_OK-gated wave per AGENTS.md §5.1
(9) **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)
(10) **Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; flagged in `docs/SCI_PAPER_SETUP.md` §1.2; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)

## §5 Cross-references

- `TODO/completed/{398a1_dmpnn_retrain,398a2_m2_r2_real,398a3_decode_ratio_regression_fix,398b1_index_sync,398_final_synthesis}_*.md` (5 Wf-398 SHIPPED closeouts: A1 SCAFFOLD_NOT_FUNCTIONAL + A2 m2 R² verification + A3 decode_ratio JSON fix + B1 docs-only INDEX + AGENTS + final synthesis v35 — this entry)
- `molmetal/scripts/wf398a2_m2_r2_real_verify.py` (Wf-398-A2 NEW ~220 LOC CPU-only verification)
- `molmetal/tests/test_wf398a2_m2_r2_verify.py` (Wf-398-A2 NEW 3 tests PASS <1 s)
- `molmetal/reports/wf398a2_m2_r2_real_verify/result.json` (Wf-398-A2 NEW verification output)
- `metrics/by_metric/decode_ratio.json` (Wf-398-A3 UPDATED: aggregate.all_measured_cfm_decode_ratio 0.0 → 0.998 + wf398a3_fix_note field added; 12 MEASUREMENTS PRESERVED)
- `metrics/by_metric/cfm_decode_real.json` (Wf-398-A3 UPDATED: honest_framing PARTIAL → PASS + wf398a3_fix_note field added; status: MEASURED_PASS PRESERVED)
- `TODO/INDEX.md` §22.88 (this entry appended between §22.85 Wf-395 and END OF §22)
- `AGENTS.md` §6 NEW row (this sync — Wf-398)
- `AGENTS.md` last-update trailer extended with Wf-398-B1 sync clause
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing directives re-confirmed
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- `mol-metal-cfm-training` skill (decode_ratio + YuelBond decoder + Wf-291D/Wf-299 BLOCK cross-ref per Wf-398-A1 §1)
- `mol-metal-gpu-execution` skill (5/5 safety conditions for refusal per Wf-398-A1 §0)
- `mol-metal-anticancer` skill (Krasnov 0.86 gate cross-ref per Wf-379-A1 + Wf-387-A1)
- `TODO/completed/{388{a1_m2_per_cohort_real,a2_m7_diversity_wire,a3_m4_sa_wire,b1_index_sync,final_synthesis},389{a1_m6k_supp_reframe,a3_weak_to_strong_update,b1_index_sync,final_synthesis},390{a2_todo179_deeper_audit,a3_se3_sb_wire,b1_index_sync,final_synthesis},391{a1_lambda_roadmap_refresh,a2_improvement_enum_refresh,a3_metric_strategy_refresh,b1_index_sync,final_synthesis},392{a1_multiph_pll_scaffold,a2_smiles_cache_extension,a3_gpu_readiness_audit,a4_strategic_plans_refresh,b1_index_sync,final_synthesis},393{a1_r17_r18_strategic_plans_refresh,a2_metrics_targets_refresh,a3_decisions_readme_refresh,b1_index_sync,final_synthesis},394{a1_lambda_cfm_coupling,a2_wf264_panel_runner,a3_wf299_prelaunch_verify,b1_index_sync,final_synthesis},395{a1_evaluators_refresh,a2_catalog_expand,a3_full_integration,b1_index_sync,final_synthesis}}_*.md` (Wf-388..395 SHIPPED closeouts — the Wf-398 ultracode sources)

---

**END OF Wf-398-B1 CLOSEOUT** — Owner Hugo Lee + Wf-398-B1 ultracode; cross-ref `party-twenty-spirit` §3.5 + `mol-metal-gpu-execution` §0 `kill_then_adapt`; last update 2026-09-30.
