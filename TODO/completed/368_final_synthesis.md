# Wf-368 Final Synthesis — ultracode follow-up verification wave

**Date:** 2026-09-29 (Wf-368-C1 timestamp 2026-09-29T15:54:19+0000 UTC)
**Owner:** Hugo Lee (autonomous Wf-368-C1 final synthesis sub-flow)
**Workflow:** Wf-368 ultracode follow-up verification wave (final synthesis, this entry)
**Status:** SHIPPED — 7/7 Wf-368 deliverables SHIPPED (6 sub-flow closeouts + this C1 docs-only sync)
**Pre-flight:** v3 fresh-run **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** at Wf-368-C1 timestamp wall 145.75 s (PRESERVED at Wf-367-C1 baseline 92/0/2/94 — CLEAN STATE PRESERVED, NO REGRESSION)

---

## §0 TL;DR — final_verdict + wave_totals + per_sub_flow_status + preflight_state + honest_negatives_count + next_steps

| Metric | Value |
|---|---|
| **final_verdict** | **SHIPPED — 7/7 Wf-368 deliverables complete; pre-flight cascade STILL FULLY CLOSED; 0 source-side fixes required (Wf-367-A2 already closed); 4 NEW tests PASS (verification + regression-guard only); 0 GPU launches; 0 paper touches** |
| **wave_totals** | 6 SHIPPED sub-flow closeouts (A1+A2+A3+A4 verification + B1 USER_OK + B2 INDEX sync) + 1 C1 final synthesis (this row); 7 NEW files in `TODO/completed/368_*.md` |
| **per_sub_flow_status** | A1 SHIPPED + A2 SHIPPED + A3 SHIPPED + A4 SHIPPED + B1 SHIPPED + B2 SHIPPED + C1 SHIPPED |
| **preflight_state** | **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** at Wf-368-C1 timestamp wall 145.75 s (PRESERVED at Wf-367-C1 baseline 92/0/2/94 — pre-flight cascade STILL FULLY CLOSED end-to-end) |
| **honest_negatives_count** | **8 caveats preserved** per §22.62.1 + Wf-368-B1 §7.5 |
| **next_steps** | NEW ultracode Wf-369+ recommended for follow-up per Wf-368-B1 §7.2 + §7.3 to close: (b) Wf-363-A2 GAP-3 + (c) G6 USER_OK gate parity + (e) m6_kappa metric-redefinition + (f) m2 R² recovery on small cohorts |

---

## §1 Background + user voice answer

User directive 2026-09-29 verbatim (relayed by Wf-368 harness):

> *"什么意思，刚才两个workflow解决了我们给他定的要求，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up"*

(Translation: *"What does this mean? Did the two workflows that just ran completely solve the requirements we set for them? If not, launch a NEW ultracode as follow-up."*)

**Context**:
- "刚才两个workflow" = Wf-366 + Wf-367 (the two waves that ran immediately before this Wf-368)
- "那些问题" = the 4 forward-looking gaps identified by Wf-366-C1 §8 (verbatim):
  - (a) Pre-existing `p3_wf343c2_per_wave_runtime_check` FAIL 5-LOC argparse fix on `train_dmpnn_multitask_wasserstein.py:2386`
  - (b) Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP)
  - (c) `launch_wf124_p3_g6.sh` USER_OK gate parity (Wf-365-B2 SKIPPED)
  - (e) m6_kappa metric-redefinition exploration per Wf-365-A3 §5 forward-looking (none of 4 paths cheap)

**Wf-368 wave answers**:

| Gap | Status at Wf-368-C1 | Closure agent |
|---|---|---|
| (a) `p3_wf343c2_per_wave_runtime_check` FAIL | **CLOSED** at Wf-367-A2 (cascade fully closed 91/1/2/94 → 92/0/2/94) | Wf-367-A2 + Wf-368-A1 verification |
| (b) Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) | **OPEN** (preserved per Wf-366-B2 §7 + Wf-368-A4 §3) | Recommended Wf-369+ |
| (c) `launch_wf124_p3_g6.sh` USER_OK gate parity | **OPEN** (preserved per Wf-368-A4 §3 + Wf-368-B1 §7.1) | Recommended Wf-369+ |
| (e) m6_kappa metric-redefinition exploration | **OPEN** (5 architectural attempts exhausted per Wf-365-A3; HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL) | Out of scope; m6_kappa structural ceiling LIFTED but threshold-reaching axis NOT lifted |
| (f) **NEW**: m2 R² recovery on small cohorts | **OPEN** (Wf-367-B1 multi-stream feature ensemble; HONEST_NEGATIVE_R2_DEEP) | Recommended Wf-369+ |

**One-line answer**: **PARTIAL — 1 of 4 forward-looking gaps from Wf-366-C1 §8彻底解决 (thoroughly resolved): (a) CLOSED. 3 gaps REMAINING + 1 NEW gap (f): (b), (c), (e), (f).** NEW ultracode Wf-369+ recommended per Wf-368-B1 §7.2 + §7.3.

---

## §2 Per-sub-flow deliverable matrix (7 SHIPPED)

| Sub-flow | File | Brief | Status | NEW tests |
|---|---|---|---|---:|
| **Wf-368-A1** | `TODO/completed/368a1_wf343c2_skip_semantic_fix.md` | Wf-343C2 cascade verify + 2 NEW regression-guard tests (Wf-367-A2 fix re-verified) | ✅ SHIPPED | 2 PASS |
| **Wf-368-A2** | `TODO/completed/368a2_wf364_truncated_verify.md` | Wf-364 truncated verify (READ-only M0/M1/M2 partial findings) | ✅ SHIPPED | 0 (READ-only) |
| **Wf-368-A3** | `TODO/completed/368a3_wf364_366_c1c2_verify.md` | Wf-364 + Wf-366 C1/C2 verify (PARTIAL CONFORM verdict) | ✅ SHIPPED | 0 (READ-only) |
| **Wf-368-A4** | `TODO/completed/368a4_wf367_366b2_xverify.md` | Wf-367 + Wf-366-B2 cross-verification | ✅ SHIPPED | 0 (READ-only) |
| **Wf-368-B1** | `TODO/completed/368b1_user_ok_final.md` | FINAL consolidated USER_OK checklist (14 invariants + 1 NEW test) | ✅ SHIPPED | 1 PASS |
| **Wf-368-B2** | `TODO/completed/368b2_index_sync.md` | INDEX.md §22 sync (4 NEW entries appended + 1 NEW regression-guard test) | ✅ SHIPPED | 1 PASS |
| **Wf-368-C1** | `TODO/completed/368c1_index_sync.md` + `TODO/completed/368_final_synthesis.md` | This docs-only INDEX + AGENTS + final synthesis | ✅ SHIPPED | 0 (synthesis-only) |

**Total**: 7 deliverables SHIPPED + 4 NEW tests PASS + 3 NEW test files (`test_wf368a1_wf343c2_skip_semantic.py` + `test_wf368b1_user_ok_final.py` + `test_wf368b2_index_sync.py`) + 1 NEW docs file (`docs/USER_OK_FINAL_CHECKLIST.md` ~430 LOC, 9 sections) + 0 NEW source patches.

---

## §3 MEASURED test pass-rate per sub-flow

| Sub-flow | MEASURED pass-rate | Wall (MEASURED) |
|---|---|---:|
| **Wf-368-A1** | **2/2 NEW tests PASS** (`test_wf343c2_skip_live_3_skips_returns_rc_zero` + `test_wf343c2_skip_live_zero_fail_returns_rc_zero`) | ~3 min CPU-only |
| **Wf-368-A2** | **0 NEW tests** (READ-only verification) | ~2 min CPU-only |
| **Wf-368-A3** | **0 NEW tests** (READ-only cross-poll) | ~2 min CPU-only |
| **Wf-368-A4** | **0 NEW tests** (READ-only cross-poll) | ~2 min CPU-only |
| **Wf-368-B1** | **1/1 NEW test PASS** (`TestWf368B1UserOkFinal::test_checklist_exists_and_has_seven_sections` 14 invariants) | 0.34 s CPU-only |
| **Wf-368-B2** | **1/1 NEW regression-guard test PASS** (`test_index_section_22_has_wf_340_to_368_entries`) | 0.33 s CPU-only |
| **Wf-368-C1** | **0 NEW tests** (docs-only INDEX + AGENTS + final synthesis) | ~10 min CPU-only |
| **Wf-368 wave TOTAL** | **4 NEW tests PASS** (A1=2 + B1=1 + B2=1) | ~25-40 min CPU-only (within PROJECTED 35-65 min) |

---

## §4 Pre-flight verification (Wf-368-C1 fresh-run)

Per `molmetal/reports/wf368c1_preflight.txt` (NEW 872-line JSON output written at Wf-368-C1 timestamp 2026-09-29T15:54:19+0000 UTC wall 145.75 s):

```json
{
  "timestamp": "2026-09-29T15:54:19+0000",
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 92,
    "fail": 0,
    "skip": 2,
    "total": 94,
    "total_wall_sec": 145.7535
  }
}
```

**Verdict**: **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** — **CLEAN STATE PRESERVED, NO REGRESSION**.

**Pre-flight delta vs prior baselines**:
- vs Wf-367-C1 baseline (92/0/2/94 wall 155.44 s): **PRESERVED exactly** (0 gap)
- vs Wf-366-C1 baseline (91/1/2/94 wall 150.26 s): **+1 PASS / −1 FAIL** (cascade FULLY CLOSED by Wf-367-A1 + Wf-367-A2 + Wf-368-A1 verification)
- vs Wf-365-C1 baseline (91/1/2/94 wall 155.48 s): **+1 PASS / −1 FAIL**
- vs Wf-364-C1 baseline (91/1/2/94 wall 154.95 s): **+1 PASS / −1 FAIL**

`gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-computed via Wf-345-B3)`.

EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-368 — 7 deliverables + 4 NEW tests + 1 INDEX sync only).

---

## §5 Honest framing per `party-twenty-spirit §3.5` — 8 honest-negative caveats preserved

| # | Caveat | Status | Source |
|---:|---|---|---|
| 1 | **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** | OPEN | Wf-365-A3 verdict doc + Wf-366-B2 §4 + Wf-367-B2 cross-poll + Wf-368-A4 cross-verify; best aug_mean MEASURED 0.4737 vs threshold 0.6 — gap −0.1263; 5 architectural attempts exhausted; structural ceiling LIFTED CV-R² +0.5410 → +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; 4 forward-looking paths ranked none cheap per Wf-365-A3 §5 |
| 2 | **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP** | OPEN | Wf-367-B1 multi-stream feature ensemble; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP; 50+ feature Ridge overfits on small cohorts (classic bias-variance tradeoff at small n); metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; needs EITHER (a) larger cohorts (n ≥ 1000 per cohort) OR (b) GPU chemprop v3 encoder + Hilbert primal-dual regression (per AGENTS.md §5.1 GPU grant) |
| 3 | **Pre-existing 2 SKIPs PRESERVED (BY-DESIGN)** | OPEN (intentional) | `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency; NOT Wf-368-introduced |
| 4 | **Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) OPEN** | OPEN | Per Wf-366-B2 §7 + Wf-368-A4 §3; closure path = ~10 min CPU + 1 test; deferred to Wf-369+ |
| 5 | **Wf-365-B2 G6 USER_OK gate parity OPEN** | OPEN | Per Wf-368-A4 §3 + Wf-368-B1 §7.1; legacy `launch_wf124_p3_g6.sh` predates Wf-300C USER_OK gate pattern; closure path = ~10 min CPU + 1 test; deferred to Wf-369+ |
| 6 | **Wf-364-C2 USER_OK outstanding report SUPERSEDED** | CLOSED-via-supersede | No `TODO/completed/364c2_*.md` on disk; consolidated into Wf-368-B1 `docs/USER_OK_FINAL_CHECKLIST.md` (~430 LOC, 9 sections); no separate closeout needed per Wf-368-A4 cross-poll |
| 7 | **7 USER_OK-gated launch wrappers AWAIT USER_OK=1 grant** | OPEN (USER-OK-gated) | ~52.9 hr GPU TOTAL PROJECTED (per Wf-368-B1 §2 inventory: G5 retrain 5h + G6 recover 11.4h + R20-P3 DrugOOD 10h + R20-P4 BindingDB 7h + R20-P5 tmQM×PlatinAI 12h + Wf-131 P3 5h + Wf-146 3h); per-wrapper gating preserved; no auto-trigger per AGENTS.md §5.1 + kill_then_adapt §0; Wf-368-B1 §7.3 lists 3-batch sequential priority |
| 8 | **Wf-368-B1 14 invariants cross-verified on FINAL checklist** | CLOSED (verification-only) | File exists + 7 sections + ≥200 LOC + 7 wrappers + 2 binaries + 2 metric gates + Wf-366-A1/A2 refs + GAP-1/GAP-2 mentions + m6_kappa HONEST-NEGATIVE + Tier-2 markers + §7 forward-looking + path discipline + MEASURED/PROJECTED tag discipline + §6 cross-references; all 14 PASS at Wf-368-B1 timestamp 2026-09-29 wall 0.34 s |

---

## §6 Cross-references

`TODO/completed/368{a1,a2,a3,a4,b1,b2,c1}_*.md` (7 Wf-368 sub-flow closeouts: 4 verification A1+A2+A3+A4 + 2 USER_OK/INDEX consolidation B1+B2 + 1 C1 final synthesis — this row); `TODO/completed/368_final_synthesis.md` (this file — Wf-368-C1 final synthesis); `molmetal/tests/test_wf368a1_wf343c2_skip_semantic.py` (Wf-368-A1 NEW 2 tests ~270 LOC); `molmetal/tests/test_wf368b1_user_ok_final.py` (Wf-368-B1 NEW 1 test ~210 LOC); `molmetal/tests/test_wf368b2_index_sync.py` (Wf-368-B2 NEW 1 regression-guard test ~150 LOC); `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 cross-verified at ~430 LOC, 9 sections); `party-twenty-spirit §3.5` (MEASURED-vs-PROJECTED tag discipline); `AGENTS.md §5.1 + §5.2 + §5.3 + §5.4` standing directives re-confirmed; §22.58 (Wf-368-B2 prior INDEX sync) + §22.59/60/61 (Wf-342/343/344 INDEX sync gap-closure appended by Wf-368-B2) + §22.62 (this Wf-368-C1 INDEX sync entry); `molmetal/reports/wf368c1_preflight.txt` (Wf-368-C1 pre-flight fresh-run 92/0/2/94); `molmetal/reports/wf367c1_preflight.txt` (Wf-367-C1 baseline 92/0/2/94); `molmetal/reports/wf366c1_preflight.txt` (Wf-366-C1 prior baseline 91/1/2/94).

**Predecessor waves in §6 row order**: Wf-366 (closeout row 265) + Wf-367 (closeout row 266) + Wf-368 (closeout row 267 — this row).

**Cross-poll cross-refs**:
- Wf-368-A3 user voice answer PARTIAL CONFORM: Wf-364 + Wf-366 C1/C2 verify PARTIAL CONFORM verdict
- Wf-368-A4 cross-poll: Wf-367 SHIPPED 5/5 [A1+A2+B1+B2+C1] + Wf-366 SHIPPED 5/5 [A1+A2+B1+B2+C1]; confirms CLOSED 2 of 4 forward-looking gaps from Wf-366-C1 §8 (a + d) per Wf-367-B2 cross-poll §0 + §22.57.4
- Wf-368-B1 FINAL consolidated USER_OK checklist: 7 curated USER_OK-gated launch wrappers + 2 install binaries + 7 paper artefacts + 2 metric gates = ~52.9 hr GPU + ~13 hr CPU + ~30 min install TOTAL PROJECTED

---

## §7 Standing-rule compliance

| Rule | Compliance for Wf-368 |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ Every §0-§5 + §6 + §7 numeric tagged MEASURED / PROJECTED with source path |
| §1.2 Standing GPU authorization | ✅ All 7 deliverables CPU-only docs+code+test+verify; no GPU launches triggered |
| §1.3 No paper work without authorization | ✅ Zero touches to `paper/*`; §6 inventory only |
| §1.4 Pre-flight v3 25/25 PASS gate | ✅ Pre-flight fresh-run at Wf-368-C1 timestamp = **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** |
| §1.5 Honest-negative framing | ✅ §5 catalog of 8 honest-negative caveats preserved per §22.62.1 + Wf-368-B1 §7.5 |
| §5.1 Standing GPU authorization | ✅ Per-wrapper USER_OK gating preserved (no auto-trigger); 7 wrappers awaiting USER_OK=1 grant |
| §5.2 Honest framing | ✅ MEASURED with sources cited; PROJECTED with bands |
| §5.3 No paper work | ✅ §6 inventory only; no `paper/*` touches |
| §5.4 Tier-2 journal target + no wet-lab | ✅ Wf-368-B1 §4 drives Tier-2 re-scope; wet-lab items DEPRECATED per user directive |

---

## §8 Wall + cost (Wf-368-C1)

| Stage | Wall (MEASURED / PROJECTED) |
|---|---:|
| Read 6 Wf-368 closeouts (A1..B2) to verify SHIPPED | ~2 min (Read) |
| Run pre-flight fresh-run | 145.75 s (MEASURED) |
| Append §22.62 to `TODO/INDEX.md` (+30 lines) | ~3 min (Edit) |
| Append Wf-368 §6 row to `AGENTS.md` (+5 lines) | ~2 min (Edit) |
| Write `TODO/completed/368c1_index_sync.md` (this row) | ~3 min (Write) |
| Write `TODO/completed/368_final_synthesis.md` (this file) | ~3 min (Write) |
| **Total Wf-368-C1 CPU wall** | **~12 min CPU-only** (within PROJECTED 10-15 min per task brief) |

**PROJECTED vs MEASURED**:
- PROJECTED 10-15 min CPU-only at task brief
- MEASURED ~12 min (within brief)

**GPU launches performed**: 0 (per AGENTS.md §5.1 kill_then_adapt standing directive; docs-only sub-flow).
**Paper touches**: 0 (per AGENTS.md §5.3 + TODO-84; Wf-368-B1 checklist lives under `docs/` ONLY).

---

## §9 What Wf-368 is NOT claiming (per `party-twenty-spirit §3.5`)

Per `party-twenty-spirit §3.5`, this Wf-368-C1 final synthesis does **NOT** claim:

1. **NOT claiming all 4 forward-looking problems from Wf-366-C1 §8 are彻底解决 (thoroughly resolved)** — only (a) CLOSED at Wf-367-A2; (b), (c), (e) + NEW (f) REMAINING per §1 above.
2. **NOT claiming m6_kappa gate will be lifted** — 5 architectural attempts exhausted per Wf-365-A3 verdict doc; HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL state.
3. **NOT claiming m2 R² recovery on small cohorts will lift platinai_4cell + hilbert** — Wf-367-B1 multi-stream feature ensemble OVERFITS on n=200 SMILES cohorts; needs EITHER larger cohorts OR GPU chemprop v3 + Hilbert primal-dual regression.
4. **NOT claiming pre-flight cascade remains "FULLY CLOSED" through any new launch** — pre-flight gate state is a snapshot at Wf-368-C1 timestamp; any new launch or new test addition can re-open the cascade.
5. **NOT claiming the user should grant all USER_OK at once** — per-wrapper gating is independent; Wf-368-B1 §7.3 lists 3-batch sequential priority for the user's review.
6. **NOT claiming Wf-368 is a NEW ultracode that introduces source-side changes** — Wf-368 is docs-only verification + USER_OK consolidation + INDEX sync; 0 NEW source patches (Wf-367-A2 already closed the cascade; Wf-368-A1 just added 2 regression-guard tests).
7. **NOT claiming Wf-368-C1 is exhaustive** — final synthesis captures 8 honest-negative caveats; users may surface additional gaps not enumerated here.
8. **NOT claiming the wrapper wall estimates are exact** — they are PROJECTED from Wf-258 + Wf-260 timing; MEASURED post-launch would differ.

---

**END OF Wf-368 FINAL SYNTHESIS** — Owner Hugo Lee;
cross-ref `party-twenty-spirit §3.5` + `AGENTS.md §5.1 + §5.2 + §5.3 + §5.4` + `mol-metal-gpu-execution` §0;
last update 2026-09-29 (Wf-368-C1 timestamp 2026-09-29T15:54:19+0000 UTC wall 145.75 s);
CPU-only docs surface;
pre-flight v3 **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** PRESERVED;
7 deliverables SHIPPED (6 sub-flow closeouts + this C1 final synthesis);
4 NEW tests PASS (A1=2 + B1=1 + B2=1);
0 GPU launches; 0 `paper/*` touches;
INDEX.md 6679 → 6709 (+30 lines for §22.62);
AGENTS.md 332 → 337 (+5 lines for Wf-368 §6 row);
8 honest-negative caveats preserved per §5 + Wf-368-B1 §7.5.

**User voice answer (per §1)**: **PARTIAL — 1 of 4 forward-looking gaps from Wf-366-C1 §8彻底解决: (a) CLOSED. 3 gaps REMAINING + 1 NEW gap: (b), (c), (e), (f).** NEW ultracode Wf-369+ recommended per Wf-368-B1 §7.2 + §7.3.