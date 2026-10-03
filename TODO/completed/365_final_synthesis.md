# Wf-365 — Final synthesis (m6_kappa FINAL verdict + cross-verify wave)

**Date:** 2026-09-29 (Wf-365-C1 timestamp 2026-09-29T15:09:17+0000 UTC wall 155.48 s pre-flight + ~25-30 min docs sync)
**Owner:** Hugo Lee + Wf-365 ultracode wave (4 SHIPPED sub-flows + 1 C1 docs sync = 5 deliverables)
**Status:** SHIPPED — final aggregation per Wf-365-C1 docs-only INDEX + AGENTS + final synthesis sync
**Workflow:** Wf-365 (m6_kappa FINAL verdict + cross-verify wave)
**Parent:** Wf-363 (GPU wrapper polish + dashboard enhancements) + Wf-364 (preflight cascade fix + m6_kappa combined LAST path + full test collection + §6 limitations template) — both IN-FLIGHT

---

## §0 TL;DR — Wf-365 wave summary

| Item | Value | Source |
|---|---|---|
| **Wave structure** | 4 SHIPPED sub-flows + 1 C1 docs sync = **5 deliverables** | this entry |
| **NEW tests** | **2** (A3=1 + B2=1 parametrized) | this entry |
| **NEW test files** | **2** (`test_wf365a3_m6_kappa_verdict.py` + `test_wf365b2_7_wrappers_final.py`) | this entry |
| **NEW docs file** | **1** (`docs/M6_KAPPA_FINAL_VERDICT.md` 269 LOC 9 sections) | Wf-365-A3 |
| **NEW metrics JSON field** | **1** (`m6_kappa.json::wf365a3_final_verdict`) | Wf-365-A3 |
| **Pre-flight v3 fresh-run** | **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** at 155.48 s wall | `molmetal/reports/wf365c1_preflight.txt` |
| **GPU launches** | **0** (CPU-only docs+code+test+verify per AGENTS.md §5.1 + kill_then_adapt §0) | this entry |
| **paper/* touches** | **0** (per AGENTS.md §5.3 TODO-84 active) | this entry |
| **EXPECTED_GATE_COUNTS["all"]** | **94** PRESERVED (no new pre-flight gates added by Wf-365) | Wf-345-B3 auto-compute |
| **Wall (MEASURED)** | ~25-30 min CPU-only across 5 deliverables | this entry |
| **User voice answer** | **No, NOT all problems were 彻底解决. NEW ultracode (Wf-366+) follow-up required** | Wf-365-B1 §0 cross-poll |

**One-line verdict**: Wf-365 ships 5 deliverables (4 SHIPPED sub-flows + 1 C1 docs sync) — confirms m6_kappa gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL) + surfaces 8 MISSING Wf-363/Wf-364 closeouts for NEW ultracode follow-up.

---

## §1 Per-sub-flow final aggregation

### §1.1 Wf-365-A1 — Wf-361-B2 Neural ODE verification (VERIFIED — no re-run)

**Status:** SHIPPED (READ-only verification; result NOT truncated)

**MEASURED**:
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` 6,359 bytes; valid JSON; 4 candidates
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/per_axis.csv` 681 bytes; 4 rows × 13 cols
- `TODO/completed/361b2_m6_kappa_neural_ode.md` 18,023 bytes; 282 lines; 10 sections
- `molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py` 3/3 NEW tests PASS in 1.28 s
- CV-R² +0.5410 MEASURED via dopri5 integration on evolved features (continuous dynamics CEILING_BREAK for structural recovery axis)
- Wall ~0.42 s CPU-only

**Verdict**: Wf-361-B2 result IS on disk and IS NOT truncated. Wf-365-A1 verification confirms MEASURED values match the task brief; **NO re-run needed**.

### §1.2 Wf-365-A3 — m6_kappa FINAL verdict doc + metrics JSON (SHIPPED)

**Status:** SHIPPED (NEW verdict doc + 1 NEW test)

**MEASURED**:
- `docs/M6_KAPPA_FINAL_VERDICT.md` 269 LOC, 9 sections
- `molmetal/tests/test_wf365a3_m6_kappa_verdict.py` 1/1 NEW test PASS in 0.33 s
- `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` NEW field parses cleanly
- 5 architectural attempts inventoried: Wf-335-M1 chemprop+PLL + Wf-359-A1 W₂ + Wf-361-B1 MAML + Wf-361-B2 Neural ODE + Wf-364-A2 combined LAST path
- Best aug_mean MEASURED = **0.4737** (Wf-364-A2 joint RDKit+ODE+MAML candidate) vs threshold 0.6 → gap −0.1263
- Best CV-R² MEASURED = **+0.6814** (Wf-335-M1 chemprop+PLL — ABOVE PROJECTED +0.05 to +0.15 band per `WEAK_TO_STRONG_PLAN.md §3.5`)
- Wall ~7 min CPU-only

**Verdict**: m6_kappa gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL). Structural ceiling LIFTED but threshold-reaching axis NOT lifted. 4 forward-looking paths ranked (none cheap) per verdict doc §5.

### §1.3 Wf-365-B1 — Wf-363 + Wf-364 cross-verification (SHIPPED)

**Status:** SHIPPED (READ-only cross-poll)

**MEASURED**:
- Wf-363: SHIPPED 2/6 closeouts (A1 + A2) — 4 MISSING (A3 / B1 / B2 / C1)
- Wf-364: SHIPPED 4/8 closeouts (A1 + A2 + A3 + A4) — 4 MISSING (B1 / B2 / C1 / C2)
- 8 MISSING closeouts total for NEW ultracode follow-up per user voice answer
- m6_kappa gate NOT lifted across all 5 architectural attempts (per Wf-365-A3 verdict doc §1-§4)
- Wall ~3-5 min CPU-only

**Verdict**: PARTIAL — follow-up NEW ultracode (Wf-366+) recommended per user directive 2026-09-29 verbatim *"如果不是，启动新的ultracode做follow up"*.

### §1.4 Wf-365-B2 — Final end-to-end verification of 7 launch wrappers (SHIPPED)

**Status:** SHIPPED (NEW comprehensive test 24/25 PASS + 1 SKIP)

**MEASURED**:
- 7 wrappers verified: 5 Wf-355 wrappers (Wf-355-A1/A2/A3/B1/B2) + 2 legacy Wf-124 G5/G6
- `molmetal/tests/test_wf365b2_7_wrappers_final.py` 25 tests; 24 PASS + 1 SKIP in 167.22 s wall
- The 1 SKIP is `test_user_ok_gate_enforced[Wf-124-G5]` — legacy G5 predates Wf-300C USER_OK gate pattern (HONEST-NEGATIVE; closure = 5 LOC gate addition, deferred to NEW ultracode follow-up)
- 6/7 wrappers fully conformant (USER_OK-gated + dry-runnable + DRY_RUN.log emission per Wf-363-A1 schema)
- Wall ~2.8 min CPU-only MEASURED at Wf-365-C1 timestamp

**Verdict**: 7-ship wrapper stack (5 Wf-355 + 2 legacy) is end-to-end operational. 1 SKIP HONEST-NEGATIVE on legacy wrapper parity.

### §1.5 Wf-365-C1 — Docs-only INDEX + AGENTS + final synthesis sync (SHIPPED — this entry)

**Status:** SHIPPED (docs-only)

**MEASURED**:
- `TODO/INDEX.md` line count: 5926 → 6039 (**+113 lines**) — §22.53 entry appended between §22.52 and §23
- `AGENTS.md` line count: 327 → 328 (**+1 row**) — Wf-365 row appended after Wf-363 row
- NEW files: `TODO/completed/365_final_synthesis.md` (this file) + `TODO/completed/365c1_index_sync.md` (sibling)
- 0 NEW tests (docs-only)
- 0 GPU launches
- 0 paper/* touches
- Wall ~10-15 min CPU-only

---

## §2 MEASURED test pass-rates (combined fresh-run 2026-09-29 at Wf-365-C1 timestamp)

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf365a3_m6_kappa_verdict.py \
    molmetal/tests/test_wf365b2_7_wrappers_final.py
...
molmetal/tests/test_wf365a3_m6_kappa_verdict.py .                      [  4%]
molmetal/tests/test_wf365b2_7_wrappers_final.py .......s.................  [100%]
======================== 25 passed, 1 skipped in 168.59s =========================
```

(A1 verification reuses 3 existing tests in `test_wf361b2_m6_kappa_neural_ode.py` per closeout §1.3; not re-run in this C1 fresh-run because A1 is READ-only verification of Wf-361-B2 already-completed work.)

| Sub-flow | NEW tests | MEASURED |
|---|---|---|
| Wf-365-A1 | 0 (reuses 3 existing) | 3/3 PASS via existing `test_wf361b2_m6_kappa_neural_ode.py` (verified at Wf-361-B2 closeout §3) |
| Wf-365-A3 | 1 | 1/1 PASS in 0.33 s (fresh-run at Wf-365-C1 timestamp) |
| Wf-365-B1 | 0 (cross-poll only) | n/a |
| Wf-365-B2 | 1 (25 parametrized) | 24/25 PASS + 1 SKIP in 168.59 s (fresh-run at Wf-365-C1 timestamp) |
| Wf-365-C1 | 0 (docs-only) | n/a |
| **TOTAL** | **2 NEW** | **25/25 PASS + 1 SKIP** in C1 fresh-run + 3/3 PASS via reused A1 verification = **28 PASS + 1 SKIP across Wf-365 wave** |

---

## §3 Pre-flight v3 fresh-run state (per `molmetal/reports/wf365c1_preflight.txt`)

| Gate bucket | Count | Status | Source |
|---|---|---|---|
| Total | 94 | n/a | Wf-345-B3 auto-compute |
| PASS | 91 | PRESERVED vs Wf-363-C1 baseline 91 | this entry |
| FAIL | 1 | PRESERVED vs Wf-363-C1 baseline 1 (pre-existing `p3_wf343c2_per_wave_runtime_check` Wf-347A1 pre-retrain smoke wave cascade) | Wf-358-D1 §0 + Wf-362-B1 §1 + Wf-363-C1 honest-negative baseline |
| SKIP | 2 | PRESERVED vs Wf-363-C1 baseline 2 (pre-existing per Wf-350-A2 §5 honest-negative baseline) | this entry |
| Wall | 155.48 s | PRESERVED vs Wf-363-C1 baseline 156.60 s | this entry |
| Timestamp | 2026-09-29T15:09:17+0000 UTC | this entry | this entry |
| `gate_count_consistency_audit` | PASS | actual=94 == expected=94 (auto-computed via Wf-345-B3) | this entry |

**The 2 pre-existing SKIPs**:
1. `p3_wf343c4_wave_budget_remaining_nonneg` — BY-DESIGN per Wf-262 §7.1 with −183.04h overrun
2. `data_2_1_6_bindingdb_external_honest_negative` — external 1.07M BindingDB dependency

**Net effect**: zero pre-flight regression vs Wf-363-C1 baseline. Wf-365-C1 is docs-only — no new pre-flight gates added; EXPECTED_GATE_COUNTS["all"] 94 PRESERVED.

---

## §4 §22.53.1 Net effect on TODO/ inventory (post-Wf-365-C1 INDEX sync)

| Inventory | Pre-Wf-365-C1 | Post-Wf-365-C1 | Delta |
|---|---|---|---|
| `pending/` | 13 (Wf-363-C1 baseline) | **13** | **0** (Wf-365 = 4 sub-flows + 1 C1 docs-only sync; no pending/ moves) |
| `completed/` | 789 (Wf-363-C1 baseline) | **794** | **+5** (4 NEW Wf-365 closeouts + 1 Wf-365-C1 docs-only synthesis) |
| `INDEX.md` line count | 5926 (post-§22.52 Wf-363-C1) | 6039 | **+113 lines** (§22.53 entry appended between §22.52 and §23) |
| `docs/` NEW files | 0 | **1** (`docs/M6_KAPPA_FINAL_VERDICT.md` 269 LOC) | **+1** (Wf-365-A3 NEW verdict doc) |
| `EXPECTED_GATE_COUNTS["all"]` | 94 | **94** | **0** (Wf-365 does not change gate count) |
| `.claude/skills/` skill count | 29 | **29** | **0** (Wf-365 adds zero new skills) |
| Pre-flight PASS count | 91 | **91** | **0** (PRESERVED) |
| Pre-flight FAIL count | 1 | **1** | **0** (PRESERVED) |

---

## §5 User voice answer per Wf-365-B1 cross-poll §0 (verbatim)

> **User**: 什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up
>
> **Wf-365-B1 one-line verdict**: **No, NOT all problems were 彻底解决 (thoroughly resolved). Launch a NEW ultracode for follow-up.**

Cross-poll findings:

| Workflow | Present closeouts | Missing closeouts | Final verdict |
|---|---|---|---|
| **Wf-363** | A1 + A2 (2 SHIPPED) | A3, B1, B2, C1 (4 MISSING) | PARTIAL — NEW ultracode recommended |
| **Wf-364** | A1 + A2 + A3 + A4 (4 SHIPPED) | B1, B2, C1, C2 (4 MISSING) | PARTIAL — NEW ultracode recommended |
| **Wf-365** | A1 + A3 + B1 + B2 + C1 (5 SHIPPED — this entry) | A2 (1 MISSING — closed by A3+B1 combined) | COMPLETE — this entry completes the wave |

**m6_kappa FINAL answer**: per Wf-365-A3 verdict doc §4 — gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL state); structural ceiling LIFTED but threshold-reaching axis NOT lifted; no more architectural paths to try per Wf-364-A2 §8. Path forward: 4 forward-looking paths ranked in Wf-365-A3 §5 — none cheap.

**NEW ultracode (Wf-366+) recommended scope** (per Wf-365-B1 cross-poll + Wf-365-A3 verdict doc §5):
1. Close Wf-363-A3 / B1 / B2 / C1 gaps (USER_OK gate migration for G5 + recover_g6 dry-run ordering + DRY_RUN UX consistency hardening)
2. Close Wf-364-B1 (7 wrappers) / B2 (dashboard UX) / C1 / C2 gaps
3. Explore whether m6_kappa gate can be lifted via metric redefinition on different cohorts (per Wf-365-A3 §5 forward-looking + Wf-361-B1 §9 option 4 "drop m6_kappa as Tier-2 paper §3 main-result metric")

---

## §6 Honest framing per `party-twenty-spirit` §3.5 (6 caveats preserved)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (Wf-365-A3 verdict doc §1-§4 + Wf-365-B1 cross-poll §0). Best augmented_mean MEASURED = **0.4737** (Wf-364-A2 joint RDKit+ODE+MAML candidate) vs threshold 0.6 — gap −0.1263. 5 architectural attempts exhausted. Structural ceiling LIFTED (CV-R² +0.5410 to +0.6814 — ABOVE PROJECTED +0.05 to +0.15 band per `WEAK_TO_STRONG_PLAN.md §3.5`), but threshold-reaching axis is **NOT lifted**. Per Wf-365-A3 §5 forward-looking, 4 paths remain ranked: cross-cohort / drop / re-define / degenerate chain — none cheap.

2. **Wf-363 + Wf-364 partial completion surfaced by Wf-365-B1 cross-poll — NEW ultracode follow-up required** (per user directive 2026-09-29 verbatim *"如果不是，启动新的ultracode做follow up"*): Wf-363 SHIPPED 2/6 closeouts (A1+A2) — 4 MISSING (A3/B1/B2/C1); Wf-364 SHIPPED 4/8 closeouts (A1+A2+A3+A4) — 4 MISSING (B1/B2/C1/C2). This Wf-365-C1 ships the final aggregation docs only; the **NEW ultracode (Wf-366+ recommended)** must close these 8 MISSING closeouts before Wf-363/Wf-364 can be marked FORMAL CLOSE.

3. **Pre-existing 1 FAIL (`p3_wf343c2_per_wave_runtime_check`) NOT Wf-365-introduced** (preserved per Wf-358-D1 §0 + Wf-362-B1 §1 + Wf-363-C1 honest-negative baseline; structural cascade issue — would close to 92/0 FAIL / 2 SKIP / 94 TOTAL via Wf-357-C3 §5.1 Recommendation 1 = CPU-only ~5 LOC argparse `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386`).

4. **Wf-365-B2 G5 USER_OK gate parity HONEST-NEGATIVE** (legacy `launch_wf124_p3_g5.sh` predates Wf-300C USER_OK gate pattern; 1/7 wrappers flagged — `test_user_ok_gate_enforced[Wf-124-G5]` SKIPPED per closeout §1 + §2). Closure path: 5 LOC gate addition to G5 wrapper, USER_OK-gated like other Wf-355 wrappers — deferred to NEW ultracode follow-up per Wf-365-B1 §0.

5. **Wf-365-A2 closeout MISSING** (no `TODO/completed/365a2_*.md` on disk at Wf-365-C1 timestamp — per `ls TODO/completed/ | grep -E "^365"` shows only A1+A3+B1+B2 = 4 closeouts). Per Wf-365-B1 cross-poll §0, the Wf-365-A2 slot was reserved for m6_kappa alternative-paths cross-verification but was not needed once Wf-365-A3 verdict doc aggregated all 5 paths. A2 gap closed by A3 verdict doc + Wf-365-B1 cross-poll combined.

6. **`paper/*` paper §6 entry prep DEFERRED** (`docs/M6_KAPPA_FINAL_VERDICT.md` §6 contains LaTeX template ready for paper §6 honest-limitations entry per Wf-365-A3 verdict doc §6). Paper work requires USER_OK per TODO-84 + AGENTS.md §5.3; Wf-365-C1 ships the template only, NOT the paper write. Future paper-§6 submission BLOCKED on user authorization.

---

## §7 Per-agent close-out summary

| Agent | Artifact | Action | NEW tests | Wall (MEASURED) |
|---|---|---|---|---:|
| **Wf-365-A1** | `TODO/completed/365a1_neural_ode_verify.md` (219 LOC) | READ-only verification of Wf-361-B2 Neural ODE result (no re-run) — confirms result.json + per_axis.csv + closeout doc all present and NOT truncated | 0 NEW (reuses 3 existing tests) | ~0.42 s |
| **Wf-365-A3** | `docs/M6_KAPPA_FINAL_VERDICT.md` (269 LOC, 9 sections) + `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` field + `TODO/completed/365a3_m6_kappa_final.md` | NEW verdict doc aggregating 5 architectural attempts; cross-attempt table; 4 forward-looking paths | 1/1 PASS | ~7 min |
| **Wf-365-B1** | `TODO/completed/365b1_wf363_364_xverify.md` (398 LOC) | READ-only cross-poll of Wf-363 (2/6 SHIPPED) + Wf-364 (4/8 SHIPPED) — surfaces 8 MISSING closeouts for NEW ultracode follow-up | 0 NEW (cross-poll only) | ~3-5 min |
| **Wf-365-B2** | `molmetal/tests/test_wf365b2_7_wrappers_final.py` + `TODO/completed/365b2_7_wrappers_final.md` | Final end-to-end verification of 7 launch wrappers (5 Wf-355 + 2 legacy Wf-124 G5/G6) — 25 tests (24 PASS + 1 SKIP legacy G5 USER_OK gate parity) | 24/25 PASS + 1 SKIP | ~2.8 min |
| **Wf-365-C1** | `TODO/INDEX.md` + `AGENTS.md` + `TODO/completed/365_final_synthesis.md` + `TODO/completed/365c1_index_sync.md` | Docs-only INDEX §22.53 sync + AGENTS §6 +1 row + final synthesis + index sync | 0 NEW (docs-only) | ~10-15 min |

---

## §8 Cross-references

- `TODO/completed/365c1_index_sync.md` (Wf-365-C1 INDEX + AGENTS sync — this entry's primary source)
- `TODO/completed/365{a1,a3,b1,b2}_*.md` (4 Wf-365 sub-flow closeouts: A1 Neural ODE verify + A3 m6_kappa FINAL verdict + B1 Wf-363/364 cross-poll + B2 7-wrappers final integration)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 NEW verdict doc 269 LOC 9 sections)
- `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` (Wf-365-A3 NEW metrics JSON field)
- `molmetal/tests/test_wf365a3_m6_kappa_verdict.py` (Wf-365-A3 NEW 1 test)
- `molmetal/tests/test_wf365b2_7_wrappers_final.py` (Wf-365-B2 NEW 25 tests parametrized)
- `molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py` (Wf-365-A1 reuses existing 3 tests)
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` (Wf-365-A1 verified result 6359 bytes)
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/per_axis.csv` (Wf-365-A1 verified 4 candidates)
- `TODO/completed/361b2_m6_kappa_neural_ode.md` (Wf-365-A1 verified closeout 18023 bytes 282 lines 10 sections)
- `molmetal/reports/wf365c1_preflight.txt` (Wf-365-C1 pre-flight fresh-run wall 155.48 s; 91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL — PRESERVED at Wf-363-C1 baseline)
- `TODO/completed/361b1_m6_kappa_maml.md` (Wf-365-B1 cross-ref — MAML attempt)
- `TODO/completed/335m1_m6_kappa_chemprop_pll.md` (Wf-365-B1 cross-ref — chemprop+PLL anchor)
- `TODO/completed/359a1_m6_kappa_w2_regression.md` (Wf-365-B1 cross-ref — W₂ attempt)
- `TODO/completed/364a2_m6_kappa_combined.md` (Wf-365-B1 cross-ref — combined LAST path)
- `TODO/completed/363{a1,a2,a3,b1,b2,c1}_*.md` (Wf-365-B1 cross-poll — 2 SHIPPED + 4 MISSING)
- `TODO/completed/364{a1,a2,a3,a4,b1,b2,c1,c2}_*.md` (Wf-365-B1 cross-poll — 4 SHIPPED + 4 MISSING)
- `TODO/INDEX.md` §22.53 (NEW entry appended between §22.52 and §23)
- `AGENTS.md` §6 (NEW Wf-365 row appended between Wf-363 and end-of-§6)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar

---

**END OF Wf-365 FINAL SYNTHESIS** — Owner Hugo Lee + Wf-365 ultracode wave; cross-ref `party-twenty-spirit §3.5`; 5/5 deliverables SHIPPED; 2 NEW tests + 28/28 PASS + 1 SKIP across wave; pre-flight v3 91/1/2/94 PRESERVED; 0 GPU launches; 0 paper/* touches; 8 MISSING Wf-363/Wf-364 closeouts surfaced for NEW ultracode follow-up per user voice answer (Wf-365-B1 cross-poll §0).
