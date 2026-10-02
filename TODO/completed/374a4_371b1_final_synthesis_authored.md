# Wf-374-A4 — Wf-371-B1 final synthesis retroactive close (docs-only ship)

**Date:** 2026-09-30 (Wf-374-A4 timestamp)
**Owner:** Hugo Lee + Wf-374-A4 ultracode agent
**Workflow:** Wf-374-A4 (single-task docs-only ship — retroactive closure of Wf-371-B1 gap)
**Parent:** Wf-371 wave (A1+A2+A3+B1) + Wf-373-A2 verification (2/4 SHIPPED + 2 MISSING) +
Wf-374 follow-up ultracode (Wf-374-A3 closed A1; this Wf-374-A4 closes B1)

---

## §0 TL;DR

| Item | Value | Source |
|---|---:|---|
| **Final synthesis file authored** | `TODO/completed/371b1_final_synthesis.md` (NEW, retroactive close of Wf-371-B1 gap) | this entry |
| **File size** | 524 lines (~400 LOC target) | this entry |
| **Sections** | 7 (§0 TL;DR + §1-§6) | task brief §4 listed 7 numbered sections |
| **NEW test file** | (none — docs-only synthesis) | this entry |
| **Tests status** | 0 NEW (docs-only aggregation only; 4 NEW tests already SHIPPED across Wf-371 A1+A2+A3) | this entry |
| **GPU launches** | 0 | AGENTS.md §5.1 kill_then_adapt |
| **paper/* touches** | 0 | AGENTS.md §5.3 TODO-84 active |
| **Pre-flight v3 gate** | **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** PRESERVED at Wf-370-B1 baseline (NOT re-run by Wf-374-A4) | `molmetal/reports/wf370b1_preflight.txt` |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED (no new pre-flight gates added by Wf-374-A4) | Wf-345-B3 auto-compute |
| **Wall (MEASURED)** | ~16 min CPU-only (1 file Write; slightly over brief 10-15 min due to 7-section comprehensive closeout) | this entry |
| **Wave structure** | 1 deliverable (this entry) | this entry |

**One-line verdict**: Wf-374-A4 retroactively closes the Wf-371-B1 gap surfaced by
Wf-373-A2 §1.4 verification by authoring `TODO/completed/371b1_final_synthesis.md`
(524 lines, 7 sections: §0 TL;DR + §1 Wf-371 wave structure + §2 What 彻底解决 +
§3 What NOT 彻底解决 + §4 Wf-374-A3+A4 follow-up closure + §5 Honest framing +
§6 Cross-references). The final synthesis aggregates Wf-371-A1 + Wf-371-A2 + Wf-371-A3
closeouts + Wf-365-A3 verdict doc + Wf-368-C1 standalone + Wf-369-A3 catalog +
Wf-370-A3 session synthesis into single entry per Wf-373-A2 §4 retroactive closure
recommendation. **Wf-371 wave SHIPS 4/4 retroactively** at Wf-374-A4 timestamp.

---

## §1 Task brief verification

Per the computed task brief for Wf-374-A4:

| Step | Brief item | Status |
|---|---|---|
| 1 | `cd /home/hugo/codes/mol-metal` | ✅ CWD already at repo root |
| 2 | Read Wf-371-A2 closeout | ✅ Read `TODO/completed/371a2_368c1_standalone_authored.md` (131 lines, 6 sections) |
| 3 | Read Wf-371-A3 closeout | ✅ Read `TODO/completed/371a3_365a2_standalone_authored.md` (227 lines, 7 sections) |
| 4 | Read Wf-374-A3 closeout (Wf-371-A1 retroactive) | ✅ Read `TODO/completed/371a1_m6_kappa_degenerate.md` (155 lines, 8 sections) |
| 5 | Read Wf-365-A3 verdict doc + Wf-368-C1 standalone closeout | ✅ Read `TODO/completed/365a3_m6_kappa_final.md` + `TODO/completed/368c1_final_synthesis.md` |
| 6 | Author NEW `TODO/completed/371b1_final_synthesis.md` (~400 LOC, 6 sections) per template | ✅ Authored with 7 sections (§0 TL;DR + §1-§6); 524 lines |
| 7 | No NEW test file (docs-only synthesis aggregation) | ✅ 0 NEW tests (consistent with docs-only nature) |
| 8 | Document this entry `TODO/completed/374a4_371b1_final_synthesis_authored.md` | ✅ This file |

### §1.1 Section count clarification

The task brief said "~400 LOC, 6 sections" but enumerated 7 numbered sections (§0 TL;DR +
§1 Wf-371 wave structure + §2 What 彻底解决 + §3 What NOT 彻底解决 + §4 Wf-374-A3+A4
closure + §5 Honest framing + §6 Cross-references). Wf-374-A4 follows the enumerated
structure with 7 sections matching the task brief section list, totaling 524 lines (~400
LOC target + ~30% cross-reference expansion).

---

## §2 Final synthesis content summary (`TODO/completed/371b1_final_synthesis.md`)

### §2.1 Section-by-section coverage

| § | Title | Coverage | Task brief alignment |
|---|---|---|---|
| §0 | TL;DR — User voice answer | 1-paragraph verdict + 10-row axis verdict table + 2 sub-tables (what 彻底解决 / what NOT 彻底解决) + user voice question verbatim + NEW ultracode follow-up recommendation | task brief §4 (§0 TL;DR) |
| §1 | Wf-371 wave structure (4 sub-flows) | 4-row sub-flow table + per-sub-flow ship evidence table (file / LOC / sections / NEW tests / wall) + retroactive closure rationale | task brief §4 (§1) |
| §2 | What 彻底解决 (thoroughly resolved) | 4 sub-sections covering Wf-368-C1 final_synthesis standalone (A2) + Wf-365-A2 winner CLI flag standalone (A3) + m6_kappa DEGENERATE chain (A1 retroactive) + Wf-371-B1 final synthesis retroactive (this entry) | task brief §4 (§2) |
| §3 | What NOT 彻底解决 (remains open HONEST_NEGATIVE) | 5 sub-sections covering m6_kappa structural floor + m2 R² small-cohort overfit + G6 ~11.4 hr launch + 6 forward-looking USER_OK items + Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity | task brief §4 (§3) |
| §4 | Wf-374-A3 + Wf-374-A4 follow-up closure (4/4 retroactive) | 4 sub-sections covering Wf-373-A2 verification verdict + Wf-374-A3 retroactive closure of A1 + Wf-374-A4 retroactive closure of B1 + retroactive 4/4 closure verdict | task brief §4 (§4) |
| §5 | Honest framing per `party-twenty-spirit §3.5` | 4 sub-sections covering MEASURED-vs-PROJECTED tag discipline + 6 honest-negative caveats + standing directives re-confirmed + PROGRESSIVE TRUTH user voice answer | task brief §4 (§5) |
| §6 | Cross-references | 10 sub-sections covering Wf-371 primary closeouts + Wf-373 + Wf-374 follow-up + m6_kappa verdict + Wf-368-C1 + Wf-365-A2 + catalog + session synthesis + pre-flight cascade closure trail + per-section source pointers + skill cross-refs + wall+cost | task brief §4 (§6) |

### §2.2 Cross-reference targets (verified)

- `TODO/completed/371a1_m6_kappa_degenerate.md` ✅ (Wf-374-A3 retroactive closeout of Wf-371-A1)
- `TODO/completed/371a2_368c1_standalone_authored.md` ✅ (Wf-371-A2 original wave closeout)
- `TODO/completed/371a3_365a2_standalone_authored.md` ✅ (Wf-371-A3 original wave closeout)
- `TODO/completed/365a3_m6_kappa_final.md` ✅ (Wf-365-A3 verdict doc; 5 attempts inventoried)
- `TODO/completed/368c1_final_synthesis.md` ✅ (Wf-371-A2 reconstructed Wf-368-C1 final_synthesis)
- `TODO/completed/368c1_index_sync.md` ✅ (Wf-368-C1 INDEX sync)
- `TODO/completed/369a3_m6_kappa_summary.md` ✅ (Wf-369-A3 §5 forward-looking recommendation #4 = DEGENERATE chain)
- `TODO/completed/373a2_wf371_xverify.md` ✅ (Wf-373-A2 verification of Wf-371 wave; 2/4 SHIPPED)
- `TODO/completed/370_final_synthesis.md` ✅ (Wf-370-B1 final ship + user voice answer)
- `TODO/completed/367c1_index_sync.md` ✅ (Wf-367-C1 pre-flight 92/0/2/94 FIRST verification post-cascade)
- `molmetal/reports/wf370b1_preflight.txt` ✅ (Wf-370-B1 fresh-run 92/0/2/94 wall 152.53 s)
- `docs/M6_KAPPA_FINAL_VERDICT.md` ✅ (Wf-365-A3 NEW 269 LOC, 9 sections, FINAL HONEST_NEGATIVE_PARTIAL_STRUCTURAL)
- `docs/HONEST_NEGATIVE_CATALOG.md` ✅ (65 entries; 9 → 65 = 7.2× expansion)
- `docs/M2_R2_FINAL_VERDICT.md` ✅ (Wf-369-A2 NEW 318 LOC, 7 sections; m2 R² FINAL per-cohort verdict)
- `docs/USER_OK_FINAL_CHECKLIST.md` ✅ (Wf-368-B1 NEW ~430 LOC, 9 sections; 7 USER_OK-gated launches)
- `docs/SESSION_FINAL_SYNTHESIS.md` ✅ (Wf-370-A3 NEW ~560 LOC, 10 sections; session-wide aggregation)
- `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` ✅
- `metrics/by_metric/m6_kappa.json::wf371a1_degenerate_chain_verify` ✅
- `molmetal/scripts/pre_flight_check.py` ✅ (gate_count_consistency_audit PASS: actual=94 == expected=94)
- `molmetal/tests/test_wf371a1_m6_kappa_degenerate.py` ✅ (Wf-371-A1 NEW 2 tests)
- `molmetal/tests/test_wf371a2_368c1_standalone.py` ✅ (Wf-371-A2 NEW 1 test)
- `molmetal/tests/test_wf371a3_365a2_standalone.py` ✅ (Wf-371-A3 NEW 1 test)

---

## §3 Wf-371 wave retroactive 4/4 verdict

### §3.1 Pre-Wf-374-A4 (at Wf-373-A2 timestamp 2026-09-30)

| Agent | Status | File |
|---|---|---|
| Wf-371-A1 (m6_kappa degenerate) | ❌ MISSING HONEST-NEGATIVE | (gap flagged) |
| Wf-371-A2 (Wf-368-C1 standalone) | ✅ SHIPPED | `TODO/completed/371a2_368c1_standalone_authored.md` |
| Wf-371-A3 (Wf-365-A2 standalone) | ✅ SHIPPED | `TODO/completed/371a3_365a2_standalone_authored.md` |
| Wf-371-B1 (final synthesis) | ❌ MISSING HONEST-NEGATIVE | (gap flagged) |
| **TOTAL** | **2/4 SHIPPED** | **2 MISSING flagged** |

### §3.2 Post-Wf-374-A4 (retroactive 4/4 SHIPPED)

| Agent | Status | File |
|---|---|---|
| Wf-371-A1 (m6_kappa degenerate) | ✅ SHIPPED via Wf-374-A3 retroactive | `TODO/completed/371a1_m6_kappa_degenerate.md` (155 lines, 8 sections, 2 NEW tests) |
| Wf-371-A2 (Wf-368-C1 standalone) | ✅ SHIPPED (preserved from original wave) | `TODO/completed/371a2_368c1_standalone_authored.md` |
| Wf-371-A3 (Wf-365-A2 standalone) | ✅ SHIPPED (preserved from original wave) | `TODO/completed/371a3_365a2_standalone_authored.md` |
| **Wf-371-B1 (final synthesis)** | ✅ **SHIPPED via Wf-374-A4 retroactive (this entry)** | **`TODO/completed/371b1_final_synthesis.md`** (524 lines, 7 sections, 0 NEW tests docs-only) |
| **TOTAL** | **4/4 SHIPPED retroactively** | **4 files on disk** |

**Wf-371 wave retroactive verdict**: **4/4 SHIPPED** — gap closure complete via
Wf-374-A3 (m6_kappa degenerate) + Wf-374-A4 (final synthesis = this entry).

---

## §4 Honest framing per `party-twenty-spirit §3.5`

### §4.1 MEASURED-vs-PROJECTED tag discipline

Every numeric value in the final synthesis traces to a JSON / report path on disk:
- Wf-371-A1 MEASURED: aug_mean 0.4493, CV-R² +0.5694, gap −0.1507, delta_mean −0.0018,
  pass_rate_lift @0.60 −0.0167 (per Wf-371-A1 §2 + result.json)
- Wf-371-A2 MEASURED: 1 NEW test PASS in 0.32 s; ~280 LOC; 9 invariants verified
  (per Wf-371-A2 §1 + §4)
- Wf-371-A3 MEASURED: 1 NEW test PASS in 0.32 s; ~14.5 KB; 12 invariants verified;
  Neural ODE 0.4495 + MAML 0.4288 + `--neural-ode-flow` CLI flag (per Wf-371-A3 §3)
- Pre-flight state PROJECTED preserved at 92/0/2/94 (NOT re-run by Wf-374-A4 docs-only
  synthesis; PRESERVED at Wf-370-B1 baseline per `molmetal/reports/wf370b1_preflight.txt`)
- Wf-371 wave combined wall PROJECTED 32-45 min → MEASURED ~32 min CPU-only

### §4.2 6 honest-negative caveats preserved

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL**: best
   aug_mean MEASURED 0.4737 (Wf-364-A2) vs threshold 0.6 — gap −0.1263; Wf-371-A1
   DEGENERATE chain aug_mean 0.4493 confirms metric-intrinsic floor; 7 architectural
   attempts exhausted per Wf-371-A1 §3 + Wf-365-A3 verdict doc; structural CV-R²
   +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP FINAL**: platinai_4cell + hilbert
   cohorts n=200 SMILES each; 50+ feature Ridge ensemble OVERFITS on small cohorts
   per Wf-367-B1; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands;
   FINAL per Wf-369-A2 verdict doc
3. **Pre-existing 2 SKIPs PRESERVED (BY-DESIGN)**: `p3_wf343c4_wave_budget_remaining_nonneg`
   per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative`
   external 1.07M dependency; NOT Wf-374-A4-introduced
4. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK**: per Wf-368-B1 §7.1; closure path =
   USER_OK=1 + VRAM budget check per AGENTS.md §5.1
5. **7 USER_OK-gated launch wrappers + 6 forward-looking items BLOCKED on USER_OK**
   (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0 totals); no auto-trigger
   per §5.1 + kill_then_adapt §0
6. **Wf-371-B1 is docs-only synthesis — 0 NEW tests**: Wf-371-B1 does NOT add pre-flight
   gates; does NOT modify source code; does NOT re-run pre-flight; only aggregates
   Wf-371 wave A1+A2+A3 closeouts into single synthesis entry per Wf-373-A2 §4 + Wf-374
   task brief

### §4.3 PROGRESSIVE TRUTH (per `party-twenty-spirit` §3.5)

The user voice question is answered as: **PARTIAL — Wf-371 wave + Wf-374-A3+A4 retroactive
closure DID彻底解决 the 2 final_synthesis/standalone reconstruction gaps (A2+A3 in
original wave; A1+B1 via retroactive Wf-374-A3+this A4). They did NOT彻底解决 the
structural/metric gates (m6_kappa + m2 R² FINAL honest-negative) or unblock the 7+6
USER_OK-gated launches (~52.9 hr GPU + 13 hr CPU + 30 min install). NEW ultracode
Wf-375+ IS the recommended follow-up per Wf-368-B1 §7.2 + Wf-370-A3 §9 with 6
USER_OK-gated sub-flows (~21.5 hr GPU + 10 min CPU total).**

---

## §5 Standing-rule compliance

| Rule | Compliance for Wf-374-A4 |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ Every §0-§6 numeric tagged MEASURED / PROJECTED with source path on disk |
| §1.2 Standing GPU authorization (5.1) | ✅ Docs-only synthesis; 0 GPU launches triggered |
| §1.3 No paper work without authorization (5.3) | ✅ 0 touches to `paper/*`; Wf-371-B1 lives under `TODO/` ONLY |
| §1.4 Pre-flight v3 92/0/2/94 gate | ✅ Pre-flight NOT re-run (docs-only); PRESERVED at Wf-370-B1 baseline |
| §1.5 Honest-negative framing (5.2) | ✅ 6 HONEST-NEGATIVE caveats preserved per §5 |
| §5.1 Standing GPU authorization | ✅ kill_then_adapt respected; 0 GPU launches |
| §5.2 Honest framing | ✅ MEASURED with sources cited; PROJECTED with bands |
| §5.3 No paper work | ✅ 0 `paper/*` touches |
| §5.4 Tier-2 journal target + no wet-lab | ✅ Re-confirmed per 2026-09-29 user directive |

---

## §6 Cross-references

- `TODO/completed/371b1_final_synthesis.md` (Wf-374-A4 NEW — Wf-371-B1 retroactive final synthesis)
- `TODO/completed/371a1_m6_kappa_degenerate.md` (Wf-374-A3 retroactive closeout of Wf-371-A1)
- `TODO/completed/371a2_368c1_standalone_authored.md` (Wf-371-A2 original wave closeout)
- `TODO/completed/371a3_365a2_standalone_authored.md` (Wf-371-A3 original wave closeout)
- `TODO/completed/373a2_wf371_xverify.md` (Wf-373-A2 verification of Wf-371 wave)
- `TODO/completed/365a3_m6_kappa_final.md` (Wf-365-A3 verdict aggregation)
- `TODO/completed/368c1_final_synthesis.md` (Wf-371-A2 reconstructed Wf-368-C1 final_synthesis)
- `TODO/completed/369a3_m6_kappa_summary.md` (Wf-369-A3 §5 forward-looking recommendation #4)
- `TODO/completed/370_final_synthesis.md` (Wf-370-B1 final ship + user voice answer)
- `TODO/completed/367c1_index_sync.md` (Wf-367-C1 pre-flight 92/0/2/94 FIRST verification post-cascade)
- `molmetal/reports/wf370b1_preflight.txt` (Wf-370-B1 fresh-run 92/0/2/94 wall 152.53 s)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 NEW 269 LOC, 9 sections)
- `docs/HONEST_NEGATIVE_CATALOG.md` (65 entries; 9 → 65 = 7.2× expansion)
- `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 NEW 318 LOC, 7 sections)
- `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 NEW ~430 LOC, 9 sections)
- `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 NEW ~560 LOC, 10 sections)
- `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` (Wf-365-A3 metrics JSON field)
- `metrics/by_metric/m6_kappa.json::wf371a1_degenerate_chain_verify` (Wf-371-A1 metrics JSON field)
- `molmetal/scripts/pre_flight_check.py` (gate_count_consistency_audit PASS)
- `molmetal/tests/test_wf371a1_m6_kappa_degenerate.py` (Wf-371-A1 NEW 2 tests)
- `molmetal/tests/test_wf371a2_368c1_standalone.py` (Wf-371-A2 NEW 1 test)
- `molmetal/tests/test_wf371a3_365a2_standalone.py` (Wf-371-A3 NEW 1 test)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §5.1 standing GPU authorization (NOT exercised; CPU-only docs) + §5.2 honest
  framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar

---

**END OF Wf-374-A4 DOCS-ONLY SHIP** — Owner Hugo Lee + Wf-374-A4 ultracode agent;
cross-ref `party-twenty-spirit §3.5`; **1 deliverable SHIPPED** (this entry); 0 NEW
tests (docs-only synthesis aggregation); Wf-371 wave closeout inventory **4/4
retroactive** (2/4 original wave + 2/4 retroactive via Wf-374-A3+A4); pre-flight v3
**92/0/2/94** PRESERVED at Wf-370-B1 baseline (NOT re-run by Wf-374-A4 docs-only);
EXPECTED_GATE_COUNTS["all"] 94 PRESERVED; 0 GPU launches; 0 paper/* touches; m6_kappa
gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL (gap −0.1263 to
threshold 0.6); m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP FINAL; 7 USER_OK-gated
launches + 6 forward-looking items BLOCKED on user OK; NEW ultracode Wf-375+ IS the
recommended follow-up per Wf-368-B1 §7.2 + Wf-370-A3 §9.
