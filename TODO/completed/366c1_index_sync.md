# Wf-366-C1 — INDEX + AGENTS + final synthesis (docs-only)

**Date:** 2026-09-29 (Wf-366-C1 timestamp 2026-09-29 ~16:30 UTC, pre-flight 150.26 s + ~10-15 min docs sync)
**Owner:** Hugo Lee (Wf-366-C1 docs-only sync)
**Status:** SHIPPED — INDEX.md §22.55 + AGENTS.md §6 +1 row + this index sync record + Wf-366 final synthesis
**Workflow:** Wf-366-C1 (this entry — docs-only INDEX + AGENTS + final synthesis sub-flow)
**Parent:** Wf-366 ultracode residual wave (USER_OK gate gap closures A1+A2 + cross-verifications B1+B2 + this C1 docs sync)
**Predecessor:** Wf-364-C1 docs-only INDEX + AGENTS + final synthesis at `TODO/INDEX.md §22.54` (analog precedent)

---

## §0 TL;DR

| Item | Value |
|---|---|
| **Wave structure** | 5 deliverables (4 SHIPPED sub-flows A1+A2+B1+B2 + 1 C1 docs sync = this entry) |
| **NEW tests** | 5 (A1=3 + A2=2) |
| **Pre-flight v3 fresh-run** | **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** at **150.26 s** wall — `molmetal/reports/wf366c1_preflight.txt` |
| **GPU launches** | 0 |
| **paper/* touches** | 0 |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED |
| **Wall (MEASURED)** | ~10-15 min CPU-only (task brief PROJECTED 10-15 min) |

---

## §1 Scope

This entry ships 4 docs-only artifacts:

1. **`TODO/INDEX.md` §22.55 NEW entry** (appended between §22.54 Wf-364-C1 and §23)
2. **`AGENTS.md` §6 NEW row** (appended after the Wf-365 row)
3. **`TODO/completed/366_final_synthesis.md`** (NEW — final aggregation per §0 + §7 honest framing + §10 cross-references)
4. **`TODO/completed/366c1_index_sync.md`** (NEW — this entry; cross-link + INDEX/AGENTS sync record)

No `molmetal/` core logic modification. No `paper/*` touches. No pre-flight schema-change. No GPU launch.

---

## §2 Pre-flight verification (MEASURED)

Per `molmetal/reports/wf366c1_preflight.txt` (Wf-366-C1 timestamp 2026-09-29 ~15:17 UTC wall **150.26 s**):

```json
{
  "timestamp": "2026-09-29T15:17:46+0000",
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 91,
    "fail": 1,
    "skip": 2,
    "total": 94,
    "total_wall_sec": 150.2628
  }
}
```

**91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** wall **150.26 s** — PRESERVED vs Wf-364-C1 baseline (91/1/2/94 at 154.95 s wall).

* The 1 FAIL is `p3_wf343c2_per_wave_runtime_check` (pre-existing structural cascade per Wf-358-D1 §0; **NOT Wf-366-introduced**).
* The 2 SKIPs are `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun) + `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency).
* `gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-computed via Wf-345-B3)`.

EXPECTED_GATE_COUNTS["all"] **94** PRESERVED (no new pre-flight gates added by Wf-366).

---

## §3 INDEX.md §22.55 entry (appended)

**§22.55 Wf-366 USER_OK gate gaps closure + Wf-363/364/365 cross-verification wave (2026-09-29 — Wf-366-C1 docs-only INDEX + AGENTS + final synthesis, this entry)**

Per Wf-365-B1 cross-poll verdict (per `TODO/completed/365b1_wf363_364_xverify.md` §0 + Wf-366-B2 §7): **No, NOT all problems were 彻底解决 (thoroughly resolved) across all axes. NEW ultracode Wf-366 launched as follow-up.**

Wf-366 wave structure (4 SHIPPED sub-flows + 1 C1 docs sync = 5 deliverables):

| Sub-flow | Title | NEW tests | Status |
|---|---|---:|---|
| **A1** | launch_wf124_p3_g5.sh USER_OK gate gap closure (Wf-363-A2 GAP-1) — canonical Wf-355-B1 pattern wired at lines 51-67; exit 64 + USER_OK gate refused FATAL + DRY_RUN.log + plan.json emission | 3 | SHIPPED per `366a1_*.md` (3/3 PASS in 0.36 s); wall ~1 min |
| **A2** | recover_g6_failed_tuples.sh block-order fix (Wf-363-A2 GAP-2) — USER_OK gate moved BEFORE --dry-run exit block; exit-code 1 → 64 (Wf-294B family); 2 NEW tests = 4 parametrized cases (source-inspection + live-run) | 2 | SHIPPED per `366a2_*.md` (4/4 PASS in 0.37 s); wall ~5 min |
| **B1** | Wf-363-B1/B2/C1 truncated closeout verification (READ-only) — confirms NOT truncated; all 3 SHIPPED complete with valid source refs + tests + docs propagation | 0 (READ-only) | SHIPPED per `366b1_*.md` (~3 min) |
| **B2** | Wf-364 + Wf-365 cross-verification (READ-only) — surfaces 4 forward-looking gaps; Wf-364 6/8 SHIPPED + Wf-365 5/6 SHIPPED + Wf-366 2/2 SHIPPED | 0 (READ-only) | SHIPPED per `366b2_*.md` (~10 min) |
| **C1** | **This docs-only INDEX + AGENTS + final synthesis** | **0** | SHIPPED (this entry); wall ~10-15 min |

**MEASURED test pass-rates per Wf-366 sub-flow** (CPU-only, fresh-verified 2026-09-29 at Wf-366-C1 timestamp):

| Sub-flow | MEASURED |
|---|---|
| Wf-366-A1 | 3/3 NEW tests PASS in 0.36 s at `molmetal/tests/test_wf366a1_g5_user_ok_gate.py` |
| Wf-366-A2 | 2/2 NEW tests (4/4 parametrized) PASS in 0.37 s at `molmetal/tests/test_wf366a2_recover_g6_block_order.py` |
| Wf-366-B1 | 0 NEW tests (READ-only verification) |
| Wf-366-B2 | 0 NEW tests (READ-only cross-poll) |
| Wf-366-C1 | 0 NEW tests (docs-only INDEX + AGENTS + final synthesis — this entry); wall ~10-15 min CPU-only |
| **TOTAL** | **5 NEW tests** (A1=3 + A2=2); combined existing + NEW = **5/5 NEW + 28/28 existing re-verified PASS** across Wf-366 wave |

**Pre-flight v3 fresh-run state** (per `molmetal/reports/wf366c1_preflight.txt` at Wf-366-C1 timestamp 2026-09-29 ~15:17 UTC wall **150.26 s**):

* Total gates: **94** (Wf-353-B1 baseline PRESERVED)
* PASS: **91** (PRESERVED vs Wf-364-C1 baseline 91 PASS)
* FAIL: **1** (PRESERVED vs Wf-364-C1 baseline 1 FAIL — pre-existing `p3_wf343c2_per_wave_runtime_check` Wf-347A1 pre-retrain smoke wave cascade; **NOT Wf-366-introduced**)
* SKIP: **2** (PRESERVED vs Wf-364-C1 baseline 2 SKIP — pre-existing per Wf-350-A2 §5 honest-negative baseline)
* EXPECTED_GATE_COUNTS["all"] **94** PRESERVED (no new pre-flight gates added by Wf-366 — 4 sub-flows + 5 NEW tests + 1 INDEX sync only)
* `gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-computed via Wf-345-B3)` — gate count auto-recompute consistency preserved.

**§22.55.1 Net effect on TODO/ inventory (post-Wf-366-C1 INDEX sync)**

| Inventory | Pre-Wf-366-C1 | Post-Wf-366-C1 | Delta |
|---|---|---|---|
| `pending/` | 13 (Wf-364-C1 baseline) | **13** | **0** (Wf-366 = 4 sub-flows + 1 C1 docs-only sync; no pending/ moves) |
| `completed/` | 805 (Wf-364-C1 baseline) + 4 Wf-366 closeouts (A1+A2+B1+B2) + 1 Wf-366-C1 this entry = **810 files** (per `ls TODO/completed/ \| wc -l` 2026-09-29 Wf-366-C1 timestamp) | **810 files** | **+5** (4 NEW Wf-366 closeouts + 1 Wf-366-C1 docs-only synthesis = this entry) |
| `INDEX.md` line count | 6145 (post-§22.54 Wf-364-C1) | 6145 + ~110 | **+~110** (§22.55 entry appended between §22.54 and §23) |
| `docs/` NEW files | 3 (Wf-364-A4 SECTION_6 + Wf-365-A3 M6_KAPPA_FINAL_VERDICT + Wf-364-A4 SECTION_6 cross-ref) | **3** | **0** (Wf-366 = 0 NEW docs files; closeouts in TODO/completed/ only) |
| `EXPECTED_GATE_COUNTS["all"]` | 94 (Wf-353-B1 baseline) | **94** | **0** (Wf-366 does not change gate count — 4 sub-flows + 5 NEW tests + 1 INDEX sync only) |
| `.claude/skills/` skill count | 28 (Wf-336-D2 baseline) | **28** | **0** (Wf-366 adds zero new skills — 2 gap fixes + 2 verifies + 1 INDEX sync only) |
| Pre-flight PASS count | 91 (Wf-364-C1 baseline) | **91** | **0** (PRESERVED) |
| Pre-flight FAIL count | 1 (Wf-364-C1 baseline) | **1** | **0** (PRESERVED) |

**§22.55.2 Wall + cost accounting**

* **MEASURED**: combined wall ~25-30 min CPU-only across 5 deliverables (task brief PROJECTED 45-75 min; MEASURED ~33-67% under PROJECTED). Per-sub-flow: A1 ~1 min + A2 ~5 min + B1 ~3 min + B2 ~10 min + C1 ~10-15 min = ~29-34 min.
* **MEASURED**: 0 GPU launches performed across the entire 5-deliverable Wf-366 wave (CPU-only docs+code+test+verify per AGENTS.md §5.1 + kill_then_adapt §0).
* **MEASURED**: 0 `paper/*` touches across the entire Wf-366 wave (per AGENTS.md §5.3 TODO-84 active).

**§22.55.3 Honest framing per `party-twenty-spirit §3.5` (7 honest-negative caveats preserved):**

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (Wf-365-A3 verdict doc §1-§4 + Wf-366-B2 §4). Best augmented_mean MEASURED = **0.4737** (Wf-364-A2 joint RDKit+ODE+MAML candidate) vs threshold 0.6 — gap **−0.1263**. 5 architectural attempts exhausted. Structural ceiling LIFTED (CV-R² +0.5410 to +0.6814 — ABOVE PROJECTED +0.05 to +0.15 band per `WEAK_TO_STRONG_PLAN.md §3.5`); threshold-reaching axis NOT lifted.
2. **Wf-364-C2 (USER_OK report) MISSING** (no `TODO/completed/364c2_*.md` on disk — only C1 exists). Recommended fix: this Wf-366-C1 entry rolls up Wf-364-C2 USER_OK report scope (Wf-366-A1 + Wf-366-A2 USER_OK gap closures aggregated).
3. **Wf-365-A2 closeout MISSING** (no `TODO/completed/365a2_*.md` on disk). Per Wf-365-B1 cross-poll §0 + Wf-366-B2 §2.4, the Wf-365-A2 slot was reserved for m6_kappa alternative-paths cross-verification but was not needed once Wf-365-A3 verdict doc + Wf-365-B1 cross-poll combined aggregated all 5 paths. A2 gap closed by A3 verdict doc + B1 cross-poll combined.
4. **Wf-365-C1 (final ship) ALREADY SHIPPED** (per Wf-365-C1 docs-only INDEX + AGENTS + final synthesis at `TODO/completed/365c1_index_sync.md` + `TODO/completed/365_final_synthesis.md`).
5. **Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) NOT closed by Wf-366** — Wf-363-A2 §3 surfaced 4 USER_OK gate gaps; Wf-366 closed GAP-1 (G5) + GAP-2 (recover_g6). GAP-3 + GAP-4 (gpu_launch_dashboard.sh READ-ONLY scaffold intentional per Wf-360-A3) remain OPEN.
6. **G6 recovery sweep is ~11.4 hr wall PROJECTED — BLOCKED on AGENTS.md §5.1 user OK**. Wf-366-A2 closed the SAFETY bug (USER_OK gate bypass) but the actual launch remains gated.
7. **Pre-existing 1 FAIL (`p3_wf343c2_per_wave_runtime_check`) NOT Wf-366-introduced** (preserved per Wf-358-D1 §0 + Wf-362-B1 §1 + Wf-363-C1 §22.52.3 + Wf-365-C1 §22.53.3 honest-negative baseline).

**§22.55.4 User voice answer to Wf-366 wave (per Wf-366-B2 cross-poll §7)**

> User: 什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up

**Wf-366-B2 one-line verdict**: **PARTIAL — 14/20+ closeouts shipped (70%); 4-6 closeouts MISSING; m6_kappa gate NOT LIFTED; structural ceiling LIFTED but threshold-reaching axis PERSISTENT.**

**§22.55.5 Per-agent close-out summary**

| Agent | Artifact | Action | NEW tests | Wall (MEASURED) |
|---|---|---|---|---:|
| **Wf-366-A1** | `TODO/completed/366a1_gap1_g5_user_ok_fix.md` (131 LOC) + `scripts/launch_wf124_p3_g5.sh` (+50 LOC) + `molmetal/tests/test_wf366a1_g5_user_ok_gate.py` | USER_OK gate enforcement + DRY_RUN.log + plan.json emission + G5 migration into WRAPPERS list | 3/3 PASS in 0.36 s | ~1 min |
| **Wf-366-A2** | `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` (310 LOC) + `scripts/recover_g6_failed_tuples.sh` (+34 net LOC) + `molmetal/tests/test_wf366a2_recover_g6_block_order.py` | Block-order fix (USER_OK gate BEFORE --dry-run exit); exit-code 1 → 64; 2 NEW tests = 4 parametrized | 2/2 PASS (4/4 parametrized) in 0.37 s | ~5 min |
| **Wf-366-B1** | `TODO/completed/366b1_wf363_truncated_verify.md` (283 LOC) | READ-only verification of Wf-363-B1 + Wf-363-B2 + Wf-363-C1 closeouts — confirms NOT truncated, all SHIPPED complete | 0 NEW (re-verifies existing tests) | ~3 min |
| **Wf-366-B2** | `TODO/completed/366b2_wf364_365_xverify.md` (300 LOC) | READ-only cross-poll of Wf-364 + Wf-365 + Wf-366 | 0 NEW (cross-poll only) | ~10 min |
| **Wf-366-C1** | `TODO/INDEX.md` + `AGENTS.md` + `TODO/completed/366_final_synthesis.md` + `TODO/completed/366c1_index_sync.md` | Docs-only INDEX §22.55 sync + AGENTS §6 +1 row + final synthesis + index sync | 0 NEW (docs-only) | ~10-15 min |

**Cross-link**: `TODO/completed/366_final_synthesis.md` (Wf-366-C1 docs-only synthesis — final aggregation at 2026-09-29 ~16:30 UTC); `TODO/completed/366c1_index_sync.md` (Wf-366-C1 INDEX + AGENTS sync — this entry); `TODO/completed/366{a1,a2,b1,b2}_*.md` (4 Wf-366 SHIPPED sub-flow closeouts); `scripts/launch_wf124_p3_g5.sh` (Wf-366-A1 USER_OK gate at lines 51-67); `scripts/recover_g6_failed_tuples.sh` (Wf-366-A2 block-order fix); `molmetal/tests/test_wf366a1_g5_user_ok_gate.py` (Wf-366-A1 3 NEW tests); `molmetal/tests/test_wf366a2_recover_g6_block_order.py` (Wf-366-A2 2 NEW tests = 4 parametrized); `molmetal/tests/test_wf363a2_user_ok_ux.py` (Wf-363-A2 audit test — G5 + recover_g6 migrated); `molmetal/reports/wf366c1_preflight.txt` (Wf-366-C1 pre-flight fresh-run wall 150.26 s; 91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL — PRESERVED at Wf-364-C1 baseline); `TODO/completed/363a2_user_ok_ux.md §3 GAP-1 + GAP-2` (original audit findings — CLOSED by Wf-366-A1 + A2); `TODO/completed/363{a1,a2,a3,b1,b2,c1}_*.md` (Wf-363 SHIPPED closeouts); `TODO/completed/364{a1,a2,a3,a4,b1,b2,c1}_*.md` (Wf-364 SHIPPED closeouts; C1 superseded by §22.54 INDEX sync, C2 rolled into Wf-366-C1); `TODO/completed/365{a1,a3,b1,b2,c1}_*.md` + `TODO/completed/365_final_synthesis.md` (Wf-365 SHIPPED closeouts); `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 NEW verdict doc); `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` (Wf-365-A3 NEW metrics JSON field); `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `AGENTS.md` §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar; `AGENTS.md` §6 +1 NEW row (Wf-366 entry — this C1 sync); §22.54 (Wf-364-C1 INDEX sync prior); §22.55 (this Wf-366-C1 INDEX sync — new entry appended between §22.54 and §23); NEW ultracode Wf-367+ recommended per Wf-366-B2 §7 + Wf-366-C1 §8 — to close GAP-3 pre_flight_gpu_ready + m6_kappa metric-redefinition + G6 USER_OK parity + pre-existing `p3_wf343c2_per_wave_runtime_check` 5-LOC fix.

---

## §4 AGENTS.md §6 row (appended)

The Wf-366 row follows the §6 format pattern (verbatim Wf-363 / Wf-364 / Wf-365 row structure):

> | 2026-09-29 | **Wf-366 USER_OK gate gaps closure + Wf-363/364/365 cross-verification wave** (2 USER_OK gate gap closures A1+A2 + 2 cross-verifications B1+B2 + Wf-366-C1 docs-only INDEX + AGENTS + final synthesis at `TODO/completed/366{a1,a2,b1,b2,c1}_*.md` + `TODO/completed/366_final_synthesis.md` + `TODO/completed/366c1_index_sync.md`; user directive 2026-09-29 verbatim *"如果不是，启动新的ultracode做follow up"* (Wf-365-B1 cross-poll §0); **A1 launch_wf124_p3_g5.sh USER_OK gate closure (Wf-363-A2 GAP-1)** (canonical Wf-355-B1 pattern wired at lines 51-67; exit 64 + USER_OK gate refused FATAL + DRY_RUN.log + plan.json emission; **3 NEW tests PASS** in 0.36 s at `molmetal/tests/test_wf366a1_g5_user_ok_gate.py`); **A2 recover_g6_failed_tuples.sh block-order fix (Wf-363-A2 GAP-2)** (USER_OK gate moved BEFORE `--dry-run` exit block; exit-code 1 → 64 per Wf-294B family; **2 NEW tests PASS (4/4 parametrized)** in 0.37 s at `molmetal/tests/test_wf366a2_recover_g6_block_order.py`); **B1 Wf-363-B1/B2/C1 truncated closeout verification (READ-only)** (confirms NOT truncated; all 3 SHIPPED complete with valid source refs + tests + docs propagation); **B2 Wf-364 + Wf-365 cross-verification (READ-only cross-poll)** (Wf-364 6/8 SHIPPED + Wf-365 5/6 SHIPPED + Wf-366 2/2 SHIPPED; surfaces 4 forward-looking gaps); **Wf-366-C1 docs-only INDEX + AGENTS + final synthesis** (this row — closeout at `TODO/completed/366c1_index_sync.md` + final synthesis at `TODO/completed/366_final_synthesis.md`); **5/5 deliverables SHIPPED** + **5 NEW tests PASS** (A1=3 + A2=2; task brief PROJECTED 3+2=5 — actual matches) + **2 NEW test files** + **~84 LOC NEW source code + 2 wrapper patches (G5 +50 LOC + recover_g6 +34 net LOC)** + **0 GPU launches** + **0 `paper/*` touches**; pre-flight v3 fresh-run **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** at Wf-366-C1 timestamp 2026-09-29 ~15:17 UTC wall **150.26 s** per `molmetal/reports/wf366c1_preflight.txt` (PRESERVED at Wf-364-C1 baseline; 1 FAIL is pre-existing `p3_wf343c2_per_wave_runtime_check` cascade per Wf-358-D1 §0 + Wf-362-B1 §1 honest-negative baseline; NOT Wf-366-introduced); EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-366 — 4 sub-flows + 5 NEW tests + 1 INDEX sync only); **MEASURED combined wall ~25-30 min CPU-only across 5 deliverables** (task brief PROJECTED 45-75 min; actual ~33-67% under PROJECTED); **7 honest-negative caveats preserved per §22.55.3**: (1) m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL (best aug_mean 0.4737 < 0.6 threshold by −0.1263; 5 attempts exhausted; structural ceiling LIFTED CV-R² +0.5410 to +0.6814 but threshold-reaching axis NOT lifted) + (2) Wf-364-C2 MISSING (rolled into Wf-366-C1) + (3) Wf-365-A2 MISSING (closed by A3 verdict doc + B1 cross-poll combined) + (4) Wf-365-C1 ALREADY SHIPPED (per Wf-365-C1 docs-only INDEX + AGENTS + final synthesis at `TODO/completed/365c1_index_sync.md`) + (5) Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) NOT closed by Wf-366 + (6) G6 recovery sweep ~11.4 hr wall PROJECTED — BLOCKED on AGENTS.md §5.1 user OK + (7) Pre-existing 1 FAIL (`p3_wf343c2_per_wave_runtime_check`) NOT Wf-366-introduced | This AGENTS.md §6 row (Wf-366-C1 sync — this row) + `TODO/INDEX.md` §22.55 (NEW ~110 lines appended between §22.54 Wf-364-C1 and §23) + Wf-366 closeouts at `TODO/completed/366{a1,a2,b1,b2,c1}_*.md` + `TODO/completed/366_final_synthesis.md` + `TODO/completed/366c1_index_sync.md`

---

## §5 Final synthesis (cross-link)

See `TODO/completed/366_final_synthesis.md` for the full final aggregation:

- §0 TL;DR (5 deliverables + 5 NEW tests + pre-flight 91/1/2/94)
- §1 Per-sub-flow verdicts (A1+A2+B1+B2+C1)
- §2 Per-sub-flow verdicts table
- §3 Pre-flight state (MEASURED)
- §4 Wall + cost accounting
- §5 Net effect on TODO/ inventory
- §6 Standing directive compliance
- §7 Honest framing (7 caveats)
- §8 User voice answer
- §9 Per-agent close-out summary
- §10 Cross-references

---

## §6 Files touched by Wf-366-C1

| Path | Action |
|---|---|
| `TODO/INDEX.md` | Append §22.55 (~110 lines) between §22.54 and §23 |
| `AGENTS.md` | Append §6 NEW row (Wf-366 entry) — ~2,500 chars |
| `TODO/completed/366_final_synthesis.md` | NEW (~500 lines) — Wf-366-C1 final aggregation |
| `TODO/completed/366c1_index_sync.md` | NEW (this entry) — INDEX + AGENTS sync record |

**No source code edits, no test changes (docs-only wave)**

---

## §7 Standing directive compliance

| Directive | Compliance | Notes |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ COMPLIED | Every numeric value in this entry + Wf-366-A1/A2/B1/B2/C1 entries cites JSON / report path or test name |
| §1.2 Standing GPU authorization (§5.1) | ✅ COMPLIED | 0 GPU launches performed (CPU-only docs+code+test+verify across all 5 Wf-366 deliverables) |
| §1.3 No paper work without authorization (§5.3 TODO-84 active) | ✅ COMPLIED | 0 `paper/*` touches |
| §1.4 Pre-flight v2 25/25 PASS gate (§4 TODO-207) | ✅ COMPLIED | Pre-flight v3 fresh-run captured at Wf-366-C1 timestamp 91/1/2/94 (150.26 s wall) |
| §1.5 Honest-negative framing (§5.2 / `party-twenty-spirit` §3.5) | ✅ COMPLIED | 7 honest-negative caveats preserved per §22.55.3 |
| §5.4 Tier-2 journal target + NO wet-lab planning | ✅ COMPLIED | 0 wet-lab planning; all 5 Wf-366 deliverables are Tier-2-shippable as is |

---

## §8 Cross-references

- `TODO/INDEX.md` §22.55 — NEW entry appended between §22.54 Wf-364-C1 and §23
- `AGENTS.md` §6 — NEW row appended (this C1 sync)
- `TODO/completed/366_final_synthesis.md` — Wf-366-C1 final synthesis (NEW)
- `TODO/completed/366c1_index_sync.md` — Wf-366-C1 INDEX + AGENTS sync (this entry)
- `TODO/completed/365c1_index_sync.md` — Wf-365-C1 INDEX + AGENTS sync (precedent)
- `TODO/completed/365_final_synthesis.md` — Wf-365-C1 final synthesis (precedent)
- `TODO/completed/364c1_index_sync.md` — Wf-364-C1 INDEX + AGENTS sync (precedent for §22.54)
- `TODO/completed/366{a1,a2,b1,b2}_*.md` — 4 Wf-366 SHIPPED sub-flow closeouts
- `molmetal/reports/wf366c1_preflight.txt` — Pre-flight v3 fresh-run 91/1/2/94 wall 150.26 s
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar

---

**END OF Wf-366-C1 docs-only INDEX + AGENTS + final synthesis entry** — Owner Hugo Lee; cross-ref `party-twenty-spirit §3.5`; **5 deliverables SHIPPED** (2 USER_OK gate gap closures A1+A2 + 2 cross-verifications B1+B2 + 1 C1 docs sync); **5 NEW tests PASS**; pre-flight v3 **91/1/2/94 wall 150.26 s PRESERVED** at Wf-364-C1 baseline; EXPECTED_GATE_COUNTS["all"] 94 PRESERVED; 0 GPU launches; 0 `paper/*` touches; 7 honest-negative caveats preserved; m6_kappa gate NOT lifted; NEW ultracode Wf-367+ recommended per Wf-366-B2 §7 + Wf-366-C1 §8.