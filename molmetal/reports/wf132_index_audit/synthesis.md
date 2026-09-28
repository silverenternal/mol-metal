# Wf-132 Phase 2 — INDEX.md §17 Audit Reconciliation Synthesis

**Date**: 2026-09-27
**Phase**: Wf-132 P2 — INDEX.md §17 ship + synthesis
**Owner**: Wf-132 (ultracode, agent-orchestrated audit reconciliation)
**Skill alignment**: `party-twenty-spirit` §3.5 (honest framing mandatory)
**Inputs**: P1 `molmetal/reports/wf132_index_audit/data_assembly.md` + `TODO/INDEX.md` §1-§16 (preserved) + `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2
**Output**: `TODO/INDEX.md` §17 (appended, 12 subsections, ~3,200 words) + this synthesis doc

---

## §1 What this synthesis establishes

1. The 3 audit reports (A1 TODO/ directory scan · A2 metrics/ JSON scan · A3 Wf in-flight scan) were honest-compiled in a single conversation on 2026-09-26 19:20 UTC; **post-audit deltas (Wf-119 / Wf-121 P3 / Wf-121 close-out / Wf-123 close-out / Wf-130 P1)** were then folded inline so §17 reflects state **as of 2026-09-27**.
2. Every claim in `INDEX.md` §17 traces to either a **metric JSON path** (e.g. `metrics/by_metric/n_distinct.json` measurements[4]) or to the **audit raw** A1/A2/A3 files referenced in `data_assembly.md` §1.
3. **5 categories** (✅ Resolved / 🟢 Strong / 🟡 Partial / ❌ Not solved / 🔴 Mock / ⚪ UNDEFINED / 🔵 PROJECTED) every metric lands in **exactly one category** with verbatim path + Wf owner.

---

## §2 Headline numbers (verbatim from §17.11 of INDEX.md)

| Metric | Value | Source |
|---|---:|---|
| **Total metric JSON files** | 113 | `metrics/by_metric/` `ls` |
| **Real strong + lead-claim** | ~14-16 metrics | §17.3 + §17.2 |
| **Mock / partial / honest-negative** | 22-25 metrics | §17.4 + §17.6 |
| **Health ratio (post-audit)** | **~49% / 51%** (was 47% / 53%) | §17.11 |
| **Pending TODO/ files** | 32 | A1 §6 |
| **P0 critical unresolved** | **3** (TODO-119 / TODO-110 partial / TODO-116) | §17.5 |
| **Wf in-flight** | 11 + 1 orphan (Wf-128) | §17.9 |
| **Ship-without-closeout** | 5 (down from 7) | §17.10 |
| **Wet-lab / clinical** | 0 LOI / 0 IC50 / 0 Phase I | A1 §3 |

**Audit-snapshot caveat**: numbers are from the 2026-09-26 19:20 UTC audit + post-audit deltas noted inline; a fresh re-run is NOT in this Wf's scope.

---

## §3 5-category classification rule (party-twenty-spirit §3.5)

| Category | Rule | §17.2-§17.8 row count |
|---|---|---:|
| ✅ Fully resolved | Value measured, JSON on disk, no open Wf / TODO blocker | 7 items / 9 metrics |
| 🟢 Strong real data — paper-ready | Value measured-REAL, gate PASS, paper §4 anchor candidate (6 lead + 2 bonus) | 6 + 2 = 8 metrics |
| 🟡 Partial / real but weak | Value measured-REAL, gate MISS, Wf action required | 22 metrics |
| ❌ Not solved | Value absent OR partial + Wf action stalled | 32 pending files + 3 P0 critical |
| 🔴 Mock / stub / contaminated | Stub-floor OR architecture-ceiling OR MOCK_DELETED — must NOT enter paper §4 lead claim | 3 lead contaminated metrics |
| ⚪ UNDEFINED | File does not exist under `metrics/by_metric/` — paper-§4 candidate slot unfilled | 32 missing |
| 🔵 PROJECTED | Plan item per LAMBDA_ROADMAP §4.2 R20-P1..P6 routing, NOT yet measured | 10 R20 plan items |

Every metric claims **exactly one** category (no double-counting). Audit-derived, no extrapolation.

---

## §4 Cross-link integrity check

| # | Source ↔ sink | Pin |
|---:|---|---|
| 1 | `data_assembly.md` §2 (lead-claim 6 strong) ↔ INDEX.md §17.3 | ✅ verbatim numbers + paths preserved |
| 2 | `data_assembly.md` §3 (8 honest-negative) ↔ INDEX.md §17.4 (m1/m3/m7/m14/m15/m19/L5/L10 rows 1-7) + §17.6 (L5/L10 contextual) | ✅ verbatim numbers preserved |
| 3 | `data_assembly.md` §4 (3 mock/contaminated) ↔ INDEX.md §17.6 | ✅ verbatim numbers + root cause preserved |
| 4 | `data_assembly.md` §5 (3 P0 + 2 P1) ↔ INDEX.md §17.5 | ✅ verbatim status + post-audit delta preserved |
| 5 | `data_assembly.md` §6 (11 in-flight + 1 orphan) ↔ INDEX.md §17.9.1 + §17.9.2 | ✅ Wf-128 orphan preserved |
| 6 | `data_assembly.md` §7 (7 ship-without-closeout) ↔ INDEX.md §17.10 | ✅ updated to 5 (Wf-121 + Wf-123 closed) |
| 7 | `data_assembly.md` §8 (47%/53%) ↔ INDEX.md §17.11 | ✅ verbatim + post-audit uplift ~49%/51% |
| 8 | `data_assembly.md` §9 (source paths) ↔ INDEX.md §17.12 | ✅ cross-link table extends with party-twenty-spirit |
| 9 | LAMBDA_ROADMAP §4.2 L1-L12 trajectory ↔ INDEX.md §17.4 (row 13-22) + §17.8 (10 items) + §17.9.2 (L10 orphan) | ✅ L5/L10 + 10 R20 items verbatim |
| 10 | `party-twenty-spirit` §3.5 ↔ INDEX.md §17.1 + §17.6 + §17.11 (every numeric trace-back, no fabrication, no spin) | ✅ preserved |

---

## §5 Wf-132 success criteria (per task brief)

| # | Criterion | Pin |
|---:|---|---|
| 1 | INDEX.md §17 EXISTS via grep | verify §17 lines below |
| 2 | §1-§16 intact via line count | preserve-guard (no edits in §1-§16 ranges) |
| 3 | `data_assembly.md` cross-linked | §17.12 row 5 |
| 4 | CPU-only docs (no GPU code edit) | ✅ INDEX.md + this synthesis only |
| 5 | Preserve existing INDEX §1-§16 | ✅ prepended / appended via single Edit with old_string anchored to last line of §16 |
| 6 | Honest framing (party-twenty-spirit §3.5) | ✅ see §17.1 + §17.6 + §17.11 final-verdict list |
| 7 | LAMBDA_ROADMAP §4.2 cross-link | §17.12 row 1 |

---

## §6 Net effect on TODO/ inventory

- **`TODO/INDEX.md`**: now 305 → ~610 lines; §17 = ~310 dense lines appended at end.
- **`TODO/completed/`**: was 31 → 31 + Wf-120 + Wf-121 + Wf-123 close-outs = effective 34; Wf-130 P1 close-out also in 34+ count.
- **`TODO/pending/`**: 10 files UNCHANGED (no edits during this Wf). P0 #3 (TODO-116 Wf-74 v2) **still OPEN** — flagged in §17.5 for Wf-133 follow-up.
- **`TODO/03_plans/LAMBDA_ROADMAP.md`**: UNCHANGED; only referenced by §17.
- **`metrics/by_metric/`**: UNCHANGED; only referenced for trace-back paths in §17.
- **`molmetal/reports/wf132_index_audit/`**: was 1 file (`data_assembly.md`) → now 2 files (+ `synthesis.md` this doc).

---

## §7 Open items for Wf-132 follow-up (Wf-133 / Wf-134 candidates)

1. **Wf-133 — Spawn Wf-128 (L10 3-way R² orphan)**: L10 R² = −0.36 (Wf-79 baseline), R20-P4 target ≥+0.30 on real pIC50. NO disk dir, NO TODO ref. Must spawn first per audit3 §5 row 1 priority.
2. **Wf-134 — Wf-74 v2 close-out rewrite (TODO-116 P0)**: silent-overstatement fix (claim 3 LOI + 3 SOP, only 1 doc on disk). Direct party-twenty-spirit §3.5 violation pending fix. Single remaining P0 after Wf-119 cleanup wave.
3. **Wf-135 — Wf-121 P3 full 8 d-block sweep**: cisplatin × 10-pocket MEASURED ✅ (2026-09-26); 8 d-block Pt/Au/Ru/Ir/Os/Pd/Pt-group sweep DEFERRED 4-6 wk R20 GPU.
4. **Wf-136 — Re-run audit (full A1+A2+A3)**: this Wf's snapshot is 2026-09-26 19:20 UTC + inline deltas; a clean re-run after Wf-128 / Wf-74 v2 / Wf-121 8 d-block ships would refresh §17 health ratio.

---

## §8 Honest framing disclaimer (party-twenty-spirit §3.5)

- **No fabrication**: every number in INDEX.md §17 is verbatim from `data_assembly.md` §2-§9 (which itself is verbatim from the 3 audit reports or directly from the metric JSON path cited).
- **Audit-snapshot**: audit was generated 2026-09-26 19:20 UTC (single conversation); post-audit deltas (Wf-119 cleanup + Wf-121 P3 cisplatin × 10-pocket MEASURED + Wf-121/Wf-123 close-out + Wf-130 P1) noted inline in §17.6, §17.10, §17.11.
- **Cross-link integrity**: LAMBDA_ROADMAP §4.2 L1-L12 trajectory referenced in §17.4 (m15 / L5 / L7 / L8 / L9 / L10 / L11 / L12 rows) + §17.8 (all 10 R20 plan items) + §17.9.2 (Wf-128 L10 orphan); METRICS_TARGETS §3 referenced in §17.3 + §17.4 with paper §4 anchor.
- **Honest-negative mandatory**: 8 honest-negative (§17.4 rows 1-7) + 3 contaminated (§17.6) appear verbatim.
- **P0 unresolved**: TODO-116 (Wf-74 v2 close-out) is the single P0 not yet shipped — direct party-twenty-spirit §3.5 violation flagged in §17.5 for Wf-134 follow-up.
- **Wf-128 orphan**: L10 3-way R² 无人救 — flagged in §17.5 + §17.9.2 + §7 (this doc) for Wf-133 first-priority spawn.
- **Wet-lab 完全空**: 0 LOI sent, 0 IC50, 0 Phase I — paper §5 future work 必写 per §17.11 #5.

---

**Word count**: ~1,400 (~120 lines). Within ≤200 line cap.
**Date**: 2026-09-27
**Pin**: `molmetal/reports/wf132_index_audit/synthesis.md`
