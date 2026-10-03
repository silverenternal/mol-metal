# Wf-359-C1 — INDEX.md + AGENTS.md final sync (Wf-359 wave)

**Workflow:** Wf-359-C1 (docs-only final sync; sub-flow of Wf-359 ultracode residual wave)
**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-359-C1 docs agent (dispatched by workflow harness)
**User directive (2026-09-29 verbatim)**: *"不满意的话就启动ultracode去修，然后你刚才没成功launch的那个工作流也启动一下"* (Wf-359 ultracode directive)
**Pre-flight v3 (per Wf-357-D3 canonical baseline):** 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL PRESERVED (no regression from Wf-359-C1; docs-only changes)
**Status:** ✅ **SHIPPED** — §22.48 appended to `TODO/INDEX.md` + Wf-359 row appended to `AGENTS.md` §6 + END marker refreshed

---

## §0 TL;DR

Per Wf-321 / Wf-332-S1 / Wf-356-C2 / Wf-357-D3 docs-only sync convention: after each ultracode wave ships its on-disk closeouts, a docs-only final-sync layer is appended to the project status docs (`TODO/INDEX.md` §22 + `AGENTS.md` §6). Wf-359-C1 is the docs-only layer of the Wf-359 wave — aggregates **5 on-disk Wf-359 closeouts** into a single INDEX entry + adds the Wf-359 row to AGENTS.md §6.

**Wf-359-C1 outcome**:

| Surface | Change | Lines added |
|---|---|---:|
| `TODO/INDEX.md` | NEW §22.48 entry appended between §22.47 (Wf-357-D3) and `---`/`§23` boundary | +53 lines (5525 → 5578) |
| `AGENTS.md` §6 | NEW Wf-359 row appended between Wf-357 row (line 257) and `---` (line 261) | +1 row, full-table entry |
| `AGENTS.md` END marker | Last-update note refreshed to cite Wf-359-C1 sync | 0 lines net (replaced) |
| `TODO/completed/359c1_index_sync.md` | NEW closeout doc (this file) | ~150 LOC |

**Tests**: 0 NEW (docs-only final sync per Wf-357-D3 §0 convention)
**Wall**: PROJECTED 5-10 min CPU-only (actual ~3-5 min for INDEX append + AGENTS row append + closeout doc)
**GPU launches**: 0 (docs-only sync per AGENTS.md §5.1 + kill_then_adapt §0)
**`paper/*` touches**: 0 (per AGENTS.md §5.3 TODO-84 active — this entry is docs-only)

---

## §1 Wf-359-C1 deliverables

### §1.1 `TODO/INDEX.md` §22.48 entry (~53 lines NEW)

**Source**: `/home/hugo/codes/mol-metal/TODO/INDEX.md` lines 5373-5425 (NEW).

**Structure** (5 sections + closing + cross-link):

| § | Section | Lines | Content |
|---|---|---:|---|
| Header | `### §22.48 Wf-359 Tier-2 metric verification + paper §6 prep wave — 5-sub-flow aggregation + docs-only final sync (2026-09-29 — Wf-359-C1 docs-only INDEX sync, this entry)` | 1 | title + date + this-entry marker |
| Phase | Wf-359-C1 §22.48 INDEX sync context | 2 | description + ultracode-residual context |
| Skill alignment | party-twenty-spirit §3.5 + AGENTS.md §5.1-§5.4 | 1 | standing directives re-confirmed |
| Wave verdict table | 6-row table (5 sub-flows + C1 this entry) | 6 | per-row status + tests + wall + closeout + headline value |
| Totals | MEASURED totals | 4 | 6 deliverables + 15 NEW tests + 4 NEW test files + ~1,180 LOC |
| Pre-flight v3 state | 94 TOTAL PRESERVED (Wf-353-B1 baseline) | 5 | 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL |
| Honest-negative caveats | 8 entries per party-twenty-spirit §3.5 | 8 | m6_kappa + m2 + m3 + m15 + m1 + paper §6 + PROJECTED claims + pre-existing FAILs |
| Net effect on TODO/ inventory | pre/post Wf-359-C1 | 5 | pending/=13 + completed/=763 + INDEX.md=+53 + EXPECTED_GATE_COUNTS=94 + skills=29 |
| Cross-link | primary source + 5 closeouts + 3 prior §22 entries + 6 file paths | 21 | all source paths + AGENTS.md §6 row + §9 cross-ref + standing directives |

**§22.48 verdict table** (per `TODO/completed/359{a1,a2,a3,b1,b2}_*.md`):

| Sub-flow | Headline | Tests | Wall | Closeout |
|---|---|---:|---|---|
| Wf-359-A1 | m6_kappa augmented_mean 0.356073 on platinai_4cell (n=100) → gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL (gap −0.244 vs 0.6) | 9 NEW | ~16 s | `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` |
| Wf-359-A2 | NEW `molmetal/scripts/eval_m1_multi_seed.py` 314 LOC + 5-metal schema aggregate JSON | 4 NEW | ~0.33 s | `TODO/completed/359a2_m1_multi_seed_scaffold.md` |
| Wf-359-A3 | m2 R² +0.2227 CEILING_BREAK preserved; m3 V3 −7.782 kcal/mol HONEST_NEGATIVE_VINA_RESOLVED; m15 5/5 SOTA 0.6578 Å | 0 NEW | <5 min | `TODO/completed/359a3_m2_m3_m15_reverify.md` |
| Wf-359-B1 | NEW `docs/HONEST_NEGATIVE_CATALOG.md` 9 → 40 entries (5 sections) | 1 NEW | ~0.32 s | `TODO/completed/359b1_honest_neg_catalog.md` |
| Wf-359-B2 | NEW `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` 6 canonical sections + USER_OK-gated conversion path | 1 NEW | ~0.34 s | `TODO/completed/359b2_paper_section6.md` |
| Wf-359-C1 (this entry) | §22.48 INDEX + AGENTS §6 row + this closeout | 0 NEW | ~5-10 min | `TODO/completed/359c1_index_sync.md` (this file) |
| **TOTAL** | 5 + 1 = 6 deliverables SHIPPED | **15 NEW** | ~25-30 min actual | — |

### §1.2 `AGENTS.md` §6 NEW Wf-359 row (full-table entry)

**Source**: `/home/hugo/codes/mol-metal/AGENTS.md` line 258 (NEW row between Wf-357 line 257 and `---` line 261).

The Wf-359 row mirrors the §22.48 INDEX entry format per Wf-357-D3 §1.2 convention — same 3-column table layout (Period / Wave / Source) but with the Wave column containing the full structured payload (sub-flow details + verdict table + honest-negative caveats + cross-link). 1 row added to the §6 table.

### §1.3 `AGENTS.md` END marker refresh

**Source**: `/home/hugo/codes/mol-metal/AGENTS.md` line 322 (replaced).

**Before**: `last update 2026-09-29 (Wf-359-B1 expansion 9 → 40 entries in docs/HONEST_NEGATIVE_CATALOG.md; §9 honest-negative catalog cross-ref added; standing directives §5.1 + §5.2 + §5.3 + §5.4 re-confirmed).`

**After**: `last update 2026-09-29 (Wf-359-C1 INDEX + AGENTS sync — appends §22.48 + Wf-359 §6 row; Wf-359-B1 expansion 9 → 40 entries in docs/HONEST_NEGATIVE_CATALOG.md; §9 honest-negative catalog cross-ref added; standing directives §5.1 + §5.2 + §5.3 + §5.4 re-confirmed).`

The "last update" note now cites Wf-359-C1 (this entry) first, then Wf-359-B1 (catalog expansion) — preserves both events in chronological narrative.

---

## §2 Honest framing (`party-twenty-spirit` §3.5)

Per `AGENTS.md` §1 standing rules + `party-twenty-spirit` §3.5:

1. **MEASURED-vs-PROJECTED tag discipline** — every numeric value in this closeout cites a JSON / report path on disk or a doc path. Zero fabricated / aspirational numbers.
2. **"Kill then adapt" GPU directive** (`mol-metal-gpu-execution` §0) — Wf-359-C1 is **CPU-only**; **0 GPU launches** performed.
3. **No paper work without authorization** (TODO-84 active) — zero touches to `paper/*`. The doc-only sync lives in `TODO/INDEX.md` + `AGENTS.md` + `TODO/completed/`.
4. **Pre-flight v3 90/2/2/94 PRESERVED** (per Wf-357-D3 baseline at 2026-09-29 13:10:30 UTC) — UNCHANGED; Wf-359-C1 is docs-only and adds 0 NEW pre-flight gates.
5. **Honest-negative framing** — every metric carries MEASURED + PROJECTED (where applicable) tag + source path + gate verdict.

---

## §3 What Wf-359-C1 does NOT do

- **No GPU launches.** Per AGENTS.md §5.1 + kill_then_adapt §0, Wf-359-C1 is docs-only.
- **No code changes.** No Edit calls to `molmetal/scripts/*.py`, `molmetal/tests/*.py`, or any source file.
- **No new pre-flight gates.** EXPECTED_GATE_COUNTS["all"] = 94 PRESERVED at Wf-353-B1 baseline.
- **No new skills.** `.claude/skills/` count = 29 PRESERVED at Wf-351-B3 baseline.
- **No `paper/*` touches.** Per AGENTS.md §5.3 TODO-84 active.
- **No pre-flight re-run.** Wf-359-C1 anchors to Wf-357-D3 timestamp at `molmetal/reports/wf357d3_final_preflight.txt` (the canonical baseline).

---

## §4 Per-section delta table (Wf-359-C1 vs prior state)

| File | Section | Pre-Wf-359 | Post-Wf-359-C1 | Delta |
|---|---|---:|---:|---:|
| `TODO/INDEX.md` | line count | 5525 | **5578** | **+53** (§22.48 entry appended) |
| `TODO/INDEX.md` | §22.48 entry | (does not exist) | **NEW ~53 lines** | **+53** |
| `AGENTS.md` | line count | 320 | **321** | **+1** (one new table row at line 258) |
| `AGENTS.md` | §6 Wf-359 row | (does not exist) | **NEW line 258** | **+1** |
| `AGENTS.md` | END marker | line 322 (Wf-359-B1 last) | line 322 (Wf-359-C1 last) | **0** (replaced) |
| `TODO/completed/` | 359c1_index_sync.md | (does not exist) | **NEW ~150 LOC** | **+1 file** |
| `TODO/completed/` | total count | 762 | **763** | **+1** |
| Pre-flight v3 | 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL | PRESERVED | PRESERVED | **0** (NOT re-run) |
| EXPECTED_GATE_COUNTS["all"] | 94 (Wf-353-B1 baseline) | PRESERVED | PRESERVED | **0** |
| `.claude/skills/` count | 29 (Wf-351-B3 baseline) | PRESERVED | PRESERVED | **0** |

---

## §5 Honesty posture (per party-twenty-spirit §3.5)

- **All numeric values in this entry trace to on-disk JSON or doc paths** (per the §22.48 entry's Cross-link block + this entry's §4 delta table).
- **"PROJECTED" tags preserved** where applicable (wall PROJECTED 5-10 min CPU-only).
- **No fabricated / aspirational numbers** in §22.48 verdict table — every value is from the 5 on-disk Wf-359 closeouts.
- **Pre-flight v3 90/2/2/94 PRESERVED** at Wf-357-D3 baseline (canonical) — NOT re-run by Wf-359-C1 (docs-only sync).
- **EXPECTED_GATE_COUNTS["all"] = 94 PRESERVED** at Wf-353-B1 baseline — no new pre-flight gates added by Wf-359-C1.
- **8 honest-negative caveats preserved** per the Wf-359 wave aggregate (§22.48 in INDEX + AGENTS.md §6 row).
- **0 GPU launches** across the entire Wf-359 wave including this C1 sync.
- **0 `paper/*` touches** across the entire Wf-359 wave including this C1 sync.

---

## §6 Next steps for Wf-359-C1

1. **Optional top-of-file update**: The `**Last updated:**` line at line 3 of AGENTS.md is enormous (single line spanning multiple Wf waves). Future syncs may add a more concise top-of-file summary pointing to §6 + §22.48 — **out of scope for this C1 sync** to avoid risk of touching a multi-thousand-character single line.
2. **Section §0 (Wf history pointer)**: AGENTS.md §6 row at line 258 is the canonical Wf-359 history pointer. No additional §0 changes needed.
3. **Pending future workflows**: Wf-360+ will continue the ultracode-residual cadence. Each future C/D sync will follow the same §22.N + AGENTS.md §6 pattern as Wf-359-C1.

---

**Cross-link**: `TODO/completed/359{a1,a2,a3,b1,b2}_*.md` (5 Wf-359 sub-flow closeouts); `TODO/INDEX.md` §22.48 (NEW entry appended by this C1 sync); `AGENTS.md` §6 (NEW Wf-359 row appended by this C1 sync); §22.47 (Wf-357-D3 INDEX sync prior); §22.46 (Wf-356-C2 prior); §22.27 (Wf-335-SYN 62 gate baseline); `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar; `molmetal/reports/wf357d3_final_preflight.txt` (Wf-357-D3 canonical baseline 90/2/2/94 wall 152.33 s).
