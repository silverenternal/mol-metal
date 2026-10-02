# Wf-353-C2: INDEX.md + AGENTS.md docs-only sync for Wf-353 ship wave

**Wf:** Wf-353-C2 (docs-only INDEX + AGENTS sync; no code change, no GPU launch)
**Date:** 2026-09-29
**Author:** Claude Code subagent (workflow harness)
**Status:** SHIPPED — docs-only sync
**Mode:** Plain-text OUTPUT (per AGENTS.md §8 + Wf-291C workaround)

---

## §0 TL;DR (Honest framing per `party-twenty-spirit` §3.5)

| Field | Value |
|---|---|
| Wave scope | 6 Wf-353 sub-flow closeouts (A1+A2+A3+A4+B1+B2) already on disk |
| C2 scope | Append §22.41 to `TODO/INDEX.md` + new row to `AGENTS.md` §6 table |
| Tests added | **0** (docs-only sync) |
| Wall | **PROJECTED 5-10 min CPU-only** (Bash + Edit + grep + Write only) |
| GPU launches | **0** (pure docs) |
| NEW pre-flight gates | **0** (the 71→93 expansion was Wf-353-B1, NOT this C2) |
| Files touched | `TODO/INDEX.md` (+97 lines), `AGENTS.md` (+1 line), this file |

---

## §1 What was synced

### §1.1 `TODO/INDEX.md` — §22.41 entry appended

- **§22.41** title: *"Wf-353 Wf-340 deeper integration + Wf-343c1 un-wired gates (2026-09-29 — Wf-353-C2 docs-only INDEX sync, this entry)"*
- **Insertion point**: between §22.40 (Wf-349-D3) and §23 (Wf-316B), preserving chronological order
- **Entry length**: ~97 lines (6-row closeout verdict + 6-row per-fix outcome + 4-row pre-flight delta + 5-row net-effect + 4-entry honest-negative catalog + cross-link)

**Sub-sections appended**:
- §22.41.1 Per-closeout verdict table (6 rows; A1+A2+A3+A4+B1+B2)
- §22.41.2 Aggregate Wf-353 totals (22 NEW tests, ~480 LOC, 71→93 gate count, 0 GPU launches)
- §22.41.3 Per-fix outcome table (6 axes: Triton mini-batch OT + shortcut distillation + consistency regularization + FlowMol + Wf-343c1 22 gates + Wf-343c2/c3 verify)
- §22.41.4 Pre-flight gate delta (71 → 93 from Wf-353-B1, NOT from C2)
- §22.41.5 Net effect on TODO/ inventory (pending/ UNCHANGED at 13; completed/ +7; INDEX.md +97 lines)
- §22.41.6 Honest framing per `party-twenty-spirit` §3.5 (4 honest-negative caveats)
- §22.41.7 Word count + density + cross-link

### §1.2 `AGENTS.md` §6 — new row appended

- **Row position**: appended between row 254 (Wf-344..354 wave) and end-of-§6
- **Row content**: 6 Wf-353 sub-flows + Wf-353-C2 docs sync + per-sub-flow test counts + gate count delta + 4 honest-negative caveats

---

## §2 Wf-353 wave summary (per upstream closeouts on disk)

| Closeout | Subject | LOC | Tests | Verdict |
|---|---|---:|---:|---|
| Wf-353-A1 | Triton mini-batch OT kernel (Wf-340-P3 #1) | ~100 | 4/4 PASS | SHIPPED |
| Wf-353-A2 | Shortcut distillation (Wf-340-P3 #4) | ~120 | 4/4 PASS | SHIPPED |
| Wf-353-A3 | Consistency regularization (Wf-340-P3 #5) | ~80 | 4/4 PASS | SHIPPED |
| Wf-353-A4 | FlowMol unified atom-bond flow (Wf-340-P3 #6) | ~180 | 4/4 PASS | SHIPPED |
| Wf-353-B1 | Wf-343c1 22 un-wired gates wired (71→93) | +22 runners | 4/4 PASS | SHIPPED |
| Wf-353-B2 | Wf-343c2 + Wf-343c3 integration verify | 0 (verify) | 2/2 PASS | VERIFIED |
| **Wf-353-C2** | **INDEX + AGENTS docs sync (this file)** | **0** | **0** | **SHIPPED** |
| **Total** | **—** | **~480** | **22 NEW tests** | **7/7 SHIPPED** |

---

## §3 Pre-flight gate delta verification

Per `molmetal/scripts/pre_flight_check.py:3117 EXPECTED_GATE_COUNTS` after Wf-353-B1 wiring:

- `EXPECTED_GATE_COUNTS["all"]` = **93** (was 71 at Wf-352-A1 baseline; +22 from Wf-353-B1)
- `EXPECTED_GATE_COUNTS["frontier"]` = 25 (UNCHANGED)
- `EXPECTED_GATE_COUNTS["chemprop_v2_v3"]` = 14 (UNCHANGED)
- `EXPECTED_GATE_COUNTS["cfm"]` = 6 (UNCHANGED)

**This C2 sync adds 0 NEW pre-flight gates** — the 71→93 expansion is from Wf-353-B1 (Wired 22 Wf-343c1 gates), not from Wf-353-C2 docs-only work.

---

## §4 Honest framing per `party-twenty-spirit` §3.5

**MEASURED today (2026-09-29, this Wf-353-C2 INDEX + AGENTS sync)**:

- 0/0 NEW tests added (C2 is docs-only)
- 0/0 GPU launches performed
- 0/0 NEW pre-flight gates added by Wf-353-C2
- +97 lines to `TODO/INDEX.md` (new §22.41 entry)
- +1 line to `AGENTS.md` (new §6 row)
- 1 NEW closeout file at `TODO/completed/353c2_index_sync.md` (this file)

**Honest-negative caveats preserved (4 entries, see §22.41.6 in INDEX.md)**:

1. **HONEST_NEGATIVE_GPU_BENCH_DEFERRED_WF353**: Wf-353-A1..A4 are CPU-only scaffolds; GPU benchmarks DEFERRED per AGENTS.md §5.1
2. **HONEST_NEGATIVE_FLOWMOL_LICENSE_GATE**: FlowMol arXiv:2412.01966 CC-BY-NC; USER_OK required for production integration
3. **HONEST_NEGATIVE_WF343C1_POST_RETRAIN_SKIP_GATES_NOT_WIRED**: 4 §5.11-§5.14 post-retrain SKIP gates intentionally NOT wired into `build_gates('all')` per honest-framing convention
4. **HONEST_NEGATIVE_PRE_FLIGHT_GATE_EXPANSION_71_93**: +22 expansion from Wf-353-B1, NOT from Wf-353-C2; pre-flight v3 NOT re-run by Wf-353-C2

---

## §5 Cross-references

- `TODO/INDEX.md` §22.41 (NEW — appended between §22.40 and §23)
- `AGENTS.md` §6 row 255 (NEW — appended between row 254 and end-of-§6)
- `TODO/completed/353{a1,a2,a3,a4,b1,b2}_*.md` (6 Wf-353 sub-flow closeouts — primary sources)
- `TODO/completed/340p3_concrete_code_changes.md` (Wf-340-P3 consolidation closeout — Wf-353 wire-up source)
- `molmetal/scripts/pre_flight_check.py:3117` (EXPECTED_GATE_COUNTS = {"all": 93, ...})
- `molmetal/scripts/wf343c1_pre_retrain_check.py` (standalone script for the 4 post-retrain SKIP gates)
- `molmetal/tests/test_wf353{a1,a2,a3,a4,b1,b2}_*.py` (6 NEW test files; 22 NEW tests total)
- AGENTS.md §1.4 pre-flight PASS gate + §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar

---

## §6 What was NOT touched in this Wf-353-C2 (per standing directives)

- No code under `molmetal/` modified (docs-only)
- No `paper/*` touched (TODO-84 active per AGENTS.md §5.3)
- No GPU launches performed (AGENTS.md §5.1 "kill then adapt" directive honored)
- No new metrics emitted
- No new scripts shipped
- No pre-flight gates added (the 71→93 expansion is from Wf-353-B1)

---

**END OF Wf-353-C2** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; Wf-353 docs-only sync complete; INDEX.md +97 lines, AGENTS.md +1 line, 0 NEW tests, 0 GPU launches.