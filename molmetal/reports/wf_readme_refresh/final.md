# wf_readme_refresh — final

**Workflow**: R17 README + INDEX refresh to reflect close-out state
**Date**: 2026-09-25
**Owner**: docs-sync (continuation of wf_docs_sync)
**Status**: COMPLETE — README.md and INDEX.md updated to R17 final state

---

## 1. Purpose

Refresh `TODO/README.md` (was R15-era stale) and `TODO/INDEX.md` (was already R17 but missing
the skills section) to reflect the R17 close-out state:

- 30/30 decode probes = 1.000 (vs R16 0/192)
- 14/14 evaluators verified on Blackwell + CadButEaas/.venv-vlm
- 20 metrics specified (Tier 1 / Tier 2 / Tier 3)
- 5 project skills shipped (was 4 in MERGE_GATE / EVALUATORS_MIGRATED / MECHANISM_METRIC_MAPPING
  pre-R17 docs; +1 with the new `mol-metal-mechanism-mapping` skill)
- 6 new top-level files added in R17 (`PROJECT_STATEMENT.md`, `INSIGHTS_FROM_WORKFLOWS.md`,
  `MERGE_GATE.md`, `EVALUATORS_MIGRATED.md`, `METRICS_TARGETS.md`, `MECHANISM_METRIC_MAPPING.md`)
- Environment updated to reflect shared `CadButEaas/.venv-vlm` interpreter (was the old
  uv-managed `mol-metal/.venv`)
- Hardware updated to NVIDIA Blackwell sm_120 (was ROCm RX 7800 XT)

---

## 2. Changes shipped

### 2.1 `TODO/README.md` (full rewrite to R17)

**File**: `/home/hugo/codes/mol-metal/TODO/README.md` (was R15-era stale; now R17 close-out)

| Section | Old state | New state |
|---|---|---|
| **Header** | `Mol-Metal: Precious-Metal Anti-Cancer Drug Design via Flow Matching + SBDD` (R15 framing) | `Mol-Metal: Click-Chemistry-Driven de novo Design of Precious-Metal Anticancer Agents` (R17 + lambda-algorithm framing per PROJECT_STATEMENT) |
| **Last updated** | `2026-09-16` | `2026-09-25` |
| **Project Overview** | Pure FM + SBDD framing | Added molecular lambda calculus + KataGo-MCTS + 18 click rules + 84-rule MLE matrix framing; cited `β-NF closure` + `atoms-as-combinators` |
| **R17 Final State table** | (absent) | NEW — 9-row table: 30/30 probes, 14/14 evaluators, 20 metrics, 20-element vocab, 84-rule MLE, 18 click rules, 5 skills, Deflex cite, 3-seed stability |
| **Environment** | `uv` + ROCm 7.2 + RX 7800 XT | Shared `CadButEaas/.venv-vlm` (Python 3.12) + Blackwell sm_120 (current) + ROCm 7.2 (legacy) |
| **Engineering Practices** | kept (R15) | +2 conventions: always reference `METRICS_TARGETS.md` + `MECHANISM_METRIC_MAPPING.md` when changing a metric; use `MERGE_GATE.md` for dual-line parallel merge |
| **Project Skills (NEW)** | (absent) | 5-row table: lambda-calculus / cfm-training / click-chemistry / anticancer / mechanism-mapping with path + ownership |
| **Directory Structure** | 11 root files, 9 pending, 28 completed | +6 R17 root files; 15 pending; 51 completed; 4 archive |
| **Strategic Core** | Lipman + Hexagonal + Reuse (R15) | +YuelBond + KataGo-MCTS + PLL + 84-rule MLE + 18 click + ASTSR + 6 shim fixes |
| **Roadmap Status table** | up to R15 | + R16 / R16b / R17 closed rows; R18 gated on dGPU cold power cycle |
| **Active Work** | 8 pending | 15 pending (R17 close-out added Track A/B spec + TODO-33 P1+P2 + TODO-37 model track + TODO-38 lit synthesis) |
| **R17 close-out summary** | (absent) | NEW — 8 advantages made full + 6 honest caveats preserved |
| **R16 + R15 + 2026-09-15 ship summaries** | (R15 only) | + R16 + 2026-09-15 preserved |

**Total**: ~190 lines old → ~395 lines new (~205 lines added).

### 2.2 `TODO/INDEX.md` (R17 close-out refresh)

**File**: `/home/hugo/codes/mol-metal/TODO/INDEX.md`

| Section | Change |
|---|---|
| **Top timestamp** | Updated to R17 close-out wording: `5 project skills shipped (...mechanism-mapping); PROJECT_STATEMENT.md + INSIGHTS_FROM_WORKFLOWS.md + MERGE_GATE.md + EVALUATORS_MIGRATED.md + METRICS_TARGETS.md + MECHANISM_METRIC_MAPPING.md added; R18 gated on dGPU cold-power-cycle` |
| **Layout** | +6 R17 root files listed (PROJECT_STATEMENT, INSIGHTS_FROM_WORKFLOWS, MERGE_GATE, EVALUATORS_MIGRATED, METRICS_TARGETS, MECHANISM_METRIC_MAPPING); pending/ count 11 → 15; completed/ count 31 → 51 |
| **`completed/` header** | `28 files` → `51 files` |
| **`completed/` archive reference** | `R3–R11b` → `R3–R17` |
| **Project Skills section (NEW)** | NEW — 5-row table with path + ownership; +73 lines added across the 5 SKILL.md files (per wf_docs_sync/final.md) |
| **Master close-out inventory** | `completed/ → 31 files` → `51 files (up from 31; +20 R17-era: 32/33/34/35/37/38 + 7 R15-era)` |

### 2.3 `molmetal/reports/wf_readme_refresh/final.md` (this file)

NEW workflow report documenting the refresh.

---

## 3. Discrepancy note: "6 skills" vs actual 5

The original task description stated "Update INDEX.md layout section to mention 6 skills (was
4)". The actual count at the time of this refresh is **5 project skills** (per
`wf_docs_sync/final.md` and the `ls .claude/skills/` listing):

1. `mol-metal-lambda-calculus`
2. `mol-metal-cfm-training`
3. `mol-metal-click-chemistry`
4. `mol-metal-anticancer`
5. `mol-metal-mechanism-mapping`

There is **no 6th project skill**. The task description likely counted `mol-metal-mechanism-mapping`
separately from the "original 4" without accounting for it being the same 5th entry; or it
anticipated a 6th skill that has not yet been added.

**Resolution**: documented the actual count (5) in both the README and the INDEX, and flagged
the discrepancy in this report. If a 6th skill is required later, it can be added with a new
SKILL.md and the table will be updated.

---

## 4. Verification

```bash
# 1. README.md timestamp + R17 framing
head -3 /home/hugo/codes/mol-metal/TODO/README.md
# Expected: 2026-09-25 (R17 close-out...) + Click-Chemistry-Driven...

# 2. INDEX.md top timestamp + 5 skills
head -3 /home/hugo/codes/mol-metal/TODO/INDEX.md
# Expected: 2026-09-25 (R17 close-out — 30/30 decode probes ...; 5 project skills shipped ...)

# 3. Skills table present in both files
grep -c "mol-metal-mechanism-mapping" /home/hugo/codes/mol-metal/TODO/README.md /home/hugo/codes/mol-metal/TODO/INDEX.md
# Expected: ≥ 2 in README, ≥ 2 in INDEX

# 4. 30/30 probe + 14/14 evaluator + 20 metrics claims in both files
grep -c "30/30\|14/14\|20 metrics\|20 个指标" /home/hugo/codes/mol-metal/TODO/README.md /home/hugo/codes/mol-metal/TODO/INDEX.md
# Expected: ≥ 5 in README (the R17 table has all three + prose); ≥ 3 in INDEX

# 5. Directory structure updated
grep -A1 "PROJECT_STATEMENT.md" /home/hugo/codes/mol-metal/TODO/README.md | head -10
# Expected: mentions each of PROJECT_STATEMENT.md / INSIGHTS_FROM_WORKFLOWS.md / MERGE_GATE.md / EVALUATORS_MIGRATED.md / METRICS_TARGETS.md / MECHANISM_METRIC_MAPPING.md
```

---

## 5. Honest framing preserved

- `mol-metal/.venv` remains a stale legacy venv (not the runtime); README flags it explicitly
- Hardware parity note: the README documents both Blackwell (current) and ROCm 7.2 (legacy RX
  7800 XT) without claiming cross-compatibility
- The 6-shim-fix list points at TODO-33 / P1+P2 rather than re-listing every file change
- The R17 honest caveats (GPU hang at step 2500, single-seed only, 30-cell pilot R²) are all
  preserved verbatim from `wf_todo33_p1_done.md` + `wf_todo33_p2_shim_fix/final.md` +
  `wf_deflex_astsr` (TODO-35)
- The "20 metrics" claim explicitly notes 3 are "future work" — not all 20 are currently
  MEASURED

---

## 6. Cross-skill coherence check

The README + INDEX + PROJECT_STATEMENT + MERGE_GATE + EVALUATORS_MIGRATED + METRICS_TARGETS +
MECHANISM_METRIC_MAPPING + 5 SKILL.md files now agree on:

- 30/30 decode probes = 1.000 (vs R16 0/192)
- 14/14 evaluators verified on Blackwell + CadButEaas/.venv-vlm
- 20 metrics specified (Tier 1 / Tier 2 / Tier 3)
- 84-rule MLE matrix + 18 click rules + 20-atom vocab
- 6 shim fixes (TODO-33 P1+P2)
- Deflex 2026 (`yu2026deflex`) cited
- 5 project skills shipped
- Environment = `CadButEaas/.venv-vlm`

No contradictions.

---

## 7. Files changed

| File | Lines before | Lines after | Delta |
|---|---|---|---|
| `TODO/README.md` | ~190 | ~395 | +205 |
| `TODO/INDEX.md` | ~350 | ~395 | +45 |
| `molmetal/reports/wf_readme_refresh/final.md` | 0 | (this file) | NEW |

---

## 8. Follow-ups (optional, not blocking)

- If a 6th project skill is added, update both files' skills table
- If R18 GPU sweep completes, refresh `Roadmap Status` table to mark R18 closed
- If `paper/main.pdf` is recompiled after the §3.5 Deflex Deep Cite edit, refresh the page
  count + size claim