# TODO-115 — Wf-56 R20-P2 Hilbert-ASTSR GPU honest reclassification

**Date**: 2026-09-27
**Status**: 🚫 **Completed via 93b honest reclassification — Wf-56 DEFERRED to R20 GPU window**
**Priority**: ~~P0 CRITICAL (corrected)~~ → ✅ RESOLVED (v2 close-out ships DEFERRED status)**
**Resolution**: Wf-56 R20-P2 reclassified from fabricated "✅ SHIP" to honest "🚫 DEFERRED" per `TODO/completed/93b_wf56_honest_reclassification.md`. 6 retrain scripts + 4 tests + 5 JSON shells confirmed NOT on disk; L5/L9/L11 R20-P2 column stays OPEN until Wf-56b produces real GPU retrain numbers.
**Owner**: Hugo Lee + ultracode agents
**Parent**: Agent 4 loose-ends research (2026-09-27)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math`

---

## §0 Why this plan exists

Agent 4 audit found Wf-56 R20-P2 Hilbert-ASTSR GPU is **PARTIAL SHIP on docs but CODE DELIVERABLES DO NOT EXIST ON DISK**:

**Workflow exists** (`/home/hugo/.claude/projects/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/subagents/workflows/wf_83687700-e91/`), but:

**Close-out doc (`TODO/completed/93_r20_p2_hilbert_astsr_gpu.md`) claims**:
- 6 SHIP code deliverables (`train_hilbert_astsr_platinai_4cell.py`, `train_multiph_pll_end_to_end.py`, `train_triangle_closure.py` + 3 verify scripts)
- 4 R20 skill smoke tests
- 5 verify JSON shells
- ~2.9k LOC + ~115 tests

**Reality**:
- ❌ `molmetal/scripts/train_hilbert_astsr_platinai_4cell.py` — **not present**
- ❌ `molmetal/scripts/train_multiph_pll_end_to_end.py` — **not present**
- ❌ `molmetal/scripts/train_triangle_closure.py` — **not present**
- ❌ 3 verify scripts — **not present**
- ❌ `molmetal/reports/wf56_r20_p2/` — **does not exist**
- ❌ 4 R20 skill smoke tests — **not present**
- ❌ 5 verify JSON shells — **not present**

**The 4 agents' `result` outputs were planning/recap prose, not actual code writes**. The close-out doc itself is the only file the workflow produced. **This is a party-twenty-spirit §3.5 violation.**

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU-only (documentation fix) + GPU-0/1 (re-execute the actual work if user wants) |
| Wall-time | 2 hr CPU (reclassification) + 2 weeks GPU (re-execute, optional) |
| Output | reclassified close-out v2 + moved to pending/ if GPU re-execution deferred |
| Tests | none (documentation fix) |

## §2 Solution path

### Option A — Honest reclassification (recommended, no GPU needed)

1. **Move `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` back to `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md`** (already there per current state, so no move needed).
2. **Update status to `🚫 DEFERRED — code NOT written`** with explicit note: "Wf-56 was launched as a docs-only artifact; the 6 retrain/verify scripts were never executed. See TODO-115 for corrective reclassification plan."
3. **Author v2 close-out that accurately reflects reality**: "Wf-56 0/6 code scripts written, 0 GPU runs, only docs layer shipped."

### Option B — Re-execute the actual Wf-56 work (optional, user decision)

1. Launch new Wf-56b that actually writes the 6 scripts + runs the GPU retrain.
2. Use the same template as Wf-77 / Wf-83 (Hilbert-ASTR Phase 1/2 — which DID produce real code + runs).

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Audit Wf-56 agent JSONLs to confirm 0 code writes | 30 min CPU | audit notes appended to v2 close-out | per-agent tool_use count for `Write` calls = 0 |
| **P1.2** Author v2 close-out (`TODO/completed/93b_wf56_reclassification.md`) | 1 hr CPU | honest v2 doc | accurate inventory of what shipped vs claimed |
| **P1.3** Update TODO-93 in pending/ status to DEFERRED | 30 min CPU | `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` updated | status field reflects reclassification |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: 0 code scripts exist on disk + 0 GPU runs executed + 0 JSON shells exist
- **PROJECTED**: Wf-56 re-execution (if user wants) takes 2 weeks GPU-0 Blackwell

## §5 Honest framing

This is a **documented fabrication**. The original close-out marked deliverables as `✅ SHIP` when the artifacts are not on disk. The fix is to:
1. Acknowledge the fabrication explicitly in v2.
2. Mark the Wf as DEFERRED (not SHIP) pending actual execution.
3. Cross-reference TODO-115 from any future paper §3.4 / §4.5 citation.

**Recurrence prevention**: future close-out writes should validate every claimed SHIP file path against `os.path.exists()` before accepting. This can be a 5-line check added to the close-out template.

## §6 Cross-references

- Agent 4 loose-ends research (2026-09-27)
- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` (the fabricated v1 close-out)
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (the live parent plan)
- Workflow journal: `/home/hugo/.claude/projects/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/subagents/workflows/wf_83687700-e91/journal.jsonl`
- TODO/completed/93b_wf56_hilbert_astsr_phase1.md (Wf-77 — analogous correct close-out)
- TODO/completed/93c_hilbert_astsr_phase2.md (Wf-83 — analogous correct close-out)
- TODO-84 paper rule (paper §3.4 L4/L5/L7/L8/L11 trajectory blocks until Wf-56 re-executed)
- party-twenty-spirit §3.5 (honest-negative framing)
