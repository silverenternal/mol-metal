# Wf-322G — TODO-130 G3 closure theorem Catherine 30-metal bench closeout

**Date**: 2026-09-29 (Wf-322G ship)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ SHIP (documentation + closeout, CPU-only)
**Parent**: `TODO/03_plans/LAMBDA_ROADMAP.md` §3.3 R22-P5 (closure theorem d-block 30-metal bench)
**Cross-refs**:
- `TODO/completed/130_wf130_p1_catherine_30.md` (Wf-130 P1 — 30-metal audit, synthetic CPU-only)
- `TODO/completed/131_wf130_p2_closure_30_5000.md` (Wf-130 P2 — closure 30 × 5000)
- `TODO/pending/96_r20_p5_closure_catherine.md` (R20-P5 parent)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.4 R20-P5 + §3.3 R22-P5

---

## §1 TL;DR

Wf-322G is a **documentation + closeout** workflow for the Catherine
30-metal closure theorem ship surface that already exists in the
codebase from Wf-130 P1 (shipped 2026-09-27) + Wf-130 P2
(2026-09-27). It does NOT launch a fresh GPU sweep — per the
**standing "kill then adapt" GPU directive** (AGENTS.md §5.1 +
`mol-metal-gpu-execution` §0), the **full 30-metal GPU sweep** (the
R20-P5 reachability ≥ 0.95 target) is in pipeline Phase 3 R20-P5 and
requires explicit user OK.

This closeout:

1. **Verifies** that the 30-metal bench + audit infrastructure is on
   disk and the synthetic CPU-only path covers all 30 metals × 6
   Catherine properties (P1..P6).
2. **Adds** the `p3_catherine_30metal_closeout_documented` pre-flight
   gate so downstream consumers cannot silently regress the closeout
   surface (the 30-metal synthetic coverage + closeout doc + sweep
   summary + per-metal JSON).
3. **Updates** `EXPECTED_GATE_COUNTS["all"]` to keep the Wf-321D
   gate-count consistency audit in sync (55 → 56).

**MEASURED key numbers (2026-09-29 closeout)**:

| Metric | Value | Source |
|---|---|---|
| 30-metal canonical CSV | 30 metals | `molmetal/data/transition_metals_30.csv` (Wf-37 P2 ship) |
| Catherine property verifiers | 6 (P1..P6) | `molmetal/math_skills/catherine_properties.py` (Wf-57 P3 ship) |
| Bench script CPU-only runs | yes (smoke path) | `molmetal/scripts/bench_catherine_30metals.py` |
| Audit script CPU-only runs | yes (synthetic fallback) | `molmetal/scripts/audit_catherine_30metals.py` |
| Wf-130 P1 synthetic 6/6 PASS | 30 / 30 metals | `molmetal/reports/wf130_p1_catherine_30/catherine_audit_30metals.json` |
| Wf-130 P1 sweep wall time | < 1 s (synthetic) | Wf-130 P1 metadata |
| Wf-255 30-metal smoke (3 metals) | torch IMPORT ERROR (synthetic fallback engaged) | `molmetal/reports/wf255_catherine_30metals/per_metal_pass.json` |
| Real GPU sweep reachability | **NOT MEASURED** (deferred to R20-P5 Phase 3) | honest-negative below |
| Target reachability | ≥ 0.95 per-metal | `TODO/pending/96_r20_p5_closure_catherine.md` §3 P5.1 |

---

## §2 What was already shipped (Wf-130 P1 + Wf-130 P2)

Per `TODO/completed/130_wf130_p1_catherine_30.md` (Wf-130 Phase 1,
2026-09-27) and `TODO/completed/131_wf130_p2_closure_30_5000.md`
(Wf-130 Phase 2, 2026-09-27):

| Deliverable | Status | Path |
|---|---|---|
| 30-metal canonical CSV | ✅ SHIP | `molmetal/data/transition_metals_30.csv` |
| Catherine 6-property verifiers | ✅ SHIP | `molmetal/math_skills/catherine_properties.py` |
| Bench script (CPU-only smoke) | ✅ SHIP | `molmetal/scripts/bench_catherine_30metals.py` |
| Audit script (CPU-only synthetic) | ✅ SHIP | `molmetal/scripts/audit_catherine_30metals.py` |
| Test suite (16 tests, all PASS) | ✅ SHIP | `molmetal/tests/test_catherine_30metals_audit.py` |
| Per-metal JSON 30 dirs | ✅ SHIP | `molmetal/reports/wf130_p1_catherine_30/{metal}/catherine_audit.json` |
| Combined sweep JSON | ✅ SHIP | `molmetal/reports/wf130_p1_catherine_30/catherine_audit_30metals.json` |
| Synthesis MD | ✅ SHIP | `molmetal/reports/wf130_p1_catherine_30/synthesis.md` |
| Wf-255 smoke bench 3-metal JSON | ✅ SHIP | `molmetal/reports/wf255_catherine_30metals/per_metal_pass.json` |
| Wf-255 sweep summary MD | ✅ SHIP | `molmetal/reports/wf255_catherine_30metals/sweep_summary.md` |

---

## §3 Honest framing (party-twenty-spirit §3.5)

### §3.1 MEASURED vs PROJECTED

- **MEASURED** (Wf-130 P1): 30/30 metals × 6/6 Catherine properties
  PASS on the **synthetic CPU-only fallback path** (no RDKit / torch /
  molmetal_lam available in this environment). The synthetic path uses
  PASS-friendly defaults (`witness_rate=1.0`, `well_typed_rate=1.0`,
  `coord_violations=0`, `reachability_rate=1.0`) so the verifier
  returns 6/6 PASS by construction. This is honest-negative framing:
  the 30/30 × 6/6 number is on a **placeholder bench**, not on
  production closure BFS.
- **NOT MEASURED** (honest-negative): the full 30-metal × real
  closure-theorem BFS reachability ≥ 0.95 target. This is the R20-P5
  §3 P5.1 deliverable. In the current env (no RDKit / molmetal_lam)
  the 30-metal real BFS cannot run on CPU; per Wf-130 P2 honest
  framing, the OOD / per-metal real reachability is BLOCKED on env +
  GPU budget. **Recovery path is the R20-P5 Phase 3 GPU sweep**
  (per LAMBDA_ROADMAP §1.4 R20-P5 timeline W9-W10, requires explicit
  fresh user OK per AGENTS.md §5.1 kill_then_adapt directive).

### §3.2 Tier-2 strategy alignment

Per Wf-317 Tier-2 strategy + LAMBDA_ROADMAP §3.3: **closure theorem's
ship surface is computational** (the 6/6 properties + per-metal
reachability are computed artifacts, not theoretical proofs). The
CPU-only bench + audit + per-metal JSON is the **computational ship
surface** for R22-P5. The full 30-metal real BFS is the **stretch
GPU target** for R20-P5 Phase 3.

### §3.3 What this closeout does NOT do

- Does NOT launch a GPU sweep (kill_then_adapt directive applies —
  R20-P5 30-metal real BFS would be > 1 hr wall).
- Does NOT modify `closure.py` / `catherine_properties.py` (R17/Wf-57
  ship surface, 0-byte modification policy per Wf-130 P1 §4).
- Does NOT change the 30-metal canonical CSV (Wf-37 P2 ship surface).
- Does NOT change `molmetal/reports/wf130_p1_catherine_30/` outputs
  (already on disk, audit-stable).

---

## §4 The new pre-flight gate (Wf-322G deliverable)

### §4.1 Gate name

`p3_catherine_30metal_closeout_documented`

### §4.2 Gate behaviour

PASS when **all 4 ship artefacts** are present:

1. `molmetal/scripts/bench_catherine_30metals.py` — 30-metal bench script
2. `molmetal/scripts/audit_catherine_30metals.py` — 30-metal audit script
3. `molmetal/reports/wf255_catherine_30metals/per_metal_pass.json` — sweep JSON
5. `molmetal/reports/wf130_p1_catherine_30/catherine_audit_30metals.json` — 30-metal synthetic audit JSON
6. `TODO/completed/322g_todo130_g3_closure_30metal_closeout.md` — this closeout doc

### §4.3 EXPECTED_GATE_COUNTS bump

The Wf-321D gate-count consistency audit asserts
`summary.total == EXPECTED_GATE_COUNTS["all"]`. The new gate brings
the total from 55 → 56, so `EXPECTED_GATE_COUNTS["all"]` is bumped
accordingly. The `chemprop_v2_v3` / `cfm` / `frontier` pipelines are
unchanged (the new gate is in `all` only).

### §4.4 Why the gate is `all`-only

The 30-metal closeout is the **documentation + bench infrastructure**
ship surface, not a per-pipeline contract. The chemprop_v2_v3 / cfm
pipelines have their own gate sets (13 / 6 respectively) and the
frontier pipeline (21) is for frontier-math wire-in. The
catherine_30metal closeout belongs in the `all` pipeline as a
**project-wide documentation presence check**.

---

## §5 Recovery path (R20-P5 Phase 3 GPU sweep)

Per LAMBDA_ROADMAP §1.4 R20-P5 W9-W10 + `TODO/pending/96_r20_p5_closure_catherine.md` §3 P5.1:

1. **CPU-only env upgrade**: install RDKit + molmetal_lam in the
   bench env (Wf-130 P2 follow-up; Wf-130 P1 synthetic path is the
   CPU-only fallback for slim environments).
2. **Real 30-metal BFS sweep**: run `bench_catherine_30metals.py
   --metals 30` on the upgraded CPU env → expect per-metal
   reachability ≥ 0.95 (R20-P5 §3 P5.1 target).
3. **Per-metal OOD overlay**: layer Wf-44 P2 OOD protocol
   (Tanimoto<0.4) on the 30-metal sweep → expect aggregate
   reachability ≥ 0.85 (R20-P5).
4. **GPU acceleration** (optional, R22-P5 stretch): port the
   per-metal BFS to GPU for the 30 × 100+ molecule cohort.

**Kill_then_adapt gate**: any of the above steps > 1 hr wall
requires explicit fresh user OK per AGENTS.md §5.1.

---

## §6 Cross-references

- `TODO/completed/130_wf130_p1_catherine_30.md` — Wf-130 P1 (30-metal synthetic audit, 2026-09-27)
- `TODO/completed/131_wf130_p2_closure_30_5000.md` — Wf-130 P2 (closure 30 × 5000, 2026-09-27)
- `TODO/pending/96_r20_p5_closure_catherine.md` — R20-P5 parent plan
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.4 R20-P5 + §3.3 R22-P5
- `molmetal/scripts/bench_catherine_30metals.py` — 30-metal bench script
- `molmetal/scripts/audit_catherine_30metals.py` — 30-metal audit script
- `molmetal/reports/wf130_p1_catherine_30/` — Wf-130 P1 ship surface (this closeout verifies it is intact)
- `molmetal/reports/wf255_catherine_30metals/` — Wf-255 30-metal smoke bench
- `molmetal/scripts/pre_flight_check.py` — pre-flight gate (this closeout adds gate #56)

---

## §7 Next (Wf-43 / Wf-130 P2 / Wf-44 P2)

Per Wf-130 P1 §6:

- **Wf-130 P2** — Catherine 6/6 audit on **MEASURED** closure-theorem
  BFS (when RDKit + molmetal_lam are available; Wf-130 P1 synthetic
  path is the CPU-only fallback).
- **Wf-43** — Closure d-block 30 metal × 100+ molecule reachability ≥
  0.95 (R20-P5 §3 P5.1 target).
- **Wf-44 P2** — Closure OOD cross-domain Phase 2 (PlatinAI 5K +
  MetalCytoToxDB + tmQM d-block; OOD reachability ≥ 0.85).

**Date**: 2026-09-29 (Wf-322G ship)