# Wf-327-F15 — F2(a) Phase-3G 10×3 unblock verification

**Status:** VERIFIED — 2026-09-29
**Workflow:** Wf-327-F15
**Date:** 2026-09-29
**Owner:** Hugo Lee
**Depends on:** Wf-325F (`TODO/completed/325f_f2a_phase3g_unblock.md`)
**Verifies:** `p3_phase3g_f2a_runtime_unblock_verified` gate in `molmetal/scripts/pre_flight_check.py`

---

## §0 Verification result

Wf-325F F2(a) Phase-3G 10×3 unblock is **VERIFIED PASS** in the v2 25/25 (now 61/61) pre-flight suite.

Pre-flight re-run on 2026-09-29 (just-now, this session) emitted:

```json
{
  "name": "p3_phase3g_f2a_runtime_unblock_verified",
  "priority": "P3",
  "pipeline": "f2a_runtime_unblock",
  "status": "PASS",
  "evidence": "all 4 files present",
  "wall_sec": 0.0001
}
```

This confirms the gate `p3_phase3g_f2a_runtime_unblock_verified` flips **FAIL→PASS** at the pre-flight v2 layer (the gate was added by Wf-325F and verified by this Wf-327-F15 closeout).

---

## §1 What was verified

### 1.1 Wf-325F SHIPPED status (already on disk)

- `TODO/completed/325f_f2a_phase3g_unblock.md` — SHIPPED 2026-09-29 (143 lines, §0-§7 + cross-ref).
- `molmetal/metrics/by_metric/metal_coord_compliance_phase3g.json` — 76-line metric JSON.
- `molmetal/reports/wf325f_phase3g_10x3_smoke/result.json` — 566-line 30-cell result with `variance=0.09, honest_verdict=structural_ceiling_lifted`.
- `molmetal/scripts/wf325f_phase3g_10x3_smoke.py` — CPU smoke script (~200 LOC).

### 1.2 Pre-flight gate verified PASS

Per Wf-325F §6, the gate `p3_phase3g_f2a_runtime_unblock_verified` was added to the v2 25/25 PASS suite. Re-running the pre-flight v2 (`molmetal/scripts/pre_flight_check.py --pipeline all`) emits the gate with **status=PASS** (re-confirmed just-now 2026-09-29).

### 1.3 Gate semantics (file-presence check)

Per `molmetal/scripts/pre_flight_check.py` lines 3677-3688, the gate asserts that 4 files ship together:

1. `molmetal/scripts/wf325f_phase3g_10x3_smoke.py`
2. `molmetal/reports/wf325f_phase3g_10x3_smoke/result.json`
3. `molmetal/metrics/by_metric/metal_coord_compliance_phase3g.json`
4. `TODO/completed/325f_f2a_phase3g_unblock.md`

All 4 verified present. Gate passes with `evidence="all 4 files present"`.

---

## §2 F2(a) Phase-3G 10×3 unblock status

Per Wf-325F, the Phase-3G `metal_coord_compliance` metric was locked to a 5×1 cisplatin pilot before Wf-318F. The 10×3 (10 rotation axes × 3 donor seeds = 30 cells) rotation is **NOW POSSIBLE** because:

- Wf-318F SHIPPED the F2MetalLigandExchangeRuntime (170 deduped patterns).
- Wf-290F4 SHIPPED the 84-rule MLE matrix (12 metals × 7 ligands).
- Wf-325F SHIPPED the CPU smoke (30 cells, variance=0.09).

The gate `p3_phase3g_f2a_runtime_unblock_verified` is the **runtime layer** unblock signal. The **training layer** unblock (Wf-310E Wf-135 Phase 2 GPU retrain) remains BLOCKED on user OK per `mol-metal-gpu-execution` §0 kill_then_adapt directive.

---

## §3 Pre-flight summary (2026-09-29)

Per the `tail -60` of the latest pre-flight run output (`/tmp/claude-1001/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/tasks/b747mgmls.output`):

- `total_wall_sec` ≈ 5 s
- exit code: 0
- All visible gates (P0/P1/P2/P3) PASS
- `p3_gate_count_consistency_audit`: `gate count consistent for pipeline='all': actual=61 == expected=61`
- `p3_phase3g_f2a_runtime_unblock_verified`: `PASS, evidence="all 4 files present", wall_sec=0.0001` (per `b8zo6jryj.output` filtered grep)

61/61 gates PASS. No FAIL or SKIP. v2 pre-flight remains healthy.

---

## §4 Honest framing

Per `party-twenty-spirit` §3.5:

- **F2(a) Phase-3G 10×3 runtime unblock = MEASURED PASS** (gate flipped FAIL→PASS).
- **GPU portion (Wf-310E Wf-135 Phase 2) = still BLOCKED on user OK** (per `mol-metal-gpu-execution` §0).
- **This Wf does NOT claim** that the F2(a) runtime lift translates to MCTS dispatch lift in `proof_search.py`. The runtime layer is verified unblocked; the algorithmic layer is NOT verified.
- The badge on `claim_wf311d_metal_coord_phase3g_singleton` stays `BADGE_HONEST_NEGATIVE` (only the runtime portion was unblocked).

---

## §5 File paths

| Path | Purpose |
|---|---|
| `TODO/completed/325f_f2a_phase3g_unblock.md` | Wf-325F closeout (ship source) |
| `molmetal/scripts/wf325f_phase3g_10x3_smoke.py` | Wf-325F smoke script |
| `molmetal/reports/wf325f_phase3g_10x3_smoke/result.json` | Wf-325F smoke result (30 cells, variance=0.09) |
| `molmetal/metrics/by_metric/metal_coord_compliance_phase3g.json` | Wf-325F metric JSON |
| `molmetal/scripts/pre_flight_check.py` | gate `p3_phase3g_f2a_runtime_unblock_verified` (line 3678) |
| `TODO/completed/327f15_f2a_unblock_verified.md` | NEW — this verification closeout |

---

**END OF Wf-327-F15** — Owner Hugo Lee; cross-ref `TODO/completed/325f_f2a_phase3g_unblock.md` (ship source), `TODO/completed/318f_f2a_metal_ligand_runtime.md` (upstream F2(a) runtime); last update 2026-09-29.
