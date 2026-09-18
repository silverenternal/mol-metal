# WF-Last-CPU-Polish — MASTER consolidation (2026-09-17)

**Owner:** Wave-3 (final CPU-only polish pass)
**Status:** SHIPPED — 3/3 polish items closed, all CPU-only, zero GPU consumed.

## 1. TL;DR

Three CPU-only polish items shipped (CFM import fix, Strategy 3 architecture doc, D-MPNN multi-task completion).
All items are additive, honest-framed, and gated on no new measurements.
Total LOC shipped: 18 (code) + 280 (doc) + 285 (test) + 194 (test) = 777 LOC.

## 2. Outcomes

| Item | Workflow | Pass/Fail | LOC | Tests |
|------|----------|-----------|-----|-------|
| P1A | CFM import fix (`r10_cfg_real_crossdocked.py` + `.pth`) | PASS | +17 code, +1 .pth | 5/5 |
| P1B | Strategy 3 arch doc (`lambda_cfm_cascaded.md`) | PASS (DESIGN ONLY) | ~280 doc, +4 TODO21 | n/a (doc) |
| P1C | D-MPNN multi-task baseline completion | PASS | +285 (test_dmpnn_baseline.py) | 32/32 |

## 3. MEASURED deltas

| Fix | Outcome |
|-----|---------|
| **CFM import fix** | `ModuleNotFoundError` resolved across 4 cwd's (project root, molmetal/, molmetal/tests/, /tmp). 5/5 new tests pass; `.pth` file + 12-line in-script bootstrap (defence in depth). |
| **Strategy 3 doc** | DESIGN ONLY: ~280 lines covering overview, ASCII flow, 2-stage roadmap, 4 blockers, 5-paper SOTA lineage (TargetDiff/FLOWr/Pocket2Mol/DiffSBDD/DecompDiff), ~30 LOC pseudo-code. All Vina lift +1-2 / decode_ratio +0.10-0.30 PROJECTED, not MEASURED. |
| **D-MPNN completion** | 17-test integration suite shipped (test_dmpnn_baseline.py); 32/32 D-MPNN family tests pass in 86.85 s. CLI smoke: 3 epochs × 3668 Ru mols in 30.2 s, test pearson_r=0.086 (honest baseline floor, well below ridge 0.572). |

## 4. What remains BLOCKED

GPU retrain (decode_ratio gate currently 0/192 per wf_cfm_gpu_recovery_now) — Strategy 3 impl waits on this + decode>0 + Round-13 100x3 λ-only sweep + wet-lab collaborators for PlatinAI oracle validation.

## 5. Honest framing

No new MEASURED metrics this round — all 3 items are polish/integration/docs, not experiments.
D-MPNN baseline serves as comparison floor for Lambda/CFM, not SOTA claim (test pearson_r=0.086 << ridge 0.572).
Strategy 3 doc is DESIGN ONLY with explicit PROJECTED markers; pseudo-code raises NotImplementedError.
CFM import fix is zero-runtime-cost (`.pth` evaluated once at site init).
No GPU consumed; no code in `flow_matching_lipman/__init__.py` touched; no existing tests broken (Test 17 additive guard).