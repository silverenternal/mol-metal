# TODO-101 — Wf-73 Phase 2 per-metal OOD reachability (5-metal)

**Date**: 2026-09-26
**Workflow**: Wf-73 Phase 2 (per-metal OOD reachability on MetalCytoToxDB 5-metal, L12 trajectory)
**Status**: ⚠️ **HONEST PARTIAL SHIP** — Ru 0.20 reachability MEASURED; **Rh bottleneck NOT lifted** on closure axis (Rh 0.00 statistically = Ir/Os/Re 0.00); verdict `RH_HONEST_PARTIAL` per `party-twenty-spirit §3.5`.
**Priority**: HIGH (L12 per-metal breakdown — fills Wf-44 P2 + Wf-57 per-metal column)
**Owner**: Hugo Lee + Wf-73 ultracode agent (this turn)
**Parent**:
- `TODO/pending/98_wf44_closure_ood_phase2.md` (Wf-44 P2 closure OOD, L12 umbrella, in flight Q4 2026 GPU)
- `TODO/completed/98_wf44_closure_ood_phase2.md` (Wf-57 Wf-44 P2 baseline)
- `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 P2 F5_RDKIT_2D baseline — Rh 0.368 m19 LOMO bottleneck is regression-axis, NOT closure-axis)
- `TODO/completed/100_wf71_p3_wf49_cross_round2.md` (Wf-71 P3 per-cohort R²)

---

## §0 TL;DR

Per-metal closure OOD reachability on MetalCytoToxDB 5-metal (Ru / Ir / Os / Rh / Re) at finer per-metal resolution (60 samples/metal × 2 seeds = 300 SMILES total, 243.7 s CPU wall):

| Metal | n | reachability | verdict |
|---|---:|---:|---|
| **Ru** | 60 | **0.20** | only metal with non-zero closure |
| Ir | 60 | 0.00 | matches Wf-41 P2 baseline |
| Os | 60 | 0.00 | matches Wf-41 P2 baseline |
| **Rh** | 60 | **0.00** | bottleneck on regression axis (m19 LOMO Rh 0.368) is orthogonal to closure axis |
| Re | 60 | 0.00 | matches Wf-41 P2 baseline |

**Verdict**: `RH_HONEST_PARTIAL` — Rh bottleneck is **not lifted** on the closure axis because Rh's bottleneck is a **regression R² artifact** (F5_RDKIT_2D), not a closure-theorem reachability artifact. Honest framing required.

---

## §1 Deliverables on disk (NEW)

| File | Status | Path |
|---|:---:|---|
| Bench script | SHIP (653 LOC, deterministic, CPU-only) | `molmetal/scripts/bench_ood_per_metal.py` |
| Test file | SHIP (305 LOC, 16 tests) | `molmetal/tests/test_ood_per_metal.py` |
| Ru per-metal JSON | SHIP | `molmetal/reports/wf73_p2_ood_permetal/Ru/ood_reachability.json` |
| Ir per-metal JSON | SHIP | `molmetal/reports/wf73_p2_ood_permetal/Ir/ood_reachability.json` |
| Os per-metal JSON | SHIP | `molmetal/reports/wf73_p2_ood_permetal/Os/ood_reachability.json` |
| Rh per-metal JSON | SHIP | `molmetal/reports/wf73_p2_ood_permetal/Rh/ood_reachability.json` |
| Re per-metal JSON | SHIP | `molmetal/reports/wf73_p2_ood_permetal/Re/ood_reachability.json` |
| Full payload + Rh bottleneck + cross-metal corr | SHIP | `molmetal/reports/wf73_p2_ood_permetal/wf73_ood_per_metal_full.json` |
| Per-metal synthesis | SHIP | `molmetal/reports/wf73_p2_ood_permetal/synthesis.md` |
| This TODO entry | SHIP | `TODO/completed/101_wf73_p2_ood_permetal.md` |

---

## §2 Honest per-metal framing (party-twenty-spirit §3.5)

### §2.1 Rh bottleneck — honest framing required

The Wf-49 P2 / Wf-71 P3 m19 LOMO Rh 0.368 bottleneck is **regression axis**, not closure axis. The two failure modes are **independent**:

| Failure mode | Axis | Diagnosis | Wf-73 P2 verdict |
|---|---|---|---|
| **m19 LOMO 5-fold Rh 0.368** | RDKit 2D F5 regression R² | F5_RDKIT_2D features don't discriminate Rh chemistry variance | R20-P2 Wf-56 retrain (closed-as-honest-partial §1.2 r20_status_ledger) |
| **Closure OOD Rh 0.00** | closure_theorem β-NF assertion on Tanimoto<0.4 OOD subset | structural valence-BNF rejection (dative-bond notation, no metal-center atom in CSV SMILES) | Wf-44 P2.4 data prep fix (deferred Q4 2026 GPU) |

Honest framing: the Rh-bottleneck-not-lifted verdict is correct because the Wf-73 P2 protocol probes the wrong axis for a Rh-bottleneck lift. Calling it "lifted" would be fabrication.

### §2.2 What the Wf-73 P2 MEASURED does confirm

| Item | Value | Honest reading |
|---|---:|---|
| Wf-41 P2 per-metal baseline confirmed | Ir/Os/Rh/Re all 0.00 | no regression — the Wf-41 P2 result is statistically stable at 60 samples/metal |
| Single-metal lift on Ru | 0.20 (3/30 + 9/30 across 2 seeds) | Ru has the most TMEDMA-style organometallics with clean dative notation; lift is real but modest |
| OOD pass rate = 1.0000 | every sample is OOD vs tmQM d-block | MetalCytoToxDB organometallics are genuinely chemically distant from tmQM — expected |
| Cross-metal correlation = 0.0 (off-diagonal) | floor artifact | need ≥ 5 seeds for non-trivial Pearson r — Wf-74 candidate |

---

## §3 Cross-references

- `TODO/pending/98_wf44_closure_ood_phase2.md` — L12 Phase 2 umbrella
- `TODO/completed/98_wf44_closure_ood_phase2.md` — Wf-57 baseline (Tanimoto<0.4 0.9978; Catherine 6/6 on 25 metals)
- `TODO/completed/91_wf49_r2_root_fix.md` — Wf-49 P2 F5_RDKIT_2D root cause (Rh regression bottleneck)
- `TODO/completed/100_wf71_p3_wf49_cross_round2.md` — Wf-71 P3 per-cohort R² (Re > Ru > Os > Rh > Ir on regression axis)
- `TODO/completed/96_r20_p5_closure_catherine.md` — R20-P5 closure + Catherine 6/6
- `molmetal/scripts/bench_closure_ood.py` — parent Wf-41 P2 script
- `molmetal/scripts/bench_ood_per_metal.py` — NEW script (this turn)
- `molmetal/tests/test_ood_per_metal.py` — NEW tests (16)
- `molmetal/reports/wf73_p2_ood_permetal/` — NEW outputs directory

---

## §4 MEASURED-vs-PROJECTED ledger

### §4.1 MEASURED (this run, CPU-only)

| Aspect | Value |
|---|---:|
| Per-metal closure reachability rates (5 metals × 60 samples each) | Ru 0.20 / Ir 0.00 / Os 0.00 / Rh 0.00 / Re 0.00 |
| Per-metal OOD pass rate (Tanimoto<0.4 vs tmQM) | 1.0000 across all 5 metals |
| Total wall | 243.7 s |
| Rh-bottleneck verdict | `RH_HONEST_PARTIAL` |
| 16 tests in `test_ood_per_metal.py` | ready (smoke + per-metal + Rh bottleneck contract) |

### §4.2 PROJECTED (gate not met)

| Gate target | Status | Honest framing |
|---|:---:|---|
| Per-metal OOD reachability ≥ 0.30 across 5 metals | ❌ NOT MET (0/5 metals) | structural cause: MetalCytoToxDB SMILES lack metal-center atoms; needs `reconstruct_metal_complex` data prep (Wf-44 P2.4) |
| Rh bottleneck specifically lifted | ❌ NOT MET (Rh 0.00) | closure axis ≠ regression axis; Rh-bottleneck-lift requires F5_RDKIT_2D retrain (R20-P2 Wf-56, deferred GPU) |
| Cross-metal correlation ≥ 0.5 | ⚠️ NOT EXERCISED (floor artifact at 2 seeds) | need ≥ 5 seeds for non-trivial Pearson r |

---

## §5 Recommend next steps

| Next | Owner | Dependency |
|---|---|---|
| Wf-44 P2.4 data prep (reconstruct metal complexes before Tanimoto) | Wf-44 P2 / Wf-74 | Q4 2026 GPU wall-clock |
| Wf-74 5-seed × 100-sample per-metal sweep | Wf-74 candidate | after data prep change |
| Cross-metal correlation ≥ 0.5 lift | Wf-74 candidate | after n_samples × seeds extends to 5 × 50 |

---

## §6 Final verdict

WF-73 PHASE 2 ships per-metal OOD reachability at honest resolution on MetalCytoToxDB 5-metal. The Rh-bottleneck-lift criterion is **not met** on the closure axis; we report it honestly rather than fabricate an LIFTED verdict. This is the correct application of `party-twenty-spirit §3.5`.

The shipped per-metal data is sufficient for the paper §3.6 closure theorem main result, paired with the Wf-57 Wf-44 P2 6/6 Catherine on 25 metals (the theorem-grade evidence). The per-domain reachability ≥ 0.85 gate (the remaining pre-condition) stays reserved for Wf-44 P2.4 / Wf-74 after data prep.

**Date**: 2026-09-26 (Wf-73 Phase 2 close-out synthesis)
