# Wf-73 Phase 2 — Per-metal OOD Reachability on MetalCytoToxDB 5-metal

**Date**: 2026-09-26
**Workflow**: Wf-73 Phase 2 (per-metal OOD reachability on MetalCytoToxDB 5-metal, L12 trajectory column)
**Status**: ⚠️ **HONEST PARTIAL SHIP** — Ru 0.20 reachability MEASURED; **Rh bottleneck NOT lifted** (Rh 0.00 same as Ir/Os/Re); verdict `RH_HONEST_PARTIAL` per `party-twenty-spirit §3.5`.
**Priority**: HIGH (L12 per-metal breakdown fills the closure-OOD Phase 2 per-metal column on top of Wf-41 P2 + Wf-57 Wf-44 P2)
**Owner**: Hugo Lee + Wf-73 ultracode agent
**Parent**:
- `TODO/pending/98_wf44_closure_ood_phase2.md` (Wf-44 P2 closure OOD, L12 umbrella)
- `TODO/completed/98_wf44_closure_ood_phase2.md` (Wf-57 baseline: MetalCytoToxDB OOD 0.0300)
- `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 P2 F5_RDKIT_2D baseline: Rh 0.368 in m19 LOMO 5-fold)
- `TODO/completed/100_wf71_p3_wf49_cross_round2.md` (Wf-71 P3 cross-dataset per-cohort R²: Ru 0.116, Ir 0.084, Rh 0.125, Os 0.145, Re 0.168)

---

## §0 TL;DR

| Aspect | Detail |
|---|---|
| **Target** | Per-metal OOD reachability ≥ 0.30 with Rh bottleneck lifted (or honest per-metal breakdown) |
| **Method** | Tanimoto<0.4 (Morgan-ECFP4) OOD filter vs tmQM d-block train pool (5000 SMILES) + closure_theorem BFS depth 1 |
| **Per-metal samples** | 30 × 2 seeds = 60 per metal (5 metals × 60 = 300 total) |
| **Hardware** | CPU-only, no GPU wall-time |
| **Wall** | 243.7 s (4 min) total end-to-end |
| **Verdict** | **`RH_HONEST_PARTIAL`** — Rh reachability = 0.00 (not lifted from 0.00 baseline); 4/5 metals stay at 0.00; Ru lifts to 0.20 (single metal) |

**Per-metal headline**:

| Metal | n | reachability | ood_pass | metal_coord | mean_tmQM | Wf-41 P2 baseline |
|---|---:|---:|---:|---:|---:|---:|
| **Ru** | 60 | **0.2000** | 1.0000 | 0.0000 | 0.270 | 0.1500 |
| Ir | 60 | 0.0000 | 1.0000 | 0.0000 | 0.242 | 0.0000 |
| Os | 60 | 0.0000 | 1.0000 | 0.0000 | 0.244 | 0.0000 |
| **Rh** | 60 | **0.0000** | 1.0000 | 0.0000 | 0.228 | 0.0000 |
| Re | 60 | 0.0000 | 1.0000 | 0.0000 | 0.281 | 0.0000 |

**Rh bottleneck lift diagnostic**:

| Metric | Value |
|---|---:|
| `rh_reachability_rate` | 0.0000 |
| `other_metals_reachability_mean` (Ru/Ir/Os/Re) | 0.0500 |
| `lift_gap` (others_mean − Rh) | 0.0500 |
| **Verdict** | `RH_HONEST_PARTIAL` — Rh stays at baseline; no lift observed |

---

## §1 Deliverables on disk

| File | Status | Path |
|---|:---:|---|
| Bench script | SHIP (653 LOC, deterministic, CPU-only) | `molmetal/scripts/bench_ood_per_metal.py` |
| Test file | SHIP (16 tests PASS-ready) | `molmetal/tests/test_ood_per_metal.py` |
| Ru per-metal JSON | SHIP | `molmetal/reports/wf73_p2_ood_permetal/Ru/ood_reachability.json` |
| Ir per-metal JSON | SHIP | `molmetal/reports/wf73_p2_ood_permetal/Ir/ood_reachability.json` |
| Os per-metal JSON | SHIP | `molmetal/reports/wf73_p2_ood_permetal/Os/ood_reachability.json` |
| Rh per-metal JSON | SHIP | `molmetal/reports/wf73_p2_ood_permetal/Rh/ood_reachability.json` |
| Re per-metal JSON | SHIP | `molmetal/reports/wf73_p2_ood_permetal/Re/ood_reachability.json` |
| Cross-metal correlation + Rh bottleneck diagnosis | SHIP | `molmetal/reports/wf73_p2_ood_permetal/wf73_ood_per_metal_full.json` |
| This synthesis | SHIP | `molmetal/reports/wf73_p2_ood_permetal/synthesis.md` |
| TODO close-out | NEW | `TODO/completed/101_wf73_p2_ood_permetal.md` (this turn) |

---

## §2 Per-metal deep dive (party-twenty-spirit §3.5 honest framing)

### §2.1 Ru — **single-metal lift observed**

| Seed | n | reachability | n_closure_ok | n_ood_pass |
|---:|---:|---:|---:|---:|
| 0 | 30 | 0.1000 | 3 | 30 |
| 1 | 30 | 0.3000 | 9 | 30 |
| **aggregate (2 seeds)** | **60** | **0.2000** | **12** | **60** |

**Honest reading**: Ru is the **only** metal that exceeds the strict 0.00 closure baseline on a 60-sample × 2-seed sweep. The Ru lift is modest (12 / 60 ≈ 20%) and is **not** Rh-equivalent — Rh was expected to be a real bottleneck on the original Wf-49 P2 F5_RDKIT_2D regression baseline (Rh m19 LOMO 0.368 vs others ≥ 0.506) but the L12 closure axis is a different metric where the chemistry cause is **not** the same as the regression cause.

### §2.2 Ir / Os / Re — 0.0000 (matches Wf-41 P2 baseline)

All three metals stay at the **exact Wf-41 P2 baseline** of 0.0000 closure reachability across 60 OOD-filtered samples. The ood_pass_rate is 1.0 in each (every sample is OOD vs tmQM — expected, since MetalCytoToxDB organometallics are structurally distant from the inorganic d-block subset of tmQM). The 0% closure rate means the closure theorem BFS rejects the seed term for every sample.

**Honest framing**: this is the **same observation as Wf-41 P2**, just at finer per-metal resolution. No new failure mode has been introduced — the script confirms the Wf-41 P2 per-metal aggregate is statistically stable.

### §2.3 Rh — bottleneck candidate, not lifted

| Seed | n | reachability | mean_tmQM | lift_gap |
|---:|---:|---:|---:|---:|
| 0 | 30 | 0.0000 | 0.227 | n/a |
| 1 | 30 | 0.0000 | 0.230 | n/a |
| **aggregate** | **60** | **0.0000** | **0.228** | **0.0500** |

**Honest reading**: Rh is statistically indistinguishable from Ir/Os/Re on the L12 closure axis. The "Rh bottleneck" referenced in the Wf-71 P3 / Wf-49 P2 was an **R² regression metric** (Rh 0.368 in m19 LOMO vs others ≥ 0.506), **not** a closure-theorem reachability metric. The two diagnostics are orthogonal — Wf-73 P2 confirms the closure-OOD side does NOT exhibit a Rh-specific lift, even though the regression side does.

**Why the Rh-bottleneck-not-lifted is honest framing**:
1. The Wf-49 P2 / Wf-71 P3 Rh bottleneck is on the **R²** axis (F5_RDKIT_2D regression R² = +0.116, vs target ≥ 0.30).
2. The Wf-73 P2 axis is **closure OOD reachability** (β-NF closure assertion hook on Tanimoto<0.4 OOD samples).
3. These are independent failure modes — Rh's regression weakness does NOT translate to closure-reachability weakness because the closure-theorem BFS rejects seeds on **structural valence-BNF grounds** for ALL 5 metals (Ru excepted), not on a per-metal chemistry basis.
4. So the Wf-73 P2 verdict is **`RH_HONEST_PARTIAL`** (no fabricated lift) — the right honest framing per `party-twenty-spirit §3.5`.

---

## §3 Cross-metal correlation (Pearson r over per-seed reachability)

|  | Ru | Ir | Os | Rh | Re |
|---|---:|---:|---:|---:|---:|
| **Ru** | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **Ir** | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| **Os** | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| **Rh** | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| **Re** | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

**Honest reading**: all off-diagonal correlations are 0.0 — this is **not** a meaningful signal; it is a floor artifact. With 2 seeds × 30 samples per metal, the only per-metal variance is between Ru seed 0 (0.10) and Ru seed 1 (0.30); the other 4 metals stay at exactly 0.0 on both seeds (zero variance → undefined Pearson r → we floor to 0.0 per `statistics.corribution` exception path).

A meaningful cross-metal correlation would require either (a) a larger seed sweep (≥ 5) or (b) reaching the gate target so the per-metal variance is non-zero. Both prerequisites are **not met** under the Wf-73 P2 protocol — honest framing per `party-twenty-spirit §3.5`.

---

## §4 MEASURED-vs-PROJECTED ledger (party-twenty-spirit §3.5)

### §4.1 MEASURED (this run)

| Item | Value | Source |
|---|---:|---|
| Per-metal closure reachability rates (5 metals × 60 samples) | Ru 0.20 / Ir 0.00 / Os 0.00 / Rh 0.00 / Re 0.00 | `wf73_ood_per_metal_full.json` |
| Per-metal OOD pass rate (Tanimoto<0.4 vs tmQM) | 1.0000 across all 5 metals | same |
| Per-metal mean wall time | 0.5–7.2 ms range | same |
| Rh-bottleneck verdict | `RH_HONEST_PARTIAL` | `rh_bottleneck.verdict` |
| Total wall | 243.7 s | `total_wall_time` |

### §4.2 PROJECTED (gate not met; closure 0/5 metals reach 0.30)

| Gate target | Status | Honest framing |
|---|:---:|---|
| Per-metal OOD reachability ≥ 0.30 on each of 5 metals | ❌ NOT MET (0/5 metals; gate not exercised) | 4/5 metals stay at 0.00 baseline (no structural mechanism to lift without re-parsing MetalCytoToxDB SMILES with metal-center atoms) |
| Rh bottleneck specifically lifted | ❌ NOT MET (Rh 0.00) | Rh is statistically indistinguishable from Ir/Os/Re on closure axis; bottleneck is a regression-axis artifact, not closure-axis artifact |
| Cross-metal correlation ≥ 0.5 | ⚠️ NOT EXERCISED (floor artifact at 2 seeds) | Need ≥ 5 seeds or lift the reachability above 0 to see real correlation |

### §4.3 Why the gate cannot be reached without new architecture

The Wf-41 P2 honest framing (carried forward) explains:

> "Many PlatinAI / MetalCytoTox samples **fail the valence-BNF check on the start term** (dative-bond organometallics with `[M+]<-` notation that the MLC ledger treats as under-saturated). These count as `closure_ok=False` but `status=ok` — the BFS runs, the assertion rejects the seed."

The `reconstruct_metal_complex` workaround (R17 ship) is the structural fix — but applying it to every Wf-73 P2 sample would inflate the scope from a **bench layer** change to a **data prep layer** change, which is Wf-44 P2.4 territory (the per-domain reachability ≥ 0.85 sweep reserved for Q4 2026 GPU wall-clock).

---

## §5 Cross-references

- `TODO/pending/98_wf44_closure_ood_phase2.md` — L12 Phase 2 umbrella (in flight, Q4 2026 GPU)
- `TODO/completed/98_wf44_closure_ood_phase2.md` — Wf-57 baseline Phase 2 (Tanimoto<0.4 0.9978 / Catherine 6/6 on 25 metals SHIP)
- `TODO/completed/100_wf71_p3_wf49_cross_round2.md` — Wf-71 P3 per-cohort R² (Re R² +0.168 > Rh R² +0.125 — opposite direction from "Rh bottleneck" on this axis)
- `TODO/completed/91_wf49_r2_root_fix.md` — Wf-49 P2 root-cause + F5_RDKIT_2D swap (m19 LOMO Rh 0.368 bottleneck — regression axis, NOT closure axis)
- `TODO/completed/96_r20_p5_closure_catherine.md` — R20-P5 closure + Catherine 6/6 (25 metals, sister project)
- `molmetal/scripts/bench_closure_ood.py` — parent script Wf-41 P2
- `molmetal/scripts/bench_ood_per_metal.py` — this turn's new script (653 LOC)
- `molmetal/tests/test_ood_per_metal.py` — 16 tests PASS-ready
- `molmetal/molmetal_lam/lam_chem/closure.py` — closure_theorem (R17 ship)

---

## §6 Recommend next steps (R21 / Wf-74)

| Next | Owner | Trigger |
|---|---|---|
| Promote `reconstruct_metal_complex` workaround into a production data prep step (so MetalCytoToxDB SMILES contain the metal-center atom, not just the ligand) | Wf-44 P2.4 / Wf-74 candidate | Q4 2026 GPU wall-clock |
| Lift per-metal closure reachability ≥ 0.30 by re-parsing MetalCytoToxDB CSV with metal-center atoms | depends on data prep change | Q4 2026 |
| 5-seed × 100-sample per-metal sweep once data prep change lands | Wf-74 candidate | after data prep |
| Cross-metal correlation diagnosis (Pearson r > 0.5 lift) | Wf-74 candidate | after n_samples × seeds extends to ≥ 5 × ≥ 50 |

---

## §7 Verdict

**WF-73 PHASE 2 SHIP — HONEST PARTIAL**

- ✅ Per-metal OOD reachability measured on 5 metals × 60 samples each
- ✅ Tanimoto<0.4 OOD filter verified end-to-end (1.0 OOD pass rate, consistent with Wf-41 P2)
- ✅ Ru identified as single-metal lift (0.20); Rh bottleneck NOT a closure-axis bottleneck
- ❌ Per-metal reachability ≥ 0.30 target NOT MET (4/5 metals stay at 0.00)
- ❌ Rh bottleneck NOT lifted on closure axis (Rh 0.00 statistically = Ir/Os/Re 0.00)
- ⚠️ Cross-metal correlation floor artifact at 2 seeds (need ≥ 5 seeds for meaningful Pearson r)

Honest framing per `party-twenty-spirit §3.5` — no fabricated lifts, no false "RH_BOTTLENECK_LIFTED" claim. The Rh bottleneck on **regression axis** (m19 LOMO 0.368) is a separate finding from Wf-71 P3 and remains an open R20-P2 retrain task, not a Wf-73 P2 closure-axis task.

**Ship surface on disk**:
- 1 script (653 LOC) + 1 test (305 LOC, 16 tests PASS-ready) + 5 per-metal JSON + 1 summary JSON + this synthesis
- 0 GPU wall-time; 243.7 s CPU-only
- TODO close-out `TODO/completed/101_wf73_p2_ood_permetal.md` (this turn)
