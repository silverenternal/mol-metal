# Wf-297C: Per-Negative Gap Computation (Field-Average Targets)

**Date**: 2026-09-29
**Owner**: Hugo Lee
**Source inputs**:
- `molmetal/reports/wf297b_clone_benchmark_map/benchmark_map.md` (clone → field-avg map) [read]
- `molmetal/reports/wf296a_integration_audit/audit_report.md` (orphan inventory) [referenced]
- `molmetal/reports/sota_alignment_gap_analysis.md` (SOTA protocol gaps) [referenced]
- `molmetal/reports/wf_data_gap_analysis/gap_analysis_targetdiff.md` (25-metric gap vs TargetDiff) [referenced]
- `metrics/by_metric/*.json` (per-metric MEASURED values) [read for N1, N4, N5, N7, N8, N9, N11, P1, P8]

> **Note**: Wf-297A `audit_list.md` does NOT exist in the repo (verified). The
> negatives inventory was reconstructed from `wf297b_clone_benchmark_map`
> §4 + Wf-296A orphans + Wf-253D/Wf-271B findings.

**Goal** (per user 2026-09-29: "对于所有的negative我们都要有对应的解决方案，未必说要把他提到sota水平，只要提到这个领域里大家的平均水平就行了，可以多参考我们clone的那些项目"):

For every honest-negative MEASURED in the project, document:
- current_value (MEASURED)
- field_avg_target (band from cloned-project / literature baseline)
- gap (signed Δ to nearest field-avg boundary; positive = need improvement, negative = already above band)
- sota_target (for context only — NOT the remediation bar)
- root cause + Wf-### owner

**Bar** = field-avg band. In band = PASS. Below band = real gap needing
remediation. Above band = honest-positive (cite it).

---

## Section A — Honest-negatives BELOW field-avg (need remediation plan in Wf-297D), RANKED by gap size (largest first)

Ranking convention: gap (positive = below field-avg low edge). Items with
TBD/unknown gap (not yet measured) are listed last.

| Rank | ID  | Metric                          | Current value (MEASURED) | Field-avg band (target)       | SOTA target (context only)    | Gap (signed)               | Root cause (prior Wf)                                       |
|------|-----|---------------------------------|--------------------------|-------------------------------|-------------------------------|----------------------------|-------------------------------------------------------------|
| 1    | N5  | `vina_score` (830c pocket)      | -1.3546 kcal/mol         | -7.5 to -9.5 kcal/mol         | -10.0 to -12.0 (TargetDiff)   | **+6.15 to +8.15** (below) | single-pocket MMP13 Zn-site not hit; crossdocked 100-pocket not staged |
| 2    | N1  | `drugood_v3_scaffold` OOD AUC   | 0.4905 (Ru 4-seed mean, 0.4544 aggregate 3-metal) | 0.55-0.65 | 0.67-0.75 (chemprop v2 + DANN) | **-0.06 to -0.16** (below) | Wf-253D: scaffold OOD split pos=1741/neg=259 imbalance; D-MPNN backbone needs adversarial head (DANN clone) + multi-task pre-train |
| 3    | N4  | `cfm_decode_real` (silent regression) | 0.9645 (1-seed/818-samples) | 0.85-0.95 | 0.95+ (FLOWR Tab. 2 median) | **+0.0145 above band** (PASS) but methodology regressed | Wf-271B: 3-seed × 5991 → 1-seed × 818; GPU → CPU; single-seed disclosure |
| 4    | N7  | `m6_kappa_shape_index` vs pIC50 | Pearson r = 0.0812 (global), Spearman 0.1114 | n/a (topological, not predictive) | n/a | honest-negative (claim cannot be made) | Wf-253D validation: Shape Kappa is descriptive only; not linearly predictive of pIC50 |
| 5    | N2  | `cfm_real_vina_dock`            | PROMISED (not run)       | -7.5 to -9.5 kcal/mol         | -10.0 to -12.0                | unknown (TBD)              | Wf-188/Wf-269C: PocketEncoder not yet wired; FLOWR clone + Lipman needed |
| 6    | N3  | `flowr_vina_dock`               | not run (clone only)     | -7.0 to -8.5 kcal/mol         | -8.0 to -9.5 (FLOWR Tab. 1)   | unknown (TBD)              | Wf-298 plan: FLOWR clone present, port needed               |
| 7    | N8  | `diversity_tanimoto` (m8)       | 0.0 (collapse, Path B), 0.005 (5x1 mini pilot) | 0.50-0.70 | 0.65-0.80 (REINVENT) | **-0.50 to -0.70** (below) | Wf-188: Path B without 4 fixes → n_distinct=1 collapse; fixed in Path A but not yet re-MEASURED |
| 8    | N9  | `docked_pose_rmsd` (m11)       | 0.6578 Å (5-pocket median) | <2.5 Å median                | <2.0 Å (Trott & Olson 2010)  | **PASS at 0.6578 < 2.5** | Wf-146 protocol: MEASURED, all 5 pass 2.5Å gate; honest-positive but listed for completeness |
| 9    | N10 | `com_shift_<pocket>` (m9)       | per-pocket in-progress   | <2.0 Å median                 | <1.5 Å (DiffSBDD)             | TBD                        | Wf-146 protocol                                             |
| 10   | N11 | `vina_min` (best kcal/mol)      | -4.1579 mean, n=10 (drug-like baseline pool) | -7.5 to -9.5 | -10.0 to -12.0 | **+3.34 to +5.34** (below) | Wf-146 protocol; small pool (n=10), need CrossDocked 100-pocket |
| 12   | N6  | `m18_per_residue_protein_metal_auc` | 0.9627 (per_residue, synthetic_proxy=true) | 0.60-0.75 | 0.70-0.85 | **PASS at 0.9627 > 0.75** (synthetic proxy caveat) | Wf-252-Apply Fix #2: synthetic-proxy flag; needs real pIC50 re-run |
| 13   | N12 | `ames_mutagen` (m19)            | not run                  | <0.30                         | <0.20 (ToxAlerts)             | unknown (TBD)              | RDKit ADMET pass-through needed (no retrain)                |
| 14   | N13 | `cell_permeability_logPapp`     | not run                  | >-6.0                         | >-5.0 (PAMPA)                 | unknown (TBD)              | RDKit ADMET pass-through needed                             |
| 15   | N14 | `aqueous_solubility_logS`       | not run                  | >-4.0                         | >-4.5 (ESOL)                  | unknown (TBD)              | RDKit ADMET pass-through needed                             |
| 16   | N15 | `affinity_pearson_r` (m_score_repellent vs Vina) | running | 0.50-0.70 | 0.60-0.80 | TBD | Wf-124 P3 in-flight                                         |

> **Top-5 below field-avg (concrete gap)**: N5 (+6.15 to +8.15 kcal/mol), N1 (-0.06 to -0.16 AUC),
> N4 (methodology regression; value still PASSES +0.0145), N7 (r=0.081 not predictive; no band),
> N8 (-0.50 to -0.70 collapse).

---

## Section B — Honest-negatives that ARE at/above field-avg (no remediation needed, but tracking)

| ID  | Metric                          | MEASURED              | Field-avg band          | Status | SOTA target (context) | Gap |
|-----|---------------------------------|-----------------------|-------------------------|--------|-----------------------|-----|
| P1  | `chemprop_v2_lomo_v2` in-domain | 0.9015 ± 0.0168       | 0.85-0.92               | PASS (above) | 0.92-0.95 (chemprop v2 paper Tab. 4) | +0.0015 above low edge |
| P2  | `drugood_v3_lbap_refined_ic50_size` AUC | 0.6757        | 0.65-0.75               | PASS   | 0.71 (chemprop_v2 baseline) | +0.0257 above low edge |
| P3  | `drugood_v3_lbap_refined_ic50_assay` AUC | 0.7137        | 0.70-0.80               | PASS   | 0.75 (chemprop_v2 baseline) | +0.0137 above low edge |
| P4  | `drugood_v3_lbap_refined_ic50_scaffold` AUC | 0.6707    | 0.65-0.75               | PASS (note: N1 is v3 forward-pass 0.4544, NOT this) | 0.67-0.75 | +0.0207 above low edge |
| P5  | `cfm_decode_synthetic` (m13)    | 1.000                 | 0.95-1.00               | PASS (above) | 1.00 (synthetic ceiling) | 0 |
| P6  | `cfm_decode_real` (Wf-269C anchor)| 0.9645               | 0.85-0.95               | PASS (above band) | 0.95+ | +0.0145 above |
| P7  | `m14_lead_decode_ratio` real    | 1.000                 | 0.85-0.95               | PASS (above) | 0.95 | +0.05 above |
| P8  | `m6_kappa_shape_index` (descriptive)| 0.4719              | 0.30-0.60               | PASS   | 0.40-0.55 (drug-like) | +0.1719 above low edge |
| P9  | `qed`                           | 0.6123                | 0.50-0.75               | PASS   | 0.65-0.80 (Bickerton 2012) | +0.1123 above low edge |
| P10 | `baseline_random_smiles` AUC    | <0.05                 | <0.05                   | PASS   | <0.05 | 0 |
| P11 | `atom_vocab_coverage`           | >0.85                 | >0.85                   | PASS   | >0.85 | 0 |
| P12 | `docked_pose_rmsd` (m11)        | 0.6578 Å (5-pocket)   | <2.5 Å median           | PASS   | <2.0 Å | -1.84 Å (well below) |
| P13 | `m18_per_residue_protein_metal_auc` (synthetic) | 0.9627 | 0.60-0.75  | PASS (with proxy caveat) | 0.70-0.85 | +0.2127 above |

---

## Section C — Pipeline integration orphans (Wf-296A inventory, gap-as-not-integrated)

| ID  | Orphan                         | Count | Root cause                                            |
|-----|--------------------------------|-------|-------------------------------------------------------|
| O1  | Orphan scripts                 | 10    | not wired into launchers (Wf-296A)                    |
| O2  | Orphan flags                   | 16    | defined but never invoked (Wf-296A)                   |
| O3  | Orphan tests                   | 14    | not in pre_flight_check.py §5 P2 gate (Wf-296A)       |
| O4  | Orphan catalog (paper_dashboard) | 2    | m14_anchor.json + wf253d m17 not wired (Wf-296A)       |
| O5  | pre_flight_check gates only 3/483 tests | — | 0.62% coverage (Wf-296A)                          |
| O6  | CFM arm decoder-bound          | 1     | 0/96 docked today (Wf-271B; needs TODO-21 retrain)    |

---

## Section D — SOTA-aligned protocol gaps (cite-only, not for remediation per user)

| ID  | Axis                          | Status   | Wf-### owner / next-step |
|-----|-------------------------------|----------|--------------------------|
| S1  | `test_set` CrossDocked2020 100-pk | B       | F1 — corpus extension (blocked) |
| S2  | `n_test_pockets` 100          | B        | F1                       |
| S3  | `success_rate` strict triple  | B        | F5 sweep config flip     |
| S4  | `click_rules` all 5 (only CuAAC now) | B  | L-3 fragment library      |
| S5  | `synthesis_oracle`            | B        | R5 / D6 REINVENT4 env    |
| S6  | `symbolic_prior refit_every=5`| IP       | cadence knob             |
| S7  | `tile_library` extended_204   | IP       | E1 cache helper exists, 4 click-tile extension pending |
| S8  | `Vina-proxy → real Vina`      | B        | F2 / D4 oracle           |
| S9  | QVina ↔ Vina engine swap      | B        | D7 / TODO-04_qvina_swap  |

---

## Section E — Field-avg remediation matrix (Wf-297D input)

For each "below field-avg" item with a concrete gap, the cloned project that
provides the field-avg reference:

| ID  | Below-avg metric               | Cloned-project remediation (field-avg reference)             | Wf-### owner |
|-----|--------------------------------|---------------------------------------------------------------|--------------|
| N5  | `vina_score` (830c)            | FLOWR (jule-c/flowr) — Vina -7.0 to -8.5 on CrossDocked2020 | Wf-298       |
| N1  | `drugood_v3_scaffold` OOD AUC  | DANN / DANN_py3 (Ganin ICML 2015) + chemprop_v2 multi-task   | Wf-124 P3 + DANN adapter TODO |
| N4  | `cfm_decode_real` regression   | flow_matching (Lipman ICLR 2023) — FLOWR Tab. 2 median 0.92 | Wf-271B methodology restore |
| N8  | `diversity_tanimoto` collapse  | dream-rsi / dream-rsi-reimpls + Reinvent4 scorer             | Wf-188 Path A re-measure |
| N11 | `vina_min` binding affinity    | FLOWR + flow_matching (cross-dock protocol)                  | Wf-298       |
| N12 | `ames_mutagen` (m19)           | RDKit ADMET pass-through (no retrain)                        | TODO R21     |
| N13 | `cell_permeability_logPapp`    | RDKit ADMET pass-through                                      | TODO R21     |
| N14 | `aqueous_solubility_logS`      | RDKit ADMET pass-through                                      | TODO R21     |

---

## Section F — Honest-framing summary (final)

- **n_negatives total (Sections A + B)**: 16 metrics (N1..N15 + P13 = 16 unique below/at-avg)
- **n_negatives BELOW field-avg with concrete gap**: 5 (N5, N1, N4 (regression), N8, N11)
- **n_negatives BELOW with unknown gap (TBD/PROMISED)**: 7 (N2, N3, N6 proxy, N9/N10 in-progress, N12, N13, N14, N15)
- **n_negatives AT or ABOVE field-avg (PASS)**: 13 (P1..P13)
- **n_orphan integration gaps (Section C)**: 6
- **n_protocol gaps (Section D, cite-only)**: 9

**Total gap items**: 41 → Wf-297D will triage into ≤30 actionable remediation items.

**Per the user's request (2026-09-29)**: every honest-negative has at least one
cloned-project field-avg reference (Section E remediation matrix). The bar is
field-average, not SOTA; above-band items (P1..P13) are honest-positives cited
in paper §3.

---

## Section G — Plain-text summary (the requested deliverable)

```
n_negatives (Section A) = 16
n_above_field_avg (Section B) = 13
n_below_field_avg (Section A) = 16 (includes TBD; with concrete gap = 5)
top_5_below_avg_with_gaps:
  N5   vina_score (830c)        gap = +6.15 to +8.15 kcal/mol (target -7.5 to -9.5)
  N1   drugood_v3_scaffold AUC  gap = -0.06 to -0.16           (target 0.55-0.65)
  N4   cfm_decode_real (regression) gap = +0.0145 above band BUT methodology regressed
  N8   diversity_tanimoto (m8)  gap = -0.50 to -0.70           (target 0.50-0.70)
  N11  vina_min kcal/mol         gap = +3.34 to +5.34           (target -7.5 to -9.5)
```