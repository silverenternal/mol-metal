# Wf-313E — MECHANISM_METRIC_MAPPING extension: 23-mechanism documentation invariant + comprehensive §12 regression-debug table

**Date**: 2026-09-29
**Workflow**: **Wf-313E** — MECHANISM_METRIC_MAPPING update M22-M44 + regression-debug table
**Status**: **SHIPPED (CPU-only documentation + tests)** — 33 / 33 new tests pass.
**Parent doc**: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` (675 → 723 LOC after §12 add)

---

## 1. TL;DR — Honest framing

Per AGENTS.md §1.5 / `party-twenty-spirit` §3.5, this document declares **what Wf-313E actually shipped** vs **what it did not ship**.

### 1.1 What Wf-313E actually shipped (MEASURED this build)

| Deliverable | Path | Status |
|---|---|---|
| §12 comprehensive regression-debug table covering all 20 metrics | `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §12 | **NEW** (this Wf) |
| 33 tests pinning the M22-M44 documentation invariant | `molmetal/tests/test_wf313e_mechanism_metric_mappings.py` | **NEW** (this Wf) |
| Verification: all 23 R18 mechanism entries (M22-M44) are documented | test `test_all_23_mechanisms_present` PASSED | **MEASURED** |
| Verification: §4 + §12 cover m1..m20 | test `test_regression_debug_table` PASSED | **MEASURED** |
| Verification: PROJECTED / MEASURED / BLOCKED / HONEST-NEGATIVE tags are consistent | test `test_projected_tagging_consistency` PASSED + `test_section_12_status_tags` PASSED | **MEASURED** |
| Verification: 3 wave-level cross-mapping tables populated | tests `test_wave{1,2,3}_cross_mapping_table` PASSED | **MEASURED** |
| Verification: §12 covers future R20-P3 / P4 / P6 metrics (m18 m19 m20) | test `test_section_12_covers_future_metrics` PASSED | **MEASURED** |

### 1.2 What Wf-313E did NOT ship (honest-negative)

| Item | Status | Reason |
|---|---|---|
| 23 NEW mechanism modules | **NOT NEW** — already shipped in Wf-18 (`TODO/completed/71_r18_mechanism_mapping.md`, 2026-09-26) | Pre-existing; Wf-313E is documentation extension, not module creation |
| Per-mechanism target gates (R18 GPU retrain) | **PROJECTED** (R18 GPU retrain is deferred — `TODO/03_plans/R18_PLAN.md` Blackwell single-card §risk) | 0 GPU retrain shipped; numbers are PROJECTED not MEASURED |
| m18 / m19 / m20 tests (R20-P3 / P4 / P6 future) | **HONEST-NEGATIVE** — scripts DEFER | `test_leave_one_metal_out.py` + `test_tumor_specific_split.py` not yet authored |
| m12 live AiZynthFinder backend | **BLOCKED** — external dependency | `conda install -c bioconda aizynthfinder` not yet executed |
| m17 MMseqs2 axis | **PROJECTED (PARTIAL)** — Tanimoto axis only | `conda install -c bioconda mmseqs2` not yet executed |

---

## 2. Per-deliverable

| # | Deliverable | Path | LOC | Status |
|---|---|---|---:|---|
| 1 | Comprehensive §12 regression-debug table | `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §12 | ~70 | **NEW (Wf-313E)** |
| 2 | Test file: 33 tests pinning M22-M44 invariant | `molmetal/tests/test_wf313e_mechanism_metric_mappings.py` | 372 | **NEW (Wf-313E)** |
| 3 | This completion doc | `TODO/completed/313e_mechanism_metric_mapping_extension.md` | ~200 | **NEW (Wf-313E)** |

**Total Wf-313E ship**: ~640 LOC of documentation + tests; 33 / 33 tests pass.

---

## 3. §12 regression-debug table — design rationale

### 3.1 Pre-existing §4 surface

The R17-shipping §4 table covered 13 / 20 metrics:
m1, m2, m3, m5, m7, m8, m11, m13, m14, m17

### 3.2 Pre-existing §5 / §6 partial coverage

§5 (R18 W2 failure-mode diagnostic) and §6 (R18 W3 failure-mode diagnostic)
added partial coverage for m3 / m4 / m15 / m21 / m7 / m11 / m18 (paper-§3-only).

### 3.3 §12 unified coverage

§12 consolidates and extends to cover **all 20 metrics** (m1..m20), with each
row declaring one of:

- **MEASURED** — symptom + mechanisms + self-check command verified by an
  existing R17 / R18 unit test.
- **PROJECTED** — R18 GPU retrain target; not yet MEASURED; per AGENTS.md §1.5.
- **BLOCKED** — external dependency required (e.g. `aizynthfinder`, `mmseqs2`).
- **HONEST-NEGATIVE** — script DEFER (e.g. `test_leave_one_metal_out.py` not
  yet authored for m19 / m20 future metrics).

Per AGENTS.md §1.5 + `party-twenty-spirit` §3.5: zero fabricated numbers;
every PROJECTED target is labelled honestly; BLOCKED + DEFER statuses are
declared rather than paraphrased as "promising".

---

## 4. Test design

### 4.1 Test inventory (33 tests)

| Test | What it pins |
|---|---|
| `test_m22_se3_sb` ... `test_m28_macroguide_ph` (7 tests) | W1 mechanism entries: theorem anchor + affected metrics + PROJECTED tag |
| `test_m29_coordination_aware_gnn` ... `test_m35_multiph_pll_integration` (7 tests) | W2 mechanism entries: module anchor + affected metrics |
| `test_m36_skeleton_jump` ... `test_m44_bass_martingale` (9 tests) | W3 mechanism entries: module anchor + affected metrics |
| `test_wave1_cross_mapping_table` ... `test_wave3_cross_mapping_table` (3 tests) | Each wave-level cross-mapping table populated |
| `test_regression_debug_table` | §4 + §12 union covers all 20 metrics (the §12-completeness invariant) |
| `test_section_12_status_tags` | §12 uses MEASURED/PROJECTED/BLOCKED/HONEST-NEGATIVE |
| `test_section_12_covers_future_metrics` | §12 explicitly addresses m18/m19/m20 (R20-P3/P4/P6 future) |
| `test_wave_regression_diagnostic_tables` | Each wave carries its own failure-mode table with 自查命令 |
| `test_all_23_mechanisms_present` | All 23 R18 mechanism subsections exist |
| `test_pareto_insight_aggregations` | Each wave has Pareto 80/20 aggregation |
| `test_projected_tagging_consistency` | GPU-blocked targets carry PROJECTED tag (or paper §3 only for M31/M32) |

### 4.2 Test execution

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf313e_mechanism_metric_mappings.py -v
============================== 33 passed, 1 warning in 0.33s ==============================
```

### 4.3 Per-mechanism source citation (MEASURED)

| Mechanism | Source (file path + section anchor) |
|---|---|
| M22 | `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` line 61 (`### M22 SE(3) Schrödinger Bridge solver`); module `molmetal/math_skills/se3_schrodinger_bridge.py`; theorem Mahmood 2026 |
| M23 | line 71 (`### M23 Transport Relaxation`); `molmetal/math_skills/transport_relaxation.py`; theorem Jiang-Xu-Zhang 2026 |
| M24 | line 80 (`### M24 Multi-Parameter Persistence`); `molmetal/math_skills/multi_parameter_ph.py`; theorem Kim & Jung 2026 |
| M25 | line 89 (`### M25 Hilbert primal-dual optimizer`); `molmetal/molmetal_lam/reward/hilbert_primal_dual.py`; theorem Sun X.-K. & Zheng L.-J. 2026 |
| M26 | line 99 (`### M26 Score-Repellent Monte Carlo`); `molmetal/molmetal_lam/search_alg/score_repellent_mc.py`; theorem Burch-Bardsley 2022 |
| M27 | line 109 (`### M27 W-Flow ODE solver`); `molmetal/adapters/flow_matching_lipman/w_flow.py`; theorem Lelarasmee 1982 + Stanford ICML'26 |
| M28 | line 118 (`### M28 MacroGuide PH`); `molmetal/molmetal_lam/lam_chem/macroguide_ph.py`; theorem Oxford ICML'26 arXiv:2602.14977 |
| M29 | line 294 (`### M29 Coordination-Aware GNN`); `molmetal/adapters/coordination_gnn.py`; theorem Huheey 1993 §4-5 + Reedijk 1987 |
| M30 | line 304 (`### M30 Quotient-Space Diffusion`); `molmetal/adapters/quotient_space_diffusion.py`; theorem Arvind-Petrov-Wang 2026 |
| M31 | line 314 (`### M31 Quantum semantics`); `molmetal/math_skills/quantum_semantics.py`; theorem Ma X.-N. 2026 |
| M32 | line 323 (`### M32 Hochschild cohomology`); `molmetal/math_skills/hochschild_cohomology.py`; theorem Li-Wang-Yang 2026 |
| M33 | line 332 (`### M33 VarianceControlledPUCT`); `molmetal/molmetal_lam/search_alg/variance_puct_integration.py`; theorem Burch-Bardsley 2022 |
| M34 | line 342 (`### M34 Fractional ODE`); `molmetal/adapters/flow_matching_lipman/fractional_ode.py`; theorem Diethelm-Ford-Freed 2002 + Lubich 1986 |
| M35 | line 352 (`### M35 Multi-PH + PLL signature integration`); `molmetal/molmetal_lam/lam_chem/multi_ph_pll_integration.py`; theorem Kim & Jung 2026 + Liu & Feng 2026 |
| M36 | line 435 (`### M36 Skeleton jump / fragment-swap`); `molmetal/molmetal_lam/lam_chem/skeleton_jump.py`; theorem Bemis & Murcko 1996 |
| M37 | line 445 (`### M37 MD relaxation stub`); `molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py`; theorem Rappé 1992 (UFF) |
| M38 | line 455 (`### M38 FG-compat / leave-group veto`); `molmetal/molmetal_lam/lam_chem/fg_compat_veto.py`; theorem March 1985 + Huheey 1993 |
| M39 | line 465 (`### M39 Layered generation`); `molmetal/molmetal_lam/lam_chem/layered_generation.py`; theorem Bemis & Murcko 1996 |
| M40 | line 475 (`### M40 Buyable-reactant gate`); `molmetal/molmetal_lam/lam_chem/buyable_gate.py`; theorem Genheden & Thakkar 2020 |
| M41 | line 485 (`### M41 OOD protocol`); `molmetal/molmetal_lam/lam_chem/ood_protocol.py`; theorem Peng 2022 Pocket2Mol + Soeding 2017 MMseqs2 |
| M42 | line 495 (`### M42 Non-click med-chem rules`); `molmetal/molmetal_lam/lam_chem/non_click_medchem_rules.py`; theorem Carey-Sundberg 2007 + Abdel-Magid 1996 |
| M43 | line 505 (`### M43 Bergman kernel`); `molmetal/math_skills/bergman_kernel.py`; theorem Ma X.-N. 2014 + Ma X.-N. 2026 |
| M44 | line 515 (`### M44 Bass martingale`); `molmetal/math_skills/sb_bass_martingale.py` + `bass_existence.py`; theorem Bass 1997 + Beiglböck-Juillet 2014 |

---

## 5. MEASURED vs PROJECTED summary

### 5.1 MEASURED (this build, 2026-09-29)

| Item | MEASURED | Source |
|---|---|---|
| §12 table LOC | ~70 lines | `MECHANISM_METRIC_MAPPING.md` §12 (post-Wf-313E) |
| Test count | 33 / 33 PASS | `pytest molmetal/tests/test_wf313e_mechanism_metric_mappings.py` |
| §12 coverage | 20 / 20 metrics | `test_regression_debug_table` (asserts §4 ∪ §12 covers m1..m20) |
| §12 status tags | 4 / 4 (MEASURED / PROJECTED / BLOCKED / HONEST-NEGATIVE) | `test_section_12_status_tags` |
| M22-M44 entries | 23 / 23 present | `test_all_23_mechanisms_present` |
| Cross-mapping tables | 3 / 3 (W1 / W2 / W3) | `test_wave{1,2,3}_cross_mapping_table` |
| Pareto insight tables | 3 / 3 (W1 / W2 / W3) | `test_pareto_insight_aggregations` |
| Wave failure-mode tables | 3 / 3 (W1 / W2 / W3) | `test_wave_regression_diagnostic_tables` |
| §12 future-metric coverage | 3 / 3 (m18 m19 m20) | `test_section_12_covers_future_metrics` |

### 5.2 PROJECTED (NOT MEASURED in this build)

| Item | PROJECTED | Reason |
|---|---|---|
| R18 GPU retrain targets | All R18 per-mechanism target gates | Blackwell single-card §risk per `TODO/03_plans/R18_PLAN.md`; 0 GPU retrain shipped |
| m4 success rate variance reduction ≥ 30% | PROJECTED | M26 + M33 ship; GPU joint training DEFER |
| m15 docking RMSD < 2 Å | PROJECTED | M22 + M30 + M37 ship; GPU retrain DEFER |
| m16 Affinity Pearson r ≥ 0.7 | PROJECTED | M44 Bass martingale T1-gated; T1 stability not yet verified |
| m17 DrugOOD OOD AUC uplift | PROJECTED (PARTIAL) | Tanimoto axis only; MMseqs2 axis DEFER |

### 5.3 BLOCKED (external dependency)

| Item | Blocked on |
|---|---|
| m12 synthesis-rate ≥ 60% with live AiZynthFinder | `conda install -c bioconda aizynthfinder` |
| m17 strict dual-axis OOD protocol (Tanimoto<0.4 AND MMseqs2<30%) | `conda install -c bioconda mmseqs2` |

### 5.4 HONEST-NEGATIVE (script DEFER)

| Item | DEFER |
|---|---|
| m19 multi-metal LOMO transfer AUC | `test_leave_one_metal_out.py` not yet authored; R20-P3 TODO-95 scaffold |
| m20 tumor-specific protein-metal binding AUC | `test_tumor_specific_split.py` not yet authored; R20-P6 TODO-97 wet-lab IC50 gate |

---

## 6. Pre-flight (TODO-207)

**Not applicable** — Wf-313E is a documentation extension + unit test addition.
No retrain / panel / ablation launch. Pre-flight v2 25/25 PASS gate is
not required for documentation-only work.

---

## 7. GPU launch (AGENTS.md §1.2 "kill then adapt")

**Not applicable** — Wf-313E ships zero GPU work. All 23 R18 mechanism
modules are CPU-only verification (308 / 308 tests pass per Wf-18 ship).
R18 GPU retrain is separately tracked under TODO-181 G5 (5 hr, user OK
required) + TODO-182 G6 (7.5 hr, user OK required) — both remain
dry-run READY per AGENTS.md §5.1.

---

## 8. Paper work (AGENTS.md §1.3)

**Not applicable** — Wf-313E did NOT touch `paper/*`. The new §12 table is
a developer-facing diagnostic surface, not a paper section. `mol-metal-paper-writing`
skill is READ-only on paper §1-§7 per AGENTS.md §5.3.

---

## 9. Standing rule conformance

| Rule | Status | Notes |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED | **CONFORM** | §5 declares MEASURED / PROJECTED / BLOCKED / HONEST-NEGATIVE for every numerical claim |
| §1.2 "Kill then adapt" GPU | **CONFORM (N/A)** | No GPU work; no user OK required |
| §1.3 No paper work without auth | **CONFORM (N/A)** | No `paper/*` touched |
| §1.4 Pre-flight v2 25/25 PASS | **CONFORM (N/A)** | No retrain / panel / ablation |
| §1.5 Honest-negative framing | **CONFORM** | §1.2 + §5.3 + §5.4 declare honest-negatives explicitly |

---

## 10. Companion files

| Path | Purpose |
|---|---|
| `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` | Main mechanism ↔ metric map (723 LOC post-Wf-313E; §12 added) |
| `molmetal/tests/test_wf313e_mechanism_metric_mappings.py` | 33 / 33 tests pinning M22-M44 documentation invariant |
| `TODO/completed/71_r18_mechanism_mapping.md` | Wf-18 ship doc (2026-09-26) — M22-M44 cross-mapping synthesis |
| `TODO/02_metrics/METRICS_TARGETS.md` | 20-metric definition surface (§2 + §6 + §7) |
| `TODO/03_plans/R18_PLAN.md` | R18 plan with Blackwell single-card §risk |

---

**END OF Wf-313E COMPLETION DOC** — Owner Hugo Lee; cross-ref AGENTS.md §1.5; honest framing mandated throughout.
